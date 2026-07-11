# pyright: strict
"""Direct tests for `ArrHttp`.

The httpx-native transport the raw arr endpoints share. Pins the fail-open
matrix (request error / non-200 / non-JSON / wrong shape -> None + ONE warning
naming the cause), the strict library-fetch matrix (the same failures RAISE
typed `ArrConnectionError`/`ArrAuthError` instead), the GET retry policy
(transient statuses retry with backoff, terminal ones don't), the POST policy
(`post_json` NEVER retries - not idempotent), the per-request GET timeout
override, and the two
security invariants: the API key rides the `X-Api-Key` header (never the
URL) and a redirect is NEVER followed (a 3xx must not replay the key at a new
location).
"""

import json
from typing import cast

import httpx
import pytest
import respx

from pearlarr.modules.arr_http import (
    GET_RETRIES,
    ArrAuthError,
    ArrConnectionError,
    ArrHttp,
    make_httpx_client,
)
from pearlarr.modules.output import Diagnostic, Severity, install_hub
from pearlarr.modules.output.recording import RecordingHub

_URL = "http://arr.test"


def _bind() -> tuple[ArrHttp, RecordingHub]:
    """A bound helper over a plain httpx client, backoff sleeps stubbed out.

    The fail-open warnings ride the hub, so each test gets a fresh RecordingHub.
    """

    recording = RecordingHub()
    install_hub(recording.hub)  # conftest teardown restores the default
    http = ArrHttp.bind(
        client=httpx.Client(),
        url=f"{_URL}/",  # trailing slash must normalize away ("//api" redirects to login)
        api_key="testkey",
        label="Sonarr",
        sleep=lambda _s: None,
    )
    return http, recording


def _one_warning(recording: RecordingHub) -> Diagnostic:
    """The single recorded Diagnostic, asserted to be a WARNING."""

    [note] = recording.of_type(Diagnostic)
    assert note.severity is Severity.WARNING
    return note


@respx.mock
def test_get_json_success_sends_key_in_header_only() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").respond(json={"ok": True})
    http, recording = _bind()

    payload = http.get_json("/api/v3/thing", params={"a": "1"}, warn="unused ({detail})")

    assert payload == {"ok": True}
    request = route.calls.last.request
    assert request.headers["X-Api-Key"] == "testkey"
    assert "testkey" not in str(request.url)
    assert "a=1" in str(request.url)
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_request_error_retries_then_fails_open_with_one_warning() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").mock(side_effect=httpx.ConnectError("boom"))
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="Could not fetch the thing ({detail})") is None
    # The initial attempt plus every retry, then ONE warning naming the error type.
    assert route.call_count == GET_RETRIES + 1
    assert _one_warning(recording).message == "Could not fetch the thing (request failed (ConnectError))"


@respx.mock
def test_retryable_status_retries_then_succeeds() -> None:
    route = respx.get(f"{_URL}/api/v3/thing")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json=[1, 2])]
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="nope ({detail})") == [1, 2]
    assert route.call_count == 2
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_terminal_status_does_not_retry() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="miss ({detail})") is None
    assert route.call_count == 1  # 404 is not transient: no retries
    assert _one_warning(recording).message == "miss (status code 404)"


@respx.mock
def test_redirect_is_never_followed() -> None:
    # A reverse-proxy login bounce: the key must NOT replay at the new location.
    respx.get(f"{_URL}/api/v3/thing").respond(status_code=302, headers={"Location": "http://evil.test/login"})
    elsewhere = respx.get("http://evil.test/login").respond(json={})
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="bounced ({detail})") is None
    assert elsewhere.call_count == 0
    assert _one_warning(recording).message == "bounced (status code 302)"


@respx.mock
def test_non_json_body_fails_open() -> None:
    respx.get(f"{_URL}/api/v3/thing").respond(content=b"<html>login</html>", content_type="text/html")
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="page ({detail})") is None
    assert _one_warning(recording).message == "page (non-JSON body)"


@respx.mock
def test_shape_helpers_reject_the_wrong_shape() -> None:
    respx.get(f"{_URL}/api/v3/object").respond(json={"a": 1})
    respx.get(f"{_URL}/api/v3/array").respond(json=[1])
    http, recording = _bind()

    assert http.get_json_list("/api/v3/object", warn="list ({detail})") is None
    assert http.get_json_dict("/api/v3/array", warn="dict ({detail})") is None
    assert http.get_json_list("/api/v3/array", warn="ok ({detail})") == [1]
    assert http.get_json_dict("/api/v3/object", warn="ok ({detail})") == {"a": 1}
    notes = recording.of_type(Diagnostic)
    assert all(note.severity is Severity.WARNING for note in notes)
    assert [note.message for note in notes] == [
        "list (unexpected payload)",
        "dict (unexpected payload)",
    ]


@respx.mock
def test_warn_template_with_literal_braces_is_safe() -> None:
    # Migration call sites embed filenames/titles, which can carry braces
    # ("Steins;{Gate}"); the fail-open warning must never crash on them.
    respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="Could not parse Steins;{Gate} ({detail})") is None
    assert _one_warning(recording).message == "Could not parse Steins;{Gate} (status code 404)"


@respx.mock
def test_warn_none_fails_open_silently() -> None:
    respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn=None) is None
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_get_timeout_override_rides_the_request() -> None:
    """`timeout=` overrides the client-level timeout for that request only
    (the 120s manual-import scans); the default rides the client's timeout.
    """

    seen: list[dict[str, float | None]] = []

    def _serve(request: httpx.Request) -> httpx.Response:
        seen.append(cast("dict[str, float | None]", request.extensions["timeout"]))
        return httpx.Response(200, json=[])

    respx.get(f"{_URL}/api/v3/manualimport").mock(side_effect=_serve)
    http, _ = _bind()

    assert http.get_json("/api/v3/manualimport", warn=None, timeout=120) == []
    assert http.get_json("/api/v3/manualimport", warn=None) == []
    assert seen[0] == {"connect": 120, "read": 120, "write": 120, "pool": 120}
    assert seen[1] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}


# --- post_json() (single attempt, never retried) ------------------------------


@respx.mock
def test_post_json_sends_body_and_key_in_header_only() -> None:
    route = respx.post(f"{_URL}/api/v3/command").respond(json={"id": 7})
    http, recording = _bind()

    payload = http.post_json("/api/v3/command", json={"name": "ManualImport"}, warn="unused ({detail})")

    assert payload == {"id": 7}
    request = route.calls.last.request
    assert json.loads(request.content) == {"name": "ManualImport"}
    assert request.headers["X-Api-Key"] == "testkey"
    assert "testkey" not in str(request.url)
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_post_json_accepts_201_created() -> None:
    respx.post(f"{_URL}/api/v3/command").respond(status_code=201, json={"id": 8})
    http, recording = _bind()

    assert http.post_json("/api/v3/command", json={"name": "X"}, warn="cmd ({detail})") == {"id": 8}
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_post_json_does_not_retry_a_retryable_status() -> None:
    # 500 retries on a GET; a POST is not idempotent, so ONE attempt only.
    route = respx.post(f"{_URL}/api/v3/command").respond(status_code=500)
    http, recording = _bind()

    assert http.post_json("/api/v3/command", json={"name": "X"}, warn="cmd ({detail})") is None
    assert route.call_count == 1
    assert _one_warning(recording).message == "cmd (status code 500)"


@respx.mock
def test_post_json_does_not_retry_a_transport_error() -> None:
    route = respx.post(f"{_URL}/api/v3/command").mock(side_effect=httpx.ConnectError("boom"))
    http, recording = _bind()

    assert http.post_json("/api/v3/command", json={"name": "X"}, warn="cmd ({detail})") is None
    assert route.call_count == 1
    assert _one_warning(recording).message == "cmd (request failed (ConnectError))"


@respx.mock
def test_post_json_non_json_body_fails_open() -> None:
    respx.post(f"{_URL}/api/v3/command").respond(content=b"<html>login</html>", content_type="text/html")
    http, recording = _bind()

    assert http.post_json("/api/v3/command", json={"name": "X"}, warn="cmd ({detail})") is None
    assert _one_warning(recording).message == "cmd (non-JSON body)"


# --- get_json_list_strict() (the fail-CLOSED library fetch) ------------------


@respx.mock
def test_strict_get_401_raises_auth_error_without_retrying() -> None:
    route = respx.get(f"{_URL}/api/v3/series").respond(status_code=401)
    http, recording = _bind()

    with pytest.raises(ArrAuthError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Sonarr at {_URL} rejected the API key (status code 401)"
    assert route.call_count == 1  # 401 is terminal: no retries
    assert recording.of_type(Diagnostic) == []  # the raise IS the report; no fail-open warning


@respx.mock
def test_strict_get_connection_error_raises_naming_the_url() -> None:
    route = respx.get(f"{_URL}/api/v3/series").mock(side_effect=httpx.ConnectError("boom"))
    http, _ = _bind()

    with pytest.raises(ArrConnectionError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Could not reach Sonarr at {_URL} (request failed (ConnectError))"
    assert route.call_count == GET_RETRIES + 1  # transport errors still retry first


@respx.mock
def test_strict_get_non_json_200_raises() -> None:
    respx.get(f"{_URL}/api/v3/series").respond(content=b"<html>login</html>", content_type="text/html")
    http, _ = _bind()

    with pytest.raises(ArrConnectionError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Could not reach Sonarr at {_URL} (non-JSON body)"


@respx.mock
def test_strict_get_non_list_payload_raises() -> None:
    respx.get(f"{_URL}/api/v3/series").respond(json={"message": "not an array"})
    http, _ = _bind()

    with pytest.raises(ArrConnectionError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Could not reach Sonarr at {_URL} (unexpected payload)"


@respx.mock
def test_strict_get_retryable_status_retries_then_succeeds() -> None:
    route = respx.get(f"{_URL}/api/v3/series")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json=[{"id": 1}])]
    http, recording = _bind()

    assert http.get_json_list_strict("/api/v3/series") == [{"id": 1}]
    assert route.call_count == 2
    assert recording.of_type(Diagnostic) == []


def test_make_httpx_client_pins_the_transport_policy() -> None:
    """The factory pins no-redirects (the key must never ride one) and the
    (connect, read) timeout split every arr call used to pass per-request.
    """

    client = make_httpx_client()
    try:
        assert client.follow_redirects is False
        assert client.timeout.connect == 5
        assert client.timeout.read == 30
    finally:
        client.close()


@pytest.mark.parametrize("url", [f"{_URL}", f"{_URL}/"])
def test_bind_normalizes_the_base_url(url: str) -> None:
    http = ArrHttp.bind(
        client=httpx.Client(),
        url=url,
        api_key="k",
        label="Sonarr",
    )
    assert http.base_url == _URL
