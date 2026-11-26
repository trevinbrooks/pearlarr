# pyright: strict
"""Tests for `ArrHttp`, the transport every Sonarr and Radarr request goes through.

A failed fail-open read gives None and one warning naming the cause, and the strict library read raises a
typed error instead. Transient GET failures retry and others don't. A POST never retries, since a repeat
could queue a command twice, and a DELETE leaves its retry to the caller's next run or poll. Repeat
failures stay quiet, re-warn on a timer and end with a recovery note. The API key travels only in the
`X-Api-Key` header, and a redirect is never followed, so a 3xx can't replay the key somewhere else.
"""

import json
import threading
from dataclasses import replace
from typing import cast

import httpx
import pytest
import respx

from pearlarr.arr_http import (
    GET_RETRIES,
    ArrAuthError,
    ArrHttp,
    ArrUnreachableError,
    DeleteOutcome,
    make_arr_client,
)
from pearlarr.output import Diagnostic, Severity

from .fakes import FakeClock, RecordingHub, bind_arr_http

_URL = "http://arr.test"


def _bind() -> tuple[ArrHttp, RecordingHub]:
    """The shared bound helper, fed the trailing slash that must normalize away ("//api" redirects to login)."""

    return bind_arr_http(f"{_URL}/")


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
    # Call sites embed filenames/display names, which can carry braces
    # ("Steins;{Gate}"). The fail-open warning must never crash on them.
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
def test_get_timeout_override_is_sent_with_the_request() -> None:
    """`timeout=` overrides the client-level timeout for that request only.

    Used for the 120s ManualImport folder fetches. The default is the client's timeout.
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


# --- retries=0 poll handle + failure coalescing --------------------------------


@respx.mock
def test_no_retry_handle_attempts_once_primary_keeps_the_budget() -> None:
    """`replace(http, retries=0)` makes exactly ONE attempt. The primary still retries.

    The wait pass's poll handle: its poll loop IS the retry mechanism.
    """

    route = respx.get(f"{_URL}/api/v3/thing").respond(status_code=500)
    http, recording = _bind()

    assert replace(http, retries=0).get_json("/api/v3/thing", warn="poll ({detail})") is None
    assert route.call_count == 1
    assert http.get_json("/api/v3/thing", warn="sweep ({detail})") is None
    assert route.call_count == 1 + GET_RETRIES + 1
    assert [note.message for note in recording.of_type(Diagnostic)] == [
        "poll (status code 500)",
        "sweep (status code 500)",
    ]


@respx.mock
def test_replace_shares_the_failure_streaks() -> None:
    """A failure on the no-retry poll handle closes with a recovery note on the primary's next success.

    `dataclasses.replace` re-runs `default_factory` for init=False fields, so a streak kept on the handle
    itself would split in two and its warnings and recovery notes double up.
    """

    route = respx.get(f"{_URL}/api/v3/thing")
    route.side_effect = [httpx.Response(404), httpx.Response(200, json=[])]
    http, recording = _bind()

    assert replace(http, retries=0).get_json("/api/v3/thing", warn="poll ({detail})") is None
    assert http.get_json("/api/v3/thing", warn="poll ({detail})") == []
    assert [note.message for note in recording.of_type(Diagnostic)] == [
        "poll (status code 404)",
        "poll - recovered after 1 failure (0s)",
    ]


@respx.mock
def test_repeat_failures_coalesce_then_recover() -> None:
    """A repeat failure stays quiet, the next success notes the total, and a failure after that warns afresh."""

    route = respx.get(f"{_URL}/api/v3/thing")
    route.side_effect = [httpx.Response(404), httpx.Response(404), httpx.Response(200, json=[1]), httpx.Response(404)]
    http, recording = _bind()

    assert http.get_json("/api/v3/thing", warn="miss ({detail})") is None
    assert http.get_json("/api/v3/thing", warn="miss ({detail})") is None
    assert http.get_json("/api/v3/thing", warn="miss ({detail})") == [1]
    assert http.get_json("/api/v3/thing", warn="miss ({detail})") is None

    assert [(note.severity, note.message) for note in recording.of_type(Diagnostic)] == [
        (Severity.WARNING, "miss (status code 404)"),
        (Severity.INFO, "miss - recovered after 2 failures (0s)"),
        (Severity.WARNING, "miss (status code 404)"),
    ]


@respx.mock
def test_interleaved_keys_keep_independent_streaks() -> None:
    """Two alternating failing templates never share a streak. Recovery is per-template."""

    route_a = respx.get(f"{_URL}/api/v3/a")
    route_a.side_effect = [httpx.Response(404), httpx.Response(404), httpx.Response(200, json=[])]
    respx.get(f"{_URL}/api/v3/b").respond(status_code=404)
    http, recording = _bind()

    assert http.get_json("/api/v3/a", warn="a-broke ({detail})") is None
    assert http.get_json("/api/v3/b", warn="b-broke ({detail})") is None
    assert http.get_json("/api/v3/a", warn="a-broke ({detail})") is None
    assert http.get_json("/api/v3/b", warn="b-broke ({detail})") is None
    assert http.get_json("/api/v3/a", warn="a-broke ({detail})") == []
    assert http.get_json("/api/v3/b", warn="b-broke ({detail})") is None

    assert [note.message for note in recording.of_type(Diagnostic)] == [
        "a-broke (status code 404)",
        "b-broke (status code 404)",
        "a-broke - recovered after 2 failures (0s)",
    ]


@respx.mock
def test_rewarns_at_the_bound_cadence() -> None:
    """A long streak re-warns ("still failing") once per re-warn interval."""

    respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    primary, recording = _bind()
    clock = FakeClock()
    http = replace(primary, clock=clock.monotonic, rewarn_s=60.0)

    for at in (0.0, 30.0, 59.0, 60.0, 90.0, 125.0):
        clock.t = at
        assert http.get_json("/api/v3/thing", warn="down ({detail})") is None

    notes = recording.of_type(Diagnostic)
    assert all(note.severity is Severity.WARNING for note in notes)
    assert [note.message for note in notes] == [
        "down (status code 404)",
        "down (status code 404; still failing - attempt 4, 1m)",
        "down (status code 404; still failing - attempt 6, 2m)",
    ]


@respx.mock
def test_concurrent_failures_keep_an_exact_streak_count() -> None:
    """Concurrent sweep threads hammer one failing key, and the recovery note counts every failure.

    `streak.count += 1` is a read-modify-write that would drop increments unlocked (the shared client pool
    serves `SONARR_FETCH_WORKERS` threads). The fake clock holds the note's age at 0s however long the
    threads take.
    """

    route = respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    primary, recording = _bind()
    http = replace(primary, clock=FakeClock().monotonic)
    workers, per_worker = 8, 25
    barrier = threading.Barrier(workers, timeout=30.0)

    def hammer() -> None:
        barrier.wait()
        for _ in range(per_worker):
            assert http.get_json("/api/v3/thing", warn="down ({detail})") is None

    threads = [threading.Thread(target=hammer, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
        assert not thread.is_alive(), "a worker never finished its reads"

    route.respond(json=[])
    assert http.get_json("/api/v3/thing", warn="down ({detail})") == []
    assert [note.message for note in recording.of_type(Diagnostic)] == [
        "down (status code 404)",
        f"down - recovered after {workers * per_worker} failures (0s)",
    ]


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
    # 500 retries on a GET. A POST is not idempotent, so ONE attempt only.
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


# --- delete() ----------------------------------------------------------------


@respx.mock
def test_delete_success_sends_key_in_header_only() -> None:
    route = respx.delete(f"{_URL}/api/v3/queue/7").respond(json={})
    http, recording = _bind()

    assert http.delete("/api/v3/queue/7", warn="unused ({detail})") is DeleteOutcome.OK
    request = route.calls.last.request
    assert request.headers["X-Api-Key"] == "testkey"
    assert "testkey" not in str(request.url)
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_delete_404_is_gone_and_quiet() -> None:
    """A 404 means the target already left: GONE with zero diagnostics, not the failure warning."""

    respx.delete(f"{_URL}/api/v3/queue/7").respond(status_code=404)
    http, recording = _bind()

    assert http.delete("/api/v3/queue/7", warn="rm ({detail})") is DeleteOutcome.GONE
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_delete_404_after_a_failure_closes_the_streak() -> None:
    """A 404 after a failed delete means the item is gone, so it ends the failure streak with a recovery note."""

    route = respx.delete(f"{_URL}/api/v3/queue/7")
    route.side_effect = [httpx.Response(500), httpx.Response(404)]
    http, recording = _bind()

    assert http.delete("/api/v3/queue/7", warn="rm ({detail})") is DeleteOutcome.FAILED
    assert http.delete("/api/v3/queue/7", warn="rm ({detail})") is DeleteOutcome.GONE
    assert [(note.severity, note.message) for note in recording.of_type(Diagnostic)] == [
        (Severity.WARNING, "rm (status code 500)"),
        (Severity.INFO, "rm - recovered after 1 failure (0s)"),
    ]


@respx.mock
@pytest.mark.parametrize(
    ("failure", "warning"),
    [
        pytest.param(httpx.Response(500), "rm (status code 500)", id="non-2xx"),
        pytest.param(httpx.ConnectError("boom"), "rm (request failed (ConnectError))", id="request-error"),
    ],
)
def test_delete_failure_returns_failed_with_one_warning(failure: httpx.Response | Exception, warning: str) -> None:
    route = respx.delete(f"{_URL}/api/v3/queue/7").mock(side_effect=[failure])
    http, recording = _bind()

    assert http.delete("/api/v3/queue/7", warn="rm ({detail})") is DeleteOutcome.FAILED
    assert route.call_count == 1  # one attempt, never retried
    assert _one_warning(recording).message == warning


# --- get_json_list_strict() (the fail-CLOSED library fetch) ------------------


@respx.mock
def test_strict_get_401_raises_auth_error_without_retrying() -> None:
    route = respx.get(f"{_URL}/api/v3/series").respond(status_code=401)
    http, recording = _bind()

    with pytest.raises(ArrAuthError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Sonarr at {_URL} rejected the API key (status code 401)"
    assert route.call_count == 1  # 401 is terminal: no retries
    assert recording.of_type(Diagnostic) == []  # the raise IS the report, no fail-open warning


@respx.mock
def test_strict_get_connection_error_raises_naming_the_url() -> None:
    route = respx.get(f"{_URL}/api/v3/series").mock(side_effect=httpx.ConnectError("boom"))
    http, _ = _bind()

    with pytest.raises(ArrUnreachableError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Could not reach Sonarr at {_URL} (request failed (ConnectError))"
    assert route.call_count == GET_RETRIES + 1  # transport errors still retry first


@respx.mock
def test_strict_get_non_json_200_raises() -> None:
    respx.get(f"{_URL}/api/v3/series").respond(content=b"<html>login</html>", content_type="text/html")
    http, _ = _bind()

    with pytest.raises(ArrUnreachableError) as excinfo:
        http.get_json_list_strict("/api/v3/series")

    assert str(excinfo.value) == f"Could not reach Sonarr at {_URL} (non-JSON body)"


@respx.mock
def test_strict_get_non_list_payload_raises() -> None:
    respx.get(f"{_URL}/api/v3/series").respond(json={"message": "not an array"})
    http, _ = _bind()

    with pytest.raises(ArrUnreachableError) as excinfo:
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


# --- history_since() (the activity read both arr clients share) --------------


@respx.mock
def test_history_since_failure_returns_none_with_one_warning() -> None:
    """A failed history read is None, never an empty window, and its one warning says what the run skips."""

    respx.get(f"{_URL}/api/v3/history/since").respond(status_code=500)
    http, recording = _bind()

    assert http.history_since("2026-06-30T08:00:00Z", include_flags={}) is None
    assert _one_warning(recording).message == (
        "Could not fetch Sonarr history (status code 500) - skipping activity detection this run"
    )


def test_make_httpx_client_pins_the_transport_policy() -> None:
    """The factory pins no-redirects: the key must never follow one.

    It also fixes the connect and read timeouts every arr call shares.
    """

    client = make_arr_client()
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
