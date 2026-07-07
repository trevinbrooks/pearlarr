# pyright: strict
"""Direct tests for :class:`~seadexarr.modules.arr_http.ArrHttp`.

The httpx-native transport the raw arr endpoints share. Pins the fail-open
matrix (request error / non-200 / non-JSON / wrong shape -> None + ONE warning
naming the cause), the GET retry policy (transient statuses retry with backoff,
terminal ones don't), and the two security invariants: the API key rides the
``X-Api-Key`` header (never the URL) and a redirect is NEVER followed (a 3xx
must not replay the key at a new location).
"""

import logging

import httpx
import pytest
import respx

from seadexarr.modules.arr_http import GET_RETRIES, ArrHttp, make_httpx_client

from .fakes import CaptureHandler

_URL = "http://arr.test"


def _capture_logger(name: str) -> tuple[logging.Logger, CaptureHandler]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    capture = CaptureHandler()
    logger.addHandler(capture)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, capture


def _bind(name: str) -> tuple[ArrHttp, CaptureHandler]:
    """A bound helper over a plain httpx client, backoff sleeps stubbed out."""

    logger, capture = _capture_logger(name)
    http = ArrHttp.bind(
        client=httpx.Client(),
        url=f"{_URL}/",  # trailing slash must normalize away ("//api" redirects to login)
        api_key="testkey",
        label="Sonarr",
        logger=logger,
        sleep=lambda _s: None,
    )
    return http, capture


@respx.mock
def test_get_json_success_sends_key_in_header_only() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").respond(json={"ok": True})
    http, capture = _bind("arr-http-success")

    payload = http.get_json("/api/v3/thing", params={"a": "1"}, warn="unused ({detail})")

    assert payload == {"ok": True}
    request = route.calls.last.request
    assert request.headers["X-Api-Key"] == "testkey"
    assert "testkey" not in str(request.url)
    assert "a=1" in str(request.url)
    assert capture.records == []


@respx.mock
def test_request_error_retries_then_fails_open_with_one_warning() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").mock(side_effect=httpx.ConnectError("boom"))
    http, capture = _bind("arr-http-conn-error")

    assert http.get_json("/api/v3/thing", warn="Could not fetch the thing ({detail})") is None
    # The initial attempt plus every retry, then ONE warning naming the error type.
    assert route.call_count == GET_RETRIES + 1
    [record] = capture.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "Could not fetch the thing (request failed (ConnectError))"


@respx.mock
def test_retryable_status_retries_then_succeeds() -> None:
    route = respx.get(f"{_URL}/api/v3/thing")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json=[1, 2])]
    http, capture = _bind("arr-http-retry-ok")

    assert http.get_json("/api/v3/thing", warn="nope ({detail})") == [1, 2]
    assert route.call_count == 2
    assert capture.records == []


@respx.mock
def test_terminal_status_does_not_retry() -> None:
    route = respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    http, capture = _bind("arr-http-404")

    assert http.get_json("/api/v3/thing", warn="miss ({detail})") is None
    assert route.call_count == 1  # 404 is not transient: no retries
    [record] = capture.records
    assert record.getMessage() == "miss (status code 404)"


@respx.mock
def test_redirect_is_never_followed() -> None:
    # A reverse-proxy login bounce: the key must NOT replay at the new location.
    respx.get(f"{_URL}/api/v3/thing").respond(status_code=302, headers={"Location": "http://evil.test/login"})
    elsewhere = respx.get("http://evil.test/login").respond(json={})
    http, capture = _bind("arr-http-redirect")

    assert http.get_json("/api/v3/thing", warn="bounced ({detail})") is None
    assert elsewhere.call_count == 0
    [record] = capture.records
    assert record.getMessage() == "bounced (status code 302)"


@respx.mock
def test_non_json_body_fails_open() -> None:
    respx.get(f"{_URL}/api/v3/thing").respond(content=b"<html>login</html>", content_type="text/html")
    http, capture = _bind("arr-http-html")

    assert http.get_json("/api/v3/thing", warn="page ({detail})") is None
    [record] = capture.records
    assert record.getMessage() == "page (non-JSON body)"


@respx.mock
def test_shape_helpers_reject_the_wrong_shape() -> None:
    respx.get(f"{_URL}/api/v3/object").respond(json={"a": 1})
    respx.get(f"{_URL}/api/v3/array").respond(json=[1])
    http, capture = _bind("arr-http-shape")

    assert http.get_json_list("/api/v3/object", warn="list ({detail})") is None
    assert http.get_json_dict("/api/v3/array", warn="dict ({detail})") is None
    assert http.get_json_list("/api/v3/array", warn="ok ({detail})") == [1]
    assert http.get_json_dict("/api/v3/object", warn="ok ({detail})") == {"a": 1}
    assert [r.getMessage() for r in capture.records] == [
        "list (unexpected payload)",
        "dict (unexpected payload)",
    ]


@respx.mock
def test_warn_none_fails_open_silently() -> None:
    respx.get(f"{_URL}/api/v3/thing").respond(status_code=404)
    http, capture = _bind("arr-http-quiet")

    assert http.get_json("/api/v3/thing", warn=None) is None
    assert capture.records == []


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
        logger=logging.getLogger("arr-http-url"),
    )
    assert http.base_url == _URL
