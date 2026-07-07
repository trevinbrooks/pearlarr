# pyright: strict
# pyright: reportPrivateUsage=false
# The verify pin reads the client's private transport chain (httpx exposes no
# public ssl-context accessor); the repo already disables the rule for tests.
"""Direct tests for the shared non-arr web client and its GET retry helper.

``make_web_client`` is the second process-wide client (tracker pages, AniList,
webhooks, mapping downloads): these pin its transport policy - redirects
followed (no credential header rides it), verification always on, the
``seadexarr/<version>`` User-Agent default, and the (5, 30) timeout split -
mirroring ``test_make_httpx_client_pins_the_transport_policy`` for the arr
client. ``get_with_retries`` replaces the urllib3 ``Retry`` adapter the old
requests session mounted, so its matrix is pinned too: transient statuses and
transport errors retry with backoff, a terminal status returns immediately,
retry exhaustion returns the response (never raises on status) but propagates
a transport error.
"""

import ssl

import httpx
import pytest
import respx

from seadexarr import __version__
from seadexarr.modules.arr_http import GET_RETRIES
from seadexarr.modules.web_client import USER_AGENT, get_with_retries, make_web_client

_URL = "https://web.test/page"


def _pool_ssl_context(client: httpx.Client) -> ssl.SSLContext:
    """The SSL context the client's connection pool verifies against."""

    transport = client._transport
    assert isinstance(transport, httpx.HTTPTransport)
    context = transport._pool._ssl_context
    assert isinstance(context, ssl.SSLContext)
    return context


def test_make_web_client_pins_the_transport_policy() -> None:
    """The factory pins redirects-followed (no credential rides this client),
    always-on verification, the identifiable UA default, and the (5, 30)
    (connect, read) timeout split every external call used to pass per-request.
    """

    client = make_web_client()
    try:
        assert client.follow_redirects is True
        assert client.headers["User-Agent"] == USER_AGENT
        assert USER_AGENT == f"seadexarr/{__version__}"
        assert client.timeout.connect == 5
        assert client.timeout.read == 30
        assert client.timeout.write == 30
        assert client.timeout.pool == 5
        assert _pool_ssl_context(client).verify_mode is ssl.CERT_REQUIRED
    finally:
        client.close()


# --- get_with_retries --------------------------------------------------------


@respx.mock
def test_get_with_retries_transient_5xx_recovers() -> None:
    route = respx.get(_URL)
    route.side_effect = [httpx.Response(503), httpx.Response(200, text="ok")]
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 200
    assert route.call_count == 2
    assert len(sleeps) == 1  # one backoff between the 503 and the recovery


@respx.mock
def test_get_with_retries_exhausted_5xx_returns_the_response() -> None:
    # raise_on_status=False parity: the still-503 response is RETURNED so the
    # call site's raise_for_status surfaces it; the helper never raises on status.
    route = respx.get(_URL).respond(status_code=503)
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 503
    assert route.call_count == GET_RETRIES + 1  # the initial attempt plus every retry
    assert len(sleeps) == GET_RETRIES


@respx.mock
def test_get_with_retries_connect_error_recovers() -> None:
    route = respx.get(_URL)
    route.side_effect = [httpx.ConnectError("boom"), httpx.Response(200)]

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=lambda _s: None)

    assert response.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_get_with_retries_exhausted_transport_error_propagates() -> None:
    # Matching urllib3 Retry exhaustion raising: a transport error that survives
    # every retry propagates (unlike a status, which always returns).
    route = respx.get(_URL).mock(side_effect=httpx.ConnectError("still down"))

    with httpx.Client() as client, pytest.raises(httpx.ConnectError):
        get_with_retries(client, _URL, sleep=lambda _s: None)

    assert route.call_count == GET_RETRIES + 1


@respx.mock
def test_get_with_retries_terminal_status_returns_immediately() -> None:
    route = respx.get(_URL).respond(status_code=404)
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 404
    assert route.call_count == 1  # 404 is not transient: no retries
    assert sleeps == []
