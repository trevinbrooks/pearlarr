# pyright: strict
# pyright: reportPrivateUsage=false
# httpx has no public read of the SSL context a client verifies against, so the verify check reads it off the
# client's private transport.
"""The shared web client and its GET and HEAD retries.

`make_web_client` builds the one client for tracker pages, AniList, webhooks and the mapping downloads. It follows
redirects, always verifies certificates, names pearlarr in its User-Agent, and gives up after 5 seconds to connect
and 30 to read. `get_with_retries` and `head_with_retries` send again after a retryable status or a transport
error, waiting longer each time, or as long as the response's Retry-After says up to a cap. A status that isn't
retryable comes back at once. The status the last try gets comes back too, and only a transport error that
outlasts every retry is raised.
"""

import ssl

import httpcore
import httpx
import pytest
import respx

from pearlarr import __version__
from pearlarr.arr_http import GET_RETRIES
from pearlarr.web_client import RETRY_AFTER_CAP_S, get_with_retries, head_with_retries, make_web_client

_URL = "https://web.test/page"


def _pool_ssl_context(client: httpx.Client) -> ssl.SSLContext:
    """The SSL context the client's connection pool verifies against."""

    transport = client._transport
    assert isinstance(transport, httpx.HTTPTransport)
    # httpx types the pool as the union of its three httpcore pools, one of which is
    # the placeholder httpcore defines when socksio is absent (no pool base, no context).
    pool = transport._pool
    assert isinstance(pool, httpcore.ConnectionPool)
    context = pool._ssl_context
    assert isinstance(context, ssl.SSLContext)
    return context


def test_make_web_client_pins_the_transport_policy() -> None:
    """Redirects are safe to follow, as no credential goes out on this client. No arr's `verify_ssl` reaches it."""

    client = make_web_client()
    try:
        assert client.follow_redirects is True
        assert client.headers["User-Agent"] == f"pearlarr/{__version__}"
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
    # The last response comes back as it is: the caller's `raise_for_status` is what raises on it.
    route = respx.get(_URL).respond(status_code=503)
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 503
    assert route.call_count == GET_RETRIES + 1  # the initial attempt plus every retry
    assert len(sleeps) == GET_RETRIES


@respx.mock
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx.ConnectError("refused"), id="refused"),
        pytest.param(httpx.ReadTimeout("no answer"), id="timed-out"),
    ],
)
def test_get_with_retries_recovers_from_a_transport_error(error: httpx.TransportError) -> None:
    route = respx.get(_URL)
    route.side_effect = [error, httpx.Response(200)]

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=lambda _s: None)

    assert response.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_get_with_retries_exhausted_transport_error_propagates() -> None:
    # A transport error that outlasts every retry is raised, where a status would come back.
    route = respx.get(_URL).mock(side_effect=httpx.ConnectError("still down"))

    with httpx.Client() as client, pytest.raises(httpx.ConnectError):
        get_with_retries(client, _URL, sleep=lambda _s: None)

    assert route.call_count == GET_RETRIES + 1


@respx.mock
def test_get_with_retries_honors_retry_after() -> None:
    # The response's own window beats the backoff: Retry-After: 7 waits 7 seconds, not half of one.
    respx.get(_URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 200
    assert sleeps == [7.0]


@respx.mock
def test_get_with_retries_caps_and_falls_back_on_bad_retry_after() -> None:
    # A window past the cap waits the cap. One that isn't a number of seconds waits that try's backoff.
    respx.get(_URL).mock(
        side_effect=[
            httpx.Response(503, headers={"Retry-After": "9999"}),
            httpx.Response(503, headers={"Retry-After": "soon"}),
            httpx.Response(200),
        ]
    )
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 200
    assert sleeps[0] == RETRY_AFTER_CAP_S
    assert 1.0 <= sleeps[1] <= 1.25  # attempt 1: 0.5 * 2**1 + jitter


@respx.mock
def test_get_with_retries_terminal_status_returns_immediately() -> None:
    route = respx.get(_URL).respond(status_code=404)
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 404
    assert route.call_count == 1  # 404 is not transient: no retries
    assert sleeps == []


@respx.mock
def test_head_with_retries_sends_a_head_under_the_same_policy() -> None:
    route = respx.head(_URL)
    route.side_effect = [httpx.Response(503), httpx.Response(200)]
    sleeps: list[float] = []

    with httpx.Client() as client:
        response = head_with_retries(client, _URL, sleep=sleeps.append)

    assert response.status_code == 200
    assert route.call_count == 2  # the route matches HEAD only, so a GET would fail the mock
    assert len(sleeps) == 1


@respx.mock
def test_a_cookie_set_by_a_retryable_response_is_sent_on_the_retry() -> None:
    cookies_sent: list[str | None] = []

    def _challenge(request: httpx.Request) -> httpx.Response:
        cookies_sent.append(request.headers.get("cookie"))
        if len(cookies_sent) == 1:
            return httpx.Response(503, headers={"Set-Cookie": "challenge=ok"})
        return httpx.Response(200)

    respx.get(_URL).mock(side_effect=_challenge)

    with httpx.Client() as client:
        response = get_with_retries(client, _URL, sleep=lambda _s: None)

    assert response.status_code == 200
    assert cookies_sent == [None, "challenge=ok"]
