"""The shared httpx client for non-arr web traffic, and its GET retry helper.

``make_web_client`` is the second process-wide client beside
:func:`~.arr_http.make_httpx_client`: the arr client stays pinned to
no-redirects and the per-arr ``verify_ssl`` knob (the ``X-Api-Key``
invariants), while external hosts - tracker pages, AniList, webhooks, the
mapping-source downloads - ride this one. ``get_with_retries`` carries the
transient-retry policy those scrapes used to get from the urllib3 ``Retry``
adapter on the old requests session.
"""

import random
import time
from collections.abc import Callable

import httpx

from .arr_http import BACKOFF_BASE_S, GET_RETRIES, RETRYABLE_STATUS
from .. import __version__

# Identify ourselves to the hosts we scrape/post to (AniList's API terms ask
# clients to be identifiable); also what a maintainer would see if this client
# ever misbehaves.
USER_AGENT = f"seadexarr/{__version__}"


def make_web_client() -> httpx.Client:
    """The pinned httpx client all non-arr web traffic shares (one per run).

    - ``follow_redirects=True``: tracker pages and GitHub release downloads
      redirect routinely, and no credential header rides this client, so
      following is safe (unlike the arr client, which must never replay its
      key at a new location).
    - ``verify=True`` always: external hosts never inherit an arr's
      ``verify_ssl`` escape hatch (truststore is injected process-wide at the
      CLI root, so the OS trust store is what backs this).
    - The default ``User-Agent`` header, so every call site is identifiable.
    - Client-level timeout mirroring the (5, 30) (connect, read) bounds every
      external call used to pass per-request, so no call site can forget one.
    - A small pool: non-arr traffic is a handful of scrapes/posts per run.
    """

    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(connect=5, read=30, write=30, pool=5),
        follow_redirects=True,
        verify=True,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    )


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    sleep: Callable[[float], None] = time.sleep,  # injectable so tests don't wait out backoffs
) -> httpx.Response:
    """GET ``url``, retrying transient failures; NEVER raises on a status.

    The web twin of ``ArrHttp._get_with_retries`` (same retry count, backoff
    shape and retryable-status set) - GETs only, so it stays safe for
    idempotent reads. Returns the final response: a success, a terminal
    non-retryable status, or the still-transient one once retries run out -
    the call site's ``raise_for_status`` surfaces a bad status (the urllib3
    policy's ``raise_on_status=False``, preserved).
    """

    for attempt in range(GET_RETRIES):
        try:
            response = client.get(url)
        except httpx.TransportError:
            pass  # transient by definition; the next attempt may recover
        else:
            if response.status_code not in RETRYABLE_STATUS:
                return response
        sleep(BACKOFF_BASE_S * 2**attempt + random.uniform(0, 0.25))
    # The last attempt: a transport error now propagates (retries exhausted,
    # matching the urllib3 Retry policy raising) and any status returns.
    return client.get(url)
