import contextlib
import time
from typing import Any, cast

import requests

from .seadex_types import AniListError, AniListMediaNode

API_URL = "https://graphql.anilist.co"

type AniListCache = dict[int, dict[str, dict[str, Any]]]
"""In-memory AniList cache: id -> raw GraphQL body ``{"data": {"Media": {...}}}``.

The cached value is the *whole* response body (what ``get_query`` /
``get_query_batch`` return), so it round-trips verbatim through the persisted
``anilist_meta`` block. ``_get_media`` extracts and parses the ``Media`` node
out of it into an :class:`AniListMediaNode`.
"""

# AniList rate-limits (HTTP 429) and occasionally returns a transient 5xx. Retry
# those a few times with a backoff that respects a Retry-After header, so a busy
# run (many series in quick succession) waits out the limit instead of treating
# the throttled response as real data.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_BACKOFF = 60

# AniList also soft-throttles by returning HTTP 200 with a GraphQL error payload
# (``{"data": null, "errors": [{"message": "Too Many Requests", "status": 429}]}")
# instead of a 429 status. These substrings flag a throttle/rate-limit error so it
# gets the same polite retry as a real 429 rather than being treated as real data.
RETRYABLE_ERROR_SUBSTRINGS = ("too many requests", "rate limit", "throttle")

# Up to this many Media can be fetched in one batched request (AniList's Page
# perPage max). Batching collapses one-request-per-id into a handful, which is
# what keeps a big run from tripping the rate limit one lookup at a time.
ANILIST_BATCH_SIZE = 50

# The Media fields both queries select. Kept in one place so the cached shape is
# identical no matter which query populated it.
_MEDIA_FIELDS = """
    id
    title {
        english
        romaji
    }
    coverImage {
        extraLarge
        large
        medium
    }
    episodes
    format
"""

# Single-id query
QUERY = (
    """
query ($id: Int) {
  Media (id: $id, type: ANIME) {
"""
    + _MEDIA_FIELDS
    + """
  }
}
"""
)

# Batched query: many Media in one request via id_in
BATCH_QUERY = (
    """
query ($ids: [Int]) {
  Page (perPage: %d) {
    media (id_in: $ids, type: ANIME) {
"""
    % ANILIST_BATCH_SIZE
    + _MEDIA_FIELDS
    + """
    }
  }
}
"""
)


def _errors_are_retryable(body: dict[str, Any] | None) -> bool:
    """True if a GraphQL body carries a throttle/rate-limit or 5xx-style error

    AniList sometimes soft-throttles with HTTP 200 and a non-empty "errors"
    array (``{"data": null, "errors": [{"message": "Too Many Requests",
    "status": 429}]}``). That should be retried like a real 429. A legitimate
    "not found" is HTTP 200 with "data" present, the entry "null" and *no*
    "errors" array, so a body without errors is never treated as retryable
    here - the caller's null-safe extraction handles it as an ordinary miss.

    Args:
        body (dict[str, Any] | None): The parsed JSON response body
    """

    for err in _parse_errors(body):
        # A 429 or 5xx status carried in the error entry is retryable.
        if err.status in RETRYABLE_STATUS:
            return True
        # Otherwise match the message (case-insensitive) for throttle wording.
        message = err.message.lower()
        if any(s in message for s in RETRYABLE_ERROR_SUBSTRINGS):
            return True

    return False


def _parse_errors(body: dict[str, Any] | None) -> list[AniListError]:
    """Parse a GraphQL body's ``errors`` array into typed :class:`AniListError`.

    The ``errors`` array is the dynamic GraphQL boundary; this maps each raw
    entry into the typed domain (skipping any non-object entry), so the caller
    reads ``err.status`` / ``err.message`` rather than untyped ``dict`` keys.

    Args:
        body (dict[str, Any] | None): The parsed JSON response body
    """

    raw_errors = (body or {}).get("errors")
    if not isinstance(raw_errors, list):
        return []
    # response.json() is untyped; map each GraphQL error OBJECT into the typed
    # AniListError, skipping any non-object entry (a soft-throttle or malformed
    # body can carry non-dict junk in the errors array).
    return [
        AniListError.from_api(cast("dict[str, Any]", err))
        for err in cast("list[Any]", raw_errors)
        if isinstance(err, dict)
    ]


def _extract(body: dict[str, Any] | None, *path: str) -> dict[str, Any]:
    """Walk a null-safe key path through a GraphQL body, yielding {} on any miss

    AniList returns {"data": null} or {"data": {"Media": null}} for an unknown
    id or a rate-limit, so each hop is guarded with "or {}" and a missing or
    null level yields an empty dict rather than raising
    "'NoneType' object has no attribute 'get'".

    Args:
        body (dict[str, Any] | None): The parsed JSON response body
        *path (str): The keys to walk, e.g. "data", "Media"
    """

    node: dict[str, Any] = body or {}
    for key in path:
        node = node.get(key) or {}
    return node


def _media_from(body: dict[str, Any] | None) -> AniListMediaNode:
    """Parse the Media node from a single-id body into an AniListMediaNode

    The raw ``{"data": {"Media": {...}}}`` body is the dynamic GraphQL boundary;
    this is where it crosses into the typed domain. A miss (``data``/``Media``
    null) yields an all-``None`` node via ``from_api({})``.

    Args:
        body (dict[str, Any] | None): The parsed JSON response body
    """

    return AniListMediaNode.from_api(_extract(body, "data", "Media"))


def _post_with_retry(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST a GraphQL query to AniList, retrying politely on rate-limits / 5xx

    On a rate-limit (HTTP 429) or a transient 5xx, AniList returns
    "{"data": null, ...}". It can also soft-throttle with HTTP 200 and a
    throttle/rate-limit error in the "errors" array (see
    ``_errors_are_retryable``); both take the same backoff path. Returning a
    throttled response untried is what surfaced downstream as
    "'NoneType' object has no attribute 'get'" when a run made many requests
    in quick succession, so we wait (honoring Retry-After when present) and
    retry before giving up. Returns the parsed JSON - which may still be an
    error payload after the final attempt - or "{}" if the response body
    wasn't JSON.
    """

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_URL,
                json={"query": query, "variables": variables},
            )
        except requests.RequestException:
            # Network blip: back off and retry, then give up with an empty result
            if attempt >= MAX_RETRIES:
                return {}
            time.sleep(min(2**attempt, MAX_BACKOFF))
            continue

        retryable = resp.status_code in RETRYABLE_STATUS

        # Parse the body so a soft-throttle (HTTP 200 + throttle error payload)
        # can take the same retry path as a 429 status. A non-JSON body yields
        # {} here and falls through to the return below. response.json() is
        # untyped (the GraphQL body is an open JSON object), so cast at the parse
        # boundary; a non-dict body is rejected by the isinstance guards below.
        try:
            body: dict[str, Any] | None = cast("dict[str, Any]", resp.json())
        except ValueError:
            body = None

        if not retryable and isinstance(body, dict) and _errors_are_retryable(body):
            retryable = True

        if retryable and attempt < MAX_RETRIES:
            # Prefer the server's Retry-After (seconds); otherwise exponential
            retry_after = resp.headers.get("Retry-After")
            wait = 2**attempt
            if retry_after is not None:
                with contextlib.suppress(TypeError, ValueError):
                    wait = float(retry_after)

            time.sleep(min(max(wait, 1), MAX_BACKOFF))
            continue

        # Final attempt (or a non-retryable response): return the parsed body,
        # or {} when the body wasn't JSON, so the caller degrades gracefully.
        return body if body is not None else {}

    return {}


def get_query(al_id: int) -> dict[str, Any]:
    """Fetch one AniList Media by id (see _post_with_retry for the retry policy)

    Args:
        al_id (int): Anilist ID
    """

    return _post_with_retry(QUERY, {"id": al_id})


def get_query_batch(al_ids: list[int]) -> AniListCache:
    """Fetch up to ANILIST_BATCH_SIZE AniList Media in a single request via id_in

    Returns "{id: {"data": {"Media": {...}}}}" mirroring the single-id shape,
    so the results can seed the same cache directly. Ids unknown to AniList are
    simply absent from the result.

    Args:
        al_ids (list[int]): Up to ANILIST_BATCH_SIZE AniList IDs
    """

    j = _post_with_retry(BATCH_QUERY, {"ids": list(al_ids)})
    # response.json() is untyped and the cache stores each Media body verbatim
    # (re-parsed on read via AniListMediaNode.from_api). Keep each Media OBJECT
    # carrying an id, skipping any non-object entry in the array.
    out: AniListCache = {}
    for raw in cast("list[Any]", _extract(j, "data", "Page").get("media") or []):
        if not isinstance(raw, dict):
            continue
        media = cast("dict[str, Any]", raw)
        media_id = media.get("id")
        if media_id is not None:
            out[media_id] = {"data": {"Media": media}}
    return out


def _get_media(
    al_id: int,
    al_cache: AniListCache | None,
) -> AniListMediaNode:
    """Fetch and parse the AniList Media node for an ID, caching successful lookups

    Centralizes the cache lookup and the once-only parse the public helpers
    share: the raw ``{"data": {"Media": {...}}}`` body is the cached value
    (a successful lookup is stored into the caller's ``al_cache`` in place), but
    callers receive a typed :class:`AniListMediaNode`. AniList returns
    {"data": {"Media": null}} (or "{"data": null}") for an unknown ID or a
    rate-limit, so a miss yields an all-``None`` node rather than raising. A miss
    is deliberately not cached, so a transient rate-limit isn't remembered as a
    permanent "unknown" for the rest of the run - the next call gets a fresh
    chance.

    Args:
        al_id (int): Anilist ID
        al_cache (AniListCache | None): Cache of prior AniList bodies, keyed by ID

    Returns:
        AniListMediaNode: The parsed node; all-``None`` on a miss.
    """

    if al_cache is None:
        al_cache = {}

    # Cache hit: parse the stored body's Media node.
    j = al_cache.get(al_id)
    if j is not None:
        return _media_from(j)

    # Miss: query AniList. Extract the raw Media dict once to gate the cache
    # store, then parse it into the typed node for the return.
    j = get_query(al_id)
    raw_media = _extract(j, "data", "Media")

    # Only remember a response that actually carried Media, so a transient
    # failure (rate-limit, network) isn't cached as a permanent miss. The
    # cached body is only ever read, so store it directly rather than copying.
    if raw_media:
        al_cache[al_id] = j

    return AniListMediaNode.from_api(raw_media)


def get_anilist_n_eps(
    al_id: int,
    al_cache: AniListCache | None = None,
) -> int | None:
    """Query AniList to get the number of episodes for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (AniListCache): Cached Anilist bodies, updated in place.
            Defaults to None (no caching across calls)
    """

    return _get_media(al_id, al_cache).episodes


def get_anilist_title(
    al_id: int,
    al_cache: AniListCache | None = None,
) -> str | None:
    """Query AniList to get a title for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (AniListCache): Cached Anilist bodies, updated in place.
            Defaults to None (no caching across calls)
    """

    media = _get_media(al_id, al_cache)

    # Prefer the English title, but fall back to romaji
    return media.title_english or media.title_romaji


def get_anilist_thumb(
    al_id: int,
    al_cache: AniListCache | None = None,
) -> str | None:
    """Query AniList to get thumbnail URL for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (AniListCache): Cached Anilist bodies, updated in place.
            Defaults to None (no caching across calls)
    """

    return _get_media(al_id, al_cache).cover_image


def get_anilist_format(
    al_id: int,
    al_cache: AniListCache | None = None,
) -> str | None:
    """Query AniList to get format for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (AniListCache): Cached Anilist bodies, updated in place.
            Defaults to None (no caching across calls)
    """

    return _get_media(al_id, al_cache).format
