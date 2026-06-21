import contextlib
import time

import requests

API_URL = "https://graphql.anilist.co"

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
QUERY = """
query ($id: Int) {
  Media (id: $id, type: ANIME) {
""" + _MEDIA_FIELDS + """
  }
}
"""

# Batched query: many Media in one request via id_in
BATCH_QUERY = """
query ($ids: [Int]) {
  Page (perPage: %d) {
    media (id_in: $ids, type: ANIME) {
""" % ANILIST_BATCH_SIZE + _MEDIA_FIELDS + """
    }
  }
}
"""


def _errors_are_retryable(body: dict | None) -> bool:
    """True if a GraphQL body carries a throttle/rate-limit or 5xx-style error

    AniList sometimes soft-throttles with HTTP 200 and a non-empty "errors"
    array (``{"data": null, "errors": [{"message": "Too Many Requests",
    "status": 429}]}``). That should be retried like a real 429. A legitimate
    "not found" is HTTP 200 with "data" present, the entry "null" and *no*
    "errors" array, so a body without errors is never treated as retryable
    here - the caller's null-safe extraction handles it as an ordinary miss.

    Args:
        body (dict): The parsed JSON response body
    """

    errors = (body or {}).get("errors")
    if not errors or not isinstance(errors, list):
        return False

    for err in errors:
        if not isinstance(err, dict):
            continue
        # A 429 or 5xx status carried in the error entry is retryable.
        status = err.get("status")
        if status in RETRYABLE_STATUS:
            return True
        # Otherwise match the message (case-insensitive) for throttle wording.
        message = str(err.get("message") or "").lower()
        if any(s in message for s in RETRYABLE_ERROR_SUBSTRINGS):
            return True

    return False


def _post_with_retry(query: str, variables: dict) -> dict:
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
                API_URL, json={"query": query, "variables": variables},
            )
        except requests.RequestException:
            # Network blip: back off and retry, then give up with an empty result
            if attempt >= MAX_RETRIES:
                return {}
            time.sleep(min(2 ** attempt, MAX_BACKOFF))
            continue

        retryable = resp.status_code in RETRYABLE_STATUS

        # Parse the body so a soft-throttle (HTTP 200 + throttle error payload)
        # can take the same retry path as a 429 status. A non-JSON body yields
        # {} here and falls through to the return below.
        try:
            body = resp.json()
        except ValueError:
            body = None

        if not retryable and isinstance(body, dict) and _errors_are_retryable(body):
            retryable = True

        if retryable and attempt < MAX_RETRIES:
            # Prefer the server's Retry-After (seconds); otherwise exponential
            retry_after = resp.headers.get("Retry-After")
            wait = 2 ** attempt
            if retry_after is not None:
                with contextlib.suppress(TypeError, ValueError):
                    wait = float(retry_after)

            time.sleep(min(max(wait, 1), MAX_BACKOFF))
            continue

        # Final attempt (or a non-retryable response): return the parsed body,
        # or {} when the body wasn't JSON, so the caller degrades gracefully.
        return body if body is not None else {}

    return {}


def get_query(al_id: int) -> dict:
    """Fetch one AniList Media by id (see _post_with_retry for the retry policy)

    Args:
        al_id (int): Anilist ID
    """

    return _post_with_retry(QUERY, {"id": al_id})


def get_query_batch(al_ids: list[int]) -> dict:
    """Fetch up to ANILIST_BATCH_SIZE AniList Media in a single request via id_in

    Returns "{id: {"data": {"Media": {...}}}}" mirroring the single-id shape,
    so the results can seed the same cache directly. Ids unknown to AniList are
    simply absent from the result.

    Args:
        al_ids (list[int]): Up to ANILIST_BATCH_SIZE AniList IDs
    """

    j = _post_with_retry(BATCH_QUERY, {"ids": list(al_ids)})
    media_list = (
        ((j or {}).get("data") or {}).get("Page") or {}
    ).get("media") or []

    return {
        m["id"]: {"data": {"Media": m}}
        for m in media_list
        if isinstance(m, dict) and m.get("id") is not None
    }


def _get_media(
    al_id: int,
    al_cache: dict | None,
) -> tuple[dict, dict]:
    """Fetch the AniList Media object for an ID, caching successful lookups

    Centralizes the cache lookup and the null-safe extraction the public helpers
    share. AniList returns {"data": {"Media": null}} (or "{"data": null}")
    for an unknown ID or a rate-limit, so every level is guarded with "or {}"
    and a miss yields an empty dict rather than raising
    "'NoneType' object has no attribute 'get'". A miss is deliberately not
    cached, so a transient rate-limit isn't remembered as a permanent "unknown"
    for the rest of the run - the next call gets a fresh chance.

    Args:
        al_id (int): Anilist ID
        al_cache (dict | None): Cache of prior AniList responses, keyed by ID

    Returns:
        tuple: (media_dict, al_cache) media_dict is {} on a miss.
    """

    if al_cache is None:
        al_cache = {}

    # Try and find query in cache
    j = al_cache.get(al_id)

    # If we don't have it, do the query
    if j is None:
        j = get_query(al_id)
        # Only remember a response that actually carried Media, so a transient
        # failure (rate-limit, network) isn't cached as a permanent miss. The
        # cached payload is only ever read (the helpers do .get() lookups, no
        # mutation), so store it directly rather than deep-copying.
        if ((j or {}).get("data") or {}).get("Media"):
            al_cache[al_id] = j

    media = ((j or {}).get("data") or {}).get("Media") or {}

    return media, al_cache


def get_anilist_n_eps(
    al_id: int,
    al_cache: dict | None = None,
) -> tuple[int | None, dict]:
    """Query AniList to get the number of episodes for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return media.get("episodes", None), al_cache


def get_anilist_title(
    al_id: int,
    al_cache: dict | None = None,
) -> tuple[str | None, dict]:
    """Query AniList to get a title for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    # Prefer the English title, but fall back to romaji
    title = media.get("title") or {}

    return (title.get("english") or title.get("romaji")), al_cache


def get_anilist_thumb(
    al_id: int,
    al_cache: dict | None = None,
) -> tuple[str | None, dict]:
    """Query AniList to get thumbnail URL for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return (media.get("coverImage") or {}).get("large", None), al_cache


def get_anilist_format(
    al_id: int,
    al_cache: dict | None = None,
) -> tuple[str | None, dict]:
    """Query AniList to get format for anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return media.get("format", None), al_cache
