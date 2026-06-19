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


def _post_with_retry(query, variables):
    """POST a GraphQL query to AniList, retrying politely on rate-limits / 5xx

    On a rate-limit (HTTP 429) or a transient 5xx, AniList returns
    ``{"data": null, ...}``. Returning that unretried is what surfaced
    downstream as ``'NoneType' object has no attribute 'get'`` when a run made
    many requests in quick succession, so we wait (honouring Retry-After when
    present) and retry before giving up. Returns the parsed JSON - which may
    still be an error payload after the final attempt - or ``{}`` if the
    response body wasn't JSON.
    """

    for attempt in range(MAX_RETRIES + 1):

        try:
            resp = requests.post(
                API_URL, json={"query": query, "variables": variables}
            )
        except requests.RequestException:
            # Network blip: back off and retry, then give up with an empty result
            if attempt >= MAX_RETRIES:
                return {}
            time.sleep(min(2 ** attempt, MAX_BACKOFF))
            continue

        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            # Prefer the server's Retry-After (seconds); otherwise exponential
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                wait = 2 ** attempt
            time.sleep(min(max(wait, 1), MAX_BACKOFF))
            continue

        try:
            return resp.json()
        except ValueError:
            return {}

    return {}


def get_query(al_id):
    """Fetch one AniList Media by id (see _post_with_retry for the retry policy)

    Args:
        al_id (int): Anilist ID
    """

    return _post_with_retry(QUERY, {"id": al_id})


def get_query_batch(al_ids):
    """Fetch up to ANILIST_BATCH_SIZE AniList Media in a single request via id_in

    Returns ``{id: {"data": {"Media": {...}}}}`` mirroring the single-id shape,
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
    al_id,
    al_cache,
):
    """Fetch the AniList Media object for an ID, caching successful lookups

    Centralises the cache lookup and the null-safe extraction the public helpers
    share. AniList returns ``{"data": {"Media": null}}`` (or ``{"data": null}``)
    for an unknown ID or a rate-limit, so every level is guarded with ``or {}``
    and a miss yields an empty dict rather than raising
    ``'NoneType' object has no attribute 'get'``. A miss is deliberately not
    cached, so a transient rate-limit isn't remembered as a permanent "unknown"
    for the rest of the run - the next call gets a fresh chance.

    Args:
        al_id (int): Anilist ID
        al_cache (dict | None): Cache of prior AniList responses, keyed by ID

    Returns:
        tuple: (media_dict, al_cache). media_dict is {} on a miss.
    """

    if al_cache is None:
        al_cache = {}

    # Try and find query in cache
    j = al_cache.get(al_id, None)

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
    al_id,
    al_cache=None,
):
    """Query AniList to get number of episodes for an anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return media.get("episodes", None), al_cache


def get_anilist_title(
    al_id,
    al_cache=None,
):
    """Query AniList to get title for an anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    # Prefer the english title, but fall back to romaji
    title = media.get("title") or {}

    return (title.get("english") or title.get("romaji")), al_cache


def get_anilist_thumb(
    al_id,
    al_cache=None,
):
    """Query AniList to get thumbnail URL for an anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return (media.get("coverImage") or {}).get("large", None), al_cache


def get_anilist_format(
    al_id,
    al_cache=None,
):
    """Query AniList to get format for an anime.

    Args:
        al_id (int): Anilist ID
        al_cache (dict): Cached Anilist requests. Defaults to None,
            which will create a dictionary
    """

    media, al_cache = _get_media(al_id, al_cache)

    return media.get("format", None), al_cache
