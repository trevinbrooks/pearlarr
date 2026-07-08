"""AniList GraphQL wire layer: the bound client, retry policy, and parse helpers."""

import contextlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from .json_narrow import is_json_list, is_json_obj
from .seadex_types import AniListError, AniListMediaNode, validation_summary

API_URL = "https://graphql.anilist.co"

type AniListCache = dict[int, dict[str, dict[str, Any]]]
"""In-memory AniList cache: id -> raw GraphQL body ``{"data": {"Media": {...}}}``.

The cached value is the *whole* response body (what :meth:`AniListClient.query`
/ :meth:`AniListClient.query_batch` return), so it round-trips verbatim through
the persisted ``anilist_meta`` block. The gateway extracts and parses the
``Media`` node out of it into an :class:`AniListMediaNode`.
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


@dataclass
class AniListRetryLog:
    """Voices ``_post_with_retry``'s waits and give-ups through the run logger.

    Without it a rate-limit backoff sleeps up to 60s with zero output (the run
    looks hung) and a final give-up returns ``{}`` silently. One instance per
    :class:`AniListClient` - which is built once per arr run - so the give-up
    warning fires once per run rather than once per title.
    """

    logger: logging.Logger
    _gave_up: bool = field(default=False, init=False)

    def waiting(self, reason: str, wait: float, retry: int, *, level: int = logging.INFO) -> None:
        """One backoff notice, so a long Retry-After wait doesn't look like a hang."""

        self.logger.log(level, f"AniList {reason}; waiting {wait:.0f}s (retry {retry}/{MAX_RETRIES})")

    def gave_up(self) -> None:
        """Warn ONCE per run that AniList lookups are degraded, then stay quiet."""

        if not self._gave_up:
            self.logger.warning(
                f"AniList request failed after {MAX_RETRIES} retries; "
                "some titles/episode counts may be missing this run",
            )
        self._gave_up = True


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
    if not is_json_list(raw_errors):
        return []
    # Validate each GraphQL error entry into the typed AniListError, dropping
    # the junk ones (a soft-throttle or malformed body can carry non-dict junk).
    errors: list[AniListError] = []
    for err in raw_errors:
        try:
            errors.append(AniListError.model_validate(err))
        except ValidationError:
            continue
    return errors


def extract_path(body: dict[str, Any] | None, *path: str) -> dict[str, Any]:
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


def media_node_from(raw: dict[str, Any], *, logger: logging.Logger | None = None) -> AniListMediaNode:
    """Validate a raw ``Media`` dict into the typed node (single-object fail-open).

    A miss (``{}``) validates to the all-``None`` node; a malformed node
    degrades to the same all-``None`` miss with one scrubbed warning (when a
    logger is provided - the production callers always pass one).

    Args:
        raw (dict[str, Any]): The raw ``Media`` dict (``{}`` on a miss).
        logger (logging.Logger | None): Voices the malformed-node warning.
    """

    try:
        return AniListMediaNode.model_validate(raw)
    except ValidationError as e:
        if logger is not None:
            logger.warning(f"Ignoring malformed AniList Media node ({validation_summary(e)})")
        return AniListMediaNode()


def media_from(body: dict[str, Any] | None, *, logger: logging.Logger | None = None) -> AniListMediaNode:
    """Parse the Media node from a single-id body into an AniListMediaNode

    The raw ``{"data": {"Media": {...}}}`` body is the dynamic GraphQL boundary;
    this is where it crosses into the typed domain. A miss (``data``/``Media``
    null) yields an all-``None`` node; so does a malformed node (see
    :func:`media_node_from`).

    Args:
        body (dict[str, Any] | None): The parsed JSON response body
        logger (logging.Logger | None): Voices the malformed-node warning.
    """

    return media_node_from(extract_path(body, "data", "Media"), logger=logger)


class AniListClient:
    """AniList GraphQL wire client: the POST + retry policy, bound once.

    The AniList analog of :class:`~.arr_http.ArrHttp`: the shared web client
    and the per-run retry narration are bound at construction, so callers ask
    for bodies by id instead of threading ``(client, retry_log)`` through
    every call. Construction is network-free. The gateway layers the run cache
    on top; this class is deliberately cache-blind.
    """

    def __init__(self, *, client: httpx.Client, logger: logging.Logger) -> None:
        """Bind the wire client to the shared web client and run logger.

        Args:
            client (httpx.Client): The shared web client every POST rides (its
                defaults carry the identifying User-Agent and timeout bounds).
            logger (logging.Logger): Voices the retry waits / give-ups via the
                bound :class:`AniListRetryLog` (one give-up warning per run).
        """

        self._client = client
        self._retry_log = AniListRetryLog(logger=logger)

    def query(self, al_id: int) -> dict[str, Any]:
        """Fetch one AniList Media by id (see _post_with_retry for the retry policy)

        Args:
            al_id (int): Anilist ID
        """

        return self._post_with_retry(QUERY, {"id": al_id})

    def query_batch(self, al_ids: list[int]) -> AniListCache:
        """Fetch up to ANILIST_BATCH_SIZE AniList Media in a single request via id_in

        Returns "{id: {"data": {"Media": {...}}}}" mirroring the single-id shape,
        so the results can seed the same cache directly. Ids unknown to AniList are
        simply absent from the result.

        Args:
            al_ids (list[int]): Up to ANILIST_BATCH_SIZE AniList IDs
        """

        j = self._post_with_retry(BATCH_QUERY, {"ids": list(al_ids)})
        # The cache stores each Media body verbatim (re-parsed on read into an
        # AniListMediaNode). Keep each Media OBJECT carrying an int id, skipping
        # junk entries in the array.
        out: AniListCache = {}
        media_list = extract_path(j, "data", "Page").get("media")
        for raw in media_list if is_json_list(media_list) else []:
            if not is_json_obj(raw):
                continue
            media_id = raw.get("id")
            if isinstance(media_id, int):
                out[media_id] = {"data": {"Media": raw}}
        return out

    def _post_with_retry(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
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
        wasn't JSON. The bound retry log narrates the waits and warns once per
        run on a final give-up. The bound client is the shared web client (its
        defaults carry the identifying User-Agent and the timeout bounds); this
        loop stays AniList's ONE retry policy - the web client's generic GET
        helper is never involved.
        """

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.post(API_URL, json={"query": query, "variables": variables})
            except httpx.HTTPError as e:
                # Network blip: back off and retry, then give up with an empty result
                if attempt >= MAX_RETRIES:
                    self._retry_log.gave_up()
                    return {}
                wait = min(2**attempt + random.uniform(0, 1), MAX_BACKOFF)
                self._retry_log.waiting(
                    f"request failed ({type(e).__name__})",
                    wait,
                    attempt + 1,
                    level=logging.DEBUG,
                )
                time.sleep(wait)
                continue

            retryable = resp.status_code in RETRYABLE_STATUS

            # Parse the body so a soft-throttle (HTTP 200 + throttle error payload)
            # can take the same retry path as a 429 status. A non-JSON body - or a
            # JSON body that isn't an object (e.g. an array) - folds to None here
            # and returns as {} below, the callers' no-data arm.
            raw_body: object
            try:
                raw_body = resp.json()
            except ValueError:
                raw_body = None
            body: dict[str, Any] | None = raw_body if is_json_obj(raw_body) else None

            if not retryable and body is not None and _errors_are_retryable(body):
                retryable = True

            if retryable and attempt < MAX_RETRIES:
                # Prefer the server's Retry-After (seconds, honored exactly);
                # otherwise exponential, jittered so concurrent clients that hit
                # the same limit window don't retry in lockstep.
                retry_after = resp.headers.get("Retry-After")
                wait = 2**attempt + random.uniform(0, 1)
                if retry_after is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        wait = float(retry_after)

                wait = min(max(wait, 1), MAX_BACKOFF)
                # A 429 / soft-throttle reads as a rate limit; a 5xx names itself.
                reason = (
                    f"returned HTTP {resp.status_code}"
                    if resp.status_code in RETRYABLE_STATUS and resp.status_code != 429
                    else "rate-limited"
                )
                self._retry_log.waiting(reason, wait, attempt + 1)
                time.sleep(wait)
                continue

            # Final attempt (or a non-retryable response): return the parsed body,
            # or {} when the body wasn't JSON, so the caller degrades gracefully.
            # Running out of attempts on a retryable response is a give-up too.
            if retryable:
                self._retry_log.gave_up()
            return body if body is not None else {}

        return {}
