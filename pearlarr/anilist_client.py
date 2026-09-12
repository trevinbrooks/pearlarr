"""AniList GraphQL wire layer: the bound client, retry policy, and parse helpers."""

import contextlib
import random
import time
from typing import Any

import httpx
from pydantic import ValidationError

from .json_narrow import is_json_list, is_json_obj
from .output import Severity, hub_note, hub_warn
from .paths import PROJECT_URL
from .seadex_types import AniListError, AniListMediaNode, validation_summary

API_URL = "https://graphql.anilist.co"

# AniList refuses a request without a Referer (HTTP 403 "temporarily disabled",
# any non-empty value passes), so every POST names the project.
REQUEST_HEADERS = {"Referer": PROJECT_URL}

type AniListBody = dict[str, dict[str, Any]]
"""One raw GraphQL body `{"data": {"Media": {...}}}`, stored verbatim."""

type AniListCache = dict[int, AniListBody]
"""In-memory AniList cache: id -> raw GraphQL body `{"data": {"Media": {...}}}`.

The cached value is the *whole* response body (what `AniListClient.query`
/ `AniListClient.query_batch` return), so it round-trips verbatim through
the persisted `anilist_meta` block. The gateway extracts and parses the
`Media` node out of it into an `AniListMediaNode`.
"""

# AniList rate-limits (HTTP 429) and occasionally returns a transient 5xx. Retry
# those a few times with a backoff that respects a Retry-After header, so a busy
# run (many series in quick succession) waits out the limit instead of treating
# the throttled response as real data.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_BACKOFF = 60

# AniList's error message is quoted in the once-per-run breaker warning, clamped to this.
_REFUSAL_MESSAGE_MAX = 160


def _exp_backoff(attempt: int) -> float:
    # Exponential, jittered so concurrent clients hitting the same limit window don't retry in lockstep.
    return min(2**attempt + random.uniform(0, 1), MAX_BACKOFF)


# AniList also soft-throttles by returning HTTP 200 with a GraphQL error payload
# (`{"data": null, "errors": [{"message": "Too Many Requests", "status": 429}]}`)
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
    bannerImage
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
    f"""
query ($ids: [Int]) {{
  Page (perPage: {ANILIST_BATCH_SIZE}) {{
    media (id_in: $ids, type: ANIME) {{
"""
    + _MEDIA_FIELDS
    + """
    }
  }
}
"""
)


def _log_wait(reason: str, wait: float, retry: int, *, severity: Severity = Severity.INFO) -> None:
    """One backoff notice, so a long Retry-After wait doesn't look like a hang."""

    # A network-blip wait rides at DEBUG, a throttle wait at INFO.
    hub_note(f"AniList {reason} - waiting {wait:.0f}s (retry {retry}/{MAX_RETRIES})", severity=severity)


def _errors_are_retryable(body: dict[str, Any] | None) -> bool:
    """True if a GraphQL body carries a throttle/rate-limit or 5xx-style error.

    AniList soft-throttles with HTTP 200 and a 429 error entry, retried like a
    real 429. An unknown id answers HTTP 404 with `Media: null` under `data`
    and a not-found error, which stays an ordinary miss.
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


def _refusal(status: int, body: dict[str, Any] | None) -> str | None:
    """The breaker reason when AniList did not answer, else None. An unknown id keeps its `data` node."""

    if body is not None and body.get("data") is not None:
        return None
    errors = _parse_errors(body)
    message = " ".join(errors[0].message.split())[:_REFUSAL_MESSAGE_MAX] if errors else ""
    if message:
        return f"refused the request (HTTP {status}: {message})"
    return f"gave no usable answer (HTTP {status})"


def _parse_errors(body: dict[str, Any] | None) -> list[AniListError]:
    """Parse a GraphQL body's `errors` array into typed `AniListError`.

    The `errors` array is the dynamic GraphQL boundary. This maps each raw
    entry into the typed domain (skipping any non-object entry), so the caller
    reads `err.status` / `err.message` rather than untyped `dict` keys.
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
    """Walk a null-safe key path through a GraphQL body, yielding {} on any miss.

    AniList returns {"data": null} or {"data": {"Media": null}} for an unknown
    id or a rate-limit, so each hop is guarded with "or {}" and a missing or
    null level yields an empty dict rather than raising
    "'NoneType' object has no attribute 'get'". The `path` keys are walked in
    order, e.g. "data", "Media".
    """

    node: dict[str, Any] = body or {}
    for key in path:
        node = node.get(key) or {}
    return node


def media_node_from(raw: dict[str, Any]) -> AniListMediaNode:
    """Validate a raw `Media` dict into the typed node (single-object fail-open).

    A miss (`{}`) validates to the all-`None` node. A malformed node
    degrades to the same all-`None` miss with one scrubbed warning.
    """

    try:
        return AniListMediaNode.model_validate(raw)
    except ValidationError as e:
        hub_warn(f"Ignoring malformed AniList Media node ({validation_summary(e)})")
        return AniListMediaNode()


def media_from(body: dict[str, Any] | None) -> AniListMediaNode:
    """Parse the Media node from a single-id body into an AniListMediaNode.

    The raw `{"data": {"Media": {...}}}` body is the dynamic GraphQL boundary,
    and this is where it crosses into the typed domain. A miss (`data`/`Media`
    null) yields an all-`None` node, and so does a malformed node (see
    `media_node_from`).
    """

    return media_node_from(extract_path(body, "data", "Media"))


class AniListClient:
    """AniList GraphQL wire client: the POST + retry policy, bound once.

    Cache-blind (the gateway layers the run cache on top). A per-run breaker trips
    on retry exhaustion or a dataless answer, after which every call returns empty.
    """

    def __init__(self, *, client: httpx.Client) -> None:
        """Bind the wire client to the shared web client (network-free)."""

        self._client = client
        # Set once AniList gave no answer or exhausted its retries: every later call short-circuits this run.
        self._outage = False

    @property
    def outage(self) -> bool:
        """True once AniList has been declared unavailable for this run."""

        return self._outage

    def _note_outage(self, detail: str) -> None:
        """Warn ONCE that AniList is unavailable, muting every later call via the flag."""

        if not self._outage:
            hub_warn(f"AniList {detail}, skipping further lookups this run")
        self._outage = True

    def query(self, al_id: int) -> dict[str, Any]:
        """Fetch one AniList Media by id (see _post_with_retry for the retry policy)."""

        return self._post_with_retry(QUERY, {"id": al_id})

    def query_batch(self, al_ids: list[int]) -> AniListCache:
        """Fetch up to ANILIST_BATCH_SIZE AniList Media in a single request via id_in.

        Returns "{id: {"data": {"Media": {...}}}}" mirroring the single-id shape,
        so the results can seed the same cache directly. Ids unknown to AniList are
        simply absent from the result.
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
        """POST a GraphQL query, retrying politely on rate limits and 5xx.

        Returns the parsed body, or `{}` when it was not JSON or the breaker has
        tripped. Retry exhaustion and any dataless answer trip the breaker for the run.
        """

        if self._outage:
            return {}

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.post(
                    API_URL, json={"query": query, "variables": variables}, headers=REQUEST_HEADERS
                )
            except httpx.HTTPError as e:
                # Network blip: back off and retry, then trip the breaker with an empty result.
                if attempt >= MAX_RETRIES:
                    self._note_outage(f"request failed after {MAX_RETRIES} retries")
                    return {}
                wait = _exp_backoff(attempt)
                _log_wait(f"request failed ({type(e).__name__})", wait, attempt + 1, severity=Severity.DEBUG)
                time.sleep(wait)
                continue

            retryable = resp.status_code in RETRYABLE_STATUS

            # Parse the body so a soft-throttle (HTTP 200 + throttle error payload)
            # can take the same retry path as a 429 status. A non-JSON body, or a
            # JSON body that isn't an object (e.g. an array), folds to None here
            # and trips the breaker below.
            raw_body: object
            try:
                raw_body = resp.json()
            except ValueError:
                raw_body = None
            body: dict[str, Any] | None = raw_body if is_json_obj(raw_body) else None

            if not retryable and body is not None and _errors_are_retryable(body):
                retryable = True

            if retryable and attempt < MAX_RETRIES:
                # Prefer the server's Retry-After (seconds, honored exactly).
                # Otherwise fall back to the shared exponential backoff.
                retry_after = resp.headers.get("Retry-After")
                wait = _exp_backoff(attempt)
                if retry_after is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        wait = float(retry_after)

                wait = min(max(wait, 1), MAX_BACKOFF)
                # A 429 / soft-throttle reads as a rate limit. A 5xx names itself.
                reason = (
                    f"returned HTTP {resp.status_code}"
                    if resp.status_code in RETRYABLE_STATUS and resp.status_code != 429
                    else "rate-limited"
                )
                _log_wait(reason, wait, attempt + 1)
                time.sleep(wait)
                continue

            # Terminal: exhausted retries and any dataless answer trip the breaker. The
            # body (possibly an error payload) still returns so the caller degrades.
            if retryable:
                self._note_outage(f"request failed after {MAX_RETRIES} retries")
            elif (detail := _refusal(resp.status_code, body)) is not None:
                self._note_outage(detail)
            return body if body is not None else {}

        return {}
