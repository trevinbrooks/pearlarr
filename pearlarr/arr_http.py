"""Shared raw-endpoint HTTP for the arr clients.

`ArrHttp` is the httpx-native transport every raw arr endpoint rides:
one bound helper per client holding the request/retry/parse/fail-open
boilerplate each raw arr endpoint would otherwise repeat (plus the strict,
fail-closed library fetch and its typed errors, and the shared history read
both arr clients delegate to).
"""

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

import httpx

from .config import strip_userinfo
from .log import LOG_NAME
from .output import hub_note, hub_warn
from .seadex_types import ARR_REQUEST_TIMEOUT_S, HistoryRecord, Json, validate_each

# Transient statuses worth another try on an idempotent GET - shared with the
# web client's `get_with_retries` so the two stacks retry the same set.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
GET_RETRIES = 3
BACKOFF_BASE_S = 0.5
# Streak re-warn cadence default. The composition root binds the configured
# `imports.digest_interval` (same default) at `bind`.
HEARTBEAT_INTERVAL_S = 300.0

# Coalesced-repeat breadcrumbs ride the stdlib channel (first-party child of
# the app logger, so the bridge adopts them into the file sink at DEBUG).
_LOG = logging.getLogger(f"{LOG_NAME}.arr_http")


def _streak_age(seconds: float) -> str:
    """Short streak age for coalesced lines: `"45s"` / `"12m"` / `"1h05m"`."""

    total = max(0, int(seconds))
    if total >= 3600:
        hours, minutes = divmod(total // 60, 60)
        return f"{hours}h{minutes:02d}m"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


@dataclass
class _Streak:
    """One (template, detail) key's consecutive-failure run (guard: `FailureStreaks.lock`)."""

    count: int
    first_at: float
    last_warned_at: float


@dataclass
class FailureStreaks:
    """The consecutive-failure ledger an arr's transport handles share.

    Mutable on purpose: `ArrHttp` stays frozen while `_fail`/`_recover`
    accumulate streak state here, and `dataclasses.replace` handles (the
    no-retry poll handle) share the one object so a streak spans both -
    separate ledgers would double-fire heartbeats and recovery notes.
    `lock` guards `active`: the shared client pool serves concurrent sweep
    threads (`SONARR_FETCH_WORKERS`).
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    active: dict[tuple[str, str], _Streak] = field(default_factory=dict[tuple[str, str], _Streak])


class ArrConnectionError(Exception):
    """An arr's load-bearing library fetch could not produce a result.

    Raised (instead of failing open) by `ArrHttp.get_json_list_strict`:
    the library list is a run's ground truth, so an unreachable arr / non-200 /
    non-JSON body / wrong payload shape aborts the leg with this clean,
    user-facing message (the CLI containment arm renders it without a
    traceback) rather than reading as an empty library.
    """


class ArrAuthError(Exception):
    """An arr rejected the API key (401/403) on the load-bearing library fetch.

    A distinct error from `ArrConnectionError` so the CLI containment arm can
    point the operator at `<arr>.api_key` specifically.
    """


def make_httpx_client(*, verify: bool = True) -> httpx.Client:
    """The pinned httpx client the arr transports share (one per run).

    - `follow_redirects=False` (httpx's default, pinned here on purpose): the
      `X-Api-Key` header must never ride a cross-host redirect, so a 3xx from
      an arr (a reverse-proxy login bounce) surfaces as a non-200 miss instead
      of silently replaying credentials elsewhere.
    - Client-level timeout mirroring `ARR_REQUEST_TIMEOUT_S`, so no call site
      can forget one.
    - Pool sized to the episode sweep's fetch concurrency
      (`SONARR_FETCH_WORKERS`), so parallel GETs don't queue.
    """

    connect_s, read_s = ARR_REQUEST_TIMEOUT_S
    return httpx.Client(
        timeout=httpx.Timeout(connect=connect_s, read=read_s, write=read_s, pool=connect_s),
        follow_redirects=False,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        verify=verify,
    )


@dataclass(frozen=True)
class ArrHttp:
    """One arr's bound HTTP surface: base url + auth header + fail-open policy.

    Frozen: `bind` is the single write point, so the auth header and base url
    can't be reassigned on a shared instance post-bind.

    Owns the boilerplate every raw endpoint would otherwise repeat: the auth header rides
    each request (the client is shared across arrs, so never on the client),
    transient GET failures retry with jittered backoff (a POST is not
    idempotent, so `post_json` never retries), EVERY body is parsed
    behind a JSON guard (a 200 HTML proxy page reads as a miss, never an
    abort), and each failure warns once through the caller's `warn` template
    with the failure detail filled in - consecutive identical failures
    coalesce (`_fail`/`_recover`): the first warns, repeats drop to DEBUG,
    a "still failing" re-warn fires every `heartbeat_s`, and the first
    success after a streak notes the recovery. The one fail-CLOSED read is
    `get_json_list_strict`, which raises typed errors for the
    load-bearing library fetch.
    """

    client: httpx.Client
    base_url: str
    """The arr base URL, no trailing slash (a "//api" join redirects to the login page)."""
    label: str
    """The arr's name ("Sonarr" / "Radarr"), for warning messages."""
    headers: Mapping[str, str]
    sleep: Callable[[float], None] = time.sleep  # injectable backoff sleep (bypassed in tests)
    retries: int = GET_RETRIES
    """In-call GET retry budget. The wait-path poll handle rides `replace(http,
    retries=0)` - its monitor loop IS the retry mechanism, so in-call backoff
    would only stretch each poll."""
    heartbeat_s: float = HEARTBEAT_INTERVAL_S
    """Seconds between "still failing" re-warns while a streak runs (bound from
    `imports.digest_interval` at the composition root - ArrHttp knows no config)."""
    clock: Callable[[], float] = time.monotonic  # injectable streak clock (faked in tests)
    streaks: FailureStreaks = field(default_factory=FailureStreaks, compare=False)
    """The failure-streak ledger. MUST stay `init=True`: `dataclasses.replace`
    re-runs `default_factory` for init=False fields, which would silently
    un-share the ledger between the primary and no-retry handles.
    `compare=False` keeps accumulated state out of `__eq__`."""

    @property
    def display_url(self) -> str:
        """The base URL as messages may show it: any embedded login masked.

        Requests ride `base_url` verbatim (a `user:pass@` login there is
        real basic auth, e.g. an arr behind a protected reverse proxy). Every
        error/warning string uses this instead - the login is a credential
        under the redaction guarantee.
        """

        return strip_userinfo(self.base_url)

    @classmethod
    def bind(
        cls,
        *,
        client: httpx.Client,
        url: str,
        api_key: str,
        label: str,
        sleep: Callable[[float], None] = time.sleep,
        heartbeat_s: float = HEARTBEAT_INTERVAL_S,
    ) -> "ArrHttp":
        """Bind the shared client to one arr's url + key.

        The key becomes the `X-Api-Key` header (never a query param, so it
        can't leak through URLs in logs/exceptions). `heartbeat_s` is the
        streak re-warn cadence, wired from config by the composition root.
        """

        return cls(
            client=client,
            base_url=url.rstrip("/"),
            label=label,
            headers={"X-Api-Key": api_key},
            sleep=sleep,
            heartbeat_s=heartbeat_s,
        )

    def _get_with_retries(
        self,
        path: str,
        params: Mapping[str, str] | None,
        *,
        timeout: float | None = None,
    ) -> tuple[httpx.Response | None, str]:
        """The retrying GET core the fail-open and strict paths share.

        Retries transient failures (connect/read errors, 429/5xx) up to
        `retries` times (the bind-time budget - the wait-path poll handle rides
        0) with jittered exponential backoff - GETs only, so
        this stays safe for idempotent reads. `timeout` overrides the
        client-level timeout per request (the 120s manual-import scans).
        None rides the client default. Returns the final response (a 200, or
        the terminal / retry-exhausted non-200) paired with a detail naming
        the failure, or `(None, detail)` when no request completed.
        """

        request_timeout = httpx.USE_CLIENT_DEFAULT if timeout is None else httpx.Timeout(timeout)
        response: httpx.Response | None = None
        detail = "request failed"
        for attempt in range(self.retries + 1):
            try:
                response = self.client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self.headers,
                    timeout=request_timeout,
                )
            except (httpx.HTTPError, httpx.InvalidURL) as e:
                # InvalidURL is NOT an HTTPError subclass. A config URL weird
                # enough to fail httpx's parser must still fail open, not abort.
                # Naming the failure type: "ConnectError" beats a bare "failed".
                detail = f"request failed ({type(e).__name__})"
                response = None
            else:
                if response.status_code == 200:
                    return response, detail
                detail = f"status code {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    return response, detail
            if attempt < self.retries:
                self.sleep(BACKOFF_BASE_S * 2**attempt + random.uniform(0, 0.25))
        return response, detail

    def _fetch_json(
        self,
        path: str,
        params: Mapping[str, str] | None,
        *,
        warn: str | None,
        timeout: float | None,
    ) -> object | None:
        """GET + JSON-parse core: the failure side of the streak ledger only.

        The public reads layer their shape checks and the `_recover` success
        hook on top - recovery must fire only when the WHOLE read succeeds, or
        a persistent wrong-shape response would flap recovery/warn each call.
        """

        response, detail = self._get_with_retries(path, params, timeout=timeout)
        if response is None or response.status_code != 200:
            return self._fail(warn, detail)
        try:
            return cast("object", response.json())
        except ValueError:
            return self._fail(warn, "non-JSON body")

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> object | None:
        """GET `path` and parse the JSON body, failing open to None with one warning.

        Rides `_get_with_retries` (transient failures retry with jittered
        backoff, up to the handle's `retries`). Any terminal failure (request
        error, non-200, non-JSON body)
        warns via `warn` - a template whose `{detail}` names the cause - and
        returns None. `warn=None` keeps a deliberate quiet path silent.
        `timeout` overrides the client-level timeout per request (seconds).
        None rides the client default.
        """

        payload = self._fetch_json(path, params, warn=warn, timeout=timeout)
        if payload is None:
            return None
        self._recover(warn)
        return payload

    def get_json_list(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> list[object] | None:
        """`get_json` narrowed to a JSON array (fails open on any other shape)."""

        payload = self._fetch_json(path, params, warn=warn, timeout=timeout)
        if payload is None:
            return None
        if not isinstance(payload, list):
            return self._fail(warn, "unexpected payload")
        self._recover(warn)
        return cast("list[object]", payload)

    def get_json_dict(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> dict[str, object] | None:
        """`get_json` narrowed to a JSON object (fails open on any other shape)."""

        payload = self._fetch_json(path, params, warn=warn, timeout=timeout)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return self._fail(warn, "unexpected payload")
        self._recover(warn)
        return cast("dict[str, object]", payload)

    def post_json(
        self,
        path: str,
        *,
        json: Json,
        warn: str | None,
    ) -> object | None:
        """POST `json` to `path` and parse the body, failing open to None with one warning.

        ONE attempt, never retried: a POST is not idempotent, so a retry could
        double-queue a command. Both 200 and 201 read as success. Any failure
        (request error, other status, non-JSON body) warns via `warn` - the
        same `{detail}` template as `get_json`, None for a deliberate quiet
        path - and returns None.
        """

        try:
            response = self.client.post(f"{self.base_url}{path}", json=json, headers=self.headers)
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            return self._fail(warn, f"request failed ({type(e).__name__})")
        if response.status_code not in (200, 201):
            return self._fail(warn, f"status code {response.status_code}")
        try:
            payload = cast("object", response.json())
        except ValueError:
            return self._fail(warn, "non-JSON body")
        self._recover(warn)
        return payload

    def get_json_list_strict(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> list[object]:
        """GET `path` and parse the JSON array body - RAISE instead of failing open.

        The strict counterpart of `get_json_list`, for the load-bearing
        library fetch where a failure must abort the leg rather than read as an
        empty library. Shares the retry core (transient statuses still retry
        first), then turns any terminal failure into a typed error: a 401/403
        raises `ArrAuthError`. A transport error, any other non-200, a
        non-JSON body or a non-array payload raises
        `ArrConnectionError` - both with a clean message naming this
        arr's base url and the failure detail.
        """

        response, detail = self._get_with_retries(path, params)
        if response is not None and response.status_code in (401, 403):
            raise ArrAuthError(f"{self.label} at {self.display_url} rejected the API key ({detail})")
        if response is None or response.status_code != 200:
            raise self._strict_error(detail)
        try:
            payload = cast("object", response.json())
        except ValueError:
            raise self._strict_error("non-JSON body")
        if not isinstance(payload, list):
            raise self._strict_error("unexpected payload")
        return cast("list[object]", payload)

    def history_since(
        self,
        date: str,
        *,
        include_flags: Mapping[str, str],
    ) -> list[HistoryRecord] | None:
        """History records since `date` (`/api/v3/history/since`, ascending), or None.

        One unfiltered call (`eventType` is single-valued server-side - the
        activity scan filters client-side), shared by both arr clients - the
        record's `seriesId`/`movieId` fold into one `item_id` at the
        model. Fails open to None through `get_json_list`'s matrix with a
        warning that states the consequence too: this read only feeds activity
        detection, and the caller (`ArrActivityMonitor.scan`) doesn't re-warn.

        Args:
            date: ISO8601 lower bound (arr-clock, inclusive).
            include_flags: The arr's include-* query params.
        """

        raw = self.get_json_list(
            "/api/v3/history/since",
            params={"date": date, **include_flags},
            warn=f"Could not fetch {self.label} history ({{detail}}) - skipping activity detection this run",
        )
        if raw is None:
            return None

        # Every field folds junk independently, so only a non-object stray skips.
        return validate_each(HistoryRecord, raw)

    def _strict_error(self, detail: str) -> ArrConnectionError:
        """The strict path's uniform could-not-reach error, naming url + cause."""

        return ArrConnectionError(f"Could not reach {self.label} at {self.display_url} ({detail})")

    def _fail(self, warn: str | None, detail: str) -> None:
        """The single fail-open tail: warn (when wanted) and return None.

        Literal replace, NOT str.format: templates will embed filenames/titles
        (which can carry braces), and the fail-open path must never crash.

        Consecutive identical failures coalesce per (template, detail): the
        first warns as before, repeats drop to a DEBUG breadcrumb with the
        running count, and a "still failing" re-warn fires every
        `heartbeat_s`. `_recover` closes the streak on the next success.
        """

        if warn is None:
            return None
        now = self.clock()
        with self.streaks.lock:
            streak = self.streaks.active.get((warn, detail))
            if streak is None:
                self.streaks.active[(warn, detail)] = _Streak(count=1, first_at=now, last_warned_at=now)
                count, age, heartbeat_due = 1, 0.0, False
            else:
                streak.count += 1
                count = streak.count
                age = now - streak.first_at
                heartbeat_due = now - streak.last_warned_at >= self.heartbeat_s
                if heartbeat_due:
                    streak.last_warned_at = now
        # Emit OUTSIDE the lock: hub/render work must not serialize sweep threads.
        if count == 1:
            hub_warn(warn.replace("{detail}", detail))
        elif heartbeat_due:
            hub_warn(warn.replace("{detail}", f"{detail}; still failing - attempt {count}, {_streak_age(age)}"))
        else:
            _LOG.debug(f"{warn.replace('{detail}', detail)} - failure {count} in a row")
        return None

    def _recover(self, warn: str | None) -> None:
        """Close any failure streaks under `warn` on the first success after them.

        The success side of `_fail`'s coalescing, called from each public
        read's success tail. Keyed by template alone so a streak whose detail
        shifted mid-outage (500 -> ConnectError) still closes as ONE note
        carrying the combined total.
        """

        if warn is None:
            return
        now = self.clock()
        with self.streaks.lock:
            ended = [self.streaks.active.pop(key) for key in list(self.streaks.active) if key[0] == warn]
        if not ended:
            return
        total = sum(streak.count for streak in ended)
        age = _streak_age(now - min(streak.first_at for streak in ended))
        noun = "failure" if total == 1 else "failures"
        hub_note(f"{warn.partition(' ({detail})')[0]} - recovered after {total} {noun} ({age})")
