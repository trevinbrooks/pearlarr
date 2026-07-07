"""Shared raw-endpoint HTTP for the arr clients.

:class:`ArrHttp` is the httpx-native transport every raw arr endpoint rides:
one bound helper per client holding the request/retry/parse/fail-open
boilerplate that used to be copied per endpoint (plus the strict, fail-closed
library fetch and its typed errors, and the shared history read both arr
clients delegate to).
"""

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import httpx

from .seadex_types import ARR_REQUEST_TIMEOUT_S, HistoryRecord, Json

# Transient statuses worth another try on an idempotent GET - the same set the
# urllib3 Retry policy on the (now torrents-only) requests session uses.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
GET_RETRIES = 3
BACKOFF_BASE_S = 0.5


class ArrConnectionError(Exception):
    """An arr's load-bearing library fetch could not produce a result.

    Raised (instead of failing open) by :meth:`ArrHttp.get_json_list_strict`:
    the library list is a run's ground truth, so an unreachable arr / non-200 /
    non-JSON body / wrong payload shape aborts the leg with this clean,
    user-facing message (the CLI containment arm renders it without a
    traceback) rather than reading as an empty library.
    """


class ArrAuthError(Exception):
    """An arr rejected the API key (401/403) on the load-bearing library fetch.

    Split from :class:`ArrConnectionError` so the CLI containment arm can point
    the user at ``<arr>.api_key`` specifically instead of the url.
    """


def make_httpx_client(*, verify: bool = True) -> httpx.Client:
    """The pinned httpx client the arr transports share (one per run).

    - ``follow_redirects=False`` (httpx's default, pinned here on purpose): the
      ``X-Api-Key`` header must never ride a cross-host redirect, so a 3xx from
      an arr (a reverse-proxy login bounce) surfaces as a non-200 miss instead
      of silently replaying credentials elsewhere.
    - Client-level timeout mirroring ``ARR_REQUEST_TIMEOUT_S``, so no call site
      can forget one.
    - Pool sized to the episode sweep's fetch concurrency
      (``SONARR_FETCH_WORKERS``), so parallel GETs don't queue.
    """

    connect_s, read_s = ARR_REQUEST_TIMEOUT_S
    return httpx.Client(
        timeout=httpx.Timeout(connect=connect_s, read=read_s, write=read_s, pool=connect_s),
        follow_redirects=False,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        verify=verify,
    )


@dataclass
class ArrHttp:
    """One arr's bound HTTP surface: base url + auth header + fail-open policy.

    Owns the boilerplate every raw endpoint used to copy: the auth header rides
    each request (the client is shared across arrs, so never on the client),
    transient GET failures retry with jittered backoff (a POST is not
    idempotent, so :meth:`post_json` never retries), EVERY body is parsed
    behind a JSON guard (a 200 HTML proxy page reads as a miss, never an
    abort), and each failure warns once through the caller's ``warn`` template
    with the failure detail filled in. The one fail-CLOSED read is
    :meth:`get_json_list_strict`, which raises typed errors for the
    load-bearing library fetch.
    """

    client: httpx.Client
    base_url: str  # no trailing slash (a "//api" join redirects to the login page)
    label: str  # "Sonarr" / "Radarr", for warnings
    logger: logging.Logger
    headers: Mapping[str, str]
    sleep: Callable[[float], None] = time.sleep  # injectable so tests don't wait out backoffs

    @classmethod
    def bind(
        cls,
        *,
        client: httpx.Client,
        url: str,
        api_key: str,
        label: str,
        logger: logging.Logger,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "ArrHttp":
        """Bind the shared client to one arr's url + key.

        The key becomes the ``X-Api-Key`` header (never a query param, so it
        can't leak through URLs in logs/exceptions).
        """

        return cls(
            client=client,
            base_url=url.rstrip("/"),
            label=label,
            logger=logger,
            headers={"X-Api-Key": api_key},
            sleep=sleep,
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
        ``GET_RETRIES`` times with jittered exponential backoff - GETs only, so
        this stays safe for idempotent reads. ``timeout`` overrides the
        client-level timeout per request (the 120s manual-import scans);
        None rides the client default. Returns the final response (a 200, or
        the terminal / retry-exhausted non-200) paired with a detail naming
        the failure, or ``(None, detail)`` when no request completed.
        """

        request_timeout = httpx.USE_CLIENT_DEFAULT if timeout is None else httpx.Timeout(timeout)
        response: httpx.Response | None = None
        detail = "request failed"
        for attempt in range(GET_RETRIES + 1):
            try:
                response = self.client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self.headers,
                    timeout=request_timeout,
                )
            except (httpx.HTTPError, httpx.InvalidURL) as e:
                # InvalidURL is NOT an HTTPError subclass; a config URL weird
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
            if attempt < GET_RETRIES:
                self.sleep(BACKOFF_BASE_S * 2**attempt + random.uniform(0, 0.25))
        return response, detail

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> object | None:
        """GET ``path`` and parse the JSON body; fail open to None with one warning.

        Rides :meth:`_get_with_retries` (transient failures retry with jittered
        backoff). Any terminal failure (request error, non-200, non-JSON body)
        warns via ``warn`` - a template whose ``{detail}`` names the cause - and
        returns None; ``warn=None`` keeps a deliberate quiet path silent.

        Args:
            path (str): Endpoint path (e.g. ``"/api/v3/queue"``).
            params (Mapping[str, str] | None): Query params. Defaults to None.
            warn (str | None): Warning template with a ``{detail}`` placeholder,
                or None to fail open silently.
            timeout (float | None): Per-request timeout override (seconds).
                Defaults to None (the client-level timeout).
        """

        response, detail = self._get_with_retries(path, params, timeout=timeout)
        if response is None or response.status_code != 200:
            return self._fail(warn, detail)
        try:
            return cast("object", response.json())
        except ValueError:
            return self._fail(warn, "non-JSON body")

    def get_json_list(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> list[object] | None:
        """:meth:`get_json` narrowed to a JSON array (fails open on any other shape)."""

        payload = self.get_json(path, params=params, warn=warn, timeout=timeout)
        if payload is None:
            return None
        if not isinstance(payload, list):
            return self._fail(warn, "unexpected payload")
        return cast("list[object]", payload)

    def get_json_dict(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        warn: str | None,
        timeout: float | None = None,
    ) -> dict[str, object] | None:
        """:meth:`get_json` narrowed to a JSON object (fails open on any other shape)."""

        payload = self.get_json(path, params=params, warn=warn, timeout=timeout)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return self._fail(warn, "unexpected payload")
        return cast("dict[str, object]", payload)

    def post_json(
        self,
        path: str,
        *,
        json: Json,
        warn: str | None,
    ) -> object | None:
        """POST ``json`` to ``path`` and parse the body; fail open to None with one warning.

        ONE attempt, never retried: a POST is not idempotent, so a retry could
        double-queue a command. Both 200 and 201 read as success; any failure
        (request error, other status, non-JSON body) warns via ``warn`` - the
        same ``{detail}`` template as :meth:`get_json` - and returns None.

        Args:
            path (str): Endpoint path (e.g. ``"/api/v3/command"``).
            json (Json): The JSON request body.
            warn (str | None): Warning template with a ``{detail}`` placeholder,
                or None to fail open silently.
        """

        try:
            response = self.client.post(f"{self.base_url}{path}", json=json, headers=self.headers)
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            return self._fail(warn, f"request failed ({type(e).__name__})")
        if response.status_code not in (200, 201):
            return self._fail(warn, f"status code {response.status_code}")
        try:
            return cast("object", response.json())
        except ValueError:
            return self._fail(warn, "non-JSON body")

    def get_json_list_strict(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> list[object]:
        """GET ``path`` and parse the JSON array body; RAISE instead of failing open.

        The strict counterpart of :meth:`get_json_list`, for the load-bearing
        library fetch where a failure must abort the leg rather than read as an
        empty library. Shares the retry core (transient statuses still retry
        first), then turns any terminal failure into a typed error: a 401/403
        raises :class:`ArrAuthError`; a transport error, any other non-200, a
        non-JSON body or a non-array payload raises
        :class:`ArrConnectionError` - both with a clean message naming this
        arr's base url and the failure detail.

        Args:
            path (str): Endpoint path (e.g. ``"/api/v3/series"``).
            params (Mapping[str, str] | None): Query params. Defaults to None.
        """

        response, detail = self._get_with_retries(path, params)
        if response is not None and response.status_code in (401, 403):
            raise ArrAuthError(f"{self.label} at {self.base_url} rejected the API key ({detail})")
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
        item_key: str,
    ) -> list[HistoryRecord] | None:
        """History records since ``date`` (``/api/v3/history/since``, ascending), or None.

        One unfiltered call (``eventType`` is single-valued server-side; the
        activity scan filters client-side), shared by both arr clients. Fails
        open to None through :meth:`get_json_list`'s matrix with a warning that
        states the consequence too: this read only feeds activity detection,
        and the caller (``ArrActivityMonitor.scan``) doesn't re-warn.

        Args:
            date (str): ISO8601 lower bound (arr-clock, inclusive).
            include_flags (Mapping[str, str]): The arr's include-* query params.
            item_key (str): The record's item-id field (``seriesId``/``movieId``).
        """

        raw = self.get_json_list(
            "/api/v3/history/since",
            params={"date": date, **include_flags},
            warn=f"Could not fetch {self.label} history ({{detail}}); skipping activity detection this run",
        )
        if raw is None:
            return None

        # Element dicts are unvalidated JSON: cast at the parse boundary, skip strays.
        return [
            HistoryRecord.from_api(cast("dict[str, Any]", record), item_key=item_key)
            for record in raw
            if isinstance(record, dict)
        ]

    def _strict_error(self, detail: str) -> ArrConnectionError:
        """The strict path's uniform could-not-reach error, naming url + cause."""

        return ArrConnectionError(f"Could not reach {self.label} at {self.base_url} ({detail})")

    def _fail(self, warn: str | None, detail: str) -> None:
        """The single fail-open tail: warn (when wanted) and return None.

        Literal replace, NOT str.format: templates will embed filenames/titles
        (which can carry braces), and the fail-open path must never crash.
        """

        if warn is not None:
            self.logger.warning(warn.replace("{detail}", detail))
        return None
