"""SkyHook wire client: Sonarr's metadata service, read for each series' TVDB episode orders."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import override

import httpx
from pydantic import ValidationError

from .arr_http import RETRYABLE_STATUS
from .output import Severity, hub_note, hub_warn
from .seadex_types import SkyHookReply, validation_summary
from .web_client import failure_detail, get_with_retries

SKYHOOK_URL = "https://skyhook.sonarr.tv/v2/tvdb/shows/eng"


class SkyHookMiss(Enum):
    """Why a fetch has no reply."""

    OUTAGE = "outage"
    """SkyHook is unreachable this run."""

    UNKNOWN = "unknown"
    """SkyHook answered the request with a client error (4xx): it has no such series or order."""


class AbstractSkyHookClient(ABC):
    """The SkyHook seam the gateway depends on."""

    @property
    @abstractmethod
    def outage(self) -> bool:
        """True once SkyHook has been declared unreachable for this run."""

    @abstractmethod
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookMiss:
        """One series' reply for one order.

        A reply that fails validation or doesn't match the request comes back with no episodes. After an outage
        every call returns `SkyHookMiss.OUTAGE` without a request.
        """


class SkyHookClient(AbstractSkyHookClient):
    """SkyHook over the shared web client: GETs with the web retry policy and a per-run outage latch."""

    def __init__(self, *, client: httpx.Client, sleep: Callable[[float], None] = time.sleep) -> None:
        """Bind the shared web client (network-free). `sleep` is the retry backoff's."""

        self._client = client
        self._sleep = sleep
        self._outage = False

    @property
    @override
    def outage(self) -> bool:
        return self._outage

    def _note_outage(self, detail: str) -> None:
        """Warn ONCE that SkyHook is unreachable, muting every later call via the flag."""

        if not self._outage:
            hub_warn(f"SkyHook request failed ({detail}) - skipping TVDB order fetches this run")
        self._outage = True

    @override
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookMiss:
        if self._outage:
            return SkyHookMiss.OUTAGE
        try:
            response = get_with_retries(self._client, f"{SKYHOOK_URL}/{tvdb_id}/{order_type}", sleep=self._sleep)
        except httpx.RequestError as e:
            self._note_outage(failure_detail(e))
            return SkyHookMiss.OUTAGE
        status = response.status_code
        if status in RETRYABLE_STATUS or status >= 500:
            self._note_outage(f"HTTP {status}")
            return SkyHookMiss.OUTAGE
        if status >= 400:
            hub_note(f"SkyHook has no {order_type} order for TVDB {tvdb_id} (HTTP {status})", severity=Severity.DEBUG)
            return SkyHookMiss.UNKNOWN
        return _read_reply(response, tvdb_id, order_type)


def _read_reply(response: httpx.Response, tvdb_id: int, order_type: str) -> SkyHookReply:
    """The validated reply, or an empty one (warned) when it doesn't validate or doesn't match the request."""

    empty = SkyHookReply(tvdb_id=tvdb_id)
    try:
        reply = SkyHookReply.model_validate_json(response.content)
    except ValidationError as e:
        _warn_malformed(tvdb_id, order_type, validation_summary(e))
        return empty
    if reply.tvdb_id != tvdb_id or not reply.lists(order_type):
        _warn_malformed(tvdb_id, order_type, "the reply is for another series or order")
        return empty
    if reply.episodes is None:
        _warn_malformed(tvdb_id, order_type, "an episode row failed validation")
        return reply.model_copy(update={"episodes": ()})
    return reply


def _warn_malformed(tvdb_id: int, order_type: str, detail: str) -> None:
    """Warn that one order's reply can't be used, so the order is stored with no episodes."""

    hub_warn(f"SkyHook sent a malformed {order_type} order for TVDB {tvdb_id} ({detail}) - storing it empty")
