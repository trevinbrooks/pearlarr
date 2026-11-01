"""SkyHook wire client: Sonarr's metadata service, read for each series' TVDB episode orders."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import override

import httpx
from pydantic import ValidationError

from .output import Severity, hub_note, hub_warn
from .seadex_types import SkyHookError, SkyHookReply, validation_summary
from .web_client import failure_detail, get_with_retries

SKYHOOK_URL = "https://skyhook.sonarr.tv/v2/tvdb/shows/eng"


class SkyHookMiss(Enum):
    """Why a fetch has no reply."""

    OUTAGE = auto()
    """SkyHook is unreachable this run, or answered with a status that says nothing about the series."""

    NOT_FOUND = auto()
    """SkyHook says TVDB has no such series."""


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

    def _note_outage(self, detail: str) -> SkyHookMiss:
        """Warn that SkyHook failed and latch the outage, muting every later call this run."""

        # The latch also trips on one series SkyHook keeps failing, so `detail` names the request.
        hub_warn(f"SkyHook failed ({detail}) - re-checking cached titles until it answers")
        self._outage = True
        return SkyHookMiss.OUTAGE

    @override
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookMiss:
        if self._outage:
            return SkyHookMiss.OUTAGE
        asked = f"for TVDB {tvdb_id} {order_type}"
        try:
            response = get_with_retries(self._client, f"{SKYHOOK_URL}/{tvdb_id}/{order_type}", sleep=self._sleep)
        except httpx.RequestError as e:
            return self._note_outage(f"{failure_detail(e)} {asked}")
        if response.is_success:
            return _read_reply(response, tvdb_id, order_type)
        if _says_not_found(response):
            hub_note(f"SkyHook has no TVDB series {tvdb_id} ({order_type} order)", severity=Severity.DEBUG)
            return SkyHookMiss.NOT_FOUND
        # Any other status is about the host, not the series: an edge block (403), a moved route (a bare 404), a
        # rate limit. Read as a miss, it would store every series empty for a week.
        return self._note_outage(f"HTTP {response.status_code} {asked}")


def _says_not_found(response: httpx.Response) -> bool:
    """Whether the response is SkyHook's own 404 for a series TVDB doesn't have."""

    if response.status_code != 404:
        return False
    try:
        return SkyHookError.model_validate_json(response.content).not_found
    except ValidationError:
        return False


def _read_reply(response: httpx.Response, tvdb_id: int, order_type: str) -> SkyHookReply:
    """The validated reply, or an empty one (warned) when it doesn't validate or doesn't match the request."""

    try:
        reply = SkyHookReply.model_validate_json(response.content)
    except ValidationError as e:
        _warn_malformed(tvdb_id, order_type, validation_summary(e))
        return SkyHookReply(tvdb_id=tvdb_id)
    if reply.tvdb_id != tvdb_id:
        # Seen when TVDB merged the series into another id and Sonarr still has the old one.
        hub_warn(f"SkyHook answered TVDB {tvdb_id} with TVDB {reply.tvdb_id}'s {order_type} order - storing it empty")
        return SkyHookReply(tvdb_id=tvdb_id)
    if not reply.lists(order_type):
        _warn_malformed(tvdb_id, order_type, "the reply doesn't list that order")
        return SkyHookReply(tvdb_id=tvdb_id)
    if reply.rows_failed:
        _warn_malformed(tvdb_id, order_type, "an episode row failed validation")
    return reply


def _warn_malformed(tvdb_id: int, order_type: str, detail: str) -> None:
    """Warn that one order's reply can't be used, so the order is stored with no episodes."""

    hub_warn(f"SkyHook sent a malformed {order_type} order for TVDB {tvdb_id} ({detail}) - storing it empty")
