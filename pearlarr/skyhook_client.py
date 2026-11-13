"""SkyHook wire client: Sonarr's metadata service, read for each series' TVDB episode orders."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, cast, override

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from .arr_models import SeasonTypeCount
from .boundary import (
    ApiModel,
    LenientStr,
    LenientTime,
    validate_all_or_none,
    validate_skipping_junk,
    validation_summary,
)
from .output import Severity, hub_note, hub_warn
from .web_client import failure_detail, get_with_retries

# --- SkyHook (`skyhook.sonarr.tv` TVDB shows) -------------------------------


ENDED_STATUS = "Ended"
"""TVDB's status for a series that has finished airing."""


class SkyHookEpisode(ApiModel):
    """One `episodes[]` row of a SkyHook show reply, numbered by the order the reply was asked for."""

    tvdb_id: int = Field(validation_alias="tvdbId")
    season_number: int = Field(validation_alias="seasonNumber")
    episode_number: int = Field(validation_alias="episodeNumber")
    air_date_utc: LenientTime = Field(default=None, validation_alias="airDateUtc")
    absolute_episode_number: int | None = Field(default=None, validation_alias="absoluteEpisodeNumber")
    """Sent on official rows only."""


class SkyHookReply(ApiModel):
    """One SkyHook show reply for one TVDB order (`/v2/tvdb/shows/eng/<tvdbId>/<type>`)."""

    tvdb_id: int = Field(validation_alias="tvdbId")
    status: LenientStr = None
    """TVDB's series status (`Ended`, `Continuing`, ...)."""
    season_types: tuple[SeasonTypeCount, ...] = Field(default=(), validation_alias="seasonTypes")
    """Every order TVDB lists for the series, whichever order was asked for."""
    episodes: tuple[SkyHookEpisode, ...] = ()
    """The asked order's rows. Empty when the reply sends none, or when any row failed validation."""
    rows_failed: bool = False
    """Whether a row failed validation and emptied `episodes`. Set by the validator, never read off the wire."""

    @model_validator(mode="before")
    @classmethod
    def _episodes_all_or_none(cls, data: object) -> object:
        """Read `episodes[]` all-or-nothing: one bad row empties it and sets `rows_failed`. A null is no rows."""

        if not isinstance(data, dict):
            return data
        body = cast("dict[str, Any]", data)
        raw = body.get("episodes")
        rows: list[SkyHookEpisode] | None = [] if raw is None else validate_all_or_none(SkyHookEpisode, raw)
        return {**body, "episodes": rows or (), "rows_failed": rows is None}

    @field_validator("season_types", mode="before")
    @classmethod
    def _lenient_season_types(cls, value: object) -> object:
        """Skip junk `seasonTypes[]` entries, never failing the reply over one."""

        return validate_skipping_junk(SeasonTypeCount, value)

    @property
    def ended(self) -> bool:
        """Whether TVDB lists the series as finished airing."""

        return self.status == ENDED_STATUS

    def lists(self, order_type: str) -> bool:
        """Whether the reply's `seasonTypes` names `order_type`."""

        return any(season_type.type == order_type for season_type in self.season_types)


class SkyHookError(ApiModel):
    """SkyHook's error body, as `{"message": "NotFound", "description": ...}` on a series TVDB doesn't have."""

    message: LenientStr = None

    @property
    def not_found(self) -> bool:
        """Whether SkyHook itself says it has no such series."""

        return self.message == "NotFound"


SKYHOOK_URL = "https://skyhook.sonarr.tv/v2/tvdb/shows/eng"


class SkyHookNoReply(Enum):
    """Why a fetch has no reply."""

    UNREACHABLE = auto()
    """SkyHook is unreachable this run, or answered with a status that says nothing about the series."""

    NOT_FOUND = auto()
    """SkyHook says TVDB has no such series."""


class AbstractSkyHookClient(ABC):
    """The SkyHook seam the gateway depends on."""

    @property
    @abstractmethod
    def unreachable(self) -> bool:
        """True once SkyHook has been declared unreachable for this run."""

    @abstractmethod
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookNoReply:
        """One series' reply for one order.

        A reply that fails validation or doesn't match the request comes back with no episodes. After an outage
        every call returns `SkyHookNoReply.UNREACHABLE` without a request.
        """


class SkyHookClient(AbstractSkyHookClient):
    """SkyHook over the shared web client: GETs with the web retry policy and a per-run outage latch."""

    def __init__(self, *, client: httpx.Client, sleep: Callable[[float], None] = time.sleep) -> None:
        """Bind the shared web client (network-free). `sleep` is the retry backoff's."""

        self._client = client
        self._sleep = sleep
        self._unreachable = False

    @property
    @override
    def unreachable(self) -> bool:
        return self._unreachable

    def _note_unreachable(self, detail: str) -> SkyHookNoReply:
        """Warn that SkyHook failed and latch the outage, muting every later call this run."""

        # The latch also trips on one series SkyHook keeps failing, so `detail` names the request.
        hub_warn(f"SkyHook failed ({detail}) - re-checking cached titles until it answers")
        self._unreachable = True
        return SkyHookNoReply.UNREACHABLE

    @override
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookNoReply:
        if self._unreachable:
            return SkyHookNoReply.UNREACHABLE
        asked = f"for TVDB {tvdb_id} {order_type}"
        try:
            response = get_with_retries(self._client, f"{SKYHOOK_URL}/{tvdb_id}/{order_type}", sleep=self._sleep)
        except httpx.RequestError as e:
            return self._note_unreachable(f"{failure_detail(e)} {asked}")
        if response.is_success:
            return _read_reply(response, tvdb_id, order_type)
        if _says_not_found(response):
            hub_note(f"SkyHook has no TVDB series {tvdb_id} ({order_type} order)", severity=Severity.DEBUG)
            return SkyHookNoReply.NOT_FOUND
        # Any other status is about the host, not the series: an edge block (403), a moved route (a bare 404), a
        # rate limit. Read as a miss, it would store every series empty for a week.
        return self._note_unreachable(f"HTTP {response.status_code} {asked}")


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
