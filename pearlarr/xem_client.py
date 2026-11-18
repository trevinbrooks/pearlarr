"""TheXEM wire client: the scene numbers releases give a series' episodes, read per TVDB series."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import override

import httpx
from pydantic import ValidationError, field_validator

from .boundary import ApiModel, BlankStr, LenientStr, validate_all_or_none, validation_summary
from .output import Severity, hub_note, hub_warn
from .scene_numbers import SceneMapping
from .web_client import failure_detail, get_with_retries

# --- TheXEM (`thexem.info` scene mappings) -----------------------------------


XEM_URL = "https://thexem.info/map/all"

_SUCCESS = "success"
_FAILURE = "failure"

_NO_MAPPINGS_MESSAGES = ("no single connection", "no show with the tvdb_id")
"""What a failure says of a series TheXEM maps to no other site, and of one it doesn't list. They are Sonarr's own
two (`XemProxy.IgnoredErrors`), looked for inside the message as Sonarr does: the second is followed by the id."""

_SAID_MAX = 160
"""What a reply says of itself is quoted in a warning, clamped to this."""


class XemNumbers(ApiModel):
    """One side of a mapping: the numbers one site gives an episode."""

    season: int
    episode: int
    absolute: int


class XemMapping(ApiModel):
    """One `data[]` row of a mappings reply: a TVDB episode, with its numbers on each site TheXEM maps it to."""

    tvdb: XemNumbers
    scene: XemNumbers | None = None
    """The numbers releases give the episode. Missing on a row that maps the episode to other sites only."""


class XemReply(ApiModel):
    """One mappings reply (`/map/all?id=<tvdbId>&origin=tvdb`)."""

    result: LenientStr = None
    """`success`, or `failure` with the reason in `message`."""
    message: BlankStr = ""
    data: tuple[XemMapping, ...] | None = None
    """The series' mappings, none on a failure. None when the reply sends no list, or when any row failed
    validation."""

    @field_validator("data", mode="before")
    @classmethod
    def _data_all_or_none(cls, value: object) -> object:
        """Read `data[]` all-or-nothing: one bad row leaves it None."""

        return validate_all_or_none(XemMapping, value)

    @property
    def says_no_mappings(self) -> bool:
        """Whether the reply is TheXEM's failure for a series it has no mappings for."""

        return self.result == _FAILURE and any(known in self.message for known in _NO_MAPPINGS_MESSAGES)

    @property
    def said(self) -> str:
        """The reply's result and message on one clamped line, for a warning."""

        return " ".join(f"{self.result or 'no result'}: {self.message or 'no message'}".split())[:_SAID_MAX]


class XemNoReply(Enum):
    """Why a read has no mappings to give."""

    UNREACHABLE = auto()
    """TheXEM is unreachable this run: a request failed, or it answered with another status than 200, or with a
    200 that isn't JSON."""

    UNUSABLE = auto()
    """TheXEM answered in JSON, and mappings can't be read from the answer. It says nothing of any other series."""


class AbstractXemClient(ABC):
    """The TheXEM interface the gateway depends on."""

    @property
    @abstractmethod
    def unreachable(self) -> bool:
        """True once TheXEM has been marked unreachable for this run."""

    @abstractmethod
    def mappings(self, tvdb_id: int) -> tuple[SceneMapping, ...] | XemNoReply:
        """One series' mappings that have a scene side, empty when TheXEM has none for the series.

        Once TheXEM is unreachable every call returns `XemNoReply.UNREACHABLE` without a request.
        """


class XemClient(AbstractXemClient):
    """TheXEM over the shared web client: GETs with the web retry policy and a per-run unreachable flag.

    A request carries the series' TVDB id and nothing else about the library.
    """

    def __init__(self, *, client: httpx.Client, sleep: Callable[[float], None] = time.sleep) -> None:
        """Bind the shared web client (network-free). `sleep` is the retry backoff's."""

        self._client = client
        self._sleep = sleep
        self._unreachable = False

    @property
    @override
    def unreachable(self) -> bool:
        return self._unreachable

    def _note_unreachable(self, detail: str) -> XemNoReply:
        """Warn that TheXEM failed and mark it unreachable, muting every later call this run."""

        hub_warn(f"TheXEM failed ({detail}) - keeping the scene numbers already stored until it answers")
        self._unreachable = True
        return XemNoReply.UNREACHABLE

    @override
    def mappings(self, tvdb_id: int) -> tuple[SceneMapping, ...] | XemNoReply:
        if self._unreachable:
            return XemNoReply.UNREACHABLE
        asked = f"for TVDB {tvdb_id}"
        try:
            response = get_with_retries(self._client, f"{XEM_URL}?id={tvdb_id}&origin=tvdb", sleep=self._sleep)
        except httpx.RequestError as e:
            return self._note_unreachable(f"{failure_detail(e)} {asked}")
        # TheXEM answers 200 for a series it doesn't list too, so any other status is about the host.
        if response.status_code != 200:
            return self._note_unreachable(f"HTTP {response.status_code} {asked}")
        try:
            reply = XemReply.model_validate_json(response.content)
        except ValidationError as e:
            # A body that isn't JSON is a page served in TheXEM's place, so it is about the host too.
            if _is_not_json(e):
                return self._note_unreachable(f"a reply that isn't JSON {asked}")
            return _warn_unusable(tvdb_id, validation_summary(e))
        return _read_reply(reply, tvdb_id)


def _is_not_json(e: ValidationError) -> bool:
    """Whether the body failed to parse as JSON at all, where any other failure is of JSON with another shape."""

    return any(error["type"] == "json_invalid" for error in e.errors(include_url=False, include_input=False))


def _read_reply(reply: XemReply, tvdb_id: int) -> tuple[SceneMapping, ...] | XemNoReply:
    """The reply's mappings that have a scene side, or `XemNoReply.UNUSABLE` (warned) when it has none to read.

    The rule is pearlarr's own, and stricter than Sonarr's, which keeps a reply's rows whatever its `result` says.
    A `failure` means no mappings only with one of the two messages TheXEM has for a series without any. Only a
    `success` has mappings to read, and every row of it must read: a number that is missing or isn't a whole number
    makes the reply unusable. A row with no scene side is dropped, as Sonarr drops it. So a failure worded another
    way leaves the series unread, where Sonarr would read it as a series without mappings.
    """

    if reply.says_no_mappings:
        hub_note(f"TheXEM has no scene numbers for TVDB {tvdb_id}", severity=Severity.DEBUG)
        return ()
    if reply.result != _SUCCESS:
        return _warn_unusable(tvdb_id, reply.said)
    if reply.data is None:
        return _warn_unusable(tvdb_id, "its mappings failed validation")
    return tuple(
        SceneMapping(row.tvdb.season, row.tvdb.episode, row.scene.season, row.scene.episode, row.scene.absolute)
        for row in reply.data
        if row.scene is not None
    )


def _warn_unusable(tvdb_id: int, detail: str) -> XemNoReply:
    """Warn that one series' reply can't be used, so nothing is stored from it."""

    hub_warn(f"TheXEM's answer for TVDB {tvdb_id} can't be used ({detail}) - keeping what is stored for the series")
    return XemNoReply.UNUSABLE
