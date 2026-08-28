"""qBittorrent adapter: parse a SeaDex release URL and add it to the client."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import NamedTuple

import httpx
import pynyaa
import qbittorrentapi
from seadex import Tracker

from .arr_categories import ArrCategoryResolver
from .seadex_types import SeadexUrlItem
from .torrent import (
    ParsedTorrent,
    TorrentParseError,
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)


class TorrentAddError(Exception):
    """qBittorrent rejected the add (a non-`"Ok."` `torrents_add` result)."""


# The expected external failures a grab can hit. Anything else is a bug and propagates.
GRAB_FAILURES: tuple[type[Exception], ...] = (
    TorrentParseError,
    TorrentAddError,
    httpx.HTTPError,
    pynyaa.PyNyaaError,
    qbittorrentapi.APIError,
)


type _Parser = Callable[[str, str | None, httpx.Client], ParsedTorrent]


def _parse_nyaa(url: str, infohash: str | None, client: httpx.Client) -> ParsedTorrent:
    del infohash, client
    return get_nyaa_torrent(url=url)


def _parse_animetosho(url: str, infohash: str | None, client: httpx.Client) -> ParsedTorrent:
    del infohash
    return get_animetosho_torrent(url=url, client=client)


def _parse_rutracker(url: str, infohash: str | None, client: httpx.Client) -> ParsedTorrent:
    return get_rutracker_torrent(url=url, infohash=infohash, client=client)


# The grab pipeline pre-filters on PARSEABLE_TRACKERS, so `add`'s raise is defensive.
_PARSERS: dict[Tracker, _Parser] = {
    Tracker.NYAA: _parse_nyaa,
    Tracker.ANIMETOSHO: _parse_animetosho,
    Tracker.RUTRACKER: _parse_rutracker,
}
PARSEABLE_TRACKERS: frozenset[Tracker] = frozenset(_PARSERS)


class AddOutcome(Enum):
    """The result of handing a release to the torrent client."""

    ADDED = auto()
    ALREADY_ADDED = auto()


class AddResult(NamedTuple):
    """One add's result."""

    outcome: AddOutcome
    name: str | None
    """The qBittorrent-reported name, falling back to the scraped release title. None when neither exists."""
    added_on: datetime | None = None
    """qBittorrent's own add time for an `ALREADY_ADDED` torrent (local-naive, like `now_stamp`), else None."""


def _row_added_on(row: object) -> datetime | None:
    """The `added_on` epoch seconds off a qBittorrent info row as a datetime, or None when absent/junk."""

    raw = getattr(row, "added_on", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    try:
        return datetime.fromtimestamp(raw)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class ReleaseOutcome:
    """One release's add result."""

    outcome: AddOutcome
    name: str | None
    group: str

    @property
    def added(self) -> bool:
        return self.outcome is AddOutcome.ADDED


class TorrentService:
    """Parse a release URL by tracker and add the torrent to qBittorrent."""

    def __init__(
        self,
        *,
        qbit: qbittorrentapi.Client | None,
        web: httpx.Client,
        categories: ArrCategoryResolver,
        tags: list[str] | None,
        logger: logging.Logger,
    ) -> None:
        """Wire the adapter to the client and the shared web HTTP client."""

        self.qbit = qbit
        self.web = web
        self.categories = categories
        self.tags = tags
        self.logger = logger

    def add(
        self,
        *,
        item: SeadexUrlItem,
        preview: bool,
    ) -> AddResult:
        """Parse a release URL by tracker and add it to qBittorrent."""

        parser = _PARSERS.get(item.tracker)
        if parser is None:
            raise ValueError(f"Unable to parse torrent links from {item.tracker}")
        parsed_url, source_name = parser(item.url, item.infohash, self.web)

        if parsed_url is None:
            raise TorrentParseError(f"Could not extract a torrent download link from {item.url}")

        added = self._add_to_qbit(item=item, torrent_url=parsed_url, preview=preview)

        return AddResult(added.outcome, added.name or source_name, added.added_on)

    def _add_to_qbit(
        self,
        *,
        item: SeadexUrlItem,
        torrent_url: str,
        preview: bool,
    ) -> AddResult:
        """Add a torrent to qBittorrent (dedup by hash, read the name back)."""

        infohash = item.infohash

        # A private torrent has no info hash to dedup by, so qBittorrent dedups it internally.
        if infohash is not None and self.qbit is not None:
            torr_info = self.qbit.torrents_info(torrent_hashes=infohash)
            if torr_info:
                self.logger.debug(f"Torrent {item.url} already in qBittorrent")
                return AddResult(AddOutcome.ALREADY_ADDED, torr_info[0].name, _row_added_on(torr_info[0]))

        if preview:
            return AddResult(AddOutcome.ADDED, None)

        # Past the preview gate there is always a client, so this raise is defensive.
        if self.qbit is None:
            raise RuntimeError("qBittorrent client not configured")

        result = self.qbit.torrents_add(
            urls=torrent_url,
            category=self.categories.grab(),
            tags=self.tags,
        )
        if result != "Ok.":
            raise TorrentAddError(f"qBittorrent rejected the torrent from {item.url} (response: {result!r})")

        torrent_name = None
        if infohash is not None:
            added_info = self.qbit.torrents_info(torrent_hashes=infohash)
            torrent_name = added_info[0].name if added_info else None

        return AddResult(AddOutcome.ADDED, torrent_name)
