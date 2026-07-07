"""qBittorrent adapter: parse a SeaDex release URL and add it to the client."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import httpx
import pynyaa
import qbittorrentapi
import requests
from seadex import Tracker

from .log import indent_string
from .torrent import (
    TorrentParseError,
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)


class TorrentAddError(Exception):
    """qBittorrent rejected the add (a non-``"Ok."`` ``torrents_add`` result)."""


# The expected external failures a grab can hit: the tracker scrape (requests /
# pynyaa's httpx), the parse itself, and the qBittorrent add. The grab pipeline
# contains these per release; anything else is a bug and propagates.
GRAB_FAILURES: tuple[type[Exception], ...] = (
    TorrentParseError,
    TorrentAddError,
    requests.RequestException,
    httpx.HTTPError,
    pynyaa.PyNyaaError,
    qbittorrentapi.APIError,
)


# Uniform parser signature: (url, infohash, session) -> (download/magnet link,
# release title scraped from the source page). Wrappers adapt the per-tracker args.
type _Parser = Callable[[str, str | None, requests.Session], tuple[str | None, str]]


def _parse_nyaa(url: str, infohash: str | None, session: requests.Session) -> tuple[str | None, str]:
    del infohash, session
    return get_nyaa_torrent(url=url)


def _parse_animetosho(url: str, infohash: str | None, session: requests.Session) -> tuple[str | None, str]:
    del infohash
    return get_animetosho_torrent(url=url, session=session)


def _parse_rutracker(url: str, infohash: str | None, session: requests.Session) -> tuple[str | None, str]:
    return get_rutracker_torrent(url=url, infohash=infohash, session=session)


# One parser per supported tracker; PARSEABLE_TRACKERS derives from this table so
# the two can't drift. The grab pipeline pre-filters on the frozenset, so an
# unparseable tracker is skipped there rather than reaching ``add``'s raise.
_PARSERS: dict[Tracker, _Parser] = {
    Tracker.NYAA: _parse_nyaa,
    Tracker.ANIMETOSHO: _parse_animetosho,
    Tracker.RUTRACKER: _parse_rutracker,
}
PARSEABLE_TRACKERS: frozenset[Tracker] = frozenset(_PARSERS)


class AddOutcome(Enum):
    """The result of handing a release to the torrent client.

    Replaces the ``"torrent_added"`` / ``"torrent_already_added"`` status strings:
    a closed two-member vocabulary, so the dispatch on it is exhaustive (no dead
    ``else: raise`` fallthrough).
    """

    ADDED = auto()  # was "torrent_added"
    ALREADY_ADDED = auto()  # was "torrent_already_added" / "already have"


class AddResult(NamedTuple):
    """One add's result: the outcome plus the best display name available.

    ``name`` is the qBittorrent-reported name, falling back to the release title
    scraped from the source page (None only when neither exists).
    """

    outcome: AddOutcome
    name: str | None


@dataclass(frozen=True)
class ReleaseOutcome:
    """One release's add result, as the engine records it for the reporter.

    Replaces the ``{"outcome", "name", "group"}`` dict the engine built and
    :meth:`RunReporter.log_seadex_action` re-read.
    """

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
        session: requests.Session,
        category: str | None,
        tags: list[str] | None,
        logger: logging.Logger,
    ) -> None:
        """Wire the adapter to the client and the shared HTTP session.

        Args:
            qbit (qbittorrentapi.Client | None): The logged-in client, or None
                when no client is configured (every add is then a preview).
            session (requests.Session): Shared keep-alive session for the
                tracker page scrapes.
            category (str | None): qBittorrent category for added torrents.
            tags (list[str] | None): qBittorrent tags for added torrents.
            logger (logging.Logger): For the "already in qBittorrent" debug line.
        """

        self.qbit = qbit
        self.session = session
        self.category = category
        self.tags = tags
        self.logger = logger

    def add(
        self,
        *,
        url: str,
        tracker: Tracker,
        infohash: str | None,
        preview: bool,
    ) -> AddResult:
        """Parse a release URL by tracker and add it to qBittorrent.

        Args:
            url (str): SeaDex release-page URL.
            tracker (Tracker): SeaDex tracker (selects the parser).
            infohash (str | None): Info hash, used to dedup / read the name
                back (None for a private torrent with no hash).
            preview (bool): When True, simulate the add without touching the
                client.

        Returns:
            AddResult: The outcome plus the best display name available.
        """

        parser = _PARSERS.get(tracker)
        if parser is None:
            raise ValueError(f"Unable to parse torrent links from {tracker}")
        parsed_url, source_name = parser(url, infohash, self.session)

        if parsed_url is None:
            raise TorrentParseError(f"Could not extract a torrent download link from {url}")

        outcome, torrent_name = self._add_to_qbit(
            url=url,
            torrent_url=parsed_url,
            infohash=infohash,
            preview=preview,
        )

        # Prefer the name qBittorrent reports; fall back to the release's
        # title from the source page rather than the raw download link.
        return AddResult(outcome, torrent_name or source_name)

    def _add_to_qbit(
        self,
        *,
        url: str,
        torrent_url: str,
        infohash: str | None,
        preview: bool,
    ) -> AddResult:
        """Add a torrent to qBittorrent (dedup by hash, read the name back).

        Args:
            url (str): SeaDex URL (for the "already added" debug line).
            torrent_url (str): Torrent / magnet link to hand the client.
            infohash (str | None): Info hash, or None for a hashless torrent.
            preview (bool): When True, report the add without touching the client.

        Returns:
            AddResult: The outcome plus the client-reported name (None when
                there's no hash to look it up by).
        """

        # A private torrent has no info hash, so we can't look it up by hash to
        # dedup or to read its name back; just add it and let qBittorrent dedup
        # internally. With a hash, skip the adding if it's already present
        if infohash is not None and self.qbit is not None:
            torr_info = self.qbit.torrents_info(torrent_hashes=infohash)
            torr_hashes = [i.hash for i in torr_info]

            if infohash in torr_hashes:
                self.logger.debug(
                    indent_string(f"Torrent {url} already in qBittorrent"),
                )
                return AddResult(AddOutcome.ALREADY_ADDED, torr_info[0].name)

        # Preview (dry run or no client): report it as added without touching the
        # client. With a client present the dedup lookup above still ran, so an
        # already-present torrent is reported accurately. There's no client-side
        # name to read back, so the caller falls back to the URL.
        if preview:
            return AddResult(AddOutcome.ADDED, None)

        # Past the preview gate there is always a client: the caller passes
        # preview=True whenever none is configured, so this narrows for type
        # safety and never raises in practice.
        if self.qbit is None:
            raise RuntimeError("qBittorrent client not configured")

        # Add the torrent
        result = self.qbit.torrents_add(
            urls=torrent_url,
            category=self.category,
            tags=self.tags,
        )
        if result != "Ok.":
            raise TorrentAddError(f"qBittorrent rejected the torrent from {url} (response: {result!r})")

        # Look the torrent back up by hash so we can report its name. A private
        # torrent has no info hash to look up, so leave the name unset and let
        # the caller fall back to the URL
        torrent_name = None
        if infohash is not None:
            added_info = self.qbit.torrents_info(torrent_hashes=infohash)
            torrent_name = added_info[0].name if added_info else None

        return AddResult(AddOutcome.ADDED, torrent_name)
