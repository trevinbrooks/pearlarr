"""qBittorrent adapter: parse a SeaDex release URL and add it to the client.

``TorrentService`` owns the per-tracker URL parsing and the qBittorrent
interaction (dedup-by-hash, add, read the name back). It returns the add status
and the resolved name; the orchestrator keeps the download-flag / public_only /
tracker filtering and the run-state bookkeeping (stats, counters, the
max-torrents cap) around it - that knot is untied in Phase 4 behind RunContext.

Extracted from ``SeaDexArr`` in Phase 3 of the refactor (see
``REFACTOR_PLAN.md``); behaviour-preserving.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import qbittorrentapi
import requests
from seadex import Tracker

from .log import indent_string
from .torrent import (
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)


class AddOutcome(Enum):
    """The result of handing a release to the torrent client.

    Replaces the ``"torrent_added"`` / ``"torrent_already_added"`` status strings:
    a closed two-member vocabulary, so the dispatch on it is exhaustive (no dead
    ``else: raise`` fallthrough).
    """

    ADDED = auto()  # was "torrent_added"
    ALREADY_ADDED = auto()  # was "torrent_already_added" / "already have"


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
        torrent_hash: str | None,
        preview: bool,
    ) -> tuple[AddOutcome, str | None]:
        """Parse a release URL by tracker and add it to qBittorrent.

        Args:
            url (str): SeaDex release-page URL.
            tracker (Tracker): SeaDex tracker (selects the parser).
            torrent_hash (str | None): Info hash, used to dedup / read the name
                back (None for a private torrent with no hash).
            preview (bool): When True, simulate the add without touching the
                client.

        Returns:
            tuple: (outcome, name) - outcome is ``AddOutcome.ADDED`` or
                ``AddOutcome.ALREADY_ADDED``; name is the qBittorrent-reported
                name, falling back to the release title scraped from the source
                page.
        """

        # Each parser returns the download/magnet link plus the release's
        # human-readable title scraped from the source page, so we always
        # have a real name to show even when the client can't report one
        # (e.g. a private torrent with no info hash, or a dry run)

        # Each tracker has its own parser with its own call shape; key the
        # dispatch on the Tracker enum members directly (no .lower()) and look
        # the parser up once. Closures capture the per-tracker call args.
        parsers = {
            Tracker.NYAA: lambda: get_nyaa_torrent(url=url),
            Tracker.ANIMETOSHO: lambda: get_animetosho_torrent(
                url=url,
                session=self.session,
            ),
            Tracker.RUTRACKER: lambda: get_rutracker_torrent(
                url=url,
                torrent_hash=torrent_hash,
                session=self.session,
            ),
        }
        parser = parsers.get(tracker)
        if parser is None:
            raise ValueError(f"Unable to parse torrent links from {tracker}")
        parsed_url, source_name = parser()

        if parsed_url is None:
            raise Exception("Have not managed to parse the torrent URL")

        status, torrent_name = self._add_to_qbit(
            url=url,
            torrent_url=parsed_url,
            torrent_hash=torrent_hash,
            preview=preview,
        )

        # Prefer the name qBittorrent reports; fall back to the release's
        # title from the source page rather than the raw download link
        if not torrent_name:
            torrent_name = source_name

        return status, torrent_name

    def _add_to_qbit(
        self,
        *,
        url: str,
        torrent_url: str,
        torrent_hash: str | None,
        preview: bool,
    ) -> tuple[AddOutcome, str | None]:
        """Add a torrent to qBittorrent (dedup by hash, read the name back).

        Args:
            url (str): SeaDex URL (for the "already added" debug line).
            torrent_url (str): Torrent / magnet link to hand the client.
            torrent_hash (str | None): Info hash, or None for a hashless torrent.
            preview (bool): When True, report the add without touching the client.

        Returns:
            tuple: (outcome, torrent_name) - outcome is ``AddOutcome.ADDED`` or
                ``AddOutcome.ALREADY_ADDED``; torrent_name is the client-reported
                name (None when there's no hash to look it up by).
        """

        # A private torrent has no info hash, so we can't look it up by hash to
        # dedup or to read its name back; just add it and let qBittorrent dedup
        # internally. With a hash, skip the adding if it's already present
        if torrent_hash is not None and self.qbit is not None:
            torr_info = self.qbit.torrents_info(torrent_hashes=torrent_hash)
            torr_hashes = [i.hash for i in torr_info]

            if torrent_hash in torr_hashes:
                self.logger.debug(
                    indent_string(f"Torrent {url} already in qBittorrent"),
                )
                return AddOutcome.ALREADY_ADDED, torr_info[0].name

        # Preview (dry run or no client): report it as added without touching the
        # client. With a client present the dedup lookup above still ran, so an
        # already-present torrent is reported accurately. There's no client-side
        # name to read back, so the caller falls back to the URL.
        if preview:
            return AddOutcome.ADDED, None

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
            raise Exception("Failed to add torrent")

        # Look the torrent back up by hash so we can report its name. A private
        # torrent has no info hash to look up, so leave the name unset and let
        # the caller fall back to the URL
        torrent_name = None
        if torrent_hash is not None:
            added_info = self.qbit.torrents_info(torrent_hashes=torrent_hash)
            torrent_name = added_info[0].name if added_info else None

        return AddOutcome.ADDED, torrent_name
