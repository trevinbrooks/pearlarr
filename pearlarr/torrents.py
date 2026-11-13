"""qBittorrent adapter: find a SeaDex release's torrent in the client, or parse its URL and add it."""

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import NamedTuple, Self

import httpx
import qbittorrentapi
from seadex import Tracker

from .arr_categories import ArrCategoryResolver
from .boundary import coerce_int
from .manual_import import normalized_leaf, torrent_errored, torrent_finished
from .output import hub_warn
from .releases import SeadexDict, SeadexUrlItem, normalized_infohash
from .torrent import (
    ParsedTorrent,
    TorrentParseError,
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)


class TorrentAddError(Exception):
    """A failed add: qBittorrent rejected it (a non-`"Ok."` `torrents_add` result), or the url's cross-seed left."""


# The expected external failures a grab can hit. Anything else is a bug and propagates.
GRAB_FAILURES: tuple[type[Exception], ...] = (
    TorrentParseError,
    TorrentAddError,
    httpx.HTTPError,
    qbittorrentapi.APIError,
)


type _Parser = Callable[[str, str | None, httpx.Client], ParsedTorrent]


def _parse_nyaa(url: str, infohash: str | None, client: httpx.Client) -> ParsedTorrent:
    del infohash
    return get_nyaa_torrent(url=url, client=client)


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
    """The qBittorrent-reported name. A new add falls back to the tracker's release title. None when neither exists."""
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


type _FileSet = Counter[tuple[str, int | None]]
"""A torrent's files as (normalized leaf name, size) pairs, counted."""


def _file_set(files: Iterable[tuple[str, int | None]]) -> _FileSet:
    """The (name, size) pairs of a torrent's files as a `_FileSet`."""

    return Counter((normalized_leaf(name), size) for name, size in files)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A torrent in this arr's categories that could be a url's cross-seed."""

    infohash: str
    name: str


@dataclass(slots=True)
class _CategoryLook:
    """The run's one look at this arr's categories (`TorrentService.mark_cross_seeds`)."""

    qbit: qbittorrentapi.Client
    by_size: Mapping[int, Sequence[_Candidate]]
    """The torrents that count, by total size, oldest first."""
    files: dict[str, _FileSet | None] = field(default_factory=dict[str, _FileSet | None])
    """Each candidate's files once read (`_files_of`)."""

    @classmethod
    def take(cls, qbit: qbittorrentapi.Client, categories: Iterable[str], *, pearlarr_imports: bool) -> Self:
        """List `categories` and keep the torrents that count.

        An errored one never does: it won't finish until someone fixes it. A finished one does only when Pearlarr
        imports.
        """

        rows = [row for category in categories for row in qbit.torrents_info(category=category)]
        by_size: dict[int, list[_Candidate]] = {}
        for row in sorted(rows, key=lambda row: _row_added_on(row) or datetime.max):
            raw_hash = getattr(row, "hash", None)
            infohash = normalized_infohash(raw_hash if isinstance(raw_hash, str) else None)
            total_size = coerce_int(getattr(row, "total_size", None))
            unusable = torrent_errored(row) or (torrent_finished(row) and not pearlarr_imports)
            if infohash is None or total_size is None or unusable:
                continue
            by_size.setdefault(total_size, []).append(_Candidate(infohash, str(getattr(row, "name", ""))))
        return cls(qbit, by_size)

    def cross_seed_of(self, item: SeadexUrlItem) -> _Candidate | None:
        """The oldest torrent that counts and holds exactly the url's files, by name and size, under another hash.

        The oldest, so every run picks the same one: the url's record is keyed by its hash. None when there's
        none, or SeaDex's own torrent is anywhere in qBittorrent. Names count, not sizes alone: sizes are the
        planner's identity for one release (`DownloadPlanner.reduce_overlapping_downloads`), but a record finds its
        files on disk by name, so a torrent is usable only when the url's names are in it.
        """

        # No hash means a private url. Those are never marked: the planner's private and fallback handling knows
        # nothing of a cross-seed.
        if item.infohash is None:
            return None
        candidates = self.by_size.get(sum(item.size), ())
        # SeaDex's own torrent in a listed category settles it with no file list read.
        if any(candidate.infohash == item.infohash for candidate in candidates):
            return None
        wanted = _file_set(zip(item.files, item.size, strict=True))
        holder = next((candidate for candidate in candidates if self._files_of(candidate) == wanted), None)
        if holder is None or self.qbit.torrents_info(torrent_hashes=item.infohash):
            return None
        return holder

    def _files_of(self, candidate: _Candidate) -> _FileSet | None:
        """The candidate's files, read once a run. None when it left qBittorrent or skips one of its files."""

        if candidate.infohash not in self.files:
            self.files[candidate.infohash] = self._read_files(candidate.infohash)
        return self.files[candidate.infohash]

    def _read_files(self, infohash: str) -> _FileSet | None:
        try:
            files = self.qbit.torrents_files(torrent_hash=infohash)
        except qbittorrentapi.NotFound404Error:
            return None
        # Priority 0 is "do not download": the torrent will never hold that file.
        if any(getattr(f, "priority", None) == 0 for f in files):
            return None
        return _file_set((str(getattr(f, "name", "")), coerce_int(getattr(f, "size", None))) for f in files)


class TorrentService:
    """Find a release's torrent in qBittorrent, or parse its URL by tracker and add it."""

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
        self._pearlarr_imports = False
        # Taken at the first `mark_cross_seeds` whose categories can be read, and kept for the run like the service.
        self._look: _CategoryLook | None = None

    def begin_run(self, *, pearlarr_imports: bool) -> None:
        """Say who imports this run's downloads: Pearlarr, or the arr itself."""

        self._pearlarr_imports = pearlarr_imports

    def mark_cross_seeds(self, seadex_dict: SeadexDict) -> None:
        """Mark each url whose exact files a cross-seed in this arr's categories holds with that torrent's hash.

        Which torrents count depends on who imports them (`begin_run`). When Pearlarr does, it's any torrent in the
        grab or post-import category, finished or not. Otherwise it's an unfinished one in the grab category, the
        only kind the arr imports itself. An errored torrent never counts. The categories are listed once a run. A
        failed qBittorrent read warns once and marks nothing more this run.
        """

        if self.qbit is None:
            return
        try:
            if self._look is None:
                categories = self.categories.download_categories(with_post_import=self._pearlarr_imports)
                if categories is None:
                    # The arr's download clients can't be read: the next title asks again.
                    return
                self._look = _CategoryLook.take(self.qbit, categories, pearlarr_imports=self._pearlarr_imports)
            for rg_item in seadex_dict.values():
                for item in rg_item.urls.values():
                    cross_seed = self._look.cross_seed_of(item)
                    if cross_seed is not None:
                        self.logger.debug(
                            f"{item.url} is already in qBittorrent as {cross_seed.name} ({cross_seed.infohash[:8]})"
                        )
                        item.cross_seed_hash = cross_seed.infohash
        except qbittorrentapi.APIError as e:
            # An empty look marks nothing more and reads nothing more.
            self._look = _CategoryLook(self.qbit, {})
            hub_warn(
                f"Could not read qBittorrent's torrents ({type(e).__name__}"
                ") - this run may add a release that another torrent already holds"
            )

    def add(
        self,
        *,
        item: SeadexUrlItem,
        preview: bool,
    ) -> AddResult:
        """Add a release to qBittorrent, parsing its tracker page only when qBittorrent doesn't have it yet.

        A url marked with a cross-seed is never added. Its torrent is the cross-seed, and the add raises when that
        torrent has left qBittorrent.
        """

        parser = _PARSERS.get(item.tracker)
        if parser is None:
            raise ValueError(f"Unable to parse torrent links from {item.tracker}")
        if (present := self._already_added(item)) is not None:
            return present
        if item.cross_seed_hash is not None:
            raise TorrentAddError("the torrent holding this release's files left qBittorrent during the run")
        parsed_url, source_name = parser(item.url, item.infohash, self.web)

        if parsed_url is None:
            raise TorrentParseError(f"Could not extract a torrent download link from {item.url}")

        name = self._add_to_qbit(item=item, torrent_url=parsed_url, preview=preview)

        return AddResult(AddOutcome.ADDED, name or source_name)

    def _already_added(self, item: SeadexUrlItem) -> AddResult | None:
        """The `ALREADY_ADDED` result when qBittorrent has the url's torrent (`SeadexUrlItem.qbit_hash`), else None."""

        qbit_hash = item.qbit_hash
        # A private torrent has no info hash to dedup by, so qBittorrent dedups it internally.
        if qbit_hash is None or self.qbit is None:
            return None
        torr_info = self.qbit.torrents_info(torrent_hashes=qbit_hash)
        if not torr_info:
            return None
        self.logger.debug(f"Torrent {item.url} already in qBittorrent")
        return AddResult(AddOutcome.ALREADY_ADDED, torr_info[0].name, _row_added_on(torr_info[0]))

    def _add_to_qbit(
        self,
        *,
        item: SeadexUrlItem,
        torrent_url: str,
        preview: bool,
    ) -> str | None:
        """Add `torrent_url` to qBittorrent and return the name it reports.

        None in preview, without a hash to read it by, or when qBittorrent doesn't list the torrent yet.
        """

        infohash = item.infohash

        if preview:
            return None

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

        if infohash is None:
            return None
        added_info = self.qbit.torrents_info(torrent_hashes=infohash)
        return added_info[0].name if added_info else None
