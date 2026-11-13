"""Pure functions that turn a series' SeaDex entries into per-torrent listings and size identities."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from seadex import EntryRecord

from .anibridge import NO_ALIASES, SpecialAliasing
from .file_paths import file_name
from .pairs import unambiguous
from .pending_import import TitleWindow
from .placement_types import EpisodeBySize, FetchedListings, TorrentListing
from .releases import carries_ignored_tag, normalized_infohash
from .video_files import is_video_candidate


@dataclass(frozen=True, slots=True)
class EntryListing:
    """One entry of the series: its SeaDex record, its episode window, and its special aliases."""

    record: EntryRecord
    """The entry's SeaDex record."""

    window: TitleWindow | None
    """The entry's window, None if its episodes couldn't be read."""

    special_aliasing: SpecialAliasing | None
    """The entry's specials aliases and their TMDB show (`MappingEntry.special_aliasing`), None when it has none."""


def fold_listings(entries: Sequence[EntryListing], ignore_tags: frozenset[str]) -> FetchedListings:
    """Build the series' `ListingsRead` from its entries.

    Each infohash gets the windows of the entries that list it and the union of their special aliases
    (`_torrent_listing`), and the size identities come from `size_identities`. An unread window marks its hashes
    and the identities unread. A size or a TMDB number that two entries map differently is dropped.
    """

    by_hash: dict[str, list[EntryListing]] = {}
    for entry in entries:
        for torrent in entry.record.torrents:
            if (infohash := normalized_infohash(torrent.infohash)) is not None:
                by_hash.setdefault(infohash, []).append(entry)
    listings = {infohash: _torrent_listing(listed) for infohash, listed in by_hash.items()}
    return FetchedListings(listings, _series_identities(entries, ignore_tags))


def _torrent_listing(entries: Sequence[EntryListing]) -> TorrentListing | None:
    """The entries' windows and the union of their special aliases, or None if any window is unread.

    When the entries pair against different TMDB shows, the hash gets no aliases.
    """

    windows: list[TitleWindow] = []
    pairs: list[tuple[int, int]] = []
    shows: set[int] = set()
    for entry in entries:
        if entry.window is None:
            return None
        windows.append(entry.window)
        if entry.special_aliasing is not None:
            shows.add(entry.special_aliasing.tmdb_id)
            pairs.extend(entry.special_aliasing.aliases.items())
    return TorrentListing(tuple(windows), unambiguous(pairs) if len(shows) == 1 else NO_ALIASES)


def _series_identities(entries: Sequence[EntryListing], ignore_tags: frozenset[str]) -> EpisodeBySize | None:
    """The size identities of every entry, or None if any window is unread."""

    pairs: list[tuple[int, int]] = []
    for entry in entries:
        if entry.window is None:
            return None
        pairs.extend(size_identities(entry.record, entry.window, ignore_tags))
    return unambiguous(pairs)


def size_identities(record: EntryRecord, window: TitleWindow, ignore_tags: frozenset[str]) -> Iterator[tuple[int, int]]:
    """Yield `(size, episode)` for each torrent with exactly one video file, when the entry covers one episode.

    All trackers count, since the size identifies the file wherever it's served. Torrents with an ignored tag
    are skipped, because the release filter never offers them.
    """

    if len(window.ordered_episode_ids) != 1:
        return
    (ep_id,) = window.ordered_episode_ids
    for torrent in record.torrents:
        if carries_ignored_tag(torrent, ignore_tags):
            continue
        videos = [f.size for f in torrent.files if is_video_candidate(file_name(f.name))]
        # SeaDex lists an unknown size as 0, so skip it.
        if len(videos) == 1 and videos[0] > 0:
            yield videos[0], ep_id
