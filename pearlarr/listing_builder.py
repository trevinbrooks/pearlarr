"""Pure functions that turn a series' SeaDex entries into per-torrent listings and episodes by size."""

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
class ListedTitle:
    """One title of the series: its SeaDex entry, its episode window, and its special aliases."""

    entry: EntryRecord
    """The title's SeaDex entry."""

    window: TitleWindow | None
    """The title's window, None if its episodes couldn't be read."""

    special_aliasing: SpecialAliasing | None
    """The title's special aliases and their TMDB show (`TitleMapping.special_aliasing`), None when it has none."""


def build_listings(titles: Sequence[ListedTitle], ignore_tags: frozenset[str]) -> FetchedListings:
    """Build the series' `FetchedListings` from its titles.

    Each infohash gets the windows of the titles whose entries list it and the union of their special aliases
    (`_torrent_listing`), and the episodes by size come from `single_episode_sizes`. An unread window marks its hashes
    and the episodes by size unread. A size or a TMDB number that two titles give differently is dropped.
    """

    by_hash: dict[str, list[ListedTitle]] = {}
    for title in titles:
        for torrent in title.entry.torrents:
            if (infohash := normalized_infohash(torrent.infohash)) is not None:
                by_hash.setdefault(infohash, []).append(title)
    listings = {infohash: _torrent_listing(listed) for infohash, listed in by_hash.items()}
    return FetchedListings(listings, _series_episode_by_size(titles, ignore_tags))


def _torrent_listing(titles: Sequence[ListedTitle]) -> TorrentListing | None:
    """The titles' windows and the union of their special aliases, or None if any window is unread.

    When the titles pair against different TMDB shows, the hash gets no aliases.
    """

    windows: list[TitleWindow] = []
    pairs: list[tuple[int, int]] = []
    shows: set[int] = set()
    for title in titles:
        if title.window is None:
            return None
        windows.append(title.window)
        if title.special_aliasing is not None:
            shows.add(title.special_aliasing.tmdb_id)
            pairs.extend(title.special_aliasing.aliases.items())
    return TorrentListing(tuple(windows), unambiguous(pairs) if len(shows) == 1 else NO_ALIASES)


def _series_episode_by_size(titles: Sequence[ListedTitle], ignore_tags: frozenset[str]) -> EpisodeBySize | None:
    """The episodes by size of every title, or None if any window is unread."""

    pairs: list[tuple[int, int]] = []
    for title in titles:
        if title.window is None:
            return None
        pairs.extend(single_episode_sizes(title.entry, title.window, ignore_tags))
    return unambiguous(pairs)


def single_episode_sizes(
    entry: EntryRecord, window: TitleWindow, ignore_tags: frozenset[str]
) -> Iterator[tuple[int, int]]:
    """Yield `(size, episode)` for each torrent with exactly one video file, when the entry covers one episode.

    All trackers count, since the size identifies the file wherever it's served. Torrents with an ignored tag
    are left out, because the release filter never offers them.
    """

    if len(window.ordered_episode_ids) != 1:
        return
    (ep_id,) = window.ordered_episode_ids
    for torrent in entry.torrents:
        if carries_ignored_tag(torrent, ignore_tags):
            continue
        videos = [f.size for f in torrent.files if is_video_candidate(file_name(f.name))]
        # SeaDex lists an unknown size as 0, so leave it out.
        if len(videos) == 1 and videos[0] > 0:
            yield videos[0], ep_id
