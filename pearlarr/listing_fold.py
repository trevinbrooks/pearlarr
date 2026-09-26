"""Pure functions that turn a series' SeaDex entries into per-torrent listings and size identities."""

from collections.abc import Iterable, Iterator
from typing import NamedTuple

from seadex import EntryRecord

from .manual_import import path_leaf, unambiguous
from .placement_types import EMPTY_LISTING, ListingsRead, TorrentListing
from .seadex_types import carries_ignored_tag, normalized_infohash
from .video_files import is_video_candidate


class EntryListing(NamedTuple):
    """One entry of the series: its SeaDex record and its episode window."""

    record: EntryRecord
    """The entry's SeaDex record."""

    window: frozenset[int] | None
    """The window's episode ids, None if its episodes couldn't be read."""


def fold_listings(entries: Iterable[EntryListing], ignore_tags: frozenset[str]) -> ListingsRead:
    """Build the series' `ListingsRead` from its entries.

    Each infohash gets the union of the windows that list it, and the size identities come from
    `size_identities`. If any window couldn't be read, the hashes it lists and the identities are marked unread.
    A size that two entries give to different episodes is dropped.
    """

    by_hash: dict[str, TorrentListing | None] = {}
    pairs: list[tuple[int, int]] = []
    unread = False
    for record, window in entries:
        for torrent in record.torrents:
            if (infohash := normalized_infohash(torrent.infohash)) is None:
                continue
            union = by_hash.get(infohash, EMPTY_LISTING)
            by_hash[infohash] = None if window is None or union is None else TorrentListing(union.ids | window)
        if window is None:
            unread = True
        else:
            pairs.extend(size_identities(record, window, ignore_tags))
    return ListingsRead(by_hash, None if unread else unambiguous(pairs))


def size_identities(
    record: EntryRecord, window: frozenset[int], ignore_tags: frozenset[str]
) -> Iterator[tuple[int, int]]:
    """Yield `(size, episode)` for each torrent with exactly one video file, when the entry covers one episode.

    All trackers count, since the size identifies the file wherever it's served. Torrents with an ignored tag
    are skipped, because the release filter never offers them.
    """

    if len(window) != 1:
        return
    (ep_id,) = window
    for torrent in record.torrents:
        if carries_ignored_tag(torrent, ignore_tags):
            continue
        videos = [f.size for f in torrent.files if is_video_candidate(path_leaf(f.name))]
        # SeaDex lists an unknown size as 0, so skip it.
        if len(videos) == 1 and videos[0] > 0:
            yield videos[0], ep_id
