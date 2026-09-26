# pyright: strict
"""Folding a series' SeaDex entries into each torrent's listing and the series' size identities."""

from seadex import EntryRecord, Tag, TorrentRecord, Tracker

from pearlarr.listing_fold import EntryListing, fold_listings, size_identities
from pearlarr.placement_types import TorrentListing
from pearlarr.seadex_types import SpecialAliasing

from .builders import make_entry_record, make_torrent_record

_H1 = "a" * 40
_H2 = "b" * 40
_NO_TAGS: frozenset[str] = frozenset()


def _torrent(infohash: str | None, *files: tuple[str, int], tags: frozenset[Tag] = frozenset()) -> TorrentRecord:
    names = tuple(name for name, _size in files)
    sizes = tuple(size for _name, size in files)
    return make_torrent_record(infohash=infohash, file_names=names, file_sizes=sizes, tags=tags)


def _entry(*torrents: TorrentRecord) -> EntryRecord:
    return make_entry_record(torrents=torrents)


def _listed(record: EntryRecord, *ids: int) -> EntryListing:
    return EntryListing(record, frozenset(ids), None)


def _aliased(record: EntryRecord, ids: frozenset[int], aliases: dict[int, int], tmdb_id: int = 900) -> EntryListing:
    """An entry over `ids` whose `aliases` count in TMDB show `tmdb_id`."""

    return EntryListing(record, ids, SpecialAliasing(tmdb_id, aliases))


class TestFoldListings:
    """Each hash gets its entries' windows and special aliases, unioned, or None if any of those windows is unread."""

    def test_the_union_over_every_entry_listing_the_torrent(self) -> None:
        one, both, two = _entry(_torrent(_H1)), _entry(_torrent(_H1), _torrent(_H2)), _entry(_torrent(_H2))

        read = fold_listings([_listed(one, 501, 502), _listed(both, 503), _listed(two, 504)], _NO_TAGS)

        assert dict(read.by_hash) == {
            _H1: TorrentListing(frozenset({501, 502, 503})),
            _H2: TorrentListing(frozenset({503, 504})),
        }

    def test_an_unread_window_marks_its_hashes_unread(self) -> None:
        entries = [
            EntryListing(_entry(_torrent(_H1)), None, None),
            _listed(_entry(_torrent(_H1), _torrent(_H2)), 503),
        ]

        read = fold_listings(entries, _NO_TAGS)

        assert dict(read.by_hash) == {_H1: None, _H2: TorrentListing(frozenset({503}))}

    def test_hashes_match_by_their_one_spelling_and_a_hash_less_torrent_lists_nothing(self) -> None:
        read = fold_listings([_listed(_entry(_torrent(_H1.upper()), _torrent(None)), 501)], _NO_TAGS)

        assert dict(read.by_hash) == {_H1: TorrentListing(frozenset({501}))}

    def test_a_hash_no_entry_lists_is_listed_nowhere(self) -> None:
        read = fold_listings([_listed(_entry(_torrent(_H1)), 501)], _NO_TAGS)

        assert read.listing(_H2) == TorrentListing(frozenset())

    def test_each_hash_unions_the_aliases_of_every_entry_listing_it(self) -> None:
        one, both = _entry(_torrent(_H1)), _entry(_torrent(_H1), _torrent(_H2))
        entries = [
            _aliased(one, frozenset({501}), {7: 7, 13: 12}),
            _aliased(both, frozenset({502}), {15: 15, 16: 17}),
        ]

        read = fold_listings(entries, _NO_TAGS)

        assert dict(read.by_hash) == {
            _H1: TorrentListing(frozenset({501, 502}), {7: 7, 13: 12, 15: 15, 16: 17}),
            _H2: TorrentListing(frozenset({502}), {15: 15, 16: 17}),
        }

    def test_a_tmdb_number_two_entries_pair_differently_is_dropped(self) -> None:
        pack = _torrent(_H1)
        entries = [
            _aliased(_entry(pack), frozenset({501}), {13: 12}),
            _aliased(_entry(pack), frozenset({502}), {13: 14, 7: 7}),
        ]

        read = fold_listings(entries, _NO_TAGS)

        assert read.listing(_H1) == TorrentListing(frozenset({501, 502}), {7: 7})

    def test_entries_whose_aliases_count_in_different_tmdb_shows_leave_the_hash_without_aliases(self) -> None:
        # Each entry's TMDB numbers count in its own show, so the two alias maps aren't one numbering.
        pack = _torrent(_H1)
        entries = [
            _aliased(_entry(pack), frozenset({501}), {7: 7}, tmdb_id=900),
            _aliased(_entry(pack), frozenset({502}), {8: 8}, tmdb_id=901),
        ]

        read = fold_listings(entries, _NO_TAGS)

        assert read.listing(_H1) == TorrentListing(frozenset({501, 502}))

    def test_an_entry_without_aliases_leaves_the_other_entrys_aliases_on_the_hash(self) -> None:
        pack = _torrent(_H1)
        entries = [_aliased(_entry(pack), frozenset({501}), {7: 7}), _listed(_entry(pack), 502)]

        read = fold_listings(entries, _NO_TAGS)

        assert read.listing(_H1) == TorrentListing(frozenset({501, 502}), {7: 7})

    def test_single_file_copies_identify_a_packs_specials(self) -> None:
        # One pack listed under three entries, and a hash-less single-file copy of each special under its own.
        # The sizes identify the specials, whatever the pack's names say.
        pack = _torrent(_H1, ("Show - S00E02.mkv", 700), ("Show - S00E03.mkv", 800))
        read = fold_listings(
            [
                _listed(_entry(pack), 101, 102),
                _listed(_entry(pack, _torrent(None, ("Show - Special B.mkv", 800))), 502),
                _listed(_entry(pack, _torrent(None, ("Show - Special A.mkv", 700))), 501),
            ],
            _NO_TAGS,
        )

        assert read.identities is not None
        assert dict(read.identities) == {800: 502, 700: 501}
        assert read.listing(_H1) == TorrentListing(frozenset({101, 102, 501, 502}))

    def test_a_size_two_entries_give_different_episodes_is_dropped(self) -> None:
        single = _torrent(None, ("Show - Special.mkv", 700))

        read = fold_listings([_listed(_entry(single), 7897), _listed(_entry(single), 7898)], _NO_TAGS)

        assert read.identities == {}

    def test_an_unread_window_leaves_the_identities_unread(self) -> None:
        # The unread entry's sizes are unknown, and one of them might clash with another entry's size.
        entries = [
            _listed(_entry(_torrent(None, ("Show - Special.mkv", 700))), 7897),
            EntryListing(_entry(_torrent(None, ("Show - Special.mkv", 700))), None, None),
        ]

        assert fold_listings(entries, _NO_TAGS).identities is None


class TestSizeIdentities:
    """A torrent with one video file, under an entry covering one episode, identifies that episode by size."""

    def test_a_lone_video_under_a_one_episode_window_identifies_its_episode(self) -> None:
        record = _entry(_torrent(_H1, ("Show - Special.mkv", 700)))

        assert list(size_identities(record, frozenset({7898}), _NO_TAGS)) == [(700, 7898)]

    def test_one_video_beside_subtitles_is_a_single_file_listing(self) -> None:
        record = _entry(_torrent(_H1, ("Show - Special.mkv", 700), ("Show - Special.ass", 5), ("Fonts/a.ttf", 9)))

        assert list(size_identities(record, frozenset({7898}), _NO_TAGS)) == [(700, 7898)]

    def test_two_videos_identify_nothing(self) -> None:
        record = _entry(_torrent(_H1, ("Show - 01.mkv", 700), ("Show - 02.mkv", 800)))

        assert list(size_identities(record, frozenset({7898}), _NO_TAGS)) == []

    def test_a_wider_window_identifies_nothing(self) -> None:
        record = _entry(_torrent(_H1, ("Show - Special.mkv", 700)))

        assert list(size_identities(record, frozenset({7897, 7898}), _NO_TAGS)) == []

    def test_a_torrent_with_an_ignored_tag_identifies_nothing(self) -> None:
        tagged = _torrent(_H1, ("Show - Special.mkv", 700), tags=frozenset({Tag.DOLBY_VISION}))
        record = _entry(tagged, _torrent(_H2, ("Show - Special v2.mkv", 800)))

        assert list(size_identities(record, frozenset({7898}), frozenset({"dolby vision"}))) == [(800, 7898)]

    def test_every_tracker_counts_and_a_hash_less_torrent_too(self) -> None:
        private = make_torrent_record(
            tracker=Tracker.ANIMEBYTES, infohash=None, file_names=("Show - Special v2.mkv",), file_sizes=(800,)
        )
        record = _entry(_torrent(None, ("Show - Special.mkv", 700)), private)

        assert list(size_identities(record, frozenset({7898}), _NO_TAGS)) == [(700, 7898), (800, 7898)]

    def test_a_zero_size_identifies_nothing(self) -> None:
        record = _entry(_torrent(_H1, ("Show - Special.mkv", 0)))

        assert list(size_identities(record, frozenset({7898}), _NO_TAGS)) == []
