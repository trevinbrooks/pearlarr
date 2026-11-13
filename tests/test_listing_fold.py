# pyright: strict
"""Folding a series' SeaDex entries into each torrent's listing and the series' size identities."""

from seadex import EntryRecord, Tag, TorrentRecord, Tracker

from pearlarr.anibridge import SpecialAliasing
from pearlarr.listing_fold import EntryListing, fold_listings, size_identities
from pearlarr.manual_import import EntryWindow
from pearlarr.placement_types import EMPTY_LISTING

from .builders import entry_window, make_entry_record, make_torrent_record, torrent_listing

_H1 = "a" * 40
_H2 = "b" * 40
_NO_TAGS: frozenset[str] = frozenset()
_ONE_SPECIAL = entry_window(1, (7898,))
"""A window of the one special 7898."""


def _torrent(infohash: str | None, *files: tuple[str, int], tags: frozenset[Tag] = frozenset()) -> TorrentRecord:
    names = tuple(name for name, _size in files)
    sizes = tuple(size for _name, size in files)
    return make_torrent_record(infohash=infohash, file_names=names, file_sizes=sizes, tags=tags)


def _entry(al_id: int, *torrents: TorrentRecord) -> EntryRecord:
    return make_entry_record(anilist_id=al_id, torrents=torrents)


def _window(record: EntryRecord, *ids: int) -> EntryWindow:
    """The entry's window over `ids`."""

    return entry_window(record.anilist_id, ids)


def _listed(record: EntryRecord, *ids: int) -> EntryListing:
    return EntryListing(record, _window(record, *ids), None)


def _aliased(record: EntryRecord, ids: tuple[int, ...], aliases: dict[int, int], tmdb_id: int = 900) -> EntryListing:
    """An entry over `ids` whose `aliases` count in TMDB show `tmdb_id`."""

    return EntryListing(record, _window(record, *ids), SpecialAliasing(tmdb_id, aliases))


class TestFoldListings:
    """Each hash gets its entries' windows and the union of their special aliases, or None if any window is unread."""

    def test_each_hash_keeps_the_window_of_every_entry_listing_it(self) -> None:
        one, both, two = _entry(1, _torrent(_H1)), _entry(2, _torrent(_H1), _torrent(_H2)), _entry(3, _torrent(_H2))

        read = fold_listings([_listed(one, 501, 502), _listed(both, 503), _listed(two, 504)], _NO_TAGS)

        assert dict(read.by_hash) == {
            _H1: torrent_listing(_window(one, 501, 502), _window(both, 503)),
            _H2: torrent_listing(_window(both, 503), _window(two, 504)),
        }
        listing = read.listing(_H1)
        assert listing is not None
        assert listing.ids == {501, 502, 503}

    def test_the_windows_run_in_anilist_id_order_one_per_entry(self) -> None:
        # SeaDex can list one hash twice under an entry: that entry still holds one window on it.
        later, twice = _entry(3, _torrent(_H1)), _entry(1, _torrent(_H1), _torrent(_H1))

        read = fold_listings([_listed(later, 504), _listed(twice, 501)], _NO_TAGS)

        listing = read.listing(_H1)
        assert listing is not None
        assert listing.entries == (_window(twice, 501), _window(later, 504))

    def test_an_unread_window_marks_its_hashes_unread(self) -> None:
        both = _entry(2, _torrent(_H1), _torrent(_H2))
        entries = [EntryListing(_entry(1, _torrent(_H1)), None, None), _listed(both, 503)]

        read = fold_listings(entries, _NO_TAGS)

        assert dict(read.by_hash) == {_H1: None, _H2: torrent_listing(_window(both, 503))}

    def test_hashes_match_by_their_one_spelling_and_a_hash_less_torrent_lists_nothing(self) -> None:
        record = _entry(1, _torrent(_H1.upper()), _torrent(None))

        read = fold_listings([_listed(record, 501)], _NO_TAGS)

        assert dict(read.by_hash) == {_H1: torrent_listing(_window(record, 501))}

    def test_a_hash_no_entry_lists_is_listed_nowhere(self) -> None:
        read = fold_listings([_listed(_entry(1, _torrent(_H1)), 501)], _NO_TAGS)

        assert read.listing(_H2) == EMPTY_LISTING

    def test_each_hash_unions_the_aliases_of_every_entry_listing_it(self) -> None:
        one, both = _entry(1, _torrent(_H1)), _entry(2, _torrent(_H1), _torrent(_H2))
        entries = [_aliased(one, (501,), {7: 7, 13: 12}), _aliased(both, (502,), {15: 15, 16: 17})]

        read = fold_listings(entries, _NO_TAGS)

        assert dict(read.by_hash) == {
            _H1: torrent_listing(_window(one, 501), _window(both, 502), aliases={7: 7, 13: 12, 15: 15, 16: 17}),
            _H2: torrent_listing(_window(both, 502), aliases={15: 15, 16: 17}),
        }

    def test_a_tmdb_number_two_entries_pair_differently_is_dropped(self) -> None:
        one, two = _entry(1, _torrent(_H1)), _entry(2, _torrent(_H1))
        entries = [_aliased(one, (501,), {13: 12}), _aliased(two, (502,), {13: 14, 7: 7})]

        read = fold_listings(entries, _NO_TAGS)

        assert read.listing(_H1) == torrent_listing(_window(one, 501), _window(two, 502), aliases={7: 7})

    def test_entries_whose_aliases_count_in_different_tmdb_shows_leave_the_hash_without_aliases(self) -> None:
        # Each entry's TMDB numbers count in its own show, so the two alias maps aren't one numbering.
        one, two = _entry(1, _torrent(_H1)), _entry(2, _torrent(_H1))
        entries = [_aliased(one, (501,), {7: 7}, tmdb_id=900), _aliased(two, (502,), {8: 8}, tmdb_id=901)]

        read = fold_listings(entries, _NO_TAGS)

        assert read.listing(_H1) == torrent_listing(_window(one, 501), _window(two, 502))

    def test_an_entry_without_aliases_leaves_the_other_entrys_aliases_on_the_hash(self) -> None:
        one, two = _entry(1, _torrent(_H1)), _entry(2, _torrent(_H1))

        read = fold_listings([_aliased(one, (501,), {7: 7}), _listed(two, 502)], _NO_TAGS)

        assert read.listing(_H1) == torrent_listing(_window(one, 501), _window(two, 502), aliases={7: 7})

    def test_single_file_copies_identify_a_packs_specials(self) -> None:
        # One pack listed under three entries, and a hash-less single-file copy of each special under its own.
        # The sizes identify the specials, whatever the pack's names say.
        pack = _torrent(_H1, ("Show - S00E02.mkv", 700), ("Show - S00E03.mkv", 800))
        season = _entry(1, pack)
        special_b = _entry(2, pack, _torrent(None, ("Show - Special B.mkv", 800)))
        special_a = _entry(3, pack, _torrent(None, ("Show - Special A.mkv", 700)))

        read = fold_listings([_listed(season, 101, 102), _listed(special_b, 502), _listed(special_a, 501)], _NO_TAGS)

        assert read.identities is not None
        assert dict(read.identities) == {800: 502, 700: 501}
        assert read.listing(_H1) == torrent_listing(
            _window(season, 101, 102), _window(special_b, 502), _window(special_a, 501)
        )

    def test_a_size_two_entries_give_different_episodes_is_dropped(self) -> None:
        single = _torrent(None, ("Show - Special.mkv", 700))

        read = fold_listings([_listed(_entry(1, single), 7897), _listed(_entry(2, single), 7898)], _NO_TAGS)

        assert read.identities == {}

    def test_an_unread_window_leaves_the_identities_unread(self) -> None:
        # The unread entry's sizes are unknown, and one of them might clash with another entry's size.
        entries = [
            _listed(_entry(1, _torrent(None, ("Show - Special.mkv", 700))), 7897),
            EntryListing(_entry(2, _torrent(None, ("Show - Special.mkv", 700))), None, None),
        ]

        assert fold_listings(entries, _NO_TAGS).identities is None


class TestSizeIdentities:
    """A torrent with one video file, under an entry covering one episode, identifies that episode by size."""

    def test_a_lone_video_under_a_one_episode_window_identifies_its_episode(self) -> None:
        record = _entry(1, _torrent(_H1, ("Show - Special.mkv", 700)))

        assert list(size_identities(record, _ONE_SPECIAL, _NO_TAGS)) == [(700, 7898)]

    def test_one_video_beside_subtitles_is_a_single_file_listing(self) -> None:
        record = _entry(1, _torrent(_H1, ("Show - Special.mkv", 700), ("Show - Special.ass", 5), ("Fonts/a.ttf", 9)))

        assert list(size_identities(record, _ONE_SPECIAL, _NO_TAGS)) == [(700, 7898)]

    def test_two_videos_identify_nothing(self) -> None:
        record = _entry(1, _torrent(_H1, ("Show - 01.mkv", 700), ("Show - 02.mkv", 800)))

        assert list(size_identities(record, _ONE_SPECIAL, _NO_TAGS)) == []

    def test_a_wider_window_identifies_nothing(self) -> None:
        record = _entry(1, _torrent(_H1, ("Show - Special.mkv", 700)))

        assert list(size_identities(record, entry_window(1, (7897, 7898)), _NO_TAGS)) == []

    def test_a_torrent_with_an_ignored_tag_identifies_nothing(self) -> None:
        tagged = _torrent(_H1, ("Show - Special.mkv", 700), tags=frozenset({Tag.DOLBY_VISION}))
        record = _entry(1, tagged, _torrent(_H2, ("Show - Special v2.mkv", 800)))

        assert list(size_identities(record, _ONE_SPECIAL, frozenset({"dolby vision"}))) == [(800, 7898)]

    def test_every_tracker_counts_and_a_hash_less_torrent_too(self) -> None:
        private = make_torrent_record(
            tracker=Tracker.ANIMEBYTES, infohash=None, file_names=("Show - Special v2.mkv",), file_sizes=(800,)
        )
        record = _entry(1, _torrent(None, ("Show - Special.mkv", 700)), private)

        assert list(size_identities(record, _ONE_SPECIAL, _NO_TAGS)) == [(700, 7898), (800, 7898)]

    def test_a_zero_size_identifies_nothing(self) -> None:
        record = _entry(1, _torrent(_H1, ("Show - Special.mkv", 0)))

        assert list(size_identities(record, _ONE_SPECIAL, _NO_TAGS)) == []
