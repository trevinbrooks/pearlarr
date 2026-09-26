# pyright: strict
"""SeriesListings: reads every entry's record and window, and folds them once per series per run."""

from collections.abc import Mapping

from seadex import EntryRecord

from pearlarr.mappings import MappingEntry
from pearlarr.placement_types import TorrentListing
from pearlarr.seadex_types import SonarrEpisode
from pearlarr.torrent_listings import SeriesListings

from .builders import (
    FakeSeaDexSource,
    make_bare_instance,
    make_entry_record,
    make_sonarr_episodes,
    make_torrent_record,
    sonarr_ep,
)
from .fakes import ScriptedEpisodes

_SPECIALS = {n: sonarr_ep(0, n, ep_id=500 + n) for n in range(1, 7)}
_H1 = "a" * 40
_H2 = "b" * 40


def _entry(al_id: int, *hashes: str | None) -> EntryRecord:
    """Entry `al_id` listing one single-file torrent per hash, sized `al_id`."""

    torrents = tuple(
        make_torrent_record(url=f"https://nyaa.si/{al_id}{i}", infohash=h, file_names=("Show.mkv",), file_size=al_id)
        for i, h in enumerate(hashes)
    )
    return make_entry_record(anilist_id=al_id, url=f"https://releases.moe/{al_id}", torrents=torrents)


def _mappings(*al_ids: int) -> dict[int, MappingEntry]:
    return {al_id: MappingEntry(anilist_id=al_id) for al_id in al_ids}


def _window(*numbers: int) -> list[SonarrEpisode]:
    return [_SPECIALS[n] for n in numbers]


def _windows(
    windows: Mapping[int, list[SonarrEpisode] | None], *, ambiguous: frozenset[int] = frozenset()
) -> ScriptedEpisodes:
    """The episode collaborator scripted with one window per AniList id, raising for an `ambiguous` one."""

    return ScriptedEpisodes(windows=windows, ambiguous=ambiguous)


def _reader(seadex: FakeSeaDexSource, episodes: ScriptedEpisodes) -> SeriesListings:
    """A reader over the given entries and scripted windows, with no ignored tags."""

    return make_bare_instance(SeriesListings, _seadex=seadex, _episodes=episodes, _ignore_tags=frozenset(), _reads={})


class TestRead:
    """Reads every entry's record and window and folds them into per-hash listings and size identities."""

    def test_every_entry_is_read_and_folded(self) -> None:
        episodes = _windows({1: _window(1, 2), 2: _window(3), 3: _window(4)})
        seadex = FakeSeaDexSource({1: _entry(1, _H1), 2: _entry(2, _H1, _H2), 3: _entry(3, _H2)})

        read = _reader(seadex, episodes).read(7, _mappings(1, 2, 3))

        assert read is not None
        assert read.listing(_H1) == TorrentListing(frozenset({501, 502, 503}))
        assert read.listing(_H2) == TorrentListing(frozenset({503, 504}))
        # Entries 2 and 3 list lone files (sized 2 and 3) under one-episode windows. Entry 1's window holds two.
        assert read.identities == {2: 503, 3: 504}
        assert episodes.calls == [(7, 1), (7, 2), (7, 3)]

    def test_a_series_is_read_once_a_run(self) -> None:
        episodes = _windows({1: _window(1)})
        seadex = FakeSeaDexSource({1: _entry(1, _H1)})
        reader = _reader(seadex, episodes)

        first = reader.read(7, _mappings(1))
        again = reader.read(7, _mappings(1))

        assert again is first
        assert (seadex.entry_calls, episodes.calls) == ([1], [(7, 1)])
        reader.reset()
        assert reader.read(7, _mappings(1)) == first
        assert seadex.entry_calls == [1, 1]

    def test_a_mapped_id_without_an_entry_is_skipped(self) -> None:
        read = _reader(FakeSeaDexSource({1: _entry(1, _H1)}), _windows({1: _window(1)})).read(7, _mappings(1, 2))

        assert read is not None
        assert read.listing(_H1) == TorrentListing(frozenset({501}))

    def test_an_outage_reads_nothing_and_is_not_asked_again_this_run(self) -> None:
        seadex = FakeSeaDexSource({1: _entry(1, _H1)}, outage=True)
        reader = _reader(seadex, _windows({1: _window(1)}))

        assert reader.read(7, _mappings(1, 2)) is None
        assert reader.read(7, _mappings(1, 2)) is None
        assert seadex.entry_calls == [1, 2]

    def test_an_unread_window_marks_its_hashes_and_the_identities_unread(self) -> None:
        seadex = FakeSeaDexSource({1: _entry(1, _H1), 2: _entry(2, _H1, _H2)})

        read = _reader(seadex, _windows({1: None, 2: _window(3)})).read(7, _mappings(1, 2))

        assert read is not None
        assert (read.listing(_H1), read.listing(_H2)) == (None, TorrentListing(frozenset({503})))
        assert read.identities is None

    def test_an_ambiguous_mapping_lists_nothing(self) -> None:
        # The entry's own run already errors on it: its window is no part of the listing, which stays read.
        episodes = _windows({1: _window(1), 2: _window(3)}, ambiguous=frozenset({1}))
        seadex = FakeSeaDexSource({1: _entry(1, _H1), 2: _entry(2, _H1, _H2)})

        read = _reader(seadex, episodes).read(7, _mappings(1, 2))

        assert read is not None
        assert (read.listing(_H1), read.listing(_H2)) == (
            TorrentListing(frozenset({503})),
            TorrentListing(frozenset({503})),
        )
        assert read.identities == {2: 503}

    def test_a_zero_id_episode_is_no_window_id(self) -> None:
        seadex = FakeSeaDexSource({1: _entry(1, _H1)})

        read = _reader(seadex, _windows({1: [_SPECIALS[1], sonarr_ep(0, 9, ep_id=0)]})).read(7, _mappings(1))

        assert read is not None
        assert read.listing(_H1) == TorrentListing(frozenset({501}))


def test_no_entry_asks_nothing_of_the_real_resolver() -> None:
    # Built over the real collaborator types: with no entry resolved, every hash is listed nowhere and the
    # resolver (bare, no Sonarr behind it) is never asked.
    read = SeriesListings(FakeSeaDexSource(), make_sonarr_episodes(), frozenset()).read(7, {})

    assert read is not None
    assert read.listing(_H1) == TorrentListing(frozenset())
