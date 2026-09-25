# pyright: strict
"""The listing a numbered specials pack is judged by: the specials union over every entry of a series listing it."""

from collections.abc import Mapping

from seadex import EntryRecord

from pearlarr.mappings import MappingEntry
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
    """Entry `al_id` listing one torrent per hash."""

    torrents = tuple(make_torrent_record(url=f"https://nyaa.si/{al_id}{i}", infohash=h) for i, h in enumerate(hashes))
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


def _listings(
    entries: Mapping[int, EntryRecord], episodes: ScriptedEpisodes, *, outage: bool = False
) -> SeriesListings:
    """The reader over the entries served and the scripted windows."""

    seadex = FakeSeaDexSource(dict(entries), outage=outage)
    return make_bare_instance(SeriesListings, _seadex=seadex, _episodes=episodes)


class TestRead:
    """One union per hash asked: every listing entry's window folded, None once any of them could not be read."""

    def test_the_union_over_every_entry_listing_the_torrent(self) -> None:
        episodes = _windows({1: _window(1, 2), 2: _window(3), 3: _window(4)})
        reader = _listings({1: _entry(1, _H1), 2: _entry(2, _H1, _H2), 3: _entry(3, _H2)}, episodes)

        unions = reader.read(7, _mappings(1, 2, 3), [_H1, _H2])

        assert unions == {_H1: {501, 502, 503}, _H2: {503, 504}}
        assert episodes.calls == [(7, 1), (7, 2), (7, 3)]

    def test_a_hash_no_entry_lists_is_listed_nowhere(self) -> None:
        episodes = _windows({1: _window(1)})
        reader = _listings({1: _entry(1, "c" * 40)}, episodes)

        assert reader.read(7, _mappings(1), [_H1]) == {_H1: frozenset()}
        # An entry listing none of the hashes asked reads no window.
        assert episodes.calls == []

    def test_a_mapped_id_without_an_entry_is_skipped(self) -> None:
        reader = _listings({1: _entry(1, _H1)}, _windows({1: _window(1)}))

        assert reader.read(7, _mappings(1, 2), [_H1]) == {_H1: {501}}

    def test_an_outage_reads_every_hash_unread(self) -> None:
        reader = _listings({1: _entry(1, _H1)}, _windows({1: _window(1)}), outage=True)

        assert reader.read(7, _mappings(1, 2), [_H1, _H2]) == {_H1: None, _H2: None}

    def test_an_unread_window_marks_its_hashes_unread(self) -> None:
        reader = _listings({1: _entry(1, _H1), 2: _entry(2, _H1, _H2)}, _windows({1: None, 2: _window(3)}))

        assert reader.read(7, _mappings(1, 2), [_H1, _H2]) == {_H1: None, _H2: {503}}

    def test_an_ambiguous_mapping_lists_nothing(self) -> None:
        # The entry's own run already errors on it: its window is no part of the listing, which stays read.
        episodes = _windows({1: _window(1), 2: _window(3)}, ambiguous=frozenset({1}))
        reader = _listings({1: _entry(1, _H1), 2: _entry(2, _H1, _H2)}, episodes)

        assert reader.read(7, _mappings(1, 2), [_H1, _H2]) == {_H1: {503}, _H2: {503}}

    def test_hashes_match_by_their_one_spelling(self) -> None:
        # The listing spells the hash upper-case, and a torrent without a hash lists nothing.
        reader = _listings({1: _entry(1, _H1.upper(), None)}, _windows({1: _window(1)}))

        assert reader.read(7, _mappings(1), [_H1]) == {_H1: {501}}

    def test_a_zero_id_episode_is_no_window_id(self) -> None:
        reader = _listings({1: _entry(1, _H1)}, _windows({1: [_SPECIALS[1], sonarr_ep(0, 9, ep_id=0)]}))

        assert reader.read(7, _mappings(1), [_H1]) == {_H1: {501}}

    def test_nothing_asked_reads_nothing(self) -> None:
        episodes = _windows({1: _window(1)})
        seadex = FakeSeaDexSource({1: _entry(1, _H1)})
        reader = make_bare_instance(SeriesListings, _seadex=seadex, _episodes=episodes)

        assert reader.read(7, _mappings(1), []) == {}
        assert (seadex.entry_calls, episodes.calls) == ([], [])


def test_no_entry_asks_nothing_of_the_real_resolver() -> None:
    # Built over the real collaborator types: with no entry resolved, every hash is listed nowhere and the
    # resolver (bare, no Sonarr behind it) is never asked.
    reader = SeriesListings(FakeSeaDexSource(), make_sonarr_episodes())

    assert reader.read(7, {}, [_H1]) == {_H1: frozenset()}
