# pyright: strict
"""The `(season, episode) -> id` index, a batch's copies and parse hinge, and the target scope's own rules."""

from collections.abc import MutableMapping
from typing import cast

import pytest

from pearlarr.episode_titles import EpisodeTitles
from pearlarr.manual_import import FileEpisodeMap
from pearlarr.placement_types import (
    EMPTY_LISTING,
    EpisodeIndex,
    ListingEvidence,
    PlacementBatch,
    PlacementVerdict,
    TargetScope,
    all_specials,
    episode_index,
)
from pearlarr.seadex_types import SONARR_MISSING_KEY, EpisodeKey, ParsedFileInfo, SonarrEpisode
from pearlarr.tvdb_orders import OrderRow, SeriesOrdering

from .builders import entry_window, parsed_info, series_index, sonarr_ep, torrent_listing


class TestEpisodeIndex:
    """`episode_index` folds an episode fetch into the import family's three facets.

    `id_by_key` maps `(season, episode)` to the first episode id, missing numbers
    folding to a sentinel key (no collision with real pairs). Zero-id episodes are
    dropped from every facet before keying. `by_id` keeps the fetch order: the
    resolved set the add flow persists rides `list(by_id)`. `orders` reads each
    other TVDB order onto the same ids through the episodes' TVDB ids. The
    lookups (`key_by_id` and the rest) derive from the facets.
    """

    def test_normal_seasoned_episodes(self) -> None:
        eps = [
            sonarr_ep(1, 1, ep_id=11, episode_file_id=0),
            sonarr_ep(1, 2, ep_id=12, episode_file_id=0),
            sonarr_ep(2, 1, ep_id=21, episode_file_id=0),
        ]
        index = episode_index(eps)
        assert index.id_by_key == {(1, 1): 11, (1, 2): 12, (2, 1): 21}
        assert tuple(index.by_id) == (11, 12, 21)

    def test_missing_season_and_episode_use_sentinel_no_collision(self) -> None:
        eps = [
            sonarr_ep(None, None, ep_id=5, episode_file_id=0),
            sonarr_ep(1, 1, ep_id=6, episode_file_id=0),
        ]
        result = episode_index(eps).id_by_key
        assert result[EpisodeKey(SONARR_MISSING_KEY, SONARR_MISSING_KEY)] == 5
        assert result[EpisodeKey(1, 1)] == 6

    def test_first_wins_on_duplicate_key(self) -> None:
        eps = [sonarr_ep(1, 1, ep_id=7, episode_file_id=0), sonarr_ep(1, 1, ep_id=8, episode_file_id=0)]
        assert episode_index(eps).id_by_key == {(1, 1): 7}

    def test_zero_id_dropped_before_keying(self) -> None:
        # A real-id twin behind a zero-id record must still win its key.
        eps = [sonarr_ep(1, 2, ep_id=0, episode_file_id=0), sonarr_ep(1, 2, ep_id=9, episode_file_id=0)]
        index = episode_index(eps)
        assert index.id_by_key == {(1, 2): 9}
        assert tuple(index.by_id) == (9,)

    def test_the_title_and_absolute_number_ride_the_record(self) -> None:
        raw = {"id": 6, "seasonNumber": 1, "episodeNumber": 1, "absoluteEpisodeNumber": 13, "title": "Beach Day"}
        ep = SonarrEpisode.model_validate({**raw, "episodeFileId": 0})
        bare = SonarrEpisode.model_validate({"id": 6, "seasonNumber": 1, "episodeNumber": 1, "episodeFileId": 0})

        assert (ep.absolute_episode_number, ep.title) == (13, "Beach Day")
        assert (bare.absolute_episode_number, bare.title) == (None, "")

    def test_another_order_reads_onto_the_episodes_by_tvdb_id(self) -> None:
        # The dvd order splits S01E01 in two and has an episode Sonarr lacks. The repeated key takes its first row.
        eps = [sonarr_ep(1, 1, ep_id=11, tvdb_id=901), sonarr_ep(1, 2, ep_id=12, tvdb_id=902)]
        dvd = (
            OrderRow(1, 1, 902, None),
            OrderRow(1, 1, 901, None),
            OrderRow(1, 2, 903, None),
        )

        index = episode_index(eps, SeriesOrdering({"dvd": dvd}, None))

        assert index.orders["dvd"].id_by_key == {(1, 1): 12}
        assert index.orders["dvd"].keys == {(1, 1), (1, 2)}
        assert episode_index(eps).orders == {}

    def test_facets_detach_and_reject_mutation(self) -> None:
        source = {EpisodeKey(1, 1): 11}
        index = EpisodeIndex(by_id={}, id_by_key=source)

        source[EpisodeKey(1, 2)] = 12

        assert EpisodeKey(1, 2) not in index.id_by_key
        with pytest.raises(TypeError):
            cast("MutableMapping[EpisodeKey, int]", index.id_by_key)[EpisodeKey(1, 3)] = 13

    def test_the_lookups_are_derived_from_the_facets(self) -> None:
        keys = {EpisodeKey(0, 1): 501, EpisodeKey(1, 1): 601, EpisodeKey(1, 2): 602}
        index = series_index(keys, absolutes={601: 1, 602: 2})

        assert index.key_by_id == {501: EpisodeKey(0, 1), 601: EpisodeKey(1, 1), 602: EpisodeKey(1, 2)}
        assert index.season_counts == {0: 1, 1: 2}
        assert index.absolute_of == {601: 1, 602: 2}
        assert index.id_by_absolute == {1: 601, 2: 602}


class TestEpisodeIndexViews:
    """`EpisodeIndex.titles` builds the episode titles once per index and set of series words."""

    _WORDS = frozenset({"show"})

    def _titled_index(self) -> EpisodeIndex:
        return series_index(
            {EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12}, titles={11: "Show: The Long Road Home", 12: "A Quiet Morning"}
        )

    def test_one_index_hands_out_one_view_per_word_set(self) -> None:
        index = self._titled_index()

        view = index.titles(self._WORDS)

        assert index.titles(self._WORDS) is view
        assert index.titles(frozenset({"other"})) is not view

    def test_a_twin_index_builds_its_own_view(self) -> None:
        index, twin = self._titled_index(), self._titled_index()

        assert twin.titles(self._WORDS) is not index.titles(self._WORDS)

    def test_the_view_is_the_fold_of_the_index(self) -> None:
        index = self._titled_index()

        assert index.titles(self._WORDS) == EpisodeTitles.of(index.by_id, index.id_by_key, self._WORDS)


class TestPlacementBatchParsesKnown:
    """`all_parses_known`: one file's missing parse makes the batch's parses unknown."""

    def test_a_transport_miss_is_unknown(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"a.mkv": parsed_info(), "b.mkv": None}
        assert PlacementBatch(["a.mkv", "b.mkv"], parsed).all_parses_known is False

    def test_served_parses_are_known(self) -> None:
        # A numberless answer from Sonarr is a real answer: known, even though it places nothing.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(),
            "b.mkv": parsed_info(season=1, episodes=(1,)),
        }
        assert PlacementBatch(["a.mkv", "b.mkv"], parsed).all_parses_known is True

    def test_an_empty_batch_is_known(self) -> None:
        # A fully seeded record parses nothing, and nothing is missing.
        assert PlacementBatch([], {}).all_parses_known is True


class TestPlacementBatchCopies:
    """The batch copies the caller's names, parses and map, the map's ids as tuples, and wraps them read-only."""

    def test_the_fields_are_copied_and_read_only(self) -> None:
        # The ids go in as a list through a cast, as an untyped caller can hand one: the `(11,)` below is the
        # batch's own tupling.
        ids = [11]
        to_place = ["a.mkv"]
        parsed: dict[str, ParsedFileInfo | None] = {"a.mkv": parsed_info()}
        batch = PlacementBatch(to_place, parsed, cast("FileEpisodeMap", {"b.mkv": ids}))

        to_place.append("c.mkv")
        parsed["c.mkv"] = None
        ids.append(12)

        assert (batch.to_place, dict(batch.parsed), dict(batch.seeded)) == (
            ("a.mkv",),
            {"a.mkv": parsed_info()},
            {"b.mkv": (11,)},
        )
        with pytest.raises(TypeError):
            cast("MutableMapping[str, tuple[int, ...]]", batch.seeded)["c.mkv"] = (12,)
        with pytest.raises(TypeError):
            cast("MutableMapping[str, ParsedFileInfo | None]", batch.parsed)["c.mkv"] = None


class TestPlacementBatchNames:
    """`names` is what the placer places: the batch's distinct names less the ones its map holds."""

    def test_a_mapped_name_is_never_placed(self) -> None:
        batch = PlacementBatch(["a.mkv", "b.mkv"], {}, {"a.mkv": (1,)})

        assert batch.names == ("b.mkv",)


class TestTargetScope:
    """The scope admits its resolved ids, or any when unscoped, and `using` takes only the ids it admits."""

    _SERIES = series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12, EpisodeKey(2, 1): 21})

    def test_using_takes_the_admitted_ids_and_detaches_the_resolved_list(self) -> None:
        resolved = [11, 12, 0]
        scope = TargetScope(resolved, self._SERIES, used=frozenset({11}))

        resolved.append(21)
        narrowed = scope.using([12, 21, 0])

        assert scope.resolved == (11, 12, 0)
        assert narrowed.used == {11, 12}
        assert (narrowed.resolved, narrowed.real_ids) == (scope.resolved, scope.real_ids)

    def test_an_unscoped_scope_uses_every_id(self) -> None:
        assert TargetScope([], self._SERIES).using([12, 21]).used == {12, 21}

    def test_using_keeps_the_listing(self) -> None:
        listing = ListingEvidence(torrent_listing(entry_window(1, (11, 12))), {"a.mkv": 11})
        scope = TargetScope([11], self._SERIES, listing=listing)

        assert scope.using([12]).listing == listing


class TestTorrentListing:
    """The listing's windows, one per AniList id in id order, and its aliases copied from the caller."""

    def test_two_windows_under_one_anilist_id_keep_the_first(self) -> None:
        first, second, other = entry_window(1, (501,)), entry_window(1, (502,)), entry_window(2, (503,))

        assert torrent_listing(other, first, second).entries == (first, other)

    def test_a_change_to_the_callers_dict_leaves_the_listing_as_it_was(self) -> None:
        aliases = {13: 12}
        listing = torrent_listing(entry_window(1, (501,)), aliases=aliases)

        aliases[14] = 13

        assert dict(listing.special_aliases) == {13: 12}


class TestListingEvidence:
    """The identified names are copied from the caller's map and wrapped read-only."""

    def test_the_identified_names_are_copied_and_read_only(self) -> None:
        identified = {"a.mkv": 11}
        evidence = ListingEvidence(EMPTY_LISTING, identified)
        identified["b.mkv"] = 12

        assert evidence.identified == {"a.mkv": 11}
        with pytest.raises(TypeError):
            cast("dict[str, int]", evidence.identified)["c.mkv"] = 13

    def test_the_ids_and_aliases_come_from_the_listing(self) -> None:
        listing = torrent_listing(entry_window(1, (501,)), aliases={13: 12})

        evidence = ListingEvidence(listing)

        assert (evidence.listing, evidence.ids, dict(evidence.special_aliases)) == (listing, {501}, {13: 12})


class TestPlacementVerdict:
    """A misnumbered file is neither placed nor excluded: it stays a leftover the summary lists for a hand import."""

    def test_misnumbered_is_an_open_leftover(self) -> None:
        verdict = PlacementVerdict.MISNUMBERED

        assert not verdict.placed
        assert not verdict.excluded
        assert verdict.refused

    def test_refused_is_held_or_misnumbered(self) -> None:
        # Both decide a name without ids: no later window may place it by its numbers.
        assert {v for v in PlacementVerdict if v.refused} == {PlacementVerdict.HELD, PlacementVerdict.MISNUMBERED}

    def test_a_hand_import_is_misnumbered_or_refuted_by_its_titles(self) -> None:
        hand_import = {PlacementVerdict.MISNUMBERED, PlacementVerdict.TITLE_REFUTED}

        assert {v for v in PlacementVerdict if v.hand_import} == hand_import

    def test_another_listed_titles_file_is_excluded_and_no_hand_import(self) -> None:
        # It waits for the other title's own grab to claim the torrent, so this record skips it and it's no hand import.
        verdict = PlacementVerdict.LISTED_ELSEWHERE

        assert verdict.excluded
        assert not verdict.placed
        assert not verdict.hand_import

    def test_a_size_match_or_a_special_alias_places(self) -> None:
        # A file placed by size, or through a specials pack's aliases, carries ids like any other placement.
        for verdict in (PlacementVerdict.IDENTIFIED, PlacementVerdict.OVERRIDDEN, PlacementVerdict.ALTERNATE):
            assert verdict.placed
            assert not verdict.excluded
            assert not verdict.refused


class TestAllSpecials:
    """Whether every id is one of the series' specials: the shape a listing must have to judge a specials pack."""

    _SERIES = series_index({EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(1, 1): 11})

    def test_specials_only(self) -> None:
        assert all_specials(self._SERIES, [501, 502])

    def test_a_seasoned_id_is_no_special(self) -> None:
        assert not all_specials(self._SERIES, [501, 11])

    def test_an_unknown_id_is_no_special(self) -> None:
        assert not all_specials(self._SERIES, [501, 999])

    def test_no_ids_vacuously(self) -> None:
        assert all_specials(self._SERIES, [])


class TestSpecialsListing:
    """The listing a numbered pack is judged by: read, all specials, covering a scoped window of specials."""

    _SERIES = series_index({EpisodeKey(0, n): 500 + n for n in range(1, 5)} | {EpisodeKey(1, 1): 11})

    def _scope(self, resolved: list[int], listed: tuple[int, ...] = ()) -> TargetScope:
        return TargetScope(resolved, self._SERIES, listing=ListingEvidence(torrent_listing(entry_window(1, listed))))

    def test_a_listing_covering_a_window_of_specials_judges(self) -> None:
        assert self._scope([502, 504], (502, 503, 504)).specials_listing() == {502, 503, 504}

    def test_a_listing_leaving_a_window_special_out_stands_down(self) -> None:
        assert self._scope([501, 502], (502, 504)).specials_listing() is None

    def test_a_listing_with_a_seasoned_id_stands_down(self) -> None:
        assert self._scope([501], (501, 11)).specials_listing() is None

    def test_no_listing_stands_down(self) -> None:
        assert self._scope([501]).specials_listing() is None

    def test_an_unscoped_window_stands_down(self) -> None:
        assert self._scope([], (501,)).specials_listing() is None

    def test_a_specials_window_is_scoped_and_specials_only(self) -> None:
        assert self._scope([501, 502]).specials_window
        assert not self._scope([501, 11]).specials_window
        assert not self._scope([]).specials_window
