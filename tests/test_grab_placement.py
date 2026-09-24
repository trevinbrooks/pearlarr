# pyright: strict
"""The grab-time placement: what `place_release` records for the planner, and what the seeds fold from it."""

from pearlarr.grab_placement import (
    EntryPlacements,
    PendingSeedContext,
    SeedFile,
    SeedScope,
    UrlPlacement,
    build_pending_seeds,
    place_release,
)
from pearlarr.manual_import import EntryNames, normalize_basename, normalized_leaf
from pearlarr.placement_types import EpisodeAssignment, PlacementVerdict, episode_index
from pearlarr.seadex_types import EpisodeRecord, MatchedEpisode, ParsedFileInfo, SeadexDict, SonarrEpisode

from .builders import rg_group, sonarr_ep, url_item

_SEASON = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0) for n in range(1, 5)]
_SPECIAL = sonarr_ep(0, 1, ep_id=501, episode_file_id=0)
_SERIES = [
    _SPECIAL,
    *_SEASON,
    sonarr_ep(2, 1, ep_id=201, episode_file_id=0),
    sonarr_ep(2, 2, ep_id=202, episode_file_id=0),
]
_RUN = [f"Show - {n:02d}.mkv" for n in range(1, 5)]


def _scope(entry: list[SonarrEpisode], names: EntryNames | None = None) -> SeedScope:
    return SeedScope(episode_index(entry), episode_index(_SERIES), names or EntryNames())


def _bare(number: int) -> ParsedFileInfo:
    """A `- 01` name Sonarr matched nothing for."""

    return ParsedFileInfo(episode_numbers=(number,))


def _matched(season: int, episode: int) -> ParsedFileInfo:
    return ParsedFileInfo(
        season_number=season,
        episode_numbers=(episode,),
        matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),),
    )


def _verdicts(placement: UrlPlacement) -> set[PlacementVerdict]:
    return {p.verdict for p in placement.assignment.placements}


class TestPlaceRelease:
    """One url's files placed by the import's own passes, its records the planner's coverage vocabulary."""

    def test_a_run_sonarr_read_nothing_of_records_the_window(self) -> None:
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON))

        assert placement.records == tuple(EpisodeRecord(1, n, 10 * n) for n in range(1, 5))
        assert placement.assignment.assigned == {normalized_leaf(name): [100 + n] for n, name in enumerate(_RUN, 1)}
        assert placement.parses_known

    def test_a_whole_series_pack_records_only_the_entrys_slice(self) -> None:
        names = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)] + ["Show - S02E01.mkv", "Show - S02E02.mkv"]
        parses = [_matched(1, n) for n in range(1, 5)] + [_matched(2, 1), _matched(2, 2)]
        files = [SeedFile(name, 1, parse) for name, parse in zip(names, parses, strict=True)]

        placement = place_release(files, _scope(_SEASON))

        assert placement.records == tuple(EpisodeRecord(1, n, 1) for n in range(1, 5))
        assert {p.name for p in placement.assignment.excluded} == {"show - s02e01.mkv", "show - s02e02.mkv"}

    def test_a_file_placed_nowhere_leaves_no_record(self) -> None:
        files = [SeedFile("Show - S02E01.mkv", 1, _matched(2, 1))]

        placement = place_release(files, _scope(_SEASON))

        assert placement.records == ()
        assert _verdicts(placement) == {PlacementVerdict.FOREIGN}

    def test_a_held_run_records_nothing_and_its_parses_are_not_known(self) -> None:
        # One parse request failed: the run is held for import time, and the title re-checks next run.
        files = [SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON))

        assert placement.records == ()
        assert not placement.parses_known
        assert _verdicts(placement) == {PlacementVerdict.HELD}

    def test_a_file_beside_a_held_run_still_places(self) -> None:
        files = [SeedFile("Show - OVA.mkv", 5, _matched(0, 1))] + [
            SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _scope([_SPECIAL, *_SEASON]))

        assert placement.records == (EpisodeRecord(0, 1, 5),)
        assert not placement.parses_known

    def test_two_files_sharing_a_leaf_keep_their_own_sizes(self) -> None:
        # The gather already stripped the folders: one leaf, placed once, two listed sizes.
        files = [SeedFile("Show - S01E01.mkv", 10, _matched(1, 1)), SeedFile("Show - S01E01.mkv", 20, _matched(1, 1))]

        placement = place_release(files, _scope(_SEASON))

        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 1, 20))
        assert placement.assignment.assigned == {"show - s01e01.mkv": [101]}

    def test_a_scope_that_cannot_place_records_nothing(self) -> None:
        # An unread series map: the seed waits for import time, and the planner judges by group and size.
        files = [SeedFile(name, 1, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        scope = SeedScope(episode_index(_SEASON), episode_index([]), EntryNames())

        placement = place_release(files, scope)

        assert not scope.can_place
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.parses_known

    def test_no_files_is_an_empty_placement_with_its_parses_known(self) -> None:
        placement = place_release([], _scope(_SEASON))

        assert placement == UrlPlacement((), EpisodeAssignment(()), (), True)


class TestEntryPlacements:
    """The per-entry fold: records onto the release dict, the groups a failed parse held, and the seeds."""

    @staticmethod
    def _entry(names: EntryNames | None = None) -> tuple[EntryPlacements, SeadexDict]:
        scope = _scope(_SEASON, names)
        run = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        held = [SeedFile(name, 1, None if n == 1 else _bare(n)) for n, name in enumerate(_RUN, start=1)]
        placed = EntryPlacements.place(scope, {"u1": run, "u2": held, "u3": []})
        seadex_dict: SeadexDict = {
            "RG": rg_group({"u1": url_item(url="u1", files=_RUN, size=[10, 20, 30, 40], infohash="h1", download=True)}),
            "Held": rg_group({"u2": url_item(url="u2", files=_RUN, size=[1] * 4, infohash="h2", download=True)}),
            "Subs": rg_group({"u3": url_item(url="u3", files=["Show.ass"], size=[1], infohash="h3", download=True)}),
        }
        return placed, seadex_dict

    def test_attach_writes_each_urls_records_and_the_group_union(self) -> None:
        placed, seadex_dict = self._entry()

        placed.attach_records(seadex_dict)

        run_records = [EpisodeRecord(1, n, 10 * n) for n in range(1, 5)]
        assert seadex_dict["RG"].urls["u1"].episodes == run_records
        assert seadex_dict["RG"].all_episodes == run_records
        assert seadex_dict["Held"].urls["u2"].episodes == []
        assert seadex_dict["Held"].all_episodes == []
        assert seadex_dict["Subs"].all_episodes == []

    def test_parse_failed_groups_name_the_held_release(self) -> None:
        placed, seadex_dict = self._entry()

        assert placed.parse_failed_groups(seadex_dict) == ("Held",)

    def test_seeds_fold_the_placement_and_carry_the_scopes_names(self) -> None:
        names = EntryNames("Show", ("Show", "Shou"))
        placed, seadex_dict = self._entry(names)
        entry = PendingSeedContext(al_id=1, series_id=7, title="Show", added_at="2026-01-01 00:00:00")

        seeds = build_pending_seeds(seadex_dict, placed, entry)

        # The subs-only url seeds nothing; the held one is seeded with an empty map for import time.
        assert set(seeds) == {"h1", "h2"}
        assert dict(seeds["h1"].file_episode_map) == {
            normalize_basename(name): (100 + n,) for n, name in enumerate(_RUN, 1)
        }
        assert seeds["h1"].seadex_files == tuple(_RUN)
        assert seeds["h1"].claims[0].names == names
        assert dict(seeds["h2"].file_episode_map) == {}
