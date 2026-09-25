# pyright: strict
"""The import-time `FileEpisodeMapper`: on-disk leaves and a record's seed through the placer."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import pytest

from pearlarr.grab_placement import SeedFile, SeedRelease, SeedScope, build_pending_seed, place_release
from pearlarr.import_files import CandidateFile
from pearlarr.manual_import import EntryNames, PendingImport, normalize_basename
from pearlarr.placement_types import EpisodeIndex, Placement, PlacementVerdict, TargetScope
from pearlarr.seadex_types import EpisodeKey, ManualImportCandidate, ParsedFileInfo
from pearlarr.sonarr_mapper import FileAssignment, FileEpisodeMapper

from .builders import entry_facts, indexes_for, parsed_info, pending_import, series_index, url_item
from .fakes import FakeSonarrClient


@dataclass(frozen=True, slots=True)
class _Disk:
    """What the mapper finds on disk: Sonarr's parse of each leaf, and the leaves (the record's files when None)."""

    parse: Callable[[str], ParsedFileInfo | None]
    files: tuple[str, ...] | None = None


def _candidates(mapper: FileEpisodeMapper, names: Iterable[str]) -> dict[str, CandidateFile]:
    """The on-disk candidates for `names`, indexed as the mapper indexes Sonarr's listing."""

    return mapper.candidate_files([ManualImportCandidate(path=f"/dl/{name}") for name in names])


def _assign_on_disk(pending: PendingImport, disk: _Disk, id_by_key: Mapping[EpisodeKey, int]) -> FileAssignment:
    """One mapper poll over `disk` against a one-series map."""

    mapper = FileEpisodeMapper(FakeSonarrClient(parse_fn=disk.parse))
    candidates = _candidates(mapper, pending.seadex_files if disk.files is None else disk.files)
    return mapper.assign(pending, candidates, indexes_for(pending, series_index(id_by_key)))


class TestMapperSeam:
    """What a poll hands back: its fresh placements beside the seeded ones, with the record itself untouched."""

    def test_fully_seeded_record_skips_out_of_scope_on_disk_leftover(self) -> None:
        # Every resolved episode is seeded, and the folder also holds a season-2 file: it is another
        # slice's, never imported through an unscoped fallback, and the seed map stays as it was.
        seed_name, leftover_name = "Show - 01 [1080p].mkv", "Show - S02E01 [1080p].mkv"
        pending = pending_import(
            file_episode_map={seed_name: [101]}, ordered_episode_ids=[101], seadex_files=[seed_name]
        )
        disk = _Disk(lambda _f: parsed_info(season=2, episodes=(1,)), (seed_name, leftover_name))

        result = _assign_on_disk(pending, disk, {EpisodeKey(1, 1): 101, EpisodeKey(2, 1): 999})

        assert 999 not in {ep_id for ids in result.assigned.values() for ep_id in ids}
        assert result.excluded == (Placement(normalize_basename(leftover_name), (), PlacementVerdict.FOREIGN),)
        assert result.placed == {}
        assert dict(pending.file_episode_map) == {seed_name: (101,)}

    def test_assign_returns_placements_without_touching_the_record(self) -> None:
        name = "Show - S01E01 [1080p].mkv"
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name])

        result = _assign_on_disk(
            pending, _Disk(lambda _f: parsed_info(season=1, episodes=(1,))), {EpisodeKey(1, 1): 101}
        )

        assert result.placed == result.assigned == {normalize_basename(name): [101]}
        assert dict(pending.file_episode_map) == {}

    def test_placed_excludes_the_seeded_entries(self) -> None:
        # A seeded entry rides `assigned` only, so the seam never re-persists what the record holds.
        seed_name, leftover_name = "Show - S01E01 [1080p].mkv", "Show - S01E02 [1080p].mkv"
        parses = {
            seed_name: parsed_info(season=1, episodes=(1,)),
            leftover_name: parsed_info(season=1, episodes=(2,)),
        }
        pending = pending_import(
            file_episode_map={seed_name: [101]}, ordered_episode_ids=[101, 102], seadex_files=[seed_name, leftover_name]
        )

        result = _assign_on_disk(pending, _Disk(parses.get), {EpisodeKey(1, 1): 101, EpisodeKey(1, 2): 102})

        assert result.placed == {normalize_basename(leftover_name): [102]}
        assert result.assigned == {normalize_basename(seed_name): [101], normalize_basename(leftover_name): [102]}


class TestSeededSharerTell:
    """A seeded file's parse still feeds the duplicate tell: a version sharing its absolute never takes a spare id."""

    _V1: ClassVar[str] = "Show - 12 [1080p].mkv"
    _V2: ClassVar[str] = "Show - 12v2 [1080p].mkv"
    _TWELVE: ClassVar[ParsedFileInfo] = parsed_info(season=0, absolutes=(12,), matched=((1, 12),))

    def test_placed_sharer_still_vetoes_on_the_next_poll(self) -> None:
        # Poll 1 places the v2 and the record folds it in. Poll 2, on the same mapper and parse cache,
        # must not let the now-seeded v2 hide the shared absolute from the tell.
        mapper = FileEpisodeMapper(FakeSonarrClient(parse_fn={self._V1: self._TWELVE, self._V2: self._TWELVE}.get))
        pending = pending_import(
            file_episode_map={}, ordered_episode_ids=[2586, 2587], seadex_files=[self._V1, self._V2]
        )
        candidates = _candidates(mapper, (self._V1, self._V2))
        indexes = indexes_for(pending, series_index({EpisodeKey(1, 12): 2586}))

        first = mapper.assign(pending, candidates, indexes)
        second = mapper.assign(pending.with_placements(first.placed), candidates, indexes)

        assert first.placed == {normalize_basename(self._V2): [2586]}
        assert normalize_basename(self._V1) not in second.assigned
        assert second.excluded == (Placement(normalize_basename(self._V1), (), PlacementVerdict.DUPLICATE),)
        assert dict(pending.file_episode_map) == {}

    def test_seeded_sharer_parse_blip_fails_closed(self) -> None:
        # A later run's parse of the seeded v1 blips, so the tell's input is incomplete and the v2 stays
        # refused. Nothing proves it a duplicate either, so it is a skip, re-asked next poll.
        pending = pending_import(
            file_episode_map={self._V1: [2586]}, ordered_episode_ids=[2586, 2587], seadex_files=[self._V1, self._V2]
        )

        result = _assign_on_disk(pending, _Disk({self._V2: self._TWELVE}.get), {EpisodeKey(1, 12): 2586})

        assert normalize_basename(self._V2) not in result.assigned
        assert result.skipped == (normalize_basename(self._V2),)
        assert not result.settled

    def test_seeded_dual_numbered_sharer_offline_fallback_fails_closed(self) -> None:
        # The seeded v1 is dual-numbered. Its parse blips and the offline stand-in loses the absolute,
        # which the tell treats as unknown rather than letting the v2 slide onto the spare id.
        v1 = "Show - S01E12 - 12 [1080p].mkv"
        pending = pending_import(
            file_episode_map={v1: [2586]}, ordered_episode_ids=[2586, 2587], seadex_files=[v1, self._V2]
        )

        result = _assign_on_disk(
            pending, _Disk({self._V2: self._TWELVE}.get), {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}
        )

        assert normalize_basename(self._V2) not in result.assigned
        assert result.excluded == (Placement(normalize_basename(self._V2), (), PlacementVerdict.DUPLICATE),)

    def test_moved_out_seeded_sharer_still_vetoes(self) -> None:
        # The seeded v1 already imported and left the folder. Sonarr's parse is name-based, so the
        # tell keeps seeing absolute 12 and refuses the v2 the spare id.
        pending = pending_import(
            file_episode_map={self._V1: [2586]}, ordered_episode_ids=[2586, 2999], seadex_files=[self._V1, self._V2]
        )
        disk = _Disk({self._V1: self._TWELVE, self._V2: self._TWELVE}.get, (self._V2,))

        result = _assign_on_disk(pending, disk, {EpisodeKey(1, 12): 2586})

        assert normalize_basename(self._V2) not in result.assigned
        assert result.excluded == (Placement(normalize_basename(self._V2), (), PlacementVerdict.DUPLICATE),)

    def test_none_parse_v2_never_rides_the_single_file_fallback(self) -> None:
        # The blip lands on the v2 itself: no parse is no evidence, so the spare id stays open.
        pending = pending_import(
            file_episode_map={self._V1: [2586]}, ordered_episode_ids=[2586, 2587], seadex_files=[self._V1, self._V2]
        )

        result = _assign_on_disk(pending, _Disk({self._V1: self._TWELVE}.get), {EpisodeKey(1, 12): 2586})

        assert normalize_basename(self._V2) not in result.assigned
        assert normalize_basename(self._V2) in result.skipped


class TestAssignDuplicateLeaves:
    """One basename in two folders collapses in the basename-keyed pool.

    Only one physical file can ever import, so the unmatched warning must
    follow the map: a placed name is never also reported skipped, and an
    unplaced one is reported once.
    """

    def test_placed_duplicate_leaf_is_not_reported_skipped(self) -> None:
        # The second occurrence of a placed name defers off the used set, and must not surface as a skip.
        name = "Show - 01 [1080p].mkv"
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name, name])

        result = _assign_on_disk(
            pending, _Disk(lambda _f: parsed_info(season=1, episodes=(1,))), {EpisodeKey(1, 1): 101}
        )

        assert result.assigned == {normalize_basename(name): [101]}
        assert result.placed == {normalize_basename(name): [101]}
        assert result.skipped == ()

    def test_unplaced_duplicate_leaf_is_reported_once(self) -> None:
        name = "Extra.mkv"
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], seadex_files=[name, name])

        result = _assign_on_disk(pending, _Disk(lambda _f: parsed_info()), {})

        assert result.assigned == {}
        assert result.skipped == (normalize_basename(name),)


class TestAssignSettled:
    """`settled`: a skip is a verdict only when every parse was served and the episode index was."""

    _PAIR: ClassVar[tuple[str, ...]] = ("Movie Part 1.mkv", "Movie Part 2.mkv")

    @pytest.mark.parametrize(
        ("parse", "id_by_key", "settled"),
        [
            pytest.param(None, {EpisodeKey(1, 1): 101}, False, id="a parse miss"),
            # A failed episode fetch serves an empty index: the exact pass could not have matched anything.
            pytest.param(parsed_info(), {}, False, id="an empty episode index"),
            pytest.param(parsed_info(), {EpisodeKey(1, 1): 101}, True, id="served parses over a served index"),
        ],
    )
    def test_a_skip_settles_only_over_served_parses_and_a_served_index(
        self, parse: ParsedFileInfo | None, id_by_key: dict[EpisodeKey, int], settled: bool
    ) -> None:
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=list(self._PAIR))

        result = _assign_on_disk(pending, _Disk(lambda _f: parse), id_by_key)

        assert sorted(result.skipped) == sorted(normalize_basename(name) for name in self._PAIR)
        assert result.settled is settled

    def test_a_fully_seeded_batch_is_settled_without_a_parse(self) -> None:
        name = "Show - 01 [1080p].mkv"
        pending = pending_import(file_episode_map={name: [101]}, ordered_episode_ids=[101])
        sonarr = FakeSonarrClient()
        mapper = FileEpisodeMapper(sonarr)

        result = mapper.assign(
            pending, _candidates(mapper, (name,)), indexes_for(pending, series_index({EpisodeKey(1, 1): 101}))
        )

        assert result.skipped == ()
        assert result.settled is True
        assert sonarr.parse_calls == []


class TestSeedEqualsMapper:
    """One batch, two entry points: the grab-time seed and the import-time mapper agree file for file."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(1, 1): 601,
    }
    _NAMES: ClassVar[list[str]] = ["Show - S00E01 [1080p].mkv", "Show - S00E02 [1080p].mkv", "Show - S01E01 [BD].mkv"]

    @classmethod
    def _parses(cls) -> dict[str, ParsedFileInfo | None]:
        return {
            cls._NAMES[0]: parsed_info(season=0, episodes=(1,)),
            cls._NAMES[1]: parsed_info(season=0, episodes=(2,)),
            cls._NAMES[2]: parsed_info(season=1, episodes=(1,)),
        }

    @classmethod
    def _index(cls) -> EpisodeIndex:
        return series_index({EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502})

    def test_seed_scope_targets_the_entrys_ids_over_the_series_map(self) -> None:
        scope = SeedScope(1, self._index(), series_index(self._MAP), EntryNames())

        assert scope.target() == TargetScope([501, 502], series_index(self._MAP))

    def test_the_seed_and_the_mapper_place_and_exclude_alike(self) -> None:
        parses = self._parses()
        scope = SeedScope(1, self._index(), series_index(self._MAP), EntryNames())
        release = SeedRelease(
            release_group="grp",
            url_item=url_item(url="u", infohash="h"),
            infohash="h",
            placed=place_release([SeedFile(name, 1000, parses[name]) for name in self._NAMES], scope, None),
        )

        seed = build_pending_seed(release, scope, entry_facts(al_id=1, series_id=2, title="t"))
        pending = pending_import(
            file_episode_map={}, ordered_episode_ids=list(scope.entry.by_id), seadex_files=self._NAMES
        )
        live = _assign_on_disk(pending, _Disk(parses.get), self._MAP)

        assert seed.placements == live.assigned
        assert seed.excluded == tuple(p.name for p in live.excluded)
        # A real placement, not two empty maps agreeing.
        assert seed.placements == {
            normalize_basename(self._NAMES[0]): [501],
            normalize_basename(self._NAMES[1]): [502],
        }
        assert seed.excluded == (normalize_basename(self._NAMES[2]),)
        # The record the pipeline persists carries the same map and exclusions.
        record = seed.record_at("2026-01-01 00:00:00", fresh=True)
        assert dict(record.file_episode_map) == {name: tuple(ids) for name, ids in live.assigned.items()}
        assert record.excluded_files == seed.excluded
