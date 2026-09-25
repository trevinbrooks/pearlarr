# pyright: strict
"""Tests over JSON captured verbatim from a live Sonarr (`tests/fixtures/sonarr/`).

Quality, `ParsedFileInfo`, the queue, the command list, and the specials end to end through `import_completed`.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pearlarr.config import AppConfig
from pearlarr.import_quality import ParsedQuality, quality_axes_from_model, resolve_quality
from pearlarr.manual_import import AttemptKind, ImportProgress, PendingImport, normalize_basename
from pearlarr.probe_verdicts import (
    ContentPaths,
    DownloadMatch,
    QueueVerdict,
    classify_queue,
    manual_import_in_flight,
    started_disk_commands,
)
from pearlarr.release_names import parse_se_from_filename
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import (
    CommandResource,
    ManualImportCandidate,
    ParsedFileInfo,
    QualityDefinition,
    QualitySource,
    QueueRecord,
    SonarrEpisode,
)

from .builders import FakeCacheStore, make_config, make_sonarr_sync, pending_import
from .fakes import FakeSonarrClient

_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"
_BODY: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _load_models[M: BaseModel](name: str, model: type[M]) -> list[M]:
    """One captured list body, each item validated as the client boundary does."""

    items: list[object] = json.loads(_read(name))
    return [model.model_validate(item) for item in items]


def _load_parse(name: str) -> ParsedFileInfo:
    return ParsedFileInfo.model_validate_json(_read(name))


class _QueuePage(BaseModel):
    """The captured `/queue` page: only its records are read."""

    records: list[QueueRecord]


class TestQualityResolution:
    """Quality is matched by the structured `(source, resolution)` pair.

    The candidate read runs on a live capture. `qualitydefinitions.json` is a hand-authored stand-in mirroring
    real Sonarr, so dropping a live `/api/v3/qualitydefinition` capture in its place re-runs these unchanged.
    """

    def test_qualitydefinition_fixture_has_the_shape_the_matcher_needs(self) -> None:
        # A contract on the stand-in (or a capture swapped in for it), not proof the live instance
        # serializes both fields: the matcher keys on (source, resolution), so every definition carries both.
        defs = _load_models("qualitydefinitions.json", QualityDefinition)
        assert defs
        for definition in defs:
            quality = definition.quality
            assert quality is not None
            assert isinstance(quality.resolution, int)
            assert isinstance(quality.source, str)
            if quality.name != "Unknown":
                assert QualitySource.parse(quality.source) is not None

    def test_bd_remux_resolves_against_full_def_list(self) -> None:
        # Sonarr parses a 1080p BD remux as (blurayRaw, 1080): against the full list that pair
        # resolves to the "Bluray-1080p Remux" definition, never an omitted quality.
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _load_models("qualitydefinitions.json", QualityDefinition),
            candidate_model=None,
        )
        quality = model.quality
        assert quality is not None
        assert quality.name == "Bluray-1080p Remux"
        assert quality.source == "blurayRaw"
        assert quality.resolution == 1080

    def test_structured_read_on_real_manualimport_candidate(self) -> None:
        candidates = _load_models("manualimport_yamada.json", ManualImportCandidate)
        dvd = next(
            c
            for c in candidates
            if c.quality is not None and c.quality.quality is not None and c.quality.quality.name == "DVD"
        )
        assert quality_axes_from_model(dvd.quality) == ParsedQuality(
            source=QualitySource.DVD,
            resolution=480,
        )


class TestParsedFileInfoFromRealBodies:
    """The load-bearing claim: `parsedEpisodeInfo` populates even when `episodes` (series-matched) is empty."""

    def test_special_has_season_episode_despite_no_series_match(self) -> None:
        # The series-matched array is empty for a release Sonarr can't match to the series.
        assert _BODY.validate_json(_read("parse_yamada_s00e01.json"))["episodes"] == []

        info = _load_parse("parse_yamada_s00e01.json")
        assert info.season_number == 0
        assert info.episode_numbers == (1,)
        assert info.absolute_episode_numbers == ()

    def test_absolute_numbered_file_reports_absolute_not_season_episode(self) -> None:
        info = _load_parse("parse_glimmerzu_abs14.json")
        assert info.episode_numbers == ()
        assert info.absolute_episode_numbers == (14,)

    def test_missing_parsed_info_is_all_empty(self) -> None:
        info = ParsedFileInfo.model_validate({})
        assert info == ParsedFileInfo()

    def test_full_season_flag_reads_through(self) -> None:
        info = ParsedFileInfo.model_validate({"parsedEpisodeInfo": {"fullSeason": True}})
        assert info.full_season is True

    def test_junk_matched_entry_poisons_the_whole_array(self) -> None:
        # One malformed episodes[] entry folds the WHOLE array to (): dropping
        # just the bad one would shorten a span into a partial placement.
        body: dict[str, object] = {
            "episodes": [
                {"seasonNumber": 1, "episodeNumber": 1, "id": 501},
                {"seasonNumber": 1, "id": 502},
            ],
        }
        info = ParsedFileInfo.model_validate(body)
        assert info.matched_episodes == ()


class TestParseSeFromFilename:
    """`parse_se_from_filename` extracts an offline SxxExx pattern, never guessing a bare absolute number."""

    def test_sxxexx_extracted(self) -> None:
        info = parse_se_from_filename("Show.Name.S00E05.480p.mkv")
        assert info is not None
        assert info.season_number == 0
        assert info.episode_numbers == (5,)
        # Marked offline: the regex is blind to absolutes, so the absolute
        # zip's duplicate tell must treat this stand-in as unknown.
        assert info.offline is True

    def test_dash_separated_sxxexx(self) -> None:
        info = parse_se_from_filename("Show - S2E3 [1080p].mkv")
        assert info is not None
        assert (info.season_number, info.episode_numbers) == (2, (3,))

    def test_bare_absolute_number_is_not_guessed(self) -> None:
        # "01" alone is NOT an SxxExx: it is left to Sonarr's parse and the absolute zip.
        assert parse_se_from_filename("Show - 01 [1080p].mkv") is None


class TestResolvedIds:
    """`resolved_ids`: the claim's window when the record carries one, else the seeds' ids sorted."""

    def test_the_claims_window_wins(self) -> None:
        pending = pending_import(file_episode_map={"a.mkv": [7]}, ordered_episode_ids=[3, 1, 2])

        assert pending.resolved_ids() == [3, 1, 2]

    def test_an_unscoped_claim_falls_back_to_its_seeds_sorted(self) -> None:
        pending = pending_import(
            file_episode_map={"a.mkv": [7, 0], "b.mkv": [5]},
            ordered_episode_ids=[],
        )

        assert pending.resolved_ids() == [5, 7]

    def test_a_record_with_no_targets_reads_empty(self) -> None:
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[])

        assert pending.resolved_ids() == []


class TestClassifyRealQueue:
    """The real queue had a paused download (wait) + two importBlocked (step in)."""

    @staticmethod
    def _records_by_download() -> dict[str, list[QueueRecord]]:
        records: dict[str, list[QueueRecord]] = {}
        for record in _QueuePage.model_validate_json(_read("queue.json")).records:
            records.setdefault(record.download_id or "", []).append(record)
        return records

    def test_import_blocked_steps_in(self) -> None:
        records = self._records_by_download()
        unmatched = records["1111111111111111111111111111111111111111"]
        assert classify_queue(unmatched) is QueueVerdict.STEP_IN

    def test_paused_download_waits(self) -> None:
        records = self._records_by_download()
        paused = records["B7640FF13A2ADCA981B821D03CEBD1B569798459"]
        assert classify_queue(paused) is QueueVerdict.WAIT


class TestPendingImportOrderedIds:
    """The claim's `ordered_episode_ids` round-trips through JSON.

    A claim missing the key rehydrates unscoped.
    """

    def test_round_trip_preserves_ordered_episode_ids(self) -> None:
        rec = pending_import(ordered_episode_ids=[8030, 8031, 8032])

        again = PendingImport.from_json(rec.to_json(), guards={})

        assert again.claims[0].ordered_episode_ids == (8030, 8031, 8032)
        assert again == rec

    def test_claim_without_ordered_ids_rehydrates_unscoped(self) -> None:
        raw = pending_import().to_json()
        del raw["claims"][0]["ordered_episode_ids"]

        assert PendingImport.from_json(raw, guards={}).claims[0].ordered_episode_ids == ()


_STACKED_DOWNLOAD_ID = "3333333333333333333333333333333333333333"
"""The capture's stacked ManualImport commands share this downloadId (a duplicate-import loop)."""


def _commands() -> list[CommandResource]:
    """The captured command list: stacked ManualImports, a folder import with no downloadId, and one other command."""

    return _load_models("command_list.json", CommandResource)


class TestCommandResourceFixture:
    """`CommandResource.model_validate` parses name, status, message, and body.files."""

    def test_started_manual_import_parses_message_and_files(self) -> None:
        started = next(c for c in _commands() if c.name == "ManualImport" and c.status == "started")
        assert started.message == "Processing file 4 of 8"
        assert started.files
        first = started.files[0]
        assert first.download_id == _STACKED_DOWNLOAD_ID
        assert first.series_id == 169
        assert first.episode_ids == (6605,)

    def test_completed_manual_import_parses(self) -> None:
        completed = next(c for c in _commands() if c.status == "completed")
        assert completed.name == "ManualImport"
        assert completed.message == "Manually imported 10 files"
        assert completed.result == "successful"

    def test_folder_import_has_no_download_id(self) -> None:
        # A season-pack folder import: its files carry a folderName and path but NO downloadId,
        # so the guard must fall back to the path.
        folder = next(c for c in _commands() if c.files and c.files[0].series_id == 153)
        assert folder.files[0].download_id is None
        assert "Vodes" in (folder.files[0].path or "")

    def test_non_manual_import_command_parsed_without_files(self) -> None:
        proc = next(c for c in _commands() if c.name == "ProcessMonitoredDownloads")
        assert proc.files == ()


class TestManualImportInFlightFixture:
    """Both command-list guards read the real captured list to close the loop."""

    def test_matching_download_id_is_in_flight(self) -> None:
        # A started and a queued ManualImport share the downloadId: a fresh import would stack a duplicate.
        assert manual_import_in_flight(
            _commands(),
            DownloadMatch(_STACKED_DOWNLOAD_ID, ContentPaths(raw="/downloads", sonarr_visible="/downloads"), set()),
        )

    def test_unrelated_download_id_is_not_in_flight(self) -> None:
        assert not manual_import_in_flight(
            _commands(),
            DownloadMatch(
                "ffffffffffffffffffffffffffffffffffffffff",
                ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"),
                set(),
            ),
        )

    def test_folder_import_matches_by_episode_id(self) -> None:
        # The folder import carries no downloadId. Episode 5645 is ours.
        assert manual_import_in_flight(
            _commands(),
            DownloadMatch("no-such-hash", ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"), {5645}),
        )

    def test_disk_guard_defers_only_on_the_started_command(self) -> None:
        # The capture's one STARTED command (a ManualImport) defers. The queued remainder,
        # the parked ProcessMonitoredDownloads included, never does.
        commands = _commands()
        assert started_disk_commands(commands)
        queued_only = [c for c in commands if c.status != "started"]
        assert not started_disk_commands(queued_only)


def _replay_specials_parse(raw_base: str) -> ParsedFileInfo | None:
    """Replay the captured /parse bodies for the two specials by basename."""

    if "S00E01" in raw_base:
        return _load_parse("parse_yamada_s00e01.json")
    if "S00E02" in raw_base:
        return _load_parse("parse_yamada_s00e02.json")
    return None


class _Specials(NamedTuple):
    """The captured specials download wired into a bare `SonarrSync` and its scripted fake."""

    strat: SonarrSync
    sonarr: FakeSonarrClient
    files: list[str]
    """The on-disk basenames of the captured candidates."""

    def pending(self, **overrides: Any) -> PendingImport:
        """A record for the download (its series, group, and files) with `overrides` applied."""

        defaults: dict[str, Any] = {"series_id": 213, "release_group": "Headpatter", "seadex_files": self.files}
        return pending_import(**{**defaults, **overrides})


def _specials(config: AppConfig | None = None) -> _Specials:
    """The captured episode list, candidates, and parses replayed by a fresh fake, under `config`."""

    candidates = _load_models("manualimport_yamada.json", ManualImportCandidate)
    sonarr = FakeSonarrClient(
        queue=[],  # not tracked, so the poll steps in
        episodes=_load_models("episodes_213_yamada.json", SonarrEpisode),
        candidates=candidates,
        parse_fn=_replay_specials_parse,
        refresh_count=7,
        command_status=CommandResource(status="completed"),
        quality_defs=[],
        languages=[],
        execute_command_id=99,
    )
    strat = make_sonarr_sync(sonarr=sonarr, config=config or make_config(), cache_store=FakeCacheStore())
    return _Specials(strat, sonarr, [c.path.rsplit("/", 1)[-1] for c in candidates if c.path])


class TestCapturedSpecialsEndToEnd:
    """Drive `import_completed` with the real fixtures for a specials download Sonarr couldn't match."""

    @pytest.mark.parametrize(
        ("ordered_episode_ids", "title"),
        [
            # The torrent carries only E01/E02 of the entry's three specials, so only those two place.
            pytest.param([8030, 8031, 8032], "Yamada-kun and the Seven Witches", id="the resolved set"),
            # No resolved set: the exact pass falls back to the live series map, so the record imports
            # rather than retrying forever.
            pytest.param([], "Yamada and the Seven Witches (OVA)", id="an empty set"),
        ],
    )
    def test_specials_import_to_their_episode_ids(self, ordered_episode_ids: list[int], title: str) -> None:
        specials = _specials()
        pending = specials.pending(
            infohash="1111111111111111111111111111111111111111",
            title=title,
            file_episode_map={},
            ordered_episode_ids=ordered_episode_ids,
        )

        probe = specials.strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        # The copy is async, so nothing is present yet.
        assert probe.files_present is False
        assert probe.command_issued is True
        assert len(specials.sonarr.execute_calls) == 1
        files = specials.sonarr.execute_calls[0][0]
        assert {f.episodeIds[0] for f in files} == {8030, 8031}
        assert all(f.seriesId == 213 for f in files)

    @pytest.mark.parametrize(
        ("config", "mode"),
        [
            pytest.param(make_config(), "auto", id="the default"),
            pytest.param(make_config(import_mode="move"), "move", id="move"),
        ],
    )
    def test_import_mode_propagates_from_config(self, config: AppConfig, mode: str) -> None:
        # A "move" deletes the source files, so a wrong or ignored mode must never be silent.
        specials = _specials(config)
        pending = specials.pending(
            infohash="1111111111111111111111111111111111111111",
            file_episode_map={},
            ordered_episode_ids=[8030, 8031, 8032],
        )

        specials.strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert len(specials.sonarr.execute_calls) == 1
        assert specials.sonarr.execute_calls[0][1] == mode

    def test_import_completed_probe_carries_seed_complete_counts(self) -> None:
        # A complete seed map gives the probe determinate counts over the seed set, none landed yet.
        specials = _specials()
        ep_map = {name: [8030 + i] for i, name in enumerate(specials.files)}
        pending = specials.pending(
            infohash="2222222222222222222222222222222222222222",
            file_episode_map=ep_map,
            ordered_episode_ids=[v[0] for v in ep_map.values()],
        )

        probe = specials.strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert probe.target_count == len(specials.files)
        assert probe.imported_count == 0

    def test_import_progress_is_read_only_and_counts_seed_targets(self) -> None:
        # The fast poll: a determinate count over the seed targets, reading ONLY the episode files,
        # never the refresh, queue, or execute pipeline.
        specials = _specials()
        ep_map = {name: [8030 + i] for i, name in enumerate(specials.files)}
        pending = specials.pending(
            infohash="4444444444444444444444444444444444444444",
            file_episode_map=ep_map,
            ordered_episode_ids=[v[0] for v in ep_map.values()],
        )

        progress = specials.strat.import_progress(pending)

        assert progress.determinate is True
        assert progress.total == len(specials.files)
        assert progress.done == 0
        assert specials.sonarr.episodes_calls
        assert specials.sonarr.execute_calls == []
        assert specials.sonarr.refresh_calls == 0
        assert specials.sonarr.queue_calls == 0

    def test_import_progress_indeterminate_when_seed_map_incomplete(self) -> None:
        # No seed map gives an indeterminate zero without a fetch: promotion is left to the heavy poll.
        specials = _specials()
        pending = specials.pending(
            infohash="3333333333333333333333333333333333333333",
            file_episode_map={},
            ordered_episode_ids=[8030, 8031, 8032],
        )

        progress = specials.strat.import_progress(pending)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert specials.sonarr.episodes_calls == []
        assert specials.sonarr.execute_calls == []

    def test_import_progress_indeterminate_for_a_listless_record(self) -> None:
        # A window with no SeaDex file list (a migrated legacy row folds its flat ids into the window)
        # has nothing to measure completeness against, so the row stays indeterminate.
        specials = _specials()
        pending = specials.pending(
            infohash="6666666666666666666666666666666666666666",
            file_episode_map={},
            ordered_episode_ids=[8030],
            seadex_files=[],
        )

        progress = specials.strat.import_progress(pending)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert specials.sonarr.execute_calls == []

    def test_excluded_files_make_the_heavy_counts_determinate_but_never_promote(self) -> None:
        # Map and exclusions account for every file of a pack carrying another slice's files, so the heavy
        # probe counts OUR slice. The fast poll can promote, so a grab-time exclusion keeps it indeterminate.
        specials = _specials()
        pending = specials.pending(
            infohash="5555555555555555555555555555555555555555",
            file_episode_map={specials.files[0]: [8030]},
            ordered_episode_ids=[8030],
            excluded_files=[normalize_basename(name) for name in specials.files[1:]],
        )

        progress = specials.strat.import_progress(pending)
        probe = specials.strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert (probe.imported_count, probe.target_count) == (0, 1)
