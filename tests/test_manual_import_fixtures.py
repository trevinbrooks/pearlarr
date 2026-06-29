"""Real-API-fixture tests for the resolved-mapping manual import.

These pin the behaviour the *old* code got wrong, using JSON captured verbatim
from a live Sonarr (``tests/fixtures/sonarr/``). The headline failure that
motivated the rewrite: a specials/alias release Sonarr can't match to a series
(``Yamada-kun and the Seven Witches`` vs ``... (2015)``) returns an empty
series-*matched* ``episodes`` array, so the import silently mapped nothing. The
fix reads the series-*agnostic* ``parsedEpisodeInfo`` and assigns it into OUR
resolved episode set - so identity comes from the same mapping the add flow
already trusts, never from Sonarr's title match.

The pure :func:`assign_episode_ids` tests encode the three cases the user raised
(correctly-named specials, mis-numbered specials, multi-season "To Love-Ru"); the
end-to-end test drives the real Yamada fixtures through ``import_completed``.
"""

import json
import types
from pathlib import Path
from typing import Any
from unittest import mock

from seadexarr.modules.manual_import import (
    CandidateFile,
    EpisodeAssignment,
    ImportProgress,
    ImportReadiness,
    ParsedQuality,
    QueueRecordView,
    QueueVerdict,
    assign_episode_ids,
    classify_queue,
    manual_import_in_flight,
    normalize_basename,
    parse_se_from_filename,
    quality_axes_from_model,
    resolve_quality,
)
from seadexarr.modules.seadex_types import (
    CommandResource,
    ManualImportCandidate,
    ParsedFileInfo,
    QualityDefinition,
    QualitySource,
    SonarrEpisode,
)

from .builders import (
    FakeCacheStore,
    make_config,
    make_logger,
    make_sonarr_mapper,
    make_sonarr_sync,
    pending_import,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"


def load_fixture(name: str) -> Any:
    """Parse one captured Sonarr response from ``tests/fixtures/sonarr``."""

    return json.loads((_FIXTURES / name).read_text())


def _pinfo(
    *,
    season: int | None = None,
    episodes: tuple[int, ...] = (),
    absolutes: tuple[int, ...] = (),
) -> ParsedFileInfo:
    """Shorthand ParsedFileInfo for the pure-assignment tests."""

    return ParsedFileInfo(
        season_number=season,
        episode_numbers=episodes,
        absolute_episode_numbers=absolutes,
    )


# --------------------------------------------------------------------------- #
# Quality resolution - the (source, resolution) match, on real bodies
# --------------------------------------------------------------------------- #
class TestQualityResolution:
    """The quality fix's load-bearing claims.

    Quality is matched by the structured ``(source, resolution)`` pair. The
    candidate-read test runs on a verbatim live-Sonarr capture; the
    qualitydefinition list is a hand-authored STAND-IN (``qualitydefinitions.json``)
    mirroring real Sonarr - the live ``/api/v3/qualitydefinition`` capture is owed
    (the user's instance sits behind an auth proxy). Dropping a real capture in
    place of the stand-in re-runs these against reality unchanged.
    """

    def test_qualitydefinition_fixture_has_the_shape_the_matcher_needs(self) -> None:
        # CONTRACT, not validation: the matcher keys on (source, resolution), so
        # every definition must carry both. This guards the stand-in (and any real
        # capture swapped in for it) - it does NOT by itself prove the live
        # instance serializes the fields; that capture is owed to the user.
        defs: list[QualityDefinition] = load_fixture("qualitydefinitions.json")
        assert defs
        for definition in defs:
            quality = definition.get("quality")
            assert quality is not None
            assert isinstance(quality.get("resolution"), int)
            assert isinstance(quality.get("source"), str)
            if quality.get("name") != "Unknown":
                assert QualitySource.parse(quality.get("source")) is not None

    def test_bd_remux_resolves_against_full_def_list(self) -> None:
        # The original failure: a 1080p BD remux. Sonarr parses it as
        # (blurayRaw, 1080); against the full definition list that must resolve to
        # the "Bluray-1080p Remux" definition (valid id+name) - never omitted.
        defs: list[QualityDefinition] = load_fixture("qualitydefinitions.json")
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            defs,
            candidate_model=None,
        )
        quality = model.get("quality")
        assert quality is not None
        assert quality.get("name") == "Bluray-1080p Remux"
        assert quality.get("source") == "blurayRaw"
        assert quality.get("resolution") == 1080

    def test_structured_read_on_real_manualimport_candidate(self) -> None:
        # quality_axes_from_model reads (source, resolution) off a candidate
        # captured verbatim from a live Sonarr - proving the read works on real
        # output, not just hand-written dicts.
        raw = load_fixture("manualimport_yamada.json")
        candidates = [ManualImportCandidate.from_api(c) for c in raw]
        dvd = next(
            c for c in candidates if c.quality is not None and (c.quality.get("quality") or {}).get("name") == "DVD"
        )
        assert quality_axes_from_model(dvd.quality) == ParsedQuality(
            source=QualitySource.DVD,
            resolution=480,
        )


# --------------------------------------------------------------------------- #
# ParsedFileInfo.from_parse_resource - the series-agnostic field, on real bodies
# --------------------------------------------------------------------------- #
class TestParsedFileInfoFromRealBodies:
    """The fix's load-bearing claim: parsedEpisodeInfo is populated when the
    series-matched ``episodes`` array is empty."""

    def test_yamada_special_has_season_episode_despite_no_series_match(self) -> None:
        body = load_fixture("parse_yamada_s00e01.json")
        # The OLD code read this (series-matched) array and got nothing:
        assert body["episodes"] == []

        info = ParsedFileInfo.from_parse_resource(body)
        assert info.season_number == 0
        assert info.episode_numbers == (1,)
        assert info.absolute_episode_numbers == ()

    def test_absolute_numbered_file_reports_absolute_not_season_episode(self) -> None:
        info = ParsedFileInfo.from_parse_resource(load_fixture("parse_toloveru_abs14.json"))
        assert info.episode_numbers == ()
        assert info.absolute_episode_numbers == (14,)

    def test_missing_parsed_info_is_all_empty(self) -> None:
        info = ParsedFileInfo.from_parse_resource({})
        assert info == ParsedFileInfo()


# --------------------------------------------------------------------------- #
# parse_se_from_filename - the offline SxxExx fallback
# --------------------------------------------------------------------------- #
class TestParseSeFromFilename:
    def test_sxxexx_extracted(self) -> None:
        info = parse_se_from_filename("Show.Name.S00E05.480p.mkv")
        assert info is not None
        assert info.season_number == 0
        assert info.episode_numbers == (5,)

    def test_dash_separated_sxxexx(self) -> None:
        info = parse_se_from_filename("Show - S2E3 [1080p].mkv")
        assert info is not None
        assert (info.season_number, info.episode_numbers) == (2, (3,))

    def test_bare_absolute_number_is_not_guessed(self) -> None:
        # "01" alone is NOT an SxxExx - left to Sonarr's parse / the absolute leg,
        # never guessed as S?E01 here.
        assert parse_se_from_filename("Show - 01 [1080p].mkv") is None


# --------------------------------------------------------------------------- #
# assign_episode_ids - the three cases the user raised, plus guards
# --------------------------------------------------------------------------- #
class TestAssignExactSeason:
    """Leg 1: a correctly-named file Sonarr just couldn't match to the series."""

    def test_yamada_specials_assigned_by_exact_season_episode(self) -> None:
        # Resolved set is the entry's S00 episodes (ids 8030..8032); the two files
        # carry S00E01 / S00E02 and land on 8030 / 8031.
        files = ["s00e01.mkv", "s00e02.mkv"]
        parsed = {
            "s00e01.mkv": _pinfo(season=0, episodes=(1,)),
            "s00e02.mkv": _pinfo(season=0, episodes=(2,)),
        }
        ep_id_map = {(0, 1): 8030, (0, 2): 8031, (0, 3): 8032, (1, 1): 8033}

        result = assign_episode_ids(files, parsed, [8030, 8031, 8032], ep_id_map)

        assert result == EpisodeAssignment(
            assigned={"s00e01.mkv": [8030], "s00e02.mkv": [8031]},
            skipped=[],
        )

    def test_exact_parse_outside_resolved_set_is_skipped(self) -> None:
        # File parses to S01E01 (id 8033) but the resolved set is only S00 -> never
        # imported (the over-grab guard: identity must land INSIDE our set).
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {(0, 1): 8030, (1, 1): 8033}

        result = assign_episode_ids(["x.mkv"], parsed, [8030], ep_id_map)

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_empty_resolved_set_places_correctly_named_specials(self) -> None:
        # The stuck-record case: NO resolved set (an empty ordered_episode_ids, e.g.
        # a record whose grab-time specials resolution found nothing). The exact leg
        # falls back to the live series map, so a correctly-named file lands on its
        # real episode instead of sticking forever.
        files = ["s00e01.mkv", "s00e02.mkv"]
        parsed = {
            "s00e01.mkv": _pinfo(season=0, episodes=(1,)),
            "s00e02.mkv": _pinfo(season=0, episodes=(2,)),
        }
        ep_id_map = {(0, 1): 8030, (0, 2): 8031, (0, 3): 8032}

        result = assign_episode_ids(files, parsed, [], ep_id_map)

        assert result == EpisodeAssignment(
            assigned={"s00e01.mkv": [8030], "s00e02.mkv": [8031]},
            skipped=[],
        )


class TestAssignAbsolute:
    """Leg 2: absolute-number index onto the resolved set."""

    def test_mis_numbered_specials_map_positionally(self) -> None:
        # The user's case: files on disk are "01".."05" but are really S00E05..E09.
        # The release numbers never decide identity - they only ORDER the files onto
        # the resolved set, so "01" -> the first resolved episode (8034 = S00E05).
        files = [f"{n:02d}.mkv" for n in range(1, 6)]
        parsed = {name: _pinfo(absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = [8034, 8035, 8036, 8037, 8038]  # S00E05..E09 ids

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.skipped == []
        assert result.assigned == {
            "01.mkv": [8034],
            "02.mkv": [8035],
            "03.mkv": [8036],
            "04.mkv": [8037],
            "05.mkv": [8038],
        }

    def test_continuous_absolute_batch_spans_seasons(self) -> None:
        # A continuous absolute batch (1..4) maps cleanly onto a season-sorted
        # multi-season resolved set - this is the only multi-season pack we trust.
        files = ["e1.mkv", "e2.mkv", "e3.mkv", "e4.mkv"]
        parsed = {f"e{i}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 5)}
        resolved = [501, 502, 601, 602]  # S05E01-02, S06E01-02

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.assigned == {
            "e1.mkv": [501],
            "e2.mkv": [502],
            "e3.mkv": [601],
            "e4.mkv": [602],
        }

    def test_overlord_absolute_ova_pack_maps_onto_resolved_set(self) -> None:
        # releases.moe/101083 ("Overlord II - Ple Ple Pleiades 2"): 13 OVA files
        # named "- 01".."- 13", all parsed season 0 / absolute-only. The add flow
        # resolves this entry (anibridge tvdb_mappings {0: [(16, 28)]}) to the 13
        # season-0 episodes S00E16..E28 (live ids 2090..2102), so the absolute leg
        # places each file onto its season-sorted id (count-matched 13:13, no-dup) -
        # "- 01" -> S00E16, "- 13" -> S00E28. No grab-time change needed.
        files = [f"{n:02d}.mkv" for n in range(1, 14)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = list(range(2090, 2103))  # S00E16..E28 ids, season order

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.skipped == []
        assert result.assigned == {f"{n:02d}.mkv": [2089 + n] for n in range(1, 14)}


class TestAssignGuards:
    """Leg 3: refuse to guess - skip + warn instead."""

    def test_toloveru_per_title_restart_is_refused(self) -> None:
        # One torrent spanning two sub-series whose numbering BOTH restart at 1:
        # the shared absolutes are the tell of a season-boundary scramble, so the
        # whole absolute leg is refused rather than mis-assigned.
        main = {f"main-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        dark = {f"dark-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        parsed = {**main, **dark}
        files = list(parsed)
        resolved = [501, 502, 503, 601, 602, 603]

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)


def _cand(basename: str) -> CandidateFile:
    return CandidateFile(
        basename=basename,
        path=f"/dl/{basename}",
        quality=None,
        is_sample=False,
        is_already_imported=False,
    )


class TestAssignScopeGate:
    """CB3: allow_unscoped must key off the FULL resolved set, not the post-seed remainder."""

    def test_explicit_allow_unscoped_false_keeps_scope_on_empty_set(self) -> None:
        # An empty resolved set but allow_unscoped pinned False (what the mapper passes
        # for a fully-seeded record): a correctly-named but out-of-scope file is
        # skipped, NOT placed on the live map.
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {(1, 1): 8033}

        result = assign_episode_ids(["x.mkv"], parsed, [], ep_id_map, allow_unscoped=False)

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_fully_seeded_record_skips_out_of_scope_on_disk_leftover(self) -> None:
        # A fully-seeded record (every resolved episode already seeded) whose batch
        # folder also holds an OUT-OF-SCOPE file (a season-2 file in a season-1 grab).
        # The leftover must be skipped - not imported via the allow_unscoped fallback -
        # and the grab-time seed map must stay un-contaminated by the self-heal.
        seed_name = "Show - 01 [1080p].mkv"
        leftover_name = "Show - S02E01 [1080p].mkv"
        pending = pending_import(
            file_episode_map={seed_name: [101]},
            episode_ids=[101],
            ordered_episode_ids=[101],
            seadex_files=[seed_name],
        )
        sonarr = mock.MagicMock()
        sonarr.parse_episode_info.return_value = _pinfo(season=2, episodes=(1,))
        mapper = make_sonarr_mapper(sonarr=sonarr)

        candidates = {
            normalize_basename(seed_name): _cand(seed_name),
            normalize_basename(leftover_name): _cand(leftover_name),
        }
        ep_id_map = {(1, 1): 101, (2, 1): 999}  # 999 is OUTSIDE the resolved {101}

        merged, skipped = mapper.assign(pending, candidates, ep_id_map)

        placed_ids = {i for ids in merged.values() for i in ids}
        assert 999 not in placed_ids
        assert normalize_basename(leftover_name) in skipped
        assert pending.file_episode_map == {seed_name: [101]}

    def test_count_mismatch_skips(self) -> None:
        # Two absolute files but three resolved ids -> not a clean 1:1 -> skip both.
        parsed = {"a.mkv": _pinfo(absolutes=(1,)), "b.mkv": _pinfo(absolutes=(2,))}

        result = assign_episode_ids(["a.mkv", "b.mkv"], parsed, [1, 2, 3], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv"]

    def test_empty_resolved_set_skips_absolute_only_files(self) -> None:
        # With NO resolved set, the absolute leg has nothing to index into, so an
        # absolute-only pack (Overlord-style "- 01".."- 03") is left for the user
        # rather than guessed - absolute numbers are never trusted to decide identity
        # on their own (the To Love-Ru safety posture).
        files = [f"{n:02d}.mkv" for n in range(1, 4)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = assign_episode_ids(files, parsed, [], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)

    def test_single_unparseable_file_single_target_is_placed(self) -> None:
        # Degenerate positional: one leftover file, one leftover episode -> it's that
        # one even with no usable parse (the single-file fallback, resolved-set form).
        result = assign_episode_ids(["only.mkv"], {"only.mkv": None}, [900], {})

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == []

    def test_mixed_exact_then_leftover_absolute(self) -> None:
        # One file names its season (placed by leg 1); the remaining absolute file
        # maps onto the one leftover id.
        parsed = {
            "s01e01.mkv": _pinfo(season=1, episodes=(1,)),
            "extra.mkv": _pinfo(absolutes=(2,)),
        }
        ep_id_map = {(1, 1): 8033}

        result = assign_episode_ids(
            ["s01e01.mkv", "extra.mkv"],
            parsed,
            [8033, 8044],
            ep_id_map,
        )

        assert result.assigned == {"s01e01.mkv": [8033], "extra.mkv": [8044]}
        assert result.skipped == []


# --------------------------------------------------------------------------- #
# classify_queue on the real captured queue
# --------------------------------------------------------------------------- #
class TestClassifyRealQueue:
    """The real queue had a paused download (wait) + two importBlocked (step in)."""

    @staticmethod
    def _views_by_download() -> dict[str, list[QueueRecordView]]:
        views: dict[str, list[QueueRecordView]] = {}
        for rec in load_fixture("queue.json")["records"]:
            view = QueueRecordView(
                state=rec.get("trackedDownloadState", ""),
                status=rec.get("trackedDownloadStatus", ""),
            )
            views.setdefault(rec["downloadId"], []).append(view)
        return views

    def test_import_blocked_steps_in(self) -> None:
        views = self._views_by_download()
        yamada = views["1111111111111111111111111111111111111111"]
        assert classify_queue(yamada) is QueueVerdict.STEP_IN

    def test_paused_download_waits(self) -> None:
        views = self._views_by_download()
        paused = views["B7640FF13A2ADCA981B821D03CEBD1B569798459"]
        assert classify_queue(paused) is QueueVerdict.WAIT


# --------------------------------------------------------------------------- #
# PendingImport round-trip carries the new resolved set (with back-compat)
# --------------------------------------------------------------------------- #
class TestPendingImportOrderedIds:
    def test_round_trip_preserves_ordered_episode_ids(self) -> None:
        rec = pending_import(ordered_episode_ids=[8030, 8031, 8032])
        from seadexarr.modules.manual_import import PendingImport

        again = PendingImport.from_json(rec.to_json())
        assert again.ordered_episode_ids == [8030, 8031, 8032]
        assert again == rec

    def test_legacy_record_without_ordered_ids_rehydrates_empty(self) -> None:
        from seadexarr.modules.manual_import import PendingImport

        raw = pending_import().to_json()
        del raw["ordered_episode_ids"]
        assert PendingImport.from_json(raw).ordered_episode_ids == []


# --------------------------------------------------------------------------- #
# CommandResource.from_api on the real captured /api/v3/command list
# --------------------------------------------------------------------------- #
# The capture is the bug-2 evidence: stacked ManualImport commands sharing one
# downloadId (a duplicate-import loop), plus a folder import with no downloadId
# and a non-ManualImport command. Scrubbed for the public fixture (infohash +
# server path root), matching the rest of tests/fixtures/sonarr/.
_SAO_DOWNLOAD_ID = "3333333333333333333333333333333333333333"


class TestCommandResourceFixture:
    """CommandResource.from_api parses name / status / message / body.files."""

    @staticmethod
    def _commands() -> list[CommandResource]:
        return [CommandResource.from_api(c) for c in load_fixture("command_list.json")]

    def test_started_manual_import_parses_message_and_files(self) -> None:
        started = next(c for c in self._commands() if c.name == "ManualImport" and c.status == "started")
        assert started.message == "Processing file 4 of 8"
        assert started.files  # body.files were parsed
        first = started.files[0]
        assert first.download_id == _SAO_DOWNLOAD_ID
        assert first.series_id == 169
        assert first.episode_ids == (6605,)

    def test_completed_manual_import_parses(self) -> None:
        completed = next(c for c in self._commands() if c.status == "completed")
        assert completed.name == "ManualImport"
        assert completed.message == "Manually imported 10 files"
        assert completed.result == "successful"

    def test_folder_import_has_no_download_id(self) -> None:
        # The Tensei Vodes season-pack import is folder-based: its files carry a
        # folderName + path but NO downloadId, so the guard must fall back to path.
        folder = next(c for c in self._commands() if c.files and c.files[0].series_id == 153)
        assert folder.files[0].download_id is None
        assert "Vodes" in (folder.files[0].path or "")

    def test_non_manual_import_command_parsed_without_files(self) -> None:
        proc = next(c for c in self._commands() if c.name == "ProcessMonitoredDownloads")
        assert proc.files == ()


class TestManualImportInFlightFixture:
    """manual_import_in_flight reads the real command list to close the loop."""

    @staticmethod
    def _commands() -> list[CommandResource]:
        return [CommandResource.from_api(c) for c in load_fixture("command_list.json")]

    def test_matching_download_id_is_in_flight(self) -> None:
        # The SAO download has a started + queued ManualImport sharing its
        # downloadId -> a fresh import for it would stack a duplicate.
        assert manual_import_in_flight(
            self._commands(),
            _SAO_DOWNLOAD_ID,
            "/downloads",
            set(),
        )

    def test_unrelated_download_id_is_not_in_flight(self) -> None:
        # A different infohash with no path/episode overlap -> proceed.
        assert not manual_import_in_flight(
            self._commands(),
            "ffffffffffffffffffffffffffffffffffffffff",
            "/nowhere",
            set(),
        )

    def test_folder_import_matches_by_episode_id(self) -> None:
        # The Vodes folder import carries no downloadId; episode 5645 is ours.
        assert manual_import_in_flight(
            self._commands(),
            "no-such-hash",
            "/nowhere",
            {5645},
        )


# --------------------------------------------------------------------------- #
# End-to-end: the real Yamada failure now imports to the resolved S00 ids
# --------------------------------------------------------------------------- #
def _yamada_parse_side_effect(raw_base: str) -> ParsedFileInfo | None:
    """Replay the captured /parse bodies for the two Yamada specials by basename."""

    if "S00E01" in raw_base:
        return ParsedFileInfo.from_parse_resource(load_fixture("parse_yamada_s00e01.json"))
    if "S00E02" in raw_base:
        return ParsedFileInfo.from_parse_resource(load_fixture("parse_yamada_s00e02.json"))
    return None


def _yamada_strat() -> tuple[Any, mock.MagicMock, list[str]]:
    """The real Yamada fixtures wired into a bare SonarrSync + its scripted mock.

    Returns the strategy, its ``sonarr`` MagicMock (replaying the captured episode
    list / manual-import candidates / per-file parse), and the on-disk basenames.
    """

    episodes = [SonarrEpisode.from_api(e) for e in load_fixture("episodes_213_yamada.json")]
    candidates = [ManualImportCandidate.from_api(c) for c in load_fixture("manualimport_yamada.json")]
    seadex_files = [c.path.rsplit("/", 1)[-1] for c in candidates if c.path]

    sonarr = mock.MagicMock()
    sonarr.queue.return_value = []  # not tracked -> STEP_IN
    sonarr.episodes.return_value = episodes
    sonarr.manual_import_candidates.return_value = candidates
    sonarr.parse_episode_info.side_effect = _yamada_parse_side_effect
    sonarr.refresh_monitored_downloads.return_value = 7
    sonarr.command_status.return_value = types.SimpleNamespace(status="completed")
    sonarr.quality_definitions.return_value = []
    sonarr.languages.return_value = []
    sonarr.manual_import_execute.return_value = 99

    strat = make_sonarr_sync(
        sonarr=sonarr,
        logger=make_logger(),
        log_fmt=mock.MagicMock(),
        _config=make_config(),
        _last_refresh_monotonic=None,
        _ep_list_cache={},
        _parse_info_cache={},
        _warned_unplaceable=set(),
        cache_store=FakeCacheStore(),
    )
    return strat, sonarr, seadex_files


class TestYamadaEndToEnd:
    """Drive import_completed with the real fixtures for the failing queue item."""

    def test_specials_import_to_resolved_episode_ids(self) -> None:
        strat, sonarr, seadex_files = _yamada_strat()

        # Resolved set = the entry's S00 episodes (8030, 8031, 8032); the torrent
        # only carries E01/E02, so only those two get placed.
        pending = pending_import(
            infohash="1111111111111111111111111111111111111111",
            series_id=213,
            title="Yamada-kun and the Seven Witches",
            release_group="Headpatter",
            file_episode_map={},  # the real grab-time failure: nothing seeded
            episode_ids=[],
            ordered_episode_ids=[8030, 8031, 8032],
            seadex_files=seadex_files,
            seadex_sizes=[0] * len(seadex_files),
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        # The command was issued (copy is async -> RETRY + command_issued).
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        sonarr.manual_import_execute.assert_called_once()

        files = sonarr.manual_import_execute.call_args.kwargs["files"]
        assigned = {f["episodeIds"][0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f["seriesId"] == 213 for f in files)

    def test_import_completed_probe_carries_seed_complete_counts(self) -> None:
        # A complete seed map -> the probe carries the determinate "files inserted"
        # counts (none landed yet here -> 0 / N), pinned to the seed set.
        strat, sonarr, seadex_files = _yamada_strat()
        del sonarr
        ep_map = {name: [8030 + i] for i, name in enumerate(seadex_files)}
        pending = pending_import(
            infohash="2222222222222222222222222222222222222222",
            series_id=213,
            release_group="Headpatter",
            file_episode_map=ep_map,
            episode_ids=[],
            ordered_episode_ids=[v[0] for v in ep_map.values()],
            seadex_files=seadex_files,
            seadex_sizes=[0] * len(seadex_files),
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        assert probe.target_count == len(seadex_files)
        assert probe.imported_count == 0

    def test_import_progress_is_read_only_and_counts_seed_targets(self) -> None:
        # The Tier-2 fast poll: a determinate count over the seed targets, reading
        # ONLY the episode files - never the refresh / queue / execute pipeline.
        strat, sonarr, seadex_files = _yamada_strat()
        ep_map = {name: [8030 + i] for i, name in enumerate(seadex_files)}
        pending = pending_import(
            infohash="4444444444444444444444444444444444444444",
            series_id=213,
            release_group="Headpatter",
            file_episode_map=ep_map,
            episode_ids=[],
            ordered_episode_ids=[v[0] for v in ep_map.values()],
            seadex_files=seadex_files,
            seadex_sizes=[0] * len(seadex_files),
        )

        progress = strat.import_progress(pending)

        assert progress.determinate is True
        assert progress.total == len(seadex_files)
        assert progress.done == 0  # no episode holds a recommended file yet
        sonarr.episodes.assert_called()  # the one read it does make
        sonarr.manual_import_execute.assert_not_called()
        sonarr.refresh_monitored_downloads.assert_not_called()
        sonarr.queue.assert_not_called()

    def test_import_progress_indeterminate_when_seed_map_incomplete(self) -> None:
        # No (or partial) seed map -> indeterminate zero, and it never even fetches:
        # the importing row stays a spinner, promotion is left to the heavy poll.
        strat, sonarr, seadex_files = _yamada_strat()
        pending = pending_import(
            infohash="3333333333333333333333333333333333333333",
            series_id=213,
            release_group="Headpatter",
            file_episode_map={},  # the real grab-time gap
            episode_ids=[],
            ordered_episode_ids=[8030, 8031, 8032],
            seadex_files=seadex_files,
            seadex_sizes=[0] * len(seadex_files),
        )

        progress = strat.import_progress(pending)

        assert progress == ImportProgress(0, 0, determinate=False)
        sonarr.episodes.assert_not_called()
        sonarr.manual_import_execute.assert_not_called()

    def test_specials_import_with_empty_resolved_set(self) -> None:
        # THE headline regression: the ACTUAL on-disk stuck record is pre-fix - EMPTY
        # everything (no ordered_episode_ids, no seed map). Before the fix this fell
        # to the legacy path, mapped nothing (Sonarr's series-matched episodes are
        # empty), and retried forever. Now the empty-set exact fallback places the
        # two specials onto the live series episodes, so it imports with no re-grab.
        strat, sonarr, seadex_files = _yamada_strat()

        pending = pending_import(
            infohash="1111111111111111111111111111111111111111",
            series_id=213,
            title="Yamada and the Seven Witches (OVA)",
            release_group="Headpatter",
            file_episode_map={},
            episode_ids=[],
            ordered_episode_ids=[],  # the pre-fix stuck record
            seadex_files=seadex_files,
            seadex_sizes=[0] * len(seadex_files),
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        sonarr.manual_import_execute.assert_called_once()

        files = sonarr.manual_import_execute.call_args.kwargs["files"]
        assigned = {f["episodeIds"][0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f["seriesId"] == 213 for f in files)
