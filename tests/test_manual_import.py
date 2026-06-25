"""Characterization tests for the pure manual-import helpers.

Pins the deterministic decision helpers in
:mod:`seadexarr.modules.manual_import`: the ``(season, episode) -> id`` map, the
filename normalization, the authoritative file->episode mapping and import
planning (strict-honor + never-overwrite + never-skip), the queue classifier and
episode-file status, the filename quality parse, the layered quality/language
selection, the wait-mode resolution, and the :class:`PendingImport` JSON
round-trip. All pure, no network or disk; :class:`SonarrEpisode` is built directly
via :meth:`SonarrEpisode.from_api`.
"""

from typing import cast

from seadexarr.modules.manual_import import (
    CandidateFile,
    EpisodeFileStatus,
    ImportProbe,
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
    PendingState,
    QualitySelection,
    QueueRecordView,
    QueueVerdict,
    WaitOutcome,
    all_targets_done,
    build_authoritative_map,
    build_episode_id_map,
    classify_pending,
    classify_queue,
    derive_languages,
    episode_file_statuses,
    episode_ids_for_parsed,
    normalize_basename,
    normalize_group,
    parse_quality_from_filename,
    plan_import_files,
    resolve_language_objects,
    resolve_wait_mode,
    select_quality,
    targets_needing_import,
)
from seadexarr.modules.seadex_types import (
    SONARR_MISSING_KEY,
    QualityModel,
    SonarrEpisode,
)


def _ep(
    *,
    ep_id: int,
    season: int | None = 1,
    episode: int | None = 1,
    file_id: int = 0,
    group: str | None = None,
) -> SonarrEpisode:
    """A ``SonarrEpisode`` from the raw fields the helpers read."""

    raw: dict = {"id": ep_id, "seasonNumber": season, "episodeNumber": episode}
    if file_id:
        raw["episodeFileId"] = file_id
        raw["episodeFile"] = {"releaseGroup": group}
    return SonarrEpisode.from_api(raw)


class TestBuildEpisodeIdMap:
    def test_normal_seasoned_episodes(self) -> None:
        eps = [
            _ep(ep_id=11, season=1, episode=1),
            _ep(ep_id=12, season=1, episode=2),
            _ep(ep_id=21, season=2, episode=1),
        ]
        assert build_episode_id_map(eps) == {(1, 1): 11, (1, 2): 12, (2, 1): 21}

    def test_missing_season_and_episode_use_sentinel_no_collision(self) -> None:
        eps = [
            _ep(ep_id=5, season=None, episode=None),
            _ep(ep_id=6, season=1, episode=1),
        ]
        result = build_episode_id_map(eps)
        assert result[(SONARR_MISSING_KEY, SONARR_MISSING_KEY)] == 5
        assert result[(1, 1)] == 6

    def test_first_wins_on_duplicate_key(self) -> None:
        eps = [_ep(ep_id=7, season=1, episode=1), _ep(ep_id=8, season=1, episode=1)]
        assert build_episode_id_map(eps) == {(1, 1): 7}

    def test_zero_id_skipped(self) -> None:
        eps = [_ep(ep_id=0, season=1, episode=1), _ep(ep_id=9, season=1, episode=2)]
        assert build_episode_id_map(eps) == {(1, 2): 9}


class TestNormalize:
    def test_nfc_nfd_match(self) -> None:
        # Same text, NFC (composed) vs NFD (decomposed) "é"; both fold equal.
        nfc = "Café - 01.mkv"
        nfd = "Café - 01.mkv"
        assert normalize_basename(nfc) == normalize_basename(nfd)

    def test_strips_and_casefolds(self) -> None:
        assert normalize_basename("  Show - 01.MKV  ") == "show - 01.mkv"

    def test_group_casefold(self) -> None:
        assert normalize_group("SubGroup") == normalize_group("subgroup")


class TestEpisodeIdsForParsed:
    def test_maps_via_index(self) -> None:
        idx = {(1, 1): 11, (1, 2): 12}
        parsed = [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]
        assert episode_ids_for_parsed(parsed, idx) == [11, 12]

    def test_drops_unknown_and_none(self) -> None:
        idx = {(1, 1): 11}
        parsed = [
            {"season": 1, "episode": 1},
            {"season": 9, "episode": 9},
            {"season": None, "episode": 1},
        ]
        assert episode_ids_for_parsed(parsed, idx) == [11]


class TestBuildAuthoritativeMap:
    def test_seed_wins_over_repair(self) -> None:
        merged = build_authoritative_map({"a.mkv": [11]}, {"a.mkv": [99], "b.mkv": [12]})
        assert merged == {"a.mkv": [11], "b.mkv": [12]}

    def test_drops_empty_id_lists(self) -> None:
        merged = build_authoritative_map({"a.mkv": [0]}, {"b.mkv": []})
        assert merged == {}


class TestEpisodeFileStatuses:
    def test_absent_recommended_other_unknown(self) -> None:
        episodes = {
            1: _ep(ep_id=1, file_id=0),
            2: _ep(ep_id=2, file_id=20, group="SubGroup"),
            3: _ep(ep_id=3, file_id=30, group="OtherGroup"),
            4: _ep(ep_id=4, file_id=40, group=None),
        }
        statuses = episode_file_statuses([1, 2, 3, 4], episodes, {"subgroup"})
        assert statuses == {
            1: EpisodeFileStatus.ABSENT,
            2: EpisodeFileStatus.RECOMMENDED,
            3: EpisodeFileStatus.OTHER_GROUP,
            4: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_missing_episode_is_absent(self) -> None:
        statuses = episode_file_statuses([99], {}, {"subgroup"})
        assert statuses == {99: EpisodeFileStatus.ABSENT}

    def test_all_targets_done_only_when_all_recommended(self) -> None:
        rec = {1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.RECOMMENDED}
        mixed = {1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.OTHER_GROUP}
        assert all_targets_done(rec) is True
        assert all_targets_done(mixed) is False
        assert all_targets_done({}) is False

    def test_targets_needing_import_excludes_only_recommended(self) -> None:
        statuses = {
            1: EpisodeFileStatus.ABSENT,
            2: EpisodeFileStatus.RECOMMENDED,
            3: EpisodeFileStatus.OTHER_GROUP,
            4: EpisodeFileStatus.UNKNOWN_GROUP,
        }
        assert targets_needing_import(statuses) == {1, 3, 4}


def _candidate(basename: str, *, sample: bool = False, already: bool = False) -> CandidateFile:
    return CandidateFile(
        basename=basename,
        path=f"/dl/{basename}",
        quality=None,
        is_sample=sample,
        is_already_imported=already,
    )


class TestPlanImportFiles:
    def test_imports_only_needing_episodes(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv"), "b.mkv": _candidate("b.mkv")}
        # episode 12 already holds a recommended file -> only 11 needs import.
        decisions = plan_import_files(amap, cands, needing_import={11})
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["a.mkv"].episode_ids == [11]
        assert by_base["b.mkv"].action == "skip_done"

    def test_candidate_not_in_map_is_never_imported(self) -> None:
        amap = {"a.mkv": [11]}
        cands = {"a.mkv": _candidate("a.mkv"), "rogue.mkv": _candidate("rogue.mkv")}
        decisions = plan_import_files(amap, cands, needing_import={11})
        # Only our mapped file is decided on; the rogue on-disk file is ignored.
        assert {d.basename for d in decisions} == {"a.mkv"}

    def test_intended_file_missing_from_disk_is_flagged_not_dropped(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv")}
        decisions = plan_import_files(amap, cands, needing_import={11, 12})
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["b.mkv"].action == "missing"

    def test_sample_and_already_are_not_imported(self) -> None:
        amap = {"s.mkv": [11], "i.mkv": [12]}
        cands = {
            "s.mkv": _candidate("s.mkv", sample=True),
            "i.mkv": _candidate("i.mkv", already=True),
        }
        decisions = plan_import_files(amap, cands, needing_import={11, 12})
        actions = {d.basename: d.action for d in decisions}
        assert actions == {"s.mkv": "sample", "i.mkv": "already"}


class TestParseQualityFromFilename:
    def test_2160p_webdl(self) -> None:
        assert parse_quality_from_filename("Show.S01E01.2160p.WEB-DL.x265.mkv") == "WEBDL-2160p"

    def test_1080p_bluray(self) -> None:
        assert parse_quality_from_filename("[Group] Show - 01 [BluRay 1080p HEVC].mkv") == "Bluray-1080p"

    def test_no_resolution_returns_none(self) -> None:
        assert parse_quality_from_filename("Show.S01E01.WEB-DL.mkv") is None

    def test_remux_maps_to_remux_name(self) -> None:
        assert parse_quality_from_filename("Show.2160p.BluRay.Remux.mkv") == "Remux-2160p"

    def test_no_source_defaults_to_webdl(self) -> None:
        assert parse_quality_from_filename("Show - 01 [1080p].mkv") == "WEBDL-1080p"


class TestSelectQuality:
    def test_ours_wins(self) -> None:
        sel = select_quality(
            our_name="Bluray-2160p",
            candidate_quality={"quality": {"name": "WEBDL-1080p"}},
            default_name="HDTV-720p",
        )
        assert sel == QualitySelection(name="Bluray-2160p", model=None)

    def test_sonarr_in_context_when_no_ours(self) -> None:
        candidate: QualityModel = {"quality": {"name": "WEBDL-1080p"}}
        sel = select_quality(our_name=None, candidate_quality=candidate, default_name="HDTV-720p")
        assert sel == QualitySelection(name=None, model=candidate)

    def test_unknown_candidate_falls_through_to_default(self) -> None:
        sel = select_quality(
            our_name=None,
            candidate_quality={"quality": {"name": "Unknown"}},
            default_name="HDTV-720p",
        )
        assert sel == QualitySelection(name="HDTV-720p", model=None)

    def test_unknown_when_nothing_available(self) -> None:
        sel = select_quality(our_name=None, candidate_quality=None, default_name=None)
        assert sel == QualitySelection(name=None, model=None)


class TestDeriveLanguages:
    def test_dual_audio_returns_dual(self) -> None:
        assert derive_languages(True, ["Japanese", "English"], ["Japanese"]) == [
            "Japanese",
            "English",
        ]

    def test_single_audio_returns_single(self) -> None:
        assert derive_languages(False, ["Japanese", "English"], ["Japanese"]) == ["Japanese"]


class TestResolveWaitMode:
    def test_cli_wins_over_config(self) -> None:
        assert resolve_wait_mode(ImportWaitMode.OFF, ImportWaitMode.HYBRID) is ImportWaitMode.OFF

    def test_config_used_when_no_cli(self) -> None:
        assert resolve_wait_mode(None, ImportWaitMode.BLOCKING) is ImportWaitMode.BLOCKING

    def test_default_off_when_neither(self) -> None:
        assert resolve_wait_mode(None, None) is ImportWaitMode.OFF


class TestPendingImportRoundTrip:
    def test_to_json_from_json_round_trip(self) -> None:
        pending = PendingImport(
            infohash="abc123",
            series_id=55,
            file_episode_map={"ep1.mkv": [11], "ep2.mkv": [12]},
            episode_ids=[11, 12],
            release_group="Era-Raws",
            is_dual_audio=True,
            season_number=2,
            seadex_files=["ep1.mkv", "ep2.mkv"],
            seadex_sizes=[1000, 2000],
            title="Some Show",
            added_at="2026-06-24 12:00:00",
            coverage="S02 E01-E12",
            url="https://releases.moe/1",
        )
        assert PendingImport.from_json(pending.to_json()) == pending

    def test_from_json_tolerates_missing_keys(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h", "series_id": 1})
        assert rebuilt.infohash == "h"
        assert rebuilt.file_episode_map == {}
        assert rebuilt.seadex_sizes == []
        assert rebuilt.title is None

    def test_old_record_without_seadex_sizes_rehydrates(self) -> None:
        # Back-compat: a record persisted before seadex_sizes existed still loads.
        raw = {
            "infohash": "h",
            "series_id": 1,
            "file_episode_map": {"a.mkv": [1]},
            "episode_ids": [1],
            "release_group": "RG",
            "is_dual_audio": False,
            "season_number": 1,
            "seadex_files": ["a.mkv"],
            "title": "T",
            "added_at": "2026-06-24 00:00:00",
        }
        assert PendingImport.from_json(raw).seadex_sizes == []

    def test_old_record_without_coverage_url_defaults_to_none(self) -> None:
        # Migration-safe: a record persisted before coverage/url existed loads with
        # both defaulting to None (via from_json's .get).
        raw = {"infohash": "h", "series_id": 1}
        rebuilt = PendingImport.from_json(raw)
        assert rebuilt.coverage is None
        assert rebuilt.url is None

    def test_coverage_url_default_none_on_dataclass(self) -> None:
        # The dataclass defaults coverage/url to None so callers (and old records)
        # need not supply them.
        pending = PendingImport(
            infohash="h",
            series_id=1,
            file_episode_map={},
            episode_ids=[],
            release_group="RG",
            is_dual_audio=False,
            season_number=None,
            seadex_files=[],
            seadex_sizes=[],
            title=None,
            added_at="2026-06-24 00:00:00",
        )
        assert pending.coverage is None
        assert pending.url is None


class TestPendingStateAndProbe:
    """The shared carried-over status vocabulary + the import probe value object."""

    def test_pending_state_members(self) -> None:
        assert {s.name for s in PendingState} == {
            "QUEUED", "IMPORTING", "IMPORTED", "ERRORED", "MISSING",
        }

    def test_pending_state_is_its_string(self) -> None:
        assert PendingState.IMPORTING == "importing"
        assert PendingState.QUEUED == "queued"

    def test_import_probe_holds_readiness_and_flags(self) -> None:
        probe = ImportProbe(
            readiness=ImportReadiness.RETRY, files_present=False, command_issued=True,
        )
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.files_present is False
        assert probe.command_issued is True


class TestClassifyPending:
    """classify_pending folds a poll's outcome + the files-present flag into a state."""

    def test_missing(self) -> None:
        assert classify_pending(WaitOutcome.MISSING, False) is PendingState.MISSING

    def test_errored(self) -> None:
        assert classify_pending(WaitOutcome.ERRORED, False) is PendingState.ERRORED

    def test_still_downloading_is_queued(self) -> None:
        assert classify_pending(None, False) is PendingState.QUEUED

    def test_timed_out_is_queued(self) -> None:
        # A non-COMPLETE terminal that isn't missing/errored still reads queued.
        assert classify_pending(WaitOutcome.TIMED_OUT, False) is PendingState.QUEUED

    def test_complete_and_files_present_is_imported(self) -> None:
        assert classify_pending(WaitOutcome.COMPLETE, True) is PendingState.IMPORTED

    def test_complete_without_files_is_importing(self) -> None:
        # The copy is still in flight -> importing, never imported, until the
        # files are verified present.
        assert classify_pending(WaitOutcome.COMPLETE, False) is PendingState.IMPORTING


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in WaitOutcome} == {"COMPLETE", "ERRORED", "TIMED_OUT", "MISSING"}


def test_import_readiness_members_exist() -> None:
    assert {o.name for o in ImportReadiness} == {"IMPORTED", "RETRY", "LEAVE"}


def _qrecord(state: str, status: str = "ok", *, messages: bool = False) -> QueueRecordView:
    return QueueRecordView(state=state, status=status, has_messages=messages)


class TestClassifyQueue:
    """classify_queue reads state + status + statusMessages into one verdict."""

    def test_empty_steps_in(self) -> None:
        assert classify_queue([]) is QueueVerdict.STEP_IN

    def test_import_blocked_steps_in(self) -> None:
        assert classify_queue([_qrecord("importBlocked", "warning")]) is QueueVerdict.STEP_IN

    def test_failed_steps_in(self) -> None:
        assert classify_queue([_qrecord("failed", "error")]) is QueueVerdict.STEP_IN

    def test_clean_pending_is_pending_clean(self) -> None:
        assert classify_queue([_qrecord("importPending", "ok")]) is QueueVerdict.PENDING_CLEAN

    def test_pending_with_warning_is_pending_clean(self) -> None:
        # Any importPending waits (PENDING_CLEAN), even with a warning: stepping in
        # on a still-pending record races Sonarr's import and double-imports.
        assert classify_queue([_qrecord("importPending", "warning")]) is QueueVerdict.PENDING_CLEAN

    def test_pending_with_messages_is_pending_clean(self) -> None:
        assert classify_queue([_qrecord("importPending", "ok", messages=True)]) is QueueVerdict.PENDING_CLEAN

    def test_downloading_waits(self) -> None:
        assert classify_queue([_qrecord("downloading")]) is QueueVerdict.WAIT

    def test_in_motion_beats_blocked_to_avoid_racing(self) -> None:
        # Something is actively importing -> wait, don't race it, even if a sibling
        # record is blocked; a later poll re-evaluates once the import settles.
        assert (
            classify_queue([_qrecord("importing"), _qrecord("importBlocked", "warning")])
            is QueueVerdict.WAIT
        )

    def test_case_insensitive(self) -> None:
        assert classify_queue([_qrecord("IMPORTBLOCKED", "WARNING")]) is QueueVerdict.STEP_IN


class TestResolveLanguageObjectsDefensive:
    """resolve_language_objects survives a blank/None or malformed name list."""

    def test_none_names_returns_empty(self) -> None:
        names = cast("list[str]", None)
        assert resolve_language_objects(names, [{"id": 8, "name": "Japanese"}]) == []

    def test_non_string_names_are_skipped(self) -> None:
        defs = [{"id": 8, "name": "Japanese"}]
        names = cast("list[str]", [None, 5, "Japanese"])
        assert resolve_language_objects(names, defs) == [{"id": 8, "name": "Japanese"}]
