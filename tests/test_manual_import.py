# pyright: strict
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

from seadexarr.modules.manual_import import (
    CandidateFile,
    EpisodeFileStatus,
    ImportProbe,
    ImportReadiness,
    ImportWaitMode,
    ParsedQuality,
    PendingImport,
    PendingState,
    QueueRecordView,
    QueueVerdict,
    WaitOutcome,
    all_targets_done,
    build_episode_id_map,
    classify_pending,
    classify_queue,
    derive_languages,
    episode_file_statuses,
    episode_ids_for_parsed,
    manual_import_in_flight,
    normalize_basename,
    normalize_group,
    parse_quality_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
    resolve_wait_mode,
    targets_needing_import,
)
from seadexarr.modules.planner import normalize_rg
from seadexarr.modules.seadex_types import (
    SONARR_MISSING_KEY,
    CommandResource,
    Language,
    QualityDefinition,
    QualityModel,
    QualitySource,
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

    raw: dict[str, object] = {"id": ep_id, "seasonNumber": season, "episodeNumber": episode}
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

    def test_group_dash_wrapped_agrees_with_planner(self) -> None:
        # normalize_group is the single source of truth normalize_rg delegates to;
        # a dash-wrapped group must compare equal on both ends or a release the
        # planner grabbed gets re-imported over by the overwrite guard.
        assert normalize_group("-Aergia-") == "aergia"
        assert normalize_group("-Aergia-") == normalize_rg("-Aergia-")
        assert normalize_group("Aergia") == normalize_rg("-Aergia-")


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

    def test_dash_wrapped_group_counts_as_recommended(self) -> None:
        # Sonarr can report a file's group dash-wrapped ("-Aergia-"); the overwrite
        # guard must still match it against the recommended set built from "Aergia".
        episodes = {5: _ep(ep_id=5, file_id=50, group="-Aergia-")}
        statuses = episode_file_statuses([5], episodes, {normalize_group("Aergia")})
        assert statuses == {5: EpisodeFileStatus.RECOMMENDED}

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

    def test_sample_is_not_imported(self) -> None:
        # A sample is never our intended file regardless of need.
        amap = {"s.mkv": [11]}
        cands = {"s.mkv": _candidate("s.mkv", sample=True)}
        decisions = plan_import_files(amap, cands, needing_import={11})
        assert decisions[0].action == "sample"

    def test_already_imported_does_not_skip_a_needed_target(self) -> None:
        # Bug fix: Sonarr's "already imported" rejection fires whenever the episode
        # already holds ANY file (including a missing-group one we grabbed to
        # replace). It must NOT veto importing a target that still needs our file.
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, needing_import={12})
        assert decisions[0].action == "import"
        assert decisions[0].episode_ids == [12]

    def test_already_imported_skips_only_when_no_target_needs_us(self) -> None:
        # When every target already holds a recommended file (none in needing),
        # Sonarr's rejection and our episode-file check agree -> the more specific
        # ``already`` (never overwrite).
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, needing_import=set())
        assert decisions[0].action == "already"

    def test_not_needed_without_rejection_is_skip_done(self) -> None:
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv")}
        decisions = plan_import_files(amap, cands, needing_import=set())
        assert decisions[0].action == "skip_done"


def _command(
    *,
    name: str = "ManualImport",
    status: str = "started",
    files: list[dict[str, object]] | None = None,
) -> CommandResource:
    """A ``CommandResource`` from the raw command fields the guard reads."""

    return CommandResource.from_api(
        {"name": name, "status": status, "body": {"files": files or []}},
    )


class TestManualImportInFlight:
    """The pure in-flight guard over the /api/v3/command list."""

    def test_matching_download_id_is_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "ABC", "episodeIds": [1]}])]
        # Case-insensitive match on the infohash.
        assert manual_import_in_flight(cmds, "abc", "/d", set())

    def test_completed_command_is_not_in_flight(self) -> None:
        cmds = [_command(status="completed", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, "abc", "/d", set())

    def test_non_manual_import_command_ignored(self) -> None:
        cmds = [_command(name="ProcessMonitoredDownloads", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, "abc", "/d", set())

    def test_unrelated_download_id_not_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "OTHER"}])]
        assert not manual_import_in_flight(cmds, "abc", "/d", set())

    def test_queued_status_counts_as_in_flight(self) -> None:
        cmds = [_command(status="queued", files=[{"downloadId": "ABC"}])]
        assert manual_import_in_flight(cmds, "abc", "/d", set())

    def test_folder_import_matches_by_path_prefix(self) -> None:
        # No downloadId on the files -> fall back to the content_path prefix.
        cmds = [_command(files=[{"path": "/d/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, "no-hash", "/d/folder", set())

    def test_folder_import_matches_by_episode_overlap(self) -> None:
        cmds = [_command(files=[{"path": "/elsewhere/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, "no-hash", "/other", {9})

    def test_download_id_command_not_swept_by_path_overlap(self) -> None:
        # A command that DOES carry a (different) downloadId is never matched by
        # path/episode overlap - only the no-downloadId folder case falls back.
        cmds = [_command(files=[{"downloadId": "OTHER", "path": "/d/x.mkv", "episodeIds": [9]}])]
        assert not manual_import_in_flight(cmds, "abc", "/d", {9})

    def test_empty_command_list_not_in_flight(self) -> None:
        assert not manual_import_in_flight([], "abc", "/d", {9})


# A realistic subset of Sonarr's /api/v3/qualitydefinition, matched on the
# structured (source, resolution) pair (never on the display name).
_DEFS: list[QualityDefinition] = [
    {"quality": {"id": 4, "name": "HDTV-720p", "source": "television", "resolution": 720}},
    {"quality": {"id": 6, "name": "Bluray-720p", "source": "bluray", "resolution": 720}},
    {"quality": {"id": 9, "name": "HDTV-1080p", "source": "television", "resolution": 1080}},
    {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
    {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
    {"quality": {"id": 20, "name": "Bluray-1080p Remux", "source": "blurayRaw", "resolution": 1080}},
    {"quality": {"id": 19, "name": "Bluray-2160p", "source": "bluray", "resolution": 2160}},
    {"quality": {"id": 21, "name": "Bluray-2160p Remux", "source": "blurayRaw", "resolution": 2160}},
]


def _resolved_name(model: QualityModel) -> str | None:
    """The emitted quality's display name, for asserting which definition won."""

    return (model.get("quality") or {}).get("name")


class TestParseQualityFromFilename:
    def test_2160p_webdl(self) -> None:
        assert parse_quality_from_filename(
            "Show.S01E01.2160p.WEB-DL.x265.mkv",
        ) == ParsedQuality(source=QualitySource.WEB, resolution=2160)

    def test_1080p_bluray(self) -> None:
        assert parse_quality_from_filename(
            "[Group] Show - 01 [BluRay 1080p HEVC].mkv",
        ) == ParsedQuality(source=QualitySource.BLURAY, resolution=1080)

    def test_bd_remux_1080p_is_blurayraw(self) -> None:
        # The bug case: a BD remux must parse to (blurayRaw, 1080) - never the
        # bogus "Remux-1080p" name the old joiner produced.
        assert parse_quality_from_filename(
            "The.Seven.Deadly.Sins.1080p.Dual.Audio.BD.Remux.DTS-HD.MA-TTGA.mkv",
        ) == ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)

    def test_no_resolution_still_keeps_source(self) -> None:
        assert parse_quality_from_filename(
            "Show.S01E01.WEB-DL.mkv",
        ) == ParsedQuality(source=QualitySource.WEB, resolution=None)

    def test_no_source_leaves_source_none(self) -> None:
        # No recognized source token -> source stays None (NOT defaulted to WEB),
        # so the configured default fills the axis.
        assert parse_quality_from_filename(
            "Show - 01 [1080p].mkv",
        ) == ParsedQuality(source=None, resolution=1080)


class TestQualityAxesFromModel:
    def test_reads_structured_source_and_resolution(self) -> None:
        model: QualityModel = {
            "quality": {"name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
        }
        assert quality_axes_from_model(model) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=1080,
        )

    def test_unknown_source_and_zero_resolution_are_undetermined(self) -> None:
        model: QualityModel = {
            "quality": {"name": "Unknown", "source": "unknown", "resolution": 0},
        }
        assert quality_axes_from_model(model) == ParsedQuality(source=None, resolution=None)

    def test_none_model_is_empty(self) -> None:
        assert quality_axes_from_model(None) == ParsedQuality()


class TestQualityAxesFromName:
    def test_resolves_default_name_to_axes(self) -> None:
        assert quality_axes_from_name("Bluray-2160p", _DEFS) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=2160,
        )

    def test_unset_or_unmatched_is_empty(self) -> None:
        assert quality_axes_from_name(None, _DEFS) == ParsedQuality()
        assert quality_axes_from_name("Not-A-Quality", _DEFS) == ParsedQuality()


class TestResolveQuality:
    def test_sonarr_wins_over_ours_and_default(self) -> None:
        # Sonarr parsed (web, 1080); our filename parse and the default disagree.
        sonarr = ParsedQuality(source=QualitySource.WEB, resolution=1080)
        ours = ParsedQuality(source=QualitySource.BLURAY, resolution=2160)
        default = ParsedQuality(source=QualitySource.TELEVISION, resolution=720)
        model = resolve_quality(sonarr, ours, default, _DEFS, candidate_model=None)
        assert _resolved_name(model) == "WEBDL-1080p"

    def test_bd_remux_resolves_to_remux_definition(self) -> None:
        # The headline fix: (blurayRaw, 1080) -> "Bluray-1080p Remux" (valid id+name).
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-1080p Remux"
        assert (model.get("quality") or {}).get("id") == 20

    def test_per_axis_fill_from_default(self) -> None:
        # User's example: we parsed (None, 1080); default is Bluray-2160p ->
        # import as Bluray-1080p, NOT WEBDL-1080p and NOT Unknown.
        ours = ParsedQuality(source=None, resolution=1080)
        default = ParsedQuality(source=QualitySource.BLURAY, resolution=2160)
        model = resolve_quality(
            ParsedQuality(),
            ours,
            default,
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-1080p"

    def test_blurayraw_downgrades_to_bluray_when_no_remux_def(self) -> None:
        # 720p has no remux definition; a (blurayRaw, 720) gracefully downgrades
        # to Bluray-720p rather than failing.
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=720)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-720p"

    def test_no_match_falls_back_to_candidate_verbatim(self) -> None:
        # Nothing determined, but Sonarr's candidate carries a real quality:
        # re-emit it verbatim rather than omit the quality key.
        candidate: QualityModel = {
            "quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
            "revision": {"version": 1, "real": 0, "isRepack": False},
        }
        model = resolve_quality(
            ParsedQuality(),
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate,
        )
        assert model == candidate

    def test_nothing_resolves_synthesizes_explicit_unknown(self) -> None:
        # No axes and no candidate quality: emit an explicit Unknown object (never
        # omit the key - the omitted key is what crashed Sonarr's FileNameBuilder).
        model = resolve_quality(
            ParsedQuality(),
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert model.get("quality") == {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}
        assert "revision" in model

    def test_candidate_revision_is_preserved(self) -> None:
        # A repack/proper revision on the candidate carries onto the resolved model.
        sonarr = ParsedQuality(source=QualitySource.WEB, resolution=1080)
        candidate: QualityModel = {
            "quality": {"name": "WEBDL-1080p", "source": "web", "resolution": 1080},
            "revision": {"version": 2, "real": 0, "isRepack": True},
        }
        model = resolve_quality(sonarr, ParsedQuality(), ParsedQuality(), _DEFS, candidate)
        assert model.get("revision") == {"version": 2, "real": 0, "isRepack": True}


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
            seadex_files=["ep1.mkv", "ep2.mkv"],
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
        assert rebuilt.title is None

    def test_old_record_with_unknown_keys_rehydrates(self) -> None:
        # Back-compat: a record persisted with since-removed keys still loads
        # (from_json reads only the known keys and ignores the rest).
        raw = {
            "infohash": "h",
            "series_id": 1,
            "file_episode_map": {"a.mkv": [1]},
            "episode_ids": [1],
            "release_group": "RG",
            "is_dual_audio": False,
            "seadex_files": ["a.mkv"],
            "seadex_sizes": [1000],
            "title": "T",
            "added_at": "2026-06-24 00:00:00",
        }
        assert PendingImport.from_json(raw).infohash == "h"

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
            seadex_files=[],
            title=None,
            added_at="2026-06-24 00:00:00",
        )
        assert pending.coverage is None
        assert pending.url is None


class TestPendingStateAndProbe:
    """The shared carried-over status vocabulary + the import probe value object."""

    def test_pending_state_members(self) -> None:
        assert {s.name for s in PendingState} == {
            "QUEUED",
            "IMPORTING",
            "IMPORTED",
            "ERRORED",
            "MISSING",
        }

    def test_pending_state_is_its_string(self) -> None:
        assert PendingState.IMPORTING == "importing"
        assert PendingState.QUEUED == "queued"

    def test_import_probe_holds_readiness_and_flags(self) -> None:
        probe = ImportProbe(
            readiness=ImportReadiness.RETRY,
            files_present=False,
            command_issued=True,
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

    def test_complete_and_files_present_is_imported(self) -> None:
        assert classify_pending(WaitOutcome.COMPLETE, True) is PendingState.IMPORTED

    def test_complete_without_files_is_importing(self) -> None:
        # The copy is still in flight -> importing, never imported, until the
        # files are verified present.
        assert classify_pending(WaitOutcome.COMPLETE, False) is PendingState.IMPORTING


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in WaitOutcome} == {"COMPLETE", "ERRORED", "MISSING"}


def test_import_readiness_members_exist() -> None:
    assert {o.name for o in ImportReadiness} == {"IMPORTED", "RETRY", "LEAVE"}


def _qrecord(state: str, status: str = "ok") -> QueueRecordView:
    return QueueRecordView(state=state, status=status)


class TestClassifyQueue:
    """classify_queue buckets queue records by trackedDownloadState into one verdict."""

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

    def test_pending_with_error_status_is_pending_clean(self) -> None:
        # An importPending record waits regardless of status, including "error":
        # bucketing is by state, so it never routes to STEP_IN (which would race
        # Sonarr's import and double-import). Sonarr couples error with failedPending,
        # so this also hardens the invariant against a future queue-state shape.
        assert classify_queue([_qrecord("importPending", "error")]) is QueueVerdict.PENDING_CLEAN

    def test_downloading_waits(self) -> None:
        assert classify_queue([_qrecord("downloading")]) is QueueVerdict.WAIT

    def test_in_motion_beats_blocked_to_avoid_racing(self) -> None:
        # Something is actively importing -> wait, don't race it, even if a sibling
        # record is blocked; a later poll re-evaluates once the import settles.
        assert classify_queue([_qrecord("importing"), _qrecord("importBlocked", "warning")]) is QueueVerdict.WAIT

    def test_case_insensitive(self) -> None:
        assert classify_queue([_qrecord("IMPORTBLOCKED", "WARNING")]) is QueueVerdict.STEP_IN


class TestResolveLanguageObjectsDefensive:
    """resolve_language_objects survives a blank/None or malformed name list."""

    def test_none_names_returns_empty(self) -> None:
        defs: list[Language] = [{"id": 8, "name": "Japanese"}]
        assert resolve_language_objects(None, defs) == []

    def test_non_string_names_are_skipped(self) -> None:
        defs: list[Language] = [{"id": 8, "name": "Japanese"}]
        names: list[object] = [None, 5, "Japanese"]
        assert resolve_language_objects(names, defs) == [{"id": 8, "name": "Japanese"}]
