# pyright: strict
"""Characterization tests for the pure manual-import helpers.

Pins the deterministic decision helpers in
`manual_import` (the wait vocabulary, the normalizers,
the `PendingImport` JSON round-trip) and
`sonarr_import_plan` (the `(season, episode) -> id`
map, the authoritative file->episode mapping and import planning (strict-honor
+ never-overwrite + never-skip), the queue classifier and episode-file status,
the filename quality parse, and the layered quality/language selection). All
pure, no network or disk. `SonarrEpisode` is built directly via
`SonarrEpisode.model_validate`.
"""

import pytest

from pearlarr.manual_import import (
    ImportProbe,
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
    PendingState,
    TorrentTelemetry,
    WaitOutcome,
    classify_pending,
    normalize_basename,
    normalize_group,
    resolve_wait_mode,
    sanitize_torrent_telemetry,
)
from pearlarr.planner import normalize_rg
from pearlarr.seadex_types import (
    SONARR_MISSING_KEY,
    CommandResource,
    HistoryRecord,
    ParsedEpisode,
    Quality,
    QualityDefinition,
    QualityModel,
    QualitySource,
    RemotePathMapping,
    Revision,
    SonarrEpisode,
)
from pearlarr.sonarr_import_plan import (
    CandidateFile,
    ContentPaths,
    EpisodeFileStatus,
    EpisodeSnapshot,
    ParsedQuality,
    QueueVerdict,
    all_targets_done,
    build_episode_id_map,
    classify_download_history,
    classify_queue,
    derive_languages,
    episode_file_statuses,
    episode_ids_for_parsed,
    manual_import_in_flight,
    parse_quality_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_quality,
    targets_needing_import,
    translate_download_path,
)

from .builders import SEP


def _ep(
    *,
    ep_id: int,
    season: int | None = 1,
    episode: int | None = 1,
    file_id: int = 0,
    group: str | None = None,
) -> SonarrEpisode:
    """A `SonarrEpisode` from the raw fields the helpers read."""

    raw: dict[str, object] = {"id": ep_id, "seasonNumber": season, "episodeNumber": episode}
    if file_id:
        raw["episodeFileId"] = file_id
        raw["episodeFile"] = {"releaseGroup": group}
    return SonarrEpisode.model_validate(raw)


class TestBuildEpisodeIdMap:
    """`build_episode_id_map` maps `(season, episode)` to the first episode id.

    Missing season/episode fold to a sentinel key (no collision with real pairs), and id 0 is skipped.
    """

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
    """`normalize_basename`/`normalize_group` casefold and strip for stable comparison.

    Interior dashes in a group name are never stripped, only wrapping ones.
    """

    def test_nfc_nfd_match(self) -> None:
        # Same text, NFC (composed) vs NFD (decomposed) "é". Both fold equal.
        nfc = "Café - 01.mkv"
        nfd = "Café - 01.mkv"
        assert normalize_basename(nfc) == normalize_basename(nfd)

    def test_strips_and_casefolds(self) -> None:
        assert normalize_basename("  Show - 01.MKV  ") == "show - 01.mkv"

    def test_group_casefold(self) -> None:
        assert normalize_group("SubGroup") == normalize_group("subgroup")

    def test_group_strip_only_removes_wrapping_dashes(self) -> None:
        # MUTATION PIN: strip("-") widened to a multi-char strip set would eat
        # the X off an X-edged group. Only wrapping dashes (and whitespace) go,
        # and interior dashes always stay.
        assert normalize_group("Xrays-") == "xrays"
        assert normalize_group("X-Raws") == "x-raws"

    def test_group_dash_wrapped_agrees_with_planner(self) -> None:
        # normalize_group is the single source of truth normalize_rg delegates to.
        # A dash-wrapped group must compare equal on both ends or a release the
        # planner grabbed gets re-imported over by the overwrite guard.
        assert normalize_group("-Aergia-") == "aergia"
        assert normalize_group("-Aergia-") == normalize_rg("-Aergia-")
        assert normalize_group("Aergia") == normalize_rg("-Aergia-")


class TestEpisodeIdsForParsed:
    """`episode_ids_for_parsed` maps parsed `(season, episode)` pairs to ids via the index, dropping unknowns."""

    def test_maps_via_index(self) -> None:
        idx = {(1, 1): 11, (1, 2): 12}
        parsed = [ParsedEpisode(season=1, episode=1), ParsedEpisode(season=1, episode=2)]
        assert episode_ids_for_parsed(parsed, idx) == [11, 12]

    def test_drops_unknown(self) -> None:
        idx = {(1, 1): 11}
        parsed = [
            ParsedEpisode(season=1, episode=1),
            ParsedEpisode(season=9, episode=9),
        ]
        assert episode_ids_for_parsed(parsed, idx) == [11]


class TestEpisodeFileStatuses:
    """`episode_file_statuses` classifies each episode's file, and the two summary helpers derive from that.

    `all_targets_done` is true only when every status is recommended. `targets_needing_import` excludes only
    the recommended ones.
    """

    def test_absent_recommended_other_unknown(self) -> None:
        episodes = {
            1: _ep(ep_id=1, file_id=0),
            2: _ep(ep_id=2, file_id=20, group="SubGroup"),
            3: _ep(ep_id=3, file_id=30, group="OtherGroup"),
            4: _ep(ep_id=4, file_id=40, group=None),
        }
        statuses = episode_file_statuses([1, 2, 3, 4], EpisodeSnapshot(episodes, {"subgroup"}))
        assert statuses == {
            1: EpisodeFileStatus.ABSENT,
            2: EpisodeFileStatus.RECOMMENDED,
            3: EpisodeFileStatus.OTHER_GROUP,
            4: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_missing_episode_is_absent(self) -> None:
        statuses = episode_file_statuses([99], EpisodeSnapshot({}, {"subgroup"}))
        assert statuses == {99: EpisodeFileStatus.ABSENT}

    def test_dash_wrapped_group_counts_as_recommended(self) -> None:
        # Sonarr can report a file's group dash-wrapped ("-Aergia-"). The overwrite
        # guard must still match it against the recommended set built from "Aergia".
        episodes = {5: _ep(ep_id=5, file_id=50, group="-Aergia-")}
        statuses = episode_file_statuses([5], EpisodeSnapshot(episodes, {normalize_group("Aergia")}))
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
    """`plan_import_files` decides each candidate's action from the file->episode map and the needing-import set.

    Actions: import, skip_done, missing, sample, already.
    """

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
        # Only our mapped file is decided on. The rogue on-disk file is ignored.
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
        # `already` (never overwrite).
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
    """A `CommandResource` from the raw command fields the guard reads."""

    return CommandResource.model_validate(
        {"name": name, "status": status, "body": {"files": files or []}},
    )


def _paths(raw: str, sonarr_visible: str | None = None) -> ContentPaths:
    """A `ContentPaths` pair. The Sonarr view defaults to the raw path (untranslated)."""

    return ContentPaths(raw=raw, sonarr_visible=sonarr_visible if sonarr_visible is not None else raw)


class TestManualImportInFlight:
    """The pure in-flight guard over the /api/v3/command list."""

    def test_matching_download_id_is_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "ABC", "episodeIds": [1]}])]
        # Case-insensitive match on the infohash.
        assert manual_import_in_flight(cmds, "abc", _paths("/d"), set())

    def test_completed_command_is_not_in_flight(self) -> None:
        cmds = [_command(status="completed", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, "abc", _paths("/d"), set())

    def test_non_manual_import_command_ignored(self) -> None:
        cmds = [_command(name="ProcessMonitoredDownloads", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, "abc", _paths("/d"), set())

    def test_unrelated_download_id_not_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "OTHER"}])]
        assert not manual_import_in_flight(cmds, "abc", _paths("/d"), set())

    def test_queued_status_counts_as_in_flight(self) -> None:
        cmds = [_command(status="queued", files=[{"downloadId": "ABC"}])]
        assert manual_import_in_flight(cmds, "abc", _paths("/d"), set())

    def test_folder_import_matches_by_path_prefix(self) -> None:
        # No downloadId on the files -> fall back to the content_path prefix.
        cmds = [_command(files=[{"path": "/d/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, "no-hash", _paths("/d/folder"), set())

    def test_folder_import_matches_by_translated_prefix(self) -> None:
        # A dead-tracked folder import POSTs the TRANSLATED path. The raw
        # qBittorrent prefix matches nothing, the Sonarr-visible one must.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": []}])]
        assert manual_import_in_flight(
            cmds,
            "no-hash",
            _paths("/home/u/torrents/tv/folder", "/remote/tv/folder"),
            set(),
        )

    def test_folder_import_matches_by_episode_overlap(self) -> None:
        cmds = [_command(files=[{"path": "/elsewhere/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, "no-hash", _paths("/other"), {9})

    def test_translated_command_with_empty_seed_matches_by_episode_arm(self) -> None:
        # The empty-seed edge: nothing to match by path (both views miss), the
        # episode-id arm still guards - it is translation-immune.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, "no-hash", _paths("/nowhere"), {9})

    def test_download_id_command_not_swept_by_path_overlap(self) -> None:
        # A command that DOES carry a (different) downloadId is never matched by
        # path/episode overlap - only the no-downloadId folder case falls back.
        cmds = [_command(files=[{"downloadId": "OTHER", "path": "/d/x.mkv", "episodeIds": [9]}])]
        assert not manual_import_in_flight(cmds, "abc", _paths("/d"), {9})

    def test_empty_command_list_not_in_flight(self) -> None:
        assert not manual_import_in_flight([], "abc", _paths("/d"), {9})


def _history(*events: tuple[str, str]) -> list[HistoryRecord]:
    """History records from `(eventType, date)` pairs, newest first (as the probe reads)."""

    return [
        HistoryRecord.model_validate({"eventType": event, "date": date, "id": len(events) - index})
        for index, (event, date) in enumerate(events)
    ]


class TestClassifyDownloadHistory:
    """The dead-tracked probe: the newest relevant event decides, others are skipped."""

    def test_newest_imported_is_dead_tracked(self) -> None:
        verdict = classify_download_history(
            _history(("downloadFolderImported", "2026-06-20T06:15:30Z"), ("grabbed", "2026-06-19T00:00:00Z")),
        )
        assert verdict.dead_tracked
        assert verdict.event == "imported"
        assert verdict.date == "2026-06-20T06:15:30Z"

    def test_newest_failed_is_dead_tracked(self) -> None:
        verdict = classify_download_history(_history(("downloadFailed", "2026-01-01T00:00:00Z")))
        assert verdict.dead_tracked
        assert verdict.event == "failed"

    def test_newest_ignored_is_dead_tracked(self) -> None:
        verdict = classify_download_history(_history(("downloadIgnored", "2026-01-01T00:00:00Z")))
        assert verdict.dead_tracked
        assert verdict.event == "ignored"

    def test_grabbed_after_old_failure_is_clean(self) -> None:
        # Sonarr itself re-grabbed the hash after an old failure: genuinely
        # Downloading - the noisy branch must not claim it.
        verdict = classify_download_history(
            _history(("grabbed", "2026-07-01T00:00:00Z"), ("downloadFailed", "2026-01-01T00:00:00Z")),
        )
        assert not verdict.dead_tracked
        assert verdict.event is None

    def test_irrelevant_events_are_skipped_not_decided_on(self) -> None:
        # episodeFileDeleted is newer than the import but is NOT one of the four
        # tracked-state events - the verdict must come from the import below it.
        verdict = classify_download_history(
            _history(
                ("episodeFileDeleted", "2026-07-15T00:00:00Z"),
                ("downloadFolderImported", "2026-06-20T00:00:00Z"),
            ),
        )
        assert verdict.dead_tracked
        assert verdict.event == "imported"

    def test_none_of_the_four_is_clean(self) -> None:
        verdict = classify_download_history(_history(("episodeFileDeleted", "2026-07-15T00:00:00Z")))
        assert not verdict.dead_tracked

    def test_empty_history_is_clean(self) -> None:
        assert not classify_download_history([]).dead_tracked

    def test_event_type_matches_casefolded(self) -> None:
        assert classify_download_history(_history(("DOWNLOADFOLDERIMPORTED", "d"))).dead_tracked


def _mapping(remote: str, local: str, *, host: str | None = None) -> RemotePathMapping:
    """One remote path mapping from the raw API field names."""

    return RemotePathMapping.model_validate({"host": host, "remotePath": remote, "localPath": local})


class TestTranslateDownloadPath:
    """The remote-path translation behind the folder-scan fallback."""

    def test_no_mappings_is_a_no_op(self) -> None:
        assert translate_download_path("/d/folder", [], "qbit") == "/d/folder"

    def test_prefix_translates_and_suffix_survives(self) -> None:
        # The live incident mapping: trailing slash on both stored paths.
        mappings = [_mapping("/home/u/torrents/4k-tv/", "/remote/torrents/4k-tv/")]
        assert (
            translate_download_path("/home/u/torrents/4k-tv/Show S01", mappings, None)
            == "/remote/torrents/4k-tv/Show S01"
        )

    def test_exact_match_translates_to_local_root(self) -> None:
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads", mappings, None) == "/data"

    def test_separator_boundary_is_respected(self) -> None:
        # /downloads must NOT prefix-match /downloads-x/f.
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads-x/f", mappings, None) == "/downloads-x/f"

    def test_trailing_slash_tolerated_on_either_side(self) -> None:
        assert translate_download_path("/d/f", [_mapping("/d/", "/l")], None) == "/l/f"
        assert translate_download_path("/d/f", [_mapping("/d", "/l/")], None) == "/l/f"

    def test_suffix_case_is_preserved(self) -> None:
        # Compare case-insensitively but never fold the suffix - POSIX targets
        # are case-sensitive.
        mappings = [_mapping("/Downloads", "/data")]
        assert translate_download_path("/downloads/Show S01/Ep.MKV", mappings, None) == "/data/Show S01/Ep.MKV"

    def test_windows_backslash_remote_path(self) -> None:
        mappings = [_mapping("C:\\torrents\\", "/data/torrents")]
        assert translate_download_path("C:\\torrents\\Show\\ep.mkv", mappings, None) == "/data/torrents/Show/ep.mkv"

    def test_longest_prefix_wins(self) -> None:
        mappings = [
            _mapping("/d", "/short"),
            _mapping("/d/tv", "/long"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, None) == "/long/Show"

    def test_host_equality_tiebreaks_equal_prefixes(self) -> None:
        mappings = [
            _mapping("/d", "/other-client", host="other"),
            _mapping("/d", "/ours", host="qbit.local"),
        ]
        assert translate_download_path("/d/Show", mappings, "QBIT.LOCAL") == "/ours/Show"

    def test_host_mismatch_never_excludes(self) -> None:
        # Sonarr's host is the download-client host as SONARR knows it -
        # routinely a different string from our qBittorrent host.
        mappings = [_mapping("/d", "/data", host="sonarr-view-of-qbit")]
        assert translate_download_path("/d/Show", mappings, "localhost") == "/data/Show"

    def test_longer_prefix_beats_host_match(self) -> None:
        mappings = [
            _mapping("/d", "/host-matched", host="qbit"),
            _mapping("/d/tv", "/longer", host="other"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, "qbit") == "/longer/Show"

    def test_mapping_missing_either_path_is_skipped(self) -> None:
        mappings = [_mapping("", "/data"), _mapping("/d", "")]
        assert translate_download_path("/d/Show", mappings, None) == "/d/Show"

    def test_single_file_content_path_translates(self) -> None:
        # A single-FILE torrent's content_path is the file itself.
        mappings = [_mapping("/d/", "/data/")]
        assert translate_download_path("/d/Show - 01.mkv", mappings, None) == "/data/Show - 01.mkv"


# A realistic subset of Sonarr's /api/v3/qualitydefinition, matched on the
# structured (source, resolution) pair (never on the display name). Validated
# from the raw wire shape, as the client boundary does.
_RAW_DEFS: list[dict[str, object]] = [
    {"quality": {"id": 4, "name": "HDTV-720p", "source": "television", "resolution": 720}},
    {"quality": {"id": 6, "name": "Bluray-720p", "source": "bluray", "resolution": 720}},
    {"quality": {"id": 9, "name": "HDTV-1080p", "source": "television", "resolution": 1080}},
    {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
    {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
    {"quality": {"id": 20, "name": "Bluray-1080p Remux", "source": "blurayRaw", "resolution": 1080}},
    {"quality": {"id": 19, "name": "Bluray-2160p", "source": "bluray", "resolution": 2160}},
    {"quality": {"id": 21, "name": "Bluray-2160p Remux", "source": "blurayRaw", "resolution": 2160}},
]
_DEFS: list[QualityDefinition] = [QualityDefinition.model_validate(d) for d in _RAW_DEFS]


def _resolved_name(model: QualityModel) -> str | None:
    """The emitted quality's display name, for asserting which definition won."""

    return model.quality.name if model.quality is not None else None


class TestParseQualityFromFilename:
    """`parse_quality_from_filename` extracts the `(source, resolution)` pair, including the blurayRaw remux case."""

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
    """`quality_axes_from_model` reads the `(source, resolution)` pair off a `QualityModel`.

    An unknown source or zero resolution is treated as undetermined, as is a None model.
    """

    def test_reads_structured_source_and_resolution(self) -> None:
        model = QualityModel.model_validate(
            {"quality": {"name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
        )
        assert quality_axes_from_model(model) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=1080,
        )

    def test_unknown_source_and_zero_resolution_are_undetermined(self) -> None:
        model = QualityModel.model_validate(
            {"quality": {"name": "Unknown", "source": "unknown", "resolution": 0}},
        )
        assert quality_axes_from_model(model) == ParsedQuality(source=None, resolution=None)

    def test_none_model_is_empty(self) -> None:
        assert quality_axes_from_model(None) == ParsedQuality()


class TestQualityAxesFromName:
    """`quality_axes_from_name` resolves a display name to its `(source, resolution)` pair via the definitions."""

    def test_resolves_default_name_to_axes(self) -> None:
        assert quality_axes_from_name("Bluray-2160p", _DEFS) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=2160,
        )

    def test_unset_or_unmatched_is_empty(self) -> None:
        assert quality_axes_from_name(None, _DEFS) == ParsedQuality()
        assert quality_axes_from_name("Not-A-Quality", _DEFS) == ParsedQuality()


class TestResolveQuality:
    """`resolve_quality` picks the winning value per axis - Sonarr's own parse, then ours, then the default.

    It resolves the pair to a definition, falling back to the candidate verbatim or an explicit Unknown
    when nothing resolves.
    """

    def test_sonarr_wins_over_ours_and_default(self) -> None:
        # Sonarr parsed (web, 1080). Our filename parse and the default disagree.
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
        assert model.quality is not None
        assert model.quality.id == 20

    def test_per_axis_fill_from_default(self) -> None:
        # User's example: we parsed (None, 1080). Default is Bluray-2160p ->
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
        # 720p has no remux definition. A (blurayRaw, 720) gracefully downgrades
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
        candidate = QualityModel.model_validate(
            {
                "quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
                "revision": {"version": 1, "real": 0, "isRepack": False},
            },
        )
        model = resolve_quality(
            ParsedQuality(),
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate,
        )
        assert model is candidate

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
        assert model.quality == Quality(id=0, name="Unknown", source="unknown", resolution=0)
        assert model.revision is not None

    def test_candidate_revision_is_preserved(self) -> None:
        # A repack/proper revision on the candidate carries onto the resolved model.
        sonarr = ParsedQuality(source=QualitySource.WEB, resolution=1080)
        candidate = QualityModel.model_validate(
            {
                "quality": {"name": "WEBDL-1080p", "source": "web", "resolution": 1080},
                "revision": {"version": 2, "real": 0, "isRepack": True},
            },
        )
        model = resolve_quality(sonarr, ParsedQuality(), ParsedQuality(), _DEFS, candidate)
        assert model.revision == Revision(version=2, real=0, isRepack=True)


class TestDeriveLanguages:
    """`derive_languages` returns both audio languages for a dual-audio release, or just the primary otherwise."""

    def test_dual_audio_returns_dual(self) -> None:
        assert derive_languages(True, ["Japanese", "English"], ["Japanese"]) == [
            "Japanese",
            "English",
        ]

    def test_single_audio_returns_single(self) -> None:
        assert derive_languages(False, ["Japanese", "English"], ["Japanese"]) == ["Japanese"]


class TestResolveWaitMode:
    """`resolve_wait_mode` prefers the CLI override over the config value, defaulting to OFF when neither is set."""

    def test_cli_wins_over_config(self) -> None:
        assert resolve_wait_mode(ImportWaitMode.OFF, ImportWaitMode.HYBRID) is ImportWaitMode.OFF

    def test_config_used_when_no_cli(self) -> None:
        assert resolve_wait_mode(None, ImportWaitMode.BLOCKING) is ImportWaitMode.BLOCKING

    def test_default_off_when_neither(self) -> None:
        assert resolve_wait_mode(None, None) is ImportWaitMode.OFF


class TestPendingImportRoundTrip:
    """`PendingImport`'s JSON round-trip tolerates missing/unknown keys and defaults coverage/url to None.

    `display_label` falls back title -> infohash when the title/group is absent.
    """

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

    def test_display_label_is_title_dot_group_with_fallbacks(self) -> None:
        # The group disambiguates a series that grabbed several torrents. A
        # groupless record shows the bare title, a titleless one its infohash.
        rebuilt = PendingImport.from_json({"infohash": "h", "series_id": 1})
        assert rebuilt.display_label == "h"
        titled = PendingImport.from_json({"infohash": "h", "series_id": 1, "title": "Show"})
        assert titled.display_label == "Show"
        grouped = PendingImport.from_json(
            {"infohash": "h", "series_id": 1, "title": "Show", "release_group": "Era-Raws"},
        )
        assert grouped.display_label == f"Show{SEP}Era-Raws"

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


class TestSanitizeTorrentTelemetry:
    """MUTATION PIN: pins the pure telemetry sanitizer's clamps and sentinel folds.

    Covers the numeric-string `_as_float` path too - a cluster of ~10 surviving mutants.
    """

    @pytest.mark.parametrize(
        ("progress", "dlspeed", "eta", "completed", "size", "expected"),
        [
            # All-None getattr reads fold to the empty telemetry.
            (None, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # A clean row passes through untouched.
            (0.64, 3_200_000, 130, 1_800, 2_900, TorrentTelemetry(0.64, 3_200_000, 130, 1_800, 2_900)),
            # NaN progress (float and string) folds to 0.0, not a poisoned bar.
            (float("nan"), 100, 130, 50, 200, TorrentTelemetry(0.0, 100, 130, 50, 200)),
            ("nan", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Numeric-string progress parses. Junk folds to 0.0.
            ("0.75", None, None, None, None, TorrentTelemetry(0.75, None, None, None, None)),
            ("fast", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Progress clamps to [0, 1] on both ends.
            (1.5, None, None, None, None, TorrentTelemetry(1.0, None, None, None, None)),
            (-0.25, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Idle (0) and negative speeds read as "no speed", never a 0 B/s row.
            (0.5, 0, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, -5, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            # qBittorrent's 8_640_000 "infinite" eta and a 0/negative eta are unknown.
            # The last finite second still renders.
            (0.5, 100, 8_640_000, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 0, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 8_639_999, None, None, TorrentTelemetry(0.5, 100, 8_639_999, None, None)),
            # Zero/negative byte counts are unknown, not empty-progress readings.
            (0.5, None, None, 0, 0, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, None, None, -3, -1, TorrentTelemetry(0.5, None, None, None, None)),
            # An over-count clamps done to the total, never a >100% bar.
            (0.5, None, None, 500, 200, TorrentTelemetry(0.5, None, None, 200, 200)),
            # Bytes done without a known total still renders.
            (0.5, None, None, 100, None, TorrentTelemetry(0.5, None, None, 100, None)),
        ],
    )
    def test_edge_inputs(
        self,
        progress: object,
        dlspeed: object,
        eta: object,
        completed: object,
        size: object,
        expected: TorrentTelemetry,
    ) -> None:
        assert sanitize_torrent_telemetry(progress, dlspeed, eta, completed, size) == expected


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in WaitOutcome} == {"COMPLETE", "ERRORED", "MISSING"}


def test_import_readiness_members_exist() -> None:
    assert {o.name for o in ImportReadiness} == {"IMPORTED", "RETRY", "LEAVE"}


class TestClassifyQueue:
    """classify_queue buckets trackedDownloadState values into one verdict."""

    def test_empty_steps_in(self) -> None:
        assert classify_queue([]) is QueueVerdict.STEP_IN

    def test_import_blocked_steps_in(self) -> None:
        assert classify_queue(["importBlocked"]) is QueueVerdict.STEP_IN

    def test_failed_steps_in(self) -> None:
        assert classify_queue(["failed"]) is QueueVerdict.STEP_IN

    def test_pending_is_pending_clean(self) -> None:
        # The importPending-always-waits invariant is structural now: the input
        # carries ONLY states, so no trackedDownloadStatus (warning/error) can
        # ever route a pending record to STEP_IN (a double-import).
        assert classify_queue(["importPending"]) is QueueVerdict.PENDING_CLEAN

    def test_downloading_waits(self) -> None:
        assert classify_queue(["downloading"]) is QueueVerdict.WAIT

    def test_in_motion_beats_blocked_to_avoid_racing(self) -> None:
        # Something is actively importing -> wait, don't race it, even if a sibling
        # record is blocked. A later poll re-evaluates once the import settles.
        assert classify_queue(["importing", "importBlocked"]) is QueueVerdict.WAIT

    def test_case_insensitive(self) -> None:
        assert classify_queue(["IMPORTBLOCKED"]) is QueueVerdict.STEP_IN
