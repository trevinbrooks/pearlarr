# pyright: strict
"""The pure verdicts over the import probes' Sonarr reads: queue, in-flight import, commands, history, paths."""

from pearlarr.manual_import import Deferral, PendingImport
from pearlarr.probe_verdicts import (
    CommandBlock,
    ContentPaths,
    DownloadMatch,
    HistoryImport,
    QueueVerdict,
    classify_commands,
    classify_download_history,
    classify_queue,
    manual_import_in_flight,
    placements_from_history,
    sonarr_process_pass_running,
    started_disk_commands,
    translate_download_path,
)
from pearlarr.seadex_types import CommandResource, HistoryPage, RemotePathMapping

from .builders import pending_import, queue_record


class TestClassifyQueue:
    """`classify_queue` buckets queue records (state + pending status) into one verdict."""

    def test_empty_steps_in(self) -> None:
        assert classify_queue([]) is QueueVerdict.STEP_IN

    def test_import_blocked_steps_in(self) -> None:
        assert classify_queue([queue_record("A", "importBlocked")]) is QueueVerdict.STEP_IN

    def test_failed_steps_in(self) -> None:
        assert classify_queue([queue_record("A", "failed")]) is QueueVerdict.STEP_IN

    def test_clean_pending_is_pending_clean(self) -> None:
        assert classify_queue([queue_record("A", "importPending", status="ok")]) is QueueVerdict.PENDING_CLEAN

    def test_statusless_pending_is_pending_clean(self) -> None:
        # No reported status folds to clean: only an explicit warning/error flag
        # routes a pending record away from the defer-to-Sonarr path.
        assert classify_queue([queue_record("A", "importPending", status=None)]) is QueueVerdict.PENDING_CLEAN

    def test_flagged_pending_steps_in(self) -> None:
        # warning/error on importPending = Sonarr's own import attempt failed and
        # it won't reliably retry (observed live: flagged-pending torrents sat
        # 10+ minutes untouched) - so waiting only burns the readiness deadline.
        assert classify_queue([queue_record("A", "importPending", status="warning")]) is QueueVerdict.STEP_IN
        assert classify_queue([queue_record("A", "importPending", status="error")]) is QueueVerdict.STEP_IN

    def test_downloading_waits(self) -> None:
        assert classify_queue([queue_record("A", "downloading")]) is QueueVerdict.WAIT

    def test_in_motion_beats_blocked_to_avoid_racing(self) -> None:
        # Something is actively importing -> wait, don't race it, even if a sibling
        # record is blocked. A later poll re-evaluates once the import settles.
        records = [queue_record("A", "importing"), queue_record("A", "importBlocked")]
        assert classify_queue(records) is QueueVerdict.IMPORTING

    def test_importing_outranks_a_clean_pending_sibling(self) -> None:
        # The copy itself is Sonarr's wait (credited), and it outranks every other reading.
        records = [queue_record("A", "importPending", status="ok"), queue_record("A", "importing")]
        assert classify_queue(records) is QueueVerdict.IMPORTING

    def test_downloading_beats_blocked_as_a_plain_wait(self) -> None:
        # In motion but not copying: Sonarr's view lags the finished torrent, so
        # the wait is the record's own (uncredited).
        records = [queue_record("A", "downloading"), queue_record("A", "importBlocked")]
        assert classify_queue(records) is QueueVerdict.WAIT

    def test_case_insensitive(self) -> None:
        assert classify_queue([queue_record("A", "IMPORTBLOCKED")]) is QueueVerdict.STEP_IN
        assert classify_queue([queue_record("A", "importPending", status="WARNING")]) is QueueVerdict.STEP_IN


def _command(
    *,
    name: str = "ManualImport",
    status: str = "started",
    files: list[dict[str, object]] | None = None,
    command_id: int = 0,
) -> CommandResource:
    """A `CommandResource` from the raw command fields the guards read."""

    return CommandResource.model_validate(
        {"id": command_id, "name": name, "status": status, "body": {"files": files or []}},
    )


def _paths(raw: str, sonarr_visible: str | None = None) -> ContentPaths:
    """A `ContentPaths` pair. The Sonarr view defaults to the raw path (untranslated)."""

    return ContentPaths(raw=raw, sonarr_visible=sonarr_visible if sonarr_visible is not None else raw)


class TestManualImportInFlight:
    """The pure in-flight guard over the /api/v3/command list."""

    def test_matching_download_id_is_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "ABC", "episodeIds": [1]}])]
        # Case-insensitive match on the infohash.
        assert manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), set()))

    def test_completed_command_is_not_in_flight(self) -> None:
        cmds = [_command(status="completed", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), set()))

    def test_non_manual_import_command_ignored(self) -> None:
        cmds = [_command(name="ProcessMonitoredDownloads", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), set()))

    def test_unrelated_download_id_not_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "OTHER"}])]
        assert not manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), set()))

    def test_queued_status_counts_as_in_flight(self) -> None:
        cmds = [_command(status="queued", files=[{"downloadId": "ABC"}])]
        assert manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), set()))

    def test_folder_import_matches_by_path_prefix(self) -> None:
        # No downloadId on the files -> fall back to the content_path prefix.
        cmds = [_command(files=[{"path": "/d/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadMatch("no-hash", _paths("/d/folder"), set()))

    def test_folder_import_matches_by_translated_prefix(self) -> None:
        # A dead-tracked folder import POSTs the TRANSLATED path. The raw
        # qBittorrent prefix matches nothing, the Sonarr-visible one must.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": []}])]
        assert manual_import_in_flight(
            cmds,
            DownloadMatch("no-hash", _paths("/home/u/torrents/tv/folder", "/remote/tv/folder"), set()),
        )

    def test_folder_import_matches_by_episode_overlap(self) -> None:
        cmds = [_command(files=[{"path": "/elsewhere/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadMatch("no-hash", _paths("/other"), {9}))

    def test_translated_command_with_empty_seed_matches_by_episode_arm(self) -> None:
        # The empty-seed edge: nothing to match by path (both views miss), the
        # episode-id arm still guards - it is translation-immune.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadMatch("no-hash", _paths("/nowhere"), {9}))

    def test_download_id_command_not_swept_by_path_overlap(self) -> None:
        # A command that DOES carry a (different) downloadId is never matched by
        # path/episode overlap - only the no-downloadId folder case falls back.
        cmds = [_command(files=[{"downloadId": "OTHER", "path": "/d/x.mkv", "episodeIds": [9]}])]
        assert not manual_import_in_flight(cmds, DownloadMatch("abc", _paths("/d"), {9}))

    def test_empty_command_list_not_in_flight(self) -> None:
        assert not manual_import_in_flight([], DownloadMatch("abc", _paths("/d"), {9}))


class TestClassifyCommands:
    """The verdict layer: precedence + ownership over one command snapshot."""

    def test_clear_snapshot_steps_in(self) -> None:
        assert classify_commands([], DownloadMatch("abc", _paths("/d"), set()), lambda _: False) is None

    def test_download_id_match_is_our_import_without_issued_ids(self) -> None:
        # The download-id proof alone owns the copy (survives restarts).
        cmds = [_command(files=[{"downloadId": "ABC"}])]
        block = classify_commands(cmds, DownloadMatch("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.OWN_IMPORT

    def test_unproven_folder_match_blocks_without_a_claim(self) -> None:
        # Possibly foreign: it blocks (and the wait is credited) but never claims the import.
        cmds = [_command(files=[{"path": "/d/ep.mkv", "episodeIds": []}])]
        block = classify_commands(cmds, DownloadMatch("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.IN_FLIGHT_IMPORT

    def test_issued_id_owns_an_unproven_folder_match(self) -> None:
        cmds = [_command(command_id=7, files=[{"path": "/d/ep.mkv", "episodeIds": []}])]
        block = classify_commands(cmds, DownloadMatch("abc", _paths("/d"), set()), lambda cid: cid == 7)
        assert block is CommandBlock.OWN_IMPORT

    def test_own_disk_command_blocks_without_claiming_the_import(self) -> None:
        cmds = [_command(command_id=7, name="RenameFiles")]
        block = classify_commands(cmds, DownloadMatch("abc", _paths("/d"), set()), lambda cid: cid == 7)
        assert block is CommandBlock.DISK_COMMAND

    def test_foreign_disk_command_blocks(self) -> None:
        cmds = [_command(command_id=9, name="ProcessMonitoredDownloads")]
        block = classify_commands(cmds, DownloadMatch("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.DISK_COMMAND

    def test_block_properties(self) -> None:
        # Only our own import claims `command_issued`; every import blocks as an
        # IMPORT deferral, a disk command as BUSY.
        assert [block.claims_import for block in CommandBlock] == [True, False, False]
        assert CommandBlock.OWN_IMPORT.deferral is Deferral.IMPORT
        assert CommandBlock.IN_FLIGHT_IMPORT.deferral is Deferral.IMPORT
        assert CommandBlock.DISK_COMMAND.deferral is Deferral.BUSY


class TestStartedDiskCommands:
    """The pure disk-command guard over the same /api/v3/command list."""

    def test_started_process_monitored_downloads_defers(self) -> None:
        assert started_disk_commands([_command(name="ProcessMonitoredDownloads")])

    def test_queued_pass_never_defers(self) -> None:
        # A queued pass is near-permanently present during a wait (Sonarr pushes
        # one after every rescan, including ours), so deferring on it would
        # starve the step-in entirely.
        assert not started_disk_commands([_command(name="ProcessMonitoredDownloads", status="queued")])

    def test_completed_pass_never_defers(self) -> None:
        assert not started_disk_commands([_command(name="ProcessMonitoredDownloads", status="completed")])

    def test_legacy_folder_scan_defers(self) -> None:
        assert started_disk_commands([_command(name="DownloadedEpisodesScan")])

    def test_rename_sweep_defers(self) -> None:
        # Any started RequiresDiskAccess command queue-blocks a fresh
        # ManualImport, opening the stale-replay window - not just the passes.
        assert started_disk_commands([_command(name="RenameFiles")])

    def test_running_manual_import_defers(self) -> None:
        # A foreign ManualImport blocks ours the same way (our own is also
        # caught by manual_import_in_flight; this guard needs no file match).
        assert started_disk_commands([_command(name="ManualImport")])

    def test_case_folded_match(self) -> None:
        assert started_disk_commands([_command(name="processMONITOREDdownloads", status="Started")])

    def test_non_disk_commands_ignored(self) -> None:
        cmds = [_command(name="RefreshMonitoredDownloads"), _command(name="RssSync")]
        assert not started_disk_commands(cmds)

    def test_empty_command_list(self) -> None:
        assert not started_disk_commands([])


class TestSonarrProcessPassRunning:
    """The narrower rescan-absorb predicate: only the self-inflicted pass."""

    def test_started_pass_matches(self) -> None:
        assert sonarr_process_pass_running([_command(name="processMonitoredDownloads", status="Started")])

    def test_queued_pass_ignored(self) -> None:
        assert not sonarr_process_pass_running([_command(name="ProcessMonitoredDownloads", status="queued")])

    def test_foreign_disk_commands_ignored(self) -> None:
        # A started rename/import is the disk-command guard's job to defer on -
        # absorbing it in the rescan would stall the whole poll for its bound.
        cmds = [_command(name="RenameFiles"), _command(name="ManualImport")]
        assert not sonarr_process_pass_running(cmds)


def _event_row(event: str, date: str) -> dict[str, object]:
    """A bare history row."""

    return {"eventType": event, "date": date}


def _import_row(episode_id: int, path: str, date: str) -> dict[str, object]:
    """A `downloadFolderImported` row landing `path` on `episode_id` for series 7."""

    return {
        **_event_row("downloadFolderImported", date),
        "seriesId": 7,
        "episodeId": episode_id,
        "data": {"droppedPath": path},
    }


def _history(*rows: tuple[str, str] | dict[str, object], total_records: int = 0) -> HistoryPage:
    """A history page from `(eventType, date)` pairs or raw rows, newest first (as the probe reads)."""

    records = [
        {**(row if isinstance(row, dict) else _event_row(*row)), "id": len(rows) - index}
        for index, row in enumerate(rows)
    ]
    return HistoryPage.model_validate({"records": records, "totalRecords": total_records})


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
        assert not classify_download_history(HistoryPage()).dead_tracked

    def test_event_type_matches_casefolded(self) -> None:
        assert classify_download_history(_history(("DOWNLOADFOLDERIMPORTED", "d"))).dead_tracked

    def test_imported_verdict_carries_its_rows(self) -> None:
        # The newer file-deleted row is skipped over; the import rows above the
        # grab come back in page order, each carrying the series id.
        verdict = classify_download_history(
            _history(
                ("episodeFileDeleted", "2026-01-04T00:00:00Z"),
                _import_row(102, "/d/Show/Show - 02.mkv", "2026-01-03T00:00:00Z"),
                _import_row(101, "/d/Show/Show - 01.mkv", "2026-01-03T00:00:00Z"),
                ("grabbed", "2026-01-02T00:00:00Z"),
            ),
        )
        assert verdict.dead_tracked
        assert verdict.import_rows == (
            HistoryImport("/d/Show/Show - 02.mkv", 102, 7),
            HistoryImport("/d/Show/Show - 01.mkv", 101, 7),
        )

    def test_a_multi_episode_file_repeats_its_path(self) -> None:
        verdict = classify_download_history(
            _history(
                _import_row(102, "/d/Show - 01-02.mkv", "d"),
                _import_row(101, "/d/Show - 01-02.mkv", "d"),
                ("grabbed", "d"),
            ),
        )
        assert verdict.import_rows == (
            HistoryImport("/d/Show - 01-02.mkv", 102, 7),
            HistoryImport("/d/Show - 01-02.mkv", 101, 7),
        )

    def test_rows_stop_at_the_grab_below(self) -> None:
        # An older cycle's import rows below the grab belong to another import.
        verdict = classify_download_history(
            _history(
                _import_row(101, "/d/new.mkv", "d"),
                ("grabbed", "d"),
                _import_row(101, "/d/old.mkv", "d"),
                ("grabbed", "d"),
            ),
        )
        assert verdict.import_rows == (HistoryImport("/d/new.mkv", 101, 7),)

    def test_failed_and_bare_import_rows_carry_nothing(self) -> None:
        failed = classify_download_history(_history(("downloadFailed", "d"), ("grabbed", "d")))
        bare = classify_download_history(_history(("downloadFolderImported", "d"), ("grabbed", "d")))
        assert (failed.dead_tracked, failed.import_rows) == (True, ())
        assert (bare.dead_tracked, bare.import_rows) == (True, ())

    def test_a_cut_page_without_the_grab_carries_nothing(self) -> None:
        # The page holds fewer rows than the envelope counts and no grab bounds
        # the cycle: the rows may go on past the page, so none are trusted.
        verdict = classify_download_history(_history(_import_row(101, "/d/a.mkv", "d"), total_records=150))
        assert verdict.dead_tracked
        assert verdict.import_rows == ()

    def test_a_cut_page_with_the_grab_inside_is_trusted(self) -> None:
        verdict = classify_download_history(
            _history(_import_row(101, "/d/a.mkv", "d"), ("grabbed", "d"), total_records=150),
        )
        assert verdict.import_rows == (HistoryImport("/d/a.mkv", 101, 7),)


def _unplaced(**overrides: object) -> PendingImport:
    """A record with an empty map over a one-file listing resolved to episode 101."""

    fields: dict[str, object] = {
        "file_episode_map": {},
        "episode_ids": [],
        "ordered_episode_ids": [101],
        "seadex_files": ["Show - 01.mkv"],
    }
    fields.update(overrides)
    return pending_import(**fields)


class TestPlacementsFromHistory:
    """Sonarr's import rows fill only the listing files the record's own map leaves unplaced."""

    def test_a_row_outside_the_listing_is_dropped(self) -> None:
        rows = [HistoryImport("/d/Show - 01.mkv", 101, 7), HistoryImport("/d/Extra.mkv", 101, 7)]
        assert placements_from_history(rows, _unplaced()) == {"show - 01.mkv": [101]}

    def test_an_excluded_file_is_never_claimed(self) -> None:
        pending = _unplaced(
            ordered_episode_ids=[],
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            excluded_files=["show - 02.mkv"],
        )
        assert placements_from_history([HistoryImport("/d/Show - 02.mkv", 102, 7)], pending) == {}

    def test_a_mapped_file_is_left_to_its_seed(self) -> None:
        pending = _unplaced(
            file_episode_map={"Show - 01.mkv": [101]},
            ordered_episode_ids=[101, 102],
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
        )
        rows = [HistoryImport("/d/Show - 01.mkv", 102, 7), HistoryImport("/d/Show - 02.mkv", 102, 7)]
        assert placements_from_history(rows, pending) == {"show - 02.mkv": [102]}

    def test_another_series_row_is_dropped(self) -> None:
        assert placements_from_history([HistoryImport("/d/Show - 01.mkv", 101, 8)], _unplaced()) == {}

    def test_the_resolved_set_scopes_the_ids_when_present(self) -> None:
        # A row outside the resolved set is a sibling's slice (or a parse we
        # would not have made); an empty set, the legacy or specials shape,
        # accepts every in-series row.
        rows = [HistoryImport("/d/Show - 01.mkv", 999, 7)]
        assert placements_from_history(rows, _unplaced()) == {}
        assert placements_from_history(rows, _unplaced(ordered_episode_ids=[])) == {"show - 01.mkv": [999]}

    def test_a_multi_episode_file_groups_its_sorted_ids(self) -> None:
        pending = _unplaced(ordered_episode_ids=[101, 102], seadex_files=["Show - 01-02.mkv"])
        rows = [HistoryImport("/d/Show - 01-02.mkv", 102, 7), HistoryImport("/d/Show - 01-02.mkv", 101, 7)]
        assert placements_from_history(rows, pending) == {"show - 01-02.mkv": [101, 102]}

    def test_a_remote_path_folds_to_the_normalized_leaf(self) -> None:
        rows = [HistoryImport("C:\\downloads\\Show S01/SHOW - 01.MKV", 101, 7)]
        assert placements_from_history(rows, _unplaced()) == {"show - 01.mkv": [101]}

    def test_no_rows_give_no_placements(self) -> None:
        assert placements_from_history([], _unplaced()) == {}


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
