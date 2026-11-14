# pyright: strict
"""The pure states over the import attempts' Sonarr reads: queue, in-flight import, commands, history."""

from datetime import datetime, timedelta

import pytest

from pearlarr.arr_models import CommandResource, HistoryPage
from pearlarr.pending_import import PendingImport
from pearlarr.sonarr_state import (
    CommandBlock,
    ContentPaths,
    DownloadIdentifiers,
    HistoryImport,
    QueueState,
    classify_commands,
    classify_download_history,
    classify_queue,
    disk_commands_changed_since,
    in_flight_disk_commands,
    manual_import_in_flight,
    placements_from_history,
    rescanned_since_add,
    sonarr_restarted,
)
from pearlarr.wait_types import PauseReason

from .builders import PENDING_IMPORT_ADDED, PENDING_IMPORT_STAMP, pending_import, queue_record, title_claim


class TestClassifyQueue:
    """`classify_queue` buckets queue records (state + importPending status) into one state."""

    def test_empty_is_not_in_queue(self) -> None:
        assert classify_queue([]) is QueueState.NOT_IN_QUEUE

    def test_import_blocked_gets_an_auto_import(self) -> None:
        assert classify_queue([queue_record("A", "importBlocked")]) is QueueState.STUCK

    def test_failed_gets_an_auto_import(self) -> None:
        assert classify_queue([queue_record("A", "failed")]) is QueueState.STUCK

    def test_clean_pending_is_pending_clean(self) -> None:
        assert classify_queue([queue_record("A", "importPending", status="ok")]) is QueueState.IMPORT_PENDING_OK

    def test_statusless_pending_is_pending_clean(self) -> None:
        # No reported status reads as clean: only an explicit warning/error mark
        # routes an importPending queue record away from the leave-it-to-Sonarr path.
        assert classify_queue([queue_record("A", "importPending", status=None)]) is QueueState.IMPORT_PENDING_OK

    def test_marked_pending_gets_an_auto_import(self) -> None:
        # warning/error on importPending = Sonarr's own import attempt failed and
        # it won't reliably retry (observed live: marked importPending torrents sat
        # 10+ minutes untouched) - so waiting only burns the import timeout.
        assert classify_queue([queue_record("A", "importPending", status="warning")]) is QueueState.STUCK
        assert classify_queue([queue_record("A", "importPending", status="error")]) is QueueState.STUCK

    def test_downloading_waits(self) -> None:
        assert classify_queue([queue_record("A", "downloading")]) is QueueState.WAIT

    def test_in_motion_beats_blocked_to_avoid_racing(self) -> None:
        # Something is actively importing -> wait, don't race it, even if a sibling
        # queue record is blocked. A later poll re-evaluates once the import settles.
        records = [queue_record("A", "importing"), queue_record("A", "importBlocked")]
        assert classify_queue(records) is QueueState.IMPORTING

    def test_importing_outranks_a_clean_pending_sibling(self) -> None:
        # The copy itself is Sonarr's work (a pause), and it outranks every other state.
        records = [queue_record("A", "importPending", status="ok"), queue_record("A", "importing")]
        assert classify_queue(records) is QueueState.IMPORTING

    def test_downloading_beats_blocked_as_a_plain_wait(self) -> None:
        # In motion but not copying: Sonarr's view lags the finished torrent, so
        # the wait is the pending import's own (no pause).
        records = [queue_record("A", "downloading"), queue_record("A", "importBlocked")]
        assert classify_queue(records) is QueueState.WAIT

    def test_case_insensitive(self) -> None:
        assert classify_queue([queue_record("A", "IMPORTBLOCKED")]) is QueueState.STUCK
        assert classify_queue([queue_record("A", "importPending", status="WARNING")]) is QueueState.STUCK


def _command(
    *,
    name: str = "ManualImport",
    status: str = "started",
    files: list[dict[str, object]] | None = None,
    command_id: int = 0,
    started: datetime | None = None,
) -> CommandResource:
    """A `CommandResource` from the raw command fields the guards read."""

    raw: dict[str, object] = {"id": command_id, "name": name, "status": status, "body": {"files": files or []}}
    if started is not None:
        raw["started"] = started.isoformat()
    return CommandResource.model_validate(raw)


def _paths(raw: str, sonarr_visible: str | None = None) -> ContentPaths:
    """A `ContentPaths` pair. The Sonarr view defaults to the raw path (untranslated)."""

    return ContentPaths(raw=raw, sonarr_visible=sonarr_visible if sonarr_visible is not None else raw)


class TestManualImportInFlight:
    """The pure in-flight guard over the /api/v3/command list."""

    def test_matching_download_id_is_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "ABC", "episodeIds": [1]}])]
        # Case-insensitive match on the hash.
        assert manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), set()))

    def test_completed_command_is_not_in_flight(self) -> None:
        cmds = [_command(status="completed", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), set()))

    def test_non_manual_import_command_ignored(self) -> None:
        cmds = [_command(name="ProcessMonitoredDownloads", files=[{"downloadId": "ABC"}])]
        assert not manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), set()))

    def test_unrelated_download_id_not_in_flight(self) -> None:
        cmds = [_command(files=[{"downloadId": "OTHER"}])]
        assert not manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), set()))

    def test_queued_status_counts_as_in_flight(self) -> None:
        cmds = [_command(status="queued", files=[{"downloadId": "ABC"}])]
        assert manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), set()))

    def test_folder_import_matches_by_path_prefix(self) -> None:
        # No downloadId on the files -> fall back to the content_path prefix.
        cmds = [_command(files=[{"path": "/d/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadIdentifiers("no-hash", _paths("/d/folder"), set()))

    def test_folder_import_matches_by_translated_prefix(self) -> None:
        # A folder import for a closed history POSTs the TRANSLATED path. The raw
        # qBittorrent prefix matches nothing, the Sonarr-visible one must.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": []}])]
        assert manual_import_in_flight(
            cmds,
            DownloadIdentifiers("no-hash", _paths("/home/u/torrents/tv/folder", "/remote/tv/folder"), set()),
        )

    def test_folder_import_matches_by_episode_overlap(self) -> None:
        cmds = [_command(files=[{"path": "/elsewhere/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadIdentifiers("no-hash", _paths("/other"), {9}))

    def test_translated_command_with_empty_file_map_matches_by_episode_branch(self) -> None:
        # The empty-file-map edge: nothing to match by path (both views fail), the
        # episode-id branch still guards - it is translation-immune.
        cmds = [_command(files=[{"path": "/remote/tv/folder/ep.mkv", "episodeIds": [9]}])]
        assert manual_import_in_flight(cmds, DownloadIdentifiers("no-hash", _paths("/nowhere"), {9}))

    def test_download_id_command_not_swept_by_path_overlap(self) -> None:
        # A command that DOES carry a (different) downloadId is never matched by
        # path/episode overlap - only the no-downloadId folder case falls back.
        cmds = [_command(files=[{"downloadId": "OTHER", "path": "/d/x.mkv", "episodeIds": [9]}])]
        assert not manual_import_in_flight(cmds, DownloadIdentifiers("abc", _paths("/d"), {9}))

    def test_empty_command_list_not_in_flight(self) -> None:
        assert not manual_import_in_flight([], DownloadIdentifiers("abc", _paths("/d"), {9}))


class TestClassifyCommands:
    """The state layer: precedence + ownership over one command snapshot."""

    def test_clear_snapshot_allows_the_auto_import(self) -> None:
        assert classify_commands([], DownloadIdentifiers("abc", _paths("/d"), set()), lambda _: False) is None

    def test_download_id_match_is_an_auto_import_without_issued_ids(self) -> None:
        # The download-id proof alone owns the copy (survives restarts).
        cmds = [_command(files=[{"downloadId": "ABC"}])]
        block = classify_commands(cmds, DownloadIdentifiers("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.AUTO_IMPORT

    def test_unproven_folder_match_blocks_without_a_claim(self) -> None:
        # Possibly foreign: it blocks (and the import clock pauses) but never claims the import.
        cmds = [_command(files=[{"path": "/d/ep.mkv", "episodeIds": []}])]
        block = classify_commands(cmds, DownloadIdentifiers("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.IN_FLIGHT_IMPORT

    def test_issued_id_owns_an_unproven_folder_match(self) -> None:
        cmds = [_command(command_id=7, files=[{"path": "/d/ep.mkv", "episodeIds": []}])]
        block = classify_commands(cmds, DownloadIdentifiers("abc", _paths("/d"), set()), lambda cid: cid == 7)
        assert block is CommandBlock.AUTO_IMPORT

    def test_pearlarrs_disk_command_blocks_without_claiming_the_import(self) -> None:
        cmds = [_command(command_id=7, name="RenameFiles")]
        block = classify_commands(cmds, DownloadIdentifiers("abc", _paths("/d"), set()), lambda cid: cid == 7)
        assert block is CommandBlock.DISK_COMMAND

    def test_another_downloads_disk_command_blocks(self) -> None:
        cmds = [_command(command_id=9, name="ProcessMonitoredDownloads")]
        block = classify_commands(cmds, DownloadIdentifiers("abc", _paths("/d"), set()), lambda _: False)
        assert block is CommandBlock.DISK_COMMAND


_BLOCK_WAITS = {
    CommandBlock.AUTO_IMPORT: (True, PauseReason.IMPORT),
    CommandBlock.IN_FLIGHT_IMPORT: (False, PauseReason.IMPORT),
    CommandBlock.DISK_COMMAND: (False, PauseReason.BUSY),
    CommandBlock.DISK_COMMAND_CHANGED: (False, PauseReason.BUSY),
    CommandBlock.SONARR_RESTARTED: (False, PauseReason.NONE),
}


class TestCommandBlock:
    """The waiting attempt result each command block calls for."""

    @pytest.mark.parametrize("block", list(CommandBlock))
    def test_waiting(self, block: CommandBlock) -> None:
        probe = block.waiting()

        assert (probe.command_issued, probe.pause_reason) == _BLOCK_WAITS[block]


class TestInFlightDiskCommands:
    """The pure disk-command guard over the same /api/v3/command list."""

    def test_started_process_monitored_downloads_blocks(self) -> None:
        assert in_flight_disk_commands([_command(name="ProcessMonitoredDownloads")])

    def test_queued_pass_blocks(self) -> None:
        assert in_flight_disk_commands([_command(name="ProcessMonitoredDownloads", status="queued")])

    def test_completed_pass_never_blocks(self) -> None:
        assert not in_flight_disk_commands([_command(name="ProcessMonitoredDownloads", status="completed")])

    def test_legacy_folder_scan_blocks(self) -> None:
        assert in_flight_disk_commands([_command(name="DownloadedEpisodesScan")])

    def test_rename_sweep_blocks(self) -> None:
        # Any RequiresDiskAccess command blocks, not just the import commands.
        assert in_flight_disk_commands([_command(name="RenameFiles")])

    def test_running_manual_import_blocks(self) -> None:
        # A foreign ManualImport blocks an auto import the same way (pearlarr's is also
        # caught by manual_import_in_flight; this guard needs no file match).
        assert in_flight_disk_commands([_command(name="ManualImport")])

    def test_case_folded_match(self) -> None:
        assert in_flight_disk_commands([_command(name="processMONITOREDdownloads", status="Started")])

    def test_non_disk_commands_ignored(self) -> None:
        cmds = [_command(name="RefreshMonitoredDownloads"), _command(name="RssSync")]
        assert not in_flight_disk_commands(cmds)

    def test_empty_command_list(self) -> None:
        assert not in_flight_disk_commands([])


def _rescan(*, started: datetime | None, status: str = "completed") -> CommandResource:
    """A RefreshMonitoredDownloads command with the given start time."""

    return _command(name="RefreshMonitoredDownloads", status=status, started=started)


class TestRescannedSinceAdd:
    """The untracked proof: a completed rescan that started a minute or more after the add."""

    def test_a_minute_after_the_add_proves(self) -> None:
        assert rescanned_since_add(
            [_rescan(started=PENDING_IMPORT_ADDED + timedelta(seconds=60))], PENDING_IMPORT_STAMP
        )

    def test_under_a_minute_after_the_add_does_not_prove(self) -> None:
        assert not rescanned_since_add(
            [_rescan(started=PENDING_IMPORT_ADDED + timedelta(seconds=59))], PENDING_IMPORT_STAMP
        )

    def test_a_junk_stamp_does_not_prove(self) -> None:
        assert not rescanned_since_add([_rescan(started=PENDING_IMPORT_ADDED + timedelta(days=1))], "junk")

    def test_a_rescan_without_a_start_does_not_prove(self) -> None:
        assert not rescanned_since_add([_rescan(started=None)], PENDING_IMPORT_STAMP)

    def test_a_running_rescan_does_not_prove(self) -> None:
        assert not rescanned_since_add(
            [_rescan(started=PENDING_IMPORT_ADDED + timedelta(days=1), status="started")], PENDING_IMPORT_STAMP
        )

    def test_another_completed_command_does_not_prove(self) -> None:
        rss = _command(name="RssSync", status="completed", started=PENDING_IMPORT_ADDED + timedelta(days=1))
        assert not rescanned_since_add([rss], PENDING_IMPORT_STAMP)


@pytest.mark.parametrize(
    ("commands", "restarted"),
    [
        ([], True),
        ([_rescan(started=PENDING_IMPORT_ADDED, status="started")], True),
        ([_rescan(started=PENDING_IMPORT_ADDED)], False),
    ],
    ids=["empty", "started-only", "completed"],
)
def test_sonarr_restarted(commands: list[CommandResource], restarted: bool) -> None:
    assert sonarr_restarted(commands) is restarted


class TestDiskCommandsChangedSince:
    """The pre-POST check: disk commands whose `(id, status)` pair the baseline did not list."""

    def test_a_pass_with_a_new_id_is_listed(self) -> None:
        new_pass = _command(name="ProcessMonitoredDownloads", command_id=9)
        assert disk_commands_changed_since([], [new_pass]) == [new_pass]

    def test_the_same_pass_in_both_is_not_listed(self) -> None:
        listed = [_command(name="ProcessMonitoredDownloads", command_id=9)]
        assert disk_commands_changed_since(listed, listed) == []

    def test_a_pass_that_moved_from_queued_to_completed_is_listed(self) -> None:
        before = [_command(name="ProcessMonitoredDownloads", status="queued", command_id=9)]
        after = [_command(name="ProcessMonitoredDownloads", status="completed", command_id=9)]
        assert disk_commands_changed_since(before, after) == after

    def test_a_new_non_disk_command_is_not_listed(self) -> None:
        assert disk_commands_changed_since([], [_command(name="RssSync", command_id=9)]) == []

    def test_an_id_less_pass_new_since_the_baseline_is_listed(self) -> None:
        new_pass = _command(name="ProcessMonitoredDownloads")
        assert disk_commands_changed_since([], [new_pass]) == [new_pass]


def _event_row(event: str, date: str) -> dict[str, object]:
    """A plain history row."""

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
    """A history page from `(eventType, date)` pairs or raw rows, newest first (as the attempt reads)."""

    records = [
        {**(row if isinstance(row, dict) else _event_row(*row)), "id": len(rows) - index}
        for index, row in enumerate(rows)
    ]
    return HistoryPage.model_validate({"records": records, "totalRecords": total_records})


class TestClassifyDownloadHistory:
    """The history-closed check: the newest relevant event decides, others are skipped."""

    def test_newest_imported_closes_the_history(self) -> None:
        verdict = classify_download_history(
            _history(("downloadFolderImported", "2026-06-20T06:15:30Z"), ("grabbed", "2026-06-19T00:00:00Z")),
        )
        assert verdict.history_closed
        assert verdict.event == "imported"
        assert verdict.date == "2026-06-20T06:15:30Z"

    def test_newest_failed_closes_the_history(self) -> None:
        verdict = classify_download_history(_history(("downloadFailed", "2026-01-01T00:00:00Z")))
        assert verdict.history_closed
        assert verdict.event == "failed"

    def test_newest_ignored_closes_the_history(self) -> None:
        verdict = classify_download_history(_history(("downloadIgnored", "2026-01-01T00:00:00Z")))
        assert verdict.history_closed
        assert verdict.event == "ignored"

    def test_grabbed_after_old_failure_is_clean(self) -> None:
        # Sonarr itself re-grabbed the hash after an old failure: genuinely
        # Downloading - the noisy branch must not claim it.
        verdict = classify_download_history(
            _history(("grabbed", "2026-07-01T00:00:00Z"), ("downloadFailed", "2026-01-01T00:00:00Z")),
        )
        assert not verdict.history_closed
        assert verdict.event is None

    def test_irrelevant_events_are_skipped_not_decided_on(self) -> None:
        # episodeFileDeleted is newer than the import but is NOT one of the four
        # tracked-state events - the state must come from the import below it.
        verdict = classify_download_history(
            _history(
                ("episodeFileDeleted", "2026-07-15T00:00:00Z"),
                ("downloadFolderImported", "2026-06-20T00:00:00Z"),
            ),
        )
        assert verdict.history_closed
        assert verdict.event == "imported"

    def test_none_of_the_four_is_clean(self) -> None:
        verdict = classify_download_history(_history(("episodeFileDeleted", "2026-07-15T00:00:00Z")))
        assert not verdict.history_closed

    def test_empty_history_is_clean(self) -> None:
        assert not classify_download_history(HistoryPage()).history_closed

    def test_event_type_matches_casefolded(self) -> None:
        assert classify_download_history(_history(("DOWNLOADFOLDERIMPORTED", "d"))).history_closed

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
        assert verdict.history_closed
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
        # An older import's rows below the grab belong to another import.
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
        assert (failed.history_closed, failed.import_rows) == (True, ())
        assert (bare.history_closed, bare.import_rows) == (True, ())

    def test_a_cut_page_without_the_grab_carries_nothing(self) -> None:
        # The page holds fewer rows than the envelope counts and no grab bounds
        # the import: the rows may go on past the page, so none are trusted.
        verdict = classify_download_history(_history(_import_row(101, "/d/a.mkv", "d"), total_records=150))
        assert verdict.history_closed
        assert verdict.import_rows == ()

    def test_a_cut_page_with_the_grab_inside_is_trusted(self) -> None:
        verdict = classify_download_history(
            _history(_import_row(101, "/d/a.mkv", "d"), ("grabbed", "d"), total_records=150),
        )
        assert verdict.import_rows == (HistoryImport("/d/a.mkv", 101, 7),)


def _unplaced(**overrides: object) -> PendingImport:
    """A pending import with an empty file map over a one-file listing resolved to episode 101."""

    fields: dict[str, object] = {
        "file_episode_map": {},
        "ordered_episode_ids": [101],
        "seadex_files": ["Show - 01.mkv"],
    }
    fields.update(overrides)
    return pending_import(**fields)


class TestPlacementsFromHistory:
    """Sonarr's import rows place only the listing files the pending import's own file map leaves unplaced."""

    def test_a_row_outside_the_listing_is_dropped(self) -> None:
        rows = [HistoryImport("/d/Show - 01.mkv", 101, 7), HistoryImport("/d/Extra.mkv", 101, 7)]
        assert placements_from_history(rows, _unplaced()) == {"show - 01.mkv": (101,)}

    def test_an_excluded_file_is_never_claimed(self) -> None:
        pending = _unplaced(
            ordered_episode_ids=[],
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            excluded_files=["show - 02.mkv"],
        )
        assert placements_from_history([HistoryImport("/d/Show - 02.mkv", 102, 7)], pending) == {}

    def test_a_mapped_file_is_left_to_the_file_map(self) -> None:
        pending = _unplaced(
            file_episode_map={"Show - 01.mkv": [101]},
            ordered_episode_ids=[101, 102],
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
        )
        rows = [HistoryImport("/d/Show - 01.mkv", 102, 7), HistoryImport("/d/Show - 02.mkv", 102, 7)]
        assert placements_from_history(rows, pending) == {"show - 02.mkv": (102,)}

    def test_another_series_row_is_dropped(self) -> None:
        assert placements_from_history([HistoryImport("/d/Show - 01.mkv", 101, 8)], _unplaced()) == {}

    def test_a_row_is_admitted_by_the_claim_on_its_series(self) -> None:
        # A row goes only where a claim on ITS series admits the episode: the other series' claim
        # holding the same id never admits it.
        pending = _unplaced(
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            claims=(
                title_claim(anilist_id=1, series_id=7, ordered_episode_ids=[101]),
                title_claim(anilist_id=2, series_id=8, ordered_episode_ids=[201]),
            ),
        )
        rows = [HistoryImport("/d/Show - 01.mkv", 201, 8), HistoryImport("/d/Show - 02.mkv", 201, 7)]
        assert placements_from_history(rows, pending) == {"show - 01.mkv": (201,)}

    def test_the_claims_window_scopes_the_ids_when_present(self) -> None:
        # A row outside the claim's window is a sibling's claimed coverage (or a parse pearlarr
        # would not have made). A claim with no window, the specials shape, accepts
        # every in-series row.
        rows = [HistoryImport("/d/Show - 01.mkv", 999, 7)]
        assert placements_from_history(rows, _unplaced()) == {}
        assert placements_from_history(rows, _unplaced(ordered_episode_ids=[])) == {"show - 01.mkv": (999,)}

    def test_a_multi_episode_file_groups_its_sorted_ids(self) -> None:
        pending = _unplaced(ordered_episode_ids=[101, 102], seadex_files=["Show - 01-02.mkv"])
        rows = [HistoryImport("/d/Show - 01-02.mkv", 102, 7), HistoryImport("/d/Show - 01-02.mkv", 101, 7)]
        assert placements_from_history(rows, pending) == {"show - 01-02.mkv": (101, 102)}

    def test_a_remote_path_converts_to_the_normalized_file_name(self) -> None:
        rows = [HistoryImport("C:\\downloads\\Show S01/SHOW - 01.MKV", 101, 7)]
        assert placements_from_history(rows, _unplaced()) == {"show - 01.mkv": (101,)}

    def test_no_rows_give_no_placements(self) -> None:
        assert placements_from_history([], _unplaced()) == {}
