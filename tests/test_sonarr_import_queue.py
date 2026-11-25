# pyright: strict
"""Tests for what Sonarr's queue and commands make an import attempt do, and the queue record's removal.

An attempt waits, rescans or posts the import. Once the download is imported, its queue record is removed. An import
runs over a scripted Sonarr and is judged by the commands posted to it and the attempt result.
"""

import pytest

from pearlarr.arr_http import DeleteOutcome
from pearlarr.arr_models import CommandResource, QueueRecord
from pearlarr.output import Severity
from pearlarr.wait_types import AttemptKind, EffectStatus, PauseReason

from .builders import (
    FakeCacheStore,
    FakeClock,
    make_config,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
    queue_record,
)
from .fakes import FakeSonarrClient, diagnostic_messages, install_recording_hub
from .sonarr_import_builders import inflight_manual_import, make_sonarr_for_import, posted_imports, started_process_pass

_IMPORTING = [queue_record("ABC123", "importing")]


_CLEAN_PENDING = [queue_record("ABC123", "importPending", status="ok")]


class TestImportCompletedQueueState:
    """Sonarr's queue, its running commands and its settings decide whether an import attempt waits or posts."""

    @pytest.mark.parametrize(
        ("queue", "commands", "attempt_kind", "outcome"),
        [
            # Sonarr is copying this download. That is Sonarr's wait, so the clock pauses, and nothing is posted
            # over it even at the deadline.
            pytest.param(_IMPORTING, [], AttemptKind.ORDINARY, (PauseReason.IMPORT, False, 0), id="importing"),
            pytest.param(_IMPORTING, [], AttemptKind.FINAL, (PauseReason.IMPORT, False, 0), id="importing-final"),
            # Sonarr's view lags qBittorrent's "complete": the pending import's own wait, on a running clock.
            pytest.param(
                [queue_record("ABC123", "downloading")],
                [],
                AttemptKind.ORDINARY,
                (PauseReason.NONE, False, 0),
                id="downloading",
            ),
            # A clean importPending is left to Sonarr until the final attempt, which posts the import itself.
            pytest.param(_CLEAN_PENDING, [], AttemptKind.ORDINARY, (PauseReason.NONE, False, 0), id="clean-pending"),
            pytest.param(
                _CLEAN_PENDING, [], AttemptKind.FINAL, (PauseReason.ISSUED, True, 1), id="clean-pending-final"
            ),
            # Sonarr imports torrents one at a time. While its own pass runs, this download may be next in line,
            # so the clock pauses and even the final attempt posts nothing past it.
            pytest.param(
                _CLEAN_PENDING,
                [started_process_pass()],
                AttemptKind.ORDINARY,
                (PauseReason.BUSY, False, 0),
                id="clean-pending-behind-a-running-pass",
            ),
            pytest.param(
                _CLEAN_PENDING,
                [started_process_pass()],
                AttemptKind.FINAL,
                (PauseReason.BUSY, False, 0),
                id="clean-pending-behind-a-running-pass-final",
            ),
            pytest.param(
                [], [started_process_pass()], AttemptKind.FINAL, (PauseReason.BUSY, False, 0), id="untracked-final"
            ),
            # A warning on the importPending row is an import Sonarr tried and won't reliably retry.
            pytest.param(
                [queue_record("ABC123", "importPending", status="warning")],
                [],
                AttemptKind.ORDINARY,
                (PauseReason.ISSUED, True, 1),
                id="pending-with-a-warning",
            ),
            # A ManualImport of this download is already running, as after a restart: no second one is stacked on
            # it, and the wait reads as an import in flight. A finished one holds nothing back.
            pytest.param(
                [],
                [inflight_manual_import("ABC123")],
                AttemptKind.FINAL,
                (PauseReason.IMPORT, True, 0),
                id="its-own-import-running-final",
            ),
            pytest.param(
                [],
                [inflight_manual_import("abc123", status="completed")],
                AttemptKind.ORDINARY,
                (PauseReason.ISSUED, True, 1),
                id="its-own-import-completed",
            ),
            pytest.param(
                [],
                [inflight_manual_import("OTHERHASH", status="queued")],
                AttemptKind.ORDINARY,
                (PauseReason.BUSY, False, 0),
                id="another-downloads-import-queued",
            ),
        ],
    )
    def test_the_queue_state_and_running_commands_decide_between_a_wait_and_an_auto_import(
        self,
        queue: list[QueueRecord],
        commands: list[CommandResource],
        attempt_kind: AttemptKind,
        outcome: tuple[PauseReason, bool, int],
    ) -> None:
        # `outcome` is the pause the attempt asks for, whether an import command for this download is in flight,
        # and how many imports were posted.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")], queue=queue, commands=commands
        )

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert (result.imported_count, result.file_map_episode_count) == (0, 1)
        assert (result.pause_reason, result.command_issued, len(sonarr.execute_calls)) == outcome

    def test_a_clean_import_pending_is_imported_at_once_when_sonarr_imports_no_completed_download(self) -> None:
        # With completed download handling off, Sonarr leaves a finished download at importPending for good, so a
        # wait on it would only run out the deadline.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=_CLEAN_PENDING,
            commands=[],
            completed_download_handling=False,
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (result.pause_reason, result.command_issued) == (PauseReason.ISSUED, True)
        assert posted_imports(sonarr) == [[("/d/Show - 01 [1080p].mkv", [101])]]

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_unreachable_retries_instead_of_an_auto_import(self, attempt_kind: AttemptKind) -> None:
        # A failed queue read must never read as "not tracked" (an auto import would
        # race an import Sonarr may be running). Waiting with the clock running
        # fails closed on BOTH attempt kinds: the poll loop retries, and a queue that stays
        # unreachable burns the import deadline so the pending import finishes still not imported.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
        )
        sonarr.queue_return = None

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert result.paused is False
        assert sonarr.execute_calls == []

    def test_auto_import_post_is_a_paused_wait(self) -> None:
        # The accepted POST is pearlarr's copy in flight (ISSUED, paused once by the
        # wait pass), so a final auto import is not finished on the same poll. The
        # next poll finds the command started under its issued id (IMPORT).
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cmd_id=42,
        )

        issued = strat.import_completed(pending, "/d", AttemptKind.FINAL)
        sonarr.commands_return = [
            CommandResource.model_validate(
                {
                    "id": 42,
                    "name": "ManualImport",
                    "status": "started",
                    "body": {"files": [{"path": "/d/Show - 01 [1080p].mkv"}]},
                },
            ),
        ]
        in_flight = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert (issued.command_issued, issued.pause_reason) == (True, PauseReason.ISSUED)
        assert (in_flight.command_issued, in_flight.pause_reason) == (True, PauseReason.IMPORT)
        assert len(sonarr.execute_calls) == 1


class TestImportCompletedRescan:
    """A rescan counts once Sonarr completes it, and is asked for again when it never does or a restart orphans it.

    The throttle between two pending imports' rescans runs from the end of the last one, a rejected one included.
    """

    @pytest.mark.parametrize(
        ("command_status", "command_status_script", "refresh_calls"),
        [
            pytest.param(
                None,
                [CommandResource(status="started"), CommandResource(status="completed")],
                1,
                id="slow-completes",
            ),
            pytest.param(CommandResource(status="started"), None, 2, id="never-completes"),
        ],
    )
    def test_rescan_marker_lands_only_on_a_completed_rescan(
        self,
        command_status: CommandResource | None,
        command_status_script: list[CommandResource] | None,
        refresh_calls: int,
    ) -> None:
        clock = FakeClock()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            command_status=command_status,
            command_status_script=command_status_script,
            clock=clock,
        )
        pending = pending_import(infohash="abc123")

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == refresh_calls

    def test_rescan_asks_again_when_a_restart_orphans_it(self) -> None:
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            command_status_script=[CommandResource(status="orphaned")],
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 2

    def test_rescan_throttle_measures_from_the_rescan_end(self) -> None:
        sonarr = FakeSonarrClient(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            command_status_script=[CommandResource(status="started")] * 3,
            execute_command_id=42,
        )
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(),
            cache_store=FakeCacheStore(),
            clock=FakeClock(step=10),
        )

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="def456"), "/e", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    def test_rescan_throttle_holds_after_a_rejected_post(self) -> None:
        # The stamp lands on every exit, a rejected rescan included, so the
        # next pending import in the wait pass never retries it inside the window.
        sonarr = FakeSonarrClient(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            execute_command_id=42,
            refresh_count=None,
        )
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(),
            cache_store=FakeCacheStore(),
            clock=FakeClock(step=10),
        )

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="def456"), "/e", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1


class TestCloseTracked:
    """close_tracked dismisses the download's remaining queue record through any one row's id."""

    def test_dismisses_by_the_first_usable_row_id(self) -> None:
        # Sonarr closes the whole tracked download off any one queue id, so the first row with a usable id is
        # deleted, exactly once.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=queue_id) for queue_id in (0, 11, 12)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]

    def test_other_downloads_rows_are_not_touched(self) -> None:
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("FFFF", "importPending", queue_id=9)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_unknown_series_rows_are_left_for_sonarr(self) -> None:
        # Dismissing an unknown-series queue record 500s in Sonarr without recording
        # the ignore; the row drops once the torrent leaves the watched category.
        # SKIPPED, never FAILED: an unknown-series queue record can't strand the pending import.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", status="warning", queue_id=11, series_id=0)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_queue_unreachable_fails_the_effect_without_a_guessed_delete(self) -> None:
        # A failed read already retried inside the transport: FAILED (kept for a
        # later run), never an in-run RETRY and never a guessed delete.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.queue_return = None

        assert strat.remove_from_queue(pending) is EffectStatus.FAILED
        assert sonarr.queue_delete_calls == []

    def test_successful_close_notes_the_removal(self) -> None:
        recording = install_recording_hub()
        strat, _ = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )

        status = strat.remove_from_queue(pending_import(infohash="abc123", title="Show"))

        assert status is EffectStatus.DONE
        assert diagnostic_messages(recording, Severity.INFO) == [
            "Removed the imported download Show · SubGroup from Sonarr's queue"
        ]

    def test_failed_delete_asks_for_a_retry_and_stays_quiet(self) -> None:
        # The client's fail-open warning already fired; no success note on top.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.FAILED

        status = strat.remove_from_queue(pending_import(infohash="abc123"))

        assert status is EffectStatus.RETRY
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording, Severity.INFO) == []

    def test_gone_queue_record_is_quiet_success(self) -> None:
        # Sonarr already dismissed the queue record (its refresh cleared it first): DONE
        # with zero diagnostics, not the coalesced could-not-remove warning.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.GONE

        status = strat.remove_from_queue(pending_import(infohash="abc123"))

        assert status is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording) == []
