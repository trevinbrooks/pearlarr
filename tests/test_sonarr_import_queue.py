# pyright: strict
"""Tests for what Sonarr's queue, its commands and its episode lists make one import attempt do.

An attempt waits for Sonarr, asks it to rescan its downloads, or posts the import itself. It runs over a scripted
Sonarr and is judged by its result and by what it posted.
"""

import pytest

from pearlarr.arr_models import CommandResource, QueueRecord, RemotePathMapping, SonarrEpisode
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import NO_PROGRESS, AttemptKind, PauseReason

from .builders import (
    FakeCacheStore,
    make_config,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
    queue_record,
    sonarr_ep,
)
from .fakes import FakeClock, FakeSonarrClient, HookedSonarrClient
from .sonarr_import_builders import (
    RESCAN_AFTER_ADD,
    RESCAN_BEFORE_ADD,
    closed_history,
    inflight_manual_import,
    make_sonarr_for_import,
    posted_imports,
    started_process_pass,
)

_IMPORTING = [queue_record("ABC123", "importing")]


_CLEAN_PENDING = [queue_record("ABC123", "importPending", status="ok")]


def _strategy(queue: list[QueueRecord], clock: FakeClock | None = None) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A strategy and its hooked fake under `queue`, the by-id fetch serving the pending import's one file."""

    return make_sonarr_for_import(candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")], queue=queue, clock=clock)


def _blocked_row() -> QueueRecord:
    """The pending import's queue record in a state Sonarr won't progress."""

    return queue_record("ABC123", "importBlocked", status="warning")


def _tv_mapping() -> RemotePathMapping:
    """The incident-shaped mapping: qBittorrent's `/d` is Sonarr's `/remote/tv`."""

    return RemotePathMapping.model_validate({"host": "seedbox", "remotePath": "/d/", "localPath": "/remote/tv/"})


def _process_pass(command_id: int, status: str) -> CommandResource:
    """A ProcessMonitoredDownloads command with this id and status."""

    return CommandResource.model_validate({"id": command_id, "name": "ProcessMonitoredDownloads", "status": status})


def _episode_with_torrent_file() -> list[SonarrEpisode]:
    """The pending import's one episode with this torrent's file."""

    return [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]


def _strategy_whose_fetch_meets_a_pass(clock: FakeClock) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A tracked pending import's strategy: the by-id fetch starts a ProcessMonitoredDownloads, later ones fail."""

    strat, sonarr = _strategy([_blocked_row()], clock)

    def start_a_pass() -> None:
        sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]
        sonarr.candidates_return = None

    sonarr.on_fetch = start_a_pass
    return strat, sonarr


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


class TestRescanOnce:
    """A lagging view gets one rescan per pending import per run, and none once Sonarr provably looked."""

    def test_a_pass_the_rescan_starts_blocks_the_auto_import(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_AFTER_ADD, started_process_pass()]

        sonarr.on_rescan = start_a_pass
        pending = pending_import()

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        clock.tick(30)
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (first.pause_reason, second.pause_reason) == (PauseReason.BUSY, PauseReason.BUSY)
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 1

    def test_proof_is_read_before_the_queue(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = [RESCAN_BEFORE_ADD]

        def rescan_meanwhile() -> None:
            sonarr.commands_return = [RESCAN_AFTER_ADD]

        sonarr.on_queue = rescan_meanwhile

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    @pytest.mark.parametrize(
        ("commands", "refresh_calls"),
        [
            pytest.param((RESCAN_AFTER_ADD,), 0, id="rescanned-after-the-add"),
            pytest.param((RESCAN_BEFORE_ADD,), 1, id="not-since-the-add"),
        ],
    )
    def test_pending_import_with_no_row_skips_the_rescan_when_sonarr_already_looked(
        self, commands: tuple[CommandResource, ...], refresh_calls: int
    ) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = list(commands)

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == refresh_calls

    def test_restart_signature_resets_the_rescan_markers(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        first = sonarr.refresh_calls
        sonarr.commands_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        restarted = sonarr.refresh_calls
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (first, restarted, sonarr.refresh_calls) == (1, 2, 2)

    @pytest.mark.parametrize(
        ("state", "status"),
        [("importBlocked", "warning"), ("importPending", "ok"), ("importing", "ok")],
        ids=["blocked", "pending", "importing"],
    )
    def test_tracked_row_gets_no_rescan(self, state: str, status: str) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", state, status=status)])

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 0

    def test_in_motion_row_rescans_then_reads_the_queue_again(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])

        def block_the_import() -> None:
            sonarr.queue_return = [_blocked_row()]

        sonarr.on_rescan = block_the_import

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1
        assert len(sonarr.execute_calls) == 1

    def test_throttled_pending_import_keeps_its_turn(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)
        throttled = pending_import(infohash="def456")

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(throttled, "/e", AttemptKind.ORDINARY)
        inside_the_interval = sonarr.refresh_calls
        clock.tick(30)
        strat.import_completed(throttled, "/e", AttemptKind.ORDINARY)

        assert (inside_the_interval, sonarr.refresh_calls) == (1, 2)


class TestAutoImportGuard:
    """The command list read before the fetch, right after it, and right before the POST."""

    def test_pass_started_during_the_fetch_blocks_the_auto_import(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, _process_pass(9, "started")]

        sonarr.on_fetch = start_a_pass

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    def test_pass_started_during_an_empty_fetch_reads_busy(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=[], queue=[_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, _process_pass(9, "started")]

        sonarr.on_fetch = start_a_pass

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY

    def test_pass_completed_during_the_fetch_does_not_block_the_post(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def run_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_fetch = run_a_pass

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1

    def test_pass_run_during_the_episode_re_read_blocks_the_auto_import(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        reads = 0

        def run_a_pass_on_the_re_read() -> None:
            nonlocal reads
            reads += 1
            if reads == 2:
                sonarr.commands_return = [RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_episodes = run_a_pass_on_the_re_read

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    def test_pass_started_before_the_post_blocks_the_auto_import(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_languages = start_a_pass

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    def test_pass_completed_before_the_post_blocks_the_auto_import(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([_blocked_row()], clock)

        def run_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_languages = run_a_pass
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        posted_while_blocked = len(sonarr.execute_calls)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert blocked.pause_reason is PauseReason.BUSY
        assert (posted_while_blocked, len(sonarr.execute_calls)) == (0, 1)

    def test_import_of_the_download_started_before_the_post_reads_as_an_auto_import(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_an_import() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, inflight_manual_import("ABC123", command_id=77)]

        sonarr.on_languages = start_an_import

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.IMPORT
        assert result.command_issued is True
        assert sonarr.execute_calls == []

    def test_restart_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def restart() -> None:
            sonarr.commands_return = []

        sonarr.on_languages = restart

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []

    def test_folder_import_in_flight_matches_its_translated_path_on_the_first_poll(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = closed_history()
        sonarr.folder_candidates_return = [manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")]
        sonarr.path_mappings_return = [_tv_mapping()]
        folder_import = CommandResource.model_validate(
            {
                "id": 7000,
                "name": "ManualImport",
                "status": "started",
                "body": {"files": [{"path": "/remote/tv/Show/Show - 01 [1080p].mkv", "episodeIds": [202]}]},
            },
        )

        def start_a_folder_import() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, folder_import]

        sonarr.on_fetch = start_a_folder_import

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.IMPORT
        assert result.command_issued is False
        assert sonarr.execute_calls == []

    def test_unreadable_command_list_waits(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = None

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 0

    def test_unreadable_command_list_after_the_fetch_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.commands_return = None

        def recover() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD]

        sonarr.on_fetch = fail_the_next_read
        sonarr.on_languages = recover

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []

    def test_unreadable_command_list_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.commands_return = None

        sonarr.on_languages = fail_the_next_read

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []


class TestEpisodeReads:
    """Episode files read this poll decide what is imported, never the last known list."""

    def test_files_a_pass_imports_during_the_fetch_read_as_imported(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def import_torrent_file() -> None:
            sonarr.episodes_return = _episode_with_torrent_file()
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_fetch = import_torrent_file

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert result.imported_count == 1
        assert sonarr.execute_calls == []

    def test_unreadable_episode_re_read_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.episodes_return = None

        sonarr.on_episodes = fail_the_next_read

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []

    def test_stale_cache_never_reads_as_imported(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = _episode_with_torrent_file()
        pending = pending_import()

        imported = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = None
        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert imported.all_done is True
        assert result.all_done is False
        assert sonarr.execute_calls == []

    def test_failed_read_keeps_the_last_read_counts(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])
        sonarr.episodes_return = [*_episode_with_torrent_file(), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101], "Show - 02 [1080p].mkv": [102]})

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = None
        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (result.imported_count, result.file_map_episode_count) == (1, 2)

    def test_import_progress_is_indeterminate_on_a_failed_read(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = None

        assert strat.import_progress(pending_import()) == NO_PROGRESS


class TestKeptCandidates:
    """A by-id fetch from a blocked poll serves the pending import's next poll, once."""

    def test_blocked_poll_keeps_its_candidates_for_the_next_poll(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_fetch_meets_a_pass(clock)
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert blocked.pause_reason is PauseReason.BUSY
        assert len(sonarr.execute_calls) == 1

    def test_empty_by_id_answer_is_never_kept(self) -> None:
        clock = FakeClock()
        strat, sonarr = make_sonarr_for_import(candidates=[], queue=[_blocked_row()], clock=clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_fetch = start_a_pass
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        sonarr.candidates_return = [manual_candidate("/d/Show - 01 [1080p].mkv")]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1

    def test_a_fresh_fetch_discards_the_kept_candidates(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([_blocked_row()], clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_fetch = start_a_pass
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        sonarr.queue_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.queue_return = [_blocked_row()]
        clock.tick(25)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert len(sonarr.candidate_calls) == 3

    def test_kept_candidates_are_not_used_without_a_queue_row(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_fetch_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.queue_return = []
        sonarr.commands_return = [RESCAN_AFTER_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_kept_candidates_expire_after_two_poll_intervals(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_fetch_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        clock.tick(90)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_kept_candidates_are_not_used_under_another_content_path(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_fetch_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/elsewhere", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_kept_candidates_serve_one_poll_only(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_fetch_meets_a_pass(clock)
        pending = pending_import()

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        sonarr.on_queue = start_a_pass
        clock.tick(30)
        taken = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.on_queue = None
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert taken.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []
