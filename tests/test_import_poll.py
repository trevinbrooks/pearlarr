# pyright: strict
# pyright: reportPrivateUsage=false
# These reuse the seam module's private builders. Strict re-flags that and the repo disables it for tests.
"""The import poll: one rescan per record, the command reads around the step-in, fresh episode reads, the kept scan."""

import pytest

from pearlarr.manual_import import NO_PROGRESS, AttemptKind, Deferral
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import CommandResource, QueueRecord, SonarrEpisode

from .builders import manual_candidate, pending_import, queue_record, sonarr_ep
from .fakes import FakeClock, HookedSonarrClient
from .test_strategy_seam import (
    _RESCAN_AFTER_ADD,
    _RESCAN_BEFORE_ADD,
    _dead_history,
    _inflight_manual_import,
    _make_sonarr_for_import,
    _started_process_pass,
    _tv_mapping,
)


def _strategy(queue: list[QueueRecord], clock: FakeClock | None = None) -> tuple[SonarrSync, HookedSonarrClient]:
    """A strategy and its hooked fake under `queue`, the by-id scan serving the record's one file."""

    return _make_sonarr_for_import(candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")], queue=queue, clock=clock)


def _blocked_row() -> QueueRecord:
    """The record's queue row in a state Sonarr won't progress."""

    return queue_record("ABC123", "importBlocked", status="warning")


def _process_pass(command_id: int, status: str) -> CommandResource:
    """A ProcessMonitoredDownloads command with this id and status."""

    return CommandResource.model_validate({"id": command_id, "name": "ProcessMonitoredDownloads", "status": status})


def _our_file_placed() -> list[SonarrEpisode]:
    """The record's one target episode holding its own file."""

    return [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]


def _strategy_whose_scan_meets_a_pass(clock: FakeClock) -> tuple[SonarrSync, HookedSonarrClient]:
    """A tracked record's strategy whose by-id scan starts a pass and leaves every later by-id scan failing."""

    strat, sonarr = _strategy([_blocked_row()], clock)

    def start_a_pass() -> None:
        sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]
        sonarr.candidates_return = None

    sonarr.on_scan = start_a_pass
    return strat, sonarr


class TestRescanOnce:
    """A lagging view gets one rescan per record per run, and none once Sonarr provably looked."""

    def test_pass_our_rescan_starts_defers_the_step_in(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_AFTER_ADD, _started_process_pass()]

        sonarr.on_rescan = start_a_pass
        pending = pending_import()

        first = strat.import_completed(pending, "/d", AttemptKind.POLL)
        clock.tick(30)
        second = strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert (first.deferral, second.deferral) == (Deferral.BUSY, Deferral.BUSY)
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 1

    def test_proof_is_read_before_the_queue(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]

        def rescan_meanwhile() -> None:
            sonarr.commands_return = [_RESCAN_AFTER_ADD]

        sonarr.on_queue = rescan_meanwhile

        strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert sonarr.refresh_calls == 1

    @pytest.mark.parametrize(
        ("commands", "refresh_calls"),
        [
            pytest.param((_RESCAN_AFTER_ADD,), 0, id="rescanned-after-the-add"),
            pytest.param((_RESCAN_BEFORE_ADD,), 1, id="not-since-the-add"),
        ],
    )
    def test_no_row_record_skips_the_rescan_when_sonarr_already_looked(
        self, commands: tuple[CommandResource, ...], refresh_calls: int
    ) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = list(commands)

        strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert sonarr.refresh_calls == refresh_calls

    def test_restart_signature_resets_the_rescan_markers(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        first = sonarr.refresh_calls
        sonarr.commands_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)
        restarted = sonarr.refresh_calls
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert (first, restarted, sonarr.refresh_calls) == (1, 2, 2)

    @pytest.mark.parametrize(
        ("state", "status"),
        [("importBlocked", "warning"), ("importPending", "ok"), ("importing", "ok")],
        ids=["blocked", "pending", "importing"],
    )
    def test_tracked_row_gets_no_rescan(self, state: str, status: str) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", state, status=status)])

        strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert sonarr.refresh_calls == 0

    def test_in_motion_row_rescans_then_reads_the_queue_again(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])

        def block_the_import() -> None:
            sonarr.queue_return = [_blocked_row()]

        sonarr.on_rescan = block_the_import

        strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert sonarr.refresh_calls == 1
        assert len(sonarr.execute_calls) == 1

    def test_throttled_record_keeps_its_turn(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock)
        throttled = pending_import(infohash="def456")

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.POLL)
        strat.import_completed(throttled, "/e", AttemptKind.POLL)
        inside_the_interval = sonarr.refresh_calls
        clock.tick(30)
        strat.import_completed(throttled, "/e", AttemptKind.POLL)

        assert (inside_the_interval, sonarr.refresh_calls) == (1, 2)


class TestStepInGuard:
    """The command list read before the scan, right after it, and right before the POST."""

    def test_pass_started_during_the_scan_defers_the_step_in(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _process_pass(9, "started")]

        sonarr.on_scan = start_a_pass

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.BUSY
        assert sonarr.execute_calls == []

    def test_pass_started_during_an_empty_scan_reads_busy(self) -> None:
        strat, sonarr = _make_sonarr_for_import(candidates=[], queue=[_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _process_pass(9, "started")]

        sonarr.on_scan = start_a_pass

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.BUSY

    def test_pass_completed_during_the_scan_does_not_block_the_post(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def run_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_scan = run_a_pass

        strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert len(sonarr.execute_calls) == 1

    def test_pass_run_during_the_episode_re_read_defers_the_step_in(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        reads = 0

        def run_a_pass_on_the_re_read() -> None:
            nonlocal reads
            reads += 1
            if reads == 2:
                sonarr.commands_return = [_RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_episodes = run_a_pass_on_the_re_read

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.BUSY
        assert sonarr.execute_calls == []

    def test_pass_started_before_the_post_defers_the_step_in(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        sonarr.on_languages = start_a_pass

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.BUSY
        assert sonarr.execute_calls == []

    def test_pass_completed_before_the_post_defers_the_step_in(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([_blocked_row()], clock)

        def run_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _process_pass(9, "completed")]

        sonarr.on_languages = run_a_pass
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d", AttemptKind.POLL)
        posted_while_blocked = len(sonarr.execute_calls)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert blocked.deferral is Deferral.BUSY
        assert (posted_while_blocked, len(sonarr.execute_calls)) == (0, 1)

    def test_import_of_the_download_started_before_the_post_reads_as_ours(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def start_an_import() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _inflight_manual_import("ABC123", command_id=77)]

        sonarr.on_languages = start_an_import

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.IMPORT
        assert probe.command_issued is True
        assert sonarr.execute_calls == []

    def test_restart_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def restart() -> None:
            sonarr.commands_return = []

        sonarr.on_languages = restart

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.NONE
        assert sonarr.execute_calls == []

    def test_folder_import_in_flight_matches_its_translated_path_on_the_first_poll(self) -> None:
        strat, sonarr = _make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = _dead_history()
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
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, folder_import]

        sonarr.on_scan = start_a_folder_import

        probe = strat.import_completed(pending_import(), "/d/Show", AttemptKind.POLL)

        assert probe.deferral is Deferral.IMPORT
        assert probe.command_issued is False
        assert sonarr.execute_calls == []

    def test_unreadable_command_list_waits(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = None

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.NONE
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 0

    def test_unreadable_command_list_after_the_scan_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.commands_return = None

        def recover() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD]

        sonarr.on_scan = fail_the_next_read
        sonarr.on_languages = recover

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.NONE
        assert sonarr.execute_calls == []

    def test_unreadable_command_list_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.commands_return = None

        sonarr.on_languages = fail_the_next_read

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.deferral is Deferral.NONE
        assert sonarr.execute_calls == []


class TestEpisodeReads:
    """Episode files read this poll decide what is imported, never the last known list."""

    def test_files_a_pass_imports_during_the_scan_read_as_imported(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def import_our_file() -> None:
            sonarr.episodes_return = _our_file_placed()
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        sonarr.on_scan = import_our_file

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.files_present is True
        assert probe.imported_count == 1
        assert sonarr.execute_calls == []

    def test_unreadable_re_seed_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.episodes_return = None

        sonarr.on_episodes = fail_the_next_read

        probe = strat.import_completed(pending_import(), "/d", AttemptKind.POLL)

        assert probe.files_present is False
        assert probe.deferral is Deferral.NONE
        assert sonarr.execute_calls == []

    def test_stale_cache_never_reads_as_imported(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = _our_file_placed()
        pending = pending_import()

        imported = strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.episodes_return = None
        probe = strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert imported.files_present is True
        assert probe.files_present is False
        assert sonarr.execute_calls == []

    def test_failed_read_keeps_the_last_read_counts(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])
        sonarr.episodes_return = [*_our_file_placed(), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101], "Show - 02 [1080p].mkv": [102]})

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.episodes_return = None
        probe = strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert (probe.imported_count, probe.target_count) == (1, 2)

    def test_import_progress_is_indeterminate_on_a_failed_read(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = None

        assert strat.import_progress(pending_import()) == NO_PROGRESS


class TestKeptScan:
    """A by-id scan from a blocked poll serves the record's next poll, once."""

    def test_blocked_poll_keeps_its_scan_for_the_next_poll(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_scan_meets_a_pass(clock)
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert blocked.deferral is Deferral.BUSY
        assert len(sonarr.execute_calls) == 1

    def test_empty_by_id_answer_is_never_kept(self) -> None:
        clock = FakeClock()
        strat, sonarr = _make_sonarr_for_import(candidates=[], queue=[_blocked_row()], clock=clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        sonarr.on_scan = start_a_pass
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.on_scan = None
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        sonarr.candidates_return = [manual_candidate("/d/Show - 01 [1080p].mkv")]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert len(sonarr.execute_calls) == 1

    def test_a_fresh_scan_discards_the_kept_scan(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([_blocked_row()], clock)

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        sonarr.on_scan = start_a_pass
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.on_scan = None
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        sonarr.queue_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.queue_return = [_blocked_row()]
        clock.tick(25)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert len(sonarr.candidate_calls) == 3

    def test_kept_scan_is_not_used_without_a_queue_row(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_scan_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.queue_return = []
        sonarr.commands_return = [_RESCAN_AFTER_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert sonarr.execute_calls == []

    def test_kept_scan_expires_after_two_poll_intervals(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_scan_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        clock.tick(90)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert sonarr.execute_calls == []

    def test_kept_scan_is_not_used_under_another_content_path(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_scan_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/elsewhere", AttemptKind.POLL)

        assert sonarr.execute_calls == []

    def test_kept_scan_serves_one_poll_only(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_scan_meets_a_pass(clock)
        pending = pending_import()

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        sonarr.on_queue = start_a_pass
        clock.tick(30)
        taken = strat.import_completed(pending, "/d", AttemptKind.POLL)
        sonarr.on_queue = None
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.POLL)

        assert taken.deferral is Deferral.BUSY
        assert sonarr.execute_calls == []
