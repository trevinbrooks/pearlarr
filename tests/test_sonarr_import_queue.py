# pyright: strict
"""Tests for what Sonarr's queue, its commands and its episode lists make one import attempt do.

An attempt waits for Sonarr, asks it to rescan its downloads, or posts the import itself. It runs over a scripted
Sonarr and is judged by its result and by what it posted.

A pass is Sonarr's own ProcessMonitoredDownloads command, the run in which it imports its finished downloads. A
hook on the language read changes Sonarr just before the post: the languages are the last read the fake can hook
before the command list is read again, and a run reads them once, so only its first post meets the hook. Pending
imports are polled every 30 seconds by default, so a 30 second tick is the next poll.
"""

from collections.abc import Callable

import pytest

from pearlarr.arr_models import CommandResource, QueueRecord, RemotePathMapping, SonarrEpisode
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import NO_PROGRESS, AttemptKind, PauseReason

from .builders import manual_candidate, pending_import, queue_record, sonarr_ep
from .fakes import NEUTRAL_RESCAN, FakeClock, HookedSonarrClient
from .sonarr_import_builders import (
    RESCAN_AFTER_ADD,
    RESCAN_BEFORE_ADD,
    closed_history,
    inflight_manual_import,
    make_sonarr_for_import,
    posted_imports,
    started_process_pass,
)

type _Imports = list[list[tuple[str, list[int]]]]
"""The `(path, episode ids)` of every file in each import posted, as `posted_imports` reads them."""

_FILE = "/d/Show - 01 [1080p].mkv"
_POSTED: _Imports = [[(_FILE, [101])]]
"""The import an attempt posts for the pending import: its one file, on episode 101."""

_IMPORTING = [queue_record("ABC123", "importing")]
_CLEAN_PENDING = [queue_record("ABC123", "importPending", status="ok")]


def _strategy(
    queue: list[QueueRecord], *, commands: list[CommandResource] | None = None, clock: FakeClock | None = None
) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A strategy over a Sonarr that serves the pending import's one file by its download id and lists `commands`."""

    return make_sonarr_for_import(candidates=[manual_candidate(_FILE)], queue=queue, commands=commands, clock=clock)


def _blocked_row() -> QueueRecord:
    """The download's queue record in a state Sonarr won't progress, so the attempt posts the import itself."""

    return queue_record("ABC123", "importBlocked", status="warning")


def _completed_pass() -> CommandResource:
    """Sonarr's own ProcessMonitoredDownloads command, finished."""

    return CommandResource(name="ProcessMonitoredDownloads", status="completed")


def _then_lists(sonarr: HookedSonarrClient, *commands: CommandResource) -> Callable[[], None]:
    """A hook after which Sonarr lists `commands`, beside a rescan it completed long before the add."""

    def change() -> None:
        sonarr.commands_return = [NEUTRAL_RESCAN, *commands]

    return change


def _then_command_list_fails(sonarr: HookedSonarrClient) -> Callable[[], None]:
    """A hook after which Sonarr's command list can't be read."""

    def fail() -> None:
        sonarr.commands_return = None

    return fail


def _imported_episode() -> list[SonarrEpisode]:
    """Episode 101 holding this torrent's file."""

    return [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]


class TestQueueRecord:
    """The download's queue record decides whether an attempt leaves the import to Sonarr or posts it itself."""

    @pytest.mark.parametrize(
        ("queue", "attempt_kind", "pause_reason", "posted"),
        [
            # Sonarr is copying this download. That is Sonarr's wait, so the clock pauses, and nothing is posted
            # over it even at the deadline.
            pytest.param(_IMPORTING, AttemptKind.ORDINARY, PauseReason.IMPORT, [], id="importing"),
            pytest.param(_IMPORTING, AttemptKind.FINAL, PauseReason.IMPORT, [], id="importing-final"),
            # Sonarr's view lags qBittorrent's "complete": the pending import's own wait, on a running clock.
            pytest.param(
                [queue_record("ABC123", "downloading")], AttemptKind.ORDINARY, PauseReason.NONE, [], id="downloading"
            ),
            # A clean importPending is left to Sonarr until the final attempt, which posts the import itself.
            pytest.param(_CLEAN_PENDING, AttemptKind.ORDINARY, PauseReason.NONE, [], id="clean-pending"),
            pytest.param(_CLEAN_PENDING, AttemptKind.FINAL, PauseReason.ISSUED, _POSTED, id="clean-pending-final"),
            # A warning on the importPending row is an import Sonarr tried and won't reliably retry.
            pytest.param(
                [queue_record("ABC123", "importPending", status="warning")],
                AttemptKind.ORDINARY,
                PauseReason.ISSUED,
                _POSTED,
                id="pending-with-a-warning",
            ),
        ],
    )
    def test_the_record_decides_between_a_wait_and_a_post(
        self,
        queue: list[QueueRecord],
        attempt_kind: AttemptKind,
        pause_reason: PauseReason,
        posted: _Imports,
    ) -> None:
        strat, sonarr = _strategy(queue)

        result = strat.import_completed(pending_import(), "/d", attempt_kind)

        assert (result.all_done, result.imported_count, result.file_map_episode_count) == (False, 0, 1)
        assert result.pause_reason is pause_reason
        assert posted_imports(sonarr) == posted
        # A wait never claims the import as pearlarr's own. Only a post does.
        assert result.command_issued is bool(posted)

    def test_a_clean_import_pending_is_posted_at_once_when_sonarr_imports_no_completed_download(self) -> None:
        # With completed download handling off, Sonarr leaves a finished download at importPending for good, so a
        # wait on it would only run out the deadline.
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(_FILE)], queue=_CLEAN_PENDING, completed_download_handling=False
        )

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.ISSUED
        assert posted_imports(sonarr) == _POSTED


class TestRescan:
    """An attempt asks Sonarr to rescan its downloads while Sonarr's view trails the download.

    That is while there is no queue record, or one still downloading. A pending import gets one completed rescan a
    run, again after Sonarr restarts, and none once Sonarr lists a rescan from after the add. Between pending
    imports, a rescan waits a poll interval from the end of the last one.
    """

    @pytest.mark.parametrize(
        ("state", "status"),
        [("importBlocked", "warning"), ("importPending", "ok"), ("importing", "ok")],
        ids=["blocked", "pending", "importing"],
    )
    def test_a_record_past_downloading_gets_no_rescan(self, state: str, status: str) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", state, status=status)])

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 0

    def test_a_record_still_downloading_is_rescanned_and_the_attempt_goes_by_the_queue_after_it(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])

        def block_the_import() -> None:
            sonarr.queue_return = [_blocked_row()]

        sonarr.on_rescan = block_the_import

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1
        assert posted_imports(sonarr) == _POSTED

    @pytest.mark.parametrize(
        ("rescan", "refresh_calls"),
        [
            pytest.param(RESCAN_AFTER_ADD, 0, id="rescanned-after-the-add"),
            pytest.param(RESCAN_BEFORE_ADD, 1, id="not-since-the-add"),
        ],
    )
    def test_a_download_with_no_record_is_rescanned_unless_sonarr_rescanned_since_the_add(
        self, rescan: CommandResource, refresh_calls: int
    ) -> None:
        strat, sonarr = _strategy([], commands=[rescan])

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == refresh_calls

    def test_a_rescan_completed_after_the_queue_read_is_no_proof(self) -> None:
        # The queue is read after the command list, so a rescan the list shows completed is one the queue already
        # reflects. This one completes after the queue read, so the attempt still asks for its own.
        strat, sonarr = _strategy([])
        sonarr.on_queue = _then_lists(sonarr, RESCAN_AFTER_ADD)

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    def test_a_rescan_that_completes_is_not_asked_for_again(self) -> None:
        clock = FakeClock()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(_FILE)],
            command_status_script=[CommandResource(status="started"), CommandResource(status="completed")],
            clock=clock,
        )
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    @pytest.mark.parametrize("status", ["failed", "started"], ids=["fails", "never-completes"])
    def test_a_rescan_that_does_not_complete_is_asked_for_again_on_the_next_poll(self, status: str) -> None:
        clock = FakeClock()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(_FILE)], command_status=CommandResource(status=status), clock=clock
        )
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 2

    def test_a_rescan_a_restart_cuts_short_is_asked_for_again_at_once(self) -> None:
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(_FILE)], command_status_script=[CommandResource(status="orphaned")]
        )

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 2

    def test_after_sonarr_restarts_a_pending_import_is_rescanned_once_more(self) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy([], clock=clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        before_the_restart = sonarr.refresh_calls
        # A restarted Sonarr lists no completed rescan at all.
        sonarr.commands_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (before_the_restart, sonarr.refresh_calls) == (1, 2)

    def test_the_next_pending_imports_rescan_waits_a_poll_interval_from_the_end_of_the_last(self) -> None:
        # Sonarr answers "started" three times, and each wait for its next answer takes ten seconds here, so the
        # first rescan ends a full poll interval after it began.
        clock = FakeClock(step=10)
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(_FILE)],
            command_status_script=[CommandResource(status="started")] * 3,
            clock=clock,
        )
        held = pending_import(infohash="def456")

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)
        strat.import_completed(held, "/e", AttemptKind.ORDINARY)
        inside_the_interval = sonarr.refresh_calls
        clock.tick(30)
        strat.import_completed(held, "/e", AttemptKind.ORDINARY)

        assert (inside_the_interval, sonarr.refresh_calls) == (1, 2)

    def test_a_rescan_sonarr_rejects_still_holds_the_next_pending_imports_rescan(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.refresh_count = None

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="def456"), "/e", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1


class TestSonarrBusy:
    """Sonarr's own disk work holds the post, at the deadline too, with the clock paused.

    That is a pass or another download's import listed when the attempt starts, and any pass that starts or ends
    after the attempt reads the command list again past the fetch, since its episode read may then miss what the
    pass did. A pass that ended before that read holds nothing.
    """

    @pytest.mark.parametrize(
        ("command", "attempt_kind"),
        [
            pytest.param(started_process_pass(), AttemptKind.ORDINARY, id="a-pass"),
            pytest.param(started_process_pass(), AttemptKind.FINAL, id="a-pass-final"),
            pytest.param(
                inflight_manual_import("OTHERHASH", status="queued"), AttemptKind.ORDINARY, id="another-import"
            ),
        ],
    )
    def test_a_disk_command_listed_when_the_attempt_starts_holds_it(
        self, command: CommandResource, attempt_kind: AttemptKind
    ) -> None:
        # Sonarr imports one download at a time, so this one may be next in line. Its record is a clean importPending,
        # which an ordinary attempt would otherwise just wait on, so only the command list read at the start can hold
        # it as busy.
        strat, sonarr = _strategy(_CLEAN_PENDING, commands=[command])

        result = strat.import_completed(pending_import(), "/d", attempt_kind)

        assert (result.all_done, result.imported_count, result.file_map_episode_count) == (False, 0, 1)
        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    def test_a_pass_started_during_a_fetch_that_found_no_files_reads_as_busy(self) -> None:
        # Not as files missing: the pass may be importing them.
        strat, sonarr = make_sonarr_for_import(candidates=[], queue=[_blocked_row()])
        sonarr.on_fetch = _then_lists(sonarr, started_process_pass())

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY

    def test_a_pass_that_completed_during_the_fetch_does_not_hold_the_post(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        sonarr.on_fetch = _then_lists(sonarr, _completed_pass())

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == _POSTED

    def test_a_pass_that_ends_while_the_episodes_are_read_again_holds_the_post(self) -> None:
        # The episode read past the fetch was answered before the pass ended, so it may miss what the pass imported.
        strat, sonarr = _strategy([_blocked_row()])

        def end_a_pass_on_the_next_episode_read() -> None:
            sonarr.on_episodes = _then_lists(sonarr, _completed_pass())

        sonarr.on_fetch = end_a_pass_on_the_next_episode_read

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    @pytest.mark.parametrize("command", [started_process_pass(), _completed_pass()], ids=["running", "ended"])
    def test_a_pass_listed_just_before_the_post_holds_it(self, command: CommandResource) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        sonarr.on_languages = _then_lists(sonarr, command)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []


class TestImportInFlight:
    """An import of this download that Sonarr is running holds the post, with the clock paused.

    One that carries the download id, or whose command id pearlarr's post got back, reads as pearlarr's own.
    """

    def test_an_import_of_the_download_running_when_the_attempt_starts_reads_as_pearlarrs_own(self) -> None:
        # As after pearlarr restarts with its earlier post still running. Even the final attempt stacks no second
        # import on it.
        strat, sonarr = _strategy([], commands=[inflight_manual_import("ABC123")])

        result = strat.import_completed(pending_import(), "/d", AttemptKind.FINAL)

        assert (result.pause_reason, result.command_issued) == (PauseReason.IMPORT, True)
        assert sonarr.execute_calls == []

    def test_a_finished_import_of_the_download_holds_nothing_back(self) -> None:
        strat, sonarr = _strategy([], commands=[inflight_manual_import("ABC123", status="completed")])

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == _POSTED

    def test_an_import_of_the_download_started_just_before_the_post_reads_as_pearlarrs_own(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        sonarr.on_languages = _then_lists(sonarr, inflight_manual_import("ABC123"))

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert (result.pause_reason, result.command_issued) == (PauseReason.IMPORT, True)
        assert sonarr.execute_calls == []

    def test_an_import_naming_the_file_by_sonarrs_path_holds_the_post_from_the_first_poll(self) -> None:
        # Sonarr sees qBittorrent's /d as /remote/tv, and the import names the file by that path. It is on episode 202,
        # not the file's 101, so only the path can match it. The fetch reads the mapping, so the reads after it match
        # the import on this first poll. Pearlarr didn't post it, so it isn't read as pearlarr's own.
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = closed_history()
        sonarr.folder_candidates_return = [manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")]
        sonarr.path_mappings_return = [
            RemotePathMapping.model_validate({"host": "seedbox", "remotePath": "/d/", "localPath": "/remote/tv/"})
        ]
        folder_import = CommandResource.model_validate(
            {
                "name": "ManualImport",
                "status": "started",
                "body": {"files": [{"path": "/remote/tv/Show/Show - 01 [1080p].mkv", "episodeIds": [202]}]},
            },
        )
        sonarr.on_fetch = _then_lists(sonarr, folder_import)

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert (result.pause_reason, result.command_issued) == (PauseReason.IMPORT, False)
        assert sonarr.execute_calls == []

    def test_the_next_poll_reads_pearlarrs_folder_import_as_its_own_by_the_command_id(self) -> None:
        # Sonarr's history of the download is closed, so the post names the files by path with no download id. The
        # command id the post got back is what marks the running import as pearlarr's. The copy runs on after the
        # post, so the attempt that posts waits too, on a paused clock.
        strat, sonarr = make_sonarr_for_import(candidates=None, cmd_id=42)
        sonarr.history_page_return = closed_history()
        sonarr.folder_candidates_return = [manual_candidate(_FILE)]
        sonarr.path_mappings_return = []
        pending = pending_import()

        posting = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [
            NEUTRAL_RESCAN,
            CommandResource.model_validate(
                {"id": 42, "name": "ManualImport", "status": "started", "body": {"files": [{"path": _FILE}]}}
            ),
        ]
        in_flight = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (posting.pause_reason, posting.command_issued) == (PauseReason.ISSUED, True)
        assert (in_flight.pause_reason, in_flight.command_issued) == (PauseReason.IMPORT, True)
        assert posted_imports(sonarr) == _POSTED


class TestUnreadableSonarr:
    """A failed read of Sonarr's queue or command list waits for the next poll and posts nothing.

    So does a command list that shows Sonarr just restarted. None of these pauses the clock, so a Sonarr that stays
    unreadable runs out the deadline with the download not imported.
    """

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_an_unreadable_queue_waits(self, attempt_kind: AttemptKind) -> None:
        # Never read as no record: a post could race an import Sonarr may be running.
        strat, sonarr = _strategy([])
        sonarr.queue_return = None

        result = strat.import_completed(pending_import(), "/d", attempt_kind)

        assert (result.all_done, result.pause_reason) == (False, PauseReason.NONE)
        assert sonarr.execute_calls == []

    def test_an_unreadable_command_list_waits_without_a_rescan(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.commands_return = None

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 0

    def test_a_command_list_unreadable_after_the_fetch_waits(self) -> None:
        # The list reads fine again just before the post, so only the failed read after the fetch holds it.
        strat, sonarr = _strategy([_blocked_row()])
        sonarr.on_fetch = _then_command_list_fails(sonarr)
        sonarr.on_languages = _then_lists(sonarr)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []

    def test_a_command_list_unreadable_just_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])
        sonarr.on_languages = _then_command_list_fails(sonarr)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []

    def test_a_restart_just_before_the_post_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def restart() -> None:
            sonarr.commands_return = []

        sonarr.on_languages = restart

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.NONE
        assert sonarr.execute_calls == []


class TestEpisodeReads:
    """The episode files read on this poll decide what is imported, and a failed read never reads as imported."""

    def test_files_a_pass_imports_during_the_fetch_read_as_imported(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def import_the_file() -> None:
            sonarr.episodes_return = _imported_episode()
            sonarr.commands_return = [NEUTRAL_RESCAN, started_process_pass()]

        sonarr.on_fetch = import_the_file

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert (result.all_done, result.imported_count) == (True, 1)
        assert sonarr.execute_calls == []

    def test_an_episode_read_that_fails_after_the_fetch_waits(self) -> None:
        strat, sonarr = _strategy([_blocked_row()])

        def fail_the_next_read() -> None:
            sonarr.episodes_return = None

        sonarr.on_episodes = fail_the_next_read

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert (result.all_done, result.pause_reason) == (False, PauseReason.NONE)
        assert sonarr.execute_calls == []

    def test_a_failed_episode_read_never_reads_as_imported_from_the_last_one(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = _imported_episode()
        pending = pending_import()

        imported = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = None
        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (imported.all_done, result.all_done) == (True, False)
        assert sonarr.execute_calls == []

    def test_a_failed_episode_read_keeps_the_counts_of_the_last_one(self) -> None:
        strat, sonarr = _strategy([queue_record("ABC123", "downloading")])
        sonarr.episodes_return = [*_imported_episode(), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101], "Show - 02 [1080p].mkv": [102]})

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = None
        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (result.imported_count, result.file_map_episode_count) == (1, 2)

    def test_the_progress_on_a_failed_episode_read_is_indeterminate(self) -> None:
        strat, sonarr = _strategy([])
        sonarr.episodes_return = None

        assert strat.import_progress(pending_import()) == NO_PROGRESS


def _strategy_whose_first_fetch_meets_a_pass(clock: FakeClock) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A blocked download whose first fetch sees Sonarr start a pass, and whose every later fetch fails.

    So its first poll is held, and only the fetch that poll kept can post the files later.
    """

    strat, sonarr = _strategy([_blocked_row()], clock=clock)

    def start_a_pass() -> None:
        sonarr.commands_return = [NEUTRAL_RESCAN, started_process_pass()]
        sonarr.candidates_return = None
        sonarr.on_fetch = None

    sonarr.on_fetch = start_a_pass
    return strat, sonarr


class TestKeptFetch:
    """A poll Sonarr holds keeps its fetch, and the pending import's next poll posts those files without fetching again.

    The kept fetch serves one poll, under the same content path, within two poll intervals. A poll that finds no queue
    record fetches afresh and drops it. A fetch that found no files is never kept.
    """

    @pytest.mark.parametrize(
        ("seconds_later", "content_path", "posted"),
        [
            # Kept up to two poll intervals, so a late next poll still finds it.
            pytest.param(50, "/d", _POSTED, id="a-late-next-poll"),
            pytest.param(90, "/d", [], id="past-two-poll-intervals"),
            pytest.param(30, "/elsewhere", [], id="another-content-path"),
        ],
    )
    def test_the_next_poll_posts_the_kept_files_only_soon_enough_and_under_the_same_path(
        self, seconds_later: int, content_path: str, posted: _Imports
    ) -> None:
        clock = FakeClock()
        strat, sonarr = _strategy_whose_first_fetch_meets_a_pass(clock)
        pending = pending_import()

        held = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [NEUTRAL_RESCAN]
        clock.tick(seconds_later)
        strat.import_completed(pending, content_path, AttemptKind.ORDINARY)

        assert held.pause_reason is PauseReason.BUSY
        assert posted_imports(sonarr) == posted

    def test_a_fetch_that_found_no_files_is_never_kept(self) -> None:
        # The next poll fetches again and finds the files.
        clock = FakeClock()
        strat, sonarr = make_sonarr_for_import(candidates=[], queue=[_blocked_row()], clock=clock)
        sonarr.on_fetch = _then_lists(sonarr, started_process_pass())
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.commands_return = [NEUTRAL_RESCAN]
        sonarr.candidates_return = [manual_candidate(_FILE)]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == _POSTED

    def test_a_poll_with_no_queue_record_drops_the_kept_fetch(self) -> None:
        # Nothing is posted, then or on the poll after it, when the record is back and the kept fetch still fresh.
        clock = FakeClock()
        strat, sonarr = _strategy_whose_first_fetch_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [NEUTRAL_RESCAN]
        sonarr.queue_return = []
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.queue_return = [_blocked_row()]
        clock.tick(25)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_the_kept_fetch_serves_one_poll_only(self) -> None:
        # The poll that takes it is held too, and the poll after it finds nothing kept.
        clock = FakeClock()
        strat, sonarr = _strategy_whose_first_fetch_meets_a_pass(clock)
        pending = pending_import()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [NEUTRAL_RESCAN]
        sonarr.on_queue = _then_lists(sonarr, started_process_pass())
        clock.tick(30)
        taken = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.on_queue = None
        sonarr.commands_return = [NEUTRAL_RESCAN]
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert taken.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []
