# pyright: strict
"""Tests for the deferred end pass, one poll over every carried-over pending import.

They cover the outcome and the state each reading gives, and the tally that counts the ones still waiting.
"""

from typing import override

import pytest
import qbittorrentapi

from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.pending_import import PendingImport
from pearlarr.wait_types import (
    NO_ATTEMPT,
    NO_PROGRESS,
    PENDING_IMPORT_STATE_FOR_OUTCOME,
    AttemptResult,
    ImportProgress,
    Outcome,
    PendingImportState,
)

from .builders import FakeCacheStore, attempt_result, pending_import
from .fakes import FakeClock
from .wait_fakes import (
    FakeQbit,
    FakeTorrent,
    QbitStep,
    RecordingStrategy,
    make_orchestration_manager,
    stored_awaiting_cleanup,
)


class _AutoTickClock(FakeClock):
    """Time passing between reads: every `now` reading advances by `step`."""

    @override
    def monotonic(self) -> float:
        t = super().monotonic()
        self.t += self._step
        return t


class TestRunDeferred:
    """run_deferred runs ONE non-blocking poll over every carried-over pending import."""

    @pytest.mark.parametrize(
        ("attempts", "progress"),
        [
            pytest.param([attempt_result(all_done=True)], [NO_PROGRESS], id="the-attempt-confirms"),
            pytest.param(
                [attempt_result(all_done=False, command_issued=True)],
                [ImportProgress(1, 1, determinate=True)],
                id="the-progress-read-confirms",
            ),
        ],
    )
    def test_each_ready_carried_over_import_is_counted_and_dropped_and_a_this_run_grab_is_left(
        self, attempts: list[AttemptResult], progress: list[ImportProgress]
    ) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({key: [FakeTorrent(is_complete=True, content_path="/d")] for key in ("h1", "h2", "new")}),
            strategy=RecordingStrategy(attempts=attempts, progress=progress),
            store=store,
            stored_pending_imports=[pending_import(infohash="h1"), pending_import(infohash="h2")],
            grabbed=[pending_import(infohash="new")],
        )

        result = mgr.run_deferred()

        assert result is not None
        assert result.carried_over_imported == 2
        assert stored_awaiting_cleanup(store) == {"new": False}

    @pytest.mark.parametrize(
        ("torrent", "attempts", "outcome", "state"),
        [
            pytest.param(
                FakeTorrent(progress=0.5),
                [NO_ATTEMPT],
                Outcome.STILL_DOWNLOADING,
                PendingImportState.DOWNLOADING,
                id="still-downloading",
            ),
            # A dropped connection or a re-auth in flight is a read that didn't happen, not a finish.
            pytest.param(
                qbittorrentapi.APIConnectionError("down"),
                [NO_ATTEMPT],
                Outcome.QBITTORRENT_UNREACHABLE,
                PendingImportState.DOWNLOADING,
                id="qbittorrent-unreachable",
            ),
            pytest.param(
                FakeTorrent(is_complete=True),
                [NO_ATTEMPT],
                Outcome.NO_CONTENT_PATH,
                PendingImportState.DOWNLOADED,
                id="no-content-path",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                [attempt_result(all_done=False, command_issued=True)],
                Outcome.IMPORT_IN_PROGRESS,
                PendingImportState.DOWNLOADED,
                id="import-in-progress",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                [attempt_result(all_done=False)],
                Outcome.AWAITING_IMPORT,
                PendingImportState.DOWNLOADED,
                id="awaiting-import",
            ),
            # A torrent at 100% is done downloading even before qBittorrent flags it complete.
            pytest.param(
                FakeTorrent(progress=1.0, content_path="/d"),
                [attempt_result(all_done=False)],
                Outcome.AWAITING_IMPORT,
                PendingImportState.DOWNLOADED,
                id="full-progress-without-the-complete-flag",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                [RuntimeError("boom")],
                Outcome.ATTEMPT_FAILED,
                PendingImportState.DOWNLOADED,
                id="attempt-failed",
            ),
            pytest.param(
                FakeTorrent(is_errored=True),
                [NO_ATTEMPT],
                Outcome.DOWNLOAD_ERRORED,
                PendingImportState.ERRORED,
                id="download-errored",
            ),
            # A finished torrent whose files went missing reads errored at 100%: no import is tried.
            pytest.param(
                FakeTorrent(is_errored=True, progress=1.0, content_path="/d"),
                [attempt_result(all_done=True)],
                Outcome.DOWNLOAD_ERRORED,
                PendingImportState.ERRORED,
                id="errored-at-full-progress",
            ),
        ],
    )
    def test_each_reading_gives_the_outcome_and_the_state_the_tally_counts(
        self,
        torrent: QbitStep,
        attempts: list[AttemptResult | Exception],
        outcome: Outcome,
        state: PendingImportState,
    ) -> None:
        # No series report read the pending import earlier this run.
        store = FakeCacheStore()
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=RecordingStrategy(attempts=attempts),
            store=store,
            state=run,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        result = mgr.run_deferred()

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(outcome, True)]
        assert run.pending_import_states == {"h": state}
        assert stored_awaiting_cleanup(store) == {"h": False}

    def test_the_end_pass_state_replaces_the_one_the_series_report_noted(self) -> None:
        # The series report read the torrent still downloading, and it finished before the end pass.
        run = ArrRunState(arr=Arr.SONARR, pending_import_states={"h": PendingImportState.DOWNLOADING})
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=False)]),
            state=run,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        mgr.run_deferred()

        assert run.pending_import_states == {"h": PendingImportState.DOWNLOADED}

    def test_a_deferred_pass_never_times_out_a_download(self) -> None:
        # The download clock starts when the pass is built, so a timeout here would only measure the pass.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.5)]}),
            strategy=RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="h")],
            download_timeout=1,
            clock=_AutoTickClock(step=30),
        )

        result = mgr.run_deferred()

        assert result is not None
        assert [row.outcome for row in result.rows] == [Outcome.STILL_DOWNLOADING]


class TestPendingImportStateConversion:
    """The pinned Outcome -> PendingImportState conversion a wait pass classifies stored pending imports through."""

    def test_conversion_covers_exactly_the_non_dropped_outcomes(self) -> None:
        # IMPORTED and GONE leave the store, so the conversion never sees them.
        assert set(PENDING_IMPORT_STATE_FOR_OUTCOME) == {o for o in Outcome if not o.dropped}


class TestTallyCarriedOverIntoStats:
    """tally_carried_over_into_stats counts each still-waiting pending import once."""

    def test_a_carried_over_pending_import_counts_by_the_state_the_run_noted(self) -> None:
        # Both counters add up, a pending import nothing observed this run counts as downloading, and an
        # errored one is counted nowhere.
        state = ArrRunState(
            arr=Arr.SONARR,
            pending_import_states={
                "q": PendingImportState.DOWNLOADING,
                "d1": PendingImportState.DOWNLOADED,
                "d2": PendingImportState.DOWNLOADED,
                "e": PendingImportState.ERRORED,
            },
        )
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash=key) for key in ("q", "d1", "d2", "e", "untouched")],
            state=state,
        )

        mgr.tally_carried_over_into_stats()

        assert (state.stats.downloading, state.stats.downloaded) == (2, 2)

    @pytest.mark.parametrize(
        ("stored", "grabbed", "seen_again", "counted"),
        [
            # Found in qBittorrent again without a grab, it still counts.
            pytest.param([pending_import(infohash="h")], [], ("h",), (1, 0), id="seen-again-this-run"),
            # A grab this run counts as grabbed instead.
            pytest.param([], [pending_import(infohash="h")], (), (0, 0), id="grabbed-this-run"),
            # It already imported and only its cleanup is owed.
            pytest.param([pending_import(infohash="h", awaiting_cleanup=True)], [], (), (0, 0), id="awaiting-cleanup"),
        ],
    )
    def test_counts_each_carried_over_pending_import_still_waiting(
        self,
        stored: list[PendingImport],
        grabbed: list[PendingImport],
        seen_again: tuple[str, ...],
        counted: tuple[int, int],
    ) -> None:
        state = ArrRunState(arr=Arr.SONARR, reacquired_keys=set(seen_again))
        mgr = make_orchestration_manager(
            qbit=None, strategy=RecordingStrategy(), stored_pending_imports=stored, grabbed=grabbed, state=state
        )

        mgr.tally_carried_over_into_stats()

        assert (state.stats.downloading, state.stats.downloaded) == counted
