# pyright: strict
"""Tests for how a wait row ends when the attempt result names files the import left behind.

Unplaced files end the row on the poll that returned them, in either wait mode. Files left for a hand import end
it as a partial import that stays stored.
"""

from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.wait_types import AttemptKind, Outcome, PendingImportState

from .builders import (
    FakeCacheStore,
    attempt_result,
    files_left_for_hand_import,
    order_only_hand_import_files,
    pending_import,
)
from .fakes import FakeClock
from .wait_fakes import (
    CategoryQbit,
    FakeQbit,
    FakeTorrent,
    RecordingStrategy,
    RecordingWaitView,
    make_orchestration_manager,
    run_blocking_wait,
    stored_awaiting_cleanup,
)


class TestUnplacedRowFinish:
    """A result naming unplaced files finishes on the poll that returned it, in either wait mode, and stays stored."""

    _NAMES = ("a.mkv", "b.mkv")

    @classmethod
    def _strategy(cls) -> RecordingStrategy:
        return RecordingStrategy(attempts=[attempt_result(all_done=False, unmatched_files=cls._NAMES)])

    def test_run_deferred_finishes_unplaced_and_keeps_the_pending_import(self) -> None:
        strategy = self._strategy()
        store = FakeCacheStore()
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=store,
            state=run,
            stored_pending_imports=[pending_import(infohash="h")],
        )
        view = RecordingWaitView()

        result = mgr.run_deferred(view=view)

        assert result is not None
        assert [(row.outcome, row.left_for_hand_import) for row in result.rows] == [
            (Outcome.NOTHING_PLACED, files_left_for_hand_import(*self._NAMES))
        ]
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY]
        assert view.final("h").left_for_hand_import.names == self._NAMES
        assert run.pending_import_states == {"h": PendingImportState.DOWNLOADED}
        assert stored_awaiting_cleanup(store) == {"h": False}

    def test_run_blocking_finishes_unplaced_on_the_first_poll(self) -> None:
        # The result ends the wait on the poll that establishes it: no final attempt, no import clock.
        strategy = self._strategy()
        store, view = run_blocking_wait(strategy, FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}))

        final = view.final("h")
        assert final.outcome is Outcome.NOTHING_PLACED
        assert final.left_for_hand_import.names == self._NAMES
        assert final.phase_elapsed_s == 0
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY]
        assert stored_awaiting_cleanup(store) == {"h": False}

    def test_later_polls_never_re_poll_a_finished_row(self) -> None:
        # A slower sibling keeps the wait pass running past the first poll. The finished row is not asked again.
        strategy = self._strategy()
        qbit = FakeQbit(
            {
                "h": [FakeTorrent(is_complete=True, content_path="/d")],
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            grabbed=[pending_import(infohash="h"), pending_import(infohash="slow")],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert [call.infohash for call in strategy.import_calls] == ["h", "slow"]
        assert view.final("h").outcome is Outcome.NOTHING_PLACED
        assert view.final("slow").outcome is Outcome.NOTHING_PLACED
        assert stored_awaiting_cleanup(store) == {"h": False, "slow": False}


class TestLeftForHandImportRowFinish:
    """A result naming hand-import files finishes as a partial import: kept for the hand import, never cleaned up."""

    def test_each_run_names_the_hand_import_files_and_leaves_the_torrent_in_its_category(self) -> None:
        left = order_only_hand_import_files()
        store = FakeCacheStore()
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        run1 = make_orchestration_manager(
            qbit=qbit,
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=False, left_for_hand_import=left)]),
            store=store,
            stored_pending_imports=[pending_import(infohash="h")],
            post_import_category="pearlarr-done",
        )
        # The same store, next run.
        run2_state = ArrRunState(arr=Arr.SONARR)
        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=False, left_for_hand_import=left)]),
            store=store,
            state=run2_state,
            post_import_category="pearlarr-done",
        )

        first = run1.run_deferred(view=RecordingWaitView())
        second = run2.run_deferred(view=RecordingWaitView())

        assert first is not None
        assert [(row.outcome, row.left_for_hand_import) for row in first.rows] == [(Outcome.PARTIAL, left)]
        assert second is not None
        assert [(row.outcome, row.left_for_hand_import) for row in second.rows] == [(Outcome.PARTIAL, left)]
        assert run2_state.pending_import_states == {"h": PendingImportState.DOWNLOADED}
        assert stored_awaiting_cleanup(store) == {"h": False}
        assert qbit.set_category_calls == []
