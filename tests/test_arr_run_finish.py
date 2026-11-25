# pyright: strict
# pyright: reportPrivateUsage=false
# These tests still call the arr run loop's private finish on a bare loop and read its state.
# Strict re-flags that, and the repo disables reportPrivateUsage for tests.
"""Tests for the arr run loop's finish: the order of its steps, what a raise leaves, and its `imported` count.

The reporter, store and wait manager are typed recorders, so the order is read off one list of markers. The
finalize site that calls the finish is pinned in `test_run_finalize`.
"""

from typing import override

import pytest

from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.wait_types import ImportWaitMode, Outcome, PendingImportState
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import CLIENT_SENTINEL, FakeCacheStore, make_bare_instance, make_config, make_logger, make_services
from .fakes import FakeStrategy


class _FinalizeReporter:
    """Records the arr run finish's summary + the two close boundaries as markers."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def scan_finished(self, arr: Arr) -> None:
        self._calls.append("scan_finished")

    def log_arr_run_summary(
        self,
        state: ArrRunState,
        *,
        dry_run: bool,
        has_client: bool,
    ) -> bool:
        self._calls.append("summary")
        return True

    def arr_run_finished(self, arr: Arr) -> None:
        self._calls.append("run_finished")


class _RecordingCacheStore(FakeCacheStore):
    """A `FakeCacheStore` whose `save` appends a `"save"` ordering marker."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    @override
    def save(self, *, dry_run: bool) -> None:
        self._calls.append("save")


class _FinalizeWaitManager:
    """A fake wait manager for the finish-ordering tests.

    Every step appends its ordering marker. The real ones return early on the
    empty working set these tests build - recording nothing - so a recording
    fake is what makes each step observable. `raise_on` scripts one step to
    fail, for the unwind pins (the raise escapes `_finish_arr_run`, exactly as a
    real failure would, and bootstrap's finally is what closes the run).
    `check_result` / `blocking_result` script what the wait passes return, so the
    finish's `imported` bump off the result rows is observable too.
    """

    def __init__(
        self,
        calls: list[str],
        *,
        raise_on: str | None = None,
        check_result: WaitResult | None = None,
        blocking_result: WaitResult | None = None,
    ) -> None:
        self._calls = calls
        self._raise_on = raise_on
        self._check_result = check_result
        self._blocking_result = blocking_result

    def _mark(self, name: str) -> None:
        self._calls.append(name)
        if name == self._raise_on:
            raise RuntimeError(f"{name} exploded")

    def finish_cleanups(self) -> None:
        self._mark("cleanup")

    def run_deferred(self) -> WaitResult | None:
        self._mark("check")
        return self._check_result

    def tally_carried_over_into_stats(self) -> None:
        self._mark("tally")

    def run_blocking(self) -> WaitResult | None:
        self._mark("monitor")
        return self._blocking_result


def _finalize_engine(
    calls: list[str],
    *,
    qbit: object,
    mode: ImportWaitMode,
    raise_on: str | None = None,
    supports_blocking_wait: bool = True,
    check_result: WaitResult | None = None,
    blocking_result: WaitResult | None = None,
) -> ArrRunLoop:
    """A bare arr run loop whose every finish step appends a marker to `calls`.

    The reporter, cache store, and wait manager are typed recorders (a
    `_FinalizeReporter` / `_RecordingCacheStore` / `_FinalizeWaitManager`), so
    `_finish_arr_run`'s ordering is asserted on the recorded `calls` list without a
    live Sonarr/qBittorrent. The finish's dry-run fact comes off the services hub,
    so a bare one shares the arr run loop's state (and the `qbit` that decides dry run).
    `raise_on` names the wait-manager step that should blow up. `supports_blocking_wait`
    scripts the active strategy's blocking-wait flag - True is a Sonarr-shaped
    strategy, False a Radarr one (reconciles pre-summary, never runs a blocking wait pass).
    """

    state = ArrRunState(arr=Arr.SONARR, import_wait_mode=mode)
    return make_bare_instance(
        ArrRunLoop,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
        ),
        _reporter=_FinalizeReporter(calls),
        cache_store=_RecordingCacheStore(calls),
        _wait_manager=_FinalizeWaitManager(
            calls,
            raise_on=raise_on,
            check_result=check_result,
            blocking_result=blocking_result,
        ),
        _services=make_services(qbit=qbit, _state=state),
        _active_strategy=FakeStrategy(items=[], anilist_ids={}, supports_blocking_wait=supports_blocking_wait),
        _state=state,
    )


class TestFinishArrRunOrdering:
    """_finish_arr_run brackets the finish with its close boundaries, summary before the blocking wait pass.

    `scan_finished` leads on every path - so the reconcile/tally diagnostics that
    follow print at run level, never inside the last title - and `arr_run_finished`
    trails the save, so it is the arr run's last event. Finishing the cleanups leads the
    wait steps in every active mode, before any wait pass reads the store.
    """

    @pytest.mark.parametrize(
        ("qbit", "mode", "supports_blocking_wait", "steps"),
        [
            # The arr run summary must print before the blocking wait pass, and the save must be dead
            # last (so the store reflects the wait pass's drops).
            pytest.param(
                CLIENT_SENTINEL,
                ImportWaitMode.BLOCKING,
                True,
                ["scan_finished", "cleanup", "tally", "summary", "monitor", "save", "run_finished"],
                id="blocking",
            ),
            pytest.param(
                CLIENT_SENTINEL,
                ImportWaitMode.DEFERRED,
                True,
                ["scan_finished", "cleanup", "check", "tally", "summary", "save", "run_finished"],
                id="deferred",
            ),
            # A Radarr-shaped strategy (no blocking wait) reconciles pre-summary in
            # BLOCKING too - the blocking wait pass would otherwise be its only reconcile - and
            # never enters the wait region.
            pytest.param(
                CLIENT_SENTINEL,
                ImportWaitMode.BLOCKING,
                False,
                ["scan_finished", "cleanup", "check", "tally", "summary", "save", "run_finished"],
                id="blocking-without-a-wait-pass",
            ),
            # A dry run (no client) runs no wait step.
            pytest.param(
                None, ImportWaitMode.BLOCKING, True, ["scan_finished", "summary", "save", "run_finished"], id="dry-run"
            ),
        ],
    )
    def test_each_mode_runs_its_wait_steps_in_order(
        self, qbit: object, mode: ImportWaitMode, supports_blocking_wait: bool, steps: list[str]
    ) -> None:
        calls: list[str] = []
        engine = _finalize_engine(calls, qbit=qbit, mode=mode, supports_blocking_wait=supports_blocking_wait)

        engine._finish_arr_run()

        assert calls == steps


class TestFinishArrRunUnwind:
    """A raise in the finish leaves `arr_run_finished` to bootstrap's unwind emit.

    The scan is already closed either way, so the arr-run-fatal error the composition
    root logs can never render inside a leftover title. The
    finally spans the whole finish, so a raise still saves - the run's staged writes
    survive rather than rolling back when bootstrap closes the store.
    """

    def test_reconcile_raise_closes_the_scan_but_not_the_run(self) -> None:
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.DEFERRED,
            raise_on="check",
        )

        with pytest.raises(RuntimeError, match="check exploded"):
            engine._finish_arr_run()

        # The raise still escapes (bootstrap closes the run), but the finally now
        # saves so the run's staged writes persist rather than rolling back.
        assert calls == ["scan_finished", "cleanup", "check", "save"]

    def test_blocking_wait_raise_still_saves_but_leaves_the_run_open(self) -> None:
        # arr_run_finished sits OUTSIDE the save's finally on purpose: emitting it there
        # would double up with bootstrap's unwind emit on this very path.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
            raise_on="monitor",
        )

        with pytest.raises(RuntimeError, match="monitor exploded"):
            engine._finish_arr_run()

        assert calls == ["scan_finished", "cleanup", "tally", "summary", "monitor", "save"]

    def test_tally_raise_still_saves_but_leaves_the_run_open(self) -> None:
        # A pre-summary raise also lands in the finally, so the run's staged writes
        # persist. Deferred so reconcile precedes tally and pins the whole prefix.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.DEFERRED,
            raise_on="tally",
        )

        with pytest.raises(RuntimeError, match="tally exploded"):
            engine._finish_arr_run()

        assert calls == ["scan_finished", "cleanup", "check", "tally", "save"]


class TestFinishArrRunImportedBump:
    """_finish_arr_run owns the tally's `imported` bump, filtered to carried-over rows."""

    def test_deferred_counts_scan_report_and_check_imports(self) -> None:
        calls: list[str] = []
        check = WaitResult(
            (
                WaitOutcomeRow("carried", Outcome.IMPORTED, carried_over=True),
                WaitOutcomeRow("fresh", Outcome.IMPORTED),
                WaitOutcomeRow("left", Outcome.STILL_DOWNLOADING, carried_over=True),
            ),
            elapsed_s=1.0,
        )
        engine = _finalize_engine(calls, qbit=CLIENT_SENTINEL, mode=ImportWaitMode.DEFERRED, check_result=check)
        engine._state.pending_import_states["snap"] = PendingImportState.IMPORTED
        engine._state.pending_import_states["kept"] = PendingImportState.DOWNLOADED

        engine._finish_arr_run()

        # One import the report confirmed plus one carried-over deferred-wait import. The
        # fresh row stays `added` and the non-imported ones stay out.
        assert engine._state.stats.imported == 2

    def test_blocking_counts_carried_over_waited_imports(self) -> None:
        calls: list[str] = []
        wait_result = WaitResult(
            (
                WaitOutcomeRow("carried", Outcome.IMPORTED, carried_over=True),
                WaitOutcomeRow("fresh", Outcome.IMPORTED),
            ),
            elapsed_s=1.0,
        )
        engine = _finalize_engine(
            calls, qbit=CLIENT_SENTINEL, mode=ImportWaitMode.BLOCKING, blocking_result=wait_result
        )

        engine._finish_arr_run()

        assert engine._state.stats.imported == 1
