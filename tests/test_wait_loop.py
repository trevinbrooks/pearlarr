"""Unit tests for the engine's qBittorrent completion wait/poll machinery.

These pin :meth:`SeaDexArr._poll_torrent` (the single-shot state read) and
:meth:`SeaDexArr._wait_for_completion` (the poll loop) against a scripted
``FakeQbit``. The clock and sleep are injected into the wait loop so it never
actually waits - real foreground ``sleep`` is blocked in this env, so the fakes
are mandatory, not just a speed-up. The engine is built bare (``object.__new__``
via ``make_bare_instance``) so no live qBittorrent login or disk I/O happens.
"""

import types
from typing import cast, override
from unittest import mock

import qbittorrentapi

from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import (
    ImportReadiness,
    ImportWaitMode,
    PendingState,
    WaitOutcome,
)
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.wait_view import WaitView

from .builders import (
    import_probe,
    make_bare_instance,
    make_config,
    make_logger,
    pending_import,
)


class FakeStateEnum:
    """Mimics qBittorrent's ``state_enum`` (the ``is_*`` booleans the poll reads)."""

    def __init__(self, *, is_complete: bool = False, is_errored: bool = False) -> None:
        self.is_complete = is_complete
        self.is_errored = is_errored


class FakeTorrent:
    """Mimics one qBittorrent torrent info row (the fields ``_poll_torrent`` reads)."""

    def __init__(
        self,
        *,
        is_complete: bool = False,
        is_errored: bool = False,
        progress: float = 0.0,
        content_path: str | None = None,
    ) -> None:
        self.state_enum = FakeStateEnum(is_complete=is_complete, is_errored=is_errored)
        self.progress = progress
        self.content_path = content_path


class FakeQbit:
    """A scriptable qBittorrent client returning queued ``torrents_info`` results.

    Each call to ``torrents_info`` pops the next scripted result; once the script
    is exhausted the last result repeats (so a steady "downloading" state can be
    polled indefinitely). A scripted value may be an exception instance, which is
    raised to exercise the transient-error path.
    """

    def __init__(self, results: list[object]) -> None:
        self._results = results
        self.calls = 0

    def torrents_info(self, *, torrent_hashes: str) -> object:
        self.calls += 1
        result = self._results[min(self.calls - 1, len(self._results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def make_engine(qbit: FakeQbit) -> SeaDexArr:
    """A bare ``SeaDexArr`` wired only with the ``qbit`` the wait path reads."""

    return make_bare_instance(SeaDexArr, qbit=qbit)


class TestPollTorrent:
    """_poll_torrent maps a single qBittorrent read to (outcome, path, progress)."""

    def test_missing_on_empty_list(self) -> None:
        engine = make_engine(FakeQbit([[]]))

        assert engine._poll_torrent("h") == (WaitOutcome.MISSING, None, 0.0)

    def test_errored(self) -> None:
        engine = make_engine(FakeQbit([[FakeTorrent(is_errored=True)]]))

        assert engine._poll_torrent("h") == (WaitOutcome.ERRORED, None, 0.0)

    def test_complete_carries_content_path(self) -> None:
        torrent = FakeTorrent(is_complete=True, content_path="/data/show")
        engine = make_engine(FakeQbit([[torrent]]))

        assert engine._poll_torrent("h") == (WaitOutcome.COMPLETE, "/data/show", 0.0)

    def test_complete_on_full_progress_without_flag(self) -> None:
        # progress == 1.0 counts as complete even if the state flag is unset.
        torrent = FakeTorrent(progress=1.0, content_path="/data/movie")
        engine = make_engine(FakeQbit([[torrent]]))

        assert engine._poll_torrent("h") == (WaitOutcome.COMPLETE, "/data/movie", 1.0)

    def test_none_while_downloading_carries_progress(self) -> None:
        engine = make_engine(FakeQbit([[FakeTorrent(progress=0.5)]]))

        assert engine._poll_torrent("h") == (None, None, 0.5)

    def test_none_on_transient_api_error(self) -> None:
        # A dropped connection / re-auth in flight is "still waiting", not terminal.
        engine = make_engine(FakeQbit([qbittorrentapi.APIConnectionError("boom")]))

        assert engine._poll_torrent("h") == (None, None, 0.0)

    def test_none_when_no_client(self) -> None:
        engine = make_bare_instance(SeaDexArr, qbit=None)

        assert engine._poll_torrent("h") == (None, None, 0.0)


class FakeClock:
    """A monotonic clock the wait loop reads; advances by a fixed step per sleep."""

    def __init__(self, step: float) -> None:
        self.t = 0.0
        self._step = step

    def now(self) -> float:
        return self.t

    def sleep(self, _seconds: float) -> None:
        # Ignore the requested duration; advance our own clock so the loop's
        # deadline arithmetic is exercised without ever really sleeping.
        self.t += self._step


class TestWaitForCompletion:
    """_wait_for_completion loops _poll_torrent against an injected clock."""

    def test_returns_complete_immediately(self) -> None:
        torrent = FakeTorrent(is_complete=True, content_path="/data/show")
        engine = make_engine(FakeQbit([[torrent]]))
        clock = FakeClock(step=10)

        outcome, path = engine._wait_for_completion(
            "h", timeout_s=3600, poll_s=30, now=clock.now, sleep=clock.sleep,
        )

        assert (outcome, path) == (WaitOutcome.COMPLETE, "/data/show")

    def test_returns_errored_immediately(self) -> None:
        engine = make_engine(FakeQbit([[FakeTorrent(is_errored=True)]]))
        clock = FakeClock(step=10)

        outcome, path = engine._wait_for_completion(
            "h", timeout_s=3600, poll_s=30, now=clock.now, sleep=clock.sleep,
        )

        assert (outcome, path) == (WaitOutcome.ERRORED, None)

    def test_returns_missing_immediately(self) -> None:
        engine = make_engine(FakeQbit([[]]))
        clock = FakeClock(step=10)

        outcome, path = engine._wait_for_completion(
            "h", timeout_s=3600, poll_s=30, now=clock.now, sleep=clock.sleep,
        )

        assert (outcome, path) == (WaitOutcome.MISSING, None)

    def test_times_out_after_deadline_while_downloading(self) -> None:
        # Always downloading -> the loop keeps polling/sleeping until the injected
        # clock crosses the deadline, then returns TIMED_OUT (never really sleeps).
        qbit = FakeQbit([[FakeTorrent(progress=0.5)]])
        engine = make_engine(qbit)
        clock = FakeClock(step=30)

        outcome, path = engine._wait_for_completion(
            "h", timeout_s=60, poll_s=30, now=clock.now, sleep=clock.sleep,
        )

        assert (outcome, path) == (WaitOutcome.TIMED_OUT, None)
        # Polls at t=0 (sleep->30) and t=30 (sleep->60), then the t=60 poll sees
        # 60 >= the 60s deadline -> timed out: three polls in all.
        assert qbit.calls == 3

    def test_completes_after_some_downloading_polls(self) -> None:
        # First two polls show downloading, the third shows complete; the loop
        # should return COMPLETE rather than timing out.
        qbit = FakeQbit(
            [
                [FakeTorrent(progress=0.3)],
                [FakeTorrent(progress=0.7)],
                [FakeTorrent(is_complete=True, content_path="/data/done")],
            ],
        )
        engine = make_engine(qbit)
        clock = FakeClock(step=30)

        outcome, path = engine._wait_for_completion(
            "h", timeout_s=3600, poll_s=30, now=clock.now, sleep=clock.sleep,
        )

        assert (outcome, path) == (WaitOutcome.COMPLETE, "/data/done")
        assert qbit.calls == 3


# A timestamp far enough in the past/future that the TTL verdict is fixed no
# matter when the suite runs (cutoff = now - import_pending_max_age_days).
_FRESH = "2999-01-01 00:00:00"
_EXPIRED = "2000-01-01 00:00:00"


def make_orchestration_engine(
    *,
    qbit: object,
    strategy: mock.MagicMock,
    store_records: list[dict] | None = None,
    pending: list | None = None,
    **config_overrides: object,
) -> SeaDexArr:
    """A bare ``SeaDexArr`` wired for the pending-import orchestration paths.

    Seeds the durable per-arr store (via the engine's own ``_pending_store``) and
    the in-memory ``_ctx.pending_imports`` list so ``_prune_expired_pending``,
    ``_snapshot_pending_for_series``, ``_reconcile_remaining`` and ``_run_monitor``
    can be driven without a live Sonarr/qBittorrent.
    """

    engine = make_bare_instance(
        SeaDexArr,
        qbit=qbit,
        logger=make_logger(),
        log_fmt=mock.MagicMock(),
        _config=make_config(**config_overrides),
        _active_strategy=strategy,
        _reporter=mock.MagicMock(),
        cache_store=types.SimpleNamespace(data={}),
    )
    engine._ctx = RunContext(arr=Arr.SONARR, pending_imports=list(pending or []))
    store = engine._pending_store()
    for record in store_records or []:
        store[record["infohash"]] = record
    return engine


class TestPruneExpiredPending:
    """_prune_expired_pending drops aged-out / unparseable records only."""

    def test_drops_expired_and_unparseable_keeps_fresh(self) -> None:
        records = [
            pending_import(infohash="fresh", added_at=_FRESH).to_json(),
            pending_import(infohash="old", added_at=_EXPIRED).to_json(),
            pending_import(infohash="bad", added_at="not-a-timestamp").to_json(),
        ]
        engine = make_orchestration_engine(
            qbit=None, strategy=mock.MagicMock(), store_records=records,
        )

        engine._prune_expired_pending()

        assert set(engine._pending_store()) == {"fresh"}


class FakeWaitView(WaitView):
    """Records the WaitView calls the engine makes, for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    @override
    def start(self, torrents: list[tuple[str, str]]) -> None:
        self.events.append(("start", torrents))

    @override
    def download(self, key: str, pct: float, elapsed: float, timeout: float) -> None:
        self.events.append(("download", key, pct))

    @override
    def importing(self, key: str, elapsed: float, timeout: float) -> None:
        self.events.append(("importing", key))

    @override
    def done(self, key: str, outcome: str) -> None:
        self.events.append(("done", key, outcome))

    @override
    def close(self) -> None:
        self.events.append(("close",))


class TestSnapshotPendingForSeries:
    """_snapshot_pending_for_series reports CARRIED-OVER records inline, no double-report."""

    def test_carried_over_imported_drops_and_counts(self) -> None:
        # A prior run's download finished: COMPLETE + verified files -> the record
        # is reported imported, dropped, and stats.imported bumped.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = import_probe(
            ImportReadiness.IMPORTED, files_present=True,
        )
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH).to_json()],
        )

        engine._snapshot_pending_for_series(7)

        assert engine._pending_store() == {}
        assert engine._ctx.stats.imported == 1
        assert engine._ctx.pending_states["h"] is PendingState.IMPORTED
        # Forced (CDH-off safe) but NOT at the deadline (no loud warning).
        assert strategy.import_completed.call_args.kwargs.get("force") is True
        assert strategy.import_completed.call_args.kwargs.get("at_deadline") is False

    def test_carried_over_downloading_is_queued_and_kept(self) -> None:
        # Still downloading -> queued, record kept, no import attempt.
        strategy = mock.MagicMock()
        qbit = FakeQbit([[FakeTorrent(progress=0.5)]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH).to_json()],
        )

        engine._snapshot_pending_for_series(7)

        strategy.import_completed.assert_not_called()
        assert set(engine._pending_store()) == {"h"}
        assert engine._ctx.pending_states["h"] is PendingState.QUEUED
        assert engine._ctx.stats.imported == 0

    def test_this_run_grab_is_skipped_no_double_report(self) -> None:
        # REGRESSION (double-report): a torrent grabbed THIS run lives in
        # _ctx.pending_imports AND the store. The snapshot must skip it entirely -
        # no poll, no row, no counter, no state - so it's only ever `added`.
        strategy = mock.MagicMock()
        this_run = pending_import(infohash="h", series_id=7, added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[this_run.to_json()], pending=[this_run],
        )

        engine._snapshot_pending_for_series(7)

        strategy.import_completed.assert_not_called()
        cast("mock.MagicMock", engine._reporter).log_pending_snapshot.assert_not_called()
        assert engine._ctx.pending_states == {}
        assert engine._ctx.stats.imported == 0
        assert set(engine._pending_store()) == {"h"}

    def test_other_series_record_is_not_touched(self) -> None:
        # The snapshot is series-scoped: a record for a different series is left
        # alone (the deferred reconcile / monitor handles it later).
        strategy = mock.MagicMock()
        qbit = FakeQbit([[FakeTorrent(progress=0.5)]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="other", series_id=99, added_at=_FRESH).to_json()],
        )

        engine._snapshot_pending_for_series(7)

        assert "other" not in engine._ctx.pending_states
        assert set(engine._pending_store()) == {"other"}


class TestReconcileRemaining:
    """_reconcile_remaining force-polls carried-over records not snapshotted this run."""

    def test_imports_ready_record_not_yet_snapshotted(self) -> None:
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = import_probe(
            ImportReadiness.IMPORTED, files_present=True,
        )
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_remaining()

        assert engine._pending_store() == {}
        assert engine._ctx.stats.imported == 1
        assert strategy.import_completed.call_args.kwargs.get("force") is True
        assert strategy.import_completed.call_args.kwargs.get("at_deadline") is False

    def test_skips_already_snapshotted(self) -> None:
        # A record the inline snapshot already touched must not be re-polled.
        strategy = mock.MagicMock()
        engine = make_orchestration_engine(
            qbit=FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]]),
            strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )
        engine._ctx.pending_states["h"] = PendingState.QUEUED

        engine._reconcile_remaining()

        strategy.import_completed.assert_not_called()

    def test_skips_this_run_grabs(self) -> None:
        strategy = mock.MagicMock()
        this_run = pending_import(infohash="h", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]]),
            strategy=strategy,
            store_records=[this_run.to_json()], pending=[this_run],
        )

        engine._reconcile_remaining()

        strategy.import_completed.assert_not_called()


class TestTallyCarriedOverIntoStats:
    """_tally_carried_over_into_stats counts each still-pending record once."""

    def test_counts_known_states_and_defaults_to_queued(self) -> None:
        engine = make_orchestration_engine(
            qbit=None, strategy=mock.MagicMock(),
            store_records=[
                pending_import(infohash="q", added_at=_FRESH).to_json(),
                pending_import(infohash="i", added_at=_FRESH).to_json(),
                pending_import(infohash="untouched", added_at=_FRESH).to_json(),
            ],
        )
        engine._ctx.pending_states = {
            "q": PendingState.QUEUED,
            "i": PendingState.IMPORTING,
        }

        engine._tally_carried_over_into_stats()

        assert engine._ctx.stats.queued == 2  # explicit q + defaulted untouched
        assert engine._ctx.stats.importing == 1

    def test_excludes_this_run_grabs(self) -> None:
        this_run = pending_import(infohash="h", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=None, strategy=mock.MagicMock(),
            store_records=[this_run.to_json()], pending=[this_run],
        )

        engine._tally_carried_over_into_stats()

        assert engine._ctx.stats.queued == 0
        assert engine._ctx.stats.importing == 0


class TestRunMonitor:
    """_run_monitor: interleaved, copy-aware wait+import over ALL pending."""

    def test_interleaved_fast_and_slow(self) -> None:
        # Two torrents: "fast" completes + imports first cycle (files present);
        # "slow" is still downloading, then completes + imports a later cycle. Both
        # advance each cycle (interleaved), so the fast one isn't stuck behind slow.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = import_probe(
            ImportReadiness.RETRY, files_present=True,
        )

        class TwoTorrentQbit:
            def __init__(self) -> None:
                self.calls: dict[str, int] = {}

            def torrents_info(self, *, torrent_hashes: str):
                n = self.calls.get(torrent_hashes, 0)
                self.calls[torrent_hashes] = n + 1
                if torrent_hashes == "fast":
                    return [FakeTorrent(is_complete=True, content_path="/fast")]
                # slow: downloading on the first cycle, complete after.
                if n == 0:
                    return [FakeTorrent(progress=0.5)]
                return [FakeTorrent(is_complete=True, content_path="/slow")]

        fast = pending_import(infohash="fast", added_at=_FRESH)
        slow = pending_import(infohash="slow", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=TwoTorrentQbit(), strategy=strategy,
            store_records=[fast.to_json(), slow.to_json()],
            pending=[fast, slow],
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        # Both ultimately imported and dropped.
        assert engine._pending_store() == {}
        assert ("done", "fast", "imported") in view.events
        assert ("done", "slow", "imported") in view.events
        # slow showed a downloading heartbeat before it completed.
        assert ("download", "slow", 0.5) in view.events

    def test_imported_only_when_files_present_two_cycles(self) -> None:
        # The copy is async: cycle 1 issues the command (RETRY + command_issued,
        # files NOT present) -> reads `importing`; cycle 2 verifies files present
        # -> `imported`. imported is gated on verified files, never command accept.
        strategy = mock.MagicMock()
        strategy.import_completed.side_effect = [
            import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
            import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
        ]
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.import_completed.call_count == 2
        assert ("importing", "h") in view.events  # cycle 1, copy in flight
        assert ("done", "h", "imported") in view.events  # cycle 2, files landed
        assert engine._pending_store() == {}

    def test_importing_at_deadline_left_without_warning(self) -> None:
        # The copy never lands within import_ready_timeout: the final attempt
        # (at_deadline) leaves it pending with "still importing; left" - no drop.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = import_probe(
            ImportReadiness.RETRY, files_present=False, command_issued=True,
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_ready_timeout=60, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert ("done", "h", "still importing; left") in view.events
        assert set(engine._pending_store()) == {"h"}  # left, not dropped
        # The final in-bound poll forces AND flags the deadline.
        last = strategy.import_completed.call_args_list[-1]
        assert last.kwargs.get("force") is True
        assert last.kwargs.get("at_deadline") is True

    def test_missing_drops_record(self) -> None:
        strategy = mock.MagicMock()
        pending = pending_import(infohash="h", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=FakeQbit([[]]), strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        strategy.import_completed.assert_not_called()
        assert ("done", "h", "gone from qBittorrent") in view.events
        assert engine._pending_store() == {}

    def test_errored_leaves_record(self) -> None:
        strategy = mock.MagicMock()
        pending = pending_import(infohash="h", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=FakeQbit([[FakeTorrent(is_errored=True)]]), strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        strategy.import_completed.assert_not_called()
        assert ("done", "h", "download errored; left") in view.events
        assert set(engine._pending_store()) == {"h"}

    def test_wait_scope_all_includes_store_only_carried_over(self) -> None:
        # REGRESSION (complaint 4 - "exited right away"): a carried-over record
        # that is ONLY in the store (NOT a this-run grab) is still monitored. Here
        # there are no this-run grabs at all, yet the store record is driven.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = import_probe(
            ImportReadiness.RETRY, files_present=True,
        )
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="carried", added_at=_FRESH).to_json()],
            pending=[],  # nothing grabbed this run
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        strategy.import_completed.assert_called()
        assert ("done", "carried", "imported") in view.events
        assert engine._pending_store() == {}

    def test_import_exception_is_swallowed_and_record_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate and
        # abort _finalize_run's cache save; the record is left pending instead.
        strategy = mock.MagicMock()
        strategy.import_completed.side_effect = RuntimeError("boom")
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_ready_timeout=60, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        engine._run_monitor(now=clock.now, sleep=clock.sleep, view=view)  # must not raise

        assert set(engine._pending_store()) == {"h"}


class TestImportWaitModeProperty:
    """The engine exposes the run's RESOLVED wait mode (cli > config).

    The Sonarr strategy's seed-building gate reads this instead of the raw config
    so a CLI override that turns the feature on over an ``off`` config still
    builds seeds (otherwise the whole pass silently no-ops).
    """

    def test_reflects_resolved_mode(self) -> None:
        engine = make_bare_instance(SeaDexArr, _import_wait_mode=ImportWaitMode.HYBRID)

        assert engine.import_wait_mode is ImportWaitMode.HYBRID


def _finalize_engine(calls: list[str], *, qbit: object, mode: ImportWaitMode) -> SeaDexArr:
    """A bare engine whose summary / save each append a marker to ``calls``.

    The reporter and cache_store are MagicMocks (so ``side_effect`` is well-typed),
    so ``_finalize_run``'s ordering can be asserted without a live
    Sonarr/qBittorrent. ``_run_monitor`` is patched per-test (a recording stub).
    """

    reporter = mock.MagicMock()
    reporter.log_run_summary.side_effect = lambda *a, **k: calls.append("summary")
    cache_store = mock.MagicMock()
    cache_store.data = {}
    cache_store.save.side_effect = lambda *a, **k: calls.append("save")
    engine = make_bare_instance(
        SeaDexArr,
        qbit=qbit,
        logger=make_logger(),
        log_fmt=mock.MagicMock(),
        _config=make_config(
            import_wait_timeout=3600, import_ready_timeout=600, import_poll_interval=30,
        ),
        _active_strategy=mock.MagicMock(),
        _reporter=reporter,
        cache_store=cache_store,
        _import_wait_mode=mode,
        dry_run=False,
    )
    engine._ctx = RunContext(arr=Arr.SONARR)
    return engine


class TestFinalizeRunOrdering:
    """_finalize_run prints the summary BEFORE the blocking monitor runs."""

    def test_blocking_summary_precedes_monitor_then_saves_last(self) -> None:
        # The scoreboard must print before the monitor, and the save must be dead
        # last (so the store reflects the monitor's drops).
        calls: list[str] = []
        engine = _finalize_engine(
            calls, qbit=mock.MagicMock(), mode=ImportWaitMode.BLOCKING,
        )

        with mock.patch.object(
            engine, "_run_monitor", side_effect=lambda *a, **k: calls.append("monitor"),
        ):
            engine._finalize_run(Arr.SONARR)

        assert calls == ["summary", "monitor", "save"]

    def test_deferred_does_not_run_monitor(self) -> None:
        # Deferred reconciles pre-summary but NEVER runs the blocking monitor.
        calls: list[str] = []
        engine = _finalize_engine(
            calls, qbit=mock.MagicMock(), mode=ImportWaitMode.DEFERRED,
        )

        with mock.patch.object(
            engine, "_run_monitor", side_effect=lambda *a, **k: calls.append("monitor"),
        ):
            engine._finalize_run(Arr.SONARR)

        assert "monitor" not in calls
        assert calls == ["summary", "save"]

    def test_preview_skips_monitor_and_tally(self) -> None:
        # A preview (no client) short-circuits reconcile/tally/monitor.
        calls: list[str] = []
        engine = _finalize_engine(calls, qbit=None, mode=ImportWaitMode.BLOCKING)

        with mock.patch.object(
            engine, "_run_monitor", side_effect=lambda *a, **k: calls.append("monitor"),
        ):
            engine._finalize_run(Arr.SONARR)

        assert "monitor" not in calls


class TestDropPending:
    """_drop_pending removes a record from both the durable store and run list."""

    def test_removes_from_store_and_ctx(self) -> None:
        keep = pending_import(infohash="keep", added_at=_FRESH)
        drop = pending_import(infohash="drop", added_at=_FRESH)
        engine = make_orchestration_engine(
            qbit=None, strategy=mock.MagicMock(),
            store_records=[keep.to_json(), drop.to_json()],
            pending=[keep, drop],
        )

        engine._drop_pending("drop")

        assert set(engine._pending_store()) == {"keep"}
        assert engine._ctx.pending_imports == [keep]
