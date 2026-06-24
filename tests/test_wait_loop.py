"""Unit tests for the engine's qBittorrent completion wait/poll machinery.

These pin :meth:`SeaDexArr._poll_torrent` (the single-shot state read) and
:meth:`SeaDexArr._wait_for_completion` (the poll loop) against a scripted
``FakeQbit``. The clock and sleep are injected into the wait loop so it never
actually waits - real foreground ``sleep`` is blocked in this env, so the fakes
are mandatory, not just a speed-up. The engine is built bare (``object.__new__``
via ``make_bare_instance``) so no live qBittorrent login or disk I/O happens.
"""

import types
from unittest import mock

import qbittorrentapi

from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import ImportReadiness, ImportWaitMode, WaitOutcome
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr

from .builders import make_bare_instance, make_config, make_logger, pending_import


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
    ``_reconcile_pending_imports`` and ``_run_blocking_imports`` can be driven
    without a live Sonarr/qBittorrent.
    """

    engine = make_bare_instance(
        SeaDexArr,
        qbit=qbit,
        logger=make_logger(),
        log_fmt=mock.MagicMock(),
        _config=make_config(**config_overrides),
        _active_strategy=strategy,
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


class TestReconcilePendingImports:
    """_reconcile_pending_imports polls once and imports the ready records."""

    def test_complete_and_verified_drops_record(self) -> None:
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.IMPORTED
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_pending_imports()

        strategy.import_completed.assert_called_once()
        assert engine._pending_store() == {}

    def test_complete_but_not_ready_leaves_record(self) -> None:
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.RETRY
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_pending_imports()

        assert set(engine._pending_store()) == {"h"}

    def test_complete_forces_step_in(self) -> None:
        # Reconcile runs a prior run's download, so Sonarr has had a full cycle: a
        # still-absent target means it won't import on its own -> force our import.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.IMPORTED
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_pending_imports()

        assert strategy.import_completed.call_args.kwargs.get("force") is True

    def test_missing_drops_record(self) -> None:
        strategy = mock.MagicMock()
        engine = make_orchestration_engine(
            qbit=FakeQbit([[]]), strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_pending_imports()

        strategy.import_completed.assert_not_called()
        assert engine._pending_store() == {}

    def test_downloading_leaves_record(self) -> None:
        strategy = mock.MagicMock()
        engine = make_orchestration_engine(
            qbit=FakeQbit([[FakeTorrent(progress=0.5)]]), strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH).to_json()],
        )

        engine._reconcile_pending_imports()

        strategy.import_completed.assert_not_called()
        assert set(engine._pending_store()) == {"h"}


class TestRunBlockingImports:
    """_run_blocking_imports waits on this run's records and imports/leaves them."""

    def test_complete_and_verified_drops_from_store_and_ctx(self) -> None:
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.IMPORTED
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
        )

        engine._run_blocking_imports()

        assert engine._pending_store() == {}
        assert engine._ctx.pending_imports == []

    def test_retry_then_imported_drops_record(self) -> None:
        # Phase B keeps polling Sonarr while it's not ready (RETRY), then drops the
        # record once the import lands (IMPORTED). The injected clock/sleep mean no
        # real waiting; FakeQbit reports the download already complete.
        strategy = mock.MagicMock()
        strategy.import_completed.side_effect = [
            ImportReadiness.RETRY,
            ImportReadiness.IMPORTED,
        ]
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_ready_timeout=600, import_poll_interval=30,
        )
        clock = FakeClock(step=30)

        engine._wait_and_import_one(pending, now=clock.now, sleep=clock.sleep)

        assert strategy.import_completed.call_count == 2
        assert engine._pending_store() == {}
        assert engine._ctx.pending_imports == []

    def test_forces_import_at_deadline(self) -> None:
        # Sonarr defers (RETRY) until the readiness deadline, where ONE forced poll
        # drives our own import - so a download Sonarr won't import (CDH off) still
        # imports rather than waiting forever.
        strategy = mock.MagicMock()
        strategy.import_completed.side_effect = [
            ImportReadiness.RETRY,
            ImportReadiness.RETRY,
            ImportReadiness.IMPORTED,
        ]
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_ready_timeout=60, import_poll_interval=30,
        )
        clock = FakeClock(step=30)

        engine._wait_and_import_one(pending, now=clock.now, sleep=clock.sleep)

        calls = strategy.import_completed.call_args_list
        assert calls[0].kwargs.get("force") is False
        assert calls[-1].kwargs.get("force") is True
        assert engine._pending_store() == {}

    def test_retry_until_ready_timeout_leaves_record(self) -> None:
        # Sonarr never becomes ready (always RETRY); once the readiness deadline
        # passes the record is left pending for a later run, not dropped.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.RETRY
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_ready_timeout=60, import_poll_interval=30,
        )
        clock = FakeClock(step=30)

        engine._wait_and_import_one(pending, now=clock.now, sleep=clock.sleep)

        assert set(engine._pending_store()) == {"h"}
        assert engine._ctx.pending_imports == [pending]

    def test_errored_leaves_record(self) -> None:
        strategy = mock.MagicMock()
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_errored=True)]])
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
        )

        engine._run_blocking_imports()

        strategy.import_completed.assert_not_called()
        assert set(engine._pending_store()) == {"h"}
        assert engine._ctx.pending_imports == [pending]

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
        )

        engine._run_blocking_imports()  # must not raise

        assert set(engine._pending_store()) == {"h"}
        assert engine._ctx.pending_imports == [pending]


class FakeWaitView:
    """Records the WaitView calls the engine makes, for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def start(self, torrents: list[tuple[str, str]]) -> None:
        self.events.append(("start", torrents))

    def download(self, key: str, pct: float, elapsed: float, timeout: float) -> None:
        self.events.append(("download", key, pct))

    def phase_sonarr(self, key: str, elapsed: float, timeout: float) -> None:
        self.events.append(("phase_sonarr", key))

    def done(self, key: str, outcome: str) -> None:
        self.events.append(("done", key, outcome))

    def close(self) -> None:
        self.events.append(("close",))


class TestWaitViewRouting:
    """The engine routes download/import progress + outcomes through the view."""

    def test_download_then_import_drives_view(self) -> None:
        # One downloading poll, then complete; Sonarr imports on the first poll.
        strategy = mock.MagicMock()
        strategy.import_completed.return_value = ImportReadiness.IMPORTED
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            [
                [FakeTorrent(progress=0.4)],
                [FakeTorrent(is_complete=True, content_path="/d")],
            ],
        )
        engine = make_orchestration_engine(
            qbit=qbit, strategy=strategy,
            store_records=[pending.to_json()], pending=[pending],
            import_wait_timeout=3600, import_poll_interval=30,
        )
        view = FakeWaitView()
        clock = FakeClock(step=30)

        outcome = engine._wait_and_import_one(
            pending, view=view, now=clock.now, sleep=clock.sleep,
        )

        assert outcome == "imported"
        kinds = [e[0] for e in view.events]
        assert "download" in kinds  # the downloading poll heartbeat
        assert ("done", "h", "imported") in view.events
    """The engine exposes the run's RESOLVED wait mode (cli > config).

    The Sonarr strategy's seed-building gate reads this instead of the raw config
    so a CLI override that turns the feature on over an ``off`` config still
    builds seeds (otherwise the whole pass silently no-ops).
    """

    def test_reflects_resolved_mode(self) -> None:
        engine = make_bare_instance(SeaDexArr, _import_wait_mode=ImportWaitMode.HYBRID)

        assert engine.import_wait_mode is ImportWaitMode.HYBRID


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
