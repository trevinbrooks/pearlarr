# pyright: strict
# pyright: reportPrivateUsage=false
# These access the wait manager's + engine's private members (mgr._reporter,
# mgr._pending_store, mgr._ctx, engine._wait_manager, ...); strict re-flags that and
# the repo disables reportPrivateUsage for tests.
"""Unit tests for the completion wait/poll machinery (:class:`ImportWaitManager`).

These pin :meth:`ImportWaitManager.poll_torrent` (the single-shot state read)
and :meth:`ImportWaitManager.run_monitor` (the poll loop) against a scripted
``FakeQbit``. The clock and sleep are injected into the monitor loop so it never
actually waits - real foreground ``sleep`` is blocked in this env, so the fakes
are mandatory, not just a speed-up. The manager is built bare (``object.__new__``
via ``make_bare_instance``) so no live qBittorrent login or disk I/O happens. The
engine's ``_finalize_run`` orchestration (which drives the manager's passes) is
pinned via a real engine with an attached manager at the bottom of the file.

Every collaborator the manager drives - the strategy's import hooks, the snapshot
reporter, qBittorrent - is a small typed fake recording what a test asserts, so
the contracts are pinned by recorded state.
"""

from dataclasses import dataclass
from typing import Protocol, override

import qbittorrentapi

from seadexarr.modules.config import Arr
from seadexarr.modules.import_wait import ImportWaitManager
from seadexarr.modules.manual_import import (
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    ImportWaitMode,
    Outcome,
    PendingImport,
    PendingState,
    TorrentProbe,
    WaitOutcome,
)
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.torrents import AddOutcome
from seadexarr.modules.wait_view import Phase, TorrentView, WaitResult, WaitSnapshot, WaitView

from .builders import (
    CLIENT_SENTINEL,
    FakeCacheStore,
    FakeTorrents,
    import_probe,
    make_bare_instance,
    make_config,
    make_grab_pipeline,
    make_import_wait_manager,
    make_logger,
    one_release_dict,
    pending_import,
)
from .fakes import FakeStrategy


class FakeStateEnum:
    """Mimics qBittorrent's ``state_enum`` (the ``is_*`` booleans the poll reads)."""

    def __init__(self, *, is_complete: bool = False, is_errored: bool = False) -> None:
        self.is_complete = is_complete
        self.is_errored = is_errored


class FakeTorrent:
    """Mimics one qBittorrent torrent info row (the fields ``poll_torrent`` reads).

    The telemetry fields (``dlspeed`` / ``eta`` / ``completed`` / ``size``) default
    to None so the common monitor tests don't have to set them; ``poll_torrent``
    reads them via ``getattr`` and sanitizes None to None.
    """

    def __init__(
        self,
        *,
        is_complete: bool = False,
        is_errored: bool = False,
        progress: float = 0.0,
        content_path: str | None = None,
        dlspeed: int | None = None,
        eta: int | None = None,
        completed: int | None = None,
        size: int | None = None,
    ) -> None:
        self.state_enum = FakeStateEnum(is_complete=is_complete, is_errored=is_errored)
        self.progress = progress
        self.content_path = content_path
        self.dlspeed = dlspeed
        self.eta = eta
        self.completed = completed
        self.size = size


class FakeQbit:
    """A scriptable qBittorrent client returning queued ``torrents_info`` results.

    Each call to ``torrents_info`` pops the next scripted result; once the script
    is exhausted the last result repeats (so a steady "downloading" state can be
    polled indefinitely). A scripted value may be an exception instance, which is
    raised to exercise the transient-error path.
    """

    def __init__(self, results: list[list[FakeTorrent] | Exception]) -> None:
        self._results = results
        self.calls = 0

    def torrents_info(self, *, torrent_hashes: str) -> list[FakeTorrent]:
        self.calls += 1
        result = self._results[min(self.calls - 1, len(self._results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def make_wait_manager(qbit: FakeQbit) -> ImportWaitManager:
    """A bare ``ImportWaitManager`` wired only with the ``qbit`` the poll reads."""

    return make_bare_instance(ImportWaitManager, qbit=qbit)


class TestPollTorrent:
    """poll_torrent maps a single qBittorrent read to a sanitized TorrentProbe."""

    def test_missing_on_empty_list(self) -> None:
        mgr = make_wait_manager(FakeQbit([[]]))

        assert mgr.poll_torrent("h") == TorrentProbe(WaitOutcome.MISSING, None, 0.0)

    def test_errored(self) -> None:
        mgr = make_wait_manager(FakeQbit([[FakeTorrent(is_errored=True)]]))

        assert mgr.poll_torrent("h") == TorrentProbe(WaitOutcome.ERRORED, None, 0.0)

    def test_complete_carries_content_path(self) -> None:
        torrent = FakeTorrent(is_complete=True, content_path="/data/show")
        mgr = make_wait_manager(FakeQbit([[torrent]]))

        assert mgr.poll_torrent("h") == TorrentProbe(
            WaitOutcome.COMPLETE,
            "/data/show",
            0.0,
        )

    def test_complete_on_full_progress_without_flag(self) -> None:
        # progress == 1.0 counts as complete even if the state flag is unset.
        torrent = FakeTorrent(progress=1.0, content_path="/data/movie")
        mgr = make_wait_manager(FakeQbit([[torrent]]))

        assert mgr.poll_torrent("h") == TorrentProbe(
            WaitOutcome.COMPLETE,
            "/data/movie",
            1.0,
        )

    def test_none_while_downloading_carries_progress(self) -> None:
        mgr = make_wait_manager(FakeQbit([[FakeTorrent(progress=0.5)]]))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.5)

    def test_none_on_transient_api_error(self) -> None:
        # A dropped connection / re-auth in flight is "still waiting", not terminal.
        mgr = make_wait_manager(FakeQbit([qbittorrentapi.APIConnectionError("boom")]))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.0)

    def test_none_when_no_client(self) -> None:
        mgr = make_bare_instance(ImportWaitManager, qbit=None)

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.0)

    def test_carries_live_download_telemetry(self) -> None:
        torrent = FakeTorrent(
            progress=0.64,
            dlspeed=3_200_000,
            eta=130,
            completed=1_800_000_000,
            size=2_900_000_000,
        )
        mgr = make_wait_manager(FakeQbit([[torrent]]))

        assert mgr.poll_torrent("h") == TorrentProbe(
            None,
            None,
            0.64,
            3_200_000,
            130,
            1_800_000_000,
            2_900_000_000,
        )

    def test_sanitizes_junk_telemetry(self) -> None:
        # qB's "∞" eta sentinel, an idle (0) speed and a zero size all sanitize to
        # None so the wait view never renders a nonsense countdown / negative bar.
        torrent = FakeTorrent(progress=0.5, dlspeed=0, eta=8_640_000, completed=0, size=0)
        mgr = make_wait_manager(FakeQbit([[torrent]]))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.5)


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


# A timestamp far enough in the past/future that the TTL verdict is fixed no
# matter when the suite runs (cutoff = now - import_pending_max_age_days).
_FRESH = "2999-01-01 00:00:00"
_EXPIRED = "2000-01-01 00:00:00"


@dataclass(frozen=True)
class _ImportCall:
    """One recorded ``import_completed`` call: its record/path + force/deadline flags."""

    pending: PendingImport
    content_path: str
    force: bool
    at_deadline: bool


class _RecordingStrategy(FakeStrategy):
    """A :class:`FakeStrategy` that records + scripts the two import hooks the manager drives.

    ``import_completed`` records each call's force/at_deadline flags (asserted on
    ``import_calls``) and dispenses a scripted :class:`ImportProbe` - a single
    ``completed`` repeated, a ``completed_sequence`` advanced per call (clamped to its
    last), or a ``completed_error`` raised (the swallowed-import path).
    ``import_progress`` likewise records (``progress_calls``) and dispenses an
    :class:`ImportProgress`, defaulting to an indeterminate zero - the Tier-2
    fast-poll no-op the heavy-poll tests rely on.
    """

    def __init__(
        self,
        *,
        completed: ImportProbe | None = None,
        completed_sequence: list[ImportProbe] | None = None,
        completed_error: Exception | None = None,
        progress: ImportProgress | None = None,
        progress_sequence: list[ImportProgress] | None = None,
    ) -> None:
        super().__init__(items=[], anilist_ids={})
        self._completed = completed
        self._completed_sequence = completed_sequence
        self._completed_error = completed_error
        self._completed_index = 0
        self._progress = progress
        self._progress_sequence = progress_sequence
        self._progress_index = 0
        self.import_calls: list[_ImportCall] = []
        self.progress_calls: list[PendingImport] = []

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        self.import_calls.append(_ImportCall(pending, content_path, force, at_deadline))
        if self._completed_error is not None:
            raise self._completed_error
        if self._completed_sequence is not None:
            idx = min(self._completed_index, len(self._completed_sequence) - 1)
            self._completed_index += 1
            return self._completed_sequence[idx]
        if self._completed is not None:
            return self._completed
        return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        self.progress_calls.append(pending)
        if self._progress_sequence is not None:
            idx = min(self._progress_index, len(self._progress_sequence) - 1)
            self._progress_index += 1
            return self._progress_sequence[idx]
        if self._progress is not None:
            return self._progress
        return ImportProgress(0, 0, determinate=False)


@dataclass(frozen=True)
class _SnapshotCall:
    """One recorded ``log_pending_snapshot`` call's reported fields."""

    state: PendingState
    title: str
    coverage: str | None
    url: str | None


class _RecordingReporter:
    """Records ``log_pending_snapshot`` calls, so the inline-snapshot / no-double-report
    contracts are asserted on recorded state."""

    def __init__(self) -> None:
        self.snapshot_calls: list[_SnapshotCall] = []

    def log_pending_snapshot(
        self,
        ctx: RunContext,
        state: PendingState,
        title: str,
        coverage: str | None,
        url: str | None,
    ) -> bool:
        self.snapshot_calls.append(_SnapshotCall(state, title, coverage, url))
        return True


class _PollableQbit(Protocol):
    """The single qBittorrent read ``poll_torrent`` makes - the seam the scriptable
    ``FakeQbit`` and the per-test multi-torrent fakes satisfy structurally."""

    def torrents_info(self, *, torrent_hashes: str) -> list[FakeTorrent]: ...


def make_orchestration_manager(
    *,
    qbit: _PollableQbit | None,
    strategy: _RecordingStrategy,
    store_records: list[PendingImport] | None = None,
    pending: list[PendingImport] | None = None,
    reporter: _RecordingReporter | None = None,
    **config_overrides: object,
) -> ImportWaitManager:
    """A bare ``ImportWaitManager`` wired for the pending-import orchestration paths.

    Seeds the durable per-arr store (via the manager's own ``cache_store``) and the
    in-memory ``_ctx.pending_imports`` list so ``prune_expired_pending``,
    ``snapshot_pending_for_series``, ``reconcile_remaining`` and ``run_monitor`` can
    be driven without a live Sonarr/qBittorrent. The strategy is a recording
    ``_RecordingStrategy`` (its import hooks scripted per test) and the reporter a
    recording ``_RecordingReporter`` (a test that asserts on the snapshot reporter
    passes in its own ``reporter`` to read it back).
    """

    mgr = make_bare_instance(
        ImportWaitManager,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(**config_overrides),
        _active_strategy=strategy,
        _reporter=reporter or _RecordingReporter(),
        cache_store=FakeCacheStore(),
    )
    mgr._ctx = RunContext(arr=Arr.SONARR, pending_imports=list(pending or []))
    for record in store_records or []:
        mgr.cache_store.put_pending(Arr.SONARR, record.infohash, record.to_json())
    return mgr


class TestPruneExpiredPending:
    """prune_expired_pending drops aged-out / unparseable records only."""

    def test_drops_expired_and_unparseable_keeps_fresh(self) -> None:
        records = [
            pending_import(infohash="fresh", added_at=_FRESH),
            pending_import(infohash="old", added_at=_EXPIRED),
            pending_import(infohash="bad", added_at="not-a-timestamp"),
        ]
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=records,
        )

        mgr.prune_expired_pending()

        assert set(mgr._pending_store()) == {"fresh"}


class RecordingWaitView(WaitView):
    """Records every snapshot the manager pushes, for assertion.

    Replaces the old call-tuple FakeWaitView: the view is now a pure function of
    the pushed :class:`WaitSnapshot`, so the tests assert on the recorded snapshot
    state (each torrent's phase / outcome) rather than imperative method calls.
    """

    def __init__(self) -> None:
        self.snapshots: list[WaitSnapshot] = []
        self.closed = False

    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        self.snapshots.append(snapshot)

    @override
    def close(self) -> None:
        self.closed = True

    def final(self, key: str) -> TorrentView:
        """The torrent's row in the last recorded snapshot."""

        return next(t for t in self.snapshots[-1].torrents if t.key == key)

    def saw(self, key: str, phase: Phase) -> bool:
        """Whether any recorded snapshot showed ``key`` in ``phase``."""

        return any(t.key == key and t.phase is phase for snap in self.snapshots for t in snap.torrents)


class TestSnapshotPendingForSeries:
    """snapshot_pending_for_series reports CARRIED-OVER records inline, no double-report."""

    def test_carried_over_imported_drops_and_counts(self) -> None:
        # A prior run's download finished: COMPLETE + verified files -> the record
        # is reported imported, dropped, and stats.imported bumped.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        reporter = _RecordingReporter()
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert mgr._pending_store() == {}
        assert mgr._ctx.stats.imported == 1
        assert mgr._ctx.pending_states["h"] is PendingState.IMPORTED
        # The record is reported inline with its reconciled state + title (the gap a
        # bare `snapshot_calls != []` check left open: wrong state/title slid through).
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingState.IMPORTED
        assert reporter.snapshot_calls[0].title == "Show"
        # Forced (CDH-off safe) but NOT at the deadline (no loud warning).
        assert strategy.import_calls[-1].force is True
        assert strategy.import_calls[-1].at_deadline is False

    def test_carried_over_downloading_is_queued_and_kept(self) -> None:
        # Still downloading -> queued, record kept, no import attempt.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit([[FakeTorrent(progress=0.5)]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert set(mgr._pending_store()) == {"h"}
        assert mgr._ctx.pending_states["h"] is PendingState.QUEUED
        assert mgr._ctx.stats.imported == 0
        # The carried-over record is still reported inline, with the QUEUED state.
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingState.QUEUED
        assert reporter.snapshot_calls[0].title == "Show"

    def test_this_run_grab_is_skipped_no_double_report(self) -> None:
        # REGRESSION (double-report): a torrent grabbed THIS run lives in
        # _ctx.pending_imports AND the store. The snapshot must skip it entirely -
        # no poll, no row, no counter, no state - so it's only ever `added`.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        this_run = pending_import(infohash="h", series_id=7, added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[this_run],
            pending=[this_run],
        )

        mgr.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert reporter.snapshot_calls == []
        assert mgr._ctx.pending_states == {}
        assert mgr._ctx.stats.imported == 0
        assert set(mgr._pending_store()) == {"h"}

    def test_other_series_record_is_not_touched(self) -> None:
        # The snapshot is series-scoped: a record for a different series is left
        # alone (the deferred reconcile / monitor handles it later).
        strategy = _RecordingStrategy()
        qbit = FakeQbit([[FakeTorrent(progress=0.5)]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending_import(infohash="other", series_id=99, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert "other" not in mgr._ctx.pending_states
        assert set(mgr._pending_store()) == {"other"}


class TestReconcileRemaining:
    """reconcile_remaining force-polls carried-over records not snapshotted this run."""

    def test_imports_ready_record_not_yet_snapshotted(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH)],
        )

        mgr.reconcile_remaining()

        assert mgr._pending_store() == {}
        assert mgr._ctx.stats.imported == 1
        assert strategy.import_calls[-1].force is True
        assert strategy.import_calls[-1].at_deadline is False

    def test_skips_already_snapshotted(self) -> None:
        # A record the inline snapshot already touched must not be re-polled.
        strategy = _RecordingStrategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]]),
            strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH)],
        )
        mgr._ctx.pending_states["h"] = PendingState.QUEUED

        mgr.reconcile_remaining()

        assert strategy.import_calls == []

    def test_skips_this_run_grabs(self) -> None:
        strategy = _RecordingStrategy()
        this_run = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]]),
            strategy=strategy,
            store_records=[this_run],
            pending=[this_run],
        )

        mgr.reconcile_remaining()

        assert strategy.import_calls == []


class TestTallyCarriedOverIntoStats:
    """tally_carried_over_into_stats counts each still-pending record once."""

    def test_counts_known_states_and_defaults_to_queued(self) -> None:
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[
                pending_import(infohash="q", added_at=_FRESH),
                pending_import(infohash="i", added_at=_FRESH),
                pending_import(infohash="untouched", added_at=_FRESH),
            ],
        )
        mgr._ctx.pending_states = {
            "q": PendingState.QUEUED,
            "i": PendingState.IMPORTING,
        }

        mgr.tally_carried_over_into_stats()

        assert mgr._ctx.stats.queued == 2  # explicit q + defaulted untouched
        assert mgr._ctx.stats.importing == 1

    def test_excludes_this_run_grabs(self) -> None:
        this_run = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[this_run],
            pending=[this_run],
        )

        mgr.tally_carried_over_into_stats()

        assert mgr._ctx.stats.queued == 0
        assert mgr._ctx.stats.importing == 0


class TestRunMonitor:
    """run_monitor: interleaved, copy-aware wait+import over ALL pending."""

    def test_interleaved_fast_and_slow(self) -> None:
        # Two torrents: "fast" completes + imports first cycle (files present);
        # "slow" is still downloading, then completes + imports a later cycle. Both
        # advance each cycle (interleaved), so the fast one isn't stuck behind slow.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )

        class TwoTorrentQbit:
            def __init__(self) -> None:
                self.calls: dict[str, int] = {}

            def torrents_info(self, *, torrent_hashes: str) -> list[FakeTorrent]:
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
        mgr = make_orchestration_manager(
            qbit=TwoTorrentQbit(),
            strategy=strategy,
            store_records=[fast, slow],
            pending=[fast, slow],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        # Both ultimately imported and dropped.
        assert mgr._pending_store() == {}
        assert view.final("fast").outcome is Outcome.IMPORTED
        assert view.final("slow").outcome is Outcome.IMPORTED
        # Each torrent's OWN content_path reached import_completed - a run_monitor bug
        # forwarding the wrong torrent's path would still import, so pin the pairing.
        by_hash = {c.pending.infohash: c.content_path for c in strategy.import_calls}
        assert by_hash["fast"] == "/fast"
        assert by_hash["slow"] == "/slow"
        # slow showed a downloading heartbeat (fraction 0.5) before it completed.
        assert any(
            t.key == "slow" and t.phase is Phase.DOWNLOADING and t.fraction == 0.5
            for snap in view.snapshots
            for t in snap.torrents
        )

    def test_imported_only_when_files_present_two_cycles(self) -> None:
        # The copy is async: cycle 1 issues the command (RETRY + command_issued,
        # files NOT present) -> reads `importing`; cycle 2 verifies files present
        # -> `imported`. imported is gated on verified files, never command accept.
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
                import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert len(strategy.import_calls) == 2
        assert view.saw("h", Phase.IMPORTING)  # cycle 1, copy in flight
        assert view.final("h").outcome is Outcome.IMPORTED  # cycle 2, files landed
        assert mgr._pending_store() == {}

    def test_tier2_fast_poll_fills_bar_and_promotes_before_next_heavy_poll(self) -> None:
        # Between heavy polls the cheap progress poll fills the "files inserted" bar
        # as files land and promotes the row to IMPORTED the instant all are present
        # - no second (heavy) import_completed poll, no RefreshMonitoredDownloads.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
            progress_sequence=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),
                ImportProgress(3, 3, determinate=True),  # all present -> promote
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
            progress_poll_interval=5,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=5)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        # Only ONE heavy poll; the fast poll did the rest.
        assert len(strategy.import_calls) == 1
        assert len(strategy.progress_calls) == 3
        # The bar advanced (2/3 seen) before the row finished.
        assert any(
            t.key == "h" and t.import_done == 2 and t.import_total == 3
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_store() == {}

    def test_tier2_disabled_skips_the_fast_poll(self) -> None:
        # progress_poll_interval=0 -> no cheap poll at all; the heavy poll alone
        # drives completion (the bar simply steps once per poll).
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
                import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
            progress_poll_interval=0,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.progress_calls == []
        assert len(strategy.import_calls) == 2
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_importing_at_deadline_left_without_warning(self) -> None:
        # The copy never lands within import_ready_timeout: the final attempt
        # (at_deadline) leaves it pending with "still importing; left" - no drop.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=60,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_store()) == {"h"}  # left, not dropped
        # The final in-bound poll forces AND flags the deadline, for THIS torrent's path.
        last = strategy.import_calls[-1]
        assert last.content_path == "/d"
        assert last.force is True
        assert last.at_deadline is True

    def test_missing_drops_record(self) -> None:
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit([[]]),
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.import_calls == []
        assert view.final("h").outcome is Outcome.MISSING
        assert mgr._pending_store() == {}

    def test_errored_leaves_record(self) -> None:
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit([[FakeTorrent(is_errored=True)]]),
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.import_calls == []
        assert view.final("h").outcome is Outcome.DOWNLOAD_ERRORED
        assert set(mgr._pending_store()) == {"h"}

    def test_wait_scope_all_includes_store_only_carried_over(self) -> None:
        # REGRESSION (complaint 4 - "exited right away"): a carried-over record
        # that is ONLY in the store (NOT a this-run grab) is still monitored. Here
        # there are no this-run grabs at all, yet the store record is driven.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending_import(infohash="carried", added_at=_FRESH)],
            pending=[],  # nothing grabbed this run
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.import_calls  # the store-only record was driven
        assert view.final("carried").outcome is Outcome.IMPORTED
        assert mgr._pending_store() == {}

    def test_keyboard_interrupt_breaks_and_leaves_records(self) -> None:
        # Ctrl-C during the poll nap must break the loop (not propagate), so the
        # caller's finally still restores the terminal + saves the cache; the
        # in-flight record is left pending and a WaitResult is still returned.
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit([[FakeTorrent(progress=0.3)]]),
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()

        def interrupt(_seconds: float) -> None:
            raise KeyboardInterrupt

        result = mgr.run_monitor(  # must not raise
            now=lambda: 0.0,
            sleep=interrupt,
            view=view,
        )

        assert result is not None and result.waited == 0
        assert set(mgr._pending_store()) == {"h"}  # left pending for next run
        assert view.saw("h", Phase.DOWNLOADING)

    def test_import_exception_is_swallowed_and_record_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate and
        # abort _finalize_run's cache save; the record is left pending instead.
        strategy = _RecordingStrategy(completed_error=RuntimeError("boom"))
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit([[FakeTorrent(is_complete=True, content_path="/d")]])
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=60,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)  # must not raise

        assert view.final("h").outcome is Outcome.NOTHING_TO_IMPORT
        assert set(mgr._pending_store()) == {"h"}


class TestImportWaitModeProperty:
    """The engine exposes the run's RESOLVED wait mode (cli > config).

    The Sonarr strategy's seed-building gate reads this instead of the raw config
    so a CLI override that turns the feature on over an ``off`` config still
    builds seeds (otherwise the whole pass silently no-ops).
    """

    def test_reflects_resolved_mode(self) -> None:
        engine = make_bare_instance(
            SeaDexArr,
            _ctx=RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.HYBRID),
        )

        assert engine.import_wait_mode is ImportWaitMode.HYBRID


def _attach_wait_manager(engine: SeaDexArr) -> None:
    """Attach an ``ImportWaitManager`` sharing the engine's run state.

    The wait/poll machinery lives on the manager now, so a finalize/snapshot test
    on a bare engine must wire one bound to the SAME ``_ctx`` / ``cache_store`` /
    client / strategy the engine holds - exactly as ``__init__`` + ``begin_run`` do.
    """

    engine._wait_manager = make_import_wait_manager(
        _config=engine._config,
        cache_store=engine.cache_store,
        _reporter=engine._reporter,
        logger=engine.logger,
        qbit=engine.qbit,
        _ctx=engine._ctx,
        _active_strategy=engine._active_strategy,
    )


class _FinalizeReporter:
    """Records the engine's end-of-run summary as a ``"summary"`` ordering marker."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def log_run_summary(
        self,
        ctx: RunContext,
        *,
        is_preview: bool,
        has_client: bool,
    ) -> bool:
        self._calls.append("summary")
        return True


class _RecordingCacheStore(FakeCacheStore):
    """A :class:`FakeCacheStore` whose ``save`` appends a ``"save"`` ordering marker."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    @override
    def save(self, *, preview: bool) -> None:
        self._calls.append("save")


class _FinalizeWaitManager:
    """A stand-in wait manager for the finalize-ordering tests.

    Its reconcile/tally passes are silent no-ops (mirroring the real ones reading
    an empty store and appending nothing), and ``run_monitor`` appends the
    ``"monitor"`` ordering marker. The real ``run_monitor`` returns early on the
    empty working set these tests build - recording nothing - so a recording
    stand-in is what makes the monitor step observable.
    """

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def reconcile_remaining(self) -> None:
        pass

    def tally_carried_over_into_stats(self) -> None:
        pass

    def run_monitor(self) -> WaitResult | None:
        self._calls.append("monitor")
        return None


def _finalize_engine(calls: list[str], *, qbit: object, mode: ImportWaitMode) -> SeaDexArr:
    """A bare engine whose summary / save / monitor each append a marker to ``calls``.

    The reporter, cache store, and wait manager are typed recorders (a
    ``_FinalizeReporter`` / ``_RecordingCacheStore`` / ``_FinalizeWaitManager``), so
    ``_finalize_run``'s ordering is asserted on the recorded ``calls`` list without a
    live Sonarr/qBittorrent. The fake wait manager's reconcile/tally are silent
    no-ops and its ``run_monitor`` records the ``"monitor"`` marker.
    """

    engine = make_bare_instance(
        SeaDexArr,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        ),
        _reporter=_FinalizeReporter(calls),
        cache_store=_RecordingCacheStore(calls),
        _wait_manager=_FinalizeWaitManager(calls),
    )
    engine._ctx = RunContext(arr=Arr.SONARR, import_wait_mode=mode)
    return engine


class TestFinalizeRunOrdering:
    """_finalize_run prints the summary BEFORE the blocking monitor runs."""

    def test_blocking_summary_precedes_monitor_then_saves_last(self) -> None:
        # The scoreboard must print before the monitor, and the save must be dead
        # last (so the store reflects the monitor's drops).
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
        )

        engine._finalize_run()

        assert calls == ["summary", "monitor", "save"]

    def test_deferred_does_not_run_monitor(self) -> None:
        # Deferred reconciles pre-summary but NEVER runs the blocking monitor.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.DEFERRED,
        )

        engine._finalize_run()

        assert "monitor" not in calls
        assert calls == ["summary", "save"]

    def test_preview_skips_monitor_and_tally(self) -> None:
        # A preview (no client) short-circuits reconcile/tally/monitor.
        calls: list[str] = []
        engine = _finalize_engine(calls, qbit=None, mode=ImportWaitMode.BLOCKING)

        engine._finalize_run()

        assert "monitor" not in calls
        # Even short-circuited, the summary still prints and the save still runs last.
        assert calls == ["summary", "save"]


class TestDropPending:
    """_drop_pending removes a record from both the durable store and run list."""

    def test_removes_from_store_and_ctx(self) -> None:
        keep = pending_import(infohash="keep", added_at=_FRESH)
        drop = pending_import(infohash="drop", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[keep, drop],
            pending=[keep, drop],
        )

        mgr.drop_pending("drop")

        assert set(mgr._pending_store()) == {"keep"}
        assert mgr._ctx.pending_imports == [keep]


def make_add_engine(
    *,
    torrents: FakeTorrents,
    strategy: _RecordingStrategy,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config_overrides: object,
) -> SeaDexArr:
    """A bare ``SeaDexArr`` + an attached ``GrabPipeline`` + ``ImportWaitManager``.

    The produce side lives on :class:`GrabPipeline` and the consume side on
    :class:`ImportWaitManager` now, so the engine gets both wired to the SAME
    ``_ctx`` / ``cache_store`` / client it holds - an add through
    ``engine._grab_pipeline`` registers into exactly the state the manager's
    consume passes (``snapshot_pending_for_series`` / ``_monitor_working_set``)
    read back. ``_active_strategy`` is the test's recording ``_RecordingStrategy``
    and ``_reporter`` a recording ``_RecordingReporter`` so the snapshot can be
    driven afterwards and asserted on recorded state.
    """

    engine = make_bare_instance(
        SeaDexArr,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(**config_overrides),
        _torrents=torrents,
        _active_strategy=strategy,
        _reporter=_RecordingReporter(),
        cache_store=FakeCacheStore(),
    )
    engine._ctx = RunContext(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode)
    engine._grab_pipeline = make_grab_pipeline(
        _config=engine._config,
        _torrents=torrents,
        cache_store=engine.cache_store,
        qbit=qbit,
        _ctx=engine._ctx,
    )
    _attach_wait_manager(engine)
    return engine


class TestRegisteredGrabSurvivesSnapshot:
    """A this-run grab registered by the pipeline is owned by the monitor, not re-polled."""

    def test_registered_already_added_survives_snapshot(self) -> None:
        # Integration: an ALREADY_ADDED registered THIS run is a this-run grab, so
        # the per-series snapshot skips it (no re-poll, no drop) and the end-of-run
        # monitor owns it via the working set.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show")})
        strategy = _RecordingStrategy()
        engine = make_add_engine(torrents=torrents, strategy=strategy)
        seeds = {"h1": pending_import(infohash="h1", series_id=7, added_at=_FRESH)}

        engine._grab_pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )
        engine._wait_manager.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert set(engine._wait_manager._pending_store()) == {"h1"}
        assert [p.infohash for p in engine._wait_manager._monitor_working_set()] == ["h1"]
