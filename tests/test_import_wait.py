# pyright: strict
# pyright: reportPrivateUsage=false
# These access the wait manager's + engine's private members (mgr._reporter,
# mgr._pending_records, mgr._ctx, engine._wait_manager, ...). Strict re-flags that,
# and the repo disables reportPrivateUsage for tests.
"""Unit tests for the completion wait/poll machinery (`ImportWaitManager`).

These pin `ImportWaitManager.poll_torrent` (the single-shot state read)
and `ImportWaitManager.run_monitor` (the poll loop) against a scripted
`FakeQbit`. The clock and sleep are injected into the monitor loop so it never
actually waits - real foreground `sleep` is blocked in this env, so the fakes
are mandatory, not just a speed-up. The manager is built bare (`object.__new__`
via `make_bare_instance`) so no live qBittorrent login or disk I/O happens. The
engine's `_finalize_run` orchestration (which drives the manager's passes) is
pinned via a real engine with an attached manager at the bottom of the file.

Every collaborator the manager drives - the strategy's import hooks, the snapshot
reporter, qBittorrent - is a small typed fake recording what a test asserts, so
the contracts are pinned by recorded state.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import override

import pytest
import qbittorrentapi

from pearlarr.cache import UPDATED_AT_STR_FORMAT
from pearlarr.config import Arr
from pearlarr.grab_pipeline import GrabPipeline
from pearlarr.import_wait import ImportWaitManager, MonitorPass
from pearlarr.log import LOG_NAME
from pearlarr.manual_import import (
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    ImportWaitMode,
    Outcome,
    PendingImport,
    PendingKey,
    PendingState,
    TorrentProbe,
    TorrentTelemetry,
    WaitOutcome,
)
from pearlarr.output import SPARK_SAMPLES, Diagnostic, Phase, Severity, TorrentView, WaitSnapshot
from pearlarr.reporter import RunContext
from pearlarr.run_loop import RunLoop
from pearlarr.seadex_types import HistoryRecord
from pearlarr.torrents import AddOutcome
from pearlarr.wait_view import WaitResult, WaitView

from .builders import (
    CLIENT_SENTINEL,
    PENDING_AL_ID,
    SEP,
    FakeCacheStore,
    FakeTorrents,
    import_probe,
    make_bare_instance,
    make_config,
    make_grab_pipeline,
    make_import_wait_manager,
    make_logger,
    make_radarr_sync,
    make_services,
    one_release_dict,
    pending_import,
)
from .fakes import CaptureHandler, FakeRadarrClient, FakeStrategy, install_recording_hub


def pk(infohash: str, al_id: int = PENDING_AL_ID) -> PendingKey:
    """The composite key of a `pending_import`-built record (builder-default al_id)."""

    return PendingKey(infohash, al_id)


def rk(infohash: str, al_id: int = PENDING_AL_ID) -> str:
    """The snapshot row key (`TorrentView.key`) of a `pending_import`-built record."""

    return pk(infohash, al_id).row_key


class FakeStateEnum:
    """Mimics qBittorrent's `state_enum` (the `is_*` booleans the poll reads)."""

    def __init__(self, *, is_complete: bool = False, is_errored: bool = False) -> None:
        self.is_complete = is_complete
        self.is_errored = is_errored


class FakeTorrent:
    """Mimics one qBittorrent torrent info row (the fields `poll_torrent` reads).

    The telemetry fields (`dlspeed` / `eta` / `completed` / `size`) default
    to None so the common monitor tests don't have to set them. `poll_torrent`
    reads them via `getattr` and sanitizes None to None. `hash` is only read
    by the batched `poll_telemetry` (which keys its result off it). A blank
    hash is filled in from the script key when served from a `FakeQbit`
    telemetry script, so only a deliberate-mismatch test needs to set it.
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
        torrent_hash: str = "",
    ) -> None:
        self.state_enum = FakeStateEnum(is_complete=is_complete, is_errored=is_errored)
        self.progress = progress
        self.content_path = content_path
        self.dlspeed = dlspeed
        self.eta = eta
        self.completed = completed
        self.size = size
        self.hash = torrent_hash


type QbitStep = FakeTorrent | Exception
"""One scripted qBittorrent reading: an info row, or an error instance to raise."""


class FakeQbit:
    """A scriptable qBittorrent client with per-hash, per-lane scripts.

    `torrents` scripts the heavy per-hash poll: each infohash maps to its own
    ordered lifecycle of readings, and every single-hash `torrents_info` call
    advances THAT hash's script by one (clamping at the last, so a steady state
    repeats indefinitely). An unscripted hash reads as gone (`[]`). An
    Exception step is raised to exercise the transient-error path. `telemetry`
    separately scripts the fast batched poll the same way - a batch read never
    consumes a heavy script (and vice versa), so a test scripts torrent
    lifecycles declaratively without caring which lane polls first. Hashes with
    no telemetry script simply don't appear in a batch. Hash matching is
    case-insensitive on both lanes (real qBittorrent is). `calls` counts every
    `torrents_info` call, either lane.
    """

    def __init__(
        self,
        torrents: dict[str, list[QbitStep]] | None = None,
        *,
        telemetry: dict[str, list[QbitStep]] | None = None,
    ) -> None:
        self._torrents = {h.casefold(): list(steps) for h, steps in (torrents or {}).items()}
        self._telemetry: dict[str, list[QbitStep]] = {}
        for h, steps in (telemetry or {}).items():
            for step in steps:
                # A blank row hash keys back to its script key, so common
                # telemetry scripts don't have to repeat the hash.
                if isinstance(step, FakeTorrent) and not step.hash:
                    step.hash = h
            self._telemetry[h.casefold()] = list(steps)
        self._heavy_index: dict[str, int] = {}
        self._batch_index: dict[str, int] = {}
        self.calls = 0

    @staticmethod
    def _next(steps: list[QbitStep], index: dict[str, int], key: str) -> FakeTorrent:
        """Advance `key`'s cursor through `steps` (clamped). Raise Exception steps."""

        i = index.get(key, 0)
        index[key] = i + 1
        step = steps[min(i, len(steps) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[FakeTorrent]:
        self.calls += 1
        if isinstance(torrent_hashes, str):
            steps = self._torrents.get(torrent_hashes.casefold())
            if steps is None:
                return []
            return [self._next(steps, self._heavy_index, torrent_hashes.casefold())]
        rows: list[FakeTorrent] = []
        for infohash in torrent_hashes:
            steps = self._telemetry.get(infohash.casefold())
            if steps is not None:
                rows.append(self._next(steps, self._batch_index, infohash.casefold()))
        return rows


class _InterruptOnHash(FakeQbit):
    """Raises KeyboardInterrupt when the heavy poll reaches the given hash.

    Models a Ctrl-C landing mid-advance-loop, after earlier records already advanced.
    """

    def __init__(self, torrents: dict[str, list[QbitStep]], *, interrupt_on: str) -> None:
        super().__init__(torrents)
        self._interrupt_on = interrupt_on.casefold()

    @override
    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[FakeTorrent]:
        if isinstance(torrent_hashes, str) and torrent_hashes.casefold() == self._interrupt_on:
            raise KeyboardInterrupt
        return super().torrents_info(torrent_hashes=torrent_hashes)


def make_wait_manager(qbit: FakeQbit) -> ImportWaitManager:
    """A bare `ImportWaitManager` wired only with the `qbit` the poll reads."""

    return make_bare_instance(ImportWaitManager, qbit=qbit)


class TestPollTorrent:
    """poll_torrent maps a single qBittorrent read to a sanitized TorrentProbe."""

    def test_missing_on_empty_list(self) -> None:
        mgr = make_wait_manager(FakeQbit({}))

        assert mgr.poll_torrent("h") == TorrentProbe(WaitOutcome.MISSING, None, 0.0)

    def test_errored(self) -> None:
        mgr = make_wait_manager(FakeQbit({"h": [FakeTorrent(is_errored=True)]}))

        assert mgr.poll_torrent("h") == TorrentProbe(WaitOutcome.ERRORED, None, 0.0)

    def test_complete_carries_content_path(self) -> None:
        torrent = FakeTorrent(is_complete=True, content_path="/data/show")
        mgr = make_wait_manager(FakeQbit({"h": [torrent]}))

        assert mgr.poll_torrent("h") == TorrentProbe(
            WaitOutcome.COMPLETE,
            "/data/show",
            0.0,
        )

    def test_complete_on_full_progress_without_flag(self) -> None:
        # progress == 1.0 counts as complete even if the state flag is unset.
        torrent = FakeTorrent(progress=1.0, content_path="/data/movie")
        mgr = make_wait_manager(FakeQbit({"h": [torrent]}))

        assert mgr.poll_torrent("h") == TorrentProbe(
            WaitOutcome.COMPLETE,
            "/data/movie",
            1.0,
        )

    def test_none_while_downloading_carries_progress(self) -> None:
        mgr = make_wait_manager(FakeQbit({"h": [FakeTorrent(progress=0.5)]}))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.5)

    def test_none_on_transient_api_error(self) -> None:
        # A dropped connection / re-auth in flight is "still waiting", not terminal.
        # The un-observed flag tells the monitor the zeroed telemetry is a
        # placeholder (keep the last real bar), not a reading.
        mgr = make_wait_manager(FakeQbit({"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.0, observed=False)

    def test_none_when_no_client(self) -> None:
        mgr = make_bare_instance(ImportWaitManager, qbit=None)

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.0, observed=False)

    def test_carries_live_download_telemetry(self) -> None:
        torrent = FakeTorrent(
            progress=0.64,
            dlspeed=3_200_000,
            eta=130,
            completed=1_800_000_000,
            size=2_900_000_000,
        )
        mgr = make_wait_manager(FakeQbit({"h": [torrent]}))

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
        mgr = make_wait_manager(FakeQbit({"h": [torrent]}))

        assert mgr.poll_torrent("h") == TorrentProbe(None, None, 0.5)


class TestPollTelemetry:
    """poll_telemetry: ONE batched, read-only info call for the fast cockpit refresh."""

    def test_batches_and_matches_hashes_case_insensitively(self) -> None:
        # qBittorrent lowercases response hashes. The result must key back to OUR
        # infohash spelling so the monitor's views dict matches.
        row = FakeTorrent(progress=0.5, dlspeed=200, completed=100, size=1000)
        qbit = FakeQbit(telemetry={"ABC123": [row]})  # blank row hash fills in as "ABC123"
        mgr = make_wait_manager(qbit)

        telemetry = mgr.poll_telemetry(["abc123"])

        assert qbit.calls == 1  # one call covers the whole set
        assert telemetry == {"abc123": TorrentTelemetry(0.5, 200, None, 100, 1000)}

    def test_one_call_covers_many_hashes(self) -> None:
        qbit = FakeQbit(
            telemetry={
                "A1": [FakeTorrent(progress=0.2, dlspeed=10)],
                "B2": [FakeTorrent(progress=0.8, dlspeed=20)],
            },
        )
        mgr = make_wait_manager(qbit)

        telemetry = mgr.poll_telemetry(["a1", "b2"])

        assert qbit.calls == 1  # genuinely batched, not per-hash
        assert set(telemetry) == {"a1", "b2"}

    def test_empty_set_makes_no_call(self) -> None:
        qbit = FakeQbit({})
        mgr = make_wait_manager(qbit)

        assert mgr.poll_telemetry([]) == {}
        assert qbit.calls == 0

    def test_transient_error_yields_nothing(self) -> None:
        # The rows just keep their last telemetry until the next heavy poll.
        mgr = make_wait_manager(FakeQbit(telemetry={"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert mgr.poll_telemetry(["h"]) == {}

    def test_unasked_rows_are_ignored(self) -> None:
        # The batch answers for "h" with a row whose hash reads back "other" -
        # poll_telemetry must drop what it can't key back to an asked hash.
        mgr = make_wait_manager(FakeQbit(telemetry={"h": [FakeTorrent(torrent_hash="other", progress=0.4)]}))

        assert mgr.poll_telemetry(["h"]) == {}


class FakeClock:
    """A monotonic clock the wait loop reads. Advances by a fixed step per sleep."""

    def __init__(self, step: float) -> None:
        self.t = 0.0
        self._step = step

    def now(self) -> float:
        return self.t

    def sleep(self, _seconds: float) -> None:
        # Ignore the requested duration. Advance our own clock so the loop's
        # deadline arithmetic is exercised without ever really sleeping.
        self.t += self._step


# A timestamp far enough in the past/future that the TTL verdict is fixed no
# matter when the suite runs (cutoff = now - imports.pending_max_age_days).
_FRESH = "2999-01-01 00:00:00"
_EXPIRED = "2000-01-01 00:00:00"


@dataclass(frozen=True)
class _ImportCall:
    """One recorded `import_completed` call: its record/path + force/deadline flags."""

    pending: PendingImport
    content_path: str
    force: bool
    at_deadline: bool


class _RecordingStrategy(FakeStrategy):
    """A `FakeStrategy` that records + scripts the two import hooks the manager drives.

    `import_completed` records each call's force/at_deadline flags (asserted on
    `import_calls`) and dispenses a scripted `ImportProbe` - a single
    `completed` repeated, a `completed_sequence` advanced per call (clamped to its
    last), or a `completed_error` raised (the swallowed-import path).
    `import_progress` likewise records (`progress_calls`) and dispenses an
    `ImportProgress`, defaulting to an indeterminate zero - the Tier-2
    fast-poll no-op the heavy-poll tests rely on. `progress_error` is raised
    ONCE on the first call (the fast-lane containment path), then cleared.
    """

    def __init__(
        self,
        *,
        completed: ImportProbe | None = None,
        completed_sequence: list[ImportProbe] | None = None,
        completed_error: Exception | None = None,
        progress: ImportProgress | None = None,
        progress_sequence: list[ImportProgress] | None = None,
        progress_error: Exception | None = None,
    ) -> None:
        super().__init__(items=[], anilist_ids={})
        self._completed = completed
        self._completed_sequence = completed_sequence
        self._completed_error = completed_error
        self._completed_index = 0
        self._progress = progress
        self._progress_sequence = progress_sequence
        self._progress_error = progress_error
        self._progress_index = 0
        self.import_calls: list[_ImportCall] = []
        self.progress_calls: list[PendingImport] = []
        self.close_calls: list[PendingImport] = []

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
    def close_tracked(self, pending: PendingImport) -> None:
        self.close_calls.append(pending)

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        self.progress_calls.append(pending)
        if self._progress_error is not None:
            error = self._progress_error
            self._progress_error = None  # one-shot: the next fast poll succeeds
            raise error
        if self._progress_sequence is not None:
            idx = min(self._progress_index, len(self._progress_sequence) - 1)
            self._progress_index += 1
            return self._progress_sequence[idx]
        if self._progress is not None:
            return self._progress
        return ImportProgress(0, 0, determinate=False)


@dataclass(frozen=True)
class _SnapshotCall:
    """One recorded `log_pending_snapshot` call's reported fields."""

    state: PendingState
    title: str
    coverage: str | None
    url: str | None


class _RecordingReporter:
    """Records `log_pending_snapshot` calls so the inline-snapshot/no-double-report contracts are checkable."""

    def __init__(self) -> None:
        self.snapshot_calls: list[_SnapshotCall] = []

    def log_pending_snapshot(
        self,
        state: PendingState,
        pending: PendingImport,
    ) -> bool:
        self.snapshot_calls.append(_SnapshotCall(state, pending.display_label, pending.coverage, pending.url))
        return True


def make_orchestration_manager(
    *,
    qbit: FakeQbit | None,
    strategy: _RecordingStrategy,
    store_records: list[PendingImport] | None = None,
    pending: list[PendingImport] | None = None,
    reporter: _RecordingReporter | None = None,
    **config_overrides: object,
) -> ImportWaitManager:
    """A bare `ImportWaitManager` wired for the pending-import orchestration paths.

    Seeds the durable per-arr store (via the manager's own `cache_store`) and the
    in-memory `_ctx.pending_imports` list so `prune_expired_pending`,
    `snapshot_pending_for_series`, `reconcile_remaining` and `run_monitor` can
    be driven without a live Sonarr/qBittorrent. The strategy is a recording
    `_RecordingStrategy` (its import hooks scripted per test) and the reporter a
    recording `_RecordingReporter` (a test that asserts on the snapshot reporter
    passes in its own `reporter` to read it back).
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
        mgr.cache_store.put_pending(Arr.SONARR, record.key, record.to_json())
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
        recording = install_recording_hub()

        mgr.prune_expired_pending()

        assert set(mgr._pending_records()) == {pk("fresh")}
        # Only the aged drop is announced (an INFO hub Diagnostic). The
        # unparseable-stamp drop stays DEBUG chatter on the logger.
        (aged,) = recording.of_type(Diagnostic)
        assert aged.severity is Severity.INFO
        assert aged.origin == LOG_NAME
        assert "is older than" in aged.message
        assert "giving up on it" in aged.message

    def test_ttl_direction_keeps_recent_drops_aged(self) -> None:
        # MUTATION PIN: `cutoff = now() - timedelta` flipped to `+` survived the
        # century-scale _FRESH/_EXPIRED stamps above. Real near-now stamps pin the
        # sign: 1h old with a 30-day TTL is KEPT, 31 days old is dropped.
        hour_old = (datetime.now() - timedelta(hours=1)).strftime(UPDATED_AT_STR_FORMAT)
        month_old = (datetime.now() - timedelta(days=31)).strftime(UPDATED_AT_STR_FORMAT)
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[
                pending_import(infohash="recent", added_at=hour_old),
                pending_import(infohash="aged", added_at=month_old),
            ],
            import_pending_max_age_days=30,
        )

        mgr.prune_expired_pending()

        assert set(mgr._pending_records()) == {pk("recent")}


class RecordingWaitView(WaitView):
    """Records every snapshot the manager pushes, for assertion.

    Replaces the old call-tuple FakeWaitView: the view is now a pure function of
    the pushed `WaitSnapshot`, so the tests assert on the recorded snapshot
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
        """Whether any recorded snapshot showed `key` in `phase`."""

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
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert mgr._pending_records() == {}
        assert mgr._ctx.stats.imported == 1
        assert mgr._ctx.pending_states[pk("h")] is PendingState.IMPORTED
        # The record is reported inline with its reconciled state + title (the gap a
        # bare `snapshot_calls != []` check left open: wrong state/title slid through).
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingState.IMPORTED
        # The inline row is labeled title · group (the group disambiguates a
        # series that grabbed several torrents).
        assert reporter.snapshot_calls[0].title == f"Show{SEP}SubGroup"
        # Forced (CDH-off safe) but NOT at the deadline (no loud warning).
        assert strategy.import_calls[-1].force is True
        assert strategy.import_calls[-1].at_deadline is False

    def test_carried_over_downloading_is_queued_and_kept(self) -> None:
        # Still downloading -> queued, record kept, no import attempt.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(progress=0.5)]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert set(mgr._pending_records()) == {pk("h")}
        assert mgr._ctx.pending_states[pk("h")] is PendingState.QUEUED
        assert mgr._ctx.stats.imported == 0
        # The carried-over record is still reported inline, with the QUEUED state.
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingState.QUEUED
        assert reporter.snapshot_calls[0].title == f"Show{SEP}SubGroup"

    def test_this_run_grab_is_skipped_no_double_report(self) -> None:
        # REGRESSION (double-report): a torrent grabbed THIS run lives in
        # _ctx.pending_imports AND the store. The snapshot must skip it entirely -
        # no poll, no row, no counter, no state - so it's only ever `added`.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        this_run = pending_import(infohash="h", series_id=7, added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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
        assert set(mgr._pending_records()) == {pk("h")}

    def test_other_series_record_is_not_touched(self) -> None:
        # The snapshot is series-scoped: a record for a different series is left
        # alone (the deferred reconcile / monitor handles it later).
        strategy = _RecordingStrategy()
        qbit = FakeQbit({"other": [FakeTorrent(progress=0.5)]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending_import(infohash="other", series_id=99, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert pk("other") not in mgr._ctx.pending_states
        assert set(mgr._pending_records()) == {pk("other")}

    def test_complete_without_content_path_stays_importing(self) -> None:
        # MUTATION PIN (_reconcile_one): COMPLETE with an empty content_path must
        # NOT attempt an import (`and` -> `or`) and must classify IMPORTING, kept -
        # never IMPORTED/dropped (the default probe's files_present=False -> True).
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True)]})  # no content_path
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert mgr._ctx.pending_states[pk("h")] is PendingState.IMPORTING
        assert set(mgr._pending_records()) == {pk("h")}
        assert mgr._ctx.stats.imported == 0
        assert [c.state for c in reporter.snapshot_calls] == [PendingState.IMPORTING]


class TestReconcileRemaining:
    """reconcile_remaining force-polls carried-over records not snapshotted this run."""

    def test_imports_ready_record_not_yet_snapshotted(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH)],
        )

        mgr.reconcile_remaining()

        assert mgr._pending_records() == {}
        assert mgr._ctx.stats.imported == 1
        assert strategy.import_calls[-1].force is True
        assert strategy.import_calls[-1].at_deadline is False

    def test_skips_already_snapshotted(self) -> None:
        # A record the inline snapshot already touched must not be re-polled.
        strategy = _RecordingStrategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store_records=[pending_import(infohash="h", added_at=_FRESH)],
        )
        mgr._ctx.pending_states[pk("h")] = PendingState.QUEUED

        mgr.reconcile_remaining()

        assert strategy.import_calls == []

    def test_skips_this_run_grabs(self) -> None:
        strategy = _RecordingStrategy()
        this_run = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store_records=[this_run],
            pending=[this_run],
        )

        mgr.reconcile_remaining()

        assert strategy.import_calls == []

    def test_two_imports_both_counted(self) -> None:
        # MUTATION PIN: `stats.imported += 1` degraded to `= 1` clamps at one.
        # Two carried-over imports in one pass must tally 2.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        qbit = FakeQbit(
            {
                "h1": [FakeTorrent(is_complete=True, content_path="/d1")],
                "h2": [FakeTorrent(is_complete=True, content_path="/d2")],
            },
        )
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[
                pending_import(infohash="h1", added_at=_FRESH),
                pending_import(infohash="h2", added_at=_FRESH),
            ],
        )

        mgr.reconcile_remaining()

        assert mgr._ctx.stats.imported == 2
        assert mgr._pending_records() == {}


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
            pk("q"): PendingState.QUEUED,
            pk("i"): PendingState.IMPORTING,
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

    def test_two_importing_records_both_tallied(self) -> None:
        # MUTATION PIN: `stats.importing += 1` degraded to `= 1` clamps at one.
        # Two known-IMPORTING records must tally 2.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[
                pending_import(infohash="i1", added_at=_FRESH),
                pending_import(infohash="i2", added_at=_FRESH),
            ],
        )
        mgr._ctx.pending_states = {
            pk("i1"): PendingState.IMPORTING,
            pk("i2"): PendingState.IMPORTING,
        }

        mgr.tally_carried_over_into_stats()

        assert mgr._ctx.stats.importing == 2
        assert mgr._ctx.stats.queued == 0


class TestMonitorWorkingSet:
    """_monitor_working_set dedups per record key, the in-memory record winning."""

    def test_in_memory_record_wins_the_store_collision(self) -> None:
        # MUTATION PIN: the first loop's `not in seen` flipped to `in seen` skips
        # every this-run grab, so the store's (staler) copy would be monitored
        # instead. One record key in BOTH places, differing by title: exactly one
        # record survives and it is the in-memory one.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[pending_import(infohash="h", title="StoreCopy", added_at=_FRESH)],
            pending=[pending_import(infohash="h", title="InMemory", added_at=_FRESH)],
        )

        records = mgr._monitor_working_set()

        assert [p.title for p in records] == ["InMemory"]

    def test_sibling_records_on_one_torrent_are_both_monitored(self) -> None:
        # Two AniList entries share one torrent: two records, two working-set
        # rows - the old bare-infohash dedup shadowed the second entry's slice.
        first = pending_import(infohash="h", al_id=11, title="Cour 1", added_at=_FRESH)
        second = pending_import(infohash="h", al_id=22, title="Cour 2", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store_records=[first, second],
            pending=[first, second],
        )

        records = mgr._monitor_working_set()

        assert sorted(p.title or "" for p in records) == ["Cour 1", "Cour 2"]
        # The store round-trip keeps them distinct too (composite-keyed rehydration).
        rehydrated = mgr._pending_records()
        assert {key: p.title for key, p in rehydrated.items()} == {
            PendingKey("h", 11): "Cour 1",
            PendingKey("h", 22): "Cour 2",
        }


class TestRunMonitor:
    """run_monitor: interleaved, copy-aware wait+import over ALL pending."""

    def test_interleaved_fast_and_slow(self) -> None:
        # Two torrents: "fast" completes + imports first cycle (files present).
        # "slow" is still downloading, then completes + imports a later cycle. Both
        # advance each cycle (interleaved), so the fast one isn't stuck behind slow.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        qbit = FakeQbit(
            {
                "fast": [FakeTorrent(is_complete=True, content_path="/fast")],
                # slow: downloading on the first cycle, complete after.
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        fast = pending_import(infohash="fast", added_at=_FRESH)
        slow = pending_import(infohash="slow", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=qbit,
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
        assert mgr._pending_records() == {}
        assert view.final(rk("fast")).outcome is Outcome.IMPORTED
        assert view.final(rk("slow")).outcome is Outcome.IMPORTED
        # Each torrent's OWN content_path reached import_completed - a run_monitor bug
        # forwarding the wrong torrent's path would still import, so pin the pairing.
        by_hash = {c.pending.infohash: c.content_path for c in strategy.import_calls}
        assert by_hash["fast"] == "/fast"
        assert by_hash["slow"] == "/slow"
        # slow showed a downloading heartbeat (fraction 0.5) before it completed.
        assert any(
            t.key == rk("slow") and t.phase is Phase.DOWNLOADING and t.fraction == 0.5
            for snap in view.snapshots
            for t in snap.torrents
        )

    def test_imported_only_when_files_present_two_cycles(self) -> None:
        # The copy is async: cycle 1 issues the command (RETRY + command_issued,
        # files NOT present) -> reads `importing`. Cycle 2 verifies files present
        # -> `imported`. imported is gated on verified files, never command accept.
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
                import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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
        assert view.saw(rk("h"), Phase.IMPORTING)  # cycle 1, copy in flight
        assert view.final(rk("h")).outcome is Outcome.IMPORTED  # cycle 2, files landed
        assert mgr._pending_records() == {}

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
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        # Only ONE heavy poll. The fast poll did the rest.
        assert len(strategy.import_calls) == 1
        assert len(strategy.progress_calls) == 3
        # The bar advanced (2/3 seen) before the row finished.
        assert any(
            t.key == rk("h") and t.import_done == 2 and t.import_total == 3
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final(rk("h")).outcome is Outcome.IMPORTED
        assert mgr._pending_records() == {}

    def test_tier2_disabled_skips_the_fast_poll(self) -> None:
        # progress_poll_interval=0 -> no cheap poll at all. The heavy poll alone
        # drives completion (the bar simply steps once per poll).
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
                import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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
        assert view.final(rk("h")).outcome is Outcome.IMPORTED

    def test_importing_at_deadline_left_without_warning(self) -> None:
        # The copy never lands within imports.ready_timeout: the final attempt
        # (at_deadline) leaves it pending with "still importing; left" - no drop.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        assert view.final(rk("h")).outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_records()) == {pk("h")}  # left, not dropped
        # The final in-bound poll forces AND flags the deadline, for THIS torrent's path.
        last = strategy.import_calls[-1]
        assert last.content_path == "/d"
        assert last.force is True
        assert last.at_deadline is True

    def test_landing_files_re_anchor_the_ready_deadline(self) -> None:
        # A season pack Sonarr copies file-by-file: a baseline (t=0) then a rise
        # each heavy poll. Every rise re-anchors the ready deadline, so the pass
        # outlives ready_timeout (60s here) and finishes imported instead of
        # cutting the import off mid-copy.
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(
                    ImportReadiness.RETRY,
                    files_present=False,
                    command_issued=True,
                    imported_count=done,
                    target_count=3,
                )
                for done in (0, 1, 2)
            ]
            + [import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True, target_count=3)],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        assert view.final(rk("h")).outcome is Outcome.IMPORTED
        assert mgr._pending_records() == {}
        # Four polls (t=0/30/60/90), none forced: the deadline never fired.
        assert len(strategy.import_calls) == 4
        assert all(c.at_deadline is False for c in strategy.import_calls)

    def test_deadline_attempt_rescued_by_a_same_cycle_landing(self) -> None:
        # The forced deadline attempt itself reports a fresh rise (0/3 -> 1/3 at
        # t=60): the row must NOT terminal that cycle - the landing re-anchors
        # the deadline instead - and only the NEXT quiet deadline (t=120) ends it.
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(
                    ImportReadiness.RETRY,
                    files_present=False,
                    command_issued=True,
                    imported_count=done,
                    target_count=3,
                )
                for done in (0, 0, 1)
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        assert view.final(rk("h")).outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_records()) == {pk("h")}
        # t=0/30 in-bound, t=60 forced-but-rescued, t=90 in-bound, t=120 forced.
        assert [c.at_deadline for c in strategy.import_calls] == [False, False, True, False, True]

    def test_static_done_count_never_extends_the_deadline(self) -> None:
        # A genuinely stalled import: the first determinate reading (1/3) is a
        # baseline, not progress, and a count that never rises past it must not
        # move the anchor - the deadline still fires on schedule.
        strategy = _RecordingStrategy(
            completed=import_probe(
                ImportReadiness.RETRY,
                files_present=False,
                command_issued=True,
                imported_count=1,
                target_count=3,
            ),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        assert view.final(rk("h")).outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_records()) == {pk("h")}
        # Three polls (t=0/30/60): the third is the forced deadline attempt.
        assert len(strategy.import_calls) == 3
        assert strategy.import_calls[-1].at_deadline is True

    def test_fast_lane_landing_re_anchors_the_deadline(self) -> None:
        # The cheap Tier-2 poll sees a file land (1/3 -> 2/3) even though the
        # heavy poll's counts stay indeterminate. That rise alone must push the
        # deadline out one more heavy cycle (t=90, not t=60).
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
            progress_sequence=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),  # the rise, at t=10
            ],
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=60,
            import_poll_interval=30,
            progress_poll_interval=5,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=5)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert view.final(rk("h")).outcome is Outcome.STILL_IMPORTING
        # Heavy polls at t=0/30/60/90: the t=60 poll is in-bound only because the
        # t=10 fast-lane rise re-anchored the deadline to fire at t=70.
        assert len(strategy.import_calls) == 4
        assert strategy.import_calls[-1].at_deadline is True

    def test_missing_drops_record(self) -> None:
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),
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
        assert view.final(rk("h")).outcome is Outcome.MISSING
        assert mgr._pending_records() == {}

    def test_errored_leaves_record(self) -> None:
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_errored=True)]}),
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
        assert view.final(rk("h")).outcome is Outcome.DOWNLOAD_ERRORED
        assert set(mgr._pending_records()) == {pk("h")}

    def test_download_timeout_is_terminal_and_leaves_record(self) -> None:
        # The torrent never finishes: past imports.wait_timeout the row terminates
        # DOWNLOAD_TIMED_OUT (deferred), the record stays for a later run, and the
        # post-import category hook is never touched (SUCCESS-only).
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = CategoryQbit({"h": [FakeTorrent(progress=0.5)]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            import_wait_timeout=60,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert strategy.import_calls == []
        assert view.final(rk("h")).outcome is Outcome.DOWNLOAD_TIMED_OUT
        assert set(mgr._pending_records()) == {pk("h")}  # left pending, not dropped
        assert qbit.set_category_calls == []  # only a verified import moves category

    def test_complete_without_content_path_gets_its_own_outcome(self) -> None:
        # qBittorrent reports COMPLETE but hands back no content_path to import
        # from: terminal NO_CONTENT_PATH (not a misleading "timed out" - the
        # download finished fine), never an import attempt, record kept.
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True)]}),
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
        assert view.final(rk("h")).outcome is Outcome.NO_CONTENT_PATH
        assert set(mgr._pending_records()) == {pk("h")}

    def test_tier2_progress_poll_error_is_contained(self) -> None:
        # A raising import_progress during the fast lane is debug-logged and
        # skipped - the row survives to the next heavy poll, which lands the import.
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),
                import_probe(ImportReadiness.RETRY, files_present=True, command_issued=True),
            ],
            progress_error=RuntimeError("progress boom"),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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
        handler = CaptureHandler()
        mgr.logger.addHandler(handler)
        mgr.logger.setLevel(logging.DEBUG)
        view = RecordingWaitView()
        clock = FakeClock(step=5)
        try:
            mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)  # must not raise
        finally:
            mgr.logger.removeHandler(handler)
            mgr.logger.setLevel(logging.WARNING)

        assert view.final(rk("h")).outcome is Outcome.IMPORTED
        assert len(strategy.import_calls) == 2  # the heavy poll still ran and landed it
        assert len(strategy.progress_calls) >= 2  # the fast lane kept polling after the raise
        assert any(r.levelno == logging.DEBUG and "progress poll" in r.getMessage() for r in handler.records)

    def test_wait_scope_all_includes_store_only_carried_over(self) -> None:
        # REGRESSION (complaint 4 - "exited right away"): a carried-over record
        # that is ONLY in the store (NOT a this-run grab) is still monitored. Here
        # there are no this-run grabs at all, yet the store record is driven.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        qbit = FakeQbit({"carried": [FakeTorrent(is_complete=True, content_path="/d")]})
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
        assert view.final(rk("carried")).outcome is Outcome.IMPORTED
        assert mgr._pending_records() == {}

    def test_keyboard_interrupt_breaks_and_leaves_records(self) -> None:
        # Ctrl-C during the poll nap must break the loop (not propagate), so the
        # caller's finally still restores the terminal + saves the cache. The
        # in-flight record is left pending and a WaitResult is still returned.
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        def interrupt(_seconds: float) -> None:
            raise KeyboardInterrupt

        result = mgr.run_monitor(  # must not raise
            now=lambda: 0.0,
            sleep=interrupt,
            view=view,
        )

        assert result is not None and result.waited == 0
        assert set(mgr._pending_records()) == {pk("h")}  # left pending for next run
        assert view.saw(rk("h"), Phase.DOWNLOADING)
        assert view.closed is False  # injected views are the caller's to close (own_view seam)
        # The break is announced as an INFO hub Diagnostic.
        (note,) = recording.of_type(Diagnostic)
        assert note.severity is Severity.INFO
        assert note.message == "Wait interrupted - 1 left pending"
        assert note.origin == LOG_NAME

    def test_interrupt_mid_cycle_still_graduates_that_cycles_terminals(self) -> None:
        # A torrent that turned terminal in the SAME cycle the interrupt lands in
        # must still reach the view (the except arm pushes one final snapshot), so
        # the console tally/ledger can never undercount the returned WaitResult.
        first = pending_import(infohash="h1", added_at=_FRESH)
        second = pending_import(infohash="h2", added_at=_FRESH)
        # h1 is unscripted -> gone -> MISSING terminal on its first advance. The
        # interrupt then lands on h2's poll, before that cycle's snapshot push.
        qbit = _InterruptOnHash({"h2": [FakeTorrent(progress=0.3)]}, interrupt_on="h2")
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            store_records=[first, second],
            pending=[first, second],
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()

        result = mgr.run_monitor(now=lambda: 0.0, sleep=lambda _s: None, view=view)  # must not raise

        assert result is not None and result.waited == 1
        assert view.final(rk("h1")).outcome is Outcome.MISSING  # the interrupt-time push carried it

    def test_import_exception_is_swallowed_and_record_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate and
        # abort _finalize_run's cache save. The record is left pending instead.
        strategy = _RecordingStrategy(completed_error=RuntimeError("boom"))
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        assert view.final(rk("h")).outcome is Outcome.NOTHING_TO_IMPORT
        assert set(mgr._pending_records()) == {pk("h")}

    def test_swallowed_import_announces_an_error_and_leaves(self) -> None:
        # The observable half of the swallow above: the failure is announced as
        # an ERROR Diagnostic carrying its traceback, and the probe says LEAVE.
        strategy = _RecordingStrategy(completed_error=RuntimeError("boom"))
        mgr = make_orchestration_manager(qbit=None, strategy=strategy)
        recording = install_recording_hub()

        probe = mgr.try_import_completed(pending_import(infohash="h", added_at=_FRESH), "/downloads/x")

        assert probe == ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert error.message == "Manual import failed for Show · SubGroup - leaving it for a later run"
        assert error.trace is not None

    def test_imported_terminal_row_carries_files_count(self) -> None:
        # The graduation ledger states "(N files · elapsed)": a terminal imported
        # row carries the verified files count (the probe's seed target_count).
        strategy = _RecordingStrategy(
            completed=import_probe(
                ImportReadiness.RETRY,
                files_present=True,
                imported_count=3,
                target_count=3,
            ),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
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

        final = view.final(rk("h"))
        assert final.outcome is Outcome.IMPORTED
        assert final.import_total == 3


def _monitor_pass(
    qbit: FakeQbit,
    record: PendingImport,
) -> MonitorPass:
    """A fresh `MonitorPass` over one record, wired to a scripted qBittorrent."""

    mgr = make_orchestration_manager(qbit=qbit, strategy=_RecordingStrategy())
    clock = FakeClock(step=5)
    return MonitorPass(mgr, [record], now=clock.now, dl_timeout=3600, import_timeout=600)


class TestMonitorFastTelemetry:
    """refresh_telemetry: the fast-lane download refresh - telemetry only, no verdicts."""

    def test_progress_wait_drives_the_telemetry_refresh(self) -> None:
        # The wiring test: between heavy polls, _progress_wait's fast lane pushes
        # a snapshot with fresh download telemetry (the row moves to 60% at
        # 999 B/s well before the next heavy cycle).
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.3, dlspeed=100),
                    FakeTorrent(is_complete=True, content_path="/d", progress=1.0),
                ]
            },
            telemetry={"h": [FakeTorrent(progress=0.6, dlspeed=999)]},  # the fast batch reading
        )
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

        assert any(
            t.key == rk("h") and t.phase is Phase.DOWNLOADING and t.fraction == 0.6 and t.speed_bps == 999
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final(rk("h")).outcome is Outcome.IMPORTED

    def test_views_that_render_no_telemetry_skip_the_read(self) -> None:
        # A view with wants_telemetry=False (the non-TTY digest) must not cost a
        # qBittorrent read every fast slice: only the two heavy polls hit the client.
        class NoTelemetryView(RecordingWaitView):
            wants_telemetry = False

        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.3, dlspeed=100),
                    FakeTorrent(is_complete=True, content_path="/d", progress=1.0),
                ]
            },
        )
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
        clock = FakeClock(step=5)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=NoTelemetryView())

        assert qbit.calls == 2

    def test_updates_downloading_row_without_a_phase_transition(self) -> None:
        record = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
            telemetry={"h": [FakeTorrent(progress=1.0, dlspeed=250)]},
        )
        mp = _monitor_pass(qbit, record)
        mp.advance(record)
        assert mp.views[rk("h")].phase is Phase.DOWNLOADING

        changed = mp.refresh_telemetry()

        assert changed is True
        view = mp.views[rk("h")]
        # Telemetry moved (even to 100%) but the phase did NOT change - terminal
        # decisions belong to the heavy poll alone.
        assert view.phase is Phase.DOWNLOADING
        assert view.fraction == 1.0
        assert view.speed_bps == 250
        # The sparkline window is heavy-poll-sampled. The fast refresh adds nothing.
        assert view.speed_history == (100,)
        assert rk("h") in mp.active

    def test_unchanged_telemetry_reports_no_change(self) -> None:
        record = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
            telemetry={"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
        )
        mp = _monitor_pass(qbit, record)
        mp.advance(record)

        assert mp.refresh_telemetry() is False


class TestMonitorTransientPoll:
    """A transient qBittorrent blip keeps the row's last real state on screen."""

    def test_unobserved_poll_keeps_bar_and_history(self) -> None:
        record = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.3, dlspeed=100),
                    qbittorrentapi.APIConnectionError("blip"),
                    FakeTorrent(progress=0.4, dlspeed=200),
                ],
            },
        )
        mp = _monitor_pass(qbit, record)

        mp.run_cycle()
        mp.run_cycle()  # the blip cycle
        assert mp.views[rk("h")].fraction == 0.3  # last real reading kept, no 0% flash
        mp.run_cycle()

        view = mp.views[rk("h")]
        assert view.fraction == 0.4
        # Only the two real readings are in the sparkline window - the blip never
        # injected a fake stall sample.
        assert view.speed_history == (100, 200)


class TestMonitorSpeedHistory:
    """The downloading rows' sparkline window: one sample per heavy poll, bounded."""

    def test_accumulates_per_heavy_poll_with_stalls_as_zero(self) -> None:
        record = pending_import(infohash="h", added_at=_FRESH)
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.1, dlspeed=100),
                    FakeTorrent(progress=0.2),  # stalled (no speed) -> a 0 sample
                    FakeTorrent(progress=0.3, dlspeed=300),
                ],
            },
        )
        mp = _monitor_pass(qbit, record)

        for _ in range(3):
            mp.run_cycle()

        assert mp.views[rk("h")].speed_history == (100, 0, 300)

    def test_window_is_bounded_to_spark_samples(self) -> None:
        record = pending_import(infohash="h", added_at=_FRESH)
        mp = _monitor_pass(FakeQbit({"h": [FakeTorrent(progress=0.1, dlspeed=100)]}), record)

        for _ in range(SPARK_SAMPLES + 3):
            mp.run_cycle()

        assert mp.views[rk("h")].speed_history == (100,) * SPARK_SAMPLES


class TestImportWaitModeProperty:
    """The services hub exposes the run's RESOLVED wait mode (cli > config).

    The Sonarr strategy's seed-building gate reads this instead of the raw config
    so a CLI override that turns the feature on over an `off` config still
    builds seeds (otherwise the whole pass silently no-ops).
    """

    def test_reflects_resolved_mode(self) -> None:
        services = make_services(
            _ctx=RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.HYBRID),
        )

        assert services.import_wait_mode is ImportWaitMode.HYBRID


def _attach_wait_manager(engine: RunLoop) -> None:
    """Attach an `ImportWaitManager` sharing the engine's run state.

    The wait/poll machinery lives on the manager now, so a finalize/snapshot test
    on a bare engine must wire one bound to the SAME `_ctx` / `cache_store` /
    client / strategy the engine holds - exactly as `__init__` + `begin_run` do.
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
    """Records the run tail's summary + the two close boundaries as markers."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def scan_finished(self, arr: Arr) -> None:
        self._calls.append("scan_finished")

    def log_run_summary(
        self,
        ctx: RunContext,
        *,
        preview: bool,
        has_client: bool,
    ) -> bool:
        self._calls.append("summary")
        return True

    def run_finished(self, arr: Arr) -> None:
        self._calls.append("run_finished")


class _RecordingCacheStore(FakeCacheStore):
    """A `FakeCacheStore` whose `save` appends a `"save"` ordering marker."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    @override
    def save(self, *, preview: bool) -> None:
        self._calls.append("save")


class _FinalizeWaitManager:
    """A stand-in wait manager for the finalize-ordering tests.

    Every pass appends its ordering marker. The real ones return early on the
    empty working set these tests build - recording nothing - so a recording
    stand-in is what makes each step observable. `raise_on` scripts one pass to
    fail, for the unwind pins (the raise escapes `_finalize_run`, exactly as a
    real failure would, and bootstrap's finally is what closes the run).
    """

    def __init__(self, calls: list[str], *, raise_on: str | None = None) -> None:
        self._calls = calls
        self._raise_on = raise_on

    def _mark(self, name: str) -> None:
        self._calls.append(name)
        if name == self._raise_on:
            raise RuntimeError(f"{name} exploded")

    def reconcile_remaining(self) -> None:
        self._mark("reconcile")

    def tally_carried_over_into_stats(self) -> None:
        self._mark("tally")

    def run_monitor(self) -> WaitResult | None:
        self._mark("monitor")
        return None


def _finalize_engine(
    calls: list[str],
    *,
    qbit: object,
    mode: ImportWaitMode,
    raise_on: str | None = None,
    supports_monitor: bool = True,
) -> RunLoop:
    """A bare engine whose every run-tail step appends a marker to `calls`.

    The reporter, cache store, and wait manager are typed recorders (a
    `_FinalizeReporter` / `_RecordingCacheStore` / `_FinalizeWaitManager`), so
    `_finalize_run`'s ordering is asserted on the recorded `calls` list without a
    live Sonarr/qBittorrent. The finalize's preview fact comes off the services hub,
    so a bare one shares the engine's ctx (and the `qbit` that decides preview).
    `raise_on` names the wait-manager pass that should blow up. `supports_monitor`
    scripts the active strategy's blocking-monitor flag - True is a Sonarr-shaped
    strategy, False a Radarr one (reconciles pre-summary, never monitors).
    """

    ctx = RunContext(arr=Arr.SONARR, import_wait_mode=mode)
    return make_bare_instance(
        RunLoop,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        ),
        _reporter=_FinalizeReporter(calls),
        cache_store=_RecordingCacheStore(calls),
        _wait_manager=_FinalizeWaitManager(calls, raise_on=raise_on),
        _services=make_services(qbit=qbit, _ctx=ctx),
        _active_strategy=FakeStrategy(items=[], anilist_ids={}, supports_blocking_monitor=supports_monitor),
        _ctx=ctx,
    )


class TestFinalizeRunOrdering:
    """_finalize_run brackets the tail with its close boundaries, summary before monitor.

    `scan_finished` leads on every path - so the reconcile/tally diagnostics that
    follow place at run level, never inside the last entry - and `run_finished`
    trails the save, so it is the leg's last event.
    """

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

        assert calls == ["scan_finished", "tally", "summary", "monitor", "save", "run_finished"]

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
        assert calls == ["scan_finished", "reconcile", "tally", "summary", "save", "run_finished"]

    def test_preview_skips_monitor_and_tally(self) -> None:
        # A preview (no client) short-circuits reconcile/tally/monitor.
        calls: list[str] = []
        engine = _finalize_engine(calls, qbit=None, mode=ImportWaitMode.BLOCKING)

        engine._finalize_run()

        assert "monitor" not in calls
        # Even short-circuited, the summary still prints and the save still runs last.
        assert calls == ["scan_finished", "summary", "save", "run_finished"]

    def test_non_monitor_strategy_reconciles_in_blocking_without_monitoring(self) -> None:
        # A Radarr-shaped strategy (no blocking monitor) reconciles pre-summary in
        # BLOCKING too - the monitor would otherwise be its only reconcile - and
        # never enters the monitor cockpit.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
            supports_monitor=False,
        )

        engine._finalize_run()

        assert "monitor" not in calls
        assert calls == ["scan_finished", "reconcile", "tally", "summary", "save", "run_finished"]

    def test_non_monitor_strategy_reconciles_in_hybrid_without_monitoring(self) -> None:
        # HYBRID gets the same treatment for a non-monitor strategy: reconcile, no monitor.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.HYBRID,
            supports_monitor=False,
        )

        engine._finalize_run()

        assert "monitor" not in calls
        assert calls == ["scan_finished", "reconcile", "tally", "summary", "save", "run_finished"]


class TestFinalizeRunUnwind:
    """A raise in the tail leaves `run_finished` to bootstrap's unwind emit.

    The scan is already closed either way, so the leg-fatal error the composition
    root logs can never render inside a stale entry (Band D review finding #7). The
    finally spans the whole tail, so a raise still saves - the run's staged writes
    survive rather than rolling back when bootstrap closes the store.
    """

    def test_reconcile_raise_closes_the_scan_but_not_the_run(self) -> None:
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.DEFERRED,
            raise_on="reconcile",
        )

        with pytest.raises(RuntimeError, match="reconcile exploded"):
            engine._finalize_run()

        # The raise still escapes (bootstrap closes the run), but the finally now
        # saves so the run's staged writes persist rather than rolling back.
        assert calls == ["scan_finished", "reconcile", "save"]

    def test_monitor_raise_still_saves_but_leaves_the_run_open(self) -> None:
        # run_finished sits OUTSIDE the save's finally on purpose: emitting it there
        # would double up with bootstrap's unwind emit on this very path.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
            raise_on="monitor",
        )

        with pytest.raises(RuntimeError, match="monitor exploded"):
            engine._finalize_run()

        assert calls == ["scan_finished", "tally", "summary", "monitor", "save"]

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
            engine._finalize_run()

        assert calls == ["scan_finished", "reconcile", "tally", "save"]


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

        mgr.drop_pending(drop)

        assert set(mgr._pending_records()) == {pk("keep")}
        assert mgr._ctx.pending_imports == [keep]


class CategoryQbit(FakeQbit):
    """A `FakeQbit` recording the category writes the post-import move makes.

    `set_errors` scripts per-call `torrents_set_category` failures (popped in
    order), so the create-on-409 retry and the best-effort warn path can be driven.
    """

    def __init__(
        self,
        torrents: dict[str, list[QbitStep]] | None = None,
        *,
        set_errors: list[Exception] | None = None,
    ) -> None:
        super().__init__(torrents)
        self._set_errors = list(set_errors or [])
        self.set_category_calls: list[tuple[str, str]] = []
        self.created_categories: list[str] = []

    def torrents_set_category(self, *, category: str, torrent_hashes: str) -> None:
        if self._set_errors:
            raise self._set_errors.pop(0)
        self.set_category_calls.append((category, torrent_hashes))

    def torrents_create_category(self, *, name: str) -> None:
        self.created_categories.append(name)


class TestPostImportCategory:
    """apply_post_import_category: verified imports move to imports.post_import_category."""

    @staticmethod
    def _imported_manager(
        qbit: CategoryQbit,
        post_import_category: str | None = None,
    ) -> ImportWaitManager:
        """A manager whose one carried-over record reconciles straight to IMPORTED.

        A `None` category routes through `make_config` to the blank-drop, so it
        exercises the same "left unset" default a real config file yields.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
            ),
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
            post_import_category=post_import_category,
        )

    def test_reconcile_moves_imported_torrent(self) -> None:
        # The inline snapshot confirms the import -> the torrent moves category
        # BEFORE the record is dropped.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._imported_manager(qbit, "pearlarr-done")

        mgr.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert qbit.created_categories == []  # no 409 -> no create
        assert mgr._pending_records() == {}

    def test_monitor_moves_imported_torrent(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert view.final(rk("h")).outcome is Outcome.IMPORTED
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_missing_torrent_is_not_recategorized(self) -> None:
        # MISSING also drops the record, but only a verified IMPORT moves category.
        pending = pending_import(infohash="h", added_at=_FRESH)
        qbit = CategoryQbit({})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            store_records=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert view.final(rk("h")).outcome is Outcome.MISSING
        assert mgr._pending_records() == {}  # still dropped
        assert qbit.set_category_calls == []

    def test_creates_category_on_409_and_retries(self) -> None:
        # qBittorrent 409s an unknown category: create it, then re-apply.
        qbit = CategoryQbit(
            {"h": [FakeTorrent(is_complete=True, content_path="/d")]},
            set_errors=[qbittorrentapi.Conflict409Error("unknown category")],
        )
        mgr = self._imported_manager(qbit, "pearlarr-done")

        mgr.snapshot_pending_for_series(7)

        assert qbit.created_categories == ["pearlarr-done"]
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_client_error_warns_and_import_still_lands(self) -> None:
        # Best-effort: a client error must not undo the import - the record is
        # still dropped and counted, the category is just left as-is.
        qbit = CategoryQbit(
            {"h": [FakeTorrent(is_complete=True, content_path="/d")]},
            set_errors=[qbittorrentapi.APIConnectionError("down")],
        )
        mgr = self._imported_manager(qbit, "pearlarr-done")

        mgr.snapshot_pending_for_series(7)  # must not raise

        assert qbit.set_category_calls == []
        assert mgr._pending_records() == {}
        assert mgr._ctx.stats.imported == 1

    def test_unconfigured_category_makes_no_call(self) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._imported_manager(qbit)  # post_import_category left blank

        mgr.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == []
        assert qbit.created_categories == []
        assert mgr._pending_records() == {}

    def test_configured_category_without_client_is_a_silent_noop(self) -> None:
        # Category configured but no qBittorrent client (preview): nothing to
        # call and nothing emitted - not even the best-effort warning.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            post_import_category="pearlarr-done",
        )
        recording = install_recording_hub()
        mgr.apply_post_import_category(pending_import(infohash="h"))  # must not raise

        assert recording.of_type(Diagnostic) == []

    def test_reconcile_missing_is_not_recategorized(self) -> None:
        # The reconcile twin of the monitor-path MISSING test: the record is
        # dropped, but only a verified IMPORT moves category.
        qbit = CategoryQbit({})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
            post_import_category="pearlarr-done",
        )

        mgr.snapshot_pending_for_series(7)

        assert mgr._pending_records() == {}  # MISSING still drops the record
        assert qbit.set_category_calls == []
        assert qbit.created_categories == []


class TestPostImportCategorySiblingGate:
    """The move waits for EVERY record sharing the torrent: last verified import moves.

    SeaDex can list one torrent on several AniList entries, each with its own
    pending record for its own episode slice. Users key delete-with-data cleanup
    off the category, so a move on the FIRST record's import would flag a torrent
    whose sibling slices are still waiting.
    """

    @staticmethod
    def _siblings() -> tuple[PendingImport, PendingImport]:
        """Two records claiming one torrent, one per AniList entry (multi-cour)."""

        return (
            pending_import(infohash="h", al_id=11, series_id=7, title="Cour 1", added_at=_FRESH),
            pending_import(infohash="h", al_id=22, series_id=7, title="Cour 2", added_at=_FRESH),
        )

    def test_monitor_moves_once_after_both_siblings_import(self) -> None:
        # Cycle 1: Cour 1 verifies -> dropped, but Cour 2 still claims the hash,
        # so NO move fires. Cycle 2: Cour 2 verifies -> last claim gone -> exactly
        # ONE move. Ungated per-record code would have moved twice (and early).
        first, second = self._siblings()
        strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.IMPORTED, files_present=True),  # Cour 1, cycle 1
                import_probe(ImportReadiness.RETRY, files_present=False, command_issued=True),  # Cour 2, cycle 1
                import_probe(ImportReadiness.RETRY, files_present=True),  # Cour 2, cycle 2
            ],
        )
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store_records=[first, second],
            pending=[first, second],
            post_import_category="pearlarr-done",
            import_wait_timeout=3600,
            import_ready_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        clock = FakeClock(step=30)

        mgr.run_monitor(now=clock.now, sleep=clock.sleep, view=view)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert mgr._pending_records() == {}
        # Two tracked rows for the one torrent, each labeled by its own entry.
        assert view.final(rk("h", 11)).outcome is Outcome.IMPORTED
        assert view.final(rk("h", 22)).outcome is Outcome.IMPORTED
        assert view.final(rk("h", 11)).label == f"Cour 1{SEP}SubGroup"
        assert view.final(rk("h", 22)).label == f"Cour 2{SEP}SubGroup"

    def test_reconcile_gate_holds_across_runs_until_the_sibling_imports(self) -> None:
        # Run 1's inline snapshot imports Cour 1 only (Cour 2's slice not landed):
        # no move. Run 2 reconciles the carried-over Cour 2 -> the move fires once.
        first, second = self._siblings()
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        run1 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed_sequence=[
                    import_probe(ImportReadiness.IMPORTED, files_present=True),  # Cour 1
                    import_probe(ImportReadiness.RETRY, files_present=False),  # Cour 2: not landed
                ],
            ),
            store_records=[first, second],
            post_import_category="pearlarr-done",
        )

        run1.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == []  # the sibling still claims the hash
        assert set(run1._pending_records()) == {pk("h", 22)}

        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
            ),
            post_import_category="pearlarr-done",
        )
        run2.cache_store = run1.cache_store  # the same durable store, next run

        run2.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert run2._pending_records() == {}

    def test_ttl_expiry_of_the_last_sibling_releases_the_gate(self) -> None:
        # The aged-out sibling's claim is pruned, so the survivor's verified
        # import still moves the torrent (the gate cannot deadlock on a corpse).
        survivor = pending_import(infohash="h", al_id=11, series_id=7, added_at=_FRESH)
        expired = pending_import(infohash="h", al_id=22, series_id=7, added_at=_EXPIRED)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
            ),
            store_records=[survivor, expired],
            post_import_category="pearlarr-done",
        )

        mgr.prune_expired_pending()
        mgr.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert mgr._pending_records() == {}

    def test_all_siblings_expired_never_moves(self) -> None:
        # Every claim aged out un-imported: nothing was "verified complete", so
        # the category must never move.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            store_records=[
                pending_import(infohash="h", al_id=11, series_id=7, added_at=_EXPIRED),
                pending_import(infohash="h", al_id=22, series_id=7, added_at=_EXPIRED),
            ],
            post_import_category="pearlarr-done",
        )

        mgr.prune_expired_pending()
        mgr.snapshot_pending_for_series(7)

        assert mgr._pending_records() == {}
        assert qbit.set_category_calls == []

    def test_missing_sibling_drop_releases_the_gate(self) -> None:
        # Run 1: only the sibling's record exists and the torrent is gone from
        # qBittorrent -> MISSING drop, no move. Run 2: the other entry's record
        # (a later-run re-grab) imports -> no remaining claim -> the move fires.
        sibling = pending_import(infohash="h", al_id=22, series_id=7, added_at=_FRESH)
        gone = CategoryQbit({})  # an unscripted hash polls as MISSING
        run1 = make_orchestration_manager(
            qbit=gone,
            strategy=_RecordingStrategy(),
            store_records=[sibling],
            post_import_category="pearlarr-done",
        )

        run1.snapshot_pending_for_series(7)

        assert run1._pending_records() == {}
        assert gone.set_category_calls == []

        revived = pending_import(infohash="h", al_id=11, series_id=7, added_at=_FRESH)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
            ),
            post_import_category="pearlarr-done",
        )
        run2.cache_store = run1.cache_store
        run2.cache_store.put_pending(Arr.SONARR, revived.key, revived.to_json())

        run2.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_later_run_re_registration_moves_again_idempotently(self) -> None:
        # A record re-created on a later run (a SeaDex update re-grabs the same
        # torrent) re-applies the move after its own import - idempotent in
        # qBittorrent, so a second move is harmless and correct.
        record = pending_import(infohash="h", series_id=7, added_at=_FRESH)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        imported = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        run1 = make_orchestration_manager(
            qbit=qbit,
            strategy=imported,
            store_records=[record],
            post_import_category="pearlarr-done",
        )

        run1.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]

        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
            ),
            post_import_category="pearlarr-done",
        )
        run2.cache_store = run1.cache_store
        run2.cache_store.put_pending(Arr.SONARR, record.key, record.to_json())

        run2.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")] * 2
        assert run2._pending_records() == {}


def _movie_import(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` import record keyed by the torrent's downloadId."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


def _radarr_reconcile_manager(
    *,
    qbit: CategoryQbit | None,
    history: list[HistoryRecord] | None,
    radarr_records: list[PendingImport] | None = None,
    sonarr_records: list[PendingImport] | None = None,
    store: FakeCacheStore | None = None,
    post_import_category: str | None = None,
) -> tuple[ImportWaitManager, FakeCacheStore]:
    """A manager wired for a Radarr run: a real `RadarrSync` reconciling off scripted history.

    The manager and the strategy share one durable store, so a record the
    reconcile drops is gone for the cross-arr category count. `history` is set on
    the client directly (None = a Radarr outage; `[]` = readable-but-empty). The
    `_ctx.arr` is RADARR, so the arr-scoped passes read the Radarr records.
    """

    store = store if store is not None else FakeCacheStore()
    config = make_config(post_import_category=post_import_category)
    radarr = FakeRadarrClient()
    radarr.history_since_return = history
    strat = make_radarr_sync(radarr=radarr, config=config, cache_store=store)
    mgr = make_bare_instance(
        ImportWaitManager,
        qbit=qbit,
        logger=make_logger(),
        _config=config,
        _active_strategy=strat,
        _reporter=_RecordingReporter(),
        cache_store=store,
    )
    mgr._ctx = RunContext(arr=Arr.RADARR)
    for record in radarr_records or []:
        store.put_pending(Arr.RADARR, record.key, record.to_json())
    for record in sonarr_records or []:
        store.put_pending(Arr.SONARR, record.key, record.to_json())
    return mgr, store


class TestCloseTrackedDownload:
    """close_tracked_download: the last imported record dismisses Sonarr's leftover queue entry.

    Sonarr auto-closes a tracked download only when ONE import covers the
    grab's full episode count, so a download finished across several passes
    stays parked in its queue where completed-download handling would
    re-import it. The engine asks the strategy to close it once no record of
    this arr still claims the torrent.
    """

    def test_reconcile_import_closes_the_queue_entry(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        pending = pending_import(infohash="h", series_id=7, added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store_records=[pending],
        )

        mgr.snapshot_pending_for_series(7)

        assert [c.key for c in strategy.close_calls] == [pending.key]

    def test_monitor_import_closes_the_queue_entry(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.RETRY, files_present=True),
        )
        pending = pending_import(infohash="h", added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
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

        assert view.final(rk("h")).outcome is Outcome.IMPORTED
        assert [c.key for c in strategy.close_calls] == [pending.key]

    def test_sibling_record_defers_the_close_to_the_last_import(self) -> None:
        # Run 1: Cour 1 imports but Cour 2 still claims the torrent -> no close
        # (a close now would mark the download ignored while Sonarr may still
        # need to import Cour 2's slice). Run 2: Cour 2 imports -> ONE close.
        first = pending_import(infohash="h", al_id=11, series_id=7, title="Cour 1", added_at=_FRESH)
        second = pending_import(infohash="h", al_id=22, series_id=7, title="Cour 2", added_at=_FRESH)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        run1_strategy = _RecordingStrategy(
            completed_sequence=[
                import_probe(ImportReadiness.IMPORTED, files_present=True),  # Cour 1
                import_probe(ImportReadiness.RETRY, files_present=False),  # Cour 2: not landed
            ],
        )
        run1 = make_orchestration_manager(qbit=qbit, strategy=run1_strategy, store_records=[first, second])

        run1.snapshot_pending_for_series(7)

        assert run1_strategy.close_calls == []
        assert set(run1._pending_records()) == {pk("h", 22)}

        run2_strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        run2 = make_orchestration_manager(qbit=qbit, strategy=run2_strategy)
        run2.cache_store = run1.cache_store  # the same durable store, next run

        run2.snapshot_pending_for_series(7)

        assert [c.key for c in run2_strategy.close_calls] == [second.key]

    def test_remove_from_queue_off_never_closes(self) -> None:
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
            remove_from_queue=False,
        )

        mgr.snapshot_pending_for_series(7)

        assert strategy.close_calls == []
        assert mgr._pending_records() == {}  # the import itself still drops

    def test_missing_record_is_not_closed(self) -> None:
        # MISSING also drops, but nothing was imported: the queue entry (if
        # any) stays Sonarr's to resolve.
        strategy = _RecordingStrategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),  # an unscripted hash polls as MISSING
            strategy=strategy,
            store_records=[pending_import(infohash="h", series_id=7, added_at=_FRESH)],
        )

        mgr.snapshot_pending_for_series(7)

        assert mgr._pending_records() == {}
        assert strategy.close_calls == []

    def test_radarr_sibling_does_not_hold_the_sonarr_close(self) -> None:
        # The gate is PER-ARR (unlike the cross-arr category gate): Sonarr's
        # queue entry only blocks Sonarr's completed-download handling, so a
        # Radarr record sharing the torrent must not keep the close window open.
        strategy = _RecordingStrategy(
            completed=import_probe(ImportReadiness.IMPORTED, files_present=True),
        )
        sonarr_record = pending_import(infohash="h", al_id=11, series_id=7, added_at=_FRESH)
        radarr_record = pending_import(infohash="h", al_id=22, series_id=0, added_at=_FRESH)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store_records=[sonarr_record],
        )
        mgr.cache_store.put_pending(Arr.RADARR, radarr_record.key, radarr_record.to_json())

        mgr.snapshot_pending_for_series(7)

        assert [c.key for c in strategy.close_calls] == [sonarr_record.key]


class TestRadarrReconcile:
    """Radarr records reconcile off import history: a matching event imports, else they wait."""

    def test_matching_event_drops_and_moves_category(self) -> None:
        # The torrent is complete in qBittorrent and Radarr's history shows the
        # import -> the record drops and the category move fires through the gate.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            radarr_records=[pending_import(infohash="h", series_id=0, added_at=_FRESH)],
            post_import_category="pearlarr-done",
        )

        mgr.reconcile_remaining()

        assert mgr._pending_records() == {}
        assert mgr._ctx.stats.imported == 1
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_no_event_keeps_record_and_defers_category(self) -> None:
        # Complete download, but Radarr hasn't imported it yet (empty history):
        # the record stays and the category is NOT moved.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[],
            radarr_records=[pending_import(infohash="h", series_id=0, added_at=_FRESH)],
            post_import_category="pearlarr-done",
        )

        mgr.reconcile_remaining()

        assert set(mgr._pending_records()) == {pk("h")}
        assert qbit.set_category_calls == []
        assert mgr._ctx.stats.imported == 0

    def test_history_outage_keeps_record_and_never_moves(self) -> None:
        # history_since -> None (Radarr down): fail-open, never move on no evidence.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=None,
            radarr_records=[pending_import(infohash="h", series_id=0, added_at=_FRESH)],
            post_import_category="pearlarr-done",
        )

        mgr.reconcile_remaining()

        assert set(mgr._pending_records()) == {pk("h")}
        assert qbit.set_category_calls == []

    def test_ttl_expiry_of_a_radarr_record_releases_the_gate(self) -> None:
        # Two Radarr records share one torrent; one aged out. Prune drops the
        # corpse, so the survivor's verified import still moves the category.
        survivor = pending_import(infohash="h", al_id=11, series_id=0, added_at=_FRESH)
        expired = pending_import(infohash="h", al_id=22, series_id=0, added_at=_EXPIRED)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            radarr_records=[survivor, expired],
            post_import_category="pearlarr-done",
        )

        mgr.prune_expired_pending()
        mgr.reconcile_remaining()

        assert mgr._pending_records() == {}
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_tally_counts_a_carried_over_radarr_record(self) -> None:
        # A complete-but-not-yet-imported Radarr record reconciles to IMPORTING and
        # the pre-summary tally folds it into the importing counter (no double poll).
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[],
            radarr_records=[pending_import(infohash="h", series_id=0, added_at=_FRESH)],
        )

        mgr.reconcile_remaining()
        mgr.tally_carried_over_into_stats()

        assert mgr._ctx.stats.importing == 1
        assert mgr._ctx.stats.queued == 0


class TestCrossArrCategoryGate:
    """A torrent shared by a Sonarr grab and a Radarr grab moves only after BOTH import."""

    def test_sonarr_import_defers_until_radarr_reconciles(self) -> None:
        # One torrent, two records: a Sonarr slice (imports first) and a Radarr
        # movie. The Sonarr import must NOT move the category while the Radarr
        # record still claims the hash; the later Radarr reconcile fires the move.
        store = FakeCacheStore()
        sonarr_record = pending_import(infohash="h", al_id=11, series_id=7, title="Series", added_at=_FRESH)
        radarr_record = pending_import(infohash="h", al_id=22, series_id=0, title="Movie", added_at=_FRESH)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})

        # Sonarr run: its slice verifies and drops, but the Radarr record still
        # claims the hash cross-arr -> the move is deferred.
        sonarr_mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(completed=import_probe(ImportReadiness.IMPORTED, files_present=True)),
            post_import_category="pearlarr-done",
        )
        sonarr_mgr.cache_store = store
        store.put_pending(Arr.SONARR, sonarr_record.key, sonarr_record.to_json())
        store.put_pending(Arr.RADARR, radarr_record.key, radarr_record.to_json())

        sonarr_mgr.snapshot_pending_for_series(7)

        assert qbit.set_category_calls == []  # the Radarr record still claims the hash
        assert set(store.get_pending(Arr.RADARR)) == {pk("h", 22)}

        # Radarr run: the movie import verifies -> no claim remains -> the move fires.
        radarr_mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            store=store,
            post_import_category="pearlarr-done",
        )

        radarr_mgr.reconcile_remaining()

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert store.get_pending(Arr.RADARR) == {}


def make_add_engine(
    *,
    torrents: FakeTorrents,
    strategy: _RecordingStrategy,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config_overrides: object,
) -> tuple[RunLoop, GrabPipeline]:
    """A bare `RunLoop` + a `GrabPipeline` + an attached `ImportWaitManager`.

    The produce side lives on `GrabPipeline` (held by the services hub in
    production) and the consume side on `ImportWaitManager`, so both are
    wired to the SAME `_ctx` / `cache_store` / client the engine holds - an
    add through the returned pipeline registers into exactly the state the
    manager's consume passes (`snapshot_pending_for_series` /
    `_monitor_working_set`) read back. `_active_strategy` is the test's
    recording `_RecordingStrategy` and `_reporter` a recording
    `_RecordingReporter` so the snapshot can be driven afterwards and asserted
    on recorded state.
    """

    engine = make_bare_instance(
        RunLoop,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(**config_overrides),
        _active_strategy=strategy,
        _reporter=_RecordingReporter(),
        cache_store=FakeCacheStore(),
    )
    engine._ctx = RunContext(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode)
    pipeline = make_grab_pipeline(
        _config=engine._config,
        _torrents=torrents,
        cache_store=engine.cache_store,
        qbit=qbit,
        _ctx=engine._ctx,
    )
    _attach_wait_manager(engine)
    return engine, pipeline


class TestRegisteredGrabSurvivesSnapshot:
    """A this-run grab registered by the pipeline is owned by the monitor, not re-polled."""

    def test_registered_already_added_survives_snapshot(self) -> None:
        # Integration: an ALREADY_ADDED registered THIS run is a this-run grab, so
        # the per-series snapshot skips it (no re-poll, no drop) and the end-of-run
        # monitor owns it via the working set.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show")})
        strategy = _RecordingStrategy()
        engine, pipeline = make_add_engine(torrents=torrents, strategy=strategy)
        seeds = {"h1": pending_import(infohash="h1", series_id=7, added_at=_FRESH)}

        pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )
        engine._wait_manager.snapshot_pending_for_series(7)

        assert strategy.import_calls == []
        assert set(engine._wait_manager._pending_records()) == {pk("h1")}
        assert [p.infohash for p in engine._wait_manager._monitor_working_set()] == ["h1"]
