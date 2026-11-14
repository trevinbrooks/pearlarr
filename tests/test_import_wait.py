# pyright: strict
# pyright: reportPrivateUsage=false
# These access the wait manager's + arr run loop's private members (mgr._reporter,
# mgr._pending_imports, mgr._cleanup, mgr._state, engine._wait_manager, ...). Strict
# re-flags that, and the repo disables reportPrivateUsage for tests.
"""Unit tests for the completion wait/poll machinery (`ImportWaitManager`).

These pin `WaitPollers.poll_torrent` (the single-shot state read)
and `ImportWaitManager.run_blocking` (the poll loop) against a scripted
`FakeQbit`. A `FakeClock` is injected at manager construction so the poll
loop never actually waits. Real foreground `sleep` is blocked in this env, so
the fakes are mandatory, not just a speed-up. The manager is built bare (`object.__new__`
via `make_bare_instance`) so no live qBittorrent login or disk I/O happens. The
arr run loop's `_finish_arr_run` orchestration (which drives the manager's wait passes) is
pinned via a real arr run loop with an attached manager at the bottom of the file.

Every collaborator the manager drives - the strategy's import hooks, the snapshot
reporter, qBittorrent - is a small typed fake recording what a test asserts, so
the contracts are pinned by recorded state.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast, override

import httpx
import pytest
import qbittorrentapi
import respx

from pearlarr.arr_categories import CategorySources
from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.grab_pipeline import GrabPipeline
from pearlarr.import_wait import ImportWaitManager, WaitPass, WaitPollers
from pearlarr.log import LOG_NAME
from pearlarr.output import SPARK_SAMPLES, Diagnostic, Phase, Severity, TorrentView, WaitKind, WaitSnapshot
from pearlarr.pending_import import OwnedAtGrab, PendingImport
from pearlarr.torrent_service import AddStatus
from pearlarr.wait_types import (
    NO_ATTEMPT,
    PENDING_IMPORT_STATE_FOR_OUTCOME,
    AttemptKind,
    AttemptResult,
    DownloadProgress,
    EffectStatus,
    ImportProgress,
    ImportWaitMode,
    Outcome,
    PauseReason,
    PendingImportState,
    TorrentPoll,
    TorrentStatus,
)
from pearlarr.wait_view import WaitOutcomeRow, WaitResult, WaitView

from .builders import (
    CLIENT_SENTINEL,
    SEP,
    FakeCacheStore,
    FakeTorrents,
    attempt_result,
    claim_anilist_ids,
    files_left_for_hand_import,
    grab_request,
    make_bare_instance,
    make_categories,
    make_config,
    make_fetched_categories,
    make_grab_pipeline,
    make_import_wait_manager,
    make_logger,
    make_radarr_sync,
    make_services,
    one_release_dict,
    order_only_hand_import_files,
    pending_import,
    pending_import_update,
    title_claim,
    two_claim_pending_import,
)
from .fakes import (
    CaptureHandler,
    FakeClock,
    FakeRadarrClient,
    FakeStrategy,
    bind_arr_http,
    diagnostic_messages,
    install_recording_hub,
)


class FakeStateEnum:
    """Mimics qBittorrent's `state_enum` (the `is_*` booleans the poll reads)."""

    def __init__(self, *, is_complete: bool = False, is_errored: bool = False) -> None:
        self.is_complete = is_complete
        self.is_errored = is_errored


class FakeTorrent:
    """Mimics one qBittorrent torrent info row (the fields `poll_torrent` reads).

    The download progress fields (`dlspeed` / `eta` / `completed` / `size`) default
    to None so the common blocking-wait tests don't have to set them. `poll_torrent`
    reads them via `getattr` and sanitizes None to None. `hash` is only read
    by the batched `poll_download_progress` (which keys its result off it). A blank
    hash is filled in from the script key when served from a `FakeQbit`
    download progress script, so only a deliberate-mismatch test needs to set it.
    `category` is read by the post-import move alone, and blank is an
    uncategorized torrent, as qBittorrent reports one.
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
        category: str = "",
    ) -> None:
        self.state_enum = FakeStateEnum(is_complete=is_complete, is_errored=is_errored)
        self.category = category
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
    """A scriptable qBittorrent client with per-hash, per-read scripts.

    `torrents` scripts the per-hash poll: each hash has its own
    ordered lifecycle of readings, and every single-hash `torrents_info` call
    advances THAT hash's script by one (clamping at the last, so a steady state
    repeats indefinitely). An unscripted hash reads as gone (`[]`). An
    Exception step is raised to exercise the transient-error path. `download_progress`
    separately scripts the batched download progress read the same way - a batch read never
    consumes a poll script (and vice versa), so a test scripts torrent
    lifecycles declaratively without caring which read comes first. Hashes with
    no download progress script simply don't appear in a batch. Hash matching is
    case-insensitive on both reads (real qBittorrent is). `calls` counts every
    `torrents_info` call, either read.
    """

    def __init__(
        self,
        torrents: dict[str, list[QbitStep]] | None = None,
        *,
        download_progress: dict[str, list[QbitStep]] | None = None,
    ) -> None:
        self._torrents = {h.casefold(): list(steps) for h, steps in (torrents or {}).items()}
        self._download_progress: dict[str, list[QbitStep]] = {}
        for h, steps in (download_progress or {}).items():
            for step in steps:
                # A blank row hash keys back to its script key, so common
                # download progress scripts don't have to repeat the hash.
                if isinstance(step, FakeTorrent) and not step.hash:
                    step.hash = h
            self._download_progress[h.casefold()] = list(steps)
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
            steps = self._download_progress.get(infohash.casefold())
            if steps is not None:
                rows.append(self._next(steps, self._batch_index, infohash.casefold()))
        return rows


class _InterruptOnHash(FakeQbit):
    """Raises KeyboardInterrupt when the poll reaches the given hash.

    Models a Ctrl-C landing mid-advance-loop, after earlier pending imports already advanced.
    """

    def __init__(self, torrents: dict[str, list[QbitStep]], *, interrupt_on: str) -> None:
        super().__init__(torrents)
        self._interrupt_on = interrupt_on.casefold()

    @override
    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[FakeTorrent]:
        if isinstance(torrent_hashes, str) and torrent_hashes.casefold() == self._interrupt_on:
            raise KeyboardInterrupt
        return super().torrents_info(torrent_hashes=torrent_hashes)


def _sdk_client(qbit: FakeQbit) -> qbittorrentapi.Client:
    """A fake as the SDK client it stands in for (the sanctioned leaf-client hatch: no interface to subclass)."""

    return cast("qbittorrentapi.Client", qbit)


def make_pollers(qbit: FakeQbit | None) -> WaitPollers:
    """Bare `WaitPollers` wired only with the `qbit` the polls read (the ctor takes a real client only)."""

    return make_bare_instance(WaitPollers, _qbit=qbit, _logger=make_logger(), strategy=None)


class TestPollTorrent:
    """poll_torrent turns a single qBittorrent read into a sanitized TorrentPoll."""

    def test_gone_on_empty_list(self) -> None:
        pollers = make_pollers(FakeQbit({}))

        assert pollers.poll_torrent("h") == TorrentPoll(TorrentStatus.GONE, None)

    def test_errored(self) -> None:
        pollers = make_pollers(FakeQbit({"h": [FakeTorrent(is_errored=True)]}))

        assert pollers.poll_torrent("h") == TorrentPoll(TorrentStatus.ERRORED, None)

    def test_complete_carries_content_path(self) -> None:
        torrent = FakeTorrent(is_complete=True, content_path="/data/show")
        pollers = make_pollers(FakeQbit({"h": [torrent]}))

        assert pollers.poll_torrent("h") == TorrentPoll(TorrentStatus.COMPLETE, "/data/show")

    def test_complete_on_full_progress_without_flag(self) -> None:
        # progress == 1.0 counts as complete even if the state flag is unset.
        torrent = FakeTorrent(progress=1.0, content_path="/data/movie")
        pollers = make_pollers(FakeQbit({"h": [torrent]}))

        assert pollers.poll_torrent("h") == TorrentPoll(
            TorrentStatus.COMPLETE,
            "/data/movie",
            DownloadProgress(1.0, None, None, None, None),
        )

    def test_none_while_downloading_carries_progress(self) -> None:
        pollers = make_pollers(FakeQbit({"h": [FakeTorrent(progress=0.5)]}))

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, DownloadProgress(0.5, None, None, None, None))

    def test_none_on_transient_api_error(self) -> None:
        # A dropped connection / re-auth in flight is "still waiting", not finished.
        # The un-observed flag tells the wait pass the zeroed download progress is a
        # placeholder (keep the last real bar), not a reading.
        pollers = make_pollers(FakeQbit({"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, observed=False)

    def test_none_when_no_client(self) -> None:
        pollers = make_pollers(None)

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, observed=False)

    def test_carries_live_download_progress(self) -> None:
        torrent = FakeTorrent(
            progress=0.64,
            dlspeed=3_200_000,
            eta=130,
            completed=1_800_000_000,
            size=2_900_000_000,
        )
        pollers = make_pollers(FakeQbit({"h": [torrent]}))

        assert pollers.poll_torrent("h") == TorrentPoll(
            None,
            None,
            DownloadProgress(0.64, 3_200_000, 130, 1_800_000_000, 2_900_000_000),
        )

    def test_sanitizes_junk_download_progress(self) -> None:
        # qB's "∞" eta sentinel, an idle (0) speed and a zero size all sanitize to
        # None so the wait view never renders a nonsense countdown / negative bar.
        torrent = FakeTorrent(progress=0.5, dlspeed=0, eta=8_640_000, completed=0, size=0)
        pollers = make_pollers(FakeQbit({"h": [torrent]}))

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, DownloadProgress(0.5, None, None, None, None))


class TestPollDownloadProgress:
    """poll_download_progress: ONE batched, read-only info call for the wait region refresh."""

    def test_batches_and_matches_hashes_case_insensitively(self) -> None:
        # qBittorrent lowercases response hashes. The result must key back to PEARLARR's
        # hash spelling so the wait pass's views dict matches.
        row = FakeTorrent(progress=0.5, dlspeed=200, completed=100, size=1000)
        qbit = FakeQbit(download_progress={"ABC123": [row]})  # blank row hash fills in as "ABC123"
        pollers = make_pollers(qbit)

        progress = pollers.poll_download_progress(["abc123"])

        assert qbit.calls == 1  # one call covers the whole set
        assert progress == {"abc123": DownloadProgress(0.5, 200, None, 100, 1000)}

    def test_one_call_covers_many_hashes(self) -> None:
        qbit = FakeQbit(
            download_progress={
                "A1": [FakeTorrent(progress=0.2, dlspeed=10)],
                "B2": [FakeTorrent(progress=0.8, dlspeed=20)],
            },
        )
        pollers = make_pollers(qbit)

        progress = pollers.poll_download_progress(["a1", "b2"])

        assert qbit.calls == 1  # genuinely batched, not per-hash
        assert set(progress) == {"a1", "b2"}

    def test_empty_set_makes_no_call(self) -> None:
        qbit = FakeQbit({})
        pollers = make_pollers(qbit)

        assert pollers.poll_download_progress([]) == {}
        assert qbit.calls == 0

    def test_transient_error_yields_nothing(self) -> None:
        # The rows just keep their last download progress until the next poll.
        pollers = make_pollers(FakeQbit(download_progress={"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert pollers.poll_download_progress(["h"]) == {}

    def test_unasked_rows_are_ignored(self) -> None:
        # The batch answers for "h" with a row whose hash reads back "other" -
        # poll_download_progress must drop what it can't key back to an asked hash.
        pollers = make_pollers(FakeQbit(download_progress={"h": [FakeTorrent(torrent_hash="other", progress=0.4)]}))

        assert pollers.poll_download_progress(["h"]) == {}


class _AutoTickClock(FakeClock):
    """Time passing between reads: every `now` reading advances by `step`."""

    @override
    def monotonic(self) -> float:
        t = self.t
        self.t += self._step
        return t


class _InterruptSleepClock(FakeClock):
    """A clock whose sleep raises KeyboardInterrupt (Ctrl-C mid-wait)."""

    @override
    def sleep(self, seconds: float) -> None:
        raise KeyboardInterrupt


class _StopSleepClock(FakeClock):
    """A clock whose sleep raises `SystemExit` (a SIGTERM stop mid-wait)."""

    @override
    def sleep(self, seconds: float) -> None:
        raise SystemExit(0)


def _two_claim_pending_import(**overrides: Any) -> PendingImport:
    """A stored pending import on hash "h" claimed by two titles with no window (multi-cour), each named by its cour."""

    fields: dict[str, Any] = {"infohash": "h"}
    fields.update(overrides)
    return two_claim_pending_import(anilist_ids=(11, 22), windows=((), ()), titles=("Cour 1", "Cour 2"), **fields)


@dataclass(frozen=True)
class _ImportCall:
    """One recorded `import_completed` call: its pending import/path + attempt kind."""

    pending: PendingImport
    content_path: str
    attempt_kind: AttemptKind


class _RecordingStrategy(FakeStrategy):
    """A `FakeStrategy` that records + scripts the two import hooks the manager drives.

    `import_completed` records each call's attempt kind (asserted on
    `import_calls`) and dispenses a scripted `AttemptResult` - a single
    `completed` repeated, a `completed_sequence` advanced per call (clamped to its
    last), or a `completed_error` raised (the swallowed-import path).
    `import_progress` likewise records (`progress_calls`) and dispenses an
    `ImportProgress`, defaulting to an indeterminate zero - the
    progress-poll no-op the poll tests rely on. `progress_error` is raised
    ONCE on the first call (the progress-poll containment path), then cleared.
    `remove_from_queue` records (`close_calls`) and dispenses a single
    `close_return` or a `close_sequence` advanced per call (clamped to its last).
    """

    def __init__(
        self,
        *,
        completed: AttemptResult | None = None,
        completed_sequence: list[AttemptResult] | None = None,
        completed_error: Exception | None = None,
        progress: ImportProgress | None = None,
        progress_sequence: list[ImportProgress] | None = None,
        progress_error: Exception | None = None,
        close_return: EffectStatus = EffectStatus.DONE,
        close_sequence: list[EffectStatus] | None = None,
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
        self._close_return = close_return
        self._close_sequence = close_sequence
        self._close_index = 0
        self.import_calls: list[_ImportCall] = []
        self.progress_calls: list[PendingImport] = []
        self.close_calls: list[PendingImport] = []

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt_kind: AttemptKind,
    ) -> AttemptResult:
        self.import_calls.append(_ImportCall(pending, content_path, attempt_kind))
        if self._completed_error is not None:
            raise self._completed_error
        if self._completed_sequence is not None:
            idx = min(self._completed_index, len(self._completed_sequence) - 1)
            self._completed_index += 1
            return self._completed_sequence[idx]
        if self._completed is not None:
            return self._completed
        return NO_ATTEMPT

    @override
    def remove_from_queue(self, pending: PendingImport) -> EffectStatus:
        self.close_calls.append(pending)
        if self._close_sequence is not None:
            idx = min(self._close_index, len(self._close_sequence) - 1)
            self._close_index += 1
            return self._close_sequence[idx]
        return self._close_return

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        self.progress_calls.append(pending)
        if self._progress_error is not None:
            error = self._progress_error
            self._progress_error = None  # one-shot: the next progress poll succeeds
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
    """One recorded `log_pending_import` call's reported fields."""

    state: PendingImportState
    title: str
    coverage: str | None
    url: str | None


class _RecordingReporter:
    """Records `log_pending_import` calls so the inline-snapshot/no-double-report contracts are checkable."""

    def __init__(self) -> None:
        self.snapshot_calls: list[_SnapshotCall] = []

    def log_pending_import(self, state: PendingImportState, pending: PendingImport, series_id: int) -> None:
        claim = pending.claim_for(series_id)
        self.snapshot_calls.append(_SnapshotCall(state, pending.display_label, claim.window_coverage, claim.url))


def make_orchestration_manager(
    *,
    qbit: FakeQbit | None,
    strategy: _RecordingStrategy,
    stored_pending_imports: list[PendingImport] | None = None,
    pending: list[PendingImport] | None = None,
    reporter: _RecordingReporter | None = None,
    clock: FakeClock | None = None,
    store: FakeCacheStore | None = None,
    **config_overrides: object,
) -> ImportWaitManager:
    """An `ImportWaitManager` wired for the pending-import orchestration paths.

    Sets up the durable per-arr store (`store`, a fresh fake unless a test shares one
    across runs) and the run list (`pending`, keyed into `_state.grabbed_this_run`) so
    `report_imports_for_series`, `run_deferred` and `run_blocking` can be driven
    without a live Sonarr/qBittorrent. The strategy is a recording
    `_RecordingStrategy` (its import hooks scripted per test) and the reporter a
    recording `_RecordingReporter` (a test that asserts on the snapshot reporter
    passes in its own `reporter` to read it back).
    """

    cache_store = store if store is not None else FakeCacheStore()
    mgr = make_import_wait_manager(
        qbit=qbit,
        cache_store=cache_store,
        reporter=reporter or _RecordingReporter(),
        clock=clock if clock is not None else FakeClock(),
        strategy=strategy,
        state=ArrRunState(arr=Arr.SONARR, grabbed_this_run={p.infohash: p for p in pending or []}),
        **config_overrides,
    )
    for stored in stored_pending_imports or []:
        cache_store.put_pending_import_row(Arr.SONARR, stored.infohash, stored.to_json())
    return mgr


class TestPendingImportsGuardLoading:
    """The bug-fix pin: every claim of one title loads the SAME guard row.

    Claims of one title on torrents grabbed across different runs used to carry
    divergent frozen copies. The title's single `guard_facts` row now feeds them all.
    """

    def test_claims_share_the_title_row_and_others_stay_empty(self) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            store=store,
            stored_pending_imports=[
                pending_import(infohash="run1", anilist_id=5),
                pending_import(infohash="run2", anilist_id=5),
                pending_import(infohash="other", anilist_id=6),
            ],
        )
        facts = OwnedAtGrab(entry_groups=("NewPick",), stale_groups=("OldPick",))
        store.put_guards(Arr.SONARR, 5, facts)

        pending_imports = mgr._pending_imports.active_imports()
        assert [c.guards for c in pending_imports["run1"].claims] == [facts]
        assert [c.guards for c in pending_imports["run2"].claims] == [facts]
        # No row = empty facts, the designed state (Radarr, degradation).
        assert [c.guards for c in pending_imports["other"].claims] == [OwnedAtGrab()]


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


class TestReportImportsForSeries:
    """report_imports_for_series reports CARRIED-OVER pending imports inline, no double-report."""

    def test_carried_over_imported_drops_and_classifies(self) -> None:
        # A prior run's download finished with every file done: the
        # pending import is reported imported, dropped, and classified IMPORTED for the
        # finalize tally.
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.report_imports_for_series(7)

        assert mgr._pending_imports.rows() == {}
        assert mgr._state.pending_import_states["h"] is PendingImportState.IMPORTED
        # Read-only report: no import attempt, and the `imported` bump belongs to
        # _finish_arr_run (it counts the IMPORTED state).
        assert strategy.import_calls == []
        assert mgr._state.stats.imported == 0
        # The pending import is reported inline with its reconciled state + title (the gap a
        # plain `snapshot_calls != []` check left open: wrong state/title slid through).
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingImportState.IMPORTED
        # The inline row is labeled title · group (the group disambiguates a
        # series that grabbed several torrents).
        assert reporter.snapshot_calls[0].title == f"Show{SEP}SubGroup"

    def test_carried_over_unfinished_torrent_is_downloading_and_kept(self) -> None:
        # Still downloading -> DOWNLOADING, pending import kept, no import attempt.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(progress=0.5)]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.report_imports_for_series(7)

        assert strategy.import_calls == []
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADING
        assert mgr._state.stats.imported == 0
        # The carried-over pending import is still reported inline, with the DOWNLOADING state.
        assert len(reporter.snapshot_calls) == 1
        assert reporter.snapshot_calls[0].state is PendingImportState.DOWNLOADING
        assert reporter.snapshot_calls[0].title == f"Show{SEP}SubGroup"

    def test_this_run_grab_is_skipped_no_double_report(self) -> None:
        # REGRESSION (double-report): a torrent grabbed THIS run lives in
        # _state.grabbed_this_run AND the store. The report must skip it entirely -
        # no poll, no row, no counter, no state - so it's only ever `added`.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        this_run = pending_import(infohash="h", series_id=7)
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[this_run],
            pending=[this_run],
        )

        mgr.report_imports_for_series(7)

        assert strategy.import_calls == []
        assert reporter.snapshot_calls == []
        assert mgr._state.pending_import_states == {}
        assert mgr._state.stats.imported == 0
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_reacquired_pending_import_is_skipped(self) -> None:
        # A stored pending import re-seen in qBittorrent this run was already reported by
        # its title block: no poll, no row, no state. Only the tally counts it.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        mgr._state.reacquired_keys = {"h"}

        mgr.report_imports_for_series(7)

        assert qbit.calls == 0
        assert reporter.snapshot_calls == []
        assert mgr._state.pending_import_states == {}
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_other_series_pending_import_is_not_touched(self) -> None:
        # The report is series-scoped: a pending import for a different series is left
        # alone (the deferred or blocking wait pass handles it later).
        strategy = _RecordingStrategy()
        qbit = FakeQbit({"other": [FakeTorrent(progress=0.5)]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="other", series_id=99)],
        )

        mgr.report_imports_for_series(7)

        assert "other" not in mgr._state.pending_import_states
        assert set(mgr._pending_imports.rows()) == {"other"}

    @pytest.mark.parametrize(
        ("scan_order", "coverage", "url"),
        [
            pytest.param((7, 8), "S01 E01-E12", "https://releases.moe/11", id="first-series-scans-first"),
            pytest.param((8, 7), "S02 E01-E12", "https://releases.moe/22", id="second-series-scans-first"),
        ],
    )
    def test_two_series_pending_import_is_reported_once_under_the_first_scan(
        self,
        scan_order: tuple[int, int],
        coverage: str,
        url: str,
    ) -> None:
        # One torrent claimed by titles on two series: observed and reported ONCE,
        # under whichever series scans first, with THAT claim's coverage and url.
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(progress=0.5)]})
        pending = pending_import(
            infohash="h",
            claims=(
                title_claim(anilist_id=11, series_id=7, window_coverage="S01 E01-E12", url="https://releases.moe/11"),
                title_claim(anilist_id=22, series_id=8, window_coverage="S02 E01-E12", url="https://releases.moe/22"),
            ),
        )
        mgr = make_orchestration_manager(
            qbit=qbit, strategy=strategy, reporter=reporter, stored_pending_imports=[pending]
        )

        for series_id in scan_order:
            mgr.report_imports_for_series(series_id)

        assert qbit.calls == 1
        assert mgr._state.pending_import_states == {"h": PendingImportState.DOWNLOADING}
        assert [(c.state, c.coverage, c.url) for c in reporter.snapshot_calls] == [
            (PendingImportState.DOWNLOADING, coverage, url),
        ]

    def test_complete_without_content_path_stays_importing(self) -> None:
        # MUTATION PIN (_reconcile_one): COMPLETE with an empty content_path must
        # NOT attempt an import (`and` -> `or`) and must classify IMPORTING, kept -
        # never IMPORTED/dropped (the default attempt result's all_done=False -> True).
        strategy = _RecordingStrategy()
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True)]})  # no content_path
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.report_imports_for_series(7)

        assert strategy.import_calls == []
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADED
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert mgr._state.stats.imported == 0
        assert [c.state for c in reporter.snapshot_calls] == [PendingImportState.DOWNLOADED]


class TestGoneTorrentHasItsTitlesCheckedAgain:
    """Dropping a pending import whose torrent left qBittorrent clears its titles' stamps: the next run checks them."""

    _CACHED_ON = "2021-06-05 00:00:00"

    @classmethod
    def _store(cls) -> FakeCacheStore:
        """Titles 5, 6 and 9 cached."""

        store = FakeCacheStore()
        for anilist_id in (5, 6, 9):
            store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": cls._CACHED_ON})
        return store

    @staticmethod
    def _record() -> PendingImport:
        """Torrent "h", claimed by title 5 on series 7 and title 6 on series 8."""

        return two_claim_pending_import(anilist_ids=(5, 6), series_ids=(7, 8), infohash="h")

    @staticmethod
    def _stamps(store: FakeCacheStore) -> dict[int, str | None]:
        cached_titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (5, 6, 9)}
        return {anilist_id: cached.updated_at for anilist_id, cached in cached_titles.items() if cached is not None}

    def test_the_scan_report_clears_the_stamp_of_every_claiming_title(self) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}), strategy=_RecordingStrategy(), store=store, stored_pending_imports=[self._record()]
        )

        mgr.report_imports_for_series(7)

        assert mgr._pending_imports.rows() == {}
        # Title 9 never claimed the torrent.
        assert self._stamps(store) == {5: None, 6: None, 9: self._CACHED_ON}

    def test_the_end_pass_clears_the_stamp_of_every_claiming_title(self) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}), strategy=_RecordingStrategy(), store=store, stored_pending_imports=[self._record()]
        )

        mgr.run_deferred(view=RecordingWaitView())

        assert mgr._pending_imports.rows() == {}
        assert self._stamps(store) == {5: None, 6: None, 9: self._CACHED_ON}

    def test_an_imported_pending_import_leaves_its_titles_cached(self) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            store=store,
            stored_pending_imports=[self._record()],
        )

        mgr.report_imports_for_series(7)

        assert mgr._pending_imports.rows() == {}
        assert self._stamps(store) == dict.fromkeys((5, 6, 9), self._CACHED_ON)

    def test_the_end_pass_leaves_an_imported_pending_imports_titles_cached(self) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(completed=attempt_result(all_done=True)),
            store=store,
            stored_pending_imports=[self._record()],
        )

        mgr.run_deferred(view=RecordingWaitView())

        assert mgr._pending_imports.rows() == {}
        assert self._stamps(store) == dict.fromkeys((5, 6, 9), self._CACHED_ON)

    def test_the_other_arrs_row_keeps_its_stamp(self) -> None:
        store = self._store()
        store.update_cached_title(Arr.RADARR, 5, {"updated_at": self._CACHED_ON})
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}), strategy=_RecordingStrategy(), store=store, stored_pending_imports=[self._record()]
        )

        mgr.report_imports_for_series(7)

        radarr_row = store.get_cached_title(Arr.RADARR, 5)
        assert radarr_row is not None
        assert radarr_row.updated_at == self._CACHED_ON

    def test_a_confirmed_import_whose_torrent_is_gone_leaves_its_titles_cached(self) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),
            strategy=_RecordingStrategy(),
            store=store,
            stored_pending_imports=[replace(self._record(), awaiting_cleanup=True)],
        )

        mgr._cleanup.finish_cleanups()

        assert mgr._pending_imports.rows() == {}
        assert self._stamps(store) == dict.fromkeys((5, 6, 9), self._CACHED_ON)


class TestRunDeferred:
    """run_deferred runs ONE non-blocking poll over every carried-over pending import."""

    def test_imports_ready_carried_over_pending_import(self) -> None:
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 1
        assert mgr._pending_imports.rows() == {}
        # The `imported` bump belongs to _finish_arr_run, off the result rows.
        assert mgr._state.stats.imported == 0

    def test_skips_this_run_grabs(self) -> None:
        strategy = _RecordingStrategy()
        this_run = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[this_run],
            pending=[this_run],
        )

        assert mgr.run_deferred(view=RecordingWaitView()) is None
        assert strategy.import_calls == []

    def test_two_imports_both_reported(self) -> None:
        # MUTATION PIN: BOTH carried-over imports must land on the result rows
        # (_finish_arr_run's `imported` bump reads them).
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
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
            stored_pending_imports=[
                pending_import(infohash="h1"),
                pending_import(infohash="h2"),
            ],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 2
        assert mgr._pending_imports.rows() == {}

    @pytest.mark.parametrize(
        ("torrent", "make_strategy", "outcome", "state"),
        [
            pytest.param(
                FakeTorrent(progress=0.5),
                _RecordingStrategy,
                Outcome.STILL_DOWNLOADING,
                PendingImportState.DOWNLOADING,
                id="still-downloading",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                lambda: _RecordingStrategy(completed=attempt_result(all_done=False)),
                Outcome.AWAITING_IMPORT,
                PendingImportState.DOWNLOADED,
                id="awaiting-import",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                lambda: _RecordingStrategy(completed_error=RuntimeError("boom")),
                Outcome.ATTEMPT_FAILED,
                PendingImportState.DOWNLOADED,
                id="attempt-failed",
            ),
            pytest.param(
                FakeTorrent(is_errored=True),
                _RecordingStrategy,
                Outcome.DOWNLOAD_ERRORED,
                PendingImportState.ERRORED,
                id="download-errored",
            ),
        ],
    )
    def test_check_outcome_refreshes_pending_import_state(
        self,
        torrent: FakeTorrent,
        make_strategy: Callable[[], _RecordingStrategy],
        outcome: Outcome,
        state: PendingImportState,
    ) -> None:
        # The deferred wait pass re-observes even a reported pending import: its outcome overwrites
        # the outdated state, so the tally buckets what the wait pass saw last.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=make_strategy(),
            stored_pending_imports=[pending_import(infohash="h")],
        )
        mgr._state.pending_import_states["h"] = PendingImportState.DOWNLOADING

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(outcome, True)]
        assert mgr._state.pending_import_states["h"] is state
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_check_never_applies_the_download_timeout(self) -> None:
        # The download clock starts at wait pass construction, so its elapsed says
        # nothing about the torrent: even with time passing between reads and a
        # tiny imports.download_timeout, a still-downloading pending import finishes
        # STILL_DOWNLOADING, never DOWNLOAD_TIMED_OUT.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.5)]}),
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="h")],
            download_timeout=1,
            clock=_AutoTickClock(step=30),
        )
        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.STILL_DOWNLOADING, True)]
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADING
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_check_progress_poll_confirms_within_the_single_poll(self) -> None:
        # The one deferred poll owns the cheap bar refresh too: the poll
        # issues the command, the progress read confirms every file, and the
        # pending import finishes IMPORTED instead of "import in progress".
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True),
            progress=ImportProgress(2, 2, determinate=True),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 1
        assert len(strategy.progress_calls) == 1
        assert mgr._pending_imports.rows() == {}


class TestPendingImportStateConversion:
    """The pinned Outcome -> PendingImportState conversion a wait pass classifies stored pending imports through."""

    @pytest.mark.parametrize(
        ("outcome", "state"),
        [
            (Outcome.STILL_DOWNLOADING, PendingImportState.DOWNLOADING),
            (Outcome.DOWNLOAD_TIMED_OUT, PendingImportState.DOWNLOADING),
            (Outcome.QBITTORRENT_UNREACHABLE, PendingImportState.DOWNLOADING),
            (Outcome.AWAITING_IMPORT, PendingImportState.DOWNLOADED),
            (Outcome.IMPORT_IN_PROGRESS, PendingImportState.DOWNLOADED),
            (Outcome.STILL_IMPORTING, PendingImportState.DOWNLOADED),
            (Outcome.SONARR_BUSY, PendingImportState.DOWNLOADED),
            (Outcome.NOT_READY, PendingImportState.DOWNLOADED),
            (Outcome.NOTHING_PLACED, PendingImportState.DOWNLOADED),
            (Outcome.PARTIAL, PendingImportState.DOWNLOADED),
            (Outcome.ATTEMPT_FAILED, PendingImportState.DOWNLOADED),
            (Outcome.NO_CONTENT_PATH, PendingImportState.DOWNLOADED),
            (Outcome.DOWNLOAD_ERRORED, PendingImportState.ERRORED),
        ],
    )
    def test_each_non_dropped_outcome_lands_in_its_bucket(self, outcome: Outcome, state: PendingImportState) -> None:
        mgr = make_orchestration_manager(qbit=None, strategy=_RecordingStrategy())

        mgr.note_pending_import_state("h", outcome)

        assert mgr._state.pending_import_states["h"] is state

    def test_conversion_covers_exactly_the_non_dropped_outcomes(self) -> None:
        # IMPORTED and MISSING leave the store, so the conversion never sees them.
        assert set(PENDING_IMPORT_STATE_FOR_OUTCOME) == {o for o in Outcome if not o.dropped}


class TestUnmatchedRowFinish:
    """A result naming unplaced files finishes on the poll that returned it, in either wait mode, and stays stored."""

    _NAMES = ("a.mkv", "b.mkv")

    @classmethod
    def _strategy(cls) -> _RecordingStrategy:
        return _RecordingStrategy(completed=attempt_result(all_done=False, unmatched_files=cls._NAMES))

    def test_run_deferred_finishes_unmatched_and_keeps_the_pending_import(self) -> None:
        strategy = self._strategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
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
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADED
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_run_blocking_finishes_unmatched_on_the_first_poll(self) -> None:
        # The result ends the wait on the poll that establishes it: no final attempt, no import clock.
        strategy = self._strategy()
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=600,
        )

        final = view.final("h")
        assert final.outcome is Outcome.NOTHING_PLACED
        assert final.left_for_hand_import.names == self._NAMES
        assert final.phase_elapsed_s < 600
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY]
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_later_polls_never_re_poll_a_finished_row(self) -> None:
        # A slower sibling keeps the wait pass running past the first poll. The finished row is not asked again.
        strategy = self._strategy()
        qbit = FakeQbit(
            {
                "h": [FakeTorrent(is_complete=True, content_path="/d")],
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        pending_imports = [pending_import(infohash="h"), pending_import(infohash="slow")]
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=pending_imports,
            pending=pending_imports,
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert [call.pending.infohash for call in strategy.import_calls] == ["h", "slow"]
        assert view.final("h").outcome is Outcome.NOTHING_PLACED
        assert view.final("slow").outcome is Outcome.NOTHING_PLACED
        assert set(mgr._pending_imports.rows()) == {"h", "slow"}


class TestLeftForHandImportRowFinish:
    """A result naming hand-import files finishes as a partial import: kept for the hand import, never cleaned up."""

    _LEFT = order_only_hand_import_files()

    def _run(self, store: FakeCacheStore, qbit: FakeQbit) -> tuple[ImportWaitManager, WaitResult | None]:
        """One run's deferred wait pass over the shared store, every poll naming the hand-import files."""

        strategy = _RecordingStrategy(completed=attempt_result(all_done=False, left_for_hand_import=self._LEFT))
        mgr = make_orchestration_manager(
            qbit=qbit, strategy=strategy, store=store, post_import_category="pearlarr-done"
        )
        return mgr, mgr.run_deferred(view=RecordingWaitView())

    def test_each_run_names_the_hand_import_files_and_leaves_the_torrent_in_its_category(self) -> None:
        store = FakeCacheStore()
        store.put_pending_import_row(Arr.SONARR, "h", pending_import(infohash="h").to_json())
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})

        _, first = self._run(store, qbit)
        mgr, second = self._run(store, qbit)

        for result in (first, second):
            assert result is not None
            assert [(row.outcome, row.left_for_hand_import) for row in result.rows] == [(Outcome.PARTIAL, self._LEFT)]
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADED
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert qbit.set_category_calls == []


class TestBlockingWaitRowDiscriminator:
    """Wait rows carry `carried_over`, so _finish_arr_run can tell them from fresh grabs."""

    def test_wait_rows_tell_fresh_from_carried_over(self) -> None:
        # One fresh grab + one carried-over pending import, both import: only the
        # carried-over row is marked, even though the fresh import leaves
        # `grabbed_this_run` mid-pass.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        fresh = pending_import(infohash="f", title="Fresh")
        carried = pending_import(infohash="c", title="Carried")
        qbit = FakeQbit(
            {
                "f": [FakeTorrent(is_complete=True, content_path="/f")],
                "c": [FakeTorrent(is_complete=True, content_path="/c")],
            },
        )
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[fresh, carried],
            pending=[fresh],
            clock=FakeClock(step=30),
        )

        result = mgr.run_blocking(view=RecordingWaitView())

        assert result is not None
        assert {row.label: row.carried_over for row in result.rows} == {
            f"Fresh{SEP}SubGroup": False,
            f"Carried{SEP}SubGroup": True,
        }
        # Imported rows leave the store, so neither converts to a pending import state, and
        # the `imported` bump stays _finish_arr_run's.
        assert mgr._state.pending_import_states == {}
        assert mgr._state.stats.imported == 0

    def test_fresh_timeout_row_is_not_converted(self) -> None:
        # A fresh grab's finished state never enters the carried-over states.
        fresh = pending_import(infohash="f")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"f": [FakeTorrent(progress=0.1)]}),
            strategy=_RecordingStrategy(),
            stored_pending_imports=[fresh],
            pending=[fresh],
            download_timeout=1,
            clock=FakeClock(step=30),
        )

        result = mgr.run_blocking(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.DOWNLOAD_TIMED_OUT, False)]
        assert mgr._state.pending_import_states == {}


class TestTallyCarriedOverIntoStats:
    """tally_carried_over_into_stats counts each still-waiting pending import once."""

    def test_counts_states_defaults_downloading_and_skips_errored(self) -> None:
        # MUTATION PIN: both counters accumulate (a `+=` turned into `=` clamps
        # at one), the unobserved pending import defaults to DOWNLOADING, and ERRORED is
        # deliberately uncounted.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[
                pending_import(infohash="q"),
                pending_import(infohash="d1"),
                pending_import(infohash="d2"),
                pending_import(infohash="e"),
                pending_import(infohash="untouched"),
            ],
        )
        mgr._state.pending_import_states = {
            "q": PendingImportState.DOWNLOADING,
            "d1": PendingImportState.DOWNLOADED,
            "d2": PendingImportState.DOWNLOADED,
            "e": PendingImportState.ERRORED,
        }

        mgr.tally_carried_over_into_stats()

        assert mgr._state.stats.downloading == 2  # explicit q + defaulted untouched
        assert mgr._state.stats.downloaded == 2

    def test_excludes_this_run_grabs(self) -> None:
        this_run = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[this_run],
            pending=[this_run],
        )

        mgr.tally_carried_over_into_stats()

        assert mgr._state.stats.downloading == 0
        assert mgr._state.stats.downloaded == 0

    def test_counts_reacquired_pending_imports(self) -> None:
        # A reacquired pending import is skipped by the inline report but still counts
        # into the tally under downloading/downloaded like any carried-over one.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[
                pending_import(infohash="rq"),
                pending_import(infohash="rd"),
            ],
        )
        mgr._state.reacquired_keys = {"rq", "rd"}
        mgr._state.pending_import_states = {"rd": PendingImportState.DOWNLOADED}

        mgr.tally_carried_over_into_stats()

        assert mgr._state.stats.downloading == 1
        assert mgr._state.stats.downloaded == 1

    def test_cleanup_rows_are_counted_nowhere(self) -> None:
        # The raw awaiting_cleanup flag alone skips a row (no pending_import_states entry
        # needed): the no-phantom invariant holds locally, without any distant state write.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="flagged", awaiting_cleanup=True)],
        )

        mgr.tally_carried_over_into_stats()

        assert mgr._state.stats.downloading == 0
        assert mgr._state.stats.downloaded == 0


class TestBlockingWaitWorkingSet:
    """_imports_to_wait_on dedups per hash, the in-memory pending import winning."""

    def test_in_memory_pending_import_wins_the_store_collision(self) -> None:
        # MUTATION PIN: the first loop's `not in seen` flipped to `in seen` skips
        # every this-run grab, so the store's (older) copy would be waited on
        # instead. One hash in BOTH places, differing by title: exactly one
        # pending import survives and it is the in-memory one.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="h", title="StoreCopy")],
            pending=[pending_import(infohash="h", title="InMemory")],
        )

        pending_imports = mgr._imports_to_wait_on()

        assert [c.title for p in pending_imports for c in p.claims] == ["InMemory"]

    def test_one_torrent_claimed_twice_is_one_row(self) -> None:
        # Two AniList titles share one torrent: ONE pending import carrying both claims,
        # so the working set holds one row labeled by every claim.
        pending = _two_claim_pending_import()
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
        )

        pending_imports = mgr._imports_to_wait_on()

        assert [p.display_label for p in pending_imports] == [f"Cour 1 & Cour 2{SEP}SubGroup"]
        # The store round-trip keeps both claims on the one row.
        rehydrated = mgr._pending_imports.load(mgr._pending_imports.rows())
        assert {key: claim_anilist_ids(p) for key, p in rehydrated.items()} == {"h": (11, 22)}


def _run_blocking_wait(
    strategy: _RecordingStrategy,
    qbit: FakeQbit,
    *,
    step: float = 30,
    download_timeout: int = 3600,
    import_timeout: int = 600,
    progress_poll_interval: int = 5,
    post_import_category: str | None = None,
) -> tuple[ImportWaitManager, RecordingWaitView]:
    """Drive `run_blocking` to completion over one fresh-grab pending import on hash "h".

    The pending import is both a this-run grab and stored (the common blocking-wait
    shape). The poll interval is fixed at 30; the knob defaults mirror
    the plain-config shape the class's tests share.
    """

    pending = pending_import(infohash="h")
    mgr = make_orchestration_manager(
        qbit=qbit,
        strategy=strategy,
        stored_pending_imports=[pending],
        pending=[pending],
        download_timeout=download_timeout,
        import_timeout=import_timeout,
        import_poll_interval=30,
        progress_poll_interval=progress_poll_interval,
        post_import_category=post_import_category,
        clock=FakeClock(step=step),
    )
    view = RecordingWaitView()
    mgr.run_blocking(view=view)
    return mgr, view


class _ExplodingDropStore(FakeCacheStore):
    """A store whose pending import drop always fails, for the finished-ordering pin."""

    @override
    def drop_pending_import_row(self, arr: Arr, infohash: str) -> None:
        raise RuntimeError("drop boom")


class _ExplodingPutStore(FakeCacheStore):
    """A store whose pending import upsert always fails, for the keep-path ordering pin."""

    @override
    def put_pending_import_row(self, arr: Arr, infohash: str, row: dict[str, Any]) -> None:
        raise RuntimeError("put boom")


class TestRunBlocking:
    """run_blocking: interleaved, copy-aware wait+import over ALL pending imports."""

    def test_interleaved_fast_and_slow(self) -> None:
        # Two torrents: "fast" completes + imports on the first poll (files done).
        # "slow" is still downloading, then completes + imports on a later poll. Both
        # advance each poll (interleaved), so the fast one isn't stuck behind slow.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        qbit = FakeQbit(
            {
                "fast": [FakeTorrent(is_complete=True, content_path="/fast")],
                # slow: downloading on the first poll, complete after.
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        fast = pending_import(infohash="fast")
        slow = pending_import(infohash="slow")
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[fast, slow],
            pending=[fast, slow],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        # Both ultimately imported and dropped.
        assert mgr._pending_imports.rows() == {}
        assert view.final("fast").outcome is Outcome.IMPORTED
        assert view.final("slow").outcome is Outcome.IMPORTED
        # Each torrent's OWN content_path reached import_completed - a run_blocking bug
        # forwarding the wrong torrent's path would still import, so pin the pairing.
        by_hash = {c.pending.infohash: c.content_path for c in strategy.import_calls}
        assert by_hash["fast"] == "/fast"
        assert by_hash["slow"] == "/slow"
        # slow showed a downloading update (fraction 0.5) before it completed.
        assert any(
            t.key == "slow" and t.phase is Phase.DOWNLOADING and t.fraction == 0.5
            for snap in view.snapshots
            for t in snap.torrents
        )

    def test_imported_only_when_all_done_two_polls(self) -> None:
        # The copy is async: poll 1 issues the command (command_issued,
        # files NOT done) -> reads `importing`. Poll 2 confirms the files done
        # -> `imported`. imported is gated on done files, never command accept.
        # progress_poll_interval=0 also pins the progress poll off: the poll
        # alone drives completion.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True, command_issued=True),
            ],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            progress_poll_interval=0,
        )

        assert len(strategy.import_calls) == 2
        assert strategy.progress_calls == []  # interval 0 disables the progress poll
        assert view.saw("h", Phase.IMPORTING)  # poll 1, copy in flight
        assert view.final("h").outcome is Outcome.IMPORTED  # poll 2, files done
        assert mgr._pending_imports.rows() == {}

    def test_progress_poll_fills_bar_and_promotes_before_the_next_poll(self) -> None:
        # Between polls the cheap progress poll fills the "files inserted" bar
        # as files get done and promotes the row to IMPORTED the instant all are done
        # - no second import_completed poll, no RefreshMonitoredDownloads.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True),
            progress_sequence=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),
                ImportProgress(3, 3, determinate=True),  # all done -> promote
            ],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            progress_poll_interval=5,
        )

        # Only ONE poll. The progress poll did the rest.
        assert len(strategy.import_calls) == 1
        assert len(strategy.progress_calls) == 3
        # The bar advanced (2/3 seen) before the row finished.
        assert any(
            t.key == "h" and t.import_done == 2 and t.import_total == 3
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}

    def test_importing_at_deadline_left_without_warning(self) -> None:
        # The copy never lands within imports.import_timeout: the final attempt
        # (at_deadline) leaves it pending with "still importing; left" - no drop.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True),
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}  # left, not dropped
        # The final in-bound poll forces AND marks the deadline, for THIS torrent's path.
        last = strategy.import_calls[-1]
        assert last.content_path == "/d"
        assert last.attempt_kind is AttemptKind.FINAL

    def test_each_rise_re_anchors_the_import_timeout(self) -> None:
        # A season pack Sonarr copies file-by-file: a baseline (t=0) then a rise
        # each poll. Every rise re-anchors the import timeout, so the wait pass
        # outlives import_timeout (60s here) and finishes imported instead of
        # cutting the import off mid-copy.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(
                    all_done=False,
                    command_issued=True,
                    imported_count=done,
                    file_map_episode_count=3,
                )
                for done in (0, 1, 2)
            ]
            + [attempt_result(all_done=True, command_issued=True, file_map_episode_count=3)],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}
        # Four polls (t=0/30/60/90), none forced: the deadline never fired.
        assert len(strategy.import_calls) == 4
        assert all(not c.attempt_kind.at_deadline for c in strategy.import_calls)

    def test_final_attempt_rescued_by_a_same_poll_rise(self) -> None:
        # The forced deadline attempt itself reports a fresh rise (0/3 -> 1/3 at
        # t=60): the row must NOT finish that poll - the done file re-anchors
        # the deadline instead - and only the NEXT quiet deadline (t=120) ends it.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(
                    all_done=False,
                    command_issued=True,
                    imported_count=done,
                    file_map_episode_count=3,
                )
                for done in (0, 0, 1)
            ],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}
        # t=0/30 in-bound, t=60 forced-but-rescued, t=90 in-bound, t=120 forced.
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, False, True]

    def test_stalled_pending_import_walks_away_not_ready(self) -> None:
        # No pause ever: the stall bound finishes the row "not ready" at
        # the final attempt, and the pending import stays for the next run.
        strategy = _RecordingStrategy(completed=attempt_result(all_done=False))
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.NOT_READY
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True]

    def test_sonarr_side_pause_waits_out_a_serial_backlog(self) -> None:
        # Sonarr importing the download itself (or the release ahead of it): no
        # command of pearlarr's, yet every poll is paused, so the row waits the
        # backlog out to IMPORTED without a forced attempt.
        strategy = _RecordingStrategy(
            completed_sequence=[attempt_result(all_done=False, pause_reason=PauseReason.IMPORT) for _ in range(4)]
            + [attempt_result(all_done=True)],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}
        assert len(strategy.import_calls) == 5
        assert all(not c.attempt_kind.at_deadline for c in strategy.import_calls)

    def test_final_auto_import_waits_its_own_copy_out(self) -> None:
        # Two stalled polls, then the final attempt POSTs (ISSUED, paused):
        # the row is not finished "still importing" on that same poll, the
        # copy in flight keeps pausing the clock, and the done-check ends IMPORTED.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(all_done=False),
                attempt_result(all_done=False),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                attempt_result(all_done=True, command_issued=True),
            ],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, True, True]

    def test_fresh_post_pauses_once_per_row(self) -> None:
        # A POST whose files all fail inside Sonarr is re-POSTed every poll:
        # only the first pauses the clock, so the retry loop finishes at the
        # ordinary deadline (t=60), never going on to the cap.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True]

    def test_paused_polls_never_walk_away_mid_copy(self) -> None:
        # The auto import's copy in flight across the import timeout: every paused poll
        # gives its interval back, so the row waits the copy out to IMPORTED
        # instead of walking away.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT) for _ in range(3)
            ]
            + [attempt_result(all_done=True, command_issued=True)],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}
        # Four polls (t=0/30/60/90), none forced: the paused clock never expired.
        assert len(strategy.import_calls) == 4
        assert all(not c.attempt_kind.at_deadline for c in strategy.import_calls)

    def test_pause_reason_pauses_the_import_clock_never_resets_it(self) -> None:
        # Two stalled polls burn the clock (t=0/30), the t=60 deadline poll
        # pauses: the pause spans only the paused interval, so the deadline
        # fires on the next clear shot (t=90) - a pause postpones the
        # walk-away by exactly the time spent waiting on the auto import's work.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                attempt_result(all_done=False, command_issued=True),
            ],
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}
        # t=60 reaches the deadline but pauses (no finish); t=90 is the real
        # forced attempt.
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, True]

    def test_pause_is_capped_so_a_wedged_command_ends_the_watch(self) -> None:
        # A command wedged in flight forever: paused polls stop pausing the
        # clock at the cap, the ordinary deadline resumes, and the watch ends
        # bounded ("still importing; left for the next run") instead of hanging until
        # Ctrl-C. Ready 60s + cap (6x) 360s -> finished by t=420.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_pause_cap_reads_still_importing_for_an_import_pause(self) -> None:
        # Sonarr's own import (or an unproven one) still running at the cap:
        # no command of pearlarr's, yet the label says "still importing", not "not
        # ready". Ready 60s + cap (6x) 360s -> finished by t=420.
        strategy = _RecordingStrategy(completed=attempt_result(all_done=False, pause_reason=PauseReason.IMPORT))
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert strategy.import_calls[-1].attempt_kind.at_deadline is True
        assert len(strategy.import_calls) == 15

    def test_pause_cap_reads_sonarr_busy_for_another_downloads_disk_command(self) -> None:
        # A foreign disk command holding the line to the cap: "sonarr busy",
        # left for the next run, never a false "still importing".
        strategy = _RecordingStrategy(completed=attempt_result(all_done=False, pause_reason=PauseReason.BUSY))
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.SONARR_BUSY
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_static_done_count_never_extends_the_deadline(self) -> None:
        # A genuinely stalled import: the first determinate reading (1/3) is a
        # baseline, not progress, and a count that never rises past it must not
        # move the anchor - the deadline still fires on schedule.
        strategy = _RecordingStrategy(
            completed=attempt_result(
                all_done=False,
                command_issued=True,
                imported_count=1,
                file_map_episode_count=3,
            ),
        )
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert set(mgr._pending_imports.rows()) == {"h"}
        # Three polls (t=0/30/60): the third is the forced deadline attempt.
        assert len(strategy.import_calls) == 3
        assert strategy.import_calls[-1].attempt_kind is AttemptKind.FINAL

    def test_a_progress_poll_rise_re_anchors_the_deadline(self) -> None:
        # The cheap progress poll sees a file get done (1/3 -> 2/3) even though the
        # poll's counts stay indeterminate. That rise alone must push the
        # deadline out one more poll (t=90, not t=60).
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True),
            progress_sequence=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),  # the rise, at t=10
            ],
        )
        _, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            import_timeout=60,
            progress_poll_interval=5,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        # Polls at t=0/30/60/90: the t=60 poll is in-bound only because the
        # t=10 progress-poll rise re-anchored the deadline to fire at t=70.
        assert len(strategy.import_calls) == 4
        assert strategy.import_calls[-1].attempt_kind is AttemptKind.FINAL

    @pytest.mark.parametrize(
        ("torrents", "outcome", "kept"),
        [
            pytest.param({}, Outcome.GONE, False, id="missing-drops"),
            pytest.param({"h": [FakeTorrent(is_errored=True)]}, Outcome.DOWNLOAD_ERRORED, True, id="errored-leaves"),
            # Never finishes: past imports.download_timeout the row finishes
            # DOWNLOAD_TIMED_OUT (left), the pending import stays for a later run.
            pytest.param({"h": [FakeTorrent(progress=0.5)]}, Outcome.DOWNLOAD_TIMED_OUT, True, id="timeout-leaves"),
            # COMPLETE but no content_path to import from: its own outcome (not
            # a misleading "timed out" - the download finished fine).
            pytest.param({"h": [FakeTorrent(is_complete=True)]}, Outcome.NO_CONTENT_PATH, True, id="no-path-leaves"),
        ],
    )
    def test_a_finish_without_import_keeps_or_drops_the_pending_import(
        self,
        torrents: dict[str, list[QbitStep]],
        outcome: Outcome,
        kept: bool,
    ) -> None:
        # None of these reach an import attempt, and none may touch the
        # post-import category hook (IMPORTED-only).
        strategy = _RecordingStrategy()
        qbit = CategoryQbit(torrents)
        mgr, view = _run_blocking_wait(
            strategy,
            qbit,
            post_import_category="pearlarr-done",
            download_timeout=60,
        )

        assert strategy.import_calls == []
        assert view.final("h").outcome is outcome
        assert set(mgr._pending_imports.rows()) == ({"h"} if kept else set())
        assert qbit.set_category_calls == []

    def test_tier2_progress_poll_error_is_contained(self) -> None:
        # A raising import_progress during the progress poll is debug-logged and
        # skipped - the row survives to the next poll, which gets the import done.
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True, command_issued=True),
            ],
            progress_error=RuntimeError("progress boom"),
        )
        pending = pending_import(infohash="h")
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        handler = CaptureHandler()
        mgr.logger.addHandler(handler)
        mgr.logger.setLevel(logging.DEBUG)
        view = RecordingWaitView()
        try:
            mgr.run_blocking(view=view)  # must not raise
        finally:
            mgr.logger.removeHandler(handler)
            mgr.logger.setLevel(logging.WARNING)

        assert view.final("h").outcome is Outcome.IMPORTED
        assert len(strategy.import_calls) == 2  # the poll still ran and got it done
        assert len(strategy.progress_calls) >= 2  # the progress poll kept polling after the raise
        assert any(r.levelno == logging.DEBUG and "progress poll" in r.getMessage() for r in handler.records)

    def test_wait_scope_all_includes_store_only_carried_over(self) -> None:
        # REGRESSION (complaint 4 - "exited right away"): a carried-over pending import
        # that is ONLY in the store (NOT a this-run grab) is still waited on. Here
        # there are no this-run grabs at all, yet the stored pending import is driven.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        qbit = FakeQbit({"carried": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="carried")],
            pending=[],  # nothing grabbed this run
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert strategy.import_calls  # the store-only pending import was driven
        assert view.final("carried").outcome is Outcome.IMPORTED
        assert mgr._pending_imports.rows() == {}

    def test_keyboard_interrupt_breaks_and_leaves_pending_imports(self) -> None:
        # Ctrl-C during the poll nap must break the loop (not propagate), so the
        # caller's finally still restores the terminal + saves the cache. The
        # in-flight pending import is left and a WaitResult is still returned.
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=_InterruptSleepClock(),
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        result = mgr.run_blocking(view=view)  # must not raise

        assert result is not None and result.waited == 0
        assert set(mgr._pending_imports.rows()) == {"h"}  # left for next run
        assert view.saw("h", Phase.DOWNLOADING)
        assert view.closed is False  # injected views are the caller's to close (made_view)
        # The break is announced as an INFO hub Diagnostic with the left count.
        (note,) = recording.of_type(Diagnostic)
        assert note.severity is Severity.INFO
        assert "1 left for the next run" in note.message
        assert note.origin == LOG_NAME

    def test_a_sigterm_stop_notes_what_is_left_and_propagates(self) -> None:
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=_StopSleepClock(),
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        with pytest.raises(SystemExit):
            mgr.run_blocking(view=view)

        assert set(mgr._pending_imports.rows()) == {"h"}
        (note,) = recording.of_type(Diagnostic)
        assert "1 left for the next run" in note.message

    def test_interrupt_mid_poll_still_finishes_that_polls_finished_rows(self) -> None:
        # A torrent that finished in the SAME poll the interrupt lands in
        # must still reach the view (the except handler pushes one final snapshot), so
        # the console tally/outcome lines can never undercount the returned WaitResult.
        first = pending_import(infohash="h1")
        second = pending_import(infohash="h2")
        # h1 is unscripted -> gone -> MISSING finish on its first advance. The
        # interrupt then lands on h2's poll, before that poll's snapshot push.
        qbit = _InterruptOnHash({"h2": [FakeTorrent(progress=0.3)]}, interrupt_on="h2")
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[first, second],
            pending=[first, second],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()

        result = mgr.run_blocking(view=view)  # must not raise

        assert result is not None and result.waited == 1
        assert view.final("h1").outcome is Outcome.GONE  # the interrupt-time push carried it

    def test_import_exception_is_swallowed_announced_and_pending_import_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate
        # and abort _finish_arr_run's cache save: the failure lands as an ERROR
        # Diagnostic carrying its traceback and the pending import is left.
        strategy = _RecordingStrategy(completed_error=RuntimeError("boom"))
        recording = install_recording_hub()
        mgr, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.ATTEMPT_FAILED
        assert set(mgr._pending_imports.rows()) == {"h"}
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert "Auto import failed" in error.message
        assert error.trace is not None

    def test_imported_finished_row_carries_files_count(self) -> None:
        # The finished row's outcome line states "(N files · elapsed)": a finished imported
        # row carries the done files count (the attempt result's file_map_episode_count).
        strategy = _RecordingStrategy(
            completed=attempt_result(
                all_done=True,
                imported_count=3,
                file_map_episode_count=3,
            ),
        )
        _, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
        )

        final = view.final("h")
        assert final.outcome is Outcome.IMPORTED
        assert final.import_total == 3

    def test_failed_store_drop_leaves_no_result_row(self) -> None:
        # The store drop precedes the result-row append: a drop that raises
        # must not record a phantom IMPORTED row for _finish_arr_run to count.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=_ExplodingDropStore(),
            clock=FakeClock(step=5),
        )
        wait_pass = WaitPass(mgr, [pending_import(infohash="h")], kind=WaitKind.BLOCKING)

        with pytest.raises(RuntimeError, match="drop boom"):
            wait_pass.run_poll()

        assert wait_pass.results == []
        assert wait_pass.rows["h"].active

    def test_failed_awaiting_cleanup_write_leaves_no_result_row(self) -> None:
        # The keep path's twin: the awaiting-cleanup upsert precedes the result-row append
        # exactly like the drop, so a raising put adds no phantom IMPORTED row.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
            close_return=EffectStatus.FAILED,
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=_ExplodingPutStore(),
            clock=FakeClock(step=5),
        )
        wait_pass = WaitPass(mgr, [pending_import(infohash="h")], kind=WaitKind.BLOCKING)

        with pytest.raises(RuntimeError, match="put boom"):
            wait_pass.run_poll()

        assert wait_pass.results == []
        assert wait_pass.rows["h"].active


def _wait_pass(
    qbit: FakeQbit,
    pending: PendingImport,
) -> WaitPass:
    """A fresh `WaitPass` over one pending import, wired to a scripted qBittorrent."""

    mgr = make_orchestration_manager(qbit=qbit, strategy=_RecordingStrategy(), clock=FakeClock(step=5))
    return WaitPass(mgr, [pending], kind=WaitKind.BLOCKING)


class TestBlockingWaitDownloadProgressRefresh:
    """refresh_download_progress: the refresh between polls - download progress only, no outcomes."""

    def test_progress_wait_drives_the_download_progress_refresh(self) -> None:
        # The wiring test: between polls, _sleep_between_polls's download progress refresh pushes
        # a snapshot with fresh download progress (the row moves to 60% at
        # 999 B/s well before the next poll).
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        pending = pending_import(infohash="h")
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.3, dlspeed=100),
                    FakeTorrent(is_complete=True, content_path="/d", progress=1.0),
                ]
            },
            download_progress={"h": [FakeTorrent(progress=0.6, dlspeed=999)]},  # the batched read
        )
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert any(
            t.key == "h" and t.phase is Phase.DOWNLOADING and t.fraction == 0.6 and t.speed_bps == 999
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_views_that_render_no_download_progress_skip_the_read(self) -> None:
        # A view with wants_download_progress=False (the non-TTY progress line) must not cost a
        # qBittorrent read every refresh between polls: only the two polls hit the client.
        class NoDownloadProgressView(RecordingWaitView):
            wants_download_progress = False

        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        pending = pending_import(infohash="h")
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
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )

        mgr.run_blocking(view=NoDownloadProgressView())

        assert qbit.calls == 2

    def test_updates_downloading_row_without_a_phase_transition(self) -> None:
        pending = pending_import(infohash="h")
        qbit = FakeQbit(
            {"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
            download_progress={"h": [FakeTorrent(progress=1.0, dlspeed=250)]},
        )
        wait_pass = _wait_pass(qbit, pending)
        wait_pass._advance(wait_pass.rows["h"])
        assert wait_pass.rows["h"].view.phase is Phase.DOWNLOADING

        changed = wait_pass.refresh_download_progress()

        assert changed is True
        view = wait_pass.rows["h"].view
        # Download progress moved (even to 100%) but the phase did NOT change - finish
        # decisions belong to the poll alone.
        assert view.phase is Phase.DOWNLOADING
        assert view.fraction == 1.0
        assert view.speed_bps == 250
        # The sparkline window is sampled per poll. The refresh between polls adds nothing.
        assert view.speed_history == (100,)
        assert wait_pass.rows["h"].active

    def test_unchanged_download_progress_reports_no_change(self) -> None:
        pending = pending_import(infohash="h")
        qbit = FakeQbit(
            {"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
            download_progress={"h": [FakeTorrent(progress=0.3, dlspeed=100)]},
        )
        wait_pass = _wait_pass(qbit, pending)
        wait_pass._advance(wait_pass.rows["h"])

        assert wait_pass.refresh_download_progress() is False


class TestBlockingWaitTransientPoll:
    """A transient qBittorrent blip keeps the row's last real state on screen."""

    def test_unobserved_poll_keeps_bar_and_history(self) -> None:
        pending = pending_import(infohash="h")
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.3, dlspeed=100),
                    qbittorrentapi.APIConnectionError("blip"),
                    FakeTorrent(progress=0.4, dlspeed=200),
                ],
            },
        )
        wait_pass = _wait_pass(qbit, pending)

        wait_pass.run_poll()
        wait_pass.run_poll()  # the blip poll
        assert wait_pass.rows["h"].view.fraction == 0.3  # last real reading kept, no 0% flash
        wait_pass.run_poll()

        view = wait_pass.rows["h"].view
        assert view.fraction == 0.4
        # Only the two real readings are in the sparkline window - the blip never
        # injected a fake stall sample.
        assert view.speed_history == (100, 200)


class TestBlockingWaitSpeedHistory:
    """The downloading rows' sparkline window: one sample per poll, bounded."""

    def test_accumulates_per_poll_with_stalls_as_zero(self) -> None:
        pending = pending_import(infohash="h")
        qbit = FakeQbit(
            {
                "h": [
                    FakeTorrent(progress=0.1, dlspeed=100),
                    FakeTorrent(progress=0.2),  # stalled (no speed) -> a 0 sample
                    FakeTorrent(progress=0.3, dlspeed=300),
                ],
            },
        )
        wait_pass = _wait_pass(qbit, pending)

        for _ in range(3):
            wait_pass.run_poll()

        assert wait_pass.rows["h"].view.speed_history == (100, 0, 300)

    def test_window_is_bounded_to_spark_samples(self) -> None:
        pending = pending_import(infohash="h")
        wait_pass = _wait_pass(FakeQbit({"h": [FakeTorrent(progress=0.1, dlspeed=100)]}), pending)

        for _ in range(SPARK_SAMPLES + 3):
            wait_pass.run_poll()

        assert wait_pass.rows["h"].view.speed_history == (100,) * SPARK_SAMPLES


class TestImportWaitModeProperty:
    """The services hub exposes the run's RESOLVED wait mode (cli > config).

    The Sonarr strategy's file-map-building gate reads this instead of the raw config
    so a CLI override that turns the feature on over an `off` config still
    builds file maps (otherwise the whole wait pass silently no-ops).
    """

    def test_reflects_resolved_mode(self) -> None:
        services = make_services(
            _state=ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING),
        )

        assert services.import_wait_mode is ImportWaitMode.BLOCKING


def _attach_wait_manager(engine: ArrRunLoop) -> None:
    """Attach an `ImportWaitManager` sharing the arr run loop's state.

    The wait/poll machinery lives on the manager now, so a finish/snapshot test
    on a bare arr run loop must wire one bound to the SAME `_state` / `cache_store` /
    client / strategy the arr run loop holds - exactly as `__init__` + `bind_arr_run` do.
    """

    engine._wait_manager = make_import_wait_manager(
        config=engine._config,
        cache_store=engine.cache_store,
        reporter=engine._reporter,
        logger=engine.logger,
        qbit=engine.qbit,
        state=engine._state,
        strategy=engine._active_strategy,
    )


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
    """A stand-in wait manager for the finish-ordering tests.

    Every step appends its ordering marker. The real ones return early on the
    empty working set these tests build - recording nothing - so a recording
    stand-in is what makes each step observable. `raise_on` scripts one step to
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

    def test_blocking_summary_precedes_the_blocking_wait_then_saves_last(self) -> None:
        # The arr run summary must print before the blocking wait pass, and the save must be dead
        # last (so the store reflects the wait pass's drops).
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
        )

        engine._finish_arr_run()

        assert calls == ["scan_finished", "cleanup", "tally", "summary", "monitor", "save", "run_finished"]

    def test_deferred_does_not_run_blocking(self) -> None:
        # Deferred reconciles pre-summary but NEVER runs the blocking wait pass.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.DEFERRED,
        )

        engine._finish_arr_run()

        assert "monitor" not in calls
        assert calls == ["scan_finished", "cleanup", "check", "tally", "summary", "save", "run_finished"]

    def test_dry_run_skips_the_blocking_wait_and_tally(self) -> None:
        # A dry run (no client) short-circuits reconcile/tally/blocking wait.
        calls: list[str] = []
        engine = _finalize_engine(calls, qbit=None, mode=ImportWaitMode.BLOCKING)

        engine._finish_arr_run()

        assert "monitor" not in calls
        # Even short-circuited, the summary still prints and the save still runs last.
        assert calls == ["scan_finished", "summary", "save", "run_finished"]

    def test_strategy_without_blocking_wait_reconciles_in_blocking_without_a_wait_pass(self) -> None:
        # A Radarr-shaped strategy (no blocking wait) reconciles pre-summary in
        # BLOCKING too - the blocking wait pass would otherwise be its only reconcile - and
        # never enters the wait region.
        calls: list[str] = []
        engine = _finalize_engine(
            calls,
            qbit=CLIENT_SENTINEL,
            mode=ImportWaitMode.BLOCKING,
            supports_blocking_wait=False,
        )

        engine._finish_arr_run()

        assert "monitor" not in calls
        assert calls == ["scan_finished", "cleanup", "check", "tally", "summary", "save", "run_finished"]


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


_PLACED = {"show - 01 [1080p].mkv": (101,)}
"""The placement a poll makes for the one-file pending import with no file map the placement tests share."""

_PLACED_ROW = {"show - 01 [1080p].mkv": [101]}
"""`_PLACED` as the stored row holds it, the ids as JSON lists."""


def _pending_import_with_no_file_map(**overrides: object) -> PendingImport:
    """A stored pending import on hash "h" whose grab-time file map is empty over a one-file listing."""

    fields: dict[str, object] = {
        "infohash": "h",
        "file_episode_map": {},
        "seadex_files": ["Show - 01 [1080p].mkv"],
        "ordered_episode_ids": [101],
    }
    fields.update(overrides)
    return pending_import(**fields)


class TestWaitPassStoresPlacements:
    """The placements a poll confirmed from Sonarr's history reach the pending import before any outcome reads it."""

    def test_check_pass_persists_the_placements(self) -> None:
        # A carried-over pending import (stored, not a this-run grab): the
        # placing poll's placements reach the durable row before it is kept.
        strategy = _RecordingStrategy(
            completed=attempt_result(
                all_done=False,
                command_issued=True,
                pause_reason=PauseReason.ISSUED,
                placements=_PLACED,
            ),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[_pending_import_with_no_file_map()],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [row.outcome for row in result.rows] == [Outcome.IMPORT_IN_PROGRESS]
        assert mgr._pending_imports.rows()["h"]["file_episode_map"] == _PLACED_ROW

    def test_a_pending_import_with_placements_reaches_the_next_poll_and_a_failed_cleanup_keeps_it(self) -> None:
        # The next poll runs on the updated pending import (placed, so the fast
        # path is open to it), and the row kept awaiting cleanup carries the file map too.
        pending = _pending_import_with_no_file_map()
        strategy = _RecordingStrategy(
            completed_sequence=[
                attempt_result(
                    all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED, placements=_PLACED
                ),
                attempt_result(all_done=True),
            ],
            close_return=EffectStatus.FAILED,
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )

        mgr.run_blocking(view=RecordingWaitView())

        first, second = strategy.import_calls
        assert dict(first.pending.file_episode_map) == {}
        assert dict(second.pending.file_episode_map) == _PLACED
        assert second.pending.file_map_coverage().file_map is True
        row = mgr._pending_imports.rows()["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (True, _PLACED_ROW)

    def test_no_placements_writes_nothing(self) -> None:
        # Scripted attempt results carry no placements: `save_history_placements` must not stage a blob
        # write per poll, so a store whose put raises never fires here.
        strategy = _RecordingStrategy(completed=attempt_result(all_done=False))
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=_ExplodingPutStore(),
            clock=FakeClock(step=5),
        )
        wait_pass = WaitPass(mgr, [_pending_import_with_no_file_map()], kind=WaitKind.BLOCKING)

        wait_pass.run_poll()  # must not raise

        assert wait_pass.rows["h"].active


class TestPendingImportWrites:
    """`PendingImportStore.drop` / `save` / `save_history_placements` keep the durable store and the run list in step."""

    def test_storing_an_attempt_result_persists_the_placements(self) -> None:
        pending = pending_import(
            infohash="h",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            seadex_files=["Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"],
        )
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
        )
        result = AttemptResult(
            all_done=False,
            command_issued=True,
            placements={"show - 02 [1080p].mkv": (102,)},
        )

        healed = mgr._pending_imports.save_history_placements(pending, result)

        expected = {"show - 01 [1080p].mkv": [101], "show - 02 [1080p].mkv": [102]}
        assert healed is not pending
        assert dict(healed.file_episode_map) == {"show - 01 [1080p].mkv": (101,), "show - 02 [1080p].mkv": (102,)}
        assert mgr._pending_imports.rows()["h"]["file_episode_map"] == expected
        assert mgr._state.grabbed_this_run[pending.infohash] is healed

    def test_storing_an_attempt_result_that_adds_nothing_is_a_no_op(self) -> None:
        # Placements the pending import already has: an exploding store proves no write fires.
        pending = pending_import(
            infohash="h",
            file_episode_map={"show - 01 [1080p].mkv": [101]},
        )
        mgr = make_orchestration_manager(qbit=None, strategy=_RecordingStrategy(), store=_ExplodingPutStore())
        result = AttemptResult(
            all_done=False,
            command_issued=False,
            placements={"show - 01 [1080p].mkv": (101,)},
        )

        assert (
            mgr._pending_imports.save_history_placements(pending, AttemptResult(all_done=True, command_issued=False))
            is pending
        )
        assert mgr._pending_imports.save_history_placements(pending, result) is pending

    def test_drop_removes_from_store_and_run_list(self) -> None:
        keep = pending_import(infohash="keep")
        drop = pending_import(infohash="drop")
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[keep, drop],
            pending=[keep, drop],
        )

        mgr._pending_imports.drop(drop.infohash)

        assert set(mgr._pending_imports.rows()) == {"keep"}
        assert mgr._state.grabbed_this_run == {keep.infohash: keep}

    def test_save_refreshes_the_run_list_copy(self) -> None:
        # The write twin: a saved pending import replaces its outdated `grabbed_this_run`
        # copy, so a fresh grab awaiting cleanup is never re-read without its flag from the run list.
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
        )

        mgr._pending_imports.save(replace(pending, awaiting_cleanup=True))

        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert [p.awaiting_cleanup for p in mgr._state.grabbed_this_run.values()] == [True]

    def test_save_never_inserts_an_absent_run_list_key(self) -> None:
        # A dict upsert here would silently convert a reacquire or demote into a
        # fresh grab, breaking the carried-over tally and finish_cleanups' recount.
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(qbit=None, strategy=_RecordingStrategy())

        mgr._pending_imports.save(pending)

        assert set(mgr._pending_imports.rows()) == {"h"}
        assert mgr._state.grabbed_this_run == {}

    def test_save_writes_no_guard_row(self) -> None:
        # A claim's guard row is its grab's to write: a pending import saved from a loaded copy never re-puts one.
        store = FakeCacheStore()
        pending = pending_import(infohash="h", anilist_id=5, guards=OwnedAtGrab(entry_groups=("G",)))
        mgr = make_orchestration_manager(qbit=None, strategy=_RecordingStrategy(), store=store)

        mgr._pending_imports.save(pending)

        assert set(mgr._pending_imports.rows()) == {"h"}
        assert store.get_guards(Arr.SONARR) == {}


class CategoryQbit(FakeQbit):
    """A `FakeQbit` recording the category writes the post-import move makes.

    `set_errors` scripts per-call `torrents_set_category` failures (popped in
    order), so the create-on-409 retry and the best-effort warn path can be driven.
    `preferences_error` fails every settings read. Without one, subcategories read as on.
    """

    def __init__(
        self,
        torrents: dict[str, list[QbitStep]] | None = None,
        *,
        set_errors: list[Exception] | None = None,
        preferences_error: Exception | None = None,
    ) -> None:
        super().__init__(torrents)
        self._set_errors = list(set_errors or [])
        self._preferences_error = preferences_error
        self.set_category_calls: list[tuple[str, str]] = []
        self.created_categories: list[str] = []

    def torrents_set_category(self, *, category: str, torrent_hashes: str) -> None:
        if self._set_errors:
            raise self._set_errors.pop(0)
        self.set_category_calls.append((category, torrent_hashes))

    def torrents_create_category(self, *, name: str) -> None:
        self.created_categories.append(name)

    def app_preferences(self) -> dict[str, object]:
        if self._preferences_error is not None:
            raise self._preferences_error
        return {"use_subcategories": True}


class TestPostImportCategory:
    """_apply_post_import_category: confirmed imports move to the arr run's post_import_category."""

    @staticmethod
    def _imported_manager(
        qbit: CategoryQbit,
        post_import_category: str | None = None,
    ) -> ImportWaitManager:
        """A manager whose one carried-over pending import's files are done (report -> IMPORTED).

        A `None` category routes through `make_config` to the blank-drop, so it
        exercises the same "left unset" default a real config file yields.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category=post_import_category,
        )

    @pytest.mark.parametrize(
        ("category", "set_errors", "expected_sets", "expected_created"),
        [
            # No 409 -> no create. Also pins the gate's scope: the pending import is stored
            # during the move (drop-last), and only the OTHER arr's row puts it off.
            pytest.param("pearlarr-done", [], [("pearlarr-done", "h")], [], id="clean-move"),
            # qBittorrent 409s an unknown category: create it, then re-apply.
            pytest.param(
                "pearlarr-done",
                [qbittorrentapi.Conflict409Error("unknown category")],
                [("pearlarr-done", "h")],
                ["pearlarr-done"],
                id="409-creates-then-retries",
            ),
            # One transient client error recovers on the in-run retry: the move still lands.
            pytest.param(
                "pearlarr-done",
                [qbittorrentapi.APIConnectionError("down")],
                [("pearlarr-done", "h")],
                [],
                id="client-error-recovers-on-retry",
            ),
            # Category left unset (the real config-file default): no call at all.
            pytest.param(None, [], [], [], id="unconfigured-no-call"),
        ],
    )
    def test_reconcile_move_outcomes(
        self,
        category: str | None,
        set_errors: list[Exception],
        expected_sets: list[tuple[str, str]],
        expected_created: list[str],
    ) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}, set_errors=set_errors)
        mgr = self._imported_manager(qbit, category)

        mgr.report_imports_for_series(7)  # must not raise

        assert qbit.set_category_calls == expected_sets
        assert qbit.created_categories == expected_created
        # The import itself always goes through: the pending import drops and classifies IMPORTED
        # (the `imported` bump is _finish_arr_run's, off that state).
        assert mgr._pending_imports.rows() == {}
        assert mgr._state.pending_import_states["h"] is PendingImportState.IMPORTED

    def test_exhausted_move_retries_keep_the_pending_import_for_cleanup(self) -> None:
        # Every in-run attempt fails: the import still counts IMPORTED, but the
        # pending import survives marked `awaiting_cleanup` with ONE keep warn naming the
        # failed effect (the old per-move warn is gone).
        recording = install_recording_hub()
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * 3
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}, set_errors=errors)
        mgr = self._imported_manager(qbit, "pearlarr-done")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert mgr._state.pending_import_states["h"] is PendingImportState.IMPORTED
        raw = mgr._pending_imports.rows()["h"]
        assert raw.get("awaiting_cleanup") is True
        warnings = [d.message for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert len(warnings) == 1
        assert "post-import cleanup" in warnings[0]
        assert "(category move)" in warnings[0]
        assert "h" in mgr._cleanup._awaiting_cleanup_keys

    def test_blocking_wait_moves_imported_torrent(self) -> None:
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        pending = pending_import(infohash="h")
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert view.final("h").outcome is Outcome.IMPORTED
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_missing_torrent_is_not_recategorized(self) -> None:
        # MISSING also drops the pending import, but only a confirmed IMPORT moves category.
        pending = pending_import(infohash="h")
        qbit = CategoryQbit({})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert view.final("h").outcome is Outcome.GONE
        assert mgr._pending_imports.rows() == {}  # still dropped
        assert qbit.set_category_calls == []

    def test_configured_category_without_client_is_a_silent_noop(self) -> None:
        # Category configured but no qBittorrent client (dry run): nothing to
        # call and nothing emitted - not even the best-effort warning.
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=_RecordingStrategy(),
            post_import_category="pearlarr-done",
        )
        recording = install_recording_hub()
        mgr._cleanup._apply_post_import_category(pending_import(infohash="h"), "pearlarr-done")  # must not raise

        assert recording.of_type(Diagnostic) == []

    def test_reconcile_missing_is_not_recategorized(self) -> None:
        # The reconcile twin of the blocking-wait MISSING test: the pending import is
        # dropped, but only a confirmed IMPORT moves category.
        qbit = CategoryQbit({})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )

        mgr.report_imports_for_series(7)

        assert mgr._pending_imports.rows() == {}  # MISSING still drops the pending import
        assert qbit.set_category_calls == []
        assert qbit.created_categories == []


def _movie_import(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` import record keyed by the torrent's downloadId."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


def _radarr_reconcile_manager(
    *,
    qbit: CategoryQbit | None,
    history: list[HistoryRecord] | None,
    radarr_pending_imports: list[PendingImport] | None = None,
    sonarr_pending_imports: list[PendingImport] | None = None,
    store: FakeCacheStore | None = None,
    post_import_category: str | None = None,
) -> tuple[ImportWaitManager, FakeCacheStore]:
    """A manager wired for a Radarr run: a real `RadarrStrategy` reconciling off scripted history.

    The manager and the strategy share one durable store, so a pending import the
    reconcile drops is gone for the cross-arr category count. `history` is set on
    the client directly (None = Radarr unreachable; `[]` = readable-but-empty). The
    `_state.arr` is RADARR, so the arr-scoped steps read the Radarr pending imports.
    """

    store = store if store is not None else FakeCacheStore()
    config = make_config(radarr_post_import_category=post_import_category)
    radarr = FakeRadarrClient()
    radarr.history_since_return = history
    strat = make_radarr_sync(radarr=radarr, config=config, cache_store=store)
    mgr = make_import_wait_manager(
        config=config,
        qbit=qbit,
        cache_store=store,
        categories=make_categories(config, Arr.RADARR),
        strategy=strat,
        reporter=_RecordingReporter(),
        state=ArrRunState(arr=Arr.RADARR),
    )
    for pending in radarr_pending_imports or []:
        store.put_pending_import_row(Arr.RADARR, pending.infohash, pending.to_json())
    for pending in sonarr_pending_imports or []:
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
    return mgr, store


class TestPostImportCategoryOtherArrGate:
    """The move waits for the OTHER arr's pending import on the torrent: whichever arr imports last moves.

    SeaDex can list one torrent on a Sonarr entry and a Radarr one, each arr storing its
    own pending import. Users key delete-with-data cleanup off the category, so a move on the
    first arr's import would mark a torrent the other arr still has to import. Within
    one arr every title goes in the ONE pending import, so no same-arr sibling exists to wait for.
    """

    def test_move_is_put_off_while_the_other_arr_holds_the_hash(self) -> None:
        # The Sonarr import is confirmed and its pending import drops, but the Radarr pending import still
        # claims the hash (the one lowercase key both arrs store): no move, while the close still runs.
        radarr_pending = pending_import(infohash="h", anilist_id=22, series_id=0, title="Movie")
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )
        store.put_pending_import_row(Arr.RADARR, "h", radarr_pending.to_json())

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}
        assert mgr._state.pending_import_states["h"] is PendingImportState.IMPORTED
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}

    def test_several_claims_on_one_pending_import_move_once(self) -> None:
        # Two AniList titles (multi-cour) go in the one Sonarr pending import and no other arr
        # holds the hash: the confirmed import moves exactly once, under one wait region row.
        pending = _two_claim_pending_import()
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert mgr._pending_imports.rows() == {}
        final = view.final("h")
        assert final.outcome is Outcome.IMPORTED
        assert final.label == f"Cour 1 & Cour 2{SEP}SubGroup"
        assert [claim_anilist_ids(c) for c in strategy.close_calls] == [(11, 22)]

    def test_other_arr_missing_drop_releases_the_gate(self) -> None:
        # Run 1 (Radarr): only its pending import exists and the torrent is gone from
        # qBittorrent, a MISSING drop with no move. Run 2 (Sonarr): the torrent's
        # re-grab imports with no other-arr claim left, so the move fires.
        gone = CategoryQbit({})  # an unscripted hash polls as MISSING
        store = FakeCacheStore()
        radarr_mgr, _ = _radarr_reconcile_manager(
            qbit=gone,
            history=[],
            radarr_pending_imports=[pending_import(infohash="h", series_id=0)],
            store=store,
            post_import_category="pearlarr-done",
        )

        radarr_mgr.run_deferred(view=RecordingWaitView())

        assert store.get_pending_import_rows(Arr.RADARR) == {}
        assert gone.set_category_calls == []

        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        sonarr_mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )

        sonarr_mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_later_run_re_registration_moves_again_idempotently(self) -> None:
        # A pending import re-created on a later run (a SeaDex update re-grabs the same
        # torrent) re-applies the move after its own import - idempotent in
        # qBittorrent, so a second move is harmless and correct.
        pending = pending_import(infohash="h", series_id=7)
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        store = FakeCacheStore()
        run1 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            store=store,
            stored_pending_imports=[pending],
            post_import_category="pearlarr-done",
        )

        run1.report_imports_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]

        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            store=store,
            stored_pending_imports=[pending],
            post_import_category="pearlarr-done",
        )

        run2.report_imports_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")] * 2
        assert run2._pending_imports.rows() == {}


class TestRemoveFromQueue:
    """_remove_from_queue: a confirmed import dismisses Sonarr's leftover queue record.

    One pending import spans every title on the torrent, so the close consults no sibling
    (see `ImportCompleter.remove_from_queue`). Only the category move waits on the other arr.
    """

    def test_reconcile_import_closes_the_queue_record(self) -> None:
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        pending = pending_import(infohash="h", series_id=7)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending],
        )

        mgr.report_imports_for_series(7)

        assert [c.infohash for c in strategy.close_calls] == [pending.infohash]

    def test_blocking_wait_import_closes_the_queue_record(self) -> None:
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
        )
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert view.final("h").outcome is Outcome.IMPORTED
        assert [c.infohash for c in strategy.close_calls] == [pending.infohash]

    def test_pending_import_with_two_claims_closes_once_on_cleanup(self) -> None:
        # Two titles (multi-cour) go in the one pending import: its confirmed import closes
        # the tracked download once, with no sibling left to wait for.
        pending = _two_claim_pending_import()
        strategy = _RecordingStrategy(progress=ImportProgress(2, 2, determinate=True))
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending],
        )

        mgr.report_imports_for_series(7)

        assert [claim_anilist_ids(c) for c in strategy.close_calls] == [(11, 22)]
        assert mgr._pending_imports.rows() == {}

    def test_remove_from_queue_off_never_closes(self) -> None:
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            remove_from_queue=False,
        )

        mgr.report_imports_for_series(7)

        assert strategy.close_calls == []
        assert mgr._pending_imports.rows() == {}  # the import itself still drops

    def test_missing_pending_import_is_not_closed(self) -> None:
        # MISSING also drops, but nothing was imported: the queue record (if
        # any) stays Sonarr's to resolve.
        strategy = _RecordingStrategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),  # an unscripted hash polls as MISSING
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.report_imports_for_series(7)

        assert mgr._pending_imports.rows() == {}
        assert strategy.close_calls == []

    def test_radarr_sibling_does_not_hold_the_sonarr_close(self) -> None:
        # The close reads no other pending import (unlike the category move's other-arr
        # gate): Sonarr's queue record only blocks Sonarr's completed-download
        # handling, so a Radarr pending import sharing the torrent must not keep it open.
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        sonarr_pending = pending_import(infohash="h", anilist_id=11, series_id=7)
        radarr_pending = pending_import(infohash="h", anilist_id=22, series_id=0)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=store,
            stored_pending_imports=[sonarr_pending],
        )
        store.put_pending_import_row(Arr.RADARR, radarr_pending.infohash, radarr_pending.to_json())

        mgr.report_imports_for_series(7)

        assert [c.infohash for c in strategy.close_calls] == [sonarr_pending.infohash]


class TestCleanupEffectOrder:
    """clean_up_and_drop: category move first, queue close second, drop last."""

    def test_move_runs_before_close(self) -> None:
        order: list[str] = []

        class _OrderQbit(CategoryQbit):
            @override
            def torrents_set_category(self, *, category: str, torrent_hashes: str) -> None:
                super().torrents_set_category(category=category, torrent_hashes=torrent_hashes)
                order.append("move")

        class _OrderStrategy(_RecordingStrategy):
            @override
            def remove_from_queue(self, pending: PendingImport) -> EffectStatus:
                order.append("close")
                return super().remove_from_queue(pending)

        qbit = _OrderQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        strategy = _OrderStrategy(progress=ImportProgress(1, 1, determinate=True))
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )

        mgr.report_imports_for_series(7)

        assert order == ["move", "close"]
        assert mgr._pending_imports.rows() == {}


_ARR_URL = "http://arr.test"


class TestCleanupCategoryResolve:
    """clean_up_and_drop resolves the post-import category at most once, feeding both effects."""

    @respx.mock
    def test_transient_fetch_failure_resolves_once_and_the_close_still_runs(self) -> None:
        # The failed fetch resolves no category: the move skips and the close's unwatched
        # gate short-circuits on the absent category, so the delete runs off ONE fetch.
        route = respx.get(f"{_ARR_URL}/api/v3/downloadclient").mock(side_effect=[httpx.Response(500)])
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h")
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        http, _ = bind_arr_http(_ARR_URL)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending],
            categories=make_categories(
                make_config(), sources=CategorySources(replace(http, retries=0), _sdk_client(qbit))
            ),
        )

        assert mgr._cleanup.clean_up_and_drop(pending) is True

        assert route.call_count == 1
        assert qbit.set_category_calls == []
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}

    def test_no_client_with_a_configured_category_still_runs_the_explicit_close(self) -> None:
        # qbit None resolves NO category at all: a configured unwatched category
        # must not make the close skip on is_unwatched when no move can run.
        strategy = _RecordingStrategy()
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=None,
            strategy=strategy,
            stored_pending_imports=[pending],
            torrent_category="tv-sonarr",
            post_import_category="sonarr-done",
        )

        assert mgr._cleanup.clean_up_and_drop(pending) is True

        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}


class TestCloseSkipOnMoveToUnwatched:
    """A move to an unwatched category supersedes the explicit queue removal.

    The gate reads `is_unwatched`, never this run's move outcome: firing the
    delete (a permanent "download ignored" history row) in a transiently
    failed-move window is exactly what the skip exists to prevent.
    """

    @staticmethod
    def _manager(
        *,
        grab: str | None,
        post_import: str | None,
        qbit: FakeQbit,
        strategy: _RecordingStrategy,
    ) -> ImportWaitManager:
        """A manager whose resolver already fetched the arr's client categories (preloaded, transportless)."""

        return make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            categories=make_fetched_categories(
                make_config(), grab=grab, post_import=post_import, qbit=_sdk_client(qbit)
            ),
        )

    def test_move_to_unwatched_skips_the_explicit_close(self) -> None:
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._manager(grab="tv-sonarr", post_import="sonarr-done", qbit=qbit, strategy=strategy)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == [("sonarr-done", "h")]
        assert strategy.close_calls == []
        assert mgr._pending_imports.rows() == {}

    def test_skip_holds_even_when_the_move_failed_this_run(self) -> None:
        # A transiently failed move keeps the pending import for the move effect only: no
        # delete fires meanwhile, so no Ignored marker can land in the window.
        recording = install_recording_hub()
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * 3
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}, set_errors=errors)
        mgr = self._manager(grab="tv-sonarr", post_import="sonarr-done", qbit=qbit, strategy=strategy)

        mgr.report_imports_for_series(7)

        assert strategy.close_calls == []
        raw = mgr._pending_imports.rows()["h"]
        assert raw.get("awaiting_cleanup") is True
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "(category move)" in warn
        assert "queue removal" not in warn

    def test_fetched_blank_grab_keeps_the_explicit_close(self) -> None:
        # A blank watched category means Sonarr sees everything: no move clears
        # the queue record, so the explicit close must still run.
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._manager(grab=None, post_import="sonarr-done", qbit=qbit, strategy=strategy)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == [("sonarr-done", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]

    def test_post_equal_to_watched_keeps_the_explicit_close(self) -> None:
        # Moving within the watched category clears nothing: the queue record stays.
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="shared")]})
        mgr = self._manager(grab="shared", post_import="shared", qbit=qbit, strategy=strategy)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == [("shared", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]

    def test_unknown_watched_category_keeps_the_explicit_close(self) -> None:
        # Neither a fetched pair nor a configured grab: conservative False.
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]


class TestMoveKeepsAnUnseenTorrentOut:
    """The post-import move never carries a torrent Sonarr can't see into a category Sonarr watches.

    Sonarr has no record of that download. Once the torrent showed up in its category, Sonarr would
    import the files again by its own reading of their names.
    """

    @staticmethod
    def _try_clean_up(qbit: CategoryQbit, *, post_import: str) -> tuple[bool, ImportWaitManager]:
        """Try to clean up one confirmed pending import, with Sonarr's one client read as watching `tv-sonarr`."""

        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            categories=make_fetched_categories(
                make_config(), grab="tv-sonarr", post_import=post_import, qbit=_sdk_client(qbit)
            ),
        )
        return mgr._cleanup.clean_up_and_drop(pending), mgr

    @classmethod
    def _clean_up(cls, qbit: CategoryQbit, *, post_import: str) -> ImportWaitManager:
        """`_try_clean_up`, for a pending import whose cleanup goes through."""

        retired, mgr = cls._try_clean_up(qbit, post_import=post_import)
        assert retired is True
        return mgr

    @pytest.mark.parametrize("post_import", ["tv-sonarr", "tv-sonarr/done"], ids=["watched", "its-subcategory"])
    def test_a_torrent_outside_sonarrs_categories_is_not_moved_into_one(self, post_import: str) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="manual")]})

        mgr = self._clean_up(qbit, post_import=post_import)

        assert qbit.set_category_calls == []
        # The skip is no failure: the pending import is dropped.
        assert mgr._pending_imports.rows() == {}

    def test_a_torrent_sonarr_already_sees_still_moves_within_its_categories(self) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="tv-sonarr")]})

        self._clean_up(qbit, post_import="tv-sonarr/done")

        assert qbit.set_category_calls == [("tv-sonarr/done", "h")]

    def test_a_failed_category_read_is_retried_and_never_moves_blind(self) -> None:
        unseen = FakeTorrent(is_complete=True, content_path="/d", category="manual")
        qbit = CategoryQbit({"h": [qbittorrentapi.APIConnectionError("down"), unseen]})

        mgr = self._clean_up(qbit, post_import="tv-sonarr")

        assert qbit.set_category_calls == []
        assert mgr._pending_imports.rows() == {}

    @pytest.mark.parametrize(
        ("waiting_in", "post_import"),
        [("manual", "tv-sonarr/done"), ("tv-sonarr/sub", "tv-sonarr")],
        ids=["into-a-subcategory", "out-of-a-subcategory"],
    )
    def test_an_unreadable_subcategory_setting_holds_the_move_it_would_decide(
        self,
        waiting_in: str,
        post_import: str,
    ) -> None:
        qbit = CategoryQbit(
            {"h": [FakeTorrent(is_complete=True, content_path="/d", category=waiting_in)]},
            preferences_error=qbittorrentapi.APIConnectionError("down"),
        )

        retired, mgr = self._try_clean_up(qbit, post_import=post_import)

        assert retired is False
        assert qbit.set_category_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True

    @staticmethod
    def _clean_up_unread(qbit: CategoryQbit) -> tuple[bool, ImportWaitManager]:
        """Clean up one confirmed pending import bound for `sonarr-done` while Sonarr's clients can't be read."""

        respx.get(f"{_ARR_URL}/api/v3/downloadclient").respond(status_code=500)
        pending = pending_import(infohash="h")
        config = make_config(torrent_category="anime", post_import_category="sonarr-done")
        http, _ = bind_arr_http(_ARR_URL)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            config=config,
            categories=make_categories(config, sources=CategorySources(http, _sdk_client(qbit))),
        )
        return mgr._cleanup.clean_up_and_drop(pending), mgr

    @respx.mock
    def test_unreadable_sonarr_clients_hold_the_move_for_a_later_run(self) -> None:
        # Whether `sonarr-done` is watched can't be known, so the move waits and the pending import awaits cleanup.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="manual")]})

        retired, mgr = self._clean_up_unread(qbit)

        assert retired is False
        assert qbit.set_category_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True

    @respx.mock
    @pytest.mark.parametrize(
        "torrents",
        [{"h": [FakeTorrent(is_complete=True, content_path="/d", category="sonarr-done")]}, {}],
        ids=["already-there", "gone-from-qbittorrent"],
    )
    def test_unreadable_sonarr_clients_never_hold_a_move_that_carries_nothing(
        self,
        torrents: dict[str, list[QbitStep]],
    ) -> None:
        retired, mgr = self._clean_up_unread(CategoryQbit(torrents))

        assert retired is True
        assert mgr._pending_imports.rows() == {}


class TestCloseRetrySchedule:
    """The close effect goes through the bounded in-run retry schedule off the injected clock."""

    @staticmethod
    def _drive(strategy: _RecordingStrategy, clock: FakeClock) -> ImportWaitManager:
        """Confirm one carried-over pending import through the report, close scripted by `strategy`."""

        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            clock=clock,
        )
        mgr.report_imports_for_series(7)
        return mgr

    def test_transient_retry_recovers_after_one_pause(self) -> None:
        clock = FakeClock()
        strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_sequence=[EffectStatus.RETRY, EffectStatus.DONE],
        )

        mgr = self._drive(strategy, clock)

        assert len(strategy.close_calls) == 2
        assert clock.sleeps == [1.0]
        assert mgr._pending_imports.rows() == {}

    def test_exhausted_retries_keep_the_pending_import(self) -> None:
        recording = install_recording_hub()
        clock = FakeClock()
        strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_return=EffectStatus.RETRY,
        )

        mgr = self._drive(strategy, clock)

        assert len(strategy.close_calls) == 3
        assert clock.sleeps == [1.0, 3.0]
        raw = mgr._pending_imports.rows()["h"]
        assert raw.get("awaiting_cleanup") is True
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "(queue removal)" in warn

    def test_failed_close_is_final_after_one_call(self) -> None:
        # FAILED (the queue read already retried in the transport) never re-attempts in-run.
        clock = FakeClock()
        strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_return=EffectStatus.FAILED,
        )

        mgr = self._drive(strategy, clock)

        assert len(strategy.close_calls) == 1
        assert clock.sleeps == []
        raw = mgr._pending_imports.rows()["h"]
        assert raw.get("awaiting_cleanup") is True


class TestCleanupRetry:
    """Pending imports awaiting cleanup: the observe steps skip them, `finish_cleanups` finishes them once per run."""

    def test_keep_then_next_run_finishes_the_cleanup(self) -> None:
        # Run 1: the import is confirmed but the close fails -> awaiting cleanup, still
        # counted IMPORTED this run. Run 2: finishing the cleanup closes and drops.
        run1_strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_return=EffectStatus.FAILED,
        )
        store = FakeCacheStore()
        run1 = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=run1_strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        run1.report_imports_for_series(7)

        assert run1._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert run1._state.pending_import_states["h"] is PendingImportState.IMPORTED

        recording = install_recording_hub()
        run2_strategy = _RecordingStrategy()
        # The same durable store, next run.
        run2 = make_orchestration_manager(qbit=FakeQbit({}), strategy=run2_strategy, store=store)

        run2._cleanup.finish_cleanups()

        assert [c.infohash for c in run2_strategy.close_calls] == ["h"]
        assert run2._pending_imports.rows() == {}
        assert any(
            "Finished the leftover post-import cleanup" in m for m in diagnostic_messages(recording, Severity.INFO)
        )

    def test_a_cleanup_that_still_fails_keeps_the_pending_import_uncounted(self) -> None:
        recording = install_recording_hub()
        strategy = _RecordingStrategy(close_return=EffectStatus.FAILED)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", awaiting_cleanup=True)],
        )

        mgr._cleanup.finish_cleanups()

        assert "h" not in mgr._state.pending_import_states
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "post-import cleanup" in warn
        # The failed cleanup finish left the pending import awaiting cleanup again, so a second call retries nothing.
        strategy.close_calls.clear()
        mgr._cleanup.finish_cleanups()
        assert strategy.close_calls == []

    def test_check_pass_keep_counts_via_the_result_row_only(self) -> None:
        # A deferred-wait keep counts via its result row's carried_over_imported
        # bump. No state is written, so the finish's IMPORTED sum never sees the key.
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True),
            close_return=EffectStatus.FAILED,
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 1
        assert "h" not in mgr._state.pending_import_states
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True

    def test_the_arr_run_finish_leaves_alone_a_pending_import_kept_this_run(self) -> None:
        # A pending import kept mid-scan is in the cleanup's `_awaiting_cleanup_keys`: `finish_cleanups`
        # must not retry it in the same run.
        strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_return=EffectStatus.FAILED,
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        mgr.report_imports_for_series(7)
        assert len(strategy.close_calls) == 1

        mgr._cleanup.finish_cleanups()

        assert len(strategy.close_calls) == 1  # no same-run second attempt

    def test_pending_imports_awaiting_cleanup_are_invisible_to_every_observe_pass(self) -> None:
        reporter = _RecordingReporter()
        strategy = _RecordingStrategy()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=True)],
            reporter=reporter,
        )

        mgr.report_imports_for_series(7)

        assert reporter.snapshot_calls == []
        assert mgr.run_deferred(view=RecordingWaitView()) is None
        assert mgr.run_blocking(view=RecordingWaitView()) is None
        assert strategy.import_calls == []
        assert set(mgr._pending_imports.rows()) == {"h"}

    def test_other_arr_pending_import_awaiting_cleanup_still_puts_off_the_move(self) -> None:
        # The other arr's pending import awaiting cleanup still claims the hash: the confirmed cleanup
        # puts off the move, runs the close, and still drops this arr's pending import.
        radarr_awaiting_cleanup = pending_import(infohash="h", anilist_id=22, series_id=0, awaiting_cleanup=True)
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            post_import_category="pearlarr-done",
        )
        store.put_pending_import_row(Arr.RADARR, radarr_awaiting_cleanup.infohash, radarr_awaiting_cleanup.to_json())

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}

    def test_cleanup_with_the_other_arr_on_the_hash_leaves_out_the_move_and_drops(self) -> None:
        # A later Radarr grab claims the hash: that pending import owns the eventual move, so
        # the pending import confirmed again by the cleanup finish just closes and drops.
        radarr_pending = pending_import(infohash="h", anilist_id=22, series_id=0)
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=True)],
            post_import_category="pearlarr-done",
        )
        store.put_pending_import_row(Arr.RADARR, radarr_pending.infohash, radarr_pending.to_json())

        mgr._cleanup.finish_cleanups()

        assert qbit.set_category_calls == []
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}

    def test_keep_counts_once_when_kept_and_zero_when_the_cleanup_finishes(self) -> None:
        # Entering run: the IMPORTED state feeds the finalize sum, the tally
        # skips the row awaiting cleanup. The run that finishes the cleanup: nothing counts anywhere.
        strategy = _RecordingStrategy(
            progress=ImportProgress(1, 1, determinate=True),
            close_return=EffectStatus.FAILED,
        )
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        mgr.report_imports_for_series(7)
        mgr.tally_carried_over_into_stats()

        assert mgr._state.pending_import_states["h"] is PendingImportState.IMPORTED
        assert (mgr._state.stats.downloading, mgr._state.stats.downloaded) == (0, 0)

        second_manager = make_orchestration_manager(qbit=FakeQbit({}), strategy=_RecordingStrategy(), store=store)
        second_manager._cleanup.finish_cleanups()
        second_manager.tally_carried_over_into_stats()

        assert second_manager._state.pending_import_states == {}
        assert (second_manager._state.stats.downloading, second_manager._state.stats.downloaded) == (0, 0)
        assert second_manager._state.stats.imported == 0


class TestCleanupRetryVerification:
    """The cleanup finish re-checks through the standard import check before any effect fires."""

    @staticmethod
    def _manager_with_awaiting_cleanup(
        strategy: _RecordingStrategy,
        *,
        qbit: FakeQbit | None = None,
        post_import_category: str | None = None,
    ) -> ImportWaitManager:
        """One stored pending import awaiting cleanup, its torrent complete unless `qbit` scripts otherwise."""

        return make_orchestration_manager(
            qbit=qbit if qbit is not None else FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=True)],
            post_import_category=post_import_category,
        )

    def test_done_files_run_the_effects_at_the_final_attempt(self) -> None:
        # Fresh evidence first: ONE attempt through the same check as any pending
        # import, at FINAL, then both effects fire and the pending import drops.
        recording = install_recording_hub()
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._manager_with_awaiting_cleanup(strategy, qbit=qbit, post_import_category="pearlarr-done")

        mgr._cleanup.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}
        assert any(
            "Finished the leftover post-import cleanup" in m for m in diagnostic_messages(recording, Severity.INFO)
        )

    def test_unobserved_qbittorrent_stays_awaiting_cleanup_without_effects(self) -> None:
        strategy = _RecordingStrategy()
        mgr = self._manager_with_awaiting_cleanup(
            strategy,
            qbit=FakeQbit({"h": [qbittorrentapi.APIConnectionError("down")]}),
        )

        mgr._cleanup.finish_cleanups()

        assert strategy.import_calls == []
        assert strategy.close_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert "h" not in mgr._state.pending_import_states

    def test_transient_retry_attempt_result_stays_awaiting_cleanup(self) -> None:
        # A waiting attempt result with no command is ambiguous (a blip, a busy queue): keep it awaiting cleanup and
        # retry next run.
        strategy = _RecordingStrategy(completed=attempt_result(all_done=False))
        mgr = self._manager_with_awaiting_cleanup(strategy)

        mgr._cleanup.finish_cleanups()

        assert strategy.close_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert "h" not in mgr._state.pending_import_states

    @pytest.mark.parametrize(
        ("result", "awaiting_cleanup"),
        [
            pytest.param(attempt_result(all_done=True, placements=_PLACED), True, id="failed-retire-keeps"),
            pytest.param(
                attempt_result(
                    all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED, placements=_PLACED
                ),
                False,
                id="reissue-demotes",
            ),
        ],
    )
    def test_cleanup_pass_stores_placements_before_the_effects(
        self, result: AttemptResult, awaiting_cleanup: bool
    ) -> None:
        # The placements are saved first, so the row awaiting cleanup and the demoted
        # row both carry the updated file map.
        strategy = _RecordingStrategy(completed=result, close_return=EffectStatus.FAILED)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            stored_pending_imports=[_pending_import_with_no_file_map(series_id=7, awaiting_cleanup=True)],
        )

        mgr._cleanup.finish_cleanups()

        row = mgr._pending_imports.rows()["h"]
        assert row.get("awaiting_cleanup") is awaiting_cleanup
        assert row["file_episode_map"] == _PLACED_ROW

    def test_files_gone_and_reimport_issued_demotes_to_a_carryover(self) -> None:
        # The files vanished and a fresh re-import command just went in (the
        # ISSUED reason): the pending import demotes to an ordinary carryover, no effect
        # fires, and this same run's wait pass takes it up.
        recording = install_recording_hub()
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
        )
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = self._manager_with_awaiting_cleanup(strategy, qbit=qbit, post_import_category="pearlarr-done")

        mgr._cleanup.finish_cleanups()

        assert qbit.set_category_calls == []
        assert strategy.close_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is False
        assert mgr._state.pending_import_states["h"] is PendingImportState.DOWNLOADED
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "no longer has every file imported" in warn

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.IMPORT_IN_PROGRESS, True)]

    def test_preowned_all_done_attempt_result_cleans_up_without_reimporting(self) -> None:
        # The arr's file moved to another trusted group: the shared statuses
        # machinery reports all-done with NO command issued, so the cleanup finish cleans
        # up and never imports the old file over the new one (and never demotes).
        recording = install_recording_hub()
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True, command_issued=False),
        )
        mgr = self._manager_with_awaiting_cleanup(strategy)

        mgr._cleanup.finish_cleanups()

        assert len(strategy.import_calls) == 1
        assert mgr._pending_imports.rows() == {}
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_auto_import_command_in_flight_holds_without_a_false_demote(self) -> None:
        # An auto import command from an earlier wait pass is still running: files may
        # be fine, so no demote and no "files no longer has every file imported" warn.
        recording = install_recording_hub()
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
        )
        mgr = self._manager_with_awaiting_cleanup(strategy)

        mgr._cleanup.finish_cleanups()

        assert strategy.close_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert "h" not in mgr._state.pending_import_states
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_final_attempt_steps_past_a_clean_pending_queue_record(self) -> None:
        # A clean importPending holds every ORDINARY attempt (the strategy leaves it to
        # Sonarr), so an ORDINARY cleanup finish would wedge for good. The FINAL
        # attempt makes an auto import, confirms it, and cleans up.
        class _AttemptGatedStrategy(_RecordingStrategy):
            @override
            def import_completed(
                self,
                pending: PendingImport,
                content_path: str,
                attempt_kind: AttemptKind,
            ) -> AttemptResult:
                super().import_completed(pending, content_path, attempt_kind)
                if attempt_kind.at_deadline:
                    return attempt_result(all_done=True)
                return attempt_result(all_done=False)

        strategy = _AttemptGatedStrategy()
        mgr = self._manager_with_awaiting_cleanup(strategy)

        mgr._cleanup.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert mgr._pending_imports.rows() == {}


class TestRadarrReconcile:
    """Radarr pending imports reconcile off import history: a matching event imports, else they wait."""

    def test_matching_event_drops_and_moves_category(self) -> None:
        # The torrent is complete in qBittorrent and Radarr's history shows the
        # import -> the pending import drops and the category move fires through the gate.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            radarr_pending_imports=[pending_import(infohash="h", series_id=0)],
            post_import_category="pearlarr-done",
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 1
        assert mgr._pending_imports.rows() == {}
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_no_event_keeps_pending_import_and_puts_off_the_category_move(self) -> None:
        # Complete download, but Radarr hasn't imported it yet (empty history):
        # the pending import stays and the category is NOT moved.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[],
            radarr_pending_imports=[pending_import(infohash="h", series_id=0)],
            post_import_category="pearlarr-done",
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 0
        assert set(mgr._pending_imports.rows()) == {"h"}
        assert qbit.set_category_calls == []

    def test_history_unreachable_keeps_pending_import_and_never_moves(self) -> None:
        # history_since -> None (Radarr unreachable): fail-open, never move on no evidence.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=None,
            radarr_pending_imports=[pending_import(infohash="h", series_id=0)],
            post_import_category="pearlarr-done",
        )

        mgr.run_deferred(view=RecordingWaitView())

        assert set(mgr._pending_imports.rows()) == {"h"}
        assert qbit.set_category_calls == []

    def test_a_pending_import_awaiting_cleanup_finishes_it_off_a_history_event(self) -> None:
        # The cleanup finish confirms a Radarr pending import the way a carryover checks out:
        # through import history. A matching event cleans it up with its effects.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            radarr_pending_imports=[pending_import(infohash="h", series_id=0, awaiting_cleanup=True)],
            post_import_category="pearlarr-done",
        )

        mgr._cleanup.finish_cleanups()

        assert mgr._pending_imports.rows() == {}
        assert qbit.set_category_calls == [("pearlarr-done", "h")]

    def test_a_pending_import_awaiting_cleanup_holds_on_an_unreadable_history(self) -> None:
        # No evidence, no effects: Radarr unreachable keeps the flag for a later run.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, store = _radarr_reconcile_manager(
            qbit=qbit,
            history=None,
            radarr_pending_imports=[pending_import(infohash="h", series_id=0, awaiting_cleanup=True)],
            post_import_category="pearlarr-done",
        )

        mgr._cleanup.finish_cleanups()

        assert store.get_pending_import_rows(Arr.RADARR)["h"].get("awaiting_cleanup") is True
        assert qbit.set_category_calls == []

    def test_tally_counts_a_carried_over_radarr_pending_import(self) -> None:
        # A complete-but-not-yet-imported Radarr pending import checks out AWAITING_IMPORT
        # and the pre-summary tally counts it into the downloaded counter.
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[],
            radarr_pending_imports=[pending_import(infohash="h", series_id=0)],
        )

        mgr.run_deferred(view=RecordingWaitView())
        mgr.tally_carried_over_into_stats()

        assert mgr._state.stats.downloaded == 1
        assert mgr._state.stats.downloading == 0


class TestCrossArrCategoryGate:
    """A torrent shared by a Sonarr grab and a Radarr grab moves only after BOTH import."""

    def test_sonarr_import_puts_off_the_move_until_radarr_reconciles(self) -> None:
        # One torrent, two pending imports: a Sonarr title's part (imports first) and a Radarr
        # movie. The Sonarr import must NOT move the category while the Radarr
        # pending import still claims the hash; the later Radarr reconcile fires the move.
        store = FakeCacheStore()
        sonarr_pending = pending_import(infohash="h", anilist_id=11, series_id=7, title="Series")
        radarr_pending = pending_import(infohash="h", anilist_id=22, series_id=0, title="Movie")
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})

        # Sonarr run: its part is confirmed (files done) and drops, but the Radarr
        # pending import still claims the hash cross-arr -> the move is put off.
        sonarr_mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=ImportProgress(1, 1, determinate=True)),
            store=store,
            stored_pending_imports=[sonarr_pending],
            post_import_category="pearlarr-done",
        )
        store.put_pending_import_row(Arr.RADARR, radarr_pending.infohash, radarr_pending.to_json())

        sonarr_mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []  # the Radarr pending import still claims the hash
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}

        # Radarr run: the movie import is confirmed -> no claim remains -> the move fires.
        radarr_mgr, _ = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            store=store,
            post_import_category="pearlarr-done",
        )

        radarr_mgr.run_deferred(view=RecordingWaitView())

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert store.get_pending_import_rows(Arr.RADARR) == {}


def make_add_engine(
    *,
    torrents: FakeTorrents,
    strategy: _RecordingStrategy,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config_overrides: object,
) -> tuple[ArrRunLoop, GrabPipeline]:
    """A bare `ArrRunLoop` + a `GrabPipeline` + an attached `ImportWaitManager`.

    The produce side lives on `GrabPipeline` (kept by the services hub in
    production) and the consume side on `ImportWaitManager`, so both are
    wired to the SAME `_state` / `cache_store` / client the arr run loop holds - an
    add through the returned pipeline registers into exactly the state the
    manager's consume steps (`report_imports_for_series` /
    `_imports_to_wait_on`) read back. `_active_strategy` is the test's
    recording `_RecordingStrategy` and `_reporter` a recording
    `_RecordingReporter` so the snapshot can be driven afterwards and asserted
    on recorded state.
    """

    engine = make_bare_instance(
        ArrRunLoop,
        qbit=qbit,
        logger=make_logger(),
        _config=make_config(**config_overrides),
        _active_strategy=strategy,
        _reporter=_RecordingReporter(),
        cache_store=FakeCacheStore(),
    )
    engine._state = ArrRunState(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode)
    pipeline = make_grab_pipeline(
        _config=engine._config,
        _torrents=torrents,
        cache_store=engine.cache_store,
        qbit=qbit,
        _state=engine._state,
    )
    _attach_wait_manager(engine)
    return engine, pipeline


class TestRegisteredGrabSurvivesTheSeriesReport:
    """A this-run grab registered by the pipeline is owned by the blocking wait pass, not re-polled."""

    def test_registered_already_in_qbit_survives_the_series_report(self) -> None:
        # Integration: an ALREADY_IN_QBIT registered THIS run is a this-run grab, so
        # the per-series report skips it (no re-poll, no drop) and the end-of-run
        # blocking wait pass owns it via the working set.
        torrents = FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")})
        strategy = _RecordingStrategy()
        engine, pipeline = make_add_engine(torrents=torrents, strategy=strategy)
        updates = {"h1": pending_import_update("h1", series_id=7)}

        pipeline.grab_releases(
            grab_request(seadex_dict=one_release_dict(srg="NAN0", infohash="h1"), pending_import_updates=updates),
        )
        engine._wait_manager.report_imports_for_series(7)

        assert strategy.import_calls == []
        assert set(engine._wait_manager._pending_imports.rows()) == {"h1"}
        assert [p.infohash for p in engine._wait_manager._imports_to_wait_on()] == ["h1"]
