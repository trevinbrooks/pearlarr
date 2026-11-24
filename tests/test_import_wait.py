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
pinned via a bare arr run loop over a recording stand-in for the manager (`_finalize_engine`).

Every collaborator the manager drives - the strategy's import hooks, the snapshot
reporter, qBittorrent - is a small typed fake recording what a test asserts, so
the contracts are pinned by recorded state.
"""

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast, override

import pytest
import qbittorrentapi

from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.import_wait import ImportWaitManager, WaitPass, WaitPollers
from pearlarr.output import SPARK_SAMPLES, Diagnostic, Phase, Severity, TorrentView, WaitKind, WaitSnapshot
from pearlarr.pending_follow import follow
from pearlarr.pending_import import PendingImport
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
    attempt_result,
    files_left_for_hand_import,
    make_bare_instance,
    make_categories,
    make_config,
    make_fetched_categories,
    make_import_wait_manager,
    make_logger,
    make_radarr_sync,
    make_services,
    order_only_hand_import_files,
    pending_import,
    sonarr_ep,
    title_claim,
    two_claim_pending_import,
)
from .fakes import (
    CaptureHandler,
    FakeClock,
    FakeRadarrClient,
    FakeStrategy,
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

    def test_errored(self) -> None:
        pollers = make_pollers(FakeQbit({"h": [FakeTorrent(is_errored=True)]}))

        assert pollers.poll_torrent("h") == TorrentPoll(TorrentStatus.ERRORED, None)

    def test_complete_on_full_progress_without_flag(self) -> None:
        # progress == 1.0 counts as complete even if the state flag is unset.
        torrent = FakeTorrent(progress=1.0, content_path="/data/movie")
        pollers = make_pollers(FakeQbit({"h": [torrent]}))

        assert pollers.poll_torrent("h") == TorrentPoll(
            TorrentStatus.COMPLETE,
            "/data/movie",
            DownloadProgress(1.0, None, None, None, None),
        )

    def test_none_on_transient_api_error(self) -> None:
        # A dropped connection / re-auth in flight is "still waiting", not finished.
        # The un-observed flag tells the wait pass the zeroed download progress is a
        # placeholder (keep the last real bar), not a reading.
        pollers = make_pollers(FakeQbit({"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, observed=False)

    def test_none_when_no_client(self) -> None:
        pollers = make_pollers(None)

        assert pollers.poll_torrent("h") == TorrentPoll(None, None, observed=False)


class TestPollDownloadProgress:
    """poll_download_progress: ONE batched, read-only info call for the wait region refresh."""

    def test_empty_set_makes_no_call(self) -> None:
        qbit = FakeQbit({})
        pollers = make_pollers(qbit)

        assert pollers.poll_download_progress([]) == {}
        assert qbit.calls == 0

    def test_transient_error_yields_nothing(self) -> None:
        # The rows just keep their last download progress until the next poll.
        pollers = make_pollers(FakeQbit(download_progress={"h": [qbittorrentapi.APIConnectionError("boom")]}))

        assert pollers.poll_download_progress(["h"]) == {}

    def test_each_asked_hash_gets_its_own_reading(self) -> None:
        qbit = FakeQbit(
            download_progress={
                "a": [FakeTorrent(progress=0.2, dlspeed=10)],
                "b": [FakeTorrent(progress=0.8, dlspeed=20)],
            }
        )

        assert make_pollers(qbit).poll_download_progress(["a", "b"]) == {
            "a": DownloadProgress(0.2, 10, None, None, None),
            "b": DownloadProgress(0.8, 20, None, None, None),
        }

    def test_unasked_rows_are_ignored(self) -> None:
        # The batch answers for "h" with a row whose hash reads back "other" -
        # poll_download_progress must drop what it can't key back to an asked hash.
        pollers = make_pollers(FakeQbit(download_progress={"h": [FakeTorrent(torrent_hash="other", progress=0.4)]}))

        assert pollers.poll_download_progress(["h"]) == {}


class _AutoTickClock(FakeClock):
    """Time passing between reads: every `now` reading advances by `step`."""

    @override
    def monotonic(self) -> float:
        t = super().monotonic()
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
    state: ArrRunState | None = None,
    **config_overrides: object,
) -> ImportWaitManager:
    """An `ImportWaitManager` wired for the pending-import orchestration paths.

    Sets up the durable per-arr store (`store`, a fresh fake unless a test shares one
    across runs) and the run list (`pending`, keyed into `_state.grabbed_this_run`) so
    `report_imports_for_series`, `run_deferred` and `run_blocking` can be driven
    without a live Sonarr/qBittorrent. The strategy is a recording
    `_RecordingStrategy` (its import hooks scripted per test) and the reporter a
    recording `_RecordingReporter` (a test that asserts on the snapshot reporter
    passes in its own `reporter` to read it back). A test that reads the run's state
    passes in its own `state`, which then carries the run list in place of `pending`.
    """

    cache_store = store if store is not None else FakeCacheStore()
    if state is None:
        state = ArrRunState(arr=Arr.SONARR, grabbed_this_run={p.infohash: p for p in pending or []})
    mgr = make_import_wait_manager(
        qbit=qbit,
        cache_store=cache_store,
        reporter=reporter or _RecordingReporter(),
        clock=clock if clock is not None else FakeClock(),
        strategy=strategy,
        state=state,
        **config_overrides,
    )
    for stored in stored_pending_imports or []:
        cache_store.put_pending_import_row(Arr.SONARR, stored.infohash, stored.to_json())
    return mgr


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

    @pytest.mark.parametrize(
        ("series_id", "state"),
        [
            # A stored pending import re-seen in qBittorrent this run was already reported by
            # its title block: no poll, no row, no state. Only the tally counts it.
            pytest.param(7, ArrRunState(arr=Arr.SONARR, reacquired_keys={"h"}), id="seen-again-this-run"),
            pytest.param(
                7,
                ArrRunState(arr=Arr.SONARR, grabbed_this_run={"h": pending_import(infohash="h", series_id=7)}),
                id="grabbed-this-run",
            ),
            pytest.param(99, ArrRunState(arr=Arr.SONARR), id="on-another-series"),
        ],
    )
    def test_a_pending_import_the_series_report_skips_is_left_alone(self, series_id: int, state: ArrRunState) -> None:
        reporter = _RecordingReporter()
        qbit = FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(),
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=series_id)],
            state=state,
        )

        mgr.report_imports_for_series(7)

        assert qbit.calls == 0
        assert reporter.snapshot_calls == []
        assert state.pending_import_states == {}
        assert set(mgr._pending_imports.rows()) == {"h"}

    @pytest.mark.parametrize(
        ("torrent", "progress", "state", "kept"),
        [
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                ImportProgress(1, 1, determinate=True),
                PendingImportState.IMPORTED,
                False,
                id="every-file-imported",
            ),
            pytest.param(
                FakeTorrent(is_complete=True), None, PendingImportState.DOWNLOADED, True, id="files-not-in-yet"
            ),
        ],
    )
    def test_a_carried_over_pending_import_is_reported_as_qbittorrent_and_the_files_read(
        self, torrent: FakeTorrent, progress: ImportProgress | None, state: PendingImportState, kept: bool
    ) -> None:
        # The report only reads: it never attempts an import, and the `imported` count is the arr run finish's.
        strategy = _RecordingStrategy(progress=progress)
        reporter = _RecordingReporter()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=strategy,
            reporter=reporter,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.report_imports_for_series(7)

        assert strategy.import_calls == []
        assert mgr._state.pending_import_states["h"] is state
        assert set(mgr._pending_imports.rows()) == ({"h"} if kept else set())
        assert mgr._state.stats.imported == 0
        assert [(c.state, c.title) for c in reporter.snapshot_calls] == [(state, f"Show{SEP}SubGroup")]

    def test_a_two_series_pending_import_is_reported_once_under_the_series_that_scans_first(self) -> None:
        reporter = _RecordingReporter()
        pending = pending_import(
            infohash="h",
            claims=(
                title_claim(anilist_id=11, series_id=7, window_coverage="S01 E01-E12", url="https://releases.moe/11"),
                title_claim(anilist_id=22, series_id=8, window_coverage="S02 E01-E12", url="https://releases.moe/22"),
            ),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.5)]}),
            strategy=_RecordingStrategy(),
            reporter=reporter,
            stored_pending_imports=[pending],
        )

        mgr.report_imports_for_series(8)
        mgr.report_imports_for_series(7)

        assert [(c.state, c.coverage, c.url) for c in reporter.snapshot_calls] == [
            (PendingImportState.DOWNLOADING, "S02 E01-E12", "https://releases.moe/22"),
        ]


def _report_series_7(mgr: ImportWaitManager) -> None:
    mgr.report_imports_for_series(7)


def _run_end_pass(mgr: ImportWaitManager) -> None:
    mgr.run_deferred(view=RecordingWaitView())


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

    @pytest.mark.parametrize(
        ("in_qbittorrent", "awaiting_cleanup", "run", "checked_again"),
        [
            pytest.param(False, False, _report_series_7, True, id="gone-at-the-scan-report"),
            pytest.param(False, False, _run_end_pass, True, id="gone-at-the-end-pass"),
            pytest.param(True, False, _report_series_7, False, id="imported-at-the-scan-report"),
            pytest.param(True, False, _run_end_pass, False, id="imported-at-the-end-pass"),
            pytest.param(False, True, ImportWaitManager.finish_cleanups, False, id="gone-after-a-confirmed-import"),
        ],
    )
    def test_a_drop_clears_the_claiming_titles_stamps_only_when_nothing_was_imported(
        self,
        in_qbittorrent: bool,
        awaiting_cleanup: bool,
        run: Callable[[ImportWaitManager], None],
        checked_again: bool,
    ) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]} if in_qbittorrent else {}),
            strategy=_RecordingStrategy(
                completed=attempt_result(all_done=True), progress=ImportProgress(1, 1, determinate=True)
            ),
            store=store,
            stored_pending_imports=[replace(self._record(), awaiting_cleanup=awaiting_cleanup)],
        )

        run(mgr)

        assert store.get_pending_import_rows(Arr.SONARR) == {}
        claiming = None if checked_again else self._CACHED_ON
        # Title 9 never claimed the torrent.
        assert self._stamps(store) == {5: claiming, 6: claiming, 9: self._CACHED_ON}

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


class TestRunDeferred:
    """run_deferred runs ONE non-blocking poll over every carried-over pending import."""

    @pytest.mark.parametrize(
        "make_strategy",
        [
            pytest.param(
                lambda: _RecordingStrategy(completed=attempt_result(all_done=True)), id="the-attempt-confirms"
            ),
            pytest.param(
                lambda: _RecordingStrategy(
                    completed=attempt_result(all_done=False, command_issued=True),
                    progress=ImportProgress(1, 1, determinate=True),
                ),
                id="the-progress-read-confirms",
            ),
        ],
    )
    def test_each_ready_carried_over_import_is_counted_and_dropped_and_a_this_run_grab_is_left(
        self, make_strategy: Callable[[], _RecordingStrategy]
    ) -> None:
        this_run = pending_import(infohash="new")
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({key: [FakeTorrent(is_complete=True, content_path="/d")] for key in ("h1", "h2", "new")}),
            strategy=make_strategy(),
            store=store,
            stored_pending_imports=[pending_import(infohash="h1"), pending_import(infohash="h2"), this_run],
            pending=[this_run],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 2
        assert set(store.get_pending_import_rows(Arr.SONARR)) == {"new"}

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
                qbittorrentapi.APIConnectionError("down"),
                _RecordingStrategy,
                Outcome.QBITTORRENT_UNREACHABLE,
                PendingImportState.DOWNLOADING,
                id="qbittorrent-unreachable",
            ),
            pytest.param(
                FakeTorrent(is_complete=True),
                _RecordingStrategy,
                Outcome.NO_CONTENT_PATH,
                PendingImportState.DOWNLOADED,
                id="no-content-path",
            ),
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                lambda: _RecordingStrategy(completed=attempt_result(all_done=False, command_issued=True)),
                Outcome.IMPORT_IN_PROGRESS,
                PendingImportState.DOWNLOADED,
                id="import-in-progress",
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
        torrent: QbitStep,
        make_strategy: Callable[[], _RecordingStrategy],
        outcome: Outcome,
        state: PendingImportState,
    ) -> None:
        # The deferred wait pass re-observes even a reported pending import: its outcome overwrites
        # the outdated state, so the tally buckets what the wait pass saw last.
        # The download clock starts when the pass is built, so a deferred pass never applies the download timeout,
        # however small it is and however much time passes between reads.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=make_strategy(),
            stored_pending_imports=[pending_import(infohash="h")],
            download_timeout=1,
            clock=_AutoTickClock(step=30),
        )
        mgr._state.pending_import_states["h"] = PendingImportState.DOWNLOADING

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(outcome, True)]
        assert mgr._state.pending_import_states["h"] is state
        assert set(mgr._pending_imports.rows()) == {"h"}


class TestPendingImportStateConversion:
    """The pinned Outcome -> PendingImportState conversion a wait pass classifies stored pending imports through."""

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


class TestTallyCarriedOverIntoStats:
    """tally_carried_over_into_stats counts each still-waiting pending import once."""

    @pytest.mark.parametrize(
        ("stored", "state", "counted"),
        [
            # Both counters add up, a pending import nothing observed this run counts as downloading, and an
            # errored one is counted nowhere.
            pytest.param(
                [pending_import(infohash=key) for key in ("q", "d1", "d2", "e", "untouched")],
                ArrRunState(
                    arr=Arr.SONARR,
                    pending_import_states={
                        "q": PendingImportState.DOWNLOADING,
                        "d1": PendingImportState.DOWNLOADED,
                        "d2": PendingImportState.DOWNLOADED,
                        "e": PendingImportState.ERRORED,
                    },
                ),
                (2, 2),
                id="each-state",
            ),
            pytest.param(
                [pending_import(infohash="h")],
                ArrRunState(arr=Arr.SONARR, grabbed_this_run={"h": pending_import(infohash="h")}),
                (0, 0),
                id="grabbed-this-run",
            ),
            pytest.param(
                [pending_import(infohash="h")],
                ArrRunState(arr=Arr.SONARR, reacquired_keys={"h"}),
                (1, 0),
                id="seen-again-this-run",
            ),
            pytest.param(
                [pending_import(infohash="h", awaiting_cleanup=True)],
                ArrRunState(arr=Arr.SONARR),
                (0, 0),
                id="awaiting-cleanup",
            ),
        ],
    )
    def test_counts_each_carried_over_pending_import_still_waiting(
        self, stored: list[PendingImport], state: ArrRunState, counted: tuple[int, int]
    ) -> None:
        mgr = make_orchestration_manager(
            qbit=None, strategy=_RecordingStrategy(), stored_pending_imports=stored, state=state
        )

        mgr.tally_carried_over_into_stats()

        assert (state.stats.downloading, state.stats.downloaded) == counted


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

    def test_two_torrents_each_import_from_their_own_path(self) -> None:
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = FakeQbit(
            {
                "fast": [FakeTorrent(is_complete=True, content_path="/fast")],
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        pending_imports = [pending_import(infohash="fast"), pending_import(infohash="slow")]
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=pending_imports,
            pending=pending_imports,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        paths = {call.pending.infohash: call.content_path for call in strategy.import_calls}
        assert paths == {"fast": "/fast", "slow": "/slow"}
        assert [view.final(key).outcome for key in ("fast", "slow")] == [Outcome.IMPORTED, Outcome.IMPORTED]
        assert store.get_pending_import_rows(Arr.SONARR) == {}

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

    @pytest.mark.parametrize(
        ("clock", "ending"),
        [
            pytest.param(_InterruptSleepClock(), nullcontext(), id="ctrl-c-ends-the-wait-and-the-run-goes-on"),
            pytest.param(_StopSleepClock(), pytest.raises(SystemExit), id="a-sigterm-stop-propagates"),
        ],
    )
    def test_an_interrupted_wait_notes_what_is_left(
        self, clock: FakeClock, ending: AbstractContextManager[object]
    ) -> None:
        pending = pending_import(infohash="h")
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending],
            pending=[pending],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=clock,
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        with ending:
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

    @pytest.mark.parametrize(
        "wanted", [pytest.param(True, id="a-view-with-bars"), pytest.param(False, id="a-view-without-bars")]
    )
    def test_progress_wait_drives_the_download_progress_refresh(self, wanted: bool) -> None:
        # The wiring test: between polls, _sleep_between_polls's download progress refresh pushes
        # a snapshot with fresh download progress (the row moves to 60% at
        # 999 B/s well before the next poll). A view that shows no download progress is never refreshed.
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
        view.wants_download_progress = wanted

        mgr.run_blocking(view=view)

        shown = any(
            t.key == "h" and t.phase is Phase.DOWNLOADING and t.fraction == 0.6 and t.speed_bps == 999
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert shown is wanted
        assert view.final("h").outcome is Outcome.IMPORTED

    _POLLED = FakeTorrent(progress=0.3, dlspeed=100, eta=130, completed=300, size=1000)

    @pytest.mark.parametrize(
        ("refreshed", "moved"),
        [
            pytest.param(_POLLED, False, id="unchanged"),
            # Even at 100% the row stays downloading: only a poll finishes it.
            pytest.param(FakeTorrent(progress=1.0, dlspeed=250, eta=5, completed=1000, size=1000), True, id="moved"),
        ],
    )
    def test_a_refresh_between_polls_moves_the_bar_and_nothing_else(self, refreshed: FakeTorrent, moved: bool) -> None:
        pending = pending_import(infohash="h")
        qbit = FakeQbit({"h": [self._POLLED]}, download_progress={"h": [refreshed]})
        wait_pass = _wait_pass(qbit, pending)
        wait_pass.run_poll()

        assert wait_pass.refresh_download_progress() is moved
        view = wait_pass.rows["h"].view
        assert (view.phase, view.fraction, view.speed_bps) == (Phase.DOWNLOADING, refreshed.progress, refreshed.dlspeed)
        assert (view.eta_s, view.bytes_done, view.bytes_total) == (refreshed.eta, refreshed.completed, refreshed.size)
        # The sparkline is sampled once per poll, never by the refresh.
        assert view.speed_history == (100,)


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

    @pytest.mark.parametrize(
        ("polls", "history"),
        [
            pytest.param(3, (100, 0, 300), id="a-stall-is-a-zero"),
            pytest.param(SPARK_SAMPLES + 3, (300,) * SPARK_SAMPLES, id="the-oldest-samples-fall-off"),
        ],
    )
    def test_accumulates_per_poll_with_stalls_as_zero(self, polls: int, history: tuple[int, ...]) -> None:
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

        for _ in range(polls):
            wait_pass.run_poll()

        assert wait_pass.rows["h"].view.speed_history == history


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


_MOVED_DOWN_ONE = [
    sonarr_ep(1, number, ep_id=100 + number, tvdb_id=8999 + number, episode_file_id=0) for number in (1, 2, 3)
]
"""A read of series 7 with TVDB episode 9001 on row 102 and 9002 on row 103."""

_FOLLOWED_ROW = {"show - 01 [1080p].mkv": [102], "show - 02 [1080p].mkv": [103]}
"""The file map `_followed` gives `_two_titles_stored`, as the stored row holds it."""


def _two_titles_stored(*, awaiting_cleanup: bool = False) -> PendingImport:
    """Torrent "h" of series 7, claimed by title 5 on row 101 and title 6 on row 102.

    Stored while row 101 held TVDB episode 9001 and row 102 held 9002.
    """

    return two_claim_pending_import(
        anilist_ids=(5, 6),
        windows=((101,), (102,)),
        infohash="h",
        file_episode_map={"show - 01 [1080p].mkv": (101,), "show - 02 [1080p].mkv": (102,)},
        seadex_files=("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"),
        tvdb_ids={101: 9001, 102: 9002},
        awaiting_cleanup=awaiting_cleanup,
    )


def _followed(pending: PendingImport) -> PendingImport:
    """`pending` as an attempt that read `_MOVED_DOWN_ONE` ends with it."""

    followed = follow(pending, {7: _MOVED_DOWN_ONE})
    assert followed is not None
    return followed.pending


class TestWaitSavesTheFollowedPendingImport:
    """The pending import an attempt followed is what every save of the wait writes."""

    @staticmethod
    def _manager(strategy: _RecordingStrategy, store: FakeCacheStore, pending: PendingImport) -> ImportWaitManager:
        """A manager over `store` with `pending` stored and its torrent complete in qBittorrent."""

        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        return make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}), strategy=strategy, store=store
        )

    @pytest.mark.parametrize(
        ("all_done", "awaiting_cleanup"),
        [pytest.param(True, True, id="failed-retire-keeps"), pytest.param(False, False, id="reissue-demotes")],
    )
    def test_the_cleanup_pass_after_an_attempt_that_followed_keeps_the_followed_ids(
        self, all_done: bool, awaiting_cleanup: bool
    ) -> None:
        pending = _two_titles_stored(awaiting_cleanup=True)
        result = attempt_result(
            all_done=all_done,
            command_issued=not all_done,
            pause_reason=PauseReason.NONE if all_done else PauseReason.ISSUED,
            pending=_followed(pending),
            followed=True,
        )
        strategy = _RecordingStrategy(completed=result, close_return=EffectStatus.FAILED)
        mgr = self._manager(strategy, FakeCacheStore(), pending)

        mgr._cleanup.finish_cleanups()

        row = mgr._pending_imports.rows()["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (awaiting_cleanup, _FOLLOWED_ROW)

    @pytest.mark.parametrize(
        ("stored", "ended", "followed", "uncached"),
        [
            pytest.param(_two_titles_stored(), _followed(_two_titles_stored()), True, [5, 6], id="followed-ids"),
            pytest.param(
                pending_import(infohash="h", anilist_id=5, file_episode_map={}),
                pending_import(infohash="h", anilist_id=5),
                False,
                [],
                id="placements-alone",
            ),
        ],
    )
    def test_a_wait_attempt_stores_what_it_ended_with_and_uncaches_only_titles_whose_ids_moved(
        self, stored: PendingImport, ended: PendingImport, followed: bool, uncached: list[int]
    ) -> None:
        store = FakeCacheStore()
        for anilist_id in (5, 6, 9):
            store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": "2021-06-05 00:00:00"})
        issued = attempt_result(
            all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED, pending=ended, followed=followed
        )
        mgr = self._manager(_RecordingStrategy(completed=issued), store, stored)

        mgr.run_deferred(view=RecordingWaitView())

        assert store.get_pending_import_rows(Arr.SONARR)["h"] == ended.to_json()
        titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (5, 6, 9)}
        assert [key for key, title in titles.items() if title is not None and title.updated_at is None] == uncached

    def test_a_wait_import_whose_cleanup_fails_is_kept_with_the_followed_ids_and_still_counted(self) -> None:
        pending = _two_titles_stored()
        imported = attempt_result(all_done=True, pending=_followed(pending), followed=True)
        store = FakeCacheStore()
        mgr = self._manager(_RecordingStrategy(completed=imported, close_return=EffectStatus.FAILED), store, pending)

        result = mgr.run_deferred(view=RecordingWaitView())

        row = store.get_pending_import_rows(Arr.SONARR)["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (True, _FOLLOWED_ROW)
        assert result is not None
        assert result.carried_over_imported == 1


class TestPendingImportWrites:
    """`PendingImportStore.drop` keeps the durable store and the run list in step."""

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


_COMPLETE: dict[str, list[QbitStep]] = {"h": [FakeTorrent(is_complete=True, content_path="/d")]}
"""Torrent "h" complete in qBittorrent, as a `FakeQbit` script."""


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

    def test_a_blocking_wait_import_of_a_two_title_torrent_moves_and_closes_once(self) -> None:
        pending = two_claim_pending_import(infohash="h")
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending],
            pending=[pending],
            post_import_category="pearlarr-done",
            clock=FakeClock(step=30),
        )

        mgr.run_blocking(view=RecordingWaitView())

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}


def _movie_import(download_id: str) -> HistoryRecord:
    """One Radarr `/history/since` import record keyed by the torrent's downloadId."""

    return HistoryRecord.model_validate({"eventType": "movieFolderImported", "downloadId": download_id, "movieId": 5})


def _radarr_reconcile_manager(
    *, qbit: CategoryQbit, history: list[HistoryRecord] | None, store: FakeCacheStore, post_import_category: str
) -> ImportWaitManager:
    """A manager wired for a Radarr run: a real `RadarrStrategy` reconciling off scripted history.

    The manager and the strategy share `store`, so a pending import the reconcile
    drops is gone for the cross-arr category count. The `_state.arr` is RADARR, so
    the arr-scoped steps read the Radarr pending imports.
    """

    config = make_config(radarr_post_import_category=post_import_category)
    radarr = FakeRadarrClient()
    radarr.history_since_return = history
    return make_import_wait_manager(
        config=config,
        qbit=qbit,
        cache_store=store,
        categories=make_categories(config, Arr.RADARR),
        strategy=make_radarr_sync(radarr=radarr, config=config, cache_store=store),
        reporter=_RecordingReporter(),
        state=ArrRunState(arr=Arr.RADARR),
    )


class TestRemoveFromQueue:
    """_remove_from_queue: a confirmed import dismisses Sonarr's leftover queue record.

    One pending import spans every title on the torrent, so the close consults no sibling
    (see `ImportCompleter.remove_from_queue`). Only the category move waits on the other arr.
    """

    @pytest.mark.parametrize(
        ("torrents", "remove_from_queue", "moves", "closes"),
        [
            pytest.param(_COMPLETE, True, [("pearlarr-done", "h")], ["h"], id="imported-closes"),
            pytest.param(_COMPLETE, False, [("pearlarr-done", "h")], [], id="turned-off-never-closes"),
            pytest.param({}, True, [], [], id="gone-torrent-is-left-alone"),
        ],
    )
    def test_only_a_confirmed_import_closes_the_queue_record_and_only_when_turned_on(
        self,
        torrents: dict[str, list[QbitStep]],
        remove_from_queue: bool,
        moves: list[tuple[str, str]],
        closes: list[str],
    ) -> None:
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        qbit = CategoryQbit(torrents)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            remove_from_queue=remove_from_queue,
            post_import_category="pearlarr-done",
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert [c.infohash for c in strategy.close_calls] == closes
        assert mgr._pending_imports.rows() == {}  # imported or gone, the pending import drops


class TestCleanupCategoryResolve:
    """clean_up_and_drop resolves no post-import category when there is no qBittorrent client."""

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

    @pytest.mark.parametrize(
        ("grab", "failed_moves", "moves", "closes", "kept"),
        [
            pytest.param("tv-sonarr", 0, [("sonarr-done", "h")], [], [], id="moved-out-of-sight"),
            pytest.param("tv-sonarr", 3, [], [], ["h"], id="move-failed-this-run"),
            # A blank category on Sonarr's client watches every category, so no move clears the queue record.
            pytest.param(None, 0, [("sonarr-done", "h")], ["h"], [], id="sonarr-watches-every-category"),
        ],
    )
    def test_only_a_move_out_of_sonarrs_sight_takes_the_place_of_the_explicit_close(
        self, grab: str | None, failed_moves: int, moves: list[tuple[str, str]], closes: list[str], kept: list[str]
    ) -> None:
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True))
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * failed_moves
        qbit = CategoryQbit(_COMPLETE, set_errors=errors)
        mgr = self._manager(grab=grab, post_import="sonarr-done", qbit=qbit, strategy=strategy)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert [c.infohash for c in strategy.close_calls] == closes
        assert list(mgr._pending_imports.rows()) == kept


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

    @pytest.mark.parametrize(
        ("waiting_in", "moves"),
        [
            pytest.param("manual", [], id="unseen-stays-out"),
            pytest.param("tv-sonarr", [("tv-sonarr/done", "h")], id="seen-moves"),
        ],
    )
    def test_only_a_torrent_sonarr_already_sees_moves_into_a_watched_subcategory(
        self, waiting_in: str, moves: list[tuple[str, str]]
    ) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category=waiting_in)]})

        self._clean_up(qbit, post_import="tv-sonarr/done")

        assert qbit.set_category_calls == moves

    @pytest.mark.parametrize(
        "torrents",
        [{"h": [FakeTorrent(is_complete=True, content_path="/d", category="tv-sonarr/done")]}, {}],
        ids=["already-there", "gone-from-qbittorrent"],
    )
    def test_an_unreadable_subcategory_setting_never_holds_a_move_that_carries_nothing(
        self, torrents: dict[str, list[QbitStep]]
    ) -> None:
        qbit = CategoryQbit(torrents, preferences_error=qbittorrentapi.APIConnectionError("down"))

        retired, _ = self._try_clean_up(qbit, post_import="tv-sonarr/done")

        assert retired is True


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

    @pytest.mark.parametrize(
        ("close_results", "sleeps", "kept"),
        [
            pytest.param([EffectStatus.RETRY, EffectStatus.DONE], [1.0], [], id="recovers-after-one-pause"),
            pytest.param([EffectStatus.RETRY] * 3, [1.0, 3.0], [True], id="retries-run-out"),
            # FAILED is a queue read that already retried inside the transport.
            pytest.param([EffectStatus.FAILED], [], [True], id="failed-is-final"),
        ],
    )
    def test_the_close_retries_on_the_schedule_and_keeps_the_pending_import_when_it_gives_up(
        self, close_results: list[EffectStatus], sleeps: list[float], kept: list[bool]
    ) -> None:
        recording = install_recording_hub()
        clock = FakeClock()
        strategy = _RecordingStrategy(progress=ImportProgress(1, 1, determinate=True), close_sequence=close_results)

        mgr = self._drive(strategy, clock)

        assert len(strategy.close_calls) == len(close_results)
        assert clock.sleeps == sleeps
        assert [raw.get("awaiting_cleanup") for raw in mgr._pending_imports.rows().values()] == kept
        assert ["(queue removal)" in warn for warn in diagnostic_messages(recording, Severity.WARNING)] == kept


class TestCleanupRetry:
    """Pending imports awaiting cleanup: the observe steps skip them, `finish_cleanups` finishes them once per run."""

    @pytest.mark.parametrize(
        ("run2_close", "finished"),
        [
            pytest.param(EffectStatus.DONE, True, id="next-run-finishes"),
            pytest.param(EffectStatus.FAILED, False, id="next-run-fails-again"),
        ],
    )
    def test_keep_then_next_run_finishes_the_cleanup_or_keeps_again(
        self, run2_close: EffectStatus, finished: bool
    ) -> None:
        # Run 1: the import is confirmed but the close fails -> awaiting cleanup, still
        # counted IMPORTED this run. Run 2: finishing the cleanup closes and drops,
        # or keeps the pending import again when the close still fails.
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
        run1.finish_cleanups()

        assert run1._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert run1._state.pending_import_states["h"] is PendingImportState.IMPORTED
        # The finish leaves a cleanup that failed this run for the next one.
        assert run1_strategy.import_calls == []

        recording = install_recording_hub()
        run2_strategy = _RecordingStrategy(close_return=run2_close)
        # The same durable store, next run.
        run2 = make_orchestration_manager(qbit=FakeQbit({}), strategy=run2_strategy, store=store)

        run2.finish_cleanups()
        run2.tally_carried_over_into_stats()

        assert [c.infohash for c in run2_strategy.close_calls] == ["h"]
        assert (store.get_pending_import_rows(Arr.SONARR) == {}) is finished
        assert (
            any("Finished the leftover post-import cleanup" in m for m in diagnostic_messages(recording, Severity.INFO))
            is finished
        )
        # Run 1 counted the import, so run 2 counts nothing for it either way.
        stats = run2._state.stats
        assert (run2._state.pending_import_states, stats.downloading, stats.downloaded) == ({}, 0, 0)

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

    def test_done_files_run_the_effects_at_the_final_attempt(self) -> None:
        strategy = _RecordingStrategy(completed=attempt_result(all_done=True))
        qbit = CategoryQbit(_COMPLETE)
        mgr = self._manager_with_awaiting_cleanup(strategy, qbit=qbit, post_import_category="pearlarr-done")

        mgr.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert mgr._pending_imports.rows() == {}

    @pytest.mark.parametrize(
        "waiting",
        [
            pytest.param(attempt_result(all_done=False), id="no-command"),
            pytest.param(
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                id="command-still-in-flight",
            ),
        ],
    )
    def test_transient_retry_attempt_result_stays_awaiting_cleanup(self, waiting: AttemptResult) -> None:
        # Only a re-import issued just now says the files are gone. A blip, a busy queue or a command
        # still in flight is ambiguous: keep it awaiting cleanup and retry next run.
        strategy = _RecordingStrategy(completed=waiting)
        mgr = self._manager_with_awaiting_cleanup(strategy)

        mgr.finish_cleanups()

        assert strategy.close_calls == []
        assert mgr._pending_imports.rows()["h"].get("awaiting_cleanup") is True
        assert "h" not in mgr._state.pending_import_states

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
        radarr_mgr = _radarr_reconcile_manager(
            qbit=qbit,
            history=[_movie_import("h")],
            store=store,
            post_import_category="pearlarr-done",
        )

        radarr_mgr.run_deferred(view=RecordingWaitView())

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert store.get_pending_import_rows(Arr.RADARR) == {}

    @pytest.mark.parametrize("awaiting_cleanup", [False, True], ids=["first-import", "cleanup-finish"])
    def test_radarr_on_the_torrent_holds_the_move_and_never_the_queue_removal(self, awaiting_cleanup: bool) -> None:
        strategy = _RecordingStrategy(
            completed=attempt_result(all_done=True), progress=ImportProgress(1, 1, determinate=True)
        )
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=awaiting_cleanup)],
            post_import_category="pearlarr-done",
        )
        store.put_pending_import_row(Arr.RADARR, "h", pending_import(infohash="h", series_id=0).to_json())

        mgr.report_imports_for_series(7)
        mgr.finish_cleanups()

        assert qbit.set_category_calls == []
        assert [c.infohash for c in strategy.close_calls] == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}


class TestRadarrReconcile:
    """A Radarr pending import is done once Radarr's history shows its import, and waits until then."""

    @pytest.mark.parametrize(
        ("history", "awaiting_cleanup", "moves", "kept"),
        [
            pytest.param([], False, [], [False], id="no-event-yet"),
            pytest.param(None, False, [], [False], id="history-outage"),
            pytest.param(None, True, [], [True], id="history-outage-while-awaiting-cleanup"),
            pytest.param([_movie_import("h")], True, [("pearlarr-done", "h")], [], id="event-finishes-the-cleanup"),
        ],
    )
    def test_only_an_import_event_in_radarrs_history_moves_the_torrent_and_drops_the_pending_import(
        self,
        history: list[HistoryRecord] | None,
        awaiting_cleanup: bool,
        moves: list[tuple[str, str]],
        kept: list[bool],
    ) -> None:
        store = FakeCacheStore()
        movie = pending_import(infohash="h", series_id=0, awaiting_cleanup=awaiting_cleanup)
        store.put_pending_import_row(Arr.RADARR, "h", movie.to_json())
        qbit = CategoryQbit(_COMPLETE)
        mgr = _radarr_reconcile_manager(qbit=qbit, history=history, store=store, post_import_category="pearlarr-done")

        mgr.finish_cleanups()
        mgr.run_deferred(view=RecordingWaitView())

        assert qbit.set_category_calls == moves
        assert [bool(row.get("awaiting_cleanup")) for row in store.get_pending_import_rows(Arr.RADARR).values()] == kept
