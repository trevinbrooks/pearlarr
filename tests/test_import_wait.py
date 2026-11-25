# pyright: strict
# pyright: reportPrivateUsage=false
# The arr run finish tests still call the arr run loop's private finish on a bare loop and read its
# state. Strict re-flags that, and the repo disables reportPrivateUsage for tests.
"""Tests for the end-of-run import wait and the post-import cleanup.

The wait manager is built by its real constructor over a scripted qBittorrent, a recording strategy and a
`FakeClock`, so no wait sleeps for real. Each area drives a public entry and checks what reaches Sonarr and
qBittorrent, what is stored, and what the user is shown: the series report, the deferred and blocking wait
passes, the carried-over tally, the arr run's finish, and the cleanup that moves a torrent and removes its
queue record.
"""

import copy
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast, override

import httpx
import pytest
import qbittorrentapi
import respx

from pearlarr.arr_categories import ArrCategoryResolver, CategorySources
from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_state import ArrRunState
from pearlarr.arr_strategy import ImportCompleter
from pearlarr.config import AppConfig, Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.output import (
    SPARK_SAMPLES,
    Diagnostic,
    Phase,
    Severity,
    TitleHeader,
    TorrentView,
    WaitSnapshot,
)
from pearlarr.output.events import TitleState
from pearlarr.pending_follow import follow
from pearlarr.pending_import import PendingImport
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.torrent_service import AddStatus
from pearlarr.wait_types import (
    NO_ATTEMPT,
    NO_PROGRESS,
    PENDING_IMPORT_STATE_FOR_OUTCOME,
    AttemptKind,
    AttemptResult,
    EffectStatus,
    ImportProgress,
    ImportWaitMode,
    Outcome,
    PauseReason,
    PendingImportState,
)
from pearlarr.wait_view import WaitOutcomeRow, WaitResult, WaitView

from .builders import (
    CLIENT_SENTINEL,
    SEP,
    FakeCacheStore,
    FakeTorrents,
    attempt_result,
    download_client_json,
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
    sonarr_client_fields,
    sonarr_ep,
    title_claim,
    two_claim_pending_import,
)
from .fakes import (
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
    by the batched `poll_download_progress` (which keys its result off it), and
    `FakeQbit`'s batch read answers each row under the hash it was asked for.
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
        self.hash = ""


type QbitStep = FakeTorrent | BaseException
"""One scripted qBittorrent reading: an info row, or an error to raise (a KeyboardInterrupt is a Ctrl-C landing there)."""


class FakeQbit:
    """A scriptable qBittorrent client with per-hash, per-read scripts.

    `torrents` scripts the per-hash poll: each hash has its own
    ordered lifecycle of readings, and every single-hash `torrents_info` call
    advances THAT hash's script by one (clamping at the last, so a steady state
    repeats indefinitely). An unscripted hash reads as gone (`[]`). An
    Exception step is raised to exercise the transient-error path.

    The batched download progress read answers, as qBittorrent does, for every asked torrent the
    fake holds, each row under the hash it was asked for. `download_progress` scripts that read
    the same way. A held torrent with no such script reads as qBittorrent holds it now: its last
    poll reading, or its first before any poll. One whose last poll step was an error is left out.
    A batch read never consumes a poll script (and vice versa), so a test scripts torrent
    lifecycles declaratively without caring which read comes first. Hash matching is
    case-insensitive on both reads (real qBittorrent is).
    """

    def __init__(
        self,
        torrents: dict[str, list[QbitStep]] | None = None,
        *,
        download_progress: dict[str, list[QbitStep]] | None = None,
    ) -> None:
        self._torrents = {h.casefold(): list(steps) for h, steps in (torrents or {}).items()}
        self._download_progress = {h.casefold(): list(steps) for h, steps in (download_progress or {}).items()}
        self._heavy_index: dict[str, int] = {}
        self._batch_index: dict[str, int] = {}

    @staticmethod
    def _next(steps: list[QbitStep], index: dict[str, int], key: str) -> FakeTorrent:
        """Advance `key`'s cursor through `steps` (clamped). Raise Exception steps."""

        i = index.get(key, 0)
        index[key] = i + 1
        step = steps[min(i, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step

    def _held_now(self, key: str) -> FakeTorrent | None:
        """Torrent `key` as its poll script last read it, or None when that read was an error."""

        steps = self._torrents[key]
        step = steps[min(max(self._heavy_index.get(key, 0) - 1, 0), len(steps) - 1)]
        return step if isinstance(step, FakeTorrent) else None

    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[FakeTorrent]:
        if isinstance(torrent_hashes, str):
            steps = self._torrents.get(torrent_hashes.casefold())
            if steps is None:
                return []
            return [self._next(steps, self._heavy_index, torrent_hashes.casefold())]
        if not torrent_hashes:
            # qBittorrent reads an empty hash filter as no filter and answers with every torrent it holds.
            pytest.fail("qBittorrent was asked for torrents with an empty hash filter, which returns all of them")
        rows: list[FakeTorrent] = []
        for infohash in torrent_hashes:
            key = infohash.casefold()
            if key in self._download_progress:
                reading = self._next(self._download_progress[key], self._batch_index, key)
            elif key in self._torrents:
                reading = self._held_now(key)
            else:
                # Not a torrent qBittorrent holds.
                continue
            if reading is not None:
                row = copy.copy(reading)
                row.hash = infohash
                rows.append(row)
        return rows


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


@dataclass(frozen=True)
class _ImportCall:
    """One import attempt the wait asked the arr for."""

    infohash: str
    content_path: str
    attempt_kind: AttemptKind


class _Script[T]:
    """Answers handed out one per call, the last one repeating."""

    def __init__(self, answers: Sequence[T]) -> None:
        self._answers = list(answers)
        self._taken = 0

    def take(self) -> T:
        answer = self._answers[min(self._taken, len(self._answers) - 1)]
        self._taken += 1
        return answer


class _RecordingStrategy(FakeStrategy):
    """The arr's side of the wait: each call answers from its script, one answer per call, the last repeating.

    `attempts` answers `import_completed`, `progress` the import progress read and `removals` `remove_from_queue`.
    An exception in `attempts` or `progress` is raised, as a failed request would be. The attempts and the queue
    removals change something in the arr, so each is recorded. The progress read changes nothing and isn't.
    """

    def __init__(
        self,
        *,
        attempts: Sequence[AttemptResult | Exception] = (NO_ATTEMPT,),
        progress: Sequence[ImportProgress | Exception] = (NO_PROGRESS,),
        removals: Sequence[EffectStatus] = (EffectStatus.DONE,),
    ) -> None:
        super().__init__(items=[], anilist_ids={})
        self._attempts = _Script(attempts)
        self._progress = _Script(progress)
        self._removals = _Script(removals)
        self.import_calls: list[_ImportCall] = []
        self.removal_calls: list[str] = []
        """The hash of each pending import whose queue record the wait asked the arr to remove."""

    @override
    def import_completed(self, pending: PendingImport, content_path: str, attempt_kind: AttemptKind) -> AttemptResult:
        self.import_calls.append(_ImportCall(pending.infohash, content_path, attempt_kind))
        answer = self._attempts.take()
        if isinstance(answer, Exception):
            raise answer
        return answer

    @override
    def remove_from_queue(self, pending: PendingImport) -> EffectStatus:
        self.removal_calls.append(pending.infohash)
        return self._removals.take()

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        answer = self._progress.take()
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_orchestration_manager(
    *,
    qbit: FakeQbit | None,
    strategy: ImportCompleter,
    stored_pending_imports: list[PendingImport] | None = None,
    grabbed: list[PendingImport] | None = None,
    clock: FakeClock | None = None,
    store: FakeCacheStore | None = None,
    state: ArrRunState | None = None,
    **config_overrides: object,
) -> ImportWaitManager:
    """A wait manager over carried-over pending imports and this run's grabs, the reporter real.

    `stored_pending_imports` are the rows earlier runs left. `grabbed` are this run's grabs, stored too and on
    the run list. The store is a fresh fake and the state a fresh Sonarr arr run unless a test passes the one it
    reads back. Every other keyword is a config setting.
    """

    cache_store = store if store is not None else FakeCacheStore()
    state = state if state is not None else ArrRunState(arr=Arr.SONARR)
    state.grabbed_this_run.update({pending.infohash: pending for pending in grabbed or []})
    mgr = make_import_wait_manager(
        qbit=qbit,
        cache_store=cache_store,
        clock=clock if clock is not None else FakeClock(),
        strategy=strategy,
        state=state,
        **config_overrides,
    )
    for stored in [*(stored_pending_imports or []), *(grabbed or [])]:
        cache_store.put_pending_import_row(state.arr, stored.infohash, stored.to_json())
    return mgr


def _stored(store: FakeCacheStore, arr: Arr = Arr.SONARR) -> dict[str, bool]:
    """Each pending import `store` holds for `arr`, by hash, with whether it awaits cleanup."""

    return {key: bool(raw.get("awaiting_cleanup")) for key, raw in store.get_pending_import_rows(arr).items()}


class RecordingWaitView(WaitView):
    """Records every snapshot the wait pushes to the screen."""

    def __init__(self) -> None:
        self.snapshots: list[WaitSnapshot] = []

    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        self.snapshots.append(snapshot)

    @override
    def close(self) -> None:
        pass

    def final(self, key: str) -> TorrentView:
        """The torrent's row in the last recorded snapshot."""

        return next(t for t in self.snapshots[-1].torrents if t.key == key)

    def seen(self, key: str, phase: Phase) -> list[TorrentView]:
        """The torrent's row in each recorded snapshot that showed it in `phase`, in push order."""

        return [t for snap in self.snapshots for t in snap.torrents if t.key == key and t.phase is phase]


def _run_blocking_wait(
    strategy: _RecordingStrategy,
    qbit: FakeQbit,
    *,
    step: float = 30,
    download_timeout: int = 3600,
    import_timeout: int = 600,
    progress_poll_interval: int = 5,
    post_import_category: str | None = None,
) -> tuple[FakeCacheStore, RecordingWaitView]:
    """Drive `run_blocking` to completion over this run's one grab, on hash "h", polled every 30s.

    Returns the store the wait wrote and the view it pushed to.
    """

    store = FakeCacheStore()
    mgr = make_orchestration_manager(
        qbit=qbit,
        strategy=strategy,
        store=store,
        grabbed=[pending_import(infohash="h")],
        download_timeout=download_timeout,
        import_timeout=import_timeout,
        import_poll_interval=30,
        progress_poll_interval=progress_poll_interval,
        post_import_category=post_import_category,
        clock=FakeClock(step=step),
    )
    view = RecordingWaitView()
    mgr.run_blocking(view=view)
    return store, view


class TestReportImportsForSeries:
    """The series report shows each carried-over pending import of the series once, as qBittorrent and the files read.

    It only reads: it never attempts an import, and the `imported` count is the arr run finish's.
    """

    @pytest.mark.parametrize(
        ("torrent", "progress", "state", "shown", "stored"),
        [
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                ImportProgress(1, 1, determinate=True),
                PendingImportState.IMPORTED,
                [(TitleState.IMPORTED, f"Show{SEP}SubGroup")],
                {},
                id="every-file-imported",
            ),
            pytest.param(
                FakeTorrent(is_complete=True),
                NO_PROGRESS,
                PendingImportState.DOWNLOADED,
                [(TitleState.DOWNLOADED, f"Show{SEP}SubGroup")],
                {"h": False},
                id="files-not-in-yet",
            ),
            # The wait pass shows the error on its row. The tally counts an errored torrent nowhere.
            pytest.param(
                FakeTorrent(is_errored=True), NO_PROGRESS, PendingImportState.ERRORED, [], {"h": False}, id="errored"
            ),
        ],
    )
    def test_a_carried_over_pending_import_is_reported_as_qbittorrent_and_the_files_read(
        self,
        torrent: FakeTorrent,
        progress: ImportProgress,
        state: PendingImportState,
        shown: list[tuple[TitleState, str]],
        stored: dict[str, bool],
    ) -> None:
        strategy = _RecordingStrategy(progress=[progress])
        store = FakeCacheStore()
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=strategy,
            store=store,
            state=run,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert [(h.state, h.title) for h in recording.of_type(TitleHeader)] == shown
        assert run.pending_import_states == {"h": state}
        assert _stored(store) == stored
        assert strategy.import_calls == []
        assert run.stats.imported == 0

    def test_a_carried_over_torrent_gone_from_qbittorrent_is_warned_about_and_dropped(self) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),
            strategy=_RecordingStrategy(),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert f"Show{SEP}SubGroup is gone from qBittorrent" in warning
        assert _stored(store) == {}

    @pytest.mark.parametrize(
        ("series_id", "state"),
        [
            # A stored pending import re-seen in qBittorrent this run was already reported by its title block.
            # Only the tally counts it.
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
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=series_id)],
            state=state,
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert recording.of_type(TitleHeader) == []
        assert state.pending_import_states == {}
        assert _stored(store) == {"h": False}

    def test_a_two_series_pending_import_is_reported_once_under_the_series_that_scans_first(self) -> None:
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
            stored_pending_imports=[pending],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(8)
        mgr.report_imports_for_series(7)

        assert [(h.state, h.coverage, h.url) for h in recording.of_type(TitleHeader)] == [
            (TitleState.DOWNLOADING, "S02 E01-E12", "https://releases.moe/22"),
        ]

    def test_a_torrent_the_grab_found_already_in_qbittorrent_is_left_to_the_end_pass_and_imports_there(self) -> None:
        # The grab starts its pending import and its title block reports it, so the series report leaves it be.
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)
        store = FakeCacheStore()
        pipeline = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), cache_store=store, _state=state
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h1": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            store=store,
            state=state,
            clock=FakeClock(step=30),
        )
        pipeline.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_import_updates={"h1": pending_import_update("h1", series_id=7)},
            ),
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)
        result = mgr.run_blocking(view=RecordingWaitView())

        assert recording.of_type(TitleHeader) == []
        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.IMPORTED, True)]
        assert _stored(store) == {}


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
                attempts=[attempt_result(all_done=True)], progress=[ImportProgress(1, 1, determinate=True)]
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
            strategy=_RecordingStrategy(attempts=attempts, progress=progress),
            store=store,
            stored_pending_imports=[pending_import(infohash="h1"), pending_import(infohash="h2")],
            grabbed=[pending_import(infohash="new")],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert result.carried_over_imported == 2
        assert _stored(store) == {"new": False}

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
            strategy=_RecordingStrategy(attempts=attempts),
            store=store,
            state=run,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(outcome, True)]
        assert run.pending_import_states == {"h": state}
        assert _stored(store) == {"h": False}

    def test_the_end_pass_state_replaces_the_one_the_series_report_noted(self) -> None:
        # The series report read the torrent still downloading, and it finished before the end pass.
        run = ArrRunState(arr=Arr.SONARR, pending_import_states={"h": PendingImportState.DOWNLOADING})
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=False)]),
            state=run,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        mgr.run_deferred(view=RecordingWaitView())

        assert run.pending_import_states == {"h": PendingImportState.DOWNLOADED}

    def test_a_deferred_pass_never_times_out_a_download(self) -> None:
        # The download clock starts when the pass is built, so a timeout here would only measure the pass.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.5)]}),
            strategy=_RecordingStrategy(),
            stored_pending_imports=[pending_import(infohash="h")],
            download_timeout=1,
            clock=_AutoTickClock(step=30),
        )

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [row.outcome for row in result.rows] == [Outcome.STILL_DOWNLOADING]


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
        return _RecordingStrategy(attempts=[attempt_result(all_done=False, unmatched_files=cls._NAMES)])

    def test_run_deferred_finishes_unmatched_and_keeps_the_pending_import(self) -> None:
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
        assert _stored(store) == {"h": False}

    def test_run_blocking_finishes_unmatched_on_the_first_poll(self) -> None:
        # The result ends the wait on the poll that establishes it: no final attempt, no import clock.
        strategy = self._strategy()
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=600,
        )

        final = view.final("h")
        assert final.outcome is Outcome.NOTHING_PLACED
        assert final.left_for_hand_import.names == self._NAMES
        assert final.phase_elapsed_s < 600
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY]
        assert _stored(store) == {"h": False}

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
        assert _stored(store) == {"h": False, "slow": False}


class TestLeftForHandImportRowFinish:
    """A result naming hand-import files finishes as a partial import: kept for the hand import, never cleaned up."""

    def test_each_run_names_the_hand_import_files_and_leaves_the_torrent_in_its_category(self) -> None:
        left = order_only_hand_import_files()
        store = FakeCacheStore()
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]})
        run1 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=False, left_for_hand_import=left)]),
            store=store,
            stored_pending_imports=[pending_import(infohash="h")],
            post_import_category="pearlarr-done",
        )
        # The same store, next run.
        run2_state = ArrRunState(arr=Arr.SONARR)
        run2 = make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=False, left_for_hand_import=left)]),
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
        assert _stored(store) == {"h": False}
        assert qbit.set_category_calls == []


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


class _InterruptedDropStore(FakeCacheStore):
    """A store where a Ctrl-C lands in the pending import drop."""

    @override
    def drop_pending_import_row(self, arr: Arr, infohash: str) -> None:
        raise KeyboardInterrupt


class _InterruptedPutStore(FakeCacheStore):
    """A store where a Ctrl-C lands in the write that marks a pending import awaiting cleanup."""

    @override
    def put_pending_import_row(self, arr: Arr, infohash: str, row: dict[str, Any]) -> None:
        if row.get("awaiting_cleanup"):
            raise KeyboardInterrupt
        super().put_pending_import_row(arr, infohash, row)


class TestBlockingWaitRowDiscriminator:
    """Wait rows carry `carried_over`, so _finish_arr_run can tell them from fresh grabs."""

    def test_wait_rows_tell_fresh_from_carried_over(self) -> None:
        # One fresh grab + one carried-over pending import, both import: only the
        # carried-over row is marked, even though the fresh import leaves
        # `grabbed_this_run` mid-pass.
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=True)],
        )
        fresh = pending_import(infohash="f", title="Fresh")
        carried = pending_import(infohash="c", title="Carried")
        qbit = FakeQbit(
            {
                "f": [FakeTorrent(is_complete=True, content_path="/f")],
                "c": [FakeTorrent(is_complete=True, content_path="/c")],
            },
        )
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            state=run,
            stored_pending_imports=[carried],
            grabbed=[fresh],
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
        assert run.pending_import_states == {}
        assert run.stats.imported == 0


class TestRunBlocking:
    """run_blocking: interleaved, copy-aware wait+import over ALL pending imports."""

    def test_two_torrents_each_import_from_their_own_path(self) -> None:
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=True)])
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
            grabbed=pending_imports,
            import_poll_interval=30,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        paths = {call.infohash: call.content_path for call in strategy.import_calls}
        assert paths == {"fast": "/fast", "slow": "/slow"}
        assert [view.final(key).outcome for key in ("fast", "slow")] == [Outcome.IMPORTED, Outcome.IMPORTED]
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    def test_with_the_progress_refresh_off_only_the_next_poll_confirms_an_import(self) -> None:
        # A progress read between the polls would find every file in and finish the row before poll 2.
        strategy = _RecordingStrategy(
            attempts=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True, command_issued=True),
            ],
            progress=[ImportProgress(1, 1, determinate=True)],
        )
        _, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            progress_poll_interval=0,
        )

        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY, AttemptKind.ORDINARY]
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_progress_poll_fills_bar_and_promotes_before_the_next_poll(self) -> None:
        # Between polls the cheap progress poll fills the "files inserted" bar
        # as files get done and promotes the row to IMPORTED the instant all are done
        # - no second import_completed poll, no RefreshMonitoredDownloads.
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True)],
            progress=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),
                ImportProgress(3, 3, determinate=True),  # all done -> promote
            ],
        )
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            progress_poll_interval=5,
        )

        # Only ONE poll. The progress poll did the rest.
        assert len(strategy.import_calls) == 1
        # The bar advanced (2/3 seen) before the row finished.
        assert any(
            t.key == "h" and t.import_done == 2 and t.import_total == 3
            for snap in view.snapshots
            for t in snap.torrents
        )
        assert view.final("h").outcome is Outcome.IMPORTED
        assert _stored(store) == {}

    def test_final_attempt_rescued_by_a_same_poll_rise(self) -> None:
        # The forced deadline attempt itself reports a fresh rise (0/3 -> 1/3 at
        # t=60): the row must NOT finish that poll - the done file re-anchors
        # the deadline instead - and only the NEXT quiet deadline (t=120) ends it.
        strategy = _RecordingStrategy(
            attempts=[
                attempt_result(
                    all_done=False,
                    command_issued=True,
                    imported_count=done,
                    file_map_episode_count=3,
                )
                for done in (0, 0, 1)
            ],
        )
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert _stored(store) == {"h": False}
        # t=0/30 in-bound, t=60 forced-but-rescued, t=90 in-bound, t=120 forced.
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, False, True]

    def test_final_auto_import_waits_its_own_copy_out(self) -> None:
        # Two stalled polls, then the final attempt POSTs (ISSUED, paused):
        # the row is not finished "still importing" on that same poll, the
        # copy in flight keeps pausing the clock, and the done-check ends IMPORTED.
        strategy = _RecordingStrategy(
            attempts=[
                attempt_result(all_done=False),
                attempt_result(all_done=False),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                attempt_result(all_done=True, command_issued=True),
            ],
        )
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert _stored(store) == {}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, True, True]

    def test_fresh_post_pauses_once_per_row(self) -> None:
        # A POST whose files all fail inside Sonarr is re-POSTed every poll:
        # only the first pauses the clock, so the retry loop finishes at the
        # ordinary deadline (t=60), never going on to the cap.
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED)],
        )
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert _stored(store) == {"h": False}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True]

    def test_pause_cap_reads_still_importing_for_an_import_pause(self) -> None:
        # Sonarr's own import (or an unproven one) still running at the cap:
        # no command of pearlarr's, yet the label says "still importing", not "not
        # ready". Ready 60s + cap (6x) 360s -> finished by t=420.
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=False, pause_reason=PauseReason.IMPORT)])
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert _stored(store) == {"h": False}
        assert strategy.import_calls[-1].attempt_kind.at_deadline is True
        assert len(strategy.import_calls) == 15

    def test_pause_cap_reads_sonarr_busy_for_another_downloads_disk_command(self) -> None:
        # A foreign disk command holding the line to the cap: "sonarr busy",
        # left for the next run, never a false "still importing".
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=False, pause_reason=PauseReason.BUSY)])
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.SONARR_BUSY
        assert _stored(store) == {"h": False}

    def test_a_progress_poll_rise_re_anchors_the_deadline(self) -> None:
        # The cheap progress poll sees a file get done (1/3 -> 2/3) even though the
        # poll's counts stay indeterminate. That rise alone must push the
        # deadline out one more poll (t=90, not t=60).
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True)],
            progress=[
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
        "reading",
        [
            pytest.param(FakeTorrent(progress=0.5), id="still-downloading"),
            # qBittorrent down for the whole wait still ends it at the download timeout.
            pytest.param(qbittorrentapi.APIConnectionError("down"), id="qbittorrent-unreachable"),
        ],
    )
    def test_a_download_that_never_finishes_times_out_and_is_left_for_the_next_run(self, reading: QbitStep) -> None:
        strategy = _RecordingStrategy()
        qbit = CategoryQbit({"h": [reading]})
        store, view = _run_blocking_wait(
            strategy,
            qbit,
            post_import_category="pearlarr-done",
            download_timeout=60,
        )

        assert strategy.import_calls == []
        assert view.final("h").outcome is Outcome.DOWNLOAD_TIMED_OUT
        assert _stored(store) == {"h": False}
        assert qbit.set_category_calls == []

    def test_a_failed_progress_read_is_skipped_and_the_next_poll_imports(self) -> None:
        strategy = _RecordingStrategy(
            attempts=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True, command_issued=True),
            ],
            progress=[RuntimeError("progress boom"), NO_PROGRESS],
        )
        _, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            progress_poll_interval=5,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY, AttemptKind.ORDINARY]

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
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=_RecordingStrategy(),
            store=store,
            grabbed=[pending_import(infohash="h")],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
            clock=clock,
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        with ending:
            mgr.run_blocking(view=view)

        assert _stored(store) == {"h": False}
        assert diagnostic_messages(recording) == ["Wait interrupted - 1 left for the next run"]

    def test_interrupt_mid_poll_still_finishes_that_polls_finished_rows(self) -> None:
        # A torrent that finished in the SAME poll the interrupt lands in
        # must still reach the view (the except handler pushes one final snapshot), so
        # the console tally/outcome lines can never undercount the returned WaitResult.
        # h1 is gone, so it finishes on its first poll. The Ctrl-C lands on h2's poll, before that poll's push.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h2": [KeyboardInterrupt()]}),
            strategy=_RecordingStrategy(),
            grabbed=[pending_import(infohash="h1"), pending_import(infohash="h2")],
            download_timeout=3600,
            import_timeout=600,
            import_poll_interval=30,
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        result = mgr.run_blocking(view=view)  # must not raise

        assert result is not None and result.waited == 1
        assert view.final("h1").outcome is Outcome.GONE  # the interrupt-time push carried it
        assert diagnostic_messages(recording) == ["Wait interrupted - 1 left for the next run"]

    def test_import_exception_is_swallowed_announced_and_pending_import_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate
        # and abort _finish_arr_run's cache save: the failure lands as an ERROR
        # Diagnostic carrying its traceback and the pending import is left.
        strategy = _RecordingStrategy(attempts=[RuntimeError("boom")])
        recording = install_recording_hub()
        store, view = _run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.ATTEMPT_FAILED
        assert _stored(store) == {"h": False}
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert "Auto import failed" in error.message
        assert error.trace is not None

    @pytest.mark.parametrize(
        ("store_type", "removals"),
        [
            pytest.param(_InterruptedDropStore, [EffectStatus.DONE], id="in-the-drop"),
            # The queue removal fails, so the import is kept awaiting cleanup: that write is where the Ctrl-C lands.
            pytest.param(_InterruptedPutStore, [EffectStatus.FAILED], id="in-the-awaiting-cleanup-write"),
        ],
    )
    def test_a_ctrl_c_in_the_store_write_of_an_import_counts_no_import(
        self, store_type: type[FakeCacheStore], removals: list[EffectStatus]
    ) -> None:
        # The run's tally counts the result rows, so an import counts only once its store write went through.
        store = store_type()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=True)], removals=removals),
            store=store,
            stored_pending_imports=[pending_import(infohash="h")],
        )
        view = RecordingWaitView()

        result = mgr.run_blocking(view=view)

        assert result is not None
        assert result.rows == ()
        assert view.final("h").outcome is None
        assert _stored(store) == {"h": False}


@dataclass(frozen=True)
class _Bar:
    """A downloading row's download bar as the wait screen shows it."""

    fraction: float
    speed_bps: int | None
    eta_s: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    speed_history: tuple[int, ...] = ()


def _bars(view: RecordingWaitView, key: str) -> list[_Bar]:
    """The torrent's download bar in each pushed snapshot that showed it downloading, in push order."""

    return [
        _Bar(t.fraction, t.speed_bps, t.eta_s, t.bytes_done, t.bytes_total, t.speed_history)
        for t in view.seen(key, Phase.DOWNLOADING)
    ]


_DONE = FakeTorrent(is_complete=True, content_path="/d", progress=1.0)
"""A torrent's last reading: downloaded, ready to import."""


class TestBlockingWaitDownloadBars:
    """Each poll shows the downloading rows' bars, and a refresh between polls moves them and nothing else."""

    _POLLED = FakeTorrent(progress=0.3, dlspeed=100, eta=130, completed=300, size=1000)
    _POLLED_BAR = _Bar(0.3, 100, 130, 300, 1000, (100,))

    @pytest.mark.parametrize(
        ("wants_bars", "refreshed", "bars"),
        [
            pytest.param(
                True,
                FakeTorrent(progress=0.6, dlspeed=999, eta=40, completed=600, size=1000),
                [_POLLED_BAR, _Bar(0.6, 999, 40, 600, 1000, (100,))],
                id="moved",
            ),
            # Even at 100% the row stays downloading: only a poll finishes it.
            pytest.param(
                True,
                FakeTorrent(progress=1.0, dlspeed=250, eta=5, completed=1000, size=1000),
                [_POLLED_BAR, _Bar(1.0, 250, 5, 1000, 1000, (100,))],
                id="moved-to-full",
            ),
            pytest.param(True, _POLLED, [_POLLED_BAR], id="unchanged"),
            pytest.param(True, qbittorrentapi.APIConnectionError("down"), [_POLLED_BAR], id="the-read-fails"),
            pytest.param(
                False,
                FakeTorrent(progress=0.6, dlspeed=999, eta=40, completed=600, size=1000),
                [_POLLED_BAR],
                id="a-view-without-bars",
            ),
        ],
    )
    def test_a_refresh_between_polls_moves_only_the_download_bar(
        self, wants_bars: bool, refreshed: QbitStep, bars: list[_Bar]
    ) -> None:
        # The speed history is sampled once per poll, never by a refresh.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [self._POLLED, _DONE]}, download_progress={"h": [refreshed]}),
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            grabbed=[pending_import(infohash="h")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()
        view.wants_download_progress = wants_bars

        mgr.run_blocking(view=view)

        assert _bars(view, "h") == bars
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_a_refresh_moves_each_downloading_rows_own_bar(self) -> None:
        mgr = make_orchestration_manager(
            qbit=FakeQbit(
                {
                    "a": [FakeTorrent(progress=0.3, dlspeed=100), _DONE],
                    "b": [FakeTorrent(progress=0.3, dlspeed=100), _DONE],
                },
                download_progress={
                    "a": [FakeTorrent(progress=0.6, dlspeed=999)],
                    "b": [FakeTorrent(progress=0.8, dlspeed=500)],
                },
            ),
            strategy=_RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            grabbed=[pending_import(infohash="a"), pending_import(infohash="b")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert _bars(view, "a") == [_Bar(0.3, 100, speed_history=(100,)), _Bar(0.6, 999, speed_history=(100,))]
        assert _bars(view, "b") == [_Bar(0.3, 100, speed_history=(100,)), _Bar(0.8, 500, speed_history=(100,))]

    def test_a_refresh_leaves_an_importing_rows_bar_alone(self) -> None:
        # "imp" is importing with 1 file of 3 in while "dl" still downloads. qBittorrent reads "imp" at 100%,
        # but an importing row's bar counts the files Sonarr has in.
        importing = attempt_result(all_done=False, command_issued=True, imported_count=1, file_map_episode_count=3)
        mgr = make_orchestration_manager(
            qbit=FakeQbit(
                {"dl": [FakeTorrent(progress=0.3, dlspeed=100), _DONE], "imp": [_DONE]},
                download_progress={"dl": [FakeTorrent(progress=0.6, dlspeed=999)]},
            ),
            strategy=_RecordingStrategy(attempts=[importing, attempt_result(all_done=True)]),
            grabbed=[pending_import(infohash="dl"), pending_import(infohash="imp")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        # The first poll's frame, then the one the refresh pushed when "dl"'s bar moved.
        assert [(t.fraction, t.import_done, t.import_total) for t in view.seen("imp", Phase.IMPORTING)] == [
            (1 / 3, 1, 3),
            (1 / 3, 1, 3),
        ]

    def test_a_poll_qbittorrent_misses_keeps_the_last_bar_and_samples_no_speed(self) -> None:
        # No 0% flash and no fake stall in the sparkline: the blip poll shows the last real reading again.
        store, view = _run_blocking_wait(
            _RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            FakeQbit(
                {
                    "h": [
                        FakeTorrent(progress=0.3, dlspeed=100),
                        qbittorrentapi.APIConnectionError("blip"),
                        FakeTorrent(progress=0.4, dlspeed=200),
                        _DONE,
                    ]
                }
            ),
            progress_poll_interval=0,
        )

        assert _bars(view, "h") == [
            _Bar(0.3, 100, speed_history=(100,)),
            _Bar(0.3, 100, speed_history=(100,)),
            _Bar(0.4, 200, speed_history=(100, 200)),
        ]
        assert _stored(store) == {}

    @pytest.mark.parametrize(
        ("readings", "history"),
        [
            pytest.param(
                [
                    FakeTorrent(progress=0.1, dlspeed=100),
                    FakeTorrent(progress=0.2),
                    FakeTorrent(progress=0.3, dlspeed=300),
                ],
                (100, 0, 300),
                id="a-stall-is-a-zero",
            ),
            pytest.param(
                [FakeTorrent(progress=0.1, dlspeed=100), *[FakeTorrent(progress=0.3, dlspeed=300)] * SPARK_SAMPLES],
                (300,) * SPARK_SAMPLES,
                id="the-oldest-samples-fall-off",
            ),
        ],
    )
    def test_the_speed_sparkline_takes_one_sample_per_poll(
        self, readings: list[QbitStep], history: tuple[int, ...]
    ) -> None:
        _, view = _run_blocking_wait(
            _RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            FakeQbit({"h": [*readings, _DONE]}),
            progress_poll_interval=0,
        )

        assert _bars(view, "h")[-1].speed_history == history


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
        strategy = _RecordingStrategy(attempts=[result], removals=[EffectStatus.FAILED])
        store = FakeCacheStore()
        mgr = self._manager(strategy, store, pending)

        mgr.finish_cleanups()

        row = store.get_pending_import_rows(Arr.SONARR)["h"]
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
        mgr = self._manager(_RecordingStrategy(attempts=[issued]), store, stored)

        mgr.run_deferred(view=RecordingWaitView())

        assert store.get_pending_import_rows(Arr.SONARR)["h"] == ended.to_json()
        titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (5, 6, 9)}
        assert [key for key, title in titles.items() if title is not None and title.updated_at is None] == uncached

    def test_a_wait_import_whose_cleanup_fails_is_kept_with_the_followed_ids_and_still_counted(self) -> None:
        pending = _two_titles_stored()
        imported = attempt_result(all_done=True, pending=_followed(pending), followed=True)
        store = FakeCacheStore()
        mgr = self._manager(_RecordingStrategy(attempts=[imported], removals=[EffectStatus.FAILED]), store, pending)

        result = mgr.run_deferred(view=RecordingWaitView())

        row = store.get_pending_import_rows(Arr.SONARR)["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (True, _FOLLOWED_ROW)
        assert result is not None
        assert result.carried_over_imported == 1


def test_dropping_a_pending_import_takes_it_out_of_the_store_and_the_run_list() -> None:
    keep = pending_import(infohash="keep")
    drop = pending_import(infohash="drop")
    store = FakeCacheStore()
    state = ArrRunState(arr=Arr.SONARR)
    pending_imports = PendingImportStore(store)
    pending_imports.bind_arr_run(state)
    for grabbed in (keep, drop):
        pending_imports.insert_grabbed(grabbed, title_claim())

    pending_imports.drop("drop")

    assert set(store.get_pending_import_rows(Arr.SONARR)) == {"keep"}
    assert state.grabbed_this_run == {"keep": keep}


_COMPLETE: dict[str, list[QbitStep]] = {"h": [FakeTorrent(is_complete=True, content_path="/d")]}
"""Torrent "h" complete in qBittorrent, as a `FakeQbit` script."""


def _sdk_client(qbit: FakeQbit) -> qbittorrentapi.Client:
    """A fake as the SDK client it stands in for (the sanctioned leaf-client hatch: no interface to subclass)."""

    return cast("qbittorrentapi.Client", qbit)


_ARR_URL = "http://arr.test"


def _reading_sonarrs_clients(config: AppConfig, qbit: FakeQbit) -> ArrCategoryResolver:
    """A resolver reading Sonarr's download clients from `_ARR_URL`, with no retry inside a read, like a run's."""

    http, _ = bind_arr_http(_ARR_URL)
    return make_categories(config, sources=CategorySources(replace(http, retries=0), _sdk_client(qbit)))


class TestPostImportCategory:
    """A confirmed import moves its torrent to the arr's post-import category."""

    @staticmethod
    def _imported_manager(
        qbit: CategoryQbit, *, store: FakeCacheStore, state: ArrRunState, post_import_category: str | None
    ) -> ImportWaitManager:
        """A manager whose one carried-over pending import's files are done, so the series report confirms it.

        A `None` category routes through `make_config` to the blank-drop, so it
        exercises the same "left unset" default a real config file yields.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)]),
            store=store,
            state=state,
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
    def test_a_confirmed_import_moves_its_torrent_to_the_post_import_category_when_one_is_set(
        self,
        category: str | None,
        set_errors: list[Exception],
        expected_sets: list[tuple[str, str]],
        expected_created: list[str],
    ) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}, set_errors=set_errors)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = self._imported_manager(qbit, store=store, state=state, post_import_category=category)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == expected_sets
        assert qbit.created_categories == expected_created
        # The import itself always goes through: the pending import drops and the arr run counts it imported.
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert state.pending_import_states == {"h": PendingImportState.IMPORTED}

    def test_a_move_that_fails_every_retry_keeps_the_pending_import_for_a_later_cleanup(self) -> None:
        # The import still counts, and one warning names the effect left to finish.
        recording = install_recording_hub()
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * 3
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}, set_errors=errors)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = self._imported_manager(qbit, store=store, state=state, post_import_category="pearlarr-done")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert state.pending_import_states == {"h": PendingImportState.IMPORTED}
        assert _stored(store) == {"h": True}
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "post-import cleanup" in warning
        assert "(category move)" in warning

    def test_a_blocking_wait_import_of_a_two_title_torrent_moves_and_closes_once(self) -> None:
        pending = two_claim_pending_import(infohash="h")
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=True)])
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            grabbed=[pending],
            post_import_category="pearlarr-done",
            clock=FakeClock(step=30),
        )

        mgr.run_blocking(view=RecordingWaitView())

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}


class TestRemoveFromQueue:
    """A confirmed import removes the queue record Sonarr leaves behind.

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
        strategy = _RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)])
        qbit = CategoryQbit(torrents)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            remove_from_queue=remove_from_queue,
            post_import_category="pearlarr-done",
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert strategy.removal_calls == closes
        assert store.get_pending_import_rows(Arr.SONARR) == {}  # imported or gone, the pending import drops


class TestCleanupCategoryResolve:
    """The cleanup reads the post-import category once, and the move and the queue removal both act on that read."""

    @respx.mock
    def test_a_failed_client_read_moves_nothing_and_the_queue_removal_still_runs(self) -> None:
        # Sonarr answers a second read, naming a post-import category it doesn't watch. Had the removal read
        # again, it would skip for a move that never ran.
        respx.get(f"{_ARR_URL}/api/v3/downloadclient").mock(
            side_effect=[httpx.Response(500), httpx.Response(200, json=[download_client_json(sonarr_client_fields())])]
        )
        strategy = _RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)])
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            categories=_reading_sonarrs_clients(make_config(), qbit),
        )

        mgr.report_imports_for_series(7)

        # What happens today, not a decision: with no category read there is no move to make, so the cleanup
        # finishes and drops the pending import, and the torrent never reaches the post-import category.
        assert qbit.set_category_calls == []
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}


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
        store: FakeCacheStore,
    ) -> ImportWaitManager:
        """A manager whose resolver already fetched the arr's client categories (preloaded, transportless)."""

        return make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
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
        strategy = _RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)])
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * failed_moves
        qbit = CategoryQbit(_COMPLETE, set_errors=errors)
        store = FakeCacheStore()
        mgr = self._manager(grab=grab, post_import="sonarr-done", qbit=qbit, strategy=strategy, store=store)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert strategy.removal_calls == closes
        assert list(store.get_pending_import_rows(Arr.SONARR)) == kept


class TestMoveKeepsAnUnseenTorrentOut:
    """The post-import move never carries a torrent Sonarr can't see into a category Sonarr watches.

    Sonarr has no record of that download. Once the torrent showed up in its category, Sonarr would
    import the files again by its own reading of their names.
    """

    @staticmethod
    def _manager(
        qbit: CategoryQbit, store: FakeCacheStore, *, post_import: str, awaiting_cleanup: bool = False
    ) -> ImportWaitManager:
        """A manager over one stored pending import of series 7 whose files are in, Sonarr's clients watching `tv-sonarr`.

        With `awaiting_cleanup`, the import went through on an earlier run and only its cleanup is owed.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=_RecordingStrategy(
                attempts=[attempt_result(all_done=True)], progress=[ImportProgress(1, 1, determinate=True)]
            ),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=awaiting_cleanup)],
            categories=make_fetched_categories(
                make_config(), grab="tv-sonarr", post_import=post_import, qbit=_sdk_client(qbit)
            ),
        )

    def test_a_failed_category_read_is_retried_and_never_moves_blind(self) -> None:
        unseen = FakeTorrent(is_complete=True, content_path="/d", category="manual")
        # The poll reads the torrent first, then the move's read fails once.
        qbit = CategoryQbit({"h": [unseen, qbittorrentapi.APIConnectionError("down"), unseen]})
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}

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
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import=post_import)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert _stored(store) == {"h": True}

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
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr/done")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @pytest.mark.parametrize(
        "torrents",
        [{"h": [FakeTorrent(is_complete=True, content_path="/d", category="tv-sonarr/done")]}, {}],
        ids=["already-there", "gone-from-qbittorrent"],
    )
    def test_an_unreadable_subcategory_setting_never_holds_a_move_that_carries_nothing(
        self, torrents: dict[str, list[QbitStep]]
    ) -> None:
        # An owed cleanup, the one a torrent gone from qBittorrent still reaches.
        qbit = CategoryQbit(torrents, preferences_error=qbittorrentapi.APIConnectionError("down"))
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr/done", awaiting_cleanup=True)

        mgr.finish_cleanups()

        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @respx.mock
    def test_unreadable_sonarr_clients_hold_the_move_for_a_later_run(self) -> None:
        # Only Sonarr's clients say whether it watches `sonarr-done`. The configured grab category stands in
        # for them at the queue removal, never at the move.
        respx.get(f"{_ARR_URL}/api/v3/downloadclient").respond(status_code=500)
        config = make_config(torrent_category="anime", post_import_category="sonarr-done")
        strategy = _RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)])
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="manual")]})
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            config=config,
            categories=_reading_sonarrs_clients(config, qbit),
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == []
        assert store.get_pending_import_rows(Arr.SONARR)["h"].get("awaiting_cleanup") is True


class TestCloseRetrySchedule:
    """The close effect goes through the bounded in-run retry schedule off the injected clock."""

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
        strategy = _RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)], removals=close_results)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(_COMPLETE),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            clock=clock,
        )

        mgr.report_imports_for_series(7)

        assert len(strategy.removal_calls) == len(close_results)
        assert clock.sleeps == sleeps
        assert list(_stored(store).values()) == kept
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
            progress=[ImportProgress(1, 1, determinate=True)],
            removals=[EffectStatus.FAILED],
        )
        store = FakeCacheStore()
        run1_state = ArrRunState(arr=Arr.SONARR)
        run1 = make_orchestration_manager(
            qbit=FakeQbit(_COMPLETE),
            strategy=run1_strategy,
            store=store,
            state=run1_state,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        run1.report_imports_for_series(7)
        run1.finish_cleanups()

        assert _stored(store) == {"h": True}
        assert run1_state.pending_import_states == {"h": PendingImportState.IMPORTED}
        # The finish leaves a cleanup that failed this run for the next one.
        assert run1_strategy.import_calls == []

        recording = install_recording_hub()
        run2_strategy = _RecordingStrategy(removals=[run2_close])
        run2_state = ArrRunState(arr=Arr.SONARR)
        # The same durable store, next run.
        run2 = make_orchestration_manager(qbit=FakeQbit({}), strategy=run2_strategy, store=store, state=run2_state)

        run2.finish_cleanups()
        run2.tally_carried_over_into_stats()

        assert run2_strategy.removal_calls == ["h"]
        assert (store.get_pending_import_rows(Arr.SONARR) == {}) is finished
        assert (
            any("Finished the leftover post-import cleanup" in m for m in diagnostic_messages(recording, Severity.INFO))
            is finished
        )
        # Run 1 counted the import, so run 2 counts nothing for it either way.
        stats = run2_state.stats
        assert (run2_state.pending_import_states, stats.downloading, stats.downloaded) == ({}, 0, 0)

    def test_pending_imports_awaiting_cleanup_are_invisible_to_every_observe_pass(self) -> None:
        strategy = _RecordingStrategy()
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=True)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert recording.of_type(TitleHeader) == []
        assert mgr.run_deferred(view=RecordingWaitView()) is None
        assert mgr.run_blocking(view=RecordingWaitView()) is None
        assert strategy.import_calls == []
        assert _stored(store) == {"h": True}

    def test_the_cleanup_finish_leaves_a_pending_import_that_owes_no_cleanup_to_the_wait(self) -> None:
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=True)])
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(_COMPLETE),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        mgr.finish_cleanups()

        assert strategy.import_calls == []
        assert _stored(store) == {"h": False}


_OWED_CLEANUP = pending_import(infohash="h", series_id=7, awaiting_cleanup=True)
"""Pending import "h" of series 7, imported on an earlier run with its cleanup still owed."""


class TestCleanupRetryVerification:
    """The cleanup finish re-checks through the standard import check before any effect fires."""

    def test_unobserved_qbittorrent_stays_awaiting_cleanup_without_effects(self) -> None:
        strategy = _RecordingStrategy()
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [qbittorrentapi.APIConnectionError("down")]}),
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
        )

        mgr.finish_cleanups()

        assert strategy.import_calls == []
        assert strategy.removal_calls == []
        assert _stored(store) == {"h": True}
        assert state.pending_import_states == {}

    def test_done_files_run_the_effects_at_the_final_attempt(self) -> None:
        strategy = _RecordingStrategy(attempts=[attempt_result(all_done=True)])
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[_OWED_CLEANUP],
            post_import_category="pearlarr-done",
        )

        mgr.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}

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
        strategy = _RecordingStrategy(attempts=[waiting])
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit(_COMPLETE),
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
        )

        mgr.finish_cleanups()

        assert strategy.removal_calls == []
        assert _stored(store) == {"h": True}
        assert state.pending_import_states == {}

    def test_files_gone_and_reimport_issued_demotes_to_a_carryover(self) -> None:
        # The files vanished and a fresh re-import command just went in (the
        # ISSUED reason): the pending import demotes to an ordinary carryover, no effect
        # fires, and this same run's wait pass takes it up.
        recording = install_recording_hub()
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED)],
        )
        qbit = CategoryQbit(_COMPLETE)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
            post_import_category="pearlarr-done",
        )

        mgr.finish_cleanups()

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == []
        assert _stored(store) == {"h": False}
        # The blocking wait tallies before its pass, so the summary counts the row downloaded.
        assert state.pending_import_states == {"h": PendingImportState.DOWNLOADED}
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "no longer has every file imported" in warn

        result = mgr.run_deferred(view=RecordingWaitView())

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.IMPORT_IN_PROGRESS, True)]


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
        state=ArrRunState(arr=Arr.RADARR),
    )


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
            strategy=_RecordingStrategy(progress=[ImportProgress(1, 1, determinate=True)]),
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

    @pytest.mark.parametrize(
        ("awaiting_cleanup", "radarr_awaiting_cleanup"),
        [
            pytest.param(False, False, id="first-import"),
            pytest.param(True, False, id="cleanup-finish"),
            # Radarr's import went through but its cleanup is still to finish, and that cleanup makes the move.
            pytest.param(False, True, id="radarr-awaiting-its-own-cleanup"),
        ],
    )
    def test_radarr_on_the_torrent_holds_the_move_and_never_the_queue_removal(
        self, awaiting_cleanup: bool, radarr_awaiting_cleanup: bool
    ) -> None:
        strategy = _RecordingStrategy(
            attempts=[attempt_result(all_done=True)], progress=[ImportProgress(1, 1, determinate=True)]
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
        movie = pending_import(infohash="h", series_id=0, awaiting_cleanup=radarr_awaiting_cleanup)
        store.put_pending_import_row(Arr.RADARR, "h", movie.to_json())

        mgr.report_imports_for_series(7)
        mgr.finish_cleanups()

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == ["h"]
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
