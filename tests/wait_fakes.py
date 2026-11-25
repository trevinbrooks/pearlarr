# pyright: strict
"""Fakes and a manager builder shared by the import wait tests.

`make_orchestration_manager` builds the wait manager bare, without its constructor, over a scripted qBittorrent
(`FakeQbit`), a recording strategy and a `FakeClock`, so no wait sleeps for real.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import pytest

from pearlarr.arr_run_state import ArrRunState
from pearlarr.arr_strategy import ImportCompleter
from pearlarr.config import Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.output import Phase, TorrentView, WaitSnapshot
from pearlarr.pending_import import PendingImport
from pearlarr.wait_types import NO_ATTEMPT, NO_PROGRESS, AttemptKind, AttemptResult, EffectStatus, ImportProgress
from pearlarr.wait_view import WaitView

from .builders import FakeCacheStore, make_import_wait_manager, pending_import
from .fakes import FakeClock, FakeStrategy


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


TORRENT_H_COMPLETE: dict[str, list[QbitStep]] = {"h": [FakeTorrent(is_complete=True, content_path="/d")]}
"""Torrent "h" complete in qBittorrent, as a `FakeQbit` script."""


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


class RecordingStrategy(FakeStrategy):
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


def stored_awaiting_cleanup(store: FakeCacheStore, arr: Arr = Arr.SONARR) -> dict[str, bool]:
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


def run_blocking_wait(
    strategy: RecordingStrategy,
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
