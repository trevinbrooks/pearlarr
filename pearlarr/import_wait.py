"""The pending imports' "consume" side: poll, observe, and the wait pass."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import qbittorrentapi

from .arr_categories import Watch
from .arr_run_services import ArrRunDeps
from .arr_run_state import ArrRunState
from .arr_strategy import ImportCompleter
from .clock import Clock
from .config import ImportsSettings
from .output import SPARK_SAMPLES, Phase, TorrentView, WaitKind, WaitSnapshot, hub_error, hub_note, hub_warn
from .pending_import import PendingImport, is_awaiting_cleanup
from .pending_import_store import PendingImportStore
from .wait_types import (
    NO_ATTEMPT,
    NO_HAND_IMPORT_FILES,
    NO_PROGRESS,
    PENDING_IMPORT_STATE_FOR_OUTCOME,
    AttemptKind,
    AttemptResult,
    CleanupEffect,
    DownloadProgress,
    EffectStatus,
    HandImportFile,
    ImportProgress,
    LeftForHandImport,
    Outcome,
    PauseReason,
    PendingImportState,
    TorrentPoll,
    TorrentStatus,
    classify_pending_import,
    download_progress_of,
    torrent_errored,
    torrent_finished,
)
from .wait_view import WaitOutcomeRow, WaitResult, WaitView, make_wait_view

_EFFECT_BACKOFF_S: tuple[float, float] = (1.0, 3.0)
"""Pauses between in-run cleanup-effect retries (3 attempts, 4s worst case per effect)."""


class WaitPollers:
    """The arr run's fail-open pollers: qBittorrent reads plus the strategy's import hooks."""

    strategy: ImportCompleter | None
    """The arr run's ONE strategy binding (the manager and cleanup read it here). None until an arr run binds one."""

    def __init__(self, *, qbit: qbittorrentapi.Client | None, logger: logging.Logger) -> None:
        self._qbit = qbit
        self._logger = logger
        self.strategy = None

    def bind_arr_run(self, strategy: ImportCompleter | None) -> None:
        """Bind the active strategy whose import hooks the pollers drive."""

        self.strategy = strategy

    def poll_torrent(self, infohash: str) -> TorrentPoll:
        """Poll qBittorrent once for a torrent's status.

        `outcome=None` means keep waiting: still downloading, or a transient failure marked `observed=False`.
        """

        if self._qbit is None:
            return TorrentPoll(None, None, observed=False)
        try:
            info = self._qbit.torrents_info(torrent_hashes=infohash)
        except qbittorrentapi.APIError:
            # Transient (a dropped connection or a re-auth in flight): keep waiting.
            return TorrentPoll(None, None, observed=False)

        if not info:
            return TorrentPoll(TorrentStatus.GONE, None)

        t = info[0]
        progress = download_progress_of(t)
        if torrent_errored(t):
            return TorrentPoll(TorrentStatus.ERRORED, None, progress)
        if torrent_finished(t):
            return TorrentPoll(TorrentStatus.COMPLETE, t.content_path, progress)
        return TorrentPoll(None, None, progress)

    def poll_download_progress(self, infohashes: list[str]) -> dict[str, DownloadProgress]:
        """One batched, read-only qBittorrent info read for the wait region's download progress.

        Response hashes are matched case-insensitively (qBittorrent lowercases them).
        """

        if self._qbit is None or not infohashes:
            return {}
        try:
            infos = self._qbit.torrents_info(torrent_hashes=infohashes)
        except qbittorrentapi.APIError:
            return {}
        keys = {infohash.casefold(): infohash for infohash in infohashes}
        progress: dict[str, DownloadProgress] = {}
        for t in infos:
            key = keys.get(str(getattr(t, "hash", "")).casefold())
            if key is None:
                continue
            progress[key] = download_progress_of(t)
        return progress

    def try_import_completed(
        self,
        pending: PendingImport,
        path: str,
        attempt_kind: AttemptKind,
    ) -> AttemptResult:
        """Drive the strategy's `import_completed`, swallowing any error.

        Fail-open: an exception leaves the pending import as it is (`NO_ATTEMPT`) instead of aborting the run.
        """

        if self.strategy is None:
            return NO_ATTEMPT
        try:
            return self.strategy.import_completed(pending, path, attempt_kind)
        except Exception as e:
            hub_error(f"Manual import failed for {pending.display_label} - leaving it for a later run", exc=e)
            return NO_ATTEMPT

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only count of the imported files (the progress poll), never raising."""

        if self.strategy is None:
            return NO_PROGRESS
        try:
            return self.strategy.import_progress(pending)
        except Exception:
            self._logger.debug(f"import progress poll for {pending.infohash} failed", exc_info=True)
            return NO_PROGRESS


class PostImportCleanup:
    """Owns the post-import effects (category move, queue removal) and finishing the cleanups still owed."""

    _state: ArrRunState
    """The current arr run's state. Never bound at construction: `bind_arr_run` rebinds it every arr run."""

    def __init__(self, deps: ArrRunDeps, pending_imports: PendingImportStore, pollers: WaitPollers) -> None:
        self._imports = deps.config.imports
        self._categories = deps.categories
        self._qbit = deps.qbit
        self._clock = deps.clock
        self._logger = deps.logger
        self._pending_imports = pending_imports
        # Also the strategy source: the queue removal drives the hook bound on the pollers.
        self._pollers = pollers
        self._awaiting_cleanup_keys: set[str] = set()

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the fresh arr run state, forgetting which pending imports the prior arr run left awaiting cleanup."""

        self._state = state
        self._awaiting_cleanup_keys = set()

    def clean_up_and_drop(self, pending: PendingImport) -> bool:
        """Clean up an imported pending import: run both cleanup effects, then drop it.

        Effects-first, drop-last: a FAILED effect stores the pending import as awaiting cleanup
        (`awaiting_cleanup`) instead, so the next run finishes the cleanup.
        """

        # No client resolves NO category: a configured unwatched one must not skip the queue removal then.
        category = self._categories.post_import_category() if self._qbit is not None else None
        # Both effects run regardless of the other's status, in pinned order (move, then queue removal).
        statuses = (
            (CleanupEffect.CATEGORY_MOVE, self._apply_post_import_category(pending, category)),
            (CleanupEffect.QUEUE_REMOVAL, self._remove_from_queue(pending, category)),
        )
        failed = tuple(effect for effect, status in statuses if status is EffectStatus.FAILED)
        if failed:
            self._mark_awaiting_cleanup(pending, failed)
            return False
        self._pending_imports.drop(pending.infohash)
        return True

    def finish_cleanups(self) -> None:
        """Finish the cleanup of pending imports an earlier run left awaiting cleanup.

        Runs once per arr run, at the top of the arr run's finish, with a fresh attempt for each pending import
        before any effect fires (`_retry_cleanup`). Pending imports left awaiting cleanup THIS run are skipped
        (`_awaiting_cleanup_keys`), so a fresh one is never retried in its own run.
        """

        # Filter the raw rows first: the common case has none awaiting cleanup, so nothing loads.
        awaiting_cleanup = {
            key: raw
            for key, raw in self._pending_imports.awaiting_cleanup_rows().items()
            if key not in self._awaiting_cleanup_keys
        }
        for pending in self._pending_imports.load(awaiting_cleanup).values():
            self._retry_cleanup(pending)

    def _mark_awaiting_cleanup(self, pending: PendingImport, failed: tuple[CleanupEffect, ...]) -> None:
        """Store a pending import as awaiting cleanup: its files are imported, its `failed` effects not finished.

        `finish_cleanups` retries the cleanup on later runs until it finishes.
        """

        kept = replace(pending, awaiting_cleanup=True)
        self._pending_imports.save(kept)
        self._awaiting_cleanup_keys.add(kept.infohash)
        hub_warn(
            f"Could not finish the post-import cleanup for {pending.display_label} "
            f"({' and '.join(effect.value for effect in failed)}) - "
            "keeping its record to retry next run"
        )

    def _retry_effect(self, attempt_kind: Callable[[], EffectStatus]) -> EffectStatus:
        """Drive one cleanup effect through the bounded in-run retry schedule."""

        for pause in _EFFECT_BACKOFF_S:
            status = attempt_kind()
            if status is not EffectStatus.RETRY:
                return status
            self._clock.sleep(pause)
        status = attempt_kind()
        return EffectStatus.FAILED if status is EffectStatus.RETRY else status

    def _remove_from_queue(self, pending: PendingImport, moved_to: str | None) -> EffectStatus:
        """Remove the arr's leftover queue record once `pending`'s torrent is fully imported.

        Import-only: dropping a gone torrent removes nothing. Skipped outright whenever the post-import move is
        to an unwatched `moved_to`, whatever came of this run's move: the queue record then clears on its own,
        without the permanent "download ignored" history marker an explicit delete writes.
        """

        strategy = self._pollers.strategy
        if not self._imports.remove_from_queue or strategy is None:
            return EffectStatus.SKIPPED
        if moved_to and self._categories.is_unwatched(moved_to):
            self._logger.debug(
                f"{pending.display_label}: the category move clears the queue entry on its own - "
                "skipping the explicit removal",
            )
            return EffectStatus.SKIPPED
        # Each retry re-runs the whole removal: the fresh queue read self-corrects a queue record cleared meanwhile.
        return self._retry_effect(lambda: strategy.remove_from_queue(pending))

    def _apply_post_import_category(self, pending: PendingImport, category: str | None) -> EffectStatus:
        """Move an imported torrent to `category` (the cleanup's one resolve).

        Gated on the other arr holding no pending import on the hash (a torrent both arrs grabbed moves once
        the last of them imports). Creates the category (qBittorrent 409s an unknown one). A torrent in
        a category the arr can't see stays out of the ones it watches (`_move_blocked_by`).
        """

        qbit = self._qbit
        if qbit is None or category is None:
            return EffectStatus.SKIPPED
        if self._pending_imports.other_arr_holds(pending.infohash):
            self._logger.debug(
                f"{pending.display_label}: the other arr's record is still pending on this torrent - "
                "deferring the category move",
            )
            return EffectStatus.SKIPPED
        return self._retry_effect(lambda: self._attempt_move(pending, category, qbit))

    def _attempt_move(self, pending: PendingImport, category: str, qbit: qbittorrentapi.Client) -> EffectStatus:
        """One category-move attempt (create-on-409), DEBUG-logging its own failure detail."""

        infohash = pending.infohash
        try:
            if (blocked := self._move_blocked_by(pending, category)) is not None:
                return blocked
            try:
                qbit.torrents_set_category(category=category, torrent_hashes=infohash)
            except qbittorrentapi.Conflict409Error:
                qbit.torrents_create_category(name=category)
                qbit.torrents_set_category(category=category, torrent_hashes=infohash)
        except qbittorrentapi.APIError as e:
            # Give-up stays DEBUG too: the single user-visible warning is the awaiting-cleanup one.
            self._logger.debug(f"category move to {category!r} failed ({str(e) or type(e).__name__})")
            return EffectStatus.RETRY
        return EffectStatus.DONE

    def _move_blocked_by(self, pending: PendingImport, category: str) -> EffectStatus | None:
        """The status that ends a move attempt before the move, or None when the torrent may go to `category`.

        A move from a category the arr can't see into one it watches is skipped: the arr has no queue record of
        that download, and would import it again by its own parse of the names. While that can't be
        ruled out, the attempt is one to retry. A failed qBittorrent read raises `qbittorrentapi.APIError`.
        """

        target = self._categories.category_watch(category)
        if target is Watch.UNWATCHED:
            return None
        current = self._categories.torrent_category(pending.infohash)
        if current is None or current == category:
            # Gone from qBittorrent, or already there: the move carries nothing anywhere.
            return None
        seen = self._categories.category_watch(current)
        if seen is Watch.WATCHED:
            return None
        arr = self._state.arr.capitalize()
        if target is Watch.WATCHED and seen is Watch.UNWATCHED:
            self._logger.debug(
                f"{pending.display_label}: its torrent is in {current!r}, a category {arr} doesn't watch, "
                f"and {category!r} is one it does - leaving the torrent where it is"
            )
            return EffectStatus.SKIPPED
        self._logger.debug(
            f"{pending.display_label}: could not tell which categories {arr} watches - "
            f"not moving its torrent from {current!r} to {category!r} yet"
        )
        return EffectStatus.RETRY

    def _retry_cleanup(self, pending: PendingImport) -> None:
        """Confirm one pending import awaiting cleanup is still imported, then run its owed cleanup effects.

        Its evidence is a prior run's: no effect fires without a fresh files check, except for a gone
        torrent (below). A third reduction of a `TorrentPoll` after `_observe_one` and `_advance`: a fourth
        branch means extracting a shared helper first.
        """

        poll = self._pollers.poll_torrent(pending.infohash)
        if poll.outcome is TorrentStatus.GONE:
            # A gone torrent leaves no content path to confirm with, the import was done once,
            # and keeping it awaiting cleanup would wedge the pending import for good while the leftover
            # queue record invites Sonarr's failed-download handling. The move is vacuous and the
            # queue removal clears the leftover record: the one effects-without-fresh-check case.
            self._finish_cleanup(pending)
            return
        path = poll.ready_path
        if path is None:
            # Unobserved, errored, or still downloading: no evidence, no effects. The row stays
            # awaiting cleanup and out of `pending_import_states`, so nothing counts it (its import already did).
            return
        # FINAL, not ORDINARY: finishing the cleanup is one attempt per run with no clock to escalate it,
        # and a partial file map is only done past a clean importPending through an auto import.
        result = self._pollers.try_import_completed(pending, path, AttemptKind.FINAL)
        pending = self._pending_imports.save_history_placements(pending, result)
        if result.all_done:
            self._finish_cleanup(pending)
        elif result.pause_reason is PauseReason.ISSUED:
            # A fresh re-import just started, so the "import done" premise is dead: demote to
            # an ordinary carryover and let this arr run's wait pass track and recount it. Only a
            # fresh issue demotes. A blip or pearlarr's command still in flight keeps it awaiting cleanup.
            self._pending_imports.save(replace(pending, awaiting_cleanup=False))
            # The blocking wait mode tallies before the wait pass: store DOWNLOADED so the summary does
            # not default the never-snapshotted key to DOWNLOADING.
            self._state.pending_import_states[pending.infohash] = PendingImportState.DOWNLOADED
            hub_warn(f"Imported files for {pending.display_label} are no longer present - re-importing")

    def _finish_cleanup(self, pending: PendingImport) -> None:
        """Run the owed effects of a pending import awaiting cleanup through `clean_up_and_drop`, noting the finish."""

        if self.clean_up_and_drop(pending):
            hub_note(f"Completed the deferred post-import cleanup for {pending.display_label}")


class ImportWaitManager:
    """Polls, observes, and runs the wait pass for one arr run."""

    imports: ImportsSettings
    """The imports config submodel: the wait pass's timeouts and poll intervals."""
    clock: Clock
    """The wait passes' clock."""
    pollers: WaitPollers
    """The arr run's fail-open pollers and the one strategy binding, shared with the cleanup (to finish cleanups)."""
    _state: ArrRunState
    """The current arr run's state. `__init__` sets it through the first `bind_arr_run`, which rebinds every arr run."""

    def __init__(self, *, deps: ArrRunDeps, state: ArrRunState, pending_imports: PendingImportStore) -> None:
        self.imports = deps.config.imports
        self.clock = deps.clock
        self.logger = deps.logger
        self._reporter = deps.reporter
        # The pending import store, shared with the grab pipeline and bound by `ArrRunServices` each arr run: one run
        # list, one binding.
        self._pending_imports = pending_imports
        self.pollers = WaitPollers(qbit=deps.qbit, logger=deps.logger)
        self._cleanup = PostImportCleanup(deps, self._pending_imports, self.pollers)
        self.bind_arr_run(state, None)

    def bind_arr_run(self, state: ArrRunState, strategy: ImportCompleter | None) -> None:
        """Bind the fresh arr run state to the manager and its sub-objects, the strategy to the pollers alone.

        Rebound EVERY arr run (`start_arr_run` makes a fresh state), so none of them can
        operate on a dead state after the first arr run. The pending import store is `ArrRunServices`' to bind.
        """

        self._state = state
        self.pollers.bind_arr_run(strategy)
        self._cleanup.bind_arr_run(state)

    def fresh_grab_keys(self) -> set[str]:
        """Keys of the pending imports written THIS run, tallied as `added` and never carried over."""

        return self._pending_imports.fresh_keys()

    def _reported_in_title_blocks(self) -> set[str]:
        """Keys a title's block already reported this run, which `report_imports_for_series` skips.

        A pending import spanning several series is observed once, under whichever of its series scans first.
        """

        return self.fresh_grab_keys() | self._state.reacquired_keys | set(self._state.pending_import_states)

    def note_pending_import_state(self, key: str, outcome: Outcome) -> None:
        """Note a wait pass's outcome for a carried-over pending import it didn't drop in `pending_import_states`."""

        self._state.pending_import_states[key] = PENDING_IMPORT_STATE_FOR_OUTCOME[outcome]

    def save_history_placements(self, pending: PendingImport, result: AttemptResult) -> PendingImport:
        """Persist the files a poll's Sonarr history placed onto the pending import and return the updated copy."""

        return self._pending_imports.save_history_placements(pending, result)

    def apply_outcome(self, pending: PendingImport, outcome: Outcome, *, carried_over: bool) -> None:
        """One outcome's store effects: clean up on IMPORTED, drop on MISSING, else note the state.

        A MISSING drop also uncaches its titles, so the next run checks them again.

        Callers set their finished frame / result row only after this returns: a raising
        cleanup or drop must leave no phantom row for _finish_arr_run to count.
        """

        if outcome is Outcome.IMPORTED:
            # A failed cleanup leaves the pending import awaiting cleanup: the result row still counts the import,
            # and `awaiting_cleanup` keeps every later counter off the key.
            self._cleanup.clean_up_and_drop(pending)
        elif outcome is Outcome.MISSING:
            self._pending_imports.drop_unimported(pending)
        elif carried_over:
            # Still stored: note the outcome so the deferred wait pass's tally buckets
            # it truthfully (the blocking wait pass runs post-tally, where the note is never read).
            self.note_pending_import_state(pending.infohash, outcome)

    def _observe_one(self, pending: PendingImport) -> PendingImportState:
        """Observe one carried-over pending import (never an import) and reduce it to a `PendingImportState`."""

        poll = self.pollers.poll_torrent(pending.infohash)
        all_done = poll.outcome is TorrentStatus.COMPLETE and self.pollers.import_progress(pending).all_done
        state = classify_pending_import(poll.outcome, all_done)
        # Store effects precede the state write: a raising cleanup/drop must not
        # leave a state for _finish_arr_run to count off an unwritten store.
        if state is PendingImportState.IMPORTED:
            # _finish_arr_run counts the IMPORTED state into stats. No local bump.
            # A pending import left awaiting cleanup still counts here: the import happened this run.
            self._cleanup.clean_up_and_drop(pending)
        elif state is PendingImportState.GONE:
            self._pending_imports.drop_unimported(pending)
            hub_warn(
                f"Pending import {pending.display_label} is gone from qBittorrent - dropping its record "
                "(title checked again next run)"
            )
        elif state is PendingImportState.ERRORED:
            self.logger.debug(f"Pending import {pending.display_label} errored in qBittorrent - left for a later run")
        self._state.pending_import_states[pending.infohash] = state
        return state

    def report_imports_for_series(self, series_id: int) -> None:
        """Report this series' carried-over pending imports inline, cleaning up any newly imported one."""

        if self.pollers.strategy is None:
            return

        reported = self._reported_in_title_blocks()
        rows = {
            key: raw
            for key, raw in self._pending_imports.rows_for_series(series_id).items()
            if key not in reported and not is_awaiting_cleanup(raw)
        }
        for pending in self._pending_imports.load(rows).values():
            state = self._observe_one(pending)
            self._reporter.log_pending_import(state, pending, series_id)

    def tally_carried_over_into_stats(self) -> None:
        """Count each carried-over pending import not yet done into `downloading` / `downloaded`.

        Reacquired pending imports count here too. Only a this-run grab is skipped (it stays `added`).
        The raw `active_rows` view leaves out rows awaiting cleanup by the stored flag, so the
        no-phantom-count invariant holds locally instead of resting on distant state writes.
        """

        # A raw read on purpose: this loop needs only the keys and `pending_import_states`,
        # so loading every pending import would build a full map only to discard it.
        for key in self._pending_imports.active_rows():
            state = self._state.pending_import_states.get(key, PendingImportState.DOWNLOADING)
            if state is PendingImportState.DOWNLOADED:
                self._state.stats.downloaded += 1
            elif state is PendingImportState.DOWNLOADING:
                self._state.stats.downloading += 1

    def finish_cleanups(self) -> None:
        """Finish the cleanups earlier runs left owed, at the arr run's finish."""

        self._cleanup.finish_cleanups()

    def run_blocking(self, *, view: WaitView | None = None) -> WaitResult | None:
        """The blocking (or hybrid) wait pass: waiting and importing interleaved over every pending import."""

        if self.pollers.strategy is None:
            return None

        pending_imports = self._imports_to_wait_on()
        if not pending_imports:
            return None

        wait_pass = WaitPass(self, pending_imports, kind=WaitKind.MONITOR)

        def body(view: WaitView) -> None:
            while wait_pass.active_count:
                wait_pass.run_poll()
                view.update(wait_pass.snapshot())
                if wait_pass.active_count:
                    self._sleep_between_polls(wait_pass, view)

        return self._run_wait_pass(wait_pass, view=view, body=body)

    def run_deferred(self, *, view: WaitView | None = None) -> WaitResult | None:
        """The deferred wait pass: one non-blocking poll over the carried-over pending imports."""

        if self.pollers.strategy is None:
            return None

        pending_imports = list(self._pending_imports.active_imports().values())
        if not pending_imports:
            return None

        wait_pass = WaitPass(self, pending_imports, kind=WaitKind.CHECK)

        def body(view: WaitView) -> None:
            wait_pass.run_single_poll()
            view.update(wait_pass.snapshot())

        return self._run_wait_pass(wait_pass, view=view, body=body)

    def _run_wait_pass(
        self,
        wait_pass: "WaitPass",
        *,
        view: WaitView | None,
        body: Callable[[WaitView], None],
    ) -> WaitResult:
        """The shared scaffold of both wait modes' passes: view lifecycle, first push, interrupts.

        Ctrl-C ends the wait pass and the run goes on. A SIGTERM stop gets the same note, then propagates.
        """

        made_view = view is None
        if view is None:
            view = make_wait_view(
                self.logger,
                poll_s=self.imports.poll_interval,
                progress_line_interval=self.imports.digest_interval,
                kind=wait_pass.kind,
            )
        try:
            view.update(wait_pass.snapshot())
            try:
                body(view)
            except KeyboardInterrupt:
                self._note_interrupted(wait_pass, view)
            except SystemExit:
                self._note_interrupted(wait_pass, view)
                raise
        finally:
            if made_view:
                view.close()
        return WaitResult(tuple(wait_pass.results), elapsed_s=wait_pass.elapsed())

    @staticmethod
    def _note_interrupted(wait_pass: "WaitPass", view: WaitView) -> None:
        """Push the rows as they stand and note how many pending imports the interrupted wait pass left."""

        view.update(wait_pass.snapshot())
        hub_note(f"{wait_pass.kind.interrupt_noun} interrupted - {wait_pass.active_count} left pending")

    def _sleep_between_polls(self, wait_pass: "WaitPass", view: WaitView) -> None:
        """Sleep one poll interval, refreshing the live rows with progress polls in between.

        The progress polls run only the cheap reads (import progress and download progress). A ready
        row can still promote to IMPORTED here, with the store effects that implies.
        """

        poll_s = self.imports.poll_interval
        progress_s = self.imports.progress_poll_interval
        if progress_s <= 0 or progress_s >= poll_s:
            self.clock.sleep(poll_s)
            return
        deadline = self.clock.monotonic() + poll_s
        while wait_pass.active_count:
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                return
            self.clock.sleep(min(progress_s, remaining))
            if not wait_pass.active_count:
                return
            # The import bar first: it can promote or finish rows.
            progressed = wait_pass.refresh_progress()
            progress_moved = view.wants_download_progress and wait_pass.refresh_download_progress()
            if progressed or progress_moved:
                view.update(wait_pass.snapshot())

    def _imports_to_wait_on(self) -> list[PendingImport]:
        """This run's fresh grabs (run-list order, first) plus the loaded carried-over pending imports.

        No dedup needed: `active_imports` excludes the fresh keys, so the in-memory copy always wins.
        """

        pending_imports = [pending for pending in self._state.grabbed_this_run.values() if pending.infohash]
        pending_imports.extend(self._pending_imports.active_imports().values())
        return pending_imports


# Cap on pause time, in import timeouts per row: Sonarr work wedged forever (a command in flight, an import that
# never ends) cannot hold the wait pass open, yet a long serial backlog gets through.
_PAUSE_CAP_MULT = 6


@dataclass(slots=True)
class _ImportClock:
    """One row's import-timeout state machine.

    `timeout` bounds a STALL, not the whole copy: the anchor re-stamps on each determinate done-count rise.
    """

    timeout: float
    anchor: float
    """The first COMPLETE, then moved only by `note_progress` and `pause`."""
    started: float
    """The import phase's start (the first COMPLETE), fixed for elapsed display."""
    _seen: int | None = None
    _poll_gap: float = 0.0
    _poll_at: float | None = None
    _paused_s: float = 0.0
    _paused_for_issued: bool = False

    def mark_poll(self, now: float) -> None:
        """Stamp a poll, remembering the interval a pause may leave uncounted."""

        self._poll_gap = now - self._poll_at if self._poll_at is not None else 0.0
        self._poll_at = now

    def at_deadline(self, now: float) -> bool:
        """Whether the stall bound has elapsed since the anchor."""

        return now - self.anchor >= self.timeout

    def can_pause(self, reason: PauseReason) -> bool:
        """Whether this pause may stop the clock: none past the cap, and a fresh POST only once per row."""

        # A POST whose files all fail inside Sonarr completes within the poll gap and is re-POSTed next poll.
        # Paused for every time, that retry loop would run to the cap. Paused for once, it retries at the ordinary
        # cadence and finishes at the ordinary deadline (a second POST for a remainder follows an import anyway).
        if reason is PauseReason.ISSUED and self._paused_for_issued:
            return False
        return self._paused_s < self.timeout * _PAUSE_CAP_MULT

    def pause(self, now: float, reason: PauseReason) -> None:
        """Pause the clock by the last poll interval."""

        if reason is PauseReason.ISSUED:
            self._paused_for_issued = True
        pause_s = min(self._poll_gap, self.timeout * _PAUSE_CAP_MULT - self._paused_s)
        self._paused_s += pause_s
        self.anchor = min(self.anchor + pause_s, now)

    def note_progress(self, done: int, total: int, now: float) -> bool:
        """Re-anchor on a rising determinate done-count, True when another file is done."""

        if total <= 0:
            return False
        last = self._seen
        self._seen = done if last is None else max(last, done)
        if last is None or done <= last:
            return False
        self.anchor = now
        return True


def _unplaced_as_hand_import_files(result: AttemptResult) -> LeftForHandImport:
    """The attempt result's unplaced files, as the download's files left for a hand import."""

    return LeftForHandImport(files=tuple(HandImportFile(name) for name in result.unplaced_files))


def _timed_out_outcome(result: AttemptResult) -> Outcome:
    """The truthful label for a row finishing with no pause to stop its clock (the deadline, or the pause cap)."""

    # `command_issued` alone covers scripted attempt results: the reconciler pairs it with a pause reason.
    if result.command_issued or result.pause_reason in (PauseReason.ISSUED, PauseReason.IMPORT):
        return Outcome.STILL_IMPORTING
    if result.pause_reason is PauseReason.BUSY:
        return Outcome.SONARR_BUSY
    return Outcome.NOT_READY


@dataclass(frozen=True, slots=True)
class _FinishDetail:
    """The detail an outcome carries: the imported files count, or the files a hand import places."""

    files: int | None = None
    left_for_hand_import: LeftForHandImport = NO_HAND_IMPORT_FILES


_NO_DETAIL = _FinishDetail()


@dataclass(slots=True)
class _WaitRow:
    """One pending import's live state within a wait pass. The row owns its `TorrentView` presentation."""

    pending: PendingImport
    """The stored pending import this row tracks, replaced after a poll whose attempt placed files from Sonarr's
    history."""
    dl_start: float
    """Download-phase clock, stamped at construction."""
    carried_over: bool
    """Whether the pending import predates this run (a fresh grab tallies as `added`)."""
    clock: _ImportClock | None = None
    """Created on the first COMPLETE poll, carrying the import phase's start."""
    view: TorrentView = field(init=False)
    """The row's current frame, snapshotted by the manager each push."""

    def __post_init__(self) -> None:
        self.view = TorrentView(
            key=self.pending.infohash, label=self.pending.display_label, phase=Phase.AWAITING_FIRST_POLL
        )

    @property
    def active(self) -> bool:
        """Still running: the finished frame is set exactly once, by `view_finished`."""

        return self.view.phase is not Phase.FINISHED

    def _import_started(self) -> float:
        """The import phase's start. Every importing frame carries its clock (`dl_start` is a dead fallback)."""

        return self.clock.started if self.clock is not None else self.dl_start

    def view_finished(self, outcome: Outcome, now: float, detail: _FinishDetail = _NO_DETAIL) -> None:
        """Set the finished frame (`phase_elapsed_s` becomes the row's final wait clock)."""

        self.view = TorrentView(
            key=self.pending.infohash,
            label=self.pending.display_label,
            phase=Phase.FINISHED,
            outcome=outcome,
            import_done=detail.files,
            import_total=detail.files,
            left_for_hand_import=detail.left_for_hand_import,
            phase_elapsed_s=now - self.dl_start,
        )

    def view_downloading(self, progress: DownloadProgress, now: float) -> None:
        """The poll's downloading frame, appending ONE speed-history sample.

        Only this method samples the history, so the sparkline stays minutes wide.
        """

        history = self.view.speed_history if self.view.phase is Phase.DOWNLOADING else ()
        self.view = TorrentView(
            key=self.pending.infohash,
            label=self.pending.display_label,
            phase=Phase.DOWNLOADING,
            fraction=progress.progress,
            speed_bps=progress.speed_bps,
            eta_s=progress.eta_s,
            bytes_done=progress.bytes_done,
            bytes_total=progress.bytes_total,
            phase_elapsed_s=now - self.dl_start,
            speed_history=(*history, progress.speed_bps or 0)[-SPARK_SAMPLES:],
        )

    def view_tick(self, now: float) -> None:
        """Advance only a downloading frame's elapsed clock (an unobserved poll keeps the last real values)."""

        if self.view.phase is Phase.DOWNLOADING:
            self.view = replace(self.view, phase_elapsed_s=now - self.dl_start)

    def view_importing(self, result: AttemptResult, now: float) -> None:
        """The poll's importing frame. The bar is determinate only when the file map is whole."""

        total = result.file_map_episode_count
        done = result.imported_count
        self.view = TorrentView(
            key=self.pending.infohash,
            label=self.pending.display_label,
            phase=Phase.IMPORTING,
            fraction=(done / total if total else 1.0),
            import_done=(done if total else None),
            import_total=(total if total else None),
            phase_elapsed_s=now - self._import_started(),
            command_issued=result.command_issued,
        )

    def view_import_progress(self, progress: ImportProgress, now: float) -> None:
        """Refresh the determinate import bar (`total > 0`, no phase change)."""

        self.view = replace(
            self.view,
            fraction=progress.done / progress.total,
            import_done=progress.done,
            import_total=progress.total,
            phase_elapsed_s=now - self._import_started(),
        )

    def view_download_progress(self, progress: DownloadProgress, now: float) -> bool:
        """Apply one download progress read, returning whether the frame moved. Never a history sample."""

        view = self.view
        current = DownloadProgress(view.fraction, view.speed_bps, view.eta_s, view.bytes_done, view.bytes_total)
        if progress == current:
            return False
        self.view = replace(
            view,
            fraction=progress.progress,
            speed_bps=progress.speed_bps,
            eta_s=progress.eta_s,
            bytes_done=progress.bytes_done,
            bytes_total=progress.bytes_total,
            phase_elapsed_s=now - self.dl_start,
        )
        return True


class WaitPass:
    """One wait pass's mutable state and per-poll advance logic."""

    kind: WaitKind
    """Which wait mode this pass runs. Only the blocking wait applies the download timeout."""
    rows: dict[str, _WaitRow]
    """Each pending import's live `_WaitRow`, keyed by hash in working-set order."""
    results: list[WaitOutcomeRow]
    """One per pending import that reached its outcome."""

    def __init__(self, manager: ImportWaitManager, pending_imports: list[PendingImport], *, kind: WaitKind) -> None:
        self._mgr = manager
        self.kind = kind
        self._clock = manager.clock
        self._dl_timeout = manager.imports.wait_timeout
        self._import_timeout = manager.imports.ready_timeout
        self._start = self._clock.monotonic()
        # Stamped at construction: an imported fresh grab leaves `grabbed_this_run` mid-pass,
        # so a later membership check would misread it as carried-over.
        fresh = manager.fresh_grab_keys()
        self.rows = {
            r.infohash: _WaitRow(pending=r, dl_start=self._start, carried_over=r.infohash not in fresh)
            for r in pending_imports
        }
        self.results = []

    @property
    def active_count(self) -> int:
        """How many rows are still running (not yet finished)."""

        return sum(1 for row in self.rows.values() if row.active)

    def run_poll(self) -> None:
        """Run one poll: advance every active row."""

        for row in self.rows.values():
            if row.active:
                self._advance(row)

    def run_single_poll(self) -> None:
        """The deferred wait pass whole: one poll, one progress poll, then finish the rest."""

        self.run_poll()
        self.refresh_progress()
        self._finish_active_rows()

    def elapsed(self) -> float:
        """Seconds since the pass started (off the injected clock)."""

        return self._clock.monotonic() - self._start

    def snapshot(self) -> WaitSnapshot:
        """The current frame: every torrent's `TorrentView`, plus elapsed."""

        return WaitSnapshot(tuple(row.view for row in self.rows.values()), elapsed_s=self.elapsed())

    def _finish_row(self, outcome: Outcome, row: _WaitRow, detail: _FinishDetail = _NO_DETAIL) -> None:
        """Finish a row with its outcome: the drop or cleanup it implies, the finished frame, then the result row."""

        pending = row.pending
        # Store effects precede the finished frame and the result row (apply_outcome's contract).
        self._mgr.apply_outcome(pending, outcome, carried_over=row.carried_over)
        row.view_finished(outcome, self._clock.monotonic(), detail)
        # Read back from the finished frame, so the frame stays the one source of the row's facts.
        self.results.append(
            WaitOutcomeRow(
                label=pending.display_label,
                outcome=outcome,
                carried_over=row.carried_over,
                left_for_hand_import=row.view.left_for_hand_import,
            )
        )

    def _finish_active_rows(self) -> None:
        """Finish every still-active row with a truthful outcome that leaves it for the next run."""

        for row in self.rows.values():
            if not row.active:
                continue
            phase = row.view.phase
            if phase is Phase.IMPORTING:
                outcome = Outcome.IMPORT_IN_PROGRESS if row.view.command_issued else Outcome.AWAITING_IMPORT
            elif phase is Phase.DOWNLOADING:
                outcome = Outcome.STILL_DOWNLOADING
            else:
                outcome = Outcome.NOT_CHECKED
            self._finish_row(outcome, row)

    def _advance(self, row: _WaitRow) -> None:
        """Advance one row by one poll (download, or drive the import and confirm it)."""

        pending = row.pending

        poll = self._mgr.pollers.poll_torrent(pending.infohash)

        if poll.outcome is None:
            # Blocking only: `dl_start` is this pass's construction, so a deferred pass's timeout would
            # mislabel a still-downloading pending import. `_finish_active_rows` words it truthfully there.
            if self.kind is WaitKind.MONITOR and self._clock.monotonic() - row.dl_start >= self._dl_timeout:
                self._finish_row(Outcome.DOWNLOAD_TIMED_OUT, row)
                return
            if not poll.observed:
                # Transient qBittorrent error: the zeroed poll is a placeholder, not a real read. Keep the row's
                # last real state (no 0% bar flash, no fake stall sample) and let its clock tick.
                row.view_tick(self._clock.monotonic())
                return
            row.view_downloading(poll.telemetry, self._clock.monotonic())
            return
        if poll.outcome is TorrentStatus.GONE:
            self._finish_row(Outcome.MISSING, row)
            return
        if poll.outcome is TorrentStatus.ERRORED:
            self._finish_row(Outcome.DOWNLOAD_ERRORED, row)
            return
        path = poll.ready_path
        if path is None:
            # COMPLETE but qBittorrent reported no save path: its own outcome, not a misleading "timed out".
            self._finish_row(Outcome.NO_CONTENT_PATH, row)
            return

        # COMPLETE: drive the auto import and confirm it.
        now_ts = self._clock.monotonic()
        clock = row.clock
        if clock is None:
            clock = row.clock = _ImportClock(timeout=self._import_timeout, anchor=now_ts, started=now_ts)
        clock.mark_poll(now_ts)
        at_deadline = clock.at_deadline(now_ts)
        result = self._mgr.pollers.try_import_completed(
            pending,
            path,
            AttemptKind.FINAL if at_deadline else AttemptKind.ORDINARY,
        )
        # The placements are stored before any outcome: the cleanup, the awaiting-cleanup save, the next poll, and the
        # progress poll's bar read them.
        pending = row.pending = self._mgr.save_history_placements(pending, result)
        progressed = clock.note_progress(result.imported_count, result.file_map_episode_count, now_ts)
        # Waiting on Sonarr's import work is not this pending import stalling (a poll that got a file done already
        # re-anchored harder than a pause would).
        paused = result.paused and clock.can_pause(result.pause_reason)
        if paused and not progressed:
            clock.pause(now_ts, result.pause_reason)
        if result.all_done:
            self._finish_row(Outcome.IMPORTED, row, _FinishDetail(files=result.file_map_episode_count or None))
        elif result.unplaced_files:
            # Nothing to wait for: the deadline would only repeat the answer. Left for a hand import, reported again
            # next run.
            self._finish_row(
                Outcome.UNMATCHED, row, _FinishDetail(left_for_hand_import=_unplaced_as_hand_import_files(result))
            )
        elif result.left_for_hand_import.files:
            # The placed files are in and the rest wait on a hand import: left, reported again next run.
            self._finish_row(Outcome.PARTIAL, row, _FinishDetail(left_for_hand_import=result.left_for_hand_import))
        elif at_deadline and not progressed and not paused:
            self._finish_row(_timed_out_outcome(result), row)
        elif not result.attempted:
            self._finish_row(Outcome.ATTEMPT_FAILED, row)
        else:
            # Still waiting, or the auto import's copy is in flight.
            row.view_importing(result, now_ts)

    def refresh_progress(self) -> bool:
        """The progress poll over the IMPORTING rows: refresh the import bar, promote once every file is done.

        No rescan, no queue read, no command, and no phase transition.
        """

        changed = False
        for row in self.rows.values():
            if not row.active or row.view.phase is not Phase.IMPORTING:
                continue
            # An IMPORTING row always carries its clock (stamped on the COMPLETE transition).
            clock = row.clock
            if clock is None:
                continue
            progress = self._mgr.pollers.import_progress(row.pending)
            # Indeterminate (a partial file map) means no bar and no promotion. The poll's done-check
            # finishes the row.
            if not progress.determinate or progress.total <= 0:
                continue
            clock.note_progress(progress.done, progress.total, self._clock.monotonic())
            if progress.all_done:
                self._finish_row(Outcome.IMPORTED, row, _FinishDetail(files=progress.total))
                changed = True
            elif (progress.done, progress.total) != (row.view.import_done, row.view.import_total):
                row.view_import_progress(progress, self._clock.monotonic())
                changed = True
        return changed

    def refresh_download_progress(self) -> bool:
        """Refresh each downloading row's live download progress, returning whether it moved.

        Download progress only: no outcomes, no phase transitions, and no speed-history sample.
        """

        downloading = [row for row in self.rows.values() if row.active and row.view.phase is Phase.DOWNLOADING]
        # One batch read for every downloading row (one row per torrent).
        hashes = [row.pending.infohash for row in downloading]
        by_hash = self._mgr.pollers.poll_download_progress(hashes)
        changed = False
        for row in downloading:
            progress = by_hash.get(row.pending.infohash)
            if progress is not None and row.view_download_progress(progress, self._clock.monotonic()):
                changed = True
        return changed
