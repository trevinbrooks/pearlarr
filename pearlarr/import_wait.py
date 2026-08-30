"""The completion-wait "consume" side: poll, observe, and the end-of-run import pass."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import qbittorrentapi

from .cache import parse_stamp, pending_cutoff
from .clock import Clock
from .log import count_noun
from .manual_import import (
    LEAVE_PROBE,
    NO_PROGRESS,
    PENDING_STATE_FOR_OUTCOME,
    AttemptKind,
    EffectStatus,
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    Outcome,
    PendingImport,
    PendingKey,
    PendingState,
    TorrentProbe,
    TorrentTelemetry,
    WaitOutcome,
    classify_pending,
    sanitize_torrent_telemetry,
)
from .output import SPARK_SAMPLES, Phase, TorrentView, WaitKind, WaitSnapshot, hub_error, hub_note, hub_warn
from .protocols import ImportCompleter
from .reporter import RunContext
from .run_services import RunDeps
from .wait_view import WaitOutcomeRow, WaitResult, WaitView, make_wait_view

_EFFECT_BACKOFF_S: tuple[float, float] = (1.0, 3.0)
"""Pauses between in-run cleanup-effect retries (3 attempts, 4s worst case per effect)."""


class ImportWaitManager:
    """Polls, observes, and runs the end-of-run import pass for one Arr run."""

    def __init__(self, *, deps: RunDeps, ctx: RunContext) -> None:
        self._config = deps.config
        self._categories = deps.categories
        self.cache_store = deps.cache_store
        self._reporter = deps.reporter
        self.logger = deps.logger
        self.qbit = deps.qbit
        self._clock = deps.clock
        self._ctx = ctx
        # The None placeholder before begin_run binds the run's strategy.
        self._active_strategy: ImportCompleter | None = None

    def begin_run(self, ctx: RunContext, strategy: ImportCompleter | None) -> None:
        """Bind the run context + active strategy the wait passes read/drive."""

        self._ctx = ctx
        self._active_strategy = strategy

    def _pending_records(self) -> dict[PendingKey, PendingImport]:
        """A rehydrated snapshot of the per-Arr `{PendingKey -> PendingImport}` store."""

        guard_rows = self.cache_store.get_guards(self._ctx.arr)
        return {
            key: PendingImport.from_json(raw, guards=guard_rows.get(key.al_id))
            for key, raw in self.cache_store.get_pending(self._ctx.arr).items()
        }

    def poll_torrent(self, infohash: str) -> TorrentProbe:
        """Poll qBittorrent once for a torrent's terminal state.

        `outcome=None` means keep waiting: still downloading, or a transient failure flagged `observed=False`.
        """

        if self.qbit is None:
            return TorrentProbe(None, None, observed=False)
        try:
            info = self.qbit.torrents_info(torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError):
            # Transient (a dropped connection or a re-auth in flight): keep waiting.
            return TorrentProbe(None, None, observed=False)

        if not info:
            return TorrentProbe(WaitOutcome.MISSING, None)

        t = info[0]
        telemetry = sanitize_torrent_telemetry(t)
        if t.state_enum.is_errored:
            return TorrentProbe(WaitOutcome.ERRORED, None, telemetry)
        if t.state_enum.is_complete or telemetry.progress >= 1.0:
            return TorrentProbe(WaitOutcome.COMPLETE, t.content_path, telemetry)
        return TorrentProbe(None, None, telemetry)

    def poll_telemetry(self, infohashes: list[str]) -> dict[str, TorrentTelemetry]:
        """One batched, read-only qBittorrent info read for the fast cockpit refresh.

        Response hashes are matched case-insensitively (qBittorrent lowercases them).
        """

        if self.qbit is None or not infohashes:
            return {}
        try:
            infos = self.qbit.torrents_info(torrent_hashes=infohashes)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError):
            return {}
        keys = {infohash.casefold(): infohash for infohash in infohashes}
        telemetry: dict[str, TorrentTelemetry] = {}
        for t in infos:
            key = keys.get(str(getattr(t, "hash", "")).casefold())
            if key is None:
                continue
            telemetry[key] = sanitize_torrent_telemetry(t)
        return telemetry

    def try_import_completed(
        self,
        pending: PendingImport,
        path: str,
        attempt: AttemptKind,
    ) -> ImportProbe:
        """Drive the strategy's `import_completed`, swallowing any error.

        Fail-open: an exception leaves the record pending (a `LEAVE` probe) instead of aborting the run.
        """

        if self._active_strategy is None:
            return LEAVE_PROBE
        try:
            return self._active_strategy.import_completed(pending, path, attempt)
        except Exception as e:
            hub_error(f"Manual import failed for {pending.display_label} - leaving it for a later run", exc=e)
            return LEAVE_PROBE

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only files-landed count (the Tier-2 poll), never raising."""

        if self._active_strategy is None:
            return NO_PROGRESS
        try:
            return self._active_strategy.import_progress(pending)
        except Exception:
            self.logger.debug(f"import progress poll for {pending.key.row_key} failed", exc_info=True)
            return NO_PROGRESS

    def fresh_grab_keys(self) -> set[PendingKey]:
        """Keys of the records written THIS run, tallied as `added` and never carried-over."""

        return {p.key for p in self._ctx.pending_imports}

    def _entry_reported_keys(self) -> set[PendingKey]:
        """Keys an entry block already reported this run, which the snapshot skips."""

        return self.fresh_grab_keys() | self._ctx.reacquired_keys

    def retire_imported(self, pending: PendingImport) -> bool:
        """Retire a verified-imported record: run both cleanup effects, then drop it.

        Effects-first, drop-last: a FAILED effect keeps the record (flagged
        `awaiting_cleanup`) instead, so the next run finishes the cleanup.
        """

        category = self._categories.post_import() if self.qbit is not None else None
        move = self.apply_post_import_category(pending, category)
        close = self.close_tracked_download(pending, category)
        failed = [
            noun
            for status, noun in ((move, "category move"), (close, "queue removal"))
            if status is EffectStatus.FAILED
        ]
        if failed:
            self._keep_for_cleanup(pending, " and ".join(failed))
            return False
        self.drop_pending(pending)
        return True

    def _keep_for_cleanup(self, pending: PendingImport, what: str) -> None:
        """Keep a record whose import verified but whose cleanup (`what`) has not finished.

        The round-trip preserves `added_at`, so the TTL still bounds a never-healing record.
        """

        kept = replace(pending, awaiting_cleanup=True)
        self.cache_store.put_pending(self._ctx.arr, kept.key, kept.to_json())
        self._ctx.cleanup_kept_keys.add(kept.key)
        hub_warn(
            f"Could not finish the post-import cleanup for {pending.display_label} ({what}) - "
            "keeping its record to retry next run"
        )

    def _retry_effect(self, attempt: Callable[[], EffectStatus]) -> EffectStatus:
        """Drive one cleanup effect through the bounded in-run retry schedule."""

        for pause in _EFFECT_BACKOFF_S:
            status = attempt()
            if status is not EffectStatus.RETRY:
                return status
            self._clock.sleep(pause)
        status = attempt()
        return EffectStatus.FAILED if status is EffectStatus.RETRY else status

    def _observe_one(self, pending: PendingImport) -> PendingState:
        """Observe one carried-over record (never an import) and fold it to a `PendingState`."""

        poll = self.poll_torrent(pending.infohash)
        files_present = poll.outcome is WaitOutcome.COMPLETE and self.import_progress(pending).files_present
        state = classify_pending(poll.outcome, files_present)
        # Store effects precede the state write: a raising retire/drop must not
        # leave a state for _finalize_run to count off an unwritten store.
        if state is PendingState.IMPORTED:
            # _finalize_run counts the IMPORTED entry into stats. No local bump.
            # A kept-for-cleanup record still counts here: the import happened this run.
            self.retire_imported(pending)
        elif state is PendingState.MISSING:
            self.drop_pending(pending)
            hub_warn(f"Pending import {pending.display_label} is gone from qBittorrent - dropping its record")
        elif state is PendingState.ERRORED:
            self.logger.debug(f"Pending import {pending.display_label} errored in qBittorrent - left for a later run")
        self._ctx.pending_states[pending.key] = state
        return state

    def snapshot_pending_for_series(self, series_id: int) -> None:
        """Report this series' carried-over pending records inline, read-only, in every mode."""

        if self._active_strategy is None:
            return

        reported = self._entry_reported_keys()
        guard_rows = self.cache_store.get_guards(self._ctx.arr)
        for key, raw in self.cache_store.get_pending_for_series(self._ctx.arr, series_id).items():
            if key in reported:
                continue
            pending = PendingImport.from_json(raw, guards=guard_rows.get(key.al_id))
            if pending.awaiting_cleanup:
                # The finalize heal pass owns these. The import already reported.
                continue
            state = self._observe_one(pending)
            self._reporter.log_pending_snapshot(state, pending)

    def tally_carried_over_into_stats(self) -> None:
        """Fold each still-pending carried-over record into `queued` / `downloaded`.

        Reacquired records count here too. Only a this-run grab is skipped (it stays `added`).
        Cleanup-flagged rows are skipped off the raw flag, so the no-phantom-count invariant
        holds locally instead of resting on distant state writes.
        """

        fresh = self.fresh_grab_keys()
        # Iterate the raw stored rows, not `_pending_records()`: this loop reads only the key, the
        # cleanup flag and `pending_states`, so rehydrating every record would build a full map
        # only to discard it.
        for key, raw in self.cache_store.get_pending(self._ctx.arr).items():
            if key in fresh or raw.get("awaiting_cleanup"):
                continue
            state = self._ctx.pending_states.get(key, PendingState.QUEUED)
            if state is PendingState.DOWNLOADED:
                self._ctx.stats.downloaded += 1
            elif state is PendingState.QUEUED:
                self._ctx.stats.queued += 1

    def run_monitor(self, *, view: WaitView | None = None) -> WaitResult | None:
        """The blocking/hybrid end-of-run pass: interleaved wait+import over every pending record."""

        if self._active_strategy is None:
            return None

        records = self._monitor_working_set()
        if not records:
            return None

        mp = self._monitor_pass(records, WaitKind.MONITOR)

        def body(view: WaitView) -> None:
            while mp.active_count:
                mp.run_cycle()
                view.update(mp.snapshot())
                if mp.active_count:
                    self._progress_wait(mp, view)

        return self._run_pass(mp, view=view, body=body)

    def check_once(self, *, view: WaitView | None = None) -> WaitResult | None:
        """The deferred end-of-run pass: one non-blocking check cycle over the carried-over records."""

        if self._active_strategy is None:
            return None

        records = self._carried_over_records()
        if not records:
            return None

        mp = self._monitor_pass(records, WaitKind.CHECK)

        def body(view: WaitView) -> None:
            mp.run_check()
            view.update(mp.snapshot())

        return self._run_pass(mp, view=view, body=body)

    def note_pending_state(self, key: PendingKey, outcome: Outcome) -> None:
        """Fold a pass's non-dropped outcome for a carried-over record into `pending_states`."""

        self._ctx.pending_states[key] = PENDING_STATE_FOR_OUTCOME[outcome]

    def _monitor_pass(self, records: list[PendingImport], kind: WaitKind) -> "MonitorPass":
        """A fresh `MonitorPass` of `kind` over `records` under the run's two per-torrent timeouts."""

        return MonitorPass(
            self,
            records,
            kind=kind,
            clock=self._clock,
            dl_timeout=self._config.imports.wait_timeout,
            import_timeout=self._config.imports.ready_timeout,
        )

    def _run_pass(
        self,
        mp: "MonitorPass",
        *,
        view: WaitView | None,
        body: Callable[[WaitView], None],
    ) -> WaitResult:
        """The shared scaffold of both end-of-run passes: view lifecycle, first push, Ctrl-C."""

        own_view = view is None
        if view is None:
            view = make_wait_view(
                self.logger,
                poll_s=self._config.imports.poll_interval,
                digest_interval=self._config.imports.digest_interval,
                kind=mp.kind,
            )
        try:
            view.update(mp.snapshot())
            try:
                body(view)
            except KeyboardInterrupt:
                view.update(mp.snapshot())
                hub_note(f"{mp.kind.interrupt_noun} interrupted - {mp.active_count} left pending")
        finally:
            if own_view:
                view.close()
        return WaitResult(tuple(mp.results), elapsed_s=mp.elapsed())

    def _progress_wait(self, mp: "MonitorPass", view: WaitView) -> None:
        """Sleep one heavy-poll interval, refreshing the live rows between slices.

        The slices run only the cheap fast-lane reads (progress and telemetry). A ready
        row can still promote to IMPORTED here, with the store effects that implies.
        """

        poll_s = self._config.imports.poll_interval
        progress_s = self._config.imports.progress_poll_interval
        if progress_s <= 0 or progress_s >= poll_s:
            self._clock.sleep(poll_s)
            return
        deadline = self._clock.now() + poll_s
        while mp.active_count:
            remaining = deadline - self._clock.now()
            if remaining <= 0:
                return
            self._clock.sleep(min(progress_s, remaining))
            if not mp.active_count:
                return
            # The import bar first: it can promote or retire rows.
            progressed = mp.refresh_progress()
            telemetry_moved = view.wants_telemetry and mp.refresh_telemetry()
            if progressed or telemetry_moved:
                view.update(mp.snapshot())

    def _carried_over_records(self) -> list[PendingImport]:
        """Every store record that is not a this-run fresh grab or a cleanup-only leftover."""

        fresh = self.fresh_grab_keys()
        return [
            pending
            for key, pending in self._pending_records().items()
            if key not in fresh and not pending.awaiting_cleanup
        ]

    def _monitor_working_set(self) -> list[PendingImport]:
        """This run's fresh grabs (first) plus `_carried_over_records`, deduped by `PendingKey`."""

        records: list[PendingImport] = []
        seen: set[PendingKey] = set()
        for pending in self._ctx.pending_imports:
            if pending.infohash and pending.key not in seen:
                seen.add(pending.key)
                records.append(pending)
        records.extend(self._carried_over_records())
        return records

    def prune_expired_pending(self) -> None:
        """Drop durable pending records past `imports.pending_max_age_days` (or with an unparseable stamp)."""

        cutoff = pending_cutoff(self._config.imports.pending_max_age_days)

        # Keyed raw reads: the age check needs only the stamp, so a record rehydrates
        # (guard-less) solely on its drop.
        for key, raw in self.cache_store.get_pending(self._ctx.arr).items():
            try:
                added_at = parse_stamp(raw.get("added_at", ""))
            except (TypeError, ValueError):
                self.logger.debug(
                    f"Pending import {key.infohash} has an unparseable timestamp; dropping as expired",
                )
                self.drop_pending(PendingImport.from_json(raw))
                continue
            if added_at < cutoff:
                pending = PendingImport.from_json(raw)
                # A flagged record's import succeeded. Only the cleanup is being abandoned.
                goal = "its post-import cleanup" if pending.awaiting_cleanup else "it"
                hub_note(
                    f"Pending import {pending.display_label} is older than "
                    f"{count_noun(self._config.imports.pending_max_age_days, 'day')} - giving up on {goal}",
                )
                self.drop_pending(pending)

    def drop_pending(self, pending: PendingImport) -> None:
        """Remove ONE record (`PendingKey`-scoped, never its siblings) from the store and the run list."""

        self.cache_store.drop_pending(self._ctx.arr, pending.key)
        self._ctx.pending_imports = [p for p in self._ctx.pending_imports if p.key != pending.key]

    def close_tracked_download(self, pending: PendingImport, category: str | None) -> EffectStatus:
        """Dismiss Sonarr's leftover queue entry once `pending`'s torrent is fully imported.

        Per-arr sibling gate (the category gate's is cross-arr), import-only: a TTL or MISSING drop closes nothing.
        Skipped outright whenever the post-import move to `category` untracks the
        torrent, whatever this run's move outcome: the entry then clears on its own, without the permanent
        "download ignored" history marker an explicit delete writes.
        """

        strategy = self._active_strategy
        if not self._config.imports.remove_from_queue or strategy is None:
            return EffectStatus.SKIPPED
        if category and self._categories.move_untracks(category):
            self.logger.debug(
                f"{pending.display_label}: the category move clears the queue entry on its own - "
                "skipping the explicit removal",
            )
            return EffectStatus.SKIPPED
        target = pending.infohash.casefold()
        # Keyed read: the gate needs only the stored keys, never a rehydrated record. Self is
        # excluded explicitly: the retiring record is still resident under drop-last.
        if any(
            key != pending.key and key.infohash.casefold() == target
            for key in self.cache_store.get_pending(self._ctx.arr)
        ):
            self.logger.debug(
                f"{pending.display_label}: sibling records still pending on this torrent - leaving its queue entry",
            )
            return EffectStatus.SKIPPED
        # Each retry re-runs the whole close: the fresh queue read self-corrects an entry cleared meanwhile.
        return self._retry_effect(lambda: strategy.close_tracked(pending))

    def apply_post_import_category(self, pending: PendingImport, category: str | None) -> EffectStatus:
        """Move a verified-imported torrent to `category` (the retire's one resolve).

        Gated on no OTHER record in either arr claiming the hash (the retiring record is still
        resident under drop-last). Creates the category (qBittorrent 409s an unknown one).
        """

        qbit = self.qbit
        if qbit is None:
            return EffectStatus.SKIPPED
        remaining = self.cache_store.count_pending_for_infohash(
            pending.infohash,
            excluding=(self._ctx.arr, pending.key),
        )
        if remaining:
            self.logger.debug(
                f"{pending.display_label}: {count_noun(remaining, 'sibling record')} still pending on "
                "this torrent - deferring the category move",
            )
            return EffectStatus.SKIPPED
        if not category:
            return EffectStatus.SKIPPED

        def attempt() -> EffectStatus:
            detail = self._try_move(qbit, category, pending.infohash)
            if detail is None:
                return EffectStatus.DONE
            # Give-up stays DEBUG too: the single user-visible warn is the keep warn.
            self.logger.debug(f"category move to {category!r} failed ({detail}) - retrying")
            return EffectStatus.RETRY

        return self._retry_effect(attempt)

    def _try_move(self, qbit: qbittorrentapi.Client, category: str, infohash: str) -> str | None:
        """One category-move attempt (create-on-409), returning the failure detail or None on success."""

        try:
            try:
                qbit.torrents_set_category(category=category, torrent_hashes=infohash)
            except qbittorrentapi.Conflict409Error:
                qbit.torrents_create_category(name=category)
                qbit.torrents_set_category(category=category, torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError) as e:
            return str(e) or type(e).__name__
        return None

    def _cleanup_records(self) -> list[PendingImport]:
        """Store records flagged `awaiting_cleanup` (never a fresh grab)."""

        return [pending for pending in self._pending_records().values() if pending.awaiting_cleanup]

    def retry_cleanup_records(self) -> None:
        """Heal records whose import verified in an earlier run but whose cleanup is still owed.

        Runs once per run, at the top of the finalize tail, re-verifying each record through the
        standard import check before any effect fires (`_heal_one`). Records kept THIS run are
        skipped (`cleanup_kept_keys`), so a fresh keep is never retried in its own run.
        """

        for record in self._cleanup_records():
            if record.key in self._ctx.cleanup_kept_keys:
                continue
            self._heal_one(record)

    def _heal_one(self, record: PendingImport) -> None:
        """Re-verify one flagged record's import, then run its owed cleanup effects.

        The flag's evidence is a prior run's: no effect fires without a fresh files check, except
        for a MISSING torrent (below). A third `TorrentProbe` fold after `_observe_one` and
        `_advance`: a fourth branch means extracting a shared fold helper first.
        """

        poll = self.poll_torrent(record.infohash)
        if poll.outcome is WaitOutcome.MISSING:
            # A gone torrent leaves no content path to verify with, the import verified once,
            # and holding the flag would wedge the record until the TTL while the stale queue
            # entry invites Sonarr's failed-download handling. The move is vacuous and the
            # close clears the leftover entry: the one effects-without-fresh-check case.
            self._finish_cleanup(record)
            return
        if poll.outcome is not WaitOutcome.COMPLETE or not poll.content_path:
            # Unobserved, errored, or still downloading: no evidence, no effects.
            self.note_cleanup(record.key)
            return
        # DEADLINE, not POLL: the heal is one attempt per run with no clock to escalate it,
        # and a partial seed map only verifies past a clean importPending by stepping in.
        probe = self.try_import_completed(record, poll.content_path, AttemptKind.DEADLINE)
        if probe.files_present:
            self._finish_cleanup(record)
        elif probe.command_issued and not probe.deferred:
            # A fresh re-import just started, so the "import done" premise is dead: demote to
            # an ordinary carryover and let this run's end pass track and recount it. Only a
            # fresh issue demotes. A blip or an own command still in flight holds the flag.
            demoted = replace(record, awaiting_cleanup=False)
            self.cache_store.put_pending(self._ctx.arr, demoted.key, demoted.to_json())
            # Blocking mode tallies before the monitor: fold DOWNLOADED so the summary does
            # not default the never-snapshotted key to queued.
            self._ctx.pending_states[demoted.key] = PendingState.DOWNLOADED
            hub_warn(f"Imported files for {record.display_label} are no longer present - re-importing")
        else:
            self.note_cleanup(record.key)

    def _finish_cleanup(self, record: PendingImport) -> None:
        """Run a flagged record's owed effects through `retire_imported`, noting the outcome either way."""

        if self.retire_imported(record):
            hub_note(f"Completed the deferred post-import cleanup for {record.display_label}")
        else:
            self.note_cleanup(record.key)

    def note_cleanup(self, key: PendingKey) -> None:
        """Fold a kept-for-cleanup record as CLEANUP: untallied and un-summed, so its import counts once."""

        self._ctx.pending_states[key] = PendingState.CLEANUP


# Cap on deferral credit, in ready timeouts per row: a command wedged in flight forever cannot hold the watch
# open, yet a long multi-file copy rides through.
_DEFERRAL_CREDIT_CAP_MULT = 6


@dataclass(slots=True)
class _ReadyClock:
    """One row's ready-deadline state machine.

    `timeout` bounds a STALL, not the whole copy: the anchor re-stamps on each determinate done-count rise.
    """

    timeout: float
    anchor: float
    """The first COMPLETE, then moved only by `note_progress` and `credit_deferral`."""
    started: float
    """The import phase's start (the first COMPLETE), fixed for elapsed display."""
    _seen: int | None = None
    _poll_gap: float = 0.0
    _poll_at: float | None = None
    _credited: float = 0.0

    def mark_poll(self, now: float) -> None:
        """Stamp a heavy poll, remembering the interval a deferral may credit back."""

        self._poll_gap = now - self._poll_at if self._poll_at is not None else 0.0
        self._poll_at = now

    def at_deadline(self, now: float) -> bool:
        """Whether the stall bound has elapsed since the anchor."""

        return now - self.anchor >= self.timeout

    def can_defer(self) -> bool:
        """Whether deferral credit remains (exhaustion resumes the ordinary deadline)."""

        return self._credited < self.timeout * _DEFERRAL_CREDIT_CAP_MULT

    def credit_deferral(self, now: float) -> None:
        """Pause the clock by the last poll interval."""

        credit = min(self._poll_gap, self.timeout * _DEFERRAL_CREDIT_CAP_MULT - self._credited)
        self._credited += credit
        self.anchor = min(self.anchor + credit, now)

    def note_progress(self, done: int, total: int, now: float) -> bool:
        """Re-anchor on a rising determinate done-count, True when another file landed."""

        if total <= 0:
            return False
        last = self._seen
        self._seen = done if last is None else max(last, done)
        if last is None or done <= last:
            return False
        self.anchor = now
        return True


@dataclass(slots=True)
class _MonitorRow:
    """One record's live state within a monitor pass. The row owns its `TorrentView` presentation."""

    record: PendingImport
    """The durable record this row tracks."""
    dl_start: float
    """Download-phase clock, stamped at construction."""
    carried_over: bool
    """Whether the record predates this run (a fresh grab tallies as `added`)."""
    clock: _ReadyClock | None = None
    """Created on the first COMPLETE poll, carrying the import phase's start."""
    view: TorrentView = field(init=False)
    """The row's current frame entry, snapshotted by the manager each push."""

    def __post_init__(self) -> None:
        self.view = TorrentView(key=self.record.key.row_key, label=self.record.display_label, phase=Phase.QUEUED)

    @property
    def active(self) -> bool:
        """Still running: the terminal frame is frozen exactly once, by `view_terminal`."""

        return self.view.phase is not Phase.TERMINAL

    def _import_started(self) -> float:
        """The import phase's start. Every importing frame carries its clock (`dl_start` is a dead fallback)."""

        return self.clock.started if self.clock is not None else self.dl_start

    def view_terminal(self, outcome: Outcome, now: float, *, files: int | None = None) -> None:
        """Freeze the terminal frame (`phase_elapsed_s` becomes the row's final wait clock)."""

        self.view = TorrentView(
            key=self.record.key.row_key,
            label=self.record.display_label,
            phase=Phase.TERMINAL,
            outcome=outcome,
            import_done=files,
            import_total=files,
            phase_elapsed_s=now - self.dl_start,
        )

    def view_downloading(self, telemetry: TorrentTelemetry, now: float) -> None:
        """The heavy poll's downloading frame, appending ONE speed-history sample.

        Only this method samples the history, so the sparkline window stays minutes wide.
        """

        history = self.view.speed_history if self.view.phase is Phase.DOWNLOADING else ()
        self.view = TorrentView(
            key=self.record.key.row_key,
            label=self.record.display_label,
            phase=Phase.DOWNLOADING,
            fraction=telemetry.progress,
            speed_bps=telemetry.speed_bps,
            eta_s=telemetry.eta_s,
            bytes_done=telemetry.bytes_done,
            bytes_total=telemetry.bytes_total,
            phase_elapsed_s=now - self.dl_start,
            speed_history=(*history, telemetry.speed_bps or 0)[-SPARK_SAMPLES:],
        )

    def view_tick(self, now: float) -> None:
        """Advance only a downloading frame's elapsed clock (an unobserved poll keeps the last real reading)."""

        if self.view.phase is Phase.DOWNLOADING:
            self.view = replace(self.view, phase_elapsed_s=now - self.dl_start)

    def view_importing(self, probe: ImportProbe, now: float) -> None:
        """The heavy poll's importing frame. The bar is determinate only when the seed map is whole."""

        total = probe.target_count
        done = probe.imported_count
        self.view = TorrentView(
            key=self.record.key.row_key,
            label=self.record.display_label,
            phase=Phase.IMPORTING,
            fraction=(done / total if total else 1.0),
            import_done=(done if total else None),
            import_total=(total if total else None),
            phase_elapsed_s=now - self._import_started(),
            command_issued=probe.command_issued,
        )

    def view_import_progress(self, progress: ImportProgress, now: float) -> None:
        """Refresh the determinate files-inserted bar in place (`total > 0`, no phase change)."""

        self.view = replace(
            self.view,
            fraction=progress.done / progress.total,
            import_done=progress.done,
            import_total=progress.total,
            phase_elapsed_s=now - self._import_started(),
        )

    def view_telemetry(self, telemetry: TorrentTelemetry, now: float) -> bool:
        """Apply one fast-lane reading, returning whether the frame moved. Never a history sample."""

        view = self.view
        current = TorrentTelemetry(view.fraction, view.speed_bps, view.eta_s, view.bytes_done, view.bytes_total)
        if telemetry == current:
            return False
        self.view = replace(
            view,
            fraction=telemetry.progress,
            speed_bps=telemetry.speed_bps,
            eta_s=telemetry.eta_s,
            bytes_done=telemetry.bytes_done,
            bytes_total=telemetry.bytes_total,
            phase_elapsed_s=now - self.dl_start,
        )
        return True


class MonitorPass:
    """One end-of-run pass's mutable state and per-cycle advance logic."""

    kind: WaitKind
    """Which end-of-run pass this is. Only the monitor applies the download timeout."""
    rows: dict[str, _MonitorRow]
    """Each record's live `_MonitorRow`, keyed by `PendingKey.row_key` in working-set order."""
    results: list[WaitOutcomeRow]
    """One per record that reached a terminal outcome."""

    def __init__(
        self,
        manager: "ImportWaitManager",
        records: list[PendingImport],
        *,
        kind: WaitKind,
        clock: Clock,
        dl_timeout: int,
        import_timeout: int,
    ) -> None:
        self._mgr = manager
        self.kind = kind
        self._clock = clock
        self.dl_timeout = dl_timeout
        self.import_timeout = import_timeout
        self.start = clock.now()
        # Stamped at construction: an imported fresh grab leaves `pending_imports` mid-pass,
        # so a later membership check would misread it as carried-over.
        fresh = manager.fresh_grab_keys()
        self.rows = {
            r.key.row_key: _MonitorRow(record=r, dl_start=self.start, carried_over=r.key not in fresh) for r in records
        }
        # Per-cycle heavy-poll memo: sibling records share ONE qBittorrent read per cycle.
        self._cycle_polls: dict[str, TorrentProbe] = {}
        self.results = []

    @property
    def active_count(self) -> int:
        """How many rows are still running (not yet terminal)."""

        return sum(1 for row in self.rows.values() if row.active)

    def run_cycle(self) -> None:
        """Run one heavy-poll cycle: clear the per-hash memo, then advance every active row."""

        self._cycle_polls.clear()
        for row in self.rows.values():
            if row.active:
                self._advance(row)

    def run_check(self) -> None:
        """The check pass's whole cycle: one heavy poll, the cheap bar refresh, then graduate the rest."""

        self.run_cycle()
        self.refresh_progress()
        self._retire_active()

    def _poll(self, infohash: str) -> TorrentProbe:
        """The hash's heavy poll for this cycle, read once and shared by siblings."""

        probe = self._cycle_polls.get(infohash)
        if probe is None:
            probe = self._mgr.poll_torrent(infohash)
            self._cycle_polls[infohash] = probe
        return probe

    def elapsed(self) -> float:
        """Seconds since the pass started (off the injected clock)."""

        return self._clock.now() - self.start

    def snapshot(self) -> WaitSnapshot:
        """The current frame: every torrent's `TorrentView`, plus elapsed."""

        return WaitSnapshot(tuple(row.view for row in self.rows.values()), elapsed_s=self.elapsed())

    def _terminal(self, outcome: Outcome, row: _MonitorRow, *, files: int | None = None) -> None:
        """Record a terminal outcome: the drop or retire it implies, the frozen frame, then the result row."""

        record = row.record
        # Store effects precede the frozen frame and the result row: a failed drop
        # must leave the row active and add no phantom row for _finalize_run to count.
        if outcome is Outcome.IMPORTED:
            if not self._mgr.retire_imported(record) and row.carried_over:
                # The kept record counts via this row's carried_over_imported bump.
                # CLEANUP keeps every other counter off it.
                self._mgr.note_cleanup(record.key)
        elif outcome.dropped:
            self._mgr.drop_pending(record)
        elif row.carried_over:
            # Still store-resident: fold the outcome so the check pass's tally buckets
            # it truthfully (the monitor runs post-tally, where the fold is unread).
            self._mgr.note_pending_state(record.key, outcome)
        row.view_terminal(outcome, self._clock.now(), files=files)
        self.results.append(WaitOutcomeRow(label=record.display_label, outcome=outcome, carried_over=row.carried_over))

    def _retire_active(self) -> None:
        """Graduate every still-active row with a truthful pending outcome."""

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
            self._terminal(outcome, row)

    def _advance(self, row: _MonitorRow) -> None:
        """Advance one row one monitor cycle (download, or drive and verify the import)."""

        record = row.record

        poll = self._poll(record.infohash)

        if poll.outcome is None:
            # Monitor-only: `dl_start` is this pass's construction, so a check-pass timeout would
            # mislabel a still-downloading record. `_retire_active` words it truthfully there.
            if self.kind is WaitKind.MONITOR and self._clock.now() - row.dl_start >= self.dl_timeout:
                self._terminal(Outcome.DOWNLOAD_TIMED_OUT, row)
                return
            if not poll.observed:
                # Transient qBittorrent error: the zeroed probe is a placeholder, not a reading. Keep the row's
                # last real state (no 0% bar flash, no fake stall sample) and let its clock tick.
                row.view_tick(self._clock.now())
                return
            row.view_downloading(poll.telemetry, self._clock.now())
            return
        if poll.outcome is WaitOutcome.MISSING:
            self._terminal(Outcome.MISSING, row)
            return
        if poll.outcome is WaitOutcome.ERRORED:
            self._terminal(Outcome.DOWNLOAD_ERRORED, row)
            return
        if not poll.content_path:
            # COMPLETE but qBittorrent reported no save path: its own outcome, not a misleading "timed out".
            self._terminal(Outcome.NO_CONTENT_PATH, row)
            return

        # COMPLETE: drive and verify our import.
        now_ts = self._clock.now()
        clock = row.clock
        if clock is None:
            clock = row.clock = _ReadyClock(timeout=self.import_timeout, anchor=now_ts, started=now_ts)
        clock.mark_poll(now_ts)
        at_deadline = clock.at_deadline(now_ts)
        probe = self._mgr.try_import_completed(
            record,
            poll.content_path,
            AttemptKind.DEADLINE if at_deadline else AttemptKind.POLL,
        )
        landed = clock.note_progress(probe.imported_count, probe.target_count, now_ts)
        # Waiting on our own work is not this record stalling (a landed poll already re-anchored harder
        # than a pause would).
        deferred = probe.deferred and clock.can_defer()
        if deferred and not landed:
            clock.credit_deferral(now_ts)
        if probe.files_present:
            self._terminal(Outcome.IMPORTED, row, files=probe.target_count or None)
        elif at_deadline and not landed and not deferred:
            self._terminal(
                Outcome.STILL_IMPORTING if probe.command_issued else Outcome.NOT_READY,
                row,
            )
        elif probe.readiness is ImportReadiness.LEAVE:
            self._terminal(Outcome.ATTEMPT_FAILED, row)
        else:
            # RETRY or copy in flight.
            row.view_importing(probe, now_ts)

    def refresh_progress(self) -> bool:
        """Cheap Tier-2 pass over the IMPORTING rows: refresh the "files inserted" bar, promote on verified files.

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
            progress = self._mgr.import_progress(row.record)
            # Indeterminate (a partial seed map) means no bar and no promotion. The heavy poll's repaired
            # done-check finishes the row.
            if not progress.determinate or progress.total <= 0:
                continue
            clock.note_progress(progress.done, progress.total, self._clock.now())
            if progress.files_present:
                self._terminal(Outcome.IMPORTED, row, files=progress.total)
                changed = True
            elif (progress.done, progress.total) != (row.view.import_done, row.view.import_total):
                row.view_import_progress(progress, self._clock.now())
                changed = True
        return changed

    def refresh_telemetry(self) -> bool:
        """Cheap fast-lane pass: refresh each downloading row's live telemetry, returning whether it moved.

        Telemetry only: no outcomes, no phase transitions, and no speed-history sample.
        """

        downloading = [row for row in self.rows.values() if row.active and row.view.phase is Phase.DOWNLOADING]
        # One batch read per underlying torrent (sibling rows share the reading).
        hashes = list(dict.fromkeys(row.record.infohash for row in downloading))
        by_hash = self._mgr.poll_telemetry(hashes)
        changed = False
        for row in downloading:
            telemetry = by_hash.get(row.record.infohash)
            if telemetry is not None and row.view_telemetry(telemetry, self._clock.now()):
                changed = True
        return changed
