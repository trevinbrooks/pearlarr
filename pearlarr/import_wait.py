"""The completion-wait "consume" side: poll, observe, and the end-of-run import pass."""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime

import qbittorrentapi

from .cache import UPDATED_AT_STR_FORMAT, pending_cutoff
from .log import count_noun
from .manual_import import (
    PENDING_STATE_FOR_OUTCOME,
    AttemptKind,
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


def _info_row_telemetry(row: object) -> TorrentTelemetry:
    """Sanitized telemetry off one qBittorrent info row."""

    return sanitize_torrent_telemetry(
        getattr(row, "progress", None),
        getattr(row, "dlspeed", None),
        getattr(row, "eta", None),
        getattr(row, "completed", None),
        getattr(row, "size", None),
    )


class ImportWaitManager:
    """Polls, observes, and runs the end-of-run import pass for one Arr run."""

    def __init__(
        self,
        *,
        deps: RunDeps,
        ctx: RunContext,
        strategy: ImportCompleter | None = None,
    ) -> None:
        self._config = deps.config
        self._categories = deps.categories
        self.cache_store = deps.cache_store
        self._reporter = deps.reporter
        self.logger = deps.logger
        self.qbit = deps.qbit
        self._ctx = ctx
        self._active_strategy = strategy

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
            return TorrentProbe(None, None, 0.0, observed=False)
        try:
            info = self.qbit.torrents_info(torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError):
            # Transient (a dropped connection or a re-auth in flight): keep waiting.
            return TorrentProbe(None, None, 0.0, observed=False)

        if not info:
            return TorrentProbe(WaitOutcome.MISSING, None, 0.0)

        t = info[0]
        telemetry = _info_row_telemetry(t)
        # TorrentTelemetry's fields match TorrentProbe's telemetry tail one-for-one (the splats below).
        if t.state_enum.is_errored:
            return TorrentProbe(WaitOutcome.ERRORED, None, *telemetry)
        if t.state_enum.is_complete or telemetry.progress >= 1.0:
            return TorrentProbe(WaitOutcome.COMPLETE, t.content_path, *telemetry)
        return TorrentProbe(None, None, *telemetry)

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
            telemetry[key] = _info_row_telemetry(t)
        return telemetry

    def try_import_completed(
        self,
        pending: PendingImport,
        path: str,
        attempt: AttemptKind = AttemptKind.POLL,
    ) -> ImportProbe:
        """Drive the strategy's `import_completed`, swallowing any error.

        Fail-open: an exception leaves the record pending (a `LEAVE` probe) instead of aborting the run.
        """

        if self._active_strategy is None:
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        try:
            return self._active_strategy.import_completed(pending, path, attempt)
        except Exception as e:
            hub_error(f"Manual import failed for {pending.display_label} - leaving it for a later run", exc=e)
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only files-landed count (the Tier-2 poll), never raising."""

        if self._active_strategy is None:
            return ImportProgress(0, 0, determinate=False)
        try:
            return self._active_strategy.import_progress(pending)
        except Exception:
            self.logger.debug(f"import progress poll for {pending.key.row_key} failed", exc_info=True)
            return ImportProgress(0, 0, determinate=False)

    def fresh_grab_keys(self) -> set[PendingKey]:
        """Keys of the records written THIS run, tallied as `added` and never carried-over."""

        return {p.key for p in self._ctx.pending_imports}

    def _entry_reported_keys(self) -> set[PendingKey]:
        """Keys an entry block already reported this run, which the snapshot skips."""

        return self.fresh_grab_keys() | self._ctx.reacquired_keys

    def retire_imported(self, pending: PendingImport) -> None:
        """Retire a verified-imported record: drop it, then the queue close and category move."""

        # The drop runs first so both sibling gates count only remaining records.
        self.drop_pending(pending)
        self.close_tracked_download(pending)
        self.apply_post_import_category(pending)

    def _observe_one(self, pending: PendingImport) -> PendingState:
        """Observe one carried-over record (never an import) and fold it to a `PendingState`."""

        poll = self.poll_torrent(pending.infohash)
        files_present = poll.outcome is WaitOutcome.COMPLETE and self.import_progress(pending).files_present
        state = classify_pending(poll.outcome, files_present)
        self._ctx.pending_states[pending.key] = state
        if state is PendingState.IMPORTED:
            # _finalize_run counts the IMPORTED entry into stats. No local bump.
            self.retire_imported(pending)
        elif state is PendingState.MISSING:
            self.drop_pending(pending)
            hub_warn(f"Pending import {pending.display_label} is gone from qBittorrent - dropping its record")
        elif state is PendingState.ERRORED:
            self.logger.debug(f"Pending import {pending.display_label} errored in qBittorrent - left for a later run")
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
            state = self._observe_one(pending)
            self._reporter.log_pending_snapshot(state, pending)

    def tally_carried_over_into_stats(self) -> None:
        """Fold each still-pending carried-over record into `queued` / `downloaded`.

        Reacquired records count here too. Only a this-run grab is skipped (it stays `added`).
        """

        fresh = self.fresh_grab_keys()
        # Iterate the raw stored keys, not `_pending_records()`: this loop reads only the key and
        # `pending_states`, so rehydrating every record would build a full map only to discard it.
        for key in self.cache_store.get_pending(self._ctx.arr):
            if key in fresh:
                continue
            state = self._ctx.pending_states.get(key, PendingState.QUEUED)
            if state is PendingState.DOWNLOADED:
                self._ctx.stats.downloaded += 1
            elif state is PendingState.QUEUED:
                self._ctx.stats.queued += 1

    def run_monitor(
        self,
        *,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        view: WaitView | None = None,
    ) -> WaitResult | None:
        """The blocking/hybrid end-of-run pass: interleaved wait+import over every pending record."""

        if self._active_strategy is None:
            return None

        records = self._monitor_working_set()
        if not records:
            return None

        nap = sleep if sleep is not None else time.sleep
        mp = self._monitor_pass(records, now, WaitKind.MONITOR)

        def body(view: WaitView) -> None:
            while mp.active_count:
                mp.run_cycle()
                view.update(mp.snapshot())
                if mp.active_count:
                    self._progress_wait(mp, view, nap)

        return self._run_pass(mp, view=view, body=body)

    def check_once(
        self,
        *,
        now: Callable[[], float] | None = None,
        view: WaitView | None = None,
    ) -> WaitResult | None:
        """The deferred end-of-run pass: one non-blocking check cycle over the carried-over records."""

        if self._active_strategy is None:
            return None

        records = self._carried_over_records()
        if not records:
            return None

        mp = self._monitor_pass(records, now, WaitKind.CHECK)

        def body(view: WaitView) -> None:
            mp.run_check()
            view.update(mp.snapshot())

        return self._run_pass(mp, view=view, body=body)

    def note_pending_state(self, key: PendingKey, outcome: Outcome) -> None:
        """Fold a pass's non-dropped outcome for a carried-over record into `pending_states`."""

        self._ctx.pending_states[key] = PENDING_STATE_FOR_OUTCOME[outcome]

    def _monitor_pass(
        self,
        records: list[PendingImport],
        now: Callable[[], float] | None,
        kind: WaitKind,
    ) -> "MonitorPass":
        """A fresh `MonitorPass` of `kind` over `records` under the run's two per-torrent timeouts."""

        return MonitorPass(
            self,
            records,
            kind=kind,
            now=now if now is not None else time.monotonic,
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
                word = "Wait" if mp.kind is WaitKind.MONITOR else "Check"
                hub_note(f"{word} interrupted - {mp.active_count} left pending")
        finally:
            if own_view:
                view.close()
        return WaitResult(tuple(mp.results), elapsed_s=mp.elapsed())

    def _progress_wait(
        self,
        mp: "MonitorPass",
        view: WaitView,
        nap: Callable[[float], None],
    ) -> None:
        """Sleep one heavy-poll interval, refreshing the live rows between slices.

        The slices run only the cheap fast-lane reads, never a rescan, queue, command, or phase transition.
        """

        poll_s = self._config.imports.poll_interval
        progress_s = self._config.imports.progress_poll_interval
        if progress_s <= 0 or progress_s >= poll_s:
            nap(poll_s)
            return
        deadline = mp.now() + poll_s
        while mp.active_count:
            remaining = deadline - mp.now()
            if remaining <= 0:
                return
            nap(min(progress_s, remaining))
            if not mp.active_count:
                return
            # The import bar first: it can promote or retire rows.
            progressed = mp.refresh_progress()
            telemetry_moved = view.wants_telemetry and mp.refresh_telemetry()
            if progressed or telemetry_moved:
                view.update(mp.snapshot())

    def _carried_over_records(self) -> list[PendingImport]:
        """Every store record that is not a this-run fresh grab."""

        fresh = self.fresh_grab_keys()
        return [pending for key, pending in self._pending_records().items() if key not in fresh]

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

        for pending in self._pending_records().values():
            try:
                added_at = datetime.strptime(pending.added_at, UPDATED_AT_STR_FORMAT)
            except (TypeError, ValueError):
                self.logger.debug(
                    f"Pending import {pending.infohash} has an unparseable timestamp; dropping as expired",
                )
                self.drop_pending(pending)
                continue
            if added_at < cutoff:
                hub_note(
                    f"Pending import {pending.display_label} is older than "
                    f"{count_noun(self._config.imports.pending_max_age_days, 'day')} - giving up on it",
                )
                self.drop_pending(pending)

    def drop_pending(self, pending: PendingImport) -> None:
        """Remove ONE record (`PendingKey`-scoped, never its siblings) from the store and the run list."""

        self.cache_store.drop_pending(self._ctx.arr, pending.key)
        self._ctx.pending_imports = [p for p in self._ctx.pending_imports if p.key != pending.key]

    def close_tracked_download(self, pending: PendingImport) -> None:
        """Dismiss Sonarr's leftover queue entry once `pending`'s torrent is fully imported.

        Per-arr sibling gate (the category gate's is cross-arr), import-only: a TTL or MISSING drop closes nothing.
        """

        if not self._config.imports.remove_from_queue or self._active_strategy is None:
            return
        target = pending.infohash.casefold()
        if any(p.infohash.casefold() == target for p in self._pending_records().values()):
            self.logger.debug(
                f"{pending.display_label}: sibling records still pending on this torrent - leaving its queue entry",
            )
            return
        self._active_strategy.close_tracked(pending)

    def apply_post_import_category(self, pending: PendingImport) -> None:
        """Move a verified-imported torrent to this arr's resolved post-import category.

        Gated on no record in EITHER arr claiming the hash. Creates the category (qBittorrent 409s an unknown one).
        """

        if self.qbit is None:
            return
        remaining = self.cache_store.count_pending_for_infohash(pending.infohash)
        if remaining:
            self.logger.debug(
                f"{pending.display_label}: {count_noun(remaining, 'sibling record')} still pending on "
                "this torrent - deferring the category move",
            )
            return
        category = self._categories.post_import()
        if not category:
            return
        label = pending.display_label
        infohash = pending.infohash
        try:
            try:
                self.qbit.torrents_set_category(category=category, torrent_hashes=infohash)
            except qbittorrentapi.Conflict409Error:
                self.qbit.torrents_create_category(name=category)
                self.qbit.torrents_set_category(category=category, torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError) as e:
            hub_warn(
                f"Could not move imported torrent {label} to category {category!r} ({e}) - "
                "leaving its category unchanged"
            )


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
    """One record's live state within a monitor pass."""

    record: PendingImport
    """The durable record this row tracks."""
    view: TorrentView
    """The row's current frame entry, snapshotted by the manager each push."""
    dl_start: float
    """Download-phase clock, stamped at construction."""
    carried_over: bool
    """Whether the record predates this run (a fresh grab tallies as `added`)."""
    active: bool = True
    """Still running (not yet terminal)."""
    clock: _ReadyClock | None = None
    """Created on the first COMPLETE poll, carrying the import phase's start."""


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
        now: Callable[[], float],
        dl_timeout: int,
        import_timeout: int,
    ) -> None:
        self._mgr = manager
        self.kind = kind
        self.now = now
        self.dl_timeout = dl_timeout
        self.import_timeout = import_timeout
        self.start = now()
        # Stamped at construction: an imported fresh grab leaves `pending_imports` mid-pass,
        # so a later membership check would misread it as carried-over.
        fresh = manager.fresh_grab_keys()
        self.rows = {
            r.key.row_key: _MonitorRow(
                record=r,
                view=TorrentView(key=r.key.row_key, label=r.display_label, phase=Phase.QUEUED),
                dl_start=self.start,
                carried_over=r.key not in fresh,
            )
            for r in records
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

        return self.now() - self.start

    def snapshot(self) -> WaitSnapshot:
        """The current frame: every torrent's `TorrentView`, plus elapsed."""

        return WaitSnapshot(tuple(row.view for row in self.rows.values()), elapsed_s=self.elapsed())

    def _terminal(self, outcome: Outcome, row: _MonitorRow, *, files: int | None = None) -> None:
        """Record a terminal outcome: snapshot row, the drop or retire it implies, then the result row."""

        record = row.record
        label = record.display_label
        row.view = TorrentView(
            key=record.key.row_key,
            label=label,
            phase=Phase.TERMINAL,
            outcome=outcome,
            import_done=files,
            import_total=files,
            phase_elapsed_s=self.now() - row.dl_start,
        )
        # Store effects precede the result row: a failed drop must not leave a
        # phantom row for _finalize_run's imported bump to count.
        if outcome is Outcome.IMPORTED:
            self._mgr.retire_imported(record)
        elif outcome.dropped:
            self._mgr.drop_pending(record)
        elif row.carried_over:
            # Still store-resident: fold the outcome so the run tally buckets it truthfully.
            self._mgr.note_pending_state(record.key, outcome)
        self.results.append(WaitOutcomeRow(label=label, outcome=outcome, carried_over=row.carried_over))
        row.active = False

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
        label = record.display_label

        poll = self._poll(record.infohash)

        if poll.outcome is None:
            # Monitor-only: `dl_start` is this pass's construction, so a check-pass timeout would
            # mislabel a still-downloading record. `_retire_active` words it truthfully there.
            if self.kind is WaitKind.MONITOR and self.now() - row.dl_start >= self.dl_timeout:
                self._terminal(Outcome.DOWNLOAD_TIMED_OUT, row)
                return
            prior = row.view
            if not poll.observed:
                # Transient qBittorrent error: the zeroed probe is a placeholder, not a reading. Keep the row's
                # last real state (no 0% bar flash, no fake stall sample) and let its clock tick.
                if prior.phase is Phase.DOWNLOADING:
                    row.view = replace(prior, phase_elapsed_s=self.now() - row.dl_start)
                return
            # Speed history advances once per heavy poll. The fast telemetry refresh deliberately never samples
            # it, so the sparkline window stays minutes wide.
            history = prior.speed_history if prior.phase is Phase.DOWNLOADING else ()
            row.view = TorrentView(
                key=record.key.row_key,
                label=label,
                phase=Phase.DOWNLOADING,
                fraction=poll.progress,
                speed_bps=poll.speed_bps,
                eta_s=poll.eta_s,
                bytes_done=poll.bytes_done,
                bytes_total=poll.bytes_total,
                phase_elapsed_s=self.now() - row.dl_start,
                speed_history=(*history, poll.speed_bps or 0)[-SPARK_SAMPLES:],
            )
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
        now_ts = self.now()
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
            # RETRY or copy in flight: the bar is determinate only when the seed map is whole.
            total = probe.target_count
            done = probe.imported_count
            row.view = TorrentView(
                key=record.key.row_key,
                label=label,
                phase=Phase.IMPORTING,
                fraction=(done / total if total else 1.0),
                import_done=(done if total else None),
                import_total=(total if total else None),
                phase_elapsed_s=self.now() - clock.started,
                command_issued=probe.command_issued,
            )

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
            clock.note_progress(progress.done, progress.total, self.now())
            if progress.files_present:
                self._terminal(Outcome.IMPORTED, row, files=progress.total)
                changed = True
            elif (progress.done, progress.total) != (row.view.import_done, row.view.import_total):
                row.view = replace(
                    row.view,
                    fraction=progress.done / progress.total,
                    import_done=progress.done,
                    import_total=progress.total,
                    phase_elapsed_s=self.now() - clock.started,
                )
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
            if telemetry is None:
                continue
            view = row.view
            current = TorrentTelemetry(
                view.fraction,
                view.speed_bps,
                view.eta_s,
                view.bytes_done,
                view.bytes_total,
            )
            if telemetry == current:
                continue
            row.view = replace(
                view,
                fraction=telemetry.progress,
                speed_bps=telemetry.speed_bps,
                eta_s=telemetry.eta_s,
                bytes_done=telemetry.bytes_done,
                bytes_total=telemetry.bytes_total,
                phase_elapsed_s=self.now() - row.dl_start,
            )
            changed = True
        return changed
