"""The completion-wait "consume" side: poll, reconcile, and the blocking monitor.

Extracted from :class:`~.seadex_arr.SeaDexArr`. ``ImportWaitManager`` owns the
wait-for-completion machinery: one-shot qBittorrent polls, the carried-over
pending-record reconciliation (the per-series inline snapshot + the deferred
reconcile + the pre-summary tally), the durable-store TTL prune, and the
interleaved end-of-run blocking monitor that drives + verifies each import. The
engine keeps ``_finalize_run`` / ``run_sync`` as the orchestrator, calling the
manager's passes in order.

Binds the run :class:`RunContext` AND the active strategy via :meth:`begin_run`
(the same objects the engine holds): the strategy's ``import_completed`` is the
only thing the engine drives off the strategy, so the manager holds it under the
narrow :class:`~.protocols.ImportCompleter` protocol.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import qbittorrentapi

from .cache import UPDATED_AT_STR_FORMAT, AbstractCacheStore
from .config import AppConfig
from .manual_import import (
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    Outcome,
    PendingImport,
    PendingState,
    TorrentProbe,
    WaitOutcome,
    classify_pending,
    sanitize_torrent_telemetry,
)
from .protocols import ImportCompleter
from .reporter import RunContext, RunReporter
from .wait_view import (
    Phase,
    TorrentView,
    WaitOutcomeRow,
    WaitResult,
    WaitSnapshot,
    WaitView,
    make_wait_view,
)


class ImportWaitManager:
    """Polls, reconciles, and runs the blocking monitor for one Arr run.

    Constructed once per run in :class:`~.seadex_arr.SeaDexArr` from the unpacked
    deps + the placeholder ctx; :meth:`begin_run` rebinds the ctx + active strategy
    each run. The five passes the engine drives are the public surface: ``run_sync``
    calls :meth:`prune_expired_pending` (run start) and
    :meth:`snapshot_pending_for_series` (per item); ``_finalize_run`` calls
    :meth:`reconcile_remaining` / :meth:`tally_carried_over_into_stats` /
    :meth:`run_monitor`. The poll/import/drop helpers stay private to the subsystem.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        cache_store: AbstractCacheStore,
        reporter: RunReporter,
        logger: logging.Logger,
        qbit: qbittorrentapi.Client | None,
        ctx: RunContext,
        strategy: ImportCompleter | None = None,
    ) -> None:
        self._config = config
        self.cache_store = cache_store
        self._reporter = reporter
        self.logger = logger
        self.qbit = qbit
        # Seeded with the engine's placeholder ctx + the (initially None) strategy;
        # both rebound each run via begin_run (the same objects the engine holds, so
        # the reconcile/monitor see this run's grabs + drive this run's strategy).
        self._ctx = ctx
        self._active_strategy = strategy

    def begin_run(self, ctx: RunContext, strategy: ImportCompleter | None) -> None:
        """Bind the run context + active strategy the wait passes read/drive."""

        self._ctx = ctx
        self._active_strategy = strategy

    def _pending_store(self) -> dict[str, dict[str, Any]]:
        """A snapshot of the per-Arr ``{infohash -> record}`` pending store.

        Thin read wrapper over :meth:`CacheStore.get_pending`: returns a fresh
        plain-dict copy each call (mutating it does NOT touch the store - the two
        mutators go straight through the facade, :meth:`CacheStore.put_pending`
        in ``_register_pending_import`` and :meth:`CacheStore.drop_pending` in
        ``drop_pending``). Every other caller only iterates it read-only.

        Each value is the JSON form of a :class:`PendingImport`, rehydrated via
        :meth:`PendingImport.from_json`. :meth:`CacheStore.get_pending` already
        types its values precisely (``dict[str, dict[str, Any]]``), so they pass
        straight through with no widening or per-value narrowing.
        """

        return self.cache_store.get_pending(self._ctx.arr)

    def poll_torrent(self, infohash: str) -> TorrentProbe:
        """Poll qBittorrent once for a torrent's terminal/in-progress state.

        Returns a :class:`TorrentProbe`: a terminal :class:`WaitOutcome` (COMPLETE
        carries the ``content_path``; ERRORED/MISSING carry None), or
        ``outcome=None`` for "still waiting" - either the torrent is still
        downloading or the qBittorrent call failed transiently (auto-reauth /
        connection drop), which the wait loop treats as keep-waiting. This is the
        ONE place that reads qBittorrent AND the one place that sanitizes its junk
        telemetry (via :func:`sanitize_torrent_telemetry`), so nothing downstream
        ever sees a sentinel ETA / idle-speed / over-count.

        Args:
            infohash (str): The qBittorrent tracking key to poll.
        """

        if self.qbit is None:
            return TorrentProbe(None, None, 0.0)
        try:
            info = self.qbit.torrents_info(torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError):
            # Transient: a dropped connection or a re-auth in flight. Treat as
            # still-waiting so the caller keeps polling until the deadline.
            return TorrentProbe(None, None, 0.0)

        if not info:
            return TorrentProbe(WaitOutcome.MISSING, None, 0.0)

        t = info[0]
        telemetry = sanitize_torrent_telemetry(
            getattr(t, "progress", None),
            getattr(t, "dlspeed", None),
            getattr(t, "eta", None),
            getattr(t, "completed", None),
            getattr(t, "size", None),
        )
        # TorrentTelemetry's fields match TorrentProbe's telemetry tail one-for-one.
        if t.state_enum.is_errored:
            return TorrentProbe(WaitOutcome.ERRORED, None, *telemetry)
        if t.state_enum.is_complete or telemetry.progress >= 1.0:
            return TorrentProbe(WaitOutcome.COMPLETE, t.content_path, *telemetry)
        return TorrentProbe(None, None, *telemetry)

    def try_import_completed(
        self,
        pending: PendingImport,
        path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Drive the strategy's ``import_completed``, swallowing any error.

        The import does live Sonarr HTTP work; a malformed response (a 200 with
        a non-JSON body, a candidate missing ``path``, ...) must not abort the
        run and skip the end-of-run ``cache_store.save`` in :meth:`_finalize_run`.
        On any exception the record is left pending (returns a ``LEAVE`` probe) and
        the run continues; a real terminal failure is just retried next run / TTL'd.

        ``force`` is threaded through so the engine can tell the strategy to stop
        deferring to Sonarr on a clean ``importPending`` (the snapshot/reconcile
        passes and the final in-bound monitor poll); ``at_deadline`` flags the final
        attempt so a still-missing file warns loudly rather than at debug.
        """

        if self._active_strategy is None:
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        try:
            return self._active_strategy.import_completed(
                pending,
                path,
                force=force,
                at_deadline=at_deadline,
            )
        except Exception:
            self.logger.error(
                f"Manual import for pending {pending.infohash} raised; leaving it pending for a later run",
                exc_info=True,
            )
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only files-landed count for the wait bar (Tier-2 poll).

        Delegates to the active strategy; never refreshes downloads, reads the
        queue, or issues a command (the strategy contract enforces that). An
        indeterminate zero when no strategy is bound.
        """

        if self._active_strategy is None:
            return ImportProgress(0, 0, determinate=False)
        return self._active_strategy.import_progress(pending)

    def _this_run_infohashes(self) -> set[str]:
        """Infohashes grabbed THIS run - excluded from the carried-over passes.

        A this-run grab is reported as ``added``; the snapshot / reconcile / tally
        skip these so a record is never double-reported as queued/importing/imported.
        """

        return {p.infohash for p in self._ctx.pending_imports}

    def _reconcile_one(
        self,
        infohash: str,
        raw: dict[str, Any],
    ) -> tuple[PendingImport, PendingState]:
        """Poll one carried-over record once and fold it to a :class:`PendingState`.

        Shared by the inline snapshot and the deferred reconcile: one non-blocking
        :meth:`poll_torrent`; on COMPLETE drive one forced (CDH-off safe),
        non-deadline import attempt (so a still-missing file never warns). The
        outcome + the probe's verified-files flag fold through
        :func:`classify_pending` into one state, stashed per infohash for the
        pre-summary tally; a terminal IMPORTED is dropped + counted, a MISSING is
        dropped. Returns the rehydrated record (for the caller's inline report) and
        its classified state.
        """

        pending = PendingImport.from_json(raw)
        poll = self.poll_torrent(infohash)
        probe = ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        if poll.outcome is WaitOutcome.COMPLETE and poll.content_path:
            probe = self.try_import_completed(
                pending,
                poll.content_path,
                force=True,
                at_deadline=False,
            )

        state = classify_pending(poll.outcome, probe.files_present)
        self._ctx.pending_states[infohash] = state
        if state is PendingState.IMPORTED:
            self.drop_pending(infohash)
            self._ctx.stats.imported += 1
        elif state is PendingState.MISSING:
            self.drop_pending(infohash)
        return pending, state

    def snapshot_pending_for_series(self, series_id: int) -> None:
        """Reconcile + report this series' CARRIED-OVER pending records inline.

        For each durable record for ``series_id`` that is NOT a this-run grab (its
        infohash is absent from ``_ctx.pending_imports`` - those are already shown
        as ``added``, so including them here would double-report), do one
        non-blocking :meth:`poll_torrent`; on COMPLETE drive one forced (CDH-off
        safe) import attempt with ``at_deadline=False`` (so a still-missing file
        never warns). The poll's outcome + the probe's verified-files flag fold
        through :func:`classify_pending` into one :class:`PendingState`, which is
        rendered inline (``log_pending_snapshot``) and stashed per infohash for the
        pre-summary tally. On IMPORTED the record is dropped and ``stats.imported``
        bumped (the inline-reconciled case); other states are left pending and
        counted by the tally.
        """

        if self._active_strategy is None:
            return

        run_grabs = self._this_run_infohashes()
        # Fresh per call and SQL-filtered to this series (so a record dropped earlier
        # this run is already absent) - replaces a full get_pending scan + Python
        # series filter once per series. The ``record ->> 'series_id'`` match only
        # returns JSON objects, so each value is already a typed record.
        for infohash, record in self.cache_store.get_pending_for_series(self._ctx.arr, series_id).items():
            # Skip this-run grabs: they're already reported as `added`, so a
            # `queued`/`importing`/`imported` row here would be a double report.
            if infohash in run_grabs:
                continue
            pending, state = self._reconcile_one(infohash, record)
            self._reporter.log_pending_snapshot(
                self._ctx,
                state,
                pending.title or infohash,
                pending.coverage,
                pending.url,
            )

    def reconcile_remaining(self) -> None:
        """Non-blocking force-poll of carried-over records NOT snapshotted this run.

        The deferred-mode pre-summary step (relocated from the old startup
        reconcile): one non-blocking poll per durable record whose infohash wasn't
        already touched by the per-series inline snapshot and isn't a this-run grab
        (those stay ``added``). COMPLETE drives one forced, non-deadline import
        attempt - the download finished a prior cycle, so a still-absent target
        means Sonarr won't import on its own (CDH off) and we step in. The ready
        ones are dropped + counted; the rest record their status for the tally.
        Quiet (no live region; deferred never blocks).
        """

        if self._active_strategy is None:
            return

        run_grabs = self._this_run_infohashes()
        for infohash, raw in list(self._pending_store().items()):
            if infohash in self._ctx.pending_states:
                continue
            if infohash in run_grabs:
                continue
            self._reconcile_one(infohash, raw)

    def tally_carried_over_into_stats(self) -> None:
        """Bump queued/importing from each carried-over record's known status.

        ``imported`` is bumped at the point a record is reconciled+dropped (in the
        snapshot / reconcile), so here we only fold the records still in the store
        into ``queued`` / ``importing``: a record touched this run uses its known
        :class:`PendingState`; an un-touched store record (e.g. another series, in
        pure blocking where no reconcile ran) defaults to ``QUEUED`` without an
        extra poll. This-run grabs are excluded throughout (they're ``added``), so
        no record is ever double-counted.
        """

        run_grabs = self._this_run_infohashes()
        for infohash in self._pending_store():
            if infohash in run_grabs:
                continue
            state = self._ctx.pending_states.get(infohash, PendingState.QUEUED)
            if state is PendingState.IMPORTING:
                self._ctx.stats.importing += 1
            elif state is PendingState.QUEUED:
                self._ctx.stats.queued += 1

    def run_monitor(
        self,
        *,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        view: WaitView | None = None,
    ) -> WaitResult | None:
        """Interleaved, copy-aware wait+import over ALL pending, after the summary.

        The blocking/hybrid end-of-run pass, run dead last (after the scoreboard is
        printed). The working set is every pending record - this run's grabs
        (``_ctx.pending_imports``) AND carried-over store records, deduped by
        infohash - so a single-series run still finishes other-series carried-over
        downloads (the user's "monitor ALL" choice). Each cycle advances every
        active torrent once (so a fast torrent isn't stuck behind a slow one) into a
        ``dict[infohash -> TorrentView]``, then pushes ONE :class:`WaitSnapshot` to
        the view (which graduates any newly-terminal torrent to scrollback).
        ``imported`` is reported ONLY when the episode files are verified present
        (``probe.files_present``), so an in-flight remote-mount copy reads
        ``importing`` until it lands. Per-torrent timeouts: ``import_wait_timeout``
        for the download, ``import_ready_timeout`` for the import (from the first
        COMPLETE). Ctrl-C breaks the loop (the ``finally`` restores the terminal and
        the caller still saves the cache); the terminal outcomes are returned as a
        :class:`WaitResult` for the run report + completion notification. The clock /
        sleep / view are injectable for tests.
        """

        if self._active_strategy is None:
            return None

        records = self._monitor_working_set()
        if not records:
            return None

        clock = now if now is not None else time.monotonic
        nap = sleep if sleep is not None else time.sleep
        own_view = view is None
        if view is None:
            view = make_wait_view(
                self.logger,
                poll_s=self._config.imports.poll_interval,
                digest_interval=self._config.imports.digest_interval,
                time_source=clock,
            )

        poll_s = self._config.imports.poll_interval

        # Fresh-per-call behavioral object: it owns the per-cycle accumulators (so
        # there's nothing to reset between calls) and the advance logic; the loop
        # here keeps only the view lifecycle, the cycle pacing, and the Ctrl-C break.
        mp = MonitorPass(
            self,
            records,
            now=clock,
            dl_timeout=self._config.imports.wait_timeout,
            import_timeout=self._config.imports.ready_timeout,
        )

        try:
            view.update(mp.snapshot(0.0))
            while mp.active:
                try:
                    for record in records:
                        if record.infohash not in mp.active:
                            continue
                        mp.advance(record)
                    view.update(mp.snapshot(clock() - mp.start))
                    if mp.active:
                        self._progress_wait(mp, records, view, clock, nap, poll_s)
                except KeyboardInterrupt:
                    self.logger.info(f"Wait interrupted; {len(mp.active)} left pending")
                    break
        finally:
            if own_view:
                view.close()

        return WaitResult(tuple(mp.results), elapsed_s=clock() - mp.start)

    def _progress_wait(
        self,
        mp: "MonitorPass",
        records: list[PendingImport],
        view: WaitView,
        clock: Callable[[], float],
        nap: Callable[[float], None],
        poll_s: int,
    ) -> None:
        """Sleep one heavy-poll interval, refreshing the "files inserted" bar between.

        Splits the inter-poll ``nap`` into ``progress_poll_interval`` slices: between
        the heavy cycles it re-reads only the cheap episode-file count (never the
        throttled refresh / queue / qBittorrent) to advance each importing row's bar
        and promote a row the instant its files all land. Falls back to one plain
        ``nap(poll_s)`` when the fast poll is disabled (<= 0) or no faster than the
        heavy poll. A ``KeyboardInterrupt`` during a slice propagates to the caller's
        break, as a plain ``nap`` would.
        """

        progress_s = self._config.imports.progress_poll_interval
        if progress_s <= 0 or progress_s >= poll_s:
            nap(poll_s)
            return
        deadline = clock() + poll_s
        while mp.active:
            remaining = deadline - clock()
            if remaining <= 0:
                return
            nap(min(progress_s, remaining))
            if not mp.active:
                return
            if mp.refresh_progress(records):
                view.update(mp.snapshot(clock() - mp.start))

    def _monitor_working_set(self) -> list[PendingImport]:
        """Dedup ``_ctx.pending_imports`` + rehydrated store records by infohash.

        This-run grabs first (so their richer in-memory record wins a collision),
        then every durable store record not already present - the union the monitor
        waits on so ALL pending (this run's + carried-over) is monitored.
        """

        records: list[PendingImport] = []
        seen: set[str] = set()
        for pending in self._ctx.pending_imports:
            if pending.infohash and pending.infohash not in seen:
                seen.add(pending.infohash)
                records.append(pending)
        for infohash, raw in self._pending_store().items():
            if infohash in seen:
                continue
            seen.add(infohash)
            records.append(PendingImport.from_json(raw))
        return records

    def prune_expired_pending(self) -> None:
        """Drop durable pending records past their TTL (or with a bad stamp).

        Runs at the start of every non-off, non-preview run - including pure
        ``blocking``, which never reconciles - so a never-completing torrent
        can't pile up in the cache forever. A record past
        ``import_pending_max_age_days`` (or with an unparseable ``added_at``) is
        dropped from the durable store.
        """

        cutoff = datetime.now() - timedelta(
            days=self._config.imports.pending_max_age_days,
        )

        for infohash, raw in list(self._pending_store().items()):
            pending = PendingImport.from_json(raw)
            try:
                added_at = datetime.strptime(pending.added_at, UPDATED_AT_STR_FORMAT)
            except (TypeError, ValueError):
                self.logger.debug(
                    f"Pending import {infohash} has an unparseable timestamp; dropping as expired",
                )
                self.drop_pending(infohash)
                continue
            if added_at < cutoff:
                self.logger.info(
                    f"Pending import {infohash} older than {self._config.imports.pending_max_age_days} days; dropping",
                )
                self.drop_pending(infohash)

    def drop_pending(self, infohash: str) -> None:
        """Remove a pending record from both the durable store and the run list."""

        self.cache_store.drop_pending(self._ctx.arr, infohash)
        self._ctx.pending_imports = [p for p in self._ctx.pending_imports if p.infohash != infohash]


class MonitorPass:
    """One blocking-monitor invocation's mutable state + per-cycle advance logic.

    Built fresh at the top of :meth:`ImportWaitManager.run_monitor` from the
    manager, the working-set records, the clock, and the two per-torrent timeouts.
    It owns the five accumulators the loop used to thread - ``views`` (the frame
    the manager snapshots), ``results`` (terminal rows), ``active`` (still-running
    infohashes), ``dl_start`` / ``import_start`` (per-phase clocks) - as fields,
    so :meth:`advance` takes only the record and there is nothing to reset between
    runs (the object IS the per-invocation scope). It calls back to the manager's
    ``poll_torrent`` / ``try_import_completed`` / ``drop_pending`` (those are
    shared with the reconcile passes, so they stay on the manager).
    """

    def __init__(
        self,
        manager: "ImportWaitManager",
        records: list[PendingImport],
        *,
        now: Callable[[], float],
        dl_timeout: int,
        import_timeout: int,
    ) -> None:
        self._mgr = manager
        self.now = now
        self.dl_timeout = dl_timeout
        self.import_timeout = import_timeout
        # Sampled once here (was ``start = clock()`` atop run_monitor); the download
        # clock for every record starts now, and the manager reads ``start`` back for
        # the snapshot / WaitResult elapsed.
        self.start = now()
        self.dl_start: dict[str, float] = {r.infohash: self.start for r in records}
        self.import_start: dict[str, float] = {}
        self.active: set[str] = {r.infohash for r in records}
        self.views: dict[str, TorrentView] = {
            r.infohash: TorrentView(
                key=r.infohash,
                label=r.title or r.infohash,
                phase=Phase.QUEUED,
            )
            for r in records
        }
        self.results: list[WaitOutcomeRow] = []

    def snapshot(self, elapsed_s: float) -> WaitSnapshot:
        """The current frame: every torrent's :class:`TorrentView`, plus elapsed."""

        return WaitSnapshot(tuple(self.views.values()), elapsed_s=elapsed_s)

    def _terminal(self, outcome: Outcome, h: str, label: str) -> None:
        """Record a terminal outcome: snapshot row + result + (maybe) drop + retire.

        Drops the durable record when (and only when) ``outcome.dropped`` - True for
        exactly IMPORTED and MISSING, so the displayed word and the store mutation
        can't diverge.
        """

        self.views[h] = TorrentView(
            key=h,
            label=label,
            phase=Phase.TERMINAL,
            outcome=outcome,
        )
        self.results.append(WaitOutcomeRow(label=label, outcome=outcome))
        if outcome.dropped:
            self._mgr.drop_pending(h)
        self.active.discard(h)

    def advance(self, record: PendingImport) -> None:
        """Advance one torrent one monitor cycle (download or drive/verify import).

        Writes this torrent's current :class:`TorrentView` into ``views`` (the frame
        the caller snapshots) and, on a terminal outcome, retires it via
        :meth:`_terminal`. ``import_start`` is stamped on the first COMPLETE.
        ``imported`` is gated on verified episode files, so a freshly-issued import
        command reads ``importing`` until the copy lands; the final in-bound attempt
        (``at_deadline``) both forces and warns.
        """

        h = record.infohash
        label = record.title or record.infohash

        poll = self._mgr.poll_torrent(h)

        if poll.outcome is None:
            if self.now() - self.dl_start[h] >= self.dl_timeout:
                self._terminal(Outcome.DOWNLOAD_TIMED_OUT, h, label)
            else:
                self.views[h] = TorrentView(
                    key=h,
                    label=label,
                    phase=Phase.DOWNLOADING,
                    fraction=poll.progress,
                    speed_bps=poll.speed_bps,
                    eta_s=poll.eta_s,
                    bytes_done=poll.bytes_done,
                    bytes_total=poll.bytes_total,
                    phase_elapsed_s=self.now() - self.dl_start[h],
                    phase_timeout_s=self.dl_timeout,
                )
            return
        if poll.outcome is WaitOutcome.MISSING:
            self._terminal(Outcome.MISSING, h, label)
            return
        if poll.outcome is WaitOutcome.ERRORED:
            self._terminal(Outcome.DOWNLOAD_ERRORED, h, label)
            return
        if not poll.content_path:
            self._terminal(Outcome.DOWNLOAD_TIMED_OUT, h, label)
            return

        # COMPLETE: drive / verify our import, gating `imported` on verified files.
        self.import_start.setdefault(h, self.now())
        at_deadline = self.now() - self.import_start[h] >= self.import_timeout
        probe = self._mgr.try_import_completed(
            record,
            poll.content_path,
            force=at_deadline,
            at_deadline=at_deadline,
        )
        if probe.files_present:
            self._terminal(Outcome.IMPORTED, h, label)
        elif at_deadline:
            self._terminal(
                Outcome.STILL_IMPORTING if probe.command_issued else Outcome.NOT_READY,
                h,
                label,
            )
        elif probe.readiness is ImportReadiness.LEAVE:
            self._terminal(Outcome.NOTHING_TO_IMPORT, h, label)
        else:
            # RETRY / copy in flight: the command was accepted but the files
            # haven't landed yet (or Sonarr is still scanning). Seed the "files
            # inserted" bar from the probe counts - determinate only when the seed
            # map is whole (target_count > 0); otherwise an indeterminate row.
            total = probe.target_count
            done = probe.imported_count
            self.views[h] = TorrentView(
                key=h,
                label=label,
                phase=Phase.IMPORTING,
                fraction=(done / total if total else 1.0),
                import_done=(done if total else None),
                import_total=(total if total else None),
                phase_elapsed_s=self.now() - self.import_start[h],
                phase_timeout_s=self.import_timeout,
                command_issued=probe.command_issued,
            )

    def refresh_progress(self, records: list[PendingImport]) -> bool:
        """Cheap Tier-2 pass: refresh each importing row's "files inserted" bar.

        For every still-active row currently in the IMPORTING phase, asks the
        strategy for a read-only files-landed count (no refresh / queue / command)
        and either PROMOTES the row to IMPORTED the instant every intended file is
        present - the same verified-files signal the heavy poll gates on, only seen
        sooner - or advances its bar when the count changed. Returns whether
        anything changed, so the caller re-pushes a snapshot only when there is
        something new. Never raises: a failed progress poll is skipped, leaving the
        row's last bar in place.
        """

        changed = False
        for record in records:
            h = record.infohash
            if h not in self.active:
                continue
            view = self.views.get(h)
            if view is None or view.phase is not Phase.IMPORTING:
                continue
            try:
                progress = self._mgr.import_progress(record)
            except Exception:
                self._mgr.logger.debug(f"import progress poll for {h} failed", exc_info=True)
                continue
            # Indeterminate (partial seed map) -> no bar, no promotion; leave the
            # row to the heavy poll's repaired done-check.
            if not progress.determinate or progress.total <= 0:
                continue
            label = record.title or record.infohash
            if progress.done >= progress.total:
                self._terminal(Outcome.IMPORTED, h, label)
                changed = True
            elif (progress.done, progress.total) != (view.import_done, view.import_total):
                self.views[h] = replace(
                    view,
                    fraction=progress.done / progress.total,
                    import_done=progress.done,
                    import_total=progress.total,
                    phase_elapsed_s=self.now() - self.import_start.get(h, self.now()),
                )
                changed = True
        return changed
