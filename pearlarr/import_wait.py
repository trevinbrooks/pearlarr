"""The completion-wait "consume" side: poll, reconcile, and the blocking monitor.

`ImportWaitManager` owns the wait-for-completion machinery: one-shot
qBittorrent polls, the carried-over pending-record reconciliation (the
per-series inline snapshot + the deferred reconcile + the pre-summary tally),
the durable-store TTL prune, and the interleaved end-of-run blocking monitor
that drives + verifies each import. The engine calls the manager's passes in
order.

Binds the run `RunContext` AND the active strategy via `begin_run`
(the same objects the engine holds): the strategy's `import_completed` is the
only thing the engine drives off the strategy, so the manager holds it under the
narrow `ImportCompleter` ABC.
"""

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import qbittorrentapi

from .cache import UPDATED_AT_STR_FORMAT
from .log import count_noun
from .manual_import import (
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    Outcome,
    OutcomeCategory,
    PendingImport,
    PendingKey,
    PendingState,
    TorrentProbe,
    TorrentTelemetry,
    WaitOutcome,
    classify_pending,
    sanitize_torrent_telemetry,
)
from .output import SPARK_SAMPLES, Phase, TorrentView, WaitSnapshot, hub_error, hub_note, hub_warn
from .protocols import ImportCompleter
from .reporter import RunContext
from .run_services import RunDeps
from .wait_view import WaitOutcomeRow, WaitResult, WaitView, make_wait_view


def _info_row_telemetry(row: object) -> TorrentTelemetry:
    """Sanitized telemetry off one qBittorrent info row.

    The one place the getattr-read field list lives, shared by the heavy
    `ImportWaitManager.poll_torrent` and the batched
    `ImportWaitManager.poll_telemetry`.
    """

    return sanitize_torrent_telemetry(
        getattr(row, "progress", None),
        getattr(row, "dlspeed", None),
        getattr(row, "eta", None),
        getattr(row, "completed", None),
        getattr(row, "size", None),
    )


class ImportWaitManager:
    """Polls, reconciles, and runs the blocking monitor for one Arr run.

    Constructed once per run in `RunLoop` from the unpacked
    deps + the placeholder ctx. `begin_run` rebinds the ctx + active strategy
    each run. The five passes the engine drives are the public surface: `run_sync`
    calls `prune_expired_pending` (run start) and
    `snapshot_pending_for_series` (per item). `_finalize_run` calls
    `reconcile_remaining` / `tally_carried_over_into_stats` /
    `run_monitor`. The poll/import/drop helpers stay private to the subsystem.
    """

    def __init__(
        self,
        *,
        deps: RunDeps,
        ctx: RunContext,
        strategy: ImportCompleter | None = None,
    ) -> None:
        self._config = deps.config
        self.cache_store = deps.cache_store
        self._reporter = deps.reporter
        self.logger = deps.logger
        self.qbit = deps.qbit
        # Seeded with the engine's placeholder ctx + the (initially None) strategy.
        # Both rebound each run via begin_run (the same objects the engine holds, so
        # the reconcile/monitor see this run's grabs + drive this run's strategy).
        self._ctx = ctx
        self._active_strategy = strategy

    def begin_run(self, ctx: RunContext, strategy: ImportCompleter | None) -> None:
        """Bind the run context + active strategy the wait passes read/drive."""

        self._ctx = ctx
        self._active_strategy = strategy

    def _pending_records(self) -> dict[PendingKey, PendingImport]:
        """A rehydrated snapshot of the per-Arr `{PendingKey -> PendingImport}` store.

        Thin read wrapper over `CacheStore.get_pending` (the raw-JSON SQLite
        boundary): a fresh copy each call, with every value rehydrated ONCE via
        `PendingImport.from_json` so the wait passes only ever handle typed
        records. Keyed per record, so siblings sharing one torrent all surface.
        Read-only - the two mutators go straight through the facade
        (`CacheStore.put_pending` in `_register_pending_import` and
        `CacheStore.drop_pending` in `drop_pending`).
        """

        return {key: PendingImport.from_json(raw) for key, raw in self.cache_store.get_pending(self._ctx.arr).items()}

    def poll_torrent(self, infohash: str) -> TorrentProbe:
        """Poll qBittorrent once for a torrent's terminal/in-progress state.

        Returns a `TorrentProbe`: a terminal `WaitOutcome` (COMPLETE
        carries the `content_path` - ERRORED/MISSING carry None), or
        `outcome=None` for "still waiting" - either the torrent is still
        downloading or the qBittorrent call failed transiently (auto-reauth /
        connection drop), which the wait loop treats as keep-waiting. This is the
        ONE place that reads qBittorrent AND the one place that sanitizes its junk
        telemetry (via `sanitize_torrent_telemetry`), so nothing downstream
        ever sees a sentinel ETA / idle-speed / over-count.
        """

        if self.qbit is None:
            return TorrentProbe(None, None, 0.0, observed=False)
        try:
            info = self.qbit.torrents_info(torrent_hashes=infohash)
        except (qbittorrentapi.APIError, qbittorrentapi.APIConnectionError):
            # Transient: a dropped connection or a re-auth in flight. Treat as
            # still-waiting so the caller keeps polling until the deadline. The
            # un-observed flag keeps the row's last real telemetry on screen.
            return TorrentProbe(None, None, 0.0, observed=False)

        if not info:
            return TorrentProbe(WaitOutcome.MISSING, None, 0.0)

        t = info[0]
        telemetry = _info_row_telemetry(t)
        # TorrentTelemetry's fields match TorrentProbe's telemetry tail one-for-one.
        if t.state_enum.is_errored:
            return TorrentProbe(WaitOutcome.ERRORED, None, *telemetry)
        if t.state_enum.is_complete or telemetry.progress >= 1.0:
            return TorrentProbe(WaitOutcome.COMPLETE, t.content_path, *telemetry)
        return TorrentProbe(None, None, *telemetry)

    def poll_telemetry(self, infohashes: list[str]) -> dict[str, TorrentTelemetry]:
        """One batched, read-only qBittorrent info read for the fast cockpit refresh.

        The cheap sibling of `poll_torrent`: ONE `torrents_info` call
        covers every in-flight download (vs one call per torrent on the heavy
        cycle), and only sanitized telemetry comes back - no outcomes, no content
        paths - so the fast lane can never race the heavy poll's terminal
        decisions. A transient qBittorrent error or a missing row simply yields no
        entry (the row keeps its last telemetry until the next heavy poll).
        Response hashes are matched case-insensitively (qBittorrent lowercases).
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
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Drive the strategy's `import_completed`, swallowing any error.

        The import does live Sonarr HTTP work. A malformed response (a 200 with
        a non-JSON body, a candidate missing `path`, ...) must not abort the
        run and skip the end-of-run `cache_store.save` in `_finalize_run`.
        On any exception the record is left pending (returns a `LEAVE` probe) and
        the run continues. A real terminal failure is just retried next run / TTL'd.

        `force` is threaded through so the engine can tell the strategy to stop
        deferring to Sonarr on a clean `importPending` (the snapshot/reconcile
        passes and the final in-bound monitor poll). `at_deadline` flags the final
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
        except Exception as e:
            hub_error(f"Manual import failed for {pending.display_label} - leaving it for a later run", exc=e)
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only files-landed count for the wait bar (Tier-2 poll).

        Delegates to the active strategy. Never refreshes downloads, reads the
        queue, or issues a command (the strategy contract enforces that). An
        indeterminate zero when no strategy is bound.
        """

        if self._active_strategy is None:
            return ImportProgress(0, 0, determinate=False)
        return self._active_strategy.import_progress(pending)

    def _this_run_keys(self) -> set[PendingKey]:
        """Record keys grabbed THIS run - excluded from the carried-over passes.

        A this-run grab is reported as `added`. The snapshot / reconcile / tally
        skip these so a record is never double-reported as queued/importing/imported.
        Keyed per record, not per hash: a carried-over sibling on the same torrent
        (a prior run's record for another entry) must still be reconciled.
        """

        return {p.key for p in self._ctx.pending_imports}

    def _reconcile_one(self, pending: PendingImport) -> PendingState:
        """Poll one carried-over record once and fold it to a `PendingState`.

        Shared by the inline snapshot and the deferred reconcile: one non-blocking
        `poll_torrent`, then on COMPLETE drive one forced (CDH-off safe),
        non-deadline import attempt (so a still-missing file never warns). The
        outcome + the probe's verified-files flag fold through
        `classify_pending` into one state, stashed per record key for the
        pre-summary tally. A terminal IMPORTED is dropped + counted (drop FIRST,
        so the category gate sees only genuinely-remaining records), a MISSING is
        dropped. Returns the classified state.
        """

        poll = self.poll_torrent(pending.infohash)
        probe = ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        if poll.outcome is WaitOutcome.COMPLETE and poll.content_path:
            probe = self.try_import_completed(
                pending,
                poll.content_path,
                force=True,
                at_deadline=False,
            )

        state = classify_pending(poll.outcome, probe.files_present)
        self._ctx.pending_states[pending.key] = state
        if state is PendingState.IMPORTED:
            self.drop_pending(pending)
            self.apply_post_import_category(pending)
            self._ctx.stats.imported += 1
        elif state is PendingState.MISSING:
            self.drop_pending(pending)
        return state

    def snapshot_pending_for_series(self, series_id: int) -> None:
        """Reconcile + report this series' CARRIED-OVER pending records inline.

        For each durable record for `series_id` that is NOT a this-run grab (its
        infohash is absent from `_ctx.pending_imports` - those are already shown
        as `added`, so including them here would double-report), reconciles via
        `_reconcile_one` and renders the result inline (`log_pending_snapshot`).
        """

        if self._active_strategy is None:
            return

        run_grabs = self._this_run_keys()
        # Fresh per call and SQL-filtered to this series (so a record dropped earlier
        # this run is already absent) - replaces a full get_pending scan + Python
        # series filter once per series. The `record ->> 'series_id'` match only
        # returns JSON objects, rehydrated here (the series-scoped raw boundary).
        for key, raw in self.cache_store.get_pending_for_series(self._ctx.arr, series_id).items():
            # Skip this-run grabs: they're already reported as `added`, so a
            # `queued`/`importing`/`imported` row here would be a double report.
            if key in run_grabs:
                continue
            pending = PendingImport.from_json(raw)
            state = self._reconcile_one(pending)
            self._reporter.log_pending_snapshot(state, pending)

    def reconcile_remaining(self) -> None:
        """Non-blocking force-poll of carried-over records NOT snapshotted this run.

        The deferred-mode pre-summary step: reconciles, via `_reconcile_one`,
        every durable record whose key wasn't already touched by the
        per-series inline snapshot and isn't a this-run grab (those stay
        `added`). Quiet (no live region, deferred never blocks).
        """

        if self._active_strategy is None:
            return

        run_grabs = self._this_run_keys()
        for key, pending in self._pending_records().items():
            if key in self._ctx.pending_states:
                continue
            if key in run_grabs:
                continue
            self._reconcile_one(pending)

    def tally_carried_over_into_stats(self) -> None:
        """Bump queued/importing from each carried-over record's known status.

        `imported` is bumped at the point a record is reconciled+dropped (in the
        snapshot / reconcile), so here we only fold the records still in the store
        into `queued` / `importing`: a record touched this run uses its known
        `PendingState`. An un-touched store record (e.g. another series, in
        pure blocking where no reconcile ran) defaults to `QUEUED` without an
        extra poll. This-run grabs are excluded throughout (they're `added`), so
        no record is ever double-counted.
        """

        run_grabs = self._this_run_keys()
        # Iterate the raw stored keys, not `_pending_records()`: the loop reads only
        # the key + `pending_states`, so rehydrating each record via
        # `PendingImport.from_json` would build a full map only to discard it.
        for key in self.cache_store.get_pending(self._ctx.arr):
            if key in run_grabs:
                continue
            state = self._ctx.pending_states.get(key, PendingState.QUEUED)
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
        (`_ctx.pending_imports`) AND carried-over store records, deduped per
        record (`PendingKey`, so siblings sharing one torrent each get a row) - so
        a single-series run still finishes other-series carried-over
        downloads (the configured "monitor ALL" choice). Each cycle advances every
        active record once (so a fast torrent isn't stuck behind a slow one) into a
        `dict[row_key -> TorrentView]`, then pushes ONE `WaitSnapshot` to
        the view (which emits a graduation per newly-terminal torrent, and the
        renderers scroll it back).
        `imported` is reported ONLY when the episode files are verified present
        (`probe.files_present`), so an in-flight remote-mount copy reads
        `importing` until it lands. Per-torrent timeouts: `imports.wait_timeout`
        for the download, `imports.ready_timeout` for the import - anchored at the
        first COMPLETE and re-anchored each time another intended file lands, so
        it bounds a stall, not a big pack's whole copy. Ctrl-C pushes one final snapshot (so that cycle's terminals
        still graduate) then breaks the loop (the `finally` restores the terminal
        and the caller still saves the cache). The terminal outcomes are returned as a
        `WaitResult` for the run report + completion notification. The clock /
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
            )

        # Fresh-per-call behavioral object: it owns the per-cycle accumulators (so
        # there's nothing to reset between calls) and the advance logic. The loop
        # here keeps only the view lifecycle, the cycle pacing, and the Ctrl-C break.
        mp = MonitorPass(
            self,
            records,
            now=clock,
            dl_timeout=self._config.imports.wait_timeout,
            import_timeout=self._config.imports.ready_timeout,
        )

        try:
            view.update(mp.snapshot())
            while mp.active:
                try:
                    mp.run_cycle()
                    view.update(mp.snapshot())
                    if mp.active:
                        self._progress_wait(mp, view, nap)
                except KeyboardInterrupt:
                    # One final push so this cycle's terminals still graduate and
                    # the tally's elapsed reads interrupt time (the view is total,
                    # so the push can't raise past the break).
                    view.update(mp.snapshot())
                    hub_note(f"Wait interrupted - {len(mp.active)} left pending")
                    break
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

        Splits the inter-poll `nap` into `imports.progress_poll_interval` slices.
        Between the heavy cycles it runs only the cheap fast-lane reads: the
        episode-file count behind each importing row's "files inserted" bar
        (promoting a row the instant its files all land) and ONE batched
        qBittorrent info read keeping the downloading rows' bar/speed/ETA live
        (telemetry only - never the throttled rescan, the queue, an import
        command, or a phase transition). Falls back to one plain `nap(poll_s)`
        when the fast poll is disabled (<= 0) or no faster than the heavy poll. A
        `KeyboardInterrupt` during a slice propagates to the caller's break, as
        a plain `nap` would.
        """

        poll_s = self._config.imports.poll_interval
        progress_s = self._config.imports.progress_poll_interval
        if progress_s <= 0 or progress_s >= poll_s:
            nap(poll_s)
            return
        deadline = mp.now() + poll_s
        while mp.active:
            remaining = deadline - mp.now()
            if remaining <= 0:
                return
            nap(min(progress_s, remaining))
            if not mp.active:
                return
            # Run both fast-lane reads: the import bar first (it can promote/
            # retire rows), then the download telemetry - skipped entirely for a
            # view that renders no per-row telemetry (the non-TTY digest).
            progressed = mp.refresh_progress()
            telemetry_moved = view.wants_telemetry and mp.refresh_telemetry()
            if progressed or telemetry_moved:
                view.update(mp.snapshot())

    def _monitor_working_set(self) -> list[PendingImport]:
        """Dedup `_ctx.pending_imports` + rehydrated store records per record key.

        This-run grabs first (so their richer in-memory record wins a collision),
        then every durable store record not already present - the union the monitor
        waits on so ALL pending (this run's + carried-over) is monitored. Deduped
        by `PendingKey`, never bare infohash: two entries sharing one torrent are
        two records, each waited on for its own episode slice.
        """

        records: list[PendingImport] = []
        seen: set[PendingKey] = set()
        for pending in self._ctx.pending_imports:
            if pending.infohash and pending.key not in seen:
                seen.add(pending.key)
                records.append(pending)
        for key, pending in self._pending_records().items():
            if key in seen:
                continue
            seen.add(key)
            records.append(pending)
        return records

    def prune_expired_pending(self) -> None:
        """Drop durable pending records past their TTL (or with a bad stamp).

        Runs at the start of every non-off, non-preview run - including pure
        `blocking`, which never reconciles - so a never-completing torrent
        can't pile up in the cache forever. A record past
        `imports.pending_max_age_days` (or with an unparseable `added_at`) is
        dropped from the durable store.
        """

        cutoff = datetime.now() - timedelta(
            days=self._config.imports.pending_max_age_days,
        )

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
        """Remove ONE record from both the durable store and the run list.

        Record-scoped (`PendingKey`): a sibling record on the same torrent -
        another entry's still-waiting episode slice - is never dropped with it.
        """

        self.cache_store.drop_pending(self._ctx.arr, pending.key)
        self._ctx.pending_imports = [p for p in self._ctx.pending_imports if p.key != pending.key]

    def apply_post_import_category(self, pending: PendingImport) -> None:
        """Move a verified-imported torrent to `imports.post_import_category`.

        Called at the two confirmed-import sites (the reconcile passes and the
        monitor's IMPORTED terminal), AFTER the finished record is dropped -
        never for MISSING or a TTL drop. Gated on the whole torrent being done:
        SeaDex can list one torrent on several AniList entries, each with its own
        record for its own episode slice, and users key delete-with-data cleanup
        off this category - so the move happens only once NO pending record (in
        either arr - see `CacheStore.count_pending_for_infohash`) still claims
        the hash. The last record to verify makes the move. Creates the category
        on first use (qBittorrent 409s an unknown one). Best-effort: the import
        already succeeded, so a client error only warns - naming the record by
        its display label, not the bare infohash.
        """

        category = self._config.imports.post_import_category
        if not category or self.qbit is None:
            return
        remaining = self.cache_store.count_pending_for_infohash(pending.infohash)
        if remaining:
            self.logger.debug(
                f"{pending.display_label}: {count_noun(remaining, 'sibling record')} still pending on "
                "this torrent - deferring the category move",
            )
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


class MonitorPass:
    """One blocking-monitor invocation's mutable state + per-cycle advance logic.

    Built fresh at the top of `ImportWaitManager.run_monitor` from the
    manager, the working-set records, the clock, and the two per-torrent timeouts,
    so `advance` takes only the record and there is nothing to reset between
    runs (the object IS the per-invocation scope). It calls back to the manager's
    `poll_torrent` / `try_import_completed` / `drop_pending` (those are
    shared with the reconcile passes, so they stay on the manager).
    """

    views: dict[str, TorrentView]
    """The current frame the manager snapshots: each record's `TorrentView`, keyed by the
    record's `PendingKey.row_key` (siblings sharing a torrent are separate rows)."""
    results: list[WaitOutcomeRow]
    """Terminal rows, one per record that reached a terminal outcome."""
    active: set[str]
    """Record row keys still running (not yet terminal)."""
    dl_start: dict[str, float]
    """Per-record download-phase clock, stamped at construction."""
    import_start: dict[str, float]
    """Per-record import-phase clock, stamped on the first COMPLETE poll."""
    import_anchor: dict[str, float]
    """Per-record ready-deadline anchor: the first COMPLETE, re-stamped each time another
    intended file lands - so `ready_timeout` bounds a stalled import, not a big season
    pack Sonarr copies file-by-file."""

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
        self.records = records
        self.now = now
        self.dl_timeout = dl_timeout
        self.import_timeout = import_timeout
        # Sampled once here. The download clock for every record starts now and
        # `elapsed` measures from it.
        self.start = now()
        self.dl_start = {}
        self.import_start = {}
        self.import_anchor = {}
        # Highest determinate done-count per row: the baseline `_note_import_progress`
        # measures rises against (a max, so a stale lower reading can't fake a rise).
        self._import_seen: dict[str, int] = {}
        self.active = set()
        self.views = {}
        # Each row's underlying torrent, for the telemetry batch (siblings share one).
        self._hash_of: dict[str, str] = {}
        for r in records:
            k = r.key.row_key
            self.dl_start[k] = self.start
            self.active.add(k)
            self.views[k] = TorrentView(
                key=k,
                label=r.display_label,
                phase=Phase.QUEUED,
            )
            self._hash_of[k] = r.infohash
        # Per-cycle heavy-poll memo: sibling records share ONE qBittorrent read
        # per cycle (and see the same reading). Cleared by `run_cycle`.
        self._cycle_polls: dict[str, TorrentProbe] = {}
        self.results = []

    def run_cycle(self) -> None:
        """Run one heavy-poll cycle: clear the per-hash memo, then advance every active record."""

        self._cycle_polls.clear()
        for record in self.records:
            if record.key.row_key not in self.active:
                continue
            self.advance(record)

    def _poll(self, infohash: str) -> TorrentProbe:
        """The hash's heavy poll for this cycle - read once, shared by siblings."""

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

        return WaitSnapshot(tuple(self.views.values()), elapsed_s=self.elapsed())

    def _terminal(self, outcome: Outcome, record: PendingImport, *, files: int | None = None) -> None:
        """Record a terminal outcome: snapshot row + result + (maybe) drop + retire.

        Drops the durable record when (and only when) `outcome.dropped` - True for
        exactly IMPORTED and MISSING, so the displayed word and the store mutation
        can't diverge. A SUCCESS-class outcome additionally gets the post-import
        category, keyed off the same pinned enum vocabulary - the drop runs FIRST
        so the category gate counts only genuinely-remaining sibling records. The
        terminal row carries the pass-elapsed clock and (for an import) the
        verified files count, so the graduation ledger can state them.
        """

        k = record.key.row_key
        label = record.display_label
        self.views[k] = TorrentView(
            key=k,
            label=label,
            phase=Phase.TERMINAL,
            outcome=outcome,
            import_done=files,
            import_total=files,
            phase_elapsed_s=self.now() - self.dl_start[k],
        )
        self.results.append(WaitOutcomeRow(label=label, outcome=outcome))
        if outcome.dropped:
            self._mgr.drop_pending(record)
        if outcome.category is OutcomeCategory.SUCCESS:
            self._mgr.apply_post_import_category(record)
        self.active.discard(k)

    def advance(self, record: PendingImport) -> None:
        """Advance one torrent one monitor cycle (download or drive/verify import).

        Writes this torrent's current `TorrentView` into `views` (the frame
        the caller snapshots) and, on a terminal outcome, retires it via
        `_terminal`. `import_start` is stamped on the first COMPLETE.
        `imported` is gated on verified episode files, so a freshly-issued import
        command reads `importing` until the copy lands. The final in-bound attempt
        (`at_deadline`) both forces and warns; a file landing that same cycle
        re-anchors the deadline instead (`_note_import_progress`).
        """

        k = record.key.row_key
        label = record.display_label

        poll = self._poll(record.infohash)

        if poll.outcome is None:
            if self.now() - self.dl_start[k] >= self.dl_timeout:
                self._terminal(Outcome.DOWNLOAD_TIMED_OUT, record)
                return
            prior = self.views.get(k)
            if not poll.observed:
                # Transient qBittorrent error: the zeroed probe is a placeholder,
                # not a reading - keep the row's last real state (no 0% bar flash,
                # no fake stall sample, an importing row stays importing) and let
                # its clock tick.
                if prior is not None and prior.phase is Phase.DOWNLOADING:
                    self.views[k] = replace(prior, phase_elapsed_s=self.now() - self.dl_start[k])
                return
            # Speed history advances once per heavy poll (stalled/None -> 0),
            # bounded to the sparkline window. The fast telemetry refresh
            # deliberately never samples it, so the window stays minutes wide.
            history = prior.speed_history if prior is not None and prior.phase is Phase.DOWNLOADING else ()
            self.views[k] = TorrentView(
                key=k,
                label=label,
                phase=Phase.DOWNLOADING,
                fraction=poll.progress,
                speed_bps=poll.speed_bps,
                eta_s=poll.eta_s,
                bytes_done=poll.bytes_done,
                bytes_total=poll.bytes_total,
                phase_elapsed_s=self.now() - self.dl_start[k],
                speed_history=(*history, poll.speed_bps or 0)[-SPARK_SAMPLES:],
            )
            return
        if poll.outcome is WaitOutcome.MISSING:
            self._terminal(Outcome.MISSING, record)
            return
        if poll.outcome is WaitOutcome.ERRORED:
            self._terminal(Outcome.DOWNLOAD_ERRORED, record)
            return
        if not poll.content_path:
            # COMPLETE but qBittorrent reported no save path: its own outcome,
            # not a misleading "timed out" (the download finished fine).
            self._terminal(Outcome.NO_CONTENT_PATH, record)
            return

        # COMPLETE: drive / verify our import, gating `imported` on verified files.
        self.import_start.setdefault(k, self.now())
        self.import_anchor.setdefault(k, self.import_start[k])
        at_deadline = self.now() - self.import_anchor[k] >= self.import_timeout
        probe = self._mgr.try_import_completed(
            record,
            poll.content_path,
            force=at_deadline,
            at_deadline=at_deadline,
        )
        landed = self._note_import_progress(k, probe.imported_count, probe.target_count)
        if probe.files_present:
            self._terminal(Outcome.IMPORTED, record, files=probe.target_count or None)
        elif at_deadline and not landed:
            self._terminal(
                Outcome.STILL_IMPORTING if probe.command_issued else Outcome.NOT_READY,
                record,
            )
        elif probe.readiness is ImportReadiness.LEAVE:
            self._terminal(Outcome.NOTHING_TO_IMPORT, record)
        else:
            # RETRY / copy in flight: the command was accepted but the files
            # haven't landed yet (or Sonarr is still scanning). Seed the "files
            # inserted" bar from the probe counts - determinate only when the seed
            # map is whole (target_count > 0). Otherwise an indeterminate row.
            total = probe.target_count
            done = probe.imported_count
            self.views[k] = TorrentView(
                key=k,
                label=label,
                phase=Phase.IMPORTING,
                fraction=(done / total if total else 1.0),
                import_done=(done if total else None),
                import_total=(total if total else None),
                phase_elapsed_s=self.now() - self.import_start[k],
                command_issued=probe.command_issued,
            )

    def _note_import_progress(self, k: str, done: int, total: int) -> bool:
        """Re-anchor the row's ready deadline when another intended file lands.

        `import_timeout` bounds a STALLED import, not a big season pack Sonarr
        copies file-by-file: each rise in the determinate done-count re-stamps
        `import_anchor`. The first determinate reading is a baseline (files
        present before we started watching are not progress), and indeterminate
        counts (`total` 0) never move the anchor.
        """

        if total <= 0:
            return False
        last = self._import_seen.get(k)
        self._import_seen[k] = done if last is None else max(last, done)
        if last is None or done <= last:
            return False
        self.import_anchor[k] = self.now()
        return True

    def refresh_progress(self) -> bool:
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
        for record in self.records:
            k = record.key.row_key
            if k not in self.active:
                continue
            view = self.views.get(k)
            if view is None or view.phase is not Phase.IMPORTING:
                continue
            try:
                progress = self._mgr.import_progress(record)
            except Exception:
                self._mgr.logger.debug(f"import progress poll for {k} failed", exc_info=True)
                continue
            # Indeterminate (partial seed map) -> no bar, no promotion. Leave the
            # row to the heavy poll's repaired done-check.
            if not progress.determinate or progress.total <= 0:
                continue
            self._note_import_progress(k, progress.done, progress.total)
            if progress.done >= progress.total:
                self._terminal(Outcome.IMPORTED, record, files=progress.total)
                changed = True
            elif (progress.done, progress.total) != (view.import_done, view.import_total):
                self.views[k] = replace(
                    view,
                    fraction=progress.done / progress.total,
                    import_done=progress.done,
                    import_total=progress.total,
                    phase_elapsed_s=self.now() - self.import_start.get(k, self.now()),
                )
                changed = True
        return changed

    def refresh_telemetry(self) -> bool:
        """Cheap fast-lane pass: refresh each downloading row's live telemetry.

        One batched qBittorrent info read across the still-active DOWNLOADING rows
        keeps their bar/speed/ETA moving between heavy polls. Telemetry only: no
        outcomes, no phase transitions (a completion just shows a full bar until
        the heavy poll steps in) and no speed-history sample (that advances once
        per heavy poll, so the sparkline window stays minutes wide). Returns
        whether anything changed, so the caller re-pushes a snapshot only when
        there is something new to draw.
        """

        downloading = [k for k, view in self.views.items() if k in self.active and view.phase is Phase.DOWNLOADING]
        # One batch read per underlying torrent (sibling rows share the reading).
        hashes = list(dict.fromkeys(self._hash_of[k] for k in downloading))
        by_hash = self._mgr.poll_telemetry(hashes)
        changed = False
        for k in downloading:
            telemetry = by_hash.get(self._hash_of[k])
            if telemetry is None:
                continue
            view = self.views.get(k)
            if view is None or view.phase is not Phase.DOWNLOADING:
                continue
            current = TorrentTelemetry(
                view.fraction,
                view.speed_bps,
                view.eta_s,
                view.bytes_done,
                view.bytes_total,
            )
            if telemetry == current:
                continue
            self.views[k] = replace(
                view,
                fraction=telemetry.progress,
                speed_bps=telemetry.speed_bps,
                eta_s=telemetry.eta_s,
                bytes_done=telemetry.bytes_done,
                bytes_total=telemetry.bytes_total,
                phase_elapsed_s=self.now() - self.dl_start[k],
            )
            changed = True
        return changed
