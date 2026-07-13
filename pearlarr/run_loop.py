"""The per-arr run loop: the `run_sync` scaffolding shared by both Arr strategies."""

import time
from typing import final

from .arr_activity import ArrActivityMonitor
from .boot_flow import BootFlow
from .import_wait import ImportWaitManager
from .log import arr_item_noun, count_noun
from .manual_import import (
    ImportWaitMode,
    resolve_wait_mode,
)
from .output import hub_error, hub_note, hub_warn
from .protocols import ArrSync, ImportCompleter
from .reporter import RunContext
from .run_services import RunDeps, RunServices
from .seadex_types import ArrItem
from .wait_view import (
    WaitResult,
)


@final
class RunLoop:
    """The Arr-agnostic run loop driving an injected strategy.

    Receives its shared collaborators as a `RunDeps` bundle plus the
    `RunServices` hub (both built and injected by the composition root
    in `bootstrap.py`) and owns the run loop, the per-run `RunContext`
    lifecycle, and the wait-pass machinery. It drives an injected
    `ArrSync` strategy (passed to `run_sync`) for the
    Arr-specific pieces; the strategy holds the `RunServices` hub as its
    `services` and calls the shared per-id pipeline through it, so it never
    sees this loop type. The loop never holds the strategy and never constructs
    its own dependencies.
    """

    def __init__(self, deps: RunDeps, services: RunServices) -> None:
        """Receive the shared collaborators + services hub and set up per-run state.

        The loop adopts the services hub's placeholder context and pushes each
        run's fresh context back into it; the hub is also injected into the
        strategy.
        """

        # Unpack the injected collaborators into the attribute names the run loop
        # methods read directly (the loop-side subset of the deps).
        self._config = deps.config
        self._arr_config = deps.arr_config
        self.qbit = deps.qbit
        self.logger = deps.logger
        self._seadex = deps.seadex
        self.cache_store = deps.cache_store
        self._anilist = deps.anilist
        self._notifier = deps.notifier
        self._reporter = deps.reporter

        self._services = services

        # The active strategy for the current run, (re)set at the top of run_sync;
        # the placeholder None here is replaced before any import hook is invoked.
        self._active_strategy: ImportCompleter | None = None

        # Per-run state lives on this context (see RunServices.__init__ for the
        # full enumeration). ADOPT the services hub's placeholder (one placeholder,
        # never a second mint) - its dry_run=False + OFF wait mode keep every
        # preview / pending-import path a safe no-op - so the object is usable
        # before run_sync.
        self._ctx = services.ctx

        # Loop-side per-run collaborator, built from the deps hub + the adopted
        # placeholder ctx. begin_run rebinds its ctx at the top of each run.
        self._wait_manager = ImportWaitManager(
            deps=deps,
            ctx=self._ctx,
            strategy=self._active_strategy,
        )
        self.begin_run(self._ctx)

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context to every per-run collaborator.

        Two-phase bind: called once with the adopted placeholder ctx in
        `__init__` (so pre-run paths are safe) and again from `run_sync`
        right after `reset_run_stats` swaps in the run's real ctx, so every
        collaborator - the services hub (and through it the filter + grab
        pipeline) and the loop's wait manager - rebinds to the fresh ctx.
        """

        self._services.begin_run(ctx)
        self._wait_manager.begin_run(ctx, self._active_strategy)

    def reset_run_stats(
        self,
        dry_run: bool,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
    ) -> None:
        """Start a fresh run context and the run clock, and rebind collaborators.

        Replaces the run-scoped state wholesale with a new RunContext - this is
        the ONLY fresh-mint site; its `arr` is read off the services hub (the
        authority) - and stamps the hub-counts mark (warning/error counts are
        diffed against it when the summary is logged). The
        `begin_run` rebind is folded in here so the ctx swap and the
        collaborator rebind can never drift apart - a missed rebind would
        silently route a collaborator's writes to the orphaned prior context.

        Args:
            dry_run: Whether this run simulates without grabbing/writing.
            import_wait_mode: The run's resolved wait mode
                (cli > config > default), stamped onto the fresh context.
        """

        self._ctx = RunContext(
            arr=self._services.arr,
            dry_run=dry_run,
            import_wait_mode=import_wait_mode,
            # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
            started_monotonic=time.monotonic(),
            counts_mark=self._reporter.counts_mark(),
        )
        self.begin_run(self._ctx)

    # --- Run orchestration (shared machinery) -------------------------------
    #
    # run_sync is the shared scaffolding both Arrs use (reset stats, fetch items,
    # optional single-id filter, AniList prefetch, the per-item loop, and the
    # end-of-run save + summary). The Arr-specific pieces are the injected
    # strategy's hooks (get_items, filter_to_single, item_anilist_ids,
    # process_al_id); the strategy holds the RunServices hub as its services and
    # calls the shared per-id head/tail (al_id_prologue / cached_entry_skip /
    # grab_and_cache) through it.

    def run_sync[ItemT: ArrItem](
        self,
        strategy: ArrSync[ItemT],
        *,
        item_id: int | None,
        dry_run: bool,
        import_wait_mode: ImportWaitMode | None = None,
        boot: BootFlow,
    ) -> None:
        """Shared run scaffolding for both Arr syncers.

        Generic in `ItemT` (the strategy's item protocol), so the body sees a
        precise `list[ItemT]` / `item: ItemT` - the same concrete type the
        strategy's hooks consume and produce. `ArrSync` is invariant in its
        item type, so a concrete (non-union) strategy must reach this call:
        the composition root (`bootstrap.py`) branches per Arr so each call
        binds one `ItemT` cleanly.

        Args:
            strategy: The Arr-specific strategy to drive (injected
                by the composition root, which picks Sonarr/Radarr at runtime). It
                already holds the shared `RunServices` hub as its services,
                so its hooks are called without passing anything back.
            item_id: If set, only run for the single item with this
                id (TMDB for Radarr, TVDB for Sonarr)
            dry_run: Simulate the run without grabbing torrents, writing
                the cache, or sending notifications
            import_wait_mode: The CLI `--import-wait-mode`
                override, resolved cli > config > default. None falls back to the
                configured `imports.wait_mode`.
            boot: The startup cockpit's producer facade; the
                library fetch and the metadata prefetch graduate into it as steps,
                and its section is capped right before the per-item scan begins
                (a no-op unless a hub renders).
        """

        # Hold the active strategy (so _finalize_run / _grab can call its import
        # hook) and resolve the effective wait mode (cli > config > default) for
        # the whole run. The loop only ever calls import_completed off it, so it
        # is held under the narrow, non-generic ImportCompleter protocol - which a
        # concrete ArrSync structurally satisfies, so no invariant-ItemT cast.
        self._active_strategy = strategy
        resolved_wait_mode = resolve_wait_mode(
            import_wait_mode,
            self._config.imports.wait_mode,
        )

        # The run's arr comes off the services hub (the authority; each fresh
        # ctx.arr is a per-run copy of it).
        arr = self._services.arr

        # Start a fresh run context (stats + clock + counter snapshot + the run's
        # dry_run / wait-mode flags); reset_run_stats rebinds the collaborators to it.
        self.reset_run_stats(dry_run=dry_run, import_wait_mode=resolved_wait_mode)

        # Tend the durable pending-import store at run start (never on a preview,
        # since waiting/importing needs a real qBittorrent client). The TTL prune
        # runs for EVERY active mode - including pure blocking - so aged-out records
        # can't accumulate forever. The reconcile/snapshot/monitor that actually
        # report and import carried-over records run AFTER the per-item loop (the
        # inline per-series snapshot) and in _finalize_run (deferred reconcile +
        # the post-summary blocking monitor), never before the banner.
        if self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._services.is_preview():
            self._wait_manager.prune_expired_pending()

        # Fetch the library (the long pre-scan network wait) inside the cockpit so
        # the spinner animates through it; the count graduates as the step's detail.
        with boot.step(f"Fetching {arr.capitalize()} library") as fetching:
            all_items: list[ItemT] = strategy.get_items()

            # If we're targeting a single item, filter down to it
            if item_id is not None:
                all_items = strategy.filter_to_single(all_items, item_id)

            n_items = len(all_items)
            fetching.note(arr_item_noun(arr, n_items))

        # Arr-side activity scan: one /history/since poll marks items whose files
        # the arr changed since the last pass dirty, so the cached-entry skip
        # re-evaluates just those (everything, on a coverage gap). Runs BEFORE
        # the prefetches so al_id_needs_scan warms exactly the dirty subset (the
        # sweep-perf invariant); skipped when ignore_seadex_update_times already
        # re-processes everything.
        monitor: ArrActivityMonitor | None = None
        if self._config.advanced.detect_arr_activity and not self._config.seadex.ignore_seadex_update_times:
            with boot.step(f"Checking {arr.capitalize()} activity") as step:
                monitor = ArrActivityMonitor(arr, self.cache_store, self.logger)
                scan = monitor.scan(strategy.history_since)
                dirty: set[int] = set()
                for item in all_items:
                    if scan.rescan_all or item.id in scan.touched:
                        dirty.update(strategy.item_anilist_ids(item, log_ignored=False))
                self._services.mark_dirty(dirty)
                if scan.rescan_all:
                    step.note("history gap - rechecking all entries")
                else:
                    step.note("none" if not dirty else count_noun(len(dirty), "changed entry", "changed entries"))

        # Warm the AniList cache before the per-item loop: reuse what past runs
        # fetched, then batch-fetch (id_in pages) everything still missing, so the
        # loop rarely hits AniList one id at a time and trips its rate limit.
        # Seed the AniList cache + collect the run's candidate ids - instant local
        # work, so no ledger line. Then surface each network/concurrent warm as its
        # own timed step instead of one opaque "Prefetching metadata".
        self._anilist.load_cache()
        prefetch_ids: set[int] = set()
        for item in all_items:
            if not item.monitored and self._arr_config.ignore_unmonitored:
                continue
            prefetch_ids.update(
                strategy.item_anilist_ids(item, log_ignored=False),
            )

        with boot.step("Fetching AniList metadata") as step:
            fetched = self._anilist.prefetch(prefetch_ids, preview=self._services.is_preview(), progress=step)
            step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Bulk-fetch SeaDex entries for the same ids in batched OR-filter queries,
        # collapsing the per-id from_id round-trips (one per library id, just to read
        # updated_at) into a handful. entry() then serves from this warmed cache.
        # An outage mid-prefetch must not claim "N entries" it never fetched.
        with boot.step("Fetching SeaDex entries") as step:
            fetched = self._seadex.prefetch(prefetch_ids, progress=step)
            if self._seadex.outage:
                step.warn("SeaDex unreachable - unfetched titles will be skipped")
            else:
                step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Warm the per-item episode lists concurrently (Sonarr only; Radarr no-ops,
        # so it gets no step). Kept FRESH - not cached across runs - so the grab/skip
        # decision still reads current Sonarr file state; this only collapses the
        # sequential per-series fetch latency.
        if strategy.warms_episodes:
            with boot.step("Fetching Sonarr episodes") as step:
                warmed = strategy.prefetch_episodes(all_items, progress=step)
                step.note("cached" if warmed == 0 else count_noun(warmed, "series", "series"))
        else:
            strategy.prefetch_episodes(all_items)

        # Tear the cockpit down BEFORE the per-item scan logs anything, so the scan
        # never reflows above a stale spinner; the "ready in Xs" capstone lands here.
        boot.end_section()

        self._reporter.log_arr_start(arr, n_items)

        # Matching preferences changed since the last vouched pass: the skip gate
        # re-checks every cached verdict this run - say so once (the ignore flag
        # already re-checks everything, so it needs no announcement). Only on a
        # full run, mirroring the vouch: a single-item run re-checks just its id,
        # so the whole-library note would overstate it and recur every time.
        if item_id is None and self._services.selection_stale and not self._config.seadex.ignore_seadex_update_times:
            hub_note("Matching settings changed - rechecking cached entries")

        # Set when a per-id grab hits max_torrents_to_add: breaks the scan and falls
        # through to the single _finalize_run site below.
        cap_reached = False
        for item_idx, item in enumerate(all_items):
            try:
                item_title = item.title

                self._reporter.log_arr_item_start(
                    arr,
                    item_title,
                    item_idx + 1,
                    n_items,
                )

                # If we're not monitored, then skip if ignore_unmonitored is switched on
                if not item.monitored and self._arr_config.ignore_unmonitored:
                    self._reporter.log_arr_item_unmonitored(self._ctx, item_title)
                    continue

                # Get the mappings from the Arr item to AniList
                al_mappings = strategy.item_anilist_ids(item)

                if len(al_mappings) == 0:
                    self._reporter.log_no_anilist_mappings(self._ctx, item_title)
                    continue

                for al_id, mapping in al_mappings.items():
                    # process_al_id returns True only when max_torrents_to_add was
                    # reached - stop the whole run. The post-loop _finalize_run (the
                    # single finalize site) still runs, so the blocking/hybrid pass
                    # imports this run's records before the save + summary. A separate
                    # post-loop max check would be redundant with this: the in-block
                    # check fires after every add, so torrents_added can't reach the
                    # cap without process_al_id stopping first.
                    try:
                        if strategy.process_al_id(
                            item=item,
                            al_id=al_id,
                            mapping=mapping,
                        ):
                            cap_reached = True
                            break
                    except Exception as e:
                        # Contain a per-id failure to THIS AniList id: a transient error
                        # on one season must not skip the item's other seasons.
                        hub_error(
                            f"{item_title} (AniList #{al_id}): unexpected error ({e}) - skipping this AniList id",
                            exc=e,
                        )
                        continue

                if cap_reached:
                    break

                # Non-blocking per-item snapshot of this series' CARRIED-OVER
                # pending records (grabbed in a prior run). Runs after all of an
                # item's AniList ids so it covers the cached/grabbed/no-entry paths
                # uniformly, and reports each carried-over record inline inside the
                # series block. Sonarr returns its series id; Radarr returns None
                # (no pending records), short-circuiting the snapshot.
                sid = strategy.pending_import_series_id(item)
                if (
                    sid is not None
                    and self._ctx.import_wait_mode is not ImportWaitMode.OFF
                    and not self._services.is_preview()
                ):
                    self._wait_manager.snapshot_pending_for_series(sid)

            except Exception as e:
                title = getattr(item, "title", "unknown title")
                hub_error(f"{title}: unexpected error ({e}) - skipping this title", exc=e)
                continue

        # Advance the history checkpoint only when the pass covered the whole
        # library (a single-item or capped run leaves later activity unseen); the
        # staged write persists only at _finalize_run's non-preview save.
        if monitor is not None and item_id is None and not cap_reached:
            monitor.commit_checkpoint()

        # Same full-coverage rule for the selection digest: an un-capped whole-
        # library pass re-checked every cached verdict under the current matching
        # settings, so vouch for them (also seeds the digest on a first run). An
        # outage run skipped whatever SeaDex never served, so it cannot vouch.
        # A per-id error contained mid-scan leaves that one title on its prior
        # verdict (retried on its next SeaDex update / dirty mark); we vouch anyway
        # - blocking on any flaky title would re-scan the whole library every run,
        # and dropping its entry to force a re-check could re-grab.
        if item_id is None and not cap_reached and not self._seadex.outage:
            self.cache_store.vouch_selection(arr, self._config.selection_digest())

        # Run the end-of-run blocking pass (blocking/hybrid only), then persist
        # the run and log the summary. Per-title update_cache calls only mutate
        # memory, so this finalize is what actually saves (and sorts by id).
        self._finalize_run()

    # --- Wait-for-completion orchestration ----------------------------------
    #
    # The completion wait/poll machinery lives on `self._wait_manager`
    # (`ImportWaitManager`); the loop keeps the run tail
    # (`_finalize_run`) that drives its passes in order plus the walk-away
    # completion notification. Every path is a no-op under preview (no client).

    def _finalize_run(self) -> None:
        """Shared run tail: reconcile + tally, print the summary, THEN block.

        Bracketed by the two run-lifecycle close boundaries (the scan closes at
        the top, the run at the very end). In order:

          1. deferred-mode pre-summary reconcile of any carried-over records not
             already snapshotted inline (non-blocking; feeds the counters);
          2. fold every still-pending carried-over record into the
             `queued`/`importing`/`imported` counters (this-run grabs stay
             `added`);
          3. print the scoreboard - so the summary reflects the pre-monitor state
             and never reports completion for this-run grabs;
          4. ONLY for blocking/hybrid, run the interleaved monitor + live region
             dead last, after the summary, so the wait/import is the live report;
          5. save the cache last in a finally spanning steps 1-4, so a raise
             anywhere still persists the run's staged writes and the store
             reflects both the inline-snapshot and the monitor drops.

        Every wait/import path is skipped on a preview (no client / dry run).
        """

        # Close the scan (and any open entry) before anything below can raise or
        # log: the reconcile/tally diagnostics are run-level facts, not details of
        # the last entry. Always ScanStarted-paired - run_sync (this method's only
        # caller) has no early return between log_arr_start and here.
        self._reporter.scan_finished(self._ctx.arr)

        preview = self._services.is_preview()
        active = self._ctx.import_wait_mode is not ImportWaitMode.OFF and not preview

        # The finally guards the whole tail: a raise in reconcile/tally/summary/
        # monitor must not let bootstrap's close roll back the run's staged
        # writes. The save trails the monitor to also capture its drops.
        try:
            if active and self._ctx.import_wait_mode is ImportWaitMode.DEFERRED:
                self._wait_manager.reconcile_remaining()
            if active:
                self._wait_manager.tally_carried_over_into_stats()

            self._reporter.log_run_summary(
                self._ctx,
                preview=preview,
                has_client=self.qbit is not None,
            )

            if active and self._ctx.import_wait_mode in (
                ImportWaitMode.BLOCKING,
                ImportWaitMode.HYBRID,
            ):
                result = self._wait_manager.run_monitor()
                # Best-effort walk-away graft, run only when something actually
                # waited. It swallows its own errors so a bad webhook can never
                # skip the cache save in the finally below.
                if result is not None and result.waited:
                    self._notify_wait_complete(result)
        finally:
            self.cache_store.save(preview=preview)

        # The leg's last event. Deliberately OUTSIDE the finally: a monitor/save
        # raise leaves it to bootstrap's unwind emit, so it lands exactly once.
        self._reporter.run_finished(self._ctx.arr)

    def _notify_wait_complete(self, result: WaitResult) -> None:
        """Push the completion notification, gated on `wait_notify`; swallow errors."""

        if not self._config.notifications.wait_notify:
            return
        try:
            _ = self._notifier.push_wait_summary(arr=self._ctx.arr, result=result)
        except Exception as e:
            hub_warn("Wait completion notification failed unexpectedly - the notification was dropped", exc=e)
