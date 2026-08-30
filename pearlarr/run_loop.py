"""The per-arr run loop shared by both Arr strategies."""

import time
from typing import final

from .arr_activity import ArrActivityMonitor
from .boot_flow import BootFlow
from .import_wait import ImportWaitManager
from .log import arr_item_noun, count_noun
from .manual_import import ImportWaitMode, PendingState
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
    """The Arr-agnostic run loop driving an injected strategy."""

    def __init__(self, deps: RunDeps, services: RunServices) -> None:
        """Receive the shared collaborators + services hub and set up per-run state."""

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

        # (Re)set at the top of run_sync. The None placeholder is replaced before any import hook runs.
        # Held under the narrow, non-generic ImportCompleter ABC (a concrete ArrSync subclasses it), so no ItemT cast.
        self._active_strategy: ImportCompleter | None = None

        # Adopt the hub's placeholder ctx, never mint a second one.
        # Its dry_run=False and OFF wait mode keep every preview and pending-import path a safe no-op before run_sync.
        self._ctx = services.ctx

        self._wait_manager = ImportWaitManager(deps=deps, ctx=self._ctx)
        self.begin_run(self._ctx)

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context to every per-run collaborator."""

        self._services.begin_run(ctx)
        self._wait_manager.begin_run(ctx, self._active_strategy)

    def reset_run_stats(
        self,
        dry_run: bool,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
    ) -> None:
        """Start a fresh run context and the run clock, and rebind collaborators."""

        self._ctx = RunContext(
            arr=self._services.arr,
            dry_run=dry_run,
            import_wait_mode=import_wait_mode,
            # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
            started_monotonic=time.monotonic(),
            counts_mark=self._reporter.counts_mark(),
        )
        self.begin_run(self._ctx)

    @property
    def _wait_active(self) -> bool:
        """Whether the wait machinery runs this pass (an active mode, not preview)."""

        return self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._services.is_preview()

    # --- Run orchestration --------------------------------------------------

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

        Args:
            strategy: The Arr-specific strategy to drive.
            item_id: If set, only run for the single item with this id (TMDB for Radarr, TVDB for Sonarr).
            dry_run: Simulate the run without grabbing torrents, writing the cache, or sending notifications.
            import_wait_mode: The CLI `--import-wait-mode` override. None falls back to `imports.wait_mode`.
            boot: The startup cockpit's producer facade.
        """

        self._active_strategy = strategy
        resolved_wait_mode = import_wait_mode if import_wait_mode is not None else self._config.imports.wait_mode

        arr = self._services.arr

        self.reset_run_stats(dry_run=dry_run, import_wait_mode=resolved_wait_mode)

        # The TTL prune runs at run start for EVERY active mode, so aged-out records can't accumulate forever.
        # Reporting carried-over records is per item, importing them is _finalize_run's, never before the banner.
        if self._wait_active:
            self._wait_manager.prune_expired_pending()

        with boot.step(f"Fetching {arr.capitalize()} library") as fetching:
            all_items: list[ItemT] = strategy.get_items()

            if item_id is not None:
                all_items = strategy.filter_to_single(all_items, item_id)

            n_items = len(all_items)
            fetching.note(arr_item_noun(arr, n_items))

        # The activity scan must run before the prefetches, which warm exactly the dirty subset it marks.
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

        # Warm in bulk: fetching one id at a time in the loop trips AniList's rate limit.
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

        # Batched OR-filter queries collapse the per-id from_id round-trips. An outage mid-prefetch
        # must not claim "N entries" it never fetched.
        with boot.step("Fetching SeaDex entries") as step:
            fetched = self._seadex.prefetch(prefetch_ids, progress=step)
            if self._seadex.outage:
                # Terse: the gateway already warned once with the failure detail.
                step.warn("unreachable")
            else:
                step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Episode lists are warmed fresh every run, never cached: a stale list under-grabs.
        if strategy.warms_episodes:
            with boot.step("Fetching Sonarr episodes") as step:
                warmed = strategy.prefetch_episodes(all_items, progress=step)
                step.note("cached" if warmed == 0 else count_noun(warmed, "series", "series"))
        else:
            strategy.prefetch_episodes(all_items)

        # Tear the cockpit down before the scan logs anything, or it reflows above a stale spinner.
        boot.end_section()

        self._reporter.log_arr_start(arr, n_items)

        # Only on a full run, mirroring the vouch: a single-item run re-checks just its id, so the
        # whole-library note would overstate it and recur every time.
        if item_id is None and self._services.selection_stale and not self._config.seadex.ignore_seadex_update_times:
            hub_note("Matching settings changed - rechecking cached entries")

        cap_reached = False
        for item_idx, item in enumerate(all_items):
            try:
                if self._scan_item(strategy, item, item_idx, n_items):
                    cap_reached = True
                    break
            except Exception as e:
                title = getattr(item, "title", "unknown title")
                hub_error(f"{title}: unexpected error ({e}) - skipping this title", exc=e)
                continue

        # ONE full-coverage predicate for both gates below. They drifted apart once: an outage run committed the
        # checkpoint, consuming drift events it never acted on.
        full_pass = item_id is None and not cap_reached and not self._seadex.outage

        # Held on outage so the next healthy run re-derives the same dirty ids (the query overlap and id
        # dedup absorb the replay). The staged write persists only at _finalize_run's non-preview save.
        if monitor is not None and full_pass:
            monitor.commit_checkpoint()

        # A contained per-id error still vouches: otherwise one flaky title re-scans the library every run.
        # Dropping its entry to force a re-check is not the answer either, that could re-grab.
        if full_pass:
            self.cache_store.vouch_selection(arr, self._config.selection_digest())

        # Per-title update_cache calls only mutate memory, so this finalize is what actually saves.
        self._finalize_run()

    def _scan_item[ItemT: ArrItem](
        self,
        strategy: ArrSync[ItemT],
        item: ItemT,
        item_idx: int,
        n_items: int,
    ) -> bool:
        """Scan one library item, returning True iff a grab hit the add cap."""

        arr = self._services.arr
        item_title = item.title

        self._reporter.log_arr_item_start(
            arr,
            item_title,
            item_idx + 1,
            n_items,
        )

        if not item.monitored and self._arr_config.ignore_unmonitored:
            self._reporter.log_arr_item_unmonitored(self._ctx, item_title)
            return False

        al_mappings = strategy.item_anilist_ids(item)

        if len(al_mappings) == 0:
            self._reporter.log_no_anilist_mappings(self._ctx, item_title)
            return False

        for al_id, mapping in al_mappings.items():
            # process_al_id returns True only when max_torrents_to_add was reached, which stops the whole run. A
            # post-loop max check would be redundant: the in-block check fires after every add.
            try:
                if strategy.process_al_id(
                    item=item,
                    al_id=al_id,
                    mapping=mapping,
                ):
                    return True
            except Exception as e:
                # Contain the failure to THIS AniList id: one bad season must not skip the item's others.
                hub_error(
                    f"{item_title} (AniList #{al_id}): unexpected error ({e}) - skipping this AniList id",
                    exc=e,
                )
                continue

        # A read-only snapshot, never an import: the end pass owns that. Radarr returns None here.
        # Placed after all of the item's AniList ids so it covers the cached, grabbed and no-entry paths alike.
        if self._wait_active and (sid := strategy.pending_import_series_id(item)) is not None:
            self._wait_manager.snapshot_pending_for_series(sid)

        return False

    # --- Wait-for-completion orchestration ----------------------------------

    def _finalize_run(self) -> None:
        """Shared run tail: the one-cycle check + tally, print the summary, THEN the waiting monitor."""

        # Close the scan before anything below logs: these diagnostics are run-level facts.
        # Always ScanStarted-paired: run_sync, the only caller, has no early return between log_arr_start and here.
        self._reporter.scan_finished(self._ctx.arr)

        preview = self._services.is_preview()

        supports_monitor = self._active_strategy is not None and self._active_strategy.supports_blocking_monitor
        end_pass_waits = supports_monitor and self._ctx.import_wait_mode in (
            ImportWaitMode.BLOCKING,
            ImportWaitMode.HYBRID,
        )

        # The finally guards the whole tail: a raise in check, tally, summary or monitor must not let bootstrap's
        # close roll back the run's staged writes. The save trails the monitor to also capture its drops.
        try:
            if self._wait_active:
                # Heal earlier runs' kept-for-cleanup records before this run's passes read the store.
                self._wait_manager.retry_cleanup_records()
            if self._wait_active and not end_pass_waits:
                check = self._wait_manager.check_once()
                if check is not None:
                    self._ctx.stats.imported += check.carried_over_imported
            if self._wait_active:
                # This owns the tally's `imported` bump: the snapshot's verified imports land in
                # `pending_states`, the passes report theirs on their carried-over result rows.
                self._ctx.stats.imported += sum(
                    1 for state in self._ctx.pending_states.values() if state is PendingState.IMPORTED
                )
                self._wait_manager.tally_carried_over_into_stats()

            self._reporter.log_run_summary(
                self._ctx,
                preview=preview,
                has_client=self.qbit is not None,
            )

            if self._wait_active and end_pass_waits:
                result = self._wait_manager.run_monitor()
                if result is not None:
                    self._ctx.stats.imported += result.carried_over_imported
                    # _notify_wait_complete swallows its own errors, so a bad webhook can never skip the save below.
                    if result.waited:
                        self._notify_wait_complete(result)
        finally:
            self.cache_store.save(preview=preview)

        # Deliberately outside the finally: on a raise, bootstrap's unwind emits it, so it lands once.
        self._reporter.run_finished(self._ctx.arr)

    def _notify_wait_complete(self, result: WaitResult) -> None:
        """Push the completion notification, gated on `wait_notify`, swallowing errors."""

        if not self._config.notifications.wait_notify:
            return
        try:
            _ = self._notifier.push_wait_summary(arr=self._ctx.arr, result=result)
        except Exception as e:
            hub_warn("Wait completion notification failed unexpectedly - the notification was dropped", exc=e)
