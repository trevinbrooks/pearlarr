"""The per-arr run loop shared by both Arr strategies."""

import time
from typing import final

from .arr_activity import ArrActivityReader
from .arr_models import ArrItem
from .arr_run_services import ArrRunDeps, ArrRunServices
from .arr_run_state import ArrRunState, unresolved_label
from .arr_strategy import ArrStrategy, ImportCompleter
from .boot_flow import BootFlow
from .import_wait import ImportWaitManager
from .output import hub_error, hub_note, hub_warn
from .output.console_text import arr_item_noun, count_noun
from .wait_types import ImportWaitMode, PendingImportState
from .wait_view import (
    WaitResult,
)


@final
class ArrRunLoop:
    """The Arr-agnostic run loop driving an injected strategy."""

    def __init__(self, deps: ArrRunDeps, services: ArrRunServices) -> None:
        """Receive the shared collaborators + services hub and set up per-run state."""

        self._config = deps.config
        self._arr_config = deps.arr_config
        self.qbit = deps.qbit
        self.logger = deps.logger
        self._seadex = deps.seadex
        self.cache_store = deps.cache_store
        self._anilist = deps.anilist
        self._skyhook = deps.skyhook
        self._notifier = deps.notifier
        self._reporter = deps.reporter

        self._services = services

        # (Re)set at the top of run_arr. The None placeholder is replaced before any import hook runs.
        # Held under the narrow, non-generic ImportCompleter ABC (a concrete ArrStrategy subclasses it), so no ItemT cast.
        self._active_strategy: ImportCompleter | None = None

        # Adopt the hub's placeholder ctx, never mint a second one.
        # Its dry_run=False and OFF wait mode keep every preview and pending-import path a safe no-op before run_arr.
        self._state = services.state

        self._wait_manager = ImportWaitManager(deps=deps, state=self._state, pending_imports=services.pending_imports)
        self.bind_arr_run(self._state)

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the fresh run context to every per-run collaborator."""

        self._services.bind_arr_run(state)
        self._wait_manager.bind_arr_run(state, self._active_strategy)

    def start_arr_run(
        self,
        dry_run: bool,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
    ) -> None:
        """Start a fresh run context and the run clock, and rebind collaborators."""

        self._state = ArrRunState(
            arr=self._services.arr,
            dry_run=dry_run,
            import_wait_mode=import_wait_mode,
            # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
            started_monotonic=time.monotonic(),
            counts_mark=self._reporter.counts_mark(),
        )
        self.bind_arr_run(self._state)

    @property
    def _wait_active(self) -> bool:
        """Whether the wait machinery runs this pass (an active mode, not preview)."""

        return self._state.import_wait_mode is not ImportWaitMode.OFF and not self._services.is_dry_run()

    # --- Run orchestration --------------------------------------------------

    def run_arr[ItemT: ArrItem](
        self,
        strategy: ArrStrategy[ItemT],
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

        self.start_arr_run(dry_run=dry_run, import_wait_mode=resolved_wait_mode)

        with boot.step(f"Fetching {arr.capitalize()} library") as fetching:
            all_items: list[ItemT] = strategy.get_items()

            if item_id is not None:
                all_items = strategy.filter_to_single(all_items, item_id)

            n_items = len(all_items)
            fetching.note(arr_item_noun(arr, n_items))

        # The activity scan must run before the prefetches, which warm exactly the dirty subset it marks.
        activity: ArrActivityReader | None = None
        if self._config.advanced.detect_arr_activity and not self._config.seadex.ignore_seadex_update_times:
            with boot.step(f"Checking {arr.capitalize()} activity") as step:
                activity = ArrActivityReader(arr, self.cache_store, self.logger)
                scan = activity.read(strategy.history_since)
                dirty: set[int] = set()
                for item in all_items:
                    if scan.rescan_all or item.id in scan.touched:
                        dirty.update(strategy.item_mappings(item, log_ignored=False))
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
                strategy.item_mappings(item, log_ignored=False),
            )

        with boot.step("Fetching AniList metadata") as step:
            fetched = self._anilist.prefetch(prefetch_ids, dry_run=self._services.is_dry_run(), progress=step)
            if self._anilist.unreachable:
                # Terse: the client already warned once with the failure detail.
                step.warn("unavailable")
            else:
                step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Batched OR-filter queries collapse the per-id from_id round-trips. An outage mid-prefetch
        # must not claim "N entries" it never fetched.
        with boot.step("Fetching SeaDex entries") as step:
            fetched = self._seadex.prefetch(prefetch_ids, progress=step)
            if self._seadex.unreachable:
                # Terse: the gateway already warned once with the failure detail.
                step.warn("unreachable")
            else:
                step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Episode lists are warmed fresh every run, never cached: a stale list under-grabs.
        if strategy.warms_episodes:
            with boot.step("Fetching Sonarr episodes") as step:
                warmed = strategy.prefetch_episodes(all_items, progress=step)
                step.note("cached" if warmed == 0 else count_noun(warmed, "series", "series"))
            # Like the episode step, after the SeaDex warm and the activity scan's dirty marks: both pick their
            # series through `title_needs_check`, which reads them.
            with boot.step("Fetching TVDB orders") as step:
                fetched = strategy.prefetch_orders(all_items, progress=step)
                if self._skyhook.unreachable:
                    # Terse: the client already warned once with the failure detail.
                    step.warn("unreachable")
                else:
                    step.note("cached" if fetched == 0 else count_noun(fetched, "series", "series"))
        else:
            strategy.prefetch_episodes(all_items)

        # Tear the cockpit down before the scan logs anything, or it reflows above a stale spinner.
        boot.end_section()

        self._reporter.log_arr_start(arr, n_items)

        # Only on a full run, mirroring the vouch: a single-item run re-checks just its id, so the
        # whole-library note would overstate it and recur every time.
        if item_id is None and self._services.selection_changed and not self._config.seadex.ignore_seadex_update_times:
            hub_note("Matching settings changed - rechecking cached entries")

        for item_idx, item in enumerate(all_items):
            try:
                self._scan_item(strategy, item, item_idx, n_items)
            except Exception as e:
                title = getattr(item, "title", "unknown title")
                hub_error(f"{title}: unexpected error ({e}) - skipping this title", exc=e)
            # Commit after each title, not inside one, where paired writes (a record and its guard row) may be half
            # done. A stop or kill loses only the title in flight. A title that raised still commits what it staged,
            # as the end save always did.
            self.cache_store.save(dry_run=self._services.is_dry_run())

        # ONE full-coverage predicate for both gates below: a checkpoint committed on an outage run would consume
        # drift events the run never acted on. A SkyHook outage counts too: a series it left unrefreshed was read by
        # its old order record, and with neither committed the next pass repeats the re-checks they drive (a settings
        # change, a file change in the arr) against the refreshed one. A title held past the cap, or one whose check
        # didn't cache it as done, holds back neither: `is_unchanged` cleared its stamp, so the next run checks
        # it whatever is committed here. An AniList outage is left out: a Sonarr title whose check read an entry
        # with no stored record of its air dates isn't cached as done, and Radarr reads AniList only for display.
        full_scan = item_id is None and not self._seadex.unreachable and not self._skyhook.unreachable

        # Held on outage so the next healthy run re-derives the same dirty ids (the query overlap and id
        # dedup absorb the replay). The staged write persists only at _finish_arr_run's non-preview save.
        if activity is not None and full_scan:
            activity.stage_checkpoint()

        # A contained per-id error still vouches: otherwise one flaky title re-scans the library every run. A title
        # whose check raised past the cached-entry gate is checked again next run all the same.
        if full_scan:
            self.cache_store.store_selection_hash(arr, self._config.selection_hash())

        # The finalize save persists the checkpoint and vouch with the tail's own writes.
        self._finish_arr_run()

    def _scan_item[ItemT: ArrItem](
        self,
        strategy: ArrStrategy[ItemT],
        item: ItemT,
        item_idx: int,
        n_items: int,
    ) -> None:
        """Scan one library item: every AniList id it maps to, then its pending-record snapshot."""

        arr = self._services.arr
        item_title = item.title
        # The display fallback for every id under this item.
        self._state.arr_title = item_title

        self._reporter.log_arr_item_start(
            arr,
            item_title,
            item_idx + 1,
            n_items,
        )

        if not item.monitored and self._arr_config.ignore_unmonitored:
            self._reporter.log_arr_item_unmonitored(self._state, item_title)
            return

        al_mappings = strategy.item_mappings(item)

        if len(al_mappings) == 0:
            self._reporter.log_no_anilist_mappings(self._state, item_title)
            return

        for al_id, mapping in al_mappings.items():
            # The previous id's pause is slept here, not at the end of that id, so a stop during it never rolls back
            # a title that finished.
            self._services.pay_pause()
            # Past the run cap a title is still checked (a torrent already downloading keeps its mapping), its grab
            # held for the next run.
            try:
                strategy.check_title(item=item, al_id=al_id, mapping=mapping)
            except Exception as e:
                # Contain the failure to THIS AniList id: one bad season must not skip the item's others.
                hub_error(
                    f"{item_title} ({unresolved_label(al_id)}): unexpected error ({e}) - skipping this AniList id",
                    exc=e,
                )
                continue

        # A snapshot, never an import: the end pass owns that. Radarr returns None here.
        # Placed after all of the item's AniList ids so it covers the cached, grabbed and no-entry paths alike.
        if self._wait_active and (sid := strategy.pending_import_series_id(item)) is not None:
            self._wait_manager.report_imports_for_series(sid)

    # --- Wait-for-completion orchestration ----------------------------------

    def _finish_arr_run(self) -> None:
        """Shared run tail: the one-cycle check + tally, print the summary, THEN the waiting monitor."""

        # Close the scan before anything below logs: these diagnostics are run-level facts.
        # Always ScanStarted-paired: run_arr, the only caller, has no early return between log_arr_start and here.
        self._reporter.scan_finished(self._state.arr)

        dry_run = self._services.is_dry_run()

        supports_blocking_wait = self._active_strategy is not None and self._active_strategy.supports_blocking_wait
        end_pass_waits = supports_blocking_wait and self._state.import_wait_mode in (
            ImportWaitMode.BLOCKING,
            ImportWaitMode.HYBRID,
        )

        # The finally guards the whole tail: a raise in check, tally, summary or monitor must not let bootstrap's
        # close roll back the run's staged writes. The save trails the monitor to also capture its drops.
        try:
            if self._wait_active:
                # Heal earlier runs' kept-for-cleanup records before this run's passes read the store.
                self._wait_manager.finish_cleanups()
                if not end_pass_waits:
                    check = self._wait_manager.run_deferred()
                    if check is not None:
                        self._state.stats.imported += check.carried_over_imported
                # This owns the tally's `imported` bump: the snapshot's verified imports land in
                # `pending_import_states`, the passes report theirs on their carried-over result rows.
                self._state.stats.imported += sum(
                    1 for state in self._state.pending_import_states.values() if state is PendingImportState.IMPORTED
                )
                self._wait_manager.tally_carried_over_into_stats()

            self._reporter.log_arr_run_summary(
                self._state,
                dry_run=dry_run,
                has_client=self.qbit is not None,
            )

            if self._wait_active and end_pass_waits:
                result = self._wait_manager.run_blocking()
                if result is not None:
                    self._state.stats.imported += result.carried_over_imported
                    # _notify_wait_complete swallows its own errors, so a bad webhook can never skip the save below.
                    if result.waited:
                        self._notify_wait_complete(result)
        finally:
            self.cache_store.save(dry_run=dry_run)

        # Deliberately outside the finally: on a raise, bootstrap's unwind emits it, so it lands once.
        self._reporter.arr_run_finished(self._state.arr)

    def _notify_wait_complete(self, result: WaitResult) -> None:
        """Push the completion notification, gated on `wait_notify`, swallowing errors."""

        if not self._config.notifications.wait_notify:
            return
        try:
            _ = self._notifier.push_wait_summary(arr=self._state.arr, result=result)
        except Exception as e:
            hub_warn("Wait completion notification failed unexpectedly - the notification was dropped", exc=e)
