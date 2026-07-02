import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

import qbittorrentapi
import requests
from requests.adapters import HTTPAdapter
from seadex import EntryRecord
from urllib3.util.retry import Retry

from .anilist_gateway import AniListGateway
from .boot_view import BootView, NullBootView
from .cache import UPDATED_AT_STR_FORMAT, AbstractCacheStore, CacheRecord, CacheStore
from .config import AppConfig, Arr, ArrSettings
from .grab_pipeline import GrabPipeline, GrabRequest
from .import_wait import ImportWaitManager
from .log import (
    EntryState,
    LogFormatter,
    count_noun,
    setup_logger,
)
from .manual_import import (
    ImportWaitMode,
    resolve_wait_mode,
)
from .mappings import MappingEntry, MappingResolver, TmdbType
from .notify import Notifier
from .planner import DownloadPlanner
from .protocols import ArrSync, ImportCompleter
from .reporter import RunContext, RunReporter, is_preview
from .seadex_filter import SeadexReleaseFilter
from .seadex_gateway import SeaDexGateway, SeaDexSource
from .seadex_types import (
    ArrItem,
    ArrReleaseDict,
    SeadexDict,
    SonarrEpisode,
)
from .torrents import TorrentService
from .wait_view import (
    WaitResult,
)


class QbitConnectionError(Exception):
    """qBittorrent auth/connection failed - a user-facing config problem.

    Raised from ``RunDeps.build`` so the cli reports it as a clean one-line message
    (wrong host / credentials) instead of a stack trace under "unexpected error".
    """


@dataclass(frozen=True)
class RunDeps:
    """The shared leaf collaborators for one Arr run, built once at the root.

    A plain value object the composition root (``cli.py``) builds via
    :meth:`build` and injects into both the :class:`SeaDexArr` run machinery and
    the Arr-specific strategy. Keeping construction here (where every collaborator
    type is already imported) and injection at the root means neither the engine
    nor the strategy constructs the other's dependencies - the engine receives
    these, the strategy receives the subset it needs. ``anime_mappings`` /
    ``anidb_mappings`` / ``anibridge`` are read off ``mappings`` by consumers, not
    stored separately. ``arr_config`` is the per-arr connection/behaviour submodel
    (``config.for_arr(arr)``); ``config`` is the shared root reused by both arrs.
    """

    config: AppConfig
    arr_config: ArrSettings
    session: requests.Session
    qbit: qbittorrentapi.Client | None
    mappings: MappingResolver
    logger: logging.Logger
    seadex: SeaDexSource
    cache_store: AbstractCacheStore
    anilist: AniListGateway
    torrents: TorrentService
    notifier: Notifier
    planner: DownloadPlanner
    log_fmt: LogFormatter
    reporter: RunReporter

    @classmethod
    def build(
        cls,
        arr: Arr,
        config: str = "config.yml",
        cache: str = "cache.db",
        logger: logging.Logger | None = None,
        *,
        mappings: MappingResolver,
        app_config: AppConfig | None = None,
        cache_legacy: str | None = None,
        boot: BootView | None = None,
    ) -> "RunDeps":
        """Construct the shared collaborators in dependency order.

        Args:
            arr (Arr): Which Arr is being run; selects the per-arr config submodel.
            config (str, optional): Path to a config file. Defaults to "config.yml".
            cache (str, optional): Path to the cache database. Defaults to "cache.db".
            logger (logging.Logger | None, optional): Logger to use. Defaults to
                None, which builds one from the config's log level.
            mappings (MappingResolver): The id-mapping resolver, built once by the
                CLI and shared across a scheduled Radarr->Sonarr cycle so the three
                large mapping sources are downloaded, parsed and indexed once.
            app_config (AppConfig | None, optional): A pre-loaded config injected by
                the CLI so a scheduled cycle reads and validates the file once per
                run. Defaults to None, which loads it here.
            cache_legacy (str | None, optional): Path to a legacy ``cache.json`` to
                migrate into ``cache.db`` when no db exists yet. Defaults to None.
            boot (BootView | None, optional): The startup cockpit; the qBittorrent
                login and cache open graduate into it as steps. Defaults to None (a
                no-op view), so the standalone path runs without a cockpit.
        """

        boot = boot if boot is not None else NullBootView()

        # Load, validate, and expose the config file as typed settings. AppConfig
        # owns the file lifecycle (copy-template-if-missing, parse, validate) and is
        # the single source of truth for every setting. The CLI may inject an
        # already-loaded config (one read shared across the Radarr->Sonarr cycle);
        # otherwise it's loaded here for the standalone path. ``arr_config`` is this
        # arr's connection/behaviour submodel, injected alongside the shared root.
        app_config = AppConfig.load(config) if app_config is None else app_config
        arr_config = app_config.for_arr(arr)

        if logger is None:
            # Standalone path (the CLI always injects a logger): logs live in the
            # data dir alongside the cache it was handed (unified layout).
            log_dir = os.path.join(os.path.dirname(os.path.abspath(cache)), "logs")
            logger = setup_logger(log_level=app_config.advanced.log_level, log_dir=log_dir)

        # Shared keep-alive session for the raw Sonarr/Radarr calls. Retries
        # transient failures on idempotent GETs only (POSTs never retry, so a
        # command can't double-fire); pool_maxsize must stay >= the sweep's fetch
        # concurrency (SONARR_FETCH_WORKERS) so parallel GETs don't queue.
        # raise_on_status=False lets a still-5xx response return for the callers'
        # status checks. Per-request timeouts are at the call sites.
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # qbit. None unless host/username/password are all set; with any unset, no
        # client is created and the app treats `qbit is None` as "no client ->
        # perpetual preview".
        qbit: qbittorrentapi.Client | None = None
        credentials = app_config.qbittorrent.credentials()
        if credentials is not None:
            host, username, password = credentials
            # `options` forwards any extra qbittorrentapi.Client kwargs (e.g.
            # VERIFY_WEBUI_CERTIFICATE for a self-signed WebUI); empty by default.
            client = qbittorrentapi.Client(
                host=host,
                username=username,
                password=password,
                **app_config.qbittorrent.options,
            )
            with boot.step("Connecting to qBittorrent"):
                try:
                    client.auth_log_in()
                except qbittorrentapi.APIConnectionError as e:
                    # LoginFailed (bad credentials) subclasses APIConnectionError, so
                    # this one arm covers both a wrong host and wrong credentials.
                    raise QbitConnectionError(
                        "qBittorrent connection failed - check the qbittorrent host and credentials in your config",
                    ) from e
            qbit = client

        # Load the cache (or create its schema) and reconcile the descriptor against
        # the current package version + config checksum. Each arr builds its own
        # store that reads the file fresh, so a scheduled Radarr->Sonarr cycle hands
        # off through cache.db rather than shared memory. A legacy cache.json (if
        # present and there's no db yet) is migrated on the first real save.
        with boot.step("Opening cache"):
            cache_store = CacheStore.load(
                cache,
                config_checksum=app_config.checksum(),
                migrate_from=cache_legacy,
                logger=logger,
            )

        # AniList client gateway: owns the in-memory meta cache (al_cache) and the
        # persisted anilist_meta block.
        anilist = AniListGateway(cache_store=cache_store, logger=logger)

        # qBittorrent adapter: parses a release URL by tracker and adds it. A None
        # qbit is treated as a perpetual preview.
        torrents = TorrentService(
            qbit=qbit,
            session=session,
            category=arr_config.torrent_category,
            tags=app_config.qbittorrent.tags,
            logger=logger,
        )

        # All aligned detail rendering goes through this formatter.
        log_fmt = LogFormatter(logger)

        return cls(
            config=app_config,
            arr_config=arr_config,
            session=session,
            qbit=qbit,
            mappings=mappings,
            logger=logger,
            # SeaDex API gateway (entry lookups, with connection-error handling)
            seadex=SeaDexGateway(logger=logger),
            cache_store=cache_store,
            anilist=anilist,
            torrents=torrents,
            # Discord notifier; a no-op when no webhook is configured.
            notifier=Notifier(
                discord_url=app_config.notifications.discord_url,
                webhook_url=app_config.notifications.wait_webhook_url,
            ),
            # Download-decision engine: flips each release's download flag.
            planner=DownloadPlanner(
                public_only=app_config.seadex.public_only,
                interactive=app_config.advanced.interactive,
                use_torrent_hash_to_filter=app_config.seadex.use_torrent_hash_to_filter,
                logger=logger,
            ),
            log_fmt=log_fmt,
            # Presentation: owns every log_* method and the end-of-run summary.
            reporter=RunReporter(
                logger=logger,
                log_fmt=log_fmt,
                cache_store=cache_store,
                anilist=anilist,
            ),
        )


@final
class SeaDexArr:
    """The Arr-agnostic run machinery driving an injected strategy.

    Receives its shared collaborators as a :class:`RunDeps` bundle (built and
    injected by the composition root in ``cli.py``) and owns the run loop, the
    per-run :class:`RunContext`, and the shared per-id pipeline. It drives an
    injected :class:`~.protocols.ArrSync` strategy (passed to :meth:`run_sync`)
    for the Arr-specific pieces; the strategy holds *this* object as its
    ``services`` and calls the pipeline through it. The engine never holds the
    strategy and never constructs its own dependencies.
    """

    def __init__(self, deps: RunDeps, arr: Arr = Arr.SONARR) -> None:
        """Receive the shared collaborators and set up per-run state.

        Args:
            deps (RunDeps): The shared collaborators
            arr (Arr, optional): Which Arr is being run. Defaults to Arr.SONARR.
        """

        # Unpack the injected collaborators into the attribute names the run loop
        # and pipeline methods read directly. The mapping sources are reached
        # through the shared resolver (self._mappings), which owns them.
        self._config = deps.config
        self._arr_config = deps.arr_config
        self.session = deps.session
        self.qbit = deps.qbit
        self._mappings = deps.mappings
        self.logger = deps.logger
        self._seadex = deps.seadex
        self.cache_store = deps.cache_store
        self._anilist = deps.anilist
        self._torrents = deps.torrents
        self._notifier = deps.notifier
        self._planner = deps.planner
        self.log_fmt = deps.log_fmt
        self._reporter = deps.reporter

        # The active strategy for the current run, (re)set at the top of run_sync;
        # the placeholder None here is replaced before any import hook is invoked.
        self._active_strategy: ImportCompleter | None = None

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the public_only skip flags, plus the
        # run's dry_run + resolved wait-mode flags) lives on this context, replaced
        # fresh at the start of each run by reset_run_stats. A placeholder is built
        # here - its dry_run=False + OFF wait mode keep every preview / pending-
        # import path a safe no-op - so the object is usable before run_sync.
        self._ctx = RunContext(arr=arr)

        # Engine-internal collaborators, built from the unpacked deps + the
        # placeholder ctx. begin_run rebinds their ctx at the top of each run.
        self._filter = SeadexReleaseFilter(
            config=self._config,
            planner=self._planner,
            cache_store=self.cache_store,
            logger=self.logger,
            log_fmt=self.log_fmt,
            ctx=self._ctx,
        )
        self._grab_pipeline = GrabPipeline(
            config=self._config,
            planner=self._planner,
            cache_store=self.cache_store,
            torrents=self._torrents,
            anilist=self._anilist,
            notifier=self._notifier,
            reporter=self._reporter,
            log_fmt=self.log_fmt,
            qbit=self.qbit,
            ctx=self._ctx,
        )
        self._wait_manager = ImportWaitManager(
            config=self._config,
            cache_store=self.cache_store,
            reporter=self._reporter,
            logger=self.logger,
            qbit=self.qbit,
            ctx=self._ctx,
            strategy=self._active_strategy,
        )
        self.begin_run(self._ctx)

    def close(self) -> None:
        """Release run-scoped resources: the HTTP session and the cache db.

        Called once per arr run from ``cli.py``'s ``finally`` (each arr owns its
        own ``CacheStore`` - no sharing - so this never double-closes). The cache
        ``close`` rolls back anything not flushed by the end-of-run save point.
        """
        # self.session is the injected RunDeps.session (a requests.Session, never
        # None), so it can be closed unconditionally.
        self.session.close()
        self.cache_store.close()

    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """Whether the cached entry matches SeaDex's last-updated timestamp."""

        return self.cache_store.check_al_id_in_cache(arr, al_id, seadex_entry)

    def al_id_needs_scan(self, al_id: int) -> bool:
        """Side-effect-free mirror of process_al_id's no-entry + cached_entry_skip
        gates: True iff the per-id loop would actually process this id.

        Lets prefetch_episodes warm only the series the loop won't short-circuit
        (no SeaDex entry, or cached and unchanged), instead of every mapped
        series. No logging / stats / backfill - purely a predicate over the
        warmed SeaDex cache and the entry cache.
        """

        sd_entry = self._seadex.entry(al_id)  # warmed cache, no network
        if sd_entry is None:
            return False
        if self._config.seadex.ignore_seadex_update_times:
            return True
        return not self.cache_store.check_al_id_in_cache(self._ctx.arr, al_id, sd_entry)

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: TmdbType = TmdbType.MOVIE,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve external Arr ids to a {AniList id -> mapping} dict

        The resolver does the mapping computation and reports which ids it
        dropped (the user's ignore list); the logging stays here so the
        presentation concern doesn't leak into the resolver.

        Args:
            tvdb_id (int | None): TVDB ID
            tmdb_id (int | None): TMDB ID
            imdb_id (str | None): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.
            log_ignored (bool): Log a ledger row for each ignored AniList ID.
                Defaults to True; pass False from the prefetch pass so ignored
                ids aren't logged twice (once there, once in the main loop)
        """

        anilist_mappings, ids_to_drop = self._mappings.get_anilist_ids(
            tvdb_id=tvdb_id,
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
            tmdb_type=tmdb_type,
        )

        # Log ignored ids per-call (not just on the cache-filling call), so the
        # main loop still logs every ignored id even after the prefetch pass ran
        if log_ignored:
            for al_id in ids_to_drop:
                self._reporter.log_ignored_anilist_id(al_id)

        return anilist_mappings

    def get_anilist_title(
        self,
        al_id: int,
    ) -> str:
        """Resolve and remember the AniList title for an ID (no logging)

        The gateway resolves the raw title (no side-effects); the empty-result
        fallback and the transitional ``current_title`` attribution live here so
        later steps can attribute grabs to the active entry. The entry header is
        logged separately by log_al_title, once episodes are known.

        Args:
            al_id (int): AniList ID
        """

        anilist_title = self._anilist.title(al_id)

        # If the lookup came back empty (e.g., AniList was rate-limiting even
        # after retries), fall back to the id so the entry is still identifiable
        # rather than showing "None"
        if not anilist_title:
            anilist_title = f"AniList #{al_id}"

        self._ctx.current_title = anilist_title

        return anilist_title

    def get_seadex_dict(self, sd_entry: EntryRecord) -> SeadexDict:
        """Parse and filter a SeaDex entry into the run's release dict (delegates)."""

        return self._filter.build(sd_entry)

    def filter_seadex_interactive(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """Interactively pick which release group(s) to grab (delegates)."""

        return self._filter.interactive_pick(seadex_dict, sd_entry)

    def filter_seadex_downloads(
        self,
        al_id: int,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> tuple[list[str | None], SeadexDict]:
        """Apply the download plan, stamping public_only skips onto ctx (delegates)."""

        return self._filter.filter_downloads(al_id, seadex_dict, arr_release_dict, ep_list)

    def _is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return is_preview(self._ctx, self.qbit)

    @property
    def import_wait_mode(self) -> ImportWaitMode:
        """The wait mode resolved for the current run (cli > config > default).

        Set at the top of ``run_sync``; the active strategy reads this (not the
        raw ``config.imports.wait_mode``) so its seed-building gate agrees with the
        engine's persist/reconcile/blocking gates - otherwise a CLI override that
        turns the feature on over an ``off`` config would build no seeds and the
        whole pass would silently no-op.
        """

        return self._ctx.import_wait_mode

    def update_cache(
        self,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> bool:
        """Merge ``cache_details`` into an entry's cache record (in-memory only).

        The run's save points flush it; see ``CacheStore.update_cache``.

        Args:
            al_id (int): AniList ID
            cache_details (CacheRecord): Details for the cache entry. Defaults
                to None
        """

        return self.cache_store.update_cache(self._ctx.arr, al_id, cache_details)

    def no_releases_skip(
        self,
        al_id: int,
        cache_details: CacheRecord,
    ) -> bool:
        """Shared no-suitable-releases tail both Arr strategies fall into.

        When SeaDex yields no usable releases for an id, every strategy does the
        same four things: log the outcome, persist what it knows into the cache,
        throttle, and report "not grabbed". Hoisted here so the two strategies
        share one definition instead of a byte-for-byte duplicated block.

        Args:
            al_id (int): AniList ID.
            cache_details (CacheRecord): Cache record assembled for this id.

        Returns:
            bool: Always ``False`` (nothing was grabbed).
        """

        self.log_no_seadex_releases()
        self.update_cache(al_id=al_id, cache_details=cache_details)
        time.sleep(self._config.advanced.sleep_time)
        return False

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context to the engine's per-run collaborators.

        Two-phase bind: called once with the placeholder ctx in ``__init__`` (so
        pre-run paths are safe) and again from ``run_sync`` right after
        ``reset_run_stats`` swaps in the run's real ctx, so every collaborator
        rebinds to the fresh ctx.
        """

        self._filter.begin_run(ctx)
        self._grab_pipeline.begin_run(ctx)
        self._wait_manager.begin_run(ctx, self._active_strategy)

    def reset_run_stats(
        self,
        arr: Arr,
        dry_run: bool,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
    ) -> bool:
        """Start a fresh run context and the run clock, and rebind collaborators

        Replaces the run-scoped state wholesale with a new RunContext and
        snapshots the logger-level counter (warning/error counts are diffed
        against this when the summary is logged). The ``begin_run`` rebind is
        folded in here so the ctx swap and the collaborator rebind can never
        drift apart - a missed rebind would silently route a collaborator's
        writes to the orphaned prior context.

        Args:
            arr (Arr): Which Arr is being run.
            dry_run (bool): Whether this run simulates without grabbing/writing.
            import_wait_mode (ImportWaitMode): The run's resolved wait mode
                (cli > config > default), stamped onto the fresh context.
        """

        counter = getattr(self.logger, "seadex_counter", None)
        self._ctx = RunContext(
            arr=arr,
            dry_run=dry_run,
            import_wait_mode=import_wait_mode,
            # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
            started_monotonic=time.monotonic(),
            log_counts_at_start=counter.snapshot() if counter else {},
        )
        self.begin_run(self._ctx)

        return True

    # --- Run orchestration (shared machinery) -------------------------------
    #
    # run_sync is the shared scaffolding both Arrs use (reset stats, fetch items,
    # optional single-id filter, AniList prefetch, the per-item loop, and the
    # end-of-run save + summary). The Arr-specific pieces are the injected
    # strategy's hooks (get_items, filter_to_single, item_anilist_ids,
    # process_al_id); the strategy holds this object as its services and calls
    # the shared per-id head/tail (al_id_prologue / cached_entry_skip /
    # grab_and_cache) through it.

    def run_sync[ItemT: ArrItem](
        self,
        strategy: ArrSync[ItemT],
        *,
        arr: Arr,
        item_id: int | None,
        dry_run: bool,
        import_wait_mode: ImportWaitMode | None = None,
        boot: BootView | None = None,
    ) -> bool:
        """Shared run scaffolding for both Arr syncers

        Generic in ``ItemT`` (the strategy's item protocol), so the body sees a
        precise ``list[ItemT]`` / ``item: ItemT`` - the same concrete type the
        strategy's hooks consume and produce - rather than the loose ``Any`` the
        run machinery used to carry. ``ArrSync`` is invariant in its item type, so
        a concrete (non-union) strategy must reach this call: the composition root
        (``cli.py``) branches per Arr so each call binds one ``ItemT`` cleanly.

        Args:
            strategy (ArrSync[ItemT]): The Arr-specific strategy to drive (injected
                by the composition root, which picks Sonarr/Radarr at runtime). It
                already holds this object as its services, so its hooks are
                called without passing self.
            arr (Arr): Which Arr is being run
            item_id (int | None): If set, only run for the single item with this
                id (TMDB for Radarr, TVDB for Sonarr)
            dry_run (bool): Simulate the run without grabbing torrents, writing
                the cache, or sending notifications
            import_wait_mode (ImportWaitMode | None): The CLI ``--import-wait-mode``
                override, resolved cli > config > default. None falls back to the
                configured ``import_wait_mode``.
            boot (BootView | None): The startup cockpit; the library fetch and the
                metadata prefetch graduate into it as steps, and it is torn down
                right before the per-item scan begins. Defaults to None (no-op view).
        """

        boot = boot if boot is not None else NullBootView()

        # Hold the active strategy (so _finalize_run / _grab can call its import
        # hook) and resolve the effective wait mode (cli > config > default) for
        # the whole run. The engine only ever calls import_completed off it, so it
        # is held under the narrow, non-generic ImportCompleter protocol - which a
        # concrete ArrSync structurally satisfies, so no invariant-ItemT cast.
        self._active_strategy = strategy
        resolved_wait_mode = resolve_wait_mode(
            import_wait_mode,
            self._config.imports.wait_mode,
        )

        # Start a fresh run context (stats + clock + counter snapshot + the run's
        # dry_run / wait-mode flags); reset_run_stats rebinds the collaborators to it.
        self.reset_run_stats(arr=arr, dry_run=dry_run, import_wait_mode=resolved_wait_mode)

        # Tend the durable pending-import store at run start (never on a preview,
        # since waiting/importing needs a real qBittorrent client). The TTL prune
        # runs for EVERY active mode - including pure blocking - so aged-out records
        # can't accumulate forever. The reconcile/snapshot/monitor that actually
        # report and import carried-over records run AFTER the per-item loop (the
        # inline per-series snapshot) and in _finalize_run (deferred reconcile +
        # the post-summary blocking monitor), never before the banner.
        if self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._is_preview():
            self._wait_manager.prune_expired_pending()

        # Fetch the library (the long pre-scan network wait) inside the cockpit so
        # the spinner animates through it; the count graduates as the step's detail.
        with boot.step(f"Connecting to {arr.capitalize()}") as connecting:
            all_items: list[ItemT] = strategy.get_items()

            # If we're targeting a single item, filter down to it
            if item_id is not None:
                all_items = strategy.filter_to_single(all_items, item_id)

            n_items = len(all_items)
            connecting.note(
                count_noun(n_items, "movie") if arr is Arr.RADARR else count_noun(n_items, "series", "series"),
            )

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
            fetched = self._anilist.prefetch(prefetch_ids, preview=self._is_preview(), progress=step)
            step.note("cached" if fetched == 0 else count_noun(fetched, "entry", "entries"))

        # Bulk-fetch SeaDex entries for the same ids in batched OR-filter queries,
        # collapsing the per-id from_id round-trips (one per library id, just to read
        # updated_at) into a handful. entry() then serves from this warmed cache.
        with boot.step("Fetching SeaDex entries") as step:
            fetched = self._seadex.prefetch(prefetch_ids, progress=step)
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
                    # imports this run's records before the save + summary. The
                    # original per-item post-loop max check is redundant with this
                    # (the in-block check fires after every add, so torrents_added
                    # can't reach the cap without process_al_id stopping first).
                    try:
                        if strategy.process_al_id(
                            item=item,
                            item_title=item_title,
                            al_id=al_id,
                            mapping=mapping,
                        ):
                            cap_reached = True
                            break
                    except Exception as e:
                        # Contain a per-id failure to THIS AniList id: a transient error
                        # on one season must not skip the item's other seasons.
                        self.logger.error(
                            f"{item_title} (anilist {al_id}): unexpected error: {e}",
                            exc_info=True,
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
                if sid is not None and self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._is_preview():
                    self._wait_manager.snapshot_pending_for_series(sid)

            except Exception as e:
                title = getattr(item, "title", "unknown title")
                self.logger.error(
                    f"{title}: unexpected error: {e}",
                    exc_info=True,
                )
                continue

        # Run the end-of-run blocking pass (blocking/hybrid only), then persist
        # the run and log the summary. Per-title update_cache calls only mutate
        # memory, so this finalize is what actually saves (and sorts by id).
        self._finalize_run()

        return True

    def al_id_prologue(self, al_id: int | None) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry

        Returns the SeaDex entry to process, or None when the id should be
        skipped (no id, or no SeaDex entry) - the caller moves to the next id.

        Args:
            al_id (int | None): AniList id being processed; defensively None-checked
                since the mapping dicts are built from external data
        """

        # Reset the per-title skip flags (and the skipped group names) before we
        # make any download decisions for this title
        self._ctx.public_only_skipped = False
        self._ctx.public_only_groups = []
        self._ctx.unsupported_tracker_skipped = False
        self._ctx.unsupported_tracker_groups = []
        self._ctx.stats.checked += 1

        if al_id is None:
            self._reporter.log_no_anilist_id()
            return None

        # Get the SeaDex entry if it exists
        sd_entry = self._seadex.entry(al_id)
        if sd_entry is None:
            self._reporter.log_no_sd_entry(self._ctx, al_id)
            return None

        return sd_entry

    def cached_entry_skip(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        sd_url: str,
        coverage: Callable[[], str],
    ) -> bool:
        """Shared cached-entry short-circuit for both Arr runners

        When the id is already cached and we're honoring SeaDex update times,
        backfill the url + coverage on legacy records that predate those fields,
        log the cached entry, and return True so the caller skips it. ``coverage``
        is a zero-arg callable so the (for Sonarr, episode-fetching) coverage
        lookup runs only on the one-time backfill, never on the common
        already-backfilled path.

        Args:
            al_id (int): AniList id being processed
            sd_entry (EntryRecord): Resolved SeaDex entry
            sd_url (str): SeaDex entry URL stored on the backfilled record
            coverage (Callable[[], str]): Lazily builds the coverage string for
                the backfill ("" for a movie, a season/episode range for a series)
        """

        # One read of the whole row serves both the freshness check and the
        # url-backfill check below (was a SELECT updated_at + a SELECT url).
        entry = self.cache_store.get_entry(self._ctx.arr, al_id)
        # Mirrors check_al_id_in_cache: absent, or a timestamp that no longer matches
        # SeaDex's, means re-process (don't skip).
        if entry is None or entry.updated_at != sd_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT):
            return False
        if self._config.seadex.ignore_seadex_update_times:
            return False

        # Backfill the enriched fields for records written before they existed,
        # so cached rows can still link to SeaDex (and, for series, show the
        # season/episode coverage). One-time per old entry.
        if not entry.url:
            self.update_cache(
                al_id=al_id,
                cache_details={"url": sd_url, "coverage": coverage()},
            )
        self._reporter.log_cached_entry(self._ctx, self._ctx.arr, al_id)
        return True

    def grab_and_cache(self, req: GrabRequest) -> bool:
        """Shared per-id tail: add torrents, notify, cache the outcome (delegates).

        Both strategies build a :class:`GrabRequest` and call this through their
        services; the produce mechanics live on
        :class:`~.grab_pipeline.GrabPipeline`. Returns True only when
        max_torrents_to_add was reached (the caller stops the whole run).
        """

        return self._grab_pipeline.grab_and_cache(req)

    # --- Wait-for-completion orchestration ----------------------------------
    #
    # The completion wait/poll machinery lives on ``self._wait_manager``
    # (:class:`~.import_wait.ImportWaitManager`); the engine keeps the run tail
    # (``_finalize_run``) that drives its passes in order plus the walk-away
    # completion notification. Every path is a no-op under preview (no client).

    def _finalize_run(self) -> None:
        """Shared run tail: reconcile + tally, print the summary, THEN block.

        The ordering corrects the old "exited right away" + detached-tally
        behaviour. In order:

          1. deferred-mode pre-summary reconcile of any carried-over records not
             already snapshotted inline (non-blocking; feeds the counters);
          2. fold every still-pending carried-over record into the
             ``queued``/``importing``/``imported`` counters (this-run grabs stay
             ``added``);
          3. print the scoreboard - so the summary reflects the pre-monitor state
             and never reports completion for this-run grabs;
          4. ONLY for blocking/hybrid, run the interleaved monitor + live region
             dead last, after the summary, so the wait/import is the live report;
          5. save the cache last, so the store reflects both the inline-snapshot
             and the monitor drops.

        Every wait/import path is skipped on a preview (no client / dry run).
        """

        preview = self._is_preview()
        active = self._ctx.import_wait_mode is not ImportWaitMode.OFF and not preview

        if active and self._ctx.import_wait_mode is ImportWaitMode.DEFERRED:
            self._wait_manager.reconcile_remaining()
        if active:
            self._wait_manager.tally_carried_over_into_stats()

        self._reporter.log_run_summary(
            self._ctx,
            self._ctx.arr,
            is_preview=preview,
            has_client=self.qbit is not None,
            import_wait_mode=self._ctx.import_wait_mode,
        )

        # The monitor is the only post-summary step that mutates the store
        # (dropping records it imports); guard the save in a finally so an
        # unexpected monitor error can't lose this run's grabs or drops from the
        # durable cache (the old order saved before the wait pass for the same
        # reason - here the save runs after, to also capture the monitor's drops).
        try:
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

    def _notify_wait_complete(self, result: WaitResult) -> None:
        """Push the completion notification, gated on ``wait_notify``; swallow errors."""

        if not self._config.notifications.wait_notify:
            return
        try:
            _ = self._notifier.push_wait_summary(arr=self._ctx.arr, result=result)
        except Exception:
            self.logger.debug("wait completion notification failed", exc_info=True)

    # --- Presentation seam (strategy-facing) ---------------------------------
    #
    # The engine logs through ``self._reporter`` directly (threading ``self._ctx``
    # and the preview/client facts so the reporter stays free of orchestrator
    # state). The only log_* methods kept here are the ones the Sonarr/Radarr
    # strategies invoke through their services view; each delegates the same way.

    def log_entry_status(
        self,
        state: EntryState,
        label: str,
        style: str | None = "grey50",
    ) -> bool:
        """Log a one-line entry status row (delegates to RunReporter)."""
        return self._reporter.log_entry_status(state, label, style=style)

    def log_anilist_item_unmonitored(self, item_title: str) -> bool:
        """Log an unmonitored-item skip (delegates to RunReporter)."""
        return self._reporter.log_arr_item_unmonitored(self._ctx, item_title)

    def log_al_title(
        self,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> bool:
        """Log the active-entry header (delegates to RunReporter)."""
        return self._reporter.log_al_title(
            self._ctx,
            anilist_title,
            sd_entry,
            coverage=coverage,
        )

    def log_cached_entry(
        self,
        arr: Arr,
        al_id: int,
        state: EntryState = EntryState.UNCHANGED,
    ) -> bool:
        """Log a cached entry (delegates to RunReporter)."""
        return self._reporter.log_cached_entry(self._ctx, arr, al_id, state=state)

    def log_no_seadex_releases(self) -> bool:
        """Log a no-suitable-releases outcome (delegates to RunReporter)."""
        return self._reporter.log_no_seadex_releases(self._ctx)
