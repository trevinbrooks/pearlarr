"""The per-run dependency bundle and the per-AniList-id services hub.

Split out of ``seadex_arr.py``: :class:`RunDeps` is the shared leaf-collaborator
bundle the composition root builds once per arr run, and :class:`RunServices` is
the services hub the Arr strategies hold as ``self._services`` and call the
shared per-id pipeline through. The run loop itself stays in
:class:`~.seadex_arr.SeaDexArr`, which adopts the hub's placeholder context and
pushes each run's fresh context down via :meth:`RunServices.begin_run` - so the
strategies depend on this module only and never see the loop type.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

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
from .log import EntryState, LogFormatter, setup_logger
from .manual_import import ImportWaitMode
from .mappings import MappingEntry, MappingResolver
from .notify import Notifier
from .planner import DownloadPlanner
from .reporter import RunContext, RunReporter, is_preview
from .seadex_filter import SeadexReleaseFilter
from .seadex_gateway import SeaDexGateway, SeaDexSource
from .seadex_types import (
    ArrReleaseDict,
    SeadexDict,
    SonarrEpisode,
)
from .torrents import TorrentService


class QbitConnectionError(Exception):
    """qBittorrent auth/connection failed - a user-facing config problem.

    Raised from ``RunDeps.build`` so the cli reports it as a clean one-line message
    (wrong host / credentials) instead of a stack trace under "unexpected error".
    """


@dataclass(frozen=True)
class RunDeps:
    """The shared leaf collaborators for one Arr run, built once at the root.

    A plain value object the composition root (``cli.py``) builds via
    :meth:`build` and injects into the :class:`RunServices` hub, the
    :class:`~.seadex_arr.SeaDexArr` run loop, and the Arr-specific strategy.
    Keeping construction here (where every collaborator type is already imported)
    and injection at the root means none of them constructs another's
    dependencies - each receives the subset it needs. ``anime_mappings`` /
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
        else:
            # No credentials -> perpetual preview. Say so on the boot ledger (a ⚠
            # step) instead of silently grabbing nothing all run.
            with boot.step("Connecting to qBittorrent") as step:
                step.warn("not configured - preview mode")

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
                logger=logger,
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

    def close(self) -> None:
        """Release run-scoped resources: the HTTP session and the cache db.

        Called once per arr run from ``cli.py``'s ``finally`` (each arr owns its
        own ``CacheStore`` - no sharing - so this never double-closes). The cache
        ``close`` rolls back anything not flushed by the end-of-run save point.
        """
        # self.session is a requests.Session (never None), so it can be closed
        # unconditionally.
        self.session.close()
        self.cache_store.close()


# Deliberately NOT @final (the old engine was): the strategy-seam tests subclass
# this with a scripted fake (_FakeRunServices), so the seam stays overridable.
class RunServices:
    """The per-AniList-id services hub the strategies call.

    Receives its shared collaborators as a :class:`RunDeps` bundle (built and
    injected by the composition root in ``cli.py``) and owns the shared per-id
    pipeline the Arr strategies reach through ``self._services``: the release
    filter, the grab tail, the cache checks, and the strategy-facing log
    delegates. ``arr`` is THE authority for which Arr is being run (``ctx.arr``
    is the per-run copy); the :class:`~.seadex_arr.SeaDexArr` run loop adopts
    the placeholder context minted here and pushes each run's fresh context
    down via :meth:`begin_run`, so the strategies never see the loop type.
    """

    def __init__(self, deps: RunDeps, arr: Arr) -> None:
        """Receive the shared collaborators and set up the per-id pipeline.

        Args:
            deps (RunDeps): The shared collaborators
            arr (Arr): Which Arr is being run. The authority for the run's arr;
                every fresh run context carries a per-run copy of it.
        """

        # Unpack the injected collaborators into the attribute names the per-id
        # service methods read directly. The mapping sources are reached
        # through the shared resolver (self._mappings), which owns them.
        self._config = deps.config
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

        self.arr = arr

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the public_only skip flags, plus the
        # run's dry_run + resolved wait-mode flags) lives on this context, replaced
        # fresh at the start of each run by the loop's reset_run_stats. The single
        # placeholder is minted here - its dry_run=False + OFF wait mode keep every
        # preview / pending-import path a safe no-op - so the object is usable
        # before run_sync; the run loop ADOPTS it (via :attr:`ctx`) at construction.
        self._ctx = RunContext(arr=arr)

        # The shared per-id collaborators, built from the unpacked deps + the
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

    @property
    def ctx(self) -> RunContext:
        """The current run context (the placeholder until a run begins).

        Read by the :class:`~.seadex_arr.SeaDexArr` run loop at construction so
        it adopts the same placeholder instead of minting a second one.
        """

        return self._ctx

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context to the hub and its per-id collaborators.

        Driven by the run loop's ``begin_run``: once with the placeholder at
        construction (so pre-run paths are safe) and again right after
        ``reset_run_stats`` mints the run's real ctx. The wait-manager rebind
        stays on the loop side (the loop owns the manager).
        """

        self._ctx = ctx
        self._filter.begin_run(ctx)
        self._grab_pipeline.begin_run(ctx)

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
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve external Arr ids to a {AniList id -> mapping} dict

        The resolver does the mapping computation and reports which ids it
        dropped (the user's ignore list); the logging stays here so the
        presentation concern doesn't leak into the resolver.

        Args:
            tvdb_id (int | None): TVDB ID
            tmdb_id (int | None): TMDB (movie) ID
            imdb_id (str | None): IMDb ID
            log_ignored (bool): Log a ledger row for each ignored AniList ID.
                Defaults to True; pass False from the prefetch pass so ignored
                ids aren't logged twice (once there, once in the main loop)
        """

        anilist_mappings, ids_to_drop = self._mappings.get_anilist_ids(
            tvdb_id=tvdb_id,
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
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

    def is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return is_preview(self._ctx, self.qbit)

    @property
    def import_wait_mode(self) -> ImportWaitMode:
        """The wait mode resolved for the current run (cli > config > default).

        Set at the top of ``run_sync``; the active strategy reads this (not the
        raw ``config.imports.wait_mode``) so its seed-building gate agrees with the
        run loop's persist/reconcile/blocking gates - otherwise a CLI override that
        turns the feature on over an ``off`` config would build no seeds and the
        whole pass would silently no-op.
        """

        return self._ctx.import_wait_mode

    def _update_cache(
        self,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Merge ``cache_details`` into an entry's cache record (in-memory only).

        The run's save points flush it; see ``CacheStore.update_cache``.

        Args:
            al_id (int): AniList ID
            cache_details (CacheRecord): Details for the cache entry. Defaults
                to None
        """

        self.cache_store.update_cache(self._ctx.arr, al_id, cache_details)

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

        self._log_no_seadex_releases()
        self._update_cache(al_id=al_id, cache_details=cache_details)
        time.sleep(self._config.advanced.sleep_time)
        return False

    def al_id_prologue(self, al_id: int) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry

        Returns the SeaDex entry to process, or None when the id has no SeaDex
        entry - the caller moves to the next id.

        Args:
            al_id (int): AniList id being processed
        """

        # Reset the per-title skip flags (and the skipped group names) before we
        # make any download decisions for this title
        self._ctx.public_only_skipped = False
        self._ctx.public_only_groups = []
        self._ctx.unsupported_tracker_skipped = False
        self._ctx.unsupported_tracker_groups = []
        self._ctx.unsupported_tracker_hashes = []
        self._ctx.stats.checked += 1

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
            self._update_cache(
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

    # --- Presentation seam (strategy-facing) ---------------------------------
    #
    # The run loop logs through the reporter directly (threading its ctx and the
    # preview/client facts so the reporter stays free of orchestrator state). The
    # only log_* methods kept here are the ones the Sonarr/Radarr strategies
    # invoke through their services view; each delegates the same way.

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

    def _log_no_seadex_releases(self) -> bool:
        """Log a no-suitable-releases outcome (delegates to RunReporter)."""
        return self._reporter.log_no_seadex_releases(self._ctx)
