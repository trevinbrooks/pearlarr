"""The per-run dependency bundle and the per-AniList-id services hub.

`RunDeps` is the shared leaf-collaborator bundle the composition root builds
once per arr run. `RunServices` is the services hub the Arr strategies hold
as `self._services` and call the shared per-id pipeline through. The run
loop (`RunLoop`) adopts the hub's placeholder context and pushes each run's
fresh context down via `RunServices.begin_run`, so the strategies depend on
this module only and never see the loop type.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import httpx
import qbittorrentapi
from seadex import EntryRecord, SeaDexEntry

from .anilist_client import AniListClient
from .anilist_gateway import AniListGateway
from .arr_categories import ArrCategoryResolver, CategorySources
from .arr_http import ARR_REQUEST_TIMEOUT_S, ArrHttp, make_arr_client
from .arr_models import ArrFiles, SonarrEpisode
from .arr_run_state import ArrRunState, PerTitleState, Title, is_dry_run, title_from_names
from .boot_flow import BootFlow
from .cache import AbstractCacheStore, CachedTitle, CachedTitleUpdate, CacheStore
from .clock import Clock, SystemClock
from .config import AppConfig, Arr, ArrSettings, PrivateReleaseAction, secret_value
from .grab_pipeline import GrabPipeline, GrabRequest
from .mappings import ExternalIds, MappingResolver, ResolvedMappings, TitleMapping
from .notify import Notifier
from .output import emit_to_hub, hub_counts
from .parse_cache import ParseCacheRows
from .pending_import_store import PendingImportStore
from .planner import GrabPlan, GrabPlanner
from .releases import SeadexDict
from .reporter import RunReporter
from .seadex_filter import SeadexReleaseFilter
from .seadex_gateway import SeaDexGateway, SeaDexMiss, SeaDexSource
from .skyhook_client import SkyHookClient
from .skyhook_gateway import SkyHookGateway
from .torrent_service import TorrentService
from .wait_types import ImportWaitMode


class QbitConnectionError(Exception):
    """qBittorrent auth/connection failed - a user-facing config problem.

    Raised from `RunDeps.build` so the cli reports it as a clean one-line message
    (wrong host / credentials) instead of a stack trace under "unexpected error".
    """


def bind_arr_http(arr: Arr, config: AppConfig, client: httpx.Client) -> ArrHttp:
    """Bind the shared client to `arr`'s configured url + key - the one production spelling.

    Raises `require_connection`'s ValueError when either key is missing - callers
    needing tolerance gate on `config.is_configured(arr)` first. Consumers derive
    tuned handles (retries, heartbeat) via `dataclasses.replace`.
    """

    url, api_key = config.require_connection(arr)
    return ArrHttp.bind(client=client, url=url, api_key=api_key, label=arr.capitalize())


@dataclass(frozen=True)
class ArrRunDeps:
    """The shared leaf collaborators for one Arr run, built once at the root.

    A plain value object the composition root (`bootstrap.py`) builds via
    `build` and injects into the `RunServices` hub, the
    `RunLoop` run loop, and the Arr-specific strategy.
    Keeping construction here (where every collaborator type is already imported)
    and injection at the root means none of them constructs another's
    dependencies - each receives the subset it needs. `anime_mappings` /
    `anidb_mappings` / `anibridge` are read off `mappings` by consumers, not
    stored separately.
    """

    config: AppConfig
    """The shared root config reused by both arrs."""

    arr_config: ArrSettings
    """This arr's connection/behavior submodel, `config.for_arr(arr)`."""

    arr_http: ArrHttp | None
    """This arr's bound transport - None when the connection keys are missing
    (the strategy's `require_connection` error then fires at its construction)."""

    categories: ArrCategoryResolver
    """The run's effective qBittorrent categories: explicit config first, then
    the arr's own download client - fetched lazily at first use (the run's
    first title checked, else an import scan or a post-import move), where
    the arr is provably up. Also says which categories the arr watches."""

    clock: Clock
    """The time seam: the wait passes' clock and the per-title throttle (the run clock reads time directly)."""

    web: httpx.Client
    http: httpx.Client
    qbit: qbittorrentapi.Client | None
    resolver: MappingResolver
    logger: logging.Logger
    seadex: SeaDexSource
    cache_store: AbstractCacheStore
    parse_cache_rows: ParseCacheRows
    """The Sonarr parse-cache leaf over `cache_store`, read by the sweep's parse step."""
    anilist: AniListGateway
    skyhook: SkyHookGateway
    """The TVDB order cache. The Sonarr strategy reads its records, and the run loop its outage."""
    torrents: TorrentService
    notifier: Notifier
    planner: GrabPlanner
    reporter: RunReporter

    @classmethod
    def build(
        cls,
        arr: Arr,
        cache: str = "cache.db",
        *,
        logger: logging.Logger,
        mappings: MappingResolver,
        app_config: AppConfig,
        web: httpx.Client,
        boot: BootFlow,
    ) -> "ArrRunDeps":
        """Construct the shared collaborators in dependency order.

        Args:
            arr: Which Arr is being run. Selects the per-arr config submodel.
            cache: Path to the cache database.
            logger: Logger to use (the CLI builds it before the
                config file can even be read, so config errors are loggable).
            mappings: The id-mapping resolver, built once by the
                CLI and shared across a scheduled Radarr->Sonarr cycle so the three
                large mapping sources are downloaded, parsed and indexed once.
            app_config: The loaded config, read and validated once by
                the CLI per run and shared across a scheduled Radarr->Sonarr cycle.
            web: The shared non-arr web client (tracker scrapes,
                AniList, webhooks), built once by the CLI per cycle and owned
                there - `close` deliberately leaves it open.
            boot: The startup cockpit's producer facade. The
                qBittorrent login and cache open graduate into it as steps
                (a no-op unless a hub renders).
        """

        # `arr_config` is this arr's connection/behavior submodel, injected
        # alongside the shared root.
        arr_config = app_config.for_arr(arr)

        # The httpx client every raw arr endpoint rides (ArrHttp binds it per
        # arr). Pinned timeouts / no-redirects / pool sizing live in its factory.
        # Verification follows THIS arr's knob (the client is per-run, per-arr).
        http = make_arr_client(verify=arr_config.verify_ssl)

        # qbit. None unless host/username/password are all set. With any unset, no
        # client is created and the app treats `qbit is None` as "no client ->
        # perpetual preview".
        qbit: qbittorrentapi.Client | None = None
        credentials = app_config.qbittorrent.credentials()
        if credentials is not None:
            host, username, password = credentials
            # `options` forwards any extra qbittorrentapi.Client kwargs (e.g.
            # VERIFY_WEBUI_CERTIFICATE for a self-signed WebUI). Empty by default.
            # Every request gets the shared Arr timeout so a wedged qBittorrent
            # socket can't hang a poll. A user REQUESTS_ARGS in `options` wins.
            options = dict(app_config.qbittorrent.options)
            options.setdefault("REQUESTS_ARGS", {"timeout": ARR_REQUEST_TIMEOUT_S})
            client = qbittorrentapi.Client(
                host=host,
                username=username,
                password=password,
                **options,
            )
            with boot.step("Connecting to qBittorrent"):
                try:
                    client.auth_log_in()
                except qbittorrentapi.APIConnectionError as e:
                    # LoginFailed (bad credentials) subclasses APIConnectionError, so
                    # this one arm covers both a wrong host and wrong credentials.
                    raise QbitConnectionError(
                        "qBittorrent connection failed - check qbittorrent.host, qbittorrent.username, and "
                        "qbittorrent.password in your config",
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
        # off through cache.db rather than shared memory.
        with boot.step("Opening cache"):
            cache_store = CacheStore.load(
                cache,
                config_hash=app_config.config_hash(),
            )

        # AniList client gateway: owns the in-memory meta cache (al_cache) and the
        # persisted anilist_meta block, over the bound wire client.
        anilist = AniListGateway(
            cache_store=cache_store,
            logger=logger,
            client=AniListClient(client=web),
        )

        # This arr's bound transport. None when the connection keys are missing:
        # build stays raise-free (the strategy's require_connection names the
        # missing key at construction). Bound even without qbit - preview runs
        # still talk to the arr.
        arr_http = bind_arr_http(arr, app_config, http) if app_config.is_configured(arr) else None

        # The category sources exist only when a real qbit does too (a preview
        # run applies no category). A single attempt per use (retries=0): a
        # miss fails open and is retried at the next use.
        sources = (
            CategorySources(replace(arr_http, retries=0), qbit) if qbit is not None and arr_http is not None else None
        )
        categories = ArrCategoryResolver(arr, arr_config, sources)

        # qBittorrent adapter: parses a release URL by tracker and adds it. A None
        # qbit is treated as a perpetual preview.
        torrents = TorrentService(
            qbit=qbit,
            web=web,
            categories=categories,
            tags=app_config.qbittorrent.tags,
            logger=logger,
        )

        return cls(
            config=app_config,
            arr_config=arr_config,
            arr_http=arr_http,
            categories=categories,
            clock=SystemClock(),
            web=web,
            parse_cache_rows=ParseCacheRows(cache_store),
            http=http,
            qbit=qbit,
            resolver=mappings,
            logger=logger,
            # SeaDex API gateway (entry lookups, with connection-error handling)
            seadex=SeaDexGateway(client=SeaDexEntry()),
            cache_store=cache_store,
            anilist=anilist,
            skyhook=SkyHookGateway(client=SkyHookClient(client=web), cache_store=cache_store),
            torrents=torrents,
            # Discord notifier, a no-op when no webhook is configured. The
            # SecretStr urls are unwrapped here, at their point of use.
            notifier=Notifier(
                discord_url=secret_value(app_config.notifications.discord_url),
                webhook_url=secret_value(app_config.notifications.wait_webhook_url),
                web=web,
            ),
            # Download-decision engine: flips each release's download flag.
            planner=GrabPlanner(
                arr=arr,
                interactive=app_config.advanced.interactive,
                logger=logger,
            ),
            # Presentation: every log_* method emits a typed output event through
            # the process hub (resolved at call time), never rendering directly.
            reporter=RunReporter(
                emit=emit_to_hub,
                counts=hub_counts,
                cache_store=cache_store,
                anilist=anilist,
            ),
        )

    def close(self) -> None:
        """Release run-scoped resources: the arr HTTP client and the cache db.

        Called once per arr run from `bootstrap.py`'s `finally` (each arr owns its
        own `CacheStore` - no sharing - so this never double-closes). The cache
        `close` rolls back anything not flushed by a save point.
        `web` is NOT closed here: the CLI owns it across the whole cycle.
        """
        self.http.close()
        self.cache_store.close()


# Deliberately NOT @final: the strategy-seam tests subclass this with a
# scripted fake (_FakeRunServices), so the seam stays overridable.
class ArrRunServices:
    """The per-AniList-id services hub the strategies call.

    Receives its shared collaborators as a `RunDeps` bundle (built and
    injected by the composition root in `bootstrap.py`) and owns the shared per-id
    pipeline the Arr strategies reach through `self._services`: the release
    filter, the grab tail, the cache checks, and the strategy-facing log
    delegates. `arr` is THE authority for which Arr is being run (`ctx.arr`
    is the per-run copy). The `RunLoop` run loop adopts
    the placeholder context minted here and pushes each run's fresh context
    down via `begin_run`, so the strategies never see the loop type.
    """

    def __init__(self, deps: ArrRunDeps, arr: Arr) -> None:
        """Receive the shared collaborators and set up the per-id pipeline.

        `arr` is the authority for the run's arr. Every fresh run context
        carries a per-run copy of it.
        """

        # Unpack the injected collaborators into the attribute names the per-id
        # service methods read directly. The mapping sources are reached
        # through the shared resolver (self._mappings), which owns them.
        self._config = deps.config
        self.qbit = deps.qbit
        self._resolver = deps.resolver
        self.logger = deps.logger
        self._seadex = deps.seadex
        self.cache_store = deps.cache_store
        self._anilist = deps.anilist
        self._torrents = deps.torrents
        self._notifier = deps.notifier
        self._planner = deps.planner
        self._reporter = deps.reporter
        self._clock = deps.clock

        self.arr = arr

        # Whether matching preferences moved since a prior full pass vouched for
        # this arr's cached verdicts. `_unchanged_cached_title` then re-checks every
        # cached entry, and the run loop re-vouches once a full sweep finishes.
        self._selection_changed = deps.cache_store.selection_changed(arr, deps.config.selection_hash())

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the private-only skip flags, plus the
        # run's dry_run + resolved wait-mode flags) lives on this context, replaced
        # fresh at the start of each run by the loop's start_arr_run. The single
        # placeholder is minted here - its dry_run=False + OFF wait mode keep every
        # preview / pending-import path a safe no-op - so the object is usable
        # before run_arr. The run loop ADOPTS it (via `ctx`) at construction.
        self._ctx = ArrRunState(arr=arr)

        # AniList ids whose arr-side files changed since the last pass (fed by the
        # run loop's activity scan). They bypass the cached-entry skip once.
        self._dirty_al_ids: set[int] = set()

        # The shared per-id collaborators, built from the deps hub + the
        # placeholder ctx. bind_arr_run rebinds their ctx at the top of each run.
        self._filter = SeadexReleaseFilter(deps=deps, ctx=self._ctx)
        # The one pending-record seam, shared by the grab pipeline, the strategies, and the wait manager.
        self._pending_imports = PendingImportStore(deps.cache_store)
        self._pending_imports.bind_arr_run(self._ctx)
        self._grab_pipeline = GrabPipeline(deps=deps, ctx=self._ctx, pending_imports=self._pending_imports)

    @property
    def ctx(self) -> ArrRunState:
        """The current run context (the placeholder until a run begins).

        Read by the `RunLoop` run loop at construction so
        it adopts the same placeholder instead of minting a second one.
        """

        return self._ctx

    @property
    def selection_changed(self) -> bool:
        """Whether matching settings changed since this arr's verdicts were vouched.

        Read by the run loop to announce the run-wide re-check
        `_skippable_entry` applies.
        """

        return self._selection_changed

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the fresh run context to the hub and its per-id collaborators.

        Driven by the run loop's `begin_run`: once with the placeholder at
        construction (so pre-run paths are safe) and again right after
        `reset_run_stats` mints the run's real ctx. The wait-manager rebind
        stays on the loop side (the loop owns the manager).
        """

        self._ctx = state
        # Per-run state: the loop's activity scan re-marks dirty ids after this.
        self._dirty_al_ids.clear()
        self._filter.bind_arr_run(state)
        self._grab_pipeline.bind_arr_run(state)
        self._pending_imports.bind_arr_run(state)
        # Radarr imports its own downloads, and so does Sonarr with the wait off.
        self._torrents.bind_arr_run(
            pearlarr_imports=self.arr is Arr.SONARR and state.import_wait_mode is not ImportWaitMode.OFF
        )

    @property
    def pending_imports(self) -> PendingImportStore:
        """The one pending-record seam bound to the current run, shared by the strategies and the wait manager."""

        return self._pending_imports

    def title_needs_check(self, al_id: int) -> bool:
        """Side-effect-free mirror of process_al_id's no-entry + cached_entry_skip gates.

        True iff the per-id loop would actually process this id. Lets
        prefetch_episodes warm only the series the loop won't short-circuit
        (no SeaDex entry, or cached and unchanged), instead of every mapped
        series. No logging / stats / backfill / stamp clear - purely a predicate
        over the warmed SeaDex cache and the entry cache.
        """

        sd_entry = self._seadex.entry(al_id)  # warmed cache, no network
        if isinstance(sd_entry, SeaDexMiss):
            return False
        return self._unchanged_cached_title(al_id, sd_entry) is None

    def _unchanged_cached_title(self, al_id: int, sd_entry: EntryRecord) -> CachedTitle | None:
        """The cached row iff the per-id loop may skip this id. None re-processes.

        The single decision BOTH cache gates share: `al_id_needs_scan` picks
        what prefetch warms and `cached_entry_skip` what the loop skips, and
        the two must agree or un-warmed ids hit AniList one at a time. Run-wide
        bypasses come first (no db read): the ignore flag, a selection-digest
        change (matching preferences moved, so every cached verdict is suspect),
        or a dirty id. Then the SeaDex-timestamp compare, then warn mode re-processes
        fallback-satisfied entries, so their private-only warning resurfaces
        after a switch back from fallback mode.
        """

        if self._config.seadex.ignore_seadex_update_times or self._selection_changed or al_id in self._dirty_al_ids:
            return None
        cached = self.cache_store.get_cached_title(self._ctx.arr, al_id)
        if cached is None or not cached.cached_under(sd_entry.updated_at):
            return None
        if cached.fallback_satisfied and self._config.seadex.private_releases is PrivateReleaseAction.WARN:
            return None
        return cached

    def mark_dirty(self, al_ids: Iterable[int]) -> None:
        """Record AniList ids whose arr-side file state changed since the last pass.

        Fed by the run loop's `ArrActivityMonitor` scan.
        `al_id_needs_scan` and `cached_entry_skip` bypass the cached-entry
        short-circuit for exactly these ids.
        """

        self._dirty_al_ids.update(al_ids)

    def title_mappings(self, ids: ExternalIds, log_ignored: bool = True) -> dict[int, TitleMapping]:
        """Resolve external Arr ids to a {AniList id -> mapping} dict, logging the ignored ones.

        The resolver maps and sets aside the ignore list's entries, and the ledger rows stay here.
        `log_ignored=False` on the prefetch pass keeps an ignored id from being logged twice.
        """

        resolved = self._resolver.resolve(ids)

        # Logged per call, not only on the cache-filling one, so the main loop still logs every ignored id.
        if log_ignored:
            for al_id in resolved.ignored:
                self._reporter.log_ignored_anilist_id(self._ctx, al_id)

        return resolved.kept

    def mappings_with_ignored(self, ids: ExternalIds) -> ResolvedMappings:
        """The entries mapped to the external ids, the ignored ones set apart, for the series' listing read.

        An ignored entry is never processed, but a torrent it lists is still placed under its window, flagged so
        `ordered_windows` runs it after the kept windows of the same kind (season or specials). Logs nothing, since
        `get_anilist_ids` logs the ignored ids.
        """

        return self._resolver.resolve(ids)

    def resolve_title(self, al_id: int) -> Title:
        """Resolve the entry's title through the ladder (resolves only: no logging, no attribution).

        The stored name is deliberately not read: the gateway serves its cached record, and
        `new_cache_details` writes the result back. `log_al_title` opens the block with it.
        """

        return title_from_names(al_id, self._anilist.titles(al_id), self._ctx.arr_title)

    def new_cache_details(self, title: Title, sd_entry: EntryRecord) -> CachedTitleUpdate:
        """The seed record a processed id accumulates into.

        `name` rides only when AniList resolved: `update_cache` merges partially, so an omitted
        name keeps what an earlier run stored instead of clobbering it with a fallback label.
        """

        details: CachedTitleUpdate = {"updated_at": sd_entry.updated_at}
        if title.anilist is not None:
            details["name"] = title.anilist
        return details

    def get_seadex_dict(self, sd_entry: EntryRecord) -> SeadexDict:
        """Parse and filter a SeaDex entry into the run's release dict, each url marked with any cross-seed of it."""

        seadex_dict = self._filter.build(sd_entry)
        self._torrents.mark_cross_seeds(seadex_dict)
        return seadex_dict

    def filter_seadex_interactive(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """Interactively pick which release group(s) to grab (delegates)."""

        return self._filter.interactive_pick(seadex_dict, sd_entry)

    def filter_seadex_downloads(
        self,
        seadex_dict: SeadexDict,
        arr_releases: ArrFiles,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> GrabPlan:
        """Apply the download plan, stamping private-only skips onto ctx (delegates)."""

        return self._filter.filter_downloads(seadex_dict, arr_releases, ep_list)

    def is_dry_run(self) -> bool:
        """A run is a no-op preview (nothing can be grabbed): explicit dry run, or qBittorrent not configured."""
        return is_dry_run(self._ctx, self.qbit)

    @property
    def import_wait_mode(self) -> ImportWaitMode:
        """The wait mode resolved for the current run (cli > config > default).

        Set at the top of `run_sync`. The active strategy reads this (not the
        raw `config.imports.wait_mode`) so its seed-building gate agrees with the
        run loop's persist/reconcile/blocking gates - otherwise a CLI override that
        turns the feature on over an `off` config would build no seeds and the
        whole pass would silently no-op.
        """

        return self._ctx.import_wait_mode

    def _update_cache(
        self,
        al_id: int,
        update: CachedTitleUpdate,
    ) -> None:
        """Merge `cache_details` into an entry's cache record (in-memory only).

        The run's save points flush it. See `CacheStore.update_cache`.
        """

        self.cache_store.update_cached_title(self._ctx.arr, al_id, update)

    def report_no_releases(
        self,
        al_id: int,
        update: CachedTitleUpdate,
        *,
        metadata_unread: bool = False,
    ) -> None:
        """Shared no-suitable-releases tail both Arr strategies fall into.

        When SeaDex yields no usable releases for an id, every strategy does the same three things: log the
        outcome, persist what it knows into the cache, and owe the pacing pause. Hoisted here so the two strategies
        share one definition instead of a byte-for-byte duplicated block. A title whose check read its entry
        without its AniList record (`metadata_unread`) is not cached, so the next run checks it again.
        """

        self._log_no_seadex_releases()
        if not metadata_unread:
            # Never fallback-satisfied: overwrite any stale True from a prior fallback run.
            update["fallback_satisfied"] = False
            self._update_cache(al_id=al_id, update=update)
        self.owe_pause()

    def report_invalid_selection(self) -> None:
        """Shared tail for an interactive pick that left zero valid selections.

        Unlike `no_releases_skip` this deliberately persists NOTHING: caching the title as done would suppress it
        forever, when the input was only fumbled, so it must re-prompt on the next run. The picker already warned
        about the empty selection. This only owes the pacing pause.
        """

        self.owe_pause()

    def owe_pause(self) -> None:
        """Owe the `advanced.sleep_time` pause for this id's queries. `pay_pause` sleeps it before the next id."""

        self._ctx.pause_owed = True

    def pay_pause(self) -> None:
        """Sleep the owed pacing pause, if any, and clear it."""

        if self._ctx.pause_owed:
            self._ctx.pause_owed = False
            self._clock.sleep(self._config.advanced.sleep_time)

    def begin_check(self, al_id: int) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry.

        Returns the entry to process, or None for a missing entry or a SeaDex outage skip.
        The two misses are reported distinctly (an outage skip must never read as "no entry").
        """

        # A fresh PerTitleState clears every flag at once, so none leaks from the prior title.
        self._ctx.per_title = PerTitleState()
        self._ctx.stats.checked += 1

        # Get the SeaDex entry if it exists
        sd_entry = self._seadex.entry(al_id)
        if isinstance(sd_entry, SeaDexMiss):
            if sd_entry is SeaDexMiss.UNREACHABLE:
                self._reporter.log_seadex_unreachable(self._ctx, al_id)
            else:
                self._reporter.log_no_sd_entry(self._ctx, al_id)
            return None

        return sd_entry

    def is_unchanged(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        coverage: Callable[[], str],
    ) -> bool:
        """The cached-entry gate both strategies pass each id through: True skips it, False sends it on to a check.

        A skip backfills what an older row lacks and logs the cached block. A title sent on has its row's stamp
        cleared here, before anything in the check can fail: `grab_and_cache` and `no_releases_skip` are the two
        writes that stamp it again, each only when the outcome may be cached as done.

        `coverage` stays a callable so the (for Sonarr, episode-fetching) lookup runs only on the
        one-time url backfill, never on the common already-backfilled path.
        """

        cached = self._unchanged_cached_title(al_id, sd_entry)
        if cached is None:
            # A forced re-check finds the row already stamped with SeaDex's current update time. Left so, a check
            # that ends in a failed grab, a hold or an error would leave the title skipped as cached from then on.
            self.cache_store.uncache_title(self._ctx.arr, al_id)
            return False

        # One-time backfills merged into a single write: the url + coverage an older record lacks,
        # and the name a record written while AniList was unreachable lacks.
        backfill: CachedTitleUpdate = {}
        if not cached.url:
            backfill["url"] = sd_entry.url
            backfill["coverage"] = coverage()
        if not cached.name and (name := self.resolve_title(al_id).anilist):
            backfill["name"] = name
        if backfill:
            self._update_cache(al_id=al_id, update=backfill)
        self._reporter.log_cached_title(self._ctx, al_id)
        return True

    def grab_and_cache(self, req: GrabRequest) -> None:
        """Shared per-id tail: add torrents, notify, cache the outcome, owe the pacing pause.

        Both strategies build a `GrabRequest` and call this through their
        services. The produce mechanics live on `GrabPipeline`.
        """

        self._grab_pipeline.grab_and_cache(req)
        self.owe_pause()

    # --- Presentation seam (strategy-facing) ---------------------------------
    #
    # The run loop logs through the reporter directly (threading its ctx and the
    # preview/client facts so the reporter stays free of orchestrator state). The
    # only log_* methods kept here are the ones the Sonarr/Radarr strategies
    # invoke through their services view. Each delegates the same way.

    def log_no_episodes(self, title: str) -> None:
        """Log a title whose mapping resolved no episodes (delegates to RunReporter)."""
        self._reporter.log_no_episodes(title)

    def log_anilist_item_unmonitored(self, item_title: str) -> None:
        """Log an unmonitored-item skip (delegates to RunReporter)."""
        self._reporter.log_arr_item_unmonitored(self._ctx, item_title)

    def open_title_block(
        self,
        title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> None:
        """Log the active-entry header (delegates to RunReporter)."""
        self._reporter.open_title_block(
            self._ctx,
            title,
            sd_entry,
            coverage=coverage,
        )

    def _log_no_seadex_releases(self) -> None:
        """Log a no-suitable-releases outcome (delegates to RunReporter)."""
        self._reporter.log_no_seadex_releases(self._ctx)
