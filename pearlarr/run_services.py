"""The per-run dependency bundle and the per-AniList-id services hub.

Split out of `run_loop.py`: `RunDeps` is the shared leaf-collaborator
bundle the composition root builds once per arr run, and `RunServices` is
the services hub the Arr strategies hold as `self._services` and call the
shared per-id pipeline through. The run loop itself stays in
`RunLoop`, which adopts the hub's placeholder context and
pushes each run's fresh context down via `RunServices.begin_run` - so the
strategies depend on this module only and never see the loop type.
"""

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import httpx
import qbittorrentapi
from seadex import EntryRecord, SeaDexEntry

from .anilist_client import AniListClient
from .anilist_gateway import AniListGateway
from .arr_http import make_httpx_client
from .boot_flow import BootFlow
from .cache import UPDATED_AT_STR_FORMAT, AbstractCacheStore, CachedEntry, CacheRecord, CacheStore
from .config import AppConfig, Arr, ArrSettings, PrivateReleaseAction, secret_value
from .grab_pipeline import GrabPipeline, GrabRequest
from .log import EntryState
from .manual_import import ImportWaitMode
from .mappings import ExternalIds, MappingEntry, MappingResolver
from .notify import Notifier
from .output import emit_to_hub, hub_counts
from .planner import DownloadPlanner
from .reporter import RunContext, RunReporter, is_preview
from .seadex_filter import FilterResult, SeadexReleaseFilter
from .seadex_gateway import SeaDexGateway, SeaDexMiss, SeaDexSource
from .seadex_types import (
    ARR_REQUEST_TIMEOUT_S,
    ArrReleaseDict,
    SeadexDict,
    SonarrEpisode,
)
from .torrents import TorrentService


class QbitConnectionError(Exception):
    """qBittorrent auth/connection failed - a user-facing config problem.

    Raised from `RunDeps.build` so the cli reports it as a clean one-line message
    (wrong host / credentials) instead of a stack trace under "unexpected error".
    """


@dataclass(frozen=True)
class RunDeps:
    """The shared leaf collaborators for one Arr run, built once at the root.

    A plain value object the composition root (`bootstrap.py`) builds via
    `build` and injects into the `RunServices` hub, the
    `RunLoop` run loop, and the Arr-specific strategy.
    Keeping construction here (where every collaborator type is already imported)
    and injection at the root means none of them constructs another's
    dependencies - each receives the subset it needs. `anime_mappings` /
    `anidb_mappings` / `anibridge` are read off `mappings` by consumers, not
    stored separately. `arr_config` is the per-arr connection/behavior submodel
    (`config.for_arr(arr)`); `config` is the shared root reused by both arrs.
    """

    config: AppConfig
    arr_config: ArrSettings
    web: httpx.Client
    http: httpx.Client
    qbit: qbittorrentapi.Client | None
    mappings: MappingResolver
    logger: logging.Logger
    seadex: SeaDexSource
    cache_store: AbstractCacheStore
    anilist: AniListGateway
    torrents: TorrentService
    notifier: Notifier
    planner: DownloadPlanner
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
    ) -> "RunDeps":
        """Construct the shared collaborators in dependency order.

        Args:
            arr: Which Arr is being run; selects the per-arr config submodel.
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
            boot: The startup cockpit's producer facade; the
                qBittorrent login and cache open graduate into it as steps
                (a no-op unless a hub renders).
        """

        # `arr_config` is this arr's connection/behavior submodel, injected
        # alongside the shared root.
        arr_config = app_config.for_arr(arr)

        # The httpx client every raw arr endpoint rides (ArrHttp binds it per
        # arr); pinned timeouts / no-redirects / pool sizing live in its factory.
        # Verification follows THIS arr's knob (the client is per-run, per-arr).
        http = make_httpx_client(verify=arr_config.verify_ssl)

        # qbit. None unless host/username/password are all set; with any unset, no
        # client is created and the app treats `qbit is None` as "no client ->
        # perpetual preview".
        qbit: qbittorrentapi.Client | None = None
        credentials = app_config.qbittorrent.credentials()
        if credentials is not None:
            host, username, password = credentials
            # `options` forwards any extra qbittorrentapi.Client kwargs (e.g.
            # VERIFY_WEBUI_CERTIFICATE for a self-signed WebUI); empty by default.
            # Every request gets the shared Arr timeout so a wedged qBittorrent
            # socket can't hang a poll; a user REQUESTS_ARGS in `options` wins.
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
                config_checksum=app_config.checksum(),
            )

        # AniList client gateway: owns the in-memory meta cache (al_cache) and the
        # persisted anilist_meta block, over the bound wire client.
        anilist = AniListGateway(
            cache_store=cache_store,
            logger=logger,
            client=AniListClient(client=web),
        )

        # qBittorrent adapter: parses a release URL by tracker and adds it. A None
        # qbit is treated as a perpetual preview.
        torrents = TorrentService(
            qbit=qbit,
            web=web,
            category=arr_config.torrent_category,
            tags=app_config.qbittorrent.tags,
            logger=logger,
        )

        return cls(
            config=app_config,
            arr_config=arr_config,
            web=web,
            http=http,
            qbit=qbit,
            mappings=mappings,
            logger=logger,
            # SeaDex API gateway (entry lookups, with connection-error handling)
            seadex=SeaDexGateway(client=SeaDexEntry()),
            cache_store=cache_store,
            anilist=anilist,
            torrents=torrents,
            # Discord notifier; a no-op when no webhook is configured. The
            # SecretStr urls are unwrapped here, at their point of use.
            notifier=Notifier(
                discord_url=secret_value(app_config.notifications.discord_url),
                webhook_url=secret_value(app_config.notifications.wait_webhook_url),
                web=web,
            ),
            # Download-decision engine: flips each release's download flag.
            planner=DownloadPlanner(
                arr=arr,
                interactive=app_config.advanced.interactive,
                use_torrent_hash_to_filter=app_config.seadex.use_torrent_hash_to_filter,
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
        `close` rolls back anything not flushed by the end-of-run save point.
        `web` is NOT closed here: the CLI owns it across the whole cycle.
        """
        self.http.close()
        self.cache_store.close()


# Deliberately NOT @final (the old engine was): the strategy-seam tests subclass
# this with a scripted fake (_FakeRunServices), so the seam stays overridable.
class RunServices:
    """The per-AniList-id services hub the strategies call.

    Receives its shared collaborators as a `RunDeps` bundle (built and
    injected by the composition root in `bootstrap.py`) and owns the shared per-id
    pipeline the Arr strategies reach through `self._services`: the release
    filter, the grab tail, the cache checks, and the strategy-facing log
    delegates. `arr` is THE authority for which Arr is being run (`ctx.arr`
    is the per-run copy); the `RunLoop` run loop adopts
    the placeholder context minted here and pushes each run's fresh context
    down via `begin_run`, so the strategies never see the loop type.
    """

    def __init__(self, deps: RunDeps, arr: Arr) -> None:
        """Receive the shared collaborators and set up the per-id pipeline.

        `arr` is the authority for the run's arr; every fresh run context
        carries a per-run copy of it.
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
        self._reporter = deps.reporter

        self.arr = arr

        # Whether matching preferences moved since a prior full pass vouched for
        # this arr's cached verdicts; `_skippable_entry` then re-checks every
        # cached entry, and the run loop re-vouches once a full sweep finishes.
        self._selection_stale = deps.cache_store.selection_stale(arr, deps.config.selection_digest())

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the private-only skip flags, plus the
        # run's dry_run + resolved wait-mode flags) lives on this context, replaced
        # fresh at the start of each run by the loop's reset_run_stats. The single
        # placeholder is minted here - its dry_run=False + OFF wait mode keep every
        # preview / pending-import path a safe no-op - so the object is usable
        # before run_sync; the run loop ADOPTS it (via `ctx`) at construction.
        self._ctx = RunContext(arr=arr)

        # AniList ids whose arr-side files changed since the last pass (fed by the
        # run loop's activity scan); they bypass the cached-entry skip once.
        self._dirty_al_ids: set[int] = set()

        # The shared per-id collaborators, built from the deps hub + the
        # placeholder ctx. begin_run rebinds their ctx at the top of each run.
        self._filter = SeadexReleaseFilter(deps=deps, ctx=self._ctx)
        self._grab_pipeline = GrabPipeline(deps=deps, ctx=self._ctx)

    @property
    def ctx(self) -> RunContext:
        """The current run context (the placeholder until a run begins).

        Read by the `RunLoop` run loop at construction so
        it adopts the same placeholder instead of minting a second one.
        """

        return self._ctx

    @property
    def selection_stale(self) -> bool:
        """Whether matching settings changed since this arr's verdicts were vouched.

        Read by the run loop to announce the run-wide re-check
        `_skippable_entry` applies.
        """

        return self._selection_stale

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context to the hub and its per-id collaborators.

        Driven by the run loop's `begin_run`: once with the placeholder at
        construction (so pre-run paths are safe) and again right after
        `reset_run_stats` mints the run's real ctx. The wait-manager rebind
        stays on the loop side (the loop owns the manager).
        """

        self._ctx = ctx
        # Per-run state: the loop's activity scan re-marks dirty ids after this.
        self._dirty_al_ids.clear()
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
        """Side-effect-free mirror of process_al_id's no-entry + cached_entry_skip gates.

        True iff the per-id loop would actually process this id. Lets
        prefetch_episodes warm only the series the loop won't short-circuit
        (no SeaDex entry, or cached and unchanged), instead of every mapped
        series. No logging / stats / backfill - purely a predicate over the
        warmed SeaDex cache and the entry cache.
        """

        sd_entry = self._seadex.entry(al_id)  # warmed cache, no network
        if isinstance(sd_entry, SeaDexMiss):
            return False
        return self._skippable_entry(al_id, sd_entry) is None

    def _skippable_entry(self, al_id: int, sd_entry: EntryRecord) -> CachedEntry | None:
        """The cached row iff the per-id loop may skip this id; None re-processes.

        The single decision BOTH cache gates share: `al_id_needs_scan` picks
        what prefetch warms and `cached_entry_skip` what the loop skips, and
        the two must agree or un-warmed ids hit AniList one at a time. Run-wide
        bypasses come first (no db read): the ignore flag, a selection-digest
        change (matching preferences moved, so every cached verdict is suspect),
        or a dirty id; then the SeaDex-timestamp compare
        (mirrors `check_al_id_in_cache`); then warn mode re-processes
        fallback-satisfied entries, so their private-only warning resurfaces
        after a switch back from fallback mode.
        """

        if self._config.seadex.ignore_seadex_update_times or self._selection_stale or al_id in self._dirty_al_ids:
            return None
        entry = self.cache_store.get_entry(self._ctx.arr, al_id)
        if entry is None or entry.updated_at != sd_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT):
            return None
        if entry.fallback_satisfied and self._config.seadex.private_releases is PrivateReleaseAction.WARN:
            return None
        return entry

    def mark_dirty(self, al_ids: Iterable[int]) -> None:
        """Record AniList ids whose arr-side file state changed since the last pass.

        Fed by the run loop's `ArrActivityMonitor` scan;
        `al_id_needs_scan` and `cached_entry_skip` bypass the cached-entry
        short-circuit for exactly these ids.
        """

        self._dirty_al_ids.update(al_ids)

    def get_anilist_ids(
        self,
        ids: ExternalIds,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve external Arr ids to a {AniList id -> mapping} dict.

        The resolver does the mapping computation and reports which ids it
        dropped (the user's ignore list); the logging stays here so the
        presentation concern doesn't leak into the resolver.

        Args:
            ids: The external Arr ids to resolve (at least one).
            log_ignored: Log a ledger row for each ignored AniList ID.
                Pass False from the prefetch pass so ignored ids aren't logged
                twice (once there, once in the main loop)
        """

        anilist_mappings, ids_to_drop = self._mappings.get_anilist_ids(ids)

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
        """Resolve and remember the AniList title for an ID (no logging).

        The gateway resolves the raw title (no side-effects); the empty-result
        fallback and the transitional `current_title` attribution live here so
        later steps can attribute grabs to the active entry. The entry header is
        logged separately by log_al_title, once episodes are known.
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
    ) -> FilterResult:
        """Apply the download plan, stamping private-only skips onto ctx (delegates)."""

        return self._filter.filter_downloads(al_id, seadex_dict, arr_release_dict, ep_list)

    def is_preview(self) -> bool:
        """A run is a no-op preview (nothing can be grabbed): explicit dry run, or qBittorrent not configured."""
        return is_preview(self._ctx, self.qbit)

    @property
    def import_wait_mode(self) -> ImportWaitMode:
        """The wait mode resolved for the current run (cli > config > default).

        Set at the top of `run_sync`; the active strategy reads this (not the
        raw `config.imports.wait_mode`) so its seed-building gate agrees with the
        run loop's persist/reconcile/blocking gates - otherwise a CLI override that
        turns the feature on over an `off` config would build no seeds and the
        whole pass would silently no-op.
        """

        return self._ctx.import_wait_mode

    def _update_cache(
        self,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Merge `cache_details` into an entry's cache record (in-memory only).

        The run's save points flush it; see `CacheStore.update_cache`.
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

        Returns:
            Always `False` (nothing was grabbed).
        """

        self._log_no_seadex_releases()
        # Never fallback-satisfied: overwrite any stale True from a prior fallback run.
        cache_details["fallback_satisfied"] = False
        self._update_cache(al_id=al_id, cache_details=cache_details)
        time.sleep(self._config.advanced.sleep_time)
        return False

    def invalid_selection_skip(self) -> bool:
        """Shared tail for an interactive pick that left zero valid selections.

        Unlike `no_releases_skip` this deliberately persists NOTHING: caching
        the title as done would suppress it forever, when the user only fumbled the
        input - it must re-prompt on the next run. The picker already warned about
        the empty selection; this just throttles and reports "not grabbed".

        Returns:
            Always `False` (nothing was grabbed).
        """

        time.sleep(self._config.advanced.sleep_time)
        return False

    def al_id_prologue(self, al_id: int) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry.

        Returns the SeaDex entry to process, or None when there's nothing to do -
        either the id has no SeaDex entry, or the lookup was skipped because
        SeaDex is unreachable this run. The two misses are reported distinctly
        (an outage skip must never read as "no entry"); the caller moves to the
        next id either way.
        """

        # Reset the per-title skip flags (and the skipped group names) before we
        # make any download decisions for this title
        self._ctx.private_only_skipped = False
        self._ctx.private_only_groups = []
        self._ctx.private_only_stale_held = False
        self._ctx.fallback_covered = False
        self._ctx.unsupported_tracker_skipped = False
        self._ctx.unsupported_tracker_groups = []
        self._ctx.unsupported_tracker_hashes = []
        self._ctx.stats.checked += 1

        # Get the SeaDex entry if it exists
        sd_entry = self._seadex.entry(al_id)
        if isinstance(sd_entry, SeaDexMiss):
            if sd_entry is SeaDexMiss.OUTAGE:
                self._reporter.log_seadex_outage_skip(self._ctx, al_id)
            else:
                self._reporter.log_no_sd_entry(self._ctx, al_id)
            return None

        return sd_entry

    def cached_entry_skip(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        coverage: Callable[[], str],
    ) -> bool:
        """Shared cached-entry short-circuit for both Arr runners.

        When the id is already cached and we're honoring SeaDex update times,
        backfill the url + coverage on legacy records that predate those fields,
        log the cached entry, and return True so the caller skips it. `coverage`
        is a zero-arg callable so the (for Sonarr, episode-fetching) coverage
        lookup runs only on the one-time backfill, never on the common
        already-backfilled path; it builds "" for a movie, a season/episode
        range for a series.
        """

        # The shared skip decision; its one row read also serves the url-backfill
        # check below (was a SELECT updated_at + a SELECT url).
        entry = self._skippable_entry(al_id, sd_entry)
        if entry is None:
            return False

        # Backfill the enriched fields for records written before they existed,
        # so cached rows can still link to SeaDex (and, for series, show the
        # season/episode coverage). One-time per old entry.
        if not entry.url:
            self._update_cache(
                al_id=al_id,
                cache_details={"url": sd_entry.url, "coverage": coverage()},
            )
        self._reporter.log_cached_entry(self._ctx, self._ctx.arr, al_id)
        return True

    def grab_and_cache(self, req: GrabRequest) -> bool:
        """Shared per-id tail: add torrents, notify, cache the outcome (delegates).

        Both strategies build a `GrabRequest` and call this through their
        services; the produce mechanics live on `GrabPipeline`. Returns True
        only when max_torrents_to_add was reached (the caller stops the whole run).
        """

        return self._grab_pipeline.grab_and_cache(req)

    # --- Presentation seam (strategy-facing) ---------------------------------
    #
    # The run loop logs through the reporter directly (threading its ctx and the
    # preview/client facts so the reporter stays free of orchestrator state). The
    # only log_* methods kept here are the ones the Sonarr/Radarr strategies
    # invoke through their services view; each delegates the same way.

    def log_entry_status(self, state: EntryState, label: str) -> None:
        """Log a one-line entry status row (delegates to RunReporter)."""
        self._reporter.log_entry_status(state, label)

    def log_anilist_item_unmonitored(self, item_title: str) -> None:
        """Log an unmonitored-item skip (delegates to RunReporter)."""
        self._reporter.log_arr_item_unmonitored(self._ctx, item_title)

    def log_al_title(
        self,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> None:
        """Log the active-entry header (delegates to RunReporter)."""
        self._reporter.log_al_title(
            self._ctx,
            anilist_title,
            sd_entry,
            coverage=coverage,
        )

    def log_cached_entry(
        self,
        arr: Arr,
        al_id: int,
        state: Literal[EntryState.UNCHANGED, EntryState.IN_RADARR] = EntryState.UNCHANGED,
    ) -> None:
        """Log a cached entry (delegates to RunReporter)."""
        self._reporter.log_cached_entry(self._ctx, arr, al_id, state=state)

    def _log_no_seadex_releases(self) -> None:
        """Log a no-suitable-releases outcome (delegates to RunReporter)."""
        self._reporter.log_no_seadex_releases(self._ctx)
