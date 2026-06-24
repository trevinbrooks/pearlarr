import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import qbittorrentapi
import requests
from seadex import EntryRecord

from seadexarr.modules.seadex_types import SeadexReleaseGroupItem, SeadexUrlItem

from . import coverage as _coverage
from .anilist_gateway import AniListGateway
from .cache import CacheField, CacheRecord, CacheStore
from .config import PRIVATE_TRACKERS, AppConfig, Arr
from .log import (
    EntryState,
    LogFormatter,
    indent_string,
    setup_logger,
)
from .mappings import MappingEntry, MappingResolver, TmdbType
from .notify import Notifier
from .planner import DownloadPlanner
from .protocols import ArrSync
from .reporter import GrabRecord, NeedsActionRecord, RunContext, RunReporter
from .seadex_gateway import SeaDexGateway
from .seadex_types import ArrItem, ArrReleaseDict, SeadexDict, SonarrEpisode
from .torrents import AddOutcome, ReleaseOutcome, TorrentService


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
    stored separately.
    """

    config: AppConfig
    config_file: str
    session: requests.Session
    qbit: qbittorrentapi.Client | None
    mappings: MappingResolver
    logger: logging.Logger
    seadex: SeaDexGateway
    cache_file: str
    cache_store: CacheStore
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
        cache: str = "cache.json",
        logger: logging.Logger | None = None,
        *,
        mappings: MappingResolver,
        app_config: AppConfig | None = None,
    ) -> "RunDeps":
        """Construct the shared collaborators in dependency order.

        Args:
            arr (Arr): Which Arr is being run; selects the arr-prefixed config keys.
            config (str, optional): Path to a config file. Defaults to "config.yml".
            cache (str, optional): Path to a cache file. Defaults to "cache.json".
            logger (logging.Logger | None, optional): Logger to use. Defaults to
                None, which builds one from the config's log level.
            mappings (MappingResolver): The id-mapping resolver, built once by the
                CLI and shared across a scheduled Radarr->Sonarr cycle so the three
                large mapping sources are downloaded, parsed and indexed once.
            app_config (AppConfig | None, optional): A pre-loaded config injected by
                the CLI so a scheduled cycle reads and template-syncs the file once
                per run. Defaults to None, which loads it here.
        """

        # Load, template-sync, and expose the config file as typed settings.
        # AppConfig owns the file lifecycle (copy-template-if-missing, parse,
        # key-order sync) and is the single source of truth for every setting. The
        # CLI may inject an already-loaded config (one read+sync shared across the
        # Radarr->Sonarr cycle); otherwise it's loaded here for the standalone path.
        app_config = AppConfig.load(config, arr) if app_config is None else app_config

        if logger is None:
            logger = setup_logger(log_level=app_config.log_level)

        # A single keep-alive session shared by the raw Sonarr/Radarr API calls.
        session = requests.Session()

        # qbit. None unless every qbit_info field has a value; with a missing block
        # or any null field, no client is created and the app treats `qbit is None`
        # as "no client -> perpetual preview".
        qbit: qbittorrentapi.Client | None = None
        qbit_info = app_config.qbit_info
        if qbit_info is not None and all(
            qbit_info.get(key, None) is not None for key in qbit_info
        ):
            client = qbittorrentapi.Client(**qbit_info)
            try:
                client.auth_log_in()
            except qbittorrentapi.LoginFailed:
                raise ValueError(
                    "qBittorrent login failed - check the qbit_info host and "
                    "credentials in your config",
                )
            qbit = client

        # Load the cache (or create its schema) and reconcile the descriptor against
        # the current package version + config checksum. Each arr builds its own
        # store that reads the file fresh, so a scheduled Radarr->Sonarr cycle hands
        # off through cache.json rather than shared memory.
        cache_store = CacheStore.load(cache, config_checksum=app_config.checksum())

        # AniList client gateway: owns the in-memory meta cache (al_cache) and the
        # persisted anilist_meta block.
        anilist = AniListGateway(cache_store=cache_store, logger=logger)

        # qBittorrent adapter: parses a release URL by tracker and adds it. A None
        # qbit is treated as a perpetual preview.
        torrents = TorrentService(
            qbit=qbit,
            session=session,
            category=app_config.torrent_category,
            tags=app_config.torrent_tags,
            logger=logger,
        )

        # All aligned detail rendering goes through this formatter.
        log_fmt = LogFormatter(logger)

        return cls(
            config=app_config,
            config_file=config,
            session=session,
            qbit=qbit,
            mappings=mappings,
            logger=logger,
            # SeaDex API gateway (entry lookups, with connection-error handling)
            seadex=SeaDexGateway(logger=logger),
            cache_file=cache,
            cache_store=cache_store,
            anilist=anilist,
            torrents=torrents,
            # Discord notifier; a no-op when no webhook is configured.
            notifier=Notifier(discord_url=app_config.discord_url),
            # Download-decision engine: flips each release's download flag.
            planner=DownloadPlanner(
                public_only=app_config.public_only,
                interactive=app_config.interactive,
                use_torrent_hash_to_filter=app_config.use_torrent_hash_to_filter,
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


class SeaDexArr:
    """The Arr-agnostic run machinery (the strategy's :class:`~.protocols.RunServices`).

    Receives its shared collaborators as a :class:`RunDeps` bundle (built and
    injected by the composition root in ``cli.py``) and owns the run loop, the
    per-run :class:`RunContext`, and the shared per-id pipeline. It drives an
    injected :class:`~.protocols.ArrSync` strategy (passed to :meth:`run_sync`)
    for the Arr-specific pieces; the strategy holds *this* object as its
    ``RunServices`` and calls the pipeline through it. The engine never holds the
    strategy and never constructs its own dependencies.
    """

    def __init__(self, deps: RunDeps, arr: Arr = Arr.SONARR) -> None:
        """Receive the shared collaborators and set up per-run state.

        Args:
            deps (RunDeps): The shared collaborators, built and injected by the
                composition root (``cli.py``). Unpacked into the attribute names
                the run loop and pipeline already read; the engine does not build
                any of them.
            arr (Arr, optional): Which Arr is being run. Defaults to Arr.SONARR.
        """

        # Unpack the injected collaborators into the attribute names the run loop
        # and pipeline methods read directly. anime_mappings / anidb_mappings /
        # anibridge are read off the shared resolver (read-only; it owns them).
        self._config = deps.config
        self.config_file = deps.config_file
        self.session = deps.session
        self.qbit = deps.qbit
        self._mappings = deps.mappings
        self.anime_mappings = deps.mappings.anime_mappings
        self.anidb_mappings = deps.mappings.anidb_mappings
        self.anibridge = deps.mappings.anibridge
        self.logger = deps.logger
        self._seadex = deps.seadex
        self.cache_file = deps.cache_file
        self.cache_store = deps.cache_store
        self._anilist = deps.anilist
        self._torrents = deps.torrents
        self._notifier = deps.notifier
        self._planner = deps.planner
        self.log_fmt = deps.log_fmt
        self._reporter = deps.reporter

        # When True, simulate a run without grabbing torrents, writing the cache,
        # or sending notifications. Set per-run by run_sync(); the no-op default
        # here keeps every method that consults it safe before run_sync is called.
        self.dry_run = False

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the public_only skip flags) lives on
        # this context, replaced fresh at the start of each run by reset_run_stats.
        # A placeholder is created here so the object is usable before run_sync.
        self._ctx = RunContext(arr=arr, dry_run=False)

    def close(self) -> None:
        """Close the shared HTTP session (release pooled connections)."""
        if self.session is not None:
            self.session.close()

    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """Whether the cached entry matches SeaDex's last-updated timestamp."""

        return self.cache_store.check_al_id_in_cache(arr, al_id, seadex_entry)

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
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
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

    def get_seadex_dict(
        self,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """Parse and filter SeaDex request

        Args:
            sd_entry: SeaDex API query
        """

        # The torrent records are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags
        ignore_tags = set(self._config.ignore_tags)
        final_torrent_list = [
            t for t in sd_entry.torrents if ignore_tags.isdisjoint(t.tags)
        ]

        # Filter down by allowed trackers
        final_torrent_list = [
            t for t in final_torrent_list if t.tracker.casefold() in self._config.trackers
        ]

        # Pull out torrents tagged as best, so long as at least one
        # is tagged as best. Keep a copy so we can fall back if audio
        # preferences otherwise downgrade quality
        best_torrents = [t for t in final_torrent_list if t.is_best]
        any_best = len(best_torrents) > 0

        # Narrow to 'best' releases when any exist
        if self._config.want_best and any_best:
            candidates = best_torrents
        else:
            candidates = final_torrent_list

        # Prefer dual-audio releases, but only when at least one exists
        if self._config.prefer_dual_audio:
            duals = [t for t in candidates if t.is_dual_audio]
            if len(duals) > 0:
                candidates = duals
        # Otherwise prefer non-dual-audio
        else:
            non_duals = [t for t in candidates if not t.is_dual_audio]
            if len(non_duals) > 0:
                candidates = non_duals

        # Pull out release groups, URLs, and various other useful info as a
        # dictionary
        seadex_release_groups: SeadexDict = {}
        for t in candidates:
            if t.release_group not in seadex_release_groups:
                seadex_release_groups[t.release_group] = SeadexReleaseGroupItem(urls={}, tags=t.tags)

            seadex_release_groups[t.release_group].urls[t.url] = SeadexUrlItem(
                url=t.url,
                files=[f.name for f in t.files],
                size=[f.size for f in t.files],
                tracker=t.tracker,
                is_public=t.tracker.is_public() and t.tracker.casefold() not in PRIVATE_TRACKERS,
                hash=t.infohash,
                download=False,
            )

        # If we only want public releases, then within each release group drop
        # any private URLs, so long as that group also has a public option. We
        # deliberately do this per-group rather than across the whole list: a
        # group that only has a private URL is kept for now and only filtered
        # out later if the Arr doesn't already have a matching download (see
        # reduce_overlapping_downloads)
        if self._config.public_only:
            for release_group_item in seadex_release_groups.values():
                urls = release_group_item.urls
                has_public = any(u.is_public for u in urls.values())
                if has_public:
                    release_group_item.urls = {
                        url: u for url, u in urls.items() if u.is_public
                    }

        return seadex_release_groups

    def filter_seadex_interactive(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """If multiple matches are found, let the user filter them interactively

        Args:
            seadex_dict: Dictionary of SeaDex releases
            sd_entry: SeaDex entry
        """

        self.logger.warning("Multiple releases found - pick which to grab")
        self.logger.info(
            indent_string("SeaDex notes:"),
        )

        notes = sd_entry.notes.split("\n")
        for n in notes:
            self.logger.warning(
                indent_string(n),
            )
        self.logger.warning(
            indent_string(""),
        )

        all_srgs = list(seadex_dict.keys())
        for s_i, s in enumerate(all_srgs):
            self.logger.warning(
                indent_string(f"[{s_i}]: {s}"),
            )

        srgs_to_grab = input(
            "Which release group(s)? Enter one number, a comma-separated list, "
            "or leave blank for all: ",
        )

        srgs_to_grab = srgs_to_grab.split(",")

        # Remove any blank entries
        while "" in srgs_to_grab:
            srgs_to_grab.remove("")

        # If we have some selections, parse down
        if len(srgs_to_grab) > 0:
            seadex_dict_filtered = {}
            for srg_idx in srgs_to_grab:

                try:
                    srg = all_srgs[int(srg_idx)]
                except IndexError:
                    self.logger.warning(
                        indent_string(f"Index {srg_idx} is out of range"),
                    )
                    continue
                seadex_dict_filtered[srg] = copy.deepcopy(seadex_dict[srg])

            seadex_dict = seadex_dict_filtered

        return seadex_dict

    def filter_seadex_downloads(
        self,
        al_id: int,
        seadex_dict: SeadexDict,
        arr: Arr,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> tuple[list[str | None], SeadexDict]:
        """Flip the switch on whether we're downloading this torrent or not

        Thin orchestrator seam over the DownloadPlanner: pass it the entry's
        cached hashes, then apply the plan's private-only skip outcome back onto
        the run state the grab/cache tail still reads (the SkipNotice log lines,
        the public_only_skipped flag, and the skipped group names). This
        translation back to ``self`` is transitional; it unwinds when the
        add_torrent / grab_and_cache knot is untied behind RunContext.

        Args:
            al_id: AniList ID
            seadex_dict: Dictionary of SeaDex releases
            arr: Type of arr instance
            arr_release_dict: Dictionary of arr release properties
            ep_list: List of episodes. Defaults to None
        """

        result = self._planner.plan(
            seadex_dict=seadex_dict,
            arr=arr,
            arr_release_dict=arr_release_dict,
            cached_hashes=self.cache_store.torrent_hashes(arr, al_id),
            ep_list=ep_list,
        )

        # The planner reports what to log rather than logging it; render each
        # private-only skip exactly as the inline call used to.
        for notice in result.skip_notices:
            self.log_fmt.detail(
                "skipped",
                f"{', '.join(notice.groups)} {notice.reason}",
                value_style="yellow",
                level=notice.level,
            )

        # Carry the skip flag/groups onto the run context (reset per title in the
        # prologue; add_torrent may append more before grab_and_cache reads them).
        if result.public_only_skipped:
            self._ctx.public_only_skipped = True
            self._ctx.public_only_groups.extend(result.public_only_groups)

        return result.torrent_hashes, result.seadex_dict

    def add_torrent(
        self,
        torrent_dict: SeadexDict,
    ) -> tuple[int, list[ReleaseOutcome]]:
        """Add torrent(s) to qBittorrent

        The per-release outcome lines (added / kept) are NOT logged here; this
        returns them so the caller (log_seadex_action) can print the whole block
        in order with a status that reflects what actually happened - "adding" if
        anything was grabbed, "keeping" if every recommended release was already
        present. The "skipped" warnings (private-only, unselected tracker) are
        still logged inline, as they're independent of that status.

        Args:
            torrent_dict (dict): Dictionary of torrent info

        Returns:
            tuple: (n_torrents_added, results), where results is a list of
                ``ReleaseOutcome``, one per release acted on, in order
        """

        n_torrents_added = 0
        results: list[ReleaseOutcome] = []
        cap = self._config.max_torrents_to_add

        for srg, srg_item in torrent_dict.items():
            for url, url_item in srg_item.urls.items():

                add_result = self._add_one_url(srg, url, url_item)
                if add_result is None:
                    continue

                results.append(add_result)
                if add_result.outcome is not AddOutcome.ADDED:
                    continue

                # Stop once max_torrents_to_add is reached
                self._ctx.torrents_added += 1
                n_torrents_added += 1
                if cap is not None and self._ctx.torrents_added >= cap:
                    return n_torrents_added, results

        return n_torrents_added, results

    def _add_one_url(
        self,
        srg: str,
        url: str,
        url_item: SeadexUrlItem,
    ) -> ReleaseOutcome | None:
        """Resolve a single SeaDex url to an add outcome (or ``None`` to skip).

        Returns ``None`` for a release that's filtered out (not flagged for
        download, private-only under ``public_only``, or an unselected tracker)
        and for a service ``add`` that neither added nor was already present. On
        an ``AddOutcome.ADDED`` the run-summary grab record is appended here; the
        caller owns the torrents_added/cap bookkeeping.
        """

        # If not flagged for download, then skip
        if not url_item.download:
            return None

        tracker = url_item.tracker

        if self._config.public_only and not url_item.is_public:
            self.log_fmt.detail(
                "skipped",
                f"{tracker} private-only (public_only on)",
                value_style="yellow",
                level=logging.WARNING,
            )
            self._ctx.public_only_skipped = True
            self._ctx.public_only_groups.append(srg)
            return None

        # Skip trackers not in the user's selected list
        if tracker.casefold() not in self._config.trackers:
            self.log_fmt.detail(
                "skipped",
                f"{url} (tracker {tracker} not in your selected list)",
                value_style="yellow",
            )
            return None

        # The service parses the release URL by tracker and adds it to
        # qBittorrent, returning the add status and a display name (the
        # client's name, or the release title scraped from the source
        # page as a fallback). A preview run simulates the add.
        success, torrent_name = self._torrents.add(
            url=url,
            tracker=tracker,
            torrent_hash=url_item.hash,
            preview=self._is_preview(),
        )

        if success is AddOutcome.ADDED:
            # Record the grab for the end-of-run summary. Prefer the
            # release's own parsed file list (precise for multi-cour /
            # per-torrent grabs); fall back to the entry-level coverage we
            # mapped from the Arr so the summary's "files" is never blank
            # when a release's filenames couldn't be parsed (e.g. an OVA).
            coverage_str = _coverage.coverage_string(url_item.episodes) or self._ctx.current_coverage
            self._ctx.stats.added.append(
                GrabRecord(
                    title=self._ctx.current_title,
                    coverage=coverage_str,
                    url=self._ctx.current_url,
                    name=torrent_name,
                    group=srg,
                ),
            )
            return ReleaseOutcome(
                outcome=AddOutcome.ADDED,
                name=torrent_name,
                group=srg,
            )

        if success is AddOutcome.ALREADY_ADDED:
            return ReleaseOutcome(
                outcome=AddOutcome.ALREADY_ADDED,
                name=torrent_name,
                group=srg,
            )

        return None

    def _is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return self.dry_run or self.qbit is None

    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> bool:
        """Merge ``cache_details`` into an entry's cache record (in-memory only).

        The run's save points flush it; see ``CacheStore.update_cache``.

        Args:
            arr (Arr): Arr instance
            al_id (int): AniList ID
            cache_details (CacheRecord): Details for the cache entry. Defaults
                to None
        """

        return self.cache_store.update_cache(arr, al_id, cache_details)

    def no_releases_skip(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord,
    ) -> bool:
        """Shared no-suitable-releases tail both Arr strategies fall into.

        When SeaDex yields no usable releases for an id, every strategy does the
        same four things: log the outcome, persist what it knows into the cache,
        throttle, and report "not grabbed". Hoisted here so the two strategies
        share one definition instead of a byte-for-byte duplicated block.

        Args:
            arr (Arr): Arr instance the entry is cached under.
            al_id (int): AniList ID.
            cache_details (CacheRecord): Cache record assembled for this id.

        Returns:
            bool: Always ``False`` (nothing was grabbed).
        """

        self.log_no_seadex_releases()
        self.update_cache(arr=arr, al_id=al_id, cache_details=cache_details)
        time.sleep(self._config.sleep_time)
        return False

    def reset_run_stats(self, arr: Arr, dry_run: bool) -> bool:
        """Start a fresh run context and the run clock

        Replaces the run-scoped state wholesale with a new RunContext and
        snapshots the logger-level counter (warning/error counts are diffed
        against this when the summary is logged).

        Args:
            arr (Arr): Which Arr is being run.
            dry_run (bool): Whether this run simulates without grabbing/writing.
        """

        counter = getattr(self.logger, "seadex_counter", None)
        self._ctx = RunContext(
            arr=arr,
            dry_run=dry_run,
            # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
            started_monotonic=time.monotonic(),
            log_counts_at_start=counter.snapshot() if counter else {},
        )

        return True

    # --- Run orchestration (shared machinery) -------------------------------
    #
    # run_sync is the shared scaffolding both Arrs use (reset stats, fetch items,
    # optional single-id filter, AniList prefetch, the per-item loop, and the
    # end-of-run save + summary). The Arr-specific pieces are the injected
    # strategy's hooks (get_items, filter_to_single, item_anilist_ids,
    # process_al_id); the strategy holds this object as its RunServices and calls
    # the shared per-id head/tail (al_id_prologue / cached_entry_skip /
    # grab_and_cache) through it.

    def run_sync[ItemT: ArrItem](
        self,
        strategy: ArrSync[ItemT],
        *,
        arr: Arr,
        item_id: int | None,
        dry_run: bool,
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
                already holds this object as its RunServices, so its hooks are
                called without passing self.
            arr (Arr): Which Arr is being run
            item_id (int | None): If set, only run for the single item with this
                id (TMDB for Radarr, TVDB for Sonarr)
            dry_run (bool): Simulate the run without grabbing torrents, writing
                the cache, or sending notifications
        """

        # Whether this is a no-op preview - consulted by the mutating helpers
        self.dry_run = dry_run

        # Start a fresh run context (stats tally + clock + counter snapshot)
        self.reset_run_stats(arr=arr, dry_run=dry_run)

        all_items: list[ItemT] = strategy.get_items()

        # If we're targeting a single item, filter down to it
        if item_id is not None:
            all_items = strategy.filter_to_single(all_items, item_id)

        n_items = len(all_items)

        self._reporter.log_arr_start(arr, n_items)

        # Warm the AniList cache before the per-item loop: reuse what past runs
        # fetched, then batch-fetch (id_in pages) everything still missing, so the
        # loop rarely hits AniList one id at a time and trips its rate limit.
        self._anilist.load_cache()
        prefetch_ids = set()
        for item in all_items:
            if not item.monitored and self._config.ignore_unmonitored:
                continue
            prefetch_ids.update(
                strategy.item_anilist_ids(item, log_ignored=False),
            )
        self._anilist.prefetch(prefetch_ids, preview=self._is_preview())

        for item_idx, item in enumerate(all_items):

            try:

                item_title = item.title

                self._reporter.log_arr_item_start(
                    arr, item_title, item_idx + 1, n_items,
                )

                # If we're not monitored, then skip if ignore_unmonitored is switched on
                if not item.monitored and self._config.ignore_unmonitored:
                    self._reporter.log_arr_item_unmonitored(self._ctx, item_title)
                    continue

                # Get the mappings from the Arr item to AniList
                al_mappings = strategy.item_anilist_ids(item)

                if len(al_mappings) == 0:
                    self._reporter.log_no_anilist_mappings(self._ctx, item_title)
                    continue

                for al_id, mapping in al_mappings.items():
                    # process_al_id returns True only when max_torrents_to_add was
                    # reached - it has already saved the cache and logged the
                    # summary - so stop the whole run here. The original per-item
                    # post-loop max check is redundant with this early return (the
                    # in-block check fires after every add, so torrents_added can
                    # never reach the cap without process_al_id stopping first),
                    # so it isn't repeated.
                    if strategy.process_al_id(
                        arr=arr,
                        item=item,
                        item_title=item_title,
                        al_id=al_id,
                        mapping=mapping,
                    ):
                        return True

            except Exception as e:
                title = getattr(item, "title", "unknown title")
                self.logger.error(
                    f"{title}: unexpected error: {e}", exc_info=True,
                )
                continue

        # Per-title update_cache calls only mutate memory now, so this end-of-run
        # save is what actually persists the run (and sorts by id on the way out)
        self.cache_store.save(preview=self._is_preview())
        self._reporter.log_run_summary(
            self._ctx, arr, is_preview=self._is_preview(), has_client=self.qbit is not None,
        )

        return True

    def al_id_prologue(self, al_id: int | None) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry

        Returns the SeaDex entry to process, or None when the id should be
        skipped (no id, or no SeaDex entry) - the caller moves to the next id.

        Args:
            al_id (int | None): AniList id being processed; defensively None-checked
                since the mapping dicts are built from external data
        """

        # Reset the per-title public_only skip flag (and the skipped group names)
        # before we make any download decisions for this title
        self._ctx.public_only_skipped = False
        self._ctx.public_only_groups = []
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
        arr: Arr,
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
            arr (Arr): Which Arr is being run
            al_id (int): AniList id being processed
            sd_entry (EntryRecord): Resolved SeaDex entry
            sd_url (str): SeaDex entry URL stored on the backfilled record
            coverage (Callable[[], str]): Lazily builds the coverage string for
                the backfill ("" for a movie, a season/episode range for a series)
        """

        if not self.check_al_id_in_cache(arr=arr, al_id=al_id, seadex_entry=sd_entry):
            return False
        if self._config.ignore_seadex_update_times:
            return False

        # Backfill the enriched fields for records written before they existed,
        # so cached rows can still link to SeaDex (and, for series, show the
        # season/episode coverage). One-time per old entry.
        if not self.cache_store.get_cached_field(arr, al_id, CacheField.URL):
            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details={"url": sd_url, "coverage": coverage()},
            )
        self._reporter.log_cached_entry(self._ctx, arr, al_id)
        return True

    def grab_and_cache(
        self,
        arr: Arr,
        al_id: int,
        item_title: str,
        anilist_title: str,
        sd_url: str,
        seadex_dict: SeadexDict,
        torrent_hashes: list[str | None],
        cache_details: CacheRecord,
        release_group: str | list[str | None] | None,
    ) -> bool:
        """Shared per-id tail: add torrents, notify, then cache the outcome

        Identical across both Arrs once the (Arr-specific) seadex_dict and
        release-group info have been resolved. Returns True only when
        max_torrents_to_add has been reached (cache saved and summary logged),
        so the caller stops the whole run; otherwise False (move to the next id).

        Args:
            arr (Arr): Which Arr is being run
            al_id (int): AniList id being processed
            item_title (str): Arr item title (Discord notification heading)
            anilist_title (str): Resolved AniList title (non-None; Discord field)
            sd_url (str): SeaDex entry URL (non-None; Discord field)
            seadex_dict (dict): Filtered SeaDex releases
            torrent_hashes (list): Hashes to remember in the cache record
            cache_details (CacheRecord): Cache record being assembled for this id
            release_group (list | str | None): Arr release group(s) for the
                Discord fields
        """

        # Check the release groups are matching, and get a bespoke list of torrents
        any_to_download = self._planner.get_any_to_download(seadex_dict)

        # Capture the running total before the add block so we can tell whether
        # THIS title actually grabbed anything
        torrents_before = self._ctx.torrents_added

        if not any_to_download:
            if not self._ctx.public_only_skipped:
                self._ctx.stats.up_to_date += 1
                self.log_fmt.detail(
                    "status",
                    "already have the recommended release",
                    value_style="blue",
                )
        elif self._grab(
            arr=arr,
            al_id=al_id,
            item_title=item_title,
            anilist_title=anilist_title,
            sd_url=sd_url,
            seadex_dict=seadex_dict,
            release_group=release_group,
        ):
            # max_torrents_to_add reached: cache saved and summary logged inside
            # _grab; stop the whole run.
            return True

        # Work out whether THIS title actually grabbed anything
        added_this_title = self._ctx.torrents_added - torrents_before

        # Update and save out the cache whenever something was grabbed for this
        # title, or when nothing was skipped at all. Leave the title uncached ONLY
        # when public_only skipped a release AND nothing else was grabbed for it -
        # so it's re-checked (and the skip re-logged as a reminder) on every run,
        # and retried once a public release appears or public_only is relaxed
        if added_this_title > 0 or not self._ctx.public_only_skipped:
            cache_details.update({"torrent_hashes": torrent_hashes})
            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details=cache_details,
            )
        elif added_this_title == 0:
            # Record the private-only skip for the summary's "needs action" list,
            # attributed to this title - but only when nothing was actually added
            # for it. The coverage is whatever log_al_title recorded as current
            # (a season/episode string for Sonarr, None for a Radarr movie).
            self._ctx.stats.needs_action.append(
                NeedsActionRecord(
                    title=self._ctx.current_title,
                    coverage=self._ctx.current_coverage,
                    group=", ".join(
                        dict.fromkeys(self._ctx.public_only_groups),
                    ),
                    url=self._ctx.current_url,
                    reason="private-only release; public_only on",
                ),
            )

        # Add in a wait, if required
        time.sleep(self._config.sleep_time)

        return False

    def _grab(
        self,
        arr: Arr,
        al_id: int,
        item_title: str,
        anilist_title: str,
        sd_url: str,
        seadex_dict: SeadexDict,
        release_group: str | list[str | None] | None,
    ) -> bool:
        """Add this title's torrents, notify, and honour the run-wide cap.

        Runs only when there's something to download. Returns True once
        max_torrents_to_add has been reached (cache saved and summary logged
        here) so the caller stops the whole run; otherwise False.
        """

        # Resolve the AniList cover thumbnail (via the gateway) and build the
        # Discord embed fields for the grab. The thumb lookup is done up front
        # to preserve ordering even though it's only used in the push below.
        anilist_thumb = self._anilist.thumb(al_id)
        fields = self._notifier.build_fields(
            arr=arr,
            release_group=release_group,
            seadex_dict=seadex_dict,
        )

        # Add torrents to qBittorrent. add_torrent runs even in a preview
        # (no client / dry run): the service simulates the add, while the
        # download-flag, public_only and tracker filters still apply, so only
        # releases that would actually be grabbed are counted.
        n_torrents_added, results = self.add_torrent(torrent_dict=seadex_dict)

        # Log the action block now the outcome is known, so the status reads
        # "adding" only when something was actually grabbed (else "keeping")
        self._reporter.log_seadex_action(
            seadex_dict, results, dry_run=self._is_preview(),
        )

        # Push a message to Discord if we've added anything (never on a
        # preview - it's an outward notification)
        if (
            self._notifier.enabled
            and n_torrents_added > 0
            and not self._is_preview()
        ):
            self._notifier.push(
                arr_title=item_title,
                al_title=anilist_title,
                seadex_url=sd_url,
                fields=fields,
                thumb_url=anilist_thumb,
            )

        cap = self._config.max_torrents_to_add
        if cap is not None and self._ctx.torrents_added >= cap:
            self._reporter.log_max_torrents_added()
            self.cache_store.save(preview=self._is_preview())
            self._reporter.log_run_summary(
                self._ctx,
                arr,
                is_preview=self._is_preview(),
                has_client=self.qbit is not None,
            )
            return True

        return False

    # --- Presentation seam (RunServices) ------------------------------------
    #
    # The engine logs through ``self._reporter`` directly (threading ``self._ctx``
    # and the preview/client facts so the reporter stays free of orchestrator
    # state). The only log_* methods kept here are the ones the Sonarr/Radarr
    # strategies invoke through their RunServices view; each delegates the same way.

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
            self._ctx, anilist_title, sd_entry, coverage=coverage,
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

