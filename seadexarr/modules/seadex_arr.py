import copy
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import qbittorrentapi
import requests
from seadex import EntryRecord

from . import coverage as _coverage
from .anilist_gateway import AniListGateway
from .cache import CacheStore
from .config import PRIVATE_TRACKERS, AppConfig
from .log import (
    LogFormatter,
    indent_string,
    setup_logger,
)
from .mappings import MappingResolver
from .notify import Notifier
from .planner import DownloadPlanner
from .reporter import RunContext, RunReporter
from .seadex_gateway import SeaDexGateway
from .torrents import TorrentService


class SeaDexArr(ABC):

    def __init__(
        self,
        arr: str = "sonarr",
        config: str = "config.yml",
        cache: str = "cache.json",
        logger: logging.Logger | None = None,
    ) -> None:
        """Base class for SeaDexArr instances

        Args:
            arr (str, optional): Which Arr is being run.
                Defaults to "sonarr".
            config (str, optional): Path to a config file.
                Defaults to "config.yml".
            cache (str, optional): Path to a cache file.
                Defaults to "cache.json".
            logger. Logging instance. Defaults to None,
                which will create one.
        """

        # Load, template-sync, and expose the config file as typed settings.
        # AppConfig owns the file lifecycle (copy-template-if-missing, parse,
        # key-order sync); self.config stays bound to the raw mapping for the
        # few arr-specific keys the subclasses still read directly.
        self._config = AppConfig.load(config, arr)
        self.config_file = config
        self.config = self._config.data

        # Ignore unmonitored flag
        self.ignore_unmonitored = self._config.ignore_unmonitored

        # A single keep-alive session shared by the raw Sonarr/Radarr API calls
        self.session = requests.Session()

        # qbit. None until a fully-configured client is built below; the rest of
        # the app reads `self.qbit is None` to mean "no client -> perpetual
        # preview", so this must always be a defined attribute.
        self.qbit: qbittorrentapi.Client | None = None
        qbit_info = self._config.qbit_info

        # Configured only when every qbit_info field has a value; with a missing
        # block or any null field, no client is created.
        if qbit_info is not None and all(
            qbit_info.get(key, None) is not None for key in qbit_info
        ):
            qbit = qbittorrentapi.Client(**qbit_info)

            try:
                qbit.auth_log_in()
            except qbittorrentapi.LoginFailed:
                raise ValueError(
                    "qBittorrent login failed - check the qbit_info host and "
                    "credentials in your config",
                )

            self.qbit = qbit

        self.ignore_seadex_update_times = self._config.ignore_seadex_update_times

        self.use_torrent_hash_to_filter = self._config.use_torrent_hash_to_filter

        # Hooks between torrents and Arts, and torrent number bookkeeping
        self.torrent_category = self._config.torrent_category
        self.torrent_tags = self._config.torrent_tags
        self.max_torrents_to_add = self._config.max_torrents_to_add

        # When True, simulate a run without grabbing torrents, writing the cache,
        # or sending notifications. Set per-run by run(); the no-op default here
        # keeps every method that consults it safe before run() is called.
        self.dry_run = False

        # All per-run state (stats tally, running torrent count, the active
        # title/url/coverage, the run clock, the public_only skip flags) lives on
        # this context, replaced fresh at the start of each run by reset_run_stats.
        # A placeholder is created here so the object is usable before run().
        self._ctx = RunContext(arr=arr, dry_run=False)

        # Flags for filtering torrents
        self.public_only = self._config.public_only
        self.prefer_dual_audio = self._config.prefer_dual_audio
        self.want_best = self._config.want_best

        self.ignore_tags = self._config.ignore_tags

        # AniList IDs to skip entirely
        self.ignore_anilist_ids = self._config.ignore_anilist_ids

        # All trackers (public + private) by default; private are filtered later,
        # after the overlap check against what's already downloaded.
        self.trackers = self._config.trackers

        # Advanced settings
        self.sleep_time = self._config.sleep_time
        self.cache_time = self._config.cache_time

        # Resolve external Arr ids -> AniList ids via the three mapping sources.
        # The resolver downloads/refreshes, parses and indexes them, and owns the
        # module-global parse memo shared across a scheduled Radarr->Sonarr cycle.
        # anime_mappings / anidb_mappings / anibridge stay bound here as
        # transitional aliases so the subclasses keep reading them directly
        # (they're read-only after construction; the resolver owns them).
        self._mappings = MappingResolver(
            cache_time=self._config.cache_time,
            ignore_anilist_ids=self._config.ignore_anilist_ids,
            anime_mappings_cfg=self._config.anime_mappings_cfg,
            anidb_mappings_cfg=self._config.anidb_mappings_cfg,
            anibridge_mappings_cfg=self._config.anibridge_mappings_cfg,
        )
        self.anime_mappings = self._mappings.anime_mappings
        self.anidb_mappings = self._mappings.anidb_mappings
        self.anibridge = self._mappings.anibridge

        self.interactive = self._config.interactive

        if logger is None:
            self.logger = setup_logger(log_level=self._config.log_level)
        else:
            self.logger = logger

        # SeaDex API gateway (entry lookups, with connection-error handling)
        self._seadex = SeaDexGateway(logger=self.logger)

        # Per-run cache of the raw Sonarr episode fetch, keyed by series id. A
        # multi-season series maps to several AniList ids, each of which would
        # otherwise re-fetch the same whole-series episode list; cache it for the
        # run so the network round-trip happens once per series. Reset per run.
        self._ep_list_cache: dict[int, list] = {}

        # Load the cache (or create its schema) and reconcile the descriptor
        # against the current package version + config checksum. Each arr builds
        # its own store that reads the file fresh, so a scheduled Radarr->Sonarr
        # cycle hands off through cache.json rather than shared memory.
        self.cache_file = cache
        self.cache_store = CacheStore.load(cache, config_checksum=self._config.checksum())
        self.cache: dict[str, Any] = self.cache_store.data

        # AniList client gateway: owns the in-memory meta cache (al_cache) and the
        # persisted anilist_meta block. al_cache is exposed via the property below
        # so the subclasses and the not-yet-extracted presentation helpers keep
        # reading and reassigning self.al_cache transparently.
        self._anilist = AniListGateway(
            cache_store=self.cache_store,
            logger=self.logger,
        )

        # qBittorrent adapter: parses a release URL by tracker and adds it. qbit
        # is None when no client is configured; the service treats a missing
        # client as a perpetual preview.
        self._torrents = TorrentService(
            qbit=self.qbit,
            session=self.session,
            category=self.torrent_category,
            tags=self.torrent_tags,
            logger=self.logger,
        )

        # Discord notifier (builds the embed fields and posts grabs); a no-op
        # when no webhook is configured.
        self._notifier = Notifier(discord_url=self._config.discord_url)

        # Download-decision engine: consumes the shaped seadex_dict + the Arr's
        # release info and flips each release's download flag, returning a
        # PlanResult (hashes to remember + the private-only skip outcome) rather
        # than mutating run state or logging from deep in the call stack.
        self._planner = DownloadPlanner(
            public_only=self._config.public_only,
            interactive=self._config.interactive,
            use_torrent_hash_to_filter=self._config.use_torrent_hash_to_filter,
            logger=self.logger,
        )

        # All aligned detail rendering goes through this formatter, so the
        # presentation primitives (kv lines, blank separators, elapsed strings)
        # live on it rather than on the orchestration class. line_length is the
        # full width used for the run's separator rules.
        self.log_fmt = LogFormatter(self.logger)

        # Presentation: owns every log_* method and the end-of-run summary,
        # reading/writing the RunContext rather than scattered self.* state. The
        # orchestrator keeps thin log_* delegators (below) that inject self._ctx,
        # so the Sonarr/Radarr adapters call the same surface as before.
        self._reporter = RunReporter(
            logger=self.logger,
            log_fmt=self.log_fmt,
            cache_store=self.cache_store,
            anilist=self._anilist,
        )

    def close(self) -> None:
        """Close the shared HTTP session (release pooled connections)."""
        if self.session is not None:
            self.session.close()

    def anidb_anime_by_id(self, anidb_id: int) -> list:
        """Return the AniDB XML <anime> element(s) for an AniDB id.

        Delegates to the MappingResolver, which owns the lazily-built index.

        Args:
            anidb_id (int): AniDB id to look up
        """

        return self._mappings.anidb_anime_by_id(anidb_id)

    def get_seadex_entry(
        self,
        al_id: int,
    ) -> EntryRecord | None:
        """Get SeaDex entry from AniList ID

        Args:
            al_id (int): AniList ID
        """

        return self._seadex.entry(al_id)

    def check_al_id_in_cache(
        self,
        arr: str,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """Whether the cached entry matches SeaDex's last-updated timestamp."""

        return self.cache_store.check_al_id_in_cache(arr, al_id, seadex_entry)

    def get_cached_name(
        self,
        arr: str,
        al_id: int,
    ) -> str | None:
        """Cached AniList title for an entry, reused without an AniList lookup."""

        return self.cache_store.get_cached_name(arr, al_id)

    def get_cached_field(
        self,
        arr: str,
        al_id: int,
        field: str,
    ) -> Any:
        """Read a single stored field from an entry's cache record, if present."""

        return self.cache_store.get_cached_field(arr, al_id, field)

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: str = "movie",
        log_ignored: bool = True,
    ) -> dict:
        """Resolve external Arr ids to a {AniList id -> mapping} dict

        The resolver does the mapping computation and reports which ids it
        dropped (the user's ignore list); the logging stays here so the
        presentation concern doesn't leak into the resolver.

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (str): TMDB type. Can be "movie" or "show"
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
                self.log_ignored_anilist_id(al_id=al_id)

        return anilist_mappings

    @property
    def al_cache(self) -> dict:
        """In-memory AniList response cache, owned by the AniList gateway.

        Exposed as a read/write property so the subclasses and the
        not-yet-extracted presentation helpers keep reading and (re)assigning
        ``self.al_cache`` while the gateway is the single owner.
        """

        return self._anilist.al_cache

    @al_cache.setter
    def al_cache(self, value: dict) -> None:
        self._anilist.al_cache = value

    def load_anilist_cache(self) -> None:
        """Seed the in-memory AniList cache from the persisted store."""

        self._anilist.load_cache()

    def prefetch_anilist(self, al_ids: Iterable[int]) -> None:
        """Warm the AniList cache for a set of ids in batched requests.

        Args:
            al_ids (iterable[int]): Candidate AniList IDs for this run
        """

        self._anilist.prefetch(al_ids, preview=self._is_preview())

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
    ) -> dict:
        """Parse and filter SeaDex request

        Args:
            sd_entry: SeaDex API query
        """

        # The torrent records are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags
        ignore_tags = set(self.ignore_tags)
        final_torrent_list = [
            t for t in sd_entry.torrents if ignore_tags.isdisjoint(t.tags)
        ]

        # Filter down by allowed trackers
        final_torrent_list = [
            t for t in final_torrent_list if t.tracker.casefold() in self.trackers
        ]

        # Pull out torrents tagged as best, so long as at least one
        # is tagged as best. Keep a copy so we can fall back if audio
        # preferences otherwise downgrade quality
        best_torrents = [t for t in final_torrent_list if t.is_best]
        any_best = len(best_torrents) > 0

        # Narrow to 'best' releases when any exist
        if self.want_best and any_best:
            candidates = best_torrents
        else:
            candidates = final_torrent_list

        # Prefer dual-audio releases, but only when at least one exists
        if self.prefer_dual_audio:
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
        seadex_release_groups = {}
        for t in candidates:

            if t.release_group not in seadex_release_groups:
                seadex_release_groups[t.release_group] = {"urls": {}}
                seadex_release_groups[t.release_group]["tags"] = t.tags

            seadex_release_groups[t.release_group]["urls"][t.url] = {
                "url": t.url,
                "files": [f.name for f in t.files],
                "size": [f.size for f in t.files],
                "tracker": t.tracker,
                "is_public": t.tracker.is_public() and t.tracker.casefold() not in PRIVATE_TRACKERS,
                "hash": t.infohash,
                "download": False,
            }

        # If we only want public releases, then within each release group drop
        # any private URLs, so long as that group also has a public option. We
        # deliberately do this per-group rather than across the whole list: a
        # group that only has a private URL is kept for now and only filtered
        # out later if the Arr doesn't already have a matching download (see
        # reduce_overlapping_downloads)
        if self.public_only:
            for release_group_item in seadex_release_groups.values():
                urls = release_group_item["urls"]
                has_public = any(u["is_public"] for u in urls.values())
                if has_public:
                    release_group_item["urls"] = {
                        url: u for url, u in urls.items() if u["is_public"]
                    }

        return seadex_release_groups

    def filter_seadex_interactive(
        self,
        seadex_dict: dict,
        sd_entry: EntryRecord,
    ) -> dict:
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

            seadex_dict = copy.deepcopy(seadex_dict_filtered)

        return seadex_dict

    def filter_seadex_downloads(
        self,
        al_id: int,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        ep_list: list | None = None,
    ) -> tuple[list, dict]:
        """Flip the switch on whether we're downloading this torrent or not

        Thin orchestrator seam over the DownloadPlanner: pass it the entry's
        cached hashes, then apply the plan's private-only skip outcome back onto
        the run state the grab/cache tail still reads (the SkipNotice log lines,
        the public_only_skipped flag, and the skipped group names). This
        translation back to ``self`` is transitional; it unwinds when the
        add_torrent / _grab_and_cache knot is untied behind RunContext.

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
        # prologue; add_torrent may append more before _grab_and_cache reads them).
        if result.public_only_skipped:
            self._ctx.public_only_skipped = True
            self._ctx.public_only_groups.extend(result.public_only_groups)

        return result.torrent_hashes, result.seadex_dict

    def coverage_string(self, episodes: list) -> str:
        """One-line season/episode coverage, e.g. "S04 E01-E12".

        Thin wrapper over :func:`coverage.coverage_string`.
        """

        return _coverage.coverage_string(episodes)

    @staticmethod
    def episodes_from_ep_list(ep_list: list | None, missing_only: bool = False) -> list:
        """Convert a Sonarr ep_list into {"season","episode"} coverage dicts.

        Thin wrapper over :func:`coverage.episodes_from_ep_list`.
        """

        return _coverage.episodes_from_ep_list(ep_list, missing_only=missing_only)

    def add_torrent(
        self,
        torrent_dict: dict,
        torrent_client: str = "qbit",
    ) -> tuple[int, list]:
        """Add torrent(s) to a torrent client

        The per-release outcome lines (added / kept) are NOT logged here; this
        returns them so the caller (log_seadex_action) can print the whole block
        in order with a status that reflects what actually happened - "adding" if
        anything was grabbed, "keeping" if every recommended release was already
        present. The "skipped" warnings (private-only, unselected tracker) are
        still logged inline, as they're independent of that status.

        Args:
            torrent_dict (dict): Dictionary of torrent info
            torrent_client (str): Torrent client to use. Options are
                "qbit" for qBittorrent. Defaults to "qbit"

        Returns:
            tuple: (n_torrents_added, results), where results is a list of
                {"outcome": "added" | "already have", "name": str, "group": str}
                dicts, one per release acted on, in order
        """

        n_torrents_added = 0
        results = []

        for srg, srg_item in torrent_dict.items():

            seadex_urls = srg_item.get("urls", {})
            for url, url_item in seadex_urls.items():

                # If not flagged for download, then skip
                download = url_item.get("download", False)
                if not download:
                    continue

                item_hash = url_item.get("hash", None)
                tracker = url_item.get("tracker", None)

                if self.public_only and not url_item.get("is_public", True):
                    self.log_fmt.detail(
                        "skipped",
                        f"{tracker} private-only (public_only on)",
                        value_style="yellow",
                        level=logging.WARNING,
                    )
                    self._ctx.public_only_skipped = True
                    self._ctx.public_only_groups.append(srg)
                    continue

                # Skip trackers not in the user's selected list
                if tracker.casefold() not in self.trackers:
                    self.log_fmt.detail(
                        "skipped",
                        f"{url} (tracker {tracker} not in your selected list)",
                        value_style="yellow",
                    )
                    continue

                if torrent_client != "qbit":
                    raise ValueError(f"Unsupported torrent client {torrent_client}")

                # The service parses the release URL by tracker and adds it to
                # qBittorrent, returning the add status and a display name (the
                # client's name, or the release title scraped from the source
                # page as a fallback). A preview run simulates the add.
                success, torrent_name = self._torrents.add(
                    url=url,
                    tracker=tracker,
                    torrent_hash=item_hash,
                    preview=self._is_preview(),
                )

                if success == "torrent_added":
                    results.append(
                        {"outcome": "added", "name": torrent_name, "group": srg},
                    )

                    # Record the grab for the end-of-run summary. Prefer the
                    # release's own parsed file list (precise for multi-cour /
                    # per-torrent grabs); fall back to the entry-level coverage we
                    # mapped from the Arr so the summary's "files" is never blank
                    # when a release's filenames couldn't be parsed (e.g. an OVA).
                    coverage_str = self.coverage_string(
                        url_item.get("episodes", []),
                    ) or self._ctx.current_coverage
                    self._ctx.stats["added"].append(
                        {
                            "title": self._ctx.current_title,
                            "coverage": coverage_str,
                            "url": self._ctx.current_url,
                            "name": torrent_name,
                            "group": srg,
                        },
                    )

                    # Stop once max_torrents_to_add is reached
                    self._ctx.torrents_added += 1
                    n_torrents_added += 1
                    if self.max_torrents_to_add is not None:
                        if self._ctx.torrents_added >= self.max_torrents_to_add:
                            return n_torrents_added, results

                elif success == "torrent_already_added":
                    results.append(
                        {"outcome": "already have", "name": torrent_name, "group": srg},
                    )

                else:
                    raise ValueError(f"Cannot handle torrent client {torrent_client}")

        return n_torrents_added, results

    def _is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return self.dry_run or self.qbit is None

    def save_cache(self, sort: bool = True) -> None:
        """Persist the in-memory cache to disk, unless this run is a preview.

        Args:
            sort (bool): Sort anilist_entries by id before writing. Defaults to
                True so the persisted file is ordered by id; pass False to skip
                the sort on a hot write path.
        """

        self.cache_store.save(preview=self._is_preview(), sort=sort)

    def update_cache(self, arr: str, al_id: int, cache_details: dict | None = None) -> bool:
        """Merge ``cache_details`` into an entry's cache record (in-memory only).

        The run's save points flush it; see ``CacheStore.update_cache``.

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            cache_details (dict): Details for the cache entry. Defaults to None
        """

        return self.cache_store.update_cache(arr, al_id, cache_details)

    def reset_run_stats(self, arr: str, dry_run: bool) -> bool:
        """Start a fresh run context and the run clock

        Replaces the run-scoped state wholesale with a new RunContext, snapshots
        the logger-level counter (warning/error counts are diffed against this
        when the summary is logged), and drops any per-run scratch.

        Args:
            arr (str): "radarr" or "sonarr" being run.
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
        # Drop any episode lists cached from a previous run so a fresh run always
        # re-reads the current Sonarr library
        self._ep_list_cache = {}

        return True

    # --- Run orchestration (shared template) --------------------------------
    #
    # Each subclass's run() is a thin wrapper over run_sync: both Arrs share the
    # whole scaffolding (reset stats, fetch items, optional single-id filter,
    # AniList prefetch, the per-item loop, and the end-of-run save + summary) and
    # differ only in how an item is fetched/identified and what the per-AniList-id
    # body does. The divergent pieces are the hooks (_get_all_items,
    # _filter_to_single_item, _item_anilist_ids, _process_al_id); the identical
    # per-id head and grab/cache tail are _al_id_prologue and _grab_and_cache.

    @abstractmethod
    def _get_all_items(self) -> list:
        """Fetch every Arr item (movie/series) that has an AniList mapping."""

    @abstractmethod
    def _filter_to_single_item(self, items: list, item_id: int) -> list:
        """Narrow the item list to the one matching a single CLI-supplied id."""

    @abstractmethod
    def _item_anilist_ids(self, item: Any, log_ignored: bool = True) -> dict:
        """Resolve the AniList ids an Arr item maps to (arr-specific id args)."""

    @abstractmethod
    def _process_al_id(
        self,
        arr: str,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: dict,
    ) -> bool:
        """Handle one AniList id for an item; return True to stop the whole run.

        The per-id middle is the genuinely Arr-specific part (Radarr's single
        file vs. Sonarr's episode coverage); the shared head and tail live in
        _al_id_prologue and _grab_and_cache.
        """

    def run_sync(self, arr: str, item_id: int | None, dry_run: bool) -> bool:
        """Shared run scaffolding for both Arr syncers

        Args:
            arr (str): "radarr" or "sonarr"
            item_id (int | None): If set, only run for the single item with this
                id (TMDB for Radarr, TVDB for Sonarr)
            dry_run (bool): Simulate the run without grabbing torrents, writing
                the cache, or sending notifications
        """

        # Whether this is a no-op preview - consulted by the mutating helpers
        self.dry_run = dry_run

        # Start a fresh run context (stats tally + clock + counter snapshot)
        self.reset_run_stats(arr=arr, dry_run=dry_run)

        all_items = self._get_all_items()

        # If we're targeting a single item, filter down to it
        if item_id is not None:
            all_items = self._filter_to_single_item(all_items, item_id)

        n_items = len(all_items)

        self.log_arr_start(arr=arr, n_items=n_items)

        # Warm the AniList cache before the per-item loop: reuse what past runs
        # fetched, then batch-fetch (id_in pages) everything still missing, so the
        # loop rarely hits AniList one id at a time and trips its rate limit.
        self.load_anilist_cache()
        prefetch_ids = set()
        for item in all_items:
            if not item.monitored and self.ignore_unmonitored:
                continue
            prefetch_ids.update(
                self._item_anilist_ids(item, log_ignored=False),
            )
        self.prefetch_anilist(prefetch_ids)

        for item_idx, item in enumerate(all_items):

            try:

                item_title = item.title

                self.log_arr_item_start(
                    arr=arr,
                    item_title=item_title,
                    n_item=item_idx + 1,
                    n_items=n_items,
                )

                # If we're not monitored, then skip if ignore_unmonitored is switched on
                if not item.monitored and self.ignore_unmonitored:
                    self.log_arr_item_unmonitored(item_title=item_title)
                    continue

                # Get the mappings from the Arr item to AniList
                al_mappings = self._item_anilist_ids(item)

                if len(al_mappings) == 0:
                    self.log_no_anilist_mappings(title=item_title)
                    continue

                for al_id, mapping in al_mappings.items():
                    # _process_al_id returns True only when max_torrents_to_add was
                    # reached - it has already saved the cache and logged the
                    # summary - so stop the whole run here. The original per-item
                    # post-loop max check is redundant with this early return (the
                    # in-block check fires after every add, so torrents_added can
                    # never reach the cap without _process_al_id stopping first),
                    # so it isn't repeated.
                    if self._process_al_id(
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
        self.save_cache()
        self.log_run_summary(arr=arr)

        return True

    def _al_id_prologue(self, al_id: int | None) -> EntryRecord | None:
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
        self._ctx.stats["checked"] += 1

        if al_id is None:
            self.log_no_anilist_id()
            return None

        # Get the SeaDex entry if it exists
        sd_entry = self.get_seadex_entry(al_id=al_id)
        if sd_entry is None:
            self.log_no_sd_entry(al_id=al_id)
            return None

        return sd_entry

    def _cached_entry_skip(
        self,
        arr: str,
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
            arr (str): "radarr" or "sonarr"
            al_id (int): AniList id being processed
            sd_entry (EntryRecord): Resolved SeaDex entry
            sd_url (str): SeaDex entry URL stored on the backfilled record
            coverage (Callable[[], str]): Lazily builds the coverage string for
                the backfill ("" for a movie, a season/episode range for a series)
        """

        if not self.check_al_id_in_cache(arr=arr, al_id=al_id, seadex_entry=sd_entry):
            return False
        if self.ignore_seadex_update_times:
            return False

        # Backfill the enriched fields for records written before they existed,
        # so cached rows can still link to SeaDex (and, for series, show the
        # season/episode coverage). One-time per old entry.
        if not self.get_cached_field(arr, al_id, "url"):
            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details={"url": sd_url, "coverage": coverage()},
            )
        self.log_cached_entry(arr=arr, al_id=al_id)
        return True

    def _grab_and_cache(
        self,
        arr: str,
        al_id: int,
        item_title: str,
        anilist_title: str,
        sd_url: str,
        seadex_dict: dict,
        torrent_hashes: list,
        cache_details: dict,
        release_group: list | str | None,
    ) -> bool:
        """Shared per-id tail: add torrents, notify, then cache the outcome

        Identical across both Arrs once the (Arr-specific) seadex_dict and
        release-group info have been resolved. Returns True only when
        max_torrents_to_add has been reached (cache saved and summary logged),
        so the caller stops the whole run; otherwise False (move to the next id).

        Args:
            arr (str): "radarr" or "sonarr"
            al_id (int): AniList id being processed
            item_title (str): Arr item title (Discord notification heading)
            anilist_title (str): Resolved AniList title (non-None; Discord field)
            sd_url (str): SeaDex entry URL (non-None; Discord field)
            seadex_dict (dict): Filtered SeaDex releases
            torrent_hashes (list): Hashes to remember in the cache record
            cache_details (dict): Cache record being assembled for this id
            release_group (list | str | None): Arr release group(s) for the
                Discord fields
        """

        # Check the release groups are matching, and get a bespoke list of torrents
        any_to_download = self._planner.get_any_to_download(seadex_dict)

        # Capture the running total before the add block so we can tell whether
        # THIS title actually grabbed anything
        torrents_before = self._ctx.torrents_added

        if any_to_download:
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
            n_torrents_added, results = self.add_torrent(
                torrent_dict=seadex_dict,
                torrent_client="qbit",
            )

            # Log the action block now the outcome is known, so the status reads
            # "adding" only when something was actually grabbed (else "keeping")
            self.log_seadex_action(
                seadex_dict=seadex_dict,
                results=results,
                dry_run=self._is_preview(),
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

            if self.max_torrents_to_add is not None:
                if self._ctx.torrents_added >= self.max_torrents_to_add:
                    self.log_max_torrents_added()
                    self.save_cache()
                    self.log_run_summary(arr=arr)
                    return True

        elif not self._ctx.public_only_skipped:
            self._ctx.stats["up_to_date"] += 1
            self.log_fmt.detail(
                "status",
                "already have the recommended release",
                value_style="blue",
            )

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
            self._ctx.stats["needs_action"].append(
                {
                    "title": self._ctx.current_title,
                    "coverage": self._ctx.current_coverage,
                    "group": ", ".join(
                        dict.fromkeys(self._ctx.public_only_groups),
                    ),
                    "url": self._ctx.current_url,
                    "reason": "private-only release; public_only on",
                },
            )

        # Add in a wait, if required
        time.sleep(self.sleep_time)

        return False

    # --- Presentation delegators --------------------------------------------
    #
    # Every log_* method now lives on RunReporter (reporter.py); these thin
    # delegators preserve the call surface the Sonarr/Radarr adapters use, and
    # inject the current run context (and, for the summary, the preview/client
    # facts) so the reporter stays free of orchestrator state.

    def log_run_summary(self, arr: str) -> bool:
        """Log the end-of-run scoreboard (delegates to RunReporter)."""
        return self._reporter.log_run_summary(
            self._ctx,
            arr,
            is_preview=self._is_preview(),
            has_client=self.qbit is not None,
        )

    def log_arr_start(self, arr: str, n_items: int) -> bool:
        """Log the run banner (delegates to RunReporter)."""
        return self._reporter.log_arr_start(arr, n_items)

    def log_entry_status(
        self,
        state: str,
        label: str,
        style: str | None = "grey50",
    ) -> bool:
        """Log a one-line entry status row (delegates to RunReporter)."""
        return self._reporter.log_entry_status(state, label, style=style)

    def log_entry_coverage(
        self,
        coverage: str | None,
        url: str | None,
        style: str | None = "grey50",
        incomplete: bool = False,
    ) -> bool:
        """Log the coverage/URL line beneath an entry (delegates to RunReporter)."""
        return self._reporter.log_entry_coverage(
            coverage, url, style=style, incomplete=incomplete,
        )

    def log_arr_item_unmonitored(self, item_title: str) -> bool:
        """Log an unmonitored-item skip (delegates to RunReporter)."""
        return self._reporter.log_arr_item_unmonitored(self._ctx, item_title)

    # Both Arrs reach the same "unmonitored" outcome, so this is just an alias
    log_anilist_item_unmonitored = log_arr_item_unmonitored

    def log_arr_item_start(
        self,
        arr: str,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> bool:
        """Log the start of an Arr item (delegates to RunReporter)."""
        return self._reporter.log_arr_item_start(arr, item_title, n_item, n_items)

    def log_no_anilist_mappings(self, title: str) -> bool:
        """Log a no-AniList-mapping outcome (delegates to RunReporter)."""
        return self._reporter.log_no_anilist_mappings(self._ctx, title)

    def log_ignored_anilist_id(self, al_id: int) -> bool:
        """Log an ignore-listed AniList id (delegates to RunReporter)."""
        return self._reporter.log_ignored_anilist_id(al_id)

    def log_no_anilist_id(self) -> bool:
        """Log a missing-AniList-id outcome (delegates to RunReporter)."""
        return self._reporter.log_no_anilist_id()

    def log_no_sd_entry(self, al_id: int) -> bool:
        """Log a missing-SeaDex-entry outcome (delegates to RunReporter)."""
        return self._reporter.log_no_sd_entry(self._ctx, al_id)

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
        arr: str,
        al_id: int,
        state: str = "unchanged",
    ) -> bool:
        """Log a cached entry (delegates to RunReporter)."""
        return self._reporter.log_cached_entry(self._ctx, arr, al_id, state=state)

    def log_no_seadex_releases(self) -> bool:
        """Log a no-suitable-releases outcome (delegates to RunReporter)."""
        return self._reporter.log_no_seadex_releases(self._ctx)

    def log_seadex_action(
        self,
        seadex_dict: dict,
        results: list,
        dry_run: bool = False,
    ) -> bool:
        """Log the per-title action block (delegates to RunReporter)."""
        return self._reporter.log_seadex_action(seadex_dict, results, dry_run=dry_run)

    def log_max_torrents_added(self) -> bool:
        """Log hitting the max-torrents cap (delegates to RunReporter)."""
        return self._reporter.log_max_torrents_added()

