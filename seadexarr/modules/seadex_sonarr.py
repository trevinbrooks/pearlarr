import time
from typing import override

from . import coverage as _coverage
from .cache import CacheRecord
from .config import Arr, secret_value
from .grab_pipeline import GrabRequest
from .log import EntryState, indent_string
from .manual_import import (
    ImportProbe,
    ImportProgress,
    ImportWaitMode,
    PendingImport,
)
from .mappings import MappingEntry, MappingSource
from .planner import get_episode_keys
from .protocols import ArrSync
from .radarr_client import (
    AbstractRadarrClient,
    collect_anime_movies,
    make_radarr_client,
)
from .run_services import RunDeps, RunServices
from .seadex_types import (
    HistoryRecord,
    ProgressSink,
    RadarrItem,
    SeadexDict,
    SonarrItem,
)
from .sonarr_client import AbstractSonarrClient, SonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import import ImportExecutor, ImportReconciler
from .sonarr_mapper import FileEpisodeMapper
from .sonarr_parse import SonarrParseCache


def get_overlapping_results(seadex_dict: SeadexDict) -> bool:
    """See if SeaDex releases have overlapping episodes

    Args:
        seadex_dict (dict): Dictionary of SeaDex releases
    """

    # Shares get_episode_keys with get_same_files_groups (planner) but
    # deliberately differs on unparsed releases: here an unparsed release is
    # assumed to overlap (we can't prove it doesn't), whereas get_same_files_groups
    # keeps it separate (so we never drop content we couldn't verify). Keep both
    # consistent if the coverage semantics change.
    episode_sets: dict[str, set[tuple[int | None, int | None]]] = {}
    for rg, rg_item in seadex_dict.items():
        all_episodes = rg_item.all_episodes or []
        episode_sets[rg] = get_episode_keys(all_episodes)

    release_groups: list[str] = list(episode_sets.keys())
    for i, rg1 in enumerate(release_groups):
        for rg2 in release_groups[i + 1 :]:
            # If either release hasn't been parsed, then we can't rule out an
            # overlap, so assume they overlap
            if len(episode_sets[rg1]) == 0 or len(episode_sets[rg2]) == 0:
                return True

            # Otherwise they overlap if they share any episode
            if episode_sets[rg1] & episode_sets[rg2]:
                return True

    return False


class SonarrSync(ArrSync[SonarrItem]):
    """Sonarr sync strategy: owns the Sonarr REST client + episode domain logic.

    Implements the :class:`~.protocols.ArrSync` hooks the run machinery drives.
    The composition root injects the shared :class:`~.run_services.RunDeps` (used
    to stand up the client and the episode domain logic) and the
    :class:`~.run_services.RunServices` hub (held as ``self._services``);
    the per-id hooks call the shared pipeline through it.
    """

    def __init__(
        self,
        deps: RunDeps,
        services: RunServices,
        *,
        sonarr_client: AbstractSonarrClient | None = None,
        radarr_client: AbstractRadarrClient | None = None,
    ) -> None:
        """Stand up the Sonarr client from the injected shared collaborators.

        Args:
            deps (RunDeps): The shared collaborators; the config/mappings
                this strategy reads directly are unpacked off it, and it's handed
                to the Sonarr collaborators for the cache/AniList gateway/log
                formatter they read.
            services (RunServices): The services hub the per-id hooks call into.
            sonarr_client (AbstractSonarrClient | None): A pre-built client to use
                instead of constructing the real :class:`SonarrClient` (which
                needs the connection keys). Defaults to None (build the real
                one); tests inject a scripted fake through this typed seam, so
                the real ``__init__`` + collaborator wiring runs without keys.
            radarr_client (AbstractRadarrClient | None): Same seam for the
                movies-in-Radarr cross-check client, whose library is fetched
                eagerly below (only consulted when
                ``sonarr.ignore_movies_in_radarr`` is on). Defaults to None (build
                the real one when the feature is on and the Radarr keys are set).
        """

        self._services = services
        self._config = deps.config
        self.logger = deps.logger
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge

        # Set up Sonarr. An injected client (tests) is used as-is; otherwise the
        # connection keys are required only now (when a Sonarr run runs) and the
        # real client is built over the run's shared httpx client (parse fires
        # one request per file, so its keep-alive removes a per-file handshake).
        if sonarr_client is not None:
            self.sonarr: AbstractSonarrClient = sonarr_client
        else:
            sonarr_url, sonarr_api_key = self._config.require_connection(Arr.SONARR)
            self.sonarr = SonarrClient(
                url=sonarr_url,
                api_key=sonarr_api_key,
                http=deps.http,
                logger=self.logger,
            )

        # Episode-domain collaborator: owns the per-run episode cache + series-id
        # fingerprint and the (series, al_id, mapping) -> episodes resolution. Built
        # from the shared deps + this client; the strategy delegates get_items /
        # prefetch_episodes to it and reads its series_fp for the parse cache.
        self._episodes = SonarrEpisodes(deps, self.sonarr, self._services)

        # Parse-cache collaborator: grab-time ``/parse`` of SeaDex filenames + the
        # durable, freshness-checked parse cache (read-through the shared
        # cache_store, so its staged writes are visible to the seed builder's reads
        # later in the same run). The run's series fingerprint is threaded per call.
        self._parse = SonarrParseCache(deps, self.sonarr)

        # Import-time file -> episode mapper: owns the gnarly assignment of on-disk
        # leaves into OUR resolved episode set + the per-run on-disk parse cache.
        # The import executor calls candidate_files/assign; assign returns the
        # unplaceable files the executor warns about (producer/consumer split).
        self._mapper = FileEpisodeMapper(self.sonarr)

        # Import-execution collaborator: builds + POSTs the authoritative manual
        # import from the mapper's resolved map, owns the per-run quality/language
        # caches + the throttled rescan, and exposes the queue/command reads
        # import_completed consults. Built from the shared deps + this client + the
        # mapper; its caches reset in get_items, the run-start hook.
        self._executor = ImportExecutor(deps, self.sonarr, self._mapper)

        # Import-reconcile collaborator: the import_completed decision + the
        # grab-time pending-seed build. Composes the episode collaborator + the
        # executor; the import_completed / process_al_id hooks delegate to it.
        self._reconciler = ImportReconciler(deps, self._episodes, self._executor)

        self.ignore_movies_in_radarr = self._config.sonarr.ignore_movies_in_radarr

        # Only when ignore_movies_in_radarr is on do we need Radarr's movie list
        # (for the specials cross-check in process_al_id). Build a lightweight
        # RadarrClient and reuse the already-built shared mappings - no full
        # RadarrSync + engine stack (which would re-run mapping parse, cache
        # load, and a qBittorrent login, all unused here).
        self.all_radarr_movies: list[RadarrItem] | None = None
        if self.ignore_movies_in_radarr:
            # None-tolerant cross-check read: the Radarr keys are optional here
            # (this is a Sonarr run), so read them directly, not require_connection.
            radarr_url = self._config.radarr.url
            radarr_api_key = secret_value(self._config.radarr.api_key)
            if radarr_client is None and radarr_url is not None and radarr_api_key is not None:
                radarr_client = make_radarr_client(
                    url=radarr_url,
                    api_key=radarr_api_key,
                    http=deps.http,
                    logger=self.logger,
                )
            if radarr_client is not None:
                self.all_radarr_movies = collect_anime_movies(
                    radarr_client,
                    self._mappings,
                    self.anibridge,
                )

    # --- ArrSync hooks ------------------------------------------------------

    @override
    def get_items(self) -> list[SonarrItem]:
        """Every Sonarr series with AniList mapping info.

        Also the run-start hook: reset the per-run import scratch here, and let the
        episode collaborator reset its own cache + re-fingerprint the series-id set
        as it enumerates (this is called once, before the per-item loop).
        """

        self._mapper.reset()
        self._executor.reset()
        return self._episodes.collect_series()

    @override
    def filter_to_single(self, items: list[SonarrItem], item_id: int) -> list[SonarrItem]:
        """Narrow the series list to a single TVDB ID."""

        filtered = [s for s in items if s.tvdbId == item_id]
        if len(filtered) == 0:
            self.logger.warning(
                f"No anime series with TVDB ID {item_id} found in Sonarr",
            )
        return filtered

    @override
    def item_anilist_ids(
        self,
        item: SonarrItem,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve AniList ids for a Sonarr series (by TVDB / IMDb id)."""

        return self._services.get_anilist_ids(
            tvdb_id=item.tvdbId,
            imdb_id=item.imdbId,
            log_ignored=log_ignored,
        )

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_episodes(self, items: list[SonarrItem], *, progress: ProgressSink | None = None) -> int:
        """Warm the per-series episode lists before the scan loop.

        Delegates to the episode collaborator's concurrent prefetch; returns how
        many series it warmed (the needs-scan subset), for the caller's ledger.
        """

        return self._episodes.prefetch(items, progress=progress)

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Sonarr history since ``date`` (delegates to the client)."""

        return self.sonarr.history_since(date)

    @override
    def process_al_id(
        self,
        item: SonarrItem,
        item_title: str,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for a Sonarr series

        The middle is the episode-aware part: resolve the relevant episode list,
        its coverage and release groups, parse the SeaDex file lists into
        episodes, then hand off to the shared grab/cache tail.
        """

        run = self._services

        sd_entry = run.al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url
        sonarr_series_id = item.id

        # Skip if already cached. The one-time backfill on a legacy record adds
        # the URL and the season/episode coverage; the coverage needs the episode
        # list, so it's resolved lazily, only when the backfill actually runs.
        if run.cached_entry_skip(
            al_id,
            sd_entry,
            sd_url,
            lambda: _coverage.coverage_string(
                _coverage.episodes_from_ep_list(
                    self._episodes.get_ep_list(
                        sonarr_series_id=sonarr_series_id,
                        al_id=al_id,
                        mapping=mapping,
                    ),
                ),
            ),
        ):
            return False

        # Also check if it's in the Radarr cache, if we have that option
        if self.ignore_movies_in_radarr and not self._config.seadex.ignore_seadex_update_times:
            al_id_in_radarr_cache = run.check_al_id_in_cache(
                arr=Arr.RADARR,
                al_id=al_id,
                seadex_entry=sd_entry,
            )
            if al_id_in_radarr_cache:
                run.log_cached_entry(
                    arr=Arr.RADARR,
                    al_id=al_id,
                    state=EntryState.IN_RADARR,
                )
                return False

        # Resolve the AniList title (logged later, once episodes give us the
        # season/episode coverage)
        anilist_title = run.get_anilist_title(al_id=al_id)

        # Setup info for cache
        cache_details: CacheRecord = {
            "name": anilist_title,
            "updated_at": sd_entry.updated_at,
            "torrent_hashes": [],
        }

        # If we don't want to add movies that are already in Radarr, do that now
        if self.ignore_movies_in_radarr and self.all_radarr_movies is not None:
            radarr_movies: list[RadarrItem] = []

            # Make sure these are flagged as specials since sometimes shows and
            # movies are all lumped together
            mapping_season = mapping.tvdb_season
            if mapping_season == 0:
                mapping_tmdb_id = mapping.tmdb_movie_id
                mapping_imdb_id = mapping.imdb_id

                for m in self.all_radarr_movies:
                    # Match by TMDB or IMDb id; one append per movie either way.
                    tmdb_match = mapping_tmdb_id is not None and m.tmdbId == mapping_tmdb_id
                    imdb_match = mapping_imdb_id is not None and m.imdbId == mapping_imdb_id
                    if tmdb_match or imdb_match:
                        radarr_movies.append(m)

            if len(radarr_movies) > 0:
                for movie in radarr_movies:
                    run.log_entry_status(
                        EntryState.IN_RADARR,
                        movie.title,
                    )

                time.sleep(self._config.advanced.sleep_time)
                return False

        # Get the episode list for all relevant episodes
        ep_list = self._episodes.get_ep_list(
            sonarr_series_id=sonarr_series_id,
            al_id=al_id,
            mapping=mapping,
        )

        if ep_list is None:
            return False

        if not ep_list:
            # Resolved zero episodes (season not in Sonarr, offset past the end, or
            # AniBridge with no ranges): skip, don't mislabel "unmonitored" or grab orphans.
            run.log_entry_status(EntryState.NO_EPISODES, anilist_title)
            if mapping.source is MappingSource.ANIBRIDGE and not mapping.tvdb_mappings:
                # Surface any AniBridge no-usable-ranges case LOUDLY (distinct from a
                # Sonarr-library gap): a WARNING under the skip row naming the cause.
                # Keys off source, so it covers BOTH an empty-{} tvdb entry (mode
                # ANIBRIDGE) and a degraded imdb/tmdb-resolved entry (mode ANIME_IDS),
                # while a legit Kometa whole-series entry (source ANIME_IDS) stays quiet.
                self.logger.warning(
                    indent_string(f"AniBridge has no usable season ranges for {anilist_title}; skipping"),
                )
            time.sleep(self._config.advanced.sleep_time)
            return False

        # If all episodes are unmonitored, then skip if ignore_unmonitored is switched on
        ep_list_monitored = [ep.monitored for ep in ep_list]
        if not any(ep_list_monitored) and self._config.sonarr.ignore_unmonitored:
            run.log_anilist_item_unmonitored(
                item_title=anilist_title,
            )
            time.sleep(self._config.advanced.sleep_time)
            return False

        # Now that we have the episodes, log the active entry with its
        # season/episode coverage + URL, and remember them for the cache so
        # future cached runs can show the same detail
        coverage = _coverage.coverage_string(
            _coverage.episodes_from_ep_list(ep_list),
        )
        run.log_al_title(
            anilist_title=anilist_title,
            sd_entry=sd_entry,
            coverage=coverage,
        )
        cache_details["coverage"] = coverage
        cache_details["url"] = sd_url

        sonarr_release_dict = self._episodes.get_sonarr_release_dict(ep_list=ep_list)
        sonarr_release_groups = list(sonarr_release_dict.keys())

        self.logger.debug(
            indent_string(
                f"Sonarr release group(s): {', '.join(rg or '(none)' for rg in sonarr_release_groups)}",
            ),
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(al_id, cache_details)

        self.logger.debug(
            indent_string(
                f"SeaDex: {', '.join(seadex_dict)}",
            ),
        )

        # Parse out filenames and check for overlaps
        seadex_dict = self._parse.parse_episodes_from_seadex(seadex_dict, series_fp=self._episodes.series_fp)
        overlapping_results = get_overlapping_results(seadex_dict=seadex_dict)

        # If we're in interactive mode and there are multiple equivalent options here, then select
        if self._config.advanced.interactive and len(seadex_dict) > 1 and overlapping_results:
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )
            # Every token was invalid: skip WITHOUT caching (grab_and_cache would
            # cache the title as done and suppress it forever) so it re-prompts
            # next run.
            if len(seadex_dict) == 0:
                return run.invalid_selection_skip()

        # Filter downloads by whether the episodes in each torrent match the release
        # group we have in Sonarr
        torrent_hashes, seadex_dict = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr_release_dict=sonarr_release_dict,
            ep_list=ep_list,
        )

        # Build the authoritative per-torrent import seeds the engine will persist
        # at the add site. Only the releases marked for download (download +
        # hash) get a seed; each carries our own (basename -> Sonarr episode ids)
        # mapping so the later manual import never trusts Sonarr's blind parse.
        # Skipped entirely when the feature is off, to avoid the per-file work.
        # Gate on the engine's RESOLVED mode (cli > config), not the raw config,
        # so a CLI override agrees with the engine's persist/reconcile/blocking
        # gates - otherwise enabling via the CLI over an off config builds no
        # seeds and the whole pass silently no-ops.
        pending_seeds: dict[str, PendingImport] | None = None
        if run.import_wait_mode is not ImportWaitMode.OFF:
            pending_seeds = self._reconciler.build_pending_seeds(
                seadex_dict=seadex_dict,
                ep_list=ep_list,
                sonarr_series_id=sonarr_series_id,
                anilist_title=anilist_title,
                coverage=coverage,
                url=sd_url,
            )

        return run.grab_and_cache(
            GrabRequest(
                al_id=al_id,
                item_title=item_title,
                anilist_title=anilist_title,
                sd_url=sd_url,
                seadex_dict=seadex_dict,
                torrent_hashes=torrent_hashes,
                cache_details=cache_details,
                release_group=sonarr_release_groups,
                pending_seeds=pending_seeds,
            ),
        )

    @override
    def pending_import_series_id(self, item: SonarrItem) -> int | None:
        """The Sonarr series id whose carried-over pending records this item owns.

        The engine's per-item snapshot hook keys off this; a Sonarr series owns
        its pending records by ``series_id``, which is the Sonarr series id.
        """

        return item.id

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """One reconcile/import poll for a completed download (delegated).

        The @abstractmethod hook stays here so the ABC instantiates; the reconcile
        decision lives on :class:`~.sonarr_import.ImportReconciler`.
        """

        return self._reconciler.import_completed(
            pending,
            content_path,
            force=force,
            at_deadline=at_deadline,
        )

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap read-only files-landed count for the wait bar (delegated)."""

        return self._reconciler.import_progress(pending)
