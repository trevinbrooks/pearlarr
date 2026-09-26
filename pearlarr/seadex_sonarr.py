"""The Sonarr strategy: series/episode coverage and per-AniList-id processing over the services hub."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import override

from . import coverage as _coverage
from .arr_http import make_httpx_client
from .config import Arr
from .grab_pipeline import NO_SEEDS, GrabRequest
from .grab_placement import (
    EntryFacts,
    EntryPlacements,
    IdentityOverride,
    PendingSeed,
    SeedScope,
    TorrentReads,
    build_pending_seeds,
    entry_hashes,
    identity_overrides,
)
from .log import EntryState, count_noun, pluralize
from .manual_import import (
    AttemptKind,
    EffectStatus,
    ImportProbe,
    ImportProgress,
    ImportWaitMode,
    PendingImport,
)
from .mappings import ExternalIds, MappingEntry, MappingSource
from .output import hub_warn
from .placement_types import EpisodeIndex, episode_index
from .planner import get_episode_keys
from .protocols import ArrSync
from .radarr_client import AbstractRadarrClient, RadarrClient, collect_anime_movies
from .run_services import RunDeps, RunServices, bind_arr_http
from .seadex_types import (
    HistoryRecord,
    ProgressSink,
    RadarrItem,
    SeadexDict,
    SonarrItem,
    flagged_urls,
    ignore_tag_set,
)
from .sonarr_client import AbstractSonarrClient, SonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import import ImportExecutor, ImportReconciler
from .sonarr_mapper import FileEpisodeMapper
from .sonarr_parse import SonarrParseCache
from .torrent_listings import SeriesListings


def get_overlapping_results(seadex_dict: SeadexDict) -> bool:
    """See if SeaDex releases have overlapping episodes."""

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


def radarr_movies_matching(
    mapping: MappingEntry,
    movies: Sequence[RadarrItem],
) -> list[RadarrItem]:
    """The Radarr movies a specials mapping already covers, matched by TMDB/IMDb id.

    Only a season-0 mapping can match (shows and movies are sometimes lumped
    together, so a movie rides along as a special). An unset mapping id never
    matches - the `is not None` guards mean None == None is not a match.
    Order is preserved. A movie matching on both ids appears once.
    """

    radarr_movies: list[RadarrItem] = []

    if mapping.tvdb_season == 0:
        mapping_tmdb_id = mapping.tmdb_movie_id
        mapping_imdb_id = mapping.imdb_id

        for m in movies:
            # Match by TMDB or IMDb id. One append per movie either way.
            tmdb_match = mapping_tmdb_id is not None and m.tmdbId == mapping_tmdb_id
            imdb_match = mapping_imdb_id is not None and m.imdbId == mapping_imdb_id
            if tmdb_match or imdb_match:
                radarr_movies.append(m)

    return radarr_movies


def _override_line(url: str, override: IdentityOverride) -> str:
    """The warning for one file placed by size over its name."""

    label = override.named.label
    return (
        f"{url}: {override.name} goes on {label} because its size matches SeaDex's {label} release "
        f"(its name reads as {override.read_as})"
    )


class SonarrSync(ArrSync[SonarrItem]):
    """Sonarr sync strategy: owns the Sonarr REST client + episode domain logic.

    See `ArrSync` for the shared DI/hook-wiring regime.
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
            deps: The shared collaborators. The config/mappings
                this strategy reads directly are unpacked off it, and it's handed
                to the Sonarr collaborators for the cache/AniList gateway/log
                formatter they read.
            services: The services hub the per-id hooks call into.
            sonarr_client: A pre-built client to use instead of constructing
                the real `SonarrClient` (which needs the connection keys).
                None builds the real one.
            radarr_client: Same seam for the
                movies-in-Radarr cross-check client, whose library is fetched
                eagerly below (only consulted when
                `sonarr.ignore_movies_in_radarr` is on). None builds the real
                one when the feature is on and the Radarr keys are set.
        """

        self._services = services
        self._config = deps.config
        self._clock = deps.clock
        self.logger = deps.logger
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge

        # Set up Sonarr. An injected client (tests) is used as-is. Otherwise the
        # connection keys are required only now (when a Sonarr run runs) and the
        # real client is built over the run's shared httpx client (parse fires
        # one request per file, so its keep-alive removes a per-file handshake).
        if sonarr_client is not None:
            self.sonarr: AbstractSonarrClient = sonarr_client
        else:
            # A None deps.arr_http means the keys are missing - the fallback
            # bind raises require_connection's error here, the same point as before.
            base = deps.arr_http or bind_arr_http(Arr.SONARR, self._config, deps.http)
            self.sonarr = SonarrClient(
                # Streak re-warns pace with the wait view's digest cadence.
                http=replace(base, heartbeat_s=self._config.imports.digest_interval),
                logger=self.logger,
            )

        # Episode-domain collaborator: owns the per-run episode cache + series-id
        # fingerprint and the (series, al_id, mapping) -> episodes resolution. Built
        # from the shared deps + this client. The strategy delegates get_items /
        # prefetch_episodes to it and reads its series_fp for the parse cache.
        self._episodes = SonarrEpisodes(deps, self.sonarr, self._services)

        # Listing collaborator: reads every SeaDex entry of the series once per run. That gives the windows a
        # specials pack is checked against, and the size identities used to place files.
        self._listings = SeriesListings(deps.seadex, self._episodes, ignore_tag_set(self._config.seadex.ignore_tags))

        # Parse-cache collaborator: grab-time `/parse` of SeaDex filenames + the
        # durable, freshness-checked parse cache (read-through the shared cache_store).
        # The run's series fingerprint is threaded per call.
        self._parse = SonarrParseCache(deps, self.sonarr)

        # Import-time file -> episode mapper: owns the gnarly assignment of on-disk
        # leaves into OUR resolved episode set + the per-run on-disk parse cache.
        # The import executor calls candidate_files/assign. assign returns the
        # unplaceable files the executor warns about (producer/consumer split).
        self._mapper = FileEpisodeMapper(self.sonarr)

        # Import-execution collaborator: builds + POSTs the authoritative manual
        # import from the mapper's resolved map, owns the per-run quality/language
        # caches + the throttled rescan, and exposes the queue/command reads
        # import_completed consults. Built from the shared deps + this client + the
        # mapper. Its caches reset in get_items, the run-start hook.
        self._executor = ImportExecutor(deps, self.sonarr, self._mapper)

        # Import-reconcile collaborator: the import_completed decision + the
        # grab-time pending-seed build. Composes the episode collaborator + the
        # executor. The import_completed / process_al_id hooks delegate to it.
        self._reconciler = ImportReconciler(self._services.records, self._episodes, self._executor)

        self.ignore_movies_in_radarr = self._config.sonarr.ignore_movies_in_radarr

        # Only when ignore_movies_in_radarr is on do we need Radarr's movie list
        # (for the specials cross-check in process_al_id). Build a lightweight
        # RadarrClient and reuse the already-built shared mappings - no full
        # RadarrSync + engine stack (which would re-run mapping parse, cache
        # load, and a qBittorrent login, all unused here).
        self.all_radarr_movies: list[RadarrItem] | None = None
        if self.ignore_movies_in_radarr:
            if radarr_client is not None:
                self.all_radarr_movies = collect_anime_movies(
                    radarr_client,
                    self._mappings,
                    self.anibridge,
                )
            elif self._config.is_configured(Arr.RADARR):
                # The Radarr keys are optional here (a Sonarr run) - the gate
                # tolerates their absence instead of require_connection raising.
                # The run's shared client is pinned to SONARR's verify_ssl. This
                # one eager fetch talks to Radarr, so honor Radarr's own knob
                # with a scoped client (closed as soon as the list is read).
                with make_httpx_client(verify=self._config.radarr.verify_ssl) as radarr_http:
                    self.all_radarr_movies = collect_anime_movies(
                        RadarrClient(http=bind_arr_http(Arr.RADARR, self._config, radarr_http)),
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
        self._listings.reset()
        return self._episodes.collect_series()

    @override
    def filter_to_single(self, items: list[SonarrItem], item_id: int) -> list[SonarrItem]:
        """Narrow the series list to a single TVDB ID."""

        filtered = [s for s in items if s.tvdbId == item_id]
        if len(filtered) == 0:
            hub_warn(f"No anime series with TVDB ID {item_id} found in Sonarr - check the --series-id value")
        return filtered

    @override
    def item_anilist_ids(
        self,
        item: SonarrItem,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve AniList ids for a Sonarr series (by TVDB / IMDb id)."""

        return self._services.get_anilist_ids(
            ExternalIds(tvdb=item.tvdbId, imdb=item.imdbId),
            log_ignored=log_ignored,
        )

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_episodes(self, items: list[SonarrItem], *, progress: ProgressSink | None = None) -> int:
        """Warm the per-series episode lists before the scan loop.

        Delegates to the episode collaborator's concurrent prefetch. Returns how
        many series it warmed (the needs-scan subset), for the caller's ledger.
        """

        return self._episodes.prefetch(items, progress=progress)

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Sonarr history since `date` (delegates to the client)."""

        return self.sonarr.history_since(date)

    @override
    def process_al_id(
        self,
        item: SonarrItem,
        al_id: int,
        mapping: MappingEntry,
    ) -> None:
        """Process one AniList id for a Sonarr series.

        The middle is the episode-aware part: resolve the relevant episode list,
        its coverage and release groups, parse the SeaDex file lists into
        episodes, then hand off to the shared grab/cache tail.
        """

        run = self._services

        sd_entry = run.al_id_prologue(al_id)
        if sd_entry is None:
            return
        sd_url = sd_entry.url
        sonarr_series_id = item.id

        # Skip if already cached. The one-time backfill on a legacy record adds
        # the URL and the season/episode coverage. The coverage needs the episode
        # list, so it's resolved lazily, only when the backfill actually runs.
        if run.cached_entry_skip(
            al_id,
            sd_entry,
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
            return

        # Also check if it's in the Radarr cache, if we have that option. Skipped
        # alongside the same re-check signals the per-id gate honors: a forced
        # re-check (ignore flag) or moved matching settings (selection stale -
        # shared seadex config, so Radarr's verdict is suspect too) must fall
        # through to the live check, not short-circuit on a cached cross-arr verdict.
        # Invariant: this dedup names Arr.RADARR explicitly while running Sonarr -
        # folding the arr param into the run's bound arr breaks the cross-arr read.
        if (
            self.ignore_movies_in_radarr
            and not self._config.seadex.ignore_seadex_update_times
            and not run.selection_stale
        ):
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
                return

        # Resolved now, logged once the episode coverage is known.
        title = run.resolve_title(al_id)

        # Setup info for cache
        cache_details = run.new_cache_details(title, sd_entry)

        # If we don't want to add movies that are already in Radarr, do that now
        if self.ignore_movies_in_radarr and self.all_radarr_movies is not None:
            radarr_movies = radarr_movies_matching(mapping, self.all_radarr_movies)

            if len(radarr_movies) > 0:
                for movie in radarr_movies:
                    run.log_entry_status(
                        EntryState.IN_RADARR,
                        movie.title,
                    )

                self._clock.sleep(self._config.advanced.sleep_time)
                return

        # Get the episode list for all relevant episodes
        ep_list = self._episodes.get_ep_list(
            sonarr_series_id=sonarr_series_id,
            al_id=al_id,
            mapping=mapping,
        )

        if ep_list is None:
            return

        if not ep_list:
            # Resolved zero episodes (season not in Sonarr, offset past the end, or
            # AniBridge with no ranges): skip, don't mislabel "unmonitored" or grab orphans.
            run.log_entry_status(EntryState.NO_EPISODES, title.display)
            if mapping.source is MappingSource.ANIBRIDGE and not mapping.tvdb_mappings:
                # Surface any AniBridge no-usable-ranges case LOUDLY (distinct from a
                # Sonarr-library gap): a WARNING under the skip row naming the cause.
                # Keys off source, so it covers BOTH an empty-{} tvdb entry (mode
                # ANIBRIDGE) and a degraded imdb/tmdb-resolved entry (mode ANIME_IDS),
                # while a legit Kometa whole-series entry (source ANIME_IDS) stays quiet.
                hub_warn(f"AniBridge has no usable season ranges for {title.display} - skipping")
            self._clock.sleep(self._config.advanced.sleep_time)
            return

        # If all episodes are unmonitored, then skip if ignore_unmonitored is switched on
        if self._config.sonarr.ignore_unmonitored and not any(ep.monitored for ep in ep_list):
            run.log_anilist_item_unmonitored(
                item_title=title.display,
            )
            self._clock.sleep(self._config.advanced.sleep_time)
            return

        # Now that we have the episodes, log the active entry with its
        # season/episode coverage + URL, and remember them for the cache so
        # future cached runs can show the same detail
        coverage = _coverage.coverage_string(
            _coverage.episodes_from_ep_list(ep_list),
        )
        run.log_al_title(
            title=title.display,
            sd_entry=sd_entry,
            coverage=coverage,
        )
        cache_details["coverage"] = coverage
        cache_details["url"] = sd_url

        sonarr_releases = self._episodes.get_sonarr_releases(ep_list=ep_list)

        self.logger.debug(
            f"Sonarr release {pluralize(sonarr_releases.group_count(), 'group')}: {sonarr_releases.groups_label()}"
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            run.no_releases_skip(al_id, cache_details)
            return

        self.logger.debug(f"SeaDex: {', '.join(seadex_dict)}")

        # Place every listed file where the import will put it, so the grab is judged by the map the import
        # runs: a torrent already downloading under a stored record is placed as that record's leftover, under
        # every claim's window, and a specials pack against the windows of every entry listing it. The series
        # maps are the whole-series lists (a per-run cache hit each).
        waits_on_imports = run.import_wait_mode is not ImportWaitMode.OFF
        hashes = entry_hashes(seadex_dict)
        stored: dict[str, PendingImport] = run.records.stored_records(hashes) if waits_on_imports else {}
        indexes = self._series_indexes(
            {sonarr_series_id, *(sid for record in stored.values() for sid in record.series_ids)}
        )
        listings = self._listings.read(sonarr_series_id, self.item_anilist_ids(item, log_ignored=False))
        scope = SeedScope(al_id, episode_index(ep_list), indexes.get(sonarr_series_id, episode_index([])), title.names)
        placed = EntryPlacements.place(
            scope,
            self._parse.parsed_files(seadex_dict, series_fp=self._episodes.series_fp),
            TorrentReads(stored, indexes, listings).known(seadex_dict),
        )
        placed.attach_placements(seadex_dict)
        self._log_placements(placed)

        # If we're in interactive mode and there are multiple equivalent options here, then select
        if (
            self._config.advanced.interactive
            and len(seadex_dict) > 1
            and get_overlapping_results(seadex_dict=seadex_dict)
        ):
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )
            # Every token was invalid: skip WITHOUT caching (grab_and_cache would
            # cache the title as done and suppress it forever) so it re-prompts
            # next run.
            if len(seadex_dict) == 0:
                run.invalid_selection_skip()
                return

        # Filter downloads by whether the episodes in each torrent match the release
        # group we have in Sonarr
        plan = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr_releases=sonarr_releases,
            ep_list=ep_list,
        )
        torrent_hashes, seadex_dict = plan.torrent_hashes, plan.seadex_dict
        self._warn_overrides(placed, seadex_dict)

        # Build the per-torrent seeds the engine persists at the add site: one per release marked for download
        # (download + hash), carrying our own (basename -> Sonarr episode ids) map so the later manual import never
        # trusts Sonarr's blind parse. Gated on the engine's RESOLVED mode (cli > config), not the raw config, so
        # a CLI override agrees with the engine's persist/reconcile/blocking gates.
        pending_seeds: Mapping[str, PendingSeed] = NO_SEEDS
        if waits_on_imports:
            pending_seeds = build_pending_seeds(
                seadex_dict,
                placed,
                EntryFacts(
                    al_id=al_id,
                    series_id=sonarr_series_id,
                    title=title.display,
                    coverage=coverage,
                    url=sd_url,
                    guards=plan.guards,
                ),
            )

        run.grab_and_cache(
            GrabRequest(
                al_id=al_id,
                arr_title=item.title,
                entry_title=title.display,
                entry=sd_entry,
                seadex_dict=seadex_dict,
                torrent_hashes=torrent_hashes,
                cache_details=cache_details,
                replaced_groups=sonarr_releases.replaced_groups(),
                coverage=coverage,
                pending_seeds=pending_seeds,
                input_missing_groups=placed.input_missing_groups(seadex_dict),
            ),
        )

    def _series_indexes(self, series_ids: Iterable[int]) -> dict[int, EpisodeIndex]:
        """One index per series whose whole list served this run (a cold read fetches once); an unread one is absent."""

        return {
            sid: episode_index(episodes)
            for sid in series_ids
            if (episodes := self._episodes.cached_episodes(sid)) is not None
        }

    def _warn_overrides(self, placed: EntryPlacements, seadex_dict: SeadexDict) -> None:
        """Warn once per url this run grabs if any of its files were placed by size over their names.

        The warning shows the first such file, and the rest go to the debug log. `seadex_dict` must be the
        download plan's, so urls the run doesn't grab never warn.
        """

        for flagged in flagged_urls(seadex_dict):
            if overrides := identity_overrides(placed.by_url[flagged.url], placed.scope.series):
                first, *rest = overrides
                more = f" ({count_noun(len(rest), 'more file')} in the debug log)" if rest else ""
                hub_warn(f"{_override_line(flagged.url, first)}{more}")
                for override in rest:
                    self.logger.debug(_override_line(flagged.url, override))

    def _log_placements(self, placed: EntryPlacements) -> None:
        """One debug line per url with video files: what placed (the coverage the plan reads), what was set aside."""

        for url, placement in placed.by_url.items():
            if not placement.files:
                continue
            # The hold and the failed read explain an empty coverage on a listed release.
            aside = ", ".join(p.name for p in placement.assignment.excluded)
            hold = "" if placement.hold is None else f" (not grabbed: {placement.hold})"
            self.logger.debug(
                f"{url}: placed {_coverage.coverage_string(list(placement.records)) or 'nothing'}"
                f"{'' if placement.inputs_known else ' (a read failed)'}{hold}"
                f"{f'; set aside (other slice / duplicate): {aside}' if aside else ''}"
            )

    @override
    def pending_import_series_id(self, item: SonarrItem) -> int | None:
        """The Sonarr series id whose carried-over pending records this item owns.

        The engine's per-item snapshot hook keys off this. A Sonarr series owns
        its pending records by `series_id`, which is the Sonarr series id.
        """

        return item.id

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt: AttemptKind,
    ) -> ImportProbe:
        """One reconcile/import poll for a completed download (delegated).

        The @abstractmethod hook stays here so the ABC instantiates. The reconcile
        decision lives on `ImportReconciler`.
        """

        return self._reconciler.import_completed(pending, content_path, attempt)

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap read-only files-landed count for the wait bar (delegated)."""

        return self._reconciler.import_progress(pending)

    @override
    def close_tracked(self, pending: PendingImport) -> EffectStatus:
        """Dismiss the torrent's leftover Sonarr queue entry (see `ImportExecutor.close_tracked`)."""

        return self._executor.close_tracked(pending)

    @property
    @override
    def supports_blocking_monitor(self) -> bool:
        """Sonarr owns the interleaved end-of-run wait/import monitor cockpit."""

        return True
