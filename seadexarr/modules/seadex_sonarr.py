import os
import time
from datetime import datetime
from typing import Any, override

from . import coverage as _coverage
from .cache import UPDATED_AT_STR_FORMAT, CacheRecord
from .config import Arr
from .log import EntryState, indent_string
from .manual_import import (
    ImportProbe,
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
    QueueVerdict,
    all_targets_done,
    build_episode_id_map,
    classify_queue,
    episode_file_statuses,
    episode_ids_for_parsed,
    manual_import_in_flight,
    normalize_basename,
    normalize_group,
)
from .mappings import MappingEntry
from .planner import get_episode_keys
from .protocols import ArrSync, EpisodeProgress
from .radarr_client import (
    collect_anime_movies,
    make_radarr_client,
)
from .seadex_arr import GrabRequest, RunDeps, SeaDexArr
from .seadex_types import (
    RadarrItem,
    SeadexDict,
    SonarrEpisode,
    SonarrItem,
)
from .sonarr_client import SonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import import ImportExecutor
from .sonarr_mapper import FileEpisodeMapper
from .sonarr_parse import SonarrParseCache, is_video_candidate


def get_overlapping_results(seadex_dict: SeadexDict) -> bool:
    """See if SeaDex releases have overlapping episodes

    Args:
        seadex_dict (dict): Dictionary of SeaDex releases
    """

    # Shares get_episode_keys with get_same_files_groups (seadex_arr) but
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
    The composition root injects the shared :class:`~.seadex_arr.RunDeps` (used to
    stand up the client and the episode domain logic) and the
    :class:`~.protocols.RunServices` run machinery (held as ``self._services``);
    the per-id hooks call the shared pipeline through it.
    """

    def __init__(self, deps: RunDeps, services: SeaDexArr) -> None:
        """Stand up the Sonarr client from the injected shared collaborators.

        Args:
            deps (RunDeps): The shared collaborators; the config/session/mappings/
                cache/AniList gateway/log formatter this strategy needs are read
                off it.
            services (RunServices): The run machinery the per-id hooks call into.
        """

        self._services = services
        self._config = deps.config
        self.session = deps.session
        self.logger = deps.logger
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge
        self.cache_store = deps.cache_store
        self.log_fmt = deps.log_fmt

        # Set up Sonarr (connection keys are required only now, when a Sonarr run runs)
        sonarr_url, sonarr_api_key = self._config.require_connection(Arr.SONARR)

        # self.session (a shared keep-alive requests.Session) comes from the
        # injected deps and is handed to the client; parse in particular fires one
        # request per file, so reusing it removes a per-file handshake.
        self.sonarr = SonarrClient(
            url=sonarr_url,
            api_key=sonarr_api_key,
            session=self.session,
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

        self.ignore_movies_in_radarr = self._config.sonarr.ignore_movies_in_radarr

        # Only when ignore_movies_in_radarr is on do we need Radarr's movie list
        # (for the specials cross-check in process_al_id). Build a lightweight
        # RadarrClient and reuse the already-built shared mappings - no nested
        # SeaDexRadarr (which would re-run the whole engine __init__: mapping
        # parse, cache load, and a qBittorrent login, all unused here).
        self.all_radarr_movies: list[RadarrItem] | None = None
        # None-tolerant cross-check read: the Radarr keys are optional here (this is a
        # Sonarr run), so read them directly rather than require_connection.
        radarr_url = self._config.radarr.url
        radarr_api_key = self._config.radarr.api_key

        if self.ignore_movies_in_radarr and radarr_url is not None and radarr_api_key is not None:
            radarr_client = make_radarr_client(
                url=radarr_url,
                api_key=radarr_api_key,
                session=self.session,
                logger=self.logger,
            )
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
    def prefetch_episodes(self, items: list[SonarrItem], *, progress: EpisodeProgress | None = None) -> int:
        """Warm the per-series episode lists before the scan loop.

        Delegates to the episode collaborator's concurrent prefetch; returns how
        many series it warmed (the needs-scan subset), for the caller's ledger.
        """

        return self._episodes.prefetch(items, progress=progress)

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
                    # Check by TMDB IDs
                    if mapping_tmdb_id is not None and m.tmdbId == mapping_tmdb_id and m not in radarr_movies:
                        radarr_movies.append(m)

                    # Check by IMDb IDs
                    if mapping_imdb_id is not None and m.imdbId == mapping_imdb_id and m not in radarr_movies:
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
                f"Sonarr release group(s): {', '.join(str(rg) for rg in sonarr_release_groups)}",
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
            pending_seeds = self._build_pending_seeds(
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

    def _build_pending_seeds(
        self,
        *,
        seadex_dict: SeadexDict,
        ep_list: list[SonarrEpisode],
        sonarr_series_id: int,
        anilist_title: str,
        coverage: str | None = None,
        url: str | None = None,
    ) -> dict[str, PendingImport]:
        """Build ``infohash -> PendingImport`` for every release marked to grab.

        For each downloadable url with a hash, seed our authoritative
        ``normalized basename -> episode ids`` map from the cached ``/parse``
        results and the ``(season, episode) -> id`` index. The map is best-effort
        at grab time (the series may not be fully in Sonarr yet); it self-heals at
        import time, when the files are on disk and the series exists, so a record
        is seeded for every grabbed torrent that carries at least one video file -
        not only the ones already fully mapped.

        Args:
            seadex_dict (SeadexDict): The filtered releases; ``url_item.download``
                marks the ones the engine will add.
            ep_list (list[SonarrEpisode]): The relevant Sonarr episodes (carry ids).
            sonarr_series_id (int): The Sonarr series id the files belong to.
            anilist_title (str): Display title for the record (logging only).
            coverage (str | None): The entry's season/episode coverage, persisted
                so a carried-over record can render its inline ``files`` line next
                run without re-deriving it.
            url (str | None): The SeaDex entry URL, persisted for the carried-over
                record's inline ``link`` line.

        Returns:
            dict[str, PendingImport]: Seeds keyed by infohash (empty when nothing
            downloadable carries a video file).
        """

        ep_id_map = build_episode_id_map(ep_list)
        # The resolved episode ids for this entry, in season order - persisted onto
        # every record so import-time assignment maps files into OUR set (the same
        # mapping the add flow resolved) instead of re-deriving identity from
        # Sonarr's title parse.
        ordered_episode_ids = [ep.id for ep in ep_list if ep.id]
        # Per-file parse records are read straight from the cache facade
        # (``get_sonarr_parse``): each is the persisted parse entry
        # ``{"fetched_at": str, "episodes": [...]}`` written by
        # ``parse_episodes_from_seadex`` in the same run (staged writes are visible
        # to reads on the same connection).
        added_at = datetime.now().strftime(UPDATED_AT_STR_FORMAT)

        pending_seeds: dict[str, PendingImport] = {}

        for srg, srg_item in seadex_dict.items():
            for url_item in srg_item.urls.values():
                if not (url_item.download and url_item.hash):
                    continue

                # The video files this torrent should import (subs / fonts / NCED
                # dropped), paired with their sizes for the order-based last resort.
                video_files: list[str] = []
                video_sizes: list[int] = []
                for seadex_file, size in zip(
                    url_item.files,
                    url_item.size,
                    strict=False,
                ):
                    base = os.path.basename(seadex_file)
                    if is_video_candidate(base):
                        video_files.append(base)
                        video_sizes.append(size)

                # No importable video files at all -> nothing to track.
                if not video_files:
                    continue

                # Best-effort grab-time mapping, keyed by NORMALIZED basename so it
                # matches the on-disk leaves at import time (NFC/NFD-safe).
                file_episode_map: dict[str, list[int]] = {}
                seasons: set[int] = set()
                for base in video_files:
                    record = self.cache_store.get_sonarr_parse(base)
                    if not record:
                        continue
                    parsed: list[dict[str, Any]] = record.get("episodes", [])
                    file_ids = episode_ids_for_parsed(parsed, ep_id_map)
                    if file_ids:
                        file_episode_map[normalize_basename(base)] = file_ids
                        seasons.update(ep["season"] for ep in parsed if ep.get("season") is not None)

                season_number = seasons.pop() if len(seasons) == 1 else None

                # The flat fallback is a legitimate guess ONLY for a genuine
                # single-file torrent; a multi-file pack leaves it empty so the
                # single-file rule can never stamp a whole season onto one file.
                episode_ids: list[int] = []
                if len(video_files) == 1 and file_episode_map:
                    episode_ids = next(iter(file_episode_map.values()))

                pending_seeds[url_item.hash] = PendingImport(
                    infohash=url_item.hash,
                    series_id=sonarr_series_id,
                    file_episode_map=file_episode_map,
                    episode_ids=episode_ids,
                    release_group=srg,
                    is_dual_audio=url_item.is_dual_audio,
                    season_number=season_number,
                    seadex_files=video_files,
                    seadex_sizes=video_sizes,
                    title=anilist_title,
                    added_at=added_at,
                    coverage=coverage,
                    url=url,
                    ordered_episode_ids=ordered_episode_ids,
                )

        return pending_seeds

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """One reconcile/import poll for a completed download.

        Reads the current episode files and Sonarr's (refreshed) queue as the
        source of truth - never the cache:

          * every intended episode already holds the recommended release ->
            ``IMPORTED`` + ``files_present`` (drop the record).
          * Sonarr is genuinely importing right now -> ``RETRY`` (don't race it).
          * a clean ``importPending`` -> ``RETRY`` until ``force`` (the engine
            forces on the snapshot/reconcile passes and on the final in-bound
            monitor poll, so a download Sonarr will never import - e.g. Completed
            Download Handling off, which parks it in ``importPending`` forever -
            is still imported rather than waited on indefinitely).
          * otherwise (``importBlocked`` / ``failed`` / not tracked / forced clean
            pending) -> drive our authoritative series-pinned manual import.

        Args:
            pending (PendingImport): The durable record for the completed torrent.
            content_path (str): The qBittorrent ``content_path`` to import from.
            force (bool): Stop deferring to Sonarr on a clean ``importPending``.
            at_deadline (bool): The final attempt - a still-missing intended file
                is terminal, so warn loudly (off the deadline it's debug).
        """

        label = pending.title or pending.infohash

        # Rescan (throttled) so the queue we read reflects the finished torrent.
        self._executor.refresh_downloads()

        # Episode files are the source of truth for "already imported"; fetch the
        # series episodes once and reuse them for the manual import below.
        episodes = self._episodes.episodes_for_series(pending.series_id)
        episodes_by_id = {ep.id: ep for ep in episodes if ep.id}
        recommended = self._recommended_groups(pending.series_id, pending.release_group)

        # Fast path: when our grab-time map already covers every video file, the
        # done-check is trustworthy without scanning the folder. An incomplete map
        # falls through to the manual import, which repairs it from the on-disk
        # files and re-checks against the complete set.
        seeded_targets = self._pending_target_ids(pending)
        if seeded_targets and self._seed_map_is_complete(pending):
            statuses = episode_file_statuses(seeded_targets, episodes_by_id, recommended)
            if all_targets_done(statuses):
                self.logger.debug(
                    indent_string(f"{label}: already imported (recommended files present)"),
                )
                return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        _download_id, queue_records = self._executor.queue_record_views(pending.infohash)
        verdict = classify_queue(queue_records)
        if verdict is QueueVerdict.WAIT:
            self.logger.debug(indent_string(f"{label}: Sonarr is importing; waiting"))
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)
        if verdict is QueueVerdict.PENDING_CLEAN and not force:
            self.logger.debug(indent_string(f"{label}: Sonarr has it pending; waiting"))
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # A ManualImport we (or a prior run) already POSTed may still be running
        # server-side after Sonarr dropped the torrent from the regular queue - so
        # the queue reads "empty -> step in" and we'd stack a duplicate every poll.
        # NOT gated on ``force``: the carried-over reconcile path always forces, and
        # that is exactly the path that loops; an in-flight command must suppress a
        # re-issue regardless (``force`` overrides Sonarr's clean-pending deferral,
        # a different state). A false positive only waits (bounded by the deadline).
        if manual_import_in_flight(
            self._executor.list_commands(),
            pending.infohash,
            content_path,
            set(seeded_targets),
        ):
            self.logger.debug(
                indent_string(f"{label}: a ManualImport is already in flight; waiting"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # STEP_IN, an empty queue, or a forced clean-pending: drive our import.
        return self._executor.run_manual_import(
            pending,
            content_path,
            episodes_by_id=episodes_by_id,
            recommended_groups=recommended,
            at_deadline=at_deadline,
        )

    def _series_pending_records(self, series_id: int) -> list[dict[str, Any]]:
        """Raw durable pending records for one series (any release group).

        Each record is the genuinely-open cache JSON form of a
        :class:`PendingImport` (``to_json``/``from_json``), so it is typed
        ``dict[str, Any]``.
        """

        # ``get_pending_for_series`` returns a fresh snapshot ``{infohash -> record}``
        # already filtered to this series in SQL (so a record dropped earlier this run
        # is absent). The ``record ->> 'series_id'`` match only returns JSON objects,
        # so every value is a typed record - no defensive isinstance/widen needed.
        return list(self.cache_store.get_pending_for_series(Arr.SONARR, series_id).values())

    def _recommended_groups(self, series_id: int, this_group: str) -> set[str]:
        """Normalized recommended groups for the series (the overwrite-guard set).

        The union of this torrent's group and the group of every other pending
        record we grabbed for the same series, so an episode our mapping assigned
        to another preferred torrent is never overwritten by this one.
        """

        groups: set[str] = set()
        if this_group:
            groups.add(normalize_group(this_group))
        for raw in self._series_pending_records(series_id):
            group = raw.get("release_group")
            if group:
                groups.add(normalize_group(group))
        return groups

    @staticmethod
    def _pending_target_ids(pending: PendingImport) -> list[int]:
        """Our intended episode ids for a record (map values + single-file fallback)."""

        ids: list[int] = []
        seen: set[int] = set()
        for file_ids in pending.file_episode_map.values():
            for ep_id in file_ids:
                if ep_id and ep_id not in seen:
                    seen.add(ep_id)
                    ids.append(ep_id)
        for ep_id in pending.episode_ids:
            if ep_id and ep_id not in seen:
                seen.add(ep_id)
                ids.append(ep_id)
        return ids

    @staticmethod
    def _seed_map_is_complete(pending: PendingImport) -> bool:
        """Whether the grab-time map already covers every video file we grabbed."""

        return bool(pending.seadex_files) and len(pending.file_episode_map) >= len(
            pending.seadex_files,
        )
