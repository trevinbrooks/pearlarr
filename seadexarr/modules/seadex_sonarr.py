import os
import time
from datetime import datetime
from typing import Any, override

from . import coverage as _coverage
from .cache import UPDATED_AT_STR_FORMAT, CacheRecord
from .config import Arr
from .log import EntryState, indent_string
from .manual_import import (
    CandidateFile,
    ImportAction,
    ImportDecision,
    ImportProbe,
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
    QueueRecordView,
    QueueVerdict,
    all_targets_done,
    assign_episode_ids,
    build_episode_id_map,
    classify_queue,
    derive_languages,
    episode_file_statuses,
    episode_ids_for_parsed,
    manual_import_in_flight,
    normalize_basename,
    normalize_group,
    parse_quality_from_filename,
    parse_se_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
    targets_needing_import,
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
    CommandResource,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    ParsedFileInfo,
    QualityDefinition,
    QualitySource,
    RadarrItem,
    SeadexDict,
    SonarrEpisode,
    SonarrItem,
)
from .sonarr_client import SonarrClient
from .sonarr_episodes import SonarrEpisodes
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


# Rejection-reason substrings, matched case-insensitively against each
# rejection's reason/message text. ``ALREADY_IMPORTED`` means Sonarr already has
# the file (it imported it itself, or it exists) - seeing only these means the
# download is effectively done. ``SAMPLE`` is just a file to skip, not a sign the
# real episode imported, so the two are kept apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)

# RefreshMonitoredDownloads is quick (Sonarr re-scans its clients); poll its
# command status up to this many times, sleeping this long between, before
# proceeding regardless. Waiting means the queue we read next reflects the
# rescan; the bound means a stuck command never blocks the run.
_REFRESH_COMMAND_MAX_POLLS = 30
_REFRESH_COMMAND_POLL_S = 1
_COMMAND_TERMINAL_STATES = frozenset({"completed", "failed", "aborted", "cancelled"})


def _rejection_matches(candidate: ManualImportCandidate, tokens: tuple[str, ...]) -> bool:
    """True if any of a candidate's rejections contains one of ``tokens``.

    Best-effort and case-insensitive. Each rejection is an
    :class:`~.seadex_types.ImportRejection` view whose ``reason`` carries the
    human text (a bare-string rejection from an older Sonarr is folded into the
    same ``reason`` field at the client boundary).

    Args:
        candidate (ManualImportCandidate): The parsed candidate (reads
            ``rejections``).
        tokens (tuple[str, ...]): Lowercase substrings to look for.
    """

    for rejection in candidate.rejections:
        if not rejection.reason:
            continue
        lowered = rejection.reason.casefold()
        if any(token in lowered for token in tokens):
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

        # Per-run caches of the Sonarr quality-definition / language lists, used
        # to resolve a quality name / language names into the manual-import
        # payload objects. Fetched lazily on the first import and then reused for
        # the rest of the run so repeated imports don't re-hit the endpoints;
        # None means "not yet fetched" (cleared in get_items, the run-start hook).
        self._quality_defs_cache: list[QualityDefinition] | None = None
        self._languages_cache: list[Language] | None = None

        # Per-run, in-memory cache of the series-agnostic ``/parse`` of an on-disk
        # leaf (raw basename -> ParsedFileInfo | None), so the import poll loop sends
        # a given filename to Sonarr's parser at most once a run rather than every
        # poll. A None value caches a confirmed "Sonarr can't parse this" miss.
        self._parse_info_cache: dict[str, ParsedFileInfo | None] = {}

        # Infohashes for which we've already warned that some on-disk files could
        # not be placed in the resolved set, so the loud "left these for you" line
        # is logged once a run rather than every poll until the record clears.
        self._warned_unplaceable: set[str] = set()

        # Monotonic time of the last RefreshMonitoredDownloads we asked Sonarr for,
        # used to throttle the rescan: the blocking pass calls import_completed
        # every poll and may walk several torrents back-to-back, so we re-issue the
        # (global) refresh at most once per import_poll_interval rather than on
        # every call. None means "not refreshed yet this run" (reset in get_items).
        self._last_refresh_monotonic: float | None = None

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

        self._quality_defs_cache = None
        self._languages_cache = None
        self._parse_info_cache = {}
        self._warned_unplaceable = set()
        self._last_refresh_monotonic = None
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
        self._refresh_sonarr_downloads()

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

        _download_id, queue_records = self._queue_record_views(pending.infohash)
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
            self._list_commands(),
            pending.infohash,
            content_path,
            set(seeded_targets),
        ):
            self.logger.debug(
                indent_string(f"{label}: a ManualImport is already in flight; waiting"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # STEP_IN, an empty queue, or a forced clean-pending: drive our import.
        return self._manual_import(
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

    def _refresh_sonarr_downloads(self) -> None:
        """Queue RefreshMonitoredDownloads (throttled) and wait for it, best-effort.

        RefreshMonitoredDownloads is global and the blocking pass polls often (and
        may walk several torrents back-to-back), so it's re-issued at most once per
        ``import_poll_interval``. Waiting for the command to finish means the queue
        read that follows reflects the rescan; the poll bound means a stuck command
        can never block the run, and a failure to queue/confirm just leaves the
        next queue read slightly stale (a later poll corrects it).
        """

        now = time.monotonic()
        interval = self._config.imports.poll_interval
        if self._last_refresh_monotonic is not None and now - self._last_refresh_monotonic < interval:
            return
        self._last_refresh_monotonic = now

        cmd_id = self.sonarr.refresh_monitored_downloads()
        if cmd_id is None:
            return
        self.logger.debug(indent_string("Asked Sonarr to rescan its downloads"))

        for _ in range(_REFRESH_COMMAND_MAX_POLLS):
            command = self.sonarr.command_status(cmd_id)
            state = command.status or ""
            if state.casefold() in _COMMAND_TERMINAL_STATES:
                return
            time.sleep(_REFRESH_COMMAND_POLL_S)

    def _queue_record_views(self, infohash: str) -> tuple[str, list[QueueRecordView]]:
        """Reduce this download's queue records to what :func:`classify_queue` needs.

        Matches records to the torrent by ``downloadId`` (case-insensitively;
        Sonarr stores the infohash uppercased) and keeps the state, status, and
        whether status messages are present - the three signals that tell a healthy
        pending item from a stuck/blocked one. Records with no tracked state are
        dropped; an empty result means Sonarr isn't tracking the download.

        Args:
            infohash (str): The torrent infohash (the download id).
        """

        target = infohash.casefold()
        views: list[QueueRecordView] = []
        download_id = ""
        for record in self.sonarr.queue():
            dl_id = record.download_id
            if dl_id is None or dl_id.casefold() != target:
                continue
            if not record.state:
                continue
            download_id = dl_id
            views.append(
                QueueRecordView(
                    state=record.state,
                    status=record.status or "",
                    has_messages=record.has_messages,
                ),
            )
        return download_id if download_id else infohash, views

    def _list_commands(self) -> list[CommandResource]:
        """The current Sonarr command list, for the in-flight ManualImport guard.

        A thin pass-through to :meth:`SonarrClient.list_commands` (mirrors
        :meth:`_queue_record_views`' delegation to ``self.sonarr``). Fetched fresh
        every poll - never cached - since an in-flight command's status changes as
        Sonarr finishes the import.
        """

        return self.sonarr.list_commands()

    def _manual_import(
        self,
        pending: PendingImport,
        label: str,
        *,
        episodes_by_id: dict[int, SonarrEpisode],
        recommended_groups: set[str],
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Drive our authoritative series-pinned manual import for one download.

        Scans ``content_path`` for candidates (pinned to ``pending.series_id``),
        repairs our file->episode map from the actual on-disk files (re-parsing
        whatever the seed didn't cover, mapped through OUR ``(season, episode) ->
        id`` index - never Sonarr's candidate episode assignment), then imports
        EXACTLY the files our map intends: each file's episodes that don't already
        hold a recommended release (so a recommended file is never overwritten) and
        no file outside our map (so an episode our mapping gave to another preferred
        torrent is never imported here). An intended file Sonarr can't see yet is
        retried, never silently skipped.

        Returns an :class:`ImportProbe`. A manual-import command's copy is async, so
        accepting the command is NOT ``files_present`` - the probe reads
        ``RETRY`` + ``command_issued`` until a later poll verifies the episode files
        actually landed. ``files_present`` is set only when every intended episode
        already holds a recommended file (nothing left to copy).

        Args:
            pending (PendingImport): The durable record for the completed torrent.
            content_path (str): The qBittorrent ``content_path`` to import from.
            label (str): Display label for the log lines.
            episodes_by_id (dict[int, SonarrEpisode]): Current series episodes by id.
            recommended_groups (set[str]): Normalized recommended-group guard set.
            at_deadline (bool): The final attempt - a still-missing intended file
                is terminal, so warn loudly; otherwise it's an expected early-poll
                gap and only logged at debug.
        """

        candidates = self.sonarr.manual_import_candidates(
            pending=pending,
            filter_existing_files=False,
        )
        if candidates is None:
            # Transient (timeout / non-200); the client already warned. Ask again.
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        candidates_by_basename = self._candidate_files(candidates)
        ep_id_map = build_episode_id_map(list(episodes_by_id.values()))
        authoritative_map, unplaceable = self._assign_from_resolved(
            pending,
            candidates_by_basename,
            ep_id_map,
        )
        if unplaceable:
            self._warn_unplaceable_files(pending, unplaceable)

        if not authoritative_map:
            self.logger.debug(
                indent_string(f"{label}: no mappable files for {pending.title} yet"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Done-check against the COMPLETE (repaired) intended set, from the files.
        target_ids = sorted({i for ids in authoritative_map.values() for i in ids})
        statuses = episode_file_statuses(target_ids, episodes_by_id, recommended_groups)
        if all_targets_done(statuses):
            self.logger.debug(
                indent_string(f"{label}: already imported (recommended files present)"),
            )
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        needing = targets_needing_import(statuses)
        decisions = plan_import_files(authoritative_map, candidates_by_basename, needing)

        lang_objs = self._import_language_objects(pending)
        quality_defs = self._quality_definitions()

        files: list[ManualImportFile] = []
        missing: list[str] = []
        for decision in decisions:
            match decision.action:
                case ImportAction.MISSING:
                    missing.append(decision.basename)
                case ImportAction.IMPORT:
                    files.append(
                        self._build_file_entry(decision, pending, lang_objs, quality_defs, label),
                    )
                case _:
                    # SAMPLE / ALREADY / SKIP_DONE -> nothing to import for this file.
                    continue

        if missing:
            # Intended files our map covers but Sonarr can't see yet. An early poll
            # finding them absent is expected (the copy hasn't landed), so it's only
            # noisy at the deadline, where a still-missing file is terminal: warn
            # loudly only then, debug otherwise. Either way the record is retried,
            # never dropped silently.
            message = indent_string(
                f"{label}: {len(missing)} intended file(s) not visible to Sonarr for {pending.title}; will retry",
            )
            if at_deadline:
                self.logger.warning(message)
            else:
                self.logger.debug(message)

        if not files:
            # Nothing to queue this poll: retry if files are merely missing, else
            # everything intended is already satisfied (already/sample/skip_done).
            if missing:
                return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=self._config.imports.mode,
        )
        if cmd_id is None:
            self.logger.debug(
                indent_string(f"{label}: Sonarr rejected the import command; will retry"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # The command was accepted, but its copy is async - the episode files may
        # not have landed yet (a remote-mount copy isn't instant). Do NOT declare
        # the files imported on command acceptance: report RETRY + command_issued,
        # so the next monitor cycle flips to files_present once they appear.
        self.logger.debug(
            indent_string(f"{label}: queued {len(files)} file(s) for import (command {cmd_id})"),
        )
        return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=True)

    def _candidate_files(
        self,
        candidates: list[ManualImportCandidate],
    ) -> dict[str, CandidateFile]:
        """Index on-disk manual-import candidates by normalized basename.

        The candidates arrive already parsed at the Sonarr client boundary
        (:meth:`SonarrClient.manual_import_candidates`), so each is read by
        attribute and the raw DTO never reaches the decision path.
        """

        by_basename: dict[str, CandidateFile] = {}
        for candidate in candidates:
            path = candidate.path
            if not path:
                continue
            base = normalize_basename(os.path.basename(path))
            by_basename[base] = CandidateFile(
                basename=base,
                path=path,
                quality=candidate.quality,
                is_sample=_rejection_matches(candidate, _SAMPLE_TOKENS),
                is_already_imported=_rejection_matches(candidate, _ALREADY_IMPORTED_TOKENS),
            )
        return by_basename

    def _assign_from_resolved(
        self,
        pending: PendingImport,
        candidates_by_basename: dict[str, CandidateFile],
        ep_id_map: dict[tuple[int, int], int],
    ) -> tuple[dict[str, list[int]], list[str]]:
        """Build the final ``basename -> episode ids`` map from OUR resolved set.

        Identity is assigned off each file's series-agnostic parse against the live
        series episode map - never Sonarr's series-matched title parse: a file's
        ``(season, episode)`` is honored only *inside* our resolved set, an
        absolute-numbered pack is mapped positionally onto it, and anything ambiguous
        is returned as skipped (the caller warns and leaves it - the chosen safe
        posture).

        Files our grab-time ``file_episode_map`` already covers (the add-time
        assignment) are taken as-is - no need to re-parse what we resolved at grab
        time. Every other on-disk video leaf is parsed series-agnostically and handed
        to the pure :func:`assign_episode_ids`, which places it into our resolved set
        (``ordered_episode_ids``, the add-flow's season-sorted episodes - or, for a
        record predating that field, one synthesized from its seeds). When there is
        no set to scope against (an on-disk specials record whose grab-time parse
        found nothing), :func:`assign_episode_ids` falls back to the live series map
        for exactly named files (see ``allow_unscoped``). Fresh placements self-heal
        onto the record; SeaDex order keeps output and the absolute leg stable.

        Returns ``(merged_map, unplaceable_basenames)``.
        """

        on_disk = {
            norm_base: candidate
            for norm_base, candidate in candidates_by_basename.items()
            if is_video_candidate(os.path.basename(candidate.path))
        }

        # SeaDex order first (so output is stable and the absolute leg's input is
        # deterministic), then any on-disk leaf the SeaDex list didn't name.
        ordered = [norm for norm in (normalize_basename(name) for name in pending.seadex_files) if norm in on_disk]
        placed = set(ordered)
        ordered += [norm_base for norm_base in on_disk if norm_base not in placed]

        # Honor our grab-time map (OUR add-time assignment) - no need to re-parse
        # what we resolved at grab time. Intended files not yet on disk stay in the
        # map so the planner detects them missing and retries (never silent-drops);
        # only the on-disk leftovers the seed doesn't cover (e.g. a specials pack
        # whose grab-time parse found nothing) are resolved from their parse.
        seeded: dict[str, list[int]] = {}
        for name, ids in pending.file_episode_map.items():
            clean = [i for i in ids if i]
            if clean:
                seeded[normalize_basename(name)] = clean
        seeded_ids = {i for ids in seeded.values() for i in ids}

        leftover = [norm for norm in ordered if norm not in seeded]
        parsed_by_file = {
            norm_base: self._parsed_file_info(os.path.basename(on_disk[norm_base].path)) for norm_base in leftover
        }

        # The set the leftovers assign into: ordered_episode_ids, or - for a record
        # predating that field - one synthesized from its seeds (so the old
        # seed/single-file scoping survives). Ids the seed already owns are removed,
        # so a leftover file can't be handed an episode that's already placed.
        resolved_ids = pending.ordered_episode_ids or sorted(
            seeded_ids | {i for i in pending.episode_ids if i},
        )
        leftover_resolved = [i for i in resolved_ids if i not in seeded_ids]

        result = assign_episode_ids(
            leftover,
            parsed_by_file,
            leftover_resolved,
            ep_id_map,
        )

        # Self-heal: keep every fresh placement on the record for the run.
        for norm_base, ids in result.assigned.items():
            pending.file_episode_map[norm_base] = ids

        return {**seeded, **result.assigned}, result.skipped

    def _parsed_file_info(self, raw_base: str) -> ParsedFileInfo | None:
        """Series-agnostic parse of one on-disk leaf, cached per run.

        Prefers Sonarr's ``/parse`` ``parsedEpisodeInfo`` (it handles absolute
        numbering); on a transient parse failure (None) falls back to an offline
        ``SxxExx`` regex - without caching - so a momentary Sonarr hiccup neither
        strands a correctly-named file nor sticks for the rest of the run.
        """

        if raw_base in self._parse_info_cache:
            return self._parse_info_cache[raw_base]
        info = self.sonarr.parse_episode_info(raw_base)
        if info is None:
            return parse_se_from_filename(raw_base)
        self._parse_info_cache[raw_base] = info
        return info

    def _warn_unplaceable_files(
        self,
        pending: PendingImport,
        unplaceable: list[str],
    ) -> None:
        """Warn (once a run per download) about on-disk files we couldn't place.

        These are files Sonarr sees but our resolved mapping can't confidently
        assign (ambiguous numbering, an extra that slipped the skip list, a pack
        that doesn't line up 1:1). We import what we can and leave these - surfacing
        them loudly so they're never silently dropped.
        """

        if pending.infohash in self._warned_unplaceable:
            return
        self._warned_unplaceable.add(pending.infohash)
        label = pending.title or pending.infohash
        coverage = f" ({pending.coverage})" if pending.coverage else ""
        self.logger.warning(
            indent_string(
                f"{label}{coverage}: {len(unplaceable)} file(s) could not be mapped "
                f"to a resolved episode and were left unimported",
            ),
        )

    def _import_language_objects(self, pending: PendingImport) -> list[Language]:
        """Resolve the import language objects for a record (lazily cached)."""

        if self._languages_cache is None:
            self._languages_cache = self.sonarr.languages()
        lang_names = derive_languages(
            pending.is_dual_audio,
            self._config.imports.languages_dual,
            self._config.imports.languages_single,
        )
        return resolve_language_objects(lang_names, self._languages_cache)

    def _quality_definitions(self) -> list[QualityDefinition]:
        """The Sonarr quality definitions (lazily fetched + cached for the run)."""

        if self._quality_defs_cache is None:
            self._quality_defs_cache = self.sonarr.quality_definitions()
        return self._quality_defs_cache

    def _build_file_entry(
        self,
        decision: ImportDecision,
        pending: PendingImport,
        lang_objs: list[Language],
        quality_defs: list[QualityDefinition],
        label: str,
    ) -> ManualImportFile:
        """Build one ManualImport file payload from a planned ``import`` decision.

        The episode ids come straight from our authoritative map (never Sonarr's
        parse); the quality is decided per axis with precedence Sonarr's parse ->
        our filename parse -> the configured default, and always emits a real
        quality (never an omitted key), warning only when it resolves to Unknown.

        Only ``import`` decisions reach here, so ``decision.path`` is the on-disk
        candidate path (always set); the ``or decision.basename`` keeps the
        payload ``path`` a non-null ``str`` for the type.
        """

        path = decision.path or decision.basename
        entry: ManualImportFile = {
            "path": path,
            "seriesId": pending.series_id,
            "episodeIds": decision.episode_ids,
            "releaseGroup": pending.release_group,
            "downloadId": pending.infohash,
            "languages": lang_objs,
        }

        base = os.path.basename(path)
        sonarr_axes = quality_axes_from_model(decision.quality)
        our_axes = parse_quality_from_filename(base)
        default_axes = quality_axes_from_name(self._config.imports.default_quality, quality_defs)
        quality = resolve_quality(
            sonarr_axes,
            our_axes,
            default_axes,
            quality_defs,
            decision.quality,
        )
        entry["quality"] = quality
        resolved = quality.get("quality") or {}
        if QualitySource.parse(resolved.get("source")) is None:
            self.logger.warning(
                indent_string(
                    f"{label}: could not confidently resolve quality for {base}; importing as Unknown (re-grab risk)",
                ),
            )
        return entry
