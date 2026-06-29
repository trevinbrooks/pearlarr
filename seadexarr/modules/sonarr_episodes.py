"""Sonarr episode-domain collaborator: series enumeration + episode resolution.

Extracted from :class:`~.seadex_sonarr.SonarrSync` so the strategy keeps only the
``ArrSync`` hooks. ``SonarrEpisodes`` owns the per-run episode cache (and the
series-id fingerprint that pins negative ``/parse`` records) and the logic that
turns a (series, AniList id, mapping) into the relevant episode list: season /
AniBridge filtering, AniDB remaps, offset slicing, the existing-file release dict,
and the concurrent prefetch warm.
"""

import concurrent.futures
import hashlib
from collections.abc import Iterable

from . import coverage as _coverage
from .anilist import get_anilist_format, get_anilist_n_eps
from .config import AppConfig
from .mappings import MappingEntry, MappingMode, MappingSource
from .protocols import EpisodeProgress
from .radarr_client import IdField, collect_anime_items
from .seadex_arr import RunDeps, SeaDexArr
from .seadex_types import (
    ArrReleaseDict,
    SonarrEpisode,
    SonarrItem,
    TvdbMappings,
)
from .sonarr_client import SonarrClient

# Bounded concurrency for the episode/parse network fan-out. Only used when
# advanced.sleep_time == 0; must stay <= the session's pool_maxsize (RunDeps.build).
# Episode fetches are I/O-bound, but a typical Sonarr (often behind a reverse
# proxy) saturates around ~10-12 concurrent, so larger values don't help.
SONARR_FETCH_WORKERS = 12


def fetch_workers(config: AppConfig) -> int:
    """Concurrency for the episode / parse network fan-out.

    Sequential (1) whenever a rate-limit throttle is requested
    (``advanced.sleep_time > 0``), so the bounded pool never bypasses the user's
    intended throttle; otherwise the bounded pool size. Shared by the episode
    prefetch and the parse warm, so it lives at module scope, owned by neither.
    """

    if config.advanced.sleep_time > 0:
        return 1
    return SONARR_FETCH_WORKERS


def sonarr_series_fingerprint(series_ids: Iterable[int]) -> str:
    """Stable fingerprint of the current Sonarr series-id set.

    Invalidates negative ``/parse`` records: an empty parse almost always means
    the file's series isn't present, so a new series id flips the fingerprint and
    re-parses the affected entries. ``hashlib`` (not ``hash()``) for stability
    across processes.

    Args:
        series_ids (Iterable[int]): Current Sonarr series ids (sorted/de-duped).
    """

    joined = ",".join(str(i) for i in sorted(set(series_ids)))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def check_ep_by_anime_ids(
    ep: SonarrEpisode,
    tvdb_season: int,
) -> bool:
    """Check whether to include an episode by Anime ID style

    Args:
        ep (SonarrEpisode): Episode info
        tvdb_season (int): TVDB season number
    """

    # First, check by season
    season_number = ep.season_number

    # If the TVDB season is -1, this is anything but specials
    if tvdb_season == -1 and season_number == 0:
        return False

    # Else, if we have a season defined, and it doesn't match, don't include
    return not (tvdb_season != -1 and season_number != tvdb_season)


def check_ep_by_anibridge(
    ep: SonarrEpisode,
    tvdb_mappings: TvdbMappings,
) -> bool:
    """Check whether a Sonarr episode is covered by an AniBridge mapping.

    Args:
        ep (SonarrEpisode): Sonarr episode info (season_number, episode_number)
        tvdb_mappings (TvdbMappings): season (int) -> list of inclusive (start,
            end) TVDB episode ranges. An empty list matches the whole season; an
            end of None is open-ended.
    """

    ep_season = ep.season_number if ep.season_number is not None else -1
    ep_episode = ep.episode_number if ep.episode_number is not None else -1

    ranges = tvdb_mappings.get(ep_season)

    # Season isn't part of this mapping at all
    if ranges is None:
        return False

    # No explicit episode ranges -> the whole season is covered
    if not ranges:
        return True

    for start, end in ranges:
        if end is None:
            if ep_episode >= start:
                return True
        elif start <= ep_episode <= end:
            return True

    return False


class SonarrEpisodes:
    """Owns the per-run episode cache + the (series, al_id, mapping) -> episodes logic.

    Constructed once per run in :class:`~.seadex_sonarr.SonarrSync` from the shared
    :class:`~.seadex_arr.RunDeps`, the strategy's Sonarr client, and the run
    machinery (held as ``self._services`` for the prefetch needs-scan gate).
    """

    def __init__(self, deps: RunDeps, sonarr: SonarrClient, services: SeaDexArr) -> None:
        """Bind the shared collaborators the episode logic reads.

        Args:
            deps (RunDeps): The shared collaborators (config/mappings/AniList
                gateway/log formatter are unpacked off it).
            sonarr (SonarrClient): The strategy's Sonarr client (built once,
                reused so a multi-season series fetches its episodes once).
            services (SeaDexArr): The run machinery; ``prefetch`` calls into it to
                resolve AniList ids and the needs-scan gate.
        """

        self.sonarr = sonarr
        self._services = services
        self._config = deps.config
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge
        # The AniList gateway owns al_cache; we read/reassign it directly through
        # self._anilist.al_cache while resolving episode counts/formats. The
        # gateway object is the single owner (shared with the engine, which never
        # re-binds it), so reassigning .al_cache here stays visible everywhere.
        self._anilist = deps.anilist
        self.log_fmt = deps.log_fmt

        # Per-run cache of the raw Sonarr episode fetch, keyed by series id. A
        # multi-season series maps to several AniList ids, each of which would
        # otherwise re-fetch the same whole-series episode list; cache it for the
        # run so the network round-trip happens once per series. Reset at run start
        # (collect_series, driven by the strategy's get_items hook).
        self._ep_list_cache: dict[int, list[SonarrEpisode]] = {}

        # Fingerprint of the current Sonarr series-id set, recomputed each run in
        # collect_series. Pins negative ``/parse`` cache records to the library
        # state so they self-heal when a missing series is added.
        self._series_fp: str = ""

    @property
    def series_fp(self) -> str:
        """The current run's series-id fingerprint (read by the parse cache)."""

        return self._series_fp

    def collect_series(self) -> list[SonarrItem]:
        """Enumerate the AniList-mapped Sonarr series; the run-start reset point.

        Drops the per-run episode cache and re-fingerprints the series-id set so a
        fresh run always re-reads the current library. Called once per run from the
        strategy's ``get_items`` hook.
        """

        self._ep_list_cache = {}
        series = self.get_all_sonarr_series()
        # Fingerprint the current series-id set once per run for negative-parse
        # cache invalidation (see sonarr_series_fingerprint).
        self._series_fp = sonarr_series_fingerprint(s.id for s in series)
        return series

    def get_all_sonarr_series(self) -> list[SonarrItem]:
        """Get all series in Sonarr with AniList mapping info"""

        fields = (IdField("tvdb_id", "tvdbId"), IdField("imdb_id", "imdbId"))
        return collect_anime_items(
            self.sonarr.all_series,
            fields,
            tuple(self._mappings.anime_id_set(f.mapping_key) for f in fields),
            tuple(self.anibridge.id_set(f.mapping_key) if self.anibridge else set() for f in fields),
        )

    def prefetch(self, items: list[SonarrItem], *, progress: EpisodeProgress | None = None) -> int:
        """Warm the per-series episode lists CONCURRENTLY before the scan loop.

        One sequential ``/api/v3/episode`` round-trip per processed series is the
        dominant sweep cost (~30s here). Fetching them up front over a bounded
        thread pool collapses that to a few seconds while keeping every list
        FRESH - the grab/skip decision reads each episode's existing file, so the
        lists are deliberately not persisted across runs. Only the network fetch
        runs in the pool; the in-memory ``_ep_list_cache`` is populated on the
        main thread (the cache and mappings are not thread-safe). A series whose
        fetch fails is left unwarmed and re-fetched (and logged) by
        ``get_ep_list`` on the main thread during the loop.

        Only series the per-id loop would actually process are warmed: a series
        whose every AniList id the loop short-circuits (no SeaDex entry, or cached
        and unchanged - ``al_id_needs_scan``) is skipped, since ``get_ep_list``
        would never run for it. This keeps a warm run from re-fetching the episode
        lists the loop's cached-entry skip already obviates.

        Args:
            items (list[SonarrItem]): The run's series list (already narrowed for
                a single-series run).
            progress (EpisodeProgress | None): Boot cockpit step fed per-series
                fraction + "done/total" detail as each fetch completes; None
                outside the cockpit.

        Returns:
            int: How many series were warmed (attempted) - the needs-scan subset,
            for the caller's ledger detail. A series whose fetch returned None
            still counts.
        """

        # Build the needs-scan subset (see docstring): skip series the per-id loop would short-circuit on.
        run = self._services
        series_ids: list[int] = []
        seen: set[int] = set()
        for item in items:
            if item.id in seen:
                continue
            if not item.monitored and self._config.sonarr.ignore_unmonitored:
                continue
            al_ids = run.get_anilist_ids(tvdb_id=item.tvdbId, imdb_id=item.imdbId, log_ignored=False)
            if not al_ids:
                continue
            if not any(run.al_id_needs_scan(aid) for aid in al_ids):
                continue
            seen.add(item.id)
            series_ids.append(item.id)

        if not series_ids:
            return 0

        def warm(series_id: int) -> tuple[int, list[SonarrEpisode] | None]:
            # quiet: a transient miss here isn't logged from a worker; get_ep_list
            # retries and logs it on the main thread if it still fails. A RAISE (not
            # just a None return) degrades to None too, so one bad series can't abort
            # the whole concurrent sweep (the docstring's per-series degradation).
            try:
                return series_id, self.sonarr.episodes(series_id, quiet=True)
            except Exception:
                return series_id, None

        # submit + as_completed (not pool.map): advance the bar as each series
        # FINISHES, so a slow series doesn't freeze the bar then jump. max_workers=1
        # (sleep_time throttle) still runs serially. Results are consumed on the main
        # thread, so the cache write and the progress drive are both single-threaded.
        total = len(series_ids)
        workers = min(fetch_workers(self._config), total)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(warm, sid) for sid in series_ids]
            for fut in concurrent.futures.as_completed(futures):
                sid, eps = fut.result()
                if eps is not None:
                    self._ep_list_cache[sid] = eps
                done += 1
                if progress is not None:
                    progress.progress(done / total, f"{done}/{total}")
        return total

    def episodes_for_series(self, series_id: int) -> list[SonarrEpisode]:
        """Fetch the series' episodes FRESH for each import poll.

        Import verification reads the episode files as the source of truth for
        "already imported", and that state changes as Sonarr (or our own manual
        import) places files. A per-run cache would go stale across the monitor's
        repeated polls and never observe the import landing - the record would time
        out as "still importing" (or, in ``move`` mode where the file leaves the
        download folder, never be confirmed at all). So this fetches fresh every
        call and refreshes the per-run cache for any later reader; a transient
        fetch failure falls back to the last-known list (or empty) so a later poll
        simply retries.
        """

        fetched = self.sonarr.episodes(series_id)
        if fetched is None:
            return self._ep_list_cache.get(series_id, [])
        self._ep_list_cache[series_id] = fetched
        return fetched

    def get_ep_list(
        self,
        sonarr_series_id: int,
        al_id: int,
        mapping: MappingEntry,
    ) -> list[SonarrEpisode] | None:
        """Get a list of relevant episodes for an AniList mapping

        Args:
            sonarr_series_id (int): Series ID in Sonarr
            al_id (int): Anilist ID
            mapping (MappingEntry): Mapping between TVDB and AniList
        """

        # If we have any season info, pull that out now
        tvdb_season = mapping.tvdb_season

        # Check we have a sensible AL ID
        if al_id == -1:
            raise ValueError("AniList ID not defined!")

        # Get the AniDB ID
        anidb_id = mapping.anidb_id

        # Check what kind of mode we're in here,
        # it's either AniBridge or Anime IDs
        mode = mapping.mode

        # Get all the episodes for the whole series. The fetch is per-series (not
        # per-AniList-id), so a multi-season series resolving to several ids would
        # otherwise re-request the identical list; cache it per series for the run
        # and only do the per-id filtering below on the shared, read-only list.
        ep_list = self._ep_list_cache.get(sonarr_series_id)
        if ep_list is None:
            ep_list = self.sonarr.episodes(sonarr_series_id)
            if ep_list is None:
                return None
            self._ep_list_cache[sonarr_series_id] = ep_list

        # Filter down here by various things. Resolve the include test once by
        # mode rather than re-branching on a string for every episode; the
        # comprehension preserves ep_list order, exactly as the append loop did.
        if mode is MappingMode.ANIME_IDS:
            if mapping.source is MappingSource.ANIBRIDGE:
                # Degraded AniBridge entry (imdb/tmdb-resolved, so no tvdb season
                # ranges): tvdb_season=-1 would otherwise grab the wrong episodes.
                # Skip; process_al_id surfaces NO_EPISODES + the AniBridge warning.
                return []
            final_ep_list = [ep for ep in ep_list if check_ep_by_anime_ids(ep=ep, tvdb_season=tvdb_season)]
        else:
            tvdb_mappings = mapping.tvdb_mappings or {}
            if not tvdb_mappings:
                # AniBridge attached no usable per-season ranges, so nothing resolves.
                # Return empty; process_al_id surfaces the visible NO_EPISODES skip.
                return []
            final_ep_list = [ep for ep in ep_list if check_ep_by_anibridge(ep=ep, tvdb_mappings=tvdb_mappings)]

        # For OVAs and movies, the offsets can often be wrong, so if we have specific mappings
        # then take that into account here
        al_format, self._anilist.al_cache = get_anilist_format(
            al_id,
            al_cache=self._anilist.al_cache,
        )

        # Potentially pull out a bunch of mappings from AniDB. These should
        # be for anything not marked as TV, and specials as marked by
        # being in Season 0. The resolver owns the AniDB parse now: it returns the
        # season's {tvdb_ep: anidb_ep} map ({} when none) and raises on an ambiguous
        # id (the former "multiple AniDB mappings found" case).
        anidb_mapping_dict: dict[int, dict[int, int]] = {}
        if self._mappings.has_anidb and anidb_id is not None and (al_format not in ["TV"] or tvdb_season == 0):
            anidb_mapping_dict = self._mappings.anidb_mapping_dict(anidb_id, tvdb_season)

        # Prefer the AniDB mapping dict over any offsets
        if len(anidb_mapping_dict) > 0:
            anidb_final_ep_list: list[SonarrEpisode] = []

            # See if we have the mapping for each entry
            for ep in final_ep_list:
                season_number = ep.season_number
                episode_number = ep.episode_number
                if season_number is None or episode_number is None:
                    continue

                anidb_mapping_dict_entry = anidb_mapping_dict.get(
                    season_number,
                    {},
                ).get(episode_number, None)
                if anidb_mapping_dict_entry is not None:
                    anidb_final_ep_list.append(ep)

            # These episodes are read-only from here on (coverage,
            # get_sonarr_release_dict, and the planner only read them), so we
            # return references into the shared cache rather than cloning.
            final_ep_list = anidb_final_ep_list

        # No AniDB mapping: anime-id mappings still need the offset slice, while
        # AniBridge mappings are already fully filtered above (no-op).
        elif mode is MappingMode.ANIME_IDS:
            final_ep_list = self._apply_anime_id_offsets(
                final_ep_list=final_ep_list,
                al_id=al_id,
                mapping=mapping,
                tvdb_season=tvdb_season,
            )

        return final_ep_list

    def _apply_anime_id_offsets(
        self,
        final_ep_list: list[SonarrEpisode],
        al_id: int,
        mapping: MappingEntry,
        tvdb_season: int,
    ) -> list[SonarrEpisode]:
        """Slice an anime-id episode list down by its TVDB offset / AniList count.

        Args:
            final_ep_list (list): Season-filtered episodes to slice.
            al_id (int): AniList ID, used to resolve the expected episode count.
            mapping (MappingEntry): SeaDex mapping (read for ``tvdb_epoffset``).
            tvdb_season (int): TVDB season (-1 means single-season offset slice).
        """

        # Slice the list to get the correct episodes, so any potential offsets
        ep_offset = mapping.tvdb_epoffset
        n_eps, self._anilist.al_cache = get_anilist_n_eps(
            al_id,
            al_cache=self._anilist.al_cache,
        )

        # If we don't get a number of episodes, use them all
        if n_eps is None:
            n_eps = len(final_ep_list) - ep_offset

        # Check that we're including this by the episode number. This only
        # works for single-seasons, so be careful!
        if tvdb_season != -1:
            return [ep for ep in final_ep_list if 1 <= (ep.episode_number or 0) - ep_offset <= n_eps]

        return final_ep_list[ep_offset : n_eps + ep_offset]

    def get_sonarr_release_dict(
        self,
        ep_list: list[SonarrEpisode],
    ) -> ArrReleaseDict:
        """Get a dictionary of useful info for a series in Sonarr

        Args:
            ep_list: List of Sonarr episodes
        """

        # Look through, get release groups from the existing Sonarr files
        # and note any potential missing files
        sonarr_release_dict: ArrReleaseDict = {}
        missing_eps = 0
        n_eps = len(ep_list)
        for ep in ep_list:
            if ep.episode_file_id == 0:
                missing_eps += 1
                continue

            release_group = ep.episode_file.release_group if ep.episode_file else None
            if release_group is None or release_group == "":
                continue

            size = ep.episode_file.size if ep.episode_file else None
            sonarr_release_dict.setdefault(release_group, []).append(size)

        if missing_eps > 0:
            # Show which episodes are missing as ranges (e.g. "S04 E12"), not just
            # a count, so it's clear what's absent. Fall back to the count if the
            # episodes can't be condensed.
            missing_coverage = _coverage.coverage_string(
                _coverage.episodes_from_ep_list(ep_list, missing_only=True),
            )
            self.log_fmt.detail(
                "missing",
                missing_coverage or f"{missing_eps}/{n_eps}",
                value_style="yellow",
            )

        return sonarr_release_dict
