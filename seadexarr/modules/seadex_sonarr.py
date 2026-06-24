import os
import time
from datetime import datetime, timedelta
from xml.etree import ElementTree

from . import coverage as _coverage
from .anilist import (
    get_anilist_format,
    get_anilist_n_eps,
)
from .cache import UPDATED_AT_STR_FORMAT, CacheRecord, record_is_fresh
from .config import Arr
from .log import EntryState, indent_string
from .manual_import import (
    ImportWaitMode,
    PendingImport,
    assign_episode_ids,
    build_episode_id_map,
    derive_languages,
    parse_quality_from_filename,
    resolve_language_objects,
    resolve_quality_model,
    select_quality,
)
from .mappings import MappingEntry, MappingMode
from .planner import get_episode_keys
from .protocols import ArrSync
from .radarr_client import (
    IdField,
    collect_anime_items,
    collect_anime_movies,
    make_radarr_client,
)
from .seadex_arr import RunDeps, SeaDexArr
from .seadex_types import (
    ArrReleaseDict,
    EpisodeRecord,
    SeadexDict,
    SonarrEpisode,
    SonarrItem,
    TvdbMappings,
)
from .sonarr_client import SonarrClient

TORRENT_FILENAMES_TO_SKIP = [
    "NCED",
    "NCOP",
    "Creditless Ending",
    "Creditless Opening",
    "Creditless ED",
    "Creditless OP",
]

# File extensions that never map to an episode (subtitles, fonts, chapters,
# metadata, images, samples, ...). We skip these before querying Sonarr so we
# don't waste a round-trip on them. This is deliberately a deny-list rather than
# an allow-list of video extensions: the cost of missing one here is a single
# harmless API call (Sonarr just returns no episode), whereas an allow-list that
# omits an unusual container would silently drop a real episode.
NON_VIDEO_EXTENSIONS = {
    ".ass",
    ".srt",
    ".ssa",
    ".sub",
    ".idx",
    ".sup",
    ".vtt",
    ".nfo",
    ".txt",
    ".md",
    ".sfv",
    ".xml",
    ".json",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".torrent",
    ".url",
    ".rar",
    ".zip",
    ".7z",
}

# How long a persisted Sonarr /parse result stays usable before it's re-queried.
# A filename's season/episode mapping is stable, but Sonarr's /parse depends on
# the current library, so a wrong-but-non-empty match could otherwise be trusted
# forever; re-validate monthly so such an entry self-heals.
SONARR_PARSE_CACHE_TTL_DAYS = 30


def _parse_anidb_mapping_dict(
    anidb_item: ElementTree.Element,
    tvdb_season: int,
) -> dict[int, dict[int, int]]:
    """Parse an AniDB anime element's ``mapping-list`` into a season -> map dict.

    Args:
        anidb_item (ElementTree.Element): A single AniDB ``anime`` element.
        tvdb_season (int): The TVDB season AniList resolved to; only mappings
            whose ``tvdbseason`` agrees are kept.

    Returns:
        dict[int, dict[int, int]]: ``{tvdbseason: {tvdb_episode: anidb_episode}}``.
            An empty ``mapping-list`` findall intentionally yields an empty dict
            (the loops simply don't run); a repeated season is last-wins.
    """

    result: dict[int, dict[int, int]] = {}

    for ms in anidb_item.findall("mapping-list"):
        for i in ms.findall("mapping"):

            # If there's no text, continue
            if not i.text:
                continue

            # Only match things if AniList and AniDB agree on the TVDB season
            anidb_tvdbseason = int(i.attrib["tvdbseason"])
            if anidb_tvdbseason != tvdb_season:
                continue

            # Split at semicolons, then at hyphens; orientation is {end: start}
            i_split = [x.split("-") for x in i.text.strip(";").split(";")]
            result[anidb_tvdbseason] = {int(x[1]): int(x[0]) for x in i_split}

    return result


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
    episode_sets = {}
    for rg, rg_item in seadex_dict.items():
        all_episodes = rg_item.all_episodes or []
        episode_sets[rg] = get_episode_keys(all_episodes)

    release_groups = list(episode_sets.keys())
    for i, rg1 in enumerate(release_groups):
        for rg2 in release_groups[i + 1:]:

            # If either release hasn't been parsed, then we can't rule out an
            # overlap, so assume they overlap
            if len(episode_sets[rg1]) == 0 or len(episode_sets[rg2]) == 0:
                return True

            # Otherwise they overlap if they share any episode
            if episode_sets[rg1] & episode_sets[rg2]:
                return True

    return False


def check_ep_by_anime_ids(
    ep: SonarrEpisode,
    tvdb_season: int,
) -> bool:
    """Check whether to include an episode by Anime ID style

    Args:
        ep (dict): Dictionary of episode info
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
        ep (dict): Sonarr episode info (seasonNumber, episodeNumber)
        tvdb_mappings (dict): season (int) -> list of inclusive (start, end)
            TVDB episode ranges. An empty list matches the whole season; an
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


# Substrings in a manual-import rejection reason that mean "don't import this
# file": a sample, an already-imported file, or a file that already exists.
# Matched case-insensitively against each rejection's reason/message text.
_REJECTION_SKIP_TOKENS = ("sample", "already", "exist")


def _candidate_is_rejected(candidate: dict) -> bool:
    """True if a manual-import candidate is a sample / already-imported file.

    Best-effort: scans the candidate's ``rejections`` for a reason/message whose
    text contains any skip token (``sample``/``already``/``exist``), case
    insensitively. A rejection shape varies by Sonarr version (a bare string, or
    a dict with ``reason``/``message``), so both are handled.

    Args:
        candidate (dict): A raw ManualImportResource dict (reads ``rejections``).
    """

    for rejection in candidate.get("rejections") or []:
        if isinstance(rejection, str):
            text = rejection
        elif isinstance(rejection, dict):
            text = f"{rejection.get('reason', '')} {rejection.get('message', '')}"
        else:
            continue
        lowered = text.casefold()
        if any(token in lowered for token in _REJECTION_SKIP_TOKENS):
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
        self.anime_mappings = deps.mappings.anime_mappings
        self.anidb_mappings = deps.mappings.anidb_mappings
        self.anibridge = deps.mappings.anibridge
        self.cache_store = deps.cache_store
        self.log_fmt = deps.log_fmt
        # The AniList gateway owns al_cache; the strategy reads/reassigns it
        # directly through self._anilist.al_cache while resolving episode
        # counts/formats (the gateway is the single owner, shared with the engine).
        self._anilist = deps.anilist

        # Set up Sonarr
        sonarr_url = self._config.sonarr_url
        sonarr_api_key = self._config.sonarr_api_key

        # self.session (a shared keep-alive requests.Session) comes from the
        # injected deps and is handed to the client; parse in particular fires one
        # request per file, so reusing it removes a per-file handshake.
        self.sonarr = SonarrClient(
            url=sonarr_url,
            api_key=sonarr_api_key,
            session=self.session,
            logger=self.logger,
        )

        # Per-run cache of the raw Sonarr episode fetch, keyed by series id. A
        # multi-season series maps to several AniList ids, each of which would
        # otherwise re-fetch the same whole-series episode list; cache it for the
        # run so the network round-trip happens once per series. Cleared at the
        # top of each run (in get_items, the run-start hook).
        self._ep_list_cache: dict[int, list[SonarrEpisode]] = {}

        # Per-run caches of the Sonarr quality-definition / language lists, used
        # to resolve a quality name / language names into the manual-import
        # payload objects. Fetched lazily on the first import and then reused for
        # the rest of the run so repeated imports don't re-hit the endpoints;
        # None means "not yet fetched" (cleared in get_items, the run-start hook).
        self._quality_defs_cache: list[dict] | None = None
        self._languages_cache: list[dict] | None = None

        self.ignore_movies_in_radarr = self._config.ignore_movies_in_radarr

        # Only when ignore_movies_in_radarr is on do we need Radarr's movie list
        # (for the specials cross-check in process_al_id). Build a lightweight
        # RadarrClient and reuse the already-built shared mappings - no nested
        # SeaDexRadarr (which would re-run the whole engine __init__: mapping
        # parse, cache load, and a qBittorrent login, all unused here).
        self.all_radarr_movies = None
        radarr_url = self._config.radarr_url_optional
        radarr_api_key = self._config.radarr_api_key_optional

        if (
            self.ignore_movies_in_radarr
            and radarr_url is not None
            and radarr_api_key is not None
        ):
            radarr_client = make_radarr_client(
                url=radarr_url,
                api_key=radarr_api_key,
                session=self.session,
                logger=self.logger,
            )
            self.all_radarr_movies = collect_anime_movies(
                radarr_client,
                self.anime_mappings,
                self.anibridge,
            )

    # --- ArrSync hooks ------------------------------------------------------

    def get_items(self) -> list[SonarrItem]:
        """Every Sonarr series with AniList mapping info.

        Also the run-start hook: drop any episode lists cached from a previous
        run so a fresh run always re-reads the current Sonarr library (this is
        called once, before the per-item loop).
        """

        self._ep_list_cache = {}
        self._quality_defs_cache = None
        self._languages_cache = None
        return self.get_all_sonarr_series()

    def filter_to_single(self, items: list[SonarrItem], item_id: int) -> list[SonarrItem]:
        """Narrow the series list to a single TVDB ID."""

        filtered = [s for s in items if s.tvdbId == item_id]
        if len(filtered) == 0:
            self.logger.warning(
                f"No anime series with TVDB ID {item_id} found in Sonarr",
            )
        return filtered

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

    def process_al_id(
        self,
        arr: Arr,
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
            arr,
            al_id,
            sd_entry,
            sd_url,
            lambda: _coverage.coverage_string(
                _coverage.episodes_from_ep_list(
                    self.get_ep_list(
                        sonarr_series_id=sonarr_series_id,
                        al_id=al_id,
                        mapping=mapping,
                    ),
                ),
            ),
        ):
            return False

        # Also check if it's in the Radarr cache, if we have that option
        if self.ignore_movies_in_radarr and not self._config.ignore_seadex_update_times:
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
        if (
            self.ignore_movies_in_radarr
            and self.all_radarr_movies is not None
        ):

            radarr_movies = []

            # Make sure these are flagged as specials since sometimes shows and
            # movies are all lumped together
            mapping_season = mapping.tvdb_season
            if mapping_season == 0:

                mapping_tmdb_id = mapping.tmdb_movie_id
                mapping_imdb_id = mapping.imdb_id

                for m in self.all_radarr_movies:

                    # Check by TMDB IDs
                    if (
                        mapping_tmdb_id is not None
                        and m.tmdbId == mapping_tmdb_id
                        and m not in radarr_movies
                    ):
                        radarr_movies.append(m)

                    # Check by IMDb IDs
                    if (
                        mapping_imdb_id is not None
                        and m.imdbId == mapping_imdb_id
                        and m not in radarr_movies
                    ):
                        radarr_movies.append(m)

            if len(radarr_movies) > 0:

                for movie in radarr_movies:
                    run.log_entry_status(
                        EntryState.IN_RADARR,
                        movie.title,
                    )

                time.sleep(self._config.sleep_time)
                return False

        # Get the episode list for all relevant episodes
        ep_list = self.get_ep_list(
            sonarr_series_id=sonarr_series_id,
            al_id=al_id,
            mapping=mapping,
        )

        if ep_list is None:
            return False

        # If all episodes are unmonitored, then skip if ignore_unmonitored is switched on
        ep_list_monitored = [ep.monitored for ep in ep_list]
        if not any(ep_list_monitored) and self._config.ignore_unmonitored:
            run.log_anilist_item_unmonitored(
                item_title=anilist_title,
            )
            time.sleep(self._config.sleep_time)
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

        sonarr_release_dict = self.get_sonarr_release_dict(ep_list=ep_list)
        sonarr_release_groups = list(sonarr_release_dict.keys())

        self.logger.debug(
            indent_string(
                f"Sonarr release group(s): {', '.join(str(rg) for rg in sonarr_release_groups)}",
            ),
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(arr, al_id, cache_details)

        self.logger.debug(
            indent_string(
                f"SeaDex: {', '.join(seadex_dict)}",
            ),
        )

        # Parse out filenames and check for overlaps
        seadex_dict = self.parse_episodes_from_seadex(seadex_dict=seadex_dict)
        overlapping_results = get_overlapping_results(seadex_dict=seadex_dict)

        # If we're in interactive mode and there are multiple equivalent options here, then select
        if self._config.interactive and len(seadex_dict) > 1 and overlapping_results:
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )

        # Filter downloads by whether the episodes in each torrent match the release
        # group we have in Sonarr
        torrent_hashes, seadex_dict = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr=arr,
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
            )

        return run.grab_and_cache(
            arr=arr,
            al_id=al_id,
            item_title=item_title,
            anilist_title=anilist_title,
            sd_url=sd_url,
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            cache_details=cache_details,
            release_group=sonarr_release_groups,
            pending_seeds=pending_seeds,
        )

    def _build_pending_seeds(
        self,
        *,
        seadex_dict: SeadexDict,
        ep_list: list[SonarrEpisode],
        sonarr_series_id: int,
        anilist_title: str,
    ) -> dict[str, PendingImport]:
        """Build ``infohash -> PendingImport`` for every release marked to grab.

        For each downloadable url with a hash, map its SeaDex files to
        authoritative Sonarr episode ids via the cached ``/parse`` results and
        the ``(season, episode) -> id`` index, so the eventual manual import can
        override every field Sonarr would otherwise blind-parse.

        Args:
            seadex_dict (SeadexDict): The filtered releases; ``url_item.download``
                marks the ones the engine will add.
            ep_list (list[SonarrEpisode]): The relevant Sonarr episodes (carry ids).
            sonarr_series_id (int): The Sonarr series id the files belong to.
            anilist_title (str): Display title for the record (logging only).

        Returns:
            dict[str, PendingImport]: Seeds keyed by infohash (empty when nothing
            is downloadable / parseable).
        """

        ep_id_map = build_episode_id_map(ep_list)
        parse_cache: dict = self.cache_store.data.get("sonarr_parse_cache", {})
        added_at = datetime.now().strftime(UPDATED_AT_STR_FORMAT)

        pending_seeds: dict[str, PendingImport] = {}

        for srg, srg_item in seadex_dict.items():
            for url_item in srg_item.urls.values():

                if not (url_item.download and url_item.hash):
                    continue

                file_episode_map: dict[str, list[int]] = {}
                flat_ids: list[int] = []
                seasons: set[int] = set()

                for seadex_file in url_item.files:
                    f = os.path.basename(seadex_file)

                    record = parse_cache.get(f)
                    if not record:
                        continue

                    file_ids: list[int] = []
                    for ep in record.get("episodes", []):
                        season = ep.get("season")
                        episode = ep.get("episode")
                        ep_id = ep_id_map.get((season, episode))
                        if ep_id is None:
                            continue
                        file_ids.append(ep_id)
                        flat_ids.append(ep_id)
                        if season is not None:
                            seasons.add(season)

                    if file_ids:
                        file_episode_map[f] = file_ids

                # Nothing mapped to a real episode id: persisting a record here
                # would only re-poll a never-importable download every run until
                # the TTL drops it, so skip it (the release is still grabbed).
                if not (file_episode_map or flat_ids):
                    continue

                season_number = seasons.pop() if len(seasons) == 1 else None

                pending_seeds[url_item.hash] = PendingImport(
                    infohash=url_item.hash,
                    series_id=sonarr_series_id,
                    file_episode_map=file_episode_map,
                    episode_ids=flat_ids,
                    release_group=srg,
                    is_dual_audio=url_item.is_dual_audio,
                    season_number=season_number,
                    seadex_files=[os.path.basename(f) for f in url_item.files],
                    title=anilist_title,
                    added_at=added_at,
                )

        return pending_seeds

    def import_completed(self, pending: PendingImport, content_path: str) -> bool:
        """Drive the series-pinned manual import for one completed download.

        Asks Sonarr for the manual-import candidates under ``content_path``
        (pinned to ``pending.series_id`` so the parse runs in the context of the
        known series), then builds a payload that overrides every field we have
        authoritative data for - series id, episode ids, release group, download
        id - and layers the quality (ours -> Sonarr's in-context -> configured
        default) and languages (dual vs. single). Files Sonarr rejects as a
        sample / already-imported are skipped; files we can't confidently map to
        an episode are skipped.

        Returns True only when the ``ManualImport`` command was queued (so the
        engine may drop the pending record); False to leave it pending for a
        later retry.

        Args:
            pending (PendingImport): The durable record for the completed torrent.
            content_path (str): The qBittorrent ``content_path`` of the finished
                download (the folder/file the manual import reads from disk).
        """

        candidates = self.sonarr.manual_import_candidates(
            folder=content_path,
            series_id=pending.series_id,
            season_number=pending.season_number,
            filter_existing_files=False,
        )
        if not candidates:
            self.logger.info(
                indent_string(
                    f"No manual-import candidates for {pending.title or pending.infohash}; "
                    f"leaving pending",
                ),
            )
            return False

        # Lazily fetch + cache the quality-definition / language lists for the
        # run so repeated imports don't re-hit these endpoints.
        if self._quality_defs_cache is None:
            self._quality_defs_cache = self.sonarr.quality_definitions()
        if self._languages_cache is None:
            self._languages_cache = self.sonarr.languages()
        quality_defs = self._quality_defs_cache
        lang_defs = self._languages_cache

        assigned = assign_episode_ids(
            [os.path.basename(c["path"]) for c in candidates if c.get("path")],
            pending.file_episode_map,
            pending.episode_ids,
        )

        lang_names = derive_languages(
            pending.is_dual_audio,
            self._config.import_languages_dual,
            self._config.import_languages_single,
        )
        lang_objs = resolve_language_objects(lang_names, lang_defs)

        files: list[dict] = []
        for c in candidates:
            path = c.get("path")
            if not path:
                continue
            base = os.path.basename(path)

            ep_ids = assigned.get(base)
            if not ep_ids:
                self.logger.info(
                    indent_string(
                        f"Skipping {base}: no authoritative episode mapping",
                    ),
                )
                continue

            if _candidate_is_rejected(c):
                self.logger.info(
                    indent_string(
                        f"Skipping {base}: rejected by Sonarr (sample/already imported)",
                    ),
                )
                continue

            file_entry: dict = {
                "path": path,
                "seriesId": pending.series_id,
                "episodeIds": ep_ids,
                "releaseGroup": pending.release_group,
                "downloadId": pending.infohash,
                "languages": lang_objs,
            }

            our_q = parse_quality_from_filename(base)
            sel = select_quality(
                our_q, c.get("quality"), self._config.import_default_quality,
            )
            if sel.model is not None:
                file_entry["quality"] = sel.model
            elif sel.name is not None:
                quality_model = resolve_quality_model(sel.name, quality_defs)
                if quality_model is not None:
                    file_entry["quality"] = quality_model
                else:
                    self.logger.warning(
                        indent_string(
                            f"Could not resolve quality '{sel.name}' for {base}; "
                            f"importing without an explicit quality",
                        ),
                    )
            else:
                self.logger.warning(
                    indent_string(
                        f"Unknown quality for {base}; importing without an "
                        f"explicit quality (re-grab risk)",
                    ),
                )

            files.append(file_entry)

        if not files:
            self.logger.info(
                indent_string(
                    f"Nothing importable for {pending.title or pending.infohash}; "
                    f"leaving pending",
                ),
            )
            return False

        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=self._config.import_mode,
        )
        if cmd_id is None:
            return False

        self.logger.info(
            indent_string(
                f"Queued manual import of {len(files)} file(s) for "
                f"{pending.title or pending.infohash} (command {cmd_id})",
            ),
        )
        return True

    # --- Sonarr domain logic ------------------------------------------------

    def get_all_sonarr_series(self) -> list[SonarrItem]:
        """Get all series in Sonarr with AniList mapping info"""

        return collect_anime_items(
            self.sonarr.all_series,
            self.anime_mappings,
            (IdField("tvdb_id", "tvdbId"), IdField("imdb_id", "imdbId")),
            (
                self.anibridge.all_tvdb_ids if self.anibridge else set(),
                self.anibridge.all_imdb_ids if self.anibridge else set(),
            ),
        )

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
            final_ep_list = [
                ep
                for ep in ep_list
                if check_ep_by_anime_ids(ep=ep, tvdb_season=tvdb_season)
            ]
        else:
            tvdb_mappings = mapping.tvdb_mappings or {}
            final_ep_list = [
                ep
                for ep in ep_list
                if check_ep_by_anibridge(ep=ep, tvdb_mappings=tvdb_mappings)
            ]

        # For OVAs and movies, the offsets can often be wrong, so if we have specific mappings
        # then take that into account here
        al_format, self._anilist.al_cache = get_anilist_format(
            al_id,
            al_cache=self._anilist.al_cache,
        )

        # Potentially pull out a bunch of mappings from AniDB. These should
        # be for anything not marked as TV, and specials as marked by
        # being in Season 0
        anidb_mapping_dict: dict[int, dict[int, int]] = {}
        if (
            self.anidb_mappings is not None
            and anidb_id is not None
            and (al_format not in ["TV"] or tvdb_season == 0)
        ):
            anidb_item = self._mappings.anidb_anime_by_id(anidb_id)

            # If we don't find anything, no worries. If we find multiple, worries
            if len(anidb_item) > 1:
                raise ValueError(
                    "Multiple AniDB mappings found. This should not happen!",
                )

            # We want things with mapping lists in, since more regular
            # mappings will have already been picked up
            if len(anidb_item) == 1:
                anidb_mapping_dict = _parse_anidb_mapping_dict(
                    anidb_item[0],
                    tvdb_season,
                )

        # Prefer the AniDB mapping dict over any offsets
        if len(anidb_mapping_dict) > 0:
            anidb_final_ep_list = []

            # See if we have the mapping for each entry
            for ep in final_ep_list:

                season_number = ep.season_number
                episode_number = ep.episode_number
                if season_number is None or episode_number is None:
                    continue

                anidb_mapping_dict_entry = anidb_mapping_dict.get(
                    season_number, {},
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
            return [
                ep
                for ep in final_ep_list
                if 1 <= (ep.episode_number or 0) - ep_offset <= n_eps
            ]

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

    @staticmethod
    def _sonarr_parse_is_fresh(record: dict | None, cutoff: datetime) -> bool:
        """True if a persisted parse record has episodes and is within TTL

        Legacy list-form entries (pre-TTL, no timestamp) are treated as stale so
        they are re-queried once and upgraded to the timestamped form. ``cutoff``
        is computed once per call to :meth:`parse_episodes_from_seadex` and threaded
        in, so ``datetime.now()`` isn't recomputed for every SeaDex file.
        """
        return record_is_fresh(
            record,
            payload_key="episodes",
            ttl_days=SONARR_PARSE_CACHE_TTL_DAYS,
            cutoff=cutoff,
        )

    def parse_episodes_from_seadex(
        self,
        seadex_dict: SeadexDict,
    ) -> dict:
        """For files in a SeaDex release, parse this through Sonarr to get season/episode numbers

        This gets an overall episode list per-release group, and also episode lists per-torrent,
        if there are multiple

        Parsed filenames are cached (in memory and persisted to cache.json), so a
        given filename is only ever sent to Sonarr once - both within a run, where
        the same file can appear across overlapping release groups, and across
        runs. The mapping is deterministic for a SeaDex release name, so this is
        safe; only successful parses are cached, so a file becomes parseable as
        soon as its series is added to Sonarr.

        Args:
            seadex_dict (dict): Dictionary of seadex releases
        """

        # filename -> {"fetched_at": <str>, "episodes": [{"season", "episode"}]},
        # shared across runs via cache.json; fetched_at lets entries expire (TTL)
        parse_cache = self.cache_store.data.setdefault("sonarr_parse_cache", {})
        now_str = datetime.now().strftime(UPDATED_AT_STR_FORMAT)
        # Compute the TTL cutoff once for the whole run of files rather than
        # re-deriving datetime.now() in the per-file freshness check below.
        cutoff = datetime.now() - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS)

        for release_group_item in seadex_dict.values():

            # Set up an overall "all episodes" list (bound locally so the
            # appends below stay typed as list, not list | None)
            all_episodes: list[EpisodeRecord] = []
            release_group_item.all_episodes = all_episodes

            for url_item in release_group_item.urls.values():

                # Set up a list to parse episodes from files
                episodes: list[EpisodeRecord] = []
                url_item.episodes = episodes
                sizes = url_item.size

                for sd_file_idx, seadex_file in enumerate(url_item.files):

                    # Get basename from the file
                    f = os.path.basename(seadex_file)

                    # Skip filenames with things like "NCED", "NCOP"
                    if any(x in f for x in TORRENT_FILENAMES_TO_SKIP):
                        continue

                    # Skip non-video files (subtitles, fonts, images, ...) before
                    # hitting Sonarr - they never resolve to an episode
                    if os.path.splitext(f)[1].lower() in NON_VIDEO_EXTENSIONS:
                        continue

                    # Use the cached parse if it's still fresh, otherwise query
                    # Sonarr and remember the result with a timestamp so it
                    # expires (re-validates) rather than being trusted forever
                    record = parse_cache.get(f)
                    if self._sonarr_parse_is_fresh(record, cutoff):
                        parsed = record["episodes"]
                    else:
                        parsed = self.sonarr.parse(f)

                        if len(parsed) == 0:
                            self.logger.debug(
                                indent_string(
                                    f"Sonarr could not parse episode for {f}",
                                ),
                            )
                            # Deliberately not cached: a miss may just mean the
                            # series isn't in Sonarr yet
                            continue

                        parse_cache[f] = {"fetched_at": now_str, "episodes": parsed}

                    size = sizes[sd_file_idx]
                    for ep in parsed:

                        season = ep["season"]
                        episode = ep["episode"]

                        self.logger.debug(
                            indent_string(
                                f"{f} mapped to: S{season:02d}E{episode:02d}",
                            ),
                        )

                        episodes.append(
                            EpisodeRecord(
                                season=season,
                                episode=episode,
                                size=size,
                            ),
                        )
                        all_episodes.append(
                            EpisodeRecord(
                                season=season,
                                episode=episode,
                                size=size,
                            ),
                        )

        return seadex_dict
