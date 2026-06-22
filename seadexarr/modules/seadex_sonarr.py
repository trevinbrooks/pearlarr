import copy
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import arrapi.exceptions
from arrapi import SonarrAPI

from .anilist import (
    get_anilist_format,
    get_anilist_n_eps,
)
from .log import indent_string
from .seadex_arr import UPDATED_AT_STR_FORMAT, SeaDexArr, get_episode_keys
from .seadex_radarr import SeaDexRadarr

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


def get_tvdb_id(mapping: dict) -> int | None:
    """Get TVDB ID for a particular mapping

    Args:
        mapping (dict): Dictionary of SeaDex mappings

    Returns:
        int: TVDB ID
    """

    return mapping.get("tvdb_id")


def get_tvdb_season(mapping: dict) -> int:
    """Get TVDB season for a particular mapping

    Args:
        mapping (dict): Dictionary of SeaDex mappings

    Returns:
        int: TVDB season
    """

    return mapping.get("tvdb_season", -1)


def get_overlapping_results(seadex_dict: dict) -> bool:
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
        all_episodes = rg_item.get("all_episodes", [])
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
    ep: dict,
    tvdb_season: int,
) -> bool:
    """Check whether to include an episode by Anime ID style

    Args:
        ep (dict): Dictionary of episode info
        tvdb_season (int): TVDB season number
    """

    include_episode = True

    # First, check by season
    season_number = ep.get("seasonNumber")

    # If the TVDB season is -1, this is anything but specials
    if tvdb_season == -1 and season_number == 0:
        include_episode = False

    # Else, if we have a season defined, and it doesn't match, don't include
    elif tvdb_season != -1 and season_number != tvdb_season:
        include_episode = False

    return include_episode


def check_ep_by_anibridge(
    ep: dict,
    tvdb_mappings: dict,
) -> bool:
    """Check whether a Sonarr episode is covered by an AniBridge mapping.

    Args:
        ep (dict): Sonarr episode info (seasonNumber, episodeNumber)
        tvdb_mappings (dict): season (int) -> list of inclusive (start, end)
            TVDB episode ranges. An empty list matches the whole season; an
            end of None is open-ended.
    """

    ep_season = ep.get("seasonNumber", -1)
    ep_episode = ep.get("episodeNumber", -1)

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


class SeaDexSonarr(SeaDexArr):

    def __init__(
        self,
        config: str = "config.yml",
        cache: str = "cache.json",
        logger: logging.Logger | None = None,
    ) -> None:
        """Sync Sonarr instance with SeaDex

        Args:
            config (str, optional): Path to config file.
                Defaults to "config.yml".
            cache (str, optional): Path to cache file.
                Defaults to "cache.json".
            logger. Logging instance. Defaults to None,
                which will create one.
        """

        SeaDexArr.__init__(
            self,
            arr="sonarr",
            config=config,
            cache=cache,
            logger=logger,
        )

        # Set up Sonarr
        self.sonarr_url = self.config.get("sonarr_url", None)
        if not self.sonarr_url:
            raise ValueError(f"sonarr_url needs to be defined in {config}")

        self.sonarr_api_key = self.config.get("sonarr_api_key", None)
        if not self.sonarr_api_key:
            raise ValueError(f"sonarr_api_key needs to be defined in {config}")

        self.sonarr = SonarrAPI(
            url=self.sonarr_url,
            apikey=self.sonarr_api_key,
        )

        # self.session (a shared keep-alive requests.Session) is inherited from
        # SeaDexArr.__init__ above. parse_episodes_from_seadex in particular
        # fires one request per file, so reusing it removes a per-file handshake.

        self.ignore_movies_in_radarr = self.config.get("ignore_movies_in_radarr", False)

        # Only when ignore_movies_in_radarr is on do we need a Radarr instance,
        # and only then to fetch its movie list for the specials cross-check in
        # run(). Building it otherwise would re-run the whole base __init__
        # (mapping parse + index + cache load) plus a Radarr movie fetch, all of
        # which would then go unused - so gate the construction on the flag.
        self.radarr = None
        self.all_radarr_movies = None
        radarr_url = self.config.get("radarr_url", None)
        radarr_api_key = self.config.get("radarr_api_key", None)

        if (
            self.ignore_movies_in_radarr
            and radarr_url is not None
            and radarr_api_key is not None
        ):
            self.radarr = SeaDexRadarr(
                config=config,
                logger=logger,
            )
            self.all_radarr_movies = self.radarr.get_all_radarr_movies()

    def close(self) -> None:
        super().close()
        if self.radarr is not None:
            self.radarr.close()

    def run(self, tvdb_id: int | None = None, dry_run: bool = False) -> bool:
        """Run the SeaDex Sonarr Syncer

        Args:
            tvdb_id (int, optional): If set, only run for the series with this
                TVDB ID. Defaults to None, which runs for all series.
            dry_run (bool, optional): If True, simulate the run without grabbing
                torrents, writing the cache, or sending notifications.
                Defaults to False.
        """

        return self.run_sync(arr="sonarr", item_id=tvdb_id, dry_run=dry_run)

    def _get_all_items(self) -> list:
        """Every Sonarr series with AniList mapping info."""

        return self.get_all_sonarr_series()

    def _filter_to_single_item(self, items: list, item_id: int) -> list:
        """Narrow the series list to a single TVDB ID."""

        filtered = [s for s in items if s.tvdbId == item_id]
        if len(filtered) == 0:
            self.logger.warning(
                f"No anime series with TVDB ID {item_id} found in Sonarr",
            )
        return filtered

    def _item_anilist_ids(self, item: Any, log_ignored: bool = True) -> dict:
        """Resolve AniList ids for a Sonarr series (by TVDB / IMDb id)."""

        return self.get_anilist_ids(
            tvdb_id=item.tvdbId,
            imdb_id=item.imdbId,
            log_ignored=log_ignored,
        )

    def _process_al_id(
        self,
        arr: str,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: dict,
    ) -> bool:
        """Process one AniList id for a Sonarr series

        The middle is the episode-aware part: resolve the relevant episode list,
        its coverage and release groups, parse the SeaDex file lists into
        episodes, then hand off to the shared grab/cache tail.
        """

        sd_entry = self._al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url
        sonarr_series_id = item.id

        # Check if we've already got this cached
        al_id_in_cache = self.check_al_id_in_cache(
            arr=arr,
            al_id=al_id,
            seadex_entry=sd_entry,
        )

        if al_id_in_cache and not self.ignore_seadex_update_times:
            # Backfill the enriched fields (coverage + URL) for cache records
            # written before they existed, so cached rows can still show
            # season/episodes/URL. One-time per old entry.
            if not self.get_cached_field(arr, al_id, "url"):
                backfill_eps = self.get_ep_list(
                    sonarr_series_id=sonarr_series_id,
                    al_id=al_id,
                    mapping=mapping,
                )
                self.update_cache(
                    arr=arr,
                    al_id=al_id,
                    cache_details={
                        "url": sd_url,
                        "coverage": self.coverage_string(
                            self.episodes_from_ep_list(backfill_eps),
                        ),
                    },
                )
            self.log_cached_entry(arr=arr, al_id=al_id)
            return False

        # Also check if it's in the Radarr cache, if we have that option
        if self.ignore_movies_in_radarr and not self.ignore_seadex_update_times:
            al_id_in_radarr_cache = self.check_al_id_in_cache(
                arr="radarr",
                al_id=al_id,
                seadex_entry=sd_entry,
            )
            if al_id_in_radarr_cache:
                self.log_cached_entry(
                    arr="radarr",
                    al_id=al_id,
                    state="in radarr",
                )
                return False

        # Resolve the AniList title (logged later, once episodes give us the
        # season/episode coverage)
        anilist_title = self.get_anilist_title(al_id=al_id)

        # Setup info for cache
        cache_details = {
            "name": anilist_title,
            "updated_at": sd_entry.updated_at,
            "torrent_hashes": [],
        }

        # If we have a Radarr instance, and we don't want to add movies that are
        # already in Radarr, do that now
        if (
            self.radarr is not None
            and self.all_radarr_movies is not None
            and self.ignore_movies_in_radarr
        ):

            radarr_movies = []

            # Make sure these are flagged as specials since sometimes shows and
            # movies are all lumped together
            mapping_season = mapping.get("tvdb_season", -1)
            if mapping_season == 0:

                mapping_tmdb_id = mapping.get("tmdb_movie_id")
                mapping_imdb_id = mapping.get("imdb_id")

                for m in self.all_radarr_movies:

                    # Check by TMDB IDs
                    if mapping_tmdb_id is not None:
                        if (
                            m.tmdbId == mapping_tmdb_id
                            and m not in radarr_movies
                        ):
                            radarr_movies.append(m)

                    # Check by IMDb IDs
                    if mapping_imdb_id is not None:
                        if (
                            m.imdbId == mapping_imdb_id
                            and m not in radarr_movies
                        ):
                            radarr_movies.append(m)

            if len(radarr_movies) > 0:

                for movie in radarr_movies:
                    self.log_entry_status(
                        "in radarr",
                        movie.title,
                    )

                time.sleep(self.sleep_time)
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
        ep_list_monitored = [x.get("monitored", True) for x in ep_list]
        if not any(ep_list_monitored) and self.ignore_unmonitored:
            self.log_anilist_item_unmonitored(
                item_title=anilist_title,
            )
            time.sleep(self.sleep_time)
            return False

        # Now that we have the episodes, log the active entry with its
        # season/episode coverage + URL, and remember them for the cache so
        # future cached runs can show the same detail
        coverage = self.coverage_string(
            self.episodes_from_ep_list(ep_list),
        )
        self.log_al_title(
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
                f"Sonarr release group(s): {', '.join(sonarr_release_groups)}",
            ),
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = self.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            self.log_no_seadex_releases()

            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details=cache_details,
            )

            time.sleep(self.sleep_time)
            return False

        self.logger.debug(
            indent_string(
                f"SeaDex: {', '.join(seadex_dict)}",
            ),
        )

        # Parse out filenames and check for overlaps
        seadex_dict = self.parse_episodes_from_seadex(seadex_dict=seadex_dict)
        overlapping_results = get_overlapping_results(seadex_dict=seadex_dict)

        # If we're in interactive mode and there are multiple equivalent options here, then select
        if self.interactive and len(seadex_dict) > 1 and overlapping_results:
            seadex_dict = self.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )

        # Filter downloads by whether the episodes in each torrent match the release
        # group we have in Sonarr
        torrent_hashes, seadex_dict = self.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr=arr,
            arr_release_dict=sonarr_release_dict,
            ep_list=ep_list,
        )

        return self._grab_and_cache(
            arr=arr,
            al_id=al_id,
            item_title=item_title,
            anilist_title=anilist_title,
            sd_url=sd_url,
            seadex_dict=seadex_dict,
            torrent_hashes=torrent_hashes,
            cache_details=cache_details,
            release_group=sonarr_release_groups,
        )

    def get_all_sonarr_series(self) -> list:
        """Get all series in Sonarr with AniList mapping info"""

        sonarr_series = []

        all_tvdb_ids = set()
        all_imdb_ids = set()

        # Kometa Anime-IDs is a flat {anilist_id: mapping} dict we scan directly
        if self.anime_mappings:
            all_tvdb_ids.update(
                m.get("tvdb_id")
                for m in self.anime_mappings.values()
                if m.get("tvdb_id") is not None
            )
            all_imdb_ids.update(
                m.get("imdb_id")
                for m in self.anime_mappings.values()
                if m.get("imdb_id") is not None
            )

        # AniBridge exposes precomputed id sets (no per-call scan needed)
        if self.anibridge:
            all_tvdb_ids |= self.anibridge.all_tvdb_ids
            all_imdb_ids |= self.anibridge.all_imdb_ids

        # Track which series ids we've kept via a set: "s not in sonarr_series"
        # on a growing list is O(n) per check (and compares whole series objects),
        # making the scan quadratic on a large library
        seen_ids = set()
        for s in self.sonarr.all_series():

            if s.id in seen_ids:
                continue

            # Keep the series if it matches by TVDB or IMDb id
            if s.tvdbId in all_tvdb_ids or s.imdbId in all_imdb_ids:
                sonarr_series.append(s)
                seen_ids.add(s.id)

        sonarr_series.sort(key=lambda x: x.title)

        return sonarr_series

    def get_sonarr_series(self, tvdb_id: int):
        """Get Sonarr series for a given TVDB ID

        Args:
            tvdb_id (int): TVDB ID
        """

        try:
            series = self.sonarr.get_series(tvdb_id=tvdb_id)
        except arrapi.exceptions.NotFound:
            series = None

        return series

    def get_ep_list(
        self,
        sonarr_series_id: int,
        al_id: int,
        mapping: dict,
    ) -> list | None:
        """Get a list of relevant episodes for an AniList mapping

        Args:
            sonarr_series_id (int): Series ID in Sonarr
            al_id (int): Anilist ID
            mapping (dict): Mapping dictionary between TVDB and AniList
        """

        # If we have any season info, pull that out now
        tvdb_season = get_tvdb_season(mapping)

        # Check we have a sensible AL ID
        if al_id == -1:
            raise ValueError("AniList ID not defined!")

        # Get the AniDB ID
        anidb_id = mapping.get("anidb_id")

        # Check what kind of mode we're in here,
        # it's either AniBridge or Anime IDs
        if "tvdb_mappings" in mapping:
            mapping_mode = "anibridge"
        else:
            mapping_mode = "anime_ids"

        # Get all the episodes for the whole series. The fetch is per-series (not
        # per-AniList-id), so a multi-season series resolving to several ids would
        # otherwise re-request the identical list; cache it per series for the run
        # and only do the per-id filtering below on the shared, read-only list.
        ep_list = self._ep_list_cache.get(sonarr_series_id)
        if ep_list is None:
            eps_req_url = (
                f"{self.sonarr_url}/api/v3/episode?"
                f"seriesId={sonarr_series_id}&"
                f"includeImages=false&"
                f"includeEpisodeFile=true&"
                f"apikey={self.sonarr_api_key}"
            )
            eps_req = self.session.get(eps_req_url)

            if eps_req.status_code != 200:
                self.logger.warning(
                    "Could not fetch episode data from Sonarr; it may be unreachable",
                )
                return None

            # Sort by season/episode number for slicing later
            ep_list = sorted(
                eps_req.json(),
                key=lambda x: (
                    x.get("seasonNumber", None),
                    x.get("episodeNumber", None),
                ),
            )
            self._ep_list_cache[sonarr_series_id] = ep_list

        # Filter down here by various things
        final_ep_list = []
        for ep in ep_list:

            if mapping_mode == "anime_ids":
                include_episode = check_ep_by_anime_ids(
                    ep=ep,
                    tvdb_season=tvdb_season,
                )
            elif mapping_mode == "anibridge":
                tvdb_mappings = mapping.get("tvdb_mappings", {})
                include_episode = check_ep_by_anibridge(
                    ep=ep,
                    tvdb_mappings=tvdb_mappings,
                )
            else:
                raise ValueError(f"Invalid mapping mode {mapping_mode}")

            if include_episode:
                final_ep_list.append(ep)

        # For OVAs and movies, the offsets can often be wrong, so if we have specific mappings
        # then take that into account here
        al_format, self.al_cache = get_anilist_format(
            al_id,
            al_cache=self.al_cache,
        )

        # Potentially pull out a bunch of mappings from AniDB. These should
        # be for anything not marked as TV, and specials as marked by
        # being in Season 0
        anidb_mapping_dict = {}
        if (
            self.anidb_mappings is not None
            and anidb_id is not None
            and (al_format not in ["TV"] or tvdb_season == 0)
        ):
            anidb_item = self.anidb_anime_by_id(anidb_id)

            # If we don't find anything, no worries. If we find multiple, worries
            if len(anidb_item) > 1:
                raise ValueError(
                    "Multiple AniDB mappings found. This should not happen!",
                )

            if len(anidb_item) == 1:
                anidb_item = anidb_item[0]

                # We want things with mapping lists in, since more regular
                # mappings will have already been picked up
                anidb_mapping_list = anidb_item.findall("mapping-list")

                if len(anidb_mapping_list) > 0:
                    for ms in anidb_mapping_list:
                        m = ms.findall("mapping")
                        for i in m:

                            # If there's no text, continue
                            if not i.text:
                                continue

                            # Split at semicolons
                            i_split = i.text.strip(";").split(";")
                            i_split = [x.split("-") for x in i_split]

                            # Only match things if AniList and AniDB agree on the TVDB season
                            anidb_tvdbseason = int(i.attrib["tvdbseason"])
                            if anidb_tvdbseason != tvdb_season:
                                continue

                            anidb_mapping_dict[anidb_tvdbseason] = {
                                int(x[1]): int(x[0]) for x in i_split
                            }

        # Prefer the AniDB mapping dict over any offsets
        if len(anidb_mapping_dict) > 0:
            anidb_final_ep_list = []

            # See if we have the mapping for each entry
            for ep in final_ep_list:

                season_number = ep.get("seasonNumber", None)
                episode_number = ep.get("episodeNumber", None)

                anidb_mapping_dict_entry = anidb_mapping_dict.get(
                    season_number, {},
                ).get(episode_number, None)
                if anidb_mapping_dict_entry is not None:
                    anidb_final_ep_list.append(ep)

            final_ep_list = copy.deepcopy(anidb_final_ep_list)

        else:

            if mapping_mode == "anime_ids":

                # Slice the list to get the correct episodes, so any potential offsets
                ep_offset = mapping.get("tvdb_epoffset", 0)
                n_eps, self.al_cache = get_anilist_n_eps(
                    al_id,
                    al_cache=self.al_cache,
                )

                # If we don't get a number of episodes, use them all
                if n_eps is None:
                    n_eps = len(final_ep_list) - ep_offset

                # Check that we're including this by the episode number. This only
                # works for single-seasons, so be careful!
                if tvdb_season != -1:
                    final_ep_list = [
                        ep
                        for ep in final_ep_list
                        if 1 <= ep.get("episodeNumber", None) - ep_offset <= n_eps
                    ]
                else:
                    final_ep_list = final_ep_list[ep_offset : n_eps + ep_offset]

            # Or, we've got AniBridge mappings so we don't need to do anything (hooray)
            elif mapping_mode == "anibridge":
                pass

            else:
                raise ValueError(f"Invalid mapping mode {mapping_mode}")

        return final_ep_list

    def get_sonarr_release_dict(
        self,
        ep_list: list,
    ) -> dict:
        """Get a dictionary of useful info for a series in Sonarr

        Args:
            ep_list (list): List of episodes
        """

        # Look through, get release groups from the existing Sonarr files
        # and note any potential missing files
        sonarr_release_dict = {}
        missing_eps = 0
        n_eps = len(ep_list)
        for ep in ep_list:

            if ep.get("episodeFileId", 0) == 0:
                missing_eps += 1
                continue

            release_group = ep.get("episodeFile", {}).get("releaseGroup", None)
            if release_group is None or release_group == "":
                continue

            if release_group not in sonarr_release_dict:
                sonarr_release_dict[release_group] = {"size": []}
            size = ep.get("episodeFile", {}).get("size", None)
            sonarr_release_dict[release_group]["size"].append(size)

        if missing_eps > 0:
            # Show which episodes are missing as ranges (e.g. "S04 E12"), not just
            # a count, so it's clear what's absent. Fall back to the count if the
            # episodes can't be condensed.
            missing_coverage = self.coverage_string(
                self.episodes_from_ep_list(ep_list, missing_only=True),
            )
            self.log_fmt.detail(
                "missing",
                missing_coverage or f"{missing_eps}/{n_eps}",
                value_style="yellow",
            )

        return sonarr_release_dict

    def get_sonarr_parse(
        self,
        filename: str,
    ) -> list:
        """Ask Sonarr to parse a single filename into season/episode numbers

        Only the season/episode mapping is returned - the file size is filled in
        by the caller, since it comes from the SeaDex file list rather than from
        Sonarr.

        Args:
            filename (str): Filename to parse (basename, not full path)

        Returns:
            list: List of {"season", "episode"} dicts (empty if Sonarr couldn't
                parse the filename)
        """

        d = {"title": filename, "apikey": self.sonarr_api_key}
        d_enc = urlencode(d)

        # Parse through Sonarr
        parse_req_url = f"{self.sonarr_url}/api/v3/parse?{d_enc}"
        parse_req = self.session.get(parse_req_url)

        if parse_req.status_code != 200:
            self.logger.warning(
                indent_string(
                    f"Could not parse {filename} via Sonarr "
                    f"(status code {parse_req.status_code}); skipping file",
                ),
            )
            return []

        episode_info = parse_req.json().get("episodes", [])

        parsed = []
        for ep in episode_info:

            season = ep.get("seasonNumber", None)
            episode = ep.get("episodeNumber", None)

            if season is None or episode is None:
                self.logger.debug(
                    indent_string(
                        f"Season or episode came up None for {filename}; "
                        f"skipping this episode entry",
                    ),
                )
                continue

            parsed.append({"season": season, "episode": episode})

        return parsed

    @staticmethod
    def _sonarr_parse_is_fresh(record: dict | None) -> bool:
        """True if a persisted parse record has episodes and is within TTL

        Legacy list-form entries (pre-TTL, no timestamp) are treated as stale so
        they are re-queried once and upgraded to the timestamped form.
        """
        if not isinstance(record, dict):
            return False
        if not record.get("episodes"):
            return False
        try:
            stamp = datetime.strptime(
                record.get("fetched_at", ""), UPDATED_AT_STR_FORMAT,
            )
        except (TypeError, ValueError):
            return False
        return stamp >= datetime.now() - timedelta(
            days=SONARR_PARSE_CACHE_TTL_DAYS,
        )

    def parse_episodes_from_seadex(
        self,
        seadex_dict: dict,
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
        parse_cache = self.cache.setdefault("sonarr_parse_cache", {})
        now_str = datetime.now().strftime(UPDATED_AT_STR_FORMAT)

        for release_group_item in seadex_dict.values():

            # Set up an overall "all episodes" list
            release_group_item.update({"all_episodes": []})

            for url_item in release_group_item.get("urls", {}).values():

                # Set up a list to parse episodes from files
                url_item.update({"episodes": []})
                sizes = url_item.get("size", [])

                for sd_file_idx, seadex_file in enumerate(url_item.get("files", [])):

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
                    if self._sonarr_parse_is_fresh(record):
                        parsed = record["episodes"]
                    else:
                        parsed = self.get_sonarr_parse(f)

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

                        url_item["episodes"].append(
                            {
                                "season": season,
                                "episode": episode,
                                "size": size,
                            },
                        )
                        release_group_item["all_episodes"].append(
                            {
                                "season": season,
                                "episode": episode,
                                "size": size,
                            },
                        )

        return seadex_dict
