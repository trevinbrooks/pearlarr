import logging
import time
from typing import Any

import arrapi.exceptions
from arrapi import RadarrAPI

from .log import indent_string
from .seadex_arr import SeaDexArr


class SeaDexRadarr(SeaDexArr):

    def __init__(
        self,
        config: str = "config.yml",
        cache: str = "cache.json",
        logger: logging.Logger | None = None,
    ) -> None:
        """Sync Radarr instance with SeaDex

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
            arr="radarr",
            config=config,
            cache=cache,
            logger=logger,
        )

        # Set up Radarr
        self.radarr_url = self.config.get("radarr_url", None)
        if not self.radarr_url:
            raise ValueError(f"radarr_url needs to be defined in {config}")

        self.radarr_api_key = self.config.get("radarr_api_key", None)
        if not self.radarr_api_key:
            raise ValueError(f"radarr_api_key needs to be defined in {config}")

        self.radarr = RadarrAPI(
            url=self.radarr_url,
            apikey=self.radarr_api_key,
        )

    def run(self, tmdb_id: int | None = None, dry_run: bool = False) -> bool:
        """Run the SeaDex Radarr syncer

        Args:
            tmdb_id (int, optional): If set, only run for the movie with this
                TMDB ID. Defaults to None, which runs for all movies.
            dry_run (bool, optional): If True, simulate the run without grabbing
                torrents, writing the cache, or sending notifications.
                Defaults to False.
        """

        return self.run_sync(arr="radarr", item_id=tmdb_id, dry_run=dry_run)

    def _get_all_items(self) -> list:
        """Every Radarr movie that has an associated AniList ID."""

        return self.get_all_radarr_movies()

    def _filter_to_single_item(self, items: list, item_id: int) -> list:
        """Narrow the movie list to a single TMDB ID."""

        filtered = [m for m in items if m.tmdbId == item_id]
        if len(filtered) == 0:
            self.logger.warning(
                f"No anime movie with TMDB ID {item_id} found in Radarr",
            )
        return filtered

    def _item_anilist_ids(self, item: Any, log_ignored: bool = True) -> dict:
        """Resolve AniList ids for a Radarr movie (by TMDB / IMDb id)."""

        return self.get_anilist_ids(
            tmdb_id=item.tmdbId,
            imdb_id=item.imdbId,
            tmdb_type="movie",
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
        """Process one AniList id for a Radarr movie

        A movie is a single file, so the middle is simply: resolve the Radarr
        release group, pull the SeaDex releases, filter them, then hand off to
        the shared grab/cache tail. ``mapping`` is unused (movies need no episode
        mapping) but is accepted to match the shared hook signature.
        """

        sd_entry = self._al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url

        # Check if we've already got this cached
        al_id_in_cache = self.check_al_id_in_cache(
            arr=arr,
            al_id=al_id,
            seadex_entry=sd_entry,
        )

        if al_id_in_cache and not self.ignore_seadex_update_times:
            # Backfill the URL for cache records written before it was stored, so
            # cached rows can still link to SeaDex. Movies have no episode
            # coverage, so there's nothing else to add.
            if not self.get_cached_field(arr, al_id, "url"):
                self.update_cache(
                    arr=arr,
                    al_id=al_id,
                    cache_details={"url": sd_url, "coverage": ""},
                )
            self.log_cached_entry(arr=arr, al_id=al_id)
            return False

        # Resolve the AniList title, then log the active entry (a movie has no
        # episode coverage, so the line carries just the URL)
        anilist_title = self.get_anilist_title(al_id=al_id)
        self.log_al_title(anilist_title=anilist_title, sd_entry=sd_entry)

        # Setup info for cache (URL so cached runs can link to SeaDex; movies have
        # no episode coverage)
        cache_details = {
            "name": anilist_title,
            "updated_at": sd_entry.updated_at,
            "torrent_hashes": [],
            "url": sd_url,
            "coverage": "",
        }

        radarr_release_dict = self.get_radarr_release_dict(
            radarr_movie_id=item.id,
        )
        radarr_release_group = next(iter(radarr_release_dict))

        self.logger.debug(
            indent_string(
                f"Radarr release group: {radarr_release_group}",
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

        # If we're in interactive mode and there are multiple options here, then select
        if self.interactive and len(seadex_dict) > 1:
            seadex_dict = self.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )

        torrent_hashes, seadex_dict = self.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr=arr,
            arr_release_dict=radarr_release_dict,
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
            release_group=radarr_release_group,
        )

    def get_all_radarr_movies(self) -> list:
        """Get all movies in Radarr that have an associated AniList ID"""

        radarr_movies = []

        all_tmdb_ids = set()
        all_imdb_ids = set()

        # Kometa Anime-IDs is a flat {anilist_id: mapping} dict we scan directly
        if self.anime_mappings:
            all_tmdb_ids.update(
                e.get("tmdb_movie_id")
                for e in self.anime_mappings.values()
                if e.get("tmdb_movie_id") is not None
            )
            all_imdb_ids.update(
                e.get("imdb_id")
                for e in self.anime_mappings.values()
                if e.get("imdb_id") is not None
            )

        # AniBridge exposes precomputed id sets (no per-call scan needed)
        if self.anibridge:
            all_tmdb_ids |= self.anibridge.all_tmdb_movie_ids
            all_imdb_ids |= self.anibridge.all_imdb_ids

        # Track kept movie ids in a set: "m not in radarr_movies" on a growing
        # list is O(n) per check (and compares whole movie objects), making the
        # scan quadratic on a large library
        seen_ids = set()
        for m in self.radarr.all_movies():

            if m.id in seen_ids:
                continue

            # Keep the movie if it matches by TMDB or IMDb id
            if m.tmdbId in all_tmdb_ids or m.imdbId in all_imdb_ids:
                radarr_movies.append(m)
                seen_ids.add(m.id)

        radarr_movies.sort(key=lambda x: x.title)

        return radarr_movies

    def get_radarr_movie(self, tmdb_id: int | None = None, imdb_id: str | None = None):
        """Get Radarr movie for a given TMDB ID or IMDb ID

        Args:
            tmdb_id (int): TMDB movie ID
            imdb_id (str): IMDb movie ID
        """

        try:
            movie = self.radarr.get_movie(tmdb_id=tmdb_id, imdb_id=imdb_id)
        except arrapi.exceptions.NotFound:
            movie = None

        return movie

    def get_radarr_release_dict(
        self,
        radarr_movie_id: int,
    ) -> dict:
        """Get a dictionary of useful info for a Radarr movie

        Args:
            radarr_movie_id (int): ID for movie in Radarr
        """

        mov_req_url = (
            f"{self.radarr_url}/api/v3/moviefile?"
            f"movieId={radarr_movie_id}&"
            f"apikey={self.radarr_api_key}"
        )
        mov_req = self.session.get(mov_req_url)

        radarr_release_dict = {
            r.get("releaseGroup", None): {"size": r.get("size", None)}
            for r in mov_req.json()
        }

        # If we have multiple options, throw up an error
        if len(radarr_release_dict) > 1:
            raise ValueError(f"Multiple files found for movie {radarr_movie_id}")

        # If we have nothing, return None
        elif len(radarr_release_dict) == 0:
            radarr_release_dict = {None: {"size": None}}

        return radarr_release_dict
