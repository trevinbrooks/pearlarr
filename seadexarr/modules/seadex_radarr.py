import logging
import time
from typing import Any

from .log import indent_string
from .radarr_client import RadarrClient, collect_anime_movies
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
        radarr_url = self.config.get("radarr_url", None)
        if not radarr_url:
            raise ValueError(f"radarr_url needs to be defined in {config}")

        radarr_api_key = self.config.get("radarr_api_key", None)
        if not radarr_api_key:
            raise ValueError(f"radarr_api_key needs to be defined in {config}")

        self.radarr = RadarrClient(
            url=radarr_url,
            api_key=radarr_api_key,
            session=self.session,
            logger=self.logger,
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

        # Skip if already cached. Movies have no episode coverage, so the
        # one-time backfill on a legacy record is just the URL.
        if self._cached_entry_skip(arr, al_id, sd_entry, sd_url, lambda: ""):
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

        return collect_anime_movies(
            self.radarr,
            self.anime_mappings,
            self.anibridge,
        )

    def get_radarr_movie(self, tmdb_id: int | None = None, imdb_id: str | None = None):
        """Get Radarr movie for a given TMDB ID or IMDb ID

        Args:
            tmdb_id (int): TMDB movie ID
            imdb_id (str): IMDb movie ID
        """

        return self.radarr.get_movie(tmdb_id=tmdb_id, imdb_id=imdb_id)

    def get_radarr_release_dict(
        self,
        radarr_movie_id: int,
    ) -> dict:
        """Get a dictionary of useful info for a Radarr movie

        Args:
            radarr_movie_id (int): ID for movie in Radarr
        """

        radarr_release_dict = {
            r.get("releaseGroup", None): {"size": r.get("size", None)}
            for r in self.radarr.movie_files(radarr_movie_id)
        }

        # If we have multiple options, throw up an error
        if len(radarr_release_dict) > 1:
            raise ValueError(f"Multiple files found for movie {radarr_movie_id}")

        # If we have nothing, return None
        elif len(radarr_release_dict) == 0:
            radarr_release_dict = {None: {"size": None}}

        return radarr_release_dict
