import time

import arrapi.exceptions
from arrapi import RadarrAPI

from .discord import discord_push
from .log import indent_string
from .seadex_arr import SeaDexArr


class SeaDexRadarr(SeaDexArr):

    def __init__(
        self,
        config="config.yml",
        cache="cache.json",
        logger=None,
    ):
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

    def run(self, tmdb_id=None, dry_run=False):
        """Run the SeaDex Radarr syncer

        Args:
            tmdb_id (int, optional): If set, only run for the movie with this
                TMDB ID. Defaults to None, which runs for all movies.
            dry_run (bool, optional): If True, simulate the run without grabbing
                torrents, writing the cache, or sending notifications.
                Defaults to False.
        """

        # Whether this is a no-op preview - consulted by the mutating helpers
        self.dry_run = dry_run

        # Reset the per-run tally and start the run clock
        self.reset_run_stats()

        # Get all the anime movies
        all_radarr_movies = self.get_all_radarr_movies()

        # If we're targeting a single movie, filter down to that TMDB ID
        if tmdb_id is not None:
            all_radarr_movies = [
                m for m in all_radarr_movies if m.tmdbId == tmdb_id
            ]
            if len(all_radarr_movies) == 0:
                self.logger.warning(
                    f"No anime movie with TMDB ID {tmdb_id} found in Radarr"
                )

        n_radarr = len(all_radarr_movies)

        self.log_arr_start(
            arr="radarr",
            n_items=n_radarr,
        )

        # Warm the AniList cache before the per-movie loop: reuse what past runs
        # fetched, then batch-fetch (id_in pages) everything still missing, so
        # the loop rarely hits AniList one id at a time and trips its rate limit.
        self.load_anilist_cache()
        prefetch_ids = set()
        for movie in all_radarr_movies:
            if not movie.monitored and self.ignore_unmonitored:
                continue
            prefetch_ids.update(
                self.get_anilist_ids(
                    tmdb_id=movie.tmdbId,
                    imdb_id=movie.imdbId,
                    tmdb_type="movie",
                    log_ignored=False,
                )
            )
        self.prefetch_anilist(prefetch_ids)

        # Now start looping over these movies
        for radarr_idx, radarr_movie in enumerate(all_radarr_movies):

            try:

                # Pull Radarr and database info out
                tmdb_id = radarr_movie.tmdbId
                imdb_id = radarr_movie.imdbId
                radarr_title = radarr_movie.title
                radarr_movie_id = radarr_movie.id

                self.log_arr_item_start(
                    arr="radarr",
                    item_title=radarr_title,
                    n_item=radarr_idx + 1,
                    n_items=n_radarr,
                )

                # If we're not monitored, then skip if ignore_unmonitored is switched on
                if not radarr_movie.monitored and self.ignore_unmonitored:
                    self.log_arr_item_unmonitored(
                        arr="radarr",
                        item_title=radarr_title,
                    )
                    continue

                # Get the mappings from the Radarr movies to AniList
                al_mappings = self.get_anilist_ids(
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    tmdb_type="movie",
                )

                if len(al_mappings) == 0:
                    self.log_no_anilist_mappings(title=radarr_title)
                    continue

                for al_id, mapping in al_mappings.items():

                    # Reset the per-title public_only skip flag before we make
                    # any download decisions for this title
                    self.public_only_skipped = False
                    self.stats["checked"] += 1

                    # Map the TMDB ID through to AniList
                    if al_id is None:
                        self.log_no_anilist_id()
                        continue

                    # Get the SeaDex entry if it exists
                    sd_entry = self.get_seadex_entry(al_id=al_id)
                    if sd_entry is None:
                        self.log_no_sd_entry(al_id=al_id)
                        continue
                    sd_url = sd_entry.url

                    # Check if we've already got this cached
                    al_id_in_cache = self.check_al_id_in_cache(
                        arr="radarr",
                        al_id=al_id,
                        seadex_entry=sd_entry,
                    )

                    if al_id_in_cache and not self.ignore_seadex_update_times:
                        # Backfill the URL for cache records written before it was
                        # stored, so cached rows can still link to SeaDex. Movies
                        # have no episode coverage, so there's nothing else to add.
                        if not self.get_cached_field("radarr", al_id, "url"):
                            self.update_cache(
                                arr="radarr",
                                al_id=al_id,
                                cache_details={"url": sd_url, "coverage": ""},
                            )
                        self.log_cached_entry(arr="radarr", al_id=al_id)
                        continue

                    # Resolve the AniList title, then log the active entry (a movie
                    # has no episode coverage, so the line carries just the URL)
                    anilist_title = self.get_anilist_title(al_id=al_id)
                    self.log_al_title(anilist_title=anilist_title, sd_entry=sd_entry)

                    # Setup info for cache (URL so cached runs can link to SeaDex;
                    # movies have no episode coverage)
                    cache_details = {
                        "name": anilist_title,
                        "updated_at": sd_entry.updated_at,
                        "torrent_hashes": [],
                        "url": sd_url,
                        "coverage": "",
                    }

                    radarr_release_dict = self.get_radarr_release_dict(
                        radarr_movie_id=radarr_movie_id
                    )
                    radarr_release_group = list(radarr_release_dict.keys())[0]

                    self.logger.debug(
                        indent_string(
                            f"Radarr release group: {radarr_release_group}",
                        )
                    )

                    # Produce a dictionary of info from the SeaDex request
                    seadex_dict = self.get_seadex_dict(sd_entry=sd_entry)

                    if len(seadex_dict) == 0:
                        self.log_no_seadex_releases()

                        self.update_cache(
                            arr="radarr",
                            al_id=al_id,
                            cache_details=cache_details,
                        )

                        time.sleep(self.sleep_time)
                        continue

                    self.logger.debug(
                        indent_string(
                            f"SeaDex: {', '.join(seadex_dict)}",
                        )
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
                        arr="radarr",
                        arr_release_dict=radarr_release_dict,
                    )

                    # Check the release groups are matching, and get a bespoke list of torrents
                    any_to_download = self.get_any_to_download(seadex_dict=seadex_dict)

                    # Capture the running total before the add block so we can
                    # tell whether THIS title actually grabbed anything
                    torrents_before = self.torrents_added

                    if any_to_download:
                        fields, anilist_thumb = self.get_seadex_fields(
                            arr="radarr",
                            al_id=al_id,
                            release_group=radarr_release_group,
                            seadex_dict=seadex_dict,
                        )

                        # If we've got stuff, time to do something!
                        if len(seadex_dict) > 0:

                            # Keep track of how many torrents we've added
                            n_torrents_added = 0
                            results = []

                            # Add torrents to qBittorrent. add_torrent runs even
                            # in a preview (no client / dry run): add_torrent_to_qbit
                            # simulates the add, while the download-flag,
                            # public_only and tracker filters still apply, so only
                            # releases that would actually be grabbed are counted.
                            added, results = self.add_torrent(
                                torrent_dict=seadex_dict,
                                torrent_client="qbit",
                            )
                            n_torrents_added += added

                            # Log the action block now the outcome is known, so
                            # the status reads "adding" only when something was
                            # actually grabbed (else "keeping")
                            self.log_seadex_action(
                                seadex_dict=seadex_dict,
                                results=results,
                                dry_run=self._is_preview(),
                            )

                            # Push a message to Discord if we've added anything
                            # (never on a preview - it's an outward notification)
                            if (
                                self.discord_url is not None
                                and n_torrents_added > 0
                                and not self._is_preview()
                            ):
                                discord_push(
                                    url=self.discord_url,
                                    arr_title=radarr_title,
                                    al_title=anilist_title,
                                    seadex_url=sd_url,
                                    fields=fields,
                                    thumb_url=anilist_thumb,
                                )

                            if self.max_torrents_to_add is not None:
                                if self.torrents_added >= self.max_torrents_to_add:
                                    self.log_max_torrents_added()
                                    self.log_run_summary(arr="radarr")
                                    return True

                    elif not self.public_only_skipped:
                        self.stats["up_to_date"] += 1
                        self.log_detail(
                            "status",
                            "already have the recommended release",
                            value_style="blue",
                        )

                    # Work out whether THIS title actually grabbed anything
                    added_this_title = self.torrents_added - torrents_before

                    # Update and save out the cache whenever something was
                    # grabbed for this title, or when nothing was skipped at all.
                    # Leave the title uncached ONLY when public_only skipped a
                    # release AND nothing else was grabbed for it - so it's
                    # re-checked (and the skip re-logged as a reminder) on every
                    # run, and retried once a public release appears or
                    # public_only is relaxed
                    if added_this_title > 0 or not self.public_only_skipped:
                        cache_details.update({"torrent_hashes": torrent_hashes})
                        self.update_cache(
                            arr="radarr",
                            al_id=al_id,
                            cache_details=cache_details,
                        )
                    elif added_this_title == 0:
                        # Record the private-only skip for the summary's
                        # "needs action" list, attributed to this title - but
                        # only when nothing was actually added for it
                        self.stats["needs_action"].append(
                            {
                                "title": self.current_title,
                                "url": self.current_url,
                                "reason": "private-only release; public_only on",
                            }
                        )

                    # Add in a wait, if required
                    time.sleep(self.sleep_time)

                if self.max_torrents_to_add is not None:
                    if self.torrents_added >= self.max_torrents_to_add:
                        self.log_max_torrents_added()
                        self.log_run_summary(arr="radarr")
                        return True

            except Exception as e:
                title = getattr(radarr_movie, "title", "unknown title")
                self.logger.error(
                    f"{title}: unexpected error: {e}", exc_info=True
                )
                continue

        self.log_run_summary(arr="radarr")

        return True

    def get_all_radarr_movies(self):
        """Get all movies in Radarr that have an associated AniList ID"""

        radarr_movies = []

        all_tmdb_ids = []
        all_imdb_ids = []

        # Search through TMDB and IMDb IDs via Anime IDs and AniBridge mappings
        for mapping in [
            self.anime_mappings,
            self.anibridge_mappings,
        ]:
            if not mapping:
                continue

            all_tmdb_ids.extend(
                mapping[x].get("tmdb_movie_id", None)
                for x in mapping
                if "tmdb_movie_id" in mapping[x].keys()
            )

            all_imdb_ids.extend(
                mapping[x].get("imdb_id", None)
                for x in mapping
                if "imdb_id" in mapping[x].keys()
            )

        for m in self.radarr.all_movies():

            # Check by TMDB IDs
            tmdb_id = m.tmdbId
            if tmdb_id in all_tmdb_ids and m not in radarr_movies:
                radarr_movies.append(m)

            # Check by IMDb IDs
            imdb_id = m.imdbId
            if imdb_id in all_imdb_ids and m not in radarr_movies:
                radarr_movies.append(m)

        radarr_movies.sort(key=lambda x: x.title)

        return radarr_movies

    def get_radarr_movie(self, tmdb_id=None, imdb_id=None):
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
        radarr_movie_id,
    ):
        """Get a dictionary of useful info for a Radarr movie

        Args:
            radarr_movie_id (int): ID for movie in Radarr
        """

        # Get the movie file if it exists
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
