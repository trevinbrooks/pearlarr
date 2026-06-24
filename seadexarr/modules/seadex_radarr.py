from .cache import CacheRecord
from .config import Arr
from .log import indent_string
from .mappings import MappingEntry, TmdbType
from .protocols import ArrSync
from .radarr_client import collect_anime_movies, make_radarr_client
from .seadex_arr import RunDeps, SeaDexArr
from .seadex_types import ArrReleaseDict, RadarrItem


class RadarrSync(ArrSync):
    """Radarr sync strategy: owns the Radarr REST client + movie domain logic.

    Implements the :class:`~.protocols.ArrSync` hooks the run machinery drives.
    The composition root injects the shared :class:`~.seadex_arr.RunDeps` (used to
    stand up the client) and the :class:`~.protocols.RunServices` run machinery
    (held as ``self._services``); the per-id hooks call the shared pipeline
    through it.
    """

    def __init__(self, deps: RunDeps, services: SeaDexArr) -> None:
        """Stand up the Radarr client from the injected shared collaborators.

        Args:
            deps (RunDeps): The shared collaborators; the config/session/mappings
                this strategy needs are read off it.
            services (RunServices): The run machinery the per-id hooks call into.
        """

        self._services = services
        self._config = deps.config
        self.session = deps.session
        self.logger = deps.logger
        self.anime_mappings = deps.mappings.anime_mappings
        self.anibridge = deps.mappings.anibridge

        radarr_url = self._config.radarr_url
        radarr_api_key = self._config.radarr_api_key

        self.radarr = make_radarr_client(
            url=radarr_url,
            api_key=radarr_api_key,
            session=self.session,
            logger=self.logger,
        )

    # --- ArrSync hooks ------------------------------------------------------

    def get_items(self) -> list[RadarrItem]:
        """Every Radarr movie that has an associated AniList ID."""

        return self.get_all_radarr_movies()

    def filter_to_single(
        self, items: list[RadarrItem], item_id: int,
    ) -> list[RadarrItem]:
        """Narrow the movie list to a single TMDB ID."""

        filtered = [m for m in items if m.tmdbId == item_id]
        if len(filtered) == 0:
            self.logger.warning(
                f"No anime movie with TMDB ID {item_id} found in Radarr",
            )
        return filtered

    def item_anilist_ids(
        self,
        item: RadarrItem,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve AniList ids for a Radarr movie (by TMDB / IMDb id)."""

        return self._services.get_anilist_ids(
            tmdb_id=item.tmdbId,
            imdb_id=item.imdbId,
            tmdb_type=TmdbType.MOVIE,
            log_ignored=log_ignored,
        )

    def process_al_id(
        self,
        arr: Arr,
        item: RadarrItem,
        item_title: str,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for a Radarr movie

        A movie is a single file, so the middle is simply: resolve the Radarr
        release group, pull the SeaDex releases, filter them, then hand off to
        the shared grab/cache tail. ``mapping`` is unused (movies need no episode
        mapping) but is accepted to match the shared hook signature.
        """

        run = self._services

        sd_entry = run.al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url

        # Skip if already cached. Movies have no episode coverage, so the
        # one-time backfill on a legacy record is just the URL.
        if run.cached_entry_skip(arr, al_id, sd_entry, sd_url, lambda: ""):
            return False

        # Resolve the AniList title, then log the active entry (a movie has no
        # episode coverage, so the line carries just the URL)
        anilist_title = run.get_anilist_title(al_id=al_id)
        run.log_al_title(anilist_title=anilist_title, sd_entry=sd_entry)

        # Setup info for cache (URL so cached runs can link to SeaDex; movies have
        # no episode coverage)
        cache_details: CacheRecord = {
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
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(arr, al_id, cache_details)

        self.logger.debug(
            indent_string(
                f"SeaDex: {', '.join(seadex_dict)}",
            ),
        )

        # If we're in interactive mode and there are multiple options here, then select
        if self._config.interactive and len(seadex_dict) > 1:
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )

        torrent_hashes, seadex_dict = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr=arr,
            arr_release_dict=radarr_release_dict,
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
            release_group=radarr_release_group,
        )

    # --- Radarr domain logic ------------------------------------------------

    def get_all_radarr_movies(self) -> list[RadarrItem]:
        """Get all movies in Radarr that have an associated AniList ID"""

        return collect_anime_movies(
            self.radarr,
            self.anime_mappings,
            self.anibridge,
        )

    def get_radarr_release_dict(
        self,
        radarr_movie_id: int,
    ) -> ArrReleaseDict:
        """Get a dictionary of useful info for a Radarr movie

        Args:
            radarr_movie_id (int): ID for movie in Radarr
        """

        # A movie is a single file per release group; wrap its size in a
        # one-element list so the value matches the shared ArrReleaseDict shape
        # (Sonarr accumulates a per-episode list).
        radarr_release_dict: ArrReleaseDict = {
            r.get("releaseGroup", None): [r.get("size", None)]
            for r in self.radarr.movie_files(radarr_movie_id)
        }

        # If we have multiple options, throw up an error
        if len(radarr_release_dict) > 1:
            raise ValueError(f"Multiple files found for movie {radarr_movie_id}")

        # If we have nothing, return None
        elif len(radarr_release_dict) == 0:
            radarr_release_dict = {None: [None]}

        return radarr_release_dict
