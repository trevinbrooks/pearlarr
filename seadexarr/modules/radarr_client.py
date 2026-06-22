"""Radarr REST client: the HTTP surface the Radarr syncer talks to.

``RadarrClient`` wraps the high-level ``arrapi`` client (``all_movies`` /
``get_movie``) and the one raw endpoint the syncer needs (``/api/v3/moviefile``)
behind a small, independently-testable adapter, so the syncer's hook bodies stop
mixing HTTP concerns with domain logic. ``collect_anime_movies`` is the shared
"keep the movies that have an AniList mapping" scan, reused by the Radarr syncer
and by Sonarr's ``ignore_movies_in_radarr`` cross-check (which used to build a
whole nested ``SeaDexRadarr`` just to call it).

Extracted from ``SeaDexRadarr`` in Phase 5a of the refactor (see
``REFACTOR_PLAN.md``); behaviour-preserving.
"""

import logging

import arrapi.exceptions
import requests
from arrapi import RadarrAPI

from .anibridge import AniBridge


class RadarrClient:
    """Thin wrapper over the Radarr API (``arrapi`` + one raw endpoint)."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        session: requests.Session,
        logger: logging.Logger,
    ) -> None:
        """Instantiate the Radarr API client.

        Args:
            url (str): Radarr base URL.
            api_key (str): Radarr API key.
            session (requests.Session): Shared keep-alive session for the raw
                endpoints.
            logger (logging.Logger): For request warnings.
        """

        self._url = url
        self._api_key = api_key
        self._session = session
        self._logger = logger
        self._api = RadarrAPI(url=url, apikey=api_key)

    def all_movies(self) -> list:
        """Every movie in Radarr (unfiltered)."""

        return self._api.all_movies()

    def get_movie(self, tmdb_id: int | None = None, imdb_id: str | None = None):
        """Get the Radarr movie for a TMDB or IMDb id, or None if not found.

        Args:
            tmdb_id (int): TMDB movie ID.
            imdb_id (str): IMDb movie ID.
        """

        try:
            return self._api.get_movie(tmdb_id=tmdb_id, imdb_id=imdb_id)
        except arrapi.exceptions.NotFound:
            return None

    def movie_files(self, movie_id: int) -> list:
        """Raw movie-file records for a movie (``/api/v3/moviefile``).

        Args:
            movie_id (int): ID for the movie in Radarr.
        """

        mov_req_url = (
            f"{self._url}/api/v3/moviefile?"
            f"movieId={movie_id}&"
            f"apikey={self._api_key}"
        )
        return self._session.get(mov_req_url).json()


def collect_anime_movies(
    radarr_client: RadarrClient,
    anime_mappings: dict | None,
    anibridge: AniBridge | None,
) -> list:
    """Radarr movies that have an AniList mapping, sorted by title.

    Gathers the candidate TMDB/IMDb id sets from the two mapping sources, then
    keeps each Radarr movie that matches one of them. Shared by
    ``SeaDexRadarr.get_all_radarr_movies`` and Sonarr's ``ignore_movies_in_radarr``
    path.

    Args:
        radarr_client (RadarrClient): Client to fetch the movie list from.
        anime_mappings (dict | None): Kometa Anime-IDs flat {anilist_id: mapping}
            dict, scanned directly.
        anibridge (object | None): AniBridge mappings, exposing precomputed
            ``all_tmdb_movie_ids`` / ``all_imdb_ids`` sets.
    """

    radarr_movies = []

    all_tmdb_ids = set()
    all_imdb_ids = set()

    # Kometa Anime-IDs is a flat {anilist_id: mapping} dict we scan directly
    if anime_mappings:
        all_tmdb_ids.update(
            e.get("tmdb_movie_id")
            for e in anime_mappings.values()
            if e.get("tmdb_movie_id") is not None
        )
        all_imdb_ids.update(
            e.get("imdb_id")
            for e in anime_mappings.values()
            if e.get("imdb_id") is not None
        )

    # AniBridge exposes precomputed id sets (no per-call scan needed)
    if anibridge:
        all_tmdb_ids |= anibridge.all_tmdb_movie_ids
        all_imdb_ids |= anibridge.all_imdb_ids

    # Track kept movie ids in a set: "m not in radarr_movies" on a growing
    # list is O(n) per check (and compares whole movie objects), making the
    # scan quadratic on a large library
    seen_ids = set()
    for m in radarr_client.all_movies():

        if m.id in seen_ids:
            continue

        # Keep the movie if it matches by TMDB or IMDb id
        if m.tmdbId in all_tmdb_ids or m.imdbId in all_imdb_ids:
            radarr_movies.append(m)
            seen_ids.add(m.id)

    radarr_movies.sort(key=lambda x: x.title)

    return radarr_movies
