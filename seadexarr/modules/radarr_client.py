"""Radarr REST client: the HTTP surface the Radarr syncer talks to.

``RadarrClient`` wraps the high-level ``arrapi`` client (``all_movies``)
and the one raw endpoint the syncer needs (``/api/v3/moviefile``)
behind a small, independently-testable adapter, so the syncer's hook bodies stop
mixing HTTP concerns with domain logic. ``collect_anime_movies`` is the shared
"keep the movies that have an AniList mapping" scan, reused by the Radarr syncer
and by Sonarr's ``ignore_movies_in_radarr`` cross-check (which used to build a
whole nested ``SeaDexRadarr`` just to call it).

Extracted from ``SeaDexRadarr`` in Phase 5a of the refactor (see
``REFACTOR_PLAN.md``); behaviour-preserving.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import requests
from arrapi import RadarrAPI

from .anibridge import AniBridge


def make_radarr_client(
    *,
    url: str,
    api_key: str,
    session: requests.Session,
    logger: logging.Logger,
) -> "RadarrClient":
    """Build a :class:`RadarrClient` from the shared session/logger and a url/key.

    Hoisted so the two Arr strategies share one construction site. The url/key
    lookup policy stays with each caller (Radarr requires them, Sonarr reads them
    optionally for its cross-check), so they're passed in already resolved rather
    than re-derived here.

    Args:
        url (str): Radarr base URL.
        api_key (str): Radarr API key.
        session (requests.Session): Shared keep-alive session.
        logger (logging.Logger): For request warnings.
    """

    return RadarrClient(
        url=url,
        api_key=api_key,
        session=session,
        logger=logger,
    )


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


@dataclass(frozen=True)
class IdField:
    """One id space to filter an Arr library by.

    Pairs the Kometa Anime-IDs map key with the live Arr item attribute that
    holds the same id, so ``collect_anime_items`` matches each item against the
    candidate set built for that id space.
    """

    mapping_key: str  # e.g. "tmdb_movie_id" / "tvdb_id"
    item_attr: str  # e.g. "tmdbId" / "tvdbId"


def collect_anime_items(
    list_fn: Callable[[], list],
    anime_mappings: dict | None,
    fields: tuple[IdField, ...],
    anibridge_id_sets: tuple[set, ...],
) -> list:
    """Arr library items that have an AniList mapping, sorted by title.

    Builds one candidate id-set per ``fields`` entry (union of the Kometa
    Anime-IDs values for that key and the matching precomputed AniBridge set),
    then keeps each item that matches at least one id space.

    Args:
        list_fn (Callable[[], list]): Returns the unfiltered Arr item list.
        anime_mappings (dict | None): Kometa Anime-IDs flat {anilist_id: mapping}
            dict, scanned directly (once, building all id sets in one pass).
        fields (tuple[IdField, ...]): Id spaces to filter by; one set is built
            per field, in order.
        anibridge_id_sets (tuple[set, ...]): Precomputed AniBridge id sets, one
            per ``fields`` entry in the same order (pass ``set()`` when AniBridge
            is disabled).
    """

    # One candidate set per id space, built in a single pass over the mappings
    matched_sets: list[set] = [set() for _ in fields]
    if anime_mappings:
        for entry in anime_mappings.values():
            for i, field in enumerate(fields):
                value = entry.get(field.mapping_key)
                if value is not None:
                    matched_sets[i].add(value)

    # AniBridge exposes precomputed id sets (no per-call scan needed)
    for i, extra in enumerate(anibridge_id_sets):
        matched_sets[i] |= extra

    # Track kept item ids in a set: "item not in kept" on a growing list is O(n)
    # per check (and compares whole item objects), making the scan quadratic on
    # a large library
    kept: list = []
    seen_ids: set = set()
    for item in list_fn():
        if item.id in seen_ids:
            continue

        # Keep the item if it matches in any id space
        if any(
            getattr(item, field.item_attr) in matched_sets[i]
            for i, field in enumerate(fields)
        ):
            kept.append(item)
            seen_ids.add(item.id)

    kept.sort(key=lambda x: x.title)

    return kept


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

    return collect_anime_items(
        radarr_client.all_movies,
        anime_mappings,
        (IdField("tmdb_movie_id", "tmdbId"), IdField("imdb_id", "imdbId")),
        (
            anibridge.all_tmdb_movie_ids if anibridge else set(),
            anibridge.all_imdb_ids if anibridge else set(),
        ),
    )
