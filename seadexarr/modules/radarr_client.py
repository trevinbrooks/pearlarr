"""Radarr REST client: the HTTP surface the Radarr syncer talks to."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests
from arrapi import RadarrAPI

from .anibridge import AniBridge
from .seadex_types import ArrItem, MovieFile, RadarrItem


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

    def all_movies(self) -> list[RadarrItem]:
        """Every movie in Radarr (unfiltered)."""

        # arrapi ships no py.typed, so all_movies() is Unknown; the movie
        # objects expose the attribute surface of RadarrItem, so cast at this
        # client boundary into the project's typed shape.
        return cast("list[RadarrItem]", self._api.all_movies())

    def movie_files(self, movie_id: int) -> list[MovieFile]:
        """Movie-file records for a movie (``/api/v3/moviefile``).

        Each ``MovieFileResource`` is parsed into a :class:`~.seadex_types.MovieFile`
        view at this client boundary.

        Args:
            movie_id (int): ID for the movie in Radarr.
        """

        mov_req_url = f"{self._url}/api/v3/moviefile?movieId={movie_id}&apikey={self._api_key}"
        # response.json() is untyped; the moviefile endpoint returns a JSON
        # array of objects, so cast at the parse boundary, then parse each raw
        # record into the typed MovieFile view.
        raw = cast("list[dict[str, Any]]", self._session.get(mov_req_url).json())
        return [MovieFile.from_api(record) for record in raw]


@dataclass(frozen=True)
class IdField:
    """One id space to filter an Arr library by.

    Pairs the Kometa Anime-IDs map key with the live Arr item attribute that
    holds the same id, so ``collect_anime_items`` matches each item against the
    candidate set built for that id space.
    """

    mapping_key: str  # e.g. "tmdb_movie_id" / "tvdb_id"
    item_attr: str  # e.g. "tmdbId" / "tvdbId"


class AnimeIdSets(Protocol):
    """The slice of ``MappingResolver`` the library filter needs.

    A structural type so this module needn't import ``MappingResolver`` (which
    imports this one). Supplies the DISTINCT Anime-IDs id set for a given column.
    """

    def anime_id_set(self, column: str) -> set[Any]: ...


def collect_anime_items[ItemT: ArrItem](
    list_fn: Callable[[], list[ItemT]],
    fields: tuple[IdField, ...],
    anime_id_sets: tuple[set[Any], ...],
    anibridge_id_sets: tuple[set[Any], ...],
) -> list[ItemT]:
    """Arr library items that have an AniList mapping, sorted by title.

    Per ``fields`` entry, unions the precomputed Anime-IDs and AniBridge candidate
    sets, then keeps each item that matches at least one id space. Both id-set
    tuples are aligned to ``fields`` by position (the Anime-IDs sets come from
    ``MappingResolver.anime_id_set``, no longer a scan of the full map).

    Generic in ``ItemT`` (a :class:`~.seadex_types.SonarrItem` /
    :class:`~.seadex_types.RadarrItem`), so the filtered list returns the same
    concrete item type the caller fetched.

    Args:
        list_fn (Callable[[], list[ItemT]]): Returns the unfiltered Arr item list.
        fields (tuple[IdField, ...]): Id spaces to filter by, in order.
        anime_id_sets (tuple[set[Any], ...]): Anime-IDs candidate sets, one per
            ``fields`` entry in the same order.
        anibridge_id_sets (tuple[set[Any], ...]): AniBridge candidate sets, one
            per ``fields`` entry in the same order (pass ``set()`` when disabled).
    """

    # One candidate set per id space: the two sources' sets unioned.
    matched_sets: list[set[Any]] = [
        anime | bridge for anime, bridge in zip(anime_id_sets, anibridge_id_sets, strict=True)
    ]

    # Track kept item ids in a set: "item not in kept" on a growing list is O(n)
    # per check (and compares whole item objects), making the scan quadratic on
    # a large library
    kept: list[ItemT] = []
    seen_ids: set[int] = set()
    for item in list_fn():
        if item.id in seen_ids:
            continue

        # Keep the item if it matches in any id space
        if any(getattr(item, field.item_attr) in matched for field, matched in zip(fields, matched_sets, strict=True)):
            kept.append(item)
            seen_ids.add(item.id)

    kept.sort(key=lambda x: x.title)

    return kept


def collect_anime_movies(
    radarr_client: RadarrClient,
    mappings: AnimeIdSets,
    anibridge: AniBridge | None,
) -> list[RadarrItem]:
    """Radarr movies that have an AniList mapping, sorted by title.

    Gathers the candidate TMDB/IMDb id sets from the two mapping sources, then
    keeps each Radarr movie that matches one of them. Shared by
    ``RadarrSync.get_all_radarr_movies`` and Sonarr's ``ignore_movies_in_radarr``
    path.

    Args:
        radarr_client (RadarrClient): Client to fetch the movie list from.
        mappings (AnimeIdSets): Resolver exposing ``anime_id_set(column)`` for the
            Anime-IDs candidate sets.
        anibridge (AniBridge | None): AniBridge view, exposing ``id_set(mapping_key)``
            for its precomputed candidate sets.
    """

    fields = (IdField("tmdb_movie_id", "tmdbId"), IdField("imdb_id", "imdbId"))
    return collect_anime_items(
        radarr_client.all_movies,
        fields,
        tuple(mappings.anime_id_set(f.mapping_key) for f in fields),
        tuple(anibridge.id_set(f.mapping_key) if anibridge else set() for f in fields),
    )
