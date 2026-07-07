"""Radarr REST client: the HTTP surface the Radarr syncer talks to."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Protocol, cast, override

import httpx

from .anibridge import AniBridge
from .arr_http import ArrHttp
from .seadex_types import ArrItem, HistoryRecord, MovieFile, RadarrItem, RadarrMovie


def make_radarr_client(
    *,
    url: str,
    api_key: str,
    http: httpx.Client,
    logger: logging.Logger,
) -> "RadarrClient":
    """Build a :class:`RadarrClient` from the shared client/logger and a url/key.

    Hoisted so the two Arr strategies share one construction site. The url/key
    lookup policy stays with each caller (Radarr requires them, Sonarr reads them
    optionally for its cross-check), so they're passed in already resolved rather
    than re-derived here.

    Args:
        url (str): Radarr base URL.
        api_key (str): Radarr API key.
        http (httpx.Client): The run's shared client for the raw endpoints.
        logger (logging.Logger): For request warnings.
    """

    return RadarrClient(
        url=url,
        api_key=api_key,
        http=http,
        logger=logger,
    )


class AbstractRadarrClient(ABC):
    """The Radarr read surface the Radarr syncer consumes.

    The nominal seam mirroring :class:`~.sonarr_client.AbstractSonarrClient`:
    the strategy and ``collect_anime_movies`` take this type, so tests inject a
    subclassed fake that is statically checked against the real client's
    surface (a missing method is a ``reportAbstractUsage`` error, not a
    silently-absorbed ``Any``).
    """

    @abstractmethod
    def all_movies(self) -> list[RadarrItem]:
        """Every movie in Radarr (unfiltered)."""

    @abstractmethod
    def movie_files(self, movie_id: int) -> list[MovieFile]:
        """Movie-file records for a movie (``/api/v3/moviefile``)."""

    @abstractmethod
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History records since ``date`` (``/api/v3/history/since``)."""


class RadarrClient(AbstractRadarrClient):
    """Thin client over the raw Radarr v3 REST endpoints."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        http: httpx.Client,
        logger: logging.Logger,
    ) -> None:
        """Instantiate the Radarr API client.

        Construction is network-free (no connection probe): the first request
        happens on the first method call, so an unreachable Radarr surfaces as
        that call's typed error / fail-open path, never a constructor hang.

        Args:
            url (str): Radarr base URL (a trailing slash is normalized away by
                the bind - a "//api" join redirects to the login page).
            api_key (str): Radarr API key, sent as the ``X-Api-Key`` header (never
                a query param, so it can't leak through URLs in logs/exceptions).
            http (httpx.Client): The run's shared client the bound
                :class:`~.arr_http.ArrHttp` rides (shared across arrs, so the
                key travels per request, never on the client).
            logger (logging.Logger): For request warnings.
        """

        self._http = ArrHttp.bind(client=http, url=url, api_key=api_key, label="Radarr", logger=logger)

    @override
    def all_movies(self) -> list[RadarrItem]:
        """Every movie in Radarr (``/api/v3/movie``, unfiltered).

        The one fail-CLOSED read: the library list is the run's ground truth
        (an outage reading as an empty library would silently no-op the leg),
        so a failure raises the typed :mod:`~.arr_http` errors for the CLI
        containment arms instead of degrading to an empty list.
        """

        raw = self._http.get_json_list_strict("/api/v3/movie")
        # Element dicts are unvalidated JSON: cast at the parse boundary, skip strays.
        return [RadarrMovie.from_api(cast("dict[str, Any]", record)) for record in raw if isinstance(record, dict)]

    @override
    def movie_files(self, movie_id: int) -> list[MovieFile]:
        """Movie-file records for a movie (``/api/v3/moviefile``).

        Each ``MovieFileResource`` is parsed into a :class:`~.seadex_types.MovieFile`
        view at this client boundary.

        Returns an empty list (with a warning) on a non-200 or a transient request
        error, so the caller treats "couldn't read the files" as "no existing
        files" instead of unwinding the run.

        Args:
            movie_id (int): ID for the movie in Radarr.
        """

        raw = self._http.get_json_list(
            "/api/v3/moviefile",
            params={"movieId": str(movie_id)},
            warn=f"Could not fetch files for movie {movie_id} from Radarr ({{detail}}); assuming none",
        )
        if raw is None:
            return []

        # Each element is an unvalidated MovieFileResource object: cast at the
        # parse boundary into the typed MovieFile view.
        return [MovieFile.from_api(cast("dict[str, Any]", record)) for record in raw]

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History since ``date``, or None on failure (fail-open; shared helper)."""

        return self._http.history_since(
            date,
            include_flags={"includeMovie": "false"},
            item_key="movieId",
        )


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

    def anime_id_set(self, column: str) -> AbstractSet[int | str]: ...


def collect_anime_items[ItemT: ArrItem](
    list_fn: Callable[[], list[ItemT]],
    fields: tuple[IdField, ...],
    anime_id_sets: tuple[AbstractSet[int | str], ...],
    anibridge_id_sets: tuple[AbstractSet[int | str], ...],
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
        anime_id_sets (tuple[AbstractSet[int | str], ...]): Anime-IDs candidate
            sets, one per ``fields`` entry in the same order.
        anibridge_id_sets (tuple[AbstractSet[int | str], ...]): AniBridge candidate
            sets, one per ``fields`` entry in the same order (pass ``set()`` when
            disabled).
    """

    # One candidate set per id space: the two sources' sets unioned.
    matched_sets: list[AbstractSet[int | str]] = [
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
    radarr_client: AbstractRadarrClient,
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
