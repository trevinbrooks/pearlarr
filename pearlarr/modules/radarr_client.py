"""Radarr REST client: the HTTP surface the Radarr syncer talks to."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple, Protocol, override

import httpx

from .anibridge import AniBridge
from .arr_http import ArrHttp
from .mapping_store import AnimeIdColumn
from .seadex_types import ArrItem, HistoryRecord, MovieFile, RadarrItem, RadarrMovie, validate_each


def make_radarr_client(
    *,
    url: str,
    api_key: str,
    http: httpx.Client,
) -> "RadarrClient":
    """Build a `RadarrClient` from the shared client and a url/key.

    Hoisted so the two Arr strategies share one construction site (and the one
    `label="Radarr"` bind). The url/key lookup policy stays with each caller
    (Radarr requires them, Sonarr reads them optionally for its cross-check), so
    they're passed in already resolved rather than re-derived here.

    Args:
        url: Radarr base URL.
        api_key: Radarr API key.
        http: The run's shared client for the raw endpoints.
    """

    return RadarrClient(
        http=ArrHttp.bind(client=http, url=url, api_key=api_key, label="Radarr"),
    )


class AbstractRadarrClient(ABC):
    """The Radarr read surface the Radarr syncer consumes.

    The nominal seam mirroring `sonarr_client.AbstractSonarrClient`:
    the strategy and `collect_anime_movies` take this type, so tests inject a
    subclassed fake that is statically checked against the real client's
    surface (a missing method is a `reportAbstractUsage` error, not a
    silently-absorbed `Any`).
    """

    @abstractmethod
    def all_movies(self) -> list[RadarrItem]:
        """Every movie in Radarr (unfiltered)."""

    @abstractmethod
    def movie_files(self, movie_id: int) -> list[MovieFile]:
        """Movie-file records for a movie (`/api/v3/moviefile`)."""

    @abstractmethod
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History records since `date` (`/api/v3/history/since`)."""


class RadarrClient(AbstractRadarrClient):
    """Thin client over the raw Radarr v3 REST endpoints."""

    def __init__(self, *, http: ArrHttp) -> None:
        """Instantiate the Radarr API client.

        Construction is network-free (no connection probe): the first request
        happens on the first method call, so an unreachable Radarr surfaces as
        that call's typed error / fail-open path, never a constructor hang.
        Warnings ride the hub — unlike Sonarr, this client emits no lines of
        its own, so it holds no logger at all.

        Args:
            http: The transport already bound to Radarr's url + key
                (`make_radarr_client` does the `label="Radarr"` bind).
        """

        self._http = http

    @override
    def all_movies(self) -> list[RadarrItem]:
        """Every movie in Radarr (`/api/v3/movie`, unfiltered).

        The one fail-CLOSED read: the library list is the run's ground truth
        (an outage reading as an empty library would silently no-op the leg),
        so a failure raises the typed `arr_http` errors for the CLI
        containment arms instead of degrading to an empty list.
        """

        raw = self._http.get_json_list_strict("/api/v3/movie")
        # Strict validation to match: a non-empty payload with zero valid
        # records raises BoundaryContractError instead of reading as empty.
        return list[RadarrItem](validate_each(RadarrMovie, raw, strict=True))

    @override
    def movie_files(self, movie_id: int) -> list[MovieFile]:
        """Movie-file records for a movie (`/api/v3/moviefile`).

        Each `MovieFileResource` is parsed into a `seadex_types.MovieFile`
        view at this client boundary.

        Returns an empty list (with a warning) on a non-200 or a transient request
        error, so the caller treats "couldn't read the files" as "no existing
        files" instead of unwinding the run.

        Args:
            movie_id: ID for the movie in Radarr.
        """

        raw = self._http.get_json_list(
            "/api/v3/moviefile",
            params={"movieId": str(movie_id)},
            warn=f"Could not fetch files for movie {movie_id} from Radarr ({{detail}}); assuming none",
        )
        if raw is None:
            return []

        # Validate each record at this boundary (junk records skip with a warning).
        return validate_each(MovieFile, raw)

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """History since `date`, or None on failure (fail-open; shared helper)."""

        return self._http.history_since(
            date,
            include_flags={"includeMovie": "false"},
        )


@dataclass(frozen=True)
class IdField:
    """One id space to filter an Arr library by.

    Pairs the Kometa Anime-IDs map key with the live Arr item attribute that
    holds the same id, so `collect_anime_items` matches each item against the
    candidate set built for that id space.
    """

    mapping_key: AnimeIdColumn  # e.g. "tmdb_movie_id" / "tvdb_id"
    item_attr: str  # e.g. "tmdbId" / "tvdbId"


class IdFilter(NamedTuple):
    """One id space with its two candidate sets, aligned by construction.

    Replaces the former fields/anime-sets/anibridge-sets parallel tuples (which
    could only be kept aligned by position): each filter carries the id space
    AND the candidate sets built for it.
    """

    field: IdField
    anime_ids: AbstractSet[int | str]
    anibridge_ids: AbstractSet[int | str]


class AnimeIdSets(Protocol):
    """The slice of `MappingResolver` the library filter needs.

    A structural type so this module needn't import `MappingResolver` (which
    imports this one). Supplies the DISTINCT Anime-IDs id set for a given column.
    """

    def anime_id_set(self, column: AnimeIdColumn) -> AbstractSet[int | str]: ...


def collect_anime_items[ItemT: ArrItem](
    list_fn: Callable[[], list[ItemT]],
    filters: tuple[IdFilter, ...],
) -> list[ItemT]:
    """Arr library items that have an AniList mapping, sorted by title.

    Per filter, unions the precomputed Anime-IDs and AniBridge candidate sets
    (the Anime-IDs sets come from `MappingResolver.anime_id_set`, no longer a
    scan of the full map), then keeps each item that matches at least one id
    space.

    Generic in `ItemT` (a `seadex_types.SonarrItem` /
    `seadex_types.RadarrItem`), so the filtered list returns the same
    concrete item type the caller fetched.

    Args:
        list_fn: Returns the unfiltered Arr item list.
        filters: Id spaces to filter by, in order, each
            carrying its own candidate sets (pass `set()` for a disabled source).
    """

    # One candidate set per id space: the two sources' sets unioned.
    matched_by_attr: list[tuple[str, AbstractSet[int | str]]] = [
        (f.field.item_attr, f.anime_ids | f.anibridge_ids) for f in filters
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
        if any(getattr(item, attr) in matched for attr, matched in matched_by_attr):
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
    `RadarrSync.get_all_radarr_movies` and Sonarr's `ignore_movies_in_radarr`
    path.

    Args:
        radarr_client: Client to fetch the movie list from.
        mappings: Resolver exposing `anime_id_set(column)` for the
            Anime-IDs candidate sets.
        anibridge: AniBridge view, exposing `id_set(mapping_key)`
            for its precomputed candidate sets.
    """

    fields = (IdField("tmdb_movie_id", "tmdbId"), IdField("imdb_id", "imdbId"))
    return collect_anime_items(
        radarr_client.all_movies,
        tuple(
            IdFilter(
                field,
                anime_ids=mappings.anime_id_set(field.mapping_key),
                anibridge_ids=anibridge.id_set(field.mapping_key) if anibridge else set(),
            )
            for field in fields
        ),
    )
