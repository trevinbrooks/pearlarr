"""Arr-neutral anime-library id filtering.

Builds the candidate id sets for a library and keeps the items that carry an
AniList mapping - shared by both Arrs (Sonarr series and Radarr movies), so
it lives here rather than inside either client module.
"""

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple

from .anibridge import AniBridge
from .arr_models import ArrItem
from .mapping_store import IdKind
from .mappings import KnownExternalIds


@dataclass(frozen=True)
class IdField:
    """One id space to filter an Arr library by.

    Pairs the Kometa Anime-IDs mapping key with the live Arr item attribute that
    holds the same id, so `collect_anime_items` matches each item against the
    candidate set built for that id space.
    """

    id_kind: IdKind  # e.g. "tmdb_movie_id" / "tvdb_id"
    item_attr: str  # e.g. "tmdbId" / "tvdbId"


class IdFilter(NamedTuple):
    """One id space with its two candidate sets, aligned by construction.

    Each filter carries the id space AND the candidate sets built for it.
    """

    field: IdField
    anime_ids: AbstractSet[int | str]
    anibridge_ids: AbstractSet[int | str]


def collect_anime_items[ItemT: ArrItem](
    list_fn: Callable[[], list[ItemT]],
    filters: tuple[IdFilter, ...],
) -> list[ItemT]:
    """Arr library items that have an AniList mapping, sorted by name.

    Per filter, unions the precomputed Anime-IDs and AniBridge candidate sets
    (the Anime-IDs sets come from `MappingResolver.known_external_ids`), then keeps
    each item that matches at least one id space.

    Generic in `ItemT` (an `arr_models.SonarrItem` /
    `arr_models.RadarrItem`), so the filtered list returns the same
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


def build_id_filters(
    fields: tuple[IdField, ...],
    known_ids: KnownExternalIds,
    anibridge: AniBridge | None,
) -> tuple[IdFilter, ...]:
    """Build one `IdFilter` per id space for `collect_anime_items`.

    Pairs each id space's Anime-IDs candidate set with AniBridge's (empty when
    AniBridge is off). Shared by `collect_anime_movies` and Sonarr's
    `get_all_sonarr_series`, which each pass the result to `collect_anime_items`
    with their own `list_fn`.
    """

    return tuple(
        IdFilter(
            field,
            anime_ids=known_ids.known_external_ids(field.id_kind),
            anibridge_ids=anibridge.known_external_ids(field.id_kind) if anibridge else set(),
        )
        for field in fields
    )
