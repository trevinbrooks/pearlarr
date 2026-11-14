# pyright: strict
"""Characterization tests for the `TitleMapping` resolution boundary.

These pin the two mapping source paths (Kometa Anime-IDs and AniBridge) end-to-end
through `MappingResolver.resolve`, plus the `.kind` discriminant and
the field defaults. The boundary had no coverage before the TypedDict ->
dataclass deepening, so these guard the behavior the dataclass must preserve:
which kind a mapping has, and the exact `.get(..., default)` values the
former dict reads produced.
"""

import dataclasses
from types import MappingProxyType

import httpx
import pytest

from pearlarr.anibridge import AniBridgeGraph, SpecialAliasing
from pearlarr.mappings import (
    AnimeIdsByName,
    ExternalIds,
    MappingKind,
    MappingResolver,
    MappingSources,
    TitleMapping,
)


def _resolver(
    *,
    anime: AnimeIdsByName | bool | None = False,
    anibridge: AniBridgeGraph | bool | None = False,
) -> MappingResolver:
    """Build a resolver from in-memory configs, all on-disk sources disabled."""

    return MappingResolver(
        cache_time=1,
        ignore_anilist_ids=set(),
        web=httpx.Client(),
        sources=MappingSources(anime=anime, anidb=False, anibridge=anibridge),
    )


class TestKometaPath:
    """Kometa Anime-IDs objects resolve to SEASON_OFFSET mappings."""

    def test_carries_flat_fields_and_no_tvdb_mappings(self) -> None:
        resolver = _resolver(
            anime={
                "Some Show": {
                    "anilist_id": 100,
                    "tvdb_id": 200,
                    "tvdb_season": 2,
                    "tvdb_epoffset": 3,
                    "imdb_id": "tt100",
                    "anidb_id": 50,
                },
            },
        )

        resolved = resolver.resolve(ExternalIds(tvdb=200))
        mappings = resolved.kept

        assert resolved.ignored == {}
        assert list(mappings) == [100]
        mapping = mappings[100]
        assert isinstance(mapping, TitleMapping)
        assert mapping.kind is MappingKind.SEASON_OFFSET
        assert mapping.tvdb_mappings is None
        assert (mapping.anilist_id, mapping.tvdb_season, mapping.tvdb_epoffset) == (100, 2, 3)
        assert mapping.imdb_id == "tt100"
        assert mapping.anidb_id == 50

    def test_absent_keys_fall_back_to_the_old_get_defaults(self) -> None:
        resolver = _resolver(
            anime={"Minimal": {"anilist_id": 101, "tvdb_id": 201}},
        )

        mappings = resolver.resolve(ExternalIds(tvdb=201)).kept

        mapping = mappings[101]
        # Exactly the former mapping.get("tvdb_season", -1) / ("tvdb_epoffset", 0)
        assert mapping.tvdb_season == -1
        assert mapping.tvdb_epoffset == 0
        assert mapping.tvdb_mappings is None
        assert mapping.tmdb_movie_id is None
        assert mapping.imdb_id is None
        assert mapping.anidb_id is None
        assert mapping.kind is MappingKind.SEASON_OFFSET


class TestAniBridgePath:
    """An AniBridge tvdb-scoped lookup resolves to an EPISODE_RANGES mapping."""

    def test_tvdb_lookup_attaches_tvdb_mappings(self) -> None:
        # anilist:269 -> tvdb_show 74796 season 2, episodes 1-21 (target side).
        graph = {
            "anilist:269": {
                "tvdb_show:74796:s2": {"21-41": "1-21"},
                "anidb:1234": {},
                "imdb_show:tt0269": {},
            },
        }
        resolver = _resolver(anibridge=graph)

        mappings = resolver.resolve(ExternalIds(tvdb=74796)).kept

        mapping = mappings[269]
        assert mapping.kind is MappingKind.EPISODE_RANGES
        assert mapping.tvdb_mappings == {2: [(1, 21)]}
        assert mapping.anilist_id == 269
        assert mapping.anidb_id == 1234
        # AniBridge mappings carry no flat season/offset -> the dataclass defaults.
        assert mapping.tvdb_season == -1
        assert mapping.tvdb_epoffset == 0


class TestSpecialAliasing:
    """A mapping's `special_aliasing` defaults to None and keeps a read-only copy of the aliases it was built with."""

    def test_a_mapping_defaults_to_no_aliasing(self) -> None:
        assert TitleMapping(anilist_id=1).special_aliasing is None

    def test_the_aliases_are_a_read_only_copy_of_the_callers_dict(self) -> None:
        aliases = {13: 12}
        mapping = TitleMapping(anilist_id=1, special_aliasing=SpecialAliasing(900, aliases))

        aliases[14] = 13

        assert mapping.special_aliasing is not None
        assert mapping.special_aliasing == SpecialAliasing(900, {13: 12})
        assert isinstance(mapping.special_aliasing.aliases, MappingProxyType)

    def test_tvdb_lookup_attaches_the_paired_specials(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "7,12"}, "tmdb_show:900:s0": {"1-2": "7,13"}},
        }
        resolver = _resolver(anibridge=graph)

        mappings = resolver.resolve(ExternalIds(tvdb=500)).kept

        assert mappings[1].special_aliasing == SpecialAliasing(900, {7: 7, 13: 12})


class TestKindDiscriminant:
    """`kind` keys off presence, not truthiness (the empty-dict trap)."""

    def test_empty_tvdb_mappings_is_still_episode_ranges(self) -> None:
        # The former "tvdb_mappings" in mapping was True even for an empty dict.
        assert TitleMapping(anilist_id=1, tvdb_mappings={}).kind is MappingKind.EPISODE_RANGES

    def test_populated_tvdb_mappings_is_episode_ranges(self) -> None:
        assert TitleMapping(anilist_id=1, tvdb_mappings={2: [(1, 3)]}).kind is MappingKind.EPISODE_RANGES

    def test_absent_tvdb_mappings_is_season_offset(self) -> None:
        assert TitleMapping(anilist_id=1).kind is MappingKind.SEASON_OFFSET
        assert TitleMapping(anilist_id=1, tvdb_mappings=None).kind is MappingKind.SEASON_OFFSET


def test_mapping_is_frozen() -> None:
    # Immutability is what makes sharing mappings across the resolver's memo safe.
    mapping = TitleMapping(anilist_id=1)
    # Attribute name kept in a var so ruff's B010 (constant setattr) doesn't fire
    # and the type checker doesn't flag a static frozen-field assignment.
    field = "tvdb_season"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(mapping, field, 5)
