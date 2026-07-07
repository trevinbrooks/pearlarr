# pyright: strict
"""Characterization tests for the ``MappingEntry`` resolution boundary.

These pin the two producer paths (Kometa Anime-IDs and AniBridge) end-to-end
through ``MappingResolver.get_anilist_ids``, plus the ``.mode`` discriminant and
the field defaults. The boundary had no coverage before the TypedDict ->
dataclass deepening, so these guard the behaviour the dataclass must preserve:
which mode an entry drives, and the exact ``.get(..., default)`` values the
former dict reads produced.
"""

import dataclasses

import pytest

from seadexarr.modules.anibridge import AniBridgeGraph
from seadexarr.modules.mappings import AnimeIdsMap, ExternalIds, MappingEntry, MappingMode, MappingResolver


def _resolver(
    *,
    anime_mappings_cfg: AnimeIdsMap | bool | None = False,
    anibridge_mappings_cfg: AniBridgeGraph | bool | None = False,
) -> MappingResolver:
    """Build a resolver from in-memory configs, all on-disk sources disabled."""

    return MappingResolver(
        cache_time=1,
        ignore_anilist_ids=set(),
        anime_mappings_cfg=anime_mappings_cfg,
        anidb_mappings_cfg=False,
        anibridge_mappings_cfg=anibridge_mappings_cfg,
    )


class TestKometaPath:
    """Kometa Anime-IDs records resolve to ANIME_IDS-mode entries."""

    def test_carries_flat_fields_and_no_tvdb_mappings(self) -> None:
        resolver = _resolver(
            anime_mappings_cfg={
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

        mappings, dropped = resolver.get_anilist_ids(ExternalIds(tvdb=200))

        assert dropped == []
        assert list(mappings) == [100]
        entry = mappings[100]
        assert isinstance(entry, MappingEntry)
        assert entry.mode is MappingMode.ANIME_IDS
        assert entry.tvdb_mappings is None
        assert (entry.anilist_id, entry.tvdb_season, entry.tvdb_epoffset) == (100, 2, 3)
        assert entry.imdb_id == "tt100"
        assert entry.anidb_id == 50

    def test_absent_keys_fall_back_to_the_old_get_defaults(self) -> None:
        resolver = _resolver(
            anime_mappings_cfg={"Minimal": {"anilist_id": 101, "tvdb_id": 201}},
        )

        mappings, _ = resolver.get_anilist_ids(ExternalIds(tvdb=201))

        entry = mappings[101]
        # Exactly the former mapping.get("tvdb_season", -1) / ("tvdb_epoffset", 0)
        assert entry.tvdb_season == -1
        assert entry.tvdb_epoffset == 0
        assert entry.tvdb_mappings is None
        assert entry.tmdb_movie_id is None
        assert entry.imdb_id is None
        assert entry.anidb_id is None
        assert entry.mode is MappingMode.ANIME_IDS


class TestAniBridgePath:
    """An AniBridge tvdb-scoped lookup resolves to an ANIBRIDGE-mode entry."""

    def test_tvdb_lookup_attaches_tvdb_mappings(self) -> None:
        # anilist:269 -> tvdb_show 74796 season 2, episodes 1-21 (target side).
        graph = {
            "anilist:269": {
                "tvdb_show:74796:s2": {"21-41": "1-21"},
                "anidb:1234": {},
                "imdb_show:tt0269": {},
            },
        }
        resolver = _resolver(anibridge_mappings_cfg=graph)

        mappings, _ = resolver.get_anilist_ids(ExternalIds(tvdb=74796))

        entry = mappings[269]
        assert entry.mode is MappingMode.ANIBRIDGE
        assert entry.tvdb_mappings == {2: [(1, 21)]}
        assert entry.anilist_id == 269
        assert entry.anidb_id == 1234
        # AniBridge entries carry no flat season/offset -> the dataclass defaults.
        assert entry.tvdb_season == -1
        assert entry.tvdb_epoffset == 0


class TestModeDiscriminant:
    """``mode`` keys off presence, not truthiness (the empty-dict trap)."""

    def test_empty_tvdb_mappings_is_still_anibridge(self) -> None:
        # The former "tvdb_mappings" in mapping was True even for an empty dict.
        assert MappingEntry(anilist_id=1, tvdb_mappings={}).mode is MappingMode.ANIBRIDGE

    def test_populated_tvdb_mappings_is_anibridge(self) -> None:
        assert MappingEntry(anilist_id=1, tvdb_mappings={2: [(1, 3)]}).mode is MappingMode.ANIBRIDGE

    def test_absent_tvdb_mappings_is_anime_ids(self) -> None:
        assert MappingEntry(anilist_id=1).mode is MappingMode.ANIME_IDS
        assert MappingEntry(anilist_id=1, tvdb_mappings=None).mode is MappingMode.ANIME_IDS


def test_entry_is_frozen() -> None:
    # Immutability is what makes sharing entries across the resolver's memo safe.
    entry = MappingEntry(anilist_id=1)
    # Attribute name kept in a var so ruff's B010 (constant setattr) doesn't fire
    # and the type checker doesn't flag a static frozen-field assignment.
    field = "tvdb_season"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(entry, field, 5)
