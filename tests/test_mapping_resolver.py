# pyright: strict
# pyright: reportPrivateUsage=false
# Drives module-private resolver internals under test (_mapping_from_raw, m._parse_*, resolver._(maybe_)download*).
"""Parity + edge tests for the SQL-backed `MappingResolver` / `AniBridge`.

The SQL backings must reproduce the in-memory implementations exactly. AniBridge's
graph-backed view is used directly as the oracle (the unchanged parser). anime_ids
is checked against an inline reconstruction of the former reverse-index merge, and
the anidb + hash-gate behaviors are pinned. These are the safety net for the
"parse once, serve from SQL" migration - the resolver lookups had no direct
coverage before.
"""

import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import httpx
import pytest
import respx

import pearlarr.mappings as m
from pearlarr.anibridge import (
    AniBridge,
    AniBridgeGraph,
    SpecialAliasing,
)
from pearlarr.mapping_store import AnimeIdRow, MappingStore
from pearlarr.mappings import (
    ANIBRIDGE_MAPPINGS_FILE,
    ANIDB_MAPPINGS_FILE,
    ANIME_IDS_FILE,
    AnimeIdsByName,
    AnimeIdsRawMapping,
    ExternalIds,
    MappingKind,
    MappingResolver,
    MappingSource,
    MappingSources,
    TitleMapping,
    _mapping_from_raw,
)
from pearlarr.output import Severity
from pearlarr.paths import resolve_paths
from pearlarr.sonarr_episodes import TitleEpisodes, check_ep_by_anibridge

from .builders import make_bare_instance, make_sonarr_episodes, sonarr_ep
from .fakes import debug_logger, diagnostic_messages, install_recording_hub

# One inert shared client for the resolver ctors (inline sources: nothing downloads).
_WEB = httpx.Client()

_ANIBRIDGE = MappingSource.ANIBRIDGE


def _resolver(
    *,
    anime: AnimeIdsByName | bool = False,
    anibridge: AniBridgeGraph | bool = False,
    ignore: AbstractSet[int] = frozenset(),
) -> MappingResolver:
    """A resolver over inline sources, each off unless given, with AniDB off. The conftest closes it after the test."""

    return MappingResolver(
        refresh_days=1,
        ignore_anilist_ids=set(ignore),
        sources=MappingSources(anime=anime, anidb=False, anibridge=anibridge),
        web=httpx.Client(),
    )


# --------------------------------------------------------------------------- #
# AniBridge
# --------------------------------------------------------------------------- #

GRAPH: AniBridgeGraph = {
    "anilist:269": {
        "tvdb_show:74796:s2": {"21-41": "1-21"},
        "tvdb_show:74796:s3": {"42-52": "1-9,11-12"},
        "anidb:1234": {},
        "imdb_show:tt0269": {},
        # Providers a lookup doesn't read. Real graphs carry them.
        "mal:111": {},
        "tmdb_show:5000:s1": {"1-12": "1-12"},
    },
    "anilist:270": {
        "tvdb_show:74796:s1": {},
        "tmdb_movie:888": {},
        "tvdb_movie:999": {},
    },
    "anilist:271": {
        "tvdb_show:600:s1": {"1-": "1-"},
    },
    "anilist:300": {
        "tvdb_show:111:s1": {"1-5": "1-5"},
        "tvdb_show:222:s1": {"6-10": "1-5"},
        "imdb_show:tt300": {},
    },
    "anilist:400": {
        "imdb_movie:tt0400": {},
    },
}


class TestAniBridgeLookup:
    """A lookup finds every title AniBridge maps an id to, with the ranges of the show on a TVDB lookup."""

    @pytest.mark.parametrize(
        ("ids", "expected"),
        [
            pytest.param(
                ExternalIds(tvdb=74796, imdb="tt0269"),
                {
                    # Found by its IMDb id too, but the TVDB lookup's ranges stay.
                    269: TitleMapping(
                        269,
                        tvdb_mappings={2: [(1, 21)], 3: [(1, 9), (11, 12)]},
                        imdb_id="tt0269",
                        anidb_id=1234,
                        source=_ANIBRIDGE,
                    ),
                    # A season with no ranges covers the whole season, where a missing one covers none of it.
                    270: TitleMapping(270, tvdb_mappings={1: []}, tmdb_movie_id=888, source=_ANIBRIDGE),
                },
                id="series",
            ),
            pytest.param(
                ExternalIds(tvdb=600),
                {271: TitleMapping(271, tvdb_mappings={1: [(1, None)]}, source=_ANIBRIDGE)},
                id="series with an open-ended range",
            ),
            pytest.param(
                ExternalIds(tvdb=111, imdb="tt300"),
                {300: TitleMapping(300, tvdb_mappings={1: [(1, 5)]}, imdb_id="tt300", source=_ANIBRIDGE)},
                id="series one of a title's two shows",
            ),
            pytest.param(
                ExternalIds(tmdb_movie=888, imdb="tt0888888"),
                {270: TitleMapping(270, tmdb_movie_id=888, source=_ANIBRIDGE)},
                id="movie by its tmdb id",
            ),
            pytest.param(
                ExternalIds(tmdb_movie=404, imdb="tt0400"),
                {400: TitleMapping(400, imdb_id="tt0400", source=_ANIBRIDGE)},
                id="movie by its imdb id",
            ),
            pytest.param(ExternalIds(tvdb=424242, imdb="tt0424242"), {}, id="unknown"),
        ],
    )
    def test_a_lookup_finds_each_title_with_the_ranges_of_its_show(
        self, ids: ExternalIds, expected: dict[int, TitleMapping]
    ) -> None:
        assert _resolver(anibridge=GRAPH).resolve(ids).kept == expected

    def test_the_library_filter_keeps_every_id_a_title_is_found_by(self) -> None:
        anibridge = _resolver(anibridge=GRAPH).anibridge
        assert anibridge is not None

        assert anibridge.known_external_ids("tvdb_id") == {74796, 600, 111, 222}
        assert anibridge.known_external_ids("tmdb_movie_id") == {888}
        assert anibridge.known_external_ids("imdb_id") == {"tt0269", "tt300", "tt0400"}


class TestAniBridgeRanges:
    """A title's target range string says which TVDB episodes of the season it covers."""

    @pytest.mark.parametrize(
        ("episodes", "covered"),
        [
            pytest.param({"1-12": "13-24"}, list(range(13, 25)), id="closed"),
            pytest.param({"1-": "13-"}, list(range(13, 31)), id="open end"),
            pytest.param({"1": "13"}, [13], id="one episode"),
            pytest.param({"1-11": "13-17,19-24"}, [13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24], id="two pieces"),
            # The ratio says how many source episodes a target one stands for, not which ones are covered.
            pytest.param({"1-24": "13-24|2"}, list(range(13, 25)), id="ratio"),
        ],
    )
    def test_a_target_range_covers_its_episodes(self, episodes: dict[str, str], covered: list[int]) -> None:
        resolver = _resolver(anibridge={"anilist:2": {"tvdb_show:500:s1": episodes}})
        ranges = resolver.resolve(ExternalIds(tvdb=500)).kept[2].tvdb_mappings
        assert ranges is not None

        found = [n for n in range(11, 31) if check_ep_by_anibridge(ep=sonarr_ep(1, n), tvdb_mappings=ranges)]

        assert found == covered


def _aliasing_on(graph: AniBridgeGraph, tvdb_id: int) -> dict[int, SpecialAliasing]:
    """The specials aliasing of each title a lookup of TVDB show `tvdb_id` finds, for the titles that have any."""

    kept = _resolver(anibridge=graph).resolve(ExternalIds(tvdb=tvdb_id)).kept
    return {anilist_id: aliasing for anilist_id, mapping in kept.items() if (aliasing := mapping.special_aliasing)}


class TestSpecialAliases:
    """A title's TMDB specials numbers pair by position with its TVDB ones, within each source range both map."""

    def test_each_source_range_pairs_by_position(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {
                "tvdb_show:500:s0": {"1-3": "7,12-13", "4-5": "15,17"},
                "tmdb_show:900:s0": {"1-3": "7,13-14", "4-5": "15-16"},
            },
        }

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {7: 7, 13: 12, 14: 13, 15: 15, 16: 17})}

    def test_each_title_on_a_show_carries_its_own_pairs(self) -> None:
        # Two seasons' titles, each with its own special on the show.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1": "5"}, "tmdb_show:900:s0": {"1": "9"}},
            "anilist:2": {"tvdb_show:500:s0": {"1": "6"}, "tmdb_show:900:s0": {"1": "10"}},
        }

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {9: 5}), 2: SpecialAliasing(900, {10: 6})}

    def test_a_tmdb_target_without_a_show_id_pairs_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1": "5"}, "tmdb_show:x:s0": {"1": "9"}},
        }

        assert _aliasing_on(graph, 500) == {}

    def test_a_ratio_piece_pairs_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {
                "tvdb_show:500:s0": {"1-2": "3-4|2", "3": "9"},
                "tmdb_show:900:s0": {"1-2": "5-6", "3": "8"},
            },
        }

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {8: 9})}

    def test_an_open_end_pairs_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-": "3-"}, "tmdb_show:900:s0": {"1-": "5-"}},
        }

        assert _aliasing_on(graph, 500) == {}

    def test_targets_of_different_lengths_pair_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "5-6"}, "tmdb_show:900:s0": {"1-2": "5-7"}},
        }

        assert _aliasing_on(graph, 500) == {}

    def test_a_source_of_another_length_pairs_nothing(self) -> None:
        # The targets agree with each other but not with the source range they map.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-6": "11-17"}, "tmdb_show:900:s0": {"1-6": "12-18"}},
        }

        assert _aliasing_on(graph, 500) == {}

    def test_two_tvdb_specials_targets_pair_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:2": {
                "tvdb_show:500:s0": {"1": "3"},
                "tvdb_show:501:s0": {"1": "4"},
                "tmdb_show:900:s0": {"1": "5"},
            },
        }

        assert _aliasing_on(graph, 500) == {}

    def test_a_tmdb_number_paired_two_ways_is_dropped(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {
                "tvdb_show:500:s0": {"1": "5", "2": "6", "3": "7"},
                "tmdb_show:900:s0": {"1": "9", "2": "9", "3": "8"},
            },
        }

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {8: 7})}

    def test_an_alias_onto_a_special_another_title_maps_is_dropped(self) -> None:
        # Title 2 maps TVDB special 5 in its own season 0, so title 1's alias onto 5 is dropped. Its alias onto 4 stays.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "4-5"}, "tmdb_show:900:s0": {"1-2": "4-5"}},
            "anilist:2": {"tvdb_show:500:s0": {"1": "5"}},
        }

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {4: 4})}

    def test_another_title_mapping_the_whole_specials_season_drops_every_alias(self) -> None:
        # An empty dict covers the whole season, so title 2 maps every special and title 1 keeps no alias.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "4-5"}, "tmdb_show:900:s0": {"1-2": "4-5"}},
            "anilist:2": {"tvdb_show:500:s0": {}},
        }

        assert _aliasing_on(graph, 500) == {}

    def test_aliases_attach_on_a_tvdb_lookup_only(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {
                "tvdb_show:500:s0": {"1": "5"},
                "tmdb_show:900:s0": {"1": "9"},
                "imdb_show:tt1": {},
            },
        }

        # A series of another show that shares the title's IMDb id.
        by_imdb = _resolver(anibridge=graph).resolve(ExternalIds(tvdb=777, imdb="tt1")).kept

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {9: 5})}
        assert by_imdb[1].special_aliasing is None

    def test_a_reversed_piece_pairs_nothing(self) -> None:
        # If `5-4` were read as empty, `7,8` would line up with the two-number source and pair wrongly.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "5-4,7,8"}, "tmdb_show:900:s0": {"1-2": "9-10"}},
        }

        assert _aliasing_on(graph, 500) == {}

    @pytest.mark.parametrize("target", ["x,7,8", "7,,8"])
    def test_a_junk_piece_pairs_nothing(self, target: str) -> None:
        # If the junk piece were skipped, `7,8` would line up with the two-number source and pair wrongly.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": target}, "tmdb_show:900:s0": {"1-2": "9-10"}},
        }

        assert _aliasing_on(graph, 500) == {}


# --------------------------------------------------------------------------- #
# Kometa Anime-IDs, and how the sources merge
# --------------------------------------------------------------------------- #

AMAP: AnimeIdsByName = {
    "A": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 2, "tvdb_epoffset": 3, "imdb_id": "tt100", "anidb_id": 50},
    "B": {"anilist_id": 101, "tvdb_id": 200},  # same tvdb id, 2nd anilist
    "C": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 9},  # dup anilist -> first wins
    "D": {"anilist_id": 102, "imdb_id": "tt100"},  # shares imdb
    "E": {"tvdb_id": 999},  # no anilist -> filter-set only
    "F": {"anilist_id": 103, "tmdb_movie_id": 7000},
}


def _anime_oracle(amap: AnimeIdsByName) -> Callable[[ExternalIds], dict[int, TitleMapping]]:
    """Inline twin of the former reverse-index + no-clobber merge (the oracle): one index, then a lookup per call."""

    index: dict[str, dict[object, list[AnimeIdsRawMapping]]] = {
        "tvdb_id": {},
        "tmdb_movie_id": {},
        "imdb_id": {},
    }
    for rec in amap.values():
        if rec.get("anilist_id") is None:
            continue
        for field, bucket in index.items():
            value = rec.get(field)
            if value is not None:
                bucket.setdefault(value, []).append(rec)

    def lookup(ids: ExternalIds) -> dict[int, TitleMapping]:
        result: dict[int, TitleMapping] = {}
        for field, value in (("tvdb_id", ids.tvdb), ("tmdb_movie_id", ids.tmdb_movie), ("imdb_id", ids.imdb)):
            for rec in index[field].get(value, []):  # None is never a key
                aid = rec["anilist_id"]
                if aid not in result:
                    result[aid] = _mapping_from_raw(aid, rec)
        return result

    return lookup


_RECORD_A = TitleMapping(100, tvdb_season=2, tvdb_epoffset=3, imdb_id="tt100", anidb_id=50)


class TestAnimeIdsLookup:
    """A lookup finds every Kometa record its ids reach, the first record of each AniList title.

    Covers explicit nulls and the library filter's sets.
    """

    @pytest.mark.parametrize(
        ("ids", "expected"),
        [
            pytest.param(ExternalIds(tvdb=200), {100: _RECORD_A, 101: TitleMapping(101)}, id="series"),
            pytest.param(
                ExternalIds(tvdb=200, imdb="tt100"),
                {100: _RECORD_A, 101: TitleMapping(101), 102: TitleMapping(102, imdb_id="tt100")},
                id="series with an imdb id",
            ),
            pytest.param(
                ExternalIds(tmdb_movie=7000, imdb="tt7000"),
                {103: TitleMapping(103, tmdb_movie_id=7000)},
                id="movie by its tmdb id",
            ),
            pytest.param(
                ExternalIds(tmdb_movie=4242, imdb="tt100"),
                {100: _RECORD_A, 102: TitleMapping(102, imdb_id="tt100")},
                id="movie by its imdb id",
            ),
            pytest.param(ExternalIds(tvdb=424242), {}, id="unknown"),
        ],
    )
    def test_a_lookup_finds_every_record_its_ids_reach(
        self, ids: ExternalIds, expected: dict[int, TitleMapping]
    ) -> None:
        assert _resolver(anime=AMAP).resolve(ids).kept == expected

    def test_an_explicit_null_season_reads_as_no_season(self) -> None:
        # Kometa writes some records with explicit nulls. They read like a record without the keys.
        amap = {"N": {"anilist_id": 500, "tvdb_id": 5000, "tvdb_season": None, "tvdb_epoffset": None}}

        assert _resolver(anime=amap).resolve(ExternalIds(tvdb=5000)).kept == {500: TitleMapping(500)}

    def test_candidate_sets_include_objects_without_anilist(self) -> None:
        # The former full-dict walk added external ids even from anilist-less objects
        # (object E). The SQL distinct must keep that, even though lookups skip them.
        resolver = _resolver(anime=AMAP)

        assert resolver.known_external_ids("tvdb_id") == {200, 999}
        assert resolver.known_external_ids("imdb_id") == {"tt100"}
        assert resolver.known_external_ids("tmdb_movie_id") == {7000}
        assert resolver.mappings_from_anime_ids(ExternalIds(tvdb=999)) == {}  # E has no anilist


class TestResolveMerge:
    """`resolve` merges AniBridge (winning) with anime-ids fallback, sets ignored ids aside, and sorts by id."""

    def test_anibridge_wins_then_anime_fills_then_the_ignored_are_set_aside_and_sorted(self) -> None:
        # Series 74796: AniBridge's ranges for 269 win over Kometa's season 7, Kometa alone has 100, and AniBridge
        # alone has 270, which the ignore list sets aside.
        anime = {
            "x": {"anilist_id": 269, "tvdb_id": 74796, "tvdb_season": 7},
            "y": {"anilist_id": 100, "tvdb_id": 74796, "tvdb_season": 1},
        }
        resolver = _resolver(anime=anime, anibridge=GRAPH, ignore={270})

        resolved = resolver.resolve(ExternalIds(tvdb=74796))

        assert list(resolved.kept.items()) == [
            (100, TitleMapping(100, tvdb_season=1)),
            (
                269,
                TitleMapping(
                    269,
                    tvdb_mappings={2: [(1, 21)], 3: [(1, 9), (11, 12)]},
                    imdb_id="tt0269",
                    anidb_id=1234,
                    source=_ANIBRIDGE,
                ),
            ),
        ]
        assert resolved.ignored == {270: TitleMapping(270, tvdb_mappings={1: []}, tmdb_movie_id=888, source=_ANIBRIDGE)}


class TestAniBridgeTitleWithoutTvdbRanges:
    """A title AniBridge reaches by IMDb alone has no TVDB season ranges.

    A series takes Kometa's season for it when Kometa has one, and covers none of its episodes when Kometa has none.
    """

    def test_sonarr_context_kometa_overrides_a_mapping_without_tvdb_mappings(self) -> None:
        # Kometa has the season for title 400 on the series' TVDB id.
        resolver = _resolver(
            anime={"k": {"anilist_id": 400, "tvdb_id": 555, "tvdb_season": 2, "imdb_id": "tt400"}},
            anibridge={"anilist:400": {"imdb_show:tt400": {}}},
        )

        mappings = resolver.resolve(ExternalIds(tvdb=555, imdb="tt400")).kept

        assert mappings == {400: TitleMapping(400, tvdb_season=2, imdb_id="tt400")}

    def test_sonarr_movie_as_special_keeps_kometa_season_zero(self) -> None:
        # A movie the series lists as a special: Kometa maps title 600 to season 0, and AniBridge reaches it by the
        # movie's IMDb id, which the series shares. AniBridge's mapping would read as season -1 and cover no special.
        resolver = _resolver(
            anime={"k": {"anilist_id": 600, "tvdb_id": 555, "tvdb_season": 0, "imdb_id": "tt600"}},
            anibridge={"anilist:600": {"imdb_movie:tt600": {}}},
        )

        mappings = resolver.resolve(ExternalIds(tvdb=555, imdb="tt600")).kept

        assert mappings == {600: TitleMapping(600, tvdb_season=0, imdb_id="tt600")}

    def test_imdb_only_mapping_without_tvdb_mappings_makes_sonarr_skip(self) -> None:
        # With no Kometa record the title keeps AniBridge's mapping, and the series covers none of its episodes
        # rather than every season.
        resolver = _resolver(anibridge={"anilist:500": {"imdb_show:tt500": {}}})

        mappings = resolver.resolve(ExternalIds(tvdb=777, imdb="tt500")).kept

        without_tvdb_mappings = mappings[500]
        assert without_tvdb_mappings.source is MappingSource.ANIBRIDGE
        assert without_tvdb_mappings.kind is MappingKind.SEASON_OFFSET
        assert without_tvdb_mappings.tvdb_mappings is None

        episodes = make_sonarr_episodes(series_episodes={777: [sonarr_ep(s, 1) for s in (1, 2, 3)]})
        resolved = episodes.title_episodes(777, 500, without_tvdb_mappings)
        assert isinstance(resolved, TitleEpisodes)
        assert resolved.aired == ()


# --------------------------------------------------------------------------- #
# AniDB anime-lists
# --------------------------------------------------------------------------- #

ANIDB_XML = """<anime-list>
  <anime anidbid="1"><mapping-list>
    <mapping tvdbseason="1">1-1;2-2</mapping>
    <mapping tvdbseason="2">3-1;4-2</mapping>
  </mapping-list></anime>
  <anime anidbid="2"></anime>
  <anime anidbid="2"></anime>
  <anime anidbid="3"><mapping-list>
    <mapping tvdbseason="1">1-1</mapping>
    <mapping tvdbseason="1">5-5;6-6</mapping>
  </mapping-list></anime>
</anime-list>"""


def _anidb_resolver() -> MappingResolver:
    root = ElementTree.fromstring(ANIDB_XML)
    return MappingResolver(
        refresh_days=1,
        ignore_anilist_ids=set(),
        web=httpx.Client(),
        sources=MappingSources(
            anime=False,
            anidb=root,
            anibridge=False,
        ),
    )


class TestAnidbMappingDict:
    """`anidb_episode_by_tvdb` resolves season-scoped id->episode maps.

    Missing season/id reads as empty, a repeated season keeps the last one, an ambiguous id raises, and a
    disabled anidb source is always empty.
    """

    def test_season_scoped_mapping(self) -> None:
        resolver = _anidb_resolver()

        assert resolver.has_anidb
        assert resolver.anidb_episode_by_tvdb(1, 1) == {1: {1: 1, 2: 2}}
        assert resolver.anidb_episode_by_tvdb(1, 2) == {2: {1: 3, 2: 4}}

    def test_missing_season_or_id_is_empty(self) -> None:
        resolver = _anidb_resolver()

        assert resolver.anidb_episode_by_tvdb(1, 5) == {}  # id known, season absent
        assert resolver.anidb_episode_by_tvdb(404, 1) == {}  # id unknown

    def test_repeated_season_is_last_wins(self) -> None:
        resolver = _anidb_resolver()
        try:
            assert resolver.anidb_episode_by_tvdb(3, 1) == {1: {5: 5, 6: 6}}
        finally:
            resolver.close()

    def test_ambiguous_id_raises(self) -> None:
        resolver = _anidb_resolver()

        with pytest.raises(ValueError, match="appears in more than one anime-lists mapping"):
            resolver.anidb_episode_by_tvdb(2, 1)

    def test_disabled_source_is_empty_and_not_available(self) -> None:
        resolver = _resolver()

        assert not resolver.has_anidb
        assert resolver.anidb_episode_by_tvdb(1, 1) == {}


def _boot(data_dir: Path, sources: MappingSources, *, logger: logging.Logger | None = None) -> MappingResolver:
    """A resolver booted as a run boots it, its source files and mappings.db in `data_dir`, refreshing daily."""

    return MappingResolver(
        refresh_days=1,
        ignore_anilist_ids=set(),
        sources=sources,
        web=httpx.Client(),
        mappings_db=str(data_dir / "mappings.db"),
        logger=logger,
    )


def _untimed(messages: list[str]) -> list[str]:
    """Log lines with each parse time read as 0.00s."""

    return [re.sub(r"\d+\.\d\ds", "0.00s", message) for message in messages]


_COWBOY_BEBOP = '<anime anidbid="23" tvdbid="76885" defaulttvdbseason="1"><name>Cowboy Bebop</name><mapping-list>'
_ITS_SPECIALS = '<mapping anidbseason="0" tvdbseason="0">;1-1;</mapping></mapping-list></anime>'
_ONE_PIECE = (
    '<name>One Piece</name><mapping-list><mapping anidbseason="0" tvdbseason="0">;1-1;</mapping></mapping-list>'
)


class TestAnidbSkipTallies:
    """The boot line for an anime-lists file counts what it skipped by kind, and the rest still maps.

    An anime with no usable AniDB id is malformed, which healthy upstream data never has. A mapping form the reader
    doesn't take (the offset form, which has no text, or an AniDB episode spread over several TVDB ones) is
    unsupported, and a healthy file has hundreds. Each skipped piece comes before a good one, so a skip that stopped
    the read would show.
    """

    @pytest.mark.parametrize(
        ("anime_list", "tally"),
        [
            pytest.param(
                f'<anime tvdbid="81797">{_ONE_PIECE}</anime>{_COWBOY_BEBOP}{_ITS_SPECIALS}',
                ", 1 malformed skipped",
                id="missing anidbid",
            ),
            pytest.param(
                f'<anime anidbid="69x" tvdbid="81797">{_ONE_PIECE}</anime>{_COWBOY_BEBOP}{_ITS_SPECIALS}',
                ", 1 malformed skipped",
                id="non-int anidbid",
            ),
            pytest.param(
                f'{_COWBOY_BEBOP}<mapping anidbseason="1" tvdbseason="1" start="1" end="26" offset="0"/>{_ITS_SPECIALS}',
                ", 1 unsupported mapping forms",
                id="offset form",
            ),
            pytest.param(
                f'{_COWBOY_BEBOP}<mapping anidbseason="1" tvdbseason="1">;3-6+7;</mapping>{_ITS_SPECIALS}',
                ", 1 unsupported mapping forms",
                id="multi-episode span",
            ),
        ],
    )
    def test_summary_line_carries_nonzero_tallies_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, anime_list: str, tally: str
    ) -> None:
        (tmp_path / ANIDB_MAPPINGS_FILE).write_text(f"<anime-list>{anime_list}</anime-list>", encoding="utf-8")

        with caplog.at_level("DEBUG"):
            resolver = _boot(
                tmp_path,
                MappingSources(anime=False, anidb=None, anibridge=False),
                logger=debug_logger("pearlarr-anidb-tallies"),
            )

        assert _untimed(caplog.messages) == [
            "Parsing + indexing anime-lists ...",
            f"Indexed anime-lists (1 mappings, 0.00s{tally})",
        ]
        assert resolver.anidb_episode_by_tvdb(23, 0) == {0: {1: 1}}


# --------------------------------------------------------------------------- #
# The source files and the mappings database
# --------------------------------------------------------------------------- #

_FROM_FILES = MappingSources(anime=None, anidb=None, anibridge=None)
_FRIEREN: AnimeIdsByName = {"Sousou no Frieren": {"anilist_id": 154587, "tvdb_id": 424536, "tvdb_season": 1}}


class TestSourceHashGate:
    """A boot indexes a source file again only when its content changed."""

    def test_unchanged_file_is_not_reparsed_changed_file_is(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        anime_ids = tmp_path / ANIME_IDS_FILE
        anime_ids.write_text(json.dumps(_FRIEREN))
        (tmp_path / ANIBRIDGE_MAPPINGS_FILE).write_text(
            json.dumps({"anilist:182255": {"tvdb_show:424536:s2": {"1-10": "1-10"}}})
        )
        (tmp_path / ANIDB_MAPPINGS_FILE).write_text(f"<anime-list>{_COWBOY_BEBOP}{_ITS_SPECIALS}</anime-list>")
        _boot(tmp_path, _FROM_FILES).close()
        logger = debug_logger("pearlarr-source-hash-gate")

        with caplog.at_level("DEBUG"):
            warm = _boot(tmp_path, _FROM_FILES, logger=logger)

        assert caplog.messages == [
            "anime_ids.json unchanged; using cached index",
            "anime-lists unchanged; using cached index",
            "anibridge mappings unchanged; using cached index",
        ]
        assert warm.resolve(ExternalIds(tvdb=424536)).kept == {
            154587: TitleMapping(154587, tvdb_season=1),
            182255: TitleMapping(182255, tvdb_mappings={2: [(1, 10)]}, source=_ANIBRIDGE),
        }
        assert warm.anidb_episode_by_tvdb(23, 0) == {0: {1: 1}}

        anime_ids.write_text(json.dumps({"Sousou no Frieren": {**_FRIEREN["Sousou no Frieren"], "tvdb_season": 2}}))
        caplog.clear()
        with caplog.at_level("DEBUG"):
            changed = _boot(tmp_path, _FROM_FILES, logger=logger)

        assert _untimed(caplog.messages) == [
            "Parsing + indexing anime_ids.json ...",
            "Indexed anime_ids.json (1 rows, 0.00s)",
            "anime-lists unchanged; using cached index",
            "anibridge mappings unchanged; using cached index",
        ]
        assert changed.resolve(ExternalIds(tvdb=424536)).kept[154587] == TitleMapping(154587, tvdb_season=2)


class TestConstructionFailureClosesStore:
    """A non-`DatabaseError` raised during `_build` still closes the store before it propagates."""

    def test_store_closed_when_build_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-DatabaseError from _build must still close the store (the resolver
        # is never returned, so no one else can). Spy on close().
        closed = {"n": 0}
        real_close = MappingStore.close

        def spy_close(self: MappingStore) -> None:
            closed["n"] += 1
            real_close(self)

        monkeypatch.setattr(MappingStore, "close", spy_close)

        def boom(_amap: AnimeIdsByName) -> list[AnimeIdRow]:
            raise ValueError("parse blew up")

        monkeypatch.setattr(m, "_anime_ids_rows", boom)

        with pytest.raises(ValueError, match="parse blew up"):
            MappingResolver(
                refresh_days=1,
                ignore_anilist_ids=set(),
                web=_WEB,
                sources=MappingSources(
                    anime={"x": {"anilist_id": 1}},
                    anidb=False,
                    anibridge=False,
                ),
            )
        assert closed["n"] >= 1


class TestUnwritableDbFallback:
    """A mappings database that can't be written is rebuilt in memory for the run, with one warning."""

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod can't make a file read-only to its owner on Windows")
    def test_file_db_write_failure_warns_and_rebuilds_in_memory(self, tmp_path: Path) -> None:
        # Inline sources are written on every boot, so the second boot writes to the read-only file.
        inline = MappingSources(anime=_FRIEREN, anidb=False, anibridge=False)
        _boot(tmp_path, inline).close()
        db = tmp_path / "mappings.db"
        db.chmod(0o444)
        recording = install_recording_hub()

        resolver = _boot(tmp_path, inline)

        assert diagnostic_messages(recording, Severity.WARNING) == [
            (
                f"Mappings database at {db} could not be written - rebuilding it "
                "in memory for this run (slower boot, no data lost)"
            ),
        ]
        assert resolver.resolve(ExternalIds(tvdb=424536)).kept == {154587: TitleMapping(154587, tvdb_season=1)}


# --------------------------------------------------------------------------- #
# Real-data parity: SQL backings must equal the in-memory oracle over the
# actual source files (gitignored, so skipped in CI / when absent). This is the
# migration's core promise - it diffs the SQL path against the unchanged code.
# The sources are cached in the data directory (next to mappings.db).
# --------------------------------------------------------------------------- #


class _RealSources(NamedTuple):
    """The three real (gitignored) mapping source-file paths under the data dir."""

    anime_ids: str
    anidb: str
    anibridge: str


def _real_source_paths() -> _RealSources:
    """Resolve the real source paths lazily (at test time, not import).

    `TestRealDataParity` carries `@pytest.mark.real_data_dir` so the autouse tmp
    data-dir override is off for it and `resolve_paths()` sees the developer's
    real `PEARLARR_DATA_DIR`. Evaluating this at import would instead capture
    whatever dir was active before the fixtures ran.
    """

    base = resolve_paths().data_dir
    return _RealSources(
        anime_ids=os.path.join(base, m.ANIME_IDS_FILE),
        anidb=os.path.join(base, m.ANIDB_MAPPINGS_FILE),
        anibridge=os.path.join(base, m.ANIBRIDGE_MAPPINGS_FILE),
    )


@pytest.fixture
def real_sources() -> _RealSources:
    """The real mapping sources, or skip the test when any is absent (CI / clean)."""

    paths = _real_source_paths()
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("real mapping source files not present")
    return paths


def _anidb_oracle(items: list[ElementTree.Element], anidb_id: int, tvdb_season: int) -> dict[int, dict[int, int]]:
    """Guarded reimplementation of the former anidb_anime_by_id + parse + raise, over one id's `anime` elements."""

    if len(items) > 1:
        raise ValueError(f"AniDB id {anidb_id} appears in more than one anime-lists mapping")
    if len(items) != 1:
        return {}
    result: dict[int, dict[int, int]] = {}
    for ms in items[0].findall("mapping-list"):
        for i in ms.findall("mapping"):
            if not i.text:
                continue
            try:
                season = int(i.attrib["tvdbseason"])
                if season != tvdb_season:
                    continue
                split = [x.split("-") for x in i.text.strip(";").split(";")]
                result[season] = {int(x[1]): int(x[0]) for x in split}
            except (KeyError, ValueError, IndexError):
                continue
    return result


@pytest.mark.realdata
@pytest.mark.real_data_dir
class TestRealDataParity:
    """SQL backings for anibridge, anime_ids, and anidb match their in-memory oracles over the real source files."""

    def test_anibridge_sql_matches_graph_over_all_ids(self, real_sources: _RealSources) -> None:
        with open(real_sources.anibridge, encoding="utf-8") as f:
            graph: AniBridgeGraph = json.load(f)
        graph_ab = AniBridge(graph)
        store = MappingStore.open(":memory:")
        store.replace_anibridge("d", graph_ab.to_rows())
        sql_ab = AniBridge.from_store(store)
        try:
            assert sql_ab.all_tvdb_ids == graph_ab.all_tvdb_ids
            assert sql_ab.all_tmdb_movie_ids == graph_ab.all_tmdb_movie_ids
            assert sql_ab.all_imdb_ids == graph_ab.all_imdb_ids
            assert len(sql_ab) == len(graph_ab)
            for tvdb in graph_ab.all_tvdb_ids:
                assert sql_ab.lookup_by_tvdb(tvdb) == graph_ab.lookup_by_tvdb(tvdb)
            for tmdb in graph_ab.all_tmdb_movie_ids:
                assert sql_ab.lookup_by_tmdb_movie(tmdb) == graph_ab.lookup_by_tmdb_movie(tmdb)
            for imdb in graph_ab.all_imdb_ids:
                assert sql_ab.lookup_by_imdb(imdb) == graph_ab.lookup_by_imdb(imdb)
        finally:
            store.close()

    def test_anime_ids_sql_matches_oracle_sample(self, real_sources: _RealSources) -> None:
        amap = m._parse_anime_mappings(real_sources.anime_ids)
        resolver = MappingResolver(
            refresh_days=99999,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=amap,
                anidb=False,
                anibridge=False,
            ),
        )
        oracle = _anime_oracle(amap)
        try:
            tvdbs = [
                r["tvdb_id"] for r in amap.values() if r.get("tvdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for tvdb in tvdbs:
                assert resolver.mappings_from_anime_ids(ExternalIds(tvdb=tvdb)) == oracle(ExternalIds(tvdb=tvdb))
            imdbs = [
                r["imdb_id"] for r in amap.values() if r.get("imdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for imdb in imdbs:
                assert resolver.mappings_from_anime_ids(ExternalIds(imdb=imdb)) == oracle(ExternalIds(imdb=imdb))
        finally:
            resolver.close()

    def test_anidb_sql_matches_oracle_sample(self, real_sources: _RealSources) -> None:
        root = m._parse_anidb_mappings(real_sources.anidb)
        resolver = MappingResolver(
            refresh_days=99999,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=False,
                anidb=root,
                anibridge=False,
            ),
        )
        try:
            by_id: dict[str, list[ElementTree.Element]] = {}
            for a in root.findall("anime"):
                by_id.setdefault(a.get("anidbid") or "", []).append(a)
            for anidb_id in [int(aid) for aid in by_id if aid.isdigit()][:300]:
                for season in (0, 1, 2):
                    try:
                        expected = _anidb_oracle(by_id.get(str(anidb_id), []), anidb_id, season)
                    except ValueError:
                        with pytest.raises(ValueError, match="appears in more than one anime-lists mapping"):
                            resolver.anidb_episode_by_tvdb(anidb_id, season)
                        continue
                    assert resolver.anidb_episode_by_tvdb(anidb_id, season) == expected
        finally:
            resolver.close()


def _boom(*_a: object, **_k: object) -> None:
    raise OSError("connection reset by peer")


class TestMaybeDownloadFailOpen:
    """A refresh blip falls open to the cached file. A first-ever download stays fatal."""

    def test_refresh_failure_falls_open_to_cached_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A stale-but-valid cached source whose refresh download fails on a transient
        # error must NOT abort the run: fall open to the on-disk copy and warn.
        source = tmp_path / "anime_ids.json"
        source.write_text("{}")
        old = time.time() - 10 * 86400  # 10 days old, past refresh_days
        os.utime(source, (old, old))
        monkeypatch.setattr(MappingResolver, "_download_file", _boom)

        recording = install_recording_hub()
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, refresh_days=1, _web=_WEB)
        resolver._maybe_download(str(source), "https://example/anime_ids.json", "anime_ids.json")

        # Warned (not aborted): the fall-open notice goes through the hub.
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "Could not refresh anime_ids.json" in warning
        assert source.exists()  # the cached copy is left intact

    def test_first_ever_download_failure_still_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With no file on disk there is nothing to fall open to, so a first-ever
        # download failure stays fatal (the run cannot proceed without the source).
        monkeypatch.setattr(MappingResolver, "_download_file", _boom)
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, refresh_days=1, _web=_WEB)

        with pytest.raises(OSError):
            resolver._maybe_download(str(tmp_path / "missing.json"), "https://example/x.json", "x")


class TestDownloadFileSizeCap:
    """An over-cap response aborts as OSError and leaves no partial file behind."""

    @respx.mock
    def test_over_cap_download_aborts_and_cleans_up(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dest = tmp_path / "out.json"
        monkeypatch.setattr(m, "MAX_DOWNLOAD_BYTES", 1024)
        respx.get("https://example/src.json").respond(content=b"x" * 4096)
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, _web=_WEB)

        with pytest.raises(OSError, match="download cap"):
            resolver._download_file("https://example/src.json", str(dest), label="src.json")

        # Neither the destination nor the .part temp survives the abort.
        assert not dest.exists()
        assert not (tmp_path / "out.json.part").exists()


class TestDownloadFile:
    """The streamed download contract: atomic success, contained failures."""

    @respx.mock
    def test_success_streams_to_dest_and_removes_the_part_temp(self, tmp_path: Path) -> None:
        """A 200 lands the exact body at dest with the .part temp gone (atomic rename)."""

        body = bytes(range(256)) * 800  # 200 KB: several 64 KiB read chunks
        respx.get("https://example/src.json").respond(content=body)
        dest = tmp_path / "out.json"
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, _web=_WEB)

        resolver._download_file("https://example/src.json", str(dest), label="src.json")

        assert dest.read_bytes() == body
        assert not (tmp_path / "out.json.part").exists()

    @respx.mock
    def test_http_failure_raises_a_url_free_oserror_and_writes_nothing(self, tmp_path: Path) -> None:
        """A 500 raises OSError carrying only the status - never the URL - and writes no file.

        `raise_for_status` fires before the `.part` open, so nothing lands on
        disk. The exact-message assert pins the containment contract (the httpx
        message embeds the URL, pearlarr's must not).
        """

        respx.get("https://example/src.json").respond(status_code=500)
        dest = tmp_path / "out.json"
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, _web=_WEB)

        with pytest.raises(OSError, match="download failed: HTTP 500") as excinfo:
            resolver._download_file("https://example/src.json", str(dest), label="src.json")

        assert str(excinfo.value) == "download failed: HTTP 500"  # exact: no URL leaks
        assert not dest.exists()
        assert not (tmp_path / "out.json.part").exists()

    @respx.mock
    def test_mid_stream_failure_preserves_the_preexisting_dest(self, tmp_path: Path) -> None:
        """A transport error mid-body leaves the cached copy intact and cleans the .part.

        Pins the atomicity contract `_maybe_download`'s fall-open relies on: a
        failed refresh writes only the temp (one chunk lands before the wire dies,
        verified live: respx streams the generator lazily), never the dest it
        falls back to. The `finally` sweeps the temp.
        """

        def _chunks() -> Iterator[bytes]:
            yield b"y" * (1 << 16)  # one chunk reaches .part before the wire dies
            raise httpx.ReadError("boom")

        respx.get("https://example/src.json").mock(return_value=httpx.Response(200, content=_chunks()))
        dest = tmp_path / "out.json"
        dest.write_bytes(b'{"cached": true}')
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, _web=_WEB)

        with pytest.raises(OSError, match="download failed: ReadError"):
            resolver._download_file("https://example/src.json", str(dest), label="src.json")

        assert dest.read_bytes() == b'{"cached": true}'  # the cached copy is untouched
        assert not (tmp_path / "out.json.part").exists()
