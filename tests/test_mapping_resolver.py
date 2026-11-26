# pyright: strict
"""Tests for `MappingResolver`: which AniList titles an arr's external ids resolve to, and with what.

Each source is read as a run reads it, from the mappings database: AniBridge's TVDB ranges and specials aliases,
the Kometa Anime-IDs seasons and offsets, the AniDB episode maps, and how the sources merge. Then the source files
themselves: the downloads at boot, the hash check that skips an unchanged file, a database that can't be written,
and the real files a run downloaded.
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import httpx
import pytest
import respx

import pearlarr.mappings as m
from pearlarr.anibridge import AniBridge, AniBridgeGraph, SpecialAliasing
from pearlarr.mappings import (
    ANIBRIDGE_MAPPINGS_FILE,
    ANIDB_MAPPINGS_FILE,
    ANIME_IDS_FILE,
    ANIME_IDS_URL,
    AnimeIdsByName,
    AnimeIdsRawMapping,
    ExternalIds,
    MappingResolver,
    MappingSource,
    MappingSources,
    TitleMapping,
)
from pearlarr.output import Severity
from pearlarr.paths import resolve_paths
from pearlarr.sonarr_episodes import check_ep_by_anibridge

from .builders import sonarr_ep
from .fakes import RecordingProgress, debug_logger, diagnostic_messages, install_recording_hub

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
        "imdb_movie:tt0269": {},
        # Providers a lookup doesn't read. Real graphs carry them.
        "mal:111": {},
        "tmdb_show:5000:s1": {"1-12": "1-12"},
    },
    "anilist:270": {
        "tvdb_show:74796:s1": {},
        "tmdb_movie:888": {},
    },
    "anilist:300": {
        "tvdb_show:111:s1": {"1-5": "1-5"},
        "tvdb_show:222:s1": {"6-10": "3-7"},
    },
    "anilist:400": {
        "imdb_movie:tt0400": {},
    },
    # Several movies of each kind, as AniBridge's record for AniList 31 has.
    "anilist:31": {
        "tmdb_movie:21832": {},
        "tmdb_movie:54270": {},
        "imdb_movie:tt0169880": {},
        "imdb_movie:tt13645944": {},
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
                ExternalIds(tvdb=111),
                {300: TitleMapping(300, tvdb_mappings={1: [(1, 5)]}, source=_ANIBRIDGE)},
                id="one of a title's two shows",
            ),
        ],
    )
    def test_a_series_lookup_gives_each_title_the_ranges_of_that_show(
        self, ids: ExternalIds, expected: dict[int, TitleMapping]
    ) -> None:
        assert _resolver(anibridge=GRAPH).resolve(ids).kept == expected

    @pytest.mark.parametrize(
        ("ids", "found"),
        [
            pytest.param(ExternalIds(tmdb_movie=888, imdb="tt0888888"), [270], id="movie by its tmdb id"),
            pytest.param(ExternalIds(tmdb_movie=404, imdb="tt0400"), [400], id="movie by its imdb id"),
            pytest.param(ExternalIds(tmdb_movie=54270), [31], id="movie by the second of its tmdb ids"),
            pytest.param(ExternalIds(imdb="tt13645944"), [31], id="movie by the second of its imdb ids"),
            pytest.param(ExternalIds(tvdb=424242, imdb="tt0424242"), [], id="unknown"),
        ],
    )
    def test_a_lookup_finds_every_title_mapped_to_its_ids(self, ids: ExternalIds, found: list[int]) -> None:
        assert list(_resolver(anibridge=GRAPH).resolve(ids).kept) == found

    def test_the_library_filter_keeps_every_id_a_title_is_found_by(self) -> None:
        anibridge = _resolver(anibridge=GRAPH).anibridge
        assert anibridge is not None

        assert anibridge.known_external_ids("tvdb_id") == {74796, 111, 222}
        assert anibridge.known_external_ids("tmdb_movie_id") == {888, 21832, 54270}
        assert anibridge.known_external_ids("imdb_id") == {"tt0269", "tt0400", "tt0169880", "tt13645944"}


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


def _aliasing_on(graph: AniBridgeGraph, tvdb_id: int) -> dict[int, SpecialAliasing | None]:
    """The specials aliasing of each title a lookup of TVDB show `tvdb_id` finds, None for a title without any."""

    kept = _resolver(anibridge=graph).resolve(ExternalIds(tvdb=tvdb_id)).kept
    return {anilist_id: mapping.special_aliasing for anilist_id, mapping in kept.items()}


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

        assert _aliasing_on(graph, 500) == {1: None}

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

        assert _aliasing_on(graph, 500) == {1: None}

    def test_targets_of_different_lengths_pair_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "5-6"}, "tmdb_show:900:s0": {"1-2": "5-7"}},
        }

        assert _aliasing_on(graph, 500) == {1: None}

    def test_a_source_of_another_length_pairs_nothing(self) -> None:
        # The targets agree with each other but not with the source range they map.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-6": "11-17"}, "tmdb_show:900:s0": {"1-6": "12-18"}},
        }

        assert _aliasing_on(graph, 500) == {1: None}

    def test_a_source_range_only_tvdb_maps_pairs_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"2": "9"}, "tmdb_show:900:s0": {"1": "6"}},
        }

        assert _aliasing_on(graph, 500) == {1: None}

    def test_regular_season_targets_pair_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s1": {"1-2": "1-2"}, "tmdb_show:900:s1": {"1-2": "1-2"}},
        }

        assert _aliasing_on(graph, 500) == {1: None}

    def test_two_tvdb_specials_targets_pair_nothing(self) -> None:
        graph: AniBridgeGraph = {
            "anilist:2": {
                "tvdb_show:500:s0": {"1": "3"},
                "tvdb_show:501:s0": {"1": "4"},
                "tmdb_show:900:s0": {"1": "5"},
            },
        }

        assert _aliasing_on(graph, 500) == {2: None}

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

        assert _aliasing_on(graph, 500) == {1: SpecialAliasing(900, {4: 4}), 2: None}

    def test_another_title_mapping_the_whole_specials_season_drops_every_alias(self) -> None:
        # An empty dict covers the whole season, so title 2 maps every special and title 1 keeps no alias.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": "4-5"}, "tmdb_show:900:s0": {"1-2": "4-5"}},
            "anilist:2": {"tvdb_show:500:s0": {}},
        }

        assert _aliasing_on(graph, 500) == {1: None, 2: None}

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

        assert _aliasing_on(graph, 500) == {1: None}

    @pytest.mark.parametrize("target", ["x,7,8", "7,,8"])
    def test_a_junk_piece_pairs_nothing(self, target: str) -> None:
        # If the junk piece were skipped, `7,8` would line up with the two-number source and pair wrongly.
        graph: AniBridgeGraph = {
            "anilist:1": {"tvdb_show:500:s0": {"1-2": target}, "tmdb_show:900:s0": {"1-2": "9-10"}},
        }

        assert _aliasing_on(graph, 500) == {1: None}


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


class TestAnimeIdsLookup:
    """A lookup finds every Kometa record its ids reach, the first record of each AniList title.

    Covers explicit nulls and the library filter's sets.
    """

    @pytest.mark.parametrize(
        ("ids", "found"),
        [
            pytest.param(ExternalIds(tvdb=200, imdb="tt100"), [100, 101, 102], id="series with an imdb id"),
            pytest.param(ExternalIds(tmdb_movie=7000, imdb="tt7000"), [103], id="movie by its tmdb id"),
            pytest.param(ExternalIds(tmdb_movie=4242, imdb="tt100"), [100, 102], id="movie by its imdb id"),
            pytest.param(ExternalIds(tvdb=999), [], id="a record without an anilist id"),
            pytest.param(ExternalIds(tvdb=424242), [], id="unknown"),
        ],
    )
    def test_a_lookup_finds_every_record_its_ids_reach(self, ids: ExternalIds, found: list[int]) -> None:
        assert list(_resolver(anime=AMAP).resolve(ids).kept) == found

    def test_a_series_lookup_reads_each_titles_season_offset_and_ids(self) -> None:
        assert _resolver(anime=AMAP).resolve(ExternalIds(tvdb=200)).kept == {
            100: TitleMapping(100, tvdb_season=2, tvdb_epoffset=3, imdb_id="tt100", anidb_id=50),
            101: TitleMapping(101),
        }

    def test_an_explicit_null_season_reads_as_no_season(self) -> None:
        # A record whose season and offset are null reads like one without the keys.
        amap = {"N": {"anilist_id": 500, "tvdb_id": 5000, "tvdb_season": None, "tvdb_epoffset": None}}

        assert _resolver(anime=amap).resolve(ExternalIds(tvdb=5000)).kept == {
            500: TitleMapping(500, tvdb_season=-1, tvdb_epoffset=0)
        }

    def test_candidate_sets_include_objects_without_anilist(self) -> None:
        # The former full-dict walk added external ids even from anilist-less objects
        # (object E). The SQL distinct must keep that, even though lookups skip them.
        resolver = _resolver(anime=AMAP)

        assert resolver.known_external_ids("tvdb_id") == {200, 999}
        assert resolver.known_external_ids("imdb_id") == {"tt100"}
        assert resolver.known_external_ids("tmdb_movie_id") == {7000}
        assert resolver.mappings_from_anime_ids(ExternalIds(tvdb=999)) == {}  # E has no anilist


class TestResolveMerge:
    """`resolve` merges the two sources by title.

    AniBridge's title with TVDB ranges wins over Kometa's record for it, Kometa fills the titles AniBridge doesn't
    map, the ignored titles are set aside, and the rest sort by AniList id.
    """

    def test_a_lookup_merges_the_sources_by_title(self) -> None:
        # Series 74796: both sources map 269, Kometa alone maps 100, and AniBridge alone maps 270, which is ignored.
        anime = {"x": {"anilist_id": 269, "tvdb_id": 74796}, "y": {"anilist_id": 100, "tvdb_id": 74796}}
        resolver = _resolver(anime=anime, anibridge=GRAPH, ignore={270})

        resolved = resolver.resolve(ExternalIds(tvdb=74796))

        assert [(anilist_id, mapping.source) for anilist_id, mapping in resolved.kept.items()] == [
            (100, MappingSource.ANIME_IDS),
            (269, _ANIBRIDGE),
        ]
        assert list(resolved.ignored) == [270]


class TestAniBridgeTitleWithoutTvdbRanges:
    """A title AniBridge reaches by IMDb alone has no TVDB season ranges.

    A series takes Kometa's season for it when Kometa has one, and covers none of its episodes when Kometa has none.
    """

    @pytest.mark.parametrize(
        "season", [pytest.param(2, id="a later season"), pytest.param(0, id="a movie the series lists as a special")]
    )
    def test_a_series_takes_kometas_season_for_the_title(self, season: int) -> None:
        resolver = _resolver(
            anime={"k": {"anilist_id": 400, "tvdb_id": 555, "tvdb_season": season}},
            anibridge={"anilist:400": {"imdb_movie:tt400": {}}},
        )

        mappings = resolver.resolve(ExternalIds(tvdb=555, imdb="tt400")).kept

        assert mappings == {400: TitleMapping(400, tvdb_season=season)}

    def test_with_no_kometa_record_the_title_keeps_anibridges_mapping_without_ranges(self) -> None:
        # test_mapping_episodes shows such a mapping resolves to no episodes.
        resolver = _resolver(anibridge={"anilist:500": {"imdb_movie:tt500": {}}})

        mappings = resolver.resolve(ExternalIds(tvdb=777, imdb="tt500")).kept

        assert mappings == {500: TitleMapping(500, imdb_id="tt500", source=_ANIBRIDGE)}


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
_ANIME_IDS_FROM_FILE = MappingSources(anime=None, anidb=False, anibridge=False)
_FRIEREN: AnimeIdsByName = {"Sousou no Frieren": {"anilist_id": 154587, "tvdb_id": 424536, "tvdb_season": 1}}


class TestSourcesSummary:
    """The boot step's note names the sources a run reads."""

    @pytest.mark.parametrize(
        ("sources", "note"),
        [
            pytest.param(
                MappingSources(anime=_FRIEREN, anidb=ElementTree.fromstring("<anime-list/>"), anibridge=GRAPH),
                "anime-ids · anime-lists · anibridge",
                id="all three",
            ),
            pytest.param(MappingSources(anime=False, anidb=False, anibridge=GRAPH), "anibridge", id="one"),
            pytest.param(MappingSources(anime=False, anidb=False, anibridge=False), "none enabled", id="none"),
        ],
    )
    def test_the_note_names_each_source_read(self, sources: MappingSources, note: str) -> None:
        resolver = MappingResolver(refresh_days=1, ignore_anilist_ids=set(), sources=sources, web=httpx.Client())

        assert resolver.sources_summary() == note


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


def _aged(path: Path, *, days: int) -> None:
    """Set the file's modified time `days` days and a minute back."""

    then = time.time() - days * 86400 - 60
    os.utime(path, (then, then))


def _cut_off() -> httpx.Response:
    """A 200 whose body breaks off after its first 64 KiB, as a dropped connection leaves it."""

    def chunks() -> Iterator[bytes]:
        yield b" " * (1 << 16)
        raise httpx.ReadError("connection reset by peer")

    return httpx.Response(200, content=chunks())


class TestSourceDownloads:
    """A boot downloads a source file that's missing, and again once it is `refresh_days` old.

    A download lands whole or not at all: it streams into a `.part` file beside the source and replaces the source
    only once complete. Its errors name no URL.
    """

    @respx.mock
    def test_a_missing_source_is_downloaded_whole_and_indexed(self, tmp_path: Path) -> None:
        # Many 64 KiB read chunks, so a body cut short would show.
        records = {str(n): {"anilist_id": n, "tvdb_id": 70000 + n, "tvdb_season": 1} for n in range(1, 3001)}
        body = json.dumps(records, indent=2).encode()
        respx.get(ANIME_IDS_URL).respond(content=body)

        resolver = _boot(tmp_path, _ANIME_IDS_FROM_FILE)

        assert (tmp_path / ANIME_IDS_FILE).read_bytes() == body
        assert not (tmp_path / f"{ANIME_IDS_FILE}.part").exists()
        assert resolver.resolve(ExternalIds(tvdb=73000)).kept == {3000: TitleMapping(3000, tvdb_season=1)}

    @respx.mock
    def test_a_download_shows_each_megabyte_on_the_boot_bar(self, tmp_path: Path) -> None:
        body = json.dumps(_FRIEREN).encode().ljust(2 << 20)  # two megabytes, the JSON padded with spaces
        respx.get(ANIME_IDS_URL).respond(content=body)
        bar = RecordingProgress()

        MappingResolver(
            refresh_days=1,
            ignore_anilist_ids=set(),
            sources=_ANIME_IDS_FROM_FILE,
            web=httpx.Client(),
            mappings_db=str(tmp_path / "mappings.db"),
            progress=bar,
        )

        assert bar.updates == [(0.5, "anime_ids.json · 1/2 MB"), (1.0, "anime_ids.json · 2/2 MB")]

    @respx.mock
    def test_a_source_as_old_as_refresh_days_is_downloaded_again(self, tmp_path: Path) -> None:
        source = tmp_path / ANIME_IDS_FILE
        source.write_text(json.dumps(_FRIEREN))
        _aged(source, days=1)
        fresh = json.dumps(
            {**_FRIEREN, "Sousou no Frieren 2": {"anilist_id": 182255, "tvdb_id": 424536, "tvdb_season": 2}}
        )
        respx.get(ANIME_IDS_URL).respond(content=fresh.encode())

        resolver = _boot(tmp_path, _ANIME_IDS_FROM_FILE)

        assert source.read_text() == fresh
        assert list(resolver.resolve(ExternalIds(tvdb=424536)).kept) == [154587, 182255]

    @respx.mock
    @pytest.mark.parametrize(
        ("respond", "detail"),
        [
            pytest.param(lambda: httpx.Response(500), "HTTP 500", id="server error"),
            pytest.param(_cut_off, "ReadError", id="cut off mid-stream"),
        ],
    )
    def test_refresh_failure_falls_open_to_cached_file(
        self, tmp_path: Path, respond: Callable[[], httpx.Response], detail: str
    ) -> None:
        source = tmp_path / ANIME_IDS_FILE
        cached = json.dumps(_FRIEREN)
        source.write_text(cached)
        _aged(source, days=10)
        respx.get(ANIME_IDS_URL).mock(return_value=respond())
        recording = install_recording_hub()

        resolver = _boot(tmp_path, _ANIME_IDS_FROM_FILE)

        assert diagnostic_messages(recording, Severity.WARNING) == [
            f"Could not refresh anime_ids.json (download failed: {detail}) - using the cached copy"
        ]
        assert source.read_text() == cached
        assert not (tmp_path / f"{ANIME_IDS_FILE}.part").exists()
        assert resolver.resolve(ExternalIds(tvdb=424536)).kept == {154587: TitleMapping(154587, tvdb_season=1)}

    @respx.mock
    @pytest.mark.parametrize(
        ("respond", "error"),
        [
            pytest.param(lambda: httpx.Response(500), "download failed: HTTP 500", id="server error"),
            pytest.param(
                lambda: httpx.Response(200, content=b" " * (2 << 20)),
                "anime_ids.json exceeded the 1 MB download cap; aborting",
                id="over the size cap",
            ),
        ],
    )
    def test_first_ever_download_failure_still_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, respond: Callable[[], httpx.Response], error: str
    ) -> None:
        monkeypatch.setattr(m, "MAX_DOWNLOAD_BYTES", 1 << 20)  # a cap a small body can pass
        respx.get(ANIME_IDS_URL).mock(return_value=respond())

        with pytest.raises(OSError, match=f"^{re.escape(error)}$"):
            _boot(tmp_path, _ANIME_IDS_FROM_FILE)

        # No source and no part file, and the database closed: one left open keeps its -wal file beside it.
        assert [p.name for p in tmp_path.iterdir()] == ["mappings.db"]


# --------------------------------------------------------------------------- #
# The real source files a run downloaded. They're in the data directory once a run has fetched them, so CI and a
# fresh checkout skip these.
# --------------------------------------------------------------------------- #


class _RealSources(NamedTuple):
    """The three real (gitignored) mapping source-file paths under the data dir."""

    anime_ids: str
    anidb: str
    anibridge: str


def _real_source_paths() -> _RealSources:
    """Resolve the real source paths lazily (at test time, not import).

    `TestRealSourceFiles` carries `@pytest.mark.real_data_dir` so the autouse tmp
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


def _real_boot(sources: MappingSources) -> MappingResolver:
    """A resolver over the real source files, indexed in memory so nothing in the data dir is written.

    The refresh age is far off, so nothing downloads.
    """

    return MappingResolver(refresh_days=99999, ignore_anilist_ids=set(), sources=sources, web=httpx.Client())


def _each_id(record: AnimeIdsRawMapping) -> list[ExternalIds]:
    """One lookup for each id a Kometa record carries."""

    lookups = [
        ExternalIds(tvdb=record.get("tvdb_id")),
        ExternalIds(tmdb_movie=record.get("tmdb_movie_id")),
        ExternalIds(imdb=record.get("imdb_id")),
    ]
    return [ids for ids in lookups if ids != ExternalIds()]


def _is_ambiguous(resolver: MappingResolver, anidb_id: int) -> bool:
    """True when asking for the AniDB id's episodes raises for its being listed more than once."""

    try:
        resolver.anidb_episode_by_tvdb(anidb_id, 1)
    except ValueError:
        return True
    return False


@pytest.mark.realdata
@pytest.mark.real_data_dir
class TestRealSourceFiles:
    """The real files read as the cases above say, over shapes nobody wrote down."""

    def test_anibridge_sql_matches_graph_over_all_ids(self, real_sources: _RealSources) -> None:
        # The graph's own lookups read each title straight from the parsed file. A run asks the database instead.
        graph = AniBridge(json.loads(Path(real_sources.anibridge).read_text(encoding="utf-8")))
        served = _real_boot(MappingSources(anime=False, anidb=False, anibridge=None)).anibridge
        assert served is not None

        assert (served.all_tvdb_ids, served.all_tmdb_movie_ids, served.all_imdb_ids, len(served)) == (
            graph.all_tvdb_ids,
            graph.all_tmdb_movie_ids,
            graph.all_imdb_ids,
            len(graph),
        )
        assert [i for i in graph.all_tvdb_ids if served.lookup_by_tvdb(i) != graph.lookup_by_tvdb(i)] == []
        assert [
            i for i in graph.all_tmdb_movie_ids if served.lookup_by_tmdb_movie(i) != graph.lookup_by_tmdb_movie(i)
        ] == []
        assert [i for i in graph.all_imdb_ids if served.lookup_by_imdb(i) != graph.lookup_by_imdb(i)] == []

    def test_every_real_anime_ids_record_is_found_by_its_ids(self, real_sources: _RealSources) -> None:
        records: AnimeIdsByName = json.loads(Path(real_sources.anime_ids).read_text(encoding="utf-8"))
        resolver = _real_boot(MappingSources(anime=None, anidb=False, anibridge=False))

        missed = [
            (name, ids)
            for name, record in records.items()
            if (anilist_id := record.get("anilist_id")) is not None
            for ids in _each_id(record)
            if anilist_id not in resolver.resolve(ids).kept
        ]

        assert missed == []

    def test_an_anidb_id_is_ambiguous_exactly_when_the_real_file_lists_it_more_than_once(
        self, real_sources: _RealSources
    ) -> None:
        listed = Counter(
            anime.get("anidbid") for anime in ElementTree.parse(real_sources.anidb).getroot().iter("anime")
        )
        resolver = _real_boot(MappingSources(anime=False, anidb=None, anibridge=False))

        wrong = [
            anidb_id
            for anidb_id, times in listed.items()
            if anidb_id is not None and anidb_id.isdigit() and _is_ambiguous(resolver, int(anidb_id)) != (times > 1)
        ]

        assert wrong == []
