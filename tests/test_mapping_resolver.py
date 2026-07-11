# pyright: strict
# pyright: reportPrivateUsage=false
# Drives module-private resolver internals under test (_entry_from_raw, m._parse_*, resolver._(maybe_)download*).
"""Parity + edge tests for the SQL-backed ``MappingResolver`` / ``AniBridge``.

The SQL backings must reproduce the in-memory implementations exactly. AniBridge's
graph-backed view is used directly as the oracle (the unchanged parser); anime_ids
is checked against an inline reconstruction of the former reverse-index merge; and
the anidb + digest-gate behaviors are pinned. These are the safety net for the
"parse once, serve from SQL" migration - the resolver lookups had no direct
coverage before.
"""

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from xml.etree import ElementTree

import httpx
import pytest
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

import pearlarr.modules.mappings as m
from pearlarr.modules.anibridge import AniBridge, AniBridgeGraph, _parse_ranges
from pearlarr.modules.mapping_store import AnidbMappingRow, AnimeIdRow, MappingStore
from pearlarr.modules.mappings import (
    AnimeIdsMap,
    AnimeIdsRecord,
    ExternalIds,
    MappingEntry,
    MappingMode,
    MappingResolver,
    MappingSource,
    MappingSources,
    _entry_from_raw,
)
from pearlarr.modules.output import Severity
from pearlarr.modules.paths import resolve_paths
from pearlarr.modules.seadex_types import TvdbMappings
from pearlarr.modules.sonarr_episodes import SonarrEpisodes, check_ep_by_anibridge

from .builders import make_bare_instance, sonarr_ep
from .fakes import CaptureHandler, diagnostic_messages, install_recording_hub

# One inert shared client for the resolver ctors (inline sources: nothing downloads).
_WEB = httpx.Client()

# --------------------------------------------------------------------------- #
# AniBridge: SQL backing must equal the graph backing (the oracle)
# --------------------------------------------------------------------------- #

GRAPH: AniBridgeGraph = {
    "anilist:269": {
        "tvdb_show:74796:s2": {"21-41": "1-21"},
        "tvdb_show:74796:s3": {"42-50": "1-9,11-12"},  # multi-segment range
        "anidb:1234": {},
        "imdb_show:tt0269": {},
        "mal:111": {},  # ignored provider: present in real data, must not break parsing
        "tmdb_show:5000:s1": {"1-12": "1-12"},  # ignored provider
    },
    "anilist:270": {
        "tvdb_show:74796:s1": {},  # same tvdb id, present-but-empty season
        "tmdb_movie:888": {},
        "tvdb_movie:999": {},
    },
    "anilist:271": {
        "tvdb_show:600:s1": {"1-": "1-"},  # open-ended range (end None)
    },
    "anilist:300": {
        "tvdb_show:111:s1": {"1-5": "1-5"},  # two tvdb shows -> first-pick matters
        "tvdb_show:222:s1": {"1-5": "1-5"},
        "imdb_show:tt300": {},
    },
}


def _ab_pair(graph: AniBridgeGraph) -> tuple[AniBridge, AniBridge, MappingStore]:
    """(graph-backed oracle, SQL-backed view, store) over the same graph."""

    graph_ab = AniBridge(graph)
    store = MappingStore.open(":memory:")
    store.replace_anibridge("d", graph_ab.to_rows())
    return graph_ab, AniBridge.from_store(store), store


class TestAniBridgeParity:
    def test_lookups_and_sets_match_graph(self) -> None:
        graph_ab, sql_ab, store = _ab_pair(GRAPH)
        try:
            for tvdb in (74796, 600, 111, 222, 424242):
                assert sql_ab.lookup_by_tvdb(tvdb) == graph_ab.lookup_by_tvdb(tvdb)
            for tmdb in (888, 4242):
                assert sql_ab.lookup_by_tmdb(tmdb) == graph_ab.lookup_by_tmdb(tmdb)
            for imdb in ("tt0269", "tt300", "ttZZZ"):
                assert sql_ab.lookup_by_imdb(imdb) == graph_ab.lookup_by_imdb(imdb)

            assert sql_ab.all_tvdb_ids == graph_ab.all_tvdb_ids
            assert sql_ab.all_tmdb_movie_ids == graph_ab.all_tmdb_movie_ids
            assert sql_ab.all_imdb_ids == graph_ab.all_imdb_ids
            assert len(sql_ab) == len(graph_ab)
            assert bool(sql_ab) == bool(graph_ab)
        finally:
            store.close()

    def test_open_ended_range_roundtrips_as_none(self) -> None:
        _, sql_ab, store = _ab_pair(GRAPH)
        try:
            assert sql_ab.lookup_by_tvdb(600)[271]["tvdb_mappings"] == {1: [(1, None)]}
        finally:
            store.close()

    def test_present_but_empty_season_roundtrips(self) -> None:
        # The {season: []} ("whole season covered") vs {} ("not covered") distinction
        # is opposite behavior downstream; it must survive the SQL round-trip.
        graph_ab, sql_ab, store = _ab_pair(GRAPH)
        try:
            assert graph_ab.lookup_by_tvdb(74796)[270]["tvdb_mappings"] == {1: []}
            assert sql_ab.lookup_by_tvdb(74796)[270]["tvdb_mappings"] == {1: []}
        finally:
            store.close()

    def test_empty_graph_is_falsey(self) -> None:
        _, sql_ab, store = _ab_pair({})
        try:
            assert not sql_ab
            assert len(sql_ab) == 0
        finally:
            store.close()


# --------------------------------------------------------------------------- #
# Property-based: the invariants the example tables above sample only a few
# points of. TestRealDataParity proves graph<->SQL parity over the real files
# (gitignored, skipped in CI); these Hypothesis properties make the SAME parity
# CI-enforced over generated graphs, and pin the range-containment boundaries
# a `<=`->`<` off-by-one would silently drop.
# --------------------------------------------------------------------------- #

_SMALL_ID = st.integers(min_value=1, max_value=5)  # small pool -> shared ids across AniList entries
_IMDB_ID = st.sampled_from(("tt1", "tt2", "tt3"))
_SEASON = st.integers(min_value=0, max_value=3)
_EP = st.integers(min_value=1, max_value=40)


@st.composite
def _tgt_range(draw: DrawFn) -> str:
    """A target episode-range string: single / closed / open-ended / multi-segment."""

    a = draw(_EP)
    b = draw(_EP)
    kind = draw(st.sampled_from(("single", "closed", "open", "multi")))
    if kind == "single":
        return str(a)
    if kind == "open":
        return f"{a}-"
    if kind == "closed":
        return f"{min(a, b)}-{max(a, b)}"
    c = draw(_EP)
    d = draw(_EP)
    return f"{min(a, b)}-{max(a, b)},{min(c, d)}-{max(c, d)}"


@st.composite
def _anibridge_graph(draw: DrawFn) -> AniBridgeGraph:
    """A structurally-valid anibridge graph over a small, collision-prone id pool.

    The tiny external-id range forces shared tvdb/tmdb/imdb ids across AniList
    entries (reverse-index sets + first-pick), and every tvdb season is drawn
    either empty (present-but-empty ``{season: []}``) or range-bearing - the exact
    round-trip dimensions ``to_rows``/``from_store`` must preserve.
    """

    graph: AniBridgeGraph = {}
    anilist_ids = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=6, unique=True))
    for anilist_id in anilist_ids:
        targets: dict[str, dict[str, str]] = {}
        for _ in range(draw(st.integers(min_value=0, max_value=3))):
            season = draw(_SEASON)
            ep_map: dict[str, str] = {} if draw(st.booleans()) else {"1-99": draw(_tgt_range())}
            targets[f"tvdb_show:{draw(_SMALL_ID)}:s{season}"] = ep_map
        if draw(st.booleans()):
            # Ignored provider (like mal below): real graphs carry it; parity must hold.
            targets[f"tmdb_show:{draw(_SMALL_ID)}:s1"] = {}
        if draw(st.booleans()):
            targets[f"tmdb_movie:{draw(_SMALL_ID)}"] = {}
        if draw(st.booleans()):
            targets[f"imdb_show:{draw(_IMDB_ID)}"] = {}
        if draw(st.booleans()):
            targets[f"imdb_movie:{draw(_IMDB_ID)}"] = {}
        if draw(st.booleans()):
            targets[f"anidb:{draw(_SMALL_ID)}"] = {}
        if draw(st.booleans()):
            targets[f"mal:{draw(_SMALL_ID)}"] = {}
        graph[f"anilist:{anilist_id}"] = targets
    return graph


# The autouse data-dir / store-closing fixtures (conftest) are function-scoped, so
# @given tests trip Hypothesis's function-scoped-fixture health check; suppress it
# (these :memory: cases never touch the data dir the fixture guards).
_ALLOW_FIXTURES = settings(suppress_health_check=[HealthCheck.function_scoped_fixture])


class TestAniBridgeParityProperty:
    """graph<->SQL parity over generated graphs - the CI-enforced twin of
    TestRealDataParity (which only runs when the gitignored real files exist)."""

    @_ALLOW_FIXTURES
    @given(graph=_anibridge_graph())
    def test_sql_backing_matches_graph_over_all_ids(self, graph: AniBridgeGraph) -> None:
        graph_ab = AniBridge(graph)
        store = MappingStore.open(":memory:")
        try:
            store.replace_anibridge("d", graph_ab.to_rows())
            sql_ab = AniBridge.from_store(store)

            assert sql_ab.all_tvdb_ids == graph_ab.all_tvdb_ids
            assert sql_ab.all_tmdb_movie_ids == graph_ab.all_tmdb_movie_ids
            assert sql_ab.all_imdb_ids == graph_ab.all_imdb_ids
            assert len(sql_ab) == len(graph_ab)
            assert bool(sql_ab) == bool(graph_ab)

            for tvdb in graph_ab.all_tvdb_ids:
                assert sql_ab.lookup_by_tvdb(tvdb) == graph_ab.lookup_by_tvdb(tvdb)
            for tmdb_movie in graph_ab.all_tmdb_movie_ids:
                assert sql_ab.lookup_by_tmdb(tmdb_movie) == graph_ab.lookup_by_tmdb(tmdb_movie)
            for imdb in graph_ab.all_imdb_ids:
                assert sql_ab.lookup_by_imdb(imdb) == graph_ab.lookup_by_imdb(imdb)
        finally:
            store.close()


class TestAniBridgeRangeContainment:
    """``_parse_ranges`` -> ``check_ep_by_anibridge`` boundary classification.

    ``_parse_ranges`` has no direct test and ``check_ep_by_anibridge`` only trivial
    ones; a ``<=``->``<`` regression drops the last episode of every closed cour and
    passes the example suite. These pin the boundary set for both range shapes.
    """

    @_ALLOW_FIXTURES
    @given(start=_EP, length=st.integers(min_value=0, max_value=20), season=_SEASON)
    def test_closed_range_covers_start_through_end_only(self, start: int, length: int, season: int) -> None:
        end = start + length
        ranges = _parse_ranges(f"{start}-{end}")
        assert ranges == [(start, end)]  # _parse_ranges pins the closed-range parse
        mappings: TvdbMappings = {season: ranges}

        # start and end included; the immediate neighbours excluded (catches <=/<).
        assert check_ep_by_anibridge(ep=sonarr_ep(season, start), tvdb_mappings=mappings) is True
        assert check_ep_by_anibridge(ep=sonarr_ep(season, end), tvdb_mappings=mappings) is True
        assert check_ep_by_anibridge(ep=sonarr_ep(season, start - 1), tvdb_mappings=mappings) is False
        assert check_ep_by_anibridge(ep=sonarr_ep(season, end + 1), tvdb_mappings=mappings) is False

    @_ALLOW_FIXTURES
    @given(start=_EP, episode=st.integers(min_value=0, max_value=80), season=_SEASON)
    def test_open_ended_range_covers_from_start_upwards(self, start: int, episode: int, season: int) -> None:
        ranges = _parse_ranges(f"{start}-")
        assert ranges == [(start, None)]  # _parse_ranges pins the open-ended parse
        mappings: TvdbMappings = {season: ranges}

        covered = check_ep_by_anibridge(ep=sonarr_ep(season, episode), tvdb_mappings=mappings)
        assert covered is (episode >= start)


# --------------------------------------------------------------------------- #
# anime_ids: SQL lookups must equal the former reverse-index merge
# --------------------------------------------------------------------------- #

AMAP: AnimeIdsMap = {
    "A": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 2, "tvdb_epoffset": 3, "imdb_id": "tt100", "anidb_id": 50},
    "B": {"anilist_id": 101, "tvdb_id": 200},  # same tvdb id, 2nd anilist
    "C": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 9},  # dup anilist -> first wins
    "D": {"anilist_id": 102, "imdb_id": "tt100"},  # shares imdb
    "E": {"tvdb_id": 999},  # no anilist -> filter-set only
    "F": {"anilist_id": 103, "tmdb_movie_id": 7000},
}


def _anime_oracle(amap: AnimeIdsMap, ids: ExternalIds) -> dict[int, MappingEntry]:
    """Inline twin of the former reverse-index + no-clobber merge (the oracle)."""

    index: dict[str, dict[object, list[AnimeIdsRecord]]] = {
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

    result: dict[int, MappingEntry] = {}

    def merge(field: str, value: object) -> None:
        for rec in index[field].get(value, []):
            aid = rec["anilist_id"]
            if aid not in result:
                result[aid] = _entry_from_raw(aid, rec)

    if ids.tvdb is not None:
        merge("tvdb_id", ids.tvdb)
    if ids.tmdb is not None:
        merge("tmdb_movie_id", ids.tmdb)
    if ids.imdb is not None:
        merge("imdb_id", ids.imdb)
    return result


class TestAnimeIdsParity:
    @pytest.mark.parametrize(
        "ids",
        [
            ExternalIds(tvdb=200),
            ExternalIds(imdb="tt100"),
            ExternalIds(tmdb=7000),
            ExternalIds(tvdb=200, imdb="tt100"),
            ExternalIds(tvdb=424242),  # no match -> empty
        ],
    )
    def test_lookup_matches_oracle(self, ids: ExternalIds) -> None:
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=AMAP,
                anidb=False,
                anibridge=False,
            ),
        )
        try:
            assert resolver.get_mappings_from_anime_mappings(ids) == _anime_oracle(AMAP, ids)
        finally:
            resolver.close()

    def test_explicit_null_season_coalesces_to_sentinel(self) -> None:
        # A present-but-null tvdb_season / tvdb_epoffset (an explicit JSON null, not
        # an absent key) must not abort the populate against the NOT NULL columns; it
        # coalesces to the same -1 / 0 sentinel an absent key gets, so the run still
        # works exactly as the pre-SQL code (which carried the None through harmlessly).
        amap = {"N": {"anilist_id": 500, "tvdb_id": 5000, "tvdb_season": None, "tvdb_epoffset": None}}
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=amap,
                anidb=False,
                anibridge=False,
            ),
        )
        try:
            entry = resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=5000))[500]
            assert entry.tvdb_season == -1
            assert entry.tvdb_epoffset == 0
        finally:
            resolver.close()

    def test_candidate_sets_include_records_without_anilist(self) -> None:
        # The former full-map scan added external ids even from anilist-less records
        # (record E). The SQL distinct must keep that, even though lookups skip them.
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=AMAP,
                anidb=False,
                anibridge=False,
            ),
        )
        try:
            assert resolver.anime_id_set("tvdb_id") == {200, 999}
            assert resolver.anime_id_set("imdb_id") == {"tt100"}
            assert resolver.anime_id_set("tmdb_movie_id") == {7000}
            assert resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=999)) == {}  # E has no anilist
        finally:
            resolver.close()


class TestGetAnilistIdsMerge:
    def test_anibridge_wins_then_anime_fills_then_drop_and_sort(self) -> None:
        # 269 resolves in both sources for tvdb 74796; AniBridge must win (it is
        # queried first). 270 is AniBridge-only. Ignored ids are dropped; result
        # is sorted by AniList id.
        amap = {"x": {"anilist_id": 269, "tvdb_id": 74796, "tvdb_season": 7}}
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids={270},
            web=_WEB,
            sources=MappingSources(
                anime=amap,
                anidb=False,
                anibridge=GRAPH,
            ),
        )
        try:
            mappings, dropped = resolver.get_anilist_ids(ExternalIds(tvdb=74796))
            assert dropped == [270]
            assert list(mappings) == [269]  # 270 dropped; sorted
            # AniBridge won for 269 -> ANIBRIDGE mode (not the anime-ids season 7).
            assert mappings[269].mode is MappingMode.ANIBRIDGE
            assert mappings[269].tvdb_mappings == {2: [(1, 21)], 3: [(1, 9), (11, 12)]}
        finally:
            resolver.close()


class TestVu1DegradedAniBridge:
    """VU1: an AniBridge id reachable via imdb but not tvdb is 'degraded' (no tvdb
    season ranges). In the Sonarr/tvdb context it must defer to Kometa (which has
    the precise season) and, with no Kometa fallback, skip rather than grab the
    wrong episodes - while Radarr's AniBridge-primary precedence stays untouched.
    """

    def test_sonarr_context_kometa_overrides_degraded(self) -> None:
        # AniBridge resolves AniList 400 ONLY via imdb (imdb_show:tt400, no tvdb_show)
        # -> a degraded entry. Kometa carries the precise season for the same id on
        # the series' tvdb. In the Sonarr context (tvdb_id passed) Kometa must win.
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime={"k": {"anilist_id": 400, "tvdb_id": 555, "tvdb_season": 2, "imdb_id": "tt400"}},
                anidb=False,
                anibridge={"anilist:400": {"imdb_show:tt400": {}}},
            ),
        )
        try:
            mappings, _dropped = resolver.get_anilist_ids(ExternalIds(tvdb=555, imdb="tt400"))
            entry = mappings[400]
            assert entry.source is MappingSource.ANIME_IDS  # Kometa won, not the degraded AniBridge entry
            assert entry.tvdb_season == 2  # the precise season is no longer shadowed
        finally:
            resolver.close()

    def test_sonarr_movie_as_special_keeps_kometa_season_zero(self) -> None:
        # A movie-as-special: Kometa attaches AniList 600 to series 555 at
        # tvdb_season=0, while AniBridge carries it via the movie's imdb (degraded)
        # that the series' imdbId collides with. The Sonarr-context override must
        # restore Kometa's season 0 - without it the degraded -1 shadows season 0,
        # and check_ep_by_anime_ids DROPS every s0 episode (`-1 & season 0 -> False`),
        # so the special is never grabbed. Guards that the fix does not break (it
        # RESTORES) Sonarr grabbing movies as specials.
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime={"k": {"anilist_id": 600, "tvdb_id": 555, "tvdb_season": 0, "imdb_id": "tt600"}},
                anidb=False,
                anibridge={"anilist:600": {"imdb_movie:tt600": {}}},
            ),
        )
        try:
            mappings, _dropped = resolver.get_anilist_ids(ExternalIds(tvdb=555, imdb="tt600"))
            entry = mappings[600]
            assert entry.source is MappingSource.ANIME_IDS  # Kometa restored, not the degraded -1
            assert entry.tvdb_season == 0  # season 0 preserved -> the special is still selectable
        finally:
            resolver.close()

    def test_radarr_context_preserves_anibridge_precedence(self) -> None:
        # The SAME degraded shape for a movie (tmdb_movie/imdb, never tvdb-scoped):
        # in the Radarr context (NO tvdb_id) the override must NOT fire, so AniBridge
        # stays the winning source - the documented "AniBridge is primary" invariant.
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime={"k": {"anilist_id": 401, "tmdb_movie_id": 888, "imdb_id": "tt401"}},
                anidb=False,
                anibridge={"anilist:401": {"tmdb_movie:888": {}, "imdb_movie:tt401": {}}},
            ),
        )
        try:
            mappings, _dropped = resolver.get_anilist_ids(ExternalIds(tmdb=888, imdb="tt401"))
            assert mappings[401].source is MappingSource.ANIBRIDGE  # AniBridge stays primary
        finally:
            resolver.close()

    def test_imdb_only_degraded_entry_makes_sonarr_skip(self) -> None:
        # End-to-end: the resolver emits a degraded entry (no Kometa fallback), and
        # feeding THAT real entry to get_ep_list returns [] - the series is skipped,
        # not whole-series-grabbed. Anchors the fix to the state the pipeline emits.
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=False,  # no Kometa fallback
                anidb=False,
                anibridge={"anilist:500": {"imdb_show:tt500": {}}},
            ),
        )
        try:
            mappings, _dropped = resolver.get_anilist_ids(ExternalIds(tvdb=777, imdb="tt500"))
        finally:
            resolver.close()
        degraded = mappings[500]
        assert degraded.source is MappingSource.ANIBRIDGE
        assert degraded.mode is MappingMode.ANIME_IDS  # no tvdb ranges -> would have grabbed wrong eps
        assert degraded.tvdb_mappings is None

        episodes = make_bare_instance(
            SonarrEpisodes,
            _ep_list_cache={777: [SimpleNamespace(season_number=s, episode_number=1) for s in (1, 2, 3)]},
        )
        # A non-empty series, yet the degraded entry resolves to NO episodes (skip).
        assert episodes.get_ep_list(sonarr_series_id=777, al_id=500, mapping=degraded) == []


# --------------------------------------------------------------------------- #
# anidb: contract change (anidb_mapping_dict) edges
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
        cache_time=1,
        ignore_anilist_ids=set(),
        web=_WEB,
        sources=MappingSources(
            anime=False,
            anidb=root,
            anibridge=False,
        ),
    )


class TestAnidbMappingDict:
    def test_season_scoped_mapping(self) -> None:
        resolver = _anidb_resolver()
        try:
            assert resolver.has_anidb
            assert resolver.anidb_mapping_dict(1, 1) == {1: {1: 1, 2: 2}}
            assert resolver.anidb_mapping_dict(1, 2) == {2: {1: 3, 2: 4}}
        finally:
            resolver.close()

    def test_missing_season_or_id_is_empty(self) -> None:
        resolver = _anidb_resolver()
        try:
            assert resolver.anidb_mapping_dict(1, 5) == {}  # id known, season absent
            assert resolver.anidb_mapping_dict(404, 1) == {}  # id unknown
        finally:
            resolver.close()

    def test_repeated_season_is_last_wins(self) -> None:
        resolver = _anidb_resolver()
        try:
            assert resolver.anidb_mapping_dict(3, 1) == {1: {5: 5, 6: 6}}
        finally:
            resolver.close()

    def test_ambiguous_id_raises(self) -> None:
        resolver = _anidb_resolver()
        try:
            with pytest.raises(ValueError, match="appears in multiple anime-list entries"):
                resolver.anidb_mapping_dict(2, 1)
        finally:
            resolver.close()

    def test_disabled_source_is_empty_and_not_available(self) -> None:
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=False,
                anidb=False,
                anibridge=False,
            ),
        )
        try:
            assert not resolver.has_anidb
            assert resolver.anidb_mapping_dict(1, 1) == {}
        finally:
            resolver.close()


_HEALTHY_ANIME = '<anime anidbid="1"><mapping-list><mapping tvdbseason="1">;1-1;</mapping></mapping-list></anime>'


class TestAnidbSkipTallies:
    """``_anidb_rows`` tallies what it skips, split by kind.

    Id-level drops (missing/non-int anidbid) are *malformed* - 0 on healthy
    upstream data; mapping-level drops are *unsupported* forms the parser
    deliberately doesn't consume (the offset form's empty text, multi-episode
    spans) - present in the hundreds on a healthy file. A healthy sibling anime
    rides along in every case to pin that its rows survive the skip.
    """

    @staticmethod
    def _parse(*anime: str) -> m._AnidbParse:
        xml = f"<anime-list>{''.join(anime)}{_HEALTHY_ANIME}</anime-list>"
        return m._anidb_rows(ElementTree.fromstring(xml))

    @staticmethod
    def _assert_healthy_rows_only(parsed: m._AnidbParse) -> None:
        assert parsed.rows == [AnidbMappingRow(1, 1, 1, 1)]
        assert parsed.ambiguous == []

    def test_missing_anidbid_is_malformed(self) -> None:
        parsed = self._parse("<anime><mapping-list/></anime>")
        assert (parsed.malformed, parsed.unsupported) == (1, 0)
        self._assert_healthy_rows_only(parsed)

    def test_non_int_anidbid_is_malformed(self) -> None:
        parsed = self._parse('<anime anidbid="oops"><mapping-list/></anime>')
        assert (parsed.malformed, parsed.unsupported) == (1, 0)
        self._assert_healthy_rows_only(parsed)

    def test_offset_form_empty_text_is_unsupported(self) -> None:
        # The self-closing offset form (start/end/offset attributes, no text).
        parsed = self._parse(
            '<anime anidbid="2"><mapping-list>'
            '<mapping tvdbseason="0" start="1" end="4" offset="10"/>'
            "</mapping-list></anime>",
        )
        assert (parsed.malformed, parsed.unsupported) == (0, 1)
        self._assert_healthy_rows_only(parsed)

    def test_multi_episode_span_is_unsupported(self) -> None:
        # The multi-episode form (;3-6+7;): int("6+7") fails, the mapping is skipped.
        parsed = self._parse(
            '<anime anidbid="2"><mapping-list><mapping tvdbseason="1">;3-6+7;</mapping></mapping-list></anime>',
        )
        assert (parsed.malformed, parsed.unsupported) == (0, 1)
        self._assert_healthy_rows_only(parsed)

    def test_summary_line_carries_nonzero_tallies_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # File-based load (the digest-gated path): the DEBUG summary appends each
        # tally only when nonzero - here 1 unsupported form, 0 malformed.
        source = tmp_path / "anime-list.xml"
        source.write_text(
            "<anime-list>"
            '<anime anidbid="2"><mapping-list><mapping tvdbseason="0" start="1" end="4"/></mapping-list></anime>'
            f"{_HEALTHY_ANIME}</anime-list>",
            encoding="utf-8",
        )
        monkeypatch.setattr(m, "ANIDB_MAPPINGS_FILE", str(source))
        logger = logging.getLogger("pearlarr-test-anidb-summary")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        capture = CaptureHandler()
        logger.addHandler(capture)

        try:
            MappingResolver(
                cache_time=1,
                ignore_anilist_ids=set(),
                web=_WEB,
                sources=MappingSources(anime=False, anidb=None, anibridge=False),
                mappings_db=str(tmp_path / "mappings.db"),
                logger=logger,
            ).close()
        finally:
            logger.removeHandler(capture)

        [summary] = [r.getMessage() for r in capture.records if r.getMessage().startswith("Indexed")]
        assert summary.endswith(", 1 unsupported mapping forms)")
        assert "malformed" not in summary  # zero tallies stay off the line


# --------------------------------------------------------------------------- #
# Digest gate end-to-end: an unchanged source file is not re-parsed
# --------------------------------------------------------------------------- #


class TestDigestGate:
    def test_unchanged_file_is_not_reparsed_changed_file_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "anime_ids.json"
        source.write_text(json.dumps({"A": {"anilist_id": 1, "tvdb_id": 10}}))
        db = str(tmp_path / "mappings.db")

        # Point the resolver at the temp file; it already exists and is fresh, so
        # _maybe_download never fetches (no network).
        monkeypatch.setattr(m, "ANIME_IDS_FILE", str(source))

        calls = {"n": 0}
        real_parse = m._parse_anime_mappings

        def counting_parse(path: str) -> AnimeIdsMap:
            calls["n"] += 1
            return real_parse(path)

        monkeypatch.setattr(m, "_parse_anime_mappings", counting_parse)

        def build() -> MappingResolver:
            return MappingResolver(
                cache_time=1,
                ignore_anilist_ids=set(),
                web=_WEB,
                sources=MappingSources(
                    anime=None,
                    anidb=False,
                    anibridge=False,
                ),
                mappings_db=db,
            )

        build().close()
        assert calls["n"] == 1  # cold: parsed once
        build().close()
        assert calls["n"] == 1  # warm + unchanged digest: served from SQL, no re-parse

        # New content -> digest changes -> re-parse.
        source.write_text(json.dumps({"A": {"anilist_id": 2, "tvdb_id": 20}}))
        resolver = build()
        try:
            assert calls["n"] == 2
            assert resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=20))[2].anilist_id == 2
            assert resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=10)) == {}  # old content gone
        finally:
            resolver.close()


class TestConstructionFailureClosesStore:
    def test_store_closed_when_build_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-DatabaseError from _build must still close the store (the resolver
        # is never returned, so no one else can). Spy on close().
        closed = {"n": 0}
        real_close = MappingStore.close

        def spy_close(self: MappingStore) -> None:
            closed["n"] += 1
            real_close(self)

        monkeypatch.setattr(MappingStore, "close", spy_close)

        def boom(_amap: AnimeIdsMap) -> list[AnimeIdRow]:
            raise ValueError("parse blew up")

        monkeypatch.setattr(m, "_anime_ids_rows", boom)

        with pytest.raises(ValueError, match="parse blew up"):
            MappingResolver(
                cache_time=1,
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
    """An unwritable on-disk mapping db falls back to ``:memory:`` with one warning."""

    def test_file_db_write_failure_warns_and_rebuilds_in_memory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The first store write (the file-backed db) raises; the :memory: retry
        # runs the same _build and must serve the mappings as if nothing failed.
        real_replace = MappingStore.replace_anime_ids
        calls = {"n": 0}

        def flaky_replace(self: MappingStore, digest: str, rows: list[AnimeIdRow]) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("attempt to write a readonly database")
            real_replace(self, digest, rows)

        monkeypatch.setattr(MappingStore, "replace_anime_ids", flaky_replace)
        db = str(tmp_path / "mappings.db")
        recording = install_recording_hub()

        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(anime={"x": {"anilist_id": 1, "tvdb_id": 10}}, anidb=False, anibridge=False),
            mappings_db=db,
        )
        try:
            assert diagnostic_messages(recording, Severity.WARNING) == [
                f"Mapping cache at {db} could not be written; rebuilding it "
                "in memory for this run (slower startup, no data lost)",
            ]
            assert resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=10))[1].anilist_id == 1
        finally:
            resolver.close()

    def test_memory_db_write_failure_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ":memory:" IS the fallback, so it has none: the error propagates.
        def broken_replace(self: MappingStore, digest: str, rows: list[AnimeIdRow]) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(MappingStore, "replace_anime_ids", broken_replace)

        with pytest.raises(sqlite3.OperationalError):
            MappingResolver(
                cache_time=1,
                ignore_anilist_ids=set(),
                web=_WEB,
                sources=MappingSources(anime={"x": {"anilist_id": 1}}, anidb=False, anibridge=False),
            )


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

    ``TestRealDataParity`` carries ``@pytest.mark.real_data_dir`` so the autouse tmp
    data-dir override is off for it and ``resolve_paths()`` sees the developer's
    real ``PEARLARR_DATA_DIR``; evaluating this at import would instead capture
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


def _anidb_oracle(root: ElementTree.Element, anidb_id: int, tvdb_season: int) -> dict[int, dict[int, int]]:
    """Guarded reimplementation of the former anidb_anime_by_id + parse + raise."""

    items = [a for a in root.findall("anime") if a.get("anidbid") == str(anidb_id)]
    if len(items) > 1:
        raise ValueError(f"AniDB id {anidb_id} appears in multiple anime-list entries")
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
                assert sql_ab.lookup_by_tmdb(tmdb) == graph_ab.lookup_by_tmdb(tmdb)
            for imdb in graph_ab.all_imdb_ids:
                assert sql_ab.lookup_by_imdb(imdb) == graph_ab.lookup_by_imdb(imdb)
        finally:
            store.close()

    def test_anime_ids_sql_matches_oracle_sample(self, real_sources: _RealSources) -> None:
        amap = m._parse_anime_mappings(real_sources.anime_ids)
        resolver = MappingResolver(
            cache_time=99999,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=amap,
                anidb=False,
                anibridge=False,
            ),
        )
        try:
            tvdbs = [
                r["tvdb_id"] for r in amap.values() if r.get("tvdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for tvdb in tvdbs:
                assert resolver.get_mappings_from_anime_mappings(ExternalIds(tvdb=tvdb)) == _anime_oracle(
                    amap, ExternalIds(tvdb=tvdb)
                )
            imdbs = [
                r["imdb_id"] for r in amap.values() if r.get("imdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for imdb in imdbs:
                assert resolver.get_mappings_from_anime_mappings(ExternalIds(imdb=imdb)) == _anime_oracle(
                    amap, ExternalIds(imdb=imdb)
                )
        finally:
            resolver.close()

    def test_anidb_sql_matches_oracle_sample(self, real_sources: _RealSources) -> None:
        root = m._parse_anidb_mappings(real_sources.anidb)
        resolver = MappingResolver(
            cache_time=99999,
            ignore_anilist_ids=set(),
            web=_WEB,
            sources=MappingSources(
                anime=False,
                anidb=root,
                anibridge=False,
            ),
        )
        try:
            anidb_ids: list[int] = []
            for a in root.findall("anime"):
                aid = a.get("anidbid")
                if aid is not None and aid.isdigit():
                    anidb_ids.append(int(aid))
                if len(anidb_ids) >= 300:
                    break
            for anidb_id in anidb_ids:
                for season in (0, 1, 2):
                    try:
                        expected = _anidb_oracle(root, anidb_id, season)
                    except ValueError:
                        with pytest.raises(ValueError, match="appears in multiple anime-list entries"):
                            resolver.anidb_mapping_dict(anidb_id, season)
                        continue
                    assert resolver.anidb_mapping_dict(anidb_id, season) == expected
        finally:
            resolver.close()


def _boom(*_a: object, **_k: object) -> None:
    raise OSError("connection reset by peer")


class TestMaybeDownloadFailOpen:
    """A refresh blip falls open to the cached file; a first-ever download stays fatal."""

    def test_refresh_failure_falls_open_to_cached_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A stale-but-valid cached source whose refresh download fails on a transient
        # error must NOT abort the run: fall open to the on-disk copy and warn.
        source = tmp_path / "anime_ids.json"
        source.write_text("{}")
        old = time.time() - 10 * 86400  # 10 days old, past cache_time
        os.utime(source, (old, old))
        monkeypatch.setattr(MappingResolver, "_download_file", _boom)

        recording = install_recording_hub()
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, cache_time=1, _web=_WEB)
        resolver._maybe_download(str(source), "https://example/anime_ids.json", "anime_ids.json")

        # Warned (not aborted): the fall-open notice rides the hub.
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "Could not refresh anime_ids.json" in warning
        assert source.exists()  # the cached copy is left intact

    def test_first_ever_download_failure_still_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With no file on disk there is nothing to fall open to, so a first-ever
        # download failure stays fatal (the run cannot proceed without the source).
        monkeypatch.setattr(MappingResolver, "_download_file", _boom)
        resolver = make_bare_instance(MappingResolver, logger=None, _progress=None, cache_time=1, _web=_WEB)

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

        ``raise_for_status`` fires before the ``.part`` open, so nothing lands on
        disk; the exact-message assert pins the containment contract (the httpx
        message embeds the URL, ours must not).
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

        Pins the atomicity contract ``_maybe_download``'s fall-open relies on: a
        failed refresh writes only the temp (one chunk lands before the wire dies,
        verified live: respx streams the generator lazily), never the dest it
        falls back to; the ``finally`` sweeps the temp.
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
