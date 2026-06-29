"""Parity + edge tests for the SQL-backed ``MappingResolver`` / ``AniBridge``.

The SQL backings must reproduce the in-memory implementations exactly. AniBridge's
graph-backed view is used directly as the oracle (the unchanged parser); anime_ids
is checked against an inline reconstruction of the former reverse-index merge; and
the anidb + digest-gate behaviours are pinned. These are the safety net for the
"parse once, serve from SQL" migration - the resolver lookups had no direct
coverage before.
"""

import json
import os
import time
from unittest import mock
from xml.etree import ElementTree

import pytest

import seadexarr.modules.mappings as m
from seadexarr.modules.anibridge import AniBridge
from seadexarr.modules.mapping_store import MappingStore
from seadexarr.modules.mappings import (
    MappingMode,
    MappingResolver,
    TmdbType,
    _entry_from_raw,
)
from seadexarr.modules.paths import resolve_paths
from tests.builders import make_bare_instance

# --------------------------------------------------------------------------- #
# AniBridge: SQL backing must equal the graph backing (the oracle)
# --------------------------------------------------------------------------- #

GRAPH = {
    "anilist:269": {
        "tvdb_show:74796:s2": {"21-41": "1-21"},
        "tvdb_show:74796:s3": {"42-50": "1-9,11-12"},  # multi-segment range
        "anidb:1234": {},
        "imdb_show:tt0269": {},
        "mal:111": {},
        "tmdb_show:5000:s1": {"1-12": "1-12"},
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


def _ab_pair(graph: dict) -> tuple[AniBridge, AniBridge, MappingStore]:
    """(graph-backed oracle, SQL-backed view, store) over the same graph."""

    graph_ab = AniBridge(graph)
    store = MappingStore.open(":memory:")
    store.replace_anibridge("d", *graph_ab.to_rows())
    return graph_ab, AniBridge.from_store(store), store


class TestAniBridgeParity:
    def test_lookups_and_sets_match_graph(self) -> None:
        graph_ab, sql_ab, store = _ab_pair(GRAPH)
        try:
            for tvdb in (74796, 600, 111, 222, 424242):
                assert sql_ab.lookup_by_tvdb(tvdb) == graph_ab.lookup_by_tvdb(tvdb)
            for tmdb in (5000, 4242):
                assert sql_ab.lookup_by_tmdb(tmdb, "show") == graph_ab.lookup_by_tmdb(tmdb, "show")
            for tmdb in (888, 4242):
                assert sql_ab.lookup_by_tmdb(tmdb, "movie") == graph_ab.lookup_by_tmdb(tmdb, "movie")
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
        # is opposite behaviour downstream; it must survive the SQL round-trip.
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
# anime_ids: SQL lookups must equal the former reverse-index merge
# --------------------------------------------------------------------------- #

AMAP = {
    "A": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 2, "tvdb_epoffset": 3, "imdb_id": "tt100", "anidb_id": 50},
    "B": {"anilist_id": 101, "tvdb_id": 200},  # same tvdb id, 2nd anilist
    "C": {"anilist_id": 100, "tvdb_id": 200, "tvdb_season": 9},  # dup anilist -> first wins
    "D": {"anilist_id": 102, "imdb_id": "tt100"},  # shares imdb
    "E": {"tvdb_id": 999},  # no anilist -> filter-set only
    "F": {"anilist_id": 103, "tmdb_movie_id": 7000},
}


def _anime_oracle(amap: dict, **kw: object) -> dict[int, object]:
    """Inline twin of the former reverse-index + no-clobber merge (the oracle)."""

    index: dict[str, dict[object, list[dict]]] = {
        "tvdb_id": {},
        "tmdb_movie_id": {},
        "tmdb_show_id": {},
        "imdb_id": {},
    }
    for rec in amap.values():
        if rec.get("anilist_id") is None:
            continue
        for field, bucket in index.items():
            value = rec.get(field)
            if value is not None:
                bucket.setdefault(value, []).append(rec)

    result: dict[int, object] = {}

    def merge(field: str, value: object) -> None:
        for rec in index[field].get(value, []):
            aid = rec["anilist_id"]
            if aid not in result:
                result[aid] = _entry_from_raw(aid, rec)

    tmdb_type = kw.get("tmdb_type", TmdbType.MOVIE)
    if kw.get("tvdb_id") is not None:
        merge("tvdb_id", kw["tvdb_id"])
    if kw.get("tmdb_id") is not None:
        merge(f"tmdb_{tmdb_type}_id", kw["tmdb_id"])
    if kw.get("imdb_id") is not None:
        merge("imdb_id", kw["imdb_id"])
    return result


class TestAnimeIdsParity:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tvdb_id": 200},
            {"imdb_id": "tt100"},
            {"tmdb_id": 7000, "tmdb_type": TmdbType.MOVIE},
            {"tvdb_id": 200, "imdb_id": "tt100"},
            {"tvdb_id": 424242},  # no match -> empty
        ],
    )
    def test_lookup_matches_oracle(self, kwargs: dict) -> None:
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            anime_mappings_cfg=AMAP,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=False,
        )
        try:
            assert resolver.get_mappings_from_anime_mappings(**kwargs) == _anime_oracle(AMAP, **kwargs)
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
            anime_mappings_cfg=amap,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=False,
        )
        try:
            entry = resolver.get_mappings_from_anime_mappings(tvdb_id=5000)[500]
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
            anime_mappings_cfg=AMAP,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=False,
        )
        try:
            assert resolver.anime_id_set("tvdb_id") == {200, 999}
            assert resolver.anime_id_set("imdb_id") == {"tt100"}
            assert resolver.anime_id_set("tmdb_movie_id") == {7000}
            assert resolver.get_mappings_from_anime_mappings(tvdb_id=999) == {}  # E has no anilist
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
            anime_mappings_cfg=amap,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=GRAPH,
        )
        try:
            mappings, dropped = resolver.get_anilist_ids(tvdb_id=74796)
            assert dropped == [270]
            assert list(mappings) == [269]  # 270 dropped; sorted
            # AniBridge won for 269 -> ANIBRIDGE mode (not the anime-ids season 7).
            assert mappings[269].mode is MappingMode.ANIBRIDGE
            assert mappings[269].tvdb_mappings == {2: [(1, 21)], 3: [(1, 9), (11, 12)]}
        finally:
            resolver.close()


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
        anime_mappings_cfg=False,
        anidb_mappings_cfg=root,
        anibridge_mappings_cfg=False,
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
            with pytest.raises(ValueError, match="Multiple AniDB mappings"):
                resolver.anidb_mapping_dict(2, 1)
        finally:
            resolver.close()

    def test_disabled_source_is_empty_and_not_available(self) -> None:
        resolver = MappingResolver(
            cache_time=1,
            ignore_anilist_ids=set(),
            anime_mappings_cfg=False,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=False,
        )
        try:
            assert not resolver.has_anidb
            assert resolver.anidb_mapping_dict(1, 1) == {}
        finally:
            resolver.close()


# --------------------------------------------------------------------------- #
# Digest gate end-to-end: an unchanged source file is not re-parsed
# --------------------------------------------------------------------------- #


class TestDigestGate:
    def test_unchanged_file_is_not_reparsed_changed_file_is(self, tmp_path, monkeypatch) -> None:
        source = tmp_path / "anime_ids.json"
        source.write_text(json.dumps({"A": {"anilist_id": 1, "tvdb_id": 10}}))
        db = str(tmp_path / "mappings.db")

        # Point the resolver at the temp file; it already exists and is fresh, so
        # _maybe_download never fetches (no network).
        monkeypatch.setattr(m, "ANIME_IDS_FILE", str(source))

        calls = {"n": 0}
        real_parse = m._parse_anime_mappings

        def counting_parse(path: str) -> object:
            calls["n"] += 1
            return real_parse(path)

        monkeypatch.setattr(m, "_parse_anime_mappings", counting_parse)

        def build() -> MappingResolver:
            return MappingResolver(
                cache_time=1,
                ignore_anilist_ids=set(),
                anime_mappings_cfg=None,
                anidb_mappings_cfg=False,
                anibridge_mappings_cfg=False,
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
            assert resolver.get_mappings_from_anime_mappings(tvdb_id=20)[2].anilist_id == 2
            assert resolver.get_mappings_from_anime_mappings(tvdb_id=10) == {}  # old content gone
        finally:
            resolver.close()


class TestConstructionFailureClosesStore:
    def test_store_closed_when_build_raises(self, monkeypatch) -> None:
        # A non-DatabaseError from _build must still close the store (the resolver
        # is never returned, so no one else can). Spy on close().
        closed = {"n": 0}
        real_close = MappingStore.close

        def spy_close(self: MappingStore) -> None:
            closed["n"] += 1
            real_close(self)

        monkeypatch.setattr(MappingStore, "close", spy_close)

        def boom(_amap: object) -> list:
            raise ValueError("parse blew up")

        monkeypatch.setattr(m, "_anime_ids_rows", boom)

        with pytest.raises(ValueError, match="parse blew up"):
            MappingResolver(
                cache_time=1,
                ignore_anilist_ids=set(),
                anime_mappings_cfg={"x": {"anilist_id": 1}},
                anidb_mappings_cfg=False,
                anibridge_mappings_cfg=False,
            )
        assert closed["n"] >= 1


# --------------------------------------------------------------------------- #
# Real-data parity: SQL backings must equal the in-memory oracle over the
# actual source files (gitignored, so skipped in CI / when absent). This is the
# migration's core promise - it diffs the SQL path against the unchanged code.
# The sources are cached in the data directory (next to mappings.db).
# --------------------------------------------------------------------------- #

_REAL_DIR = resolve_paths().data_dir
_REAL_ANIME_IDS = os.path.join(_REAL_DIR, m.ANIME_IDS_FILE)
_REAL_ANIDB = os.path.join(_REAL_DIR, m.ANIDB_MAPPINGS_FILE)
_REAL_ANIBRIDGE = os.path.join(_REAL_DIR, m.ANIBRIDGE_MAPPINGS_FILE)
_HAVE_REAL = all(os.path.exists(f) for f in (_REAL_ANIME_IDS, _REAL_ANIDB, _REAL_ANIBRIDGE))


def _anidb_oracle(root: ElementTree.Element, anidb_id: int, tvdb_season: int) -> dict:
    """Guarded reimplementation of the former anidb_anime_by_id + parse + raise."""

    items = [a for a in root.findall("anime") if a.get("anidbid") == str(anidb_id)]
    if len(items) > 1:
        raise ValueError("Multiple AniDB mappings found. This should not happen!")
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


@pytest.mark.skipif(not _HAVE_REAL, reason="real mapping source files not present")
class TestRealDataParity:
    def test_anibridge_sql_matches_graph_over_all_ids(self) -> None:
        with open(_REAL_ANIBRIDGE) as f:
            graph = json.load(f)
        graph_ab = AniBridge(graph)
        store = MappingStore.open(":memory:")
        store.replace_anibridge("d", *graph_ab.to_rows())
        sql_ab = AniBridge.from_store(store)
        try:
            assert sql_ab.all_tvdb_ids == graph_ab.all_tvdb_ids
            assert sql_ab.all_tmdb_movie_ids == graph_ab.all_tmdb_movie_ids
            assert sql_ab.all_imdb_ids == graph_ab.all_imdb_ids
            assert len(sql_ab) == len(graph_ab)
            for tvdb in graph_ab.all_tvdb_ids:
                assert sql_ab.lookup_by_tvdb(tvdb) == graph_ab.lookup_by_tvdb(tvdb)
            for tmdb in set(graph_ab.tmdb_show_index):
                assert sql_ab.lookup_by_tmdb(tmdb, "show") == graph_ab.lookup_by_tmdb(tmdb, "show")
            for tmdb in graph_ab.all_tmdb_movie_ids:
                assert sql_ab.lookup_by_tmdb(tmdb, "movie") == graph_ab.lookup_by_tmdb(tmdb, "movie")
            for imdb in graph_ab.all_imdb_ids:
                assert sql_ab.lookup_by_imdb(imdb) == graph_ab.lookup_by_imdb(imdb)
        finally:
            store.close()

    def test_anime_ids_sql_matches_oracle_sample(self) -> None:
        amap = m._parse_anime_mappings(_REAL_ANIME_IDS)
        resolver = MappingResolver(
            cache_time=99999,
            ignore_anilist_ids=set(),
            anime_mappings_cfg=amap,
            anidb_mappings_cfg=False,
            anibridge_mappings_cfg=False,
        )
        try:
            tvdbs = [
                r["tvdb_id"] for r in amap.values() if r.get("tvdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for tvdb in tvdbs:
                assert resolver.get_mappings_from_anime_mappings(tvdb_id=tvdb) == _anime_oracle(amap, tvdb_id=tvdb)
            imdbs = [
                r["imdb_id"] for r in amap.values() if r.get("imdb_id") is not None and r.get("anilist_id") is not None
            ][:200]
            for imdb in imdbs:
                assert resolver.get_mappings_from_anime_mappings(imdb_id=imdb) == _anime_oracle(amap, imdb_id=imdb)
        finally:
            resolver.close()

    def test_anidb_sql_matches_oracle_sample(self) -> None:
        root = m._parse_anidb_mappings(_REAL_ANIDB)
        resolver = MappingResolver(
            cache_time=99999,
            ignore_anilist_ids=set(),
            anime_mappings_cfg=False,
            anidb_mappings_cfg=root,
            anibridge_mappings_cfg=False,
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
                        with pytest.raises(ValueError, match="Multiple AniDB mappings"):
                            resolver.anidb_mapping_dict(anidb_id, season)
                        continue
                    assert resolver.anidb_mapping_dict(anidb_id, season) == expected
        finally:
            resolver.close()


def _boom(*_a: object, **_k: object) -> None:
    raise OSError("connection reset by peer")


class TestMaybeDownloadFailOpen:
    """A refresh blip falls open to the cached file; a first-ever download stays fatal."""

    def test_refresh_failure_falls_open_to_cached_file(self, tmp_path, monkeypatch) -> None:
        # A stale-but-valid cached source whose refresh download fails on a transient
        # error must NOT abort the run: fall open to the on-disk copy and warn.
        source = tmp_path / "anime_ids.json"
        source.write_text("{}")
        old = time.time() - 10 * 86400  # 10 days old, past cache_time
        os.utime(source, (old, old))
        monkeypatch.setattr(m, "_download_file", _boom)

        logger = mock.MagicMock()
        resolver = make_bare_instance(MappingResolver, logger=logger, _progress=None, cache_time=1)
        resolver._maybe_download(str(source), "https://example/anime_ids.json", "anime_ids.json")

        assert logger.warning.called
        assert source.exists()  # the cached copy is left intact

    def test_first_ever_download_failure_still_propagates(self, tmp_path, monkeypatch) -> None:
        # With no file on disk there is nothing to fall open to, so a first-ever
        # download failure stays fatal (the run cannot proceed without the source).
        monkeypatch.setattr(m, "_download_file", _boom)
        resolver = make_bare_instance(MappingResolver, logger=mock.MagicMock(), _progress=None, cache_time=1)

        with pytest.raises(OSError):
            resolver._maybe_download(str(tmp_path / "missing.json"), "https://example/x.json", "x")
