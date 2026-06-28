"""Tests for the Sonarr sweep speedups: negative parse-cache, the series-id
fingerprint, the worker gating, and the concurrent fresh episode prefetch."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest import mock

from seadexarr.modules.cache import UPDATED_AT_STR_FORMAT
from seadexarr.modules.seadex_sonarr import (
    SONARR_FETCH_WORKERS,
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_NEG_CACHE_TTL_DAYS,
    SonarrClient,
    SonarrSync,
    sonarr_series_fingerprint,
)
from tests.builders import (
    FakeCacheStore,
    make_bare_instance,
    make_config,
    make_logger,
    make_sonarr_sync,
    rg_group,
    url_item,
)

_NOW = datetime(2026, 6, 28, 12, 0, 0)
_POS_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS)
_NEG_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_NEG_CACHE_TTL_DAYS)


def _stamp(days_ago: float) -> str:
    return (_NOW - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _fresh(record: dict[str, Any], *, series_fp: str = "fp") -> bool:
    return SonarrSync._sonarr_parse_is_fresh(
        record,
        cutoff=_POS_CUTOFF,
        neg_cutoff=_NEG_CUTOFF,
        series_fp=series_fp,
    )


class TestSeriesFingerprint:
    def test_order_and_duplicate_independent(self) -> None:
        assert sonarr_series_fingerprint([3, 1, 2]) == sonarr_series_fingerprint([2, 2, 1, 3])

    def test_different_sets_differ(self) -> None:
        assert sonarr_series_fingerprint([1, 2]) != sonarr_series_fingerprint([1, 2, 3])

    def test_empty_is_stable(self) -> None:
        assert sonarr_series_fingerprint([]) == sonarr_series_fingerprint(iter(()))


class TestSonarrParseIsFresh:
    def test_positive_within_ttl_is_fresh(self) -> None:
        assert _fresh({"fetched_at": _stamp(5), "episodes": [{"season": 1, "episode": 1}]})

    def test_positive_beyond_ttl_is_stale(self) -> None:
        assert not _fresh({"fetched_at": _stamp(40), "episodes": [{"season": 1, "episode": 1}]})

    def test_negative_fresh_when_fp_matches_and_within_backstop(self) -> None:
        rec = {"fetched_at": _stamp(2), "episodes": [], "series_fp": "fp"}
        assert _fresh(rec, series_fp="fp")

    def test_negative_stale_on_fp_mismatch(self) -> None:
        rec = {"fetched_at": _stamp(2), "episodes": [], "series_fp": "old"}
        assert not _fresh(rec, series_fp="fp")

    def test_negative_stale_beyond_backstop_ttl(self) -> None:
        rec = {"fetched_at": _stamp(SONARR_PARSE_NEG_CACHE_TTL_DAYS + 1), "episodes": [], "series_fp": "fp"}
        assert not _fresh(rec, series_fp="fp")

    def test_legacy_empty_without_fp_is_stale(self) -> None:
        # The migrated empty rows: re-parsed once, then re-stamped with a fp.
        assert not _fresh({"fetched_at": _stamp(1), "episodes": []})

    def test_non_dict_is_stale(self) -> None:
        assert not _fresh(None)  # type: ignore[arg-type]


class TestParseClientTransientVsEmpty:
    def _client(self, *, status: int, body: dict[str, Any] | None = None, boom: bool = False) -> SonarrClient:
        session = mock.MagicMock()
        if boom:
            import requests

            session.get.side_effect = requests.ConnectionError("down")
        else:
            resp = mock.MagicMock()
            resp.status_code = status
            resp.json.return_value = body or {}
            session.get.return_value = resp
        return make_bare_instance(
            SonarrClient,
            _url="http://sonarr",
            _api_key="k",
            _session=session,
            _logger=make_logger(),
        )

    def test_clean_200_with_episodes_returns_list(self) -> None:
        body = {"episodes": [{"seasonNumber": 1, "episodeNumber": 2}]}
        assert self._client(status=200, body=body).parse("f.mkv") == [{"season": 1, "episode": 2}]

    def test_clean_200_no_episodes_returns_empty_not_none(self) -> None:
        # Genuine no-match: cacheable as a negative.
        assert self._client(status=200, body={"episodes": []}).parse("f.mkv") == []

    def test_non_200_returns_none(self) -> None:
        # Transient: must NOT be cached as a negative.
        assert self._client(status=503).parse("f.mkv") is None

    def test_connection_error_returns_none(self) -> None:
        assert self._client(status=200, boom=True).parse("f.mkv") is None


class TestFetchWorkers:
    def test_concurrent_when_no_sleep(self) -> None:
        strat = make_sonarr_sync(_config=make_config(sleep_time=0))
        assert strat._fetch_workers() == SONARR_FETCH_WORKERS

    def test_sequential_when_throttled(self) -> None:
        strat = make_sonarr_sync(_config=make_config(sleep_time=2))
        assert strat._fetch_workers() == 1


class _Series:
    """A stand-in Sonarr series satisfying the ``SonarrItem`` protocol surface."""

    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tvdbId: int
    tmdbId: int

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


def _item(series_id: int, *, monitored: bool = True) -> _Series:
    return _Series(
        id=series_id,
        title=f"Series {series_id}",
        imdbId=None,
        monitored=monitored,
        tvdbId=series_id,
        tmdbId=series_id,
    )


class TestPrefetchEpisodes:
    def _strat(self, *, mapped: set[int], sleep_time: int = 0) -> tuple[SonarrSync, mock.MagicMock]:
        sonarr = mock.MagicMock()
        sonarr.episodes.side_effect = lambda sid, quiet=False: [f"ep{sid}"] if sid in mapped else None
        services = mock.MagicMock()
        # Only "mapped" series resolve to a non-empty AniList mapping.
        services.get_anilist_ids.side_effect = lambda **kw: {1: object()} if kw["tvdb_id"] in mapped else {}
        strat = make_sonarr_sync(
            sonarr=sonarr,
            _services=services,
            _config=make_config(sleep_time=sleep_time),
            _ep_list_cache={},
            logger=make_logger(),
        )
        return strat, sonarr

    def test_warms_only_mapped_series(self) -> None:
        strat, _ = self._strat(mapped={1, 2})
        strat.prefetch_episodes([_item(1), _item(2), _item(3)])
        assert strat._ep_list_cache == {1: ["ep1"], 2: ["ep2"]}

    def test_skips_unmonitored_when_ignored(self) -> None:
        strat, _ = self._strat(mapped={1, 2})
        strat._config = make_config(sleep_time=0, ignore_unmonitored=True)
        strat.prefetch_episodes([_item(1, monitored=False), _item(2)])
        assert strat._ep_list_cache == {2: ["ep2"]}

    def test_dedups_series_ids(self) -> None:
        strat, sonarr = self._strat(mapped={1})
        strat.prefetch_episodes([_item(1), _item(1)])
        assert sonarr.episodes.call_count == 1

    def test_none_result_not_cached(self) -> None:
        # series 9 is a candidate (resolves a mapping) but episodes() returns None.
        strat, sonarr = self._strat(mapped={9})
        sonarr.episodes.side_effect = lambda sid, quiet=False: None
        strat.prefetch_episodes([_item(9)])
        assert strat._ep_list_cache == {}

    def test_sequential_path_matches_concurrent(self) -> None:
        strat, _ = self._strat(mapped={1, 2}, sleep_time=2)
        strat.prefetch_episodes([_item(1), _item(2)])
        assert strat._ep_list_cache == {1: ["ep1"], 2: ["ep2"]}


class TestParseEpisodesNegativeCache:
    def _strat(
        self,
        *,
        parse_result: Any,
        sleep_time: int = 0,
        sonarr_parse: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[SonarrSync, mock.MagicMock]:
        sonarr = mock.MagicMock()
        sonarr.parse.return_value = parse_result
        strat = make_sonarr_sync(
            sonarr=sonarr,
            _config=make_config(sleep_time=sleep_time),
            cache_store=FakeCacheStore(sonarr_parse=sonarr_parse or {}),
            _series_fp="fp",
            _ep_list_cache={},
            logger=make_logger(),
        )
        return strat, sonarr

    @staticmethod
    def _dict(*files: str) -> dict[str, Any]:
        return {"GroupA": rg_group({"u": url_item(files=list(files), size=[100] * len(files))})}

    def test_genuine_empty_is_negative_cached_with_fp(self) -> None:
        strat, _ = self._strat(parse_result=[])
        strat.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"))
        rec = strat.cache_store.get_sonarr_parse("[X] Show - 01.mkv")
        assert rec is not None
        assert rec["episodes"] == []
        assert rec["series_fp"] == "fp"

    def test_transient_none_is_not_cached(self) -> None:
        strat, _ = self._strat(parse_result=None)
        strat.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"))
        assert strat.cache_store.get_sonarr_parse("[X] Show - 01.mkv") is None

    def test_fresh_negative_hit_skips_network(self) -> None:
        seeded = {
            "[X] Show - 01.mkv": {
                "fetched_at": datetime.now().strftime(UPDATED_AT_STR_FORMAT),
                "episodes": [],
                "series_fp": "fp",
            },
        }
        strat, sonarr = self._strat(parse_result=[], sonarr_parse=seeded)
        strat.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"))
        sonarr.parse.assert_not_called()

    def test_audio_file_never_parsed(self) -> None:
        strat, sonarr = self._strat(parse_result=[])
        strat.parse_episodes_from_seadex(self._dict("[X] OST - 01.flac"))
        sonarr.parse.assert_not_called()

    def test_concurrent_pass_negative_caches_each_file(self) -> None:
        strat, _ = self._strat(parse_result=[], sleep_time=0)
        strat.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv", "[X] Show - 02.mkv"))
        for name in ("[X] Show - 01.mkv", "[X] Show - 02.mkv"):
            rec = strat.cache_store.get_sonarr_parse(name)
            assert rec is not None
            assert rec["episodes"] == []
            assert rec["series_fp"] == "fp"
