# pyright: strict
# pyright: reportPrivateUsage=false
# These read the episode collaborator's private per-run state (eps._ep_list_cache /
# eps._config) and call the private SonarrParseCache._sonarr_parse_is_fresh; strict
# re-flags that and the repo disables reportPrivateUsage for tests.
"""Tests for the Sonarr sweep speedups: negative parse-cache, the series-id
fingerprint, the worker gating, and the concurrent fresh episode prefetch."""

from __future__ import annotations

from datetime import datetime, timedelta

import requests
from seadex import EntryRecord

from seadexarr.modules.cache import UPDATED_AT_STR_FORMAT
from seadexarr.modules.config import Arr
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.run_services import RunServices
from seadexarr.modules.seadex_gateway import SeaDexMiss
from seadexarr.modules.seadex_types import SeadexDict, SonarrEpisode
from seadexarr.modules.sonarr_client import SonarrClient
from seadexarr.modules.sonarr_episodes import (
    SONARR_FETCH_WORKERS,
    SonarrEpisodes,
    fetch_workers,
    sonarr_series_fingerprint,
)
from seadexarr.modules.sonarr_parse import (
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_NEG_CACHE_TTL_DAYS,
    ParseWindow,
    SonarrParseCache,
)

from .builders import (
    FakeCacheStore,
    make_bare_instance,
    make_config,
    make_entry_record,
    make_logger,
    make_services,
    make_sonarr_episodes,
    make_sonarr_parse,
    rg_group,
    sonarr_ep,
    url_item,
)

_NOW = datetime(2026, 6, 28, 12, 0, 0)
_POS_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS)
_NEG_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_NEG_CACHE_TTL_DAYS)


def _stamp(days_ago: float) -> str:
    return (_NOW - timedelta(days=days_ago)).strftime(UPDATED_AT_STR_FORMAT)


def _fresh(record: dict[str, object], *, series_fp: str = "fp") -> bool:
    return SonarrParseCache._sonarr_parse_is_fresh(
        record,
        window=ParseWindow(
            now_str=_NOW.strftime(UPDATED_AT_STR_FORMAT),
            cutoff=_POS_CUTOFF,
            neg_cutoff=_NEG_CUTOFF,
            series_fp=series_fp,
        ),
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
        rec: dict[str, object] = {"fetched_at": _stamp(2), "episodes": [], "series_fp": "fp"}
        assert _fresh(rec, series_fp="fp")

    def test_negative_stale_on_fp_mismatch(self) -> None:
        rec: dict[str, object] = {"fetched_at": _stamp(2), "episodes": [], "series_fp": "old"}
        assert not _fresh(rec, series_fp="fp")

    def test_negative_stale_beyond_backstop_ttl(self) -> None:
        rec: dict[str, object] = {
            "fetched_at": _stamp(SONARR_PARSE_NEG_CACHE_TTL_DAYS + 1),
            "episodes": [],
            "series_fp": "fp",
        }
        assert not _fresh(rec, series_fp="fp")

    def test_legacy_empty_without_fp_is_stale(self) -> None:
        # The migrated empty rows: re-parsed once, then re-stamped with a fp.
        assert not _fresh({"fetched_at": _stamp(1), "episodes": []})


class _FakeResponse:
    """A minimal ``requests``-style response: ``status_code`` + a JSON body."""

    def __init__(self, status_code: int, body: dict[str, list[dict[str, int]]]) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, list[dict[str, int]]]:
        return self._body


class _FakeSession:
    """A ``requests.Session`` stand-in scripting one ``get`` outcome.

    ``boom`` raises a ``ConnectionError`` (the transient path ``parse`` swallows);
    otherwise ``get`` returns the scripted status + body.
    """

    def __init__(self, *, status: int, body: dict[str, list[dict[str, int]]], boom: bool) -> None:
        self._status = status
        self._body = body
        self._boom = boom

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: tuple[int, int] | None = None,
    ) -> _FakeResponse:
        del url, headers, timeout
        if self._boom:
            raise requests.ConnectionError("down")
        return _FakeResponse(self._status, self._body)


class TestParseClientTransientVsEmpty:
    def _client(
        self,
        *,
        status: int,
        body: dict[str, list[dict[str, int]]] | None = None,
        boom: bool = False,
    ) -> SonarrClient:
        session = _FakeSession(status=status, body=body or {}, boom=boom)
        return make_bare_instance(
            SonarrClient,
            _url="http://sonarr",
            _headers={"X-Api-Key": "k"},
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
        assert fetch_workers(make_config(sleep_time=0)) == SONARR_FETCH_WORKERS

    def test_sequential_when_throttled(self) -> None:
        assert fetch_workers(make_config(sleep_time=2)) == 1


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


def _eps_for(series_id: int) -> list[SonarrEpisode]:
    """A distinguishable one-episode list per series (episode_number == series id)."""

    return [sonarr_ep(1, series_id)]


def _ids(*al_ids: int) -> dict[int, MappingEntry]:
    """A ``{al_id -> mapping}`` dict; only the keys are read by the prefetch gate."""

    return {aid: MappingEntry(anilist_id=aid) for aid in al_ids}


class _Recorder:
    """A ``ProgressSink`` that records every ``progress`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.calls.append((fraction, detail))


class _Sonarr:
    """A scripted Sonarr client for the prefetch warm.

    By default ``episodes(sid)`` returns a distinguishable one-episode list per
    series (``_eps_for``); ``return_none`` degrades every fetch to a transient
    miss, and ``raise_on`` makes the listed ids raise (the worker-degradation
    case). Records each ``(series_id, quiet)`` call so the dedup / not-fetched /
    quiet assertions read recorded state.
    """

    def __init__(self, *, return_none: bool = False, raise_on: set[int] | None = None) -> None:
        self.return_none = return_none
        self.raise_on: set[int] = set(raise_on or set())
        self.calls: list[tuple[int, bool]] = []

    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        self.calls.append((series_id, quiet))
        if series_id in self.raise_on:
            raise ValueError("boom")
        if self.return_none:
            return None
        return _eps_for(series_id)


class _Services:
    """A stand-in for the run machinery the prefetch consults.

    ``get_anilist_ids`` resolves a series' tvdb id to its ``{al_id -> mapping}``
    dict (``identity`` returns ``{tvdb_id: mapping}`` for any series, mirroring the
    always-mapped helper); ``al_id_needs_scan`` is the per-id needs-scan gate
    (``needs_scan=None`` reports every id as scannable).
    """

    def __init__(
        self,
        *,
        mapping: dict[int, dict[int, MappingEntry]] | None = None,
        identity: bool = False,
        needs_scan: set[int] | None = None,
    ) -> None:
        self._mapping = mapping or {}
        self._identity = identity
        self._needs_scan = needs_scan

    def get_anilist_ids(
        self,
        *,
        tvdb_id: int,
        imdb_id: str | None = None,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        del imdb_id, log_ignored
        if self._identity:
            return {tvdb_id: MappingEntry(anilist_id=tvdb_id)}
        return self._mapping.get(tvdb_id, {})

    def al_id_needs_scan(self, al_id: int) -> bool:
        if self._needs_scan is None:
            return True
        return al_id in self._needs_scan


class TestPrefetchEpisodes:
    def _eps(self, *, mapped: set[int], sleep_time: int = 0) -> tuple[SonarrEpisodes, _Sonarr]:
        sonarr = _Sonarr()
        # Only "mapped" series resolve to a non-empty AniList mapping; needs_scan
        # defaults to "every id scannable".
        services = _Services(mapping={sid: _ids(1) for sid in mapped})
        eps = make_sonarr_episodes(
            sonarr=sonarr,
            _services=services,
            _config=make_config(sleep_time=sleep_time),
        )
        return eps, sonarr

    def test_warms_only_mapped_series(self) -> None:
        eps, _ = self._eps(mapped={1, 2})
        eps.prefetch([_item(1), _item(2), _item(3)])
        assert eps._ep_list_cache == {1: _eps_for(1), 2: _eps_for(2)}

    def test_skips_unmonitored_when_ignored(self) -> None:
        eps, _ = self._eps(mapped={1, 2})
        eps._config = make_config(sleep_time=0, ignore_unmonitored=True)
        eps.prefetch([_item(1, monitored=False), _item(2)])
        assert eps._ep_list_cache == {2: _eps_for(2)}

    def test_dedups_series_ids(self) -> None:
        eps, sonarr = self._eps(mapped={1})
        eps.prefetch([_item(1), _item(1)])
        assert len(sonarr.calls) == 1

    def test_none_result_not_cached(self) -> None:
        # series 9 is a candidate (resolves a mapping) but episodes() returns None.
        eps, sonarr = self._eps(mapped={9})
        sonarr.return_none = True
        eps.prefetch([_item(9)])
        assert eps._ep_list_cache == {}

    def test_raising_series_does_not_abort_sweep(self) -> None:
        # CB5: a worker that RAISES (e.g. a non-JSON 200 response) must not abort the
        # whole concurrent sweep; that series is left unwarmed, the rest still warm.
        eps, sonarr = self._eps(mapped={1, 2})
        sonarr.raise_on = {1}
        warmed = eps.prefetch([_item(1), _item(2)])

        assert eps._ep_list_cache == {2: _eps_for(2)}  # 1 raised -> unwarmed; 2 warmed
        assert warmed == 2  # both attempted

    def test_sequential_path_matches_concurrent(self) -> None:
        eps, _ = self._eps(mapped={1, 2}, sleep_time=2)
        eps.prefetch([_item(1), _item(2)])
        assert eps._ep_list_cache == {1: _eps_for(1), 2: _eps_for(2)}

    def test_returns_warmed_count(self) -> None:
        # Only mapped, monitored series are warmed: 3 is unmapped, so 2 warmed.
        eps, _ = self._eps(mapped={1, 2})
        assert eps.prefetch([_item(1), _item(2), _item(3)]) == 2

    def test_drives_progress_per_series(self) -> None:
        eps, _ = self._eps(mapped={1, 2})
        rec = _Recorder()
        eps.prefetch([_item(1), _item(2), _item(3)], progress=rec)
        # One drive per warmed series, ending complete. Completion order is
        # nondeterministic, so assert on the count + the final value, not the
        # intermediate sequence.
        assert len(rec.calls) == 2
        assert rec.calls[-1] == (1.0, "2/2")

    def test_count_is_attempted_not_cached(self) -> None:
        # series 9 is a candidate (resolves a mapping) but episodes() returns None.
        # It's still attempted, so it counts toward the return value + the bar.
        eps, sonarr = self._eps(mapped={9})
        sonarr.return_none = True
        rec = _Recorder()
        assert eps.prefetch([_item(9)], progress=rec) == 1
        assert eps._ep_list_cache == {}  # nothing cached
        assert rec.calls[-1] == (1.0, "1/1")  # but progress still completed


def _entry(dt: datetime) -> EntryRecord:
    """A real SeaDex entry stamped at ``dt`` (only ``updated_at`` is read)."""

    return make_entry_record(updated_at=dt)


class _Seadex:
    """A SeaDex gateway stand-in returning one fixed entry (or a miss) for any al_id."""

    def __init__(self, entry: EntryRecord | None) -> None:
        self._entry = entry

    def entry(self, al_id: int) -> EntryRecord | SeaDexMiss:
        del al_id
        return self._entry if self._entry is not None else SeaDexMiss.NO_ENTRY


class TestAlIdNeedsScan:
    """``RunServices.al_id_needs_scan``: the side-effect-free mirror of the per-id
    loop's no-entry + ``cached_entry_skip`` gates, so ``prefetch_episodes`` warms
    only the series the loop would actually process (the SeaDex-modification-times
    fix). Pinned against the same cases as ``cached_entry_skip``."""

    @staticmethod
    def _run(*, entry: EntryRecord | None, cache: FakeCacheStore, **cfg: object) -> RunServices:
        return make_services(_seadex=_Seadex(entry), cache_store=cache, **cfg)

    def test_no_seadex_entry_does_not_need_scan(self) -> None:
        run = self._run(entry=None, cache=FakeCacheStore())
        assert run.al_id_needs_scan(7) is False

    def test_uncached_entry_needs_scan(self) -> None:
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=FakeCacheStore())
        assert run.al_id_needs_scan(7) is True

    def test_cached_and_matching_does_not_need_scan(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=cache)
        assert run.al_id_needs_scan(7) is False

    def test_cached_but_stale_needs_scan(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2022, 6, 6)), cache=cache)
        assert run.al_id_needs_scan(7) is True

    def test_ignore_update_times_forces_scan_when_entry_exists(self) -> None:
        # A matching cached entry is normally skipped; ignore_seadex_update_times
        # makes the loop re-process it, so the predicate must report needs-scan.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(
            entry=_entry(datetime(2021, 1, 1)),
            cache=cache,
            ignore_seadex_update_times=True,
        )
        assert run.al_id_needs_scan(7) is True

    def test_ignore_update_times_still_skips_when_no_entry(self) -> None:
        # No SeaDex entry -> al_id_prologue would skip regardless of the flag.
        run = self._run(entry=None, cache=FakeCacheStore(), ignore_seadex_update_times=True)
        assert run.al_id_needs_scan(7) is False

    def test_dirty_id_needs_scan_despite_matching_cache(self) -> None:
        # An arr-side file change re-warms the id even when the cache matches.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=cache)
        run.mark_dirty([7])
        assert run.al_id_needs_scan(7) is True

    def test_dirty_id_without_seadex_entry_still_skips(self) -> None:
        # The no-entry short-circuit stays first: dirty or not, no entry -> no scan.
        run = self._run(entry=None, cache=FakeCacheStore())
        run.mark_dirty([7])
        assert run.al_id_needs_scan(7) is False

    @staticmethod
    def _marked_cache() -> FakeCacheStore:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1), "fallback_satisfied": True})
        return cache

    def test_warn_mode_fallback_marker_needs_scan(self) -> None:
        # Prefetch must agree with cached_entry_skip's warn-mode resurfacing of
        # fallback-satisfied entries, or the reprocessed id goes un-warmed.
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=self._marked_cache(), private_releases="warn")
        assert run.al_id_needs_scan(7) is True

    def test_fallback_mode_fallback_marker_does_not_need_scan(self) -> None:
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=self._marked_cache(), private_releases="fallback")
        assert run.al_id_needs_scan(7) is False


class TestPrefetchSkipsUnchanged:
    """``prefetch_episodes`` warms only series with at least one scannable id, so a
    series whose every SeaDex entry is unchanged (or absent) is no longer fetched -
    the regression this change fixes."""

    def _eps(self, *, needs_scan: set[int]) -> tuple[SonarrEpisodes, _Sonarr]:
        # Each series maps to a single al_id equal to its id, so a series is warmed
        # iff that id is in ``needs_scan``.
        sonarr = _Sonarr()
        services = _Services(identity=True, needs_scan=needs_scan)
        eps = make_sonarr_episodes(
            sonarr=sonarr,
            _services=services,
            _config=make_config(sleep_time=0),
        )
        return eps, sonarr

    def test_skips_series_with_no_scannable_id(self) -> None:
        eps, sonarr = self._eps(needs_scan={1})
        assert eps.prefetch([_item(1), _item(2)]) == 1
        assert eps._ep_list_cache == {1: _eps_for(1)}  # series 2 never fetched
        assert sonarr.calls == [(1, True)]

    def test_warms_none_when_all_unchanged(self) -> None:
        eps, sonarr = self._eps(needs_scan=set())
        assert eps.prefetch([_item(1), _item(2)]) == 0
        assert eps._ep_list_cache == {}
        assert sonarr.calls == []

    def test_warms_series_with_any_scannable_id(self) -> None:
        # A series whose mapping carries a stale id alongside a fresh one is warmed.
        sonarr = _Sonarr()
        services = _Services(mapping={5: _ids(10, 11)}, needs_scan={11})
        eps = make_sonarr_episodes(
            sonarr=sonarr,
            _services=services,
            _config=make_config(sleep_time=0),
        )
        assert eps.prefetch([_item(5)]) == 1
        assert eps._ep_list_cache == {5: _eps_for(5)}


class _ParseSonarr:
    """A scripted Sonarr ``/parse`` client recording each parsed filename.

    ``parse_episodes_from_seadex`` only touches ``sonarr.parse``; this scripts the
    one result and records the calls so the not-parsed assertions read recorded
    state.
    """

    def __init__(self, result: list[dict[str, int]] | None) -> None:
        self._result = result
        self.calls: list[str] = []

    def parse(self, filename: str) -> list[dict[str, int]] | None:
        self.calls.append(filename)
        return self._result


class TestParseEpisodesNegativeCache:
    def _parse(
        self,
        *,
        parse_result: list[dict[str, int]] | None,
        sleep_time: int = 0,
        sonarr_parse: dict[str, dict[str, object]] | None = None,
    ) -> tuple[SonarrParseCache, _ParseSonarr]:
        sonarr = _ParseSonarr(parse_result)
        parse = make_sonarr_parse(
            sonarr=sonarr,
            _config=make_config(sleep_time=sleep_time),
            cache_store=FakeCacheStore(sonarr_parse=sonarr_parse or {}),
            logger=make_logger(),
        )
        return parse, sonarr

    @staticmethod
    def _dict(*files: str) -> SeadexDict:
        return {"GroupA": rg_group({"u": url_item(files=list(files), size=[100] * len(files))})}

    def test_genuine_empty_is_negative_cached_with_fp(self) -> None:
        parse, _ = self._parse(parse_result=[])
        parse.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"), series_fp="fp")
        rec = parse.cache_store.get_sonarr_parse("[X] Show - 01.mkv")
        assert rec is not None
        assert rec["episodes"] == []
        assert rec["series_fp"] == "fp"

    def test_transient_none_is_not_cached(self) -> None:
        parse, _ = self._parse(parse_result=None)
        parse.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"), series_fp="fp")
        assert parse.cache_store.get_sonarr_parse("[X] Show - 01.mkv") is None

    def test_fresh_negative_hit_skips_network(self) -> None:
        seeded: dict[str, dict[str, object]] = {
            "[X] Show - 01.mkv": {
                "fetched_at": datetime.now().strftime(UPDATED_AT_STR_FORMAT),
                "episodes": [],
                "series_fp": "fp",
            },
        }
        parse, sonarr = self._parse(parse_result=[], sonarr_parse=seeded)
        parse.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv"), series_fp="fp")
        assert sonarr.calls == []

    def test_audio_file_never_parsed(self) -> None:
        parse, sonarr = self._parse(parse_result=[])
        parse.parse_episodes_from_seadex(self._dict("[X] OST - 01.flac"), series_fp="fp")
        assert sonarr.calls == []

    def test_concurrent_pass_negative_caches_each_file(self) -> None:
        parse, _ = self._parse(parse_result=[], sleep_time=0)
        parse.parse_episodes_from_seadex(self._dict("[X] Show - 01.mkv", "[X] Show - 02.mkv"), series_fp="fp")
        for name in ("[X] Show - 01.mkv", "[X] Show - 02.mkv"):
            rec = parse.cache_store.get_sonarr_parse(name)
            assert rec is not None
            assert rec["episodes"] == []
            assert rec["series_fp"] == "fp"
