# pyright: strict
# pyright: reportPrivateUsage=false
# These read the episode collaborator's private per-run state (eps._ep_list_cache /
# eps._config) and call the module-private _parse_is_fresh / SonarrParseCache._parse_for.
# Strict re-flags that and the repo disables reportPrivateUsage for tests.
"""Tests for the Sonarr sweep speedups.

Covers the parse cache rows (freshness, the leaf's reads and writes), the
series-id hash, worker gating, and concurrent fresh episode prefetch.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import override

import pytest
from seadex import EntryRecord

from pearlarr.arr_models import AlternateTitle, MatchedEpisode, ParsedFileInfo, SeasonTypeCount, SonarrEpisode
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import Arr
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import ExternalIds, TitleMapping
from pearlarr.parse_cache import (
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_UNMATCHED_TTL_DAYS,
    ParseCacheRows,
    ParseFreshness,
    _parse_is_fresh,
    parse_to_json,
)
from pearlarr.releases import SeadexDict
from pearlarr.seadex_gateway import SeaDexNoEntry
from pearlarr.sonarr_episodes import (
    SONARR_FETCH_WORKERS,
    SonarrEpisodes,
    fetch_workers,
    sonarr_series_hash,
)
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.stamps import STAMP_FORMAT
from pearlarr.tvdb_orders import AIRED_ORDER
from pearlarr.video_files import is_video_candidate

from .builders import (
    FakeCacheStore,
    make_config,
    make_entry_record,
    make_logger,
    make_release,
    make_services,
    make_sonarr_episodes,
    make_sonarr_parse,
    rg_group,
    sonarr_ep,
)

_NOW = datetime(2026, 6, 28, 12, 0, 0)
_MATCHED_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS)
_UNMATCHED_CUTOFF = _NOW - timedelta(days=SONARR_PARSE_UNMATCHED_TTL_DAYS)

# The parse Sonarr returns when it matched no series at all: the cacheable
# unmatched row, pinned to the series hash on write.
_UNMATCHED = ParsedFileInfo()


def _matched(season: int, episode: int) -> ParsedFileInfo:
    """A parse Sonarr resolved to one series episode (ids omitted, as a cached row reads back)."""

    return ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),))


def _stamp(days_ago: float) -> str:
    return (_NOW - timedelta(days=days_ago)).strftime(STAMP_FORMAT)


def _window(series_hash: str = "fp") -> ParseFreshness:
    """The parse freshness anchored at `_NOW` (never the wall clock)."""

    return ParseFreshness(
        now_str=_NOW.strftime(STAMP_FORMAT),
        matched_cutoff=_MATCHED_CUTOFF,
        unmatched_cutoff=_UNMATCHED_CUTOFF,
        series_hash=series_hash,
    )


def _record(info: ParsedFileInfo, *, days_ago: float = 1, series_hash: str | None = None) -> dict[str, object]:
    """One persisted parse row, stamped `days_ago` and optionally pinned to a series hash."""

    row: dict[str, object] = {"fetched_at": _stamp(days_ago), "parse": parse_to_json(info)}
    if series_hash is not None:
        row["series_fp"] = series_hash
    return row


def _untitled(info: ParsedFileInfo) -> dict[str, object]:
    """A row stamped a day ago whose parse has no `series_title` key, as older builds wrote it."""

    parse = parse_to_json(info)
    del parse["series_title"]
    return {"fetched_at": _stamp(1), "parse": parse}


def _fresh(row: dict[str, object], *, series_hash: str = "fp") -> bool:
    return _parse_is_fresh(row, window=_window(series_hash))


class TestSeriesHash:
    """`sonarr_series_hash` is order- and duplicate-independent and differs across different id sets.

    It is stable for the empty set.
    """

    def test_order_and_duplicate_independent(self) -> None:
        assert sonarr_series_hash([3, 1, 2]) == sonarr_series_hash([2, 2, 1, 3])

    def test_different_sets_differ(self) -> None:
        assert sonarr_series_hash([1, 2]) != sonarr_series_hash([1, 2, 3])

    def test_empty_is_stable(self) -> None:
        assert sonarr_series_hash([]) == sonarr_series_hash(iter(()))


class TestParseIsFresh:
    """`_parse_is_fresh` dispatches on the series hash key: a matched row gets the 30-day TTL.

    An unmatched one (pinned with `series_fp`) is fresh only under the same
    series hash and the short backstop. A legacy row carrying no whole parse is
    never fresh.
    """

    def test_matched_within_ttl_is_fresh(self) -> None:
        assert _fresh(_record(_matched(1, 1), days_ago=5))

    def test_matched_beyond_ttl_is_stale(self) -> None:
        assert not _fresh(_record(_matched(1, 1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS + 1))

    def test_unmatched_fresh_when_series_hash_matches_and_within_backstop(self) -> None:
        assert _fresh(_record(_UNMATCHED, days_ago=2, series_hash="fp"), series_hash="fp")

    def test_unmatched_stale_on_series_hash_mismatch(self) -> None:
        assert not _fresh(_record(_UNMATCHED, days_ago=2, series_hash="old"), series_hash="fp")

    def test_unmatched_stale_beyond_backstop_ttl(self) -> None:
        aged = _record(_UNMATCHED, days_ago=SONARR_PARSE_UNMATCHED_TTL_DAYS + 1, series_hash="fp")
        assert not _fresh(aged, series_hash="fp")

    def test_unmatched_uses_the_backstop_not_the_matched_ttl(self) -> None:
        # Inside the 30-day matched TTL, past the 7-day backstop: the pin decides.
        assert not _fresh(_record(_UNMATCHED, days_ago=10, series_hash="fp"), series_hash="fp")

    def test_legacy_row_without_a_parse_is_stale_inside_the_ttl(self) -> None:
        # The pre-reshape rows carry an episode list, not a whole parse: they
        # re-parse once however recently they were stamped.
        assert not _fresh({"fetched_at": _stamp(1), "episodes": [{"season": 1, "episode": 1}]})


class TestParseCacheRows:
    """The parse-cache leaf: fresh-only reads and shape-owning writes."""

    @staticmethod
    def _parse_cache_rows() -> tuple[ParseCacheRows, FakeCacheStore]:
        store = FakeCacheStore()
        return ParseCacheRows(store), store

    def test_round_trip_keeps_the_matched_ids(self) -> None:
        rows, _ = self._parse_cache_rows()
        info = ParsedFileInfo(
            season_number=1,
            episode_numbers=(1,),
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=8476),),
        )

        rows.write("show - 01.mkv", info, window=_window())

        assert rows.read("show - 01.mkv", window=_window()) == info

    def test_unmatched_write_is_pinned_to_the_series_hash(self) -> None:
        rows, store = self._parse_cache_rows()

        rows.write("show - 01.mkv", _UNMATCHED, window=_window("fp"))

        row = store.get_sonarr_parse("show - 01.mkv")
        assert row is not None
        assert row["series_fp"] == "fp"
        assert rows.read("show - 01.mkv", window=_window("other")) is None

    def test_matched_write_carries_no_series_hash(self) -> None:
        rows, store = self._parse_cache_rows()

        rows.write("show - 01.mkv", _matched(1, 1), window=_window("fp"))

        row = store.get_sonarr_parse("show - 01.mkv")
        assert row is not None
        assert "series_fp" not in row

    def test_unreadable_parse_reads_as_a_miss(self) -> None:
        # A row whose values this build cannot validate re-parses: every field is present, so the key check passes
        # it and validation is what misses.
        row: dict[str, object] = {
            "fetched_at": _stamp(1),
            "parse": {**parse_to_json(_matched(1, 1)), "episode_numbers": "one"},
        }
        store = FakeCacheStore(sonarr_parse={"show - 01.mkv": row})

        assert ParseCacheRows(store).read("show - 01.mkv", window=_window()) is None

    def test_a_parse_without_a_series_title_is_stale_for_the_warm_check_and_the_read(self) -> None:
        # The model would read the missing title as "", so the key decides, inside the TTL.
        store = FakeCacheStore(sonarr_parse={"show - 01.mkv": _untitled(_matched(1, 1))})
        rows = ParseCacheRows(store)

        assert not rows.is_fresh("show - 01.mkv", window=_window())
        assert rows.read("show - 01.mkv", window=_window()) is None


class TestFetchWorkers:
    """`fetch_workers` returns the concurrent worker count when unthrottled, else 1 when sleep-throttled."""

    def test_concurrent_when_no_sleep(self) -> None:
        assert fetch_workers(make_config(sleep_time=0)) == SONARR_FETCH_WORKERS

    def test_sequential_when_throttled(self) -> None:
        assert fetch_workers(make_config(sleep_time=2)) == 1


class _Series:
    """A stand-in Sonarr series satisfying the `SonarrItem` protocol surface."""

    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tvdbId: int
    tmdbId: int
    seasonType: str = AIRED_ORDER
    seasonTypes: tuple[SeasonTypeCount, ...] = ()
    alternateTitles: tuple[AlternateTitle, ...] = ()

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


def _ids(*anilist_ids: int) -> dict[int, TitleMapping]:
    """A `{anilist_id -> mapping}` dict. Only the keys are read by the prefetch gate."""

    return {aid: TitleMapping(anilist_id=aid) for aid in anilist_ids}


class _Recorder:
    """A `ProgressCallback` that records every `progress` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        self.calls.append((fraction, detail))


class _Sonarr:
    """A scripted Sonarr client for the prefetch warm.

    By default `episodes(sid)` returns a distinguishable one-episode list per
    series (`_eps_for`). `return_none` turns every fetch into a transient
    failure, and `raise_on` makes the listed ids raise (the worker-failure
    case). Records each `(series_id, quiet)` call so the dedup / not-fetched /
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


class _GatedSonarr(_Sonarr):
    """A `_Sonarr` whose fetches after the first block until `gate` is set, so a single worker can't run ahead."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        if self.calls:
            assert self.gate.wait(30.0), "gate never opened"
        return super().episodes(series_id, quiet=quiet)


class _StopOnFirstProgress:
    """A `ProgressCallback` that opens the fetch gate, then raises `SystemExit` as the SIGTERM handler would."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def progress(self, fraction: float, detail: str | None = None) -> None:
        del fraction, detail
        self._gate.set()
        raise SystemExit(0)


class _Services:
    """A stand-in for the arr run services the prefetch consults.

    `title_mappings` resolves a series' tvdb id to its `{anilist_id -> mapping}`
    dict (`identity` returns `{tvdb_id: mapping}` for any series, mirroring the
    always-mapped helper). `title_needs_check` is the per-title needs-check gate
    (`needs_scan=None` reports every title as needing a check).
    """

    def __init__(
        self,
        *,
        mapping: dict[int, dict[int, TitleMapping]] | None = None,
        identity: bool = False,
        needs_scan: set[int] | None = None,
    ) -> None:
        self._mapping = mapping or {}
        self._identity = identity
        self._needs_scan = needs_scan

    def title_mappings(
        self,
        ids: ExternalIds,
        log_ignored: bool = True,
    ) -> dict[int, TitleMapping]:
        del log_ignored
        assert ids.tvdb is not None  # the prefetch always keys on the series' tvdb id
        if self._identity:
            return {ids.tvdb: TitleMapping(anilist_id=ids.tvdb)}
        return self._mapping.get(ids.tvdb, {})

    def title_needs_check(self, anilist_id: int) -> bool:
        if self._needs_scan is None:
            return True
        return anilist_id in self._needs_scan


class TestPrefetchEpisodes:
    """`SonarrEpisodes.prefetch` warms only mapped, monitored series with a title to check, deduped by id.

    It tolerates a raising or `None`-returning worker (that series stays
    unwarmed, the rest still warm) and reports the attempted (not cached)
    count via progress.
    """

    def _eps(self, *, mapped: set[int], sleep_time: int = 0) -> tuple[SonarrEpisodes, _Sonarr]:
        sonarr = _Sonarr()
        # Only "mapped" series resolve to a non-empty AniList mapping. needs_scan
        # defaults to "every title needs a check".
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

    def test_leaves_out_unmonitored_when_ignored(self) -> None:
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
        # whole concurrent sweep. That series is left unwarmed, the rest still warm.
        eps, sonarr = self._eps(mapped={1, 2})
        sonarr.raise_on = {1}
        warmed = eps.prefetch([_item(1), _item(2)])

        assert eps._ep_list_cache == {2: _eps_for(2)}  # 1 raised -> unwarmed, 2 warmed
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

    def test_a_stop_mid_sweep_drops_the_fetches_not_yet_started(self) -> None:
        sonarr = _GatedSonarr()
        series = range(1, 21)
        eps = make_sonarr_episodes(
            sonarr=sonarr,
            _services=_Services(mapping={sid: _ids(1) for sid in series}),
            _config=make_config(sleep_time=2),  # one worker
        )

        with pytest.raises(SystemExit):
            eps.prefetch([_item(sid) for sid in series], progress=_StopOnFirstProgress(sonarr.gate))

        # The first fetch, plus at most the one the worker had in hand.
        assert len(sonarr.calls) <= 2

    def test_count_is_attempted_not_cached(self) -> None:
        # series 9 is a candidate (resolves a mapping) but episodes() returns None.
        # It's still attempted, so it counts toward the return value + the bar.
        eps, sonarr = self._eps(mapped={9})
        sonarr.return_none = True
        rec = _Recorder()
        assert eps.prefetch([_item(9)], progress=rec) == 1
        assert eps._ep_list_cache == {}  # nothing cached
        assert rec.calls[-1] == (1.0, "1/1")  # but progress still completed


class TestCachedEpisodes:
    """`cached_episodes` fetches a cold series once per run and reports a failed fetch as None."""

    @staticmethod
    def _eps(*, return_none: bool = False) -> tuple[SonarrEpisodes, _Sonarr]:
        sonarr = _Sonarr(return_none=return_none)
        return make_sonarr_episodes(sonarr=sonarr, _config=make_config(sleep_time=0)), sonarr

    def test_second_call_serves_the_run_cache(self) -> None:
        eps, sonarr = self._eps()

        assert eps.cached_episodes(1) == _eps_for(1)
        assert eps.cached_episodes(1) == _eps_for(1)
        assert sonarr.calls == [(1, False)]

    def test_failed_fetch_is_none_and_not_cached(self) -> None:
        # None is the unreadable list, distinct from a series with no episodes:
        # every verdict that depends on the episode index places nothing on it instead of reading empty.
        eps, _ = self._eps(return_none=True)

        assert eps.cached_episodes(1) is None
        assert eps._ep_list_cache == {}


def _entry(dt: datetime) -> EntryRecord:
    """A real SeaDex entry stamped at `dt` (only `updated_at` is read)."""

    return make_entry_record(updated_at=dt)


class _Seadex:
    """A SeaDex gateway stand-in returning one fixed entry (or a not-found) for any anilist_id."""

    def __init__(self, entry: EntryRecord | None) -> None:
        self._entry = entry

    def entry(self, anilist_id: int) -> EntryRecord | SeaDexNoEntry:
        del anilist_id
        return self._entry if self._entry is not None else SeaDexNoEntry.NOT_FOUND


class TestTitleNeedsCheck:
    """`ArrRunServices.title_needs_check` mirrors the per-title loop's no-entry + `is_unchanged` gates.

    It is side-effect-free, so `prefetch_episodes` warms only the series the loop would actually
    check (the SeaDex-modification-times fix). Pinned against the same
    cases as `is_unchanged`.
    """

    @staticmethod
    def _run(*, entry: EntryRecord | None, cache: FakeCacheStore, **cfg: object) -> ArrRunServices:
        return make_services(_seadex=_Seadex(entry), cache_store=cache, **cfg)

    def test_no_seadex_entry_does_not_need_check(self) -> None:
        run = self._run(entry=None, cache=FakeCacheStore())
        assert run.title_needs_check(7) is False

    def test_uncached_title_needs_check(self) -> None:
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=FakeCacheStore())
        assert run.title_needs_check(7) is True

    def test_cached_and_matching_does_not_need_check(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=cache)
        assert run.title_needs_check(7) is False

    def test_cached_but_changed_needs_check(self) -> None:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2022, 6, 6)), cache=cache)
        assert run.title_needs_check(7) is True

    def test_ignore_update_times_forces_a_check_when_entry_exists(self) -> None:
        # A matching cached title is normally left unchecked. ignore_seadex_update_times
        # makes the loop re-check it, so the predicate must report needs-check.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(
            entry=_entry(datetime(2021, 1, 1)),
            cache=cache,
            ignore_seadex_update_times=True,
        )
        assert run.title_needs_check(7) is True

    def test_ignore_update_times_still_needs_no_check_when_no_entry(self) -> None:
        # No SeaDex entry -> begin_check would stop regardless of the flag.
        run = self._run(entry=None, cache=FakeCacheStore(), ignore_seadex_update_times=True)
        assert run.title_needs_check(7) is False

    def test_dirty_id_needs_check_despite_matching_cache(self) -> None:
        # An arr-side file change re-warms the id even when the cache matches.
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=cache)
        run.mark_dirty([7])
        assert run.title_needs_check(7) is True

    def test_dirty_id_without_seadex_entry_still_needs_no_check(self) -> None:
        # The no-entry short-circuit stays first: dirty or not, no entry -> no check.
        run = self._run(entry=None, cache=FakeCacheStore())
        run.mark_dirty([7])
        assert run.title_needs_check(7) is False

    @staticmethod
    def _marked_cache() -> FakeCacheStore:
        cache = FakeCacheStore()
        cache.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1), "fallback_satisfied": True})
        return cache

    def test_warn_mode_fallback_marker_needs_check(self) -> None:
        # Prefetch must agree with is_unchanged's warn-mode resurfacing of
        # fallback-satisfied titles, or the re-checked title goes un-warmed.
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=self._marked_cache(), private_releases="warn")
        assert run.title_needs_check(7) is True

    def test_fallback_mode_fallback_marker_does_not_need_check(self) -> None:
        run = self._run(entry=_entry(datetime(2021, 1, 1)), cache=self._marked_cache(), private_releases="fallback")
        assert run.title_needs_check(7) is False


class TestPrefetchLeavesOutUnchanged:
    """`prefetch_episodes` warms only series with at least one title to check.

    A series whose every title is unchanged (or has no entry) is no longer
    fetched, the regression this change fixes.
    """

    def _eps(self, *, needs_scan: set[int]) -> tuple[SonarrEpisodes, _Sonarr]:
        # Each series maps to a single anilist_id equal to its id, so a series is warmed
        # iff that id is in `needs_scan`.
        sonarr = _Sonarr()
        services = _Services(identity=True, needs_scan=needs_scan)
        eps = make_sonarr_episodes(
            sonarr=sonarr,
            _services=services,
            _config=make_config(sleep_time=0),
        )
        return eps, sonarr

    def test_leaves_out_series_with_no_title_to_check(self) -> None:
        eps, sonarr = self._eps(needs_scan={1})
        assert eps.prefetch([_item(1), _item(2)]) == 1
        assert eps._ep_list_cache == {1: _eps_for(1)}  # series 2 never fetched
        assert sonarr.calls == [(1, True)]

    def test_warms_none_when_all_unchanged(self) -> None:
        eps, sonarr = self._eps(needs_scan=set())
        assert eps.prefetch([_item(1), _item(2)]) == 0
        assert eps._ep_list_cache == {}
        assert sonarr.calls == []

    def test_warms_series_with_any_title_to_check(self) -> None:
        # A series whose mapping carries a title needing a check alongside an unchanged one is warmed.
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
    """A scripted Sonarr `/parse` client recording each parsed filename.

    `parsed_files` only touches `sonarr.parse`. This scripts the one result and
    records the calls so the not-parsed assertions read recorded state.
    """

    def __init__(self, result: ParsedFileInfo | None) -> None:
        self._result = result
        self.calls: list[str] = []

    def parse(self, filename: str) -> ParsedFileInfo | None:
        self.calls.append(filename)
        return self._result


class TestParsedFilesUnmatchedCache:
    """`parsed_files` caches a genuine unmatched parse with the series hash.

    It skips a fresh hit's network call, never caches a transient `None`, and
    never parses audio files.
    """

    def _parse(
        self,
        *,
        parse_result: ParsedFileInfo | None,
        sleep_time: int = 0,
        sonarr_parse: dict[str, dict[str, object]] | None = None,
    ) -> tuple[SonarrParseCache, _ParseSonarr]:
        sonarr = _ParseSonarr(parse_result)
        store = FakeCacheStore(sonarr_parse=sonarr_parse or {})
        parse = make_sonarr_parse(
            sonarr=sonarr,
            _config=make_config(sleep_time=sleep_time),
            cache_store=store,
            rows=ParseCacheRows(store),
            logger=make_logger(),
        )
        return parse, sonarr

    @staticmethod
    def _dict(*files: str) -> SeadexDict:
        return {"GroupA": rg_group({"u": make_release(files=list(files), size=[100] * len(files))})}

    @staticmethod
    def _assert_pinned(parse: SonarrParseCache, name: str) -> None:
        rec = parse.cache_store.get_sonarr_parse(name)
        assert rec is not None
        assert rec["parse"] == parse_to_json(_UNMATCHED)
        assert rec["series_fp"] == "fp"

    def test_genuine_unmatched_is_cached_with_series_hash(self) -> None:
        parse, _ = self._parse(parse_result=_UNMATCHED)
        parse.parsed_files(self._dict("[X] Show - 01.mkv"), series_hash="fp")
        self._assert_pinned(parse, "[X] Show - 01.mkv")

    def test_transient_none_is_not_cached(self) -> None:
        parse, _ = self._parse(parse_result=None)
        parse.parsed_files(self._dict("[X] Show - 01.mkv"), series_hash="fp")
        assert parse.cache_store.get_sonarr_parse("[X] Show - 01.mkv") is None

    def test_fresh_unmatched_hit_skips_network(self) -> None:
        file_map: dict[str, dict[str, object]] = {
            "[X] Show - 01.mkv": {
                "fetched_at": datetime.now().strftime(STAMP_FORMAT),
                "parse": parse_to_json(_UNMATCHED),
                "series_fp": "fp",
            },
        }
        parse, sonarr = self._parse(parse_result=_UNMATCHED, sonarr_parse=file_map)
        parse.parsed_files(self._dict("[X] Show - 01.mkv"), series_hash="fp")
        assert sonarr.calls == []

    def test_unreadable_row_is_re_fetched_and_rewritten(self) -> None:
        # A row whose parse this build cannot validate is a miss, so `_parse_for`
        # re-asks Sonarr and overwrites it with a readable row. Every field is
        # present, so the key check passes it and validation is what misses.
        file_map: dict[str, dict[str, object]] = {
            "[X] Show - 01.mkv": {
                "fetched_at": datetime.now().strftime(STAMP_FORMAT),
                "parse": {**parse_to_json(_matched(1, 1)), "episode_numbers": "one"},
            },
        }
        parse, sonarr = self._parse(parse_result=_matched(1, 1), sonarr_parse=file_map)

        info = parse._parse_for("[X] Show - 01.mkv", window=_window())

        assert info == _matched(1, 1)
        assert sonarr.calls == ["[X] Show - 01.mkv"]
        rec = parse.cache_store.get_sonarr_parse("[X] Show - 01.mkv")
        assert rec is not None
        assert rec["parse"] == parse_to_json(_matched(1, 1))

    def test_a_row_without_a_series_title_is_re_parsed_by_the_cache_warm(self) -> None:
        # The cache warm sees it stale too, so it re-parses in the pool with the cold names, not alone afterwards.
        names = ["[X] Show - 01.mkv", "[X] Show - 02.mkv"]
        parse, sonarr = self._parse(parse_result=_matched(1, 1), sonarr_parse={names[0]: _untitled(_matched(1, 1))})

        parse._warm_parse_cache(names, window=_window())

        assert sorted(sonarr.calls) == names
        rec = parse.cache_store.get_sonarr_parse(names[0])
        assert rec is not None
        assert rec["parse"] == parse_to_json(_matched(1, 1))

    def test_audio_file_never_parsed(self) -> None:
        parse, sonarr = self._parse(parse_result=_UNMATCHED)
        parse.parsed_files(self._dict("[X] OST - 01.flac"), series_hash="fp")
        assert sonarr.calls == []

    def test_parsed_files_pairs_each_video_file_with_its_size_and_parse(self) -> None:
        # Every release gets a key, video files only, each with its listed size and its parse.
        parse, _ = self._parse(parse_result=_matched(1, 1))
        gathered = parse.parsed_files(self._dict("[X] Show - 01.mkv", "[X] Show - 01.ass"), series_hash="fp")
        assert gathered == {"u": (ListedFile("[X] Show - 01.mkv", 100, _matched(1, 1)),)}

    def test_concurrent_warm_caches_each_file(self) -> None:
        parse, _ = self._parse(parse_result=_UNMATCHED, sleep_time=0)
        parse.parsed_files(self._dict("[X] Show - 01.mkv", "[X] Show - 02.mkv"), series_hash="fp")
        for name in ("[X] Show - 01.mkv", "[X] Show - 02.mkv"):
            self._assert_pinned(parse, name)


class TestIsVideoCandidate:
    """Checksum and sidecar files never enter the candidate pool.

    A seeded checksum file is never offered among Sonarr's ManualImport candidates,
    so it would stick a pending import on "intended file missing" until the deadline.
    All of these extensions were observed in real SeaDex releases.
    """

    def test_checksum_and_sidecar_extensions_rejected(self) -> None:
        for name in (
            "[Marin] Show - S01 [BD 1080p].blake3",
            "checksum.md5",
            "checksum.sha2",
            "checksum.sha256",
            "[Dae] Show - S01E01.mks",
            "Animate Benefits - 01.tif",
            "playlist.m3u8",
        ):
            assert not is_video_candidate(name)

    def test_bundled_reading_matter_rejected(self) -> None:
        # Manga volumes and books sit beside the video in some releases.
        for name in (
            "[X] Show - Volume 01.cbz",
            "[X] Show - Volume 02.cbr",
            "[X] Show - Volume 03.cb7",
            "[X] Show - Guidebook.pdf",
            "[X] Show - Novel 01.epub",
        ):
            assert not is_video_candidate(name)

    def test_video_containers_kept(self) -> None:
        assert is_video_candidate("[X] Show - S01E01.mkv")
        assert is_video_candidate("[X] Show - S01E01.mp4")
