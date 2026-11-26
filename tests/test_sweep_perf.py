# pyright: strict
# pyright: reportPrivateUsage=false
# These set the episode collaborator's private config (eps._config).
# Strict re-flags that and the repo disables reportPrivateUsage for tests.
"""Tests for the Sonarr sweep speedups.

Covers the parse cache rows (freshness, the leaf's reads and writes), the
series-id hash, worker gating, and concurrent fresh episode prefetch.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import override

import pytest
from seadex import EntryRecord

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_http import SONARR_FETCH_WORKERS, fetch_workers
from pearlarr.arr_models import (
    AlternateTitle,
    MatchedEpisode,
    ParsedFileInfo,
    SeasonTypeCount,
    SonarrEpisode,
    SonarrSeries,
)
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import Arr
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import ExternalIds, TitleMapping
from pearlarr.parse_cache import (
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_NO_EPISODES_TTL_DAYS,
    ParseCacheRows,
    ParseFreshness,
    parse_to_json,
)
from pearlarr.releases import SeadexDict
from pearlarr.seadex_gateway import SeaDexNoEntry
from pearlarr.series_reads import SeriesReads
from pearlarr.sonarr_episodes import SonarrEpisodes, sonarr_series_hash
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.stamps import stamp_of
from pearlarr.tvdb_orders import AIRED_ORDER
from pearlarr.video_files import is_video_candidate

from .builders import (
    FakeCacheStore,
    held_series_episodes,
    make_config,
    make_entry_record,
    make_release,
    make_run_deps,
    make_series_reads,
    make_services,
    make_sonarr_episodes,
    rg_group,
    sonarr_ep,
)
from .fakes import FakeSonarrClient, RecordingProgress

_NAME = "[X] Show - 01.mkv"
_OTHER_NAME = "[X] Show - 02.mkv"

# The parse Sonarr returns when it matched no series at all. It has no matched
# episodes, so its row is pinned to the series hash on write.
_NO_SERIES = ParsedFileInfo()


def _matched(season: int, episode: int) -> ParsedFileInfo:
    """A parse Sonarr resolved to one series episode, its episode id left out."""

    return ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),))


def _of_series(series_id: int) -> ParsedFileInfo:
    """A parse Sonarr matched to the first episode of the series with that id."""

    return ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
        series_title="Show",
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=11),),
        series_id=series_id,
    )


def _stamp(days_ago: int) -> str:
    """A row stamp `days_ago` days before now.

    A run's window opens on the wall clock (`ParseFreshness.open`), so each stamp sits a day or more off a cutoff.
    """

    return stamp_of(datetime.now() - timedelta(days=days_ago))


def _window(series_hash: str = "fp", *, orders: Mapping[int, str] | None = None) -> ParseFreshness:
    """A run's parse freshness, opened now under the series hash `series_hash`.

    Each series of `orders` is numbered by its order there, and every other series as aired.
    """

    return ParseFreshness.open(series_hash, orders or {})


def _record(info: ParsedFileInfo, *, days_ago: int = 1, series_hash: str | None = None) -> dict[str, object]:
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


class TestSeriesHash:
    """`sonarr_series_hash` differs across different id sets, whatever order the library lists them in."""

    def test_the_listing_order_does_not_change_it(self) -> None:
        assert sonarr_series_hash([3, 1, 2]) == sonarr_series_hash([1, 2, 3])

    def test_different_sets_differ(self) -> None:
        # Sets of one size, so a hash of how many series there are can't pass.
        assert sonarr_series_hash([1, 2]) != sonarr_series_hash([1, 3])


class TestParseFreshness:
    """A stored parse with matched episodes is read for the 30-day TTL.

    One with none (pinned with `series_fp`) is read only under the same series hash and the short backstop. A legacy
    row carrying no whole parse is never read.
    """

    @pytest.mark.parametrize(
        ("row", "read_back"),
        [
            pytest.param(
                _record(_matched(1, 1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS - 1),
                _matched(1, 1),
                id="matched-inside-the-ttl",
            ),
            pytest.param(
                _record(_matched(1, 1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS + 1), None, id="matched-past-the-ttl"
            ),
            pytest.param(
                _record(_NO_SERIES, days_ago=SONARR_PARSE_NO_EPISODES_TTL_DAYS - 1, series_hash="fp"),
                _NO_SERIES,
                id="no-matched-episode-inside-the-backstop",
            ),
            pytest.param(
                _record(_NO_SERIES, days_ago=SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1, series_hash="fp"),
                None,
                id="no-matched-episode-past-the-backstop",
            ),
            pytest.param(
                _record(_NO_SERIES, series_hash="old"), None, id="no-matched-episode-under-another-series-hash"
            ),
            # Rows older builds stored carry an episode list, not a whole parse: they
            # re-parse once however recently they were stamped.
            pytest.param(
                {"fetched_at": _stamp(1), "episodes": [{"season": 1, "episode": 1}]}, None, id="a-legacy-episode-list"
            ),
        ],
    )
    def test_a_stored_parse_is_read_only_while_fresh(
        self, row: dict[str, object], read_back: ParsedFileInfo | None
    ) -> None:
        rows = ParseCacheRows(FakeCacheStore(sonarr_parse={_NAME: row}))

        assert rows.read(_NAME, window=_window("fp")) == read_back

    def test_a_row_stored_under_one_order_is_stale_once_its_series_is_in_another_of_its_orders(self) -> None:
        rows = ParseCacheRows(FakeCacheStore())

        rows.write(_NAME, _of_series(7), window=_window(orders={7: "dvd"}))

        same_order = rows.read(_NAME, window=_window(orders={7: "dvd"}))
        another_order = rows.read(_NAME, window=_window(orders={7: "altdvd"}))
        assert (same_order, another_order) == (_of_series(7), None)


class TestParseCacheRows:
    """The parse-cache leaf: fresh-only reads and shape-owning writes."""

    @pytest.mark.parametrize(
        ("info", "under_another_hash"),
        [
            pytest.param(_of_series(7), _of_series(7), id="matched"),
            pytest.param(_NO_SERIES, None, id="no-series"),
            pytest.param(_of_series(7).with_matched(()), None, id="a-series-and-no-episode"),
        ],
    )
    def test_a_written_parse_with_no_matched_episode_reads_back_only_under_its_series_hash(
        self, info: ParsedFileInfo, under_another_hash: ParsedFileInfo | None
    ) -> None:
        rows = ParseCacheRows(FakeCacheStore())

        rows.write(_NAME, info, window=_window("fp"))

        same_hash = rows.read(_NAME, window=_window("fp"))
        another_hash = rows.read(_NAME, window=_window("other"))
        assert (same_hash, another_hash) == (info, under_another_hash)

    def test_unreadable_parse_reads_as_a_miss(self) -> None:
        # A row whose values this build cannot validate re-parses: every field is present, so the key check passes
        # it and validation is what misses.
        row: dict[str, object] = {
            "fetched_at": _stamp(1),
            "parse": {**parse_to_json(_matched(1, 1)), "episode_numbers": "one"},
        }
        store = FakeCacheStore(sonarr_parse={_NAME: row})

        assert ParseCacheRows(store).read(_NAME, window=_window()) is None

    def test_a_parse_without_a_series_title_is_stale_for_the_warm_check_and_the_read(self) -> None:
        # The model would read the missing title as "", so the key decides, inside the TTL.
        store = FakeCacheStore(sonarr_parse={_NAME: _untitled(_matched(1, 1))})
        rows = ParseCacheRows(store)

        assert not rows.is_fresh(_NAME, window=_window())
        assert rows.read(_NAME, window=_window()) is None


@pytest.mark.parametrize(
    ("sleep_time", "workers"),
    [pytest.param(0, SONARR_FETCH_WORKERS, id="unthrottled"), pytest.param(1, 1, id="throttled")],
)
def test_fetch_workers_are_one_under_a_sleep_throttle(sleep_time: int, workers: int) -> None:
    assert fetch_workers(make_config(sleep_time=sleep_time)) == workers


class _Series:
    """A fake Sonarr series satisfying the `SonarrItem` protocol surface."""

    id: int
    title: str = ""
    imdbId: str | None
    monitored: bool
    tvdbId: int
    tmdbId: int
    seasonType: str = AIRED_ORDER
    seasonTypes: tuple[SeasonTypeCount, ...] = ()
    alternateTitles: tuple[AlternateTitle, ...] = ()
    totalEpisodeCount: int | None = None
    order_sent: bool = False

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


class _Sonarr(FakeSonarrClient):
    """A scripted Sonarr client for the prefetch warm.

    By default `episodes(sid)` returns a distinguishable one-episode list per
    series (`_eps_for`). `return_none` turns every fetch into a transient
    failure, and `raise_on` makes the listed ids raise (the worker-failure
    case). Records each `(series_id, quiet)` call so the dedup / not-fetched /
    quiet assertions read recorded state.
    """

    def __init__(self, *, return_none: bool = False, raise_on: set[int] | None = None) -> None:
        super().__init__()
        self.return_none = return_none
        self.raise_on: set[int] = set(raise_on or set())
        self.calls: list[tuple[int, bool]] = []

    @override
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
    """A fake of the arr run services the prefetch consults.

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
        assert held_series_episodes(eps) == {1: _eps_for(1), 2: _eps_for(2)}

    def test_leaves_out_unmonitored_when_ignored(self) -> None:
        eps, _ = self._eps(mapped={1, 2})
        eps._config = make_config(sleep_time=0, ignore_unmonitored=True)
        eps.prefetch([_item(1, monitored=False), _item(2)])
        assert held_series_episodes(eps) == {2: _eps_for(2)}

    def test_dedups_series_ids(self) -> None:
        eps, sonarr = self._eps(mapped={1})
        eps.prefetch([_item(1), _item(1)])
        assert len(sonarr.calls) == 1

    def test_none_result_not_cached(self) -> None:
        # series 9 is a candidate (resolves a mapping) but episodes() returns None.
        eps, sonarr = self._eps(mapped={9})
        sonarr.return_none = True
        eps.prefetch([_item(9)])
        assert held_series_episodes(eps) == {}

    def test_raising_series_does_not_abort_sweep(self) -> None:
        # CB5: a worker that RAISES (e.g. a non-JSON 200 response) must not abort the
        # whole concurrent sweep. That series is left unwarmed, the rest still warm.
        eps, sonarr = self._eps(mapped={1, 2})
        sonarr.raise_on = {1}
        warmed = eps.prefetch([_item(1), _item(2)])

        assert held_series_episodes(eps) == {2: _eps_for(2)}  # 1 raised -> unwarmed, 2 warmed
        assert warmed == 2  # both attempted

    def test_sequential_path_matches_concurrent(self) -> None:
        eps, _ = self._eps(mapped={1, 2}, sleep_time=2)
        eps.prefetch([_item(1), _item(2)])
        assert held_series_episodes(eps) == {1: _eps_for(1), 2: _eps_for(2)}

    def test_returns_warmed_count(self) -> None:
        # Only mapped, monitored series are warmed: 3 is unmapped, so 2 warmed.
        eps, _ = self._eps(mapped={1, 2})
        assert eps.prefetch([_item(1), _item(2), _item(3)]) == 2

    def test_drives_progress_per_series(self) -> None:
        eps, _ = self._eps(mapped={1, 2})
        rec = RecordingProgress()
        eps.prefetch([_item(1), _item(2), _item(3)], progress=rec)
        # One drive per warmed series, ending complete. Completion order is
        # nondeterministic, so assert on the count + the final value, not the
        # intermediate sequence.
        assert len(rec.updates) == 2
        assert rec.updates[-1] == (1.0, "2/2")

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
        rec = RecordingProgress()
        assert eps.prefetch([_item(9)], progress=rec) == 1
        assert held_series_episodes(eps) == {}  # nothing cached
        assert rec.updates[-1] == (1.0, "1/1")  # but progress still completed


class TestAiredSeriesRead:
    """`aired_series` fetches a cold series once per run and reports a failed fetch as `NoView.READ_FAILED`."""

    @staticmethod
    def _reads(*, return_none: bool = False) -> tuple[SeriesReads, _Sonarr]:
        sonarr = _Sonarr(return_none=return_none)
        return make_series_reads(sonarr), sonarr

    def test_second_call_serves_the_run_cache(self) -> None:
        reads, sonarr = self._reads()

        first = reads.aired_series(1)
        assert isinstance(first, AiredSeries)
        assert first.in_sonarr == tuple(_eps_for(1))
        assert reads.aired_series(1) is first
        assert sonarr.calls == [(1, False)]

    def test_failed_fetch_is_no_view_and_not_cached(self) -> None:
        # A failed read is the unreadable list, distinct from a series with no episodes:
        # every verdict that depends on the episode index places nothing on it instead of reading empty.
        reads, _ = self._reads(return_none=True)

        assert reads.aired_series(1) is NoView.READ_FAILED
        assert reads.series_index(1) is None
        assert reads.last_known_episodes(1) == []


def _entry(dt: datetime) -> EntryRecord:
    """A real SeaDex entry stamped at `dt` (only `updated_at` is read)."""

    return make_entry_record(updated_at=dt)


class _Seadex:
    """A fake SeaDex gateway returning one fixed entry (or a not-found) for any anilist_id."""

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
        assert held_series_episodes(eps) == {1: _eps_for(1)}  # series 2 never fetched
        assert sonarr.calls == [(1, True)]

    def test_warms_none_when_all_unchanged(self) -> None:
        eps, sonarr = self._eps(needs_scan=set())
        assert eps.prefetch([_item(1), _item(2)]) == 0
        assert held_series_episodes(eps) == {}
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
        assert held_series_episodes(eps) == {5: _eps_for(5)}


_LIBRARY = (SonarrSeries(id=1),)
_HASH = sonarr_series_hash([1])
"""The series hash of a run over `_LIBRARY`."""


def _parse_cache(sonarr: FakeSonarrClient, cache: FakeCacheStore) -> SonarrParseCache:
    """A run's parse cache over the stored parses in `cache`, its library `_LIBRARY`, built as a run builds it."""

    return SonarrParseCache(make_run_deps(cache_store=cache), sonarr, make_series_reads(sonarr, library=_LIBRARY))


def _listing(files: Mapping[str, int]) -> SeadexDict:
    """One group's release `u`, listing each of `files` with its size."""

    return {"GroupA": rg_group({"u": make_release(files=list(files), size=list(files.values()))})}


class TestParsedFilesCache:
    """`parsed_files` asks Sonarr once for each name, then reads its stored answer while it is fresh.

    It never stores a failed answer, and it parses only the video files.
    """

    @pytest.mark.parametrize(
        ("answer", "asked"),
        [
            pytest.param(_matched(1, 1), [_NAME], id="matched"),
            pytest.param(_NO_SERIES, [_NAME], id="no-matched-episode"),
            pytest.param(None, [_NAME, _NAME], id="no-answer"),
        ],
    )
    def test_a_later_run_asks_sonarr_again_only_after_no_answer(
        self, answer: ParsedFileInfo | None, asked: list[str]
    ) -> None:
        cache = FakeCacheStore()
        sonarr = FakeSonarrClient(parse_fn=lambda _name: answer)

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100}))
        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100}))

        assert sonarr.parse_calls == asked

    def test_a_stored_parse_with_no_matched_episodes_is_asked_again_sooner(self) -> None:
        # The same age for both: past the backstop of a parse with no matched episodes, inside the TTL of one with.
        days_ago = SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1
        cache = FakeCacheStore(
            sonarr_parse={
                _NAME: _record(_matched(1, 1), days_ago=days_ago),
                _OTHER_NAME: _record(_NO_SERIES, days_ago=days_ago, series_hash=_HASH),
            }
        )
        sonarr = FakeSonarrClient()

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert sonarr.parse_calls == [_OTHER_NAME]

    def test_parsed_files_pairs_each_video_file_with_its_size_and_parse(self) -> None:
        # Video files only, each with its own listed size and its parse.
        answers = {_NAME: _matched(1, 1), _OTHER_NAME: _matched(1, 2)}
        sonarr = FakeSonarrClient(parse_fn=answers.get)
        listing = _listing({_NAME: 100, "[X] Show - 01.ass": 1, "[X] OST - 01.flac": 2, _OTHER_NAME: 200})

        gathered = _parse_cache(sonarr, FakeCacheStore()).parsed_files(listing)

        assert gathered == {"u": (ListedFile(_NAME, 100, _matched(1, 1)), ListedFile(_OTHER_NAME, 200, _matched(1, 2)))}

    def test_names_not_yet_stored_go_to_sonarr_at_once(self) -> None:
        # Each answer waits, 30 seconds at most, for the other name to be asked, so asking one at a time fails.
        both_asked = threading.Barrier(2, timeout=30.0)

        def answer(name: str) -> ParsedFileInfo:
            del name
            both_asked.wait()
            return _matched(1, 1)

        sonarr = FakeSonarrClient(parse_fn=answer)

        gathered = _parse_cache(sonarr, FakeCacheStore()).parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert gathered == {"u": (ListedFile(_NAME, 100, _matched(1, 1)), ListedFile(_OTHER_NAME, 100, _matched(1, 1)))}


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
