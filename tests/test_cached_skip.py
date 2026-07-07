# pyright: strict
# pyright: reportPrivateUsage=false
# These read the hub's private run context (run._ctx); strict re-flags that and
# the repo disables reportPrivateUsage for tests.
"""The cached-entry short-circuit (``RunServices.cached_entry_skip``).

Pins the skip decision after it was folded onto a single ``CacheStore.get_entry``
read (was a ``check_al_id_in_cache`` + a per-field ``get_cached_field``): a cached
entry whose SeaDex ``updated_at`` still matches is skipped (and its url/coverage
backfilled once if the record predates those fields); an absent or stale entry is
re-processed.
"""

from datetime import datetime

from seadexarr.modules.config import Arr
from seadexarr.modules.log import EntryState
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.run_services import RunServices

from .builders import FakeCacheStore, FakeSeaDexSource, make_entry_record, make_services


class _RecordingReporter:
    """Records ``log_cached_entry`` calls, so the cross-arr param is asserted on
    recorded state."""

    def __init__(self) -> None:
        self.calls: list[tuple[RunContext, Arr, int, EntryState]] = []

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        state: EntryState = EntryState.UNCHANGED,
    ) -> bool:
        self.calls.append((ctx, arr, al_id, state))
        return True


class TestCachedEntrySkip:
    @staticmethod
    def _run(cache: FakeCacheStore) -> RunServices:
        # cached_entry_skip touches only cache_store + _config (real, default
        # ignore_seadex_update_times=False) + the reporter; ctx defaults to a SONARR
        # RunContext (make_services), which cached_entry_skip now reads for the arr.
        return make_services(cache_store=cache, _reporter=_RecordingReporter())

    def test_skips_when_cached_and_timestamp_matches(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        reporter = _RecordingReporter()
        run = make_services(cache_store=cache, _reporter=reporter)
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True
        assert len(reporter.calls) == 1

    def test_does_not_skip_when_entry_absent(self) -> None:
        run = self._run(FakeCacheStore())
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_does_not_skip_when_timestamp_is_stale(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        # SeaDex entry now carries a newer updated_at -> stale -> re-process.
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2022, 6, 6)), lambda: "") is False

    def test_backfills_url_and_coverage_when_url_missing(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})  # legacy: no url yet
        run = self._run(cache)
        assert (
            run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1), url="sd-url"), lambda: "S01")
            is True
        )
        backfilled = cache.get_entry(Arr.SONARR, 7)
        assert backfilled is not None
        assert (backfilled.url, backfilled.coverage) == ("sd-url", "S01")

    def test_ignore_update_times_reprocesses_even_when_fresh(self) -> None:
        # The config escape hatch: a fresh, matching timestamp is still re-processed.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            ignore_seadex_update_times=True,
        )
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_dirty_id_reprocesses_even_when_fresh(self) -> None:
        # An arr-side file change bypasses the skip despite a matching timestamp.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([7])
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is False

    def test_non_dirty_sibling_still_skips(self) -> None:
        # Marking one id dirty must not widen the bypass to other cached ids.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        run.mark_dirty([8])
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True

    def test_begin_run_clears_the_dirty_set(self) -> None:
        # Dirty ids are per-run state: the next run's rebind must reset them.
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = make_services(
            cache_store=cache,
            _reporter=_RecordingReporter(),
            _filter=_CtxBind(),
            _grab_pipeline=_CtxBind(),
        )
        run.mark_dirty([7])
        run.begin_run(run.ctx)
        assert run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "") is True


class TestFallbackSatisfiedResurfacing:
    """The fallback-satisfied marker bypasses the cache skip in warn mode only."""

    @staticmethod
    def _cache(*, marker: bool) -> FakeCacheStore:
        cache = FakeCacheStore()
        cache.update_cache(
            Arr.SONARR,
            7,
            {"url": "u", "updated_at": datetime(2021, 1, 1), "fallback_satisfied": marker},
        )
        return cache

    @staticmethod
    def _skips(cache: FakeCacheStore, private_releases: str) -> bool:
        run = make_services(cache_store=cache, _reporter=_RecordingReporter(), private_releases=private_releases)
        return run.cached_entry_skip(7, make_entry_record(updated_at=datetime(2021, 1, 1)), lambda: "")

    def test_warn_mode_reprocesses_a_marked_entry(self) -> None:
        # Fresh timestamp, but the title was satisfied by a fallback: warn mode
        # re-processes it so the private-only warning resurfaces.
        assert self._skips(self._cache(marker=True), "warn") is False

    def test_fallback_mode_still_skips_a_marked_entry(self) -> None:
        assert self._skips(self._cache(marker=True), "fallback") is True

    def test_warn_mode_skips_an_unmarked_entry(self) -> None:
        assert self._skips(self._cache(marker=False), "warn") is True

    def test_no_releases_skip_clears_a_preseeded_marker(self) -> None:
        # A title that stops yielding usable releases is never fallback-satisfied:
        # the shared tail overwrites a stale True.
        cache = self._cache(marker=True)
        run = make_services(cache_store=cache, _reporter=_TailReporter(), sleep_time=0)
        assert run.no_releases_skip(7, {"name": "Title"}) is False
        entry = cache.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is False


class _CtxBind:
    """A no-op ctx-bind collaborator for driving ``begin_run`` on a bare hub."""

    def begin_run(self, ctx: RunContext) -> None:
        del ctx


class TestCrossArrLookupHonorsParam:
    """``check_al_id_in_cache`` / ``log_cached_entry`` read the ``arr`` PARAMETER,
    never ``self._ctx.arr``.

    The run-arr consolidation drops ``arr`` from methods that only ever act on the
    run's own arr, but these two stay parameterised because the Sonarr run's
    ``ignore_movies_in_radarr`` dedup calls them with ``Arr.RADARR`` to hit the
    Radarr cache while ``ctx.arr`` is SONARR. If either ever read ``ctx.arr``
    instead of the param, that cross-arr check silently reads the wrong cache -
    a regression the all-SONARR tests above physically cannot catch.
    """

    def test_check_al_id_in_cache_honors_explicit_arr_over_ctx(self) -> None:
        cache = FakeCacheStore()
        # Present in the RADARR cache only; the run's ctx.arr is SONARR.
        cache.update_cache(Arr.RADARR, 7, {"updated_at": datetime(2021, 1, 1)})
        run = make_services(cache_store=cache, _ctx=RunContext(arr=Arr.SONARR))
        entry = make_entry_record(updated_at=datetime(2021, 1, 1))
        # The explicit RADARR param must select the Radarr cache (hit)...
        assert run.check_al_id_in_cache(Arr.RADARR, 7, entry) is True
        # ...and the run's own SONARR arr must miss (no Sonarr entry exists).
        assert run.check_al_id_in_cache(Arr.SONARR, 7, entry) is False

    def test_log_cached_entry_forwards_param_arr_not_ctx(self) -> None:
        reporter = _RecordingReporter()
        run = make_services(_reporter=reporter, _ctx=RunContext(arr=Arr.SONARR))
        run.log_cached_entry(Arr.RADARR, 7, state=EntryState.IN_RADARR)
        # The reporter delegate receives the explicit cross-arr value, not ctx.arr.
        assert reporter.calls == [(run._ctx, Arr.RADARR, 7, EntryState.IN_RADARR)]


class _TailReporter:
    """Records the no-releases / no-entry / outage-skip outcomes the shared tails report."""

    def __init__(self) -> None:
        self.no_releases_ctxs: list[RunContext] = []
        self.no_sd_entry_ids: list[int] = []
        self.outage_skip_ids: list[int] = []

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        self.no_releases_ctxs.append(ctx)
        return True

    def log_no_sd_entry(self, ctx: RunContext, al_id: int) -> bool:
        del ctx
        self.no_sd_entry_ids.append(al_id)
        return True

    def log_seadex_outage_skip(self, ctx: RunContext, al_id: int) -> bool:
        del ctx
        self.outage_skip_ids.append(al_id)
        return True


class TestNoReleasesSkip:
    def test_logs_persists_and_reports_not_grabbed(self) -> None:
        # The real four-step tail (log + cache write + throttle + False). Its body
        # previously had fake-only coverage: the seam tests script it, nothing
        # drove the real thing.
        cache = FakeCacheStore()
        reporter = _TailReporter()
        run = make_services(cache_store=cache, _reporter=reporter, sleep_time=0)

        assert run.no_releases_skip(7, {"name": "Title", "url": "u"}) is False

        persisted = cache.get_entry(Arr.SONARR, 7)
        assert persisted is not None
        assert (persisted.name, persisted.url) == ("Title", "u")
        assert len(reporter.no_releases_ctxs) == 1


class TestAlIdPrologue:
    def test_no_seadex_entry_reports_and_returns_none(self) -> None:
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(), _reporter=reporter)

        assert run.al_id_prologue(5) is None
        assert reporter.no_sd_entry_ids == [5]
        assert reporter.outage_skip_ids == []
        assert run._ctx.stats.checked == 1

    def test_outage_skip_reports_distinctly_and_returns_none(self) -> None:
        # A SeaDex-unreachable skip must never be reported as "no entry" - it
        # takes the outage reporter hook and its own tally.
        reporter = _TailReporter()
        run = make_services(_seadex=FakeSeaDexSource(outage=True), _reporter=reporter)

        assert run.al_id_prologue(5) is None
        assert reporter.outage_skip_ids == [5]
        assert reporter.no_sd_entry_ids == []
        assert run._ctx.stats.checked == 1

    def test_entry_found_resets_skip_flags_and_tallies(self) -> None:
        entry = make_entry_record()
        run = make_services(_seadex=FakeSeaDexSource({5: entry}), _reporter=_TailReporter())
        run._ctx.private_only_skipped = True  # stale flag from a previous title

        assert run.al_id_prologue(5) is entry
        assert run._ctx.private_only_skipped is False
        assert run._ctx.stats.checked == 1
