"""The cached-entry short-circuit (``SeaDexArr.cached_entry_skip``).

Pins the skip decision after it was folded onto a single ``CacheStore.get_entry``
read (was a ``check_al_id_in_cache`` + a per-field ``get_cached_field``): a cached
entry whose SeaDex ``updated_at`` still matches is skipped (and its url/coverage
backfilled once if the record predates those fields); an absent or stale entry is
re-processed.
"""

from datetime import datetime
from typing import Any
from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.log import EntryState
from seadexarr.modules.reporter import RunContext
from tests.builders import FakeCacheStore, make_arr


def _entry(dt: datetime) -> Any:
    """A stand-in SeaDex entry exposing only ``updated_at`` (typed Any)."""

    class _Entry:
        updated_at = dt

    return _Entry()


class TestCachedEntrySkip:
    @staticmethod
    def _run(cache: FakeCacheStore):
        # cached_entry_skip touches only cache_store + _config (real, default
        # ignore_seadex_update_times=False) + a mocked reporter; ctx defaults to a
        # SONARR RunContext (make_arr), which cached_entry_skip now reads for the arr.
        return make_arr(cache_store=cache, _reporter=mock.MagicMock())

    def test_skips_when_cached_and_timestamp_matches(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        # Hold the reporter mock locally (typed MagicMock) so the call assertion is
        # type-clean - run._reporter is statically the real RunReporter.
        reporter = mock.MagicMock()
        run = make_arr(cache_store=cache, _reporter=reporter)
        assert run.cached_entry_skip(7, _entry(datetime(2021, 1, 1)), "u", lambda: "") is True
        reporter.log_cached_entry.assert_called_once()

    def test_does_not_skip_when_entry_absent(self) -> None:
        run = self._run(FakeCacheStore())
        assert run.cached_entry_skip(7, _entry(datetime(2021, 1, 1)), "u", lambda: "") is False

    def test_does_not_skip_when_timestamp_is_stale(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"url": "u", "updated_at": datetime(2021, 1, 1)})
        run = self._run(cache)
        # SeaDex entry now carries a newer updated_at -> stale -> re-process.
        assert run.cached_entry_skip(7, _entry(datetime(2022, 6, 6)), "u", lambda: "") is False

    def test_backfills_url_and_coverage_when_url_missing(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 1, 1)})  # legacy: no url yet
        run = self._run(cache)
        assert run.cached_entry_skip(7, _entry(datetime(2021, 1, 1)), "sd-url", lambda: "S01") is True
        backfilled = cache.get_entry(Arr.SONARR, 7)
        assert backfilled is not None
        assert (backfilled.url, backfilled.coverage) == ("sd-url", "S01")


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
        run = make_arr(cache_store=cache, _ctx=RunContext(arr=Arr.SONARR))
        entry = _entry(datetime(2021, 1, 1))
        # The explicit RADARR param must select the Radarr cache (hit)...
        assert run.check_al_id_in_cache(Arr.RADARR, 7, entry) is True
        # ...and the run's own SONARR arr must miss (no Sonarr entry exists).
        assert run.check_al_id_in_cache(Arr.SONARR, 7, entry) is False

    def test_log_cached_entry_forwards_param_arr_not_ctx(self) -> None:
        reporter = mock.MagicMock()
        run = make_arr(_reporter=reporter, _ctx=RunContext(arr=Arr.SONARR))
        run.log_cached_entry(Arr.RADARR, 7, state=EntryState.IN_RADARR)
        # The reporter delegate receives the explicit cross-arr value, not ctx.arr.
        reporter.log_cached_entry.assert_called_once_with(
            run._ctx, Arr.RADARR, 7, state=EntryState.IN_RADARR
        )
