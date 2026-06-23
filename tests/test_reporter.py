"""Characterization tests for RunReporter + RunContext (Phase 4b).

The run loop and the end-of-run summary had no coverage before this phase, so
these pin the run-state contracts the orchestrator depends on: the stats-tally
counters each ``log_*`` method bumps, the active-title attribution set by
``log_al_title``, and that the summary renders without error on both the real
and dry-run paths. Presentation goes through a NullHandler logger, so the tests
assert on the :class:`RunContext` mutations rather than exact log strings.
"""

import time
from typing import Any

from seadexarr.modules.config import Arr
from seadexarr.modules.log import LogFormatter
from seadexarr.modules.reporter import RunContext, RunReporter, fresh_stats
from tests.builders import make_logger


class _FakeCacheStore:
    """Minimal stand-in for CacheStore: the two reads the reporter makes."""

    def __init__(self, name: str | None = None, fields: dict | None = None) -> None:
        self._name = name
        self._fields = fields or {}

    def get_cached_name(self, arr: str, al_id: int) -> str | None:
        del arr, al_id
        return self._name

    def get_cached_field(self, arr: str, al_id: int, field: str):
        del arr, al_id
        return self._fields.get(field)


class _FakeAniList:
    """Owns just the al_cache attribute the reporter reads/reassigns."""

    def __init__(self) -> None:
        self.al_cache: dict = {}


class _FakeEntry:
    """SeaDex entry stand-in (the two fields log_al_title reads)."""

    def __init__(self, url: str = "https://releases.moe/1", is_incomplete: bool = False) -> None:
        self.url = url
        self.is_incomplete = is_incomplete


def _make_reporter(cache_store: Any = None) -> RunReporter:
    logger = make_logger()
    # The fakes are duck-typed; pass them as Any so they satisfy the reporter's
    # CacheStore / AniListGateway parameters without a per-call cast.
    cache: Any = cache_store or _FakeCacheStore()
    anilist: Any = _FakeAniList()
    return RunReporter(
        logger=logger,
        log_fmt=LogFormatter(logger),
        cache_store=cache,
        anilist=anilist,
    )


def test_fresh_stats_shape() -> None:
    s = fresh_stats()
    assert s["checked"] == 0
    assert s["added"] == []
    assert s["needs_action"] == []
    assert s["unmonitored"] == 0


class TestStatsCounters:
    def test_unmonitored(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        _make_reporter().log_arr_item_unmonitored(ctx, "Title")
        assert ctx.stats["unmonitored"] == 1

    def test_no_mappings(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        _make_reporter().log_no_anilist_mappings(ctx, "Title")
        assert ctx.stats["no_mappings"] == 1

    def test_no_releases(self) -> None:
        ctx = RunContext(arr=Arr.RADARR)
        _make_reporter().log_no_seadex_releases(ctx)
        assert ctx.stats["no_releases"] == 1

    def test_cached_uses_stored_name(self) -> None:
        # A stored name short-circuits the AniList lookup (no network)
        reporter = _make_reporter(
            _FakeCacheStore(name="Cached", fields={"coverage": "S01", "url": "u"}),
        )
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_cached_entry(ctx, Arr.SONARR, 1)
        assert ctx.stats["cached"] == 1

    def test_no_sd_entry_increments_and_threads_al_cache(self, monkeypatch) -> None:
        reporter = _make_reporter()
        monkeypatch.setattr(
            "seadexarr.modules.reporter.get_anilist_title",
            lambda al_id, al_cache: ("Resolved", {**al_cache, al_id: "Resolved"}),
        )
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_no_sd_entry(ctx, 42)
        assert ctx.stats["no_seadex_entry"] == 1
        # The al_cache reassignment side-effect is preserved through the gateway
        assert reporter.anilist.al_cache == {42: "Resolved"}


class TestActiveTitle:
    def test_log_al_title_sets_current(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        entry: Any = _FakeEntry(url="https://releases.moe/9")
        _make_reporter().log_al_title(ctx, "Steins;Gate", entry, coverage="S01 E01-E24")
        assert ctx.current_title == "Steins;Gate"
        assert ctx.current_url == "https://releases.moe/9"
        assert ctx.current_coverage == "S01 E01-E24"


class TestRunSummary:
    def _ctx_with_data(self) -> RunContext:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats["checked"] = 3
        ctx.torrents_added = 1
        ctx.started_monotonic = time.monotonic() - 1.0  # exercise the elapsed line
        ctx.stats["added"] = [
            {"title": "A", "coverage": "S01", "url": "u", "name": "A.mkv", "group": "G"},
        ]
        ctx.stats["needs_action"] = [
            {
                "title": "B",
                "coverage": "S02",
                "group": "Priv",
                "url": "u2",
                "reason": "private-only release; public_only on",
            },
        ]
        return ctx

    def test_real_run_renders(self) -> None:
        assert _make_reporter().log_run_summary(
            self._ctx_with_data(), Arr.SONARR, is_preview=False, has_client=True,
        )

    def test_dry_run_renders(self) -> None:
        assert _make_reporter().log_run_summary(
            self._ctx_with_data(), Arr.SONARR, is_preview=True, has_client=False,
        )
