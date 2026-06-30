# pyright: strict
"""Characterization tests for RunReporter + RunContext.

The run loop and the end-of-run summary had no coverage before this phase, so
these pin the run-state contracts the orchestrator depends on: the stats-tally
counters each ``log_*`` method bumps, the active-title attribution set by
``log_al_title``, and that the summary renders without error on both the real
and dry-run paths. Presentation goes through a NullHandler logger, so the tests
assert on the :class:`RunContext` mutations rather than exact log strings.

The collaborators are real, not mocks: the reporter is built with the shared
in-memory :class:`FakeCacheStore` (typed by ``AbstractCacheStore``) and a real
:class:`AniListGateway` whose own cache store is faked - the "construct the
composite, fake its leaves" seam - so the whole file type-checks at strict.
"""

import logging
import time
from typing import override

import pytest

from seadexarr.modules.anilist_gateway import AniListGateway
from seadexarr.modules.cache import AbstractCacheStore, CacheRecord
from seadexarr.modules.config import Arr
from seadexarr.modules.log import LogFormatter
from seadexarr.modules.manual_import import ImportWaitMode, PendingState
from seadexarr.modules.reporter import (
    GrabRecord,
    NeedsActionRecord,
    RunContext,
    RunReporter,
    RunStats,
)
from seadexarr.modules.torrents import AddOutcome, ReleaseOutcome

from .builders import FakeCacheStore, make_entry_record, make_logger


class _CaptureHandler(logging.Handler):
    """Collects every emitted record so summary/action rows can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _make_reporter(cache_store: AbstractCacheStore | None = None) -> RunReporter:
    logger = make_logger()
    store: AbstractCacheStore = cache_store if cache_store is not None else FakeCacheStore()
    # A real gateway with a faked cache store: the reporter only reads/reassigns
    # its ``al_cache`` dict, so the real wiring is exercised without a network.
    anilist = AniListGateway(cache_store=FakeCacheStore(), logger=logger)
    return RunReporter(
        logger=logger,
        log_fmt=LogFormatter(logger),
        cache_store=store,
        anilist=anilist,
    )


def _seeded_store(*, name: str, coverage: str, url: str) -> FakeCacheStore:
    """A FakeCacheStore with one entry row (arr=Sonarr, al_id=1) preseeded."""

    store = FakeCacheStore()
    store.update_cache(Arr.SONARR, 1, CacheRecord(name=name, coverage=coverage, url=url))
    return store


def test_run_stats_shape() -> None:
    s = RunStats()
    assert s.checked == 0
    assert s.added == []
    assert s.needs_action == []
    assert s.unmonitored == 0


class TestStatsCounters:
    def test_unmonitored(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        _make_reporter().log_arr_item_unmonitored(ctx, "Title")
        assert ctx.stats.unmonitored == 1

    def test_no_mappings(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        _make_reporter().log_no_anilist_mappings(ctx, "Title")
        assert ctx.stats.no_mappings == 1

    def test_no_releases(self) -> None:
        ctx = RunContext(arr=Arr.RADARR)
        _make_reporter().log_no_seadex_releases(ctx)
        assert ctx.stats.no_releases == 1

    def test_cached_uses_stored_name(self) -> None:
        # A stored name short-circuits the AniList lookup (no network)
        reporter = _make_reporter(_seeded_store(name="Cached", coverage="S01", url="u"))
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_cached_entry(ctx, Arr.SONARR, 1)
        assert ctx.stats.cached == 1

    def test_no_sd_entry_increments_and_threads_al_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _make_reporter()
        monkeypatch.setattr(
            "seadexarr.modules.reporter.get_anilist_title",
            _fake_get_title,
        )
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_no_sd_entry(ctx, 42)
        assert ctx.stats.no_seadex_entry == 1
        # The al_cache reassignment side-effect is preserved through the gateway
        assert reporter.anilist.al_cache == {42: "Resolved"}


def _fake_get_title(al_id: int, al_cache: dict[int, str]) -> tuple[str, dict[int, str]]:
    """Stand-in for ``get_anilist_title``: resolves a fixed title, threads cache."""

    return "Resolved", {**al_cache, al_id: "Resolved"}


class TestActiveTitle:
    def test_log_al_title_sets_current(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        entry = make_entry_record(url="https://releases.moe/9")
        _make_reporter().log_al_title(ctx, "Steins;Gate", entry, coverage="S01 E01-E24")
        assert ctx.current_title == "Steins;Gate"
        assert ctx.current_url == "https://releases.moe/9"
        assert ctx.current_coverage == "S01 E01-E24"


class TestRunSummary:
    def _ctx_with_data(self) -> RunContext:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.checked = 3
        ctx.torrents_added = 1
        ctx.started_monotonic = time.monotonic() - 1.0  # exercise the elapsed line
        ctx.stats.added = [
            GrabRecord(title="A", coverage="S01", url="u", name="A.mkv", group="G"),
        ]
        ctx.stats.needs_action = [
            NeedsActionRecord(
                title="B",
                coverage="S02",
                group="Priv",
                url="u2",
                reason="private-only release; public_only on",
            ),
        ]
        return ctx

    def test_real_run_renders(self) -> None:
        assert _make_reporter().log_run_summary(
            self._ctx_with_data(),
            Arr.SONARR,
            is_preview=False,
            has_client=True,
        )

    def test_dry_run_renders(self) -> None:
        assert _make_reporter().log_run_summary(
            self._ctx_with_data(),
            Arr.SONARR,
            is_preview=True,
            has_client=False,
        )


def _summary_messages(
    reporter: RunReporter,
    ctx: RunContext,
    *,
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
) -> list[str]:
    """Capture every log message log_run_summary emits, for row assertions."""

    handler = _CaptureHandler()
    reporter.logger.addHandler(handler)
    reporter.logger.setLevel(logging.DEBUG)
    try:
        reporter.log_run_summary(
            ctx,
            Arr.SONARR,
            is_preview=False,
            has_client=True,
            import_wait_mode=import_wait_mode,
        )
    finally:
        reporter.logger.removeHandler(handler)
    return [r.getMessage() for r in handler.records]


class TestPendingSnapshot:
    """log_pending_snapshot renders the inline carried-over row + bumps NO counter."""

    def test_renders_and_bumps_no_counter(self) -> None:
        reporter = _make_reporter()
        ctx = RunContext(arr=Arr.SONARR)

        rendered = reporter.log_pending_snapshot(
            ctx,
            PendingState.IMPORTED,
            "My Show",
            "S01 E01-E13",
            "https://releases.moe/1",
        )

        assert rendered is True
        # The reporter never touches the counters - the engine owns drop/count.
        assert ctx.stats.imported == 0
        assert ctx.stats.queued == 0
        assert ctx.stats.importing == 0

    def test_missing_state_renders_nothing(self) -> None:
        reporter = _make_reporter()
        ctx = RunContext(arr=Arr.SONARR)

        assert (
            reporter.log_pending_snapshot(
                ctx,
                PendingState.MISSING,
                "Gone",
                None,
                None,
            )
            is False
        )


class TestSummaryPendingCounters:
    """The carried-over counters render only when feature-on and non-zero."""

    def test_counters_render_when_feature_on_and_non_zero(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.queued = 2
        ctx.stats.importing = 1
        ctx.stats.imported = 3

        messages = _summary_messages(
            _make_reporter(),
            ctx,
            import_wait_mode=ImportWaitMode.BLOCKING,
        )
        joined = "\n".join(messages)

        assert any("queued" in m for m in messages)
        assert any("importing" in m for m in messages)
        assert "imported" in joined

    def test_counters_hidden_when_feature_off(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.queued = 2
        ctx.stats.importing = 1
        ctx.stats.imported = 3

        messages = _summary_messages(
            _make_reporter(),
            ctx,
            import_wait_mode=ImportWaitMode.OFF,
        )

        assert not any("queued" in m for m in messages)
        assert not any("importing" in m for m in messages)

    def test_zero_counters_not_rendered_even_when_on(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)  # all counters zero

        messages = _summary_messages(
            _make_reporter(),
            ctx,
            import_wait_mode=ImportWaitMode.BLOCKING,
        )

        assert not any("queued" in m for m in messages)
        assert not any("importing" in m for m in messages)

    def test_this_run_grab_shows_added_only_no_queued(self) -> None:
        # REGRESSION (double-report): a this-run grab is `added`; with the counters
        # left at 0 (the engine never bumps them for this-run grabs), the summary
        # shows `added` but no `queued` row for the same torrent.
        ctx = RunContext(arr=Arr.SONARR)
        ctx.torrents_added = 1
        ctx.stats.added = [
            GrabRecord(title="A", coverage="S01", url="u", name="A.mkv", group="G"),
        ]
        # counters stay 0 -> no queued/importing rows

        messages = _summary_messages(
            _make_reporter(),
            ctx,
            import_wait_mode=ImportWaitMode.BLOCKING,
        )

        assert any("added" in m for m in messages)
        assert not any("queued" in m for m in messages)


def _action_messages(
    reporter: RunReporter,
    results: list[ReleaseOutcome],
    *,
    dry_run: bool = False,
    monitor_active: bool = False,
) -> tuple[bool, list[str]]:
    """Capture the status + per-release rows log_seadex_action emits.

    Passes an empty ``seadex_dict`` so the recommended-group rows are skipped and
    the assertions key only on the status line and the per-release outcome rows.
    """

    handler = _CaptureHandler()
    reporter.logger.addHandler(handler)
    reporter.logger.setLevel(logging.DEBUG)
    try:
        logged = reporter.log_seadex_action({}, results, dry_run=dry_run, monitor_active=monitor_active)
    finally:
        reporter.logger.removeHandler(handler)
    return logged, [r.getMessage() for r in handler.records]


class TestLogSeadexAction:
    """log_seadex_action renders adding / already-downloading / keeping distinctly."""

    def test_added_status_and_label(self) -> None:
        logged, messages = _action_messages(
            _make_reporter(),
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name="Show-NAN0", group="NAN0")],
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "adding a better release" in joined
        assert "added" in joined
        assert "Show-NAN0" in joined

    def test_added_hashless_release_shows_group_not_none(self) -> None:
        # A hashless/private torrent has name=None; the row must fall back to the
        # release group, never render the literal "None".
        logged, messages = _action_messages(
            _make_reporter(),
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name=None, group="NAN0")],
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "None" not in joined  # the bug rendered "added : None"

    def test_already_downloading_status_monitor_active(self) -> None:
        logged, messages = _action_messages(
            _make_reporter(),
            [ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="Show-NAN0", group="NAN0")],
            monitor_active=True,
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "already downloading" in joined
        assert "waiting to import" in joined
        assert "downloading" in joined and "Show-NAN0" in joined
        # The misleading "keeping it" / "kept" wording is gone for this case.
        assert "keeping it" not in joined
        assert "kept" not in joined

    def test_already_downloading_status_monitor_inactive(self) -> None:
        # Feature off / preview: state the fact, but don't promise an import.
        _, messages = _action_messages(
            _make_reporter(),
            [ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="X", group="G")],
            monitor_active=False,
        )
        joined = "\n".join(messages)

        assert "already downloading" in joined
        assert "waiting to import" not in joined

    def test_mixed_added_and_already_added_reads_adding(self) -> None:
        # A fresh grab happened, so the headline is "adding"; the per-release rows
        # disambiguate (one added, one still downloading).
        _, messages = _action_messages(
            _make_reporter(),
            [
                ReleaseOutcome(outcome=AddOutcome.ADDED, name="new", group="G"),
                ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="old", group="G"),
            ],
            monitor_active=True,
        )
        joined = "\n".join(messages)

        assert "adding a better release" in joined
        assert "already downloading" not in joined
        assert "new" in joined and "old" in joined

    def test_dry_run_reads_adding(self) -> None:
        logged, messages = _action_messages(_make_reporter(), [], dry_run=True)

        assert logged is True
        assert "adding a better release" in "\n".join(messages)

    def test_all_skipped_returns_false_and_emits_nothing(self) -> None:
        logged, messages = _action_messages(_make_reporter(), [])

        assert logged is False
        assert messages == []
