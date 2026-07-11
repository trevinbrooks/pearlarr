# pyright: strict
"""Characterization tests for RunReporter + RunContext.

The run loop and the end-of-run summary had no coverage before this phase, so
these pin the run-state contracts the orchestrator depends on: the stats-tally
counters each ``log_*`` method bumps, the active-title attribution set by
``log_al_title``, and that the summary renders without error on both the real
and dry-run paths. Presentation is captured as EMITTED events (via the ``emit``
seam) and, where a test pins output, re-derived to lines through the shipped
builders; most tests just assert on the :class:`RunContext` mutations.

The collaborators are real, not mocks: the reporter is built with the shared
in-memory :class:`FakeCacheStore` (typed by ``AbstractCacheStore``) and a real
:class:`AniListGateway` whose own cache store is faked - the "construct the
composite, fake its leaves" seam - so the whole file type-checks at strict.
"""

import time
from typing import Any, override

import httpx
from seadex import Tag

from pearlarr.modules.anilist_client import AniListClient
from pearlarr.modules.anilist_gateway import AniListGateway
from pearlarr.modules.cache import AbstractCacheStore, CacheRecord
from pearlarr.modules.config import Arr
from pearlarr.modules.manual_import import ImportWaitMode, PendingState
from pearlarr.modules.output import (
    EntryHeader,
    Event,
    GrabAction,
    NeedsActionCause,
    RunFinished,
    RunSummaryReady,
    ScanFinished,
    ScopeClosed,
    ScopeOpened,
    Severity,
    SeverityCounts,
)
from pearlarr.modules.reporter import (
    GrabRecord,
    NeedsActionKind,
    NeedsActionRecord,
    RunContext,
    RunReporter,
    RunStats,
)
from pearlarr.modules.torrents import AddOutcome, ReleaseOutcome

from .builders import FakeCacheStore, make_entry_record, make_logger, pending_import, rg_group, url_item
from .fakes import scan_lines_from_events


def _record(
    cache_store: AbstractCacheStore | None = None,
    client: AniListClient | None = None,
) -> tuple[RunReporter, list[Event]]:
    """A real RunReporter (faked cache leaf, real gateway) plus its recorded events.

    The reporter EMITS events through the recorder; tests that pin output re-derive
    lines from the events. A real gateway with a faked cache store: the reporter
    only reads/updates its ``al_cache`` dict, so the real wiring runs without a
    network.
    """

    store: AbstractCacheStore = cache_store if cache_store is not None else FakeCacheStore()
    anilist = AniListGateway(
        cache_store=FakeCacheStore(),
        logger=make_logger(),
        client=client if client is not None else AniListClient(client=httpx.Client()),
    )
    events: list[Event] = []
    counts = SeverityCounts()
    reporter = RunReporter(emit=events.append, counts=lambda: counts, cache_store=store, anilist=anilist)
    return reporter, events


def _make_reporter(
    cache_store: AbstractCacheStore | None = None,
    client: AniListClient | None = None,
) -> RunReporter:
    """The reporter alone, for the ctx/stats-mutation tests (events discarded)."""

    reporter, _events = _record(cache_store, client)
    return reporter


def _event_messages(events: list[Event]) -> list[str]:
    """The plain messages a recorded event stream re-derives to (shipped builders)."""

    return [line.message for line in scan_lines_from_events(events)]


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

    def test_no_sd_entry_increments_and_caches_title(self) -> None:
        client = _ScriptedTitleClient()
        reporter = _make_reporter(client=client)
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_no_sd_entry(ctx, 42)
        assert ctx.stats.no_seadex_entry == 1
        # Routed through the gateway: its al_cache is warmed in place and the
        # lookup rode its bound wire client (which carries the transport and the
        # per-run retry narration by construction).
        assert 42 in reporter.anilist.al_cache
        assert client.query_calls == [42]

    def test_cached_without_stored_name_falls_back_to_gateway_title(self) -> None:
        # A legacy record predating name storage: the title fallback routes
        # through the gateway too, never a bare wire query.
        store = FakeCacheStore()
        store.update_cache(Arr.SONARR, 1, CacheRecord(coverage="S01", url="u"))
        client = _ScriptedTitleClient()
        reporter = _make_reporter(store, client=client)
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_cached_entry(ctx, Arr.SONARR, 1)
        assert ctx.stats.cached == 1
        assert client.query_calls == [1]

    def test_outage_skip_tallies_and_renders_distinctly(self) -> None:
        # A SeaDex-unreachable skip lands in its own counter and its ledger row
        # reads "skipped" with the reason - never "no entry". An UNCACHED id has
        # no stored name, so the title still comes from the gateway lookup.
        client = _ScriptedTitleClient()
        reporter, events = _record(client=client)
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_seadex_outage_skip(ctx, 42)

        assert ctx.stats.seadex_unreachable == 1
        assert ctx.stats.no_seadex_entry == 0
        joined = "\n".join(_event_messages(events))
        assert "skipped" in joined and "Resolved" in joined
        assert "lookup skipped (SeaDex unreachable)" in joined
        assert "no entry" not in joined
        assert client.query_calls == [42]

    def test_outage_skip_prefers_cached_name_over_anilist(self) -> None:
        # A previously-processed title's name sits in the cache row: the outage
        # skip must render it from there with NO AniList lookup - in a compound
        # SeaDex+AniList outage a lookup would pay retry backoff per title.
        client = _ScriptedTitleClient()
        reporter, events = _record(_seeded_store(name="Stored Title", coverage="S01", url="u"), client=client)
        ctx = RunContext(arr=Arr.SONARR)
        reporter.log_seadex_outage_skip(ctx, 1)

        assert ctx.stats.seadex_unreachable == 1
        joined = "\n".join(_event_messages(events))
        assert "Stored Title" in joined
        assert "lookup skipped (SeaDex unreachable)" in joined
        assert client.query_calls == []  # the stored name spared the lookup


class _ScriptedTitleClient(AniListClient):
    """Checked scripted ``AniListClient``: a fixed resolvable title, queries recorded.

    Injected into the gateway under the reporter, so a title lookup exercises
    the REAL gateway get-or-fetch (cache warm + store) over a canned wire body.
    """

    def __init__(self) -> None:
        super().__init__(client=httpx.Client())
        self.query_calls: list[int] = []

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        self.query_calls.append(al_id)
        return {"data": {"Media": {"id": al_id, "title": {"english": "Resolved"}}}}


class TestActiveTitle:
    def test_log_al_title_sets_current(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        entry = make_entry_record(url="https://releases.moe/9")
        _make_reporter().log_al_title(ctx, "Steins;Gate", entry, coverage="S01 E01-E24")
        assert ctx.current_title == "Steins;Gate"
        assert ctx.current_url == "https://releases.moe/9"
        assert ctx.current_coverage == "S01 E01-E24"


class TestEntryHeaderAlId:
    """G1: ``al_id`` on the emitted EntryHeader is producer-unverified - garbage
    in either header-opening path passed the whole suite - so pin it on both."""

    def _header(self, events: list[Event]) -> EntryHeader:
        return next(e for e in events if isinstance(e, EntryHeader))

    def test_log_al_title_carries_the_seadex_entrys_anilist_id(self) -> None:
        reporter, events = _record()
        entry = make_entry_record(anilist_id=12345)
        reporter.log_al_title(RunContext(arr=Arr.SONARR), "Steins;Gate", entry)
        assert self._header(events).al_id == 12345

    def test_log_cached_entry_carries_the_al_id_param(self) -> None:
        store = FakeCacheStore()
        store.update_cache(Arr.SONARR, 4242, CacheRecord(name="Cached", coverage="S01", url="u"))
        reporter, events = _record(store)
        reporter.log_cached_entry(RunContext(arr=Arr.SONARR), Arr.SONARR, 4242)
        assert self._header(events).al_id == 4242


class TestCloseBoundaries:
    """``scan_finished`` / ``run_finished`` close the open entry, then state the boundary.

    Closing entry-first is what keeps a later diagnostic (a reconcile warning, a
    leg-fatal error) from rendering inside the entry the scan happened to end on.
    """

    def test_scan_finished_closes_the_open_entry_first(self) -> None:
        reporter, events = _record()
        reporter.log_al_title(RunContext(arr=Arr.SONARR), "Steins;Gate", make_entry_record())

        reporter.scan_finished(Arr.SONARR)

        assert [type(e) for e in events] == [ScopeOpened, EntryHeader, ScopeClosed, ScanFinished]

    def test_run_finished_without_an_open_entry_emits_only_the_boundary(self) -> None:
        reporter, events = _record()

        reporter.run_finished(Arr.SONARR)

        assert events == [RunFinished(arr=Arr.SONARR)]

    def test_the_boundaries_are_idempotent_about_the_entry(self) -> None:
        # scan_finished already closed it, so run_finished's defensive close is a no-op:
        # a second ScopeClosed would demote every later fact on that stale scope.
        reporter, events = _record()
        reporter.log_al_title(RunContext(arr=Arr.SONARR), "Steins;Gate", make_entry_record())

        reporter.scan_finished(Arr.SONARR)
        reporter.run_finished(Arr.SONARR)

        assert [type(e) for e in events].count(ScopeClosed) == 1


class TestCompleteBlocksSelfClose:
    """D6: a cached / carried-over-pending block is COMPLETE on return (nothing
    follows it - the header carries the whole row), so it self-closes. A gap
    diagnostic (e.g. a retry WARNING while resolving the next title) then rides
    col 0 rather than indenting under the finished block. Contrast a CHECKING
    entry, which legitimately accrues followers and stays open until a boundary.
    """

    def test_cached_entry_emits_scope_closed_before_returning(self) -> None:
        reporter, events = _record(_seeded_store(name="Cached", coverage="S01", url="u"))
        reporter.log_cached_entry(RunContext(arr=Arr.SONARR), Arr.SONARR, 1)
        assert [type(e) for e in events] == [ScopeOpened, EntryHeader, ScopeClosed]

    def test_pending_snapshot_emits_scope_closed_before_returning(self) -> None:
        reporter, events = _record()
        assert reporter.log_pending_snapshot(
            PendingState.IMPORTED,
            pending_import(title="My Show", coverage="S01 E01-E13", url="https://releases.moe/1"),
        )
        assert [type(e) for e in events] == [ScopeOpened, EntryHeader, ScopeClosed]


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
                reason="private-only release; private releases not supported",
                kind=NeedsActionKind.PRIVATE_ONLY,
            ),
        ]
        return ctx

    def _summary_of(self, events: list[Event]) -> RunSummaryReady:
        return next(e for e in events if isinstance(e, RunSummaryReady))

    def test_real_run_emits_the_scoreboard(self) -> None:
        reporter, events = _record()
        reporter.log_run_summary(self._ctx_with_data(), preview=False, has_client=True)

        summary = self._summary_of(events).summary
        assert summary.dry_run is False and summary.dry_run_note is None
        assert summary.added_count == 1
        assert summary.tally.checked == 3
        assert [f.cause for f in summary.tally.needs_action] == [NeedsActionCause.PRIVATE_ONLY]
        assert summary.tip is NeedsActionCause.PRIVATE_ONLY
        assert summary.elapsed_s is not None and summary.elapsed_s > 0
        joined = "\n".join(_event_messages(events))
        assert "run complete" in joined and "needs action" in joined

    def test_dry_run_note_wording_tracks_the_client(self) -> None:
        reporter, events = _record()
        reporter.log_run_summary(self._ctx_with_data(), preview=True, has_client=False)
        summary = self._summary_of(events).summary
        assert summary.dry_run is True
        assert summary.dry_run_note == "qBittorrent not configured; nothing grabbed"

        reporter, events = _record()
        reporter.log_run_summary(self._ctx_with_data(), preview=True, has_client=True)
        assert self._summary_of(events).summary.dry_run_note == "nothing grabbed"

    def test_counts_mark_stays_bound_to_the_counter_it_was_stamped_on(self) -> None:
        """A counter swapped in between mark and summary (a re-installed hub) can't
        feed the diff: the mark carries the ORIGINAL counter."""

        anilist = AniListGateway(
            cache_store=FakeCacheStore(),
            logger=make_logger(),
            client=AniListClient(client=httpx.Client()),
        )
        counters = [SeverityCounts()]
        events: list[Event] = []
        reporter = RunReporter(
            emit=events.append,
            counts=lambda: counters[-1],
            cache_store=FakeCacheStore(),
            anilist=anilist,
        )

        ctx = self._ctx_with_data()
        ctx.counts_mark = reporter.counts_mark()
        counters[-1].record(Severity.WARNING)  # the run's own post-mark warning

        # The "hub swap": the source now resolves to a fresh counter soaking up
        # unrelated errors.
        counters.append(SeverityCounts())
        counters[-1].record(Severity.ERROR)
        counters[-1].record(Severity.ERROR)

        reporter.log_run_summary(ctx, preview=False, has_client=True)
        summary = self._summary_of(events).summary
        assert summary.warnings == 1  # the original counter's record, not garbage
        assert summary.errors == 0  # the swapped-in counter's errors never leak


def _summary_messages(
    ctx: RunContext,
    *,
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
) -> list[str]:
    """The messages log_run_summary re-derives to, for row-presence assertions."""

    ctx.import_wait_mode = import_wait_mode
    reporter, events = _record()
    reporter.log_run_summary(ctx, preview=False, has_client=True)
    return _event_messages(events)


class TestPendingSnapshot:
    """log_pending_snapshot renders the inline carried-over row + bumps NO counter."""

    def test_renders_and_bumps_no_counter(self) -> None:
        reporter = _make_reporter()
        ctx = RunContext(arr=Arr.SONARR)

        rendered = reporter.log_pending_snapshot(
            PendingState.IMPORTED,
            pending_import(title="My Show", coverage="S01 E01-E13", url="https://releases.moe/1"),
        )

        assert rendered is True
        # The reporter never touches the counters - the engine owns drop/count.
        assert ctx.stats.imported == 0
        assert ctx.stats.queued == 0
        assert ctx.stats.importing == 0

    def test_missing_state_renders_nothing(self) -> None:
        reporter = _make_reporter()

        assert (
            reporter.log_pending_snapshot(
                PendingState.MISSING,
                pending_import(title="Gone"),
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
            ctx,
            import_wait_mode=ImportWaitMode.OFF,
        )

        assert not any("queued" in m for m in messages)
        assert not any("importing" in m for m in messages)

    def test_zero_counters_not_rendered_even_when_on(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)  # all counters zero

        messages = _summary_messages(
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
            ctx,
            import_wait_mode=ImportWaitMode.BLOCKING,
        )

        assert any("added" in m for m in messages)
        assert not any("queued" in m for m in messages)


class TestSummaryNoReleaseRow:
    """The "no release" row is gated on non-zero, like its no-mapping/no-entry siblings."""

    def test_zero_count_renders_no_row(self) -> None:
        messages = _summary_messages(RunContext(arr=Arr.SONARR))

        assert not any("no release" in m for m in messages)

    def test_non_zero_count_renders_the_row(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.no_releases = 2

        messages = _summary_messages(ctx)

        assert any("no release" in m and "2" in m for m in messages)


class TestSummarySeadexDownRow:
    """Outage skips get their own gated "seadex down" row - never the alarming
    (and untrue) "no entry" count."""

    def test_zero_count_renders_no_row(self) -> None:
        messages = _summary_messages(RunContext(arr=Arr.SONARR))

        assert not any("seadex down" in m for m in messages)

    def test_non_zero_count_renders_the_row_not_no_entry(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.seadex_unreachable = 3

        messages = _summary_messages(ctx)

        assert any("seadex down" in m and "3" in m for m in messages)
        assert not any("no entry" in m for m in messages)


class TestPrivateOnlyTip:
    """The private-only guidance tip gates on the record's KIND, never on the
    display ``reason`` text (rewording the string must not kill the tip)."""

    def _needs_ctx(self, kind: NeedsActionKind) -> RunContext:
        ctx = RunContext(arr=Arr.SONARR)
        ctx.stats.needs_action = [
            NeedsActionRecord(
                title="B",
                coverage="S02",
                group="Priv",
                url="u2",
                reason="a reworded reason with no magic words",
                kind=kind,
            ),
        ]
        return ctx

    def test_private_only_kind_renders_tip_despite_reworded_reason(self) -> None:
        messages = _summary_messages(self._needs_ctx(NeedsActionKind.PRIVATE_ONLY))
        assert any("private_releases: fallback" in m for m in messages)
        assert not any("private_releases: allow" in m for m in messages)

    def test_no_fallback_kind_tip_omits_the_fallback_suggestion(self) -> None:
        # Fallback mode found nothing public: suggesting private_releases: fallback
        # (already on) would be nonsense, so that kind's tip names no setting at all.
        messages = _summary_messages(self._needs_ctx(NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK))
        assert not any("private_releases" in m for m in messages)
        assert any("re-checked every run" in m for m in messages)

    def test_stale_kind_renders_the_owned_stale_tip(self) -> None:
        # The fallback-never-replaces-an-owned-copy hold: the tip names the two
        # ways out (update or delete) and, with fallback already on, no setting.
        messages = _summary_messages(self._needs_ctx(NeedsActionKind.PRIVATE_ONLY_STALE))
        assert not any("private_releases" in m for m in messages)
        assert any(
            "your copies of these releases are outdated (their file sizes no longer match); "
            "update them from their private tracker, or delete the outdated files to let the "
            "public fallback stand in." in m
            for m in messages
        )

    def test_unsupported_tracker_kind_renders_no_tip(self) -> None:
        messages = _summary_messages(self._needs_ctx(NeedsActionKind.UNSUPPORTED_TRACKER))
        assert not any("private_releases" in m for m in messages)


def _action_messages(
    results: list[ReleaseOutcome],
    *,
    dry_run: bool = False,
    monitor_active: bool = False,
) -> tuple[bool, list[str]]:
    """The status + per-release rows log_seadex_action re-derives to.

    Passes an empty ``seadex_dict`` so the recommended-group rows are skipped and
    the assertions key only on the status line and the per-release outcome rows.
    """

    reporter, events = _record()
    logged = reporter.log_seadex_action({}, results, dry_run=dry_run, monitor_active=monitor_active)
    return logged, _event_messages(events)


class TestLogSeadexAction:
    """log_seadex_action renders adding / already-downloading / keeping distinctly."""

    def test_added_status_and_label(self) -> None:
        logged, messages = _action_messages(
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name="Show-NAN0", group="NAN0")],
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "adding SeaDex's recommended release" in joined
        assert "added" in joined
        assert "Show-NAN0" in joined

    def test_added_hashless_release_shows_group_not_none(self) -> None:
        # A hashless/private torrent has name=None; the row must fall back to the
        # release group, never render the literal "None".
        logged, messages = _action_messages(
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name=None, group="NAN0")],
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "None" not in joined  # the bug rendered "added : None"

    def test_already_downloading_status_monitor_active(self) -> None:
        logged, messages = _action_messages(
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
            [
                ReleaseOutcome(outcome=AddOutcome.ADDED, name="new", group="G"),
                ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="old", group="G"),
            ],
            monitor_active=True,
        )
        joined = "\n".join(messages)

        assert "adding SeaDex's recommended release" in joined
        assert "already downloading" not in joined
        assert "new" in joined and "old" in joined

    def test_dry_run_reads_would_add(self) -> None:
        # A dry run must never read like a real grab: the status says "would add
        # ... (dry run)" and the per-release row is labeled "would add".
        logged, messages = _action_messages(
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name="Show-NAN0", group="NAN0")],
            dry_run=True,
        )
        joined = "\n".join(messages)

        assert logged is True
        assert "would add SeaDex's recommended release (dry run)" in joined
        assert any("would add" in m and "Show-NAN0" in m for m in messages)
        assert "adding SeaDex's recommended release" not in joined

    def test_all_skipped_returns_false_and_emits_nothing(self) -> None:
        logged, messages = _action_messages([])

        assert logged is False
        assert messages == []

    def test_recommended_group_tags_are_sorted(self) -> None:
        # G2: the tags ride a frozenset (StrEnum hash is per-process random), so
        # the producer sorts them; feed several unordered and pin the emitted order.
        reporter, events = _record()
        seadex_dict = {
            "NAN0": rg_group(
                {"u": url_item(download=True)},
                tags=frozenset({Tag.HDR, Tag.DOLBY_VISION, Tag.BROKEN}),
            ),
        }
        reporter.log_seadex_action(
            seadex_dict,
            [ReleaseOutcome(outcome=AddOutcome.ADDED, name="Show-NAN0", group="NAN0")],
        )

        action = next(e for e in events if isinstance(e, GrabAction))
        assert action.groups[0].tags == ("Broken", "Dolby Vision", "HDR")
