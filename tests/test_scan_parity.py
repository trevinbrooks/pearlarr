# pyright: strict, reportPrivateUsage=false
# reportPrivateUsage is off for the dual-list parity pins alone: the producer's
# _TIP_PRECEDENCE and the builder's _TIP_TEXTS are deliberately private twins.
"""Golden harness for the scan output: the grammar contract.

Every golden in `scan_goldens` was captured by RUNNING the current
`RunReporter` (not hand-derived): each scenario pins the exact
`(level, message, payload)` records the reporter emits today. The harness tests
in this file re-drive the real reporter and assert byte/payload equality, so the
goldens stay honest. The builder/console tests in `test_output_scan_render`
reuse the same data, pinning the new event-driven rendering layer to this
grammar. The producer rewrite (a later band) must keep every one of these green.

Each scenario also carries its EVENT form (the `output.events` value a future
producer will emit), so the golden is one shared source: reporter call -> lines,
event -> the same lines.
"""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, get_args

import pytest
from rich.text import Span, Text
from seadex import Tag

from pearlarr.arr_run_state import ArrRunState
from pearlarr.cache import AbstractCacheStore, CachedTitleUpdate
from pearlarr.config import Arr
from pearlarr.output import (
    Accent,
    CapReached,
    Event,
    GrabAction,
    GrabFailed,
    ReleaseSkipped,
    ScopeClosed,
    ScopeOpened,
    Severity,
    SeverityCounter,
    StyledValue,
    TitleDetail,
    TitleHeader,
    TitleRow,
)
from pearlarr.output.console_text import KeyValue
from pearlarr.output.scan_lines import _TIP_TEXTS, ScanEvent
from pearlarr.releases import AnotherTitlesCounts, SeadexDict, SkipReason
from pearlarr.reporter import _TIP_PRECEDENCE, RunReporter
from pearlarr.torrent_service import AddStatus, GrabResult
from pearlarr.wait_types import ImportWaitMode, PendingImportState

from .builders import (
    FakeCacheStore,
    ScriptedAniListClient,
    make_anilist_gateway,
    make_entry_record,
    make_release,
    pending_import,
    rg_group,
)
from .fakes import SCAN_EVENT_TYPES, scan_lines_from_events
from .scan_goldens import (
    ACTION_DOWNLOADING_NO_WAIT_LINES,
    ACTION_DOWNLOADING_WAITING_LINES,
    ACTION_DRY_LINES,
    ACTION_FRESH_LINES,
    ACTION_MIXED_LINES,
    CACHED_BARE_LINES,
    CACHED_FULL_LINES,
    CAP_REACHED_LINES,
    CHECKING_FULL_LINES,
    CHECKING_URL_ONLY_LINES,
    IGNORED_LINES,
    IGNORED_TITLED_LINES,
    ITEM_STARTED_RADARR_LINES,
    ITEM_STARTED_SONARR_LINES,
    NO_ENTRY_ARR_TITLE_LINES,
    NO_ENTRY_RESOLVED_LINES,
    NO_ENTRY_UNRESOLVED_LINES,
    NO_EPISODES_LINES,
    NO_MAPPING_LINES,
    NO_RELEASES_LINES,
    PENDING_DOWNLOADED_LINES,
    PENDING_IMPORTED_LINES,
    PENDING_QUEUED_LINES,
    SCAN_STARTED_RADARR_LINES,
    SCAN_STARTED_SONARR_LINES,
    SUMMARY_DRY_HAS_CLIENT_LINES,
    SUMMARY_DRY_NO_CLIENT_LINES,
    SUMMARY_MINIMAL_LINES,
    SUMMARY_RICH_LINES,
    SUMMARY_WAIT_OFF_LINES,
    UNMONITORED_LINES,
    UNREACHABLE_LINES,
    Line,
    dry_run_stats,
    dry_stats,
    rich_stats,
    srow,
    torrent_text,
    wait_off_stats,
)

_I = logging.INFO
_W = logging.WARNING

# The one tag the action-block goldens use. Deliberately a single tag: today's
# reporter joins a frozenset, so multi-tag order is hash-random across processes.
_HDR_TAG = Tag.HDR

# The pending import behind the PENDING_* goldens.
_PENDING_IMPORT_KWARGS = {"title": "My Show", "window_coverage": "S01 E01-E13", "url": "https://releases.moe/1"}


# --- the harness: drive the REAL reporter, assert the goldens --------------------------


class _Harness:
    """A real RunReporter (real gateway, faked leaves) recording emitted events."""

    def __init__(self, store: AbstractCacheStore | None = None, title: str | None = None) -> None:
        # The parity lines come from the recorded EVENTS below, never from log records.
        self.events: list[Event] = []
        # The summary's issues row diffs this bound counter (scripted directly).
        self.counts = SeverityCounter()
        self.reporter = RunReporter(
            emit=self.events.append,
            counts=lambda: self.counts,
            cache_store=store if store is not None else FakeCacheStore(),
            anilist=make_anilist_gateway(ScriptedAniListClient(title)),
        )

    def lines(self) -> tuple[Line, ...]:
        """Re-derive the scan lines from the recorded events through the shipped builders."""

        return tuple((line.level, line.message, line.payload) for line in scan_lines_from_events(self.events))


def _seeded_store() -> FakeCacheStore:
    store = FakeCacheStore()
    store.update_cached_title(
        Arr.SONARR,
        1,
        CachedTitleUpdate(name="Cached Show", coverage="S01 E01-E12", url="https://releases.moe/20997"),
    )
    store.update_cached_title(Arr.SONARR, 2, CachedTitleUpdate(name="Bare Show"))
    return store


class TestBannerParity:
    """`log_arr_start` / `log_arr_item_start` reproduce the pinned banner/item-start golden lines per arr."""

    def test_arr_start(self) -> None:
        harness = _Harness()
        harness.reporter.log_arr_start(Arr.SONARR, 3)
        assert harness.lines() == SCAN_STARTED_SONARR_LINES

        harness.events.clear()
        harness.reporter.log_arr_start(Arr.RADARR, 1)
        assert harness.lines() == SCAN_STARTED_RADARR_LINES

    def test_item_start(self) -> None:
        harness = _Harness()
        harness.reporter.log_arr_item_start(Arr.SONARR, "Frieren: Beyond Journey's End", 2, 3)
        assert harness.lines() == ITEM_STARTED_SONARR_LINES

        harness.events.clear()
        harness.reporter.log_arr_item_start(Arr.RADARR, "Perfect Blue", 1, 1)
        assert harness.lines() == ITEM_STARTED_RADARR_LINES


class TestTitleRowParity:
    """The self-contained title rows (unmonitored, no-mapping, no-episodes, ignored) reproduce their golden lines."""

    def test_unmonitored(self) -> None:
        harness = _Harness()
        harness.reporter.log_arr_item_unmonitored(ArrRunState(arr=Arr.SONARR), "Unmonitored Show")
        assert harness.lines() == UNMONITORED_LINES

    def test_no_mapping(self) -> None:
        harness = _Harness()
        harness.reporter.log_no_anilist_mappings(ArrRunState(arr=Arr.SONARR), "Unmapped Show")
        assert harness.lines() == NO_MAPPING_LINES

    def test_no_episodes(self) -> None:
        harness = _Harness()
        harness.reporter.log_no_episodes("Episodeless Show")
        assert harness.lines() == NO_EPISODES_LINES

    def test_ignored(self) -> None:
        harness = _Harness()
        harness.reporter.log_ignored_anilist_id(ArrRunState(arr=Arr.SONARR), 123)
        assert harness.lines() == IGNORED_LINES

    def test_ignored_with_the_arr_title(self) -> None:
        harness = _Harness()
        harness.reporter.log_ignored_anilist_id(ArrRunState(arr=Arr.SONARR, arr_title="Ignored Show"), 123)
        assert harness.lines() == IGNORED_TITLED_LINES


class TestTitleHeaderParity:
    """`open_title_block` / `log_cached_title` / `log_pending_import` reproduce the pinned title-header golden lines.

    Covers the checking, cached, and pending states.
    """

    def test_checking_with_coverage_url_and_incomplete(self) -> None:
        harness = _Harness()
        entry = make_entry_record(url="https://releases.moe/111852", is_incomplete=True)
        harness.reporter.open_title_block(ArrRunState(arr=Arr.SONARR), "Frieren", entry, coverage="S01 E01-E28")
        assert harness.lines() == CHECKING_FULL_LINES

    def test_checking_url_only(self) -> None:
        harness = _Harness()
        entry = make_entry_record(url="https://releases.moe/437")
        harness.reporter.open_title_block(ArrRunState(arr=Arr.RADARR), "Perfect Blue", entry)
        assert harness.lines() == CHECKING_URL_ONLY_LINES

    def test_cached_full_row(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_cached_title(ArrRunState(arr=Arr.SONARR), 1)
        assert harness.lines() == CACHED_FULL_LINES

    def test_cached_without_coverage_or_url_renders_just_the_row(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_cached_title(ArrRunState(arr=Arr.SONARR), 2)
        assert harness.lines() == CACHED_BARE_LINES

    def test_pending_import_snapshots(self) -> None:
        harness = _Harness()
        pending = pending_import(**_PENDING_IMPORT_KWARGS)
        for state, expected in (
            (PendingImportState.DOWNLOADING, PENDING_QUEUED_LINES),
            (PendingImportState.DOWNLOADED, PENDING_DOWNLOADED_LINES),
            (PendingImportState.IMPORTED, PENDING_IMPORTED_LINES),
        ):
            harness.events.clear()
            harness.reporter.log_pending_import(state, pending, pending.claims[0].series_id)
            assert harness.lines() == expected


class TestTitledRowParity:
    """`log_no_sd_entry` / `log_seadex_unreachable` reproduce the pinned golden lines.

    A title's name is resolved via the AniList gateway, else the arr item's own name, else the id form alone.
    """

    def test_no_entry_with_resolved_name(self) -> None:
        harness = _Harness(title="Resolved Title")
        harness.reporter.log_no_sd_entry(ArrRunState(arr=Arr.SONARR), 42)
        assert harness.lines() == NO_ENTRY_RESOLVED_LINES

    def test_no_entry_with_the_arr_title_fallback(self) -> None:
        harness = _Harness(title=None)
        state = ArrRunState(arr=Arr.SONARR)
        state.arr_title = "Arr Title"
        harness.reporter.log_no_sd_entry(state, 42)
        assert harness.lines() == NO_ENTRY_ARR_TITLE_LINES

    def test_no_entry_without_a_name(self) -> None:
        harness = _Harness(title=None)
        harness.reporter.log_no_sd_entry(ArrRunState(arr=Arr.SONARR), 77)
        assert harness.lines() == NO_ENTRY_UNRESOLVED_LINES

    def test_seadex_unreachable_with_cached_name(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_seadex_unreachable(ArrRunState(arr=Arr.SONARR), 1)
        assert harness.lines() == UNREACHABLE_LINES


class TestDetailParity:
    """`log_no_seadex_releases` reproduces the pinned no-suitable-releases golden line."""

    def test_no_suitable_releases(self) -> None:
        harness = _Harness()
        harness.reporter.log_no_seadex_releases(ArrRunState(arr=Arr.SONARR))
        assert harness.lines() == NO_RELEASES_LINES


def _xdetail(level: int, message: str, key: str, value: str, style: str) -> Line:
    """An external detail row at the site's own level (else exactly the shape of the detail rows in `scan_goldens`)."""

    return (level, message, KeyValue(key=key, value=value, key_width=9, value_style=style, indent=2, sep=""))


@dataclass(frozen=True, slots=True)
class _DetailSite:
    """One migrated producer site: the `reporter.detail` args + its pinned line."""

    label: str
    value: StyledValue
    severity: Severity
    golden: Line


@dataclass(frozen=True, slots=True)
class _FactSite:
    """One flipped producer site: the typed fact `reporter.post` takes + its pinned line."""

    fact: ReleaseSkipped | GrabFailed
    golden: Line


class TestExternalDetailParity:
    """The external producer sites: grab pipeline, release filter, Sonarr episode coverage.

    Each golden was captured by running the pre-migration `LogFormatter.detail` with the site's literal args at
    fc58aef. The reporter must reproduce every line byte-identically through the shipped `scan_lines` builders (the
    grammar every rendering surface consumes). The four grab-pipeline sites emit typed `reporter.post` facts - the
    golden `Line` tuples stay verbatim. The planner notice / missing / status sites stay `reporter.detail`-driven.
    """

    FACT_SITES: tuple[_FactSite, ...] = (
        # grab_pipeline._grab_release: a private-only release.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.PRIVATE_ONLY, url="https://x/1"),
            _xdetail(
                _W,
                "    skipped   GroupA on Nyaa (private-only)",
                "skipped",
                "GroupA on Nyaa (private-only)",
                "yellow",
            ),
        ),
        # grab_pipeline._grab_release: a tracker outside the configured tracker list.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.TRACKER_NOT_SELECTED, url="https://x/1"),
            _xdetail(
                _I,
                "    skipped   https://x/1 (tracker Nyaa not in your selected list)",
                "skipped",
                "https://x/1 (tracker Nyaa not in your selected list)",
                "yellow",
            ),
        ),
        # grab_pipeline._grab_release: a tracker with no parser yet.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.UNSUPPORTED_TRACKER, url="https://x/1"),
            _xdetail(
                _W,
                "    skipped   https://x/1 (tracker Nyaa not yet supported)",
                "skipped",
                "https://x/1 (tracker Nyaa not yet supported)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: a pack whose file numbers disagree, or a specials pack numbered another way.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.MISNUMBERED, url="https://x/1"),
            _xdetail(
                _W,
                "    skipped   https://x/1 (numbering does not match its SeaDex listing)",
                "skipped",
                "https://x/1 (numbering does not match its SeaDex listing)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: a pack whose file titles contradict its numbers, none of it placed.
        _FactSite(
            ReleaseSkipped(
                group="GroupA", tracker="Nyaa", reason=SkipReason.EPISODE_TITLES_CONTRADICT, url="https://x/1"
            ),
            _xdetail(
                _W,
                "    skipped   https://x/1 (episode titles contradict its numbers)",
                "skipped",
                "https://x/1 (episode titles contradict its numbers)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: a release whose files place under another title, some of them on none.
        _FactSite(
            ReleaseSkipped(
                group="GroupA",
                tracker="Nyaa",
                reason=SkipReason.PLACED_ELSEWHERE,
                url="https://x/1",
                counts=AnotherTitlesCounts(another_titles=12, unplaced=6),
            ),
            _xdetail(
                _W,
                "    skipped   https://x/1 (12 files are another title's, 6 could not be placed)",
                "skipped",
                "https://x/1 (12 files are another title's, 6 could not be placed)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: a release whose one file is another title's.
        _FactSite(
            ReleaseSkipped(
                group="GroupA",
                tracker="Nyaa",
                reason=SkipReason.PLACED_ELSEWHERE,
                url="https://x/1",
                counts=AnotherTitlesCounts(another_titles=1, unplaced=0),
            ),
            _xdetail(
                _W,
                "    skipped   https://x/1 (1 file is another title's)",
                "skipped",
                "https://x/1 (1 file is another title's)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: a release whose files are of aired episodes Sonarr's order has no row for.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.AIRED_ONLY, url="https://x/1"),
            _xdetail(
                _I,
                "    skipped   https://x/1 (its episodes are not in Sonarr's order of the series)",
                "skipped",
                "https://x/1 (its episodes are not in Sonarr's order of the series)",
                "yellow",
            ),
        ),
        # grab_pipeline._may_grab: the listing the placement judges a pack by could not be read.
        _FactSite(
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.PLACEMENT_INPUT_UNREAD, url="https://x/1"),
            _xdetail(
                _W,
                "    skipped   https://x/1 (a read the placement needs failed)",
                "skipped",
                "https://x/1 (a read the placement needs failed)",
                "yellow",
            ),
        ),
        # grab_pipeline._grab_release: a contained grab failure.
        _FactSite(
            GrabFailed(group="GroupA", url="https://x/1", error="tracker down"),
            _xdetail(
                _W,
                "    failed    could not grab https://x/1: tracker down; will retry next run",
                "failed",
                "could not grab https://x/1: tracker down; will retry next run",
                "yellow",
            ),
        ),
    )

    SITES: tuple[_DetailSite, ...] = (
        # grab_pipeline.grab_and_cache: nothing to grab (the blue NOTE accent).
        _DetailSite(
            "status",
            StyledValue("already have the recommended release", Accent.NOTE),
            Severity.INFO,
            _xdetail(
                _I,
                "    status    already have the recommended release",
                "status",
                "already have the recommended release",
                "blue",
            ),
        ),
        # seadex_filter.plan_grabs: a planner PrivateOnlyNotice (WARNING form).
        _DetailSite(
            "skipped",
            StyledValue("GroupA, GroupB private-only (private releases not supported)", Accent.CAUTION),
            Severity.WARNING,
            _xdetail(
                _W,
                "    skipped   GroupA, GroupB private-only (private releases not supported)",
                "skipped",
                "GroupA, GroupB private-only (private releases not supported)",
                "yellow",
            ),
        ),
        # arr_run_services.log_missing_episodes: missing-episode coverage.
        _DetailSite(
            "missing",
            StyledValue("S01 E12", Accent.CAUTION),
            Severity.INFO,
            _xdetail(
                _I,
                "    missing   S01 E12",
                "missing",
                "S01 E12",
                "yellow",
            ),
        ),
    )

    def test_reporter_post_reproduces_the_goldens(self) -> None:
        # The flipped sites post typed facts. The shipped builders re-derive the
        # exact (level, message, payload) lines the detail-era sites pinned.
        harness = _Harness()

        for site in self.FACT_SITES:
            harness.reporter.post(site.fact)

        assert harness.lines() == tuple(site.golden for site in self.FACT_SITES)

    def test_reporter_detail_reproduces_the_goldens(self) -> None:
        # The real reporter emits TitleDetail events. The shipped builders
        # re-derive the exact (level, message, payload) lines the sites pinned.
        harness = _Harness()

        for site in self.SITES:
            harness.reporter.detail(site.label, site.value, severity=site.severity)

        assert harness.lines() == tuple(site.golden for site in self.SITES)

    def test_post_carries_the_open_title_scope_then_emits_scope_free(self) -> None:
        # Inside an open title the fact carries the title scope. After a boundary
        # closes it, a post emits scope-free (col-0 by design, like _post).
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.open_title_block(state, "Show", make_entry_record())
        harness.reporter.post(self.FACT_SITES[0].fact)
        harness.reporter.log_arr_item_start(Arr.SONARR, "Next", 1, 1)  # closes the title scope
        harness.reporter.post(next(site.fact for site in self.FACT_SITES if isinstance(site.fact, GrabFailed)))

        opened = next(e for e in harness.events if isinstance(e, ScopeOpened))
        skipped = next(e for e in harness.events if isinstance(e, ReleaseSkipped))
        failed = next(e for e in harness.events if isinstance(e, GrabFailed))
        assert skipped.scope == opened.scope
        assert failed.scope is None


class TestActionBlockParity:
    """`log_seadex_action` / `log_max_torrents_added` reproduce the pinned action-block golden lines.

    Covers fresh, dry-run, already-in-qBittorrent, and mixed-outcome renders.
    """

    _SEADEX_DICT: ClassVar[SeadexDict] = {
        "GroupA": rg_group({"https://nyaa.si/view/1": make_release(download=True)}, tags=frozenset({_HDR_TAG})),
        "GroupB": rg_group({"https://nyaa.si/view/2": make_release(download=True)}),
    }
    _FRESH_RESULTS: ClassVar[list[GrabResult]] = [
        GrabResult(outcome=AddStatus.ADDED, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
        GrabResult(outcome=AddStatus.ADDED, name=None, group="GroupB"),
    ]

    def test_fresh_add(self) -> None:
        harness = _Harness()
        harness.reporter.log_seadex_action(
            self._SEADEX_DICT, self._FRESH_RESULTS, dry_run=False, waiting_to_import=False
        )
        assert harness.lines() == ACTION_FRESH_LINES

    def test_dry_run(self) -> None:
        harness = _Harness()
        harness.reporter.log_seadex_action(
            self._SEADEX_DICT, self._FRESH_RESULTS, dry_run=True, waiting_to_import=False
        )
        assert harness.lines() == ACTION_DRY_LINES

    def test_already_in_qbit_with_and_without_waiting_to_import(self) -> None:
        harness = _Harness()
        seadex_dict = {
            "GroupA": rg_group({"https://nyaa.si/view/1": make_release(download=True)}, tags=frozenset({_HDR_TAG})),
        }
        results = [
            GrabResult(outcome=AddStatus.ALREADY_IN_QBIT, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
        ]
        harness.reporter.log_seadex_action(seadex_dict, results, dry_run=False, waiting_to_import=True)
        assert harness.lines() == ACTION_DOWNLOADING_WAITING_LINES

        harness.events.clear()
        harness.reporter.log_seadex_action(seadex_dict, results, dry_run=False, waiting_to_import=False)
        assert harness.lines() == ACTION_DOWNLOADING_NO_WAIT_LINES

    def test_mixed_results_group_added_before_already_in_qbit(self) -> None:
        # Already-in-qBittorrent-first input still renders added-first: the grouping, not
        # results order, is the pinned contract.
        harness = _Harness()
        results = [
            GrabResult(outcome=AddStatus.ALREADY_IN_QBIT, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
            GrabResult(outcome=AddStatus.ADDED, name="[GroupB] Frieren S01 1080p.mkv", group="GroupB"),
        ]
        harness.reporter.log_seadex_action(self._SEADEX_DICT, results, dry_run=False, waiting_to_import=False)
        assert harness.lines() == ACTION_MIXED_LINES

    def test_max_torrents(self) -> None:
        harness = _Harness()
        harness.reporter.log_max_torrents_added(5)
        assert harness.lines() == CAP_REACHED_LINES


def _summary_lines(
    state: ArrRunState,
    *,
    dry_run: bool,
    has_client: bool,
    warnings: int = 0,
    errors: int = 0,
) -> tuple[Line, ...]:
    """Capture exactly what log_arr_run_summary emits, with scripted issue deltas."""

    harness = _Harness()
    # Production shape: the arr run's start takes the mark (bound to the harness counter),
    # then the arr run's issues accrue on it.
    state.counts_mark = harness.reporter.counts_mark()
    # Scripted issues go on the bound counter directly (the issues-row delta).
    # They produce no events, so only the summary's feed the re-derived lines.
    for _ in range(warnings):
        harness.counts.record(Severity.WARNING)
    for _ in range(errors):
        harness.counts.record(Severity.ERROR)
    harness.reporter.log_arr_run_summary(state, dry_run=dry_run, has_client=has_client)
    return harness.lines()


class TestArrRunSummaryParity:
    """`log_arr_run_summary` reproduces the pinned summary golden lines across rich/dry/minimal/wait-off states.

    Includes the issues row's style ladder (plain, yellow, bold-red).
    """

    def test_rich_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A frozen clock pins the elapsed row (started at 38s, "now" 100s -> 1m 02s).
        monkeypatch.setattr(time, "monotonic", lambda: 100.0)
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING, stats=rich_stats())
        state.torrents_added = 2
        state.started_monotonic = 38.0
        assert _summary_lines(state, dry_run=False, has_client=True, warnings=2, errors=1) == SUMMARY_RICH_LINES

    def test_dry_run_with_client(self) -> None:
        state = ArrRunState(arr=Arr.SONARR, stats=dry_stats())
        state.torrents_added = 1
        assert _summary_lines(state, dry_run=True, has_client=True) == SUMMARY_DRY_HAS_CLIENT_LINES

    def test_dry_run_without_client(self) -> None:
        state = ArrRunState(arr=Arr.RADARR, stats=dry_run_stats())
        state.torrents_added = 1
        assert _summary_lines(state, dry_run=True, has_client=False) == SUMMARY_DRY_NO_CLIENT_LINES

    def test_minimal_all_zeros(self) -> None:
        state = ArrRunState(arr=Arr.SONARR)
        assert _summary_lines(state, dry_run=False, has_client=True) == SUMMARY_MINIMAL_LINES

    def test_wait_mode_off_hides_pending_import_rows(self) -> None:
        state = ArrRunState(arr=Arr.SONARR, stats=wait_off_stats())
        assert _summary_lines(state, dry_run=False, has_client=True) == SUMMARY_WAIT_OFF_LINES

    def test_warnings_only_issues_row_renders_yellow(self) -> None:
        # The style ladder's middle rung (warnings without errors) has no golden
        # of its own: bold-red needs errors, yellow needs warnings alone.
        state = ArrRunState(arr=Arr.SONARR)
        lines = _summary_lines(state, dry_run=False, has_client=True, warnings=2)
        expected = srow("  issues       : 2 warnings, 0 errors", "issues", "2 warnings, 0 errors", style="yellow")
        assert expected in lines


class TestTorrentValuePins:
    """The summary's "torrent" Text values, pinned by content: plain text, spans, and base style.

    Golden equality above can't hide a `group_highlight` regression on its own.
    """

    def test_leading_group_is_highlighted_in_place(self) -> None:
        value = torrent_text("[GroupA] Frieren S01 1080p.mkv", "GroupA", dry=False)
        assert isinstance(value, Text)
        assert value.plain == "[GroupA] Frieren S01 1080p.mkv"
        assert value.spans == [Span(1, 7, "cyan")]
        assert value.style == "green"

    def test_name_less_grab_prepends_the_group_with_trailing_space(self) -> None:
        value = torrent_text(None, "PrivGrp", dry=False)
        assert isinstance(value, Text)
        assert value.plain == "[PrivGrp] "
        assert value.spans == [Span(1, 8, "cyan")]

    def test_dry_run_dims_group_and_base(self) -> None:
        value = torrent_text("Dry.Show.S01.1080p-GroupB", "GroupB", dry=True)
        assert isinstance(value, Text)
        assert value.plain == "[GroupB] Dry.Show.S01.1080p-GroupB"
        assert value.spans == [Span(1, 7, "grey50")]
        assert value.style == "grey50"


class TestScopeLifecycle:
    """A1/A3: the title-scope open/close discipline over multi-method sequences.

    Asserts on the RAW recorded event stream (scope boundaries included). Serials
    mint from the process-wide minter, so the checks are RELATIONAL - which scope
    opens/closes in which order, which header carries which id - never absolute
    serial values.
    """

    def test_item_title_title_item_sequence(self) -> None:
        # item -> title -> title -> item: each new title closes the previous, and
        # the closing item boundary closes the last one.
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.log_arr_item_start(Arr.SONARR, "Show", 1, 1)
        harness.reporter.open_title_block(state, "First", make_entry_record())
        harness.reporter.open_title_block(state, "Second", make_entry_record())
        harness.reporter.log_arr_item_start(Arr.SONARR, "Next", 1, 1)

        boundaries = [e for e in harness.events if isinstance(e, (ScopeOpened, ScopeClosed))]
        assert len(boundaries) == 4
        first_open, first_close, second_open, second_close = boundaries
        # Each title is closed before the next one opens (one title at a time).
        assert isinstance(first_open, ScopeOpened) and first_open.label == "First"
        assert isinstance(first_close, ScopeClosed) and first_close.scope == first_open.scope
        assert isinstance(second_open, ScopeOpened) and second_open.label == "Second"
        assert isinstance(second_close, ScopeClosed) and second_close.scope == second_open.scope
        # Every header carries the id of the scope opened immediately before it.
        headers = [e for e in harness.events if isinstance(e, TitleHeader)]
        assert [h.scope for h in headers] == [first_open.scope, second_open.scope]

    def test_bare_titled_rows_open_no_scope(self) -> None:
        # The self-contained title rows never open a title scope (col-0 rows).
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.log_arr_item_unmonitored(state, "Unmon")
        harness.reporter.log_no_anilist_mappings(state, "NoMap")
        harness.reporter.log_ignored_anilist_id(state, 7)
        harness.reporter.log_no_episodes("NoEps")

        assert not any(isinstance(e, (ScopeOpened, ScopeClosed)) for e in harness.events)
        assert all(isinstance(e, TitleRow) for e in harness.events)

    def test_close_is_idempotent(self) -> None:
        # A second boundary with no open title emits no spurious ScopeClosed.
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.open_title_block(state, "Show", make_entry_record())
        harness.reporter.log_arr_item_start(Arr.SONARR, "A", 1, 1)  # closes the title scope
        harness.reporter.log_arr_item_start(Arr.SONARR, "B", 1, 1)  # nothing left to close

        assert len([e for e in harness.events if isinstance(e, ScopeClosed)]) == 1

    def test_action_keeps_the_title_open_for_a_trailing_diagnostic(self) -> None:
        # A1(b): log_seadex_action POSTS but does not close, so the title stays among
        # the open scopes - a diagnostic emitted before the next boundary indents
        # under this title (live-verified), not col-0.
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.open_title_block(state, "Show", make_entry_record())
        harness.reporter.log_seadex_action(
            {},
            [GrabResult(outcome=AddStatus.ADDED, name="X", group="G")],
            dry_run=False,
            waiting_to_import=False,
        )

        assert not any(isinstance(e, ScopeClosed) for e in harness.events)
        opened = next(e for e in harness.events if isinstance(e, ScopeOpened))
        action = next(e for e in harness.events if isinstance(e, GrabAction))
        assert action.scope == opened.scope

    def test_fresh_reporter_has_no_open_title(self) -> None:
        # _title_scope starts None on a fresh reporter, so the first current-title
        # detail (no header before it) opens no scope and goes scope-free.
        harness = _Harness()
        harness.reporter.log_no_seadex_releases(ArrRunState(arr=Arr.SONARR))

        assert not any(isinstance(e, ScopeOpened) for e in harness.events)
        detail = next(e for e in harness.events if isinstance(e, TitleDetail))
        assert detail.scope is None

    def test_cap_closes_the_open_title(self) -> None:
        # The scan breaks at the cap and _finish_arr_run's reconcile runs before
        # the summary: a still-open TITLE scope would misplace its diagnostics.
        harness = _Harness()
        state = ArrRunState(arr=Arr.SONARR)
        harness.reporter.open_title_block(state, "Show", make_entry_record())
        harness.reporter.log_max_torrents_added(5)

        opened = next(e for e in harness.events if isinstance(e, ScopeOpened))
        closes = [e for e in harness.events if isinstance(e, ScopeClosed)]
        assert [c.scope for c in closes] == [opened.scope]
        # The close lands BEFORE the cap event in the stream.
        assert harness.events.index(closes[0]) < harness.events.index(
            next(e for e in harness.events if isinstance(e, CapReached)),
        )


# --- dual-list drift pins ---------------------------------------------------------


def test_scan_event_types_match_the_alias() -> None:
    """The fakes filter tuple can't rot behind the `ScanEvent` union.

    A new scan event missing from it would silently vanish from every re-derived golden.
    """

    members: set[object] = set(get_args(ScanEvent.__value__))
    assert set(SCAN_EVENT_TYPES) == members


def test_tip_precedence_covers_exactly_the_tip_texts() -> None:
    """Producer precedence (`_TIP_PRECEDENCE`) and builder texts (`_TIP_TEXTS`) are hand-maintained twins.

    An entry in one without the other silently drops the tip line (unranked -> tip=None, unmapped -> no text).
    """

    assert set(_TIP_PRECEDENCE) == set(_TIP_TEXTS)
