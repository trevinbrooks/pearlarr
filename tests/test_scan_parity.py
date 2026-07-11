# pyright: strict, reportPrivateUsage=false
# reportPrivateUsage is off for the dual-list parity pins alone: the producer's
# _TIP_PRECEDENCE and the builder's _TIP_TEXTS are deliberately private twins.
"""Golden harness for the scan surface: the grammar contract.

Every constant below was captured by RUNNING the current `RunReporter` (not
hand-derived): each scenario pins the exact `(level, message, payload)` records
the reporter emits today. The harness tests in this file re-drive the real
reporter and assert byte/payload equality, so the goldens stay honest; the
builder/console tests in `test_output_scan_render` reuse the same data,
pinning the new event-driven rendering layer to this grammar. The producer
rewrite (a later band) must keep every one of these green.

Each scenario also carries its EVENT form (the `output.events` value a future
producer will emit), so the golden is one shared spine: reporter call -> lines,
event -> the same lines.
"""

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, get_args, override

import httpx
import pytest
from rich.text import Span, Text
from seadex import Tag

from pearlarr.modules.anilist_client import AniListClient
from pearlarr.modules.anilist_gateway import AniListGateway
from pearlarr.modules.cache import AbstractCacheStore, CacheRecord
from pearlarr.modules.config import Arr
from pearlarr.modules.log import (
    ConsoleRender,
    EntryState,
    KvLine,
    SectionRule,
    StyledLine,
    TitledRule,
    group_highlight,
)
from pearlarr.modules.manual_import import ImportWaitMode, PendingState
from pearlarr.modules.output import (
    Accent,
    CapReached,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    NeedsActionCause,
    RecommendedGroup,
    ReleaseName,
    ReleaseSkipped,
    RunSummary,
    RunSummaryReady,
    RunTally,
    ScanStarted,
    ScopeClosed,
    ScopeOpened,
    Severity,
    SeverityCounts,
    SkipReason,
    StyledValue,
)
from pearlarr.modules.output.scan_lines import _TIP_TEXTS, ScanEvent
from pearlarr.modules.reporter import (
    _TIP_PRECEDENCE,
    GrabRecord,
    NeedsActionKind,
    NeedsActionRecord,
    RunContext,
    RunReporter,
    RunStats,
)
from pearlarr.modules.torrents import AddOutcome, ReleaseOutcome

from .builders import FakeCacheStore, make_entry_record, pending_import, rg_group, url_item
from .fakes import SCAN_EVENT_TYPES, scan_lines_from_events

type Line = tuple[int, str, ConsoleRender | None]
"""One pinned line: (levelno, plain message, ConsoleRender payload)."""

_I = logging.INFO
_W = logging.WARNING

# The one tag the action-block goldens use. Deliberately a single tag: today's
# reporter joins a frozenset, so multi-tag order is hash-random across runs.
_HDR_TAG = Tag.HDR


# --- golden-line factories (literal payload compression, messages stay verbatim) ---


def _blank() -> Line:
    return (_I, "", None)


def _titled(message: str, *, title: str | None = None, heavy: bool = False) -> Line:
    return (_I, message, TitledRule(title=title if title is not None else message, heavy=heavy))


def _styled(message: str, style: str) -> Line:
    return (_I, message, StyledLine(style=style))


def _detail(message: str, key: str, value: str, style: str | None = "grey50", tail: str | None = None) -> Line:
    """An entry-detail row: the colon-less gutter kv at indent 2, key width 9."""

    return (_I, message, KvLine(key=key, value=value, key_width=9, value_style=style, indent=2, sep="", tail=tail))


def _srow(message: str, key: str, value: str, style: str | None = None) -> Line:
    """A summary scoreboard row: "key : value" at indent 1, key width 12."""

    return (_I, message, KvLine(key=key, value=value, key_width=12, value_style=style, indent=1))


def _brow(message: str, key: str, value: str | Text, style: str) -> Line:
    """A summary per-entry block row: gutter kv at indent 3, key width 7."""

    return (_I, message, KvLine(key=key, value=value, key_width=7, value_style=style, indent=3, sep=""))


def _torrent_text(name: str | None, group: str, *, dry: bool) -> str | Text:
    """The summary "torrent" value exactly as the reporter builds it."""

    dim = "grey50"
    return group_highlight(name, group, group_style=dim if dry else "cyan", base_style=dim if dry else "green")


# --- run/item banners ---------------------------------------------------------------

SCAN_STARTED_SONARR = ScanStarted(arr=Arr.SONARR, total=3)
SCAN_STARTED_SONARR_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Starting Pearlarr (Sonarr) for 3 series", heavy=True),
)

SCAN_STARTED_RADARR = ScanStarted(arr=Arr.RADARR, total=1)
SCAN_STARTED_RADARR_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Starting Pearlarr (Radarr) for 1 movie", heavy=True),
)

ITEM_STARTED_SONARR = ItemStarted(arr=Arr.SONARR, index=2, total=3, title="Frieren: Beyond Journey's End")
ITEM_STARTED_SONARR_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("[2/3] Sonarr: Frieren: Beyond Journey's End"),
)

ITEM_STARTED_RADARR = ItemStarted(arr=Arr.RADARR, index=1, total=1, title="Perfect Blue")
ITEM_STARTED_RADARR_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("[1/1] Radarr: Perfect Blue"),
)

# --- self-contained ledger rows -------------------------------------------------------

UNMONITORED_ROW = LedgerRow(state=EntryState.UNMONITORED, label="Unmonitored Show")
UNMONITORED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  unmonitored Unmonitored Show", "grey50"),
)

NO_MAPPING_ROW = LedgerRow(state=EntryState.NO_MAPPING, label="Unmapped Show")
NO_MAPPING_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no mapping  Unmapped Show", "grey50"),
)

IGNORED_ROW = LedgerRow(state=EntryState.IGNORED, label="AniList #123")
IGNORED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  ignored     AniList #123", "grey50"),
)

# --- entry headers (row + coverage/link continuation) ---------------------------------

CHECKING_FULL = EntryHeader(
    state=EntryState.CHECKING,
    title="Frieren",
    al_id=1,
    coverage="S01 E01-E28",
    url="https://releases.moe/111852",
    incomplete=True,
)
CHECKING_FULL_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  checking    Frieren", ""),
    _detail("    files     S01 E01-E28", "files", "S01 E01-E28"),
    _detail(
        "    link      https://releases.moe/111852",
        "link",
        "https://releases.moe/111852",
        tail="(marked incomplete on SeaDex)",
    ),
)

CHECKING_URL_ONLY = EntryHeader(
    state=EntryState.CHECKING,
    title="Perfect Blue",
    al_id=1,
    url="https://releases.moe/437",
)
CHECKING_URL_ONLY_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  checking    Perfect Blue", ""),
    _detail("    link      https://releases.moe/437", "link", "https://releases.moe/437"),
)

CACHED_FULL = EntryHeader(
    state=EntryState.UNCHANGED,
    title="Cached Show",
    coverage="S01 E01-E12",
    url="https://releases.moe/20997",
)
CACHED_FULL_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  unchanged   Cached Show", "grey50"),
    _detail("    files     S01 E01-E12", "files", "S01 E01-E12"),
    _detail("    link      https://releases.moe/20997", "link", "https://releases.moe/20997"),
)

CACHED_IN_RADARR = EntryHeader(state=EntryState.IN_RADARR, title="Cached Movie", url="https://releases.moe/64")
CACHED_IN_RADARR_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  in radarr   Cached Movie", "grey50"),
    _detail("    link      https://releases.moe/64", "link", "https://releases.moe/64"),
)

CACHED_BARE = EntryHeader(state=EntryState.UNCHANGED, title="Bare Show")
CACHED_BARE_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  unchanged   Bare Show", "grey50"),
)

# --- titled entries with detail continuations (outage skip / no entry) ----------------

OUTAGE_ROW = LedgerRow(state=EntryState.SKIPPED, label="Cached Show")
OUTAGE_ANILIST_DETAIL = EntryDetail(label="anilist", value=StyledValue("1"))
OUTAGE_STATUS_DETAIL = EntryDetail(label="status", value=StyledValue("lookup skipped (SeaDex unreachable)", Accent.DIM))
OUTAGE_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  skipped     Cached Show", "grey50"),
    # Quirk: the anilist id repeats on its own detail row even when the name
    # came straight from the cache row (any resolved title gets the id line).
    _detail("    anilist   1", "anilist", "1", style=None),
    _detail("    status    lookup skipped (SeaDex unreachable)", "status", "lookup skipped (SeaDex unreachable)"),
)

NO_ENTRY_RESOLVED_ROW = LedgerRow(state=EntryState.NO_ENTRY, label="Resolved Title")
NO_ENTRY_RESOLVED_DETAIL = EntryDetail(label="anilist", value=StyledValue("42"))
NO_ENTRY_RESOLVED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no entry    Resolved Title", "grey50"),
    _detail("    anilist   42", "anilist", "42", style=None),
)

NO_ENTRY_UNRESOLVED_ROW = LedgerRow(state=EntryState.NO_ENTRY, label="AniList #77")
NO_ENTRY_UNRESOLVED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no entry    AniList #77", "grey50"),
)

# --- carried-over pending snapshots ----------------------------------------------------

_PENDING_KWARGS = {"title": "My Show", "coverage": "S01 E01-E13", "url": "https://releases.moe/1"}


def _pending_header(state: EntryState) -> EntryHeader:
    return EntryHeader(state=state, title="My Show · SubGroup", coverage="S01 E01-E13", url="https://releases.moe/1")


def _pending_lines(row: Line) -> tuple[Line, ...]:
    return (
        _blank(),
        row,
        _detail("    files     S01 E01-E13", "files", "S01 E01-E13"),
        _detail("    link      https://releases.moe/1", "link", "https://releases.moe/1"),
    )


PENDING_QUEUED = _pending_header(EntryState.QUEUED)
PENDING_QUEUED_LINES = _pending_lines(_styled("  queued      My Show · SubGroup", "grey50"))

PENDING_IMPORTING = _pending_header(EntryState.IMPORTING)
PENDING_IMPORTING_LINES = _pending_lines(_styled("  importing   My Show · SubGroup", "grey50"))

PENDING_IMPORTED = _pending_header(EntryState.IMPORTED)
PENDING_IMPORTED_LINES = _pending_lines(_styled("  imported    My Show · SubGroup", "green"))

# --- entry details -----------------------------------------------------------------------

NO_RELEASES_DETAIL = EntryDetail(label="status", value=StyledValue("no suitable releases on SeaDex", Accent.DIM))
NO_RELEASES_LINES: tuple[Line, ...] = (
    _detail("    status    no suitable releases on SeaDex", "status", "no suitable releases on SeaDex"),
)

# --- action blocks -----------------------------------------------------------------------

_FRESH_GROUPS = (RecommendedGroup(name="GroupA", tags=("HDR",)), RecommendedGroup(name="GroupB"))
# The name-less second release (a hashless/private torrent) falls back to its group.
_FRESH_ADDED = (
    ReleaseName(name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
    ReleaseName(name="", group="GroupB"),
)

ACTION_FRESH = GrabAction(status=GrabStatus.ADDING, groups=_FRESH_GROUPS, added=_FRESH_ADDED, downloading=())
ACTION_FRESH_LINES: tuple[Line, ...] = (
    _detail(
        "    status    adding SeaDex's recommended release",
        "status",
        "adding SeaDex's recommended release",
        style=None,
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail("    added     [GroupA] Frieren S01 1080p.mkv", "added", "[GroupA] Frieren S01 1080p.mkv", style="green"),
    _detail("    added     GroupB", "added", "GroupB", style="green"),
)

ACTION_DRY = GrabAction(status=GrabStatus.WOULD_ADD, groups=_FRESH_GROUPS, added=_FRESH_ADDED, downloading=())
ACTION_DRY_LINES: tuple[Line, ...] = (
    _detail(
        "    status    would add SeaDex's recommended release (dry run)",
        "status",
        "would add SeaDex's recommended release (dry run)",
        style=None,
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail(
        "    would add [GroupA] Frieren S01 1080p.mkv",
        "would add",
        "[GroupA] Frieren S01 1080p.mkv",
        style="green",
    ),
    _detail("    would add GroupB", "would add", "GroupB", style="green"),
)

_DOWNLOADING = (ReleaseName(name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),)
_ONE_GROUP = (RecommendedGroup(name="GroupA", tags=("HDR",)),)

ACTION_DOWNLOADING_WAITING = GrabAction(
    status=GrabStatus.ALREADY_DOWNLOADING,
    groups=_ONE_GROUP,
    added=(),
    downloading=_DOWNLOADING,
    waiting_to_import=True,
)
ACTION_DOWNLOADING_WAITING_LINES: tuple[Line, ...] = (
    _detail(
        "    status    SeaDex's pick is already downloading in qBittorrent - waiting to import",
        "status",
        "SeaDex's pick is already downloading in qBittorrent - waiting to import",
        style="yellow",
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    # Quirk: "downloading" (11 chars) overflows key width 9, so its value sits
    # one space past the key instead of in the aligned column.
    _detail(
        "    downloading [GroupA] Frieren S01 1080p.mkv",
        "downloading",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

ACTION_DOWNLOADING_NO_WAIT = GrabAction(
    status=GrabStatus.ALREADY_DOWNLOADING,
    groups=_ONE_GROUP,
    added=(),
    downloading=_DOWNLOADING,
)
ACTION_DOWNLOADING_NO_WAIT_LINES: tuple[Line, ...] = (
    _detail(
        "    status    SeaDex's pick is already downloading in qBittorrent",
        "status",
        "SeaDex's pick is already downloading in qBittorrent",
        style="yellow",
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail(
        "    downloading [GroupA] Frieren S01 1080p.mkv",
        "downloading",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

# Mixed outcomes render grouped - all added, then all downloading - regardless
# of results order (accepted Band-D delta; OLD interleaved in results order).
ACTION_MIXED = GrabAction(
    status=GrabStatus.ADDING,
    groups=_FRESH_GROUPS,
    added=(ReleaseName(name="[GroupB] Frieren S01 1080p.mkv", group="GroupB"),),
    downloading=_DOWNLOADING,
)
ACTION_MIXED_LINES: tuple[Line, ...] = (
    _detail(
        "    status    adding SeaDex's recommended release",
        "status",
        "adding SeaDex's recommended release",
        style=None,
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail("    added     [GroupB] Frieren S01 1080p.mkv", "added", "[GroupB] Frieren S01 1080p.mkv", style="green"),
    _detail(
        "    downloading [GroupA] Frieren S01 1080p.mkv",
        "downloading",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

CAP_REACHED = CapReached(cap=5)
CAP_REACHED_LINES: tuple[Line, ...] = (
    _styled(
        "Reached the maximum number of torrents for this run (advanced.max_torrents_to_add); stopping",
        "yellow",
    ),
)

# --- run summaries ------------------------------------------------------------------------

_RULE_LINE: Line = (_I, "=" * 80, SectionRule(char="="))

PRIVATE_ONLY_TIP = (
    "  Tip: manually grab private releases or set private_releases: fallback to automatically grab public alternatives."
)
NO_FALLBACK_TIP = "  Tip: no public alternative exists yet; the title is re-checked every run until one appears."
STALE_TIP = (
    "  Tip: your copies of these releases are outdated (their file sizes no longer match); "
    "update them from their private tracker, or delete the outdated files to let the "
    "public fallback stand in."
)


def _rich_stats() -> RunStats:
    return RunStats(
        checked=8,
        added=[
            GrabRecord(
                title="Frieren",
                coverage="S01 E01-E28",
                url="https://releases.moe/111852",
                name="[GroupA] Frieren S01 1080p.mkv",
                group="GroupA",
            ),
            GrabRecord(
                title="Hashless Movie",
                coverage=None,
                url="https://releases.moe/900",
                name=None,
                group="PrivGrp",
            ),
        ],
        up_to_date=1,
        cached=3,
        no_seadex_entry=2,
        seadex_unreachable=1,
        no_releases=1,
        no_mappings=1,
        needs_action=[
            NeedsActionRecord(
                title="Private Show",
                coverage="S02 E01-E24",
                group="PrivGrp",
                url="https://releases.moe/222",
                reason="private-only release; private releases not supported",
                kind=NeedsActionKind.PRIVATE_ONLY,
            ),
            NeedsActionRecord(
                title=None,
                coverage=None,
                group="OddTracker",
                url=None,
                reason="tracker not supported",
                kind=NeedsActionKind.UNSUPPORTED_TRACKER,
            ),
        ],
        unmonitored=1,
        queued=1,
        importing=1,
        imported=2,
    )


SUMMARY_RICH = RunSummaryReady(
    summary=RunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        added_count=2,
        tally=RunTally.from_stats(_rich_stats()),
        wait_mode_on=True,
        warnings=2,
        errors=1,
        elapsed_s=62.0,
        tip=NeedsActionCause.PRIVATE_ONLY,
    ),
)
SUMMARY_RICH_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Pearlarr (Sonarr) run complete", heavy=True),
    _blank(),
    _srow("  checked      : 8", "checked", "8"),
    _srow("  needs action : 2", "needs action", "2", style="yellow"),
    _styled("    Private Show", "yellow"),
    _brow("      files   S02 E01-E24", "files", "S02 E01-E24", "grey50"),
    _brow("      group   PrivGrp", "group", "PrivGrp", "yellow"),
    _brow(
        "      reason  private-only release; private releases not supported",
        "reason",
        "private-only release; private releases not supported",
        "yellow",
    ),
    _brow("      link    https://releases.moe/222", "link", "https://releases.moe/222", "grey50"),
    _styled("    (unknown title)", "yellow"),
    _brow("      group   OddTracker", "group", "OddTracker", "yellow"),
    _brow("      reason  tracker not supported", "reason", "tracker not supported", "yellow"),
    _srow("  added        : 2", "added", "2", style="green"),
    _styled("    Frieren", ""),
    _brow("      files   S01 E01-E28", "files", "S01 E01-E28", "grey50"),
    _brow("      link    https://releases.moe/111852", "link", "https://releases.moe/111852", "grey50"),
    _brow(
        "      torrent [GroupA] Frieren S01 1080p.mkv",
        "torrent",
        _torrent_text("[GroupA] Frieren S01 1080p.mkv", "GroupA", dry=False),
        "green",
    ),
    _styled("    Hashless Movie", ""),
    _brow("      link    https://releases.moe/900", "link", "https://releases.moe/900", "grey50"),
    # Quirk: a name-less grab renders "[group] " with a trailing space.
    _brow("      torrent [PrivGrp] ", "torrent", _torrent_text(None, "PrivGrp", dry=False), "green"),
    _srow("  queued       : 1", "queued", "1", style="grey50"),
    _srow("  importing    : 1", "importing", "1", style="yellow"),
    _srow("  imported     : 2", "imported", "2", style="green"),
    _srow("  up to date   : 1", "up to date", "1"),
    _srow("  unchanged    : 3  (since last run)", "unchanged", "3  (since last run)", style="grey50"),
    _srow("  no mapping   : 1", "no mapping", "1"),
    _srow("  no entry     : 2", "no entry", "2"),
    _srow("  seadex down  : 1", "seadex down", "1", style="yellow"),
    _srow("  no release   : 1", "no release", "1"),
    _srow("  unmonitored  : 1", "unmonitored", "1"),
    _srow("  issues       : 2 warnings, 1 error", "issues", "2 warnings, 1 error", style="bold red"),
    _srow("  elapsed      : 1m 02s", "elapsed", "1m 02s"),
    _styled(PRIVATE_ONLY_TIP, "grey50"),
    _RULE_LINE,
)


def _dry_stats() -> RunStats:
    return RunStats(
        checked=2,
        added=[
            GrabRecord(
                title="Dry Show",
                coverage="S01 E01-E12",
                url="https://releases.moe/5",
                name="Dry.Show.S01.1080p-GroupB",
                group="GroupB",
            ),
        ],
        needs_action=[
            NeedsActionRecord(
                title="NoFallback Show",
                coverage="S01 E01-E12",
                group="PrivGrp",
                url="https://releases.moe/7",
                reason="no public alternative covers these files",
                kind=NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK,
            ),
        ],
    )


SUMMARY_DRY_HAS_CLIENT = RunSummaryReady(
    summary=RunSummary(
        arr=Arr.SONARR,
        dry_run_note="nothing grabbed",
        added_count=1,
        tally=RunTally.from_stats(_dry_stats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    ),
)
SUMMARY_DRY_HAS_CLIENT_LINES: tuple[Line, ...] = (
    _blank(),
    # The DRY RUN note rides the console title only; the file keeps the plain title.
    _titled(
        "Pearlarr (Sonarr) run complete",
        title="Pearlarr (Sonarr) run complete   (DRY RUN — nothing grabbed)",
        heavy=True,
    ),
    _blank(),
    _srow("  checked      : 2", "checked", "2"),
    _srow("  needs action : 1", "needs action", "1", style="yellow"),
    _styled("    NoFallback Show", "yellow"),
    _brow("      files   S01 E01-E12", "files", "S01 E01-E12", "grey50"),
    _brow("      group   PrivGrp", "group", "PrivGrp", "yellow"),
    _brow(
        "      reason  no public alternative covers these files",
        "reason",
        "no public alternative covers these files",
        "yellow",
    ),
    _brow("      link    https://releases.moe/7", "link", "https://releases.moe/7", "grey50"),
    _srow("  added        : 1", "added", "1", style="green"),
    _styled("    Dry Show", "grey50"),
    _brow("      files   S01 E01-E12", "files", "S01 E01-E12", "grey50"),
    _brow("      link    https://releases.moe/5", "link", "https://releases.moe/5", "grey50"),
    _brow(
        "      torrent [GroupB] Dry.Show.S01.1080p-GroupB",
        "torrent",
        _torrent_text("Dry.Show.S01.1080p-GroupB", "GroupB", dry=True),
        "grey50",
    ),
    _srow("  up to date   : 0", "up to date", "0"),
    _srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    _srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _styled(NO_FALLBACK_TIP, "grey50"),
    _RULE_LINE,
)


def _preview_stats() -> RunStats:
    return RunStats(
        checked=1,
        added=[
            GrabRecord(
                title="Preview Movie",
                coverage=None,
                url="https://releases.moe/12",
                name="Preview.Movie.1080p-GroupC",
                group="GroupC",
            ),
        ],
    )


SUMMARY_DRY_NO_CLIENT = RunSummaryReady(
    summary=RunSummary(
        arr=Arr.RADARR,
        dry_run_note="qBittorrent not configured; nothing grabbed",
        added_count=1,
        tally=RunTally.from_stats(_preview_stats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=None,
    ),
)
SUMMARY_DRY_NO_CLIENT_LINES: tuple[Line, ...] = (
    _blank(),
    _titled(
        "Pearlarr (Radarr) run complete",
        title="Pearlarr (Radarr) run complete   (DRY RUN — qBittorrent not configured; nothing grabbed)",
        heavy=True,
    ),
    _blank(),
    _srow("  checked      : 1", "checked", "1"),
    _srow("  needs action : 0", "needs action", "0"),
    _srow("  added        : 1", "added", "1", style="green"),
    _styled("    Preview Movie", "grey50"),
    _brow("      link    https://releases.moe/12", "link", "https://releases.moe/12", "grey50"),
    _brow(
        "      torrent [GroupC] Preview.Movie.1080p-GroupC",
        "torrent",
        _torrent_text("Preview.Movie.1080p-GroupC", "GroupC", dry=True),
        "grey50",
    ),
    _srow("  up to date   : 0", "up to date", "0"),
    _srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    _srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _RULE_LINE,
)


SUMMARY_MINIMAL = RunSummaryReady(
    summary=RunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        added_count=0,
        tally=RunTally.from_stats(RunStats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=None,
    ),
)
SUMMARY_MINIMAL_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Pearlarr (Sonarr) run complete", heavy=True),
    _blank(),
    _srow("  checked      : 0", "checked", "0"),
    _srow("  needs action : 0", "needs action", "0"),
    _srow("  added        : 0", "added", "0"),
    _srow("  up to date   : 0", "up to date", "0"),
    _srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    _srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _RULE_LINE,
)


def _wait_off_stats() -> RunStats:
    return RunStats(
        checked=6,
        up_to_date=5,
        needs_action=[
            NeedsActionRecord(
                title="Stale Show",
                coverage="S03 E01-E12",
                group="PrivGrp",
                url="https://releases.moe/33",
                reason="your copy of the private release is outdated",
                kind=NeedsActionKind.PRIVATE_ONLY_STALE,
            ),
        ],
        # Non-zero pending counters that must stay hidden with the wait mode off.
        queued=2,
        importing=1,
        imported=3,
    )


SUMMARY_WAIT_OFF = RunSummaryReady(
    summary=RunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        added_count=0,
        tally=RunTally.from_stats(_wait_off_stats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=NeedsActionCause.PRIVATE_ONLY_STALE,
    ),
)
SUMMARY_WAIT_OFF_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Pearlarr (Sonarr) run complete", heavy=True),
    _blank(),
    _srow("  checked      : 6", "checked", "6"),
    _srow("  needs action : 1", "needs action", "1", style="yellow"),
    _styled("    Stale Show", "yellow"),
    _brow("      files   S03 E01-E12", "files", "S03 E01-E12", "grey50"),
    _brow("      group   PrivGrp", "group", "PrivGrp", "yellow"),
    _brow(
        "      reason  your copy of the private release is outdated",
        "reason",
        "your copy of the private release is outdated",
        "yellow",
    ),
    _brow("      link    https://releases.moe/33", "link", "https://releases.moe/33", "grey50"),
    _srow("  added        : 0", "added", "0"),
    _srow("  up to date   : 5", "up to date", "5"),
    _srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    _srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _styled(STALE_TIP, "grey50"),
    _RULE_LINE,
)


# --- the harness: drive the REAL reporter, assert the goldens --------------------------

_logger_ids = itertools.count()


def _fresh_logger() -> logging.Logger:
    """A uniquely-named DEBUG logger for the gateway/scripted-client collaborators."""

    logger = logging.getLogger(f"scan-parity-{next(_logger_ids)}")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


class _ScriptedTitleClient(AniListClient):
    """Scripted AniList wire client: a fixed resolvable title, or none at all."""

    def __init__(self, title: str | None) -> None:
        super().__init__(client=httpx.Client())
        self._title = title

    @override
    def query(self, al_id: int) -> dict[str, Any]:
        if self._title is None:
            return {}
        return {"data": {"Media": {"id": al_id, "title": {"english": self._title}}}}


class _Harness:
    """A real RunReporter (real gateway, faked leaves) recording emitted events."""

    def __init__(self, store: AbstractCacheStore | None = None, title: str | None = None) -> None:
        # NullHandler: the logger only serves the gateway/scripted client; the
        # parity lines come from the recorded EVENTS below, never from records.
        self.logger = _fresh_logger()
        self.logger.addHandler(logging.NullHandler())
        self.events: list[Event] = []
        # The summary's issues row diffs this bound counter (scripted directly).
        self.counts = SeverityCounts()
        self.reporter = RunReporter(
            emit=self.events.append,
            counts=lambda: self.counts,
            cache_store=store if store is not None else FakeCacheStore(),
            anilist=AniListGateway(
                cache_store=FakeCacheStore(),
                logger=self.logger,
                client=_ScriptedTitleClient(title),
            ),
        )

    def lines(self) -> tuple[Line, ...]:
        """Re-derive the scan lines from the recorded events through the shipped builders."""

        return tuple((line.level, line.message, line.payload) for line in scan_lines_from_events(self.events))


def _seeded_store() -> FakeCacheStore:
    store = FakeCacheStore()
    store.update_cache(
        Arr.SONARR,
        1,
        CacheRecord(name="Cached Show", coverage="S01 E01-E12", url="https://releases.moe/20997"),
    )
    store.update_cache(Arr.RADARR, 2, CacheRecord(name="Cached Movie", url="https://releases.moe/64"))
    store.update_cache(Arr.SONARR, 3, CacheRecord(name="Bare Show"))
    return store


class TestBannerParity:
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


class TestLedgerRowParity:
    def test_unmonitored(self) -> None:
        harness = _Harness()
        harness.reporter.log_arr_item_unmonitored(RunContext(arr=Arr.SONARR), "Unmonitored Show")
        assert harness.lines() == UNMONITORED_LINES

    def test_no_mapping(self) -> None:
        harness = _Harness()
        harness.reporter.log_no_anilist_mappings(RunContext(arr=Arr.SONARR), "Unmapped Show")
        assert harness.lines() == NO_MAPPING_LINES

    def test_ignored(self) -> None:
        harness = _Harness()
        harness.reporter.log_ignored_anilist_id(123)
        assert harness.lines() == IGNORED_LINES


class TestEntryHeaderParity:
    def test_checking_with_coverage_url_and_incomplete(self) -> None:
        harness = _Harness()
        entry = make_entry_record(url="https://releases.moe/111852", is_incomplete=True)
        harness.reporter.log_al_title(RunContext(arr=Arr.SONARR), "Frieren", entry, coverage="S01 E01-E28")
        assert harness.lines() == CHECKING_FULL_LINES

    def test_checking_url_only(self) -> None:
        harness = _Harness()
        entry = make_entry_record(url="https://releases.moe/437")
        harness.reporter.log_al_title(RunContext(arr=Arr.RADARR), "Perfect Blue", entry)
        assert harness.lines() == CHECKING_URL_ONLY_LINES

    def test_cached_full_row(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_cached_entry(RunContext(arr=Arr.SONARR), Arr.SONARR, 1)
        assert harness.lines() == CACHED_FULL_LINES

    def test_cached_in_radarr(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_cached_entry(RunContext(arr=Arr.SONARR), Arr.RADARR, 2, state=EntryState.IN_RADARR)
        assert harness.lines() == CACHED_IN_RADARR_LINES

    def test_cached_without_coverage_or_url_renders_just_the_row(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_cached_entry(RunContext(arr=Arr.SONARR), Arr.SONARR, 3)
        assert harness.lines() == CACHED_BARE_LINES

    def test_pending_snapshots(self) -> None:
        harness = _Harness()
        pending = pending_import(**_PENDING_KWARGS)
        for state, expected in (
            (PendingState.QUEUED, PENDING_QUEUED_LINES),
            (PendingState.IMPORTING, PENDING_IMPORTING_LINES),
            (PendingState.IMPORTED, PENDING_IMPORTED_LINES),
        ):
            harness.events.clear()
            assert harness.reporter.log_pending_snapshot(state, pending) is True
            assert harness.lines() == expected


class TestTitledEntryParity:
    def test_no_entry_with_resolved_title(self) -> None:
        harness = _Harness(title="Resolved Title")
        harness.reporter.log_no_sd_entry(RunContext(arr=Arr.SONARR), 42)
        assert harness.lines() == NO_ENTRY_RESOLVED_LINES

    def test_no_entry_without_a_title(self) -> None:
        harness = _Harness(title=None)
        harness.reporter.log_no_sd_entry(RunContext(arr=Arr.SONARR), 77)
        assert harness.lines() == NO_ENTRY_UNRESOLVED_LINES

    def test_outage_skip_with_cached_name(self) -> None:
        harness = _Harness(store=_seeded_store())
        harness.reporter.log_seadex_outage_skip(RunContext(arr=Arr.SONARR), 1)
        assert harness.lines() == OUTAGE_LINES


class TestDetailParity:
    def test_no_suitable_releases(self) -> None:
        harness = _Harness()
        harness.reporter.log_no_seadex_releases(RunContext(arr=Arr.SONARR))
        assert harness.lines() == NO_RELEASES_LINES


def _xdetail(level: int, message: str, key: str, value: str, style: str) -> Line:
    """An external detail row at the site's own level (else exactly `_detail`'s shape)."""

    return (level, message, KvLine(key=key, value=value, key_width=9, value_style=style, indent=2, sep=""))


@dataclass(frozen=True, slots=True)
class _DetailSite:
    """One migrated producer site: the `reporter.detail` args + its pinned record."""

    label: str
    value: StyledValue
    severity: Severity
    golden: Line


@dataclass(frozen=True, slots=True)
class _FactSite:
    """One flipped producer site: the typed fact `reporter.post` takes + its pinned record."""

    fact: ReleaseSkipped | GrabFailed
    golden: Line


class TestExternalDetailParity:
    """The external producer sites (grab pipeline / release filter / episode
    mapper). Each golden was captured by RUNNING the pre-migration
    `LogFormatter.detail` with the site's literal args at fc58aef; the reporter
    must reproduce every line byte-identically through the shipped `scan_lines`
    builders (the grammar every rendering surface consumes). The four
    grab-pipeline sites emit typed `reporter.post` facts - the golden
    `Line` tuples stay verbatim; the planner notice / missing / status sites
    stay `reporter.detail`-driven.
    """

    FACT_SITES: tuple[_FactSite, ...] = (
        # grab_pipeline._add_one_url: a private-only release.
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
        # grab_pipeline._add_one_url: a tracker outside the user's selected list.
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
        # grab_pipeline._add_one_url: a tracker with no parser yet.
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
        # grab_pipeline._add_one_url: a contained grab failure.
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
        # grab_pipeline.grab_and_cache: nothing to download (the blue NOTE accent).
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
        # seadex_filter.filter_downloads: a planner SkipNotice (WARNING form).
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
        # sonarr_episodes.get_sonarr_release_dict: missing-episode coverage.
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
        # The flipped sites post typed facts; the shipped builders re-derive the
        # exact (level, message, payload) lines the detail-era sites pinned.
        harness = _Harness()

        for site in self.FACT_SITES:
            harness.reporter.post(site.fact)

        assert harness.lines() == tuple(site.golden for site in self.FACT_SITES)

    def test_reporter_detail_reproduces_the_goldens(self) -> None:
        # The real reporter emits EntryDetail events; the shipped builders
        # re-derive the exact (level, message, payload) lines the sites pinned.
        harness = _Harness()

        for site in self.SITES:
            harness.reporter.detail(site.label, site.value, severity=site.severity)

        assert harness.lines() == tuple(site.golden for site in self.SITES)

    def test_post_rides_the_open_entry_then_emits_scope_free(self) -> None:
        # Inside an open entry the fact carries the entry scope; after a boundary
        # closes it, a post emits scope-free (col-0 by design, like _post).
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_al_title(ctx, "Show", make_entry_record(url="https://releases.moe/1"))
        harness.reporter.post(self.FACT_SITES[0].fact)
        harness.reporter.log_arr_item_start(Arr.SONARR, "Next", 1, 1)  # closes the entry
        harness.reporter.post(self.FACT_SITES[3].fact)

        opened = next(e for e in harness.events if isinstance(e, ScopeOpened))
        skipped = next(e for e in harness.events if isinstance(e, ReleaseSkipped))
        failed = next(e for e in harness.events if isinstance(e, GrabFailed))
        assert skipped.scope == opened.scope
        assert failed.scope is None


class TestActionBlockParity:
    _SEADEX_DICT = {
        "GroupA": rg_group({"https://nyaa.si/view/1": url_item(download=True)}, tags=frozenset({_HDR_TAG})),
        "GroupB": rg_group({"https://nyaa.si/view/2": url_item(download=True)}),
    }
    _FRESH_RESULTS = [
        ReleaseOutcome(outcome=AddOutcome.ADDED, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
        ReleaseOutcome(outcome=AddOutcome.ADDED, name=None, group="GroupB"),
    ]

    def test_fresh_add(self) -> None:
        harness = _Harness()
        assert harness.reporter.log_seadex_action(self._SEADEX_DICT, self._FRESH_RESULTS) is True
        assert harness.lines() == ACTION_FRESH_LINES

    def test_dry_run(self) -> None:
        harness = _Harness()
        assert harness.reporter.log_seadex_action(self._SEADEX_DICT, self._FRESH_RESULTS, dry_run=True) is True
        assert harness.lines() == ACTION_DRY_LINES

    def test_already_downloading_with_monitor(self) -> None:
        harness = _Harness()
        seadex_dict = {
            "GroupA": rg_group({"https://nyaa.si/view/1": url_item(download=True)}, tags=frozenset({_HDR_TAG})),
        }
        results = [
            ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
        ]
        assert harness.reporter.log_seadex_action(seadex_dict, results, monitor_active=True) is True
        assert harness.lines() == ACTION_DOWNLOADING_WAITING_LINES

        harness.events.clear()
        assert harness.reporter.log_seadex_action(seadex_dict, results, monitor_active=False) is True
        assert harness.lines() == ACTION_DOWNLOADING_NO_WAIT_LINES

    def test_mixed_results_group_added_before_downloading(self) -> None:
        # Downloading-first input still renders added-first: the grouping, not
        # results order, is the pinned contract.
        harness = _Harness()
        results = [
            ReleaseOutcome(outcome=AddOutcome.ALREADY_ADDED, name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
            ReleaseOutcome(outcome=AddOutcome.ADDED, name="[GroupB] Frieren S01 1080p.mkv", group="GroupB"),
        ]
        assert harness.reporter.log_seadex_action(self._SEADEX_DICT, results) is True
        assert harness.lines() == ACTION_MIXED_LINES

    def test_max_torrents(self) -> None:
        harness = _Harness()
        harness.reporter.log_max_torrents_added(5)
        assert harness.lines() == CAP_REACHED_LINES


def _summary_lines(
    ctx: RunContext,
    *,
    preview: bool,
    has_client: bool,
    warnings: int = 0,
    errors: int = 0,
) -> tuple[Line, ...]:
    """Capture exactly what log_run_summary emits, with scripted issue deltas."""

    harness = _Harness()
    # Production shape: run start stamps the mark (bound to the harness counter),
    # then the run's issues accrue on it.
    ctx.counts_mark = harness.reporter.counts_mark()
    # Scripted issues ride the bound counter directly (the issues-row delta);
    # they produce no events, so only the summary's feed the re-derived lines.
    for _ in range(warnings):
        harness.counts.record(Severity.WARNING)
    for _ in range(errors):
        harness.counts.record(Severity.ERROR)
    harness.reporter.log_run_summary(ctx, preview=preview, has_client=has_client)
    return harness.lines()


class TestRunSummaryParity:
    def test_rich_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A frozen clock pins the elapsed row (started at 38s, "now" 100s -> 1m 02s).
        monkeypatch.setattr(time, "monotonic", lambda: 100.0)
        ctx = RunContext(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING, stats=_rich_stats())
        ctx.torrents_added = 2
        ctx.started_monotonic = 38.0
        assert _summary_lines(ctx, preview=False, has_client=True, warnings=2, errors=1) == SUMMARY_RICH_LINES

    def test_dry_run_with_client(self) -> None:
        ctx = RunContext(arr=Arr.SONARR, stats=_dry_stats())
        ctx.torrents_added = 1
        assert _summary_lines(ctx, preview=True, has_client=True) == SUMMARY_DRY_HAS_CLIENT_LINES

    def test_dry_run_without_client(self) -> None:
        ctx = RunContext(arr=Arr.RADARR, stats=_preview_stats())
        ctx.torrents_added = 1
        assert _summary_lines(ctx, preview=True, has_client=False) == SUMMARY_DRY_NO_CLIENT_LINES

    def test_minimal_all_zeros(self) -> None:
        ctx = RunContext(arr=Arr.SONARR)
        assert _summary_lines(ctx, preview=False, has_client=True) == SUMMARY_MINIMAL_LINES

    def test_wait_mode_off_hides_pending_rows(self) -> None:
        ctx = RunContext(arr=Arr.SONARR, stats=_wait_off_stats())
        assert _summary_lines(ctx, preview=False, has_client=True) == SUMMARY_WAIT_OFF_LINES

    def test_warnings_only_issues_row_renders_yellow(self) -> None:
        # The style ladder's middle rung (warnings without errors) has no golden
        # of its own: bold-red needs errors, yellow needs warnings alone.
        ctx = RunContext(arr=Arr.SONARR)
        lines = _summary_lines(ctx, preview=False, has_client=True, warnings=2)
        expected = _srow("  issues       : 2 warnings, 0 errors", "issues", "2 warnings, 0 errors", style="yellow")
        assert expected in lines


class TestTorrentValuePins:
    """The summary "torrent" Text values, pinned by content (plain + spans + base
    style) so the golden equality above can't hide a group_highlight regression."""

    def test_leading_group_is_highlighted_in_place(self) -> None:
        value = _torrent_text("[GroupA] Frieren S01 1080p.mkv", "GroupA", dry=False)
        assert isinstance(value, Text)
        assert value.plain == "[GroupA] Frieren S01 1080p.mkv"
        assert value.spans == [Span(1, 7, "cyan")]
        assert value.style == "green"

    def test_name_less_grab_prepends_the_group_with_trailing_space(self) -> None:
        value = _torrent_text(None, "PrivGrp", dry=False)
        assert isinstance(value, Text)
        assert value.plain == "[PrivGrp] "
        assert value.spans == [Span(1, 8, "cyan")]

    def test_dry_run_dims_group_and_base(self) -> None:
        value = _torrent_text("Dry.Show.S01.1080p-GroupB", "GroupB", dry=True)
        assert isinstance(value, Text)
        assert value.plain == "[GroupB] Dry.Show.S01.1080p-GroupB"
        assert value.spans == [Span(1, 7, "grey50")]
        assert value.style == "grey50"


class TestScopeLifecycle:
    """A1/A3: the entry-scope open/close discipline over multi-method sequences.

    Asserts on the RAW recorded event stream (scope boundaries included). Serials
    mint from the process-wide minter, so the checks are RELATIONAL - which scope
    opens/closes in which order, which header carries which id - never absolute
    serial values.
    """

    def test_item_entry_entry_item_sequence(self) -> None:
        # item -> entry -> entry -> item: each new entry closes the previous, and
        # the closing item boundary closes the last one.
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_arr_item_start(Arr.SONARR, "Show", 1, 1)
        harness.reporter.log_al_title(ctx, "First", make_entry_record(url="https://releases.moe/1"))
        harness.reporter.log_al_title(ctx, "Second", make_entry_record(url="https://releases.moe/2"))
        harness.reporter.log_arr_item_start(Arr.SONARR, "Next", 1, 1)

        boundaries = [e for e in harness.events if isinstance(e, (ScopeOpened, ScopeClosed))]
        assert len(boundaries) == 4
        first_open, first_close, second_open, second_close = boundaries
        # Each entry is closed before the next one opens (one entry at a time).
        assert isinstance(first_open, ScopeOpened) and first_open.label == "First"
        assert isinstance(first_close, ScopeClosed) and first_close.scope == first_open.scope
        assert isinstance(second_open, ScopeOpened) and second_open.label == "Second"
        assert isinstance(second_close, ScopeClosed) and second_close.scope == second_open.scope
        # Every header carries the id of the scope opened immediately before it.
        headers = [e for e in harness.events if isinstance(e, EntryHeader)]
        assert [h.scope for h in headers] == [first_open.scope, second_open.scope]

    def test_bare_titled_rows_open_no_scope(self) -> None:
        # The self-contained ledger rows never open an entry scope (col-0 rows).
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_arr_item_unmonitored(ctx, "Unmon")
        harness.reporter.log_no_anilist_mappings(ctx, "NoMap")
        harness.reporter.log_ignored_anilist_id(7)
        harness.reporter.log_entry_status(EntryState.IN_RADARR, "Owned")

        assert not any(isinstance(e, (ScopeOpened, ScopeClosed)) for e in harness.events)
        assert all(isinstance(e, LedgerRow) for e in harness.events)

    def test_close_is_idempotent(self) -> None:
        # A second boundary with no open entry emits no spurious ScopeClosed.
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_al_title(ctx, "Show", make_entry_record(url="https://releases.moe/1"))
        harness.reporter.log_arr_item_start(Arr.SONARR, "A", 1, 1)  # closes the entry
        harness.reporter.log_arr_item_start(Arr.SONARR, "B", 1, 1)  # nothing left to close

        assert len([e for e in harness.events if isinstance(e, ScopeClosed)]) == 1

    def test_action_keeps_the_entry_open_for_a_trailing_diagnostic(self) -> None:
        # A1(b): log_seadex_action POSTS but does not close, so the entry stays on
        # the frontier - a diagnostic emitted before the next boundary indents
        # under this entry (live-verified), not col-0.
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_al_title(ctx, "Show", make_entry_record(url="https://releases.moe/1"))
        harness.reporter.log_seadex_action({}, [ReleaseOutcome(outcome=AddOutcome.ADDED, name="X", group="G")])

        assert not any(isinstance(e, ScopeClosed) for e in harness.events)
        opened = next(e for e in harness.events if isinstance(e, ScopeOpened))
        action = next(e for e in harness.events if isinstance(e, GrabAction))
        assert action.scope == opened.scope

    def test_fresh_reporter_has_no_open_entry(self) -> None:
        # A3: _entry starts None on a fresh reporter, so the first current-entry
        # detail (no header before it) opens no scope and rides scope-free.
        harness = _Harness()
        harness.reporter.log_no_seadex_releases(RunContext(arr=Arr.SONARR))

        assert not any(isinstance(e, ScopeOpened) for e in harness.events)
        detail = next(e for e in harness.events if isinstance(e, EntryDetail))
        assert detail.scope is None

    def test_cap_closes_the_open_entry(self) -> None:
        # The scan breaks at the cap and _finalize_run's reconcile runs before
        # the summary: a still-open ENTRY frontier would misplace its diagnostics.
        harness = _Harness()
        ctx = RunContext(arr=Arr.SONARR)
        harness.reporter.log_al_title(ctx, "Show", make_entry_record(url="https://releases.moe/1"))
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
    """The fakes filter tuple can't rot behind the ScanEvent union: a new scan
    event missing from it would silently vanish from every re-derived golden."""

    members: set[object] = set(get_args(ScanEvent.__value__))
    assert set(SCAN_EVENT_TYPES) == members


def test_tip_precedence_covers_exactly_the_tip_texts() -> None:
    """Producer precedence (_TIP_PRECEDENCE) and builder texts (_TIP_TEXTS) are
    hand-maintained twins; an entry in one without the other silently drops the
    tip line (unranked cause -> tip=None; unmapped cause -> no text)."""

    assert set(_TIP_PRECEDENCE) == set(_TIP_TEXTS)
