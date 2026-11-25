# pyright: strict
"""Scan output goldens: each scan event beside the exact lines it prints.

Each `Line` is one `(level, message, payload)` line as the scan line builders make it. Every one was captured by
running the real `RunReporter`, not worked out by hand. `test_scan_parity` drives the reporter and checks it still
emits them, and `test_output_scan_render` checks the line builders and the console against the same data.
"""

import logging

from rich.text import Text

from pearlarr.arr_run_state import ArrRunStats
from pearlarr.config import Arr
from pearlarr.output import (
    Accent,
    ArrRunSummary,
    ArrRunSummaryReady,
    ArrRunTally,
    CapReached,
    GrabAction,
    GrabbedRow,
    GrabStatus,
    GroupLabel,
    ItemStarted,
    NeedsActionCause,
    NeedsActionRow,
    ReleaseLabel,
    ScanStarted,
    StyledValue,
    TitleDetail,
    TitleHeader,
    TitleRow,
)
from pearlarr.output.console_text import ConsoleRender, KeyValue, SeparatorRule, StyledLine, TitledRule, group_highlight
from pearlarr.output.events import TitleState

type Line = tuple[int, str, ConsoleRender | None]
"""One pinned line: (levelno, plain message, ConsoleRender payload)."""

_I = logging.INFO


# --- golden-line factories (literal payload compression, messages stay verbatim) ---


def _blank() -> Line:
    return (_I, "", None)


def _titled(message: str, *, title: str | None = None, heavy: bool = False) -> Line:
    return (_I, message, TitledRule(title=title if title is not None else message, heavy=heavy))


def _styled(message: str, style: str) -> Line:
    return (_I, message, StyledLine(style=style))


def _detail(message: str, key: str, value: str, style: str | None = "grey50", tail: str | None = None) -> Line:
    """A title-detail row: the colon-less gutter kv at indent 2, key width 9."""

    return (_I, message, KeyValue(key=key, value=value, key_width=9, value_style=style, indent=2, sep="", tail=tail))


def srow(message: str, key: str, value: str, style: str | None = None) -> Line:
    """An arr run summary row: "key : value" at indent 1, key width 12."""

    return (_I, message, KeyValue(key=key, value=value, key_width=12, value_style=style, indent=1))


def _brow(message: str, key: str, value: str | Text, style: str) -> Line:
    """A summary per-title block row: gutter kv at indent 3, key width 7."""

    return (_I, message, KeyValue(key=key, value=value, key_width=7, value_style=style, indent=3, sep=""))


def torrent_text(name: str | None, group: str, *, dry: bool) -> str | Text:
    """The summary "torrent" value exactly as the reporter builds it."""

    dim = "grey50"
    return group_highlight(name, group, group_style=dim if dry else "cyan", base_style=dim if dry else "green")


# --- scan/item banners --------------------------------------------------------------

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

# --- self-contained title rows --------------------------------------------------------

UNMONITORED_ROW = TitleRow(state=TitleState.UNMONITORED, label="Unmonitored Show")
UNMONITORED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  unmonitored Unmonitored Show", "grey50"),
)

NO_MAPPING_ROW = TitleRow(state=TitleState.NO_MAPPING, label="Unmapped Show")
NO_MAPPING_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no mapping  Unmapped Show", "grey50"),
)

NO_EPISODES_ROW = TitleRow(state=TitleState.NO_EPISODES, label="Episodeless Show")
NO_EPISODES_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no episodes Episodeless Show", "grey50"),
)

IGNORED_ROW = TitleRow(state=TitleState.IGNORED, label="AniList #123")
IGNORED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  ignored     AniList #123", "grey50"),
)

IGNORED_TITLED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  ignored     Ignored Show", "grey50"),
    _detail("    anilist   123", "anilist", "123", style=None),
)

# --- title headers (row + coverage/link continuation) ---------------------------------

CHECKING_FULL = TitleHeader(
    state=TitleState.CHECKING,
    title="Frieren",
    anilist_id=1,
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

CHECKING_URL_ONLY = TitleHeader(
    state=TitleState.CHECKING,
    title="Perfect Blue",
    anilist_id=1,
    url="https://releases.moe/437",
)
CHECKING_URL_ONLY_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  checking    Perfect Blue", ""),
    _detail("    link      https://releases.moe/437", "link", "https://releases.moe/437"),
)

CACHED_FULL = TitleHeader(
    state=TitleState.UNCHANGED,
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

CACHED_BARE = TitleHeader(state=TitleState.UNCHANGED, title="Bare Show")
CACHED_BARE_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  unchanged   Bare Show", "grey50"),
)

# --- titled rows with detail continuations (SeaDex unreachable / no entry) ------------

UNREACHABLE_ROW = TitleRow(state=TitleState.NOT_CHECKED, label="Cached Show")
UNREACHABLE_ANILIST_DETAIL = TitleDetail(label="anilist", value=StyledValue("1"))
UNREACHABLE_STATUS_DETAIL = TitleDetail(label="status", value=StyledValue("SeaDex unreachable", Accent.DIM))
UNREACHABLE_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  not checked Cached Show", "grey50"),
    # The anilist id repeats on its own detail row under any real name, the
    # cached name included. Only the id-form label omits it (it names the id itself).
    _detail("    anilist   1", "anilist", "1", style=None),
    _detail("    status    SeaDex unreachable", "status", "SeaDex unreachable"),
)

NO_ENTRY_RESOLVED_ROW = TitleRow(state=TitleState.NO_ENTRY, label="Resolved Title")
NO_ENTRY_RESOLVED_DETAIL = TitleDetail(label="anilist", value=StyledValue("42"))
NO_ENTRY_RESOLVED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no entry    Resolved Title", "grey50"),
    _detail("    anilist   42", "anilist", "42", style=None),
)

NO_ENTRY_ARR_TITLE_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no entry    Arr Title", "grey50"),
    _detail("    anilist   42", "anilist", "42", style=None),
)

NO_ENTRY_UNRESOLVED_ROW = TitleRow(state=TitleState.NO_ENTRY, label="AniList #77")
NO_ENTRY_UNRESOLVED_LINES: tuple[Line, ...] = (
    _blank(),
    _styled("  no entry    AniList #77", "grey50"),
)

# --- carried-over pending snapshots ----------------------------------------------------


def _pending_import_header(state: TitleState) -> TitleHeader:
    return TitleHeader(state=state, title="My Show · SubGroup", coverage="S01 E01-E13", url="https://releases.moe/1")


def _pending_import_lines(row: Line) -> tuple[Line, ...]:
    return (
        _blank(),
        row,
        _detail("    files     S01 E01-E13", "files", "S01 E01-E13"),
        _detail("    link      https://releases.moe/1", "link", "https://releases.moe/1"),
    )


PENDING_IMPORT_QUEUED = _pending_import_header(TitleState.DOWNLOADING)
PENDING_QUEUED_LINES = _pending_import_lines(_styled("  downloading My Show · SubGroup", "grey50"))

PENDING_IMPORT_DOWNLOADED = _pending_import_header(TitleState.DOWNLOADED)
PENDING_DOWNLOADED_LINES = _pending_import_lines(_styled("  downloaded  My Show · SubGroup", "grey50"))

PENDING_IMPORT_IMPORTED = _pending_import_header(TitleState.IMPORTED)
PENDING_IMPORTED_LINES = _pending_import_lines(_styled("  imported    My Show · SubGroup", "green"))

# --- title details -----------------------------------------------------------------------

NO_RELEASES_DETAIL = TitleDetail(label="status", value=StyledValue("no suitable releases on SeaDex", Accent.DIM))
NO_RELEASES_LINES: tuple[Line, ...] = (
    _detail("    status    no suitable releases on SeaDex", "status", "no suitable releases on SeaDex"),
)

# --- action blocks -----------------------------------------------------------------------

_FRESH_GROUPS = (GroupLabel(name="GroupA", tags=("HDR",)), GroupLabel(name="GroupB"))
# The name-less second release (a hashless/private torrent) falls back to its group.
_FRESH_ADDED = (
    ReleaseLabel(name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),
    ReleaseLabel(name="", group="GroupB"),
)

ACTION_FRESH = GrabAction(status=GrabStatus.GRABBING, groups=_FRESH_GROUPS, grabbed=_FRESH_ADDED, in_qbit=())
ACTION_FRESH_LINES: tuple[Line, ...] = (
    _detail("    status    grabbing", "status", "grabbing", style=None),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail("    grabbed   [GroupA] Frieren S01 1080p.mkv", "grabbed", "[GroupA] Frieren S01 1080p.mkv", style="green"),
    _detail("    grabbed   GroupB", "grabbed", "GroupB", style="green"),
)

ACTION_DRY = GrabAction(status=GrabStatus.WOULD_GRAB, groups=_FRESH_GROUPS, grabbed=_FRESH_ADDED, in_qbit=())
# "would grab" (10 chars) overflows key width 9, so its value sits one space past the key.
ACTION_DRY_LINES: tuple[Line, ...] = (
    _detail("    status    would grab (dry run)", "status", "would grab (dry run)", style=None),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail(
        "    would grab [GroupA] Frieren S01 1080p.mkv",
        "would grab",
        "[GroupA] Frieren S01 1080p.mkv",
        style="green",
    ),
    _detail("    would grab GroupB", "would grab", "GroupB", style="green"),
)

_DOWNLOADING = (ReleaseLabel(name="[GroupA] Frieren S01 1080p.mkv", group="GroupA"),)
_ONE_GROUP = (GroupLabel(name="GroupA", tags=("HDR",)),)

ACTION_DOWNLOADING_WAITING = GrabAction(
    status=GrabStatus.ALREADY_IN_QBIT,
    groups=_ONE_GROUP,
    grabbed=(),
    in_qbit=_DOWNLOADING,
    waiting_to_import=True,
)
ACTION_DOWNLOADING_WAITING_LINES: tuple[Line, ...] = (
    _detail(
        "    status    already in qBittorrent - waiting to import",
        "status",
        "already in qBittorrent - waiting to import",
        style="yellow",
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail(
        "    torrent   [GroupA] Frieren S01 1080p.mkv",
        "torrent",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

ACTION_DOWNLOADING_NO_WAIT = GrabAction(
    status=GrabStatus.ALREADY_IN_QBIT,
    groups=_ONE_GROUP,
    grabbed=(),
    in_qbit=_DOWNLOADING,
)
ACTION_DOWNLOADING_NO_WAIT_LINES: tuple[Line, ...] = (
    _detail(
        "    status    already in qBittorrent",
        "status",
        "already in qBittorrent",
        style="yellow",
    ),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail(
        "    torrent   [GroupA] Frieren S01 1080p.mkv",
        "torrent",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

# Mixed outcomes render grouped - all added, then all already in qBittorrent - regardless
# of results order (accepted Band-D delta, OLD interleaved in results order).
ACTION_MIXED = GrabAction(
    status=GrabStatus.GRABBING,
    groups=_FRESH_GROUPS,
    grabbed=(ReleaseLabel(name="[GroupB] Frieren S01 1080p.mkv", group="GroupB"),),
    in_qbit=_DOWNLOADING,
)
ACTION_MIXED_LINES: tuple[Line, ...] = (
    _detail("    status    grabbing", "status", "grabbing", style=None),
    _detail("    group     GroupA [HDR]", "group", "GroupA [HDR]", style="cyan"),
    _detail("    group     GroupB", "group", "GroupB", style="cyan"),
    _detail("    grabbed   [GroupB] Frieren S01 1080p.mkv", "grabbed", "[GroupB] Frieren S01 1080p.mkv", style="green"),
    _detail(
        "    torrent   [GroupA] Frieren S01 1080p.mkv",
        "torrent",
        "[GroupA] Frieren S01 1080p.mkv",
        style="yellow",
    ),
)

CAP_REACHED = CapReached(cap=5)
CAP_REACHED_LINES: tuple[Line, ...] = (
    _styled(
        "Reached the maximum number of torrents for this run (advanced.max_torrents_to_add); "
        "checking the remaining titles without grabbing",
        "yellow",
    ),
)

# --- arr run summaries --------------------------------------------------------------------

_RULE_LINE: Line = (_I, "=" * 80, SeparatorRule(char="="))

PRIVATE_ONLY_TIP = (
    "  Tip: manually grab private releases or set private_releases: fallback to grab a public release instead."
)
NO_FALLBACK_TIP = "  Tip: no public release covers these files yet; the title is re-checked every run until one does."
OUTDATED_TIP = (
    "  Tip: your copies of these releases are outdated (their file sizes no longer match); "
    "update them from their private tracker, or delete the outdated files to let the "
    "public fallback stand in."
)


def rich_stats() -> ArrRunStats:
    return ArrRunStats(
        checked=8,
        grabbed=[
            GrabbedRow(
                title="Frieren",
                coverage="S01 E01-E28",
                url="https://releases.moe/111852",
                name="[GroupA] Frieren S01 1080p.mkv",
                group="GroupA",
            ),
            GrabbedRow(
                title="Hashless Movie",
                coverage=None,
                url="https://releases.moe/900",
                name=None,
                group="PrivGrp",
            ),
        ],
        up_to_date=1,
        unchanged=3,
        no_seadex_entry=2,
        seadex_unreachable=1,
        no_releases=1,
        no_mappings=1,
        needs_action=[
            NeedsActionRow(
                title="Private Show",
                coverage="S02 E01-E24",
                group="PrivGrp",
                url="https://releases.moe/222",
                reason="private-only release; private releases not supported",
                cause=NeedsActionCause.PRIVATE_ONLY,
            ),
            NeedsActionRow(
                title=None,
                coverage=None,
                group="OddTracker",
                url=None,
                reason="tracker not supported",
                cause=NeedsActionCause.UNSUPPORTED_TRACKER,
            ),
        ],
        unmonitored=1,
        downloading=1,
        downloaded=1,
        imported=2,
        held_by_cap=3,
    )


SUMMARY_RICH = ArrRunSummaryReady(
    summary=ArrRunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        grabbed_count=2,
        tally=ArrRunTally.from_stats(rich_stats()),
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
    srow("  checked      : 8", "checked", "8"),
    srow("  needs action : 2", "needs action", "2", style="yellow"),
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
    srow("  grabbed      : 2", "grabbed", "2", style="green"),
    _styled("    Frieren", ""),
    _brow("      files   S01 E01-E28", "files", "S01 E01-E28", "grey50"),
    _brow("      link    https://releases.moe/111852", "link", "https://releases.moe/111852", "grey50"),
    _brow(
        "      torrent [GroupA] Frieren S01 1080p.mkv",
        "torrent",
        torrent_text("[GroupA] Frieren S01 1080p.mkv", "GroupA", dry=False),
        "green",
    ),
    _styled("    Hashless Movie", ""),
    _brow("      link    https://releases.moe/900", "link", "https://releases.moe/900", "grey50"),
    # Quirk: a name-less grab renders "[group] " with a trailing space.
    _brow("      torrent [PrivGrp] ", "torrent", torrent_text(None, "PrivGrp", dry=False), "green"),
    srow("  held by cap  : 3", "held by cap", "3", style="yellow"),
    srow("  downloading  : 1", "downloading", "1", style="grey50"),
    srow("  downloaded   : 1", "downloaded", "1", style="yellow"),
    srow("  imported     : 2", "imported", "2", style="green"),
    srow("  up to date   : 1", "up to date", "1"),
    srow("  unchanged    : 3  (since last run)", "unchanged", "3  (since last run)", style="grey50"),
    srow("  no mapping   : 1", "no mapping", "1"),
    srow("  no entry     : 2", "no entry", "2"),
    srow("  not checked  : 1", "not checked", "1", style="yellow"),
    srow("  no release   : 1", "no release", "1"),
    srow("  unmonitored  : 1", "unmonitored", "1"),
    srow("  issues       : 2 warnings, 1 error", "issues", "2 warnings, 1 error", style="bold red"),
    srow("  elapsed      : 1m 02s", "elapsed", "1m 02s"),
    _styled(PRIVATE_ONLY_TIP, "grey50"),
    _RULE_LINE,
)


def dry_stats() -> ArrRunStats:
    return ArrRunStats(
        checked=2,
        grabbed=[
            GrabbedRow(
                title="Dry Show",
                coverage="S01 E01-E12",
                url="https://releases.moe/5",
                name="Dry.Show.S01.1080p-GroupB",
                group="GroupB",
            ),
        ],
        needs_action=[
            NeedsActionRow(
                title="NoFallback Show",
                coverage="S01 E01-E12",
                group="PrivGrp",
                url="https://releases.moe/7",
                reason="no fallback covers these files",
                cause=NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
            ),
        ],
    )


SUMMARY_DRY_HAS_CLIENT = ArrRunSummaryReady(
    summary=ArrRunSummary(
        arr=Arr.SONARR,
        dry_run_note="nothing grabbed",
        grabbed_count=1,
        tally=ArrRunTally.from_stats(dry_stats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    ),
)
SUMMARY_DRY_HAS_CLIENT_LINES: tuple[Line, ...] = (
    _blank(),
    # The DRY RUN note goes on the console title only. The file keeps the plain title.
    _titled(
        "Pearlarr (Sonarr) run complete",
        title="Pearlarr (Sonarr) run complete   (DRY RUN - nothing grabbed)",
        heavy=True,
    ),
    _blank(),
    srow("  checked      : 2", "checked", "2"),
    srow("  needs action : 1", "needs action", "1", style="yellow"),
    _styled("    NoFallback Show", "yellow"),
    _brow("      files   S01 E01-E12", "files", "S01 E01-E12", "grey50"),
    _brow("      group   PrivGrp", "group", "PrivGrp", "yellow"),
    _brow(
        "      reason  no fallback covers these files",
        "reason",
        "no fallback covers these files",
        "yellow",
    ),
    _brow("      link    https://releases.moe/7", "link", "https://releases.moe/7", "grey50"),
    srow("  grabbed      : 1", "grabbed", "1", style="green"),
    _styled("    Dry Show", "grey50"),
    _brow("      files   S01 E01-E12", "files", "S01 E01-E12", "grey50"),
    _brow("      link    https://releases.moe/5", "link", "https://releases.moe/5", "grey50"),
    _brow(
        "      torrent [GroupB] Dry.Show.S01.1080p-GroupB",
        "torrent",
        torrent_text("Dry.Show.S01.1080p-GroupB", "GroupB", dry=True),
        "grey50",
    ),
    srow("  up to date   : 0", "up to date", "0"),
    srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _styled(NO_FALLBACK_TIP, "grey50"),
    _RULE_LINE,
)


def dry_run_stats() -> ArrRunStats:
    return ArrRunStats(
        checked=1,
        grabbed=[
            GrabbedRow(
                title="Preview Movie",
                coverage=None,
                url="https://releases.moe/12",
                name="Preview.Movie.1080p-GroupC",
                group="GroupC",
            ),
        ],
    )


SUMMARY_DRY_NO_CLIENT = ArrRunSummaryReady(
    summary=ArrRunSummary(
        arr=Arr.RADARR,
        dry_run_note="qBittorrent not configured; nothing grabbed",
        grabbed_count=1,
        tally=ArrRunTally.from_stats(dry_run_stats()),
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
        title="Pearlarr (Radarr) run complete   (DRY RUN - qBittorrent not configured; nothing grabbed)",
        heavy=True,
    ),
    _blank(),
    srow("  checked      : 1", "checked", "1"),
    srow("  needs action : 0", "needs action", "0"),
    srow("  grabbed      : 1", "grabbed", "1", style="green"),
    _styled("    Preview Movie", "grey50"),
    _brow("      link    https://releases.moe/12", "link", "https://releases.moe/12", "grey50"),
    _brow(
        "      torrent [GroupC] Preview.Movie.1080p-GroupC",
        "torrent",
        torrent_text("Preview.Movie.1080p-GroupC", "GroupC", dry=True),
        "grey50",
    ),
    srow("  up to date   : 0", "up to date", "0"),
    srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _RULE_LINE,
)


SUMMARY_MINIMAL = ArrRunSummaryReady(
    summary=ArrRunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        grabbed_count=0,
        tally=ArrRunTally.from_stats(ArrRunStats()),
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
    srow("  checked      : 0", "checked", "0"),
    srow("  needs action : 0", "needs action", "0"),
    srow("  grabbed      : 0", "grabbed", "0"),
    srow("  up to date   : 0", "up to date", "0"),
    srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _RULE_LINE,
)


def wait_off_stats() -> ArrRunStats:
    return ArrRunStats(
        checked=6,
        up_to_date=5,
        needs_action=[
            NeedsActionRow(
                title="Stale Show",
                coverage="S03 E01-E12",
                group="PrivGrp",
                url="https://releases.moe/33",
                reason="your copy of the private release is outdated",
                cause=NeedsActionCause.PRIVATE_ONLY_OUTDATED,
            ),
        ],
        # Non-zero pending import counters that must stay hidden with the wait mode off.
        downloading=2,
        downloaded=1,
        imported=3,
    )


SUMMARY_WAIT_OFF = ArrRunSummaryReady(
    summary=ArrRunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        grabbed_count=0,
        tally=ArrRunTally.from_stats(wait_off_stats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=NeedsActionCause.PRIVATE_ONLY_OUTDATED,
    ),
)
SUMMARY_WAIT_OFF_LINES: tuple[Line, ...] = (
    _blank(),
    _titled("Pearlarr (Sonarr) run complete", heavy=True),
    _blank(),
    srow("  checked      : 6", "checked", "6"),
    srow("  needs action : 1", "needs action", "1", style="yellow"),
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
    srow("  grabbed      : 0", "grabbed", "0"),
    srow("  up to date   : 5", "up to date", "5"),
    srow("  unchanged    : 0", "unchanged", "0", style="grey50"),
    srow("  issues       : 0 warnings, 0 errors", "issues", "0 warnings, 0 errors"),
    _styled(OUTDATED_TIP, "grey50"),
    _RULE_LINE,
)
