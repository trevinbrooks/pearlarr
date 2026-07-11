# pyright: strict
"""Tests for the scan-surface rendering layer.

Two halves over ONE golden spine (the data in `test_scan_parity`, captured
from the live reporter): the pure `scan_lines` builders reproduce the exact
`(level, message, payload)` tuples per event, and the RichRenderer's scan arm
renders the same lines through the shared payload renderers on the shared
Console at LOGGER-parity gating. Plus the generalized ENTRY-indent placement
rule for diagnostics.
"""

import io
import logging

from rich.console import Console

from pearlarr.modules.config import Arr
from pearlarr.modules.log import (
    LOG_NAME,
    EntryState,
    KvLine,
    SectionRule,
    StyledLine,
    TitledRule,
)
from pearlarr.modules.output import (
    Accent,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    GrabAction,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    RecommendedGroup,
    RichRenderer,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    StyledValue,
)
from pearlarr.modules.output.scan_lines import (
    LegacyLine,
    accent_style,
    cap_reached_lines,
    entry_detail_lines,
    entry_header_lines,
    grab_action_lines,
    item_started_lines,
    ledger_row_lines,
    run_summary_lines,
    scan_started_lines,
)

from .fakes import strip_ansi
from .test_scan_parity import (
    ACTION_DOWNLOADING_NO_WAIT,
    ACTION_DOWNLOADING_NO_WAIT_LINES,
    ACTION_DOWNLOADING_WAITING,
    ACTION_DOWNLOADING_WAITING_LINES,
    ACTION_DRY,
    ACTION_DRY_LINES,
    ACTION_FRESH,
    ACTION_FRESH_LINES,
    ACTION_MIXED,
    ACTION_MIXED_LINES,
    CACHED_BARE,
    CACHED_BARE_LINES,
    CACHED_FULL,
    CACHED_FULL_LINES,
    CACHED_IN_RADARR,
    CACHED_IN_RADARR_LINES,
    CAP_REACHED,
    CAP_REACHED_LINES,
    CHECKING_FULL,
    CHECKING_FULL_LINES,
    CHECKING_URL_ONLY,
    CHECKING_URL_ONLY_LINES,
    IGNORED_LINES,
    IGNORED_ROW,
    ITEM_STARTED_RADARR,
    ITEM_STARTED_RADARR_LINES,
    ITEM_STARTED_SONARR,
    ITEM_STARTED_SONARR_LINES,
    NO_ENTRY_RESOLVED_DETAIL,
    NO_ENTRY_RESOLVED_LINES,
    NO_ENTRY_RESOLVED_ROW,
    NO_ENTRY_UNRESOLVED_LINES,
    NO_ENTRY_UNRESOLVED_ROW,
    NO_MAPPING_LINES,
    NO_MAPPING_ROW,
    NO_RELEASES_DETAIL,
    NO_RELEASES_LINES,
    OUTAGE_ANILIST_DETAIL,
    OUTAGE_LINES,
    OUTAGE_ROW,
    OUTAGE_STATUS_DETAIL,
    PENDING_IMPORTED,
    PENDING_IMPORTED_LINES,
    PENDING_IMPORTING,
    PENDING_IMPORTING_LINES,
    PENDING_QUEUED,
    PENDING_QUEUED_LINES,
    SCAN_STARTED_RADARR,
    SCAN_STARTED_RADARR_LINES,
    SCAN_STARTED_SONARR,
    SCAN_STARTED_SONARR_LINES,
    SUMMARY_DRY_HAS_CLIENT,
    SUMMARY_DRY_HAS_CLIENT_LINES,
    SUMMARY_DRY_NO_CLIENT,
    SUMMARY_DRY_NO_CLIENT_LINES,
    SUMMARY_MINIMAL,
    SUMMARY_MINIMAL_LINES,
    SUMMARY_RICH,
    SUMMARY_RICH_LINES,
    SUMMARY_WAIT_OFF,
    SUMMARY_WAIT_OFF_LINES,
    UNMONITORED_LINES,
    UNMONITORED_ROW,
    Line,
)


def _as_lines(lines: tuple[LegacyLine, ...]) -> tuple[Line, ...]:
    return tuple((line.level, line.message, line.payload) for line in lines)


# --- the pure builders vs. the reporter goldens ---------------------------------------


class TestBuilders:
    def test_scan_started(self) -> None:
        assert _as_lines(scan_started_lines(SCAN_STARTED_SONARR)) == SCAN_STARTED_SONARR_LINES
        assert _as_lines(scan_started_lines(SCAN_STARTED_RADARR)) == SCAN_STARTED_RADARR_LINES

    def test_item_started(self) -> None:
        assert _as_lines(item_started_lines(ITEM_STARTED_SONARR)) == ITEM_STARTED_SONARR_LINES
        assert _as_lines(item_started_lines(ITEM_STARTED_RADARR)) == ITEM_STARTED_RADARR_LINES

    def test_ledger_rows(self) -> None:
        for event, expected in (
            (UNMONITORED_ROW, UNMONITORED_LINES),
            (NO_MAPPING_ROW, NO_MAPPING_LINES),
            (IGNORED_ROW, IGNORED_LINES),
            (NO_ENTRY_UNRESOLVED_ROW, NO_ENTRY_UNRESOLVED_LINES),
        ):
            assert _as_lines(ledger_row_lines(event)) == expected

    def test_ledger_row_accents(self) -> None:
        plain = LedgerRow(state=EntryState.CHECKING, label="Focal", accent=Accent.PLAIN)
        assert ledger_row_lines(plain)[1].payload == StyledLine(style="")
        good = LedgerRow(state=EntryState.IMPORTED, label="Done", accent=Accent.GOOD)
        assert ledger_row_lines(good)[1].payload == StyledLine(style="green")

    def test_entry_headers(self) -> None:
        for event, expected in (
            (CHECKING_FULL, CHECKING_FULL_LINES),
            (CHECKING_URL_ONLY, CHECKING_URL_ONLY_LINES),
            (CACHED_FULL, CACHED_FULL_LINES),
            (CACHED_IN_RADARR, CACHED_IN_RADARR_LINES),
            (CACHED_BARE, CACHED_BARE_LINES),
            (PENDING_QUEUED, PENDING_QUEUED_LINES),
            (PENDING_IMPORTING, PENDING_IMPORTING_LINES),
            (PENDING_IMPORTED, PENDING_IMPORTED_LINES),
        ):
            assert _as_lines(entry_header_lines(event)) == expected

    def test_incomplete_tail_rides_the_files_row_when_url_absent(self) -> None:
        event = EntryHeader(state=EntryState.CHECKING, title="Frieren", coverage="S01 E01-E28", incomplete=True)
        last = entry_header_lines(event)[-1]
        assert last.message == "    files     S01 E01-E28"
        assert isinstance(last.payload, KvLine)
        assert last.payload.tail == "(marked incomplete on SeaDex)"

    def test_entry_details(self) -> None:
        composed = (
            _as_lines(ledger_row_lines(OUTAGE_ROW))
            + _as_lines(entry_detail_lines(OUTAGE_ANILIST_DETAIL))
            + _as_lines(entry_detail_lines(OUTAGE_STATUS_DETAIL))
        )
        assert composed == OUTAGE_LINES
        resolved = _as_lines(ledger_row_lines(NO_ENTRY_RESOLVED_ROW)) + _as_lines(
            entry_detail_lines(NO_ENTRY_RESOLVED_DETAIL),
        )
        assert resolved == NO_ENTRY_RESOLVED_LINES
        assert _as_lines(entry_detail_lines(NO_RELEASES_DETAIL)) == NO_RELEASES_LINES

    def test_entry_detail_carries_its_severity_and_tail(self) -> None:
        event = EntryDetail(
            label="status",
            value=StyledValue("size mismatch", Accent.CAUTION),
            severity=Severity.WARNING,
            tail="(re-checked next run)",
        )
        (line,) = entry_detail_lines(event)
        assert line.level == logging.WARNING
        assert line.message == "    status    size mismatch"
        assert line.payload == KvLine(
            key="status",
            value="size mismatch",
            key_width=9,
            value_style="yellow",
            indent=2,
            sep="",
            tail="(re-checked next run)",
        )

    def test_grab_actions(self) -> None:
        for event, expected in (
            (ACTION_FRESH, ACTION_FRESH_LINES),
            (ACTION_MIXED, ACTION_MIXED_LINES),
            (ACTION_DRY, ACTION_DRY_LINES),
            (ACTION_DOWNLOADING_WAITING, ACTION_DOWNLOADING_WAITING_LINES),
            (ACTION_DOWNLOADING_NO_WAIT, ACTION_DOWNLOADING_NO_WAIT_LINES),
        ):
            assert _as_lines(grab_action_lines(event)) == expected

    def test_multi_tag_group_renders_in_event_order(self) -> None:
        # The event carries an ORDERED tag tuple, fixing today's frozenset-random
        # join order; the rendered form is pinned here.
        event = GrabAction(
            status=GrabStatus.ADDING,
            groups=(RecommendedGroup(name="GroupA", tags=("HDR", "Dual Audio")),),
            added=(),
            downloading=(),
        )
        assert grab_action_lines(event)[1].message == "    group     GroupA [HDR, Dual Audio]"

    def test_cap_reached(self) -> None:
        assert _as_lines(cap_reached_lines(CAP_REACHED)) == CAP_REACHED_LINES

    def test_run_summaries(self) -> None:
        for event, expected in (
            (SUMMARY_RICH, SUMMARY_RICH_LINES),
            (SUMMARY_DRY_HAS_CLIENT, SUMMARY_DRY_HAS_CLIENT_LINES),
            (SUMMARY_DRY_NO_CLIENT, SUMMARY_DRY_NO_CLIENT_LINES),
            (SUMMARY_MINIMAL, SUMMARY_MINIMAL_LINES),
            (SUMMARY_WAIT_OFF, SUMMARY_WAIT_OFF_LINES),
        ):
            assert _as_lines(run_summary_lines(event)) == expected

    def test_accent_style_covers_every_accent(self) -> None:
        assert {accent: accent_style(accent) for accent in Accent} == {
            Accent.PLAIN: "",
            Accent.DIM: "grey50",
            Accent.GOOD: "green",
            Accent.CAUTION: "yellow",
            Accent.BAD: "red",
            Accent.ACCENT: "cyan",
            Accent.FOCUS: "bold",
            Accent.NOTE: "blue",
        }


# --- the RichRenderer scan console arm ---------------------------------------------------

_WIDTH = 100


def _renderer(width: int = _WIDTH) -> tuple[RichRenderer, io.StringIO]:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=width)
    return RichRenderer(lambda: console), stream


def _plain_lines(stream: io.StringIO) -> list[str]:
    return strip_ansi(stream.getvalue()).replace("\r", "").splitlines()


def _expected_console(*line_groups: tuple[Line, ...], width: int = _WIDTH) -> list[str]:
    """The console text the golden lines should render as (derived independently
    of the renderer: rules span the width, kv tails append, titles are the
    payload's — possibly console-only-annotated — title)."""

    out: list[str] = []
    for lines in line_groups:
        for _level, message, payload in lines:
            match payload:
                case TitledRule(title=title, heavy=heavy):
                    out.append(("━" if heavy else "─") * width)
                    out.append(title)
                case SectionRule(char=char):
                    out.append(("━" if "=" in char else "─") * width)
                case KvLine(tail=tail):
                    out.append(f"{message} {tail}" if tail else message)
                case StyledLine() | None:
                    out.append(message)
    return out


class TestRichRendererScanArm:
    def test_renders_a_scan_sequence_in_order(self) -> None:
        renderer, stream = _renderer()
        for event in (SCAN_STARTED_SONARR, ITEM_STARTED_SONARR, CHECKING_FULL, ACTION_FRESH, CAP_REACHED):
            renderer.handle(event, 0.0)
        assert _plain_lines(stream) == _expected_console(
            SCAN_STARTED_SONARR_LINES,
            ITEM_STARTED_SONARR_LINES,
            CHECKING_FULL_LINES,
            ACTION_FRESH_LINES,
            CAP_REACHED_LINES,
        )

    def test_renders_the_summary_with_the_console_only_dry_note(self) -> None:
        renderer, stream = _renderer()
        renderer.handle(SUMMARY_DRY_HAS_CLIENT, 0.0)
        lines = _plain_lines(stream)
        assert lines == _expected_console(SUMMARY_DRY_HAS_CLIENT_LINES)
        # The DRY RUN note the file log drops rides the console title.
        assert "Pearlarr (Sonarr) run complete   (DRY RUN — nothing grabbed)" in lines

    def test_renders_the_rich_summary(self) -> None:
        renderer, stream = _renderer()
        renderer.handle(SUMMARY_RICH, 0.0)
        assert _plain_lines(stream) == _expected_console(SUMMARY_RICH_LINES)

    def test_warning_level_suppresses_info_scan_lines(self) -> None:
        # LOGGER parity: at configured WARNING the INFO scan lines vanish from
        # the console exactly as they vanish from the file.
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)
        for event in (SCAN_STARTED_SONARR, CHECKING_FULL, SUMMARY_RICH):
            renderer.handle(event, 0.0)
        assert _plain_lines(stream) == []

    def test_warning_detail_renders_at_warning_level(self) -> None:
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)
        detail = EntryDetail(
            label="status", value=StyledValue("size mismatch", Accent.CAUTION), severity=Severity.WARNING
        )
        renderer.handle(detail, 0.0)
        assert _plain_lines(stream) == ["    status    size mismatch"]

    def test_without_a_console_scan_events_no_op(self) -> None:
        renderer = RichRenderer(lambda: None)
        for event in (SCAN_STARTED_SONARR, ITEM_STARTED_SONARR, CHECKING_FULL, ACTION_FRESH, SUMMARY_RICH):
            renderer.handle(event, 0.0)


class TestEntryIndentedDiagnostics:
    """The generalized placement rule: a diagnostic indents while an ENTRY scope
    is open (like boot/wait); under RUN/ITEM alone it stays column-0."""

    _ENTRY = ScopeId(ScopeKind.ENTRY, 1)

    def _open_item(self, renderer: RichRenderer) -> None:
        renderer.handle(ScanStarted(arr=Arr.SONARR, total=1), 0.0)
        renderer.handle(ItemStarted(arr=Arr.SONARR, index=1, total=1, title="Frieren"), 0.0)

    def test_diagnostic_indents_inside_an_open_entry_block(self) -> None:
        renderer, stream = _renderer()
        self._open_item(renderer)
        renderer.handle(ScopeOpened(scope=self._ENTRY, label="entry"), 0.0)
        renderer.handle(Diagnostic(severity=Severity.WARNING, message="tracker down", origin=LOG_NAME), 0.0)
        assert "  WARNING  tracker down" in _plain_lines(stream)

    def test_diagnostic_stays_column_zero_under_run_and_item_alone(self) -> None:
        renderer, stream = _renderer()
        self._open_item(renderer)
        renderer.handle(Diagnostic(severity=Severity.WARNING, message="tracker down", origin=LOG_NAME), 0.0)
        assert "WARNING  tracker down" in _plain_lines(stream)
        assert "  WARNING  tracker down" not in _plain_lines(stream)

    def test_closing_the_entry_scope_returns_diagnostics_to_column_zero(self) -> None:
        renderer, stream = _renderer()
        self._open_item(renderer)
        renderer.handle(ScopeOpened(scope=self._ENTRY, label="entry"), 0.0)
        renderer.handle(ScopeClosed(scope=self._ENTRY), 0.0)
        renderer.handle(Diagnostic(severity=Severity.WARNING, message="tracker down", origin=LOG_NAME), 0.0)
        assert "WARNING  tracker down" in _plain_lines(stream)
        assert "  WARNING  tracker down" not in _plain_lines(stream)
