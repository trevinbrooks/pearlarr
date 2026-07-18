"""The scan surface's rich-console line grammar, event-driven.

Every scan line is a `LegacyLine`: a level, a plain message, and a typed
`ConsoleRender` payload (how the rich console draws it). The pure builders
here map each scan event to its console lines, pinned by the goldens in
`tests/test_scan_parity.py`.

Two consumers: the `rich_renderer.RichRenderer`'s scan arm renders the
payloads via `render_legacy_lines` (through the shared payload renderers
`render_kv` / `render_rule` / `print_titled_rule`), and the WaitRegion's
durable prints ride the same route. The file/plain/json surfaces take the same
events through the `textline` grammar instead. Renderer-side module:
importing rich is fine here, unlike `events.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final, NamedTuple, assert_never

from rich.console import Console
from rich.text import Text

from .events import (
    Accent,
    CapReached,
    EntryDetail,
    EntryHeader,
    GrabAction,
    GrabFact,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    NeedsActionCause,
    NeedsActionFact,
    ReleaseSkipped,
    RunSummaryReady,
    ScanStarted,
    SkipReason,
    severity_of,
)
from ..log import (
    DETAIL_INDENT,
    DETAIL_KEY_WIDTH,
    ConsoleRender,
    EntryState,
    KvLine,
    SectionRule,
    StyledLine,
    TitledRule,
    arr_item_noun,
    count_noun,
    entry_string,
    format_elapsed,
    group_highlight,
    indent_string,
    kv_string,
    print_literal,
    print_titled_rule,
    render_kv,
    render_rule,
    rule_string,
)

type ScanEvent = (
    ScanStarted
    | ItemStarted
    | EntryHeader
    | EntryDetail
    | LedgerRow
    | ReleaseSkipped
    | GrabFailed
    | GrabAction
    | CapReached
    | RunSummaryReady
)
"""The event subset rendered through the legacy-line builders (both seats)."""


@dataclass(frozen=True, slots=True)
class LegacyLine:
    """One rich-console line: level (gates rendering) + plain message + payload."""

    level: int
    message: str
    payload: ConsoleRender | None = None


_BLANK = LegacyLine(logging.INFO, "")

# The summary scoreboard's key column (narrower than the entry-detail column)
# and its per-entry block column.
_SUMMARY_KEY_WIDTH: Final = 12
_BLOCK_KEY_WIDTH: Final = 7

_CAP_MESSAGE: Final = "Reached the maximum number of torrents for this run (advanced.max_torrents_to_add); stopping"

# The summary guidance tips by cause. Causes without a tip render no line. The
# PRIVATE_ONLY > NO_FALLBACK > STALE precedence is settled by whoever populates
# `RunSummary.tip` (the producer), never re-derived here.
_TIP_TEXTS: Final[dict[NeedsActionCause, str]] = {
    NeedsActionCause.PRIVATE_ONLY: (
        "Tip: manually grab private releases or set private_releases: fallback to "
        "automatically grab public alternatives."
    ),
    NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK: (
        "Tip: no public alternative exists yet; the title is re-checked every run until one appears."
    ),
    NeedsActionCause.PRIVATE_ONLY_STALE: (
        "Tip: your copies of these releases are outdated (their file sizes no longer match); "
        "update them from their private tracker, or delete the outdated files to let the "
        "public fallback stand in."
    ),
}


def accent_style(accent: Accent) -> str:
    """The rich style an `Accent` renders as on the scan surface."""

    match accent:
        case Accent.PLAIN:
            return ""
        case Accent.DIM:
            return "grey50"
        case Accent.GOOD:
            return "green"
        case Accent.CAUTION:
            return "yellow"
        case Accent.BAD:
            return "red"
        case Accent.ACCENT:
            return "cyan"
        case Accent.FOCUS:
            return "bold"
        case Accent.NOTE:
            return "blue"
    assert_never(accent)


def _info(message: str, payload: ConsoleRender) -> LegacyLine:
    return LegacyLine(logging.INFO, message, payload)


def _kv_line(payload: KvLine, *, level: int = logging.INFO) -> LegacyLine:
    """One kv line: the KvLine payload plus the exact `kv_string` message it renders."""

    message = kv_string(payload.key, payload.value, key_width=payload.key_width, indent=payload.indent, sep=payload.sep)
    return LegacyLine(level, message, payload)


def _detail_kv(
    key: str,
    value: str | Text,
    *,
    value_style: str | None,
    level: int = logging.INFO,
    tail: str | None = None,
) -> LegacyLine:
    """An entry-detail line (the colon-less gutter kv indented under an entry block)."""

    payload = KvLine(
        key=key,
        value=value,
        key_width=DETAIL_KEY_WIDTH,
        value_style=value_style,
        indent=DETAIL_INDENT,
        sep="",
        tail=tail,
    )
    return _kv_line(payload, level=level)


def _ledger_line(state: EntryState, label: str, style: str) -> LegacyLine:
    """A fixed-column ledger row, indent baked into the message."""

    return _info(indent_string(entry_string(state, label), level=1), StyledLine(style=style))


# --- the builders, one per event family ---------------------------------------------


def scan_started_lines(event: ScanStarted) -> tuple[LegacyLine, ...]:
    """The run banner: a blank, then the heavy titled rule."""

    banner = f"Starting Pearlarr ({event.arr.capitalize()}) for {arr_item_noun(event.arr, event.total)}"
    return _BLANK, _info(banner, TitledRule(title=banner, heavy=True))


def item_started_lines(event: ItemStarted) -> tuple[LegacyLine, ...]:
    """A per-item header: a blank, then the light titled rule."""

    header = f"[{event.index}/{event.total}] {event.arr.capitalize()}: {event.title}"
    return _BLANK, _info(header, TitledRule(title=header))


def entry_header_lines(event: EntryHeader) -> tuple[LegacyLine, ...]:
    """An entry block's head: the ledger row plus its files/link continuation.

    The focal "checking" row stays unstyled. "imported" reads green. Every other
    state dims. Absent coverage/url rows drop, and the incomplete note rides the
    LAST rendered detail line, console-side only.
    """

    if event.state is EntryState.CHECKING:
        style = ""
    elif event.state is EntryState.IMPORTED:
        style = "green"
    else:
        style = "grey50"
    lines = [_BLANK, _ledger_line(event.state, event.title, style)]
    rows = [(label, value) for label, value in (("files", event.coverage), ("link", event.url)) if value]
    for idx, (label, value) in enumerate(rows):
        tail = "(marked incomplete on SeaDex)" if event.incomplete and idx == len(rows) - 1 else None
        lines.append(_detail_kv(label, value, value_style="grey50", tail=tail))
    return tuple(lines)


def ledger_row_lines(event: LedgerRow) -> tuple[LegacyLine, ...]:
    """A self-contained ledger row (unmonitored / no mapping / ignored / ...)."""

    return _BLANK, _ledger_line(event.state, event.label, accent_style(event.accent))


def entry_detail_lines(event: EntryDetail) -> tuple[LegacyLine, ...]:
    """One labeled detail line under an entry. PLAIN renders a style-less kv."""

    return (
        _detail_kv(
            event.label,
            event.value.text,
            value_style=accent_style(event.value.accent) or None,
            level=int(event.severity),
            tail=event.tail,
        ),
    )


def _skip_text(event: ReleaseSkipped) -> str:
    """The skip line's text by reason (`or event.group` is the None-url fallback)."""

    match event.reason:
        case SkipReason.PRIVATE_ONLY:
            return f"{event.group} on {event.tracker} (private-only)"
        case SkipReason.TRACKER_NOT_SELECTED:
            return f"{event.url or event.group} (tracker {event.tracker} not in your selected list)"
        case SkipReason.UNSUPPORTED_TRACKER:
            return f"{event.url or event.group} (tracker {event.tracker} not yet supported)"
    assert_never(event.reason)


def release_skipped_lines(event: ReleaseSkipped) -> tuple[LegacyLine, ...]:
    """A per-release "skipped" detail row, level keyed on the reason's severity."""

    return (
        _detail_kv(
            "skipped",
            _skip_text(event),
            value_style=accent_style(Accent.CAUTION),
            level=int(event.reason.severity),
        ),
    )


def grab_failed_lines(event: GrabFailed) -> tuple[LegacyLine, ...]:
    """A contained grab failure's "failed" row, at `severity_of`'s level."""

    return (
        _detail_kv(
            "failed",
            f"could not grab {event.url}: {event.error}; will retry next run",
            value_style=accent_style(Accent.CAUTION),
            level=int(severity_of(event)),
        ),
    )


def grab_action_lines(event: GrabAction) -> tuple[LegacyLine, ...]:
    """The per-title action block: status, recommended groups, per-release rows.

    The should-anything-render gate stays producer-side. This renders whatever
    the event carries.
    """

    lines: list[LegacyLine] = []
    match event.status:
        case GrabStatus.ADDING:
            lines.append(_detail_kv("status", "adding SeaDex's recommended release", value_style=None))
        case GrabStatus.WOULD_ADD:
            lines.append(_detail_kv("status", "would add SeaDex's recommended release (dry run)", value_style=None))
        case GrabStatus.ALREADY_DOWNLOADING:
            message = "SeaDex's pick is already downloading in qBittorrent"
            if event.waiting_to_import:
                message += " - waiting to import"
            lines.append(_detail_kv("status", message, value_style="yellow"))
    for group in event.groups:
        lines.append(_detail_kv("group", group.display, value_style="cyan"))
    added_label = "would add" if event.status is GrabStatus.WOULD_ADD else "added"
    for release in event.added:
        lines.append(_detail_kv(added_label, release.display, value_style="green"))
    for release in event.downloading:
        lines.append(_detail_kv("downloading", release.display, value_style="yellow"))
    return tuple(lines)


def cap_reached_lines(event: CapReached) -> tuple[LegacyLine, ...]:
    """The cap line names the setting, not the cap value."""

    del event
    return (_info(_CAP_MESSAGE, StyledLine(style="yellow")),)


def _summary_kv(key: str, value: str, *, value_style: str | None = None) -> LegacyLine:
    return _kv_line(KvLine(key=key, value=value, key_width=_SUMMARY_KEY_WIDTH, value_style=value_style, indent=1))


class SummaryRow(NamedTuple):
    """One labeled row of a summary per-entry block. A falsy `value` is skipped."""

    label: str
    value: str | Text | None
    accent: str


def _summary_block(
    title: str,
    title_style: str | None,
    rows: Iterable[SummaryRow],
) -> Iterator[LegacyLine]:
    """A summary per-entry block: title at indent 2, labeled rows at indent 3."""

    yield _info(indent_string(title, level=2), StyledLine(style=title_style or ""))
    for row in rows:
        if not row.value:
            continue
        yield _kv_line(
            KvLine(
                key=row.label, value=row.value, key_width=_BLOCK_KEY_WIDTH, value_style=row.accent, indent=3, sep=""
            ),
        )


def _needs_block(item: NeedsActionFact) -> Iterator[LegacyLine]:
    rows = [
        SummaryRow("files", item.coverage, "grey50"),
        SummaryRow("group", item.group, "yellow"),
        SummaryRow("reason", item.reason, "yellow"),
        SummaryRow("link", item.url, "grey50"),
    ]
    yield from _summary_block(item.title or "(unknown title)", "yellow", rows)


def _added_block(item: GrabFact, *, dry_run: bool) -> Iterator[LegacyLine]:
    # A dry run dims the whole block (group accent included) so the would-be
    # grabs don't read as real. kv_string interpolates the Text to plain text
    # for the message.
    torrent_value = group_highlight(
        item.name,
        item.group,
        group_style="grey50" if dry_run else "cyan",
        base_style="grey50" if dry_run else "green",
    )
    rows = [
        SummaryRow("files", item.coverage, "grey50"),
        SummaryRow("link", item.url, "grey50"),
        SummaryRow("torrent", torrent_value, "grey50" if dry_run else "green"),
    ]
    yield from _summary_block(item.title or "(unknown title)", "grey50" if dry_run else None, rows)


def run_summary_lines(event: RunSummaryReady) -> tuple[LegacyLine, ...]:
    """The end-of-run scoreboard: checked / needs-action / added tallies plus the closing section rule."""

    summary = event.summary
    tally = summary.tally

    title = f"Pearlarr ({summary.arr.capitalize()}) run complete"
    rule_title = title
    if summary.dry_run_note is not None:
        rule_title += f"   (DRY RUN - {summary.dry_run_note})"

    lines: list[LegacyLine] = [
        _BLANK,
        # The DRY RUN note rides the rule title only. The message stays plain (the
        # text surfaces carry dry_run/note as structured fields via _fact_of).
        _info(title, TitledRule(title=rule_title, heavy=True)),
        _BLANK,
        _summary_kv("checked", str(tally.checked)),
    ]

    # Needs-action ahead of "added": anything waiting on a decision surfaces first.
    needs = tally.needs_action
    lines.append(_summary_kv("needs action", str(len(needs)), value_style="yellow" if needs else None))
    for item in needs:
        lines.extend(_needs_block(item))

    lines.append(_summary_kv("added", str(summary.added_count), value_style="green" if summary.added_count else None))
    for grab in tally.added:
        lines.extend(_added_block(grab, dry_run=summary.dry_run))

    # Carried-over pending statuses render only when the feature is on AND non-zero.
    if summary.wait_mode_on:
        if tally.queued:
            lines.append(_summary_kv("queued", str(tally.queued), value_style="grey50"))
        if tally.importing:
            lines.append(_summary_kv("importing", str(tally.importing), value_style="yellow"))
        if tally.imported:
            lines.append(_summary_kv("imported", str(tally.imported), value_style="green"))

    lines.append(_summary_kv("up to date", str(tally.up_to_date)))
    lines.append(
        _summary_kv("unchanged", f"{tally.cached}  (since last run)" if tally.cached else "0", value_style="grey50"),
    )
    if tally.no_mappings:
        lines.append(_summary_kv("no mapping", str(tally.no_mappings)))
    if tally.no_seadex_entry:
        lines.append(_summary_kv("no entry", str(tally.no_seadex_entry)))
    if tally.seadex_unreachable:
        lines.append(_summary_kv("seadex down", str(tally.seadex_unreachable), value_style="yellow"))
    if tally.no_releases:
        lines.append(_summary_kv("no release", str(tally.no_releases)))
    if tally.unmonitored:
        lines.append(_summary_kv("unmonitored", str(tally.unmonitored)))

    lines.append(
        _summary_kv(
            "issues",
            f"{count_noun(summary.warnings, 'warning')}, {count_noun(summary.errors, 'error')}",
            value_style="bold red" if summary.errors else ("yellow" if summary.warnings else None),
        ),
    )
    if summary.elapsed_s is not None:
        lines.append(_summary_kv("elapsed", format_elapsed(summary.elapsed_s)))

    tip = _TIP_TEXTS.get(summary.tip) if summary.tip is not None else None
    if tip is not None:
        lines.append(_info(indent_string(tip, level=1), StyledLine(style="grey50")))

    lines.append(_info(rule_string("="), SectionRule("=")))
    return tuple(lines)


def scan_event_lines(event: ScanEvent) -> tuple[LegacyLine, ...]:
    """Dispatch one scan event to its builder (the seats' single entry point)."""

    match event:
        case ScanStarted():
            return scan_started_lines(event)
        case ItemStarted():
            return item_started_lines(event)
        case EntryHeader():
            return entry_header_lines(event)
        case EntryDetail():
            return entry_detail_lines(event)
        case LedgerRow():
            return ledger_row_lines(event)
        case ReleaseSkipped():
            return release_skipped_lines(event)
        case GrabFailed():
            return grab_failed_lines(event)
        case GrabAction():
            return grab_action_lines(event)
        case CapReached():
            return cap_reached_lines(event)
        case RunSummaryReady():
            return run_summary_lines(event)
    assert_never(event)


def render_legacy_lines(console: Console, lines: Iterable[LegacyLine], level: int) -> None:
    """Render legacy lines on the shared console, LOGGER-parity gated.

    A line prints through the legacy payload renderers iff its level clears
    `level`, so a configured WARNING hides INFO scan lines from the console
    exactly as it hides them from the file (NOT the diagnostics' console
    floor).
    """

    for line in lines:
        if line.level < level:
            continue
        match line.payload:
            case TitledRule(title=title, style=style, heavy=heavy):
                print_titled_rule(console, title, style, heavy=heavy)
            case SectionRule(char=char):
                console.print(render_rule(char))
            case KvLine() as kv:
                print_literal(console, render_kv(kv))
            case (StyledLine() | None) as payload:
                style = payload.style if payload is not None else ""
                print_literal(console, Text(line.message, style=style))
