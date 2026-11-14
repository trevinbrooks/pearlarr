"""The scan's rich-console line grammar: one pure builder per scan event."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Final, NamedTuple, assert_never

from rich.text import Text

from .console_lines import ConsoleLine
from .console_text import (
    DETAIL_INDENT,
    DETAIL_KEY_WIDTH,
    ConsoleRender,
    KeyValue,
    SeparatorRule,
    StyledLine,
    TitledRule,
    arr_item_noun,
    count_noun,
    group_highlight,
    indent_string,
    kv_string,
    long_duration,
    rule_string,
    title_row_string,
)
from .events import (
    Accent,
    ArrRunSummaryReady,
    CapReached,
    GrabAction,
    GrabbedRow,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    NeedsActionCause,
    NeedsActionRow,
    ReleaseSkipped,
    ScanStarted,
    TitleDetail,
    TitleHeader,
    TitleRow,
    TitleState,
    severity_of,
)
from ..releases import SkipReason

type ScanEvent = (
    ScanStarted
    | ItemStarted
    | TitleHeader
    | TitleDetail
    | TitleRow
    | ReleaseSkipped
    | GrabFailed
    | GrabAction
    | CapReached
    | ArrRunSummaryReady
)


_BLANK = ConsoleLine(logging.INFO, "")

_SUMMARY_KEY_WIDTH: Final = 12
_BLOCK_KEY_WIDTH: Final = 7

_CAP_MESSAGE: Final = (
    "Reached the maximum number of torrents for this run (advanced.max_torrents_to_add); "
    "checking the remaining titles without grabbing"
)

# Causes without a tip render no line. The PRIVATE_ONLY > NO_FALLBACK > OUTDATED precedence is settled by
# whoever populates `ArrRunSummary.tip`, never re-derived here.
_TIP_TEXTS: Final[dict[NeedsActionCause, str]] = {
    NeedsActionCause.PRIVATE_ONLY: (
        "Tip: manually grab private releases or set private_releases: fallback to grab a public release instead."
    ),
    NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK: (
        "Tip: no public release covers these files yet; the title is re-checked every run until one does."
    ),
    NeedsActionCause.PRIVATE_ONLY_OUTDATED: (
        "Tip: your copies of these releases are outdated (their file sizes no longer match); "
        "update them from their private tracker, or delete the outdated files to let the "
        "public fallback stand in."
    ),
}


def accent_style(accent: Accent) -> str:
    """The rich style an `Accent` renders as in the scan lines."""

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


def _info(message: str, payload: ConsoleRender) -> ConsoleLine:
    return ConsoleLine(logging.INFO, message, payload)


def kv_line(payload: KeyValue, *, level: int = logging.INFO) -> ConsoleLine:
    """One kv line: the KeyValue payload plus the exact `kv_string` message it renders."""

    message = kv_string(payload.key, payload.value, key_width=payload.key_width, indent=payload.indent, sep=payload.sep)
    return ConsoleLine(level, message, payload)


def _detail_kv(
    key: str,
    value: str | Text,
    *,
    value_style: str | None,
    level: int = logging.INFO,
    tail: str | None = None,
) -> ConsoleLine:
    """A title detail line (the colon-less gutter kv indented under a title block)."""

    payload = KeyValue(
        key=key,
        value=value,
        key_width=DETAIL_KEY_WIDTH,
        value_style=value_style,
        indent=DETAIL_INDENT,
        sep="",
        tail=tail,
    )
    return kv_line(payload, level=level)


def _title_row_line(state: TitleState, label: str, style: str) -> ConsoleLine:
    """A fixed-column title row, indent baked into the message."""

    return _info(indent_string(title_row_string(state, label), level=1), StyledLine(style=style))


# --- the builders, one per event family ---------------------------------------------


def scan_started_lines(event: ScanStarted) -> tuple[ConsoleLine, ...]:
    """The run banner: a blank, then the heavy titled rule."""

    banner = f"Starting Pearlarr ({event.arr.capitalize()}) for {arr_item_noun(event.arr, event.total)}"
    return _BLANK, _info(banner, TitledRule(title=banner, heavy=True))


def item_started_lines(event: ItemStarted) -> tuple[ConsoleLine, ...]:
    """A per-item header: a blank, then the light titled rule."""

    header = f"[{event.index}/{event.total}] {event.arr.capitalize()}: {event.title}"
    return _BLANK, _info(header, TitledRule(title=header))


def title_header_lines(event: TitleHeader) -> tuple[ConsoleLine, ...]:
    """A title block's head (title row plus files/link rows), the incomplete note on the last row."""

    if event.state is TitleState.CHECKING:
        style = ""
    elif event.state is TitleState.IMPORTED:
        style = "green"
    else:
        style = "grey50"
    lines = [_BLANK, _title_row_line(event.state, event.title, style)]
    rows = [(label, value) for label, value in (("files", event.coverage), ("link", event.url)) if value]
    for idx, (label, value) in enumerate(rows):
        tail = "(marked incomplete on SeaDex)" if event.incomplete and idx == len(rows) - 1 else None
        lines.append(_detail_kv(label, value, value_style="grey50", tail=tail))
    return tuple(lines)


def title_row_lines(event: TitleRow) -> tuple[ConsoleLine, ...]:
    """A self-contained title row (unmonitored / no mapping / ignored / ...)."""

    return _BLANK, _title_row_line(event.state, event.label, accent_style(event.accent))


def title_detail_lines(event: TitleDetail) -> tuple[ConsoleLine, ...]:
    """One labeled detail line under a title. PLAIN renders a style-less kv."""

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
        case SkipReason.MISNUMBERED:
            return f"{event.url or event.group} (numbering does not match its SeaDex listing)"
        case SkipReason.EPISODE_TITLES_CONTRADICT:
            return f"{event.url or event.group} (episode titles contradict its numbers)"
        case SkipReason.PLACED_ELSEWHERE:
            counts = event.counts
            text = f"{count_noun(counts.another_titles, 'file')} {'is' if counts.another_titles == 1 else 'are'} another title's"
            if counts.unplaced:
                text += f", {counts.unplaced} could not be placed"
            return f"{event.url or event.group} ({text})"
        case SkipReason.PLACEMENT_INPUT_UNREAD:
            return f"{event.url or event.group} (a read the placement needs failed)"
    assert_never(event.reason)


def release_skipped_lines(event: ReleaseSkipped) -> tuple[ConsoleLine, ...]:
    """A per-release "skipped" detail row, level keyed on the reason's severity."""

    return (
        _detail_kv(
            "skipped",
            _skip_text(event),
            value_style=accent_style(Accent.CAUTION),
            level=int(event.severity),
        ),
    )


def grab_failed_lines(event: GrabFailed) -> tuple[ConsoleLine, ...]:
    """A contained grab failure's "failed" row, at `severity_of`'s level."""

    return (
        _detail_kv(
            "failed",
            f"could not grab {event.url}: {event.error}; will retry next run",
            value_style=accent_style(Accent.CAUTION),
            level=int(severity_of(event)),
        ),
    )


def grab_action_lines(event: GrabAction) -> tuple[ConsoleLine, ...]:
    """The per-title action block: status, release groups, per-release rows."""

    lines: list[ConsoleLine] = []
    match event.status:
        case GrabStatus.GRABBING:
            lines.append(_detail_kv("status", "grabbing", value_style=None))
        case GrabStatus.WOULD_GRAB:
            lines.append(_detail_kv("status", "would grab (dry run)", value_style=None))
        case GrabStatus.ALREADY_IN_QBIT:
            message = "already in qBittorrent"
            if event.waiting_to_import:
                message += " - waiting to import"
            lines.append(_detail_kv("status", message, value_style="yellow"))
    for group in event.groups:
        lines.append(_detail_kv("group", group.display, value_style="cyan"))
    grabbed_label = "would grab" if event.status is GrabStatus.WOULD_GRAB else "grabbed"
    for release in event.grabbed:
        lines.append(_detail_kv(grabbed_label, release.display, value_style="green"))
    for release in event.in_qbit:
        lines.append(_detail_kv("torrent", release.display, value_style="yellow"))
    return tuple(lines)


def cap_reached_lines(event: CapReached) -> tuple[ConsoleLine, ...]:
    """The cap line names the setting, not the cap value."""

    del event
    return (_info(_CAP_MESSAGE, StyledLine(style="yellow")),)


def _summary_kv(key: str, value: str, *, value_style: str | None = None) -> ConsoleLine:
    return kv_line(KeyValue(key=key, value=value, key_width=_SUMMARY_KEY_WIDTH, value_style=value_style, indent=1))


class SummaryRow(NamedTuple):
    """One labeled row of a summary block. A falsy `value` is skipped."""

    label: str
    value: str | Text | None
    accent: str


def _summary_block(
    title: str,
    title_style: str | None,
    rows: Iterable[SummaryRow],
) -> Iterator[ConsoleLine]:
    """A summary per-title block: title at indent 2, labeled rows at indent 3."""

    yield _info(indent_string(title, level=2), StyledLine(style=title_style or ""))
    for row in rows:
        if not row.value:
            continue
        yield kv_line(
            KeyValue(
                key=row.label, value=row.value, key_width=_BLOCK_KEY_WIDTH, value_style=row.accent, indent=3, sep=""
            ),
        )


def _needs_block(item: NeedsActionRow) -> Iterator[ConsoleLine]:
    rows = [
        SummaryRow("files", item.coverage, "grey50"),
        SummaryRow("group", item.group, "yellow"),
        SummaryRow("reason", item.reason, "yellow"),
        SummaryRow("link", item.url, "grey50"),
    ]
    yield from _summary_block(item.title or "(unknown title)", "yellow", rows)


def _grabbed_block(item: GrabbedRow, *, dry_run: bool) -> Iterator[ConsoleLine]:
    # A dry run dims the whole block (group accent included) so the would-be grabs don't read as real.
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


def arr_run_summary_lines(event: ArrRunSummaryReady) -> tuple[ConsoleLine, ...]:
    """The arr run summary: checked / needs-action / grabbed tallies plus the closing rule."""

    summary = event.summary
    tally = summary.tally

    title = f"Pearlarr ({summary.arr.capitalize()}) run complete"
    rule_title = title
    if summary.dry_run_note is not None:
        rule_title += f"   (DRY RUN - {summary.dry_run_note})"

    lines: list[ConsoleLine] = [
        _BLANK,
        # The DRY RUN note goes on the rule title only. The message stays plain (the text renderers carry
        # dry_run/note as structured fields via _fact_of).
        _info(title, TitledRule(title=rule_title, heavy=True)),
        _BLANK,
        _summary_kv("checked", str(tally.checked)),
    ]

    # Needs-action ahead of "grabbed": anything waiting on a decision surfaces first.
    needs = tally.needs_action
    lines.append(_summary_kv("needs action", str(len(needs)), value_style="yellow" if needs else None))
    for item in needs:
        lines.extend(_needs_block(item))

    lines.append(
        _summary_kv("grabbed", str(summary.grabbed_count), value_style="green" if summary.grabbed_count else None)
    )
    for grab in tally.grabbed:
        lines.extend(_grabbed_block(grab, dry_run=summary.dry_run))
    if tally.held_by_cap:
        lines.append(_summary_kv("held by cap", str(tally.held_by_cap), value_style="yellow"))

    if summary.wait_mode_on:
        if tally.downloading:
            lines.append(_summary_kv("downloading", str(tally.downloading), value_style="grey50"))
        if tally.downloaded:
            lines.append(_summary_kv("downloaded", str(tally.downloaded), value_style="yellow"))
        if tally.imported:
            lines.append(_summary_kv("imported", str(tally.imported), value_style="green"))

    lines.append(_summary_kv("up to date", str(tally.up_to_date)))
    lines.append(
        _summary_kv(
            "unchanged", f"{tally.unchanged}  (since last run)" if tally.unchanged else "0", value_style="grey50"
        ),
    )
    if tally.no_mappings:
        lines.append(_summary_kv("no mapping", str(tally.no_mappings)))
    if tally.no_seadex_entry:
        lines.append(_summary_kv("no entry", str(tally.no_seadex_entry)))
    if tally.seadex_unreachable:
        lines.append(_summary_kv("not checked", str(tally.seadex_unreachable), value_style="yellow"))
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
        lines.append(_summary_kv("elapsed", long_duration(summary.elapsed_s)))

    tip = _TIP_TEXTS.get(summary.tip) if summary.tip is not None else None
    if tip is not None:
        lines.append(_info(indent_string(tip, level=1), StyledLine(style="grey50")))

    lines.append(_info(rule_string("="), SeparatorRule("=")))
    return tuple(lines)


def scan_event_lines(event: ScanEvent) -> tuple[ConsoleLine, ...]:
    """Dispatch one scan event to its builder (the renderers' single entry point)."""

    match event:
        case ScanStarted():
            return scan_started_lines(event)
        case ItemStarted():
            return item_started_lines(event)
        case TitleHeader():
            return title_header_lines(event)
        case TitleDetail():
            return title_detail_lines(event)
        case TitleRow():
            return title_row_lines(event)
        case ReleaseSkipped():
            return release_skipped_lines(event)
        case GrabFailed():
            return grab_failed_lines(event)
        case GrabAction():
            return grab_action_lines(event)
        case CapReached():
            return cap_reached_lines(event)
        case ArrRunSummaryReady():
            return arr_run_summary_lines(event)
    assert_never(event)
