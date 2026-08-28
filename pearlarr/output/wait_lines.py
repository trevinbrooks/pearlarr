"""Pure builders and reducers for the wait pass: the rich-free live-frame model and the ledger lines."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from .events import (
    Phase,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitKind,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    clamp01,
    severity_of,
)
from .scan_lines import LegacyLine
from ..log import (
    STATE_WIDTH,
    SectionRule,
    StyledLine,
    compact_duration,
    count_noun,
    format_elapsed,
    human_bytes,
    indent_string,
    rule_string,
)
from ..manual_import import Outcome, OutcomeCategory

if TYPE_CHECKING:
    from ..console_caps import Capabilities

type WaitEvent = WaitStarted | WaitProgress | TorrentGraduated | WaitFinished

# The live cockpit never grows past this many in-flight rows, the rest collapse into a one-line
# "+ N more ..." overflow so a large carried-over backlog can't blow the region past the screen.
MAX_LIVE_ROWS = 12
MIN_LIVE_ROWS = 4
# Banner, header, overflow line and breathing room, when clamping rows to the terminal height.
_RESERVED_ROWS = 8
# The sparkline needs unicode blocks and enough width not to crowd the label.
MIN_SPARK_WIDTH = 80


def graduation_tail(outcome: Outcome, files: int | None, waited_s: float) -> str:
    """The ledger line's parenthesized coda. "" when there is nothing to say."""

    if outcome is Outcome.IMPORTED:
        parts: list[str] = []
        if files:
            parts.append(count_noun(files, "file"))
        if waited_s >= 1.0:
            parts.append(format_elapsed(waited_s))
        return " · ".join(parts)
    if outcome.category is OutcomeCategory.PENDING:
        return "checked next run"
    if not outcome.dropped:
        return "retries next run"
    # The only other dropped outcome is MISSING: the torrent is gone from qBittorrent.
    return "no longer tracked"


# --- the durable ledger-line builders (the rich console's scrollback) ----------------


def wait_start_line(event: WaitStarted) -> LegacyLine:
    """The non-TTY digest's opening line, worded by the pass kind."""

    if event.kind is WaitKind.CHECK:
        message = f"Checking {count_noun(event.total, 'carried-over download')}..."
    else:
        message = f"Waiting on {count_noun(event.total, 'download')} to complete and import..."
    return LegacyLine(logging.INFO, message)


def wait_pulse_line(snapshot: WaitSnapshot) -> LegacyLine:
    """One throttled "still waiting" aggregate pulse."""

    counts = snapshot.counts()
    message = indent_string(
        f"still waiting · {counts[Phase.DOWNLOADING]} downloading · "
        f"{counts[Phase.IMPORTING]} importing · {counts[Phase.QUEUED]} queued · "
        f"{format_elapsed(snapshot.elapsed_s)}",
    )
    return LegacyLine(logging.INFO, message)


def wait_graduation_line(event: TorrentGraduated, caps: Capabilities) -> LegacyLine:
    """A finished torrent's durable ledger line: glyph + word + label + coda, at `severity_of`'s level."""

    glyph = event.outcome.glyph(use_unicode=caps.unicode)
    line = f"{glyph} {event.outcome.word.ljust(STATE_WIDTH)} {event.label}"
    tail = graduation_tail(event.outcome, event.files, event.waited_s)
    if tail:
        line += f"  ({tail})"
    return LegacyLine(
        int(severity_of(event)),
        indent_string(line),
        StyledLine(style=event.outcome.style if caps.color else ""),
    )


def wait_tally_lines(event: WaitFinished) -> list[LegacyLine]:
    """The closing wait summary. `[]` when nothing graduated."""

    if event.imported == 0 and event.pending == 0 and event.deferred == 0 and event.failed == 0:
        return []
    parts = [f"{event.imported} imported"]
    if event.pending:
        parts.append(f"{event.pending} pending")
    if event.deferred:
        parts.append(f"{event.deferred} left")
    if event.failed:
        parts.append(f"{event.failed} failed")
    parts.append(format_elapsed(event.elapsed_s))
    head = "check complete" if event.kind is WaitKind.CHECK else "wait complete"
    return [
        LegacyLine(logging.INFO, rule_string("-"), SectionRule(char="-")),
        LegacyLine(logging.INFO, indent_string(f"{head} · " + " · ".join(parts))),
    ]


@final
class PulseThrottle:
    """The "still waiting" pulse cadence."""

    __slots__ = ("_interval", "_next", "_skip_first")

    def __init__(self) -> None:
        self._interval: float | None = None
        self._next = 0.0
        self._skip_first = False

    def arm(self, interval_s: float) -> None:
        self._interval = interval_s
        self._next = interval_s
        self._skip_first = True

    def fire(self, elapsed_s: float) -> bool:
        """Advance the cadence. True when a pulse is due at `elapsed_s`."""

        if self._interval is None:
            return False
        if self._skip_first:
            self._skip_first = False
            return False
        if elapsed_s < self._next:
            return False
        self._next = elapsed_s + self._interval
        return True

    def reset(self) -> None:
        self._interval = None
        self._next = 0.0
        self._skip_first = False


@dataclass(frozen=True, slots=True)
class RowModel:
    """One rendered in-flight row: plain strings, no rich."""

    label: str
    phase: Phase
    fraction: float
    status: str = ""
    """The status word drawn in place of the bar."""
    count: str = ""
    """Progress, e.g. "61%" or "8/12" files."""
    speed: str = ""
    """The download rate: sparkline + rate, or "stalled"."""
    time: str = ""
    """The ETA for a download, or the elapsed clock for an import."""
    size: str = ""
    """The total download size."""
    show_bar: bool = False
    """Draw a determinate block bar for `fraction`."""


@dataclass(frozen=True, slots=True)
class LiveModel:
    """A bounded, ordered, rich-free description of the live cockpit frame."""

    left_text: str
    right_text: str
    overall_fraction: float
    rows: tuple[RowModel, ...]
    overflow: str = ""


def live_model(snapshot: WaitSnapshot, caps: Capabilities, kind: WaitKind = WaitKind.MONITOR) -> LiveModel:
    """Reduce a snapshot to a bounded, ordered cockpit frame."""

    in_flight = [t for t in snapshot.torrents if t.phase is not Phase.TERMINAL]
    in_flight.sort(key=_row_sort_key)

    budget = max(MIN_LIVE_ROWS, min(MAX_LIVE_ROWS, caps.height - _RESERVED_ROWS))
    visible = in_flight[:budget]
    hidden = in_flight[budget:]

    spark = caps.unicode and caps.width >= MIN_SPARK_WIDTH
    rows = tuple(_row_model(t, spark=spark) for t in visible)
    overflow = _overflow_text(hidden)

    counts = snapshot.counts()
    verb = "checking" if kind is WaitKind.CHECK else "waiting"
    left = f"{verb} {snapshot.done()}/{snapshot.total()}"
    arrow = "↓" if caps.unicode else "dl"
    meta: list[str] = [format_elapsed(snapshot.elapsed_s)]
    agg_speed = _aggregate_speed(snapshot)
    if agg_speed:
        meta.append(f"{arrow} {human_bytes(agg_speed)}/s")
    agg_eta = _aggregate_eta(snapshot, agg_speed)
    if agg_eta is not None:
        meta.append(f"{_compact_eta(agg_eta)} left")
    if counts[Phase.IMPORTING]:
        meta.append(f"{counts[Phase.IMPORTING]} importing")

    return LiveModel(
        left_text=left,
        right_text=" · ".join(meta),
        overall_fraction=snapshot.overall_fraction(),
        rows=rows,
        overflow=overflow,
    )


# Every non-terminal Phase, drift-pinned (test_output_wait_render ties it to set(Phase)).
# _overflow_text and wait_pulse_line hand-list the same trio.
_PHASE_RANK = {Phase.IMPORTING: 0, Phase.DOWNLOADING: 1, Phase.QUEUED: 2}


def _row_sort_key(torrent: TorrentView) -> tuple[int, float]:
    """Order key: importing first, downloads by soonest ETA, queued last."""

    rank = _PHASE_RANK.get(torrent.phase, 3)
    eta = float(torrent.eta_s) if torrent.eta_s is not None else float("inf")
    return rank, eta


def _row_model(torrent: TorrentView, *, spark: bool) -> RowModel:
    """Format one in-flight torrent's cells for the cockpit."""

    if torrent.phase is Phase.DOWNLOADING:
        rate = "stalled" if torrent.speed_bps is None else f"{human_bytes(torrent.speed_bps)}/s"
        if spark and len(torrent.speed_history) >= 2:
            rate = f"{sparkline(torrent.speed_history)} {rate}"
        return RowModel(
            label=torrent.label,
            phase=torrent.phase,
            fraction=clamp01(torrent.fraction),
            count=f"{round(clamp01(torrent.fraction) * 100)}%",
            speed=rate,
            time="" if torrent.eta_s is None else _compact_eta(torrent.eta_s),
            size="" if torrent.bytes_total is None else human_bytes(torrent.bytes_total),
            show_bar=True,
        )
    if torrent.phase is Phase.IMPORTING:
        elapsed = format_elapsed(torrent.phase_elapsed_s)
        if torrent.import_total:
            # An import has no download rate, hence no speed cell.
            return RowModel(
                label=torrent.label,
                phase=torrent.phase,
                fraction=clamp01(torrent.fraction),
                count=f"{torrent.import_done}/{torrent.import_total}",
                time=elapsed,
                show_bar=True,
            )
        # Indeterminate, so no bar: the status word carries the phase, "copying" once the
        # import command's async copy is in flight and "importing" before.
        return RowModel(
            label=torrent.label,
            phase=torrent.phase,
            fraction=1.0,
            status="copying" if torrent.command_issued else "importing",
            time=elapsed,
        )
    return RowModel(label=torrent.label, phase=Phase.QUEUED, fraction=0.0, status="queued")


def _overflow_text(hidden: list[TorrentView]) -> str:
    """A "+ N more downloading · M queued" tally for the rows past the budget."""

    if not hidden:
        return ""
    counts: Counter[Phase] = Counter(t.phase for t in hidden)
    parts: list[str] = []
    if counts[Phase.IMPORTING]:
        parts.append(f"{counts[Phase.IMPORTING]} more importing")
    if counts[Phase.DOWNLOADING]:
        parts.append(f"{counts[Phase.DOWNLOADING]} more downloading")
    if counts[Phase.QUEUED]:
        parts.append(f"{counts[Phase.QUEUED]} queued")
    return "+ " + " · ".join(parts)


def _aggregate_speed(snapshot: WaitSnapshot) -> int:
    """Total download speed across the downloading rows (bytes/s)."""

    return sum(t.speed_bps for t in snapshot.torrents if t.phase is Phase.DOWNLOADING and t.speed_bps is not None)


def _aggregate_eta(snapshot: WaitSnapshot, agg_speed: int) -> int | None:
    """The remaining-bytes ETA over the shared pipe. A sub-second remainder reads as done (None)."""

    if agg_speed <= 0:
        return None
    remaining = 0
    for torrent in snapshot.torrents:
        if (
            torrent.phase is Phase.DOWNLOADING
            and torrent.bytes_total is not None
            and torrent.bytes_done is not None
            and torrent.bytes_total >= torrent.bytes_done
        ):
            remaining += torrent.bytes_total - torrent.bytes_done
    if remaining <= 0:
        return None
    return int(remaining / agg_speed) or None


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(samples: tuple[int, ...]) -> str:
    """The speed-history glyph run, scaled to the window's own peak."""

    if not samples:
        return ""
    peak = max(samples)
    top = len(_SPARK_CHARS) - 1
    if peak <= 0:
        return _SPARK_CHARS[0] * len(samples)
    return "".join(_SPARK_CHARS[round(sample / peak * top)] for sample in samples)


def _compact_eta(seconds: float) -> str:
    """A short `~` ETA, e.g. `"~2m"` / `"~1h05m"` / `"~40s"`."""

    return f"~{compact_duration(seconds)}"
