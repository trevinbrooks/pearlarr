"""Pure builders + reducers for the wait pass's rendering (PR5).

Two families, no styled look decided here. The bounded live-frame model
(:func:`live_model` and its row/aggregate helpers) and the graduation ledger's
coda (:func:`graduation_tail`) reduce the wait value types in :mod:`.events` to
a rich-free layout brain both cockpit seats share. The :class:`LegacyLine`
builders (:func:`wait_start_line` and friends) map each durable wait fact to the
exact log line the pre-hub views produced - pinned by the Band A goldens in
``tests/test_wait_parity.py`` - so the file/plain echo and the console's
durable lines can never drift. :class:`PulseThrottle` carries the non-TTY
digest cadence, shared by both echo seats so the arithmetic can't diverge.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from .events import Phase, TorrentGraduated, TorrentView, WaitFinished, WaitProgress, WaitSnapshot, WaitStarted, clamp01
from .scan_lines import LegacyLine
from ..log import (
    STATE_WIDTH,
    SectionRule,
    StyledLine,
    count_noun,
    format_elapsed,
    human_bytes,
    indent_string,
    rule_string,
)
from ..manual_import import Outcome

if TYPE_CHECKING:
    from ..console_caps import Capabilities

type WaitEvent = WaitStarted | WaitProgress | TorrentGraduated | WaitFinished
"""The event subset both wait render seats consume."""

# The live cockpit never grows past this many in-flight rows; the rest collapse
# into a one-line "+ N more ..." overflow, so a large carried-over backlog can't
# blow the region past the screen. Clamped against the real terminal height too.
MAX_LIVE_ROWS = 12
MIN_LIVE_ROWS = 4
# Rows reserved for the banner, header, overflow line and a little breathing room
# when clamping the body to the terminal height.
_RESERVED_ROWS = 8
# Speed samples a downloading row keeps for its sparkline (one per heavy poll,
# so the default 30s cadence holds the last ~4 minutes).
SPARK_SAMPLES = 8
# The sparkline needs unicode blocks and enough width not to crowd the label.
MIN_SPARK_WIDTH = 80


def graduation_tail(outcome: Outcome, files: int | None, waited_s: float) -> str:
    """The ledger line's parenthesized coda - pure, "" when there is nothing to say.

    Baked into the logged message (not a console-only ``tail`` extra), so the
    file log carries it too. An import states its scale (``files``) and how long
    the wait took; a left-pending outcome says it will be retried; a dropped
    failure says the record is gone - so no outcome word reads as a dead end.
    """

    if outcome is Outcome.IMPORTED:
        parts: list[str] = []
        if files:
            parts.append(count_noun(files, "file"))
        if waited_s >= 1.0:
            parts.append(format_elapsed(waited_s))
        return " · ".join(parts)
    if not outcome.dropped:
        return "retries next run"
    # MISSING: the torrent vanished from qBittorrent, so the record went with it.
    return "no longer tracked"


# --- the durable ledger-line builders (console scrollback == file/plain echo) --------
#
# Transliterated from wait_view's LogWaitView/_DurableWaitView; every wait line
# logs at INFO today (P7), failures included. Pinned by tests/test_wait_parity.py.


def wait_start_line(event: WaitStarted) -> LegacyLine:
    """The non-TTY digest's opening line (LogWaitView's first render)."""

    return LegacyLine(logging.INFO, f"Waiting on {count_noun(event.total, 'download')} to complete and import...")


def wait_pulse_line(snapshot: WaitSnapshot) -> LegacyLine:
    """One throttled "still waiting" aggregate pulse (LogWaitView's later renders)."""

    counts = snapshot.counts()
    message = indent_string(
        f"still waiting · {counts[Phase.DOWNLOADING]} downloading · "
        f"{counts[Phase.IMPORTING]} importing · {counts[Phase.QUEUED]} queued · "
        f"{format_elapsed(snapshot.elapsed_s)}",
    )
    return LegacyLine(logging.INFO, message)


def wait_graduation_line(event: TorrentGraduated, caps: Capabilities) -> LegacyLine:
    """A finished torrent's durable ledger line: glyph + word + label + coda."""

    glyph = event.outcome.glyph(use_unicode=caps.unicode)
    line = f"{glyph} {event.outcome.word.ljust(STATE_WIDTH)} {event.label}"
    tail = graduation_tail(event.outcome, event.files, event.waited_s)
    if tail:
        line += f"  ({tail})"
    return LegacyLine(logging.INFO, indent_string(line), StyledLine(style=event.outcome.style if caps.color else ""))


def wait_tally_lines(event: WaitFinished) -> list[LegacyLine]:
    """The closing wait summary (rule + tally); ``[]`` when nothing graduated."""

    if event.imported == 0 and event.deferred == 0 and event.failed == 0:
        return []
    parts = [f"{event.imported} imported"]
    if event.deferred:
        parts.append(f"{event.deferred} left")
    if event.failed:
        parts.append(f"{event.failed} failed")
    parts.append(format_elapsed(event.elapsed_s))
    return [
        LegacyLine(logging.INFO, rule_string("-", 80), SectionRule(char="-")),
        LegacyLine(logging.INFO, indent_string("wait complete · " + " · ".join(parts))),
    ]


@final
class PulseThrottle:
    """The non-TTY digest's pulse cadence - rich-free, deterministic, shared.

    Parity with LogWaitView's throttle: :meth:`arm` (on WaitStarted) sets the
    interval; the FIRST :meth:`fire` returns False unconditionally (the old
    view's first render printed the start line and returned, so the start
    snapshot never pulses), then a pulse is due once elapsed reaches the
    elapsed-anchored next mark. State advances regardless of log level, so both
    echo seats stay in lockstep.
    """

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
        """Advance the cadence; True when a pulse is due at ``elapsed_s``."""

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
    """One rendered in-flight row, as plain strings - the pure-render unit.

    :func:`live_model` formats every value here (no rich), so the row layout is
    unit-testable; the view turns these into styled cells. Every column keeps ONE
    meaning across all row kinds: ``count`` is progress ("61%" / "8/12" files),
    ``speed`` is the download rate (sparkline + rate, or "stalled"), ``time`` is
    the ETA for a download or the elapsed clock for an import, ``size`` is the
    total download size. A row without a bar shows its ``status`` word instead.
    """

    label: str
    phase: Phase
    fraction: float
    # The status word drawn in the bar column when there is no bar: "queued",
    # "importing", or "copying" (an accepted import command's copy in flight).
    status: str = ""
    count: str = ""
    speed: str = ""
    time: str = ""
    size: str = ""
    # Draw a determinate block bar for ``fraction`` (downloads always; an importing
    # row only when its files-inserted count is known). Else the status word.
    show_bar: bool = False


@dataclass(frozen=True, slots=True)
class LiveModel:
    """A bounded, ordered, rich-free description of the live cockpit frame."""

    left_text: str
    right_text: str
    overall_fraction: float
    rows: tuple[RowModel, ...]
    overflow: str = ""


def live_model(snapshot: WaitSnapshot, caps: Capabilities) -> LiveModel:
    """Reduce a snapshot to a bounded, ordered cockpit frame - pure, no rich.

    Orders in-flight rows ``importing`` first, then ``downloading`` by soonest
    ETA (unknown/stalled last), then ``queued``; caps the visible rows to a
    height budget and collapses the rest into an overflow tally. Terminal rows
    are excluded (they graduate to scrollback).
    """

    in_flight = [t for t in snapshot.torrents if t.phase is not Phase.TERMINAL]
    in_flight.sort(key=_row_sort_key)

    budget = max(MIN_LIVE_ROWS, min(MAX_LIVE_ROWS, caps.height - _RESERVED_ROWS))
    visible = in_flight[:budget]
    hidden = in_flight[budget:]

    spark = caps.unicode and caps.width >= MIN_SPARK_WIDTH
    rows = tuple(_row_model(t, spark=spark) for t in visible)
    overflow = _overflow_text(hidden)

    counts = snapshot.counts()
    left = f"waiting {snapshot.done()}/{snapshot.total()}"
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
            count=f"{round(torrent.fraction * 100)}%",
            speed=rate,
            time="" if torrent.eta_s is None else _compact_eta(torrent.eta_s),
            size="" if torrent.bytes_total is None else human_bytes(torrent.bytes_total),
            show_bar=True,
        )
    if torrent.phase is Phase.IMPORTING:
        elapsed = format_elapsed(torrent.phase_elapsed_s)
        if torrent.import_total:
            # Determinate "files inserted" bar (the speed column stays blank -
            # an import has no download rate).
            return RowModel(
                label=torrent.label,
                phase=torrent.phase,
                fraction=clamp01(torrent.fraction),
                count=f"{torrent.import_done}/{torrent.import_total}",
                time=elapsed,
                show_bar=True,
            )
        # Indeterminate: no bar, so the status word carries the phase - "copying"
        # once an import command's async copy is in flight, "importing" before.
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
    """An honest "downloads done" ETA: remaining bytes over the shared pipe."""

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
    return int(remaining / agg_speed)


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(samples: tuple[int, ...]) -> str:
    """The speed-history glyph run, scaled to the window's own peak.

    A wedged download reads as a decay to the floor ("▆▄▁▁"); a slow-but-moving
    one keeps a steady band. All-zero history stays on the floor glyph (never
    blank), so a stall is visible rather than invisible.
    """

    if not samples:
        return ""
    peak = max(samples)
    top = len(_SPARK_CHARS) - 1
    if peak <= 0:
        return _SPARK_CHARS[0] * len(samples)
    return "".join(_SPARK_CHARS[round(sample / peak * top)] for sample in samples)


def _compact_eta(seconds: float) -> str:
    """A short ``~`` ETA, e.g. ``"~2m"`` / ``"~1h05m"`` / ``"~40s"``."""

    total = int(seconds)
    if total >= 3600:
        hours, minutes = divmod(total // 60, 60)
        return f"~{hours}h{minutes:02d}m"
    if total >= 60:
        return f"~{total // 60}m"
    return f"~{total}s"
