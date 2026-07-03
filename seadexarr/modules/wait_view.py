"""Presentation for the wait-for-completion + manual-import blocking pass.

The engine drives a :class:`WaitView` while it waits on each grabbed torrent to
download and then import. The view is a pure function of an immutable
:class:`WaitSnapshot` the engine pushes once per poll cycle - so it is testable
without a terminal, and every method is total (a render bug degrades to a no-op,
never aborting the wait loop or the end-of-run cache save).

On an attached terminal :class:`LiveWaitView` renders a sticky "cockpit": an
aggregate header over the in-flight torrents (a block bar + speed + ETA for
downloads, an ``importing`` line for imports). When a torrent reaches a terminal
state it GRADUATES - the view logs a permanent, color/glyph-coded ledger line
(through the logger, so it lands in scrollback AND the plain-text file log) and
drops out of the bounded live region. On a non-TTY (Docker / a pipe / a dumb
terminal) :class:`LogWaitView` degrades to a calm aggregate digest plus the same
durable graduation lines, so container logs stay clean. :func:`make_wait_view`
probes the console once and picks the right one, so the engine drives a single
small interface either way.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import ClassVar, final, override

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .console_caps import (
    Capabilities,
    block_bar,
    console_of,
    detect_capabilities,
    make_live,
    spinner_name,
)
from .log import (
    INDENT,
    STATE_WIDTH,
    LogFormatter,
    count_noun,
    indent_string,
    rule_string,
)
from .manual_import import Outcome, OutcomeCategory

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


class Phase(Enum):
    """The lifecycle phase of one torrent in the wait pass.

    ``QUEUED`` -> still downloading (or not yet polled). ``DOWNLOADING`` ->
    downloading with live telemetry. ``IMPORTING`` -> the download finished and an
    import is in flight (indeterminate). ``TERMINAL`` -> a terminal outcome was
    reached (carries the :class:`~.manual_import.Outcome`); these GRADUATE to
    scrollback and leave the live region.
    """

    QUEUED = auto()
    DOWNLOADING = auto()
    IMPORTING = auto()
    TERMINAL = auto()


@dataclass(frozen=True, slots=True)
class TorrentView:
    """One torrent's state for a single frame - the engine's per-poll snapshot row.

    Immutable so a snapshot is a value: the engine rebuilds the row each cycle
    (``dataclasses.replace`` off the prior one) and the view renders it. Telemetry
    fields are already sanitized (see :class:`~.manual_import.TorrentProbe`);
    ``outcome`` is non-None iff ``phase`` is ``TERMINAL``.
    """

    key: str
    label: str
    phase: Phase = Phase.QUEUED
    fraction: float = 0.0
    speed_bps: int | None = None
    eta_s: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    phase_elapsed_s: float = 0.0
    phase_timeout_s: float = 0.0
    command_issued: bool = False
    # "Files inserted" bar for an IMPORTING row: both set -> a determinate
    # done/total bar; both None -> indeterminate (just the "importing" word).
    # On a TERMINAL imported row they carry the final files count for the ledger,
    # and phase_elapsed_s freezes as the ledger's wait clock (LiveWaitView's
    # between-poll ticking skips TERMINAL rows, so it can't drift).
    import_done: int | None = None
    import_total: int | None = None
    # Speed samples (bytes/s, stalled -> 0), one per heavy poll, newest last -
    # the sparkline showing slow-but-moving vs wedged. Bounded to SPARK_SAMPLES.
    speed_history: tuple[int, ...] = ()
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class WaitSnapshot:
    """An immutable description of the whole wait pass at one poll cycle.

    The single value the engine pushes to :meth:`WaitView.update`; the view is a
    pure function of it. Derived aggregates are computed here so they can be
    unit-tested without any rendering.
    """

    torrents: tuple[TorrentView, ...]
    elapsed_s: float = 0.0

    def counts(self) -> dict[Phase, int]:
        """Count of torrents in each phase (every phase present, 0 by default)."""

        tally: dict[Phase, int] = dict.fromkeys(Phase, 0)
        for torrent in self.torrents:
            tally[torrent.phase] += 1
        return tally

    def done(self) -> int:
        """How many torrents have reached a terminal outcome."""

        return sum(1 for t in self.torrents if t.phase is Phase.TERMINAL)

    def total(self) -> int:
        """How many torrents the pass is (or was) waiting on."""

        return len(self.torrents)

    def overall_fraction(self) -> float:
        """An aggregate 0-1 progress for the header bar (download-completion based).

        Terminal and importing rows count as a finished download (1.0); a still
        downloading/queued row contributes its download fraction. Guards /0.
        """

        if not self.torrents:
            return 0.0
        total = 0.0
        for torrent in self.torrents:
            if torrent.phase in (Phase.TERMINAL, Phase.IMPORTING):
                total += 1.0
            else:
                total += max(0.0, min(1.0, torrent.fraction))
        return total / len(self.torrents)


@dataclass(frozen=True, slots=True)
class WaitOutcomeRow:
    """One torrent's terminal result, captured by the monitor for the run report."""

    label: str
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class WaitResult:
    """The outcome of a whole wait pass - what the report + notification render.

    Returned by :meth:`ImportWaitManager.run_monitor` so the end-of-run tail can
    write a durable report and push a completion notification without re-deriving
    state.
    """

    rows: tuple[WaitOutcomeRow, ...]
    elapsed_s: float

    @property
    def waited(self) -> int:
        """How many torrents the pass reached a terminal outcome for."""

        return len(self.rows)

    @property
    def imported(self) -> int:
        """Count of imported (SUCCESS) torrents."""

        return self._count(OutcomeCategory.SUCCESS)

    @property
    def left(self) -> int:
        """Count of deferred ("left for a later run") torrents."""

        return self._count(OutcomeCategory.DEFERRED)

    @property
    def failed(self) -> int:
        """Count of failed torrents."""

        return self._count(OutcomeCategory.FAILED)

    def _count(self, category: OutcomeCategory) -> int:
        return sum(1 for row in self.rows if row.outcome.category is category)


def graduations(seen: frozenset[str], snapshot: WaitSnapshot) -> list[TorrentView]:
    """The terminal torrents not yet printed - pure, deterministic.

    A torrent graduates exactly once: the view tracks the keys it has already
    logged and this returns the newly-terminal ones in snapshot order.
    """

    return [
        torrent
        for torrent in snapshot.torrents
        if torrent.phase is Phase.TERMINAL and torrent.outcome is not None and torrent.key not in seen
    ]


def graduation_tail(torrent: TorrentView, outcome: Outcome) -> str:
    """The ledger line's parenthesized coda - pure, "" when there is nothing to say.

    Baked into the logged message (not a console-only ``tail`` extra), so the
    file log carries it too. An import states its scale and how long the wait
    took; a left-pending outcome says it will be retried; a dropped failure says
    the record is gone - so no outcome word reads as a dead end.
    """

    if outcome is Outcome.IMPORTED:
        parts: list[str] = []
        if torrent.import_total:
            parts.append(count_noun(torrent.import_total, "file"))
        if torrent.phase_elapsed_s >= 1.0:
            parts.append(LogFormatter.format_elapsed(torrent.phase_elapsed_s))
        return " · ".join(parts)
    if not outcome.dropped:
        return "retries next run"
    # MISSING: the torrent vanished from qBittorrent, so the record went with it.
    return "no longer tracked"


@dataclass(frozen=True, slots=True)
class RowModel:
    """One rendered in-flight row, as plain strings - the pure-render unit.

    ``live_model`` formats every value here (no rich), so the row layout is
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


def make_wait_view(
    logger: logging.Logger,
    *,
    poll_s: int,
    digest_interval: int = 300,
    time_source: Callable[[], float] = time.monotonic,
) -> "WaitView":
    """Build a live cockpit on a capable TTY, else a calm log digest.

    Args:
        logger (logging.Logger): The app logger; its rich console handler is reused
            so the live region and the log lines share one Console (a warning logged
            mid-wait reflows ABOVE the region).
        poll_s (int): The poll cadence - the floor for the non-TTY digest interval.
        digest_interval (int): Target seconds between non-TTY aggregate pulses.
        time_source (Callable[[], float]): Monotonic clock the live cockpit uses to
            tick its spinner/timers between polls; injectable so a test (and the
            monitor) can share one deterministic clock.
    """

    console = console_of(logger)
    caps = detect_capabilities(console)
    if console is not None and caps.live:
        return LiveWaitView(console, caps, logger, time_source=time_source)
    return LogWaitView(
        logger,
        caps,
        poll_s=poll_s,
        digest_interval=digest_interval,
    )


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
    meta: list[str] = [LogFormatter.format_elapsed(snapshot.elapsed_s)]
    agg_speed = _aggregate_speed(snapshot)
    if agg_speed:
        meta.append(f"{arrow} {_human_bytes(agg_speed)}/s")
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
        rate = "stalled" if torrent.speed_bps is None else f"{_human_bytes(torrent.speed_bps)}/s"
        if spark and len(torrent.speed_history) >= 2:
            rate = f"{sparkline(torrent.speed_history)} {rate}"
        return RowModel(
            label=torrent.label,
            phase=torrent.phase,
            fraction=max(0.0, min(1.0, torrent.fraction)),
            count=f"{round(torrent.fraction * 100)}%",
            speed=rate,
            time="" if torrent.eta_s is None else _compact_eta(torrent.eta_s),
            size="" if torrent.bytes_total is None else _human_bytes(torrent.bytes_total),
            show_bar=True,
        )
    if torrent.phase is Phase.IMPORTING:
        elapsed = LogFormatter.format_elapsed(torrent.phase_elapsed_s)
        if torrent.import_total:
            # Determinate "files inserted" bar (the speed column stays blank -
            # an import has no download rate).
            return RowModel(
                label=torrent.label,
                phase=torrent.phase,
                fraction=max(0.0, min(1.0, torrent.fraction)),
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


def _human_bytes(num: float) -> str:
    """A compact human byte size, e.g. ``"3.2 MB"`` / ``"1.8 GB"``."""

    val = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _compact_eta(seconds: float) -> str:
    """A short ``~`` ETA, e.g. ``"~2m"`` / ``"~1h05m"`` / ``"~40s"``."""

    total = int(seconds)
    if total >= 3600:
        hours, minutes = divmod(total // 60, 60)
        return f"~{hours}h{minutes:02d}m"
    if total >= 60:
        return f"~{total // 60}m"
    return f"~{total}s"


class WaitView(ABC):
    """The small interface the engine drives while waiting on downloads/imports.

    The engine pushes a full :class:`WaitSnapshot` each poll cycle; the view
    renders it. Both methods MUST be total (never raise) so a presentation bug
    can't abort the wait loop or the end-of-run cache save.
    """

    # Whether the view renders per-row download telemetry between heavy polls.
    # The engine skips the fast-lane qBittorrent read when it can't be seen
    # (the non-TTY digest shows only phase counts, so the read would be waste).
    wants_telemetry: ClassVar[bool] = True

    @abstractmethod
    def update(self, snapshot: WaitSnapshot) -> None:
        """Render the latest snapshot (graduating any newly-terminal torrents)."""

    @abstractmethod
    def close(self) -> None:
        """Tear the view down and log the closing summary (idempotent)."""


class _DurableWaitView(WaitView):
    """Shared spine: graduate terminal torrents to the log, tally, summarize.

    Both concrete views graduate a finished torrent the same way - a single
    ``logger`` call, so the durable ledger line hits the styled console (reflowed
    ABOVE any live region) AND the plain-text file log. Subclasses add only the
    live frame (:meth:`_render`) and its teardown (:meth:`_teardown`).
    """

    def __init__(self, logger: logging.Logger, caps: Capabilities) -> None:
        self._logger = logger
        self._caps = caps
        self._seen: set[str] = set()
        self._tally: Counter[OutcomeCategory] = Counter()
        self._last_elapsed = 0.0
        self._closed = False

    @final
    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        try:
            self._last_elapsed = snapshot.elapsed_s
            self._emit_new_graduations(snapshot)
            self._render(snapshot)
        except Exception:
            # Total by contract: a render bug degrades to a no-op, it never
            # aborts the engine's wait loop or the end-of-run cache save.
            self._logger.debug("wait view update failed", exc_info=True)

    @final
    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._teardown()
            self._log_summary()
        except Exception:
            self._logger.debug("wait view close failed", exc_info=True)

    def _emit_new_graduations(self, snapshot: WaitSnapshot) -> None:
        for torrent in graduations(frozenset(self._seen), snapshot):
            outcome = torrent.outcome
            if outcome is None:  # pragma: no cover - graduations() guarantees this
                continue
            self._seen.add(torrent.key)
            self._tally[outcome.category] += 1
            glyph = outcome.glyph(use_unicode=self._caps.unicode)
            line = f"{glyph} {outcome.word.ljust(STATE_WIDTH)} {torrent.label}"
            tail = graduation_tail(torrent, outcome)
            if tail:
                line += f"  ({tail})"
            extra = {"line_style": outcome.style} if self._caps.color else {}
            self._logger.info(indent_string(line), extra=extra)

    def _log_summary(self) -> None:
        imported = self._tally[OutcomeCategory.SUCCESS]
        deferred = self._tally[OutcomeCategory.DEFERRED]
        failed = self._tally[OutcomeCategory.FAILED]
        if imported == 0 and deferred == 0 and failed == 0:
            return
        parts = [f"{imported} imported"]
        if deferred:
            parts.append(f"{deferred} left")
        if failed:
            parts.append(f"{failed} failed")
        parts.append(LogFormatter.format_elapsed(self._last_elapsed))
        self._logger.info(rule_string("-"))
        self._logger.info(indent_string("wait complete · " + " · ".join(parts)))

    @abstractmethod
    def _render(self, snapshot: WaitSnapshot) -> None:
        """Draw the live frame for this snapshot (may be a no-op)."""

    @abstractmethod
    def _teardown(self) -> None:
        """Stop any live region / restore the terminal (idempotent)."""


@dataclass(frozen=True, slots=True)
class _FrameAnchor:
    """The last snapshot the engine pushed + the monotonic instant it was pushed.

    Swapped atomically by :meth:`LiveWaitView._render` and read by the refresh
    thread, so a render pairs the latest snapshot with the time to roll it forward.
    """

    snapshot: WaitSnapshot
    pushed_at: float


@final
class LiveWaitView(_DurableWaitView):
    """The sticky terminal cockpit: a self-animating ``rich.Live`` region.

    ``auto_refresh=True`` (like the boot cockpit): rich re-renders on its own
    refresh thread, so the per-row + header elapsed timers tick and the importing
    spinner animates BETWEEN the engine's polls, not only when a snapshot is pushed.
    The engine's :meth:`update` just swaps the immutable :class:`_FrameAnchor`; a
    persistent :class:`_LiveFrame` rebuilds the frame from it each tick, rolling the
    elapsed clocks forward by the time since the push. The shared Console lock
    serializes that thread against the logging handler, so a line logged mid-wait
    reflows ABOVE the region. The region holds only in-flight rows under an
    aggregate header (finished torrents graduated to scrollback); ``transient=True``
    erases the box on close, leaving the durable ledger + summary.
    """

    def __init__(
        self,
        console: Console,
        caps: Capabilities,
        logger: logging.Logger,
        *,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(logger, caps)
        self._console = console
        self._time_source = time_source
        self._live: Live | None = None
        self._spinner: Spinner | None = None
        self._anchor: _FrameAnchor | None = None

    @override
    def _render(self, snapshot: WaitSnapshot) -> None:
        # Atomic swap: the refresh thread reads whichever anchor is current, never a
        # torn one (single attribute assignment under the GIL).
        self._anchor = _FrameAnchor(snapshot, self._time_source())
        if self._live is None:
            self._spinner = Spinner(spinner_name(self._caps), style="yellow")
            self._live = make_live(self._console)
            self._live.start()
            # One persistent self-recomputing renderable; the engine only swaps the
            # anchor from here on, rich's thread re-renders this between polls.
            self._live.update(_LiveFrame(self._current_group, self._logger), refresh=True)

    @override
    def _teardown(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._spinner = None

    def _current_group(self) -> Group:
        """Build the frame for the CURRENT instant - ticks timers + spinner forward.

        Called on each of rich's refresh ticks (via :class:`_LiveFrame`). Rolls the
        last pushed snapshot's elapsed clocks forward by the time since it was
        pushed, so the timers advance between polls; the pure :func:`live_model`
        reducer still sees explicit elapsed values (no clock of its own).
        """

        anchor = self._anchor
        if anchor is None:
            return Group()
        offset = max(0.0, self._time_source() - anchor.pushed_at)
        return self._frame(live_model(self._advance(anchor.snapshot, offset), self._caps))

    @staticmethod
    def _advance(snapshot: WaitSnapshot, offset: float) -> WaitSnapshot:
        """The snapshot with its in-flight elapsed clocks rolled forward by ``offset``."""

        if offset <= 0.0:
            return snapshot
        torrents = tuple(
            torrent
            if torrent.phase is Phase.TERMINAL
            else replace(torrent, phase_elapsed_s=torrent.phase_elapsed_s + offset)
            for torrent in snapshot.torrents
        )
        return replace(snapshot, torrents=torrents, elapsed_s=snapshot.elapsed_s + offset)

    def _frame(self, model: LiveModel) -> Group:
        parts: list[Text | Table | Padding] = [self._header(model)]
        body = self._body(model)
        if body is not None:
            # The rows share the header/overflow/ledger indent (the grid itself
            # starts at column 0, so pad it) - one left edge for the whole pass.
            parts.append(Padding(body, (0, 0, 0, len(INDENT))))
        if model.overflow:
            parts.append(self._truncate(Text(indent_string(model.overflow), style="grey50")))
        return Group(*parts)

    def _header(self, model: LiveModel) -> Text:
        line = Text(indent_string(""))
        line.append(model.left_text, style="bold")
        line.append("  ")
        line.append(block_bar(model.overall_fraction, 12, self._caps))
        if model.right_text:
            line.append("  ")
            line.append(model.right_text, style="cyan")
        return self._truncate(line)

    def _body(self, model: LiveModel) -> Table | None:
        if not model.rows:
            return None
        bar_width = 16 if self._caps.width >= 90 else (10 if self._caps.width >= 70 else 0)
        show_speed = self._caps.width >= 64
        show_size = self._caps.width >= 100

        table = Table.grid(padding=(0, 1, 0, 0), expand=True)
        table.add_column(justify="left", no_wrap=True)  # marker
        table.add_column(justify="left", no_wrap=True, ratio=1, overflow="ellipsis")  # label
        if bar_width:
            table.add_column(justify="left", no_wrap=True)  # bar / status word
        table.add_column(justify="right", no_wrap=True)  # count (or degraded status)
        if show_speed:
            table.add_column(justify="right", no_wrap=True)  # speed (+ sparkline)
            table.add_column(justify="right", no_wrap=True)  # time (ETA / import elapsed)
        if show_size:
            table.add_column(justify="right", no_wrap=True)  # total size

        for row in model.rows:
            table.add_row(*self._row_cells(row, bar_width, show_speed=show_speed, show_size=show_size))
        return table

    def _row_cells(
        self,
        row: RowModel,
        bar_width: int,
        *,
        show_speed: bool,
        show_size: bool,
    ) -> list[Text | Spinner]:
        # One shared spinner animates every importing row in sync; the static glyph
        # is the fallback (no live region, or any other phase).
        marker: Text | Spinner = (
            self._spinner if row.phase is Phase.IMPORTING and self._spinner is not None else self._marker(row.phase)
        )
        cells: list[Text | Spinner] = [marker, Text(row.label)]
        if bar_width:
            cells.append(self._bar_or_status(row, bar_width))
            cells.append(Text(row.count))
        else:
            # No bar column on a narrow console: the status word degrades into
            # the count column so a barless row still says what it's doing.
            word = row.count or row.status
            cells.append(Text(word, style="" if row.count else self._status_style(row.phase)))
        if show_speed:
            cells.append(Text(row.speed, style="grey50"))
            cells.append(Text(row.time, style="grey50"))
        if show_size:
            cells.append(Text(row.size, style="grey50"))
        return cells

    def _marker(self, phase: Phase) -> Text:
        if phase is Phase.DOWNLOADING:
            return Text("↓" if self._caps.unicode else "v", style="cyan")
        if phase is Phase.IMPORTING:
            return Text("∼" if self._caps.unicode else "~", style="yellow")
        return Text("·" if self._caps.unicode else ".", style="grey50")

    def _bar_or_status(self, row: RowModel, bar_width: int) -> Text:
        if row.show_bar:
            return block_bar(row.fraction, bar_width, self._caps)
        return Text(row.status.ljust(bar_width)[:bar_width], style=self._status_style(row.phase))

    @staticmethod
    def _status_style(phase: Phase) -> str:
        return "yellow" if phase is Phase.IMPORTING else "grey50"

    def _truncate(self, text: Text) -> Text:
        text.truncate(self._caps.width, overflow="ellipsis")
        return text


@final
class _LiveFrame:
    """A self-recomputing renderable for :class:`LiveWaitView`'s ``rich.Live``.

    rich re-renders this on its background refresh thread, so it rebuilds the frame
    from the view's current anchor each tick - ticking timers and animating the
    spinner between the engine's polls. Total by contract: a render bug degrades to
    an empty frame logged at debug, it never crashes the refresh thread.
    """

    def __init__(self, get_group: Callable[[], Group], logger: logging.Logger) -> None:
        self._get_group = get_group
        self._logger = logger

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console, options
        try:
            group = self._get_group()
        except Exception:
            self._logger.debug("wait frame render failed", exc_info=True)
            return
        yield group


@final
class LogWaitView(_DurableWaitView):
    """Calm aggregate digest for a non-TTY / dumb terminal (Docker, a pipe, CI).

    No live region: one start line, a wall-clock aggregate pulse throttled to the
    digest interval (driven off ``snapshot.elapsed_s`` so it's deterministic), and
    the shared durable graduation lines + closing summary. A large carried-over
    backlog therefore produces a handful of lines, not a per-torrent flood.
    """

    # The digest renders no per-row telemetry, so the fast lane needn't fetch it.
    wants_telemetry: ClassVar[bool] = False

    def __init__(
        self,
        logger: logging.Logger,
        caps: Capabilities,
        *,
        poll_s: int,
        digest_interval: int,
    ) -> None:
        super().__init__(logger, caps)
        self._interval = float(max(poll_s, digest_interval))
        self._started = False
        self._next_pulse = 0.0

    @override
    def _render(self, snapshot: WaitSnapshot) -> None:
        if not self._started:
            self._started = True
            self._next_pulse = self._interval
            self._logger.info(
                f"Waiting on {snapshot.total()} download(s) to complete and import...",
            )
            return
        if snapshot.elapsed_s < self._next_pulse:
            return
        self._next_pulse = snapshot.elapsed_s + self._interval
        counts = snapshot.counts()
        self._logger.info(
            indent_string(
                f"still waiting · {counts[Phase.DOWNLOADING]} downloading · "
                f"{counts[Phase.IMPORTING]} importing · {counts[Phase.QUEUED]} queued · "
                f"{LogFormatter.format_elapsed(snapshot.elapsed_s)}",
            ),
        )

    @override
    def _teardown(self) -> None:
        return
