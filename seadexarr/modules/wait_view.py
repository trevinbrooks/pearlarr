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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import ClassVar, final, override

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .console_caps import (
    Capabilities,
    TerminalEnv,
    block_bar,
    console_of,
    detect_capabilities,
    make_live,
    spinner_name,
)
from .log import (
    INDENT,
    STATE_WIDTH,
    count_noun,
    format_elapsed,
    indent_string,
    log_section_rule,
    log_styled,
)
from .manual_import import Outcome, OutcomeCategory
from .output import Phase, ScopeKind, ScopeMark, TorrentView, WaitSnapshot
from .output.wait_lines import LiveModel, RowModel, graduation_tail, live_model


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


def graduations(seen: AbstractSet[str], snapshot: WaitSnapshot) -> list[TorrentView]:
    """The terminal torrents not yet printed - pure, deterministic.

    A torrent graduates exactly once: the view tracks the keys it has already
    logged and this returns the newly-terminal ones in snapshot order.
    """

    return [
        torrent
        for torrent in snapshot.torrents
        if torrent.phase is Phase.TERMINAL and torrent.outcome is not None and torrent.key not in seen
    ]


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
        return LiveWaitView(TerminalEnv(console, caps, logger, time_source))
    return LogWaitView(
        logger,
        caps,
        poll_s=poll_s,
        digest_interval=digest_interval,
    )


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
        # Marks the wait region open on the hub (B1): diagnostics fired mid-wait
        # place at the wait indent instead of column 0.
        self._scope = ScopeMark(ScopeKind.WAIT_REGION, "wait")

    @final
    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        try:
            self._last_elapsed = snapshot.elapsed_s
            self._scope.open()
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
        # After the guarded teardown, so a _teardown raise can't eat the close —
        # and never before stop(), so no col-0 mark under a visible cockpit.
        self._scope.close()

    def _emit_new_graduations(self, snapshot: WaitSnapshot) -> None:
        for torrent in graduations(self._seen, snapshot):
            outcome = torrent.outcome
            if outcome is None:  # pragma: no cover - graduations() guarantees this
                continue
            self._seen.add(torrent.key)
            self._tally[outcome.category] += 1
            glyph = outcome.glyph(use_unicode=self._caps.unicode)
            line = f"{glyph} {outcome.word.ljust(STATE_WIDTH)} {torrent.label}"
            tail = graduation_tail(outcome, torrent.import_total, torrent.phase_elapsed_s)
            if tail:
                line += f"  ({tail})"
            log_styled(
                self._logger,
                indent_string(line),
                outcome.style if self._caps.color else None,
            )

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
        parts.append(format_elapsed(self._last_elapsed))
        log_section_rule(self._logger, "-")
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


@dataclass(frozen=True, slots=True)
class _TableLayout:
    """The width-derived column plan for the cockpit table.

    A pure function of the terminal width, so it's computed once at view
    construction and read by both the column setup (``_body``) and the per-row
    cell builder (``_row_cells``) - the two sides can't disagree per frame.
    """

    bar_width: int
    show_speed: bool
    show_size: bool

    @classmethod
    def for_width(cls, width: int) -> "_TableLayout":
        return cls(
            bar_width=16 if width >= 90 else (10 if width >= 70 else 0),
            show_speed=width >= 64,
            show_size=width >= 100,
        )


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

    def __init__(self, env: TerminalEnv) -> None:
        super().__init__(env.logger, env.caps)
        self._console = env.console
        self._time_source = env.time_source
        self._layout = _TableLayout.for_width(env.caps.width)
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
            try:
                self._live.stop()
            except Exception:
                # Contained here so a stop() raise can't eat _log_summary too.
                self._logger.debug("wait live stop failed", exc_info=True)
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
        layout = self._layout

        table = Table.grid(padding=(0, 1, 0, 0), expand=True)
        table.add_column(justify="left", no_wrap=True)  # marker
        table.add_column(justify="left", no_wrap=True, ratio=1, overflow="ellipsis")  # label
        if layout.bar_width:
            table.add_column(justify="left", no_wrap=True)  # bar / status word
        table.add_column(justify="right", no_wrap=True)  # count (or degraded status)
        if layout.show_speed:
            table.add_column(justify="right", no_wrap=True)  # speed (+ sparkline)
            table.add_column(justify="right", no_wrap=True)  # time (ETA / import elapsed)
        if layout.show_size:
            table.add_column(justify="right", no_wrap=True)  # total size

        for row in model.rows:
            table.add_row(*self._row_cells(row))
        return table

    def _row_cells(self, row: RowModel) -> list[Text | Spinner]:
        layout = self._layout
        # One shared spinner animates every importing row in sync; the static glyph
        # is the fallback (no live region, or any other phase).
        marker: Text | Spinner = (
            self._spinner if row.phase is Phase.IMPORTING and self._spinner is not None else self._marker(row.phase)
        )
        cells: list[Text | Spinner] = [marker, Text(row.label)]
        if layout.bar_width:
            cells.append(self._bar_or_status(row, layout.bar_width))
            cells.append(Text(row.count))
        else:
            # No bar column on a narrow console: the status word degrades into
            # the count column so a barless row still says what it's doing.
            word = row.count or row.status
            cells.append(Text(word, style="" if row.count else self._status_style(row.phase)))
        if layout.show_speed:
            cells.append(Text(row.speed, style="grey50"))
            cells.append(Text(row.time, style="grey50"))
        if layout.show_size:
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
            # Must stay below WARNING: the bridge adopts WARNING+, and hub.emit off
            # this Console-lock-holding thread is an ABBA deadlock (revisit at PR4).
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
                f"Waiting on {count_noun(snapshot.total(), 'download')} to complete and import...",
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
                f"{format_elapsed(snapshot.elapsed_s)}",
            ),
        )

    @override
    def _teardown(self) -> None:
        return
