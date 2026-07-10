"""The rich console's wait cockpit region, event-driven (PR5).

The machinery that was ``wait_view.LiveWaitView`` now lives behind the hub:
:class:`WaitRegion` is driven by :class:`~.rich_renderer.RichRenderer`'s
exhaustive match and owns the single self-animating ``rich.Live`` cockpit
(:class:`_FrameAnchor`/:class:`_LiveFrame`/:class:`_TableLayout` + the
anchor-advance timer trick), the graduation of finished torrents to durable
scrollback lines, and the closing tally. On a live-capable console
``WaitProgress`` feeds the cockpit; a non-live console degrades the way
``LogWaitView`` did - a start line + throttled aggregate pulses, no Live. Under
plain/json there is no rich console and every event no-ops (the
:class:`~.legacy_echo.LegacyRenderer` echo carries those surfaces).

The durable lines here come from the shared :mod:`.wait_lines` builders and
render through :func:`~.scan_lines.render_legacy_lines`, so the console's
scrollback and the file/plain ledger can never drift. The single Live slot is
torn down by :meth:`section_left` when the wait region leaves the renderer's
fold frontier (whatever event evicted it), so later output never lands under a
stale region.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import assert_never, final

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .events import Phase, TorrentGraduated, WaitFinished, WaitProgress, WaitSnapshot, WaitStarted
from .scan_lines import LegacyLine, render_legacy_lines
from .wait_lines import (
    LiveModel,
    PulseThrottle,
    RowModel,
    WaitEvent,
    live_model,
    wait_graduation_line,
    wait_pulse_line,
    wait_start_line,
    wait_tally_lines,
)
from ..console_caps import (
    Capabilities,
    CapsCache,
    block_bar,
    detect_capabilities,
    make_live,
    spinner_name,
)
from ..log import INDENT, LOG_NAME, indent_string


@dataclass(frozen=True, slots=True)
class _FrameAnchor:
    """The last snapshot the producer pushed + the monotonic instant it was pushed.

    Swapped atomically by :meth:`WaitRegion._advance_frame` and read by the
    refresh thread, so a render pairs the latest snapshot with the time to roll
    it forward.
    """

    snapshot: WaitSnapshot
    pushed_at: float


@dataclass(frozen=True, slots=True)
class _TableLayout:
    """The width-derived column plan for the cockpit table.

    A pure function of the terminal width, so it's computed once at Live start
    and read by both the column setup (``_body``) and the per-row cell builder
    (``_row_cells``) - the two sides can't disagree per frame.
    """

    bar_width: int
    show_speed: bool
    show_size: bool

    @classmethod
    def for_width(cls, width: int) -> _TableLayout:
        return cls(
            bar_width=16 if width >= 90 else (10 if width >= 70 else 0),
            show_speed=width >= 64,
            show_size=width >= 100,
        )


@final
class _LiveFrame:
    """A self-recomputing renderable for :class:`WaitRegion`'s ``rich.Live``.

    rich re-renders this on its background refresh thread, so it rebuilds the
    frame from the region's current anchor each tick - ticking timers and
    animating the spinner between the producer's polls. Total by contract: a
    render bug degrades to an empty frame logged at debug, it never crashes the
    refresh thread.
    """

    def __init__(self, get_group: Callable[[], Group], logger: logging.Logger) -> None:
        self._get_group = get_group
        self._logger = logger
        self._logged = False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console, options
        try:
            group = self._get_group()
        except Exception:
            # Must stay below WARNING: the bridge adopts WARNING+, and hub.emit off
            # this Console-lock-holding thread is an ABBA deadlock (see pr5_plan P4).
            # Once per Live session (a fresh frame per start): rich retries at
            # 12.5 ticks/s, so a persistent bug would traceback-spam a DEBUG run.
            if not self._logged:
                self._logged = True
                self._logger.debug("wait frame render failed", exc_info=True)
            return
        yield group


@final
class WaitRegion:
    """One live slot + durable prints over the shared Console (PR5).

    Durable lines (start, pulses, graduations, tally) print the moment their
    event arrives - they reflow ABOVE the transient cockpit via the shared
    Console lock, exactly like the PR2 diagnostics. The Live is torn down by
    :meth:`section_left` when the renderer's fold evicts the wait-region node
    (whatever event evicted it), by a new pass's ``WaitStarted`` reset - the
    load-bearing route when back-to-back passes replace the frontier node in
    one fold step - and defensively by ``begin_cycle``/``close``.
    """

    def __init__(
        self,
        console_source: Callable[[], Console | None],
        caps_cache: CapsCache | None = None,
        *,
        level_source: Callable[[], int],
        time_source: Callable[[], float],
    ) -> None:
        self._console_source = console_source
        # Production wiring (cli) shares ONE cache with the LegacyRenderer echo so
        # both surfaces branch on the same probe; None builds a private cache.
        self._caps_cache = caps_cache if caps_cache is not None else CapsCache()
        # The RichRenderer's level store, read live (no duplicate _level here).
        self._level_source = level_source
        self._time_source = time_source
        # The non-TTY digest cadence; the echo seat runs its own copy in lockstep.
        self._throttle = PulseThrottle()
        # For the _LiveFrame debug swallow; the refresh thread logs nothing above it.
        self._logger = logging.getLogger(LOG_NAME)
        # The frame snapshot the refresh thread reads: caps/layout are set once at
        # Live start (a null-probe placeholder until then; the anchor guard means
        # a frame never builds before the first live progress).
        self._caps = detect_capabilities(None)
        self._layout = _TableLayout.for_width(self._caps.width)
        self._spinner: Spinner | None = None
        self._live: Live | None = None
        self._anchor: _FrameAnchor | None = None

    def handle(self, event: WaitEvent) -> None:
        console = self._console_source()
        if console is None:
            return
        caps = self._caps_cache.for_console(console)
        match event:
            case WaitStarted():
                # Do NOT start the Live here (the old view started it on the first
                # snapshot); just arm the digest cadence and print the start line.
                self._reset_frame()
                self._throttle.arm(event.pulse_s)
                if not caps.live:
                    self._durable(console, [wait_start_line(event)])
            case WaitProgress(snapshot=snapshot):
                if caps.live:
                    self._advance_frame(console, caps, snapshot)
                elif self._throttle.fire(snapshot.elapsed_s):
                    self._durable(console, [wait_pulse_line(snapshot)])
            case TorrentGraduated():
                self._durable(console, [wait_graduation_line(event, caps)])
            case WaitFinished():
                # Old teardown-then-summary order: stop the Live FIRST, then the
                # tally prints to clean scrollback (empty list -> nothing).
                self._stop_live()
                self._durable(console, wait_tally_lines(event))
            case _:
                assert_never(event)

    def section_left(self) -> None:
        """The wait region left the renderer's frontier: tear the live slot down.

        A safe no-op when no Live ever started (the non-TTY digest path), so the
        generalized frontier-departure loop can call it unconditionally.
        """

        self._stop_live()

    def begin_cycle(self) -> None:
        self._reset_frame()
        self._caps_cache.reset()
        self._throttle.reset()

    def close(self) -> None:
        self._stop_live()

    def _durable(self, console: Console, lines: list[LegacyLine]) -> None:
        # LOGGER-parity gating: at a configured WARNING the wait INFO lines vanish
        # from the console exactly as from the file (the scan arm's mechanism).
        render_legacy_lines(console, lines, self._level_source())

    def _advance_frame(self, console: Console, caps: Capabilities, snapshot: WaitSnapshot) -> None:
        # Atomic swap: the refresh thread reads whichever anchor is current, never a
        # torn one (single attribute assignment under the GIL).
        self._anchor = _FrameAnchor(snapshot, self._time_source())
        if self._live is None:
            # Snapshot caps/layout/spinner once at Live start; the refresh thread
            # reads these + the anchor, never the caps cache or the console source.
            self._caps = caps
            self._layout = _TableLayout.for_width(caps.width)
            self._spinner = Spinner(spinner_name(caps), style="yellow")
            self._live = make_live(console)
            self._live.start()
            # One persistent self-recomputing renderable; the producer only swaps the
            # anchor from here on, rich's thread re-renders this between polls.
            self._live.update(_LiveFrame(self._current_group, self._logger), refresh=True)

    def _reset_frame(self) -> None:
        """Drop any stale live slot + frame snapshot (per pass and per cycle)."""

        self._stop_live()
        self._anchor = None
        self._caps = detect_capabilities(None)
        self._layout = _TableLayout.for_width(self._caps.width)

    def _stop_live(self) -> None:
        # Take-and-clear; the stop() raise is contained so a failed stop can't
        # eat the tally print that follows on WaitFinished.
        live, self._live, self._spinner = self._live, None, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                self._logger.debug("wait live stop failed", exc_info=True)

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
        # is the fallback (no live region, or any other phase). Read once: the main
        # thread's _stop_live clears the attribute mid-teardown.
        spinner = self._spinner
        marker: Text | Spinner = (
            spinner if row.phase is Phase.IMPORTING and spinner is not None else self._marker(row.phase)
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
