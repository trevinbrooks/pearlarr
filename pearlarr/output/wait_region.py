"""The rich console's wait cockpit region: one self-animating `rich.Live` frame plus durable ledger lines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import assert_never, final, override

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .events import Phase, TorrentGraduated, WaitFinished, WaitKind, WaitProgress, WaitSnapshot, WaitStarted
from .live_region import LiveRegion
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
from ..log import INDENT, indent_string


@dataclass(frozen=True, slots=True)
class _FrameAnchor:
    """The last pushed snapshot, with the monotonic instant of the push."""

    snapshot: WaitSnapshot
    pushed_at: float


@dataclass(frozen=True, slots=True)
class _TableLayout:
    """The width-derived column plan for the cockpit table."""

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
    """A self-recomputing renderable for `WaitRegion`'s `rich.Live`."""

    def __init__(self, get_group: Callable[[], Group]) -> None:
        self._get_group = get_group
        self._failure: Exception | None = None
        self._latched = False

    def take_failure(self) -> Exception | None:
        """One-shot read of the latched render failure (None once collected)."""

        failure, self._failure = self._failure, None
        return failure

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console, options
        try:
            group = self._get_group()
        except Exception as exc:
            # NEVER log here: hub.emit off this Console-lock-holding refresh thread is an ABBA deadlock.
            # Latch the first failure per Live session: rich retries at 12.5 ticks/s, so a bug would spam.
            if not self._latched:
                self._latched = True
                self._failure = exc
            return
        yield group


@final
class WaitRegion(LiveRegion):
    """The wait cockpit: the animated in-flight table plus its durable scrollback lines."""

    # The frame snapshot the refresh thread reads: caps and layout are set once at Live start. Seeded by
    # _reset_frame, so __init__ and the per-pass/per-cycle resets can never drift apart.
    _caps: Capabilities
    _layout: _TableLayout
    _anchor: _FrameAnchor | None
    _live_frame: _LiveFrame | None
    _kind: WaitKind

    def __init__(
        self,
        console_source: Callable[[], Console | None],
        caps_cache: CapsCache | None = None,
        *,
        level_source: Callable[[], int],
        time_source: Callable[[], float],
    ) -> None:
        super().__init__(console_source, caps_cache, level_source=level_source)
        self._time_source = time_source
        self._throttle = PulseThrottle()
        # _reset_frame's teardown probe reads this.
        self._live_frame = None
        self._reset_frame()

    def handle(self, event: WaitEvent) -> None:
        self._collect_frame_failure()
        console = self._console_source()
        if console is None:
            return
        caps = self._caps_cache.for_console(console)
        match event:
            case WaitStarted():
                # Back-to-back passes swap the frontier node in one step, so section_left never fires.
                # Do NOT start the Live here: it starts on the first live snapshot.
                self._reset_frame()
                self._kind = event.kind
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
                # Teardown before summary: stop the Live first so the tally prints to clean scrollback.
                self._stop_live()
                self._durable(console, wait_tally_lines(event))
            case _:
                assert_never(event)

    @override
    def _reset(self) -> None:
        self._reset_frame()
        self._throttle.reset()

    def _durable(self, console: Console, lines: list[LegacyLine]) -> None:
        # LOGGER-parity gating: at a configured WARNING the wait INFO lines vanish from the console as from the file.
        render_legacy_lines(console, lines, self._level_source())

    def _advance_frame(self, console: Console, caps: Capabilities, snapshot: WaitSnapshot) -> None:
        # Atomic swap: one assignment, so the refresh thread never reads a torn anchor.
        self._anchor = _FrameAnchor(snapshot, self._time_source())
        if self._live is None:
            # The refresh thread reads these and the anchor, never the caps cache or the console.
            self._caps = caps
            self._layout = _TableLayout.for_width(caps.width)
            self._spinner = Spinner(spinner_name(caps), style="yellow")
            self._live = make_live(console)
            self._live.start()
            # Updated once: from here the producer only swaps the anchor, and rich re-renders between polls.
            self._live_frame = _LiveFrame(self._current_group)
            self._live.update(self._live_frame, refresh=True)

    @override
    def _stop_live(self) -> None:
        # Teardown also flushes the latch: a failure from the session's last ticks must not die with the frame.
        super()._stop_live()
        self._collect_frame_failure()

    def _collect_frame_failure(self) -> None:
        """Report a latched refresh-thread render failure, from the MAIN thread."""

        frame = self._live_frame
        if frame is None:
            return
        failure = frame.take_failure()
        if failure is not None:
            self._report_contained("wait frame render failed", failure)

    def _reset_frame(self) -> None:
        """Drop any stale live slot + frame snapshot (per pass and per cycle)."""

        self._stop_live()
        # The frame's lifetime is the Live slot's: a dangling one keeps _collect_frame_failure polling stale state.
        self._live_frame = None
        self._anchor = None
        self._caps = detect_capabilities(None)
        self._layout = _TableLayout.for_width(self._caps.width)
        self._kind = WaitKind.MONITOR

    def _current_group(self) -> Group:
        """Build the frame for the CURRENT instant, ticking timers and spinner forward."""

        anchor = self._anchor
        if anchor is None:
            return Group()
        offset = max(0.0, self._time_source() - anchor.pushed_at)
        return self._frame(live_model(self._advance(anchor.snapshot, offset), self._caps, self._kind))

    @staticmethod
    def _advance(snapshot: WaitSnapshot, offset: float) -> WaitSnapshot:
        """The snapshot with its in-flight elapsed clocks rolled forward by `offset`.

        A terminal row's `phase_elapsed_s` stays frozen, and is the ledger line's wait clock.
        """

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
            # The grid starts at column 0, so pad it: the rows share the header/overflow/ledger left edge.
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
        # One shared spinner keeps every importing row in sync.
        # Read once: the main thread's _stop_live can clear the attribute mid-teardown.
        spinner = self._spinner
        marker: Text | Spinner = (
            spinner if row.phase is Phase.IMPORTING and spinner is not None else self._marker(row.phase)
        )
        cells: list[Text | Spinner] = [marker, Text(row.label)]
        if layout.bar_width:
            cells.append(self._bar_or_status(row, layout.bar_width))
            cells.append(Text(row.count))
        else:
            # No bar column on a narrow console: the status word degrades into the count column.
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
