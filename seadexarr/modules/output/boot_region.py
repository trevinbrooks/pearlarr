"""The rich console's boot cockpit region, event-driven (PR3).

The machinery that was ``boot_view.LiveBootView``/``LogBootView`` now lives
behind the hub: :class:`BootRegion` is driven by :class:`~.rich_renderer.RichRenderer`'s
exhaustive match and owns the banner, the single live spinner + download bar,
the graduation of finished steps to durable scrollback lines, and the capstone.
On a live-capable console the spinner shows liveness (``BootStepSlow`` is
ignored); a non-live rich console degrades the way ``LogBootView`` did — no
Live, a one-time heads-up line per slow step. Under plain/json there is no rich
console and every event no-ops (the :class:`~.legacy_echo.LegacyRenderer` echo
carries those surfaces).

The pure line builders here are shared with the echo, so the file/plain ledger
lines and the console's durable lines can never drift.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import assert_never, final

from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from .events import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    RunStarted,
    ScopeClosed,
    ScopeKind,
)
from ..console_caps import Capabilities, CapsCache, block_bar, make_live, spinner_name
from ..log import INDENT, format_elapsed, indent_string, print_titled_rule

# Width of the live download bar (mapping refresh); only the live cockpit draws
# it, so it doesn't need to scale with the terminal like the wait cockpit's does.
_BAR_WIDTH = 16

type BootEvent = (
    RunStarted | BootStepStarted | BootStepProgressed | BootStepSlow | BootStepFinished | BootReady | ScopeClosed
)
"""The event subset the RichRenderer delegates to the boot region."""


def format_step_secs(seconds: float) -> str:
    """A compact step duration: ``"0.02s"`` / ``"1.3s"`` / ``"1m 04s"``."""

    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return format_elapsed(seconds)


# --- the shared ledger-line grammar (console scrollback == file/plain echo) --------


def banner_title(version: str) -> str:
    """The banner/title text (version may be "")."""

    return f"SeaDexArr {version}".rstrip()


def data_dir_line(data_dir: str) -> str:
    """The indented data-directory line under the banner."""

    return indent_string(f"Data directory: {data_dir}")


def _ellipsis(caps: Capabilities) -> str:
    """The working-dots suffix, degraded per console caps."""

    return "…" if caps.unicode else "..."


def slow_line(label: str, caps: Capabilities) -> str:
    """The one-time slow-step heads-up, ellipsis degraded per console caps."""

    return indent_string(f"{label}{_ellipsis(caps)}")


def graduation_line(event: BootStepFinished, caps: Capabilities) -> str:
    """A finished step's durable ledger line: glyph + label · detail · elapsed."""

    glyph = event.outcome.glyph_for(use_unicode=caps.unicode)
    text = event.label
    if event.detail:
        text += f" · {event.detail}"
    text += f" · {format_step_secs(event.elapsed_s)}"
    return indent_string(f"{glyph} {text}")


def ready_line(elapsed_s: float) -> str:
    """The boot capstone line."""

    return indent_string(f"ready in {format_step_secs(elapsed_s)}")


@final
class BootRegion:
    """One live slot + durable prints over the shared Console (PR3).

    Durable lines (banner, graduations, heads-up, capstone) print the moment
    their event arrives — they reflow ABOVE the transient spinner via the shared
    Console lock, exactly like the PR2 diagnostics. The spinner is torn down on
    the boot section's ``ScopeClosed`` (and defensively by ``begin_cycle``/
    ``close``), so scan output never lands under a stale live region.
    """

    def __init__(self, console_source: Callable[[], Console | None], caps_cache: CapsCache | None = None) -> None:
        self._console_source = console_source
        # Production wiring (cli) shares ONE cache with the LegacyRenderer echo so
        # both surfaces branch on the same probe; None builds a private cache.
        self._caps_cache = caps_cache if caps_cache is not None else CapsCache()
        self._level = logging.INFO
        self._live: Live | None = None
        self._spinner: Spinner | None = None
        # The only cross-event frame state: Progressed events carry no label.
        self._label = ""

    def handle(self, event: BootEvent) -> None:
        if isinstance(event, ScopeClosed):
            if event.scope.kind is ScopeKind.BOOT_SECTION:
                self._stop_live()
            return
        console = self._console_source()
        if console is None:
            return
        caps = self._caps_cache.for_console(console)
        match event:
            case RunStarted():
                self._banner(console, event)
            case BootStepStarted(label=label):
                self._step_started(console, caps, label)
            case BootStepProgressed(fraction=fraction, detail=detail):
                if self._spinner is not None:
                    self._spinner.update(text=self._frame_text(caps, fraction, detail))
            case BootStepSlow(label=label):
                # Live consoles show liveness via the spinner; the heads-up is
                # the non-live rich console's LogBootView-style degradation.
                if not caps.live and self._admits_durable():
                    self._print(console, Text(slow_line(label, caps)))
            case BootStepFinished():
                if self._admits_durable():
                    style = event.outcome.style if caps.color else ""
                    self._print(console, Text(graduation_line(event, caps), style=style))
            case BootReady(elapsed_s=elapsed_s):
                # Old-view order: the spinner is torn down BEFORE the capstone
                # prints (ScopeClosed stays the teardown for capstone-less ends).
                self._stop_live()
                if self._admits_durable():
                    self._print(console, Text(ready_line(elapsed_s), style="grey50" if caps.color else ""))
            case _:
                assert_never(event)

    def set_level(self, level: int) -> None:
        self._level = level

    def begin_cycle(self) -> None:
        self._stop_live()
        self._caps_cache.reset()
        self._label = ""

    def close(self) -> None:
        self._stop_live()

    def _admits_durable(self) -> bool:
        """Level parity with the logger-driven ledger: INFO lines need level <= INFO."""

        return self._level <= logging.INFO

    def _banner(self, console: Console, event: RunStarted) -> None:
        # Parity: the same console look the pre-PR3 TitledRule/StyledLine payloads produced.
        if not self._admits_durable():
            return
        print_titled_rule(console, banner_title(event.version), "bold cyan", heavy=True)
        console.print(Text(""))
        self._print(console, Text(data_dir_line(event.data_dir), style="grey50"))

    def _step_started(self, console: Console, caps: Capabilities, label: str) -> None:
        self._label = label
        if not caps.live:
            return
        if self._live is None:
            self._live = make_live(console)
            self._live.start()
        # A fresh Spinner per step restarts the animation at frame 0 (intended);
        # within a step only .text mutates so the dots keep their phase. Padding
        # holds the spinner by reference, so progress updates re-render live.
        self._spinner = Spinner(spinner_name(caps), text=self._frame_text(caps, None, None), style="cyan")
        self._live.update(Padding(self._spinner, (0, 0, 0, len(INDENT))))

    def _frame_text(self, caps: Capabilities, fraction: float | None, detail: str | None) -> Text:
        line = Text(self._label, style="bold")
        if fraction is not None:
            line.append("  ")
            line.append(block_bar(fraction, _BAR_WIDTH, caps))
            line.append(f" {round(fraction * 100)}%", style="cyan")
            if detail:
                line.append("  ")
                line.append(detail, style="grey50")
        elif detail:
            line.append("  ")
            line.append(detail, style="grey50")
        else:
            line.append(_ellipsis(caps), style="bold")
        return line

    def _stop_live(self) -> None:
        # Take-and-clear; Live.stop() itself no-ops once stopped (its _started guard).
        live, self._live, self._spinner = self._live, None, None
        if live is not None:
            live.stop()

    @staticmethod
    def _print(console: Console, text: Text) -> None:
        # Literal text (no markup/highlight): "[1/182]" stays text.
        console.print(text, highlight=False, soft_wrap=True)
