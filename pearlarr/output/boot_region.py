"""The rich console's boot cockpit region, event-driven.

`BootRegion` is driven by `rich_renderer.RichRenderer`'s exhaustive match and
owns the banner, the single live spinner + download bar, the graduation of
finished steps to durable scrollback lines, and the capstone. On a
live-capable console the spinner shows liveness (`BootStepSlow` is ignored). A
non-live rich console degrades - no Live, a one-time heads-up line per slow
step. Under plain/json there is no rich console and every event no-ops (the
hub's text sinks carry those surfaces).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import assert_never, final, override

from rich.console import Console
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from .console_text import INDENT, indent_string, long_duration, print_titled_rule
from .events import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    PearlarrStarted,
)
from .live_region import LiveRegion
from ..console_features import ConsoleFeatures, DetectedFeatures, block_bar, make_live, spinner_name
from ..log import print_literal

# Width of the live download bar (mapping refresh). Only the live cockpit draws
# it, so it doesn't need to scale with the terminal like the wait cockpit's does.
_BAR_WIDTH = 16

type BootEvent = PearlarrStarted | BootStepStarted | BootStepProgressed | BootStepSlow | BootStepFinished | BootReady
"""The event subset the RichRenderer delegates to the boot region."""


def step_seconds(seconds: float) -> str:
    """A compact step duration: `"0.02s"` / `"1.3s"` / `"1m 04s"`."""

    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return long_duration(seconds)


# --- the shared ledger-line grammar (console scrollback == file/plain echo) --------


def banner_title(version: str) -> str:
    """The banner/title text (version may be "")."""

    return f"Pearlarr {version}".rstrip()


def data_dir_line(data_dir: str) -> str:
    """The indented data-directory line under the banner."""

    return indent_string(f"Data directory: {data_dir}")


def _ellipsis(features: ConsoleFeatures) -> str:
    """The working-dots suffix, degraded per console caps."""

    return "…" if features.can_unicode else "..."


def slow_line(label: str, features: ConsoleFeatures) -> str:
    """The one-time slow-step heads-up, ellipsis degraded per console caps."""

    return indent_string(f"{label}{_ellipsis(features)}")


def step_finished_line(event: BootStepFinished, features: ConsoleFeatures) -> str:
    """A finished step's durable ledger line: glyph + label · detail · elapsed."""

    glyph = event.outcome.glyph_for(use_unicode=features.can_unicode)
    text = event.label
    if event.detail:
        text += f" · {event.detail}"
    text += f" · {step_seconds(event.elapsed_s)}"
    return indent_string(f"{glyph} {text}")


def ready_line(elapsed_s: float) -> str:
    """The boot capstone line."""

    return indent_string(f"ready in {step_seconds(elapsed_s)}")


@final
class BootRegion(LiveRegion):
    """The boot cockpit: the live spinner plus its durable step ledger.

    Durable lines (banner, graduations, heads-up, capstone) print the moment
    their event arrives - they reflow ABOVE the transient spinner via the shared
    Console lock, exactly like the diagnostic lines. The spinner is torn down by
    `scope_closed` when the renderer's fold evicts the boot-section node
    (whatever event evicted it - and defensively by `begin_run`/`close`),
    so scan output never lands under a stale live region.
    """

    def __init__(
        self,
        console_source: Callable[[], Console | None],
        detected_features: DetectedFeatures | None = None,
        *,
        level_source: Callable[[], int],
    ) -> None:
        super().__init__(console_source, detected_features, level_source=level_source)
        # The only cross-event frame state: Progressed events carry no label.
        self._label = ""

    def handle(self, event: BootEvent) -> None:
        console = self._console_source()
        if console is None:
            return
        features = self._detected_features.for_console(console)
        match event:
            case PearlarrStarted():
                self._banner(console, event)
            case BootStepStarted(label=label):
                self._step_started(console, features, label)
            case BootStepProgressed(fraction=fraction, detail=detail):
                if self._spinner is not None:
                    self._spinner.update(text=self._frame_text(features, fraction, detail))
            case BootStepSlow(label=label):
                # Live consoles show liveness via the spinner. The heads-up is
                # the non-live rich console's degraded liveness signal.
                if not features.can_redraw and self._admits_durable():
                    self._print(console, Text(slow_line(label, features)))
            case BootStepFinished():
                if self._admits_durable():
                    style = event.outcome.style if features.can_color else ""
                    self._print(console, Text(step_finished_line(event, features), style=style))
            case BootReady(elapsed_s=elapsed_s):
                # Teardown order: the spinner is torn down before the capstone
                # prints (scope_closed stays the teardown for capstone-less ends).
                self._stop_live()
                if self._admits_durable():
                    self._print(console, Text(ready_line(elapsed_s), style="grey50" if features.can_color else ""))
            case _:
                assert_never(event)

    @override
    def _reset(self) -> None:
        super()._reset()
        self._label = ""

    def _admits_durable(self) -> bool:
        """Level parity with the logger-driven ledger: INFO lines need level <= INFO.

        Deliberately NOT the diagnostics' `diagnostic_threshold` gate: at a
        configured WARNING the boot ledger vanishes, matching the file log.
        """

        return self._level_source() <= logging.INFO

    def _banner(self, console: Console, event: PearlarrStarted) -> None:
        # Parity: the same look the TitledRule/StyledLine payload route renders.
        if not self._admits_durable():
            return
        print_titled_rule(console, banner_title(event.version), "bold cyan", heavy=True)
        console.print(Text(""))
        self._print(console, Text(data_dir_line(event.data_dir), style="grey50"))

    def _step_started(self, console: Console, features: ConsoleFeatures, label: str) -> None:
        self._label = label
        if not features.can_redraw:
            return
        if self._live is None:
            self._live = make_live(console)
            self._live.start()
        # A fresh Spinner per step restarts the animation at frame 0 (intended).
        # Within a step only .text mutates so the dots keep their phase. Padding
        # holds the spinner by reference, so progress updates re-render live.
        self._spinner = Spinner(spinner_name(features), text=self._frame_text(features, None, None), style="cyan")
        self._live.update(Padding(self._spinner, (0, 0, 0, len(INDENT))))

    def _frame_text(self, features: ConsoleFeatures, fraction: float | None, detail: str | None) -> Text:
        line = Text(self._label, style="bold")
        if fraction is not None:
            line.append("  ")
            line.append(block_bar(fraction, _BAR_WIDTH, features))
            line.append(f" {round(fraction * 100)}%", style="cyan")
        if detail:
            line.append("  ")
            line.append(detail, style="grey50")
        elif fraction is None:
            line.append(_ellipsis(features), style="bold")
        return line

    @staticmethod
    def _print(console: Console, text: Text) -> None:
        print_literal(console, text)
