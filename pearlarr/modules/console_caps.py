"""Shared console-capability probing for the render seats (wait + boot).

The output package's live regions (`boot_region`,
`wait_region`) drive an optional sticky `rich.Live` region over
the SAME `Console` the logger already owns, and both must degrade to a calm
log digest on a non-TTY (Docker / a pipe / a dumb or too-narrow terminal); the
wait narrator's `wants_telemetry` probe branches on the same signals. The probe lives here once: `console_of` finds the logger's
console and `detect_capabilities` folds rich's derived signals into the
small `Capabilities` value the seats branch on.
"""

import logging
from dataclasses import dataclass
from typing import final

from rich.console import Console
from rich.live import Live
from rich.text import Text

from .log import RichConsoleHandler

# Below this console width a sticky live region can't be drawn legibly, so the
# views fall back to the log digest (the same path a non-TTY / dumb terminal
# takes).
MIN_LIVE_WIDTH = 40

# rich's own refresh cadence (frames/sec) for both live cockpits: the spinner
# animates and any timers tick at this rate on rich's background thread.
LIVE_REFRESH_PER_SECOND = 12.5


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the output stream can do, probed once - drives mode + glyph choices.

    `live` -> may we drive a sticky live region (a real, non-dumb, wide-enough
    TTY)? `color` -> may we emit ANSI color? `unicode` -> may we use `✔`/box
    glyphs, or must we fall back to ASCII? `width`/`height` -> the clamped
    render size.
    """

    live: bool
    color: bool
    unicode: bool
    width: int
    height: int


def console_of(logger: logging.Logger) -> Console | None:
    """The rich Console behind the logger's console handler, if any."""

    for handler in logger.handlers:
        if isinstance(handler, RichConsoleHandler):
            return handler.console
    return None


def detect_capabilities(console: Console | None) -> Capabilities:
    """Fold rich's derived console signals into our render capabilities.

    Reads rich's own folded flags (which already honor `NO_COLOR` / `TERM` /
    isatty / legacy Windows) rather than re-parsing the environment; the only
    things added on top are the `MIN_LIVE_WIDTH` floor and a glyph-encodability
    probe, which decide box-vs-lines and the glyph set.
    """

    if console is None:
        return Capabilities(live=False, color=False, unicode=False, width=80, height=24)
    size = console.size
    width = size.width or 80
    height = size.height or 24
    live = console.is_terminal and not console.is_dumb_terminal and width >= MIN_LIVE_WIDTH
    return Capabilities(
        live=live,
        color=console.color_system is not None,
        unicode=_supports_unicode(console),
        width=width,
        height=height,
    )


@final
class CapsCache:
    """An identity-keyed `detect_capabilities` cache for the render seats.

    The console seat's regions (boot + wait) branch on the probe (the slow
    heads-up policy); they must share ONE instance per hub, or a mid-boot resize
    across `MIN_LIVE_WIDTH` could flip one surface's decision only.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: tuple[Console, Capabilities] | None = None

    def for_console(self, console: Console | None) -> Capabilities:
        """The cached probe for `console`; a new identity re-probes and replaces."""

        if console is None:
            return detect_capabilities(None)
        state = self._state
        if state is None or state[0] is not console:
            state = (console, detect_capabilities(console))
            self._state = state
        return state[1]

    def reset(self) -> None:
        """Drop the cached probe (cycle start; idempotent)."""

        self._state = None


def block_bar(fraction: float, width: int, caps: Capabilities) -> Text:
    """A fixed-width cyan progress bar (unicode blocks, or ASCII fallback)."""

    filled = round(max(0.0, min(1.0, fraction)) * width)
    if caps.unicode:
        return Text("█" * filled + "░" * (width - filled), style="cyan")
    return Text("#" * filled + "-" * (width - filled), style="cyan")


def spinner_name(caps: Capabilities) -> str:
    """The spinner glyph set the console can draw."""

    return "dots" if caps.unicode else "line"


def make_live(console: Console) -> Live:
    """The transient, auto-refreshing `rich.Live` both cockpits drive."""

    return Live(
        console=console,
        auto_refresh=True,
        refresh_per_second=LIVE_REFRESH_PER_SECOND,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def _supports_unicode(console: Console) -> bool:
    """Whether the console can encode the glyphs/blocks the live views draw."""

    if getattr(console, "legacy_windows", False):
        return False
    encoding = console.encoding or "utf-8"
    try:
        "✔━▏░".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True
