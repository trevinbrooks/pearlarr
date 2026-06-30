# pyright: strict
# pyright: reportPrivateUsage=false
# ^ test imports/exercises boot_view internals (_DurableBootView, _frame_text).
"""Tests for the startup cockpit (``boot_view``).

The view wraps the pre-scan IO as timed steps: the composition root runs each
inside ``with view.step(...)`` and the step GRADUATES a glyph/color-coded ledger
line through the logger when it finishes (so it lands in scrollback AND the
plain-text file log). These pin the capability probe (animated cockpit on a real
TTY, calm digest otherwise), the graduation + "ready" capstone, the failure path
(``✖`` + caller exception still propagates), the non-TTY heads-up throttle, and
the no-throw contract. The live view is exercised against a forced-terminal rich
Console writing to a buffer, so no real terminal is needed; the step clock is
injected so durations are deterministic.
"""

import io
import logging
import re
from typing import override

from rich.console import Console

from seadexarr.modules.boot_view import (
    BootStep,
    LiveBootView,
    LogBootView,
    NullBootView,
    _DurableBootView,
    make_boot_view,
)
from seadexarr.modules.console_caps import Capabilities
from seadexarr.modules.log import RichConsoleHandler

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class _FakeClock:
    """A monotonic-ish clock the tests advance by hand, for stable durations."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _logger_with_console(
    *,
    force_terminal: bool,
    width: int = 100,
) -> tuple[logging.Logger, Console]:
    logger = logging.getLogger(f"boot-view-test-{force_terminal}-{width}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = Console(file=io.StringIO(), force_terminal=force_terminal, width=width)
    logger.addHandler(RichConsoleHandler(console))
    return logger, console


def _plain(console: Console) -> str:
    stream = console.file
    assert isinstance(stream, io.StringIO)
    return _ANSI.sub("", stream.getvalue()).replace("\r", "")


# --- factory / capability probe ------------------------------------------------


def test_factory_returns_log_view_without_console() -> None:
    logger = logging.getLogger("boot-view-null")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    assert isinstance(make_boot_view(logger), LogBootView)


def test_factory_returns_log_view_on_non_tty() -> None:
    logger, _ = _logger_with_console(force_terminal=False)

    assert isinstance(make_boot_view(logger), LogBootView)


def test_factory_returns_live_on_a_tty() -> None:
    logger, _ = _logger_with_console(force_terminal=True)

    assert isinstance(make_boot_view(logger), LiveBootView)


def test_factory_falls_back_to_log_view_when_too_narrow() -> None:
    logger, _ = _logger_with_console(force_terminal=True, width=20)

    assert isinstance(make_boot_view(logger), LogBootView)


# --- live cockpit --------------------------------------------------------------


def test_live_view_graduates_and_summarizes() -> None:
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    view = make_boot_view(logger, time_source=clock)

    view.banner()
    with view.step("Reading config"):
        clock.tick(0.02)
    with view.step("Connecting to Sonarr") as connecting:
        clock.tick(1.2)
        connecting.note("42 series")
    view.end_section()
    view.close()

    out = _plain(console)
    assert "SeaDexArr" in out  # brand banner
    assert "✔ Reading config" in out
    assert "✔ Connecting to Sonarr · 42 series" in out  # detail rides the ledger line
    assert "ready in" in out  # capstone


def test_live_graduations_reach_the_file_log_not_just_the_console() -> None:
    # The same defining property the wait ledger has: graduations go through the
    # LOGGER (both handlers), so a plain non-console handler - the file log stand-in
    # - sees them too. A live.console.print would skip this buffer.
    clock = _FakeClock()
    logger, _ = _logger_with_console(force_terminal=True)
    file_buffer = io.StringIO()
    logger.addHandler(logging.StreamHandler(file_buffer))

    view = make_boot_view(logger, time_source=clock)
    with view.step("Opening cache"):
        clock.tick(0.1)
    view.end_section()
    view.close()

    file_text = file_buffer.getvalue()
    assert "Opening cache" in file_text
    assert "ready in" in file_text


def test_live_failure_path_marks_and_reraises() -> None:
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    view = make_boot_view(logger, time_source=clock)

    raised = False
    try:
        with view.step("Reading config"):
            clock.tick(0.01)
            raise ValueError("bad config")
    except ValueError:
        raised = True
    view.close()

    out = _plain(console)
    assert raised  # the caller's exception is never swallowed
    assert "✖" in out  # graduated as FAILED
    assert "ready in" not in out  # no false "ready" capstone after a failure


def test_live_warn_graduates_as_warning() -> None:
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    view = make_boot_view(logger, time_source=clock)

    with view.step("Connecting to qBittorrent") as step:
        clock.tick(0.05)
        step.warn("not configured")
    view.close()

    out = _plain(console)
    assert "⚠" in out
    assert "not configured" in out


# --- non-TTY digest ------------------------------------------------------------


def test_log_view_is_calm_one_line_per_step() -> None:
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=False)
    view = make_boot_view(logger, time_source=clock)

    view.banner()
    with view.step("Refreshing mappings") as step:
        # A slow step reports many progress ticks; the digest must collapse them to
        # a single heads-up, not a per-MB flood.
        for frac in (0.25, 0.5, 0.75, 1.0):
            clock.tick(0.3)
            step.progress(frac, "anime_ids.json")
        step.note("anime-ids · anidb")
    view.end_section()
    view.close()

    out = _plain(console)
    assert out.count("Refreshing mappings…") == 1  # one heads-up, not four
    assert "✔ Refreshing mappings · anime-ids · anidb" in out
    assert "ready in" in out


# --- pure render helper --------------------------------------------------------


def test_live_frame_text_draws_a_bar_with_percent() -> None:
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
    view = LiveBootView(
        Console(file=io.StringIO(), force_terminal=True),
        caps,
        logging.getLogger("boot-frame"),
        time_source=_FakeClock(),
    )
    step = BootStep(lambda _s: None, "Refreshing mappings")
    step.progress(0.5, "anime_ids.json")

    text = view._frame_text(step).plain
    assert "50%" in text
    assert "█" in text and "░" in text  # half-filled unicode bar
    assert "anime_ids.json" in text


def test_live_frame_text_ascii_bar_without_unicode() -> None:
    caps = Capabilities(live=True, color=True, unicode=False, width=100, height=40)
    view = LiveBootView(
        Console(file=io.StringIO(), force_terminal=True),
        caps,
        logging.getLogger("boot-frame-ascii"),
        time_source=_FakeClock(),
    )
    step = BootStep(lambda _s: None, "Refreshing mappings")
    step.progress(0.5)

    text = view._frame_text(step).plain
    assert "#" in text and "-" in text  # ascii fallback bar
    assert "█" not in text


# --- NullBootView --------------------------------------------------------------


def test_null_view_is_a_total_no_op() -> None:
    view = NullBootView()
    view.banner()
    with view.step("anything") as step:
        step.progress(0.5, "x")
        step.note("y")
        step.warn()
    view.end_section()
    view.close()
    # Nothing to assert beyond "none of the above raised".


def test_null_view_step_still_propagates_caller_errors() -> None:
    view = NullBootView()
    raised = False
    try:
        with view.step("anything"):
            raise RuntimeError("boom")
    except RuntimeError:
        raised = True
    assert raised


# --- no-throw contract ---------------------------------------------------------


class _BoomBootView(_DurableBootView):
    """A view whose live hooks always raise - to prove the spine stays total."""

    @override
    def _begin(self, step: BootStep) -> None:
        raise RuntimeError("begin boom")

    @override
    def _on_change(self, step: BootStep) -> None:
        raise RuntimeError("change boom")

    @override
    def _stop_live(self) -> None:
        raise RuntimeError("stop boom")


def test_view_methods_are_total_and_never_raise() -> None:
    logger = logging.getLogger("boot-view-boom")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    caps = Capabilities(live=False, color=False, unicode=False, width=80, height=24)
    view = _BoomBootView(logger, caps, time_source=_FakeClock())

    # A presentation bug in begin/change/stop must degrade to a no-op, never abort
    # the real startup work it wraps.
    with view.step("x") as step:
        step.progress(0.5, "detail")
    view.end_section()
    view.close()


def test_boom_view_still_propagates_caller_errors() -> None:
    logger = logging.getLogger("boot-view-boom-2")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    caps = Capabilities(live=False, color=False, unicode=False, width=80, height=24)
    view = _BoomBootView(logger, caps, time_source=_FakeClock())

    raised = False
    try:
        with view.step("x"):
            raise ValueError("real error")
    except ValueError:
        raised = True
    assert raised  # presentation booms are swallowed; the caller's error is not
