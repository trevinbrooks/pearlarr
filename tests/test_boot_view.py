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
from pathlib import Path
from typing import override

import httpx
from rich.console import Console

from seadexarr.modules.boot_view import (
    BootStep,
    BootView,
    LiveBootView,
    LogBootView,
    NullBootView,
    _DurableBootView,
    make_boot_view,
)
from seadexarr.modules.config import Arr
from seadexarr.modules.console_caps import Capabilities, TerminalEnv
from seadexarr.modules.log import LogCounter, RichConsoleHandler
from seadexarr.modules.mappings import MappingResolver
from seadexarr.modules.output import ScopeClosed, ScopeKind, ScopeOpened, install_hub
from seadexarr.modules.output.recording import RecordingHub
from seadexarr.modules.run_services import RunDeps

from .builders import SEP, make_bare_instance, make_config
from .fakes import strip_ansi


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
    # The name is shared across tests: a LogCounter one test adds must not leak
    # into the next under randomized ordering.
    logger.filters.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = Console(file=io.StringIO(), force_terminal=force_terminal, width=width)
    logger.addHandler(RichConsoleHandler(console))
    return logger, console


def _plain(console: Console) -> str:
    stream = console.file
    assert isinstance(stream, io.StringIO)
    return strip_ansi(stream.getvalue()).replace("\r", "")


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
    with view.step("Fetching Sonarr library") as fetching:
        clock.tick(1.2)
        fetching.note("42 series")
    view.end_section()
    view.close()

    out = _plain(console)
    assert "SeaDexArr" in out  # brand banner
    assert "✔ Reading config" in out
    assert f"✔ Fetching Sonarr library{SEP}42 series" in out  # detail rides the ledger line
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


# --- capstone error gate ---------------------------------------------------------


def test_error_logged_mid_step_without_raise_suppresses_the_capstone() -> None:
    # An ERROR that doesn't raise still means the section isn't "ready"; the view
    # sees it through the LogCounter setup_logger attaches to production loggers.
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    logger.addFilter(LogCounter())
    view = make_boot_view(logger, time_source=clock)

    with view.step("Reading config"):
        clock.tick(0.02)
        logger.error("half-configured arr refused")
    view.end_section()
    view.close()

    out = _plain(console)
    assert "✔ Reading config" in out  # the step itself still graduates as success
    assert "ready in" not in out


def test_error_logged_between_steps_suppresses_the_capstone() -> None:
    # Errors logged in an OPEN section but outside any step (e.g. a refused
    # selection after "Reading config") gate the capstone too - the counter delta
    # spans the whole section, not just step bodies.
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    logger.addFilter(LogCounter())
    view = make_boot_view(logger, time_source=clock)

    with view.step("Reading config"):
        clock.tick(0.02)
    logger.error("no runnable arr selected")
    view.end_section()
    view.close()

    assert "ready in" not in _plain(console)


def test_deferred_warn_still_allows_the_capstone() -> None:
    # A DEFERRED (⚠) finish - qBittorrent unconfigured, a SeaDex outage - is a
    # degraded but READY section; only failures and logged errors gate the capstone.
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    logger.addFilter(LogCounter())
    view = make_boot_view(logger, time_source=clock)

    with view.step("Connecting to qBittorrent") as step:
        clock.tick(0.05)
        step.warn("not configured")
    view.end_section()
    view.close()

    out = _plain(console)
    assert "⚠" in out
    assert "ready in" in out


def test_counterless_logger_still_gets_a_capstone() -> None:
    # Bare loggers (tests, embedders) carry no LogCounter and log_counter raises
    # LookupError; the gate must read that as "no errors", not swallow the capstone.
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    view = make_boot_view(logger, time_source=clock)

    with view.step("Opening cache"):
        clock.tick(0.1)
        logger.error("uncounted error")  # invisible without a counter, by design
    view.end_section()
    view.close()

    assert "ready in" in _plain(console)


def test_error_gate_resets_with_the_section() -> None:
    # end_section resets the snapshot: an error during one arr's section must not
    # suppress the NEXT arr's capstone (multi-arr runs share the view).
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    logger.addFilter(LogCounter())
    view = make_boot_view(logger, time_source=clock)

    with view.step("First step"):
        logger.error("first section error")
    view.end_section()
    with view.step("Second step"):
        clock.tick(0.1)
    view.end_section()
    view.close()

    assert _plain(console).count("ready in") == 1  # second section only


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
        step.note(f"anime-ids{SEP}anidb")
    view.end_section()
    view.close()

    out = _plain(console)
    assert out.count("Refreshing mappings…") == 1  # one heads-up, not four
    assert f"✔ Refreshing mappings{SEP}anime-ids{SEP}anidb" in out
    assert "ready in" in out


def test_log_view_heads_up_flag_rides_the_step_not_an_id_set() -> None:
    # An id(step)-keyed dedup set can collide once a dead step's address is reused
    # (CPython freelists), swallowing a later step's one-time heads-up; the flag
    # must live on the step itself. unicode=False also pins the heads-up ellipsis
    # degrading to ASCII dots alongside the glyphs.
    logger, console = _logger_with_console(force_terminal=False)
    caps = Capabilities(live=False, color=False, unicode=False, width=100, height=40)
    view = LogBootView(logger, caps, time_source=_FakeClock())

    first = BootStep(lambda _s: None, "First step")
    view._on_change(first)
    view._on_change(first)
    assert first.announced is True  # the throttle state is the step's own flag

    second = BootStep(lambda _s: None, "Second step")  # a fresh step, a fresh flag
    view._on_change(second)

    out = _plain(console)
    assert out.count("First step...") == 1
    assert out.count("Second step...") == 1
    assert "…" not in out  # an ASCII console never sees the unicode ellipsis


# --- production wiring: the unconfigured-qBittorrent boot warning ---------------


def test_rundeps_build_warns_deferred_when_qbit_unconfigured(tmp_path: Path) -> None:
    # Missing qBittorrent credentials put the whole run in perpetual preview; the
    # boot ledger must say so via a DEFERRED (⚠) step instead of silently skipping
    # the qBittorrent step.
    clock = _FakeClock()
    logger, console = _logger_with_console(force_terminal=True)
    view = make_boot_view(logger, time_source=clock)

    deps = RunDeps.build(
        Arr.SONARR,
        cache=str(tmp_path / "cache.db"),
        logger=logger,
        mappings=make_bare_instance(MappingResolver),
        app_config=make_config(),  # no qbittorrent credentials
        web=httpx.Client(),
        boot=view,
    )
    view.close()
    deps.cache_store.close()  # don't leak the sqlite handle past the test

    assert deps.qbit is None
    out = _plain(console)
    assert f"⚠ Connecting to qBittorrent{SEP}not configured - preview mode" in out


# --- pure render helper --------------------------------------------------------


def test_live_frame_text_draws_a_bar_with_percent() -> None:
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
    view = LiveBootView(
        TerminalEnv(
            Console(file=io.StringIO(), force_terminal=True),
            caps,
            logging.getLogger("boot-frame"),
            time_source=_FakeClock(),
        ),
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
        TerminalEnv(
            Console(file=io.StringIO(), force_terminal=True),
            caps,
            logging.getLogger("boot-frame-ascii"),
            time_source=_FakeClock(),
        ),
    )
    step = BootStep(lambda _s: None, "Refreshing mappings")
    step.progress(0.5)

    text = view._frame_text(step).plain
    assert "#" in text and "-" in text  # ascii fallback bar
    assert "█" not in text


def test_live_frame_text_ascii_ellipsis_without_unicode() -> None:
    # The bare "working" frame (no progress, no detail) must degrade its ellipsis
    # to ASCII dots on a console that can't encode "…", like the glyphs do.
    caps = Capabilities(live=True, color=True, unicode=False, width=100, height=40)
    view = LiveBootView(
        TerminalEnv(
            Console(file=io.StringIO(), force_terminal=True),
            caps,
            logging.getLogger("boot-frame-ellipsis"),
            time_source=_FakeClock(),
        ),
    )
    step = BootStep(lambda _s: None, "Reading config")

    text = view._frame_text(step).plain
    assert text == "Reading config..."
    assert "…" not in text


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


class TestBootSectionScope:
    """The B1/B2 graft: the view marks its boot section open/closed on the hub,
    so bridge-adopted diagnostics fired BETWEEN steps place at the ledger indent."""

    @staticmethod
    def _view_with_hub(*, force_terminal: bool) -> tuple[BootView, RecordingHub]:
        recording = RecordingHub()
        install_hub(recording.hub)
        logger, _console = _logger_with_console(force_terminal=force_terminal)
        return make_boot_view(logger, time_source=_FakeClock()), recording

    def test_banner_opens_and_end_section_closes_the_scope(self) -> None:
        # conftest's autouse teardown uninstalls the hub after every test.
        view, recording = self._view_with_hub(force_terminal=True)
        view.banner()
        with view.step("Reading config"):
            pass
        view.end_section()

        (opened,) = recording.of_type(ScopeOpened)
        (closed,) = recording.of_type(ScopeClosed)
        assert opened.scope.kind is ScopeKind.BOOT_SECTION
        assert closed.scope == opened.scope

    def test_a_step_after_end_section_opens_a_fresh_scope(self) -> None:
        # The second arr's boot steps reopen a section (bootstrap's per-arr loop).
        view, recording = self._view_with_hub(force_terminal=False)
        view.banner()
        view.end_section()
        with view.step("Fetching library"):
            pass
        view.end_section()

        first, second = recording.of_type(ScopeOpened)
        assert first.scope != second.scope
        assert [c.scope for c in recording.of_type(ScopeClosed)] == [first.scope, second.scope]

    def test_close_is_the_scope_safety_net(self) -> None:
        # bootstrap's finally-guarded teardown: a failed section still closes.
        view, recording = self._view_with_hub(force_terminal=False)
        view.banner()
        view.close()
        view.close()  # idempotent

        assert len(recording.of_type(ScopeOpened)) == 1
        assert len(recording.of_type(ScopeClosed)) == 1
