# pyright: strict
"""Tests for ``setup_logger``'s post-flip handler graph (the log.py side).

Pins the contract:

* "rich" (and "auto" on a TTY) attaches exactly ONE :class:`RichConsoleHandler`;
  it is the only non-bridge handler — no FileHandler anywhere (the hub's
  FileLogSink owns the file), no logger filters (severity tallies live on the
  hub's ``SeverityCounts``).
* "plain"/"json" (and "auto" off a TTY) attach NO console handler at all:
  level-only configuration; the bridge is the only handler, so records still
  reach the hub and ``logging.lastResort`` can never fire.
* ``console_of`` -> None under plain/json, so the live cockpits never build -
  designed, not a bug (the hub seats LineRenderer/JsonRenderer instead).
* ``apply_log_level`` re-points the rich console handler's threshold
  (``console_level`` semantics) and forwards the raw level to the hub.
* The badge seam (S5 pin 2): the rich handler badge-renders plain WARNING+
  records UNLESS the registered console owner answers True (the bridge adopts
  them; the hub's renderer places them) — no owner or a struck-out seat keeps
  the legacy badge, so warnings can never vanish.
* The invalid-level complaint fires AFTER handler attach: on a rich console
  with no owner the legacy badge renders it; under plain/json it arrives as a
  hub Diagnostic through the bridge (advisor #17's early-record path).
"""

import io
import logging
import sys
from pathlib import Path

import pytest
from rich.console import Console

from seadexarr.modules.config import LogFormat
from seadexarr.modules.console_caps import console_of
from seadexarr.modules.log import (
    RichConsoleHandler,
    apply_log_level,
    mark_hub_console_owner,
    setup_logger,
)
from seadexarr.modules.output import (
    Diagnostic,
    OutputHub,
    Severity,
    install_bridge,
    install_hub,
    uninstall_bridge,
)
from seadexarr.modules.output.bridge import HubBridgeHandler
from seadexarr.modules.output.recording import RecordingRenderer

from .fakes import CaptureHandler, TtyStringIO, strip_ansi


@pytest.fixture
def build(app_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> "_Builder":
    del app_logger  # isolation + teardown ordering only
    return _Builder(monkeypatch)


class _Builder:
    """Build a ``setup_logger`` logger over a swapped-in stdout stream.

    Swapping ``sys.stdout`` (rather than relying on pytest capture) makes the
    TTY probe deterministic: ``StringIO.isatty()`` is False even under ``-s``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    def __call__(self, console_format: LogFormat, stream: io.StringIO | None = None) -> logging.Logger:
        self._monkeypatch.setattr(sys, "stdout", io.StringIO() if stream is None else stream)
        return setup_logger(log_level="INFO", console_format=console_format)


def _non_bridge_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if not isinstance(h, HubBridgeHandler)]


class TestHandlerGraph:
    def test_plain_attaches_no_handler_at_all(self, build: _Builder) -> None:
        # Level-only configuration: the hub's LineRenderer owns plain stdout and
        # the FileLogSink owns the file; the bridge is the only record path.
        logger = build("plain")
        assert _non_bridge_handlers(logger) == []
        assert logger.level == logging.INFO

    def test_json_attaches_no_handler_at_all(self, build: _Builder) -> None:
        logger = build("json")
        assert _non_bridge_handlers(logger) == []

    def test_rich_attaches_exactly_one_rich_console_handler(self, build: _Builder) -> None:
        logger = build("rich")
        handlers = _non_bridge_handlers(logger)
        assert len(handlers) == 1
        assert isinstance(handlers[0], RichConsoleHandler)

    def test_no_file_handler_and_no_filters_anywhere(self, build: _Builder) -> None:
        # The FileLogSink owns the file; SeverityCounts owns the tallies.
        logger = build("rich")
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert logger.filters == []

    def test_setup_logger_never_touches_the_filesystem(self, build: _Builder, tmp_path: Path) -> None:
        # No makedirs, no rotation, no log file: the data dir stays untouched.
        before = sorted(tmp_path.rglob("*"))
        build("rich")
        assert sorted(tmp_path.rglob("*")) == before

    def test_no_rich_console_under_plain_so_live_views_degrade(self, build: _Builder) -> None:
        logger = build("plain")
        assert console_of(logger) is None


class TestFormatSelection:
    def test_auto_on_a_non_tty_attaches_nothing(self, build: _Builder) -> None:
        logger = build("auto")
        assert _non_bridge_handlers(logger) == []

    def test_auto_on_a_tty_picks_rich(self, build: _Builder) -> None:
        logger = build("auto", stream=TtyStringIO())
        assert any(isinstance(h, RichConsoleHandler) for h in logger.handlers)

    def test_explicit_rich_forces_rich_off_a_tty(self, build: _Builder) -> None:
        logger = build("rich")
        assert any(isinstance(h, RichConsoleHandler) for h in logger.handlers)


class TestApplyLogLevel:
    def test_repoints_the_rich_console_threshold_but_not_below_info(self, build: _Builder) -> None:
        logger = build("rich")
        console = next(h for h in logger.handlers if isinstance(h, RichConsoleHandler))

        # Raising the level quiets the sinks but the console keeps INFO+
        # (routine progress stays visible)...
        apply_log_level(logger, "ERROR")
        assert logger.level == logging.ERROR
        assert console.level == logging.INFO
        # ...while DEBUG moves the console threshold with the logger.
        apply_log_level(logger, "DEBUG")
        assert console.level == logging.DEBUG

    def test_plain_level_only_repoint(self, build: _Builder) -> None:
        # No console handler to re-point: the logger level alone changes (the
        # hub's set_level fan-out is pinned in test_output_hub).
        logger = build("plain")
        apply_log_level(logger, "ERROR")
        assert logger.level == logging.ERROR
        assert _non_bridge_handlers(logger) == []


def _rich_setup(name: str) -> tuple[logging.Logger, io.StringIO]:
    """A DEBUG-level logger over a RichConsoleHandler writing to a buffer."""

    stream = io.StringIO()
    handler = RichConsoleHandler(Console(file=stream, width=200))
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger, stream


class TestRichHandlerSeam:
    """The badge seam: while a bridge is installed the hub owns the badge class
    outright (armed seat in-context, stderr fallback otherwise), so the handler
    stands down; with no bridge the legacy badge renders (standalone fallback)."""

    def test_plain_warning_and_error_records_skip_the_handler_when_the_hub_owns_the_console(self) -> None:
        """Plain WARNING+ (incl. exc_info) never render here while a bridge is
        installed - the hub places the badge (double render otherwise)."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-badge")
        mark_hub_console_owner()  # conftest's uninstall_bridge clears it

        logger.warning("watch out")
        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("sync failed", exc_info=True)

        assert stream.getvalue() == ""

    def test_plain_warning_renders_the_badge_without_a_console_owner(self) -> None:
        """No bridge installed (library use, the pre-install window): the fallback."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-no-owner")

        logger.warning("watch out")

        assert "WARNING  watch out" in strip_ansi(stream.getvalue())

    def test_plain_warning_skips_the_handler_even_with_the_hub_seat_inactive(self) -> None:
        """Ownership is registration, not seat state: pre-begin_cycle or after a
        strike-out the hub's STDERR fallback is the single net - the handler
        rendering here too printed the same warning twice (once per net)."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-struck-out")
        mark_hub_console_owner()

        logger.warning("watch out")

        assert stream.getvalue() == ""

    def test_plain_info_still_renders_without_a_badge(self) -> None:
        logger, stream = _rich_setup("seadexarr-test-rich-seam-info")

        logger.info("checking Frieren")

        assert stream.getvalue() == "checking Frieren\n"

    def test_debug_exc_info_renders_the_traceback_but_never_frame_locals(self) -> None:
        """The handler-side secrets pin: exc_info tracebacks render with
        show_locals=False, so a frame local's VALUE (an api key) can never leak."""

        logger, stream = _rich_setup("seadexarr-test-rich-secrets")
        sentinel = "hun" + "ter2"  # runtime-assembled: never a contiguous string here

        try:
            leaked = sentinel
            raise ValueError(f"qbit exploded holding {len(leaked)} secret bytes")
        except ValueError:
            logger.debug("boom", exc_info=True)

        out = strip_ansi(stream.getvalue())
        assert "Traceback" in out
        assert "ValueError" in out
        assert "qbit exploded" in out
        assert sentinel not in out


class TestInvalidLevelComplaint:
    """setup_logger's invalid-level critical fires AFTER handler attach, so it
    always has a route: the rich console (badge fallback) or the bridge."""

    def test_complaint_renders_on_the_rich_console_without_an_owner(
        self, app_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No bridge installed (programmatic use): the legacy badge renders the
        # complaint on the rich console (the F2 fallback).
        del app_logger
        stream = TtyStringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        setup_logger(log_level="BOGUS", console_format="rich")

        assert "CRITICAL Invalid log level 'BOGUS'" in strip_ansi(stream.getvalue())

    def test_complaint_reaches_the_hub_under_plain_and_never_last_resort(
        self, app_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Advisor #17's early-record path: with hub+bridge installed FIRST (the
        cli order) and no console handler, the CRITICAL complaint arrives as a
        visible hub Diagnostic and logging.lastResort never fires."""

        del app_logger
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        last_resort = CaptureHandler()
        monkeypatch.setattr(logging, "lastResort", last_resort)
        recorder = RecordingRenderer()
        install_hub(OutputHub([recorder]))
        install_bridge()

        try:
            setup_logger(log_level="BOGUS", console_format="plain")
        finally:
            uninstall_bridge()

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.CRITICAL
        assert "Invalid log level 'BOGUS'" in diagnostic.message
        assert not diagnostic.file_only
        assert last_resort.records == []
