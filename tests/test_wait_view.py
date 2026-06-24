"""Tests for the wait-pass presentation (``wait_view``).

Verifies the factory picks the live view on a TTY and the heartbeat view
otherwise, and that both implementations drive without raising. The live view is
exercised against a forced-terminal rich Console writing to a buffer, so no real
terminal is needed.
"""

import io
import logging

from rich.console import Console

from seadexarr.modules.log import RichConsoleHandler
from seadexarr.modules.wait_view import (
    _HeartbeatWaitView,
    _LiveWaitView,
    make_wait_view,
)


def _logger_with_console(*, force_terminal: bool) -> logging.Logger:
    logger = logging.getLogger(f"wait-view-test-{force_terminal}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = Console(file=io.StringIO(), force_terminal=force_terminal)
    logger.addHandler(RichConsoleHandler(console))
    return logger


def test_factory_returns_heartbeat_without_console() -> None:
    logger = logging.getLogger("wait-view-null")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    view = make_wait_view(logger, poll_s=30)

    assert isinstance(view, _HeartbeatWaitView)


def test_factory_returns_live_on_a_tty() -> None:
    logger = _logger_with_console(force_terminal=True)

    view = make_wait_view(logger, poll_s=30)

    assert isinstance(view, _LiveWaitView)


def test_heartbeat_view_drives_without_raising() -> None:
    logger = _logger_with_console(force_terminal=False)
    view = make_wait_view(logger, poll_s=30)

    view.start([("h1", "Show A"), ("h2", "Show B")])
    view.download("h1", 0.5, 12.0, 60.0)
    view.phase_sonarr("h1", 5.0, 30.0)
    view.done("h1", "imported")
    view.done("h2", "left for a later run")
    view.close()

    output = _console_output(logger)
    assert "downloading 50%" in output
    assert "imported" in output


def test_live_view_drives_and_writes() -> None:
    logger = _logger_with_console(force_terminal=True)
    view = make_wait_view(logger, poll_s=30)

    view.start([("h1", "Show A")])
    view.download("h1", 0.25, 10.0, 60.0)
    view.phase_sonarr("h1", 2.0, 30.0)
    view.done("h1", "imported")
    view.close()

    # A live region writes ANSI control sequences + the task description.
    assert "Show A" in _console_output(logger)


def _console_output(logger: logging.Logger) -> str:
    for handler in logger.handlers:
        if isinstance(handler, RichConsoleHandler):
            stream = handler.console.file
            assert isinstance(stream, io.StringIO)
            return stream.getvalue()
    return ""
