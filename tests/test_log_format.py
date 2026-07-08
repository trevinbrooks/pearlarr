# pyright: strict
"""Tests for the ``advanced.log_format`` console renderers (the log.py side).

Pins the knob's contract:

* "plain" installs a :class:`PlainConsoleHandler` whose lines are byte-for-byte
  the file log's (one shared timestamp+level format), so piped/Docker output
  reads like the log file.
* "json" emits one JSON object per line with stable key order (time, level,
  message, then exc when a traceback rides along); ``time`` carries a UTC
  offset so aggregators can order lines. Payload-only records keep their
  file-log message text (blank separators as ``""``) by design.
* "auto" resolves once at setup: rich on a TTY stdout, plain otherwise.
* Under plain/json there is no rich console (``console_of`` -> None), so the
  live cockpits degrade to their calm log digest - designed, not a bug.
* ``apply_log_level`` re-points the plain/json console handler exactly like
  the rich one, and never touches the file handler.
* rich's exc_info branch renders a level badge + message then a frame-capped
  traceback with ``show_locals=False`` (frame locals can hold config secrets).
"""

import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast, override

import pytest
from rich.console import Console

from seadexarr.modules.config import LogFormat
from seadexarr.modules.console_caps import console_of
from seadexarr.modules.log import (
    JsonFormatter,
    PlainConsoleHandler,
    RichConsoleHandler,
    apply_log_level,
    setup_logger,
)


class _TtyStringIO(io.StringIO):
    """An in-memory stream that claims to be a terminal (drives the "auto" TTY arm)."""

    @override
    def isatty(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class _Setup:
    """A built logger plus the in-memory stream standing in for stdout."""

    logger: logging.Logger
    stream: io.StringIO


class _Builder:
    """Build a ``setup_logger`` logger over a swapped-in stdout stream.

    Swapping ``sys.stdout`` (rather than relying on pytest capture) makes the
    TTY probe deterministic: ``StringIO.isatty()`` is False even under ``-s``.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tmp_path = tmp_path
        self._monkeypatch = monkeypatch

    def __call__(self, console_format: LogFormat, stream: io.StringIO | None = None) -> _Setup:
        stream = io.StringIO() if stream is None else stream
        self._monkeypatch.setattr(sys, "stdout", stream)
        logger = setup_logger(log_level="INFO", log_dir=str(self._tmp_path / "logs"), console_format=console_format)
        return _Setup(logger=logger, stream=stream)


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Builder:
    return _Builder(tmp_path, monkeypatch)


def _parsed_lines(setup: _Setup) -> list[dict[str, str]]:
    """Each emitted stdout line parsed as a JSON object (all values are strings)."""

    return [cast("dict[str, str]", json.loads(line)) for line in setup.stream.getvalue().splitlines()]


class TestPlainFormat:
    def test_lines_are_byte_for_byte_the_file_logs(self, build: _Builder, tmp_path: Path) -> None:
        setup = build("plain")
        setup.logger.info("hello")
        setup.logger.warning("watch out")
        for handler in setup.logger.handlers:
            handler.flush()

        out = setup.stream.getvalue()
        stamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.fullmatch(f"{stamp} INFO: hello\n{stamp} WARNING: watch out\n", out)
        # One shared format constant, so console and file can never drift.
        assert out == (tmp_path / "logs" / "SeaDexArr.log").read_text(encoding="utf-8")

    def test_no_rich_console_so_live_views_degrade(self, build: _Builder) -> None:
        setup = build("plain")
        assert not any(isinstance(h, RichConsoleHandler) for h in setup.logger.handlers)
        assert console_of(setup.logger) is None


class TestJsonFormat:
    def test_lines_are_json_with_stable_key_order(self, build: _Builder) -> None:
        setup = build("json")
        setup.logger.info("hello")
        setup.logger.info("")  # a blank separator: message "" by design (file-log parity)

        first, blank = _parsed_lines(setup)
        assert list(first) == ["time", "level", "message"]
        assert first["level"] == "INFO"
        assert first["message"] == "hello"
        assert blank["message"] == ""

    def test_time_carries_a_utc_offset(self, build: _Builder) -> None:
        setup = build("json")
        setup.logger.info("stamped")

        (line,) = _parsed_lines(setup)
        assert datetime.fromisoformat(line["time"]).utcoffset() is not None

    def test_exceptions_ride_along_as_traceback_text(self, build: _Builder) -> None:
        setup = build("json")
        try:
            raise ValueError("boom")
        except ValueError:
            setup.logger.error("failed", exc_info=True)

        (line,) = _parsed_lines(setup)
        assert list(line) == ["time", "level", "message", "exc"]
        assert line["message"] == "failed"
        assert line["exc"].startswith("Traceback")
        assert "ValueError" in line["exc"]


class TestFormatSelection:
    def test_auto_on_a_non_tty_picks_plain(self, build: _Builder) -> None:
        setup = build("auto")
        console = next(h for h in setup.logger.handlers if isinstance(h, PlainConsoleHandler))
        assert not isinstance(console.formatter, JsonFormatter)

    def test_auto_on_a_tty_picks_rich(self, build: _Builder) -> None:
        setup = build("auto", stream=_TtyStringIO())
        assert any(isinstance(h, RichConsoleHandler) for h in setup.logger.handlers)
        assert not any(isinstance(h, PlainConsoleHandler) for h in setup.logger.handlers)

    def test_explicit_rich_forces_rich_off_a_tty(self, build: _Builder) -> None:
        setup = build("rich")
        assert any(isinstance(h, RichConsoleHandler) for h in setup.logger.handlers)


class TestApplyLogLevelPlain:
    def test_repoints_the_plain_handler_but_never_the_file_handler(self, build: _Builder) -> None:
        setup = build("json")
        console = next(h for h in setup.logger.handlers if isinstance(h, PlainConsoleHandler))
        file_handler = next(h for h in setup.logger.handlers if isinstance(h, logging.FileHandler))

        # Same thresholds as the rich handler: raised levels keep INFO+ visible...
        apply_log_level(setup.logger, "ERROR")
        assert setup.logger.level == logging.ERROR
        assert console.level == logging.INFO
        # ...while DEBUG moves the console threshold with the logger.
        apply_log_level(setup.logger, "DEBUG")
        assert console.level == logging.DEBUG
        # The file handler's level is the logger's job, not the console re-point's.
        assert file_handler.level == logging.NOTSET


class TestRichExcInfoRender:
    def test_exc_info_renders_badge_and_traceback_but_never_locals(self) -> None:
        """An exc_info record renders the ERROR badge + message, then the traceback.

        Pins the secrets guarantee: ``show_locals=False`` means a frame local's
        VALUE is never rendered (locals can hold config secrets). The sentinel is
        assembled at runtime so the rendered SOURCE context lines cannot contain
        it - only rendered locals could. Also pins that the branch returns after
        the traceback (the message renders exactly once, no payload dispatch).
        """

        stream = io.StringIO()
        handler = RichConsoleHandler(Console(file=stream, width=200))
        logger = logging.getLogger("seadexarr-test-rich-exc-render")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        sentinel = "hunter2-" + "sentinel"  # never a contiguous string in this source file
        try:
            leaked = sentinel
            raise ValueError(f"qbit exploded holding {len(leaked)} secret bytes")
        except ValueError:
            logger.error("sync failed", exc_info=True)

        out = stream.getvalue()
        first_line = out.splitlines()[0]
        assert first_line.startswith("ERROR")
        assert "sync failed" in first_line
        assert "Traceback" in out
        assert "ValueError" in out
        assert "qbit exploded" in out
        assert sentinel not in out  # the frame local's value must never render
        # The branch returns after the traceback: exactly one badged message line
        # ("sync failed" also shows up as traceback source context, so count the
        # badge+message pair, which only the handler's own render produces). This
        # assert stays out of that context only because it sits past rich's
        # 3-line window around the raise - keep the distance.
        assert out.count("ERROR    sync failed") == 1
