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
* The PR2 seam (S5 pin 2): the rich handler badge-renders plain WARNING+
  records UNLESS the registered console owner answers True (the bridge adopts
  them; the hub's renderer places them) — no owner or a struck-out seat keeps
  the legacy badge, so warnings can never vanish. ``HUB_EVENT``-marked
  re-emissions are always skipped; the payload arms and plain INFO/DEBUG render
  exactly as before. The badge + traceback rendering is also pinned on the
  renderer side in test_output_rich_renderer.py.
"""

import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

from seadexarr.modules.config import LogFormat
from seadexarr.modules.console_caps import console_of
from seadexarr.modules.log import (
    CONSOLE_EXTRA,
    DETAIL_KEY_WIDTH,
    HUB_EVENT,
    JsonFormatter,
    KvLine,
    PlainConsoleHandler,
    RichConsoleHandler,
    apply_log_level,
    log_titled_rule,
    register_console_owner,
    setup_logger,
)

from .fakes import TtyStringIO, strip_ansi


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
        setup = build("auto", stream=TtyStringIO())
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
    """The PR2 seam: hub placement owns the badge class only while the registered
    console owner answers True; otherwise the legacy badge renders (fallback)."""

    def test_plain_warning_and_error_records_skip_the_handler_when_the_hub_owns_the_console(self) -> None:
        """Plain WARNING+ (incl. exc_info) never render here while the hub's
        renderer owns the console - it draws the badge (double render otherwise)."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-badge")
        register_console_owner(lambda: True)  # conftest's uninstall_bridge clears it

        logger.warning("watch out")
        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("sync failed", exc_info=True)

        assert stream.getvalue() == ""

    def test_plain_warning_renders_the_badge_without_a_console_owner(self) -> None:
        """No bridge installed (library use, the pre-install window): pre-PR2 look."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-no-owner")

        logger.warning("watch out")

        assert "WARNING  watch out" in strip_ansi(stream.getvalue())

    def test_plain_warning_renders_the_badge_when_the_console_seat_struck_out(self) -> None:
        """The quarantine fallback: a False owner (struck-out seat) means the hub
        renderer is NOT drawing badges, so the legacy badge must - never both,
        never neither."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-struck-out")
        register_console_owner(lambda: False)

        logger.warning("watch out")

        assert "WARNING  watch out" in strip_ansi(stream.getvalue())

    def test_hub_event_marked_records_skip_the_handler(self) -> None:
        """A LegacyRenderer re-emission was already placed by the hub's renderer."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-hub-event")

        logger.info("httpx: flaky pool", extra={HUB_EVENT: True})

        assert stream.getvalue() == ""

    def test_plain_info_still_renders_without_a_badge(self) -> None:
        logger, stream = _rich_setup("seadexarr-test-rich-seam-info")

        logger.info("checking Frieren")

        assert stream.getvalue() == "checking Frieren\n"

    def test_payload_arms_still_render_at_warning(self) -> None:
        """A WARNING kv line keeps its aligned, badge-free legacy render (severity
        is carried by LogCounter/the hub tally, not a column-0 badge)."""

        logger, stream = _rich_setup("seadexarr-test-rich-seam-kv")

        payload = KvLine(key="missing", value="S01E03", key_width=DETAIL_KEY_WIDTH, indent=2, sep="")
        logger.warning("missing S01E03", extra={CONSOLE_EXTRA: payload})
        log_titled_rule(logger, "Sonarr", heavy=True)

        out = stream.getvalue()
        assert "S01E03" in out
        assert "WARNING" not in out
        assert "Sonarr" in out  # the TitledRule arm is untouched

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
    """setup_logger's invalid-level critical fires BEFORE any hub/bridge exists."""

    def test_complaint_renders_on_the_rich_console_without_an_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # cli wires the hub AFTER setup_logger, so no console owner is registered
        # yet and the legacy badge renders the complaint (the F2 default).
        stream = TtyStringIO()
        monkeypatch.setattr(sys, "stdout", stream)

        setup_logger(log_level="BOGUS", log_dir=str(tmp_path / "logs"), console_format="rich")

        assert "CRITICAL Invalid log level 'BOGUS'" in strip_ansi(stream.getvalue())
