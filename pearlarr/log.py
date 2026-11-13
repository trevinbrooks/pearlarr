"""Logging setup: the app logger, its rich console handler, and the level badges that handler prints."""

import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import override

from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

from .config import LogFormat

# True while the output bridge is installed: the rich handler must stand down or every record renders twice.
# With no bridge (standalone `setup_logger`) the legacy arms still render, so a record can never vanish.
_hub_owns_console = False


def mark_hub_console_owner() -> None:
    """The output bridge is installed: the hub owns the raw-record stream (`install_bridge`)."""

    global _hub_owns_console
    _hub_owns_console = True


def clear_console_owner() -> None:
    """Release console ownership (`uninstall_bridge`, tests)."""

    global _hub_owns_console
    _hub_owns_console = False


class HubBridgeBase(logging.Handler):
    """Marker base for the output bridge."""


def print_literal(console: Console, text: Text) -> None:
    """Print `text` as literal content: no markup/highlight, whole-line soft wrap."""

    console.print(text, highlight=False, soft_wrap=True)


@dataclass(frozen=True, slots=True)
class Badge:
    """One severity's console badge."""

    glyph: str
    word: str
    style: str


# INFO/DEBUG are deliberately absent, so those records print without a badge prefix.
LEVEL_BADGES = {
    logging.WARNING: Badge("⚠", "WARNING", "yellow"),
    logging.ERROR: Badge("✖", "ERROR", "bold red"),
    logging.CRITICAL: Badge("‼", "CRITICAL", "bold white on red"),
}


def console_supports_unicode(console: Console) -> bool:
    """Whether the console can encode the glyphs/blocks the styled surfaces draw."""

    if getattr(console, "legacy_windows", False):
        return False
    encoding = console.encoding or "utf-8"
    try:
        "✔━▏░⚠✖‼".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class RichConsoleHandler(logging.Handler):
    """The rich-TTY fallback surface for raw first-party records, and the shared Console's home."""

    # rich keeps the outermost and innermost frames and elides the middle beyond this.
    MAX_TRACEBACK_FRAMES = 10

    def __init__(self, console: Console, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.console = console
        # Probed once. The console identity is fixed for this handler's lifetime.
        self._use_unicode = console_supports_unicode(console)

    def _print_line(self, record: logging.LogRecord, message: str) -> None:
        """A plain message: level badge for WARNING+."""

        if LEVEL_BADGES.get(record.levelno) is None:
            line = Text(message)
        else:
            line = badge_line(record.levelno, message, use_unicode=self._use_unicode)

        print_literal(self.console, line)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _hub_owns_console:
                return

            message = record.getMessage()

            # Frame locals are never rendered: they can hold config secrets (api keys, webhook URLs).
            if record.exc_info:
                print_literal(self.console, badge_line(record.levelno, message, use_unicode=self._use_unicode))
                exc_type, exc_value, exc_tb = record.exc_info
                if exc_type is not None and exc_value is not None:
                    self.console.print(
                        Traceback.from_exception(
                            exc_type,
                            exc_value,
                            exc_tb,
                            show_locals=False,
                            max_frames=self.MAX_TRACEBACK_FRAMES,
                        ),
                    )
                return

            self._print_line(record, message)
        except Exception:
            self.handleError(record)


def badge_line(levelno: int, message: str, *, use_unicode: bool) -> Text:
    """`message` behind its level badge: the glyph ("⚠ ...") on a unicode console, the padded word on ASCII."""

    badge = LEVEL_BADGES.get(levelno, LEVEL_BADGES[logging.ERROR])
    prefix = f"{badge.glyph} " if use_unicode else f"{badge.word:<8} "
    line = Text(prefix, style=badge.style)
    line.append(message)
    return line


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogLevel(StrEnum):
    """The accepted log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def resolve_console_format(console_format: LogFormat) -> LogFormat:
    """Fold "auto" to the tty-detected concrete format."""

    if console_format == "auto":
        return "rich" if sys.stdout.isatty() else "plain"
    return console_format


def console_level(level: int) -> int:
    """The rich-console threshold for a logger level. The text surfaces use the raw level."""

    if level in (logging.DEBUG, logging.CRITICAL):
        return level
    return logging.INFO


def apply_log_level(logger: logging.Logger, log_level: str) -> None:
    """Re-point an already-built logger at `log_level`. Unknown names fall back to INFO."""

    level = _LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logger.setLevel(level)
    # Keep root in step: the config level lands here mid-cycle, and the root-seated bridge must see
    # sub-WARNING third-party records.
    logging.getLogger().setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, RichConsoleHandler):
            handler.setLevel(console_level(level))

    # Lazy: the output package imports this module at load.
    from .output.process_hub import current_hub

    current_hub().set_level(level)


# The app logger's name, and the log file stem.
LOG_NAME = "Pearlarr"


def setup_logger(
    log_level: str,
    console_format: LogFormat = "auto",
) -> logging.Logger:
    """Configure the app logger: level, plus a rich console handler under "rich". An unknown level means INFO."""

    logger = logging.getLogger(LOG_NAME)
    logger.propagate = False

    # Close and detach handlers from a previous call first (scheduled mode re-runs this each cycle): an
    # unclosed handler leaks its descriptor. The output bridge is installed once per process and must survive.
    for old_handler in list(logger.handlers):
        if isinstance(old_handler, HubBridgeBase):
            continue
        logger.removeHandler(old_handler)
        old_handler.close()

    level = _LOG_LEVELS.get(log_level.upper())
    invalid_log_level = log_level if level is None else None
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    # The bridge lives on the ROOT logger: open root's level too, so the bridge's own gate (not stdlib's
    # WARNING default) decides third-party records.
    logging.getLogger().setLevel(level)

    # Defensive fold for programmatic callers. cli resolves before calling.
    console_format = resolve_console_format(console_format)

    if console_format == "rich":
        console_handler = RichConsoleHandler(Console(file=sys.stdout))
        console_handler.setLevel(console_level(level))
        logger.addHandler(console_handler)

    # Emitted after the handler attach, so the bridge carries it and logging.lastResort never fires.
    # Deliberately raw, the sanctioned straggler the bridge adopts (allowlisted in tests/test_logging_ban.py).
    if invalid_log_level is not None:
        logger.critical(f"Invalid log level '{invalid_log_level}' - defaulting to 'INFO'")

    return logger
