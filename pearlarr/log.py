"""Logging setup and the shared console-rendering surfaces."""

import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import override

from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.traceback import Traceback

from .config import Arr, LogFormat


@dataclass(frozen=True, slots=True)
class TitledRule:
    """The payload for a titled section header."""

    title: str
    style: str = "bold cyan"
    """Rich style applied to both the rule and the title."""
    heavy: bool = False
    """`True` draws a heavy rule ("━"), `False` a light one ("─")."""


@dataclass(frozen=True, slots=True)
class SectionRule:
    """The payload for a full-width separator rule."""

    char: str = "-"


@dataclass(frozen=True, slots=True)
class KvLine:
    """An aligned "key : value" (or gutter "key value") detail line."""

    key: str
    value: str | Text
    """A plain string or a pre-styled `Text` (the only non-str path, from `group_highlight`)."""
    key_width: int
    value_style: str | None = None
    indent: int = 1
    sep: str = " :"
    tail: str | None = None
    tail_style: str = "yellow"


@dataclass(frozen=True, slots=True)
class StyledLine:
    """A plain message with a console style."""

    style: str = ""


type ConsoleRender = TitledRule | SectionRule | KvLine | StyledLine

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


def print_titled_rule(console: Console, title: str, style: str, *, heavy: bool) -> None:
    """A titled section header: a full-width rule, then the bold title line."""

    console.print(Rule(style=style, characters="━" if heavy else "─"))
    print_literal(console, Text(title, style=f"{style} bold"))


def render_kv(kv: KvLine) -> Text:
    """Build a styled "key : value" (or gutter "key value") line from a kv payload."""

    prefix = _kv_prefix(kv.indent, kv.key, kv.key_width, kv.sep)
    line = Text(prefix, style="grey50")
    value = kv.value
    if isinstance(value, Text):
        # A pre-styled value already carries its own spans, so append it as-is. value_style would flatten them.
        if len(value):
            line.append(" ")
            line.append(value)
    elif value != "":
        line.append(" ")
        line.append(Text(value, style=kv.value_style or ""))
    if kv.tail:
        line.append(" ")
        line.append(Text(kv.tail, style=kv.tail_style or "yellow"))
    return line


def render_rule(char: str) -> Rule:
    """A full-width separator: heavy ("━") for section ("=") breaks, light ("─") for sub ("-") breaks."""

    if char == "=":
        return Rule(style="cyan", characters="━")
    return Rule(style="grey37", characters="─")


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
    from .output.runtime import current_hub

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


INDENT = "  "

# The entry-ledger state column, padded so labels align. Widest state word is "unmonitored" (11).
STATE_WIDTH = 11

# Column the entry title starts at, measured from the end of its indent prefix.
ENTRY_LABEL_OFFSET = STATE_WIDTH + 1

# Detail-line keys sit in the indent gutter so the value lands at the title column, less the space kv adds.
DETAIL_INDENT = 2
DETAIL_KEY_WIDTH = (len(INDENT) + ENTRY_LABEL_OFFSET) - (DETAIL_INDENT * len(INDENT)) - 1


class EntryState(StrEnum):
    """The outcome word in an entry-ledger row's state column. The widest value fixes STATE_WIDTH."""

    UNCHANGED = "unchanged"
    IN_RADARR = "in radarr"
    CHECKING = "checking"
    UNMONITORED = "unmonitored"
    NO_MAPPING = "no mapping"
    NO_EPISODES = "no episodes"
    IGNORED = "ignored"
    NO_ENTRY = "no entry"
    SKIPPED = "skipped"
    QUEUED = "queued"
    DOWNLOADED = "downloaded"
    IMPORTED = "imported"


def entry_string(state: EntryState, label: str) -> str:
    """Format an entry-ledger body: "<state> <label>", state padded to STATE_WIDTH."""

    return f"{state.ljust(STATE_WIDTH)} {label}"


def _kv_prefix(indent: int, key: str, key_width: int, sep: str = " :") -> str:
    """Build the shared "<indent><key><sep>" leading segment for a kv line."""

    return f"{INDENT * indent}{key.ljust(key_width)}{sep}"


def rule_string(
    rule_char: str = "-",
    total_length: int = 80,
) -> str:
    """Draw a full-width separator rule for the (flat-style) logger."""

    return rule_char * total_length


def indent_string(
    text: str,
    level: int = 1,
) -> str:
    """Indent a rendered console row by `level` levels, each `INDENT` wide."""

    return f"{INDENT * level}{text}"


def kv_string(
    key: str,
    value: str | Text,
    key_width: int,
    indent: int = 1,
    sep: str = " :",
) -> str:
    """Format an aligned "key : value" line. Pass `sep=""` for the colon-less gutter "label value" format."""

    # Shared `_kv_prefix`, so the plain message matches the console render (render_kv) exactly.
    line = _kv_prefix(indent, key, key_width, sep)

    # An empty value acts as a header for the indented block below it.
    if value == "":
        return line

    return f"{line} {value}"


def compact_duration(seconds: float) -> str:
    """Compact duration: "40s" / "2m" / "1h05m". Negative floors to "0s"."""

    total = max(0, int(seconds))
    if total >= 3600:
        hours, minutes = divmod(total // 60, 60)
        return f"{hours}h{minutes:02d}m"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


def group_highlight(
    name: str | None,
    group: str | None,
    group_style: str = "cyan",
    base_style: str = "",
) -> str | Text:
    """Build a torrent-name value with its SeaDex release group called out."""

    name = name or ""
    if not group:
        return name

    # The group counts as already shown only when it leads the name, bare or in a "[Group]" wrapper. A match
    # buried mid-name doesn't count.
    cf, gf = name.casefold(), group.casefold()
    if cf.startswith(gf):
        start = 0
    elif cf.startswith(f"[{gf}"):
        start = 1
    else:
        start = -1

    text = Text(style=base_style)
    if start >= 0:
        end = start + len(group)
        text.append(name[:start])
        text.append(name[start:end], style=group_style)
        text.append(name[end:])
    else:
        # Prepend so the group always leads. The brackets stay in base_style so a prepended "[group]"
        # matches the "[group]" already in a name rather than coloring the whole wrapper.
        text.append("[")
        text.append(group, style=group_style)
        text.append("] ")
        text.append(name)
    return text


def pluralize(n: int, singular: str, plural: str | None = None) -> str:
    """Pick the singular or plural form of a word based on a count. `plural` unset means `singular` + "s"."""

    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def count_noun(n: int, singular: str, plural: str | None = None) -> str:
    """Format a count with its correctly pluralized noun, e.g. "3 movies"."""

    return f"{n} {pluralize(n, singular, plural)}"


def arr_item_noun(arr: Arr, n: int) -> str:
    """Format a count with the arr's library noun: "3 movies" / "3 series"."""

    if arr is Arr.RADARR:
        return count_noun(n, "movie")
    return count_noun(n, "series", "series")


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as e.g. "8s", "14m 03s", or "1h 02m 03s"."""

    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def human_bytes(num: float) -> str:
    """A compact human byte size, e.g. `"3.2 MB"` / `"1.8 GB"`."""

    val = num
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"
