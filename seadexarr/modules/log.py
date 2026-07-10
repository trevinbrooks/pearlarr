import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, TextIO, override

from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.traceback import Traceback

from .config import Arr, LogFormat


@dataclass(frozen=True, slots=True)
class TitledRule:
    """A titled section header: a full-width rule, then the bold title line.

    ``heavy=True`` draws a heavy rule ("━", run boundaries); otherwise a light
    rule ("─", per-title headers). Both the rule and the title take ``style``.
    """

    title: str
    style: str = "bold cyan"
    heavy: bool = False


@dataclass(frozen=True, slots=True)
class SectionRule:
    """A full-width separator rule: heavy cyan for "=", light gray for "-"."""

    char: str = "-"


@dataclass(frozen=True, slots=True)
class KvLine:
    """An aligned "key : value" (or gutter "key value") detail line.

    Locks the producer (the ``output.scan_lines`` builders) and consumer
    (``render_kv``) to the same fields, so the two sides
    can't silently drift. ``value`` is a plain string or a pre-styled ``Text``
    (the only non-str path, from ``group_highlight``).
    """

    key: str
    value: str | Text
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
"""The typed console payload a record carries under ``CONSOLE_EXTRA``."""

# The single ``extra=`` key carrying a ConsoleRender to the console handler.
CONSOLE_EXTRA: Final = "seadex_console"

# The ``extra=`` mark on hub re-emissions (output/legacy_echo.py): the bridge and
# the rich console handler drop marked records; the file/plain paths keep them.
HUB_EVENT: Final = "seadex_hub_event"

# The second mark on file_only re-emissions (hub containment notes): the file
# keeps them, but the plain/json stdout path and LogCounter skip them.
HUB_FILE_ONLY: Final = "seadex_hub_file_only"


def hub_event_marked(record: logging.LogRecord) -> bool:
    """True when ``record`` is a hub re-emission (carries the ``HUB_EVENT`` mark)."""

    mark: object = getattr(record, HUB_EVENT, None)
    return mark is not None


def hub_file_only_marked(record: logging.LogRecord) -> bool:
    """True when ``record`` is a file-only hub re-emission (``HUB_FILE_ONLY``)."""

    mark: object = getattr(record, HUB_FILE_ONLY, None)
    return mark is not None


# The hub's console-ownership probe, registered by ``install_bridge``: when it
# answers True the hub's renderer owns the badge class, so the rich handler skips
# plain WARNING+ records; absent/False (no bridge, struck-out console seat) the
# legacy badge renders — warnings can never vanish.
_console_owner: Callable[[], bool] | None = None


def register_console_owner(owner: Callable[[], bool]) -> None:
    """Install the hub's console-ownership probe (``install_bridge``)."""

    global _console_owner
    _console_owner = owner


def clear_console_owner() -> None:
    """Remove the probe (``uninstall_bridge``, tests)."""

    global _console_owner
    _console_owner = None


class HubBridgeBase(logging.Handler):
    """Marker base for the output bridge (output/bridge.py).

    A distinct base so ``setup_logger`` can preserve the bridge across its
    per-cycle handler rebuilds without importing the output package (which
    imports this module back).
    """


def console_payload(record: logging.LogRecord) -> ConsoleRender | None:
    """The typed console payload carried on ``record``, or None for a plain line."""

    payload: object = getattr(record, CONSOLE_EXTRA, None)
    if isinstance(payload, TitledRule | SectionRule | KvLine | StyledLine):
        return payload
    return None


def print_titled_rule(console: Console, title: str, style: str, *, heavy: bool) -> None:
    """A titled section header: a full-width rule, then the bold title line.

    Shared by the ``TitledRule`` handler arm and the boot banner, so the two
    console looks can't drift.
    """

    console.print(Rule(style=style, characters="━" if heavy else "─"))
    console.print(Text(title, style=f"{style} bold"), highlight=False, soft_wrap=True)


def render_kv(kv: KvLine) -> Text:
    """Build a styled "key : value" (or gutter "key value") line from a kv payload.

    The leading "<indent><key><sep>" segment comes from the shared _kv_prefix
    helper, so this matches kv_string (the file log) exactly. Labels are a
    fixed dim grey50 so the value reads first. An optional ``tail`` (e.g., a
    "(marked incomplete)" note) is appended console-side only. Module-level so
    the hub's console seat (output/scan_lines.py) shares the handler's look.
    """

    prefix = _kv_prefix(kv.indent, kv.key, kv.key_width, kv.sep)
    line = Text(prefix, style="grey50")
    value = kv.value
    if isinstance(value, Text):
        # A pre-styled value (e.g., a torrent name with its release group
        # highlighted) already carries its own spans, so append it as-is and
        # let those stand - value_style here would flatten them.
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
    """A full-width separator: heavy ("━") for section ("=") breaks, light ("─")
    for sub ("-") breaks, so the two stay distinguishable without color."""

    if "=" in char:
        return Rule(style="cyan", characters="━")
    return Rule(style="grey37", characters="─")


class RichConsoleHandler(logging.Handler):
    """Console log handler that renders records through ``rich``.

    Routine INFO/DEBUG lines print with no level prefix, so the output reads
    as clean text. PLAIN records at WARNING+ (the badge class, including
    exc_info records) are NOT rendered here while the registered console owner
    (the hub, via ``install_bridge``) answers True: the logging bridge
    (output/bridge.py) adopts them and the hub's rich renderer places them
    in-context (S5 pin 2) - rendering here too would double them. With no
    owner, or a struck-out console seat, the legacy badge renders so a warning
    can never vanish. Hub re-emissions (``HUB_EVENT``-marked records) are
    always skipped. Aligned "key : value" lines never get a badge - see
    :class:`KvLine` below.

    Presentation is driven by one typed payload (a :data:`ConsoleRender`
    dataclass under ``CONSOLE_EXTRA``, built by the ``output.scan_lines``
    builders), so the plain message string (what the file log
    stores) stays clean while the console gets the rich treatment:

    * :class:`TitledRule` -> a titled section: a full-width rule, then the
      title text LEFT-ALIGNED on the next line (both in its ``style``; the
      title is bold). ``heavy=True`` draws a heavy rule ("━", run boundaries);
      otherwise a light rule ("─", per-title headers). Used for the run banner
      and per-title headers.
    * :class:`SectionRule` -> a full-width separator rule (heavy cyan for
      "=", light gray for "-"), distinguishable without color. Unmarked ASCII
      separators (a message of only "=" / "-") are still detected as a fallback.
    * :class:`KvLine` -> an aligned, lightly colored "key : value" detail line
      whose layout matches ``kv_string`` exactly. Labels are a fixed dim grey,
      so the value reads first; ``value_style`` accents an outcome (e.g., green
      "added"). No level badge is drawn even for WARNING+ kv lines, so the
      value stays in its aligned column; LogCounter still tallies the severity
      for the run summary.
    * :class:`StyledLine` -> a style applied to an otherwise plain message
      (used to dim no-op lines such as the collapsed "cached" one-liner).

    Messages are rendered as literal text rather than ``rich`` markup, so
    bracketed content such as "[1/1]" or "[MARKED INCOMPLETE]" is never
    mistaken for a style tag.
    """

    # Level -> (badge label, rich style). INFO/DEBUG are deliberately absent,
    # so they print without a prefix.
    LEVEL_BADGES = {
        logging.WARNING: ("WARNING", "yellow"),
        logging.ERROR: ("ERROR", "bold red"),
        logging.CRITICAL: ("CRITICAL", "bold white on red"),
    }

    # Cap how many stack frames a rendered traceback shows. An unexpected crash
    # is logged as a legible excerpt (rich keeps the outermost and innermost
    # frames and elides the middle when the stack is deeper than this) rather
    # than a full-screen wall of frames.
    MAX_TRACEBACK_FRAMES = 10

    def __init__(self, console: Console, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.console = console

    def _print_rule(self, char: str) -> None:
        """Delegates to the module-level ``render_rule`` (shared with the hub seat)."""
        self.console.print(render_rule(char))

    def _print_line(self, record: logging.LogRecord, message: str, payload: StyledLine | None) -> None:
        """A plain message: level badge for WARNING+, optional style.

        Emitted whole (soft_wrap) so the terminal handles any overflow, rather
        than rich re-wrapping with unindented continuation lines.
        """
        if self.LEVEL_BADGES.get(record.levelno) is None:
            line = Text(message, style=payload.style if payload is not None else "")
        else:
            line = badge_line(record.levelno, message)

        self.console.print(line, highlight=False, soft_wrap=True)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # A hub re-emission: the hub's console renderer owns its rendering;
            # only the file/plain surfaces carry the record.
            if hub_event_marked(record):
                return

            payload = console_payload(record)

            # The badge class moved to hub placement (S5 pin 2): the bridge
            # adopts plain WARNING+ records, the hub's renderer draws the badge —
            # but only while the hub actually owns the console; with no bridge or
            # a struck-out console seat, the legacy badge below still renders.
            if (
                payload is None
                and record.levelno >= logging.WARNING
                and _console_owner is not None
                and _console_owner()
            ):
                return

            # A titled section: a full-width rule, then the title text
            # LEFT-ALIGNED on the next line (user directive).
            if isinstance(payload, TitledRule):
                print_titled_rule(self.console, payload.title, payload.style, heavy=payload.heavy)
                return

            message = record.getMessage()

            # An unexpected exception (logged with exc_info): show a colored
            # level badge + message, then a rich traceback with a capped frame
            # count so a crash is legible at a glance. Frame locals are never
            # rendered - they can hold config secrets (api keys, webhook URLs).
            # The file handler still records the full plain-text traceback (the
            # formatter renders exc_info), so nothing is lost from the log file.
            if record.exc_info:
                self.console.print(badge_line(record.levelno, message), highlight=False, soft_wrap=True)
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

            match payload:
                case SectionRule(char=char):
                    self._print_rule(char)
                case KvLine():
                    # No level badge here, even for WARNING+ kv lines: a col-0
                    # badge would push the value past its aligned column and
                    # detach the line from the entry block it belongs under.
                    # Severity is still counted by LogCounter (a logger filter)
                    # and surfaced in the run summary's "issues" tally; here,
                    # position and value_style carry the meaning.
                    self.console.print(
                        render_kv(payload),
                        highlight=False,
                        soft_wrap=True,
                    )
                case StyledLine() | None:
                    # Fallback: a hand-drawn ASCII rule (a message of only "="
                    # or "-") from a caller without the typed payload.
                    stripped = message.strip()
                    if payload is None and stripped and set(stripped) <= {"=", "-"}:
                        self._print_rule("=" if "=" in stripped else "-")
                    else:
                        self._print_line(record, message, payload)
        except Exception:
            self.handleError(record)


def badge_line(levelno: int, message: str) -> Text:
    """``message`` behind its level badge — the ONE home of the badge column grammar.

    Levels without a badge fall back to the ERROR one (only the exc_info arm,
    which always badges, can reach that).
    """

    label, style = RichConsoleHandler.LEVEL_BADGES.get(levelno, ("ERROR", "bold red"))
    line = Text(f"{label:<8} ", style=style)
    line.append(message)
    return line


class PlainConsoleHandler(logging.StreamHandler[TextIO]):
    """Marker type for the plain/json stdout handler (the formatter differs).

    A distinct subclass so ``apply_log_level`` can re-point the console handler
    without touching the file handler (``FileHandler`` is also a
    ``StreamHandler``), and so ``console_of`` keeps returning None here: the
    live cockpits deliberately degrade to their log digest under plain/json.
    """


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, message (+ exc), in that order.

    ``time`` is local time WITH its UTC offset so aggregators can order lines.
    Payload-only records keep their file-log message text (blank separators as
    ``""``, section rules as dash strings) by design - file-log parity.
    """

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created).astimezone().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class LogCounter(logging.Filter):
    """A logging filter that tallies records by level as a side effect.

    Attached to the logger so each record that reaches it is counted. Note that
    a logger only runs its filters after ``isEnabledFor``, so records below the
    logger's effective level never get here and are NOT counted. This is fine
    because it is used only for the per-run WARNING/ERROR/CRITICAL totals, which
    are always >= INFO (and so always above the effective level). Callers
    snapshot the totals at the start of a run and diff at the end to report
    those counts without having to instrument each call site.
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[int, int] = {}

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        # File-only hub notes are containment forensics, never a user-visible tally.
        if not hub_file_only_marked(record):
            self.counts[record.levelno] = self.counts.get(record.levelno, 0) + 1
        return True

    def snapshot(self) -> dict[int, int]:
        """Return a copy of the current per-level counts"""
        return dict(self.counts)


def log_counter(logger: logging.Logger) -> LogCounter:
    """The per-run :class:`LogCounter` filter ``setup_logger`` attached to ``logger``.

    Raises ``LookupError`` when absent: every logger the readers pass was built
    by ``setup_logger`` (or is a test logger that attaches its own counter).
    """

    for log_filter in logger.filters:
        if isinstance(log_filter, LogCounter):
            return log_filter
    msg = f"logger {logger.name!r} carries no LogCounter (was it built by setup_logger?)"
    raise LookupError(msg)


def _not_hub_file_only(record: logging.LogRecord) -> bool:
    """Console-handler filter: file-only hub notes reach the file sink alone."""

    return not hub_file_only_marked(record)


# The level names the file logger honors; any other value (a typo) warns and
# falls back to INFO. Kept as an explicit table, so the string ladder isn't
# reinvented for both the logger and the console handler below.
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogLevel(StrEnum):
    """The accepted log levels, as a choices-enum for the CLI's ``--log-level``."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def console_level(level: int) -> int:
    """The console-surface threshold for a logger level (the single body; the
    output package's ``console_threshold`` delegates here).

    The console always shows INFO+ so routine progress stays visible even when
    the file logger is raised - except DEBUG, which lowers the threshold, and
    CRITICAL, which raises it to match the file logger.
    """

    if level in (logging.DEBUG, logging.CRITICAL):
        return level
    return logging.INFO


def apply_log_level(logger: logging.Logger, log_level: str) -> None:
    """Re-point an already-built logger at ``log_level``.

    The CLI bootstraps its logger before the config file can be read (config
    errors must be loggable), then calls this once the config's
    ``advanced.log_level`` is known. Unknown names fall back to INFO; the
    config validates the level, so that arm only serves programmatic callers.
    """

    level = _LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for handler in logger.handlers:
        # Whichever console handler setup_logger installed (rich, or the
        # plain/json marker type); never the file handler.
        if isinstance(handler, RichConsoleHandler | PlainConsoleHandler):
            handler.setLevel(console_level(level))

    # Forward to the output hub (S4: each surface applies its own floor).
    # Imported lazily: the output package imports this module at load.
    from .output.runtime import current_hub

    current_hub().set_level(level)


# Logger and log-file name; also the log rotation cap (.log -> .log.1 ... .log.9).
LOG_NAME = "SeaDexArr"
MAX_LOG_FILES = 9

# One text format shared by the file log and the plain console renderer, so
# piped/Docker output reads exactly like the log file.
PLAIN_LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
PLAIN_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    log_level: str,
    log_dir: str,
    console_format: LogFormat = "auto",
) -> logging.Logger:
    """
    Set up the logger.

    Parameters:
        log_level (str): The log level to use
        log_dir (str): Full path to the directory for log files (resolved by the
            caller via ``paths.log_dir``; created here if missing).
        console_format (LogFormat): Console renderer - "rich" (the live styled
            console), "plain" (the file log's timestamped lines) or "json" (one
            JSON object per line). "auto" (default) resolves once, here: rich
            when stdout is a TTY, plain otherwise (pipes, Docker logs).

    Returns:
        A logger object for logging messages.
    """

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{LOG_NAME}.log")

    logger = logging.getLogger(LOG_NAME)
    logger.propagate = False

    # Close and detach any handlers from a previous call FIRST (scheduled mode
    # re-runs this each cycle): the old file handler must release the log file
    # before it is rotated, and an unclosed handler leaks its descriptor. The
    # output bridge is installed once per process (cli) and must survive.
    for old_handler in list(logger.handlers):
        if isinstance(old_handler, HubBridgeBase):
            continue
        logger.removeHandler(old_handler)
        old_handler.close()

    # Rotate prior logs: .log -> .log.1 -> ... -> .log.<MAX_LOG_FILES>. os.replace
    # overwrites the oldest atomically, so no copy/remove dance is needed.
    if os.path.isfile(log_file):
        for i in range(MAX_LOG_FILES - 1, 0, -1):
            old_log = os.path.join(log_dir, f"{LOG_NAME}.log.{i}")
            new_log = os.path.join(log_dir, f"{LOG_NAME}.log.{i + 1}")
            if os.path.exists(old_log):
                os.replace(old_log, new_log)

        os.replace(log_file, os.path.join(log_dir, f"{LOG_NAME}.log.1"))

    # Resolve the configured level once through the name->constant table instead
    # of a hand-rolled string ladder. Only the five standard names are accepted;
    # anything else (a typo) falls back to INFO - the complaint is emitted below,
    # AFTER the handlers are attached, so it reaches the console/file rather
    # than logging.lastResort.
    level = _LOG_LEVELS.get(log_level.upper())
    invalid_log_level = log_level if level is None else None
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # The text format the file log always uses, shared with the plain console
    # renderer below so piped output reads exactly like the log file.
    text_formatter = logging.Formatter(fmt=PLAIN_LOG_FORMAT, datefmt=PLAIN_LOG_DATEFMT)

    # Create the file handler. Rotation is performed manually above (once per
    # setup_logger() call); the handler opens a fresh file each run (mode="w").
    handler = logging.FileHandler(
        log_file,
        delay=True,
        mode="w",
        encoding="utf-8",
    )
    handler.setFormatter(text_formatter)

    # Resolve "auto" once, at setup: rich for an interactive terminal, plain
    # timestamped lines when piped / under Docker.
    if console_format == "auto":
        console_format = "rich" if sys.stdout.isatty() else "plain"

    if console_format == "rich":
        # Console logging through rich: routine lines print with no level
        # prefix; warnings and errors get a colored badge (RichConsoleHandler).
        console_handler: logging.Handler = RichConsoleHandler(Console(file=sys.stdout))
    else:
        # plain/json: a bare stdout handler; plain shares the file log's
        # formatter, json emits one object per line (JsonFormatter).
        console_handler = PlainConsoleHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter() if console_format == "json" else text_formatter)
    console_handler.setLevel(console_level(level))
    # File-only hub notes stay off stdout (the FileHandler carries no filter).
    console_handler.addFilter(_not_hub_file_only)

    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Only now can the invalid-level complaint reach the console + file log.
    if invalid_log_level is not None:
        logger.critical(f"Invalid log level '{invalid_log_level}', defaulting to 'INFO'")

    # Tally records by level so a run can report its warning/error counts (read
    # back via log_counter). Replace any counter from a previous call so counts
    # don't carry over.
    for existing_filter in list(logger.filters):
        if isinstance(existing_filter, LogCounter):
            logger.removeFilter(existing_filter)
    logger.addFilter(LogCounter())

    return logger


# Number of spaces each level of the flat layout is indented by
INDENT = "  "

# Entry "ledger" columns for one-line entry statuses (unchanged / checking /
# skipped / no entry / in radarr / ...). Each line is "<state> <label>" with the
# state padded to a fixed width, so the label (usually a title) starts at the
# same column on every row regardless of state-word length. STATE_WIDTH fits the
# widest state word we use ("unmonitored" = 11). The season/episode coverage and
# the SeaDex URL ride a separate continuation line under the title (see
# log_entry_coverage), not the ledger line.
STATE_WIDTH = 11

# Column the entry label (title) starts at, measured from the end of its indent
# prefix: state + " ". Continuation lines (the season/episode/URL detail) pad to
# this so they sit directly beneath the title. Derived from STATE_WIDTH so it
# can't drift.
ENTRY_LABEL_OFFSET = STATE_WIDTH + 1

# Entry-detail lines - the dim "label value" rows under an entry (files / link /
# status / group / added / kept / missing / skipped / anilist) - sit their VALUE
# in the same column as the entry title, with the label in the indent gutter and
# no colon, so the whole entry block reads as one aligned column. The label sits
# at indent level DETAIL_INDENT; the value lands at the title column
# (len(INDENT) + ENTRY_LABEL_OFFSET). kv adds one space between the padded key
# and the value, so subtract it here. Derived from the same constants as the
# title column, so the two can't drift.
DETAIL_INDENT = 2
DETAIL_KEY_WIDTH = (len(INDENT) + ENTRY_LABEL_OFFSET) - (DETAIL_INDENT * len(INDENT)) - 1


class EntryState(StrEnum):
    """The outcome word shown in an entry-ledger row's state column.

    A ``StrEnum`` so each member still equals its rendered string
    (``EntryState.UNCHANGED == "unchanged"``) and flows through
    ``entry_string``'s ``.ljust(STATE_WIDTH)`` byte-for-byte unchanged - while
    making the documented set the only representable states, enforced and
    discoverable instead of free-form literals spelled two ways at the call
    sites. The widest value ("unmonitored") fixes STATE_WIDTH.
    """

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
    IMPORTING = "importing"
    IMPORTED = "imported"


def entry_string(state: EntryState, label: str) -> str:
    """Format the body of an entry-ledger line: "<state> <label>".

    state is padded to STATE_WIDTH so the label lines up across rows regardless
    of state-word length. No indent is applied here; the caller wraps this with
    indent_string(level=1). Season/episode/URL detail is carried on a separate
    continuation line (see log_entry_coverage), not here.
    """

    return f"{state.ljust(STATE_WIDTH)} {label}"


def _kv_prefix(indent: int, key: str, key_width: int, sep: str = " :") -> str:
    """Build the shared "<indent><key><sep>" leading segment for a kv line.

    Single source of truth so the console handler (_render_kv) and the file
    formatter (kv_string) never drift in prefix/padding/separator. ``sep`` is
    " :" for summary "key : value" lines and "" for the gutter "label value"
    entry-detail lines (see DETAIL_KEY_WIDTH).
    """

    return f"{INDENT * indent}{key.ljust(key_width)}{sep}"


def rule_string(
    rule_char: str = "-",
    total_length: int = 80,
) -> str:
    """Draw a full-width separator rule for the (flat-style) logger

    Args:
        rule_char: Character to repeat across the rule. Defaults to "-"
        total_length: Width of the rule. Defaults to 80
    """

    return rule_char * total_length


def indent_string(
    text: str,
    level: int = 1,
) -> str:
    """Format an indented detail line for the (flat-style) logger

    Args:
        text: String to format
        level: Number of indent levels (each INDENT wide). Defaults to 1
    """

    return f"{INDENT * level}{text}"


def kv_string(
    key: str,
    value: str | Text,
    key_width: int,
    indent: int = 1,
    sep: str = " :",
) -> str:
    """Format an aligned "key : value" detail line for flat-style output

    Args:
        key: Left-hand label
        value: Right-hand value
        key_width: Column width the key is padded to, so the colons line up
        indent: Number of indent levels to prefix. Defaults to 1
        sep: Separator after the padded key. Defaults to " :"; pass "" for the
            colon-less gutter "label value" entry-detail format
    """

    # Built from the shared _kv_prefix helper, so the file log matches the
    # console handler (_render_kv) exactly.
    line = _kv_prefix(indent, key, key_width, sep)

    # Allow an empty value to act as a header for an indented block below it
    if value == "":
        return line

    return f"{line} {value}"


def log_styled(
    logger: logging.Logger,
    message: str,
    style: str | None,
) -> None:
    """Log a plain message that the console renders with a style

    The file log stores the plain message; the console applies ``style`` (e.g.,
    "grey50" to dim a no-op line).

    Args:
        logger: Logger the line is emitted through
        message: The message text (file log and console)
        style: Rich style for the console line; None renders unstyled
    """

    logger.info(message, extra={CONSOLE_EXTRA: StyledLine(style=style or "")})


def group_highlight(
    name: str | None,
    group: str | None,
    group_style: str = "cyan",
    base_style: str = "",
) -> "Text | str":
    """Build a torrent-name value with its SeaDex release group called out.

    The release group is the thing worth spotting at a glance on a grab line, so
    it gets the same accent the live log gives groups (``group_style``). When the
    group already leads the torrent name (bare, or in the usual "[Group]" wrapper,
    matched case-insensitively), that span is highlighted in place; otherwise the
    group is prepended in brackets so it always reads at the front - a match
    buried mid-name doesn't count. Returns a styled rich ``Text`` for the console;
    the file log sees its plain text via ``str()`` (so a prepended "[group] "
    still shows, just without color).

    With no group (or no name), the plain name is returned unchanged, so the
    caller's own ``value_style`` applies as before.

    Args:
        name: The torrent name as reported by the client / scraped from source
        group: The recommended SeaDex release group, or None
        group_style: Rich style for the group span/prefix. Defaults to "cyan"
            (the live log's group color)
        base_style: Rich style for the rest of the name. Defaults to "" (none)
    """

    name = name or ""
    if not group:
        return name

    # Only treat the group as "already shown" when it leads the name - bare, or
    # in the usual "[Group]" wrapper. A match buried mid-name doesn't count; we
    # prepend instead, so the group always reads at the front of the line.
    cf, gf = name.casefold(), group.casefold()
    if cf.startswith(gf):
        start = 0
    elif cf.startswith(f"[{gf}"):
        start = 1
    else:
        start = -1

    text = Text(style=base_style)
    if start >= 0:
        # Highlight the existing leading group in place
        end = start + len(group)
        text.append(name[:start])
        text.append(name[start:end], style=group_style)
        text.append(name[end:])
    else:
        # Not at the front - prepend it so the group always leads. Only the group
        # name takes the accent; the brackets stay in base_style so a prepended
        # "[group]" matches the "[group]" already in a name (brackets base, name
        # accented) rather than coloring the whole wrapper.
        text.append("[")
        text.append(group, style=group_style)
        text.append("] ")
        text.append(name)
    return text


def pluralize(n: int, singular: str, plural: str | None = None) -> str:
    """Pick the singular or plural form of a word based on a count

    Args:
        n: The count
        singular: The singular form, used when n == 1
        plural: The plural form. Defaults to None, i.e., singular + "s"
    """

    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def count_noun(n: int, singular: str, plural: str | None = None) -> str:
    """Format a count with its correctly pluralized noun, e.g. "3 movies"

    Args:
        n: The count
        singular: The singular noun
        plural: The plural noun. Defaults to None, i.e., singular + "s"
    """

    return f"{n} {pluralize(n, singular, plural)}"


def arr_item_noun(arr: Arr, n: int) -> str:
    """Format a count with the arr's library noun: "3 movies" / "3 series"

    Args:
        arr: Which arr the items belong to (picks movie vs series)
        n: The count
    """

    if arr is Arr.RADARR:
        return count_noun(n, "movie")
    return count_noun(n, "series", "series")


def format_elapsed(seconds: float) -> str:
    """Format an elapsed number of seconds as e.g. "8s", "14m 03s" or "1h 02m 03s" """

    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def human_bytes(num: float) -> str:
    """A compact human byte size, e.g. ``"3.2 MB"`` / ``"1.8 GB"``."""

    val = num
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"
