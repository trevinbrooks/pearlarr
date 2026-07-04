import logging
import os
import shutil
import sys
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from typing import Protocol, TypedDict, cast, override

from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.traceback import Traceback

from .config import Arr


class KvRecord(TypedDict):
    """Schema for the ``kv`` record carried on a LogRecord via ``extra=``.

    Locks the producer (``LogFormatter.kv``) and consumer
    (``RichConsoleHandler._render_kv``) to the same key names and value types,
    so the two sides can't silently drift. ``value`` is a plain string or a
    pre-styled ``Text`` (the only non-str path, from ``group_highlight``).
    """

    key: str
    value: str | Text
    value_style: str | None
    indent: int
    key_width: int
    sep: str
    tail: str | None
    tail_style: str


class RichConsoleHandler(logging.Handler):
    """Console log handler that renders records through ``rich``.

    Routine INFO/DEBUG lines print with no level prefix, so the output reads
    as clean text. Plain WARNING/ERROR messages get a colored level badge so
    problems stand out (aligned "key : value" lines never do - see ``kv``).

    Presentation is driven by ``extra=`` attributes on the record, so the
    plain message string (what the file log stores) stays clean while the
    console gets the rich treatment:

    * ``rule_title`` (+ optional ``rule_style``, ``rule_heavy``) -> a titled
      section: a full-width rule, then the title text LEFT-ALIGNED on the next
      line (both in ``rule_style``, default "cyan"; the title is bold). Pass
      ``rule_heavy=True`` for a heavy rule ("━", run boundaries); otherwise a
      light rule ("─", per-title headers) is drawn. Used for the run banner and
      per-title headers.
    * ``rule_char`` ("=" or "-") -> a full-width separator rule (heavy cyan for
      "=", light gray for "-"), distinguishable without color. Unmarked ASCII
      separators (a message of only "=" / "-") are still detected as a fallback.
    * ``kv`` (a dict with ``key`` / ``value`` and optional ``value_style`` /
      ``indent`` / ``key_width``) -> an aligned, lightly colored "key : value"
      detail line whose layout matches ``kv_string`` exactly. Labels are a fixed
      dim grey, so the value reads first; pass ``value_style`` to accent an
      outcome (e.g., green "added"). No level badge is drawn even for WARNING+ kv
      lines, so the value stays in its aligned column; LogCounter still tallies
      the severity for the run summary.
    * ``line_style`` -> a style applied to an otherwise plain message, used to
      dim no-op lines such as the collapsed "cached" one-liner.
    * ``tail`` (+ optional ``tail_style``) -> an emphasized suffix appended to
      the message, e.g., a "(marked incomplete)" note.

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

    @staticmethod
    def _render_kv(kv: KvRecord) -> Text:
        """Build a styled "key : value" (or gutter "key value") line from a kv record.

        The leading "<indent><key><sep>" segment comes from the shared _kv_prefix
        helper, so this matches kv_string (the file log) exactly. Labels are a
        fixed dim grey50 so the value reads first. An optional ``tail`` (e.g., a
        "(marked incomplete)" note) is appended console-side only, mirroring the
        plain-message tail.
        """
        prefix = _kv_prefix(
            kv["indent"],
            kv["key"],
            kv["key_width"],
            kv["sep"],
        )
        line = Text(prefix, style="grey50")
        value = kv["value"]
        if isinstance(value, Text):
            # A pre-styled value (e.g., a torrent name with its release group
            # highlighted) already carries its own spans, so append it as-is and
            # let those stand - value_style here would flatten them.
            if len(value):
                line.append(" ")
                line.append(value)
        elif value != "":
            line.append(" ")
            line.append(Text(value, style=kv["value_style"] or ""))
        tail = kv["tail"]
        if tail:
            line.append(" ")
            line.append(Text(tail, style=kv["tail_style"] or "yellow"))
        return line

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # A titled section: a full-width rule, then the title text
            # LEFT-ALIGNED on the next line (user directive). A heavy rule
            # ("━") marks run boundaries; a light rule ("─") marks per-title
            # headers (rule_heavy chooses).
            rule_title = getattr(record, "rule_title", None)
            if rule_title is not None:
                rule_style = getattr(record, "rule_style", "cyan")
                rule_heavy = getattr(record, "rule_heavy", False)
                self.console.print(
                    Rule(style=rule_style, characters="━" if rule_heavy else "─"),
                )
                self.console.print(
                    Text(rule_title, style=f"{rule_style} bold"),
                    highlight=False,
                    soft_wrap=True,
                )
                return

            message = record.getMessage()

            # An unexpected exception (logged with exc_info): show a colored
            # level badge + message, then a rich traceback with locals and a
            # capped frame count so a crash is legible at a glance. The file
            # handler still records the full plain-text traceback (the formatter
            # renders exc_info), so nothing is lost from the log file.
            if record.exc_info:
                label, style = self.LEVEL_BADGES.get(
                    record.levelno,
                    ("ERROR", "bold red"),
                )
                line = Text(f"{label:<8} ", style=style)
                line.append(message)
                self.console.print(line, highlight=False, soft_wrap=True)
                exc_type, exc_value, exc_tb = record.exc_info
                if exc_type is not None and exc_value is not None:
                    self.console.print(
                        Traceback.from_exception(
                            exc_type,
                            exc_value,
                            exc_tb,
                            show_locals=True,
                            max_frames=self.MAX_TRACEBACK_FRAMES,
                        ),
                    )
                return

            # A separator rule. The preferred form is an explicit "rule_char"
            # marker; we also fall back to detecting a hand-drawn ASCII rule (a
            # message of only "=" or "-"). A heavy line marks section ("=")
            # breaks, and a light line marks sub ("-") breaks, so the two stay
            # distinguishable even without color (piped to a file/Docker logs).
            rule_char = getattr(record, "rule_char", None)
            if rule_char is None:
                stripped = message.strip()
                if stripped and set(stripped) <= {"=", "-"}:
                    rule_char = "=" if "=" in stripped else "-"
            if rule_char is not None:
                if "=" in rule_char:
                    self.console.print(Rule(style="cyan", characters="━"))
                else:
                    self.console.print(Rule(style="grey37", characters="─"))
                return

            # A styled key/value detail line (deliberately no level badge even
            # for WARNING+ - see below; LogCounter still tallies severity).
            kv: KvRecord | None = getattr(record, "kv", None)
            if kv is not None:
                # No level badge here, even for WARNING+ kv lines: a col-0 badge
                # would push the value past its aligned column and detach the
                # line from the entry block it belongs under. Severity is still
                # counted by LogCounter (a logger filter) and surfaced in the
                # run summary's "issues" tally; here, position and value_style
                # carry the meaning.
                self.console.print(
                    self._render_kv(kv),
                    highlight=False,
                    soft_wrap=True,
                )
                return

            # Emit messages whole (soft_wrap) and let the terminal handle any
            # overflow, rather than having rich re-wrap with unindented
            # continuation lines, as the previous logger did.
            badge = self.LEVEL_BADGES.get(record.levelno)
            if badge is None:
                line = Text(message, style=getattr(record, "line_style", "") or "")
            else:
                label, style = badge
                line = Text(f"{label:<8} ", style=style)
                line.append(message)

            # An optional emphasized suffix (e.g., an "incomplete" note)
            tail = getattr(record, "tail", None)
            if tail is not None:
                line.append(" ")
                line.append(
                    Text(str(tail), style=getattr(record, "tail_style", "yellow")),
                )

            self.console.print(line, highlight=False, soft_wrap=True)
        except Exception:
            self.handleError(record)


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
        self.counts[record.levelno] = self.counts.get(record.levelno, 0) + 1
        return True

    def snapshot(self) -> dict[int, int]:
        """Return a copy of the current per-level counts"""
        return dict(self.counts)


class _CountingLogger(Protocol):
    """Structural view of a logger that carries a per-run :class:`LogCounter`.

    ``setup_logger`` attaches ``seadex_counter`` dynamically; narrowing the
    logger to this protocol lets the assignment type-check without a cast or a
    Logger subclass, mirroring the ``getattr(..., "seadex_counter", None)`` reads
    at the call sites.
    """

    seadex_counter: LogCounter


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


def setup_logger(
    log_level: str,
    log_dir: str,
    log_name: str = "SeaDexArr",
    max_logs: int = 9,
) -> logging.Logger:
    """
    Set up the logger.

    Parameters:
        log_level (str): The log level to use
        log_dir (str): Full path to the directory for log files (resolved by the
            caller via ``paths.log_dir``; created here if missing).
        log_name (str): The name of the log file.
            Defaults to "SeaDexArr"
        max_logs (int): Maximum number of log files to keep.
            Defaults to 9

    Returns:
        A logger object for logging messages.
    """

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{log_name}.log")

    # Check if a log file already exists. Copy, then remove to avoid I/O errors
    if os.path.isfile(log_file):
        for i in range(max_logs - 1, 0, -1):
            old_log = os.path.join(f"{log_dir}", f"{log_name}.log.{i}")
            new_log = os.path.join(f"{log_dir}", f"{log_name}.log.{i + 1}")
            if os.path.exists(old_log):
                if os.path.exists(new_log):
                    os.remove(new_log)
                shutil.copy(old_log, new_log)
                os.remove(old_log)

        shutil.copy(log_file, os.path.join(log_dir, f"{log_name}.log.1"))
        os.remove(log_file)

    logger = logging.getLogger(log_name)
    logger.propagate = False

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

    # Define the log message format for the log files
    logfile_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%y %I:%M %p",
    )

    # Create the file handler. Rotation is performed manually above (the
    # copy/remove of prior logs), once per setup_logger() call; the handler
    # opens a fresh file each run (mode="w"). maxBytes is intentionally unset,
    # so size-based rollover never fires - hence no backupCount here.
    handler = RotatingFileHandler(
        log_file,
        delay=True,
        mode="w",
        encoding="utf-8",
    )
    handler.setFormatter(logfile_formatter)

    # Configure console logging through the rich. Routine lines print with no
    # level prefix; warnings and errors get a colored badge (see
    # RichConsoleHandler). rich auto-detects non-interactive output (pipes,
    # Docker logs) and drops ANSI styling there.
    console = Console(file=sys.stdout)
    console_handler = RichConsoleHandler(console)
    # The console always shows INFO+ so routine progress stays visible even when
    # the file logger is raised - except DEBUG, which lowers the threshold, and
    # CRITICAL, which raises it to match the file logger. (The original
    # DEBUG/CRITICAL/else split, preserved exactly.)
    if level == logging.DEBUG:
        console_handler.setLevel(logging.DEBUG)
    elif level == logging.CRITICAL:
        console_handler.setLevel(logging.CRITICAL)
    else:
        console_handler.setLevel(logging.INFO)

    # Replace any handlers from a previous call, then attach file + console
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Only now can the invalid-level complaint reach the console + file log.
    if invalid_log_level is not None:
        logger.critical(f"Invalid log level '{invalid_log_level}', defaulting to 'INFO'")

    # Tally records by level so a run can report its warning/error counts.
    # Replace any counter from a previous call so counts don't carry over.
    for existing_filter in list(logger.filters):
        if isinstance(existing_filter, LogCounter):
            logger.removeFilter(existing_filter)
    counter = LogCounter()
    logger.addFilter(counter)
    cast(_CountingLogger, logger).seadex_counter = counter

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


class LogFormatter:
    """Render aligned detail lines through a logger.

    Holds only the logger and the rule width - no run state - so the
    presentation primitives (an aligned "key : value" line, a blank separator,
    an elapsed-time string) live apart from the reporting layer that decides
    *what* to report. The semantic ``log_*`` methods on ``RunReporter`` keep the
    run state (via ``RunContext``) and delegate their rendering here.

    Args:
        logger (logging.Logger): Logger every line is emitted through
        line_length (int): Full width used for separator rules. Defaults to 80
    """

    def __init__(self, logger: logging.Logger, line_length: int = 80) -> None:
        self.logger = logger
        self.line_length = line_length

    def kv(
        self,
        key: str,
        value: str | Text,
        value_style: str | None = None,
        level: int = logging.INFO,
        indent: int = 1,
        *,
        key_width: int,
        sep: str = " :",
        tail: str | None = None,
        tail_style: str = "yellow",
    ) -> bool:
        """Log an aligned "key : value" (or gutter "key value") detail line

        The file log stores the plain kv_string text; on the console the label
        is dimmed so the value reads first, and an optional value_style accents
        the outcome (e.g., green for "added").

        Args:
            key: Left-hand label
            value: Right-hand value
            value_style: Optional rich style for the value (e.g. "green")
            level: Logging level. Defaults to logging.INFO
            indent: Number of indent levels. Defaults to 1
            key_width: Column width the key is padded to
            sep: Separator after the padded key. Defaults to ":"; pass "" for
                the colon-less gutter format (see detail)
            tail: Optional emphasized suffix (console only), e.g., an "incomplete"
                note. Defaults to None
            tail_style: Style for the tail. Defaults to "yellow"
        """

        record: KvRecord = {
            "key": key,
            "value": value,
            "value_style": value_style,
            "indent": indent,
            "key_width": key_width,
            "sep": sep,
            "tail": tail,
            "tail_style": tail_style,
        }
        self.logger.log(
            level,
            kv_string(key, value, key_width=key_width, indent=indent, sep=sep),
            extra={"kv": record},
        )

        return True

    def detail(
        self,
        label: str,
        value: str | Text,
        value_style: str | None = None,
        level: int = logging.INFO,
        tail: str | None = None,
        tail_style: str = "yellow",
    ) -> bool:
        """Log an entry-detail line: dim gutter label, value at the title column

        The colon-less "<label> <value>" form is used for everything indented under
        an entry (files / link / status / group / added / kept / missing /
        skipped / anilist). The value lands in the same column as the entry title,
        so the whole block reads as one aligned column; the label sits dimmed in
        the indent gutter and the value carries any accent color.

        Args:
            label: Gutter label, e.g. "files" or "added"
            value: The value text
            value_style: Optional rich style for the value (e.g. "green")
            level: Logging level. Defaults to logging.INFO
            tail: Optional emphasized suffix (console only). Defaults to None
            tail_style: Style for the tail. Defaults to "yellow"
        """

        return self.kv(
            label,
            value,
            value_style=value_style,
            level=level,
            indent=DETAIL_INDENT,
            key_width=DETAIL_KEY_WIDTH,
            sep="",
            tail=tail,
            tail_style=tail_style,
        )

    def blank(self) -> bool:
        """Emit a blank line to visually separate entries / item blocks"""

        self.logger.info("")
        return True

    @staticmethod
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
