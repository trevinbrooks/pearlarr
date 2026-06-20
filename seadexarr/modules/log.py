import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.traceback import Traceback


class RichConsoleHandler(logging.Handler):
    """Console log handler that renders records through ``rich``.

    Routine INFO/DEBUG lines print with no level prefix, so the output reads
    as clean text. Plain WARNING/ERROR messages get a coloured level badge so
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
      "=", light grey for "-"), distinguishable without colour. Unmarked ASCII
      separators (a message of only "=" / "-") are still detected as a fallback.
    * ``kv`` (a dict with ``key`` / ``value`` and optional ``value_style`` /
      ``indent`` / ``key_width``) -> an aligned, lightly coloured "key : value"
      detail line whose layout matches ``kv_string`` exactly. Labels are a fixed
      dim grey so the value reads first; pass ``value_style`` to accent an
      outcome (e.g. green "added"). No level badge is drawn even for WARNING+ kv
      lines, so the value stays in its aligned column; LogCounter still tallies
      the severity for the run summary.
    * ``line_style`` -> a style applied to an otherwise plain message, used to
      dim no-op lines such as the collapsed "cached" one-liner.
    * ``tail`` (+ optional ``tail_style``) -> an emphasised suffix appended to
      the message, e.g. a "(marked incomplete)" note.

    Messages are rendered as literal text rather than ``rich`` markup, so
    bracketed content such as "[1/1]" or "[MARKED INCOMPLETE]" is never
    mistaken for a style tag.
    """

    # Level -> (badge label, rich style). INFO/DEBUG are deliberately absent
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

    def __init__(self, console, level=logging.NOTSET):
        super().__init__(level=level)
        self.console = console

    @staticmethod
    def _render_kv(kv):
        """Build a styled "key : value" (or gutter "key value") line from a kv dict.

        The leading "<indent><key><sep>" segment comes from the shared _kv_prefix
        helper, so this matches kv_string (the file log) exactly. Labels are a
        fixed dim grey50 so the value reads first. An optional ``tail`` (e.g. a
        "(marked incomplete)" note) is appended console-side only, mirroring the
        plain-message tail.
        """
        prefix = _kv_prefix(
            kv.get("indent", 1),
            kv.get("key", ""),
            kv.get("key_width", KEY_WIDTH),
            kv.get("sep", " :"),
        )
        line = Text(prefix, style="grey50")
        value = kv.get("value", "")
        if value != "":
            line.append(" ")
            line.append(Text(str(value), style=kv.get("value_style") or ""))
        tail = kv.get("tail")
        if tail:
            line.append(" ")
            line.append(Text(str(tail), style=kv.get("tail_style") or "yellow"))
        return line

    def emit(self, record):
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

            # An unexpected exception (logged with exc_info): show a coloured
            # level badge + message, then a rich traceback with locals and a
            # capped frame count so a crash is legible at a glance. The file
            # handler still records the full plain-text traceback (the formatter
            # renders exc_info), so nothing is lost from the log file.
            if record.exc_info:
                label, style = self.LEVEL_BADGES.get(
                    record.levelno, ("ERROR", "bold red"),
                )
                line = Text(f"{label:<8} ", style=style)
                line.append(message)
                self.console.print(line, highlight=False, soft_wrap=True)
                self.console.print(
                    Traceback.from_exception(
                        *record.exc_info,
                        show_locals=True,
                        max_frames=self.MAX_TRACEBACK_FRAMES,
                    ),
                )
                return

            # A separator rule. The preferred form is an explicit ``rule_char``
            # marker; we also fall back to detecting a hand-drawn ASCII rule (a
            # message of only "=" or "-"). A heavy line marks section ("=")
            # breaks and a light line marks sub ("-") breaks, so the two stay
            # distinguishable even without colour (piped to a file/Docker logs).
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

            # A styled key/value detail line. All current callers log these at
            # INFO, but if one is WARNING+ prepend the level badge so severity
            # is not silently lost (defensive).
            kv = getattr(record, "kv", None)
            if kv is not None:
                # No level badge here, even for WARNING+ kv lines: a col-0 badge
                # would push the value past its aligned column and detach the
                # line from the entry block it belongs under. Severity is still
                # counted by LogCounter (a logger filter) and surfaced in the
                # run summary's "issues" tally; here, position and value_style
                # carry the meaning.
                self.console.print(
                    self._render_kv(kv), highlight=False, soft_wrap=True,
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

            # An optional emphasised suffix (e.g. an "incomplete" note)
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

    def __init__(self):
        super().__init__()
        self.counts = {}

    def filter(self, record):
        self.counts[record.levelno] = self.counts.get(record.levelno, 0) + 1
        return True

    def snapshot(self):
        """Return a copy of the current per-level counts"""
        return dict(self.counts)


def setup_logger(
    log_level,
    log_dir="logs",
    log_name="SeaDexArr",
    max_logs=9,
):
    """
    Set up the logger.

    Parameters:
        log_level (str): The log level to use
        log_dir (str): Directory for log files.
            Defaults to "logs"
        log_name (str): The name of the log file.
            Defaults to "SeaDexArr"
        max_logs (int): Maximum number of log files to keep.
            Defaults to 9

    Returns:
        A logger object for logging messages.
    """

    if os.environ.get("DOCKER_ENV"):
        config_dir = os.environ.get("CONFIG_DIR")
        log_dir = os.path.join(config_dir, log_dir)
    else:
        log_dir = os.path.join(os.getcwd(), log_dir)

    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define the log file path
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

    # Create a logger object with the script name
    logger = logging.getLogger(log_name)
    logger.propagate = False

    # Set the log level based on the provided parameter
    log_level = log_level.upper()
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif log_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.critical(f"Invalid log level '{log_level}', defaulting to 'INFO'")
        logger.setLevel(logging.INFO)

    # Define the log message format for the log files
    logfile_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%m/%d/%y %I:%M %p",
    )

    # Create the file handler. Rotation is performed manually above (the
    # copy/remove of prior logs), once per setup_logger() call; the handler
    # opens a fresh file each run (mode="w"). maxBytes is intentionally unset,
    # so size-based rollover never fires - hence no backupCount here.
    handler = RotatingFileHandler(
        log_file, delay=True, mode="w", encoding="utf-8",
    )
    handler.setFormatter(logfile_formatter)

    # Configure console logging through rich. Routine lines print with no
    # level prefix; warnings and errors get a coloured badge (see
    # RichConsoleHandler). rich auto-detects non-interactive output (pipes,
    # Docker logs) and drops ANSI styling there.
    console = Console(file=sys.stdout)
    console_handler = RichConsoleHandler(console)
    if log_level == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    elif log_level == "CRITICAL":
        console_handler.setLevel(logging.CRITICAL)
    else:
        console_handler.setLevel(logging.INFO)

    # Replace any handlers from a previous call, then attach file + console
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Tally records by level so a run can report its warning/error counts.
    # Replace any counter from a previous call so counts don't carry over.
    for existing_filter in list(logger.filters):
        if isinstance(existing_filter, LogCounter):
            logger.removeFilter(existing_filter)
    counter = LogCounter()
    logger.addFilter(counter)
    # pyrefly: ignore [missing-attribute]
    logger.seadex_counter = counter

    return logger


# Number of spaces each level of the flat layout is indented by
INDENT = "  "

# Default key-column width for "key : value" detail lines, with comfortable room
# for the longest key in use ("already have"), so the colons line up.
KEY_WIDTH = 16

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
# title column so the two can't drift.
DETAIL_INDENT = 2
DETAIL_KEY_WIDTH = (
    (len(INDENT) + ENTRY_LABEL_OFFSET) - (DETAIL_INDENT * len(INDENT)) - 1
)


def entry_string(state, label):
    """Format the body of an entry-ledger line: "<state> <label>".

    state is padded to STATE_WIDTH so the label lines up across rows regardless
    of state-word length. No indent is applied here; the caller wraps this with
    indent_string(level=1). Season/episode/URL detail is carried on a separate
    continuation line (see log_entry_coverage), not here.
    """

    return f"{state.ljust(STATE_WIDTH)} {label}"


def _kv_prefix(indent, key, key_width, sep=" :"):
    """Build the shared "<indent><key><sep>" leading segment for a kv line.

    Single source of truth so the console handler (_render_kv) and the file
    formatter (kv_string) never drift in prefix/padding/separator. ``sep`` is
    " :" for summary "key : value" lines and "" for the gutter "label value"
    entry-detail lines (see DETAIL_KEY_WIDTH).
    """

    return f"{INDENT * indent}{key.ljust(key_width)}{sep}"


def rule_string(
    rule_char="-",
    total_length=80,
    str_prefix="",
):
    """Draw a full-width separator rule for the (flat-style) logger

    Args:
        rule_char: Character to repeat across the rule. Defaults to "-"
        total_length: Width of the rule. Defaults to 80
        str_prefix: Will include this at the start of the string. Defaults to ""
    """

    return f"{str_prefix}{rule_char * total_length}"


def indent_string(
    text,
    level=1,
    str_prefix="",
):
    """Format an indented detail line for the (flat-style) logger

    Args:
        text: String to format
        level: Number of indent levels (each INDENT wide). Defaults to 1
        str_prefix: Will include this at the start of any string. Defaults to ""
    """

    return f"{str_prefix}{INDENT * level}{text}"


# Deprecated aliases kept during the migration to indent_string. Both
# historically accepted (and ignored) a total_length argument; neither ever
# centred anything. Prefer indent_string in new code.
def centred_string(str_to_centre, str_prefix=""):
    return indent_string(str_to_centre, str_prefix=str_prefix)


def left_aligned_string(str_to_align, str_prefix=""):
    return indent_string(str_to_align, str_prefix=str_prefix)


def kv_string(
    key,
    value,
    key_width=KEY_WIDTH,
    indent=1,
    str_prefix="",
    sep=" :",
):
    """Format an aligned "key : value" detail line for flat-style output

    Args:
        key: Left-hand label
        value: Right-hand value
        key_width: Column width the key is padded to, so the colons line up.
            Defaults to KEY_WIDTH (16)
        indent: Number of indent levels to prefix. Defaults to 1
        str_prefix: Will include this at the start of any string. Defaults to ""
        sep: Separator after the padded key. Defaults to " :"; pass "" for the
            colon-less gutter "label value" entry-detail format
    """

    # Built from the shared _kv_prefix helper so the file log matches the
    # console handler (_render_kv) exactly.
    line = f"{str_prefix}{_kv_prefix(indent, key, key_width, sep)}"

    # Allow an empty value to act as a header for an indented block below it
    if value == "":
        return line

    return f"{line} {value}"


def pluralize(n, singular, plural=None):
    """Pick the singular or plural form of a word based on a count

    Args:
        n: The count
        singular: The singular form, used when n == 1
        plural: The plural form. Defaults to None, i.e. singular + "s"
    """

    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def count_noun(n, singular, plural=None):
    """Format a count with its correctly pluralised noun, e.g. "3 movies"

    Args:
        n: The count
        singular: The singular noun
        plural: The plural noun. Defaults to None, i.e. singular + "s"
    """

    return f"{n} {pluralize(n, singular, plural)}"
