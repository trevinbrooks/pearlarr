import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.rule import Rule
from rich.text import Text


class RichConsoleHandler(logging.Handler):
    """Console log handler that renders records through ``rich``.

    Routine INFO/DEBUG lines print with no level prefix, so the output reads
    as clean text. WARNING and above get a coloured level badge so problems
    stand out. Records whose message is a full-width separator (only "=" or
    "-" characters) are drawn as a ``rich`` rule, and records carrying a
    ``rule_title`` attribute (passed via ``logger.info(..., extra=...)``) are
    drawn as a titled rule.

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

    def __init__(self, console, level=logging.NOTSET):
        super().__init__(level=level)
        self.console = console

    def emit(self, record):
        try:
            message = record.getMessage()

            # A titled rule, e.g. the run banner or a per-title header
            rule_title = getattr(record, "rule_title", None)
            if rule_title is not None:
                style = getattr(record, "rule_style", "cyan")
                self.console.print(Rule(rule_title, style=style))
                return

            # A hand-drawn separator -> draw a crisp rich rule instead. A heavy
            # line marks section ("=") breaks and a light line marks sub ("-")
            # breaks, so the two stay distinguishable even without colour (e.g.
            # piped to a file or Docker logs).
            stripped = message.strip()
            if stripped and set(stripped) <= {"=", "-"}:
                if "=" in stripped:
                    self.console.print(Rule(style="cyan", characters="━"))
                else:
                    self.console.print(Rule(style="grey37", characters="─"))
                return

            # Emit messages whole (soft_wrap) and let the terminal handle any
            # overflow, rather than having rich re-wrap with unindented
            # continuation lines, as the previous logger did.
            badge = self.LEVEL_BADGES.get(record.levelno)
            if badge is None:
                self.console.print(Text(message), highlight=False, soft_wrap=True)
            else:
                label, style = badge
                line = Text(f"{label:<8} ", style=style)
                line.append(message)
                self.console.print(line, highlight=False, soft_wrap=True)
        except Exception:
            self.handleError(record)


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
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%m/%d/%y %I:%M %p"
    )

    # Create a RotatingFileHandler for log files
    handler = RotatingFileHandler(
        log_file, delay=True, mode="w", encoding="utf-8", backupCount=max_logs
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

    return logger


# Number of spaces each level of the flat layout is indented by
INDENT = "  "


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


def centred_string(
    str_to_centre,
    total_length=80,
    str_prefix="",
):
    """Format a string for the (flat-style) logger

    Despite the historical name, this no longer draws a bordered box; text is
    emitted as a single indented line. Long lines are left to run on rather than
    being padded to a fixed width, so they can never break the layout. To draw a
    separator rule, use rule_string instead.

    Args:
        str_to_centre: String to format
        total_length: Unused, kept for call-site compatibility
        str_prefix: Will include this at the start of any string. Defaults to ""
    """

    return f"{str_prefix}{INDENT}{str_to_centre}"


def left_aligned_string(
    str_to_align,
    total_length=80,
    str_prefix="",
):
    """Format an indented detail line for the (flat-style) logger

    Args:
        str_to_align: String to format
        total_length: Unused, kept for call-site compatibility
        str_prefix: Will include this at the start of any string. Defaults to ""
    """

    return f"{str_prefix}{INDENT}{str_to_align}"


def kv_string(
    key,
    value,
    key_width=16,
    indent=1,
    str_prefix="",
):
    """Format an aligned "key : value" detail line for flat-style output

    Args:
        key: Left-hand label
        value: Right-hand value
        key_width: Column width the key is padded to, so the colons line up.
            Defaults to 16
        indent: Number of indent levels to prefix. Defaults to 1
        str_prefix: Will include this at the start of any string. Defaults to ""
    """

    line = f"{str_prefix}{INDENT * indent}{key.ljust(key_width)} :"

    # Allow an empty value to act as a header for an indented block below it
    if value == "":
        return line

    return f"{line} {value}"
