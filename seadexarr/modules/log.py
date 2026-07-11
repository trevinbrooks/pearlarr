import logging
import sys
from collections.abc import Callable
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
"""The typed console-look payload of a :class:`~.output.scan_lines.LegacyLine`.

The hub's line builders (``output.scan_lines`` / ``output.wait_lines``) pair each
plain message with one of these; ``render_legacy_lines`` draws them through the
shared payload renderers below (``render_kv`` / ``render_rule`` /
``print_titled_rule``). Lives here beside those renderers.
"""

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


def print_titled_rule(console: Console, title: str, style: str, *, heavy: bool) -> None:
    """A titled section header: a full-width rule, then the bold title line.

    Shared by ``render_legacy_lines``'s ``TitledRule`` arm and the boot banner,
    so the two console looks can't drift.
    """

    console.print(Rule(style=style, characters="━" if heavy else "─"))
    console.print(Text(title, style=f"{style} bold"), highlight=False, soft_wrap=True)


def render_kv(kv: KvLine) -> Text:
    """Build a styled "key : value" (or gutter "key value") line from a kv payload.

    The leading "<indent><key><sep>" segment comes from the shared _kv_prefix
    helper, so this matches kv_string (the legacy-line message) exactly. Labels
    are a fixed dim grey50 so the value reads first. An optional ``tail`` (e.g.,
    a "(marked incomplete)" note) is appended console-side only. Consumed by
    ``render_legacy_lines`` (output/scan_lines.py), the hub's console arm.
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
    """The rich-TTY surface for RAW first-party records only.

    Post-flip the hub's seats own every structured surface; what still arrives
    here is DEBUG chatter, unmigrated INFO one-liners, the WARNING+ fallback,
    and exc_info tracebacks. Routine INFO/DEBUG lines print with no level
    prefix, so the output reads as clean text. PLAIN records at WARNING+ (the
    badge class, including exc_info records) are NOT rendered here while the
    registered console owner (the hub, via ``install_bridge``) answers True:
    the logging bridge (output/bridge.py) adopts them and the hub's rich
    renderer places them in-context (S5 pin 2) - rendering here too would
    double them. With no owner, or a struck-out console seat, the legacy badge
    renders so a warning can never vanish.

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

    def _print_line(self, record: logging.LogRecord, message: str) -> None:
        """A plain message: level badge for WARNING+.

        Emitted whole (soft_wrap) so the terminal handles any overflow, rather
        than rich re-wrapping with unindented continuation lines.
        """
        if self.LEVEL_BADGES.get(record.levelno) is None:
            line = Text(message)
        else:
            line = badge_line(record.levelno, message)

        self.console.print(line, highlight=False, soft_wrap=True)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # The badge class moved to hub placement (S5 pin 2): the bridge
            # adopts WARNING+ records, the hub's renderer draws the badge — but
            # only while the hub actually owns the console; with no bridge or
            # a struck-out console seat, the legacy badge below still renders.
            if record.levelno >= logging.WARNING and _console_owner is not None and _console_owner():
                return

            message = record.getMessage()

            # An unexpected exception (logged with exc_info): show a colored
            # level badge + message, then a rich traceback with a capped frame
            # count so a crash is legible at a glance. Frame locals are never
            # rendered - they can hold config secrets (api keys, webhook URLs).
            # The bridge-adopted event still carries the full plain-text
            # traceback to the FileLogSink, so nothing is lost from the log file.
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

            self._print_line(record, message)
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


def resolve_console_format(console_format: LogFormat) -> LogFormat:
    """Fold "auto" to the tty-detected concrete format — the ONE fold home."""

    if console_format == "auto":
        return "rich" if sys.stdout.isatty() else "plain"
    return console_format


def console_level(level: int) -> int:
    """The RICH-console threshold for a logger level (the handlers here and the
    RichRenderer's diagnostic floor); the text surfaces use the raw level (S4).

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
    # Keep root in step (see setup_logger): the config level lands here mid-cycle,
    # and the root-seated bridge must see sub-WARNING third-party records.
    logging.getLogger().setLevel(level)
    for handler in logger.handlers:
        # The rich console handler setup_logger installed (plain/json attach
        # no console handler at all); never the bridge.
        if isinstance(handler, RichConsoleHandler):
            handler.setLevel(console_level(level))

    # Forward to the output hub (S4: each surface applies its own floor).
    # Imported lazily: the output package imports this module at load.
    from .output.runtime import current_hub

    current_hub().set_level(level)


# Logger name; MAX_LOG_FILES caps the FileLogSink's rotation cascade
# (.log -> .log.1 ... .log.9 in output/textline.py).
LOG_NAME = "SeaDexArr"
MAX_LOG_FILES = 9


def setup_logger(
    log_level: str,
    console_format: LogFormat = "auto",
) -> logging.Logger:
    """Configure the app logger: level, plus a rich console handler on a TTY.

    The hub's sinks own the file/plain/json surfaces (``output.textline``); the
    logging module is an INPUT channel (the bridge adopts records into events)
    with one exception: under "rich"/"auto"-on-a-TTY a :class:`RichConsoleHandler`
    is attached as the shared-Console owner and the TTY surface for raw
    first-party records. Under plain/json NO console handler is attached -
    level-only configuration; the bridge is the only handler.

    Parameters:
        log_level (str): The log level to use
        console_format (LogFormat): "rich" attaches the styled console handler;
            "plain"/"json" attach nothing (the hub's stdout seat renders).
            "auto" (default) resolves here for programmatic callers - cli always
            passes a resolved value: rich when stdout is a TTY, plain otherwise.

    Returns:
        A logger object for logging messages.
    """

    logger = logging.getLogger(LOG_NAME)
    logger.propagate = False

    # Close and detach any handlers from a previous call FIRST (scheduled mode
    # re-runs this each cycle): an unclosed handler leaks its descriptor. The
    # output bridge is installed once per process (cli) and must survive.
    for old_handler in list(logger.handlers):
        if isinstance(old_handler, HubBridgeBase):
            continue
        logger.removeHandler(old_handler)
        old_handler.close()

    # Resolve the configured level once through the name->constant table. Only
    # the five standard names are accepted; anything else (a typo) falls back to
    # INFO - the complaint is emitted below, AFTER the handler attach, so under
    # plain/json the bridge (the only handler) carries it to the hub and
    # logging.lastResort can never fire.
    level = _LOG_LEVELS.get(log_level.upper())
    invalid_log_level = log_level if level is None else None
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    # The bridge lives on the ROOT logger: open root's level too, so the bridge's
    # own gate (not stdlib's WARNING default) decides third-party records.
    logging.getLogger().setLevel(level)

    # Defensive fold for programmatic callers; cli resolves before calling.
    console_format = resolve_console_format(console_format)

    if console_format == "rich":
        # Console logging through rich: routine lines print with no level
        # prefix; warnings and errors get a colored badge (RichConsoleHandler).
        console_handler = RichConsoleHandler(Console(file=sys.stdout))
        console_handler.setLevel(console_level(level))
        logger.addHandler(console_handler)

    # Only now can the invalid-level complaint reach the hub (and a rich console).
    if invalid_log_level is not None:
        logger.critical(f"Invalid log level '{invalid_log_level}', defaulting to 'INFO'")

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

    Single source of truth so the console render (render_kv) and the plain
    message (kv_string) never drift in prefix/padding/separator. ``sep`` is
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

    # Built from the shared _kv_prefix helper, so the plain message matches the
    # console render (render_kv) exactly.
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
