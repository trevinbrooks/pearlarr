"""Logging setup and the console surfaces: the rich handler, plain/json formats, and the level plumbing."""

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
    """A titled section header: a full-width rule, then the bold title line.

    `heavy=True` draws a heavy rule ("━", run boundaries); otherwise a light
    rule ("─", per-title headers). Both the rule and the title take `style`.
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

    Locks the producer (the `output.scan_lines` builders) and consumer
    (`render_kv`) to the same fields, so the two sides
    can't silently drift. `value` is a plain string or a pre-styled `Text`
    (the only non-str path, from `group_highlight`).
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
"""The typed console-look payload of a `output.scan_lines.LegacyLine`.

The hub's line builders (`output.scan_lines` / `output.wait_lines`) pair each
plain message with one of these; `render_legacy_lines` draws them through the
shared payload renderers below (`render_kv` / `render_rule` /
`print_titled_rule`). Lives here beside those renderers.
"""

# True while the output bridge is installed: the hub owns every raw record —
# WARNING+ in-context on the armed console seat (stderr fallback otherwise),
# DEBUG chatter at the renderer's frontier indent — so the rich handler stands
# down entirely, or the same record renders twice (once per safety net). With
# no bridge (standalone setup_logger) the legacy arms still render, so a
# record can never vanish.
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
    """Marker base for the output bridge (output/bridge.py).

    A distinct base so `setup_logger` can preserve the bridge across its
    per-cycle handler rebuilds without importing the output package (which
    imports this module back).
    """


def print_literal(console: Console, text: Text) -> None:
    """Print `text` as literal content: no markup/highlight, whole-line soft wrap.

    THE one home of the literal-print rule - bracketed content ("[1/182]",
    "[MARKED INCOMPLETE]") stays text instead of re-styling as markup, and the
    terminal owns any overflow (rich re-wrapping loses the indent).
    """

    console.print(text, highlight=False, soft_wrap=True)


def print_titled_rule(console: Console, title: str, style: str, *, heavy: bool) -> None:
    """A titled section header: a full-width rule, then the bold title line.

    Shared by `render_legacy_lines`'s `TitledRule` arm and the boot banner,
    so the two console looks can't drift.
    """

    console.print(Rule(style=style, characters="━" if heavy else "─"))
    print_literal(console, Text(title, style=f"{style} bold"))


def render_kv(kv: KvLine) -> Text:
    """Build a styled "key : value" (or gutter "key value") line from a kv payload.

    The leading "<indent><key><sep>" segment comes from the shared _kv_prefix
    helper, so this matches kv_string (the legacy-line message) exactly. Labels
    are a fixed dim grey50 so the value reads first. An optional `tail` (e.g.,
    a "(marked incomplete)" note) is appended console-side only. Consumed by
    `render_legacy_lines` (output/scan_lines.py), the hub's console arm.
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
    """A full-width separator: heavy ("━") for section ("=") breaks, light ("─") for sub ("-") breaks.

    The two weights keep the break levels distinguishable without color.
    """

    if "=" in char:
        return Rule(style="cyan", characters="━")
    return Rule(style="grey37", characters="─")


@dataclass(frozen=True, slots=True)
class Badge:
    """One severity's console badge: the glyph, its padded-word ASCII fallback, and their style."""

    glyph: str
    word: str
    style: str


# Level -> badge. INFO/DEBUG are deliberately absent, so they print without a
# prefix. The glyphs extend the cockpit ledgers' ✔/⚠/✖ language.
LEVEL_BADGES = {
    logging.WARNING: Badge("⚠", "WARNING", "yellow"),
    logging.ERROR: Badge("✖", "ERROR", "bold red"),
    logging.CRITICAL: Badge("‼", "CRITICAL", "bold white on red"),
}


def console_supports_unicode(console: Console) -> bool:
    """Whether the console can encode the glyphs/blocks the styled surfaces draw.

    Shared by the badge surfaces here and `console_caps.detect_capabilities`
    (which imports this module), so every glyph choice degrades on one probe.
    """

    if getattr(console, "legacy_windows", False):
        return False
    encoding = console.encoding or "utf-8"
    try:
        "✔━▏░⚠✖‼".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class RichConsoleHandler(logging.Handler):
    """The rich-TTY fallback surface for RAW first-party records — and the shared Console's home.

    While a bridge is installed (`_hub_owns_console`) this handler renders
    NOTHING: the logging bridge (output/bridge.py) adopts every record and the
    hub places it — WARNING+ on the armed console seat in-context (S5 pin 2)
    or its stderr fallback, DEBUG chatter at the renderer's frontier indent —
    so rendering here too would double them. With no bridge (library use, the
    pre-install window), everything below renders so a record can never
    vanish. Either way the handler stays attached: `console_of` resolves the
    cycle's shared Console from it (the hub seats print through it).

    Messages are rendered as literal text rather than `rich` markup, so
    bracketed content such as "[1/1]" or "[MARKED INCOMPLETE]" is never
    mistaken for a style tag.
    """

    # Cap how many stack frames a rendered traceback shows. An unexpected crash
    # is logged as a legible excerpt (rich keeps the outermost and innermost
    # frames and elides the middle when the stack is deeper than this) rather
    # than a full-screen wall of frames.
    MAX_TRACEBACK_FRAMES = 10

    def __init__(self, console: Console, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.console = console
        # Probed once; the console identity is fixed for this handler's lifetime.
        self._use_unicode = console_supports_unicode(console)

    def _print_line(self, record: logging.LogRecord, message: str) -> None:
        """A plain message: level badge for WARNING+.

        Emitted whole (soft_wrap) so the terminal handles any overflow, rather
        than rich re-wrapping with unindented continuation lines.
        """
        if LEVEL_BADGES.get(record.levelno) is None:
            line = Text(message)
        else:
            line = badge_line(record.levelno, message, use_unicode=self._use_unicode)

        print_literal(self.console, line)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Every record moved to hub placement: the bridge adopts it and the
            # hub renders it (badges in-context, DEBUG at the frontier indent),
            # so this handler stands down whenever a bridge is installed
            # (rendering here too doubled the record). No bridge: the legacy
            # arms below render.
            if _hub_owns_console:
                return

            message = record.getMessage()

            # An unexpected exception (logged with exc_info): show a colored
            # level badge + message, then a rich traceback with a capped frame
            # count so a crash is legible at a glance. Frame locals are never
            # rendered - they can hold config secrets (api keys, webhook URLs).
            # The bridge-adopted event still carries the full plain-text
            # traceback to the FileLogSink, so nothing is lost from the log file.
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
    """`message` behind its level badge — the ONE home of the badge column grammar.

    A unicode console gets the glyph badge ("⚠ ...", the cockpit ledgers'
    look); ASCII falls back to the padded word ("WARNING  ..."). Levels
    without a badge fall back to the ERROR one (only the exc_info arm, which
    always badges, can reach that).
    """

    badge = LEVEL_BADGES.get(levelno, LEVEL_BADGES[logging.ERROR])
    prefix = f"{badge.glyph} " if use_unicode else f"{badge.word:<8} "
    line = Text(prefix, style=badge.style)
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
    """The accepted log levels, as a choices-enum for the CLI's `--log-level`."""

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
    """The RICH-console threshold for a logger level; the text surfaces use the raw level.

    Applied by the handlers here and as the RichRenderer's diagnostic floor.
    The console always shows INFO+ so routine progress stays visible even when
    the file logger is raised - except DEBUG, which lowers the threshold, and
    CRITICAL, which raises it to match the file logger.
    """

    if level in (logging.DEBUG, logging.CRITICAL):
        return level
    return logging.INFO


def apply_log_level(logger: logging.Logger, log_level: str) -> None:
    """Re-point an already-built logger at `log_level`.

    The CLI bootstraps its logger before the config file can be read (config
    errors must be loggable), then calls this once the config's
    `advanced.log_level` is known. Unknown names fall back to INFO; the
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
LOG_NAME = "Pearlarr"
MAX_LOG_FILES = 9


def setup_logger(
    log_level: str,
    console_format: LogFormat = "auto",
) -> logging.Logger:
    """Configure the app logger: level, plus a rich console handler on a TTY.

    The hub's sinks own the file/plain/json surfaces (`output.textline`); the
    logging module is an INPUT channel (the bridge adopts records into events)
    with one exception: under "rich"/"auto"-on-a-TTY a `RichConsoleHandler`
    is attached as the shared Console's home (`console_of`). It renders
    nothing while a bridge is installed — the hub renders everything — and is
    the raw-record TTY fallback only without one. Under plain/json NO console
    handler is attached - level-only configuration; the bridge is the only
    handler.

    Args:
        log_level: Level name, case-insensitive; an unknown name falls back
            to INFO with a logged complaint.
        console_format: "rich" attaches the styled console handler;
            "plain"/"json" attach nothing (the hub's stdout seat renders).
            "auto" resolves here for programmatic callers - cli always
            passes a resolved value: rich when stdout is a TTY, plain otherwise.
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
        # The shared Console's home (console_of) and the no-bridge fallback
        # surface; under a bridge the handler stands down and the hub renders.
        console_handler = RichConsoleHandler(Console(file=sys.stdout))
        console_handler.setLevel(console_level(level))
        logger.addHandler(console_handler)

    # Only now can the invalid-level complaint reach the hub (and a rich console).
    # Deliberately raw — the sanctioned straggler the bridge adopts (this module
    # cannot lean on the hub it configures); allowlisted in tests/test_logging_ban.py.
    if invalid_log_level is not None:
        logger.critical(f"Invalid log level '{invalid_log_level}' - defaulting to 'INFO'")

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

    A `StrEnum` so each member still equals its rendered string
    (`EntryState.UNCHANGED == "unchanged"`) and flows through
    `entry_string`'s `.ljust(STATE_WIDTH)` byte-for-byte unchanged - while
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
    message (kv_string) never drift in prefix/padding/separator. `sep` is
    " :" for summary "key : value" lines and "" for the gutter "label value"
    entry-detail lines (see DETAIL_KEY_WIDTH).
    """

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
    """Indent a rendered console row by `level` levels, each `INDENT` wide.

    For renderer surfaces and interactive prompt rows — never log messages:
    the renderer owns a diagnostic's placement, so a producer-baked indent
    double-indents the rich seat and pollutes the text-seat message field.
    """

    return f"{INDENT * level}{text}"


def kv_string(
    key: str,
    value: str | Text,
    key_width: int,
    indent: int = 1,
    sep: str = " :",
) -> str:
    """Format an aligned "key : value" detail line for flat-style output.

    `key` is padded to `key_width` so the separators line up down a block;
    `indent` counts indent levels. Pass `sep=""` for the colon-less gutter
    "label value" entry-detail format.
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
    it gets the same accent the live log gives groups (`group_style`). When the
    group already leads the torrent name (bare, or in the usual "[Group]" wrapper,
    matched case-insensitively), that span is highlighted in place; otherwise the
    group is prepended in brackets so it always reads at the front - a match
    buried mid-name doesn't count. Returns a styled rich `Text` for the console;
    the file log sees its plain text via `str()` (so a prepended "[group] "
    still shows, just without color).

    With no group (or no name), the plain name is returned unchanged, so the
    caller's own `value_style` applies as before.

    Args:
        name: The torrent name as reported by the client / scraped from source
        group: The recommended SeaDex release group
        group_style: Rich style for the group span/prefix
        base_style: Rich style for the rest of the name
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
    """Pick the singular or plural form of a word based on a count; `plural` unset means `singular` + "s"."""

    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def count_noun(n: int, singular: str, plural: str | None = None) -> str:
    """Format a count with its correctly pluralized noun, e.g. "3 movies"; `plural` unset means `singular` + "s"."""

    return f"{n} {pluralize(n, singular, plural)}"


def arr_item_noun(arr: Arr, n: int) -> str:
    """Format a count with the arr's library noun: "3 movies" / "3 series"."""

    if arr is Arr.RADARR:
        return count_noun(n, "movie")
    return count_noun(n, "series", "series")


def format_elapsed(seconds: float) -> str:
    """Format an elapsed number of seconds as e.g. "8s", "14m 03s", or "1h 02m 03s"."""

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
