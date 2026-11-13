"""The console's line types and text formatters: rules, key-value lines, title row columns, counts and durations."""

from dataclasses import dataclass

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from .events import TitleState
from ..config import Arr
from ..log import print_literal


@dataclass(frozen=True, slots=True)
class TitledRule:
    """The payload for a titled block header."""

    title: str
    style: str = "bold cyan"
    """Rich style applied to both the rule and the title."""
    heavy: bool = False
    """`True` draws a heavy rule ("━"), `False` a light one ("─")."""


@dataclass(frozen=True, slots=True)
class SeparatorRule:
    """The payload for a full-width separator rule."""

    char: str = "-"


@dataclass(frozen=True, slots=True)
class KeyValue:
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


type ConsoleRender = TitledRule | SeparatorRule | KeyValue | StyledLine


def print_titled_rule(console: Console, title: str, style: str, *, heavy: bool) -> None:
    """A titled block header: a full-width rule, then the bold title line."""

    console.print(Rule(style=style, characters="━" if heavy else "─"))
    print_literal(console, Text(title, style=f"{style} bold"))


def kv_text(kv: KeyValue) -> Text:
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


def rule_text(char: str) -> Rule:
    """A full-width separator: heavy ("━") for top-level ("=") breaks, light ("─") for sub ("-") breaks."""

    if char == "=":
        return Rule(style="cyan", characters="━")
    return Rule(style="grey37", characters="─")


INDENT = "  "

# The title row state column, padded so labels align. Widest state word is "unmonitored" (11).
STATE_WIDTH = 11

# Column the title starts at, measured from the end of its indent prefix.
TITLE_LABEL_OFFSET = STATE_WIDTH + 1

# Detail-line keys sit in the indent gutter so the value lands at the title column, less the space kv adds.
DETAIL_INDENT = 2
DETAIL_KEY_WIDTH = (len(INDENT) + TITLE_LABEL_OFFSET) - (DETAIL_INDENT * len(INDENT)) - 1


def title_row_string(state: TitleState, label: str) -> str:
    """Format a title row body: "<state> <label>", state padded to STATE_WIDTH."""

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

    # Shared `_kv_prefix`, so the plain message matches the console render (kv_text) exactly.
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


def long_duration(seconds: float) -> str:
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
