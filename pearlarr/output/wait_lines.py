"""Pure builders and reducers for the wait pass: the rich-free live-frame model and the ledger lines."""

from __future__ import annotations

import logging
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from .console_lines import ConsoleLine
from .console_text import (
    DETAIL_INDENT,
    DETAIL_KEY_WIDTH,
    STATE_WIDTH,
    KeyValue,
    SeparatorRule,
    StyledLine,
    compact_duration,
    count_noun,
    human_bytes,
    indent_string,
    long_duration,
    rule_string,
)
from .events import (
    Phase,
    RowFinished,
    TorrentView,
    WaitFinished,
    WaitKind,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    clamp01,
    severity_of,
)
from .scan_lines import kv_line
from ..coverage import coverage_string
from ..releases import EpisodeKey, ReleaseEpisode
from ..tvdb_orders import order_display_name
from ..wait_types import LeftForHandImport, Outcome, OutcomeKind

if TYPE_CHECKING:
    from ..console_caps import ConsoleFeatures

type WaitEvent = WaitStarted | WaitProgress | RowFinished | WaitFinished

# The live cockpit never grows past this many in-flight rows, the rest collapse into a one-line
# "+ N more ..." overflow so a large carried-over backlog can't blow the region past the screen.
MAX_LIVE_ROWS = 12
MIN_LIVE_ROWS = 4
# Banner, header, overflow line and breathing room, when clamping rows to the terminal height.
_RESERVED_ROWS = 8
# The sparkline needs unicode blocks and enough width not to crowd the label.
MIN_SPARK_WIDTH = 80
# The console names this many of a download's held-back files. The log and the notification name them all.
_MAX_FILE_ROWS = 5
# The console wraps an order-only record's reason at this many columns, so its rows fit a 100-column terminal.
_REASON_WIDTH = 78
# A ledger row's label starts two columns past an entry row's, behind the glyph and its space.
_ROW_KEY_WIDTH = DETAIL_KEY_WIDTH + 2


def _episodes(keys: tuple[EpisodeKey, ...]) -> str:
    """Episode keys in a row label's notation, as "S04 E01, E05-E08"."""

    return coverage_string([ReleaseEpisode(season=key.season, episode=key.episode) for key in keys])


def _order_only_reason(left: LeftForHandImport) -> str:
    """Why the order-only files are held, as a sentence with no capital and no closing mark."""

    keys = [file.order_only for file in left.files if file.order_only is not None]
    names = list(dict.fromkeys(order_display_name(key.order) for key in keys))
    orders = " and ".join(names)
    noun = "order" if len(names) == 1 else "orders"
    if len(keys) == 1:
        return (
            f"this pack is numbered by the {orders} {noun}, and 1 of its files is one of the {orders} episodes "
            "that aren't in Sonarr's list, so Pearlarr can't tell where it belongs"
        )
    return (
        f"this pack is numbered by the {orders} {noun}, and {len(keys)} of its files are {orders} episodes "
        "that aren't in Sonarr's list, so Pearlarr can't tell where they belong"
    )


def hand_import_names(left: LeftForHandImport, *, sep: str = ": ", quote: str = "") -> list[str]:
    """Each held-back file's name, an order-only one behind its order's number, as "DVD S07E01: NAME"."""

    names: list[str] = []
    for file in left.files:
        quoted = f"{quote}{file.name}{quote}"
        names.append(quoted if file.order_only is None else f"{file.order_only.label}{sep}{quoted}")
    return names


def hand_import_sentences(left: LeftForHandImport) -> tuple[str, str]:
    """A record explained by its order, in words: a sentence per kind of unfilled episode, then why its files are held.

    The reason has no closing mark.
    """

    lead: list[str] = []
    if left.missing:
        verb = "is" if len(left.missing) == 1 else "are"
        lead.append(f"{_episodes(left.missing)} {verb} still missing in Sonarr.")
    if len(left.untrusted) == 1:
        lead.append(f"{_episodes(left.untrusted)} has a file in Sonarr, but it isn't from a recommended release.")
    elif left.untrusted:
        lead.append(f"{_episodes(left.untrusted)} have files in Sonarr, but they aren't from a recommended release.")
    reason = _order_only_reason(left)
    return " ".join(lead), reason[0].upper() + reason[1:]


def _still_needed_note(left: LeftForHandImport) -> str:
    """The unfilled episodes of a record explained by its order, as its ledger row's coda."""

    parts: list[str] = []
    if left.missing:
        parts.append(f"{_episodes(left.missing)} still missing in Sonarr")
    if len(left.untrusted) == 1:
        parts.append(f"{_episodes(left.untrusted)} has a file that isn't from a recommended release")
    elif left.untrusted:
        parts.append(f"{_episodes(left.untrusted)} have files that aren't from a recommended release")
    return " · ".join(parts)


def row_finished_note(event: RowFinished) -> str:
    """The ledger line's parenthesized coda. "" when there is nothing to say."""

    outcome = event.outcome
    left = event.left_for_hand_import
    if outcome is Outcome.UNMATCHED:
        return f"{count_noun(len(left.files), 'file')} matched no episode · import by hand in Sonarr"
    if outcome is Outcome.PARTIAL:
        if left.explained_by_order:
            return _still_needed_note(left)
        return f"{count_noun(len(left.files), 'file')} held back · import by hand in Sonarr"
    if outcome is Outcome.IMPORTED:
        parts: list[str] = []
        if event.files:
            parts.append(count_noun(event.files, "file"))
        if event.waited_s >= 1.0:
            parts.append(long_duration(event.waited_s))
        return " · ".join(parts)
    if outcome.kind is OutcomeKind.IN_PROGRESS:
        return "checked again next run"
    if not outcome.dropped:
        return "retries next run"
    # The only other dropped outcome is MISSING: the torrent is gone from qBittorrent.
    return "no longer tracked · title checked again next run"


# --- the durable ledger-line builders (the rich console's scrollback) ----------------


def wait_start_line(event: WaitStarted) -> ConsoleLine:
    """The non-TTY digest's opening line, worded by the pass kind."""

    plural = "" if event.total == 1 else "s"
    return ConsoleLine(logging.INFO, event.kind.start_template.format(n=event.total, s=plural))


def wait_progress_line(snapshot: WaitSnapshot) -> ConsoleLine:
    """One throttled "still waiting" aggregate pulse."""

    counts = snapshot.counts()
    message = indent_string(
        f"still waiting · {counts[Phase.DOWNLOADING]} downloading · "
        f"{counts[Phase.IMPORTING]} importing · {counts[Phase.AWAITING_FIRST_POLL]} queued · "
        f"{long_duration(snapshot.elapsed_s)}",
    )
    return ConsoleLine(logging.INFO, message)


def _detail(key: str, value: str, style: str | None = None) -> KeyValue:
    """A labeled row under a ledger row: the key in the gutter, the value under the row's label."""

    return KeyValue(key=key, value=value, key_width=_ROW_KEY_WIDTH, value_style=style, indent=DETAIL_INDENT, sep="")


def _hand_import_details(left: LeftForHandImport, features: ConsoleFeatures, style: str) -> list[KeyValue]:
    """The rows under a held-back ledger row: the reason, when the order explains it, then `_MAX_FILE_ROWS` files."""

    rows: list[KeyValue] = []
    if left.explained_by_order:
        reason = textwrap.wrap(_order_only_reason(left), width=_REASON_WIDTH)
        rows += [_detail("" if row else "reason", text, style) for row, text in enumerate(reason)]
    names = hand_import_names(left, sep="  ")
    rows += [_detail("" if row else "files", name) for row, name in enumerate(names[:_MAX_FILE_ROWS])]
    if len(names) > _MAX_FILE_ROWS:
        rows.append(_detail("", f"{'…' if features.can_unicode else '...'} +{len(names) - _MAX_FILE_ROWS} more"))
    return rows


def row_finished_lines(event: RowFinished, features: ConsoleFeatures) -> list[ConsoleLine]:
    """A finished torrent's durable ledger line (glyph + word + label + coda), then its held-back files' rows.

    Every line is at `severity_of`'s level.
    """

    level = int(severity_of(event))
    style = event.outcome.style if features.can_color else ""
    glyph = event.outcome.glyph(use_unicode=features.can_unicode)
    line = f"{glyph} {event.outcome.word.ljust(STATE_WIDTH)} {event.label}"
    tail = row_finished_note(event)
    if tail:
        line += f"  ({tail})"
    return [
        ConsoleLine(level, indent_string(line), StyledLine(style=style)),
        *(kv_line(detail, level=level) for detail in _hand_import_details(event.left_for_hand_import, features, style)),
    ]


def wait_tally_lines(event: WaitFinished) -> list[ConsoleLine]:
    """The closing wait summary. `[]` when nothing graduated."""

    if event.imported == 0 and event.in_progress == 0 and event.left == 0 and event.failed == 0:
        return []
    parts = [f"{event.imported} imported"]
    if event.in_progress:
        parts.append(f"{event.in_progress} pending")
    if event.left:
        parts.append(f"{event.left} left")
    if event.failed:
        parts.append(f"{event.failed} failed")
    parts.append(long_duration(event.elapsed_s))
    head = event.kind.tally_head
    return [
        ConsoleLine(logging.INFO, rule_string("-"), SeparatorRule(char="-")),
        ConsoleLine(logging.INFO, indent_string(f"{head} · " + " · ".join(parts))),
    ]


@final
class ProgressLineThrottle:
    """The "still waiting" pulse cadence."""

    __slots__ = ("_interval", "_next", "_skip_first")

    def __init__(self) -> None:
        self._interval: float | None = None
        self._next = 0.0
        self._skip_first = False

    def start(self, interval_s: float) -> None:
        self._interval = interval_s
        self._next = interval_s
        self._skip_first = True

    def fire(self, elapsed_s: float) -> bool:
        """Advance the cadence. True when a pulse is due at `elapsed_s`."""

        if self._interval is None:
            return False
        if self._skip_first:
            self._skip_first = False
            return False
        if elapsed_s < self._next:
            return False
        self._next = elapsed_s + self._interval
        return True

    def reset(self) -> None:
        self._interval = None
        self._next = 0.0
        self._skip_first = False


@dataclass(frozen=True, slots=True)
class RowModel:
    """One rendered in-flight row: plain strings, no rich."""

    label: str
    phase: Phase
    fraction: float
    status: str = ""
    """The status word drawn in place of the bar."""
    count: str = ""
    """Progress, e.g. "61%" or "8/12" files."""
    speed: str = ""
    """The download rate: sparkline + rate, or "stalled"."""
    time: str = ""
    """The ETA for a download, or the elapsed clock for an import."""
    size: str = ""
    """The total download size."""
    show_bar: bool = False
    """Draw a determinate block bar for `fraction`."""


@dataclass(frozen=True, slots=True)
class LiveModel:
    """A bounded, ordered, rich-free description of the live cockpit frame."""

    left_text: str
    right_text: str
    overall_fraction: float
    rows: tuple[RowModel, ...]
    overflow: str = ""


def live_model(snapshot: WaitSnapshot, features: ConsoleFeatures, kind: WaitKind) -> LiveModel:
    """Reduce a snapshot to a bounded, ordered cockpit frame."""

    in_flight = [t for t in snapshot.torrents if t.phase is not Phase.FINISHED]
    in_flight.sort(key=_row_sort_key)

    budget = max(MIN_LIVE_ROWS, min(MAX_LIVE_ROWS, features.height - _RESERVED_ROWS))
    visible = in_flight[:budget]
    hidden = in_flight[budget:]

    spark = features.can_unicode and features.width >= MIN_SPARK_WIDTH
    rows = tuple(_row_model(t, spark=spark) for t in visible)
    overflow = _overflow_text(hidden)

    counts = snapshot.counts()
    left = f"{kind.live_verb} {snapshot.done()}/{snapshot.total()}"
    arrow = "↓" if features.can_unicode else "dl"
    meta: list[str] = [long_duration(snapshot.elapsed_s)]
    agg_speed = _aggregate_speed(snapshot)
    if agg_speed:
        meta.append(f"{arrow} {human_bytes(agg_speed)}/s")
    agg_eta = _aggregate_eta(snapshot, agg_speed)
    if agg_eta is not None:
        meta.append(f"{_rough_eta(agg_eta)} left")
    if counts[Phase.IMPORTING]:
        meta.append(f"{counts[Phase.IMPORTING]} importing")

    return LiveModel(
        left_text=left,
        right_text=" · ".join(meta),
        overall_fraction=snapshot.overall_fraction(),
        rows=rows,
        overflow=overflow,
    )


# Every non-terminal Phase, drift-pinned (test_output_wait_render ties it to set(Phase)).
# _overflow_text and wait_progress_line hand-list the same trio.
_PHASE_RANK = {Phase.IMPORTING: 0, Phase.DOWNLOADING: 1, Phase.AWAITING_FIRST_POLL: 2}


def _row_sort_key(torrent: TorrentView) -> tuple[int, float]:
    """Order key: importing first, downloads by soonest ETA, queued last."""

    rank = _PHASE_RANK.get(torrent.phase, 3)
    eta = float(torrent.eta_s) if torrent.eta_s is not None else float("inf")
    return rank, eta


def _row_model(torrent: TorrentView, *, spark: bool) -> RowModel:
    """Format one in-flight torrent's cells for the cockpit."""

    if torrent.phase is Phase.DOWNLOADING:
        rate = "stalled" if torrent.speed_bps is None else f"{human_bytes(torrent.speed_bps)}/s"
        if spark and len(torrent.speed_history) >= 2:
            rate = f"{sparkline(torrent.speed_history)} {rate}"
        return RowModel(
            label=torrent.label,
            phase=torrent.phase,
            fraction=clamp01(torrent.fraction),
            count=f"{round(clamp01(torrent.fraction) * 100)}%",
            speed=rate,
            time="" if torrent.eta_s is None else _rough_eta(torrent.eta_s),
            size="" if torrent.bytes_total is None else human_bytes(torrent.bytes_total),
            show_bar=True,
        )
    if torrent.phase is Phase.IMPORTING:
        elapsed = long_duration(torrent.phase_elapsed_s)
        if torrent.import_total:
            # An import has no download rate, hence no speed cell.
            return RowModel(
                label=torrent.label,
                phase=torrent.phase,
                fraction=clamp01(torrent.fraction),
                count=f"{torrent.import_done}/{torrent.import_total}",
                time=elapsed,
                show_bar=True,
            )
        # Indeterminate, so no bar: the status word carries the phase, "copying" once the
        # import command's async copy is in flight and "importing" before.
        return RowModel(
            label=torrent.label,
            phase=torrent.phase,
            fraction=1.0,
            status="copying" if torrent.command_issued else "importing",
            time=elapsed,
        )
    return RowModel(label=torrent.label, phase=Phase.AWAITING_FIRST_POLL, fraction=0.0, status="queued")


def _overflow_text(hidden: list[TorrentView]) -> str:
    """A "+ N more downloading · M queued" tally for the rows past the budget."""

    if not hidden:
        return ""
    counts: Counter[Phase] = Counter(t.phase for t in hidden)
    parts: list[str] = []
    if counts[Phase.IMPORTING]:
        parts.append(f"{counts[Phase.IMPORTING]} more importing")
    if counts[Phase.DOWNLOADING]:
        parts.append(f"{counts[Phase.DOWNLOADING]} more downloading")
    if counts[Phase.AWAITING_FIRST_POLL]:
        parts.append(f"{counts[Phase.AWAITING_FIRST_POLL]} queued")
    return "+ " + " · ".join(parts)


def _aggregate_speed(snapshot: WaitSnapshot) -> int:
    """Total download speed across the downloading rows (bytes/s)."""

    return sum(t.speed_bps for t in snapshot.torrents if t.phase is Phase.DOWNLOADING and t.speed_bps is not None)


def _aggregate_eta(snapshot: WaitSnapshot, agg_speed: int) -> int | None:
    """The remaining-bytes ETA over the shared pipe. A sub-second remainder reads as done (None)."""

    if agg_speed <= 0:
        return None
    remaining = 0
    for torrent in snapshot.torrents:
        if (
            torrent.phase is Phase.DOWNLOADING
            and torrent.bytes_total is not None
            and torrent.bytes_done is not None
            and torrent.bytes_total >= torrent.bytes_done
        ):
            remaining += torrent.bytes_total - torrent.bytes_done
    if remaining <= 0:
        return None
    return int(remaining / agg_speed) or None


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(samples: tuple[int, ...]) -> str:
    """The speed-history glyph run, scaled to the window's own peak."""

    if not samples:
        return ""
    peak = max(samples)
    top = len(_SPARK_CHARS) - 1
    if peak <= 0:
        return _SPARK_CHARS[0] * len(samples)
    return "".join(_SPARK_CHARS[round(sample / peak * top)] for sample in samples)


def _rough_eta(seconds: float) -> str:
    """A short `~` ETA, e.g. `"~2m"` / `"~1h05m"` / `"~40s"`."""

    return f"~{compact_duration(seconds)}"
