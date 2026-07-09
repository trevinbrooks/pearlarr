"""The shared traditional text grammar + the text-surface renderers.

One grammar — ``ts LEVEL [path] message k=v`` — feeds both :class:`LineRenderer`
(plain stdout) and :class:`FileLogSink` (the log file), byte-identical by
construction (the sole carve-out: ``file_only`` diagnostics reach the file alone;
the rule lives once, on the shared chassis). ``[path]`` is a label-only breadcrumb
from a per-sink :class:`BreadcrumbFold` (S1: never position, never layout);
diagnostics instead carry their origin in the bracket plus an advisory
``during="..." placed=frontier`` tail. Values quote iff they contain whitespace,
``"`` or ``=``; newlines are escaped in messages and quoted values — a rendered
traceback is the sole multi-line form. :class:`JsonRenderer` rides the same
chassis and writes one JSON object per event (stable key order, local time with
its UTC offset).

Every event's facts — (name, severity, message, fields) — are stated exactly once
(:func:`_fact_of`); the text and json surfaces only decorate. Sinks render each
event BEFORE folding it (so a closing event still renders with the path it is
closing), and the fold advances even when rendering raises. Admission is
line-granular on the text surfaces: a WARNING summary row reaches a WARNING-level
file while its INFO siblings drop.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Final, NamedTuple, TextIO, assert_never, final, override

from .breadcrumbs import BreadcrumbFold
from .events import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CapReached,
    CycleStarted,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFact,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    NeedsActionFact,
    NextRunScheduled,
    PlacedBy,
    RecommendedGroup,
    ReleaseName,
    ReleaseSkipped,
    RunFinished,
    RunStarted,
    RunSummary,
    RunSummaryReady,
    ScanFinished,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeOpened,
    Severity,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
    severity_of,
)
from ..log import LOG_NAME, MAX_LOG_FILES, console_level
from ..manual_import import OutcomeCategory

TS_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

type FieldValue = str | int | float | bool


class Field(NamedTuple):
    """One ``key=value`` fact on a rendered line / json object."""

    key: str
    value: FieldValue


type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def console_threshold(level: int) -> int:
    """The stdout surfaces' floor (S4); delegates to log.py's single body."""

    return console_level(level)


@dataclass(frozen=True, slots=True)
class _Line:
    """One rendered text line: level word, bracket, message, logfmt fields, trace."""

    severity: Severity
    bracket: str | None
    message: str
    fields: tuple[Field, ...] = ()
    trace: str | None = None


def _flat(text: str) -> str:
    """Escape newlines so a message can never break the one-line grammar."""

    return text.replace("\n", "\\n").replace("\r", "\\r")


def _needs_quote(text: str) -> bool:
    return text == "" or '"' in text or "=" in text or any(ch.isspace() for ch in text)


def _field_text(value: FieldValue) -> str:
    # bool before int: bool is an int subclass.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if _needs_quote(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        return f'"{escaped}"'
    return value


def _logfmt(fields: tuple[Field, ...]) -> str:
    return " ".join(f"{field.key}={_field_text(field.value)}" for field in fields)


def _render_line(line: _Line, ts: str) -> str:
    parts = [ts, line.severity.name]
    if line.bracket:
        parts.append(f"[{line.bracket}]")
    parts.append(_flat(line.message))
    if line.fields:
        parts.append(_logfmt(line.fields))
    text = " ".join(parts)
    if line.trace:
        # The traceback is the sole multi-line carve-out (forensics, verbatim).
        text += "\n" + line.trace.rstrip("\n")
    return text


# --- per-event facts: name/severity/message/fields, stated exactly once -------------


@dataclass(frozen=True, slots=True)
class _Fact:
    """One event's shared surface facts; text/json only decorate around these."""

    name: str
    severity: Severity
    message: str
    fields: tuple[Field, ...]
    scope: ScopeId | None = None
    component: str = ""


def _boot_outcome_word(category: OutcomeCategory) -> str:
    if category is OutcomeCategory.FAILED:
        return "failed"
    if category is OutcomeCategory.DEFERRED:
        return "warned"
    return "ok"


def _grab_message(status: GrabStatus) -> str:
    if status is GrabStatus.ADDING:
        return "adding recommended release"
    if status is GrabStatus.WOULD_ADD:
        return "would add recommended release (dry run)"
    return "recommended release already downloading"


def _group_text(group: RecommendedGroup) -> str:
    if group.tags:
        return f"{group.name} [{', '.join(group.tags)}]"
    return group.name


def _names_text(releases: tuple[ReleaseName, ...]) -> str:
    return "; ".join(release.name or release.group for release in releases)


def _fields_run_started(event: RunStarted) -> tuple[Field, ...]:
    fields: list[Field] = []
    if event.version:
        fields.append(Field("version", event.version))
    fields.append(Field("data_dir", event.data_dir))
    return tuple(fields)


def _fields_boot_finished(event: BootStepFinished) -> tuple[Field, ...]:
    fields: list[Field] = [Field("outcome", _boot_outcome_word(event.outcome))]
    if event.detail is not None:
        fields.append(Field("detail", event.detail))
    fields.append(Field("elapsed_s", event.elapsed_s))
    return tuple(fields)


def _fields_item_started(event: ItemStarted) -> tuple[Field, ...]:
    return (
        Field("arr", str(event.arr)),
        Field("index", event.index),
        Field("total", event.total),
        Field("title", event.title),
    )


def _fields_entry_header(event: EntryHeader) -> tuple[Field, ...]:
    fields: list[Field] = [Field("title", event.title)]
    if event.al_id is not None:
        fields.append(Field("al_id", event.al_id))
    if event.coverage:
        fields.append(Field("files", event.coverage))
    if event.url:
        fields.append(Field("link", event.url))
    if event.incomplete:
        fields.append(Field("incomplete", True))
    return tuple(fields)


def _fields_release_skipped(event: ReleaseSkipped) -> tuple[Field, ...]:
    fields: list[Field] = [
        Field("group", event.group),
        Field("tracker", event.tracker),
        Field("reason", event.reason.name.lower()),
    ]
    if event.url:
        fields.append(Field("link", event.url))
    return tuple(fields)


def _fields_grab_action(event: GrabAction) -> tuple[Field, ...]:
    fields: list[Field] = []
    if event.waiting_to_import:
        fields.append(Field("waiting_to_import", True))
    if event.groups:
        fields.append(Field("groups", "; ".join(_group_text(group) for group in event.groups)))
    if event.added:
        fields.append(Field("added", _names_text(event.added)))
    if event.downloading:
        fields.append(Field("downloading", _names_text(event.downloading)))
    return tuple(fields)


def _fields_graduated(event: TorrentGraduated) -> tuple[Field, ...]:
    fields: list[Field] = [Field("title", event.label)]
    if event.files is not None:
        fields.append(Field("files", event.files))
    fields.append(Field("waited_s", event.waited_s))
    return tuple(fields)


def _fields_wait_finished(event: WaitFinished) -> tuple[Field, ...]:
    return (
        Field("imported", event.imported),
        Field("deferred", event.deferred),
        Field("failed", event.failed),
        Field("elapsed_s", event.elapsed_s),
    )


def _fields_summary_head(summary: RunSummary) -> tuple[Field, ...]:
    tally = summary.tally
    fields: list[Field] = [Field("arr", str(summary.arr))]
    if summary.dry_run:
        fields.append(Field("dry_run", True))
        if summary.dry_run_note:
            fields.append(Field("note", summary.dry_run_note))
    fields.append(Field("checked", tally.checked))
    fields.append(Field("needs_action", len(tally.needs_action)))
    fields.append(Field("added", summary.added_count))
    if summary.wait_mode_on:
        for key, value in (
            ("queued", tally.queued),
            ("importing", tally.importing),
            ("imported", tally.imported),
        ):
            if value:
                fields.append(Field(key, value))
    fields.append(Field("up_to_date", tally.up_to_date))
    fields.append(Field("cached", tally.cached))
    for key, value in (
        ("no_mappings", tally.no_mappings),
        ("no_seadex_entry", tally.no_seadex_entry),
        ("seadex_unreachable", tally.seadex_unreachable),
        ("no_releases", tally.no_releases),
        ("unmonitored", tally.unmonitored),
    ):
        if value:
            fields.append(Field(key, value))
    fields.append(Field("warnings", summary.warnings))
    fields.append(Field("errors", summary.errors))
    if summary.elapsed_s is not None:
        fields.append(Field("elapsed_s", summary.elapsed_s))
    if summary.tip is not None:
        fields.append(Field("tip", summary.tip.name.lower()))
    return tuple(fields)


def _fields_needs_action(record: NeedsActionFact) -> tuple[Field, ...]:
    fields: list[Field] = [Field("title", record.title or "(unknown title)")]
    if record.coverage:
        fields.append(Field("files", record.coverage))
    fields.append(Field("group", record.group))
    fields.append(Field("reason", record.reason))
    fields.append(Field("kind", record.cause.name.lower()))
    if record.url:
        fields.append(Field("link", record.url))
    return tuple(fields)


def _fields_added(record: GrabFact) -> tuple[Field, ...]:
    fields: list[Field] = [Field("title", record.title or "(unknown title)")]
    if record.coverage:
        fields.append(Field("files", record.coverage))
    fields.append(Field("group", record.group))
    if record.name:
        fields.append(Field("torrent", record.name))
    if record.url:
        fields.append(Field("link", record.url))
    return tuple(fields)


def _placement_fields(event: Diagnostic, crumbs: BreadcrumbFold) -> tuple[Field, ...]:
    # Ambient placement is a guess; the record admits it (during=/placed=frontier).
    during = crumbs.during()
    if event.placed_by is PlacedBy.AMBIENT and during is not None:
        return (Field("during", during), Field("placed", "frontier"))
    return ()


def _scope_fields(scope: ScopeId) -> tuple[Field, ...]:
    return (Field("kind", scope.kind.name.lower()), Field("serial", scope.serial))


def _fact_of(event: Event, crumbs: BreadcrumbFold, severity: Severity) -> _Fact | None:
    """THE exhaustive per-event table: None = the event has no rendered form."""

    match event:
        case RunStarted():
            return _Fact("run_started", severity, "SeaDexArr started", _fields_run_started(event), None, "run")
        case CycleStarted(number=number):
            return _Fact("cycle_started", severity, "cycle started", (Field("number", number),), None, "run")
        case NextRunScheduled(at=at):
            return _Fact(
                "next_run_scheduled", severity, "next run scheduled", (Field("at", at.isoformat()),), None, "run"
            )
        case ScopeOpened(scope=scope, label=label):
            fields = (*_scope_fields(scope), Field("label", label))
            return _Fact("scope_opened", severity, "scope opened", fields, None, "scope")
        case ScopeClosed(scope=scope):
            return _Fact("scope_closed", severity, "scope closed", _scope_fields(scope), None, "scope")
        case BootStepStarted() | BootStepProgressed() | WaitProgress():
            return None
        case BootStepSlow(scope=scope):
            return _Fact("boot_step_slow", severity, "in progress", (), scope, "boot")
        case BootStepFinished(scope=scope):
            return _Fact("boot_step_finished", severity, "done", _fields_boot_finished(event), scope, "boot")
        case BootReady(elapsed_s=elapsed_s):
            return _Fact("boot_ready", severity, "ready", (Field("elapsed_s", elapsed_s),), None, "boot")
        case ScanStarted(arr=arr, total=total):
            fields = (Field("arr", str(arr)), Field("total", total))
            return _Fact("scan_started", severity, "starting", fields, None, "scan")
        case ItemStarted():
            return _Fact("item_started", severity, "item", _fields_item_started(event), None, "scan")
        case EntryHeader(scope=scope, state=state):
            return _Fact("entry_header", severity, str(state), _fields_entry_header(event), scope, "entry")
        case EntryDetail(scope=scope, label=label, value=value, tail=tail):
            fields = (Field("note", tail),) if tail is not None else ()
            return _Fact("entry_detail", severity, f"{label}: {value.text}", fields, scope, "entry")
        case LedgerRow(scope=scope, state=state, label=label):
            return _Fact("ledger_row", severity, str(state), (Field("title", label),), scope, "entry")
        case ReleaseSkipped(scope=scope):
            return _Fact("release_skipped", severity, "release skipped", _fields_release_skipped(event), scope, "entry")
        case GrabFailed(scope=scope, group=group, url=url, error=error):
            fields = (Field("group", group), Field("link", url), Field("error", error))
            return _Fact("grab_failed", severity, "grab failed", fields, scope, "entry")
        case GrabAction(scope=scope, status=status):
            return _Fact("grab_action", severity, _grab_message(status), _fields_grab_action(event), scope, "entry")
        case CapReached(cap=cap):
            return _Fact("cap_reached", severity, "torrent cap reached", (Field("cap", cap),), None, "run")
        case ScanFinished(arr=arr):
            return _Fact("scan_finished", severity, "scan finished", (Field("arr", str(arr)),), None, "scan")
        case RunSummaryReady(summary=summary):
            return _Fact("run_summary", severity, "run complete", _fields_summary_head(summary), None, "summary")
        case WaitStarted(scope=scope, total=total):
            return _Fact("wait_started", severity, "waiting", (Field("total", total),), scope, "wait")
        case TorrentGraduated(scope=scope, outcome=outcome):
            return _Fact("torrent_graduated", severity, outcome.word, _fields_graduated(event), scope, "wait")
        case WaitFinished(scope=scope):
            return _Fact("wait_finished", severity, "complete", _fields_wait_finished(event), scope, "wait")
        case RunFinished(arr=arr):
            return _Fact("run_finished", severity, "run finished", (Field("arr", str(arr)),), None, "run")
        case Diagnostic(message=message, origin=origin):
            return _Fact("diagnostic", severity, message, _placement_fields(event, crumbs), None, origin)
    assert_never(event)


# Facts with no TEXT-line form: pure boundaries (the summary is the run-end marker;
# run-close is deliberately emitted twice, B3/B4.3 — a rendered line would double).
_TEXT_SKIP: Final = frozenset({ScopeOpened, ScopeClosed, ScanFinished, RunFinished})
# Facts the JSON stream drops: the slow heads-up is a text-sink affordance only.
_JSON_SKIP: Final = frozenset({BootStepSlow})


def _scope_bracket(scope: ScopeId | None, crumbs: BreadcrumbFold, fallback: str) -> str:
    """The breadcrumb for a handle-carried ScopeId, or the fixed component word."""

    if scope is not None:
        path = crumbs.path_for(scope)
        if path is not None:
            return path
    return fallback


def _record_lines(summary: RunSummary) -> tuple[_Line, ...]:
    needs = tuple(
        _Line(Severity.WARNING, "summary", "needs action", _fields_needs_action(record))
        for record in summary.tally.needs_action
    )
    added = tuple(_Line(Severity.INFO, "summary", "added", _fields_added(record)) for record in summary.tally.added)
    return (*needs, *added)


def _lines_of(event: Event, crumbs: BreadcrumbFold, severity: Severity) -> tuple[_Line, ...]:
    """The text grammar: the shared fact plus text-only decoration."""

    fact = _fact_of(event, crumbs, severity)
    if fact is None or type(event) in _TEXT_SKIP:
        return ()
    bracket = _scope_bracket(fact.scope, crumbs, fact.component)
    trace = event.trace.plain_text() if isinstance(event, Diagnostic) and event.trace is not None else None
    head = _Line(fact.severity, bracket, fact.message, fact.fields, trace)
    if isinstance(event, RunSummaryReady):
        return (head, *_record_lines(event.summary))
    return (head,)


def format_line(event: Event, *, crumbs: BreadcrumbFold, when: datetime) -> str | None:
    """Render one event into the shared text grammar (None = no text-line form).

    The full, unthresholded rendering — the sinks apply per-line severity floors.
    """

    lines = _lines_of(event, crumbs, severity_of(event))
    if not lines:
        return None
    ts = when.strftime(TS_FORMAT)
    return "\n".join(_render_line(line, ts) for line in lines)


def _json_of(event: Event, crumbs: BreadcrumbFold, iso: str, severity: Severity) -> dict[str, JsonValue] | None:
    """The json shape: the shared fact plus json-only decoration, stable key order."""

    fact = _fact_of(event, crumbs, severity)
    if fact is None or type(event) in _JSON_SKIP:
        return None
    payload: dict[str, JsonValue] = {
        "time": iso,
        "event": fact.name,
        "level": fact.severity.name,
        "message": fact.message,
    }
    if isinstance(event, Diagnostic):
        payload["origin"] = event.origin
    if fact.scope is not None:
        path = crumbs.path_for(fact.scope)
        if path is not None:
            payload["path"] = path
    for field in fact.fields:
        payload[field.key] = field.value
    if isinstance(event, RunSummaryReady):
        tally = event.summary.tally
        payload["needs_action_records"] = [
            dict[str, JsonValue](_fields_needs_action(record)) for record in tally.needs_action
        ]
        payload["added_records"] = [dict[str, JsonValue](_fields_added(record)) for record in tally.added]
    if isinstance(event, Diagnostic) and event.trace is not None:
        payload["exc"] = event.trace.plain_text()
    return payload


# --- the sinks ------------------------------------------------------------------------


@final
class _PerSecondMemo:
    """Caches one formatted timestamp per whole second (hub stamps per emit)."""

    __slots__ = ("_fmt", "_memo")

    def __init__(self, fmt: Callable[[datetime], str]) -> None:
        self._fmt = fmt
        self._memo: tuple[int, str] = (-1, "")

    def format(self, when: float) -> str:
        second = int(when)
        if self._memo[0] != second:
            self._memo = (second, self._fmt(datetime.fromtimestamp(when)))
        return self._memo[1]


class _TextLineSink:
    """The shared sink chassis: admission, fold ordering, per-second timestamps.

    Subclasses provide only the render step; ``handle`` folds the event AFTER
    rendering — in a ``finally``, so a render/write bug can never desync the path.
    """

    # The single file_only routing rule: console-ish surfaces skip, the file keeps.
    _writes_file_only: ClassVar[bool] = False

    def __init__(self) -> None:
        self._crumbs = BreadcrumbFold()
        self._threshold: int = int(Severity.INFO)
        self._ts = _PerSecondMemo(lambda dt: dt.strftime(TS_FORMAT))

    def handle(self, event: Event, when: float) -> None:
        severity = severity_of(event)
        try:
            if self._admits(event):
                self._render(event, when, severity)
        finally:
            self._crumbs.apply(event)

    def begin_cycle(self) -> None:
        self._crumbs.reset()
        self._turn_over()

    def set_level(self, level: int) -> None:
        self._threshold = console_threshold(level)

    def close(self) -> None:
        pass

    def _admits(self, event: Event) -> bool:
        return self._writes_file_only or not (isinstance(event, Diagnostic) and event.file_only)

    def _render(self, event: Event, when: float, severity: Severity) -> None:
        raise NotImplementedError

    def _turn_over(self) -> None:
        pass


class _GrammarSink(_TextLineSink):
    """Chassis + the text grammar with line-granular admission (#7)."""

    @override
    def _render(self, event: Event, when: float, severity: Severity) -> None:
        lines = _lines_of(event, self._crumbs, severity)
        kept = tuple(line for line in lines if line.severity >= self._threshold)
        if not kept:
            return
        ts = self._ts.format(when)
        text = "\n".join(_render_line(line, ts) for line in kept) + "\n"
        self._write(text, max(line.severity for line in kept))

    def _write(self, text: str, severity: Severity) -> None:
        raise NotImplementedError


@final
class LineRenderer(_GrammarSink):
    """Plain stdout: the file grammar on the console (pipes, Docker logs).

    Flushes per line; stdout blocking under the hub lock is parity with stdlib
    logging today — revisit at PR2 if it ever matters.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream

    @override
    def _write(self, text: str, severity: Severity) -> None:
        self._stream.write(text)
        self._stream.flush()


@final
class FileLogSink(_GrammarSink):
    """The traditional structured log file; rotation mirrors setup_logger's cascade.

    Rotation is pending at construction and re-armed by begin_cycle; it runs on the
    first write after either, so an idle cycle never churns the cascade. Writes are
    buffered — flushed on WARNING+, cycle turnover, and close. A reopen after close
    appends (never a silent truncate without a pending rotation).
    """

    _writes_file_only: ClassVar[bool] = True

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self._dir = log_dir
        self._file: TextIO | None = None
        self._rotate_pending = True

    @property
    def path(self) -> str:
        return os.path.join(self._dir, f"{LOG_NAME}.log")

    @override
    def set_level(self, level: int) -> None:
        # The file honors the configured level directly (S4) - no INFO floor.
        self._threshold = level

    @override
    def close(self) -> None:
        self._close_file()

    @override
    def _turn_over(self) -> None:
        self._close_file()
        self._rotate_pending = True

    @override
    def _write(self, text: str, severity: Severity) -> None:
        stream = self._open_if_needed()
        stream.write(text)
        if severity >= Severity.WARNING:
            stream.flush()

    def _open_if_needed(self) -> TextIO:
        if self._file is None:
            os.makedirs(self._dir, exist_ok=True)
            if self._rotate_pending:
                self._rotate()
                self._rotate_pending = False
                mode = "w"
            else:
                mode = "a"
            self._file = open(self.path, mode, encoding="utf-8")
        return self._file

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _rotate(self) -> None:
        # .log -> .log.1 -> ... -> .log.9; os.replace overwrites the oldest atomically.
        if not os.path.isfile(self.path):
            return
        for i in range(MAX_LOG_FILES - 1, 0, -1):
            old_log = os.path.join(self._dir, f"{LOG_NAME}.log.{i}")
            new_log = os.path.join(self._dir, f"{LOG_NAME}.log.{i + 1}")
            if os.path.exists(old_log):
                os.replace(old_log, new_log)
        os.replace(self.path, os.path.join(self._dir, f"{LOG_NAME}.log.1"))


@final
class JsonRenderer(_TextLineSink):
    """One JSON object per event: stable key order, local time with its UTC offset."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream
        self._iso = _PerSecondMemo(lambda dt: dt.astimezone().isoformat(timespec="seconds"))

    @override
    def _render(self, event: Event, when: float, severity: Severity) -> None:
        if self._effective(event, severity) < self._threshold:
            return
        payload = _json_of(event, self._crumbs, self._iso.format(when), severity)
        if payload is not None:
            self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._stream.flush()

    @staticmethod
    def _effective(event: Event, severity: Severity) -> Severity:
        # A summary carrying user-actionable / error content admits at WARNING (#7).
        if isinstance(event, RunSummaryReady):
            summary = event.summary
            if summary.tally.needs_action or summary.errors:
                return max(severity, Severity.WARNING)
        return severity
