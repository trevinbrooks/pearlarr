"""The shared traditional text grammar + the text-surface renderers.

One grammar — `ts LEVEL [path] message k=v` — feeds both `LineRenderer`
(plain stdout) and `FileLogSink` (the log file), byte-identical by
construction (the sole carve-out: `file_only` diagnostics reach the file alone;
the rule lives once, on the shared chassis). `[path]` is a label-only breadcrumb
from a per-sink `BreadcrumbFold` (never position, never layout);
diagnostics instead carry their origin in the bracket plus an advisory
`during="..." placed=frontier` tail. Values quote iff they contain whitespace,
`"` or `=`; newlines are escaped in messages and quoted values — a rendered
traceback is the sole multi-line form. `JsonRenderer` rides the same
chassis and writes one JSON object per event (stable key order, local time with
its UTC offset).

Every event's facts — (name, severity, message, fields) — are stated exactly once
(`_fact_of`); the text and json surfaces only decorate. Sinks render each
event BEFORE folding it (so a closing event still renders with the path it is
closing), and the fold advances even when rendering raises. Admission is
line-granular on the text surfaces: a WARNING summary row reaches a WARNING-level
file while its INFO siblings drop. WaitProgress is the one stateful carve-out:
the grammar sinks throttle it into a "still waiting" pulse whose cadence is pure
event content (WaitStarted.pulse_s, snapshot.elapsed_s — never wall clock), so
the independently ticking file and plain sinks stay byte-identical.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import time
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
    CacheBackedUp,
    CacheIntegrityReported,
    CacheRemoved,
    CacheRestored,
    CacheStatsReported,
    CapReached,
    ConfigMigrated,
    ConfigUpToDate,
    ConfigValidated,
    CycleStarted,
    Diagnostic,
    EffectiveConfigShown,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFact,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    JsonValue,
    LedgerRow,
    NeedsActionFact,
    NextRunScheduled,
    PathsShown,
    Phase,
    PlacedBy,
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
    StarterConfigWritten,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
    severity_of,
)
from .hub import Renderer
from .wait_lines import PulseThrottle
from ..log import LOG_NAME
from ..manual_import import OutcomeCategory

TS_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

# A rotated backup is named after the run that wrote it (the file's own mtime).
_BACKUP_STAMP_FORMAT: Final = "%Y-%m-%d_%H%M%S"

# The rotation backstop (module attribute so tests can monkeypatch it): caps
# backups before config - and its retention days - has ever been read.
_BACKSTOP_MAX_BACKUPS = 500

# The JSON stream's envelope version. Changes within it are additive-only; a
# removal, rename, or semantic change bumps it alongside a new major release
# (docs/output.md states the policy; the event catalog there is generated).
JSON_SCHEMA_VERSION: Final = 1

# A field value is any JSON value: scalars for run events, plus lists/objects on
# the json-only cli command facts (which never reach a grammar-sink text line).
type FieldValue = JsonValue


class Field(NamedTuple):
    """One `key=value` fact on a rendered line / json object."""

    key: str
    value: FieldValue


@dataclass(frozen=True, slots=True)
class _Line:
    """One rendered text line: level word, bracket, message, logfmt fields, trace."""

    severity: Severity
    bracket: str | None
    message: str
    fields: tuple[Field, ...] = ()
    trace: str | None = None


# Control characters (C0 except tab, DEL, C1) escape to \xNN so external text -
# torrent titles, or a hostile capture replayed to a TTY - can neither break the
# one-line grammar nor drive the terminal (cursor movement rewrites prior lines).
_CTRL_TABLE: Final = str.maketrans(
    {0x0A: "\\n", 0x0D: "\\r"}
    | {code: f"\\x{code:02x}" for code in (*range(0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x7F, *range(0x80, 0xA0))}
)


def _flat(text: str) -> str:
    """Escape control characters so a message can never break or forge the one-line grammar."""

    return text.translate(_CTRL_TABLE)


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
    # Containers/None ride only the json-only cli facts (which never reach a
    # grammar-sink line); compact-encode them so the formatter stays total.
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if _needs_quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').translate(_CTRL_TABLE)
        return f'"{escaped}"'
    return text.translate(_CTRL_TABLE)


def _logfmt(fields: tuple[Field, ...]) -> str:
    return " ".join(f"{field.key}={_field_text(field.value)}" for field in fields)


def _render_line(line: _Line, ts: str) -> str:
    parts = [ts, line.severity.name]
    if line.bracket:
        # Escaped like the message: breadcrumb labels carry external titles
        # (Arr/AniList), and a newline there would break the one-line grammar.
        parts.append(f"[{_flat(line.bracket)}]")
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


def _names_text(releases: tuple[ReleaseName, ...]) -> str:
    return "; ".join(release.display for release in releases)


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
        fields.append(Field("groups", "; ".join(group.display for group in event.groups)))
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


def _pulse_fact(event: WaitProgress) -> _Fact:
    """The "still waiting" heartbeat's facts — pure; the grammar sinks decide WHEN."""

    counts = event.snapshot.counts()
    fields = (
        Field("downloading", counts[Phase.DOWNLOADING]),
        Field("importing", counts[Phase.IMPORTING]),
        Field("queued", counts[Phase.QUEUED]),
        Field("elapsed_s", event.snapshot.elapsed_s),
    )
    return _Fact("wait_pulse", Severity.INFO, "still waiting", fields, event.scope, "wait")


def _fields_summary_head(summary: RunSummary) -> tuple[Field, ...]:
    tally = summary.tally
    fields: list[Field] = [Field("arr", str(summary.arr))]
    if summary.dry_run_note is not None:
        fields.append(Field("dry_run", True))
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


def _fields_paths_shown(event: PathsShown) -> tuple[Field, ...]:
    return (
        Field("data_dir", event.data_dir),
        Field("config", event.config),
        Field("cache", event.cache),
        Field("mappings_db", event.mappings_db),
        Field("log_dir", event.log_dir),
    )


def _fields_config_validated(event: ConfigValidated) -> tuple[Field, ...]:
    # "config_path" not "path": the json envelope reserves "path" for the breadcrumb.
    migrated = event.migration_notes is not None
    fields: list[Field] = [Field("config_path", event.path), Field("migrated", migrated)]
    if event.migration_notes is not None:
        fields.append(Field("migration_notes", list[JsonValue](event.migration_notes)))
    fields.append(Field("sonarr_missing_keys", list[JsonValue](event.sonarr_missing_keys)))
    fields.append(Field("radarr_missing_keys", list[JsonValue](event.radarr_missing_keys)))
    fields.append(Field("qbit_configured", event.qbit_configured))
    return tuple(fields)


def _fields_config_migrated(event: ConfigMigrated) -> tuple[Field, ...]:
    return (
        Field("config_path", event.path),
        Field("backup_path", event.backup_path),
        Field("notes", list[JsonValue](event.notes)),
    )


def _fields_cache_stats(event: CacheStatsReported) -> tuple[Field, ...]:
    return (
        Field("entries", event.entries),
        Field("torrent_hashes", event.torrent_hashes),
        Field("anilist_meta", event.anilist_meta),
        Field("sonarr_parse", event.sonarr_parse),
        Field("pending_imports", event.pending_imports),
        Field("size_bytes", event.size_bytes),
    )


def _fact_of(event: Event, crumbs: BreadcrumbFold, severity: Severity) -> _Fact | None:
    """THE exhaustive per-event table: None = the event has no rendered form."""

    match event:
        case RunStarted():
            return _Fact("run_started", severity, "Pearlarr started", _fields_run_started(event), None, "run")
        case CycleStarted(number=number):
            return _Fact("cycle_started", severity, "cycle started", (Field("number", number),), None, "run")
        case NextRunScheduled(at=at):
            # Seconds precision keeps the offset and drops microseconds; the
            # json surface shares this Field, matching its "time" key's shape.
            fields = (Field("at", at.isoformat(timespec="seconds")),)
            return _Fact("next_run_scheduled", severity, "next run scheduled", fields, None, "run")
        case ScopeOpened(scope=scope, label=label):
            fields = (*_scope_fields(scope), Field("label", label))
            return _Fact("scope_opened", severity, "scope opened", fields, None, "scope")
        case ScopeClosed(scope=scope):
            return _Fact("scope_closed", severity, "scope closed", _scope_fields(scope), None, "scope")
        case BootStepStarted() | BootStepProgressed() | WaitProgress():
            # WaitProgress has no per-event fact; the grammar sinks pulse it (throttled).
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
        case PathsShown():
            return _Fact("paths_shown", severity, "resolved paths", _fields_paths_shown(event), None, "cli")
        case StarterConfigWritten(path=path):
            # "config_path" not "path": the json envelope reserves "path" for the breadcrumb.
            return _Fact(
                "starter_config_written", severity, "starter config written", (Field("config_path", path),), None, "cli"
            )
        case ConfigValidated():
            return _Fact("config_validated", severity, "config valid", _fields_config_validated(event), None, "cli")
        case ConfigUpToDate(path=path):
            return _Fact("config_up_to_date", severity, "config up to date", (Field("config_path", path),), None, "cli")
        case ConfigMigrated():
            return _Fact("config_migrated", severity, "config migrated", _fields_config_migrated(event), None, "cli")
        case EffectiveConfigShown(path=path, config=config):
            fields = (Field("config_path", path), Field("config", config))
            return _Fact("effective_config_shown", severity, "effective config", fields, None, "cli")
        case CacheBackedUp(backup_path=backup_path):
            return _Fact(
                "cache_backed_up", severity, "cache backed up", (Field("backup_path", backup_path),), None, "cli"
            )
        case CacheRestored(backup_path=backup_path):
            return _Fact(
                "cache_restored", severity, "cache restored", (Field("backup_path", backup_path),), None, "cli"
            )
        case CacheRemoved(path=path):
            return _Fact("cache_removed", severity, "cache removed", (Field("cache_path", path),), None, "cli")
        case CacheStatsReported():
            return _Fact("cache_stats_reported", severity, "cache stats", _fields_cache_stats(event), None, "cli")
        case CacheIntegrityReported(result=result):
            return _Fact(
                "cache_integrity_reported", severity, "cache integrity", (Field("result", result),), None, "cli"
            )
    assert_never(event)


# Facts with no TEXT-line form: pure boundaries (the summary is the run-end
# marker), plus the effective-config document (its config object has no logfmt line).
_TEXT_SKIP: Final = frozenset({ScopeOpened, ScopeClosed, ScanFinished, RunFinished, EffectiveConfigShown})
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

    # Skip first: scope boundaries fire per entry on BOTH grammar sinks, so
    # building their facts just to drop them is pure hot-path waste.
    if type(event) in _TEXT_SKIP:
        return ()
    fact = _fact_of(event, crumbs, severity)
    if fact is None:
        return ()
    bracket = _scope_bracket(fact.scope, crumbs, fact.component)
    trace = event.trace.plain if isinstance(event, Diagnostic) and event.trace is not None else None
    head = _Line(fact.severity, bracket, fact.message, fact.fields, trace)
    if isinstance(event, RunSummaryReady):
        return (head, *_record_lines(event.summary))
    return (head,)


def _json_of(event: Event, crumbs: BreadcrumbFold, iso: str, severity: Severity) -> dict[str, JsonValue] | None:
    """The json shape: the shared fact plus json-only decoration, stable key order."""

    fact = _fact_of(event, crumbs, severity)
    if fact is None or type(event) in _JSON_SKIP:
        return None
    payload: dict[str, JsonValue] = {
        "schema_version": JSON_SCHEMA_VERSION,
        "time": iso,
        "event": fact.name,
        "level": fact.severity.name,
        "message": fact.message,
    }
    # The text grammar's bracket seed (subsystem/origin word); always present, so
    # a `replay` of the stream can reconstruct the [bracket] without per-event
    # knowledge. For a Diagnostic it equals `origin`, which stays its own key.
    payload["component"] = fact.component
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
        payload["exc"] = event.trace.plain
    return payload


# The envelope + decoration keys a JSON line carries: `render_envelope_line`
# turns them into the ts/level/[bracket]/message/trace shape, and everything
# else in the object becomes the k=v tail.
_ENVELOPE_KEYS: Final = frozenset(
    {"schema_version", "time", "event", "level", "message", "component", "origin", "path", "exc"},
)


def _reformat_iso(iso: str) -> str | None:
    """The grammar's `ts` for an ISO-8601 `time` value, or None when it can't be parsed.

    The stored value is local wall time with its offset; strftime prints exactly
    the components the grammar sinks printed, so no timezone conversion is needed.
    """

    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return parsed.strftime(TS_FORMAT)


def _envelope_bracket(payload: dict[str, JsonValue], event: str) -> str:
    """The bracket seed: `path`, else `component`, else `origin`, else the event name."""

    for key in ("path", "component", "origin"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return event


def render_envelope_line(payload: dict[str, JsonValue]) -> str | None:
    """Render one JSON envelope back to a `ts LEVEL [bracket] message k=v` text line.

    The inverse of `_json_of` for post-mortem reading (`pearlarr replay`): a
    generic envelope formatter carrying NO per-event knowledge, so an unknown
    future event still renders. `time`, `level`, `message`, and `event` are
    required - a missing or non-string one (or an unparseable `time`) returns
    None, a malformed line the caller counts. The bracket is `path`, else
    `component`, else `origin`, else the event name (captures predating
    `component` fall through to origin/event); every remaining key becomes the
    k=v tail in insertion order, and an `exc` string appends the traceback
    exactly as the file sink does.
    """

    time_value = payload.get("time")
    level = payload.get("level")
    message = payload.get("message")
    event = payload.get("event")
    if not (
        isinstance(time_value, str) and isinstance(level, str) and isinstance(message, str) and isinstance(event, str)
    ):
        return None
    ts = _reformat_iso(time_value)
    if ts is None:
        return None

    parts = [ts, level]
    bracket = _envelope_bracket(payload, event)
    if bracket:
        parts.append(f"[{_flat(bracket)}]")
    parts.append(_flat(message))
    fields = tuple(Field(key, value) for key, value in payload.items() if key not in _ENVELOPE_KEYS)
    if fields:
        parts.append(_logfmt(fields))
    text = " ".join(parts)
    exc = payload.get("exc")
    if isinstance(exc, str):
        # The traceback is the sole multi-line form, appended like `_render_line`.
        text += "\n" + exc.rstrip("\n")
    return text


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


class _TextLineSink(Renderer):
    """The shared sink chassis: admission, fold ordering, per-second timestamps.

    Subclasses provide only the render step; `handle` folds the event AFTER
    rendering — in a `finally`, so a render/write bug can never desync the path.
    """

    def __init__(self) -> None:
        self._crumbs = BreadcrumbFold()
        self._threshold: int = int(Severity.INFO)

    @override
    def handle(self, event: Event, when: float) -> None:
        severity = severity_of(event)
        try:
            if self._admits(event):
                self._render(event, when, severity)
        finally:
            self._crumbs.apply(event)

    @override
    def begin_cycle(self) -> None:
        self._crumbs.reset()
        self._turn_over()

    @override
    def set_level(self, level: int) -> None:
        # Text surfaces share the file's semantics: the raw configured level.
        # The INFO floor is rich-console-only (log.py's console_level).
        self._threshold = level

    @override
    def close(self) -> None:
        pass

    def _admits(self, event: Event) -> bool:
        return self.writes_file_only or not (isinstance(event, Diagnostic) and event.file_only)

    def _render(self, event: Event, when: float, severity: Severity) -> None:
        raise NotImplementedError

    def _turn_over(self) -> None:
        pass


class _GrammarSink(_TextLineSink):
    """Chassis + the text grammar with line-granular admission.

    Owns the per-sink wait-pulse throttle. Cadence is a pure function of event
    content (WaitStarted.pulse_s, snapshot.elapsed_s — never wall clock), so the
    independently ticking file and plain sinks stay byte-identical.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pulse = PulseThrottle()
        self._ts = _PerSecondMemo(lambda dt: dt.strftime(TS_FORMAT))

    @override
    def begin_cycle(self) -> None:
        self._pulse.reset()
        super().begin_cycle()

    @override
    def _render(self, event: Event, when: float, severity: Severity) -> None:
        if isinstance(event, WaitStarted):
            # A new pass (possibly several per run) restarts the cadence.
            self._pulse.arm(event.pulse_s)
        if isinstance(event, WaitProgress):
            lines = self._pulse_lines(event)
        else:
            lines = _lines_of(event, self._crumbs, severity)
        kept = tuple(line for line in lines if line.severity >= self._threshold)
        if not kept:
            return
        ts = self._ts.format(when)
        text = "\n".join(_render_line(line, ts) for line in kept) + "\n"
        self._write(text)

    def _pulse_lines(self, event: WaitProgress) -> tuple[_Line, ...]:
        """The stateful throttle consult; the cadence advances regardless of level."""

        if not self._pulse.fire(event.snapshot.elapsed_s):
            return ()
        fact = _pulse_fact(event)
        bracket = _scope_bracket(fact.scope, self._crumbs, fact.component)
        return (_Line(fact.severity, bracket, fact.message, fact.fields),)

    def _write(self, text: str) -> None:
        raise NotImplementedError


@final
class LineRenderer(_GrammarSink):
    """Plain stdout: the file grammar on the console (pipes, Docker logs).

    Flushes per line; stdout blocking happens under the drain baton (never the
    hub lock), which is parity with stdlib logging's handler lock.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream

    @override
    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()


def _mtime_or_epoch(path: str) -> float:
    # A backup vanishing between glob and stat must not abort the prune.
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@final
class FileLogSink(_GrammarSink):
    """The traditional structured log file: dated per-run backups + age-based retention.

    Rotation is armed ONLY by begin_cycle and runs on the first write after it, so
    an idle cycle never churns a backup and a pre-cycle record appends to the
    PREVIOUS cycle's file (never a second rotation burning a backup slot). A
    rotation renames the current file to a `.log.<run's mtime stamp>` backup
    (collisions get a `.1`, `.2`, ... suffix); a config-free count backstop then
    caps the backup count, so a crash-looping scheduler can't grow the log dir
    unboundedly before config has ever loaded. Age-based retention is a separate
    step (`apply_retention_days`), applied once per cycle as soon as the run's
    `advanced.log_retention_days` is known. Every line flushes as written (crash
    fidelity: the tail is on disk when the process dies). A reopen after close
    appends (never a silent truncate without a pending rotation).
    """

    writes_file_only: ClassVar[bool] = True

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self._dir = log_dir
        self._file: TextIO | None = None
        self._rotate_pending = False

    @property
    def path(self) -> str:
        return os.path.join(self._dir, f"{LOG_NAME}.log")

    def probe(self) -> None:
        """Fail fast (OSError) if the log file can't be appended.

        cli pre-flights this at install so a root-owned/read-only file aborts
        with the clean data-dir message, instead of striking the sink mid-run.
        A file the probe itself created is removed, so the first real write
        still sees the pre-probe rotation state.
        """

        os.makedirs(self._dir, exist_ok=True)
        existed = os.path.isfile(self.path)
        with open(self.path, "a", encoding="utf-8"):
            pass
        if not existed:
            os.remove(self.path)

    @override
    def close(self) -> None:
        self._close_file()

    @override
    def _turn_over(self) -> None:
        self._close_file()
        self._rotate_pending = True

    @override
    def _write(self, text: str) -> None:
        stream = self._open_if_needed()
        stream.write(text)
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
        # Named after the run that wrote it (the file's own mtime), not a cascade
        # slot; a same-second collision falls through to a numeric suffix.
        if not os.path.isfile(self.path):
            return
        stamp = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime(_BACKUP_STAMP_FORMAT)
        target = os.path.join(self._dir, f"{LOG_NAME}.log.{stamp}")
        suffix = 0
        while os.path.exists(target):
            suffix += 1
            target = os.path.join(self._dir, f"{LOG_NAME}.log.{stamp}.{suffix}")
        os.replace(self.path, target)
        self._enforce_backstop()

    def _backup_paths(self) -> list[str]:
        # glob.escape: a data dir containing [ ] * ? must not silently match nothing.
        return glob.glob(os.path.join(glob.escape(self._dir), f"{LOG_NAME}.log.*"))

    def _enforce_backstop(self) -> None:
        # A crash-looping scheduler must not grow the log dir unboundedly before
        # config - and its configured retention - has ever loaded.
        backups = sorted(self._backup_paths(), key=_mtime_or_epoch)
        excess = len(backups) - _BACKSTOP_MAX_BACKUPS
        for path in backups[: max(excess, 0)]:
            with contextlib.suppress(OSError):
                os.remove(path)

    def apply_retention_days(self, days: int) -> None:
        """Delete dated backups older than `days` days, best-effort per file.

        Retention rides the config-application moment, never `_rotate`: the
        sink is built before config exists, and a one-shot `run single`
        rotates before config loads, so a rotation-time prune would either
        never fire or run with a default that could delete backups a longer
        configured retention wanted kept. This runs outside hub dispatch, so
        one un-deletable backup must not take down the run.
        """

        cutoff = time.time() - days * 86400
        for path in self._backup_paths():
            with contextlib.suppress(OSError):
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)


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
        # A summary carrying user-actionable / error content admits at WARNING.
        if isinstance(event, RunSummaryReady):
            summary = event.summary
            if summary.tally.needs_action or summary.errors:
                return max(severity, Severity.WARNING)
        return severity
