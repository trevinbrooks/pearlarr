# pyright: strict, reportPrivateUsage=false
# reportPrivateUsage is off for the skip-set pins alone: _TEXT_SKIP/_JSON_SKIP are
# deliberately private tables, imported to pin their exact membership.
"""Tests for the shared text grammar + text sinks (``output.textline``).

Golden-pin the ``ts LEVEL [path] message k=v`` grammar (the PR6 file contract),
the quoting/escape rules, breadcrumb labels + the advisory during=/placed=frontier
tail, line/file byte-parity (with the file_only carve-out), per-line admission,
the throttled "still waiting" pulse, the rotation cascade + append-after-close,
fold-in-finally, and the one-object-per-event json shape (stable key order,
offset-bearing time).
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import get_args, override

import pytest

from seadexarr.modules.config import Arr
from seadexarr.modules.json_narrow import is_json_list, is_json_obj
from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.output import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    BreadcrumbFold,
    CapReached,
    CycleStarted,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    Event,
    FileLogSink,
    GrabAction,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    JsonRenderer,
    LedgerRow,
    LineRenderer,
    NeedsActionCause,
    NextRunScheduled,
    Phase,
    PlacedBy,
    RecommendedGroup,
    ReleaseName,
    ReleaseSkipped,
    RunFinished,
    RunStarted,
    RunSummary,
    RunSummaryReady,
    RunTally,
    ScanFinished,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    SkipReason,
    StyledValue,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    format_line,
)
from seadexarr.modules.output.textline import _JSON_SKIP, _TEXT_SKIP
from seadexarr.modules.output.trace import CapturedTrace
from seadexarr.modules.reporter import GrabRecord, NeedsActionKind, NeedsActionRecord, RunStats
from seadexarr.modules.seadex_types import Json

_EPOCH = 1_751_990_000.0
_WHEN = datetime.fromtimestamp(_EPOCH)
_TS = _WHEN.strftime("%Y-%m-%d %H:%M:%S")

_STEP = ScopeId(ScopeKind.BOOT_STEP, 1)
_ENTRY = ScopeId(ScopeKind.ENTRY, 2)
_WAIT = ScopeId(ScopeKind.WAIT_REGION, 3)


def _format(event: Event, *events_before: Event) -> str | None:
    """format_line against a fold pre-loaded with ``events_before``."""

    fold = BreadcrumbFold()
    for before in events_before:
        fold.apply(before)
    return format_line(event, crumbs=fold, when=_WHEN)


def _entry_context() -> tuple[Event, Event, Event]:
    return (
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=3, total=182, title="Frieren"),
        ScopeOpened(scope=_ENTRY, label="Frieren: Beyond Journey's End"),
    )


# --- golden lines -----------------------------------------------------------------


def test_run_started_line() -> None:
    line = _format(RunStarted(version="v1.0.0", data_dir="/data/seadexarr"))
    assert line == f"{_TS} INFO [run] SeaDexArr started version=v1.0.0 data_dir=/data/seadexarr"


def test_run_started_omits_an_empty_version() -> None:
    line = _format(RunStarted(version="", data_dir="/data"))
    assert line == f"{_TS} INFO [run] SeaDexArr started data_dir=/data"


def test_boot_step_finished_line_keeps_the_step_path() -> None:
    finished = BootStepFinished(
        scope=_STEP,
        label="Reading config",
        outcome=OutcomeCategory.SUCCESS,
        detail=None,
        elapsed_s=0.02,
    )
    line = _format(
        finished,
        ScopeOpened(scope=ScopeId(ScopeKind.BOOT_SECTION, 9), label="boot"),
        BootStepStarted(scope=_STEP, label="Reading config"),
    )
    assert line == f"{_TS} INFO [boot › Reading config] done outcome=ok elapsed_s=0.02"


def test_warned_boot_step_stays_info_with_the_warn_on_the_outcome_field() -> None:
    # Level stays INFO for every outcome (echo/ledger parity; a WARNING severity
    # would double-count next to the caller's own logged warning) — the deferred
    # state rides outcome=warned.
    finished = BootStepFinished(
        scope=_STEP,
        label="Refreshing mappings",
        outcome=OutcomeCategory.DEFERRED,
        detail="SeaDex unreachable",
        elapsed_s=3.5,
    )
    line = _format(finished, BootStepStarted(scope=_STEP, label="Refreshing mappings"))
    assert line == (f'{_TS} INFO [Refreshing mappings] done outcome=warned detail="SeaDex unreachable" elapsed_s=3.50')


def test_boot_step_slow_heads_up_line() -> None:
    line = _format(
        BootStepSlow(scope=_STEP, label="Refreshing mappings"),
        ScopeOpened(scope=ScopeId(ScopeKind.BOOT_SECTION, 9), label="boot"),
        BootStepStarted(scope=_STEP, label="Refreshing mappings"),
    )
    assert line == f"{_TS} INFO [boot › Refreshing mappings] in progress"


def test_scan_and_item_lines() -> None:
    assert _format(ScanStarted(arr=Arr.SONARR, total=182)) == (f"{_TS} INFO [scan] starting arr=sonarr total=182")
    assert _format(ItemStarted(arr=Arr.SONARR, index=3, total=182, title="Frieren")) == (
        f"{_TS} INFO [scan] item arr=sonarr index=3 total=182 title=Frieren"
    )


def test_entry_header_line_reads_the_full_breadcrumb() -> None:
    header = EntryHeader(
        state=EntryState.CHECKING,
        title="Frieren: Beyond Journey's End",
        al_id=154587,
        coverage="S01 E01-E28",
        url="https://releases.moe/154587",
        incomplete=True,
        scope=_ENTRY,
    )
    line = _format(header, *_entry_context())
    assert line == (
        f"{_TS} INFO [sonarr › [3/182] Frieren › entry] checking "
        f'title="Frieren: Beyond Journey\'s End" al_id=154587 files="S01 E01-E28" '
        f"link=https://releases.moe/154587 incomplete=true"
    )


def test_grab_action_line_joins_groups_and_names_sink_side() -> None:
    action = GrabAction(
        status=GrabStatus.ADDING,
        groups=(RecommendedGroup("SubsPlease", ("dual audio",)), RecommendedGroup("Kowo")),
        added=(ReleaseName("[SubsPlease] Sousou no Frieren", "SubsPlease"),),
        downloading=(ReleaseName("", "Kowo"),),
        scope=_ENTRY,
    )
    line = _format(action, *_entry_context())
    assert line == (
        f"{_TS} INFO [sonarr › [3/182] Frieren › entry] adding recommended release "
        f'groups="SubsPlease [dual audio]; Kowo" added="[SubsPlease] Sousou no Frieren" downloading=Kowo'
    )


def test_grab_failed_line_escapes_embedded_quotes() -> None:
    failed = GrabFailed(group="G", url="https://x", error='tracker said "denied"', scope=_ENTRY)
    line = _format(failed, *_entry_context())
    assert line == (
        f"{_TS} WARNING [sonarr › [3/182] Frieren › entry] grab failed "
        f'group=G link=https://x error="tracker said \\"denied\\""'
    )


def test_multi_line_messages_and_field_values_stay_one_line() -> None:
    diag = Diagnostic(severity=Severity.WARNING, message="line one\nline two", origin="app")
    assert _format(diag) == f"{_TS} WARNING [app] line one\\nline two"

    failed = GrabFailed(group="G", url="u", error='said "no"\r\nthen hung up')
    assert _format(failed) == (
        f'{_TS} WARNING [entry] grab failed group=G link=u error="said \\"no\\"\\r\\nthen hung up"'
    )


def test_wait_lines_ride_the_wait_breadcrumb() -> None:
    context = (ScanStarted(arr=Arr.SONARR, total=182), ScopeOpened(scope=_WAIT, label="wait"))
    assert _format(WaitStarted(total=4, pulse_s=300.0, scope=_WAIT), *context) == (
        f"{_TS} INFO [sonarr › wait] waiting total=4"
    )
    graduated = TorrentGraduated(
        label="Sousou no Frieren",
        outcome=Outcome.DOWNLOAD_TIMED_OUT,
        files=None,
        waited_s=600.0,
        scope=_WAIT,
    )
    assert _format(graduated, *context) == (
        f'{_TS} WARNING [sonarr › wait] timed out title="Sousou no Frieren" waited_s=600.00'
    )


def test_ambient_diagnostic_admits_its_placement_guess() -> None:
    diag = Diagnostic(
        severity=Severity.WARNING,
        message="Config file /x/config.yml is readable by other users - chmod 600 /x/config.yml",
        origin="config",
    )
    line = _format(
        diag,
        ScopeOpened(scope=ScopeId(ScopeKind.BOOT_SECTION, 9), label="boot"),
        BootStepStarted(scope=_STEP, label="Reading config"),
    )
    assert line == (
        f"{_TS} WARNING [config] Config file /x/config.yml is readable by other users - "
        f'chmod 600 /x/config.yml during="Reading config" placed=frontier'
    )


def test_top_level_diagnostic_has_no_during_tail() -> None:
    line = _format(Diagnostic(severity=Severity.ERROR, message="boom", origin="runlock"))
    assert line == f"{_TS} ERROR [runlock] boom"


def test_handle_demoted_diagnostic_never_claims_frontier_placement() -> None:
    diag = Diagnostic(
        severity=Severity.INFO,
        message="x [after entry 'T' closed]",
        origin="output.late.entry",
        placed_by=PlacedBy.HANDLE,
    )
    line = _format(diag, *_entry_context())
    assert line == f"{_TS} INFO [output.late.entry] x [after entry 'T' closed]"


def test_diagnostic_trace_appends_the_full_plain_traceback() -> None:
    try:
        raise ValueError("request failed")
    except ValueError as exc:
        trace = CapturedTrace.from_exception(exc)
    diag = Diagnostic(severity=Severity.ERROR, message="boom", origin="app", trace=trace)

    line = _format(diag)
    assert line is not None
    first, rest = line.split("\n", 1)
    assert first == f"{_TS} ERROR [app] boom"
    assert "ValueError: request failed" in rest


def test_pure_boundaries_and_ephemerals_have_no_text_form() -> None:
    silent: list[Event] = [
        ScopeOpened(scope=_ENTRY, label="x"),
        ScopeClosed(scope=_ENTRY),
        BootStepStarted(scope=_STEP, label="x"),
        ScanFinished(arr=Arr.SONARR),
        RunFinished(arr=Arr.SONARR),
    ]
    for event in silent:
        assert _format(event) is None


# --- the summary block -------------------------------------------------------------


def _summary_ready() -> RunSummaryReady:
    stats = RunStats(checked=182, up_to_date=161, cached=14)
    stats.queued = 1
    stats.added.append(
        GrabRecord(
            title="Frieren: Beyond Journey's End",
            coverage="S01 E01-E28",
            url="https://releases.moe/154587",
            name="[SubsPlease] Sousou no Frieren",
            group="SubsPlease",
        ),
    )
    stats.needs_action.append(
        NeedsActionRecord(
            title="Monogatari",
            coverage=None,
            group="Okay-Subs",
            url="https://releases.moe/98765",
            reason="private tracker",
            kind=NeedsActionKind.PRIVATE_ONLY,
        ),
    )
    return RunSummaryReady(
        RunSummary(
            arr=Arr.SONARR,
            dry_run=False,
            dry_run_note=None,
            added_count=1,
            tally=RunTally.from_stats(stats),
            wait_mode_on=True,
            warnings=2,
            errors=0,
            elapsed_s=401.0,
            tip=NeedsActionCause.PRIVATE_ONLY,
        ),
    )


def _quiet_summary_ready() -> RunSummaryReady:
    """A summary with nothing actionable and no errors (stays plain INFO)."""

    return RunSummaryReady(
        RunSummary(
            arr=Arr.SONARR,
            dry_run=False,
            dry_run_note=None,
            added_count=0,
            tally=RunTally.from_stats(RunStats()),
            wait_mode_on=False,
            warnings=0,
            errors=0,
            elapsed_s=None,
            tip=None,
        ),
    )


def test_summary_renders_one_head_line_plus_one_line_per_record() -> None:
    text = _format(_summary_ready())
    assert text == "\n".join(
        [
            f"{_TS} INFO [summary] run complete arr=sonarr checked=182 needs_action=1 added=1 "
            f"queued=1 up_to_date=161 cached=14 warnings=2 errors=0 elapsed_s=401.00 tip=private_only",
            f"{_TS} WARNING [summary] needs action title=Monogatari group=Okay-Subs "
            f'reason="private tracker" kind=private_only link=https://releases.moe/98765',
            f'{_TS} INFO [summary] added title="Frieren: Beyond Journey\'s End" files="S01 E01-E28" '
            f'group=SubsPlease torrent="[SubsPlease] Sousou no Frieren" link=https://releases.moe/154587',
        ],
    )


# --- sinks: thresholds, routing, parity, rotation --------------------------------------


def _line_sink() -> tuple[LineRenderer, io.StringIO]:
    stream = io.StringIO()
    return LineRenderer(stream), stream


def test_file_sink_honors_the_configured_level_but_still_folds(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.set_level(logging.WARNING)

    for event in _entry_context():
        sink.handle(event, _EPOCH)  # all INFO: filtered, but the fold must still advance
    sink.handle(Diagnostic(severity=Severity.WARNING, message="late warning", origin="anilist"), _EPOCH)
    sink.close()

    content = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert content == (
        f'{_TS} WARNING [anilist] late warning during="Frieren: Beyond Journey\'s End" placed=frontier\n'
    )


def test_a_failed_graduation_admits_through_a_warning_level_sink(tmp_path: Path) -> None:
    # P6: severity_of(TorrentGraduated) is category-based, so a FAILED
    # graduation (ERROR) survives a raised file level while a success drops.
    sink = FileLogSink(str(tmp_path))
    sink.set_level(logging.WARNING)

    sink.handle(TorrentGraduated(label="Ok Show", outcome=Outcome.IMPORTED, files=1, waited_s=5.0), _EPOCH)
    sink.handle(TorrentGraduated(label="Bad Show", outcome=Outcome.DOWNLOAD_ERRORED, files=None, waited_s=5.0), _EPOCH)
    sink.close()

    content = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert "Ok Show" not in content
    assert f'{_TS} ERROR [wait] errored title="Bad Show" waited_s=5.00\n' in content


def test_a_warning_file_level_admits_per_line_within_the_summary(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.set_level(logging.WARNING)

    sink.handle(_summary_ready(), _EPOCH)
    sink.close()

    # Line-granular admission: the WARNING needs-action row lands; the INFO
    # head line and the INFO added row drop.
    lines = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"{_TS} WARNING [summary] needs action title=Monogatari group=Okay-Subs "
        f'reason="private tracker" kind=private_only link=https://releases.moe/98765',
    ]


def test_file_only_diagnostics_reach_the_file_but_not_stdout(tmp_path: Path) -> None:
    note = Diagnostic(severity=Severity.WARNING, message="renderer failed", origin="output.hub", file_only=True)
    line_sink, stream = _line_sink()
    file_sink = FileLogSink(str(tmp_path))

    line_sink.handle(note, _EPOCH)
    file_sink.handle(note, _EPOCH)
    file_sink.close()

    assert stream.getvalue() == ""
    assert "renderer failed" in (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")


def test_line_and_file_output_are_byte_identical(tmp_path: Path) -> None:
    events: list[Event] = [
        RunStarted(version="v1.0.0", data_dir="/data"),
        *_entry_context(),
        EntryHeader(state=EntryState.CHECKING, title="Frieren", scope=_ENTRY),
        Diagnostic(severity=Severity.WARNING, message="rate limited", origin="anilist"),
        ScanFinished(arr=Arr.SONARR),
        _summary_ready(),
        RunFinished(arr=Arr.SONARR),
    ]
    line_sink, stream = _line_sink()
    file_sink = FileLogSink(str(tmp_path))

    for event in events:
        line_sink.handle(event, _EPOCH)
        file_sink.handle(event, _EPOCH)
    file_sink.close()

    assert stream.getvalue() == (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert stream.getvalue() != ""


def test_line_and_file_parity_holds_at_a_warning_level(tmp_path: Path) -> None:
    # plain == file is structural at EVERY level (raw set_level, S4): the sole
    # divergence is the file_only carve-out.
    events: list[Event] = [
        Diagnostic(severity=Severity.WARNING, message="rate limited", origin="anilist"),
        *_entry_context(),  # INFO scan events: dropped by both surfaces
        Diagnostic(severity=Severity.WARNING, message="renderer failed", origin="output.hub", file_only=True),
    ]
    line_sink, stream = _line_sink()
    file_sink = FileLogSink(str(tmp_path))
    line_sink.set_level(logging.WARNING)
    file_sink.set_level(logging.WARNING)

    for event in events:
        line_sink.handle(event, _EPOCH)
        file_sink.handle(event, _EPOCH)
    file_sink.close()

    file_lines = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8").splitlines(keepends=True)
    assert sum("renderer failed" in line for line in file_lines) == 1
    # stdout bytes == file bytes minus exactly the file_only line.
    assert stream.getvalue() == "".join(line for line in file_lines if "renderer failed" not in line)
    assert stream.getvalue() != ""


def test_file_rotation_mirrors_the_log_cascade(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))

    sink.handle(RunStarted(version="one", data_dir="/d"), _EPOCH)
    sink.begin_cycle()
    sink.handle(RunStarted(version="two", data_dir="/d"), _EPOCH)
    sink.begin_cycle()
    sink.handle(RunStarted(version="three", data_dir="/d"), _EPOCH)
    sink.close()

    assert "three" in (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert "two" in (tmp_path / "SeaDexArr.log.1").read_text(encoding="utf-8")
    assert "one" in (tmp_path / "SeaDexArr.log.2").read_text(encoding="utf-8")


def test_an_idle_cycle_never_churns_the_cascade(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.begin_cycle()
    sink.begin_cycle()  # no writes in between: nothing to rotate
    sink.handle(RunStarted(version="only", data_dir="/d"), _EPOCH)
    sink.close()

    assert (tmp_path / "SeaDexArr.log").exists()
    assert not (tmp_path / "SeaDexArr.log.1").exists()


def test_pre_cycle_writes_append_and_rotation_fires_once_per_begin_cycle(tmp_path: Path) -> None:
    # Only begin_cycle arms rotation — construction must not, or a record in the
    # install→begin_cycle window would rotate twice and strand a one-line .log.1.
    (tmp_path / "SeaDexArr.log").write_text("previous run\n", encoding="utf-8")
    sink = FileLogSink(str(tmp_path))

    sink.handle(RunStarted(version="pre-cycle", data_dir="/d"), _EPOCH)
    assert not (tmp_path / "SeaDexArr.log.1").exists()  # appended, no rotation
    log_text = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert log_text.startswith("previous run\n") and "pre-cycle" in log_text

    sink.begin_cycle()
    sink.handle(RunStarted(version="cycle-one", data_dir="/d"), _EPOCH)
    sink.close()

    rotated = (tmp_path / "SeaDexArr.log.1").read_text(encoding="utf-8")
    assert "previous run" in rotated and "pre-cycle" in rotated  # rotated as ONE file
    assert not (tmp_path / "SeaDexArr.log.2").exists()  # exactly one rotation
    assert "cycle-one" in (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")


def test_a_reopened_file_sink_appends_after_close(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.handle(RunStarted(version="one", data_dir="/d"), _EPOCH)
    sink.close()
    sink.handle(RunStarted(version="two", data_dir="/d"), _EPOCH)
    sink.close()

    # Reopen without a pending rotation appends - never a silent truncate.
    content = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert "one" in content
    assert "two" in content
    assert not (tmp_path / "SeaDexArr.log.1").exists()


def test_every_line_is_on_disk_immediately(tmp_path: Path) -> None:
    # Per-line flush (crash fidelity): the tail is on disk before close or any WARNING.
    sink = FileLogSink(str(tmp_path))
    log = tmp_path / "SeaDexArr.log"

    sink.handle(RunStarted(version="v1.0.0", data_dir="/d"), _EPOCH)
    assert "SeaDexArr started" in log.read_text(encoding="utf-8")

    sink.handle(Diagnostic(severity=Severity.WARNING, message="crash imminent", origin="app"), _EPOCH)
    assert "crash imminent" in log.read_text(encoding="utf-8")
    sink.close()


def test_probe_leaves_no_file_behind_and_keeps_existing_content(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.probe()
    assert not (tmp_path / "SeaDexArr.log").exists()  # the probe's own file is removed

    (tmp_path / "SeaDexArr.log").write_text("previous run\n", encoding="utf-8")
    sink.probe()
    assert (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8") == "previous run\n"


def test_probe_raises_on_an_unwritable_log_file(tmp_path: Path) -> None:
    # The Docker shape: the DIRECTORY is writable, the file (another uid's) is not.
    log = tmp_path / "SeaDexArr.log"
    log.write_text("root-owned\n", encoding="utf-8")
    log.chmod(0o400)
    sink = FileLogSink(str(tmp_path))
    try:
        with pytest.raises(OSError):
            sink.probe()
    finally:
        log.chmod(0o644)


def test_file_sink_writes_utf8(tmp_path: Path) -> None:
    sink = FileLogSink(str(tmp_path))
    sink.handle(
        EntryHeader(state=EntryState.CHECKING, title="Pokémon — ポケモン", scope=None),
        _EPOCH,
    )
    sink.close()

    content = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert "Pokémon — ポケモン" in content


def test_wait_progress_stays_silent_below_the_pulse_cadence() -> None:
    line_sink, stream = _line_sink()
    line_sink.handle(WaitStarted(total=1, pulse_s=300.0, scope=None), _EPOCH)
    line_sink.handle(Diagnostic(severity=Severity.DEBUG, message="hidden", origin="app"), _EPOCH)
    before = stream.getvalue()

    line_sink.handle(WaitProgress(snapshot=WaitSnapshot(torrents=(), elapsed_s=1.0)), _EPOCH)
    assert stream.getvalue() == before
    assert "hidden" not in before  # DEBUG below the default INFO floor


# --- the throttled "still waiting" pulse ---------------------------------------------


def _wait_progress(elapsed: float) -> WaitProgress:
    torrents = (
        TorrentView(key="a", label="A", phase=Phase.DOWNLOADING, fraction=0.5),
        TorrentView(key="b", label="B", phase=Phase.IMPORTING),
        TorrentView(key="c", label="C", phase=Phase.QUEUED),
    )
    return WaitProgress(snapshot=WaitSnapshot(torrents=torrents, elapsed_s=elapsed), scope=_WAIT)


def test_grammar_sinks_pulse_still_waiting_on_the_event_cadence() -> None:
    # The anti-hang heartbeat: cadence is pure event content (pulse_s/elapsed_s,
    # never wall clock), and the start snapshot never pulses.
    line_sink, stream = _line_sink()
    context = (ScanStarted(arr=Arr.SONARR, total=182), ScopeOpened(scope=_WAIT, label="wait"))
    for event in (*context, WaitStarted(total=3, pulse_s=300.0, scope=_WAIT)):
        line_sink.handle(event, _EPOCH)
    line_sink.handle(_wait_progress(0.0), _EPOCH)  # the start snapshot never pulses
    line_sink.handle(_wait_progress(299.0), _EPOCH)  # within the interval
    before = stream.getvalue()

    line_sink.handle(_wait_progress(300.0), _EPOCH)  # due at pulse_s
    assert stream.getvalue() == before + (
        f"{_TS} INFO [sonarr › wait] still waiting downloading=1 importing=1 queued=1 elapsed_s=300.00\n"
    )


def test_pulse_lines_keep_line_and_file_byte_parity(tmp_path: Path) -> None:
    events: list[Event] = [
        ScanStarted(arr=Arr.SONARR, total=182),
        ScopeOpened(scope=_WAIT, label="wait"),
        WaitStarted(total=3, pulse_s=300.0, scope=_WAIT),
        _wait_progress(0.0),
        _wait_progress(300.0),
        _wait_progress(650.0),
        WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=700.0, scope=_WAIT),
    ]
    line_sink, stream = _line_sink()
    file_sink = FileLogSink(str(tmp_path))
    for event in events:
        line_sink.handle(event, _EPOCH)
        file_sink.handle(event, _EPOCH)
    file_sink.close()

    content = (tmp_path / "SeaDexArr.log").read_text(encoding="utf-8")
    assert stream.getvalue() == content
    assert content.count("still waiting") == 2  # 300 and 650 pulse; the start snapshot never does


def test_a_new_wait_pass_rearms_the_pulse_skip_first() -> None:
    line_sink, stream = _line_sink()
    line_sink.handle(WaitStarted(total=1, pulse_s=300.0), _EPOCH)
    line_sink.handle(_wait_progress(300.0), _EPOCH)  # skip-first: even a late start snapshot never pulses
    line_sink.handle(_wait_progress(300.0), _EPOCH)  # due

    line_sink.handle(WaitStarted(total=1, pulse_s=300.0), _EPOCH)  # a second pass this run
    line_sink.handle(_wait_progress(900.0), _EPOCH)  # skip-first again
    line_sink.handle(_wait_progress(900.0), _EPOCH)  # due again

    assert stream.getvalue().count("still waiting") == 2


def test_begin_cycle_disarms_the_pulse_until_the_next_wait_pass() -> None:
    line_sink, stream = _line_sink()
    line_sink.handle(WaitStarted(total=1, pulse_s=300.0), _EPOCH)
    line_sink.begin_cycle()

    line_sink.handle(_wait_progress(10_000.0), _EPOCH)
    line_sink.handle(_wait_progress(10_000.0), _EPOCH)  # disarmed: not even a late fire
    assert "still waiting" not in stream.getvalue()


def test_pulse_lines_drop_below_a_raised_threshold_but_the_cadence_advances() -> None:
    # Line-granular admission applies to the pulse like any INFO line; the
    # throttle state advances regardless of level (the PulseThrottle contract).
    line_sink, stream = _line_sink()
    line_sink.set_level(logging.WARNING)
    line_sink.handle(WaitStarted(total=1, pulse_s=300.0), _EPOCH)
    line_sink.handle(_wait_progress(0.0), _EPOCH)
    line_sink.handle(_wait_progress(300.0), _EPOCH)  # due, but INFO drops at WARNING
    assert stream.getvalue() == ""

    line_sink.set_level(logging.INFO)
    line_sink.handle(_wait_progress(400.0), _EPOCH)  # the dropped pulse still re-armed at 600
    assert stream.getvalue() == ""
    line_sink.handle(_wait_progress(600.0), _EPOCH)
    assert "still waiting" in stream.getvalue()


class _ExplodingStream(io.StringIO):
    """A stream whose write raises while armed (the render-bug stand-in)."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    @override
    def write(self, s: str, /) -> int:
        if self.armed:
            raise RuntimeError("stream gone")
        return super().write(s)


def test_the_fold_advances_even_when_the_write_blows_up() -> None:
    stream = _ExplodingStream()
    sink = LineRenderer(stream)
    scan, item, opened = _entry_context()
    with pytest.raises(RuntimeError):
        sink.handle(scan, _EPOCH)
    with pytest.raises(RuntimeError):
        sink.handle(item, _EPOCH)
    sink.handle(opened, _EPOCH)  # no text form: folds without writing

    stream.armed = False
    sink.handle(Diagnostic(severity=Severity.WARNING, message="late", origin="app"), _EPOCH)
    assert stream.getvalue() == (f'{_TS} WARNING [app] late during="Frieren: Beyond Journey\'s End" placed=frontier\n')


# --- the json surface -------------------------------------------------------------------


def _json_lines(events: list[Event]) -> list[dict[str, Json]]:
    stream = io.StringIO()
    renderer = JsonRenderer(stream)
    for event in events:
        renderer.handle(event, _EPOCH)
    lines: list[dict[str, Json]] = []
    for raw in stream.getvalue().splitlines():
        payload: Json = json.loads(raw)
        assert is_json_obj(payload)
        lines.append(dict[str, Json](payload))
    return lines


def test_json_emits_one_object_per_event_with_stable_key_order() -> None:
    (payload,) = _json_lines([RunStarted(version="v1.0.0", data_dir="/data")])
    assert list(payload) == ["time", "event", "level", "message", "version", "data_dir"]
    assert payload["event"] == "run_started"
    assert payload["level"] == "INFO"
    assert payload["message"] == "SeaDexArr started"


def test_json_time_carries_a_utc_offset() -> None:
    (payload,) = _json_lines([RunStarted(version="", data_dir="/d")])
    time_value = payload["time"]
    assert isinstance(time_value, str)
    offset_part = time_value[10:]
    assert "T" in time_value
    assert "+" in offset_part or "-" in offset_part


def test_json_diagnostic_shape_with_placement_and_trace() -> None:
    try:
        raise ValueError("x")
    except ValueError as exc:
        trace = CapturedTrace.from_exception(exc)
    events: list[Event] = [
        *_entry_context(),
        Diagnostic(severity=Severity.WARNING, message="rate limited", origin="anilist", trace=trace),
    ]
    payloads = _json_lines(events)
    diag = payloads[-1]
    assert list(diag) == ["time", "event", "level", "message", "origin", "during", "placed", "exc"]
    assert diag["level"] == "WARNING"
    assert diag["origin"] == "anilist"
    assert diag["placed"] == "frontier"
    exc_text = diag["exc"]
    assert isinstance(exc_text, str)
    assert "ValueError" in exc_text


def test_json_includes_scope_boundaries_but_not_ephemerals() -> None:
    events: list[Event] = [
        ScopeOpened(scope=_ENTRY, label="Frieren"),
        BootStepStarted(scope=_STEP, label="x"),
        WaitProgress(snapshot=WaitSnapshot(torrents=(), elapsed_s=1.0)),
        ScopeClosed(scope=_ENTRY),
        RunFinished(arr=Arr.SONARR),
    ]
    payloads = _json_lines(events)
    assert [p["event"] for p in payloads] == ["scope_opened", "scope_closed", "run_finished"]
    assert payloads[0]["kind"] == "entry"
    assert payloads[0]["label"] == "Frieren"


def test_json_summary_carries_nested_record_arrays() -> None:
    (payload,) = _json_lines([_summary_ready()])
    assert payload["event"] == "run_summary"
    assert payload["checked"] == 182
    needs = payload["needs_action_records"]
    assert is_json_list(needs)
    first_need = needs[0]
    assert is_json_obj(first_need)
    assert first_need["kind"] == "private_only"
    assert first_need["reason"] == "private tracker"
    added = payload["added_records"]
    assert is_json_list(added)
    first_added = added[0]
    assert is_json_obj(first_added)
    assert first_added["group"] == "SubsPlease"
    assert first_added["torrent"] == "[SubsPlease] Sousou no Frieren"


def test_json_scoped_events_carry_their_breadcrumb_path() -> None:
    events: list[Event] = [
        *_entry_context(),
        EntryHeader(state=EntryState.CHECKING, title="Frieren", scope=_ENTRY),
    ]
    payloads = _json_lines(events)
    header = payloads[-1]
    assert header["event"] == "entry_header"
    assert header["path"] == "sonarr › [3/182] Frieren › entry"


def _exemplars() -> list[Event]:
    """One exemplar of EVERY union member (pinned against the union itself below)."""

    try:
        raise ValueError("x")
    except ValueError as exc:
        trace = CapturedTrace.from_exception(exc)
    return [
        RunStarted(version="v1.0.0", data_dir="/d"),
        CycleStarted(number=1),
        NextRunScheduled(at=_WHEN),
        ScopeOpened(scope=_ENTRY, label="T"),
        ScopeClosed(scope=_ENTRY),
        BootStepStarted(scope=_STEP, label="Reading config"),
        BootStepProgressed(scope=_STEP, fraction=0.5),
        BootStepSlow(scope=_STEP, label="Reading config"),
        BootStepFinished(
            scope=_STEP,
            label="Reading config",
            outcome=OutcomeCategory.SUCCESS,
            detail=None,
            elapsed_s=0.1,
        ),
        BootReady(elapsed_s=7.0),
        ScanStarted(arr=Arr.SONARR, total=1),
        ItemStarted(arr=Arr.SONARR, index=1, total=1, title="T"),
        EntryHeader(state=EntryState.CHECKING, title="T"),
        EntryDetail(label="status", value=StyledValue("no suitable releases")),
        LedgerRow(state=EntryState.IGNORED, label="AniList #1"),
        ReleaseSkipped(group="G", tracker="AB", reason=SkipReason.PRIVATE_ONLY),
        GrabFailed(group="G", url="u", error="e"),
        GrabAction(status=GrabStatus.ADDING, groups=(), added=(), downloading=()),
        CapReached(cap=25),
        ScanFinished(arr=Arr.SONARR),
        _summary_ready(),
        WaitStarted(total=1, pulse_s=300.0),
        WaitProgress(snapshot=WaitSnapshot(torrents=(), elapsed_s=1.0)),
        TorrentGraduated(label="T", outcome=Outcome.IMPORTED, files=1, waited_s=1.0),
        WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=1.0),
        RunFinished(arr=Arr.SONARR),
        Diagnostic(severity=Severity.INFO, message="m", origin="app", trace=trace),
    ]


# Events with no format_line form: pure boundaries + per-event-fact-less ephemera
# (WaitProgress's text form is the grammar sinks' throttled pulse, tested above).
_TEXT_SILENT = {ScopeOpened, ScopeClosed, BootStepStarted, BootStepProgressed, WaitProgress, ScanFinished, RunFinished}
# Events the json stream drops: ephemera (json keeps the boundaries).
_JSON_SILENT = {BootStepStarted, BootStepProgressed, BootStepSlow, WaitProgress}


def test_every_event_type_has_a_decided_form_on_both_surfaces() -> None:
    exemplars = _exemplars()
    members: set[object] = set(get_args(Event.__value__))
    assert {type(event) for event in exemplars} == members  # the sweep can't rot

    for event in exemplars:
        text = _format(event)
        assert (text is None) == (type(event) in _TEXT_SILENT), type(event).__name__

        stream = io.StringIO()
        JsonRenderer(stream).handle(event, _EPOCH)
        assert (stream.getvalue() == "") == (type(event) in _JSON_SILENT), type(event).__name__


def test_skip_sets_are_event_members_pinned_to_their_current_membership() -> None:
    """The source skip sets are hand-maintained subsets of the Event union: (a) a
    renamed/removed event breaks the subset check loudly (import still succeeds, but
    get_args no longer holds it); (b) the exact membership is pinned, so adding or
    dropping a skip is a conscious, reviewed change — not a silent behavior shift."""

    union_members: set[object] = set(get_args(Event.__value__))
    assert _TEXT_SKIP <= union_members
    assert _JSON_SKIP <= union_members
    assert _TEXT_SKIP == {ScopeOpened, ScopeClosed, ScanFinished, RunFinished}
    assert _JSON_SKIP == {BootStepSlow}


def test_json_set_level_applies_the_raw_configured_level() -> None:
    # Raw S4 semantics: DEBUG lowers the floor, WARNING raises it past INFO.
    stream = io.StringIO()
    renderer = JsonRenderer(stream)
    renderer.set_level(logging.DEBUG)
    renderer.handle(Diagnostic(severity=Severity.DEBUG, message="verbose", origin="httpx"), _EPOCH)
    assert "verbose" in stream.getvalue()

    quiet = io.StringIO()
    quiet_renderer = JsonRenderer(quiet)
    quiet_renderer.set_level(logging.WARNING)
    quiet_renderer.handle(Diagnostic(severity=Severity.INFO, message="routine", origin="app"), _EPOCH)
    assert quiet.getvalue() == ""


def test_json_summary_admission_is_data_dependent_at_a_warning_threshold() -> None:
    stream = io.StringIO()
    renderer = JsonRenderer(stream)
    renderer.set_level(logging.WARNING)
    renderer.handle(_summary_ready(), _EPOCH)
    assert "run_summary" in stream.getvalue()  # needs-action content admits

    quiet = io.StringIO()
    quiet_renderer = JsonRenderer(quiet)
    quiet_renderer.set_level(logging.WARNING)
    quiet_renderer.handle(_quiet_summary_ready(), _EPOCH)
    assert quiet.getvalue() == ""  # nothing actionable: the routine summary drops
