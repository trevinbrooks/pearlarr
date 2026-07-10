# pyright: strict
"""Tests for the shared breadcrumb fold (``output.breadcrumbs``).

Pin the fixed transition table (B6): explicit opens/closes, the boundary events'
deterministic close-deeper behavior, unwind close-all, idempotent defensive
closes, and the label-only read APIs (path text, path_for, during).
"""

from seadexarr.modules.config import Arr
from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import OutcomeCategory
from seadexarr.modules.output import (
    KIND_DEPTH,
    BootStepFinished,
    BootStepStarted,
    BreadcrumbFold,
    CycleStarted,
    Diagnostic,
    EntryHeader,
    Event,
    ItemStarted,
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
    WaitStarted,
)
from seadexarr.modules.reporter import RunStats

_BOOT = ScopeId(ScopeKind.BOOT_SECTION, 1)
_STEP = ScopeId(ScopeKind.BOOT_STEP, 2)
_ENTRY = ScopeId(ScopeKind.ENTRY, 3)
_WAIT = ScopeId(ScopeKind.WAIT_REGION, 4)


def _fold(*events: Event) -> BreadcrumbFold:
    fold = BreadcrumbFold()
    for event in events:
        fold.apply(event)
    return fold


def _kinds(fold: BreadcrumbFold) -> list[ScopeKind]:
    return [node.kind for node in fold.nodes()]


def _summary_ready() -> RunSummaryReady:
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


# --- explicit opens/closes -----------------------------------------------------------


def test_scope_opened_pushes_and_scope_closed_pops() -> None:
    fold = _fold(ScopeOpened(scope=_BOOT, label="boot"))
    assert fold.path_text() == "boot"
    assert fold.during() == "boot"

    fold.apply(ScopeClosed(scope=_BOOT))
    assert fold.path_text() == ""
    assert fold.during() is None


def test_boot_step_events_open_and_close_the_step_node() -> None:
    fold = _fold(
        ScopeOpened(scope=_BOOT, label="boot"),
        BootStepStarted(scope=_STEP, label="Reading config"),
    )
    assert fold.path_text() == "boot › Reading config"
    assert fold.during() == "Reading config"

    fold.apply(
        BootStepFinished(
            scope=_STEP,
            label="Reading config",
            outcome=OutcomeCategory.SUCCESS,
            detail=None,
            elapsed_s=0.1,
        ),
    )
    # Between steps the BOOT node still spans the gap (B2: the motivating warnings).
    assert fold.path_text() == "boot"


def test_a_second_step_auto_closes_a_leaked_sibling() -> None:
    other = ScopeId(ScopeKind.BOOT_STEP, 9)
    fold = _fold(
        ScopeOpened(scope=_BOOT, label="boot"),
        BootStepStarted(scope=_STEP, label="one"),
        BootStepStarted(scope=other, label="two"),
    )
    assert fold.path_text() == "boot › two"


def test_entry_path_reads_run_item_entry() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=3, total=182, title="Frieren"),
        ScopeOpened(scope=_ENTRY, label="Frieren: Beyond Journey's End"),
    )
    assert fold.path_text() == "sonarr › [3/182] Frieren › entry"
    assert fold.during() == "Frieren: Beyond Journey's End"


# --- boundary events close strictly-deeper nodes deterministically --------------------


def test_scan_started_closes_the_boot_section() -> None:
    fold = _fold(
        ScopeOpened(scope=_BOOT, label="boot"),
        ScanStarted(arr=Arr.SONARR, total=182),
    )
    assert _kinds(fold) == [ScopeKind.RUN]
    assert fold.path_text() == "sonarr"


def test_item_started_closes_the_previous_item_and_entry() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=1, total=182, title="A"),
        ScopeOpened(scope=_ENTRY, label="A"),
        ItemStarted(arr=Arr.SONARR, index=2, total=182, title="B"),
    )
    assert _kinds(fold) == [ScopeKind.RUN, ScopeKind.ITEM]
    assert fold.path_text() == "sonarr › [2/182] B"


def test_scan_finished_closes_item_and_entry_but_keeps_the_run_node() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=182, total=182, title="Last"),
        ScopeOpened(scope=_ENTRY, label="Last"),
        ScanFinished(arr=Arr.SONARR),
    )
    # Reconcile-time diagnostics render at run level, never inside series #182 (B4.2).
    assert _kinds(fold) == [ScopeKind.RUN]


def test_run_summary_ready_closes_to_run_level() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=1),
        ItemStarted(arr=Arr.SONARR, index=1, total=1, title="A"),
        _summary_ready(),
    )
    assert _kinds(fold) == [ScopeKind.RUN]


def test_wait_started_closes_only_entry_depth() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=1),
        ScopeOpened(scope=_WAIT, label="wait"),
        WaitStarted(total=4, pulse_s=300.0, scope=_WAIT),
    )
    assert _kinds(fold) == [ScopeKind.RUN, ScopeKind.WAIT_REGION]
    assert fold.path_text() == "sonarr › wait"


def test_run_finished_closes_everything_and_is_idempotent() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=1),
        ItemStarted(arr=Arr.SONARR, index=1, total=1, title="A"),
        ScopeOpened(scope=_ENTRY, label="A"),
        RunFinished(arr=Arr.SONARR),
    )
    assert fold.nodes() == ()

    # The defensive teardown close (B3) re-emits it; a no-op on an empty stack.
    fold.apply(RunFinished(arr=Arr.SONARR))
    assert fold.nodes() == ()


def test_run_and_cycle_starts_close_everything() -> None:
    fold = _fold(ScopeOpened(scope=_BOOT, label="boot"), RunStarted(version="", data_dir="/d"))
    assert fold.nodes() == ()

    fold = _fold(ScanStarted(arr=Arr.SONARR, total=1), CycleStarted(number=2))
    assert fold.nodes() == ()


# --- unwind + defensive closes ---------------------------------------------------------


def test_closing_a_mid_stack_scope_closes_everything_deeper() -> None:
    fold = _fold(
        ScopeOpened(scope=_BOOT, label="boot"),
        BootStepStarted(scope=_STEP, label="Fetching library"),
        ScopeClosed(scope=_BOOT),
    )
    assert fold.nodes() == ()


def test_closing_an_unknown_scope_is_a_no_op() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=1),
        ScopeClosed(scope=ScopeId(ScopeKind.ENTRY, 99)),
    )
    assert _kinds(fold) == [ScopeKind.RUN]


def test_content_events_never_move_the_fold() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=1),
        ItemStarted(arr=Arr.SONARR, index=1, total=1, title="A"),
        ScopeOpened(scope=_ENTRY, label="A"),
    )
    entry_nodes = fold.nodes()

    fold.apply(Diagnostic(severity=Severity.WARNING, message="w"))
    fold.apply(EntryHeader(state=EntryState.CHECKING, title="A", scope=_ENTRY))
    assert fold.nodes() == entry_nodes


# --- read APIs ---------------------------------------------------------------------------


def test_path_for_returns_the_breadcrumb_down_to_that_scope() -> None:
    fold = _fold(
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=3, total=182, title="Frieren"),
        ScopeOpened(scope=_ENTRY, label="Frieren"),
    )
    assert fold.path_for(_ENTRY) == "sonarr › [3/182] Frieren › entry"
    assert fold.path_for(ScopeId(ScopeKind.ENTRY, 99)) is None


def test_reset_clears_all_open_nodes() -> None:
    fold = _fold(ScanStarted(arr=Arr.SONARR, total=1))
    fold.reset()
    assert fold.nodes() == ()
    assert fold.path_text() == ""


# --- dual-list drift pin: KIND_DEPTH <-> ScopeKind -----------------------------------


def test_kind_depth_covers_every_scope_kind_with_no_stray_keys() -> None:
    """KIND_DEPTH and ScopeKind are hand-maintained twins the fold keys on (_push /
    _close_at do ``KIND_DEPTH[kind]``): a new kind missing a depth would KeyError,
    and a stray key would rot unnoticed."""

    assert set(KIND_DEPTH) == set(ScopeKind)
