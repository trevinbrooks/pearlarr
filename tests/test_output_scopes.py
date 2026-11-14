# pyright: strict
"""Tests for the scope handles (`output.scopes`).

Pin the handle lifecycles (step timing/outcomes, title header-at-open, wait
open/close ordering), ScopeId tagging, and the runtime-total closed-handle
demotion: a late emission becomes an attributed Diagnostic (placed_by=HANDLE),
never a raise and never a mispositioned structured event.
"""

import pytest

from pearlarr.output import (
    Accent,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    Diagnostic,
    Event,
    GrabAction,
    GrabFailed,
    GrabStatus,
    PlacedBy,
    ReleaseSkipped,
    RowFinished,
    ScopeClosed,
    ScopeFactory,
    ScopeIdCounter,
    ScopeKind,
    ScopeOpened,
    Severity,
    StyledValue,
    TitleDetail,
    TitleHeader,
    TitleRow,
    WaitFinished,
    WaitKind,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
)
from pearlarr.output.events import TitleState
from pearlarr.releases import SkipReason
from pearlarr.wait_types import Outcome, OutcomeKind

from .fakes import FakeClock


class _Recorder:
    """A bare Emit target: just the ordered event list, no hub behavior."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of_type[E](self, cls: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, cls)]


def _factory() -> tuple[ScopeFactory, _Recorder, FakeClock]:
    recorder = _Recorder()
    clock = FakeClock()
    # A fresh minter per test keeps serials deterministic (production shares one).
    return ScopeFactory(recorder, clock=clock, ids=ScopeIdCounter()), recorder, clock


# --- ScopeIdCounter ---------------------------------------------------------------------


def test_scope_ids_mint_unique_increasing_serials() -> None:
    ids = ScopeIdCounter()
    a = ids.mint(ScopeKind.BOOT_STEP)
    b = ids.mint(ScopeKind.TITLE)
    assert (a.kind, a.serial) == (ScopeKind.BOOT_STEP, 1)
    assert (b.kind, b.serial) == (ScopeKind.TITLE, 2)


def test_two_default_factories_never_mint_colliding_serials() -> None:
    first, second = ScopeFactory(_Recorder()), ScopeFactory(_Recorder())

    a = first.step("one").scope_id
    b = second.step("two").scope_id

    assert a.serial != b.serial  # both use the process-wide minter


# --- StepScope --------------------------------------------------------------------


def test_step_times_and_finishes_success_with_detail() -> None:
    factory, recorder, clock = _factory()

    with factory.step("Fetching Sonarr library") as step:
        clock.tick(1.2)
        step.note("42 series")

    started = recorder.of_type(BootStepStarted)
    finished = recorder.of_type(BootStepFinished)
    assert [s.label for s in started] == ["Fetching Sonarr library"]
    assert finished == [
        BootStepFinished(
            scope=started[0].scope,
            label="Fetching Sonarr library",
            outcome=OutcomeKind.IMPORTED,
            detail="42 series",
            elapsed_s=1.2,
        ),
    ]


def test_step_warn_finishes_deferred() -> None:
    factory, recorder, _ = _factory()

    with factory.step("Refreshing mappings") as step:
        step.warn("SeaDex unreachable")

    (finished,) = recorder.of_type(BootStepFinished)
    assert finished.outcome is OutcomeKind.LEFT
    assert finished.detail == "SeaDex unreachable"


def test_step_failure_finishes_failed_and_reraises() -> None:
    factory, recorder, _ = _factory()

    with pytest.raises(ValueError, match="boom"), factory.step("Opening cache"):
        raise ValueError("boom")

    (finished,) = recorder.of_type(BootStepFinished)
    assert finished.outcome is OutcomeKind.FAILED


def test_step_progress_emits_the_slow_heads_up_exactly_once() -> None:
    factory, recorder, _ = _factory()

    with factory.step("Refreshing mappings") as step:
        step.progress(0.25, "anime-ids")
        step.progress(1.7)

    assert [s.label for s in recorder.of_type(BootStepSlow)] == ["Refreshing mappings"]
    progressed = recorder.of_type(BootStepProgressed)
    # Fractions clamp to 0-1. The last detail sticks.
    assert [(p.fraction, p.detail) for p in progressed] == [(0.25, "anime-ids"), (1.0, "anime-ids")]


def test_step_finish_is_idempotent() -> None:
    factory, recorder, _ = _factory()

    step = factory.step("Reading config")
    step.finish()
    step.finish()

    assert len(recorder.of_type(BootStepFinished)) == 1


def test_late_step_calls_demote_to_attributed_diagnostics() -> None:
    factory, recorder, _ = _factory()
    step = factory.step("Reading config")
    step.finish()

    step.note("too late")
    step.warn("way too late")
    step.progress(0.5)

    assert len(recorder.of_type(BootStepFinished)) == 1
    late = recorder.of_type(Diagnostic)
    assert [d.origin for d in late] == ["output.late.step"] * 3
    assert all(d.placed_by is PlacedBy.HANDLE for d in late)
    assert late[0].message == "note: too late [after step 'Reading config' closed]"
    assert late[1].severity is Severity.WARNING


# --- TitleScope --------------------------------------------------------------------


def test_title_opens_with_its_header_tagged() -> None:
    factory, recorder, _ = _factory()

    title_scope = factory.title(TitleHeader(state=TitleState.CHECKING, title="Frieren", anilist_id=154587))

    opened = recorder.of_type(ScopeOpened)
    headers = recorder.of_type(TitleHeader)
    assert [o.label for o in opened] == ["Frieren"]
    assert headers[0].scope == title_scope.scope_id
    assert headers[0].anilist_id == 154587
    assert recorder.events[0] == opened[0]  # header-at-open: ScopeOpened first


def test_title_post_tags_the_scope_id() -> None:
    factory, recorder, _ = _factory()
    title_scope = factory.title(TitleHeader(state=TitleState.CHECKING, title="T"))

    title_scope.post(ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY))

    (skipped,) = recorder.of_type(ReleaseSkipped)
    assert skipped.scope == title_scope.scope_id


def test_title_close_emits_scope_closed_once() -> None:
    factory, recorder, _ = _factory()
    title_scope = factory.title(TitleHeader(state=TitleState.CHECKING, title="T"))

    title_scope.close()
    title_scope.close()

    assert [c.scope for c in recorder.of_type(ScopeClosed)] == [title_scope.scope_id]


def test_late_title_post_demotes_and_keeps_the_severity() -> None:
    factory, recorder, _ = _factory()
    title_scope = factory.title(TitleHeader(state=TitleState.CHECKING, title="Frieren"))
    title_scope.close()

    title_scope.post(
        TitleDetail(
            label="skipped",
            value=StyledValue("Okay-Subs on AB (private-only)", accent=Accent.CAUTION),
            severity=Severity.WARNING,
        ),
    )
    title_scope.post(
        GrabAction(status=GrabStatus.GRABBING, groups=(), grabbed=(), in_qbit=()),
    )

    assert len(recorder.of_type(TitleDetail)) == 0
    assert len(recorder.of_type(GrabAction)) == 0
    late = recorder.of_type(Diagnostic)
    assert [d.origin for d in late] == ["output.late.title"] * 2
    assert late[0].severity is Severity.WARNING  # a demoted warn stays counted as one
    assert "[after title 'Frieren' closed]" in late[0].message
    assert "grab action: grabbing" in late[1].message


def test_every_late_fact_kind_describes_itself() -> None:
    factory, recorder, _ = _factory()
    title_scope = factory.title(TitleHeader(state=TitleState.CHECKING, title="T"))
    title_scope.close()

    title_scope.post(TitleRow(state=TitleState.IGNORED, label="AniList #1"))
    title_scope.post(ReleaseSkipped(group="G", tracker="AB", reason=SkipReason.PRIVATE_ONLY))
    title_scope.post(GrabFailed(group="G", url="u", error="tracker down"))

    messages = [d.message for d in recorder.of_type(Diagnostic)]
    assert "ignored AniList #1" in messages[0]
    assert "release skipped: G on AB (private_only)" in messages[1]
    assert messages[1].startswith("release skipped")
    assert "grab failed: G (tracker down)" in messages[2]
    # A demoted fact keeps its natural severity (the reason-derived WARNING here).
    assert recorder.of_type(Diagnostic)[1].severity is Severity.WARNING


# --- WaitScope ---------------------------------------------------------------------


def test_wait_opens_scope_then_announces_totals() -> None:
    factory, recorder, _ = _factory()

    wait = factory.wait(total=4, progress_line_s=300.0, kind=WaitKind.BLOCKING)

    assert isinstance(recorder.events[0], ScopeOpened)
    assert recorder.events[1] == WaitStarted(
        total=4, progress_line_s=300.0, kind=WaitKind.BLOCKING, scope=wait.scope_id
    )


def test_wait_progress_and_finished_rows_are_tagged() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=1, progress_line_s=300.0, kind=WaitKind.BLOCKING)

    wait.progress(WaitSnapshot(torrents=(), elapsed_s=30.0))
    wait.row_finished(RowFinished(label="T", outcome=Outcome.IMPORTED, files=12, waited_s=243.0))

    (progress,) = recorder.of_type(WaitProgress)
    (row_finished,) = recorder.of_type(RowFinished)
    assert progress.scope == wait.scope_id
    assert row_finished.scope == wait.scope_id


def test_wait_finish_emits_tally_then_closes() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=4, progress_line_s=300.0, kind=WaitKind.BLOCKING)

    wait.finish(WaitFinished(imported=3, left=1, failed=0, elapsed_s=730.0, in_progress=0, kind=WaitKind.BLOCKING))

    (finished,) = recorder.of_type(WaitFinished)
    assert finished == WaitFinished(
        imported=3, left=1, failed=0, elapsed_s=730.0, in_progress=0, kind=WaitKind.BLOCKING, scope=wait.scope_id
    )
    assert isinstance(recorder.events[-1], ScopeClosed)


def test_late_wait_emissions_demote() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=1, progress_line_s=300.0, kind=WaitKind.BLOCKING)
    wait.close()

    wait.progress(WaitSnapshot(torrents=(), elapsed_s=1.0))
    wait.row_finished(RowFinished(label="T", outcome=Outcome.IMPORTED, files=None, waited_s=1.0))
    wait.finish(WaitFinished(imported=0, left=0, failed=0, elapsed_s=1.0, in_progress=0, kind=WaitKind.BLOCKING))

    assert len(recorder.of_type(WaitProgress)) == 0
    assert len(recorder.of_type(RowFinished)) == 0
    assert len(recorder.of_type(WaitFinished)) == 0
    late = recorder.of_type(Diagnostic)
    assert [d.origin for d in late] == ["output.late.wait"] * 3
    assert "imported T" in late[1].message


# (There is no Diagnostics handle: hub_note/hub_warn/hub_error are the
# position-free one-liners. The hub's once_key dedup mechanism is
# pinned in test_output_hub.)
