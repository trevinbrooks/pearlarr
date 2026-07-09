# pyright: strict
"""Tests for the scope handles (``output.scopes``).

Pin the handle lifecycles (step timing/outcomes, entry header-at-open, wait
open/close ordering), ScopeId stamping, and the runtime-total stale-handle
demotion: a late emission becomes an attributed Diagnostic (placed_by=HANDLE),
never a raise and never a mispositioned structured event.
"""

import pytest

from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.output import (
    Accent,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    GrabStatus,
    LedgerRow,
    PlacedBy,
    ReleaseSkipped,
    ScopeClosed,
    ScopeFactory,
    ScopeIds,
    ScopeKind,
    ScopeOpened,
    Severity,
    SkipReason,
    StyledValue,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)
from seadexarr.modules.wait_view import WaitSnapshot

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
    return ScopeFactory(recorder, clock=clock, ids=ScopeIds()), recorder, clock


# --- ScopeIds ---------------------------------------------------------------------


def test_scope_ids_mint_unique_increasing_serials() -> None:
    ids = ScopeIds()
    a = ids.mint(ScopeKind.BOOT_STEP)
    b = ids.mint(ScopeKind.ENTRY)
    assert (a.kind, a.serial) == (ScopeKind.BOOT_STEP, 1)
    assert (b.kind, b.serial) == (ScopeKind.ENTRY, 2)


def test_two_default_factories_never_mint_colliding_serials() -> None:
    first, second = ScopeFactory(_Recorder()), ScopeFactory(_Recorder())

    a = first.step("one").scope_id
    b = second.step("two").scope_id

    assert a.serial != b.serial  # both ride the process-wide minter


# --- StepScope --------------------------------------------------------------------


def test_step_times_and_graduates_success_with_detail() -> None:
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
            outcome=OutcomeCategory.SUCCESS,
            detail="42 series",
            elapsed_s=1.2,
        ),
    ]


def test_step_warn_graduates_deferred() -> None:
    factory, recorder, _ = _factory()

    with factory.step("Refreshing mappings") as step:
        step.warn("SeaDex unreachable")

    (finished,) = recorder.of_type(BootStepFinished)
    assert finished.outcome is OutcomeCategory.DEFERRED
    assert finished.detail == "SeaDex unreachable"


def test_step_failure_graduates_failed_and_reraises() -> None:
    factory, recorder, _ = _factory()

    with pytest.raises(ValueError, match="boom"):
        with factory.step("Opening cache"):
            raise ValueError("boom")

    (finished,) = recorder.of_type(BootStepFinished)
    assert finished.outcome is OutcomeCategory.FAILED


def test_step_progress_emits_the_slow_heads_up_exactly_once() -> None:
    factory, recorder, _ = _factory()

    with factory.step("Refreshing mappings") as step:
        step.progress(0.25, "anime-ids")
        step.progress(1.7)

    assert [s.label for s in recorder.of_type(BootStepSlow)] == ["Refreshing mappings"]
    progressed = recorder.of_type(BootStepProgressed)
    # Fractions clamp to 0-1; the last detail sticks.
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
    assert "note: too late [after step 'Reading config' closed]" == late[0].message
    assert late[1].severity is Severity.WARNING


# --- EntryScope --------------------------------------------------------------------


def test_entry_opens_with_its_header_stamped() -> None:
    factory, recorder, _ = _factory()

    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="Frieren", al_id=154587))

    opened = recorder.of_type(ScopeOpened)
    headers = recorder.of_type(EntryHeader)
    assert [o.label for o in opened] == ["Frieren"]
    assert headers[0].scope == entry.scope_id
    assert headers[0].al_id == 154587
    assert recorder.events[0] == opened[0]  # header-at-open: ScopeOpened first


def test_entry_detail_warn_and_fail_carry_severity_and_accent() -> None:
    factory, recorder, _ = _factory()
    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="T"))

    entry.detail("files", "S01 E01-E28")
    entry.warn("skipped", "Okay-Subs on AB (private-only)")
    entry.fail("status", "manual import rejected")

    details = recorder.of_type(EntryDetail)
    assert details[0] == EntryDetail(
        label="files",
        value=StyledValue("S01 E01-E28"),
        scope=entry.scope_id,
    )
    assert details[1].severity is Severity.WARNING
    assert details[1].value.accent is Accent.CAUTION
    assert details[2].severity is Severity.ERROR
    assert details[2].value.accent is Accent.BAD


def test_entry_post_stamps_the_scope_id() -> None:
    factory, recorder, _ = _factory()
    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="T"))

    entry.post(ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY))

    (skipped,) = recorder.of_type(ReleaseSkipped)
    assert skipped.scope == entry.scope_id


def test_entry_close_emits_scope_closed_once() -> None:
    factory, recorder, _ = _factory()
    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="T"))

    entry.close()
    entry.close()

    assert [c.scope for c in recorder.of_type(ScopeClosed)] == [entry.scope_id]


def test_late_entry_post_demotes_and_keeps_the_severity() -> None:
    factory, recorder, _ = _factory()
    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="Frieren"))
    entry.close()

    entry.warn("skipped", "Okay-Subs on AB (private-only)")
    entry.post(
        GrabAction(status=GrabStatus.ADDING, groups=(), added=(), downloading=()),
    )

    assert len(recorder.of_type(EntryDetail)) == 0
    assert len(recorder.of_type(GrabAction)) == 0
    late = recorder.of_type(Diagnostic)
    assert [d.origin for d in late] == ["output.late.entry"] * 2
    assert late[0].severity is Severity.WARNING  # a demoted warn stays counted as one
    assert "[after entry 'Frieren' closed]" in late[0].message
    assert "grab action: adding" in late[1].message


def test_every_late_fact_kind_describes_itself() -> None:
    factory, recorder, _ = _factory()
    entry = factory.entry(EntryHeader(state=EntryState.CHECKING, title="T"))
    entry.close()

    entry.post(LedgerRow(state=EntryState.IGNORED, label="AniList #1"))
    entry.post(ReleaseSkipped(group="G", tracker="AB", reason=SkipReason.PRIVATE_ONLY))
    entry.post(GrabFailed(group="G", url="u", error="tracker down"))

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

    wait = factory.wait(total=4)

    assert isinstance(recorder.events[0], ScopeOpened)
    assert recorder.events[1] == WaitStarted(total=4, scope=wait.scope_id)


def test_wait_progress_and_graduations_are_stamped() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=1)

    wait.progress(WaitSnapshot(torrents=(), elapsed_s=30.0))
    wait.graduated(TorrentGraduated(label="T", outcome=Outcome.IMPORTED, files=12, waited_s=243.0))

    (progress,) = recorder.of_type(WaitProgress)
    (graduated,) = recorder.of_type(TorrentGraduated)
    assert progress.scope == wait.scope_id
    assert graduated.scope == wait.scope_id


def test_wait_finish_emits_tally_then_closes() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=4)

    wait.finish(WaitFinished(imported=3, deferred=1, failed=0, elapsed_s=730.0))

    (finished,) = recorder.of_type(WaitFinished)
    assert finished == WaitFinished(imported=3, deferred=1, failed=0, elapsed_s=730.0, scope=wait.scope_id)
    assert isinstance(recorder.events[-1], ScopeClosed)


def test_late_wait_emissions_demote() -> None:
    factory, recorder, _ = _factory()
    wait = factory.wait(total=1)
    wait.close()

    wait.progress(WaitSnapshot(torrents=(), elapsed_s=1.0))
    wait.graduated(TorrentGraduated(label="T", outcome=Outcome.IMPORTED, files=None, waited_s=1.0))
    wait.finish(WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=1.0))

    assert len(recorder.of_type(WaitProgress)) == 0
    assert len(recorder.of_type(TorrentGraduated)) == 0
    assert len(recorder.of_type(WaitFinished)) == 0
    late = recorder.of_type(Diagnostic)
    assert [d.origin for d in late] == ["output.late.wait"] * 3
    assert "imported T" in late[1].message


# --- Diagnostics ---------------------------------------------------------------------


def test_diagnostics_bind_origin_once_and_narrow_via_child() -> None:
    recorder = _Recorder()
    factory = ScopeFactory(recorder)
    diag = factory.diagnostics("arr_http")

    diag.warn("fail-open: could not fetch the queue")
    diag.child("Sonarr").error("boom", exc=ValueError("x"))
    diag.info("backing off 30s", once="anilist-backoff")

    events = recorder.of_type(Diagnostic)
    assert [(d.origin, d.severity) for d in events] == [
        ("arr_http", Severity.WARNING),
        ("arr_http:Sonarr", Severity.ERROR),
        ("arr_http", Severity.INFO),
    ]
    assert events[1].trace is not None
    assert "ValueError" in events[1].trace.plain_text()
    assert events[2].once_key == "anilist-backoff"
    assert all(d.placed_by is PlacedBy.AMBIENT for d in events)
