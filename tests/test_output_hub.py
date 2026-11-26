# pyright: strict
# pyright: reportPrivateUsage=false
"""Tests for the OutputHub (`output.hub`).

Pin the dispatch contract (every renderer, emit order), per-event containment
with N strikes + per-run reset (never process-latching), the once= dedup
store, SeverityCounter mark/counts_since, set_level fan-out, the console
renderer swap on a format change at begin_run, and the combiner: renderers
run without the hub lock, concurrent emits hand off to the active drainer, and
lifecycle calls wait out an in-flight drain.
"""

import io
import logging
import sys
import threading
from typing import ClassVar, override

import pytest

from pearlarr.cli import _install_output_hub
from pearlarr.config import Arr, LogFormat
from pearlarr.log import LOG_NAME
from pearlarr.output import (
    STRIKE_LIMIT,
    ArrRunFinished,
    Diagnostic,
    Event,
    FileLogRenderer,
    NullRenderer,
    OutputHub,
    PearlarrStarted,
    Renderer,
    Severity,
    SeverityCounter,
    SeverityTally,
    current_hub,
    emit_to_hub,
    hub_note,
    install_hub,
    uninstall_bridge,
    uninstall_hub,
)
from pearlarr.output.hub import QUEUE_CAP
from pearlarr.paths import resolve_paths

from .fakes import RecordingHub, RecordingRenderer, install_recording_hub

_EVENT = PearlarrStarted(version="v1.0.0", data_dir="/data")


class _QuietRenderer(Renderer):
    """A renderer that does nothing. Each double below overrides only what it does differently."""

    @override
    def handle(self, event: Event, when: float) -> None:
        pass

    @override
    def begin_run(self) -> None:
        pass

    @override
    def set_level(self, level: int) -> None:
        pass

    @override
    def close(self) -> None:
        pass


class _FailingRenderer(_QuietRenderer):
    """A renderer whose handle always raises (the containment test double)."""

    def __init__(self) -> None:
        self.calls = 0
        self.closes = 0

    @override
    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        raise RuntimeError("render bug")

    @override
    def close(self) -> None:
        self.closes += 1


class _InterruptingRenderer(_FailingRenderer):
    @override
    def handle(self, event: Event, when: float) -> None:
        raise KeyboardInterrupt


class _FailOnMarkerRenderer(_FailingRenderer):
    """Raises on the marker event (PearlarrStarted) only.

    Renders arr run boundary events (ArrRunFinished) cleanly, so an arr run's close between
    failures neither strikes nor resets the strike count.
    """

    @override
    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        if isinstance(event, PearlarrStarted):
            raise RuntimeError("render bug")


class _FailingCloseRenderer(_FailingRenderer):
    @override
    def close(self) -> None:
        self.closes += 1
        raise RuntimeError("close bug")


class _EmittingRenderer(_QuietRenderer):
    """Emits two follow-ups through the process hub from inside handle of the first event it's handed."""

    def __init__(self) -> None:
        self.emitted = False

    @override
    def handle(self, event: Event, when: float) -> None:
        if not self.emitted:
            self.emitted = True
            emit_to_hub(Diagnostic(severity=Severity.INFO, message="first"))
            emit_to_hub(Diagnostic(severity=Severity.INFO, message="second"))


class _GatedRenderer(_QuietRenderer):
    """Blocks inside `handle` until released, then logs the event type and the thread that handled it.

    It blocks on the first event it's handed, and again on the next one after each `arm()`. Lifecycle calls go in
    the same log.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.log: list[str] = []
        self.threads: list[threading.Thread] = []
        self._armed = True

    def arm(self) -> None:
        self.entered.clear()
        self.release.clear()
        self._armed = True

    @override
    def handle(self, event: Event, when: float) -> None:
        if self._armed:
            self._armed = False
            self.entered.set()
            # The wait must outlast any CI scheduling stall: 5 seconds once ran out on a loaded Windows runner.
            # A gate that runs out raises into the hub, which contains it, so that event never reaches the log.
            assert self.release.wait(timeout=30.0), "gate never released"
        self.log.append(type(event).__name__)
        self.threads.append(threading.current_thread())

    @override
    def begin_run(self) -> None:
        self.log.append("begin_run")

    @override
    def set_level(self, level: int) -> None:
        self.log.append("set_level")

    @override
    def close(self) -> None:
        self.log.append("close")


class _InterruptOnceRenderer(_QuietRenderer):
    """Raises KeyboardInterrupt on the first handle only (the drain-lock recovery check)."""

    def __init__(self) -> None:
        self.calls = 0

    @override
    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt


# --- dispatch ---------------------------------------------------------------------


def test_every_renderer_sees_every_event_in_order() -> None:
    first, second = RecordingRenderer(), RecordingRenderer()
    hub = OutputHub([first, second])
    events: list[Event] = [_EVENT, Diagnostic(severity=Severity.WARNING, message="w")]

    for event in events:
        hub.emit(event)

    assert first.events == events
    assert second.events == events


class _FileLog(_QuietRenderer):
    """A file log renderer: keeps every event it's handed."""

    writes_file_only: ClassVar[bool] = True

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.closed = False

    @override
    def handle(self, event: Event, when: float) -> None:
        self.events.append(event)

    @override
    def close(self) -> None:
        self.closed = True


class _FailingFileish(_FailingRenderer):
    """A failing renderer that IS the file renderer (the dying-log-file shape)."""

    writes_file_only: ClassVar[bool] = True


class _LifecycleEmitter(_QuietRenderer):
    """Emits through the process hub from inside begin_run, as a logging call there does through the bridge.

    The log holds "begin_run" once that call returns, and the type of each event it's handed.
    """

    def __init__(self) -> None:
        self.log: list[str] = []

    @override
    def handle(self, event: Event, when: float) -> None:
        self.log.append(type(event).__name__)

    @override
    def begin_run(self) -> None:
        emit_to_hub(Diagnostic(severity=Severity.INFO, message="mid-lifecycle"))
        self.log.append("begin_run")


class _EmitThenInterrupt(_QuietRenderer):
    """Emits a follow-up once, then raises KeyboardInterrupt from EVERY handle of the trigger event.

    This is the signal-mid-dispatch shape (the marker is a SIGTERM handler's
    exit line. The repeat raise models a hostile renderer on the flush's
    best-effort re-dispatch).
    """

    def __init__(self) -> None:
        self._marked = False

    @override
    def handle(self, event: Event, when: float) -> None:
        if isinstance(event, PearlarrStarted):
            if not self._marked:
                self._marked = True
                emit_to_hub(Diagnostic(severity=Severity.INFO, message="exit marker"))
            raise KeyboardInterrupt


# --- containment --------------------------------------------------------------------


def test_a_raising_renderer_never_breaks_the_survivors() -> None:
    flaky, survivor = _FailingRenderer(), _FileLog()
    hub = OutputHub([flaky, survivor])

    hub.emit(_EVENT)

    assert survivor.events[0] == _EVENT
    note = survivor.events[1]
    assert isinstance(note, Diagnostic)
    assert note.file_only  # a file renderer survives: forensic, not console spam
    assert note.severity is Severity.WARNING
    assert note.origin == "output.hub"
    assert "_FailingRenderer failed on PearlarrStarted" in note.message
    assert note.trace is not None and "RuntimeError" in note.trace.plain


def test_losing_the_file_log_renderer_escalates_the_note_to_a_counted_visible_error() -> None:
    """A run must never silently lose its whole log file.

    When the casualty IS the file renderer, the containment note goes visible
    at ERROR and counts.
    """

    flaky, survivor = _FailingFileish(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])
    mark = hub.counts.mark()

    hub.emit(_EVENT)

    note = survivor.events[1]
    assert isinstance(note, Diagnostic)
    assert not note.file_only
    assert note.severity is Severity.ERROR
    assert "_FailingFileish failed on PearlarrStarted" in note.message
    assert hub.counts.counts_since(mark).errors == 1


def test_three_strikes_quarantine_until_begin_run_restarts() -> None:
    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    for _ in range(5):
        hub.emit(_EVENT)
    assert flaky.calls == STRIKE_LIMIT  # skipped after the third strike

    hub.begin_run(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)
    assert flaky.calls == STRIKE_LIMIT + 1  # reset, not process-latched


def test_strikes_carry_across_an_arr_run_boundary_and_only_begin_run_restarts() -> None:
    """An arr run's close (ArrRunFinished) is not a run turnover.

    Strikes are a per-run budget: they accumulate across the ArrRunFinished
    boundary, so the Nth strike quarantines regardless of the boundary, and
    only `begin_run` resets them.
    """

    flaky, survivor = _FailOnMarkerRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])
    sub = hub._subs[0]

    # Arr run 1: one strike short of the limit - still active across the arr run boundary.
    for _ in range(STRIKE_LIMIT - 1):
        hub.emit(_EVENT)
    assert sub.strikes == STRIKE_LIMIT - 1

    # The arr run boundary renders cleanly: it neither strikes nor resets the count.
    hub.emit(ArrRunFinished(arr=Arr.SONARR))
    assert sub.strikes == STRIKE_LIMIT - 1
    assert flaky.closes == 0

    # Arr run 2: the carried-over count makes the next failure the Nth strike, so the
    # crossing (quarantine + close) fires across the boundary - never reset by it.
    hub.emit(_EVENT)
    assert sub.strikes == STRIKE_LIMIT
    assert flaky.closes == 1

    calls_at_quarantine = flaky.calls
    hub.emit(_EVENT)  # quarantined: skipped (not dispatched), not re-closed
    assert flaky.calls == calls_at_quarantine
    assert flaky.closes == 1

    # Only a run turnover resets the renderer - the arr run boundary never did.
    hub.begin_run(console_format="plain", level=logging.INFO)
    assert sub.strikes == 0
    hub.emit(_EVENT)
    assert sub.strikes == 1


def test_quarantine_closes_the_renderer_exactly_once_at_the_crossing() -> None:
    """This test pins the live-leak fix: striking out closes the renderer (a struck boot Live must stop repainting).

    Only the crossing fires it - never skipped events.
    """

    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)
    assert flaky.closes == 1  # closed the moment the count crossed the limit

    hub.emit(_EVENT)  # skipped while quarantined: close is not re-fired
    assert flaky.closes == 1

    hub.begin_run(console_format="plain", level=logging.INFO)
    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)
    assert flaky.closes == 2  # reset, then a fresh strike-out closes again


def test_a_raising_close_never_escapes_the_quarantine_crossing() -> None:
    flaky = _FailingCloseRenderer()
    hub = OutputHub([flaky, RecordingRenderer()])

    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)  # must not raise even though close() itself raises

    assert flaky.closes == 1


def test_the_final_strike_announces_the_quarantine() -> None:
    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)

    notes = [e for e in survivor.events if isinstance(e, Diagnostic)]
    assert "disabled until the next run" in notes[-1].message
    assert all("disabled until the next run" not in note.message for note in notes[:-1])


def test_keyboard_interrupt_still_propagates() -> None:
    hub = OutputHub([_InterruptingRenderer()])
    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)


def test_containment_notes_are_uncounted_while_a_file_renderer_survives() -> None:
    flaky, survivor = _FailingRenderer(), _FileLog()
    hub = OutputHub([flaky, survivor])
    mark = hub.counts.mark()

    hub.emit(_EVENT)  # INFO event. The forensic WARNING note must not inflate the tally

    since = hub.counts.counts_since(mark)
    assert (since.info, since.warning) == (1, 0)
    assert any(isinstance(event, Diagnostic) for event in survivor.events)  # the note went out


# --- re-entrancy -----------------------------------------------------------------------


def test_reentrant_emits_drain_fifo_before_the_outer_emit_returns() -> None:
    observer = RecordingRenderer()
    hub = OutputHub([_EmittingRenderer(), observer])
    install_hub(hub)

    hub.emit(_EVENT)

    # Queued, not dispatched inline: the observer finishes the outer event first, then both follow-ups in order.
    assert observer.events == [
        _EVENT,
        Diagnostic(severity=Severity.INFO, message="first"),
        Diagnostic(severity=Severity.INFO, message="second"),
    ]


def test_a_lifecycle_reentrant_emit_enqueues_and_drains_after_the_turnover() -> None:
    """An emit from inside a renderer's lifecycle call (a bridge-adopted teardown debug) must not start a nested drain.

    Starting one against a half-mutated subscriber list is unsafe. The
    lifecycle body holds the drain lock and drains the stragglers after.
    """

    emitter = _LifecycleEmitter()
    hub = OutputHub([emitter])
    install_hub(hub)

    hub.begin_run(console_format="plain", level=logging.INFO)

    assert emitter.log == ["begin_run", "Diagnostic"]  # enqueued, not dispatched mid-lifecycle


# --- the combiner (dispatch outside the hub lock) -------------------------------------


def _emit_held_at(hub: OutputHub, gated: _GatedRenderer) -> threading.Thread:
    """Emit _EVENT on a drainer thread and return the thread once the gated renderer holds the event mid-handle."""

    drainer = threading.Thread(target=lambda: hub.emit(_EVENT), daemon=True)
    drainer.start()
    assert gated.entered.wait(timeout=30.0)
    return drainer


def _release(gated: _GatedRenderer, drainer: threading.Thread) -> None:
    """Open the gate and wait for the drainer thread to finish."""

    gated.release.set()
    drainer.join(timeout=30.0)
    assert not drainer.is_alive()


def test_a_concurrent_emit_hands_off_to_the_active_drainer() -> None:
    gated = _GatedRenderer()
    hub = OutputHub([gated])
    drainer = _emit_held_at(hub, gated)

    # This emit returns at once without dispatching: the drainer thread still holds the drain lock.
    hub.emit(Diagnostic(severity=Severity.INFO, message="from main"))
    assert gated.log == []

    _release(gated, drainer)
    # The drainer delivered both. Had this thread's emit waited on the held renderer, the gate would have run out.
    assert gated.log == ["PearlarrStarted", "Diagnostic"]
    assert gated.threads == [drainer, drainer]


def test_an_unwinding_interrupt_still_flushes_the_queued_tail() -> None:
    survivor = RecordingRenderer()
    hub = OutputHub([_EmitThenInterrupt(), survivor])
    install_hub(hub)

    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)

    # Crash fidelity, both halves: the IN-FLIGHT event's fan-out completes on the
    # unwind path (the survivor gets the PearlarrStarted the interrupt cut short -
    # even though the hostile renderer re-raises on the best-effort re-dispatch), and
    # the QUEUED marker (a SIGTERM handler's exit line) flushes before the drain lock
    # is released.
    assert len(survivor.of_type(PearlarrStarted)) == 1
    assert [d.message for d in survivor.of_type(Diagnostic)] == ["exit marker"]
    assert hub._drainer is None


def test_a_propagating_interrupt_never_strands_the_drain_lock() -> None:
    flaky, survivor = _InterruptOnceRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)
    assert hub._drainer is None  # the finally released the drain lock

    follow = Diagnostic(severity=Severity.INFO, message="after")
    hub.emit(follow)  # a fresh emit claims the drain lock and dispatches
    assert follow in survivor.events


def test_an_interrupt_before_the_drain_loop_cannot_wedge_the_drain_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt landing before _drain runs must leave no leftover drain lock behind.

    Emit only checks-and-calls. The drain lock is taken inside _drain's try. With the
    old take-in-emit shape this KI left _drainer set forever and the follow-up
    emit dispatched nothing (the silent-dark-hub wedge).
    """

    survivor = RecordingRenderer()
    hub = OutputHub([survivor])

    def interrupted_drain() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(hub, "_drain", interrupted_drain)
    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)
    monkeypatch.undo()

    assert hub._drainer is None  # nothing to strand: the drain lock was never taken
    follow = Diagnostic(severity=Severity.INFO, message="after")
    hub.emit(follow)  # drains the stranded event AND the new one, in order
    assert [type(event).__name__ for event in survivor.events] == ["PearlarrStarted", "Diagnostic"]


def test_lifecycle_waits_for_the_active_drain_to_finish() -> None:
    gated = _GatedRenderer()
    hub = OutputHub([gated])

    drainer = _emit_held_at(hub, gated)

    lifecycle = threading.Thread(target=lambda: hub.begin_run(console_format="plain", level=logging.INFO), daemon=True)
    lifecycle.start()
    # A bounded chance for a broken (non-waiting) begin_run to interleave.
    lifecycle.join(timeout=0.2)
    assert lifecycle.is_alive()  # parked until the drain finishes
    assert gated.log == []

    _release(gated, drainer)
    lifecycle.join(timeout=30.0)
    assert gated.log == ["PearlarrStarted", "begin_run", "set_level"]


def test_overflow_sheds_newest_with_one_note_per_episode() -> None:
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])
    mark = hub.counts.mark()

    drainer = _emit_held_at(hub, gated)

    # Fill the stalled queue to the cap, then two more that must be shed.
    for i in range(QUEUE_CAP):
        hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
    hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-1"))
    hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-2"))

    _release(gated, drainer)

    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert f"q{QUEUE_CAP - 1}" in messages  # everything under the cap still renders
    assert "dropped-1" not in messages and "dropped-2" not in messages
    assert sum("overflowed" in m for m in messages) == 1  # one note, not one per drop
    # Shed events were still counted at enqueue. The note itself is never counted.
    since = hub.counts.counts_since(mark)
    assert (since.info, since.warning) == (QUEUE_CAP + 3, 0)


def test_overflow_never_sheds_a_structural_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the cap, only diagnostics shed.

    An open-scopes input (ArrRunFinished) is appended even past the cap, so the
    open scopes can never be corrupted.
    """

    cap = 3
    monkeypatch.setattr("pearlarr.output.hub.QUEUE_CAP", cap)
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])

    drainer = _emit_held_at(hub, gated)

    # _EVENT is blocked mid-handle (already popped). Fill the pending queue to the cap.
    for i in range(cap):
        hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
    structural = ArrRunFinished(arr=Arr.SONARR)
    hub.emit(structural)  # non-diagnostic at the cap: appended, never shed

    _release(gated, drainer)

    assert structural in observer.events
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert all("overflowed" not in m for m in messages)  # a structural never trips the note


def test_overflow_sheds_diagnostics_and_restarts_the_note_per_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1 kept behavior: diagnostics past the cap shed (still counted at enqueue) with one note per episode.

    The note resets once a drain empties the queue.
    """

    cap = 3
    monkeypatch.setattr("pearlarr.output.hub.QUEUE_CAP", cap)
    gate, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gate, observer])
    mark = hub.counts.mark()

    def run_overflow_episode() -> None:
        gate.arm()
        drainer = _emit_held_at(hub, gate)
        for i in range(cap):  # fill to the cap
            hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
        hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-1"))
        hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-2"))
        _release(gate, drainer)

    run_overflow_episode()
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert f"q{cap - 1}" in messages  # everything under the cap still renders
    assert "dropped-1" not in messages and "dropped-2" not in messages
    assert sum("overflowed" in m for m in messages) == 1  # one note, not one per drop
    assert hub._overflowing is False  # the drain emptied the queue and reset the note

    run_overflow_episode()
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert sum("overflowed" in m for m in messages) == 2  # a fresh note for the new episode

    # Shed diagnostics were counted at enqueue. The note itself is never counted.
    # Per episode: _EVENT + cap diagnostics + 2 dropped = cap + 3 info.
    since = hub.counts.counts_since(mark)
    assert (since.info, since.warning) == (2 * (cap + 3), 0)


# --- once= dedup ---------------------------------------------------------------------


def test_once_keys_dedup_and_clear_per_scheduled_run() -> None:
    recording = RecordingHub()
    warning = Diagnostic(severity=Severity.WARNING, message="SeaDex unreachable", once_key="seadex-outage")

    recording.emit(warning)
    recording.emit(warning)
    assert len(recording.of_type(Diagnostic)) == 1
    assert recording.hub.counts.mark().warning == 1  # the dropped copy is not counted

    recording.hub.begin_run(console_format="plain", level=logging.INFO)
    recording.emit(warning)
    assert len(recording.of_type(Diagnostic)) == 2


def test_the_same_once_key_from_different_origins_both_pass() -> None:
    recording = RecordingHub()

    recording.emit(Diagnostic(severity=Severity.WARNING, message="a", origin="anilist", once_key="backoff"))
    recording.emit(Diagnostic(severity=Severity.WARNING, message="b", origin="seadex", once_key="backoff"))

    assert [d.origin for d in recording.of_type(Diagnostic)] == ["anilist", "seadex"]


# --- severity counts -------------------------------------------------------------------


def test_severity_counts_mark_and_delta() -> None:
    counts = SeverityCounter()
    counts.record(Severity.WARNING)
    mark = counts.mark()

    counts.record(Severity.WARNING)
    counts.record(Severity.ERROR)
    counts.record(Severity.CRITICAL)

    since = counts.counts_since(mark)
    assert since == SeverityTally(warning=1, error=1, critical=1)
    assert since.warning == 1
    assert since.errors == 2  # ERROR + CRITICAL
    assert counts.mark().warning == 2  # totals stay monotonic


def test_bound_mark_diffs_only_its_own_counter() -> None:
    """A CountsMark carries its counter: records on ANOTHER counter never skew since()."""

    counts = SeverityCounter()
    mark = counts.bound_mark()

    other = SeverityCounter()
    other.record(Severity.ERROR)
    assert mark.since() == SeverityTally()

    counts.record(Severity.WARNING)
    assert mark.since() == SeverityTally(warning=1)


def test_hub_tallies_every_emitted_event() -> None:
    recording = RecordingHub()
    mark = recording.hub.counts.mark()

    recording.emit(_EVENT)
    recording.emit(Diagnostic(severity=Severity.ERROR, message="boom"))

    since = recording.hub.counts.counts_since(mark)
    assert (since.info, since.errors) == (1, 1)


def test_file_only_diagnostics_are_never_counted() -> None:
    # Counts = what a visible renderer could show. file_only forensics stay out.
    recording = RecordingHub()
    mark = recording.hub.counts.mark()

    recording.emit(Diagnostic(severity=Severity.WARNING, message="forensic note", file_only=True))
    assert recording.hub.counts.counts_since(mark).warning == 0

    recording.emit(Diagnostic(severity=Severity.WARNING, message="visible note"))
    assert recording.hub.counts.counts_since(mark).warning == 1


def test_severity_is_counted_at_enqueue_even_when_every_renderer_raises() -> None:
    hub = OutputHub([_FailingRenderer(), _FailingRenderer()])
    mark = hub.counts.mark()

    hub.emit(Diagnostic(severity=Severity.WARNING, message="boom"))

    since = hub.counts.counts_since(mark)
    assert since.warning == 1  # counted at enqueue, though nothing rendered it
    assert since.errors == 2  # with no file renderer left, both notes escalate visible


# --- levels + lifecycle -------------------------------------------------------------


def test_set_level_fans_out_to_every_renderer() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.set_level(logging.DEBUG)

    assert renderer.levels == [logging.DEBUG]


def test_begin_run_turns_over_renderers_and_applies_the_level() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.begin_run(console_format="plain", level=logging.WARNING)

    assert renderer.runs_begun == 1
    assert renderer.levels == [logging.WARNING]


def test_close_is_idempotent_and_reaches_every_renderer() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.close()
    hub.close()

    assert renderer.closed


def test_a_closed_hub_drops_emits_and_scheduled_run_turnovers() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])
    hub.close()

    hub.emit(_EVENT)
    hub.begin_run(console_format="plain", level=logging.INFO)

    assert renderer.events == []
    assert renderer.runs_begun == 0


# --- timestamps ------------------------------------------------------------------------


def test_the_hub_stamps_one_instant_per_emit_for_every_renderer() -> None:
    ticks = iter([100.0, 200.0])
    first, second = RecordingRenderer(), RecordingRenderer()
    hub = OutputHub([first, second], clock=lambda: next(ticks))

    hub.emit(_EVENT)
    hub.emit(_EVENT)

    # One clock read per emit: cross-renderer timestamps can never disagree.
    assert first.whens == [100.0, 200.0]
    assert second.whens == [100.0, 200.0]


# --- console ownership (the rich handler's skip predicate) ---------------------------


def test_visible_warnings_reach_stderr_while_no_console_renderer_is_active(capsys: pytest.CaptureFixture[str]) -> None:
    """This covers the install→`begin_run` window and a quarantined console renderer.

    A visible WARNING+ diagnostic must still reach a human, via stderr.
    """

    hub = OutputHub([_FileLog()], console_factory=lambda console_format: NullRenderer())

    hub.emit(Diagnostic(severity=Severity.CRITICAL, message="invalid level", origin="app"))

    assert "CRITICAL [app] invalid level" in capsys.readouterr().err


def test_the_stderr_fallback_skips_info_file_only_and_hubs_with_a_console_renderer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hub = OutputHub([_FileLog()], console_factory=lambda console_format: NullRenderer())

    hub.emit(Diagnostic(severity=Severity.INFO, message="quiet"))
    hub.emit(Diagnostic(severity=Severity.WARNING, message="forensic", file_only=True))
    hub.begin_run(console_format="plain", level=logging.INFO)
    hub.emit(Diagnostic(severity=Severity.WARNING, message="seated"))

    assert capsys.readouterr().err == ""


def test_factory_less_hubs_never_write_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    # The renderer-less default hub's library contract: drop silently, stay quiet.
    hub = OutputHub([])

    hub.emit(Diagnostic(severity=Severity.ERROR, message="library use"))

    assert capsys.readouterr().err == ""


def test_console_render_active_is_false_without_a_console_renderer() -> None:
    hub = OutputHub([RecordingRenderer()])

    assert hub.console_render_active() is False


def test_console_render_active_is_true_for_an_active_console_renderer() -> None:
    hub = OutputHub([], console=RecordingRenderer())

    assert hub.console_render_active() is True


def test_console_render_active_is_false_once_the_console_renderer_strikes_out() -> None:
    # The quarantine fallback: a struck-out console renderer hands the badge class
    # back to the legacy handler, so warnings can never vanish.
    hub = OutputHub([], console=_FailingRenderer(), strike_limit=1)

    hub.emit(_EVENT)

    assert hub.console_render_active() is False


# --- console renderer swap -----------------------------------------------------------


def test_begin_run_swaps_the_console_renderer_only_on_format_change() -> None:
    built: list[tuple[LogFormat, RecordingRenderer]] = []

    def factory(console_format: LogFormat) -> Renderer:
        renderer = RecordingRenderer()
        built.append((console_format, renderer))
        return renderer

    stable_renderer = RecordingRenderer()
    hub = OutputHub([stable_renderer], console_factory=factory)

    hub.begin_run(console_format="rich", level=logging.INFO)
    hub.emit(_EVENT)
    hub.begin_run(console_format="rich", level=logging.INFO)  # unchanged: no rebuild
    hub.begin_run(console_format="plain", level=logging.INFO)  # changed: swap
    hub.emit(_EVENT)

    assert [fmt for fmt, _ in built] == ["rich", "plain"]
    rich_console, plain_console = built[0][1], built[1][1]
    assert rich_console.events == [_EVENT]
    assert rich_console.closed  # the replaced console is torn down
    assert plain_console.events == [_EVENT]
    assert stable_renderer.events == [_EVENT, _EVENT]  # the stable renderer saw both


def test_a_failing_console_factory_keeps_the_old_console_renderer_and_notes_file_only() -> None:
    console, renderer = RecordingRenderer(), _FileLog()

    def factory(console_format: LogFormat) -> Renderer:
        raise RuntimeError("no tty")

    hub = OutputHub([renderer], console=console, console_factory=factory)
    hub.begin_run(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)

    assert not console.closed
    assert _EVENT in console.events  # the old console renderer keeps rendering
    notes = [e for e in renderer.events if isinstance(e, Diagnostic)]
    assert notes[0].file_only
    assert "console factory failed" in notes[0].message
    assert "keeping the current console" in notes[0].message


# --- the process registry (install/uninstall) + the cli renderer order ----------------


def test_cli_hub_installs_the_file_log_renderer_first() -> None:
    # The file-before-console dispatch pin: _subs[0] is the first-declared renderer
    # (the reader convention above), and cli declares FileLogRenderer first.
    hub, _ = _install_output_hub(resolve_paths())
    try:
        assert isinstance(hub._subs[0].renderer, FileLogRenderer)
    finally:
        uninstall_hub()
        uninstall_bridge()


def test_install_hub_closes_the_previously_installed_hub() -> None:
    first_renderer, second_renderer = RecordingRenderer(), RecordingRenderer()
    first, second = OutputHub([first_renderer]), OutputHub([second_renderer])

    install_hub(first)
    install_hub(second)
    try:
        # A repeat run single in one process must not leak the prior hub's
        # open FileLogRenderer. A re-install of the SAME hub never self-closes.
        assert first_renderer.closed
        install_hub(second)
        second.emit(_EVENT)
        assert second_renderer.events == [_EVENT]
    finally:
        uninstall_hub()


def test_uninstall_hub_closes_the_outgoing_hub_but_never_the_default() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])
    install_hub(hub)

    uninstall_hub()

    assert renderer.closed
    # The renderer-less default survives repeated uninstalls (never closed):
    # a post-uninstall emit still silently drops rather than raising.
    uninstall_hub()
    emit_to_hub(_EVENT)
    hub.emit(_EVENT)  # drop-after-close is silent
    assert renderer.events == []


# --- hub_note (the raw first-party logging replacement) -------------------------------


def test_hub_note_with_exc_captures_the_trace() -> None:
    recording = install_recording_hub()
    try:
        raise ValueError("boom")
    except ValueError as e:
        hub_note("it broke", severity=Severity.ERROR, exc=e)

    (note,) = recording.of_type(Diagnostic)
    assert note.severity is Severity.ERROR
    assert note.message == "it broke"
    assert note.origin == LOG_NAME
    assert note.trace is not None
    assert "ValueError: boom" in note.trace.plain


# --- the p4 re-review pins (hub core) --------------------------------------------


def test_a_once_key_shed_by_overflow_is_shed_whole_and_lands_on_re_emit() -> None:
    """A once-keyed diagnostic arriving during an overflow episode must not consume its dedup key (or its count).

    It sheds WHOLE, so the post-drain re-emit is the one rendered and the one
    tallied.
    """

    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])
    mark = hub.counts.mark()
    keyed = Diagnostic(severity=Severity.WARNING, message="SeaDex unreachable", once_key="seadex-outage")

    drainer = _emit_held_at(hub, gated)
    for i in range(QUEUE_CAP):
        hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
    hub.emit(keyed)  # at the cap: shed whole - key unregistered, uncounted

    _release(gated, drainer)

    hub.emit(keyed)  # the queue drained: the re-emit must land

    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert messages.count("SeaDex unreachable") == 1
    # Exactly ONE keyed warning tallied (the shed one was uncounted. The
    # overflow note is file_only forensics and never counts).
    assert hub.counts.counts_since(mark).warning == 1


class _LifecycleCaller(_QuietRenderer):
    """On each PearlarrStarted it's handed, emits a follow-up and then sets the process hub's level from inside `handle`."""

    @override
    def handle(self, event: Event, when: float) -> None:
        if isinstance(event, PearlarrStarted):
            emit_to_hub(Diagnostic(severity=Severity.INFO, message="queued behind"))
            current_hub().set_level(logging.WARNING)


def test_a_mid_dispatch_lifecycle_call_keeps_the_drain_lock_and_the_emit_order() -> None:
    """`set_level` from inside a renderer's handle (the documented mid-drain lifecycle escape) must not flush re-entrantly or release the OUTER drain's drain lock.

    The re-entrant event still dispatches AFTER the in-flight one.
    """

    recorder = RecordingRenderer()
    hub = OutputHub([_LifecycleCaller(), recorder])
    install_hub(hub)

    # On a thread of its own: without the escape, set_level waits on its own drain for good.
    emitter = threading.Thread(target=lambda: hub.emit(_EVENT), daemon=True)
    emitter.start()
    emitter.join(timeout=30.0)
    assert not emitter.is_alive(), "set_level inside handle waited on its own drain"

    # Order preserved: the in-flight PearlarrStarted completes its fan-out before the
    # re-entrant diagnostic dispatches. Nothing double-dispatches. Drain lock released.
    assert [type(e).__name__ for e in recorder.events] == ["PearlarrStarted", "Diagnostic"]
    assert hub._drainer is None


class _BrokenStderr(io.StringIO):
    """A stderr whose pipe is gone."""

    @override
    def write(self, s: str) -> int:
        raise BrokenPipeError


def test_a_broken_stderr_never_raises_through_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stderr fallback is contained like every renderer write.

    A broken pipe on stderr must not abort the run (the emit-never-raises
    contract).
    """

    hub = OutputHub([], console_factory=lambda console_format: NullRenderer())
    # No begin_run: the console renderer is inactive, so the fallback path fires.
    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    hub.emit(Diagnostic(severity=Severity.WARNING, message="watch out"))

    assert hub.counts.mark().warning == 1  # emitted, counted, contained


class _NoisyCloser(_QuietRenderer):
    """Emits a file-only teardown note through the process hub from inside its own `close`."""

    @override
    def close(self) -> None:
        emit_to_hub(Diagnostic(severity=Severity.WARNING, message="teardown failed", file_only=True))


def test_close_hands_teardown_chatter_to_the_file_log_renderer_before_closing_it() -> None:
    """`close()` closes the file renderers LAST.

    A diagnostic emitted from another renderer's close path (a contained
    teardown failure) still reaches the file.
    """

    file_recorder = _FileLog()
    hub = OutputHub([file_recorder, _NoisyCloser()])
    install_hub(hub)

    hub.close()

    assert [d.message for d in file_recorder.events if isinstance(d, Diagnostic)] == ["teardown failed"]
    assert file_recorder.closed
    hub.emit(_EVENT)  # fully closed: drops silently
    assert not any(isinstance(e, PearlarrStarted) for e in file_recorder.events)


def test_begin_run_applies_the_level_to_a_fresh_console_exactly_once() -> None:
    """The swap-time `set_level` died (`begin_run`'s tail applies the run level to every sub).

    One bug must cost the fresh console ONE strike, not two.
    """

    fresh = RecordingRenderer()
    hub = OutputHub([], console_factory=lambda console_format: fresh)

    hub.begin_run(console_format="rich", level=logging.DEBUG)

    assert fresh.levels == [logging.DEBUG]


def test_hub_note_without_exc_keeps_the_trace_none() -> None:
    recording = install_recording_hub()
    hub_note("plain note")

    (note,) = recording.of_type(Diagnostic)
    assert note.severity is Severity.INFO
    assert note.trace is None
