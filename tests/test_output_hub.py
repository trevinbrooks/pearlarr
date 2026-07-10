# pyright: strict
# pyright: reportPrivateUsage=false
"""Tests for the OutputHub (``output.hub``).

Pin the dispatch contract (every surface, emit order), per-event containment
with N strikes + per-cycle re-arm (never process-latching), the once= dedup
store, SeverityCounts mark/counts_since, set_level fan-out, the console
renderer swap on a format change at begin_cycle, and the combiner: renderers
run without the hub lock, concurrent emits hand off to the active drainer, and
lifecycle calls wait out an in-flight drain.
"""

import logging
import threading
from typing import override

import pytest

from seadexarr.modules.config import Arr, LogFormat
from seadexarr.modules.output import (
    STRIKE_LIMIT,
    Diagnostic,
    Event,
    NullRenderer,
    OutputHub,
    Renderer,
    RunFinished,
    RunStarted,
    Severity,
    SeverityCounts,
    SeverityTally,
)
from seadexarr.modules.output.hub import QUEUE_CAP
from seadexarr.modules.output.recording import RecordingHub, RecordingRenderer

_EVENT = RunStarted(version="v1.0.0", data_dir="/data")


class _FailingRenderer:
    """A renderer whose handle always raises (the containment test double)."""

    def __init__(self) -> None:
        self.calls = 0
        self.closes = 0

    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        raise RuntimeError("render bug")

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        self.closes += 1


class _InterruptingRenderer(_FailingRenderer):
    @override
    def handle(self, event: Event, when: float) -> None:
        raise KeyboardInterrupt


class _FailOnMarkerRenderer(_FailingRenderer):
    """Raises on the marker event (RunStarted) only; renders leg-boundary events
    (RunFinished) cleanly, so a leg close between failures neither strikes nor
    resets the strike count."""

    @override
    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        if isinstance(event, RunStarted):
            raise RuntimeError("render bug")


class _FailingCloseRenderer(_FailingRenderer):
    @override
    def close(self) -> None:
        self.closes += 1
        raise RuntimeError("close bug")


class _EmittingRenderer:
    """A renderer that emits a follow-up Diagnostic from inside handle (once)."""

    def __init__(self) -> None:
        self.hub: OutputHub | None = None
        self.emitted = False

    def handle(self, event: Event, when: float) -> None:
        if not self.emitted and self.hub is not None:
            self.emitted = True
            self.hub.emit(Diagnostic(severity=Severity.INFO, message="re-entrant"))

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


class _DoubleEmittingRenderer(_EmittingRenderer):
    """Emits two follow-ups from inside handle of the first event (FIFO probe)."""

    @override
    def handle(self, event: Event, when: float) -> None:
        if not self.emitted and self.hub is not None:
            self.emitted = True
            self.hub.emit(Diagnostic(severity=Severity.INFO, message="first"))
            self.hub.emit(Diagnostic(severity=Severity.INFO, message="second"))


class _LockProbeRenderer:
    """handle() probes the hub lock from a HELPER thread (the same thread would
    trivially re-acquire an RLock it owns)."""

    def __init__(self) -> None:
        self.hub: OutputHub | None = None
        self.lock_was_free: list[bool] = []

    def handle(self, event: Event, when: float) -> None:
        hub = self.hub
        assert hub is not None
        acquired: list[bool] = []

        def probe() -> None:
            if hub._lock.acquire(blocking=False):
                hub._lock.release()
                acquired.append(True)
            else:
                acquired.append(False)

        helper = threading.Thread(target=probe)
        helper.start()
        helper.join(timeout=5.0)
        self.lock_was_free.extend(acquired)

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


class _GatedRenderer:
    """Blocks inside handle (on the first event only) until released; records each
    handled event with its handling thread plus lifecycle calls, in one order log."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.handled: list[tuple[str, threading.Thread]] = []
        self.log: list[str] = []
        self._blocked_once = False

    def handle(self, event: Event, when: float) -> None:
        name = type(event).__name__
        self.handled.append((name, threading.current_thread()))
        if not self._blocked_once:
            self._blocked_once = True
            self.entered.set()
            assert self.release.wait(timeout=5.0), "gate never released"
        self.log.append(f"handled:{name}")

    def begin_cycle(self) -> None:
        self.log.append("cycle")

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


class _InterruptOnceRenderer:
    """Raises KeyboardInterrupt on the first handle only (the baton-recovery probe)."""

    def __init__(self) -> None:
        self.calls = 0

    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


class _ReArmableGate:
    """Blocks inside handle once per armed episode until released; arm() re-arms it
    for the next episode — the overflow-note-per-episode reset probe."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._armed = True

    def arm(self) -> None:
        self.entered.clear()
        self.release.clear()
        self._armed = True

    def handle(self, event: Event, when: float) -> None:
        if self._armed:
            self._armed = False
            self.entered.set()
            assert self.release.wait(timeout=5.0), "gate never released"

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


# --- dispatch ---------------------------------------------------------------------


def test_every_renderer_sees_every_event_in_order() -> None:
    first, second = RecordingRenderer(), RecordingRenderer()
    hub = OutputHub([first, second])
    events: list[Event] = [_EVENT, Diagnostic(severity=Severity.WARNING, message="w")]

    for event in events:
        hub.emit(event)

    assert first.events == events
    assert second.events == events


def test_null_renderer_satisfies_the_protocol() -> None:
    renderer: Renderer = NullRenderer()
    hub = OutputHub([renderer])
    hub.emit(_EVENT)
    hub.close()


# --- containment --------------------------------------------------------------------


def test_a_raising_renderer_never_breaks_the_survivors() -> None:
    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    hub.emit(_EVENT)

    assert survivor.events[0] == _EVENT
    note = survivor.events[1]
    assert isinstance(note, Diagnostic)
    assert note.file_only
    assert note.origin == "output.hub"
    assert "_FailingRenderer failed on RunStarted" in note.message
    assert note.trace is not None and "RuntimeError" in note.trace.plain_text()


def test_three_strikes_quarantine_until_begin_cycle_rearms() -> None:
    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    for _ in range(5):
        hub.emit(_EVENT)
    assert flaky.calls == STRIKE_LIMIT  # skipped after the third strike

    hub.begin_cycle(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)
    assert flaky.calls == STRIKE_LIMIT + 1  # re-armed, not process-latched


def test_strikes_carry_across_a_run_leg_boundary_and_only_begin_cycle_rearms() -> None:
    """G3: a leg close (RunFinished) is not a cycle turnover. Strikes are a per-cycle
    budget (S9): they accumulate across the RunFinished leg boundary, so the Nth
    strike quarantines regardless of the boundary, and only begin_cycle re-arms."""

    flaky, survivor = _FailOnMarkerRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])
    sub = hub._subs[0]

    # Leg 1: one strike short of the limit — still armed across the leg boundary.
    for _ in range(STRIKE_LIMIT - 1):
        hub.emit(_EVENT)
    assert sub.strikes == STRIKE_LIMIT - 1

    # The leg boundary renders cleanly: it neither strikes nor resets the count.
    hub.emit(RunFinished(arr=Arr.SONARR))
    assert sub.strikes == STRIKE_LIMIT - 1
    assert flaky.closes == 0

    # Leg 2: the carried-over count makes the next failure the Nth strike, so the
    # crossing (quarantine + close) fires across the boundary — never reset by it.
    hub.emit(_EVENT)
    assert sub.strikes == STRIKE_LIMIT
    assert flaky.closes == 1

    calls_at_quarantine = flaky.calls
    hub.emit(_EVENT)  # quarantined: skipped (not dispatched), not re-closed
    assert flaky.calls == calls_at_quarantine
    assert flaky.closes == 1

    # Only a cycle turnover re-arms the seat — the leg boundary never did.
    hub.begin_cycle(console_format="plain", level=logging.INFO)
    assert sub.strikes == 0
    hub.emit(_EVENT)
    assert sub.strikes == 1


def test_quarantine_closes_the_seat_exactly_once_at_the_crossing() -> None:
    """The live-leak pin: striking out closes the renderer (a struck boot Live
    must stop repainting), and only the crossing fires it — never skipped events."""

    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)
    assert flaky.closes == 1  # closed the moment the count crossed the limit

    hub.emit(_EVENT)  # skipped while quarantined: close is not re-fired
    assert flaky.closes == 1

    hub.begin_cycle(console_format="plain", level=logging.INFO)
    for _ in range(STRIKE_LIMIT):
        hub.emit(_EVENT)
    assert flaky.closes == 2  # re-armed, then a fresh strike-out closes again


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
    assert "quarantined until next cycle" in notes[-1].message
    assert all("quarantined" not in note.message for note in notes[:-1])


def test_emit_never_raises_even_when_every_renderer_fails() -> None:
    hub = OutputHub([_FailingRenderer(), _FailingRenderer()])
    hub.emit(_EVENT)  # must not raise


def test_keyboard_interrupt_still_propagates() -> None:
    hub = OutputHub([_InterruptingRenderer()])
    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)


def test_containment_notes_are_never_counted() -> None:
    flaky, survivor = _FailingRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])
    mark = hub.counts.mark()

    hub.emit(_EVENT)  # INFO event; the WARNING note must not inflate the tally

    since = hub.counts.counts_since(mark)
    assert (since.info, since.warnings) == (1, 0)
    assert any(isinstance(event, Diagnostic) for event in survivor.events)  # the note went out


# --- re-entrancy -----------------------------------------------------------------------


def test_a_reentrant_emit_is_queued_and_drained_after_the_outer_event() -> None:
    emitter, observer = _EmittingRenderer(), RecordingRenderer()
    hub = OutputHub([emitter, observer])
    emitter.hub = hub

    hub.emit(_EVENT)

    # Queued, not dispatched inline: the observer finishes the outer event first.
    assert [type(event).__name__ for event in observer.events] == ["RunStarted", "Diagnostic"]


def test_reentrant_emits_drain_fifo_before_the_outer_emit_returns() -> None:
    emitter, observer = _DoubleEmittingRenderer(), RecordingRenderer()
    hub = OutputHub([emitter, observer])
    emitter.hub = hub

    hub.emit(_EVENT)

    # Both follow-ups drained inside the same emit call, in enqueue order.
    assert isinstance(observer.events[0], RunStarted)
    assert [d.message for d in observer.of_type(Diagnostic)] == ["first", "second"]


# --- the combiner (dispatch outside the hub lock) -------------------------------------


def test_handle_runs_without_the_hub_lock_held() -> None:
    probe = _LockProbeRenderer()
    hub = OutputHub([probe])
    probe.hub = hub

    hub.emit(_EVENT)

    assert probe.lock_was_free == [True]


def test_a_concurrent_emit_hands_off_to_the_active_drainer() -> None:
    gated = _GatedRenderer()
    hub = OutputHub([gated])

    drainer = threading.Thread(target=lambda: hub.emit(_EVENT))
    drainer.start()
    assert gated.entered.wait(timeout=5.0)

    # B's emit returns immediately without dispatching: A still owns the baton.
    hub.emit(Diagnostic(severity=Severity.INFO, message="from B"))
    assert [name for name, _ in gated.handled] == ["RunStarted"]

    gated.release.set()
    drainer.join(timeout=5.0)
    assert not drainer.is_alive()
    assert [name for name, _ in gated.handled] == ["RunStarted", "Diagnostic"]
    assert gated.handled[1][1] is drainer  # A's drain loop delivered B's event


def test_a_propagating_interrupt_never_strands_the_baton() -> None:
    flaky, survivor = _InterruptOnceRenderer(), RecordingRenderer()
    hub = OutputHub([flaky, survivor])

    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)
    assert hub._drainer is None  # the finally released the baton

    follow = Diagnostic(severity=Severity.INFO, message="after")
    hub.emit(follow)  # a fresh emit claims the baton and dispatches
    assert follow in survivor.events


def test_an_interrupt_before_the_drain_loop_cannot_wedge_the_baton(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt landing before _drain runs must leave no stale baton behind.

    Emit only checks-and-calls; the baton is taken inside _drain's try. With the
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

    assert hub._drainer is None  # nothing to strand: the baton was never taken
    follow = Diagnostic(severity=Severity.INFO, message="after")
    hub.emit(follow)  # drains the stranded event AND the new one, in order
    assert [type(event).__name__ for event in survivor.events] == ["RunStarted", "Diagnostic"]


def test_drain_is_a_no_op_loser_when_another_thread_holds_the_baton() -> None:
    """The take race's loser returns without dispatching or touching the owner's baton."""

    survivor = RecordingRenderer()
    hub = OutputHub([survivor])
    other = threading.Thread(target=lambda: None)
    with hub._lock:
        hub._pending.append((_EVENT, 0.0))
        hub._drainer = other

    hub._drain()

    assert survivor.events == []  # the loser dispatched nothing
    assert hub._drainer is other  # and left the owner's baton alone
    with hub._lock:  # hand the baton back so the follow-up emit can drain
        hub._drainer = None
    hub.emit(Diagnostic(severity=Severity.INFO, message="after"))
    assert [type(event).__name__ for event in survivor.events] == ["RunStarted", "Diagnostic"]


def test_lifecycle_waits_for_the_active_drain_to_finish() -> None:
    gated = _GatedRenderer()
    hub = OutputHub([gated])

    drainer = threading.Thread(target=lambda: hub.emit(_EVENT))
    drainer.start()
    assert gated.entered.wait(timeout=5.0)

    lifecycle = threading.Thread(target=lambda: hub.begin_cycle(console_format="plain", level=logging.INFO))
    lifecycle.start()
    # A bounded chance for a broken (non-waiting) begin_cycle to interleave.
    lifecycle.join(timeout=0.2)
    assert lifecycle.is_alive()  # parked until the drain finishes
    assert "cycle" not in gated.log

    gated.release.set()
    drainer.join(timeout=5.0)
    lifecycle.join(timeout=5.0)
    assert gated.log == ["handled:RunStarted", "cycle"]


def test_overflow_sheds_newest_with_one_note_per_episode() -> None:
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])
    mark = hub.counts.mark()

    drainer = threading.Thread(target=lambda: hub.emit(_EVENT))
    drainer.start()
    assert gated.entered.wait(timeout=5.0)

    # Fill the stalled queue to the cap, then two more that must be shed.
    for i in range(QUEUE_CAP):
        hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
    hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-1"))
    hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-2"))

    gated.release.set()
    drainer.join(timeout=30.0)
    assert not drainer.is_alive()

    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert f"q{QUEUE_CAP - 1}" in messages  # everything under the cap still renders
    assert "dropped-1" not in messages and "dropped-2" not in messages
    assert sum("overflowed" in m for m in messages) == 1  # one note, not one per drop
    # Shed events were still counted at enqueue; the note itself is never counted.
    since = hub.counts.counts_since(mark)
    assert (since.info, since.warnings) == (QUEUE_CAP + 3, 0)


def test_overflow_never_sheds_a_structural_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """D1: at the cap only diagnostics shed — a fold input (RunFinished) is appended
    even past the cap, so the breadcrumb frontier can never be corrupted."""

    cap = 3
    monkeypatch.setattr("seadexarr.modules.output.hub.QUEUE_CAP", cap)
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])

    drainer = threading.Thread(target=lambda: hub.emit(_EVENT))
    drainer.start()
    assert gated.entered.wait(timeout=5.0)

    # _EVENT is blocked mid-handle (already popped); fill the pending queue to the cap.
    for i in range(cap):
        hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
    structural = RunFinished(arr=Arr.SONARR)
    hub.emit(structural)  # non-diagnostic at the cap: appended, never shed

    gated.release.set()
    drainer.join(timeout=30.0)
    assert not drainer.is_alive()

    assert structural in observer.events
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert all("overflowed" not in m for m in messages)  # a structural never trips the note


def test_overflow_sheds_diagnostics_and_re_arms_the_note_per_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1 kept behavior: diagnostics past the cap shed (still counted at enqueue) with
    one note per episode; the note re-arms once a drain empties the queue."""

    cap = 3
    monkeypatch.setattr("seadexarr.modules.output.hub.QUEUE_CAP", cap)
    gate, observer = _ReArmableGate(), RecordingRenderer()
    hub = OutputHub([gate, observer])
    mark = hub.counts.mark()

    def run_overflow_episode() -> None:
        gate.arm()
        drainer = threading.Thread(target=lambda: hub.emit(_EVENT))
        drainer.start()
        assert gate.entered.wait(timeout=5.0)
        for i in range(cap):  # fill to the cap
            hub.emit(Diagnostic(severity=Severity.INFO, message=f"q{i}"))
        hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-1"))
        hub.emit(Diagnostic(severity=Severity.INFO, message="dropped-2"))
        gate.release.set()
        drainer.join(timeout=30.0)
        assert not drainer.is_alive()

    run_overflow_episode()
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert f"q{cap - 1}" in messages  # everything under the cap still renders
    assert "dropped-1" not in messages and "dropped-2" not in messages
    assert sum("overflowed" in m for m in messages) == 1  # one note, not one per drop
    assert hub._overflowing is False  # the drain emptied the queue and re-armed the note

    run_overflow_episode()
    messages = [d.message for d in observer.of_type(Diagnostic)]
    assert sum("overflowed" in m for m in messages) == 2  # a fresh note for the new episode

    # Shed diagnostics were counted at enqueue; the note itself is never counted.
    # Per episode: _EVENT + cap diagnostics + 2 dropped = cap + 3 info.
    since = hub.counts.counts_since(mark)
    assert (since.info, since.warnings) == (2 * (cap + 3), 0)


# --- once= dedup ---------------------------------------------------------------------


def test_once_keys_dedup_and_clear_per_cycle() -> None:
    recording = RecordingHub()
    outage = Diagnostic(severity=Severity.WARNING, message="SeaDex unreachable", once_key="seadex-outage")

    recording.emit(outage)
    recording.emit(outage)
    assert len(recording.of_type(Diagnostic)) == 1
    assert recording.hub.counts.mark().warnings == 1  # the dropped copy is not counted

    recording.hub.begin_cycle(console_format="plain", level=logging.INFO)
    recording.emit(outage)
    assert len(recording.of_type(Diagnostic)) == 2


def test_the_same_once_key_from_different_origins_both_pass() -> None:
    recording = RecordingHub()

    recording.emit(Diagnostic(severity=Severity.WARNING, message="a", origin="anilist", once_key="backoff"))
    recording.emit(Diagnostic(severity=Severity.WARNING, message="b", origin="seadex", once_key="backoff"))

    assert [d.origin for d in recording.of_type(Diagnostic)] == ["anilist", "seadex"]


# --- severity counts -------------------------------------------------------------------


def test_severity_counts_mark_and_delta() -> None:
    counts = SeverityCounts()
    counts.record(Severity.WARNING)
    mark = counts.mark()

    counts.record(Severity.WARNING)
    counts.record(Severity.ERROR)
    counts.record(Severity.CRITICAL)

    since = counts.counts_since(mark)
    assert since == SeverityTally(warning=1, error=1, critical=1)
    assert since.warnings == 1
    assert since.errors == 2  # ERROR + CRITICAL
    assert counts.mark().warnings == 2  # totals stay monotonic


def test_hub_tallies_every_emitted_event() -> None:
    recording = RecordingHub()
    mark = recording.hub.counts.mark()

    recording.emit(_EVENT)
    recording.emit(Diagnostic(severity=Severity.ERROR, message="boom"))

    since = recording.hub.counts.counts_since(mark)
    assert (since.info, since.errors) == (1, 1)


def test_severity_is_counted_at_enqueue_even_when_every_renderer_raises() -> None:
    hub = OutputHub([_FailingRenderer(), _FailingRenderer()])
    mark = hub.counts.mark()

    hub.emit(Diagnostic(severity=Severity.ERROR, message="boom"))

    assert hub.counts.counts_since(mark).errors == 1


def test_cycle_counts_reads_the_delta_since_begin_cycle() -> None:
    recording = RecordingHub()
    recording.emit(Diagnostic(severity=Severity.WARNING, message="before"))

    recording.hub.begin_cycle(console_format="plain", level=logging.INFO)
    recording.emit(Diagnostic(severity=Severity.WARNING, message="after"))

    assert recording.hub.cycle_counts().warnings == 1


# --- levels + lifecycle -------------------------------------------------------------


def test_set_level_fans_out_to_every_renderer() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.set_level(logging.DEBUG)

    assert renderer.levels == [logging.DEBUG]


def test_begin_cycle_turns_over_renderers_and_applies_the_level() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.begin_cycle(console_format="plain", level=logging.WARNING)

    assert renderer.cycles == 1
    assert renderer.levels == [logging.WARNING]


def test_close_is_idempotent_and_reaches_every_renderer() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])

    hub.close()
    hub.close()

    assert renderer.closed


def test_a_closed_hub_drops_emits_and_cycle_turnovers() -> None:
    renderer = RecordingRenderer()
    hub = OutputHub([renderer])
    hub.close()

    hub.emit(_EVENT)
    hub.begin_cycle(console_format="plain", level=logging.INFO)

    assert renderer.events == []
    assert renderer.cycles == 0


# --- timestamps ------------------------------------------------------------------------


def test_the_hub_stamps_one_instant_per_emit_for_every_renderer() -> None:
    ticks = iter([100.0, 200.0])
    first, second = RecordingRenderer(), RecordingRenderer()
    hub = OutputHub([first, second], clock=lambda: next(ticks))

    hub.emit(_EVENT)
    hub.emit(_EVENT)

    # One clock read per emit: cross-sink timestamps can never disagree.
    assert first.whens == [100.0, 200.0]
    assert second.whens == [100.0, 200.0]


# --- console ownership (the rich handler's skip predicate) ---------------------------


def test_console_render_active_is_false_without_a_console_seat() -> None:
    hub = OutputHub([RecordingRenderer()])

    assert hub.console_render_active() is False


def test_console_render_active_is_true_for_an_armed_seat() -> None:
    hub = OutputHub([], console=RecordingRenderer())

    assert hub.console_render_active() is True


def test_console_render_active_is_false_once_the_seat_strikes_out() -> None:
    # The quarantine fallback: a struck-out console seat hands the badge class
    # back to the legacy handler, so warnings can never vanish.
    hub = OutputHub([], console=_FailingRenderer(), strike_limit=1)

    hub.emit(_EVENT)

    assert hub.console_render_active() is False


# --- console renderer swap (S3) ------------------------------------------------------


def test_begin_cycle_swaps_the_console_renderer_only_on_format_change() -> None:
    built: list[tuple[LogFormat, RecordingRenderer]] = []

    def factory(console_format: LogFormat) -> Renderer:
        renderer = RecordingRenderer()
        built.append((console_format, renderer))
        return renderer

    sink = RecordingRenderer()
    hub = OutputHub([sink], console_factory=factory)

    hub.begin_cycle(console_format="rich", level=logging.INFO)
    hub.emit(_EVENT)
    hub.begin_cycle(console_format="rich", level=logging.INFO)  # unchanged: no rebuild
    hub.begin_cycle(console_format="plain", level=logging.INFO)  # changed: swap
    hub.emit(_EVENT)

    assert [fmt for fmt, _ in built] == ["rich", "plain"]
    rich_console, plain_console = built[0][1], built[1][1]
    assert rich_console.events == [_EVENT]
    assert rich_console.closed  # the replaced console is torn down
    assert plain_console.events == [_EVENT]
    assert sink.events == [_EVENT, _EVENT]  # the stable sink saw both


def test_a_failing_console_factory_keeps_the_old_seat_and_notes_file_only() -> None:
    console, sink = RecordingRenderer(), RecordingRenderer()

    def factory(console_format: LogFormat) -> Renderer:
        raise RuntimeError("no tty")

    hub = OutputHub([sink], console=console, console_factory=factory)
    hub.begin_cycle(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)

    assert not console.closed
    assert _EVENT in console.events  # the old seat keeps rendering
    notes = [e for e in sink.events if isinstance(e, Diagnostic)]
    assert notes[0].file_only
    assert "console factory failed" in notes[0].message
    assert "keeping the current console" in notes[0].message
