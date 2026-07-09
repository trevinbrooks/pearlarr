# pyright: strict
"""Tests for the OutputHub (``output.hub``).

Pin the dispatch contract (every surface, emit order), per-event containment
with N strikes + per-cycle re-arm (never process-latching), the once= dedup
store, SeverityCounts mark/counts_since, set_level fan-out, and the console
renderer swap on a format change at begin_cycle.
"""

import logging
from typing import override

import pytest

from seadexarr.modules.config import LogFormat
from seadexarr.modules.output import (
    STRIKE_LIMIT,
    Diagnostic,
    Event,
    NullRenderer,
    OutputHub,
    Renderer,
    RunStarted,
    Severity,
    SeverityCounts,
    SeverityTally,
)
from seadexarr.modules.output.recording import RecordingHub, RecordingRenderer

_EVENT = RunStarted(version="v1.0.0", data_dir="/data")


class _FailingRenderer:
    """A renderer whose handle always raises (the containment test double)."""

    def __init__(self) -> None:
        self.calls = 0

    def handle(self, event: Event, when: float) -> None:
        self.calls += 1
        raise RuntimeError("render bug")

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


class _InterruptingRenderer(_FailingRenderer):
    @override
    def handle(self, event: Event, when: float) -> None:
        raise KeyboardInterrupt


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
