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
from dataclasses import replace
from typing import ClassVar, override

import pytest

from pearlarr.config import Arr, LogFormat
from pearlarr.log import LOG_NAME
from pearlarr.output import (
    ArrRunFinished,
    Diagnostic,
    Event,
    NullRenderer,
    OutputHub,
    PearlarrStarted,
    Renderer,
    Severity,
    SeverityCounter,
    SeverityTally,
    emit_to_hub,
    hub_note,
    install_hub,
    uninstall_hub,
)

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
    """Raises from every handle, and logs the type of each event it's handed and each begin_run and close."""

    def __init__(self) -> None:
        self.log: list[str] = []

    @override
    def handle(self, event: Event, when: float) -> None:
        self.log.append(type(event).__name__)
        raise RuntimeError("render bug")

    @override
    def begin_run(self) -> None:
        self.log.append("begin_run")

    @override
    def close(self) -> None:
        self.log.append("close")


class _FailOnMarkerRenderer(_FailingRenderer):
    """Raises on PearlarrStarted only, so an arr run's close (ArrRunFinished) renders cleanly between failures."""

    @override
    def handle(self, event: Event, when: float) -> None:
        self.log.append(type(event).__name__)
        if isinstance(event, PearlarrStarted):
            raise RuntimeError("render bug")


class _FailingCloseRenderer(_FailingRenderer):
    """Raises from close too."""

    @override
    def close(self) -> None:
        super().close()
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


# --- dispatch ---------------------------------------------------------------------


def test_every_renderer_sees_every_event_in_order() -> None:
    first, second, console = RecordingRenderer(), RecordingRenderer(), RecordingRenderer()
    hub = OutputHub([first, second], console=console)  # a console handed in, as a subcommand's hub has
    events: list[Event] = [_EVENT, Diagnostic(severity=Severity.WARNING, message="w")]

    for event in events:
        hub.emit(event)

    assert first.events == events
    assert second.events == events
    assert console.events == events


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


class _FailingFileLog(_FailingRenderer):
    """A file log renderer that raises from every handle, as one writing to a dying disk would."""

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


# --- containment --------------------------------------------------------------------


def _untraced(events: list[Event]) -> list[Event]:
    """The events with each diagnostic's trace dropped, so a containment note compares whole."""

    return [replace(event, trace=None) if isinstance(event, Diagnostic) else event for event in events]


def test_a_raising_renderer_leaves_the_file_log_an_uncounted_note() -> None:
    """The note is file-only while the file log survives, and uncounted: a presentation bug isn't a run warning."""

    file_log = _FileLog()
    hub = OutputHub([_FailingRenderer(), file_log])

    hub.emit(_EVENT)

    assert _untraced(file_log.events) == [
        _EVENT,
        Diagnostic(
            severity=Severity.WARNING,
            message="renderer _FailingRenderer failed on PearlarrStarted: render bug",
            origin="output.hub",
            file_only=True,
        ),
    ]
    note = file_log.events[1]
    assert isinstance(note, Diagnostic) and note.trace is not None
    assert "RuntimeError: render bug" in note.trace.plain
    assert hub.counts.mark() == SeverityTally(info=1)


def test_losing_the_file_log_renderer_escalates_the_note_to_a_counted_visible_error() -> None:
    """A run must never silently lose its whole log file."""

    survivor = RecordingRenderer()
    hub = OutputHub([_FailingFileLog(), survivor])

    hub.emit(_EVENT)

    assert _untraced(survivor.events) == [
        _EVENT,
        Diagnostic(
            severity=Severity.ERROR,
            message="renderer _FailingFileLog failed on PearlarrStarted: render bug",
            origin="output.hub",
        ),
    ]
    assert hub.counts.mark() == SeverityTally(info=1, error=1)


def test_a_warning_is_counted_even_when_every_renderer_raises_on_it() -> None:
    hub = OutputHub([_FailingRenderer(), _FailingRenderer()])

    hub.emit(Diagnostic(severity=Severity.WARNING, message="boom"))

    # Counted when it's emitted, though nothing rendered it. With no file log left, both notes are counted errors.
    assert hub.counts.mark() == SeverityTally(warning=1, error=2)


def test_a_renderer_that_fails_on_a_note_takes_a_strike_for_it() -> None:
    first, second = _FailingRenderer(), _FailingRenderer()
    hub = OutputHub([first, second])

    hub.emit(_EVENT)
    hub.emit(_EVENT)

    # Each fails on the event and on the other's note, so the second event is each one's third strike. The second
    # one's note about that strike-out then has nowhere to go: the first is already out.
    assert first.log == ["PearlarrStarted", "Diagnostic", "PearlarrStarted", "close"]


def test_a_renderer_strikes_out_on_its_third_failure_until_the_next_run() -> None:
    """Striking out closes the renderer, so a struck-out live display stops repainting. begin_run restarts it."""

    flaky = _FailingRenderer()
    hub = OutputHub([flaky, RecordingRenderer()])

    for _ in range(4):
        hub.emit(_EVENT)
    hub.begin_run(console_format="plain", level=logging.INFO)
    for _ in range(4):
        hub.emit(_EVENT)

    # Closed once at the third failure and skipped after it, in each run.
    struck_out = ["PearlarrStarted"] * 3 + ["close"]
    assert flaky.log == [*struck_out, "begin_run", *struck_out]


def test_strikes_carry_across_an_arr_run_boundary() -> None:
    """An arr run's close (ArrRunFinished) is not a run turnover: a clean render between failures resets nothing."""

    flaky = _FailOnMarkerRenderer()
    hub = OutputHub([flaky, RecordingRenderer()])

    for event in [_EVENT, _EVENT, ArrRunFinished(arr=Arr.SONARR), _EVENT, _EVENT]:
        hub.emit(event)

    assert flaky.log == ["PearlarrStarted", "PearlarrStarted", "ArrRunFinished", "PearlarrStarted", "close"]


def test_a_raising_close_never_escapes_the_strike_out() -> None:
    flaky = _FailingCloseRenderer()
    hub = OutputHub([flaky, RecordingRenderer()])

    for _ in range(3):
        hub.emit(_EVENT)  # the third failure's close raises inside the hub

    assert flaky.log == ["PearlarrStarted"] * 3 + ["close"]


def test_the_final_strike_says_the_renderer_is_disabled() -> None:
    survivor = RecordingRenderer()
    hub = OutputHub([_FailingRenderer(), survivor])

    for _ in range(3):
        hub.emit(_EVENT)

    failed = "renderer _FailingRenderer failed on PearlarrStarted: render bug"
    assert [note.message for note in survivor.of_type(Diagnostic)] == [
        failed,
        failed,
        f"{failed} (disabled until the next run)",
    ]


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


# --- interrupts ---------------------------------------------------------------------

_EXIT_LINE = Diagnostic(severity=Severity.INFO, message="Received SIGTERM - exiting")


class _SigtermMidDispatch(_QuietRenderer):
    """The SIGTERM handler landing inside this renderer's first handle: it emits its exit line, then raises SystemExit."""

    def __init__(self) -> None:
        self.signalled = False

    @override
    def handle(self, event: Event, when: float) -> None:
        if not self.signalled:
            self.signalled = True
            emit_to_hub(_EXIT_LINE)
            raise SystemExit(0)


def test_a_sigterm_mid_dispatch_still_flushes_its_exit_line() -> None:
    survivor = RecordingRenderer()
    hub = OutputHub([_SigtermMidDispatch(), survivor])
    install_hub(hub)

    with pytest.raises(SystemExit):
        hub.emit(_EVENT)

    # The event the exit cut short and the exit line queued behind it both reach the survivor.
    assert survivor.events == [_EVENT, _EXIT_LINE]


_QUEUED = Diagnostic(severity=Severity.INFO, message="queued")
_AFTER = Diagnostic(severity=Severity.INFO, message="after")


class _CtrlCMidDispatch(_QuietRenderer):
    """Ctrl-C inside the first handle after queueing a follow-up, and Ctrl-C again whenever it's handed `again_on`."""

    def __init__(self, again_on: Event) -> None:
        self.again_on = again_on
        self.queued = False

    @override
    def handle(self, event: Event, when: float) -> None:
        if not self.queued:
            self.queued = True
            emit_to_hub(_QUEUED)
            raise KeyboardInterrupt
        if event == self.again_on:
            raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("again_on", "survivor_gets"),
    [
        pytest.param(_EVENT, [_EVENT, _QUEUED, _AFTER], id="as-the-unwind-retries-the-event"),
        pytest.param(_QUEUED, [_EVENT, _AFTER], id="as-the-unwind-flushes-the-queue"),
    ],
)
def test_a_second_ctrl_c_during_the_unwind(again_on: Event, survivor_gets: list[Event]) -> None:
    """One on the retry of the cut-short event is contained, so the queue still flushes.

    One on the queue cuts the flush short, but the hub still dispatches the next emit.
    """

    survivor = RecordingRenderer()
    hub = OutputHub([_CtrlCMidDispatch(again_on), survivor])
    install_hub(hub)

    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)
    hub.emit(_AFTER)

    assert survivor.events == survivor_gets


def test_an_interrupt_before_the_drain_loop_cannot_wedge_the_drain_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt that lands after emit queues the event but before the drain starts leaves no drain lock behind."""

    survivor = RecordingRenderer()
    hub = OutputHub([survivor])

    def interrupted_drain() -> None:
        raise KeyboardInterrupt

    # That window is a few bytecodes wide, so the only way to land in it is to stand in for the drain.
    monkeypatch.setattr(hub, "_drain", interrupted_drain)
    with pytest.raises(KeyboardInterrupt):
        hub.emit(_EVENT)
    monkeypatch.undo()

    hub.emit(_AFTER)  # drains the stranded event and the new one, in order
    assert survivor.events == [_EVENT, _AFTER]


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


# --- overflow ---------------------------------------------------------------------

_FULL_QUEUE = [Diagnostic(severity=Severity.INFO, message=f"q{i}") for i in range(10_000)]
_OVERFLOW_NOTE = Diagnostic(
    severity=Severity.WARNING,
    message="output queue overflowed at 10000 events; shedding newest events until it drains",
    origin="output.hub",
    file_only=True,
)


def _hold_full_queue(hub: OutputHub, gated: _GatedRenderer) -> threading.Thread:
    """Hold _EVENT mid-handle on a drainer thread, fill the queue behind it to its cap, and return the thread."""

    drainer = _emit_held_at(hub, gated)
    for event in _FULL_QUEUE:
        hub.emit(event)
    return drainer


def test_a_full_queue_sheds_new_diagnostics_with_one_note() -> None:
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])

    drainer = _hold_full_queue(hub, gated)
    hub.emit(Diagnostic(severity=Severity.INFO, message="shed"))
    hub.emit(Diagnostic(severity=Severity.INFO, message="shed too"))
    _release(gated, drainer)

    assert observer.events == [_EVENT, *_FULL_QUEUE, _OVERFLOW_NOTE]
    # The two shed still count toward the run's tally. The note is file-only, so it doesn't.
    assert hub.counts.mark() == SeverityTally(info=10_003)


def test_a_full_queue_never_sheds_a_structural_event() -> None:
    """An arr run's close is one of the events the open scopes are built from, so it's queued past the cap."""

    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])
    structural = ArrRunFinished(arr=Arr.SONARR)

    drainer = _hold_full_queue(hub, gated)
    hub.emit(structural)
    _release(gated, drainer)

    assert observer.events == [_EVENT, *_FULL_QUEUE, structural]  # and no note


def test_each_overflow_gets_its_own_note_once_the_queue_drains() -> None:
    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])

    for _ in range(2):
        gated.arm()
        drainer = _hold_full_queue(hub, gated)
        hub.emit(Diagnostic(severity=Severity.INFO, message="shed"))
        _release(gated, drainer)

    assert observer.events.count(_OVERFLOW_NOTE) == 2


def test_a_once_keyed_diagnostic_shed_at_the_cap_lands_when_emitted_again() -> None:
    """It's shed whole: its key isn't taken and it isn't counted, so the next emit of it is the one shown and counted."""

    gated, observer = _GatedRenderer(), RecordingRenderer()
    hub = OutputHub([gated, observer])
    keyed = Diagnostic(severity=Severity.WARNING, message="SeaDex unreachable", once_key="seadex-outage")

    drainer = _hold_full_queue(hub, gated)
    hub.emit(keyed)
    _release(gated, drainer)
    hub.emit(keyed)

    assert observer.events[-2:] == [_OVERFLOW_NOTE, keyed]
    assert hub.counts.mark() == SeverityTally(info=10_001, warning=1)


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


# --- the stderr fallback ----------------------------------------------------------


def test_a_visible_warning_reaches_stderr_before_the_first_run_seats_a_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Between installing the hub and the first begin_run there's no console, so stderr is the only way to a human."""

    hub = OutputHub([_FileLog()], console_factory=lambda console_format: NullRenderer())

    hub.emit(Diagnostic(severity=Severity.WARNING, message="watch out"))

    assert capsys.readouterr().err == "WARNING [app] watch out\n"


def test_a_visible_warning_reaches_stderr_once_the_console_strikes_out(capsys: pytest.CaptureFixture[str]) -> None:
    hub = OutputHub([_FileLog()], console_factory=lambda console_format: _FailingRenderer())
    hub.begin_run(console_format="rich", level=logging.INFO)

    for _ in range(3):
        hub.emit(_EVENT)  # the console strikes out, and the notes about it are the file log's
    hub.emit(Diagnostic(severity=Severity.WARNING, message="watch out"))

    assert capsys.readouterr().err == "WARNING [app] watch out\n"


def test_losing_the_file_log_before_the_first_run_tells_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    hub = OutputHub([_FailingFileLog()], console_factory=lambda console_format: NullRenderer())

    hub.emit(_EVENT)

    # The note escalates to a visible error, and with no console seated yet, stderr is where it goes.
    failed = "renderer _FailingFileLog failed on PearlarrStarted: render bug"
    assert capsys.readouterr().err == f"ERROR [output.hub] {failed}\n"


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


class _BrokenStderr(io.StringIO):
    """A stderr whose pipe is gone."""

    @override
    def write(self, s: str) -> int:
        raise BrokenPipeError


def test_a_broken_stderr_never_raises_through_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken pipe on stderr is contained like a renderer's failure, so it can't abort the run."""

    file_log = _FileLog()
    hub = OutputHub([file_log], console_factory=lambda console_format: NullRenderer())
    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    warning = Diagnostic(severity=Severity.WARNING, message="watch out")

    hub.emit(warning)  # no console is seated yet, so this one goes to stderr too

    assert file_log.events == [warning]


# --- the console renderer ----------------------------------------------------------


def test_a_blocked_console_never_starves_the_file_log() -> None:
    """The console is handed each event after the file log, so a stuck terminal can't keep an event out of the file."""

    console, file_log = _GatedRenderer(), _FileLog()
    hub = OutputHub([file_log], console_factory=lambda console_format: console)
    hub.begin_run(console_format="rich", level=logging.INFO)

    drainer = _emit_held_at(hub, console)
    assert file_log.events == [_EVENT]
    _release(console, drainer)


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


def test_a_console_that_raises_on_close_is_still_swapped_out() -> None:
    old, new = _FailingCloseRenderer(), RecordingRenderer()
    consoles: dict[LogFormat, Renderer] = {"rich": old, "plain": new}
    hub = OutputHub([], console_factory=lambda console_format: consoles[console_format])

    hub.begin_run(console_format="rich", level=logging.INFO)
    hub.begin_run(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)

    assert old.log == ["begin_run", "close"]
    assert new.events == [_EVENT]


def test_a_failing_console_factory_keeps_the_old_console_renderer_and_notes_file_only() -> None:
    console, file_log = RecordingRenderer(), _FileLog()

    def factory(console_format: LogFormat) -> Renderer:
        if console_format == "plain":
            raise RuntimeError("no tty")
        return console

    hub = OutputHub([file_log], console_factory=factory)
    hub.begin_run(console_format="rich", level=logging.INFO)
    hub.begin_run(console_format="plain", level=logging.INFO)
    hub.emit(_EVENT)

    note = Diagnostic(
        severity=Severity.WARNING,
        message="console factory failed for format 'plain': no tty; keeping the current console",
        origin="output.hub",
        file_only=True,
    )
    assert _untraced(file_log.events) == [note, _EVENT]
    # The old console keeps rendering. It's handed the file-only note too, and a real console drops those itself.
    assert _untraced(console.events) == [note, _EVENT]
    assert not console.closed


# --- the process registry (install/uninstall) -----------------------------------


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


class _LifecycleCaller(_QuietRenderer):
    """On each PearlarrStarted it's handed, emits a follow-up and then sets the level from inside `handle`.

    It's handed its hub rather than finding the process hub. A set_level that waited on its own drain would leave a
    thread holding the drain lock for good, and closing an installed hub at teardown would then wait on it forever.
    """

    def __init__(self) -> None:
        self.hub: OutputHub | None = None

    @override
    def handle(self, event: Event, when: float) -> None:
        if isinstance(event, PearlarrStarted) and self.hub is not None:
            self.hub.emit(Diagnostic(severity=Severity.INFO, message="queued behind"))
            self.hub.set_level(logging.WARNING)


def test_a_mid_dispatch_lifecycle_call_keeps_the_drain_lock_and_the_emit_order() -> None:
    """`set_level` from inside a renderer's handle neither waits on its own drain, flushes inside it, nor keeps its drain lock."""

    recorder = RecordingRenderer()
    caller = _LifecycleCaller()
    hub = OutputHub([caller, recorder])
    caller.hub = hub

    # On a thread of its own: a set_level that waited on its own drain would never return.
    emitter = threading.Thread(target=lambda: hub.emit(_EVENT), daemon=True)
    emitter.start()
    emitter.join(timeout=30.0)
    assert not emitter.is_alive(), "set_level inside handle waited on its own drain"
    hub.emit(_AFTER)  # dispatches only if the drain lock came back

    # The in-flight event finishes its fan-out before the follow-up it queued, and nothing goes out twice.
    assert recorder.events == [_EVENT, Diagnostic(severity=Severity.INFO, message="queued behind"), _AFTER]


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
