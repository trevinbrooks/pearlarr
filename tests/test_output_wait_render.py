# pyright: strict
"""Tests for the wait surface's legacy echo + throttle (PR5 Band B).

The parity centerpiece: drive the REAL ``LegacyRenderer`` through the five Band A
scenarios as event sequences (WaitStarted/WaitProgress/TorrentGraduated/
WaitFinished built from the same scenario data) and assert the echoed
``(level, message, payload)`` records equal the ``tests/test_wait_parity``
goldens byte-for-byte - the goldens are IMPORTED, never copied. Live scenarios
force a live-capable console handler (so caps.color/unicode are deterministic,
not env-derived); the non-TTY digest scenarios run with no console handler. Plus
the shared :class:`PulseThrottle` unit contract and the builder edge cases the
goldens don't isolate.
"""

import io
import logging
from collections.abc import Iterable

from rich.console import Console

from seadexarr.modules.console_caps import detect_capabilities
from seadexarr.modules.log import RichConsoleHandler, StyledLine, console_payload, hub_event_marked
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.output import (
    Event,
    LegacyRenderer,
    Phase,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
)
from seadexarr.modules.output.wait_lines import (
    PulseThrottle,
    wait_graduation_line,
    wait_start_line,
    wait_tally_lines,
)

from .fakes import CaptureHandler
from .test_wait_parity import (
    LIVE_ALL_OUTCOMES_LINES,
    LIVE_PARTIAL_CLOSE_LINES,
    LOG_DIGEST_LINES,
    LOG_MIXED_OUTCOME_LINES,
    LOG_POLL_FLOOR_LINES,
    Line,
)

# --- event-sequence builders (the same scenario data, as the narrator emits it) ------


def _dl(label: str) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.DOWNLOADING)


def _imp(label: str) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.IMPORTING)


def _q(label: str) -> TorrentView:
    return TorrentView(key=label, label=label)


def _term(label: str, outcome: Outcome) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.TERMINAL, outcome=outcome)


def _progress(*torrents: TorrentView, elapsed: float) -> WaitProgress:
    return WaitProgress(snapshot=WaitSnapshot(torrents=torrents, elapsed_s=elapsed))


def _grad(label: str, outcome: Outcome, *, files: int | None = None, waited: float = 0.0) -> TorrentGraduated:
    return TorrentGraduated(label=label, outcome=outcome, files=files, waited_s=waited)


# --- the harness: drive the REAL LegacyRenderer, capture its echoed records ------------


def _echo(app_logger: logging.Logger, events: Iterable[Event], *, live: bool) -> list[logging.LogRecord]:
    if live:
        # An explicit color_system so caps.color is True regardless of the CI env
        # (force_terminal alone leaves color to TERM/COLORTERM detection).
        console = Console(file=io.StringIO(), force_terminal=True, width=100, color_system="truecolor")
        app_logger.addHandler(RichConsoleHandler(console))
    capture = CaptureHandler()
    app_logger.addHandler(capture)
    renderer = LegacyRenderer()
    for event in events:
        renderer.handle(event, 0.0)
    return capture.records


def _as_lines(records: Iterable[logging.LogRecord]) -> tuple[Line, ...]:
    return tuple((record.levelno, record.getMessage(), console_payload(record)) for record in records)


# The 10-outcome ledger, one graduation per Outcome (files/elapsed / elapsed-alone /
# empty / retries / no-longer-tracked tails), then the 3/5/2 tally at 900s.
_ALL_OUTCOME_GRADS: tuple[TorrentGraduated, ...] = (
    _grad("Bocchi the Rock!", Outcome.IMPORTED, files=12, waited=192),
    _grad("Frieren", Outcome.IMPORTED, waited=192),
    _grad("Mushishi", Outcome.IMPORTED),
    _grad("Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
    _grad("Lycoris Recoil", Outcome.DOWNLOAD_ERRORED),
    _grad("Dandadan", Outcome.NO_CONTENT_PATH),
    _grad("Vinland Saga", Outcome.STILL_IMPORTING),
    _grad("Heavenly Delusion", Outcome.NOT_READY),
    _grad("Zom 100", Outcome.NOTHING_TO_IMPORT),
    _grad("Oshi no Ko", Outcome.MISSING),
)


class TestLegacyWaitParity:
    def test_live_graduations_and_tally_no_start_or_pulses(self, app_logger: logging.Logger) -> None:
        events: list[Event] = [
            WaitStarted(total=2, pulse_s=300.0),
            _progress(_dl("Bocchi the Rock!"), _q("Spy x Family"), elapsed=10),  # inert on a live console
            *_ALL_OUTCOME_GRADS,
            _progress(elapsed=900),  # inert
            WaitFinished(imported=3, deferred=5, failed=2, elapsed_s=900),
        ]
        records = _echo(app_logger, events, live=True)
        assert _as_lines(records) == LIVE_ALL_OUTCOMES_LINES
        assert all(hub_event_marked(record) for record in records)

    def test_live_partial_close_tally(self, app_logger: logging.Logger) -> None:
        events: list[Event] = [
            WaitStarted(total=3, pulse_s=300.0),
            _grad("Made in Abyss", Outcome.IMPORTED, files=4, waited=150),
            _progress(_dl("Still Downloading"), _q("Still Queued"), elapsed=150),  # inert
            WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=150),
        ]
        records = _echo(app_logger, events, live=True)
        assert _as_lines(records) == LIVE_PARTIAL_CLOSE_LINES
        assert all(hub_event_marked(record) for record in records)

    def test_digest_start_throttled_pulses_graduation_before_pulse(self, app_logger: logging.Logger) -> None:
        in_flight = (_dl("Show A"), _imp("Show B"))
        events: list[Event] = [
            WaitStarted(total=2, pulse_s=300.0),  # digest interval max(30, 300)
            _progress(_q("Show A"), _q("Show B"), elapsed=0),  # the start snapshot never pulses
            _progress(*in_flight, elapsed=299),  # within the interval
            _progress(*in_flight, elapsed=300),  # first pulse; re-arms at 600
            _progress(*in_flight, elapsed=599),  # still silent
            _progress(*in_flight, elapsed=650),  # second pulse; re-arms at 950
            _grad("Show A", Outcome.IMPORTED, files=8, waited=300),
            _progress(_term("Show A", Outcome.IMPORTED), _imp("Show B"), elapsed=940),  # 940 < 950: silent
            _grad("Show B", Outcome.DOWNLOAD_TIMED_OUT),
            _progress(_term("Show A", Outcome.IMPORTED), _term("Show B", Outcome.DOWNLOAD_TIMED_OUT), elapsed=1310),
            WaitFinished(imported=1, deferred=1, failed=0, elapsed_s=1310),
        ]
        records = _echo(app_logger, events, live=False)
        assert _as_lines(records) == LOG_DIGEST_LINES
        assert all(hub_event_marked(record) for record in records)

    def test_digest_poll_floor_and_zero_tally_silence(self, app_logger: logging.Logger) -> None:
        row = (_dl("Slow Show"),)
        events: list[Event] = [
            WaitStarted(total=1, pulse_s=600.0),  # poll_s=600 is the floor over digest_interval=300
            _progress(*row, elapsed=0),
            _progress(*row, elapsed=300),  # the digest interval alone would pulse; the poll floor rules
            _progress(*row, elapsed=600),  # the poll floor pulses
            WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=600),  # nothing graduated -> no tally block
        ]
        records = _echo(app_logger, events, live=False)
        assert _as_lines(records) == LOG_POLL_FLOOR_LINES

    def test_digest_mixed_outcomes_tally(self, app_logger: logging.Logger) -> None:
        events: list[Event] = [
            WaitStarted(total=3, pulse_s=300.0),
            _progress(_q("Show A"), _q("Show B"), _q("Show C"), elapsed=0),
            _grad("Show A", Outcome.IMPORTED, waited=60),
            _grad("Show B", Outcome.DOWNLOAD_TIMED_OUT),
            _grad("Show C", Outcome.DOWNLOAD_ERRORED),
            _progress(
                _term("Show A", Outcome.IMPORTED),
                _term("Show B", Outcome.DOWNLOAD_TIMED_OUT),
                _term("Show C", Outcome.DOWNLOAD_ERRORED),
                elapsed=200,
            ),
            WaitFinished(imported=1, deferred=1, failed=1, elapsed_s=200),
        ]
        records = _echo(app_logger, events, live=False)
        assert _as_lines(records) == LOG_MIXED_OUTCOME_LINES

    def test_raised_level_suppresses_every_wait_echo(self, app_logger: logging.Logger) -> None:
        # Logger parity: at configured WARNING the INFO wait lines vanish from the
        # file exactly as they would from the console (all wait lines are INFO).
        app_logger.setLevel(logging.WARNING)
        events: list[Event] = [
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show A"), elapsed=0),
            _progress(_dl("Show A"), elapsed=300),
            _grad("Show A", Outcome.IMPORTED, files=1, waited=60),
            WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=300),
        ]
        assert _echo(app_logger, events, live=False) == []


# --- the shared PulseThrottle -----------------------------------------------------------


class TestPulseThrottle:
    def test_disarmed_never_fires(self) -> None:
        throttle = PulseThrottle()
        assert throttle.fire(0.0) is False
        assert throttle.fire(10_000.0) is False

    def test_first_fire_after_arm_is_skipped(self) -> None:
        # The old view's first render printed the start line and returned.
        throttle = PulseThrottle()
        throttle.arm(300.0)
        assert throttle.fire(0.0) is False

    def test_elapsed_anchored_cadence(self) -> None:
        throttle = PulseThrottle()
        throttle.arm(300.0)
        assert throttle.fire(0.0) is False  # skip-first
        assert throttle.fire(299.0) is False  # within the interval
        assert throttle.fire(300.0) is True  # due; re-arms at 600
        assert throttle.fire(599.0) is False
        assert throttle.fire(650.0) is True  # elapsed-anchored: re-arms at 950, not a 900 grid mark
        assert throttle.fire(940.0) is False  # 940 < 950
        assert throttle.fire(1310.0) is True

    def test_poll_floor_interval(self) -> None:
        throttle = PulseThrottle()
        throttle.arm(600.0)
        assert throttle.fire(0.0) is False  # skip-first
        assert throttle.fire(300.0) is False  # a shorter interval would fire; 600 rules
        assert throttle.fire(600.0) is True

    def test_reset_disarms(self) -> None:
        throttle = PulseThrottle()
        throttle.arm(300.0)
        assert throttle.fire(0.0) is False
        throttle.reset()
        assert throttle.fire(10_000.0) is False

    def test_re_arm_restarts_the_skip_first(self) -> None:
        throttle = PulseThrottle()
        throttle.arm(300.0)
        assert throttle.fire(0.0) is False
        assert throttle.fire(300.0) is True
        throttle.arm(600.0)  # a fresh pass
        assert throttle.fire(0.0) is False  # skip-first again
        assert throttle.fire(600.0) is True


# --- builder edge cases the goldens don't isolate ---------------------------------------


class TestWaitBuilders:
    def test_zero_graduation_tally_is_empty(self) -> None:
        assert wait_tally_lines(WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=42.0)) == []

    def test_start_line_pluralizes_and_carries_no_payload(self) -> None:
        line = wait_start_line(WaitStarted(total=1, pulse_s=300.0))
        assert line.message == "Waiting on 1 download to complete and import..."
        assert line.payload is None

    def test_graduation_style_is_dropped_without_color(self) -> None:
        # Parity with log_styled(..., None): the colorless case is style="", not None.
        line = wait_graduation_line(_grad("Show A", Outcome.IMPORTED, files=1, waited=60), detect_capabilities(None))
        assert line.payload == StyledLine(style="")
        assert line.message == "  ok imported    Show A  (1 file · 1m 00s)"
