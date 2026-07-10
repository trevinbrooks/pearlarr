# pyright: strict
"""Byte goldens for the wait pass's DURABLE logger output (PR5).

The golden constants were captured at Band A by RUNNING the then-current
``LiveWaitView`` / ``LogWaitView`` (not hand-derived); they are the CONTRACT and
never change. Since the Band C flip the same scenarios drive the REAL producer -
:class:`HubWaitView` -> a real hub -> the :class:`LegacyRenderer` echo - and
must reproduce these bytes exactly: each scenario pins the exact ``(level,
message, payload)`` records a ``CaptureHandler`` on the app logger sees. Only
the durable logger seam is pinned - the rich Live cockpit frames are ephemeral
and deliberately not golden material.

The per-mode matrix (P3): a live TTY logs NO start line and NO pulses
(graduations + tally only); the piped digest adds the start line and the
throttled "still waiting" pulses. The P7 level fact: EVERY wait line -
graduations (failures included), start/pulse, rule, and tally - logs at INFO.
"""

import io
import logging

from rich.console import Console

from seadexarr.modules.console_caps import Capabilities, detect_capabilities
from seadexarr.modules.log import ConsoleRender, RichConsoleHandler, SectionRule, StyledLine, console_payload
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.output import LegacyRenderer, OutputHub, Phase, TorrentView, WaitSnapshot, install_hub
from seadexarr.modules.wait_view import HubWaitView

from .fakes import CaptureHandler

type Line = tuple[int, str, ConsoleRender | None]
"""One pinned record: (levelno, plain message, CONSOLE_EXTRA payload)."""

_I = logging.INFO

# A capable TTY (the live-cockpit seat): color + unicode glyphs.
_LIVE_CAPS = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
# The plain/piped seat (PlainConsoleHandler -> no console): ASCII glyphs, no color.
_PIPE_CAPS = detect_capabilities(None)

_RULE: Line = (_I, "-" * 80, SectionRule(char="-"))


# --- scripted snapshot rows ---------------------------------------------------------


def _queued(key: str, label: str) -> TorrentView:
    return TorrentView(key=key, label=label)


def _downloading(key: str, label: str) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.DOWNLOADING,
        fraction=0.5,
        speed_bps=3_200_000,
        eta_s=130,
        bytes_done=1_450_000_000,
        bytes_total=2_900_000_000,
    )


def _importing(key: str, label: str) -> TorrentView:
    return TorrentView(key=key, label=label, phase=Phase.IMPORTING, command_issued=True)


def _terminal(key: str, label: str, outcome: Outcome, *, files: int | None = None, waited: float = 0.0) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.TERMINAL,
        outcome=outcome,
        import_done=files,
        import_total=files,
        phase_elapsed_s=waited,
    )


# --- goldens ------------------------------------------------------------------------

# Every Outcome member graduates once, in snapshot order, covering every tail
# shape: files+elapsed, elapsed-alone, empty, "retries next run", and the MISSING
# "no longer tracked". Tally: 3 SUCCESS / 5 DEFERRED / 2 FAILED at elapsed 900s.
_ALL_OUTCOMES = (
    _terminal("h1", "Bocchi the Rock!", Outcome.IMPORTED, files=12, waited=192),
    _terminal("h2", "Frieren", Outcome.IMPORTED, waited=192),
    _terminal("h3", "Mushishi", Outcome.IMPORTED),
    _terminal("h4", "Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
    _terminal("h5", "Lycoris Recoil", Outcome.DOWNLOAD_ERRORED),
    _terminal("h6", "Dandadan", Outcome.NO_CONTENT_PATH),
    _terminal("h7", "Vinland Saga", Outcome.STILL_IMPORTING),
    _terminal("h8", "Heavenly Delusion", Outcome.NOT_READY),
    _terminal("h9", "Zom 100", Outcome.NOTHING_TO_IMPORT),
    _terminal("h10", "Oshi no Ko", Outcome.MISSING),
)

LIVE_ALL_OUTCOMES_LINES: tuple[Line, ...] = (
    (_I, "  ✔ imported    Bocchi the Rock!  (12 files · 3m 12s)", StyledLine(style="green")),
    (_I, "  ✔ imported    Frieren  (3m 12s)", StyledLine(style="green")),
    (_I, "  ✔ imported    Mushishi", StyledLine(style="green")),
    (_I, "  ⚠ timed out   Spy x Family  (retries next run)", StyledLine(style="yellow")),
    (_I, "  ✖ errored     Lycoris Recoil  (retries next run)", StyledLine(style="bold red")),
    (_I, "  ⚠ no path     Dandadan  (retries next run)", StyledLine(style="yellow")),
    (_I, "  ⚠ unfinished  Vinland Saga  (retries next run)", StyledLine(style="yellow")),
    (_I, "  ⚠ not ready   Heavenly Delusion  (retries next run)", StyledLine(style="yellow")),
    (_I, "  ⚠ no files    Zom 100  (retries next run)", StyledLine(style="yellow")),
    (_I, "  ✖ gone        Oshi no Ko  (no longer tracked)", StyledLine(style="bold red")),
    _RULE,
    (_I, "  wait complete · 3 imported · 5 left · 2 failed · 15m 00s", None),
)

# The Ctrl-C shape: closed with two torrents still in flight after one graduation.
LIVE_PARTIAL_CLOSE_LINES: tuple[Line, ...] = (
    (_I, "  ✔ imported    Made in Abyss  (4 files · 2m 30s)", StyledLine(style="green")),
    _RULE,
    (_I, "  wait complete · 1 imported · 2m 30s", None),
)

# The piped digest: start line, pulses throttled to max(poll_s, digest_interval)
# = 300s (pulses land at elapsed 300 and 650, re-arming at elapsed + 300), the
# ASCII/uncolored graduations, and the tally. The last update graduates Show B
# AND is past the pulse due at 950: graduations always render before the pulse.
LOG_DIGEST_LINES: tuple[Line, ...] = (
    (_I, "Waiting on 2 downloads to complete and import...", None),
    (_I, "  still waiting · 1 downloading · 1 importing · 0 queued · 5m 00s", None),
    (_I, "  still waiting · 1 downloading · 1 importing · 0 queued · 10m 50s", None),
    (_I, "  ok imported    Show A  (8 files · 5m 00s)", StyledLine(style="")),
    (_I, "  ~ timed out   Show B  (retries next run)", StyledLine(style="")),
    (_I, "  still waiting · 0 downloading · 0 importing · 0 queued · 21m 50s", None),
    _RULE,
    (_I, "  wait complete · 1 imported · 1 left · 21m 50s", None),
)

# poll_s=600 > digest_interval=300: the poll cadence is the pulse floor, so
# elapsed 300 stays silent and 600 pulses. Nothing graduated -> close logs NO
# tally block at all (no rule either).
LOG_POLL_FLOOR_LINES: tuple[Line, ...] = (
    (_I, "Waiting on 1 download to complete and import...", None),
    (_I, "  still waiting · 1 downloading · 0 importing · 0 queued · 10m 00s", None),
)

# One outcome from each category: the tally renders all three segments.
LOG_MIXED_OUTCOME_LINES: tuple[Line, ...] = (
    (_I, "Waiting on 3 downloads to complete and import...", None),
    (_I, "  ok imported    Show A  (1m 00s)", StyledLine(style="")),
    (_I, "  ~ timed out   Show B  (retries next run)", StyledLine(style="")),
    (_I, "  x errored     Show C  (retries next run)", StyledLine(style="")),
    _RULE,
    (_I, "  wait complete · 1 imported · 1 left · 1 failed · 3m 20s", None),
)


# --- the harness: drive the REAL producer through the REAL echo, assert the goldens --


def _narrator(app_logger: logging.Logger, *, pulse_s: float, live: bool) -> tuple[HubWaitView, CaptureHandler]:
    """The real narrator wired to a real hub carrying ONE LegacyRenderer.

    ``live`` attaches a live-capable console handler so the echo's caps probe
    resolves the ``_LIVE_CAPS`` shape (an explicit color_system keeps caps.color
    deterministic across CI envs); without one it resolves ``_PIPE_CAPS``.
    """

    if live:
        console = Console(file=io.StringIO(), force_terminal=True, width=100, color_system="truecolor")
        app_logger.addHandler(RichConsoleHandler(console))
    capture = CaptureHandler()
    app_logger.addHandler(capture)
    install_hub(OutputHub([LegacyRenderer()]))
    return HubWaitView(app_logger, pulse_s=pulse_s, wants_telemetry=live), capture


def _lines(capture: CaptureHandler) -> tuple[Line, ...]:
    return tuple((record.levelno, record.getMessage(), console_payload(record)) for record in capture.records)


class TestLiveWaitParity:
    def test_graduates_every_outcome_then_tallies(self, app_logger: logging.Logger) -> None:
        # Drift pin: the scenario stays exhaustive if Outcome ever grows.
        assert {t.outcome for t in _ALL_OUTCOMES} == set(Outcome)
        view, capture = _narrator(app_logger, pulse_s=300.0, live=True)  # max(poll 30, digest 300)

        view.update(WaitSnapshot((_downloading("h1", "Bocchi the Rock!"), _queued("h4", "Spy x Family")), elapsed_s=10))
        assert _lines(capture) == ()  # a live pass logs NO start line and NO pulses
        view.update(WaitSnapshot(_ALL_OUTCOMES, elapsed_s=900))
        view.close()

        assert _lines(capture) == LIVE_ALL_OUTCOMES_LINES

    def test_partial_close_logs_the_partial_tally(self, app_logger: logging.Logger) -> None:
        # The Ctrl-C shape: run_monitor breaks out and its finally closes the
        # view with torrents still in flight; the tally covers the graduated.
        view, capture = _narrator(app_logger, pulse_s=300.0, live=True)

        view.update(
            WaitSnapshot(
                (
                    _terminal("t1", "Made in Abyss", Outcome.IMPORTED, files=4, waited=150),
                    _downloading("t2", "Still Downloading"),
                    _queued("t3", "Still Queued"),
                ),
                elapsed_s=150,
            ),
        )
        view.close()
        view.close()  # idempotent: a second close adds nothing

        assert _lines(capture) == LIVE_PARTIAL_CLOSE_LINES


class TestLogWaitParity:
    def test_start_throttled_pulses_graduations_tally(self, app_logger: logging.Logger) -> None:
        view, capture = _narrator(app_logger, pulse_s=300.0, live=False)  # max(poll 30, digest 300)
        in_flight = (_downloading("q1", "Show A"), _importing("q2", "Show B"))

        view.update(WaitSnapshot((_queued("q1", "Show A"), _queued("q2", "Show B")), elapsed_s=0))  # start line
        view.update(WaitSnapshot(in_flight, elapsed_s=299))  # within the interval: silent
        view.update(WaitSnapshot(in_flight, elapsed_s=300))  # first pulse; re-arms at 600
        view.update(WaitSnapshot(in_flight, elapsed_s=599))  # still silent
        view.update(WaitSnapshot(in_flight, elapsed_s=650))  # second pulse; re-arms at 950
        view.update(
            WaitSnapshot(
                (_terminal("q1", "Show A", Outcome.IMPORTED, files=8, waited=300), in_flight[1]),
                elapsed_s=940,
            ),
        )  # graduation only: 940 < 950
        view.update(
            WaitSnapshot(
                (
                    _terminal("q1", "Show A", Outcome.IMPORTED, files=8, waited=300),
                    _terminal("q2", "Show B", Outcome.DOWNLOAD_TIMED_OUT),
                ),
                elapsed_s=1310,
            ),
        )  # graduation, then the due pulse
        view.close()

        assert _lines(capture) == LOG_DIGEST_LINES

    def test_pulse_interval_is_the_max_of_poll_and_digest(self, app_logger: logging.Logger) -> None:
        view, capture = _narrator(app_logger, pulse_s=600.0, live=False)  # max(poll 600, digest 300)
        row = (_downloading("h", "Slow Show"),)

        view.update(WaitSnapshot(row, elapsed_s=0))
        view.update(WaitSnapshot(row, elapsed_s=300))  # digest interval alone would pulse here
        view.update(WaitSnapshot(row, elapsed_s=600))  # the poll floor rules
        view.close()  # nothing graduated -> no tally block

        assert _lines(capture) == LOG_POLL_FLOOR_LINES

    def test_mixed_outcomes_tally_pins_the_failed_segment(self, app_logger: logging.Logger) -> None:
        view, capture = _narrator(app_logger, pulse_s=300.0, live=False)

        view.update(
            WaitSnapshot((_queued("m1", "Show A"), _queued("m2", "Show B"), _queued("m3", "Show C")), elapsed_s=0),
        )
        view.update(
            WaitSnapshot(
                (
                    _terminal("m1", "Show A", Outcome.IMPORTED, waited=60),
                    _terminal("m2", "Show B", Outcome.DOWNLOAD_TIMED_OUT),
                    _terminal("m3", "Show C", Outcome.DOWNLOAD_ERRORED),
                ),
                elapsed_s=200,
            ),
        )
        view.close()

        assert _lines(capture) == LOG_MIXED_OUTCOME_LINES


def test_every_wait_line_logs_at_info() -> None:
    """The P7 fact, pinned whole: the wait surface has exactly one level."""

    all_lines = (
        LIVE_ALL_OUTCOMES_LINES
        + LIVE_PARTIAL_CLOSE_LINES
        + LOG_DIGEST_LINES
        + LOG_POLL_FLOOR_LINES
        + LOG_MIXED_OUTCOME_LINES
    )
    assert {level for level, _, _ in all_lines} == {logging.INFO}
