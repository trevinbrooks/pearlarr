# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the lifecycle tests read RichRenderer._wait._live and drive WaitRegion/_LiveFrame.
"""Tests for the wait cockpit's renderer side (``output.wait_region``) + routing.

The RichRenderer's wait region opens its single ``rich.Live`` on the FIRST
WaitProgress (never on WaitStarted), graduates finished torrents to durable
scrollback while the cockpit is active, stops the Live before the closing tally
prints, degrades to a start line + throttled pulses on a non-live console, and
tears its slot down whenever the wait region leaves the fold's frontier
(ScopeClosed, a RunFinished unwind). A raised level suppresses the durable lines
but never the cockpit; a raising frame build degrades to an empty frame.
"""

import io
import logging

from rich.console import Console, Group

from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.output import (
    Event,
    Phase,
    RichRenderer,
    RunFinished,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
)
from seadexarr.modules.output.wait_region import WaitRegion, _LiveFrame

from .fakes import CaptureHandler, strip_ansi

_WAIT = ScopeId(ScopeKind.WAIT_REGION, 700)
_WAIT_TWO = ScopeId(ScopeKind.WAIT_REGION, 701)


def _live_renderer(width: int = 100) -> tuple[RichRenderer, io.StringIO]:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=width)
    # A frozen clock: the cockpit's between-poll ticking stays deterministic.
    return RichRenderer(lambda: console, time_source=lambda: 0.0), stream


def _feed(renderer: RichRenderer, *events: Event) -> None:
    for event in events:
        renderer.handle(event, 0.0)


def _plain(stream: io.StringIO) -> str:
    return strip_ansi(stream.getvalue()).replace("\r", "")


def _dl(label: str) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.DOWNLOADING, fraction=0.5)


def _progress(*torrents: TorrentView, elapsed: float) -> WaitProgress:
    return WaitProgress(snapshot=WaitSnapshot(torrents=torrents, elapsed_s=elapsed))


class TestWaitCockpitLifecycle:
    def test_live_starts_on_first_progress_not_on_started(self) -> None:
        renderer, _stream = _live_renderer()

        _feed(renderer, WaitStarted(total=1, pulse_s=300.0))
        assert renderer._wait._live is None  # WaitStarted alone never opens the cockpit

        _feed(renderer, _progress(_dl("Show"), elapsed=0))
        live = renderer._wait._live
        assert live is not None and live.is_started

        renderer.close()
        assert renderer._wait._live is None and not live.is_started

    def test_graduation_prints_above_the_active_cockpit(self) -> None:
        renderer, stream = _live_renderer()

        _feed(
            renderer,
            WaitStarted(total=2, pulse_s=300.0),
            _progress(_dl("Show A"), elapsed=0),
            TorrentGraduated(label="Show A", outcome=Outcome.IMPORTED, files=8, waited_s=300),
        )
        live = renderer._wait._live
        assert live is not None and live.is_started  # the cockpit is still active

        _feed(renderer, WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=300))
        out = _plain(stream)
        assert "✔ imported    Show A  (8 files · 5m 00s)" in out  # graduated to durable scrollback
        assert "wait complete · 1 imported · 5m 00s" in out

    def test_wait_finished_stops_the_live_then_prints_the_tally(self) -> None:
        # Old teardown-then-summary order: the tally lands to clean scrollback, so
        # the observable guarantee is a stopped Live WITH the tally present.
        renderer, stream = _live_renderer()

        _feed(renderer, WaitStarted(total=1, pulse_s=300.0), _progress(_dl("Show"), elapsed=0))
        live = renderer._wait._live
        assert live is not None and live.is_started

        _feed(renderer, WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=150))
        assert renderer._wait._live is None and not live.is_started
        assert "wait complete · 1 imported · 2m 30s" in _plain(stream)

    def test_zero_graduation_finish_prints_no_tally(self) -> None:
        renderer, stream = _live_renderer()

        _feed(
            renderer,
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
            WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=120),
        )
        assert renderer._wait._live is None
        assert "wait complete" not in _plain(stream)

    def test_begin_cycle_stops_the_live(self) -> None:
        renderer, _stream = _live_renderer()

        _feed(renderer, WaitStarted(total=1, pulse_s=300.0), _progress(_dl("Show"), elapsed=0))
        assert renderer._wait._live is not None

        renderer.begin_cycle()
        assert renderer._wait._live is None


class TestNonLiveDigest:
    def test_prints_start_and_throttled_pulses_without_a_live(self) -> None:
        renderer, stream = _live_renderer(width=20)  # width < MIN_LIVE_WIDTH -> not live

        _feed(
            renderer,
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),  # the start snapshot never pulses
            _progress(_dl("Show"), elapsed=300),  # the first pulse
        )
        assert renderer._wait._live is None  # a non-live console never opens a Live

        out = _plain(stream)
        assert "Waiting on 1 download to complete and import..." in out
        assert "still waiting" in out
        assert "5m 00s" in out

    def test_raised_level_suppresses_every_durable_line(self) -> None:
        renderer, stream = _live_renderer(width=20)
        renderer.set_level(logging.WARNING)

        _feed(
            renderer,
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
            _progress(_dl("Show"), elapsed=300),
            TorrentGraduated(label="Show", outcome=Outcome.IMPORTED, files=1, waited_s=60),
            WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=60),
        )
        assert _plain(stream).strip() == ""
        assert renderer._wait._live is None


class TestFrontierTeardown:
    def test_scope_closed_tears_down_the_wait_live(self) -> None:
        renderer, _stream = _live_renderer()

        _feed(
            renderer,
            ScopeOpened(scope=_WAIT, label="wait"),
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
        )
        live = renderer._wait._live
        assert live is not None and live.is_started

        _feed(renderer, ScopeClosed(scope=_WAIT))
        assert renderer._wait._live is None and not live.is_started

    def test_run_finished_unwind_tears_down_the_wait_live(self) -> None:
        renderer, _stream = _live_renderer()

        _feed(
            renderer,
            ScopeOpened(scope=_WAIT, label="wait"),
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
        )
        live = renderer._wait._live
        assert live is not None and live.is_started

        _feed(renderer, RunFinished(arr=Arr.SONARR))
        assert renderer._wait._live is None and not live.is_started

    def test_a_second_pass_starts_a_fresh_live_slot(self) -> None:
        # Negative control against stale reuse: the evicted slot is never repainted;
        # a second pass mints a fresh Live (a leaked slot would be reused, nested).
        renderer, _stream = _live_renderer()

        _feed(
            renderer,
            ScopeOpened(scope=_WAIT, label="wait"),
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
        )
        first = renderer._wait._live
        assert first is not None and first.is_started

        _feed(renderer, ScopeClosed(scope=_WAIT))
        assert renderer._wait._live is None and not first.is_started

        _feed(
            renderer,
            ScopeOpened(scope=_WAIT_TWO, label="wait"),
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
        )
        second = renderer._wait._live
        assert second is not None and second.is_started
        assert second is not first  # a fresh slot, never the stale one

        renderer.close()
        assert renderer._wait._live is None and not second.is_started

    def test_raised_level_keeps_the_cockpit_but_drops_the_durable_lines(self) -> None:
        renderer, stream = _live_renderer()
        renderer.set_level(logging.WARNING)

        _feed(renderer, WaitStarted(total=1, pulse_s=300.0), _progress(_dl("Show"), elapsed=0))
        assert renderer._wait._live is not None  # the cockpit is NOT level-gated

        _feed(
            renderer,
            TorrentGraduated(label="Show", outcome=Outcome.IMPORTED, files=1, waited_s=60),
            WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=60),
        )
        assert renderer._wait._live is None
        out = _plain(stream)
        assert "imported" not in out  # the durable graduation + tally were suppressed
        assert "wait complete" not in out

    def test_without_a_rich_console_every_wait_event_no_ops(self) -> None:
        renderer = RichRenderer(lambda: None, time_source=lambda: 0.0)

        _feed(
            renderer,
            WaitStarted(total=1, pulse_s=300.0),
            _progress(_dl("Show"), elapsed=0),
            TorrentGraduated(label="Show", outcome=Outcome.IMPORTED, files=1, waited_s=1),
            WaitFinished(imported=1, deferred=0, failed=0, elapsed_s=1),
        )
        assert renderer._wait._live is None


class TestWaitRegionDirect:
    @staticmethod
    def _region() -> WaitRegion:
        console = Console(file=io.StringIO(), force_terminal=True, width=100)
        return WaitRegion(lambda: console, level_source=lambda: logging.INFO, time_source=lambda: 0.0)

    def test_section_left_no_ops_when_no_live_ever_started(self) -> None:
        region = self._region()
        region.section_left()  # the non-TTY digest path never started a Live
        assert region._live is None
        region.section_left()  # idempotent
        assert region._live is None

    def test_section_left_is_idempotent_after_a_live(self) -> None:
        region = self._region()
        region.handle(WaitStarted(total=1, pulse_s=300.0))
        region.handle(_progress(_dl("Show"), elapsed=0))
        live = region._live
        assert live is not None and live.is_started

        region.section_left()
        assert region._live is None and not live.is_started
        region.section_left()  # idempotent
        assert region._live is None


def _boom() -> Group:
    raise RuntimeError("frame build failed")


def test_live_frame_swallows_a_raising_group_build(app_logger: logging.Logger) -> None:
    capture = CaptureHandler()
    app_logger.addHandler(capture)
    console = Console(file=io.StringIO())
    frame = _LiveFrame(_boom, app_logger)

    # A render bug degrades to an empty frame logged at debug, never a raise off
    # the refresh thread (the ABBA-safe swallow stays below WARNING).
    assert list(frame.__rich_console__(console, console.options)) == []
    assert [record.levelno for record in capture.records] == [logging.DEBUG]
