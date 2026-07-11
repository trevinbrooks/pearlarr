# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the lifecycle tests read RichRenderer._wait._live and drive WaitRegion/_LiveFrame.
"""Tests for the wait cockpit's renderer side (`output.wait_region`) + routing.

The RichRenderer's wait region opens its single `rich.Live` on the FIRST
WaitProgress (never on WaitStarted), graduates finished torrents to durable
scrollback while the cockpit is active, stops the Live before the closing tally
prints, degrades to a start line + throttled pulses on a non-live console, and
tears its slot down whenever the wait region leaves the fold's frontier
(ScopeClosed, a RunFinished unwind). A raised level suppresses the durable lines
but never the cockpit; a raising frame build degrades to an empty frame whose
failure LATCHES for a main-thread report (the refresh thread never logs).
"""

import io
import logging

from rich.console import Console, Group
from rich.spinner import Spinner

from pearlarr.modules.config import Arr
from pearlarr.modules.manual_import import Outcome
from pearlarr.modules.output import (
    Diagnostic,
    Event,
    Phase,
    RichRenderer,
    RunFinished,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    install_hub,
)
from pearlarr.modules.output.recording import RecordingHub
from pearlarr.modules.output.wait_lines import live_model
from pearlarr.modules.output.wait_region import WaitRegion, _LiveFrame

from .fakes import CaptureHandler, install_recording_hub, strip_ansi


def _render_group(group: Group) -> str:
    """A group's plain rendering — what one refresh tick would draw."""

    # legacy_windows pinned here and file-wide: Windows CI auto-detects a
    # legacy console, dropping the caps to ASCII and breaking glyph asserts.
    stream = io.StringIO()
    Console(file=stream, force_terminal=True, legacy_windows=False, width=100).print(group)
    return strip_ansi(stream.getvalue())


_WAIT = ScopeId(ScopeKind.WAIT_REGION, 700)
_WAIT_TWO = ScopeId(ScopeKind.WAIT_REGION, 701)


def _live_renderer(width: int = 100) -> tuple[RichRenderer, io.StringIO]:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, legacy_windows=False, width=width)
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
        assert renderer._wait._live_frame is not None

        renderer.begin_cycle()
        assert renderer._wait._live is None
        # The frame dies with its Live slot: no dangling dead frame to keep polling.
        assert renderer._wait._live_frame is None


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

    def test_failed_graduation_survives_a_raised_level(self) -> None:
        # P6: the graduation line carries severity_of's category-based level,
        # so a FAILED outcome renders at ERROR through render_legacy_lines'
        # WARNING gate while the INFO wait lines around it vanish.
        renderer, stream = _live_renderer(width=40)
        renderer.set_level(logging.WARNING)

        _feed(
            renderer,
            WaitStarted(total=1, pulse_s=300.0),
            TorrentGraduated(label="Show", outcome=Outcome.DOWNLOAD_ERRORED, files=None, waited_s=60),
        )
        out = _plain(stream)
        assert "errored" in out and "Show" in out
        assert "Waiting on" not in out  # the INFO start line stayed suppressed


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
        console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False, width=100)
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

    def test_frame_rolls_the_clocks_forward_between_polls(self) -> None:
        # The self-animating tick: a refresh AFTER the push rebuilds the frame
        # with the clock's advance folded into the overall elapsed and every
        # in-flight row's phase clock (TERMINAL rows would stay frozen).
        clock = {"now": 100.0}
        console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False, width=100)
        region = WaitRegion(lambda: console, level_source=lambda: logging.INFO, time_source=lambda: clock["now"])
        importing = TorrentView(key="h", label="Copy", phase=Phase.IMPORTING, phase_elapsed_s=4.0)
        region.handle(WaitStarted(total=1, pulse_s=300.0))
        region.handle(WaitProgress(snapshot=WaitSnapshot((importing,), elapsed_s=10.0)))

        at_push = _render_group(region._current_group())
        clock["now"] = 105.0  # five refresh-thread seconds later, no new poll
        ticked = _render_group(region._current_group())

        assert "10s" in at_push and "4s" in at_push
        assert "15s" in ticked and "9s" in ticked  # 10+5 header, 4+5 row clock

    def test_wide_frame_renders_the_bar_percent_and_importing_spinner(self) -> None:
        # The RowModel -> rich-widget mapping: a download shows its block bar +
        # clamped percent; an importing row rides the shared Spinner marker with
        # its "copying" word in the bar column.
        region = self._region()
        download = TorrentView(key="d", label="Down", phase=Phase.DOWNLOADING, fraction=0.5)
        copying = TorrentView(key="c", label="Copy", phase=Phase.IMPORTING, command_issued=True)
        region.handle(WaitStarted(total=2, pulse_s=300.0))
        region.handle(WaitProgress(snapshot=WaitSnapshot((download, copying), elapsed_s=5.0)))

        model = live_model(WaitSnapshot((download, copying), elapsed_s=5.0), region._caps)
        importing_cells = region._row_cells(next(r for r in model.rows if r.phase is Phase.IMPORTING))
        assert isinstance(importing_cells[0], Spinner)  # the shared animated marker

        frame = _render_group(region._current_group())
        assert "50%" in frame and "█" in frame  # the download's percent + bar
        assert "copying" in frame  # the importing row's status word

    def test_narrow_frame_degrades_the_status_into_the_count_column(self) -> None:
        # No bar column below the width threshold: a barless row still says what
        # it is doing via the count column.
        clock = {"now": 0.0}
        console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False, width=60)
        region = WaitRegion(lambda: console, level_source=lambda: logging.INFO, time_source=lambda: clock["now"])
        copying = TorrentView(key="c", label="Copy", phase=Phase.IMPORTING, command_issued=True)
        region.handle(WaitStarted(total=1, pulse_s=300.0))
        region.handle(WaitProgress(snapshot=WaitSnapshot((copying,), elapsed_s=5.0)))

        frame = _render_group(region._current_group())
        row_line = next(line for line in frame.splitlines() if "Copy" in line)
        assert "copying" in row_line  # degraded into the count column
        assert "█" not in row_line and "░" not in row_line  # no ROW bar at this width


def _boom() -> Group:
    raise RuntimeError("frame build failed")


def test_live_frame_latches_a_raising_group_build_without_logging(app_logger: logging.Logger) -> None:
    capture = CaptureHandler()
    app_logger.addHandler(capture)
    console = Console(file=io.StringIO(), legacy_windows=False)
    frame = _LiveFrame(_boom)

    # A render bug degrades to an empty frame with NO log record: the refresh
    # thread must never reach the bridge/hub at all (adoption makes any level
    # dangerous — hub.emit under the Console lock is the ABBA inversion). The
    # failure latches for the main thread instead.
    assert list(frame.__rich_console__(console, console.options)) == []
    assert capture.records == []

    # Once per Live session: rich retries every tick, but only the FIRST
    # failure latches, and take_failure is one-shot.
    assert list(frame.__rich_console__(console, console.options)) == []
    failure = frame.take_failure()
    assert isinstance(failure, RuntimeError)
    assert frame.take_failure() is None


def test_a_raising_tick_emits_nothing_to_the_hub(app_logger: logging.Logger) -> None:
    # The latch is the real rule made structural: the refresh thread must NEVER
    # reach the hub (hub.emit off a Console-lock-holding thread is the ABBA
    # topology the pin exists to prevent).
    del app_logger
    recording = RecordingHub()
    install_hub(recording.hub)
    console = Console(file=io.StringIO(), legacy_windows=False)
    frame = _LiveFrame(_boom)

    assert list(frame.__rich_console__(console, console.options)) == []

    assert recording.events == []


def _latched_region(console: Console) -> tuple[WaitRegion, _LiveFrame]:
    """A region with an active Live whose frame builder raises on every tick."""

    region = WaitRegion(lambda: console, level_source=lambda: logging.INFO, time_source=lambda: 0.0)
    region.handle(WaitStarted(total=1, pulse_s=300.0))
    region.handle(_progress(_dl("Show"), elapsed=0))
    frame = _LiveFrame(_boom)
    region._live_frame = frame
    return region, frame


def test_wait_region_reports_a_latched_failure_once_on_the_next_handle() -> None:
    recording = install_recording_hub()
    console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False, width=100)
    region, frame = _latched_region(console)

    # One refresh tick fails silently on the refresh thread...
    assert list(frame.__rich_console__(console, console.options)) == []
    assert recording.of_type(Diagnostic) == []

    # ...and the NEXT main-thread handle() reports it exactly once: a file-only
    # WARNING Diagnostic carrying the latched exception's trace — durable at any
    # config level (the old logger.debug died at the logger gate above DEBUG).
    region.handle(_progress(_dl("Show"), elapsed=1))
    (report,) = [d for d in recording.of_type(Diagnostic) if d.message == "wait frame render failed"]
    assert report.severity is Severity.WARNING
    assert report.file_only
    assert report.origin == "output.live_region"
    assert report.trace is not None and "RuntimeError" in report.trace.plain

    region.handle(_progress(_dl("Show"), elapsed=2))  # one-shot: nothing left to report
    assert len([d for d in recording.of_type(Diagnostic) if d.message == "wait frame render failed"]) == 1
    region.close()


def test_wait_region_teardown_flushes_a_latched_failure() -> None:
    # A failure from the session's last ticks must not die with the frame: the
    # teardown routes (section_left/close/reset) also collect the latch.
    recording = install_recording_hub()
    console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False, width=100)
    region, frame = _latched_region(console)

    assert list(frame.__rich_console__(console, console.options)) == []
    region.section_left()

    (report,) = [d for d in recording.of_type(Diagnostic) if d.message == "wait frame render failed"]
    assert report.severity is Severity.WARNING and report.file_only
