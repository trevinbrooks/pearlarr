# pyright: strict
# pyright: reportPrivateUsage=false
# ^ test imports/exercises wait_view internals (_DurableWaitView, _FrameAnchor, _anchor).
"""Tests for the wait-pass presentation (``wait_view``).

The view is a pure function of an immutable :class:`WaitSnapshot`: the engine
pushes one per poll cycle and the view renders it. These pin the capability probe
(live cockpit on a real TTY, calm log digest otherwise), the pure model helpers
(``graduations`` / ``live_model``), the durable graduation-through-the-logger
behaviour (so outcomes hit the file log, not just the console), and the no-throw
contract. The live view is exercised against a forced-terminal rich Console
writing to a buffer, so no real terminal is needed.
"""

import io
import logging
import re
from typing import override

from rich.console import Console
from rich.spinner import Spinner

from seadexarr.modules.console_caps import Capabilities
from seadexarr.modules.log import RichConsoleHandler
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.wait_view import (
    LiveWaitView,
    LogWaitView,
    Phase,
    TorrentView,
    WaitOutcomeRow,
    WaitResult,
    WaitSnapshot,
    _DurableWaitView,
    _FrameAnchor,
    graduations,
    live_model,
    make_wait_view,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _logger_with_console(
    *,
    force_terminal: bool,
    width: int = 100,
) -> tuple[logging.Logger, Console]:
    logger = logging.getLogger(f"wait-view-test-{force_terminal}-{width}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = Console(file=io.StringIO(), force_terminal=force_terminal, width=width)
    logger.addHandler(RichConsoleHandler(console))
    return logger, console


def _plain(console: Console) -> str:
    stream = console.file
    assert isinstance(stream, io.StringIO)
    return _ANSI.sub("", stream.getvalue())


def _downloading(key: str, label: str, frac: float) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.DOWNLOADING,
        fraction=frac,
        speed_bps=3_200_000,
        eta_s=130,
        bytes_done=int(frac * 2_900_000_000),
        bytes_total=2_900_000_000,
    )


def _terminal(key: str, label: str, outcome: Outcome) -> TorrentView:
    return TorrentView(key=key, label=label, phase=Phase.TERMINAL, outcome=outcome)


def _importing(key: str, label: str, *, done: int, total: int, elapsed: float) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.IMPORTING,
        fraction=(done / total if total else 1.0),
        import_done=done,
        import_total=total,
        phase_elapsed_s=elapsed,
        command_issued=True,
    )


def _render_to_text(renderable: object, *, width: int = 100) -> str:
    console = Console(file=io.StringIO(), force_terminal=True, width=width)
    console.print(renderable)
    return _plain(console)


# --- factory / capability probe ------------------------------------------------


def test_factory_returns_log_view_without_console() -> None:
    logger = logging.getLogger("wait-view-null")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    assert isinstance(make_wait_view(logger, poll_s=30), LogWaitView)


def test_factory_returns_log_view_on_non_tty() -> None:
    logger, _ = _logger_with_console(force_terminal=False)

    assert isinstance(make_wait_view(logger, poll_s=30), LogWaitView)


def test_factory_returns_live_on_a_tty() -> None:
    logger, _ = _logger_with_console(force_terminal=True)

    assert isinstance(make_wait_view(logger, poll_s=30), LiveWaitView)


def test_factory_falls_back_to_log_view_when_too_narrow() -> None:
    # A real TTY too narrow for a legible box folds to the safe log path.
    logger, _ = _logger_with_console(force_terminal=True, width=20)

    assert isinstance(make_wait_view(logger, poll_s=30), LogWaitView)


# --- live cockpit --------------------------------------------------------------


def test_live_view_graduates_and_summarizes() -> None:
    logger, console = _logger_with_console(force_terminal=True)
    view = make_wait_view(logger, poll_s=30)

    view.update(WaitSnapshot((_downloading("h1", "Bocchi the Rock!", 0.6),), elapsed_s=10))
    view.update(
        WaitSnapshot(
            (
                _terminal("h1", "Bocchi the Rock!", Outcome.IMPORTED),
                _terminal("h2", "Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
            ),
            elapsed_s=900,
        ),
    )
    view.close()

    out = _plain(console)
    assert "Bocchi the Rock!" in out
    assert "imported" in out  # graduation ledger word
    assert "timed out" in out
    assert "wait complete" in out  # closing summary
    assert "1 imported" in out and "1 left" in out


def test_live_graduations_reach_the_file_log_not_just_the_console() -> None:
    # The defining fix: graduations go through the LOGGER (both handlers), so a
    # plain non-console handler - the file log stand-in - sees them too. A
    # ``live.console.print`` would skip this buffer.
    logger, _ = _logger_with_console(force_terminal=True)
    file_buffer = io.StringIO()
    logger.addHandler(logging.StreamHandler(file_buffer))

    view = make_wait_view(logger, poll_s=30)
    view.update(WaitSnapshot((_downloading("h1", "Frieren", 0.4),), elapsed_s=5))
    view.update(WaitSnapshot((_terminal("h1", "Frieren", Outcome.IMPORTED),), elapsed_s=120))
    view.close()

    file_text = file_buffer.getvalue()
    assert "Frieren" in file_text
    assert "imported" in file_text
    assert "wait complete" in file_text


def test_live_frame_ticks_timer_between_polls() -> None:
    # The cockpit's elapsed timer must advance off rich's refresh between the
    # engine's polls: _current_group rolls the last anchor forward by now-pushed_at.
    # Driven deterministically through a fake clock, NOT the background thread.
    logger, _ = _logger_with_console(force_terminal=True)
    caps = Capabilities(live=True, color=False, unicode=True, width=100, height=40)
    now = [0.0]
    view = LiveWaitView(Console(file=io.StringIO()), caps, logger, time_source=lambda: now[0])
    snap = WaitSnapshot((_importing("h", "Show", done=2, total=12, elapsed=64),), elapsed_s=64)
    view._anchor = _FrameAnchor(snap, 0.0)  # pushed at t0=0

    at_push = _render_to_text(view._current_group())
    now[0] = 5.0  # 5s later, no new snapshot pushed
    later = _render_to_text(view._current_group())

    assert "1m 04s" in at_push  # 64s at push
    assert "1m 09s" in later  # 69s after the 5s tick


def test_live_frame_renders_the_import_bar_and_count() -> None:
    logger, _ = _logger_with_console(force_terminal=True)
    caps = Capabilities(live=True, color=False, unicode=True, width=100, height=40)
    view = LiveWaitView(Console(file=io.StringIO()), caps, logger)
    snap = WaitSnapshot((_importing("h", "Show", done=8, total=12, elapsed=64),), elapsed_s=64)
    view._anchor = _FrameAnchor(snap, 0.0)

    text = _render_to_text(view._current_group())

    assert "8/12" in text
    assert "█" in text  # a determinate block bar, not the "importing" word


def test_live_view_uses_a_spinner_for_importing_rows() -> None:
    # The importing marker is the shared animated spinner (the activity indicator),
    # not the static glyph.
    logger, _ = _logger_with_console(force_terminal=True)
    view = make_wait_view(logger, poll_s=30)
    assert isinstance(view, LiveWaitView)
    view.update(WaitSnapshot((_importing("h", "Show", done=1, total=3, elapsed=5),), elapsed_s=5))
    try:
        assert view._spinner is not None
        row = live_model(
            WaitSnapshot((_importing("h", "Show", done=1, total=3, elapsed=5),), elapsed_s=5), view._caps
        ).rows[0]
        cells = view._row_cells(row, 16, show_speed=True, show_size=True)
        assert isinstance(cells[0], Spinner)
    finally:
        view.close()


def test_spinner_frame_advances_in_a_table_cell() -> None:
    # Rich draws a Spinner from console.get_time(); prove the glyph actually cycles
    # over time when the spinner is a Table.grid cell (the cockpit's layout), not
    # just the boot view's Padding. The view's timer is frozen (time_source -> 0) so
    # the ONLY thing that can differ between the two frames is the spinner itself.
    now = [0.0]
    console = Console(file=io.StringIO(), force_terminal=True, width=100, get_time=lambda: now[0])
    caps = Capabilities(live=True, color=False, unicode=True, width=100, height=40)
    logger, _ = _logger_with_console(force_terminal=True)
    view = LiveWaitView(Console(file=io.StringIO()), caps, logger, time_source=lambda: 0.0)
    view._spinner = Spinner("dots", style="yellow")
    snap = WaitSnapshot((_importing("h", "Show", done=1, total=3, elapsed=5),), elapsed_s=5)
    view._anchor = _FrameAnchor(snap, 0.0)

    def frame_at(t: float) -> str:
        now[0] = t
        stream = console.file
        assert isinstance(stream, io.StringIO)
        _ = stream.seek(0)
        stream.truncate(0)
        console.print(view._current_group())
        return _ANSI.sub("", stream.getvalue())

    assert frame_at(0.0) != frame_at(0.5)  # the dots spinner cycled across ~6 frames


# --- log digest ----------------------------------------------------------------


def test_log_view_digest_is_calm_and_durable() -> None:
    logger, console = _logger_with_console(force_terminal=False)
    view = make_wait_view(logger, poll_s=30, digest_interval=300)
    snap = WaitSnapshot(
        (
            _downloading("h1", "Show A", 0.5),
            TorrentView("h2", "Show B", Phase.IMPORTING, command_issued=True),
        ),
        elapsed_s=0,
    )

    view.update(snap)  # start line
    view.update(WaitSnapshot(snap.torrents, elapsed_s=60))  # within interval: no pulse
    view.update(WaitSnapshot(snap.torrents, elapsed_s=600))  # past interval: one pulse
    view.update(
        WaitSnapshot((_terminal("h1", "Show A", Outcome.IMPORTED),), elapsed_s=700),
    )
    view.close()

    out = _plain(console)
    assert "Waiting on 2 download(s)" in out
    assert out.count("still waiting") == 1  # one aggregate pulse, not per-torrent spam
    assert "1 downloading" in out and "1 importing" in out
    assert "imported" in out and "Show A" in out
    assert "wait complete" in out


# --- pure model helpers --------------------------------------------------------


def test_graduations_returns_only_unseen_terminals() -> None:
    snap = WaitSnapshot(
        (
            _terminal("h1", "A", Outcome.IMPORTED),
            _downloading("h2", "B", 0.3),
            _terminal("h3", "C", Outcome.DOWNLOAD_ERRORED),
        ),
    )

    assert [t.key for t in graduations(frozenset(), snap)] == ["h1", "h3"]
    assert [t.key for t in graduations(frozenset({"h1"}), snap)] == ["h3"]
    assert graduations(frozenset({"h1", "h3"}), snap) == []


def test_live_model_orders_and_bounds_the_box() -> None:
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=12)
    torrents = [_downloading(f"d{i}", f"D{i}", 0.1 * i) for i in range(20)]
    torrents.append(TorrentView("imp", "Importer", Phase.IMPORTING))
    snap = WaitSnapshot(tuple(torrents), elapsed_s=120)

    model = live_model(snap, caps)

    # height budget caps visible rows; the rest collapse to an overflow tally.
    assert len(model.rows) == 4
    assert model.rows[0].phase is Phase.IMPORTING  # importing sorts first
    assert "more downloading" in model.overflow
    assert "17" in model.overflow  # 20 downloads + 1 importing, 4 shown -> 17 hidden


def test_live_model_header_reports_aggregate() -> None:
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
    snap = WaitSnapshot(
        (
            _terminal("h1", "A", Outcome.IMPORTED),
            _downloading("h2", "B", 0.5),
        ),
        elapsed_s=125,
    )

    model = live_model(snap, caps)

    assert model.left_text == "waiting 1/2"
    assert "MB/s" in model.right_text  # aggregate download speed
    assert 0.0 < model.overall_fraction < 1.0


def test_live_model_importing_determinate_bar() -> None:
    # A known files-inserted count -> a determinate bar with a "done/total" label.
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
    snap = WaitSnapshot((_importing("h", "Show", done=8, total=12, elapsed=64),), elapsed_s=64)

    row = live_model(snap, caps).rows[0]

    assert row.show_bar is True
    assert row.pct == "8/12"
    assert 0.0 < row.fraction < 1.0
    assert row.eta == "1m 04s"  # elapsed timer rides the eta slot


def test_live_model_importing_is_indeterminate_without_a_total() -> None:
    # No seed-complete count -> the old indeterminate row (no bar, "copy in flight").
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=40)
    snap = WaitSnapshot(
        (TorrentView("h", "Show", Phase.IMPORTING, command_issued=True, phase_elapsed_s=10),),
        elapsed_s=10,
    )

    row = live_model(snap, caps).rows[0]

    assert row.show_bar is False
    assert row.pct == ""
    assert "copy in flight" in row.note


# --- WaitResult ----------------------------------------------------------------


def test_wait_result_counts_by_category() -> None:
    result = WaitResult(
        (
            WaitOutcomeRow("h1", "A", Outcome.IMPORTED),
            WaitOutcomeRow("h2", "B", Outcome.IMPORTED),
            WaitOutcomeRow("h3", "C", Outcome.DOWNLOAD_TIMED_OUT),
            WaitOutcomeRow("h4", "D", Outcome.DOWNLOAD_ERRORED),
        ),
        elapsed_s=600,
    )

    assert result.waited == 4
    assert result.imported == 2
    assert result.left == 1
    assert result.failed == 1


def test_outcome_dropped_is_exactly_imported_and_missing() -> None:
    # Pins the word<->drop parity the engine relies on (outcome.dropped drives
    # _drop_pending), so a displayed word can never diverge from the store mutation.
    assert {o for o in Outcome if o.dropped} == {Outcome.IMPORTED, Outcome.MISSING}


def test_outcome_glyph_falls_back_to_ascii() -> None:
    assert Outcome.IMPORTED.glyph(use_unicode=True) == "✔"
    assert Outcome.IMPORTED.glyph(use_unicode=False) == "ok"
    assert Outcome.IMPORTED.category is OutcomeCategory.SUCCESS


# --- no-throw contract ---------------------------------------------------------


class _BoomView(_DurableWaitView):
    """A view whose render always raises - to prove update/close stay total."""

    @override
    def _render(self, snapshot: WaitSnapshot) -> None:
        raise RuntimeError("render boom")

    @override
    def _teardown(self) -> None:
        raise RuntimeError("teardown boom")


def test_view_methods_are_total_and_never_raise() -> None:
    logger = logging.getLogger("wait-view-boom")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    caps = Capabilities(live=False, color=False, unicode=False, width=80, height=24)
    view = _BoomView(logger, caps)

    # A render/teardown bug must degrade to a no-op, never propagate (which would
    # abort the engine's wait loop or the end-of-run cache save).
    view.update(WaitSnapshot((_downloading("h1", "A", 0.5),)))
    view.close()
