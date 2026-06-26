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

from seadexarr.modules.log import RichConsoleHandler
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.wait_view import (
    Capabilities,
    LiveWaitView,
    LogWaitView,
    Phase,
    TorrentView,
    WaitOutcomeRow,
    WaitResult,
    WaitSnapshot,
    _DurableWaitView,
    graduations,
    live_model,
    make_wait_view,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _logger_with_console(
    *, force_terminal: bool, width: int = 100,
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
