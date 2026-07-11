# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the drift pin reads _PHASE_RANK and the ETA tests drive _compact_eta directly.
"""Tests for the wait surface's pure line builders + throttle + reducers.

The byte guarantees on the wait ledger live in the builders (consumed by the
WaitRegion's durable prints) and the `output.textline`
grammar; the region-side rendering is pinned in test_output_wait_region. Here:
the `PulseThrottle` unit contract, the builder edge cases (including the
category-based graduation level), and the pure cockpit reducers
(`live_model` / `sparkline` / `graduation_tail`).
"""

import logging

from pearlarr.modules.console_caps import Capabilities, detect_capabilities
from pearlarr.modules.log import StyledLine
from pearlarr.modules.manual_import import Outcome
from pearlarr.modules.output import (
    Phase,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitSnapshot,
    WaitStarted,
)
from pearlarr.modules.output.wait_lines import (
    _PHASE_RANK,
    PulseThrottle,
    _compact_eta,
    graduation_tail,
    live_model,
    sparkline,
    wait_graduation_line,
    wait_start_line,
    wait_tally_lines,
)

from .builders import SEP


def _term(label: str, outcome: Outcome) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.TERMINAL, outcome=outcome)


def _grad(label: str, outcome: Outcome, *, files: int | None = None, waited: float = 0.0) -> TorrentGraduated:
    return TorrentGraduated(label=label, outcome=outcome, files=files, waited_s=waited)


# --- the shared PulseThrottle -----------------------------------------------------------


class TestPulseThrottle:
    """PulseThrottle's contract: skip-first firing, elapsed-anchored cadence, and re-arm restarts skip-first."""

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

    def test_first_fire_is_skipped_even_past_the_interval(self) -> None:
        # skip-first as DISTINCT behavior: a late first snapshot (already past the
        # mark) still never pulses — the old view's first render printed the start
        # line and returned no matter when it ran.
        throttle = PulseThrottle()
        throttle.arm(5.0)
        assert throttle.fire(7.0) is False  # skipped unconditionally
        assert throttle.fire(7.0) is True  # the next one is already due


# --- builder edge cases the goldens don't isolate ---------------------------------------


class TestWaitBuilders:
    """Builder edge cases the goldens miss: zero tallies, pluralized start lines, graduation style/level by outcome."""

    def test_zero_graduation_tally_is_empty(self) -> None:
        assert wait_tally_lines(WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=42.0)) == []

    def test_start_line_pluralizes_and_carries_no_payload(self) -> None:
        line = wait_start_line(WaitStarted(total=1, pulse_s=300.0))
        assert line.message == "Waiting on 1 download to complete and import..."
        assert line.payload is None

    def test_graduation_style_is_dropped_without_color(self) -> None:
        # The colorless case is style="", not None (an unstyled console line).
        line = wait_graduation_line(_grad("Show A", Outcome.IMPORTED, files=1, waited=60), detect_capabilities(None))
        assert line.payload == StyledLine(style="")
        assert line.message == "  ok imported    Show A  (1 file · 1m 00s)"

    def test_graduation_line_level_follows_the_outcome_category(self) -> None:
        # P6: the rendered line and severity_of/the sink admission agree —
        # FAILED carries ERROR, DEFERRED WARNING, success INFO.
        caps = detect_capabilities(None)
        assert wait_graduation_line(_grad("A", Outcome.IMPORTED), caps).level == logging.INFO
        assert wait_graduation_line(_grad("B", Outcome.DOWNLOAD_TIMED_OUT), caps).level == logging.WARNING
        assert wait_graduation_line(_grad("C", Outcome.DOWNLOAD_ERRORED), caps).level == logging.ERROR


# --- the pure cockpit reducers ----------------------------------------------------------

_WIDE = Capabilities(live=True, color=True, unicode=True, width=100, height=40)


def _downloading_row(key: str, label: str, frac: float, history: tuple[int, ...] = ()) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.DOWNLOADING,
        fraction=frac,
        speed_bps=3_200_000,
        eta_s=130,
        bytes_done=int(frac * 2_900_000_000),
        bytes_total=2_900_000_000,
        speed_history=history,
    )


def _importing_row(key: str, label: str, *, done: int, total: int, elapsed: float) -> TorrentView:
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


def test_live_model_orders_and_bounds_the_box() -> None:
    caps = Capabilities(live=True, color=True, unicode=True, width=100, height=12)
    torrents = [_downloading_row(f"d{i}", f"D{i}", 0.1 * i) for i in range(20)]
    torrents.append(TorrentView("imp", "Importer", Phase.IMPORTING))
    snap = WaitSnapshot(tuple(torrents), elapsed_s=120)

    model = live_model(snap, caps)

    # height budget caps visible rows; the rest collapse to an overflow tally.
    assert len(model.rows) == 4
    assert model.rows[0].phase is Phase.IMPORTING  # importing sorts first
    assert "more downloading" in model.overflow
    assert "17" in model.overflow  # 20 downloads + 1 importing, 4 shown -> 17 hidden


def test_live_model_header_reports_aggregate() -> None:
    snap = WaitSnapshot(
        (
            _term("A", Outcome.IMPORTED),
            _downloading_row("h2", "B", 0.5),
        ),
        elapsed_s=125,
    )

    model = live_model(snap, _WIDE)

    assert model.left_text == "waiting 1/2"
    assert "MB/s" in model.right_text  # aggregate download speed
    assert 0.0 < model.overall_fraction < 1.0


def test_live_model_importing_determinate_bar() -> None:
    # A known files-inserted count -> a determinate bar with a "done/total" count
    # and the elapsed clock in the shared time column.
    snap = WaitSnapshot((_importing_row("h", "Show", done=8, total=12, elapsed=64),), elapsed_s=64)

    row = live_model(snap, _WIDE).rows[0]

    assert row.show_bar is True
    assert row.count == "8/12"
    assert 0.0 < row.fraction < 1.0
    assert row.time == "1m 04s"


def test_live_model_importing_is_indeterminate_without_a_total() -> None:
    # No seed-complete count -> no bar; the status word carries the phase
    # ("copying" once the import command's async copy is in flight), and the
    # elapsed clock sits in the same time column as every other row.
    snap = WaitSnapshot(
        (TorrentView("h", "Show", Phase.IMPORTING, command_issued=True, phase_elapsed_s=10),),
        elapsed_s=10,
    )

    row = live_model(snap, _WIDE).rows[0]

    assert row.show_bar is False
    assert row.count == ""
    assert row.status == "copying"
    assert row.time == "10s"


def test_live_model_importing_before_command_reads_importing() -> None:
    snap = WaitSnapshot(
        (TorrentView("h", "Show", Phase.IMPORTING, phase_elapsed_s=4),),
        elapsed_s=4,
    )

    row = live_model(snap, _WIDE).rows[0]

    assert row.status == "importing"


def test_live_model_download_row_layout() -> None:
    # One meaning per column: count is the %, time is the ETA, size is the TOTAL
    # only (the done side is already the bar + %, so "done/total" was redundant).
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5),), elapsed_s=10)

    row = live_model(snap, _WIDE).rows[0]

    assert row.show_bar is True
    assert row.count == "50%"
    assert row.time == "~2m"  # 130s ETA
    assert row.size == "2.7 GB"
    assert "/" not in row.size

    queued = live_model(WaitSnapshot((TorrentView("q", "Other"),)), _WIDE).rows[0]
    assert queued.status == "queued"


def test_sparkline_scales_to_the_window_peak() -> None:
    assert sparkline((0, 100)) == "▁█"
    assert sparkline((100, 100, 100)) == "███"
    # A wedged download decays to the floor - visible, never blank.
    assert sparkline((0, 0, 0)) == "▁▁▁"


def test_download_row_speed_carries_the_sparkline() -> None:
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(0, 3_200_000)),))

    row = live_model(snap, _WIDE).rows[0]

    assert row.speed == "▁█ 3.1 MB/s"


def test_sparkline_is_dropped_when_narrow_or_ascii() -> None:
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(0, 3_200_000)),))

    narrow = Capabilities(live=True, color=True, unicode=True, width=72, height=40)
    ascii_caps = Capabilities(live=True, color=True, unicode=False, width=100, height=40)

    assert live_model(snap, narrow).rows[0].speed == "3.1 MB/s"
    assert live_model(snap, ascii_caps).rows[0].speed == "3.1 MB/s"


def test_sparkline_needs_two_samples() -> None:
    # A single sample says nothing about the trend; the cell stays plain.
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(3_200_000,)),))

    assert live_model(snap, _WIDE).rows[0].speed == "3.1 MB/s"


def test_compact_eta_covers_every_magnitude() -> None:
    # The docstring's own examples, pinned: zero-padded minutes past an hour, a
    # bare-seconds tail, and negatives floored to ~0s (never "~-5s").
    assert _compact_eta(4000) == "~1h06m"
    assert _compact_eta(3900) == "~1h05m"
    assert _compact_eta(130) == "~2m"
    assert _compact_eta(40) == "~40s"
    assert _compact_eta(-5) == "~0s"


def test_phase_rank_covers_every_non_terminal_phase() -> None:
    # Drift pin: a new Phase member must be placed in the cockpit ordering — and
    # in the overflow/pulse tallies that hand-list the same non-terminal trio.
    assert set(_PHASE_RANK) | {Phase.TERMINAL} == set(Phase)


# --- graduation ledger coda ------------------------------------------------------


def test_graduation_tail_states_files_and_elapsed_for_an_import() -> None:
    assert graduation_tail(Outcome.IMPORTED, 12, 192) == f"12 files{SEP}3m 12s"


def test_graduation_tail_is_empty_when_an_import_has_no_detail() -> None:
    # No files count (incomplete seed) and a sub-second wait -> nothing to say.
    assert graduation_tail(Outcome.IMPORTED, None, 0.0) == ""


def test_graduation_tail_elapsed_alone_when_files_unknown() -> None:
    # An incomplete seed map hides the files count but the wait clock still shows.
    assert graduation_tail(Outcome.IMPORTED, None, 192) == "3m 12s"


def test_graduation_tail_says_left_pending_outcomes_retry() -> None:
    for outcome in (
        Outcome.DOWNLOAD_TIMED_OUT,
        Outcome.DOWNLOAD_ERRORED,
        Outcome.STILL_IMPORTING,
        Outcome.NOT_READY,
        Outcome.NOTHING_TO_IMPORT,
    ):
        assert graduation_tail(outcome, None, 0.0) == "retries next run"


def test_graduation_tail_says_a_missing_record_is_gone() -> None:
    assert graduation_tail(Outcome.MISSING, None, 0.0) == "no longer tracked"
