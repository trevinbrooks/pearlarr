# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the drift pin reads _PHASE_RANK and the ETA tests drive _rough_eta directly.
"""Tests for the wait surface's pure line builders + throttle + reducers.

The byte guarantees on the wait ledger live in the builders (consumed by the
WaitRegion's durable prints) and the `output.textline`
grammar. The region-side rendering is pinned in test_output_wait_region. Here:
the `PulseThrottle` unit contract, the builder edge cases (including the
category-based graduation level), and the pure cockpit reducers
(`live_model` / `sparkline` / `graduation_tail`).
"""

import logging

import pytest

from pearlarr.console_caps import ConsoleFeatures, detect_capabilities
from pearlarr.output import (
    Phase,
    RowFinished,
    TorrentView,
    WaitFinished,
    WaitKind,
    WaitSnapshot,
    WaitStarted,
)
from pearlarr.output.console_text import StyledLine
from pearlarr.output.wait_lines import (
    _PHASE_RANK,
    PulseThrottle,
    _rough_eta,
    hand_import_sentences,
    live_model,
    row_finished_lines,
    row_finished_note,
    sparkline,
    wait_start_line,
    wait_tally_lines,
)
from pearlarr.releases import EpisodeKey, OrderOnlyKey
from pearlarr.wait_types import NO_HAND_IMPORT_FILES, HandImportFile, LeftForHandImport, Outcome

from .builders import SEP, held_files, order_only_held_back


def _term(label: str, outcome: Outcome) -> TorrentView:
    return TorrentView(key=label, label=label, phase=Phase.FINISHED, outcome=outcome)


def _grad(
    label: str,
    outcome: Outcome,
    *,
    files: int | None = None,
    waited: float = 0.0,
    left_for_hand_import: LeftForHandImport = NO_HAND_IMPORT_FILES,
) -> RowFinished:
    return RowFinished(
        label=label, outcome=outcome, files=files, waited_s=waited, left_for_hand_import=left_for_hand_import
    )


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
        assert throttle.fire(300.0) is True  # due, re-arms at 600
        assert throttle.fire(599.0) is False
        assert throttle.fire(650.0) is True  # elapsed-anchored: re-arms at 950, not a 900 grid mark
        assert throttle.fire(940.0) is False  # 940 < 950
        assert throttle.fire(1310.0) is True

    def test_poll_floor_interval(self) -> None:
        throttle = PulseThrottle()
        throttle.arm(600.0)
        assert throttle.fire(0.0) is False  # skip-first
        assert throttle.fire(300.0) is False  # a shorter interval would fire, 600 rules
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
        # mark) still never pulses - the old view's first render printed the start
        # line and returned no matter when it ran.
        throttle = PulseThrottle()
        throttle.arm(5.0)
        assert throttle.fire(7.0) is False  # skipped unconditionally
        assert throttle.fire(7.0) is True  # the next one is already due


# --- builder edge cases the goldens don't isolate ---------------------------------------


class TestWaitBuilders:
    """Builder edge cases the goldens miss: zero tallies, per-kind wording, graduation style/level by outcome."""

    def test_zero_graduation_tally_is_empty(self) -> None:
        finished = WaitFinished(imported=0, left=0, failed=0, elapsed_s=42.0, in_progress=0, kind=WaitKind.MONITOR)
        assert wait_tally_lines(finished) == []

    @pytest.mark.parametrize(
        ("kind", "one", "two"),
        [
            (
                WaitKind.MONITOR,
                "Waiting on 1 download to complete and import...",
                "Waiting on 2 downloads to complete and import...",
            ),
            (WaitKind.CHECK, "Checking 1 carried-over download...", "Checking 2 carried-over downloads..."),
        ],
    )
    def test_start_line_words_and_pluralizes_by_kind(self, kind: WaitKind, one: str, two: str) -> None:
        line = wait_start_line(WaitStarted(total=1, pulse_s=300.0, kind=kind))
        assert line.message == one
        assert line.payload is None
        assert wait_start_line(WaitStarted(total=2, pulse_s=300.0, kind=kind)).message == two

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (WaitKind.MONITOR, "  wait complete · 1 imported · 1m 00s"),
            (WaitKind.CHECK, "  check complete · 1 imported · 2 pending · 1m 00s"),
        ],
    )
    def test_tally_head_and_pending_bucket_by_kind(self, kind: WaitKind, expected: str) -> None:
        pending = 2 if kind is WaitKind.CHECK else 0
        finished = WaitFinished(imported=1, left=0, failed=0, elapsed_s=60.0, in_progress=pending, kind=kind)
        assert wait_tally_lines(finished)[1].message == expected

    def test_graduation_style_is_dropped_without_color(self) -> None:
        # The colorless case is style="", not None (an unstyled console line).
        [line] = row_finished_lines(_grad("Show A", Outcome.IMPORTED, files=1, waited=60), detect_capabilities(None))
        assert line.payload == StyledLine(style="")
        assert line.message == "  ok imported    Show A  (1 file · 1m 00s)"

    def test_graduation_line_level_follows_the_outcome_category(self) -> None:
        # P6: the rendered line and severity_of/the sink admission agree -
        # FAILED carries ERROR, DEFERRED WARNING, success INFO.
        features = detect_capabilities(None)
        assert row_finished_lines(_grad("A", Outcome.IMPORTED), features)[0].level == logging.INFO
        assert row_finished_lines(_grad("B", Outcome.DOWNLOAD_TIMED_OUT), features)[0].level == logging.WARNING
        assert row_finished_lines(_grad("C", Outcome.DOWNLOAD_ERRORED), features)[0].level == logging.ERROR


# --- the pure cockpit reducers ----------------------------------------------------------

_WIDE = ConsoleFeatures(can_redraw=True, can_color=True, can_unicode=True, width=100, height=40)


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
    features = ConsoleFeatures(can_redraw=True, can_color=True, can_unicode=True, width=100, height=12)
    torrents = [_downloading_row(f"d{i}", f"D{i}", 0.1 * i) for i in range(20)]
    torrents.append(TorrentView("imp", "Importer", Phase.IMPORTING))
    snap = WaitSnapshot(tuple(torrents), elapsed_s=120)

    model = live_model(snap, features, WaitKind.MONITOR)

    # height budget caps visible rows. The rest collapse to an overflow tally.
    assert len(model.rows) == 4
    assert model.rows[0].phase is Phase.IMPORTING  # importing sorts first
    assert "more downloading" in model.overflow
    assert "17" in model.overflow  # 20 downloads + 1 importing, 4 shown -> 17 hidden


@pytest.mark.parametrize(("kind", "left"), [(WaitKind.MONITOR, "waiting 1/2"), (WaitKind.CHECK, "checking 1/2")])
def test_live_model_header_reports_aggregate_under_the_kinds_verb(kind: WaitKind, left: str) -> None:
    snap = WaitSnapshot(
        (
            _term("A", Outcome.IMPORTED),
            _downloading_row("h2", "B", 0.5),
        ),
        elapsed_s=125,
    )

    model = live_model(snap, _WIDE, kind)

    assert model.left_text == left
    assert "MB/s" in model.right_text  # aggregate download speed
    assert 0.0 < model.overall_fraction < 1.0


def test_live_model_importing_determinate_bar() -> None:
    # A known files-inserted count -> a determinate bar with a "done/total" count
    # and the elapsed clock in the shared time column.
    snap = WaitSnapshot((_importing_row("h", "Show", done=8, total=12, elapsed=64),), elapsed_s=64)

    row = live_model(snap, _WIDE, WaitKind.MONITOR).rows[0]

    assert row.show_bar is True
    assert row.count == "8/12"
    assert 0.0 < row.fraction < 1.0
    assert row.time == "1m 04s"


def test_live_model_importing_is_indeterminate_without_a_total() -> None:
    # No seed-complete count -> no bar. The status word carries the phase
    # ("copying" once the import command's async copy is in flight), and the
    # elapsed clock sits in the same time column as every other row.
    snap = WaitSnapshot(
        (TorrentView("h", "Show", Phase.IMPORTING, command_issued=True, phase_elapsed_s=10),),
        elapsed_s=10,
    )

    row = live_model(snap, _WIDE, WaitKind.MONITOR).rows[0]

    assert row.show_bar is False
    assert row.count == ""
    assert row.status == "copying"
    assert row.time == "10s"


def test_live_model_importing_before_command_reads_importing() -> None:
    snap = WaitSnapshot(
        (TorrentView("h", "Show", Phase.IMPORTING, phase_elapsed_s=4),),
        elapsed_s=4,
    )

    row = live_model(snap, _WIDE, WaitKind.MONITOR).rows[0]

    assert row.status == "importing"


def test_live_model_download_row_layout() -> None:
    # One meaning per column: count is the %, time is the ETA, size is the TOTAL
    # only (the done side is already the bar + %, so "done/total" was redundant).
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5),), elapsed_s=10)

    row = live_model(snap, _WIDE, WaitKind.MONITOR).rows[0]

    assert row.show_bar is True
    assert row.count == "50%"
    assert row.time == "~2m"  # 130s ETA
    assert row.size == "2.7 GB"
    assert "/" not in row.size

    queued = live_model(WaitSnapshot((TorrentView("q", "Other"),)), _WIDE, WaitKind.MONITOR).rows[0]
    assert queued.status == "queued"


def test_sparkline_scales_to_the_window_peak() -> None:
    assert sparkline((0, 100)) == "▁█"
    assert sparkline((100, 100, 100)) == "███"
    # A wedged download decays to the floor - visible, never blank.
    assert sparkline((0, 0, 0)) == "▁▁▁"


def test_download_row_speed_carries_the_sparkline() -> None:
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(0, 3_200_000)),))

    row = live_model(snap, _WIDE, WaitKind.MONITOR).rows[0]

    assert row.speed == "▁█ 3.1 MB/s"


def test_sparkline_is_dropped_when_narrow_or_ascii() -> None:
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(0, 3_200_000)),))

    narrow = ConsoleFeatures(can_redraw=True, can_color=True, can_unicode=True, width=72, height=40)
    ascii_caps = ConsoleFeatures(can_redraw=True, can_color=True, can_unicode=False, width=100, height=40)

    assert live_model(snap, narrow, WaitKind.MONITOR).rows[0].speed == "3.1 MB/s"
    assert live_model(snap, ascii_caps, WaitKind.MONITOR).rows[0].speed == "3.1 MB/s"


def test_sparkline_needs_two_samples() -> None:
    # A single sample says nothing about the trend. The cell stays plain.
    snap = WaitSnapshot((_downloading_row("h", "Show", 0.5, history=(3_200_000,)),))

    assert live_model(snap, _WIDE, WaitKind.MONITOR).rows[0].speed == "3.1 MB/s"


def test_compact_eta_covers_every_magnitude() -> None:
    # The docstring's own examples, pinned: zero-padded minutes past an hour, a
    # bare-seconds tail, and negatives floored to ~0s (never "~-5s").
    assert _rough_eta(4000) == "~1h06m"
    assert _rough_eta(3900) == "~1h05m"
    assert _rough_eta(130) == "~2m"
    assert _rough_eta(40) == "~40s"
    assert _rough_eta(-5) == "~0s"


def test_phase_rank_covers_every_non_terminal_phase() -> None:
    # Drift pin: a new Phase member must be placed in the cockpit ordering - and
    # in the overflow/pulse tallies that hand-list the same non-terminal trio.
    assert set(_PHASE_RANK) | {Phase.FINISHED} == set(Phase)


# --- graduation ledger coda ------------------------------------------------------


def test_graduation_tail_states_files_and_elapsed_for_an_import() -> None:
    assert row_finished_note(_grad("A", Outcome.IMPORTED, files=12, waited=192)) == f"12 files{SEP}3m 12s"


def test_graduation_tail_is_empty_when_an_import_has_no_detail() -> None:
    # No files count (incomplete seed) and a sub-second wait -> nothing to say.
    assert row_finished_note(_grad("A", Outcome.IMPORTED)) == ""


def test_graduation_tail_elapsed_alone_when_files_unknown() -> None:
    # An incomplete seed map hides the files count but the wait clock still shows.
    assert row_finished_note(_grad("A", Outcome.IMPORTED, waited=192)) == "3m 12s"


def test_graduation_tail_says_left_pending_outcomes_retry() -> None:
    for outcome in (
        Outcome.DOWNLOAD_TIMED_OUT,
        Outcome.DOWNLOAD_ERRORED,
        Outcome.STILL_IMPORTING,
        Outcome.SONARR_BUSY,
        Outcome.NOT_READY,
        Outcome.ATTEMPT_FAILED,
    ):
        assert row_finished_note(_grad("A", outcome)) == "retries next run"


def test_graduation_tail_counts_unmatched_files_and_names_the_action() -> None:
    # Never "retries".
    event = _grad("A", Outcome.UNMATCHED, waited=15, left_for_hand_import=held_files("a.mkv", "b.mkv"))
    assert row_finished_note(event) == "2 files matched no episode · import by hand in Sonarr"
    assert row_finished_note(_grad("A", Outcome.UNMATCHED, left_for_hand_import=held_files("a.mkv"))) == (
        "1 file matched no episode · import by hand in Sonarr"
    )


def test_graduation_tail_counts_held_back_files_and_names_the_action() -> None:
    event = _grad("A", Outcome.PARTIAL, waited=15, left_for_hand_import=held_files("a.mkv", "b.mkv"))
    assert row_finished_note(event) == "2 files held back · import by hand in Sonarr"


_SEVERAL = order_only_held_back(
    missing=(EpisodeKey(4, 1), EpisodeKey(4, 2)), untrusted=(EpisodeKey(4, 5), EpisodeKey(4, 6))
)


@pytest.mark.parametrize(
    ("left", "coda", "lead"),
    [
        pytest.param(
            order_only_held_back(),
            "S04 E01 still missing in Sonarr",
            "S04 E01 is still missing in Sonarr.",
            id="one missing",
        ),
        pytest.param(
            order_only_held_back(missing=(), untrusted=(EpisodeKey(4, 1),)),
            "S04 E01 has a file that isn't from a recommended release",
            "S04 E01 has a file in Sonarr, but it isn't from a recommended release.",
            id="one holding another file",
        ),
        pytest.param(
            _SEVERAL,
            f"S04 E01-E02 still missing in Sonarr{SEP}S04 E05-E06 have files that aren't from a recommended release",
            "S04 E01-E02 are still missing in Sonarr. "
            "S04 E05-E06 have files in Sonarr, but they aren't from a recommended release.",
            id="several of each",
        ),
    ],
)
def test_an_order_only_record_leads_with_its_unfilled_episodes(left: LeftForHandImport, coda: str, lead: str) -> None:
    assert row_finished_note(_grad("A", Outcome.PARTIAL, left_for_hand_import=left)) == coda
    assert hand_import_sentences(left)[0] == lead


def test_a_lone_order_only_file_reads_in_the_singular() -> None:
    name = "[grp] Show Ending S1 - 01 [ABCD1234].mkv"
    left = LeftForHandImport(
        files=(HandImportFile(name, OrderOnlyKey("dvd", EpisodeKey(7, 1))),), missing=(EpisodeKey(4, 1),)
    )
    assert hand_import_sentences(left)[1] == (
        "This pack is numbered by the DVD order, and 1 of its files is one of the DVD episodes "
        "that aren't in Sonarr's list, so Pearlarr can't tell where it belongs"
    )


def test_a_long_reason_wraps_onto_rows_that_fit_a_100_column_console() -> None:
    left = LeftForHandImport(files=order_only_held_back().files[:1], missing=(EpisodeKey(4, 1),))
    lines = row_finished_lines(_grad("A", Outcome.PARTIAL, left_for_hand_import=left), detect_capabilities(None))
    assert [line.message for line in lines[1:3]] == [
        "    reason      this pack is numbered by the DVD order, and 1 of its files is one of the DVD",
        "                episodes that aren't in Sonarr's list, so Pearlarr can't tell where it belongs",
    ]


def test_files_keyed_by_two_orders_name_both_orders() -> None:
    files = (
        HandImportFile("a.mkv", OrderOnlyKey("dvd", EpisodeKey(7, 1))),
        HandImportFile("b.mkv", OrderOnlyKey("altdvd", EpisodeKey(3, 1))),
    )
    reason = hand_import_sentences(LeftForHandImport(files=files, missing=(EpisodeKey(4, 1),)))[1]
    assert reason.startswith("This pack is numbered by the DVD and alternate DVD orders, and 2 of its files are")


def test_an_order_only_row_says_why_then_lists_each_file_behind_its_key() -> None:
    event = _grad("Show Ending · grp · S04 E01-E12", Outcome.PARTIAL, left_for_hand_import=order_only_held_back())
    lines = row_finished_lines(event, detect_capabilities(None))
    assert [line.message for line in lines] == [
        "  ~ partial     Show Ending · grp · S04 E01-E12  (S04 E01 still missing in Sonarr)",
        "    reason      this pack is numbered by the DVD order, and 2 of its files are DVD episodes",
        "                that aren't in Sonarr's list, so Pearlarr can't tell where they belong",
        "    files       DVD S07E01  [grp] Show Ending S1 - 01 [ABCD1234].mkv",
        "                DVD S07E02  [grp] Show Ending S1 - 02 [ABCD1234].mkv",
    ]
    assert {line.level for line in lines} == {logging.WARNING}


def test_a_held_back_row_names_five_files_and_counts_the_rest() -> None:
    names = tuple(f"[grp] Show - Part {n:02d}.mkv" for n in range(1, 8))
    lines = row_finished_lines(
        _grad("A", Outcome.UNMATCHED, left_for_hand_import=held_files(*names)), detect_capabilities(None)
    )
    assert [line.message for line in lines[1:]] == [
        "    files       [grp] Show - Part 01.mkv",
        *(f"                [grp] Show - Part {n:02d}.mkv" for n in range(2, 6)),
        "                ... +2 more",
    ]


def test_a_mixed_record_keeps_its_count_and_keys_the_order_only_file() -> None:
    left = LeftForHandImport(
        files=(order_only_held_back().files[0], HandImportFile("[grp] Show Ending S1 - 05 [ABCD1234].mkv"))
    )
    lines = row_finished_lines(_grad("A", Outcome.PARTIAL, left_for_hand_import=left), detect_capabilities(None))
    assert [line.message for line in lines] == [
        f"  ~ partial     A  (2 files held back{SEP}import by hand in Sonarr)",
        "    files       DVD S07E01  [grp] Show Ending S1 - 01 [ABCD1234].mkv",
        "                [grp] Show Ending S1 - 05 [ABCD1234].mkv",
    ]


def test_graduation_tail_says_a_missing_record_is_gone() -> None:
    assert row_finished_note(_grad("A", Outcome.MISSING)) == "no longer tracked · title checked again next run"
