# pyright: strict
"""Direct tests for the arr run state: a fresh `ArrRunStats` starts every tally empty."""

from pearlarr.arr_run_state import ArrRunStats


def test_arr_run_stats_shape() -> None:
    s = ArrRunStats()
    assert s.checked == 0
    assert s.added == []
    assert s.needs_action == []
    assert s.unmonitored == 0
