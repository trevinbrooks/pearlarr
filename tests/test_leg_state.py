# pyright: strict
"""Direct tests for the leg state: a fresh `RunStats` starts every tally empty."""

from pearlarr.leg_state import RunStats


def test_run_stats_shape() -> None:
    s = RunStats()
    assert s.checked == 0
    assert s.added == []
    assert s.needs_action == []
    assert s.unmonitored == 0
