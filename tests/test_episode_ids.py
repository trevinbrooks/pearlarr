# pyright: strict
"""Direct tests for the episode ids: which kind an id is, and a stand-in's id from and to its TVDB episode."""

import pytest

from pearlarr.episode_ids import (
    is_row_id,
    is_stand_in,
    stand_in_id,
    stand_in_tvdb_id,
    stand_ins_among,
    without_stand_ins,
)


@pytest.mark.parametrize(
    ("episode_id", "stand_in", "row"),
    [(41, False, True), (0, False, False), (-9002, True, False)],
)
def test_an_id_is_a_rows_a_stand_ins_or_neither(episode_id: int, stand_in: bool, row: bool) -> None:
    assert (is_stand_in(episode_id), is_row_id(episode_id)) == (stand_in, row)


def test_a_stand_ins_id_is_no_row_id_and_gives_back_its_tvdb_episode() -> None:
    made = stand_in_id(9002)

    assert is_stand_in(made)
    assert stand_in_tvdb_id(made) == 9002


def test_ids_split_into_the_stand_ins_and_the_rest_in_their_order() -> None:
    ids = (12, stand_in_id(9002), 0, 11, stand_in_id(9001))

    assert stand_ins_among(ids) == (stand_in_id(9002), stand_in_id(9001))
    assert without_stand_ins(ids) == (12, 0, 11)
