# pyright: strict
"""Direct tests for following a stored pending import's episode ids onto the rows that hold their TVDB episodes."""

import pytest

from pearlarr.arr_models import SonarrEpisode
from pearlarr.pending_follow import Followed, follow
from pearlarr.pending_import import NO_TVDB_IDS, PendingImport, TvdbIds

from .builders import pending_import, sonarr_ep, title_claim, two_claim_pending_import


def _rows(*pairs: tuple[int, int | None]) -> list[SonarrEpisode]:
    """One season's rows with no file, each an `(id, TVDB episode id)` pair, numbered in the order given."""

    return [
        sonarr_ep(1, number, ep_id=row_id, tvdb_id=tvdb_id, episode_file_id=0)
        for number, (row_id, tvdb_id) in enumerate(pairs, start=1)
    ]


def _follow(pending: PendingImport, *pairs: tuple[int, int | None]) -> Followed:
    """The follow of `pending`, claimed on series 7 alone, under the series' rows read as `pairs` (`_rows`)."""

    followed = follow(pending, {7: _rows(*pairs)})
    assert followed is not None
    return followed


def _followed(pending: PendingImport, *pairs: tuple[int, int | None]) -> PendingImport:
    """`pending` as `_follow` leaves it."""

    return _follow(pending, *pairs).pending


class TestRowsFollowedUnder:
    """The rows a pending import is followed under, and when it can't be."""

    def test_the_rows_of_every_claimed_series_are_read(self) -> None:
        pending = two_claim_pending_import(
            series_ids=(7, 8),
            windows=((101,), (201,)),
            file_episode_map={},
            tvdb_ids={101: 9001, 201: 9201},
        )

        followed = follow(pending, {7: _rows((111, 9001)), 8: _rows((211, 9201))})

        assert followed is not None
        assert [claim.ordered_episode_ids for claim in followed.pending.claims] == [(111,), (211,)]

    @pytest.mark.parametrize(
        "rows_by_series",
        [
            pytest.param({}, id="a-series-not-read"),
            pytest.param({7: []}, id="a-series-with-no-rows"),
            pytest.param({7: _rows((111, 9001), (112, 9001))}, id="two-rows-on-an-episode-it-holds"),
        ],
    )
    def test_a_pending_import_cannot_be_followed_under(self, rows_by_series: dict[int, list[SonarrEpisode]]) -> None:
        pending = pending_import(tvdb_ids={101: 9001})

        assert follow(pending, rows_by_series) is None

    def test_two_rows_on_an_episode_it_does_not_hold_are_passed_over(self) -> None:
        pending = pending_import(tvdb_ids={101: 9001})

        followed = _followed(pending, (111, 9001), (112, 9002), (113, 9002))

        assert followed.tvdb_ids == TvdbIds({111: 9001})

    def test_a_pending_import_stored_without_tvdb_ids_is_not_followed(self) -> None:
        pending = pending_import(tvdb_ids=None)

        assert follow(pending, {7: _rows((111, 9001))}) is None


class TestFollow:
    """`follow` puts every id a pending import holds on the row that holds its TVDB episode."""

    def test_an_id_moves_to_the_row_that_holds_its_episode(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02.mkv": [102]},
            claims=(
                title_claim(anilist_id=1, ordered_episode_ids=[101, 102], preowned_episode_ids=[102]),
                title_claim(anilist_id=2, ordered_episode_ids=[102, 103]),
            ),
            tvdb_ids={101: 9001, 102: 9002, 103: 9003},
        )

        followed = _followed(pending, (101, 9001), (112, 9002), (103, 9003))

        assert dict(followed.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (112,)}
        assert [claim.ordered_episode_ids for claim in followed.claims] == [(101, 112), (112, 103)]
        assert followed.claims[0].preowned_episode_ids == (112,)
        assert followed.tvdb_ids == TvdbIds({101: 9001, 112: 9002, 103: 9003})

    def test_two_ids_that_swapped_episodes_both_move(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02.mkv": [102]},
            ordered_episode_ids=[101, 102],
            tvdb_ids={101: 9001, 102: 9002},
        )

        followed = _followed(pending, (101, 9002), (102, 9001))

        assert dict(followed.file_episode_map) == {"show - 01.mkv": (102,), "show - 02.mkv": (101,)}
        assert followed.claims[0].ordered_episode_ids == (102, 101)
        assert followed.tvdb_ids == TvdbIds({102: 9001, 101: 9002})

    def test_an_id_whose_episode_has_no_row_becomes_its_stand_in(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02-03.mkv": [102, 103]},
            ordered_episode_ids=[101, 102, 103],
            preowned_episode_ids=[101, 102],
            tvdb_ids={101: 9001, 102: 9002, 103: 9003},
        )

        followed = _followed(pending, (101, 9001), (103, 9003))

        assert followed.claims[0].ordered_episode_ids == (101, -9002, 103)
        assert followed.claims[0].preowned_episode_ids == (101,)
        assert dict(followed.file_episode_map) == {"show - 01.mkv": (101,)}
        assert followed.excluded_files == ("show - 02-03.mkv",)
        assert followed.tvdb_ids == TvdbIds({101: 9001, 103: 9003})

    def test_a_windows_stand_in_becomes_the_id_of_the_row_its_episode_has(self) -> None:
        pending = pending_import(ordered_episode_ids=[101, -9002], tvdb_ids={101: 9001})

        followed = _followed(pending, (101, 9001), (102, 9002))

        assert followed.claims[0].ordered_episode_ids == (101, 102)
        assert followed.tvdb_ids == TvdbIds({101: 9001, 102: 9002})

    def test_a_switch_and_a_switch_back_leave_every_id_on_its_episodes_row(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02.mkv": [102]},
            ordered_episode_ids=[101, 102, 103],
            tvdb_ids={101: 9001, 102: 9002, 103: 9003},
        )

        switched = _followed(pending, (101, 9002), (102, 9001))
        back = _followed(switched, (101, 9001), (102, 9002), (113, 9003))

        assert switched.claims[0].ordered_episode_ids == (102, 101, -9003)
        assert back.claims[0].ordered_episode_ids == (101, 102, 113)
        assert dict(back.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert back.tvdb_ids == TvdbIds({101: 9001, 102: 9002, 113: 9003})

    def test_a_pending_import_whose_ids_stayed_is_returned_itself(self) -> None:
        pending = pending_import(ordered_episode_ids=[101, 102], tvdb_ids={101: 9001, 102: 9002})

        followed = _follow(pending, (101, 9001), (102, 9002), (103, 9003))

        assert followed.pending is pending
        assert followed.set_aside == {}

    def test_a_window_whose_every_episode_lost_its_row_keeps_its_length(self) -> None:
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], tvdb_ids={101: 9001, 102: 9002})

        followed = _followed(pending, (103, 9003))

        assert followed.claims[0].ordered_episode_ids == (-9001, -9002)
        assert followed.tvdb_ids == NO_TVDB_IDS

    def test_an_id_stored_with_no_tvdb_id_stays(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02.mkv": [102]},
            ordered_episode_ids=[101, 102],
            tvdb_ids={102: 9002},
        )

        followed = _followed(pending, (101, 9001), (112, 9002))

        assert dict(followed.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (112,)}
        assert followed.claims[0].ordered_episode_ids == (101, 112)
        assert followed.tvdb_ids == TvdbIds({101: None, 112: 9002})

    def test_a_file_on_a_row_outside_every_window_leaves_the_file_map_when_its_row_is_gone(self) -> None:
        # A season's opener is two halves in the order Sonarr numbered the series by when the 13 files were stored:
        # the first two sit on the halves' rows, outside the window. Sonarr sends neither row at this read.
        names = [f"[grp] show ending s1 - {n:02d} [abcd1234].mkv" for n in range(1, 14)]
        rest = {names[n]: (400 + n,) for n in range(2, 13)}
        aired = {400 + n: 7400 + n for n in range(1, 13)}
        pending = pending_import(
            file_episode_map={names[0]: (9001,), names[1]: (9002,), **rest},
            seadex_files=names,
            ordered_episode_ids=tuple(aired),
            tvdb_ids={9001: 7901, 9002: 7902, **aired},
        )

        followed = _followed(pending, *aired.items())

        assert dict(followed.file_episode_map) == rest
        assert followed.excluded_files == (names[0], names[1])
        assert followed.claims[0].ordered_episode_ids == tuple(aired)
        assert followed.tvdb_ids == TvdbIds(aired)

    def test_a_file_taken_out_is_set_aside_with_its_episodes_that_have_no_row(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101], "show - 02-03.mkv": [102, 103]},
            tvdb_ids={101: 9001, 102: 9002, 103: 9003},
        )

        followed = _follow(pending, (101, 9001), (103, 9003))

        assert followed.set_aside == {"show - 02-03.mkv": (9002,)}
