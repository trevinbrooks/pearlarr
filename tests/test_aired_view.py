# pyright: strict
"""The aired view of a series: Sonarr's rows renumbered by TVDB episode id, and a stand-in per episode they lack."""

from datetime import date

import pytest

from pearlarr.aired_view import AiredSeries, NoView, aired_series, is_stand_in
from pearlarr.arr_models import SonarrEpisode
from pearlarr.tvdb_orders import AIRED_ORDER, CachedOrders, EpisodeDetails, OrderRow, SonarrSeriesOrders

from .builders import sonarr_ep, stored_orders

_AIRED = (
    OrderRow(0, 1, 900, None),
    OrderRow(1, 1, 101, 1),
    OrderRow(1, 2, 102, 2),
    OrderRow(1, 3, 103, 3),
)
_DETAILS = {
    900: EpisodeDetails(title="Extra", air_date=date(2020, 3, 1), runtime=12),
    101: EpisodeDetails(title="First", air_date=date(2020, 1, 5), runtime=24),
    102: EpisodeDetails(title="Second", air_date=date(2020, 1, 12), runtime=24),
    103: EpisodeDetails(title="", air_date=None, runtime=None),
}


def _facts(season_type: str) -> SonarrSeriesOrders:
    return SonarrSeriesOrders(50, season_type, None)


def _view(rows: list[SonarrEpisode], aired: tuple[OrderRow, ...] = _AIRED) -> AiredSeries:
    view = aired_series(rows, _facts("dvd"), stored_orders({AIRED_ORDER: aired}, details=_DETAILS))
    assert isinstance(view, AiredSeries)
    return view


def _numbers(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, int | None, int | None, int | None]]:
    return [(ep.id, ep.season_number, ep.episode_number, ep.absolute_episode_number) for ep in episodes]


@pytest.mark.parametrize("season_type", [AIRED_ORDER, ""])
def test_a_series_in_aired_order_keeps_sonarrs_rows_untouched(season_type: str) -> None:
    # The stored aired order numbers these episodes differently and lists one more. None of it is read.
    rows = [sonarr_ep(1, 2, ep_id=7, tvdb_id=101, absolute=9), sonarr_ep(1, 1, ep_id=8, tvdb_id=102)]

    view = aired_series(rows, _facts(season_type), stored_orders({AIRED_ORDER: _AIRED}, details=_DETAILS))

    assert isinstance(view, AiredSeries)
    assert (view.order, view.in_sonarr, view.stand_ins, view.aired) == (season_type, tuple(rows), (), tuple(rows))


def test_a_row_takes_the_aired_numbers_of_its_tvdb_episode_and_keeps_the_rest() -> None:
    # Sonarr's order numbers the two episodes the other way round and gives neither an absolute number.
    rows = [
        sonarr_ep(1, 1, ep_id=7, tvdb_id=102, size=5, release_group="grp"),
        sonarr_ep(1, 2, ep_id=8, tvdb_id=101, episode_file_id=0),
    ]

    view = _view(rows)

    assert _numbers(view.in_sonarr) == [(8, 1, 1, 1), (7, 1, 2, 2)]
    assert view.in_sonarr[1] == rows[0].model_copy(update={"episode_number": 2, "absolute_episode_number": 2})


def test_a_row_only_the_series_own_order_lists_is_left_out() -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=555)]

    view = _view(rows)

    assert [ep.id for ep in view.in_sonarr] == [7]
    assert 555 not in {ep.tvdb_id for ep in view.aired}


def test_an_aired_episode_sonarr_has_no_row_for_gets_a_stand_in_in_aired_position() -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)]

    view = _view(rows)

    assert _numbers(view.stand_ins) == [(-900, 0, 1, None), (-102, 1, 2, 2)]
    assert _numbers(view.aired) == [(-900, 0, 1, None), (7, 1, 1, 1), (-102, 1, 2, 2), (8, 1, 3, 3)]
    second = view.stand_ins[1]
    assert (second.title, second.air_date, second.runtime, second.tvdb_id) == ("Second", date(2020, 1, 12), 24, 102)
    assert (second.episode_file_id, second.episode_file) == (0, None)
    assert [is_stand_in(ep.id) for ep in view.aired] == [True, False, True, False]


def test_a_stand_in_without_a_title_or_runtime_reads_as_sonarrs_rows_do() -> None:
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101)])

    untitled = view.stand_ins[-1]
    assert (untitled.tvdb_id, untitled.title, untitled.air_date, untitled.runtime) == (103, "TBA", None, 0)


def test_an_order_sharing_no_episode_with_the_aired_one_leaves_every_aired_episode_a_stand_in() -> None:
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=555), sonarr_ep(1, 2, ep_id=8, tvdb_id=556)])

    assert view.in_sonarr == ()
    assert [ep.id for ep in view.aired] == [-900, -101, -102, -103]


def test_two_aired_episodes_with_one_number_keep_the_first_listed() -> None:
    # Sonarr's order has a row for each. Only the first listed is an aired episode of the view.
    aired = (OrderRow(1, 1, 101, 1), OrderRow(1, 1, 102, 1), OrderRow(1, 2, 103, 2))
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=102), sonarr_ep(1, 2, ep_id=8, tvdb_id=101)]

    view = _view(rows, aired)

    assert _numbers(view.aired) == [(8, 1, 1, 1), (-103, 1, 2, 2)]


def test_an_aired_row_without_a_positive_tvdb_id_gets_no_stand_in() -> None:
    aired = (OrderRow(1, 1, 101, 1), OrderRow(1, 2, 0, 2), OrderRow(1, 3, -4, 3))

    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101)], aired)

    assert view.stand_ins == ()
    assert [ep.id for ep in view.aired] == [7]


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(None, id="nothing-stored"),
        pytest.param(stored_orders({"dvd": _AIRED}, details=_DETAILS), id="no-aired-order"),
        pytest.param(stored_orders({AIRED_ORDER: ()}, details={}), id="aired-order-empty"),
        pytest.param(stored_orders({AIRED_ORDER: _AIRED}, details=None), id="stored-before-details-were-kept"),
    ],
)
def test_a_series_in_another_order_has_no_view_until_its_aired_order_and_details_are_stored(
    stored: CachedOrders | None,
) -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101)]

    assert aired_series(rows, _facts("dvd"), stored) is NoView.ORDERS_UNREAD
