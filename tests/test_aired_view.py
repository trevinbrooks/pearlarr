# pyright: strict
"""The aired view of a series: Sonarr's rows renumbered by TVDB episode id, and a stand-in per episode they lack."""

from datetime import date

import pytest

from pearlarr.aired_view import (
    AiredSeries,
    NoView,
    aired_series,
    numbered_by,
)
from pearlarr.arr_models import SonarrEpisode
from pearlarr.episode_ids import is_stand_in, stand_in_id, stand_in_tvdb_id
from pearlarr.releases import EpisodeKey, PrintedKey
from pearlarr.scene_numbers import SceneNumbers
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
    rows = [
        sonarr_ep(1, 2, ep_id=7, tvdb_id=101, absolute=9, scene_season=2, scene_episode=1, scene_absolute=9),
        sonarr_ep(1, 1, ep_id=8, tvdb_id=102),
    ]

    view = aired_series(rows, _facts(season_type), stored_orders({AIRED_ORDER: _AIRED}, details=_DETAILS))

    assert isinstance(view, AiredSeries)
    assert (view.in_sonarr, view.stand_ins, view.aired) == (tuple(rows), (), tuple(rows))
    assert view.own_only == ()


def test_a_row_takes_the_aired_numbers_of_its_tvdb_episode_and_keeps_the_rest() -> None:
    # Sonarr's order numbers the two episodes the other way round and gives neither an absolute number.
    rows = [
        sonarr_ep(1, 1, ep_id=7, tvdb_id=102, size=5, release_group="grp"),
        sonarr_ep(1, 2, ep_id=8, tvdb_id=101, episode_file_id=0),
    ]

    view = _view(rows)

    assert _numbers(view.in_sonarr) == [(8, 1, 1, 1), (7, 1, 2, 2)]
    assert view.in_sonarr[1] == rows[0].renumbered(season=1, episode=2, absolute=2)


def test_a_renumbered_row_loses_its_scene_numbers() -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=102, scene_season=1, scene_episode=1, scene_absolute=1)]

    (renumbered,) = _view(rows).in_sonarr

    assert (renumbered.season_number, renumbered.episode_number) == (1, 2)
    assert renumbered.scene_season_number is None
    assert renumbered.scene_episode_number is None
    assert renumbered.scene_absolute_episode_number is None


def test_scene_numbers_are_set_by_aired_number_on_the_rows_and_stand_ins_they_name() -> None:
    # Sonarr's order has rows for the aired S01E02 and S01E03. The special and S01E01 are stand-ins.
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=102, size=5), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)])
    # Releases split the season after its second episode and count absolutely through. They don't number the special.
    numbers = {
        EpisodeKey(1, 1): SceneNumbers(season=1, episode=1, absolute=1),
        EpisodeKey(1, 2): SceneNumbers(season=1, episode=2, absolute=2),
        EpisodeKey(1, 3): SceneNumbers(season=2, episode=1, absolute=3),
    }

    numbered = view.with_scene_numbers(numbers)

    assert [
        (ep.id, ep.scene_season_number, ep.scene_episode_number, ep.scene_absolute_episode_number)
        for ep in numbered.aired
    ] == [(-900, None, None, None), (-101, 1, 1, 1), (7, 1, 2, 2), (8, 2, 1, 3)]
    assert [ep.id for ep in numbered.stand_ins] == [-900, -101]
    assert numbered.in_sonarr[0] == view.in_sonarr[0].model_copy(
        update={"scene_season_number": 1, "scene_episode_number": 2, "scene_absolute_episode_number": 2}
    )


def test_a_row_only_the_series_own_order_lists_is_left_out() -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=555)]

    view = _view(rows)

    assert [ep.id for ep in view.in_sonarr] == [7]
    assert 555 not in {ep.tvdb_id for ep in view.aired}


def test_a_row_only_the_series_own_order_lists_is_kept_apart_as_sonarr_sent_it() -> None:
    own = sonarr_ep(1, 2, ep_id=8, tvdb_id=555, absolute=2, size=5, scene_season=1, scene_episode=2)

    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), own])

    assert view.own_only == (own,)


@pytest.mark.parametrize("tvdb_id", [None, 0, -4])
def test_a_row_without_a_positive_tvdb_id_is_not_kept_apart(tvdb_id: int | None) -> None:
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=tvdb_id)])

    assert view.own_only == ()


def test_an_aired_episode_sonarr_has_no_row_for_gets_a_stand_in_in_aired_position() -> None:
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)]

    view = _view(rows)

    assert _numbers(view.stand_ins) == [(-900, 0, 1, None), (-102, 1, 2, 2)]
    assert _numbers(view.aired) == [(-900, 0, 1, None), (7, 1, 1, 1), (-102, 1, 2, 2), (8, 1, 3, 3)]
    second = view.stand_ins[1]
    assert (second.title, second.air_date, second.runtime, second.tvdb_id) == ("Second", date(2020, 1, 12), 24, 102)
    assert (second.episode_file_id, second.episode_file) == (0, None)
    assert [is_stand_in(ep.id) for ep in view.aired] == [True, False, True, False]


def test_a_stand_ins_id_gives_back_its_tvdb_episode() -> None:
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)])

    assert [stand_in_tvdb_id(ep.id) for ep in view.stand_ins] == [900, 102]
    assert [stand_in_id(tvdb_id) for tvdb_id in (900, 102)] == [ep.id for ep in view.stand_ins]


def test_a_renumbered_row_prints_by_the_number_sonarr_lists_it_under() -> None:
    # Sonarr lists the aired third episode as its second.
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)])

    assert [PrintedKey.of(ep) for ep in view.in_sonarr] == [PrintedKey(EpisodeKey(1, 1)), PrintedKey(EpisodeKey(1, 2))]


def test_a_stand_in_prints_by_its_aired_number_marked_as_aired() -> None:
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=103)])

    printed = PrintedKey.of(view.stand_ins[1])

    assert (printed, printed.label) == (PrintedKey(EpisodeKey(1, 2), aired=True), "aired S01E02")


def test_a_row_as_sonarr_sent_it_prints_by_its_own_number() -> None:
    # A row of a series in aired order, and a row the aired order lacks.
    view = _view([sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=555)])

    as_sent = [PrintedKey.of(ep) for ep in (sonarr_ep(2, 5), *view.own_only)]

    assert as_sent == [PrintedKey(EpisodeKey(2, 5)), PrintedKey(EpisodeKey(1, 2))]


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


def test_the_row_of_an_aired_episode_that_shares_a_kept_number_is_kept_apart() -> None:
    # As above: the aired order lists the row's episode second under the first episode's number.
    aired = (OrderRow(1, 1, 101, 1), OrderRow(1, 1, 102, 1), OrderRow(1, 2, 103, 2))
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=102), sonarr_ep(1, 2, ep_id=8, tvdb_id=101)]

    assert _view(rows, aired).own_only == (rows[0],)


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


_AIRED_ROWS = [
    sonarr_ep(0, 1, ep_id=6, tvdb_id=900),
    sonarr_ep(1, 1, ep_id=7, tvdb_id=101),
    sonarr_ep(1, 2, ep_id=8, tvdb_id=102),
    sonarr_ep(1, 3, ep_id=9, tvdb_id=103),
]
"""Sonarr's rows of the series, numbered by `_AIRED`."""


def _moved(rows: list[SonarrEpisode], tvdb_id: int, season: int, episode: int) -> list[SonarrEpisode]:
    """The rows with one TVDB episode's row at another season and episode number."""

    return [
        row.model_copy(update={"season_number": season, "episode_number": episode}) if row.tvdb_id == tvdb_id else row
        for row in rows
    ]


def test_rows_at_the_numbers_an_order_lists_their_episodes_at_are_numbered_by_it() -> None:
    assert numbered_by(_AIRED_ROWS, _AIRED) is True


@pytest.mark.parametrize(
    "rows",
    [
        # Another order of the series swaps the season's first two episodes.
        pytest.param(_moved(_moved(_AIRED_ROWS, 101, 1, 2), 102, 1, 1), id="another-order"),
        # Another order counts the special as the season's fourth episode, and agrees on every other row.
        pytest.param(_moved(_AIRED_ROWS, 900, 1, 4), id="one-row"),
        # Another order has the special as the second of two, and Sonarr has no row for the first.
        pytest.param(_moved(_AIRED_ROWS, 900, 0, 2), id="a-special-alone"),
    ],
)
def test_a_row_off_the_numbers_an_order_lists_its_episode_at_says_the_rows_are_not_numbered_by_it(
    rows: list[SonarrEpisode],
) -> None:
    assert numbered_by(rows, _AIRED) is False


def test_a_row_of_an_episode_the_order_lacks_says_nothing_of_the_numbering() -> None:
    # Sonarr has a row for an episode TVDB's order doesn't list, and one with no TVDB episode at all.
    rows = [*_AIRED_ROWS, sonarr_ep(1, 4, ep_id=10, tvdb_id=555), sonarr_ep(1, 5, ep_id=11)]

    assert numbered_by(rows, _AIRED) is True


def test_the_second_of_two_episodes_an_order_gives_one_number_is_not_numbered_by_it_anywhere_else() -> None:
    # The order gives two episodes the season's first number. Sonarr's rows give the second one its own.
    order = (OrderRow(1, 1, 101, 1), OrderRow(1, 1, 102, 1))
    rows = [sonarr_ep(1, 1, ep_id=7, tvdb_id=101), sonarr_ep(1, 2, ep_id=8, tvdb_id=102)]

    assert numbered_by(rows, order) is False
