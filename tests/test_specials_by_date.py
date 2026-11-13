# pyright: strict
"""`specials_by_date`: when a title's mapped specials give way to the ones AniList's dates and episode length choose."""

from datetime import date

import pytest

from pearlarr.anilist_client import AniListAiring
from pearlarr.arr_models import SonarrEpisode
from pearlarr.specials_by_date import specials_by_date

from .builders import sonarr_ep

_PREMIERE = date(2020, 8, 29)


def _special(number: int, aired: str | None, runtime: int | None) -> SonarrEpisode:
    return sonarr_ep(0, number, ep_id=number, air_date=aired, runtime=runtime)


def _film(*, end: date | None = _PREMIERE) -> AniListAiring:
    return AniListAiring(start=_PREMIERE, end=end, episodes=1, duration=95)


def _numbers(picked: list[SonarrEpisode] | None) -> list[int | None] | None:
    return None if picked is None else [ep.episode_number for ep in picked]


# A 12-minute special from the year before the film.
_SHORT = _special(1, "2019-02-20", 12)

# A film-length special aired on the film's premiere.
_FEATURE = _special(2, "2020-08-29", 95)


@pytest.mark.parametrize("end", [_PREMIERE, None])
def test_a_film_moves_to_the_special_aired_on_its_date(end: date | None) -> None:
    assert specials_by_date([_SHORT], [_SHORT, _FEATURE], _film(end=end)) == [_FEATURE]


@pytest.mark.parametrize(("mapped_aired", "expected"), [("2020-09-12", None), ("2020-09-13", [2])])
def test_a_mapped_special_within_two_weeks_of_the_date_stands(mapped_aired: str, expected: list[int] | None) -> None:
    mapped = _special(1, mapped_aired, 12)

    assert _numbers(specials_by_date([mapped], [mapped, _FEATURE], _film())) == expected


@pytest.mark.parametrize(
    ("stray_aired", "kept"),
    [
        pytest.param("2020-05-02", 2, id="the earliest, months before the start date"),
        pytest.param("2020-12-12", 1, id="the latest, months after the end date"),
    ],
)
def test_one_mapped_special_far_from_its_date_is_a_miss(stray_aired: str, kept: int) -> None:
    # Two 24-minute parts a week apart. The mapping has one of them and a special aired months away.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 9, 5), episodes=2, duration=24)
    parts = [_special(1, "2020-08-29", 24), _special(2, "2020-09-05", 24)]
    stray = _special(3, stray_aired, 24)

    assert specials_by_date([parts[kept - 1], stray], [*parts, stray], airing) == parts


def test_a_weekly_run_dated_one_airing_late_stands() -> None:
    # Three 5-minute shorts, each dated a week after AniList has it. Another short aired on AniList's start date,
    # so the three shorts on AniList's dates are the wrong ones.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 9, 12), episodes=3, duration=5)
    earlier = _special(1, "2020-08-29", 5)
    run = [_special(2, "2020-09-05", 5), _special(3, "2020-09-12", 5), _special(4, "2020-09-19", 5)]

    assert specials_by_date(run, [earlier, *run], airing) is None


@pytest.mark.parametrize(("aired", "expected"), [("2020-08-28", [2]), ("2020-08-30", [2]), ("2020-08-31", None)])
def test_a_special_may_air_a_day_off_the_date(aired: str, expected: list[int] | None) -> None:
    feature = _special(2, aired, 95)

    assert _numbers(specials_by_date([_SHORT], [_SHORT, feature], _film())) == expected


@pytest.mark.parametrize(
    "specials",
    [
        pytest.param([_SHORT], id="none on the date"),
        pytest.param([_SHORT, _FEATURE, _special(3, "2020-08-29", 95)], id="two on the date"),
    ],
)
def test_a_one_episode_title_needs_exactly_one_special_on_its_date(specials: list[SonarrEpisode]) -> None:
    assert specials_by_date([_SHORT], specials, _film()) is None


def test_two_parts_aired_on_one_day_replace_the_mapped_special() -> None:
    airing = AniListAiring(start=_PREMIERE, end=_PREMIERE, episodes=2, duration=45)
    parts = [_special(2, "2020-08-29", 45), _special(3, "2020-08-29", 45)]

    assert specials_by_date([_SHORT], [_SHORT, *parts], airing) == parts


def test_a_two_episode_title_takes_the_specials_on_its_first_and_last_dates() -> None:
    # Three 45-minute specials air between the dates, so the count alone chooses nothing. The special on the last
    # date is numbered before the one on the first date.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 11, 28), episodes=2, duration=45)
    last, between, first = _special(2, "2020-11-28", 45), _special(3, "2020-10-10", 45), _special(4, "2020-08-29", 45)

    assert specials_by_date([_SHORT], [_SHORT, last, between, first], airing) == [last, first]


@pytest.mark.parametrize(
    ("first_aired", "last_aired"),
    [
        ("2020-08-28", "2020-11-28"),
        ("2020-08-30", "2020-11-28"),
        ("2020-08-29", "2020-11-27"),
        ("2020-08-29", "2020-11-29"),
    ],
)
def test_the_first_and_last_specials_may_air_a_day_off_their_dates(first_aired: str, last_aired: str) -> None:
    # Two more 45-minute specials aired in between, so the count alone chooses nothing.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 11, 28), episodes=2, duration=45)
    first, last = _special(2, first_aired, 45), _special(5, last_aired, 45)
    between = [_special(3, "2020-10-10", 45), _special(4, "2020-10-24", 45)]

    assert specials_by_date([_SHORT], [_SHORT, first, *between, last], airing) == [first, last]


@pytest.mark.parametrize(
    "aired",
    [
        pytest.param(("2020-08-29", "2020-08-30", "2020-11-28"), id="two on the first date"),
        pytest.param(("2020-08-29", "2020-11-27", "2020-11-28"), id="two on the last date"),
    ],
)
def test_two_specials_on_one_of_its_dates_leave_a_two_episode_title(aired: tuple[str, str, str]) -> None:
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 11, 28), episodes=2, duration=45)
    specials = [_special(number, day, 45) for number, day in enumerate(aired, start=2)]

    assert specials_by_date([_SHORT], [_SHORT, *specials], airing) is None


def test_the_first_and_last_dates_pick_only_for_a_two_episode_title() -> None:
    # Three 45-minute episodes over three months, and only two such specials, one on each of the two dates.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 11, 28), episodes=3, duration=45)
    specials = [_special(2, "2020-08-29", 45), _special(3, "2020-11-28", 45)]

    assert specials_by_date([_SHORT], [_SHORT, *specials], airing) is None


def test_one_special_near_both_dates_is_no_pair() -> None:
    # The title aired over two days, and the one 45-minute special aired within a day of both.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 8, 30), episodes=2, duration=45)

    assert specials_by_date([_SHORT], [_SHORT, _special(2, "2020-08-29", 45)], airing) is None


def test_specials_that_are_already_the_mapped_ones_change_nothing() -> None:
    # The mapped pair starts five weeks after AniList's start date, and no other special aired between the dates.
    airing = AniListAiring(start=_PREMIERE, end=date(2020, 11, 28), episodes=2, duration=45)
    mapped = [_special(2, "2020-10-03", 45), _special(3, "2020-11-28", 45)]

    assert specials_by_date(mapped, [_SHORT, *mapped], airing) is None


def test_a_regular_season_episode_is_never_picked() -> None:
    episode = sonarr_ep(1, 5, ep_id=105, air_date="2020-08-29", runtime=95)

    assert specials_by_date([_SHORT], [_SHORT, episode], _film()) is None


def test_a_mapping_that_reaches_a_regular_season_is_left_alone() -> None:
    episode = sonarr_ep(1, 5, ep_id=105, air_date="2019-02-27", runtime=24)

    assert specials_by_date([_SHORT, episode], [_SHORT, _FEATURE, episode], _film()) is None


def test_an_empty_mapping_stays_empty() -> None:
    assert specials_by_date([], [_SHORT, _FEATURE], _film()) is None


@pytest.mark.parametrize(
    "airing",
    [
        pytest.param(AniListAiring(start=None, end=_PREMIERE, episodes=1, duration=95), id="no start date"),
        pytest.param(AniListAiring(start=_PREMIERE, end=_PREMIERE, episodes=None, duration=95), id="no count"),
        pytest.param(AniListAiring(start=_PREMIERE, end=_PREMIERE, episodes=1, duration=None), id="no duration"),
    ],
)
def test_a_title_anilist_gives_too_little_for_is_left_alone(airing: AniListAiring) -> None:
    assert specials_by_date([_SHORT], [_SHORT, _FEATURE], airing) is None


def test_a_mapped_special_without_an_air_date_is_left_alone() -> None:
    undated = _special(1, None, 12)

    assert specials_by_date([undated], [undated, _FEATURE], _film()) is None


def test_a_special_without_an_air_date_is_never_picked() -> None:
    assert specials_by_date([_SHORT], [_SHORT, _special(2, None, 95)], _film()) is None


@pytest.mark.parametrize("runtime", [0, None])
def test_a_special_with_no_runtime_is_never_picked(runtime: int | None) -> None:
    # A 3-minute title, which a runtime of 0 is within 3 minutes of.
    airing = AniListAiring(start=_PREMIERE, end=_PREMIERE, episodes=1, duration=3)

    assert specials_by_date([_SHORT], [_SHORT, _special(2, "2020-08-29", runtime)], airing) is None


@pytest.mark.parametrize(
    ("runtime", "duration", "picked"),
    [
        # A fifth of 95 minutes is 19.
        (76, 95, True),
        (75, 95, False),
        (114, 95, True),
        (115, 95, False),
        # A fifth of 5 minutes is under the 3-minute floor.
        (8, 5, True),
        (9, 5, False),
    ],
)
def test_a_special_is_picked_only_at_a_runtime_close_to_the_episode_length(
    runtime: int, duration: int, picked: bool
) -> None:
    airing = AniListAiring(start=_PREMIERE, end=_PREMIERE, episodes=1, duration=duration)
    dated = _special(2, "2020-08-29", runtime)

    assert (specials_by_date([_SHORT], [_SHORT, dated], airing) == [dated]) is picked
