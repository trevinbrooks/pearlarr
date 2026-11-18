# pyright: strict
"""Scene numbers worked out from TheXEM's mappings, on the shapes its replies have."""

from collections.abc import Sequence

import pytest

from pearlarr.releases import EpisodeKey
from pearlarr.scene_numbers import SceneMapping, SceneNumbers, scene_numbers


def _run(tvdb: tuple[int, int], scene: tuple[int, int, int], count: int) -> tuple[SceneMapping, ...]:
    """`count` mappings in a row from `tvdb`, each one episode on from the one before on both sides."""

    (season, episode), (scene_season, scene_episode, scene_absolute) = tvdb, scene
    return tuple(
        SceneMapping(season, episode + past, scene_season, scene_episode + past, scene_absolute + past)
        for past in range(count)
    )


def _as_special(episode: int) -> SceneMapping:
    """A special that releases number as TVDB does. TheXEM gives it no absolute number."""

    return SceneMapping(0, episode, 0, episode, 0)


def _season(season: int, count: int) -> list[EpisodeKey]:
    return [EpisodeKey(season, number) for number in range(1, count + 1)]


def _scene(season: int, episode: int, absolute: int | None) -> SceneNumbers:
    return SceneNumbers(season=season, episode=episode, absolute=absolute)


_SPLIT_SEASON = (*_run((1, 1), (1, 1, 1), 12), *_run((1, 13), (2, 1, 13), 12))
"""Releases split the one TVDB season of 24 in two and count absolutely through both."""

_HALF_MAPPED = (*_run((1, 1), (1, 1, 1), 24), *_run((2, 1), (2, 1, 1), 12))
"""Releases number as TVDB does. TheXEM has the first 12 episodes of the second season."""

_SPECIAL_INSIDE = (SceneMapping(0, 1, 1, 12, 12), *_run((1, 1), (1, 1, 1), 11), *_run((1, 12), (1, 13, 13), 13))
"""Releases number a special as the season's twelfth episode, and each episode after it one higher. TheXEM maps 24
episodes of the season."""


def test_a_mapped_episode_takes_its_mappings_scene_numbers_and_any_other_is_left_out() -> None:
    numbers = scene_numbers([EpisodeKey(0, 1), *_season(1, 24)], _SPLIT_SEASON)

    assert numbers == {
        **{EpisodeKey(1, number): _scene(1, number, number) for number in range(1, 13)},
        **{EpisodeKey(1, number): _scene(2, number - 12, number) for number in range(13, 25)},
    }


def test_a_series_without_mappings_has_no_scene_numbers() -> None:
    assert scene_numbers(_season(1, 12), ()) == {}


def test_a_mapping_of_an_episode_the_series_lacks_gives_no_numbers() -> None:
    # TVDB lists 23 episodes of the season, one fewer than TheXEM maps.
    numbers = scene_numbers([*_season(0, 3), *_season(1, 23)], _SPECIAL_INSIDE)

    assert set(numbers) == {EpisodeKey(0, 1), *_season(1, 23)}


def test_a_mapping_whose_scene_numbers_are_all_zero_gives_no_numbers() -> None:
    # Releases number four specials into the season. TheXEM lists the third special with zeros.
    mappings = (
        SceneMapping(0, 1, 1, 6, 6),
        SceneMapping(0, 2, 1, 11, 11),
        SceneMapping(0, 3, 0, 0, 0),
        SceneMapping(0, 4, 1, 17, 17),
        SceneMapping(0, 5, 1, 20, 20),
        *_run((1, 1), (1, 1, 1), 5),
        *_run((1, 6), (1, 7, 7), 4),
        *_run((1, 10), (1, 12, 12), 5),
        *_run((1, 15), (1, 18, 18), 2),
        *_run((1, 17), (1, 21, 21), 8),
    )

    numbers = scene_numbers([*_season(0, 6), *_season(1, 24)], mappings)

    assert EpisodeKey(0, 3) not in numbers
    assert numbers[EpisodeKey(0, 2)] == _scene(1, 11, 11)


@pytest.mark.parametrize(
    ("aired", "mappings", "numbered_on"),
    [
        pytest.param(
            [*_season(1, 24), *_season(2, 24)],
            _HALF_MAPPED,
            {EpisodeKey(2, number): _scene(2, number, number) for number in range(13, 25)},
            id="the-rest-of-a-half-mapped-season",
        ),
        pytest.param(
            # TVDB lists two more episodes than TheXEM has mapped.
            _season(1, 26),
            _SPLIT_SEASON,
            {EpisodeKey(1, 25): _scene(2, 13, 25), EpisodeKey(1, 26): _scene(2, 14, 26)},
            id="in-the-scene-season-of-the-last-mapping",
        ),
    ],
)
def test_an_episode_after_the_last_mapping_of_the_last_mapped_season_is_numbered_on_from_it(
    aired: list[EpisodeKey], mappings: Sequence[SceneMapping], numbered_on: dict[EpisodeKey, SceneNumbers]
) -> None:
    numbers = scene_numbers(aired, mappings)

    mapped = {mapping.tvdb for mapping in mappings}
    assert {key: scene for key, scene in numbers.items() if key not in mapped} == numbered_on


def test_the_aired_episodes_may_be_given_as_an_iterable_read_once() -> None:
    aired = [*_season(1, 24), *_season(2, 24)]

    assert scene_numbers(iter(aired), _HALF_MAPPED) == scene_numbers(aired, _HALF_MAPPED)


def test_nothing_is_numbered_on_in_a_scene_season_whose_last_episode_is_a_tvdb_special() -> None:
    # Releases number the second special as the season's 26th episode. TheXEM lists specials first, as it lists every
    # reply in TVDB's order. TVDB lists a 26th episode TheXEM has no mapping for.
    mappings = (
        _as_special(1),
        SceneMapping(0, 2, 1, 26, 26),
        *(_as_special(episode) for episode in range(3, 7)),
        *_run((1, 1), (1, 1, 1), 25),
    )

    numbers = scene_numbers([*_season(0, 6), *_season(1, 26)], mappings)

    assert EpisodeKey(1, 26) not in numbers


def test_a_special_inside_the_scene_season_and_not_last_in_it_does_not_stop_the_numbering() -> None:
    # TVDB lists a 25th episode, one more than TheXEM has mapped.
    numbers = scene_numbers([*_season(0, 3), *_season(1, 25)], _SPECIAL_INSIDE)

    assert numbers[EpisodeKey(1, 25)] == _scene(1, 26, 26)


def test_a_season_after_the_last_mapped_one_is_numbered_as_far_past_the_last_scene_season() -> None:
    # TVDB lists a second season TheXEM has no mapping for. Releases ended the first on their second.
    numbers = scene_numbers([*_season(1, 24), *_season(2, 2)], _SPLIT_SEASON)

    assert (numbers[EpisodeKey(2, 1)], numbers[EpisodeKey(2, 2)]) == (_scene(3, 1, None), _scene(3, 2, None))


def test_a_season_after_the_last_mapped_one_has_no_scene_numbers_when_releases_and_tvdb_end_on_one_season() -> None:
    # Releases number four seasons as TVDB does. TVDB lists a fifth TheXEM has no mapping for.
    lengths = {1: 13, 2: 13, 3: 26, 4: 12}
    mappings = [mapping for season, count in lengths.items() for mapping in _run((season, 1), (season, 1, 1), count)]
    aired = [*(key for season, count in lengths.items() for key in _season(season, count)), *_season(5, 12)]

    numbers = scene_numbers(aired, mappings)

    assert {key.season for key in numbers} == {1, 2, 3, 4}


def test_the_last_scene_season_is_the_highest_of_every_mapping_those_of_episodes_the_series_lacks_too() -> None:
    # TheXEM maps 24 episodes of the first TVDB season, the later 12 as the releases' second season. TVDB lists 12
    # of them, and 12 more as a second season of its own.
    mappings = (*_run((1, 1), (1, 1, 1), 12), *_run((1, 13), (2, 1, 1), 12))

    numbers = scene_numbers([*_season(1, 12), *_season(2, 12)], mappings)

    assert {key: scene for key, scene in numbers.items() if key.season == 2} == {
        EpisodeKey(2, number): _scene(3, number, None) for number in range(1, 13)
    }


def test_the_last_tvdb_season_is_the_highest_of_every_mapping_those_of_episodes_the_series_lacks_too() -> None:
    # TheXEM maps four specials, half of the first season and all of a second season TVDB doesn't list.
    mappings = (
        *(_as_special(episode) for episode in range(1, 5)),
        *_run((1, 1), (1, 1, 1), 5),
        *_run((2, 1), (2, 1, 1), 10),
    )

    numbers = scene_numbers([*_season(0, 9), *_season(1, 10)], mappings)

    assert set(numbers) == {*_season(0, 4), *_season(1, 5)}
