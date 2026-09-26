# pyright: strict
"""Parse reading: a range key Sonarr read as one episode is widened to the whole range, or left as read."""

from typing import ClassVar

import pytest

from pearlarr.parse_reading import widen_range_key
from pearlarr.placement_types import SeriesFacts
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo

from .builders import parsed_info, series_index


def _matched(info: ParsedFileInfo) -> list[tuple[int, int, int | None]]:
    """The matched pairs as `(season, episode, id)` triples."""

    return [(pair.season_number, pair.episode_number, pair.id) for pair in info.matched_episodes]


class TestWidenRangeKey:
    """`widen_range_key` turns a tight `SxxEyy-zz` range Sonarr read as its first episode into the whole range.

    When it refuses, it hands back the very same parse object.
    """

    # Season 1 has episodes 1-12 (absolutes 1-12), season 2 has 1-14 but no 6 (absolutes 13-26), season 3 has
    # 1-3, and there are ten specials.
    _KEYS: ClassVar[dict[EpisodeKey, int]] = {
        **{EpisodeKey(1, e): 100 + e for e in range(1, 13)},
        **{EpisodeKey(2, e): 200 + e for e in range(1, 15) if e != 6},
        **{EpisodeKey(3, e): 300 + e for e in range(1, 4)},
        **{EpisodeKey(0, e): 8000 + e for e in range(1, 11)},
    }
    _FACTS: ClassVar[SeriesFacts] = SeriesFacts(
        series_index(_KEYS, absolutes={**{100 + e: e for e in range(1, 13)}, **{200 + e: 12 + e for e in range(1, 15)}})
    )

    @pytest.mark.parametrize(
        ("name", "season", "episodes"),
        [
            ("show s01e05-07 [1080p].mkv", 1, (5, 6, 7)),
            ("show s01e05-e07.mkv", 1, (5, 6, 7)),
            ("Show S01E05-06v2 [1080p].mkv", 1, (5, 6)),
            ("show s03e01-02.mkv", 3, (1, 2)),
        ],
    )
    def test_a_tight_range_widens_to_every_episode(self, name: str, season: int, episodes: tuple[int, ...]) -> None:
        info = parsed_info(season=season, episodes=(episodes[0],))

        widened = widen_range_key(name, info, self._FACTS)

        assert widened.episode_numbers == episodes

    def test_a_name_that_spells_no_range_is_untouched(self) -> None:
        info = parsed_info(season=1, episodes=(5,))

        assert widen_range_key("show s01e05 - 06 - title.mkv", info, self._FACTS) is info

    @pytest.mark.parametrize(("name", "first"), [("show s01e07-05.mkv", 7), ("show s01e05-05.mkv", 5)])
    def test_a_range_that_does_not_count_up_is_untouched(self, name: str, first: int) -> None:
        # Sonarr read the first number, and the last one isn't above it.
        info = parsed_info(season=1, episodes=(first,))

        assert widen_range_key(name, info, self._FACTS) is info

    def test_a_range_wider_than_a_triple_episode_is_untouched(self) -> None:
        # Four wide is past the cap. A last number that far off is more likely an absolute than a range's end.
        info = parsed_info(season=1, episodes=(1,))

        assert widen_range_key("show s01e01-04.mkv", info, self._FACTS) is info

    def test_a_range_ending_outside_the_series_is_untouched(self) -> None:
        info = parsed_info(season=1, episodes=(11,))

        assert widen_range_key("show s01e11-13.mkv", info, self._FACTS) is info

    def test_a_range_over_an_episode_the_series_lacks_is_untouched(self) -> None:
        # Season 2 has no episode 6. Rather than a range our map could only half place, the file keeps Sonarr's read.
        info = parsed_info(season=2, episodes=(5,))

        assert widen_range_key("show s02e05-07.mkv", info, self._FACTS) is info

    def test_a_range_covering_its_whole_season_is_untouched(self) -> None:
        info = parsed_info(season=3, episodes=(1,))

        assert widen_range_key("show s03e01-03.mkv", info, self._FACTS) is info

    def test_a_range_sonarr_read_whole_is_untouched(self) -> None:
        info = parsed_info(season=1, episodes=(5, 6, 7))

        assert widen_range_key("show s01e05-07.mkv", info, self._FACTS) is info

    def test_a_range_sonarr_read_in_another_season_is_untouched(self) -> None:
        info = parsed_info(season=2, episodes=(5,))

        assert widen_range_key("show s01e05-07.mkv", info, self._FACTS) is info

    def test_a_range_sonarr_read_as_another_episode_is_untouched(self) -> None:
        info = parsed_info(season=1, episodes=(6,))

        assert widen_range_key("show s01e05-07.mkv", info, self._FACTS) is info

    def test_a_dual_numbered_name_is_untouched(self) -> None:
        # Season 2 episode 1 is absolute 3 here, so the "03" is its absolute number, not the end of a range.
        facts = SeriesFacts(series_index(self._KEYS, absolutes={201: 3}))
        info = parsed_info(season=2, episodes=(1,), absolutes=(3,), matched=((2, 1),))

        assert widen_range_key("show s02e01-03.mkv", info, facts) is info

    def test_a_lone_absolute_read_off_the_range_end_is_dropped(self) -> None:
        # Sonarr read "s01e05-06" as episode 5 plus absolute 6, but the 6 is just the range's end.
        info = parsed_info(season=1, episodes=(5,), absolutes=(6,))

        widened = widen_range_key("show s01e05-06 [grp].mkv", info, self._FACTS)

        assert (widened.episode_numbers, widened.absolute_episode_numbers) == ((5, 6), ())

    def test_any_other_absolute_is_kept(self) -> None:
        info = parsed_info(season=1, episodes=(5,), absolutes=(30,))

        widened = widen_range_key("show s01e05-06.mkv", info, self._FACTS)

        assert (widened.episode_numbers, widened.absolute_episode_numbers) == ((5, 6), (30,))

    def test_a_matched_pair_on_the_first_episode_widens_with_the_range(self) -> None:
        # Sonarr matched the first episode, so the rest of the range gets matched pairs too, with no Sonarr id.
        info = parsed_info(season=1, episodes=(5,), matched=((1, 5),))

        widened = widen_range_key("show s01e05-07.mkv", info, self._FACTS)

        assert _matched(widened) == [(1, 5, None), (1, 6, None), (1, 7, None)]

    def test_an_alias_shifted_match_is_widened_in_its_own_season(self) -> None:
        # A sequel numbered as its own season 1 that Sonarr matched into season 2, so the added pairs are in season 2.
        info = parsed_info(season=1, episodes=(7,), matched=((2, 7),))

        widened = widen_range_key("show s01e07-08.mkv", info, self._FACTS)

        assert (widened.episode_numbers, _matched(widened)) == ((7, 8), [(2, 7, None), (2, 8, None)])

    def test_a_range_running_past_its_season_widens_in_the_matched_season(self) -> None:
        # Season 1 ends at episode 12, so only season 2, where Sonarr matched the first episode, holds the range.
        info = parsed_info(season=1, episodes=(12,), matched=((2, 12),))

        widened = widen_range_key("show s01e12-13.mkv", info, self._FACTS)

        assert (widened.episode_numbers, _matched(widened)) == ((12, 13), [(2, 12, None), (2, 13, None)])

    def test_a_matched_season_missing_an_episode_does_not_block_the_range(self) -> None:
        # Season 2 has no episode 6, but the name's own season 1 holds the whole range.
        info = parsed_info(season=1, episodes=(5,), matched=((2, 5),))

        widened = widen_range_key("show s01e05-07.mkv", info, self._FACTS)

        assert widened.episode_numbers == (5, 6, 7)

    def test_a_range_fitting_neither_its_season_nor_the_matched_one_is_untouched(self) -> None:
        # Season 2 has no episode 6 and season 3 stops at episode 3, so neither season holds the range.
        info = parsed_info(season=2, episodes=(5,), matched=((3, 5),))

        assert widen_range_key("show s02e05-07.mkv", info, self._FACTS) is info

    def test_a_specials_match_on_the_first_episode_does_not_block_the_range(self) -> None:
        # A specials match is never an alias shift, so only the name's own season has to hold the range.
        info = parsed_info(season=1, episodes=(5,), matched=((0, 5),))

        widened = widen_range_key("show s01e05-06.mkv", info, self._FACTS)

        assert widened.episode_numbers == (5, 6)

    def test_a_pair_sonarr_already_matched_is_not_added_again(self) -> None:
        info = parsed_info(season=1, episodes=(7,), matched=((2, 7), (2, 8)))

        widened = widen_range_key("show s01e07-08.mkv", info, self._FACTS)

        assert _matched(widened) == [(2, 7, None), (2, 8, None)]

    def test_a_matched_pair_off_the_first_episode_is_left_as_is(self) -> None:
        # Sonarr matched the absolute it read off the range's end, not the first episode, so the match isn't widened.
        info = parsed_info(season=0, episodes=(3,), absolutes=(4,), matched=((1, 4),))

        widened = widen_range_key("show s00e03-04.mkv", info, self._FACTS)

        assert (widened.episode_numbers, widened.absolute_episode_numbers) == ((3, 4), ())
        assert widened.matched_episodes == info.matched_episodes
