# pyright: strict
"""Characterization tests for the episode-coverage formatting helpers.

These helpers live in `coverage.py`.
"""

from pearlarr.coverage import coverage_of_episodes, coverage_string
from pearlarr.releases import EpisodeKey

from .builders import sonarr_ep


class TestCoverageString:
    """`coverage_string` renders episode keys season by season in order, each season's episodes as ranges."""

    def test_no_keys_is_blank(self) -> None:
        assert coverage_string([]) == ""

    def test_one_season(self) -> None:
        assert coverage_string([EpisodeKey(1, 1), EpisodeKey(1, 2)]) == "S01 E01-E02"

    def test_seasons_run_in_order(self) -> None:
        keys = [EpisodeKey(2, 1), EpisodeKey(0, 10), EpisodeKey(2, 2)]
        assert coverage_string(keys) == "S00 E10, S02 E01-E02"

    def test_a_key_given_twice_counts_once(self) -> None:
        assert coverage_string([EpisodeKey(1, 1), EpisodeKey(1, 1)]) == "S01 E01"


class TestCoverageOfEpisodes:
    """`coverage_of_episodes` is the coverage string of Sonarr episodes."""

    def test_no_episodes_is_blank(self) -> None:
        assert coverage_of_episodes([]) == ""

    def test_reads_each_episodes_numbers(self) -> None:
        assert coverage_of_episodes([sonarr_ep(1, 3), sonarr_ep(1, 4), sonarr_ep(0, 2)]) == "S00 E02, S01 E03-E04"

    def test_an_episode_without_its_numbers_is_left_out(self) -> None:
        assert coverage_of_episodes([sonarr_ep(None, None), sonarr_ep(1, 3)]) == "S01 E03"
