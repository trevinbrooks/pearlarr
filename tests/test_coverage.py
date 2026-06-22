"""Characterization tests for the episode-coverage formatting helpers.

These move to ``coverage.py`` in Phase 1.
"""

from seadexarr.modules.coverage import format_episode_coverage
from seadexarr.modules.seadex_arr import SeaDexArr
from tests.builders import make_arr, sonarr_ep


class TestFormatEpisodeCoverage:
    def test_none_when_empty(self) -> None:
        assert format_episode_coverage([]) is None

    def test_none_when_all_keys_missing(self) -> None:
        assert format_episode_coverage([{"foo": 1}]) is None

    def test_single_season(self) -> None:
        eps = [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]
        assert format_episode_coverage(eps) == [("S01", "E01-E02")]

    def test_multi_season_sorted(self) -> None:
        eps = [{"season": 2, "episode": 1}, {"season": 0, "episode": 10}]
        assert format_episode_coverage(eps) == [
            ("S00", "E10"),
            ("S02", "E01"),
        ]


class TestCoverageString:
    def test_empty_is_blank(self) -> None:
        assert make_arr().coverage_string([]) == ""

    def test_joins_seasons(self) -> None:
        eps = [
            {"season": 0, "episode": 10},
            {"season": 2, "episode": 1},
            {"season": 2, "episode": 2},
        ]
        assert make_arr().coverage_string(eps) == "S00 E10, S02 E01-E02"


class TestEpisodesFromEpList:
    def test_none_returns_empty(self) -> None:
        assert SeaDexArr.episodes_from_ep_list(None) == []

    def test_maps_field_names(self) -> None:
        eps = [sonarr_ep(1, 3, episode_file_id=0)]
        assert SeaDexArr.episodes_from_ep_list(eps) == [{"season": 1, "episode": 3}]

    def test_missing_only_drops_episodes_with_files(self) -> None:
        eps = [
            sonarr_ep(1, 1, episode_file_id=0),   # missing (no file)
            sonarr_ep(1, 2, episode_file_id=42),  # has a file
        ]
        result = SeaDexArr.episodes_from_ep_list(eps, missing_only=True)
        assert result == [{"season": 1, "episode": 1}]
