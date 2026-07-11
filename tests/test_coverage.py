# pyright: strict
"""Characterization tests for the episode-coverage formatting helpers.

These helpers live in ``coverage.py``.
"""

from pearlarr.modules.coverage import (
    coverage_string,
    episodes_from_ep_list,
    format_episode_coverage,
)
from pearlarr.modules.seadex_types import EpisodeRecord

from .builders import sonarr_ep


class TestFormatEpisodeCoverage:
    def test_none_when_empty(self) -> None:
        assert format_episode_coverage([]) is None

    def test_none_when_all_keys_missing(self) -> None:
        assert format_episode_coverage([EpisodeRecord()]) is None

    def test_single_season(self) -> None:
        eps = [EpisodeRecord(season=1, episode=1), EpisodeRecord(season=1, episode=2)]
        assert format_episode_coverage(eps) == [("S01", "E01-E02")]

    def test_multi_season_sorted(self) -> None:
        eps = [EpisodeRecord(season=2, episode=1), EpisodeRecord(season=0, episode=10)]
        assert format_episode_coverage(eps) == [
            ("S00", "E10"),
            ("S02", "E01"),
        ]


class TestCoverageString:
    def test_empty_is_blank(self) -> None:
        assert coverage_string([]) == ""

    def test_joins_seasons(self) -> None:
        eps = [
            EpisodeRecord(season=0, episode=10),
            EpisodeRecord(season=2, episode=1),
            EpisodeRecord(season=2, episode=2),
        ]
        assert coverage_string(eps) == "S00 E10, S02 E01-E02"


class TestEpisodesFromEpList:
    def test_none_returns_empty(self) -> None:
        assert episodes_from_ep_list(None) == []

    def test_maps_field_names(self) -> None:
        eps = [sonarr_ep(1, 3, episode_file_id=0)]
        assert episodes_from_ep_list(eps) == [EpisodeRecord(season=1, episode=3)]

    def test_missing_only_drops_episodes_with_files(self) -> None:
        eps = [
            sonarr_ep(1, 1, episode_file_id=0),  # missing (no file)
            sonarr_ep(1, 2, episode_file_id=42),  # has a file
        ]
        result = episodes_from_ep_list(eps, missing_only=True)
        assert result == [EpisodeRecord(season=1, episode=1)]
