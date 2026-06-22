"""Characterization tests for the pure release-matching helpers.

These functions move to ``planner.py`` in Phase 1; pinning them here proves the
relocation is behaviour-preserving.
"""

from seadexarr.modules.coverage import format_episode_ranges
from seadexarr.modules.planner import (
    get_all_seadex_rgs_per_episode,
    get_episode_keys,
    get_same_files_groups,
    normalize_rg,
)


class TestNormalizeRg:
    def test_none_and_blank_return_none(self) -> None:
        assert normalize_rg(None) is None
        assert normalize_rg("") is None

    def test_strips_whitespace_dashes_and_casefolds(self) -> None:
        assert normalize_rg("  Era-Raws-  ") == "era-raws"
        assert normalize_rg("-SubsPlease-") == "subsplease"

    def test_internal_dashes_preserved(self) -> None:
        assert normalize_rg("Era-Raws") == "era-raws"


class TestFormatEpisodeRanges:
    def test_empty(self) -> None:
        assert format_episode_ranges([]) == ""

    def test_single(self) -> None:
        assert format_episode_ranges([5]) == "E05"

    def test_contiguous_run(self) -> None:
        assert format_episode_ranges([1, 2, 3]) == "E01-E03"

    def test_gaps_split(self) -> None:
        assert format_episode_ranges([1, 2, 3, 7, 8]) == "E01-E03, E07-E08"

    def test_unsorted_and_duplicates(self) -> None:
        assert format_episode_ranges([3, 1, 2, 2]) == "E01-E03"


class TestGetEpisodeKeys:
    def test_builds_season_episode_pairs(self) -> None:
        eps = [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]
        assert get_episode_keys(eps) == {(1, 1), (1, 2)}

    def test_missing_keys_become_none(self) -> None:
        assert get_episode_keys([{}]) == {(None, None)}


class TestGetSameFilesGroups:
    def test_no_episode_parsing_groups_together(self) -> None:
        # No "all_episodes" key -> None branch -> all collapse to one group
        seadex = {"A": {}, "B": {}}
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_unparsed_each_on_its_own(self) -> None:
        # Empty list -> "couldn't verify" -> each group kept separately
        seadex = {"A": {"all_episodes": []}, "B": {"all_episodes": []}}
        assert get_same_files_groups(seadex) == [["A"], ["B"]]

    def test_identical_coverage_grouped(self) -> None:
        seadex = {
            "A": {"all_episodes": [{"season": 1, "episode": 1}]},
            "B": {"all_episodes": [{"season": 1, "episode": 1}]},
        }
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_different_coverage_separate(self) -> None:
        seadex = {
            "A": {"all_episodes": [{"season": 1, "episode": 1}]},
            "B": {"all_episodes": [{"season": 1, "episode": 2}]},
        }
        assert get_same_files_groups(seadex) == [["A"], ["B"]]


class TestGetAllSeadexRgsPerEpisode:
    def test_single_group_short_circuits(self) -> None:
        # len(seadex_dict) <= 1 returns just the empty "all" bucket
        seadex = {"A": {"urls": {"u": {"episodes": [{"season": 1, "episode": 1}]}}}}
        assert get_all_seadex_rgs_per_episode(seadex, {(1, 1): {}}) == {"all": set()}

    def test_records_episodes_sonarr_has(self) -> None:
        seadex = {
            "Era-Raws": {"urls": {"u": {"episodes": [{"season": 1, "episode": 1}]}}},
            "Other": {"urls": {"u2": {"episodes": []}}},
        }
        result = get_all_seadex_rgs_per_episode(seadex, {(1, 1): {}})
        assert result["S01E01"] == {"era-raws"}
        # Empty episode list -> the group lands in the "all" fallback bucket
        assert result["all"] == {"other"}

    def test_ignores_episodes_sonarr_lacks(self) -> None:
        seadex = {
            "A": {"urls": {"u": {"episodes": [{"season": 1, "episode": 99}]}}},
            "B": {"urls": {"u2": {"episodes": [{"season": 1, "episode": 1}]}}},
        }
        result = get_all_seadex_rgs_per_episode(seadex, {(1, 1): {}})
        assert "S01E99" not in result
        assert result["S01E01"] == {"b"}
