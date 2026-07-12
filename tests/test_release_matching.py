# pyright: strict
"""Characterization tests for the pure release-matching helpers.

These functions move to `planner.py` in Phase 1; pinning them here proves the
relocation is behavior-preserving.
"""

from pearlarr.coverage import format_episode_ranges
from pearlarr.planner import (
    get_all_seadex_rgs_per_episode,
    get_episode_keys,
    get_same_files_groups,
    normalize_rg,
)
from pearlarr.seadex_types import (
    EpisodeRecord,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
)

from .builders import sonarr_ep


class TestNormalizeRg:
    """`normalize_rg` casefolds and strips whitespace/leading-trailing dashes; blank or `None` in yields `None`."""

    def test_none_and_blank_return_none(self) -> None:
        assert normalize_rg(None) is None
        assert normalize_rg("") is None

    def test_strips_whitespace_dashes_and_casefolds(self) -> None:
        assert normalize_rg("  Era-Raws-  ") == "era-raws"
        assert normalize_rg("-SubsPlease-") == "subsplease"

    def test_internal_dashes_preserved(self) -> None:
        assert normalize_rg("Era-Raws") == "era-raws"


class TestFormatEpisodeRanges:
    """`format_episode_ranges` renders sorted, deduped episode numbers as comma-joined contiguous `E`-ranges."""

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
    """`get_episode_keys` builds a `(season, episode)` set from episode records, missing fields becoming `(None, None)`."""

    def test_builds_season_episode_pairs(self) -> None:
        eps = [EpisodeRecord(season=1, episode=1), EpisodeRecord(season=1, episode=2)]
        assert get_episode_keys(eps) == {(1, 1), (1, 2)}

    def test_missing_keys_become_none(self) -> None:
        assert get_episode_keys([EpisodeRecord()]) == {(None, None)}


class TestGetSameFilesGroups:
    """`get_same_files_groups` groups release names by identical episode coverage.

    Unset (`None`) coverage collapses every group into one; an empty list keeps
    each group apart (coverage unverifiable); matching/differing coverage sets
    group or separate accordingly.
    """

    def test_no_episode_parsing_groups_together(self) -> None:
        # No all_episodes (None) -> no-parsing branch -> all collapse to one group
        seadex = {"A": SeadexReleaseGroupItem(), "B": SeadexReleaseGroupItem()}
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_unparsed_each_on_its_own(self) -> None:
        # Empty list -> "couldn't verify" -> each group kept separately
        seadex = {
            "A": SeadexReleaseGroupItem(all_episodes=[]),
            "B": SeadexReleaseGroupItem(all_episodes=[]),
        }
        assert get_same_files_groups(seadex) == [["A"], ["B"]]

    def test_identical_coverage_grouped(self) -> None:
        seadex = {
            "A": SeadexReleaseGroupItem(all_episodes=[EpisodeRecord(season=1, episode=1)]),
            "B": SeadexReleaseGroupItem(all_episodes=[EpisodeRecord(season=1, episode=1)]),
        }
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_different_coverage_separate(self) -> None:
        seadex = {
            "A": SeadexReleaseGroupItem(all_episodes=[EpisodeRecord(season=1, episode=1)]),
            "B": SeadexReleaseGroupItem(all_episodes=[EpisodeRecord(season=1, episode=2)]),
        }
        assert get_same_files_groups(seadex) == [["A"], ["B"]]


class TestGetAllSeadexRgsPerEpisode:
    """`get_all_seadex_rgs_per_episode` maps each Sonarr-known episode to its casefolded release-group names.

    A dict of one group short-circuits to just the empty `all` bucket; a group
    with no matched episodes falls into `all` instead of a per-episode key.
    """

    def test_single_group_short_circuits(self) -> None:
        # len(seadex_dict) <= 1 returns just the empty "all" bucket
        seadex = {
            "A": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
        }
        assert get_all_seadex_rgs_per_episode(seadex, {(1, 1): sonarr_ep(1, 1)}) == {"all": set()}

    def test_records_episodes_sonarr_has(self) -> None:
        seadex = {
            "Era-Raws": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
            "Other": SeadexReleaseGroupItem(urls={"u2": SeadexUrlItem(episodes=[])}),
        }
        result = get_all_seadex_rgs_per_episode(seadex, {(1, 1): sonarr_ep(1, 1)})
        assert result["S01E01"] == {"era-raws"}
        # Empty episode list -> the group lands in the "all" fallback bucket
        assert result["all"] == {"other"}

    def test_ignores_episodes_sonarr_lacks(self) -> None:
        seadex = {
            "A": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=99)])},
            ),
            "B": SeadexReleaseGroupItem(
                urls={"u2": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
        }
        result = get_all_seadex_rgs_per_episode(seadex, {(1, 1): sonarr_ep(1, 1)})
        assert "S01E99" not in result
        assert result["S01E01"] == {"b"}
