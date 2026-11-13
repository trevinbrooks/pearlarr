# pyright: strict
"""The pure release-matching helpers the planner uses."""

import pytest

from pearlarr.coverage import format_episode_ranges
from pearlarr.planner import (
    episode_coverage,
    get_episode_keys,
    get_same_files_groups,
)
from pearlarr.releases import EpisodeKey, GroupReleases, Release, ReleaseEpisode, normalize_group

from .builders import sonarr_ep


class TestNormalizeGroup:
    """`normalize_group` casefolds and strips whitespace/leading-trailing dashes. Blank or `None` in yields `None`."""

    def test_none_and_blank_return_none(self) -> None:
        assert normalize_group(None) is None
        assert normalize_group("") is None

    def test_strips_whitespace_dashes_and_casefolds(self) -> None:
        assert normalize_group("  Era-Raws-  ") == "era-raws"
        assert normalize_group("-SubsPlease-") == "subsplease"


class TestFormatEpisodeRanges:
    """`format_episode_ranges` renders sorted, deduped episode numbers as comma-joined contiguous `E`-ranges."""

    @pytest.mark.parametrize(
        ("numbers", "rendered"),
        [
            pytest.param([], "", id="empty"),
            pytest.param([5], "E05", id="single"),
            pytest.param([1, 2, 3], "E01-E03", id="contiguous run"),
            pytest.param([1, 2, 3, 7, 8], "E01-E03, E07-E08", id="gaps split"),
            pytest.param([3, 1, 2, 2], "E01-E03", id="unsorted and duplicates"),
        ],
    )
    def test_renders_the_ranges(self, numbers: list[int], rendered: str) -> None:
        assert format_episode_ranges(numbers) == rendered


class TestGetEpisodeKeys:
    """`get_episode_keys` builds a `(season, episode)` set from release episodes, absent fields becoming `(None, None)`."""

    def test_builds_season_episode_pairs(self) -> None:
        eps = [ReleaseEpisode(season=1, episode=1), ReleaseEpisode(season=1, episode=2)]
        assert get_episode_keys(eps) == {(1, 1), (1, 2)}

    def test_missing_keys_become_none(self) -> None:
        assert get_episode_keys([ReleaseEpisode()]) == {(None, None)}


class TestGetSameFilesGroups:
    """`get_same_files_groups` groups release names by identical episode coverage.

    Unset (`None`) coverage collapses every group into one. An empty list keeps
    each group apart (coverage unverifiable). Matching/differing coverage sets
    group or separate accordingly.
    """

    def test_no_episode_parsing_groups_together(self) -> None:
        seadex = {"A": GroupReleases(), "B": GroupReleases()}
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_unparsed_each_on_its_own(self) -> None:
        seadex = {
            "A": GroupReleases(all_episodes=[]),
            "B": GroupReleases(all_episodes=[]),
        }
        assert get_same_files_groups(seadex) == [["A"], ["B"]]

    def test_identical_coverage_grouped(self) -> None:
        seadex = {
            "A": GroupReleases(all_episodes=[ReleaseEpisode(season=1, episode=1)]),
            "B": GroupReleases(all_episodes=[ReleaseEpisode(season=1, episode=1)]),
        }
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_different_coverage_separate(self) -> None:
        seadex = {
            "A": GroupReleases(all_episodes=[ReleaseEpisode(season=1, episode=1)]),
            "B": GroupReleases(all_episodes=[ReleaseEpisode(season=1, episode=2)]),
        }
        assert get_same_files_groups(seadex) == [["A"], ["B"]]


class TestEpisodeCoverage:
    """`episode_coverage` indexes each Sonarr-known episode's casefolded covering groups.

    A dict of one group short-circuits to the empty index. A group with a release
    the placement put on none of the title's episodes lands in `blanket`
    instead of a per-episode key.
    """

    def test_a_single_group_indexes_its_own_placed_episodes(self) -> None:
        # No sibling to excuse a mismatch, but a skipped torrent of the group reads its placed episodes here.
        seadex = {
            "A": GroupReleases(
                releases={"u": Release(episodes=[ReleaseEpisode(season=1, episode=1)])},
            ),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert result.blanket == frozenset()
        assert result.by_key == {EpisodeKey(1, 1): {"a"}}

    def test_records_episodes_sonarr_has(self) -> None:
        seadex = {
            "Era-Raws": GroupReleases(
                releases={"u": Release(episodes=[ReleaseEpisode(season=1, episode=1)])},
            ),
            "Other": GroupReleases(releases={"u2": Release(episodes=[])}),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert result.by_key[EpisodeKey(1, 1)] == {"era-raws"}
        # No placed episode -> the group blanket-covers every episode
        assert result.blanket == {"other"}

    def test_a_placed_episode_outside_the_index_never_blankets(self) -> None:
        # The placement only ever records the title's own episodes, so this input cannot
        # arise. The index guard still keeps a key outside the index out, and a release with placed
        # episodes never blankets: a placed episode is proof of what the release covers.
        seadex = {
            "A": GroupReleases(
                releases={"u": Release(episodes=[ReleaseEpisode(season=1, episode=99)])},
            ),
            "B": GroupReleases(
                releases={"u2": Release(episodes=[ReleaseEpisode(season=1, episode=1)])},
            ),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert EpisodeKey(1, 99) not in result.by_key
        assert result.by_key[EpisodeKey(1, 1)] == {"b"}
        assert result.blanket == frozenset()

    def test_blank_group_name_is_indexed_nowhere(self) -> None:
        # PIN: a group whose name normalizes to None (a blank name) must not
        # enter the blanket or any per-episode set. A file the arr has with no
        # group normalizes to None too, so indexing it would read every
        # no-group file as covered and silently suppress its grabs.
        seadex = {
            "": GroupReleases(
                releases={"u": Release(episodes=[]), "u2": Release(episodes=[ReleaseEpisode(1, 1)])},
            ),
            "B": GroupReleases(
                releases={"u3": Release(episodes=[ReleaseEpisode(season=1, episode=1)])},
            ),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert result.blanket == frozenset()
        assert result.by_key[EpisodeKey(1, 1)] == {"b"}
