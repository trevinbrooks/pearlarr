# pyright: strict
"""The pure release-matching helpers the planner uses."""

import pytest

from pearlarr.coverage import format_episode_ranges
from pearlarr.planner import (
    EpisodeCoverage,
    episode_coverage,
    get_episode_keys,
    get_same_files_groups,
    normalize_rg,
)
from pearlarr.seadex_types import (
    EpisodeKey,
    EpisodeRecord,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
)

from .builders import sonarr_ep


class TestNormalizeRg:
    """`normalize_rg` casefolds and strips whitespace/leading-trailing dashes. Blank or `None` in yields `None`."""

    def test_none_and_blank_return_none(self) -> None:
        assert normalize_rg(None) is None
        assert normalize_rg("") is None

    def test_strips_whitespace_dashes_and_casefolds(self) -> None:
        assert normalize_rg("  Era-Raws-  ") == "era-raws"
        assert normalize_rg("-SubsPlease-") == "subsplease"


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
    """`get_episode_keys` builds a `(season, episode)` set from episode records, missing fields becoming `(None, None)`."""

    def test_builds_season_episode_pairs(self) -> None:
        eps = [EpisodeRecord(season=1, episode=1), EpisodeRecord(season=1, episode=2)]
        assert get_episode_keys(eps) == {(1, 1), (1, 2)}

    def test_missing_keys_become_none(self) -> None:
        assert get_episode_keys([EpisodeRecord()]) == {(None, None)}


class TestGetSameFilesGroups:
    """`get_same_files_groups` groups release names by identical episode coverage.

    Unset (`None`) coverage collapses every group into one. An empty list keeps
    each group apart (coverage unverifiable). Matching/differing coverage sets
    group or separate accordingly.
    """

    def test_no_episode_parsing_groups_together(self) -> None:
        seadex = {"A": SeadexReleaseGroupItem(), "B": SeadexReleaseGroupItem()}
        assert get_same_files_groups(seadex) == [["A", "B"]]

    def test_unparsed_each_on_its_own(self) -> None:
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


class TestEpisodeCoverage:
    """`episode_coverage` indexes each Sonarr-known episode's casefolded covering groups.

    A dict of one group short-circuits to the empty index. A group with a url
    the placement put on none of the entry's episodes lands in `blanket`
    instead of a per-episode key.
    """

    def test_single_group_short_circuits(self) -> None:
        # A single group has no sibling coverage to consult
        seadex = {
            "A": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
        }
        assert episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)}) == EpisodeCoverage(frozenset(), {})

    def test_records_episodes_sonarr_has(self) -> None:
        seadex = {
            "Era-Raws": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
            "Other": SeadexReleaseGroupItem(urls={"u2": SeadexUrlItem(episodes=[])}),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert result.by_key[EpisodeKey(1, 1)] == {"era-raws"}
        # No placed record -> the group blanket-covers every episode
        assert result.blanket == {"other"}

    def test_a_record_outside_the_index_never_blankets(self) -> None:
        # The placement only ever records the entry's own episodes, so this input cannot
        # arise. The index guard still keeps a foreign key out, and a url with records
        # never blankets: a record is proof of what the url covers.
        seadex = {
            "A": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=99)])},
            ),
            "B": SeadexReleaseGroupItem(
                urls={"u2": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert EpisodeKey(1, 99) not in result.by_key
        assert result.by_key[EpisodeKey(1, 1)] == {"b"}
        assert result.blanket == frozenset()

    def test_blank_group_name_is_indexed_nowhere(self) -> None:
        # PIN: a group whose name normalizes to None (a blank name) must not
        # enter the blanket or any per-episode set. An untagged on-disk file's
        # group also normalizes to None, so indexing it would read every
        # untagged file as covered and silently suppress its grabs.
        seadex = {
            "": SeadexReleaseGroupItem(
                urls={"u": SeadexUrlItem(episodes=[]), "u2": SeadexUrlItem(episodes=[EpisodeRecord(1, 1)])},
            ),
            "B": SeadexReleaseGroupItem(
                urls={"u3": SeadexUrlItem(episodes=[EpisodeRecord(season=1, episode=1)])},
            ),
        }
        result = episode_coverage(seadex, {EpisodeKey(1, 1): sonarr_ep(1, 1)})
        assert result.blanket == frozenset()
        assert result.by_key[EpisodeKey(1, 1)] == {"b"}
