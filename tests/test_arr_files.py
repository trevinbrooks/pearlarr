# pyright: strict
"""Tests for the files each arr already holds for a title, as the grab planner reads them.

They cover Radarr's and Sonarr's readers and the `ArrFiles` type both return.
"""

from pearlarr.arr_models import ArrFiles, MovieFile

from .builders import make_radarr_sync, make_sonarr_episodes, sonarr_ep
from .fakes import FakeRadarrClient


class TestRadarrFiles:
    """radarr_files never hard-errors: a size it can't read counts as zero."""

    def test_unreadable_size_with_no_group_converts_to_zero(self) -> None:
        # An unreadable size with no group converts to 0, which disqualifies the whole
        # ownership multiset (fail-closed, like the planner's size matching of Sonarr files with no group): the
        # readable sizes alone must not let a release claim the copy.
        radarr = FakeRadarrClient(
            movie_files=[MovieFile(release_group=None, size=500), MovieFile(release_group=None, size=None)]
        )
        strat = make_radarr_sync(radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(no_group=(500, 0))


class TestSonarrFiles:
    """sonarr_files feeds the planner no sizes for files with no group."""

    def test_episode_files_with_no_group_never_reach_no_group(self) -> None:
        # PIN: an episode file with no group contributes NOTHING to the `ArrFiles` - it
        # travels on the title's episodes into the planner's size matching instead. The
        # planner merges both arrs' no-group sources, so a file fed through
        # both would count twice and could then never be owned.
        episodes = make_sonarr_episodes()

        files = episodes.sonarr_files(
            [
                sonarr_ep(1, 1, ep_id=101, release_group=None, size=700),
                sonarr_ep(1, 2, ep_id=102, release_group="RG", size=800),
            ]
        )

        assert files == ArrFiles(by_group={"RG": (800,)})


class TestArrFiles:
    """The type's own contract: hashable, one shared conversion, one label for its groups."""

    def test_equal_arr_files_hash_equal_across_key_order(self) -> None:
        # dict eq ignores insertion order, so the hash must too.
        first = ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,))
        second = ArrFiles(by_group={"B": (2,), "A": (1,)}, no_group=(3,))

        assert first == second
        assert hash(first) == hash(second)

    def test_from_files_keeps_or_drops_no_group_by_arr(self) -> None:
        # One conversion for both arrs: only the no-group destination differs. The
        # unreadable size with a group is dropped. The group keeps its name.
        files = [
            MovieFile(release_group=None, size=700),
            MovieFile(release_group="RG", size=800),
            MovieFile(release_group="RG", size=None),
        ]

        assert ArrFiles.from_files(files, keep_no_group=True) == ArrFiles(by_group={"RG": (800,)}, no_group=(700,))
        assert ArrFiles.from_files(files, keep_no_group=False) == ArrFiles(by_group={"RG": (800,)})

    def test_groups_label_renders_no_group_and_empty(self) -> None:
        assert ArrFiles(by_group={"A": (1,)}, no_group=(2,)).groups_label() == "A, (none)"
        assert ArrFiles().groups_label() == "(no files)"

    def test_group_count_matches_the_rendered_entries(self) -> None:
        # Files with no group convert to the single `(none)` entry, so they count once.
        assert ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,)).group_count() == 3
        assert ArrFiles().group_count() == 0
