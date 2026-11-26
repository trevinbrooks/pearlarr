# pyright: strict
"""Tests for the grab planner: which SeaDex releases a check marks to grab.

`plan` decides each release against what the arr has: nothing to grab where the arr has the group's copy at a listed
size, an upgrade where it has the group's files at other sizes, and a plain grab where it lacks them. Where several
groups cover the same files, one is grabbed, a public one over a private one, and a set with only private releases
is skipped with a notice.

The same-files tests call `reduce_overlapping_grabs` on marks already set: the marks, and which of them are upgrades,
are that step's whole input, and building each from the arr's files would bury it. Marks no movie can leave are built
through `plan` on a Sonarr title instead. A group built with `rg_group` and
no episodes is a movie's, as Radarr sends it. A group's private release lists files its public releases lack: one
whose files they cover is dropped before the planner sees it. The planner reads a release's sizes and placed
episodes, never its file names.
"""

from typing import Literal

import pytest

from pearlarr.aired_view import AiredSeries, aired_series
from pearlarr.arr_models import ArrFiles, SonarrEpisode
from pearlarr.config import Arr
from pearlarr.output import Severity
from pearlarr.pending_import import OwnedAtGrab, SizeMatchedEpisode
from pearlarr.planner import GrabPlan, PrivateOnlyNotice, PrivateOnlyNotices
from pearlarr.releases import (
    GroupReleases,
    PlacedEpisode,
    PlannedGrab,
    PrintedKey,
    Release,
    SeadexDict,
    SkipReason,
    releases_to_grab,
)
from pearlarr.tvdb_orders import AIRED_ORDER, OrderRow, SonarrSeriesOrders

from .builders import arr_files, make_planner, make_release, placed_episode, rg_group, sonarr_ep, stored_orders
from .fakes import debug_logger

type Mark = Literal["grab", "upgrade"]
"""How a release is marked to grab: a plain grab, or an upgrade that replaces the arr's files."""

_PRIVATE_ONLY = "private-only (private releases not supported)"
_OUTDATED = (
    "private-only; your copy is outdated (its file size no longer matches the release) and only a fallback covers it"
)

_S1E1 = placed_episode(season=1, episode=1, size=100)
_S1E2 = placed_episode(season=1, episode=2, size=200)
_S2E1 = placed_episode(season=2, episode=1, size=300)


def _grabbing_instead(public: str) -> str:
    """The notice's reason when group `public` is grabbed in place of a private-only set."""

    return f"private-only; grabbing public release {public} instead"


def _grabs(seadex: SeadexDict) -> dict[str, Mark]:
    """Each release still marked to grab, by its key, and how."""

    return {
        key: "upgrade" if release.is_upgrade else "grab"
        for group_releases in seadex.values()
        for key, release in group_releases.releases.items()
        if release.to_grab
    }


def _reduce(seadex: SeadexDict) -> PrivateOnlyNotices:
    """The skips of reducing `seadex` to one group per set of the same files, outside interactive mode."""

    return make_planner().reduce_overlapping_grabs(seadex)


def _sonarr_group(**releases: Release) -> GroupReleases:
    """A group of `releases` by key, with every release's placed episodes, as the placement leaves a Sonarr group."""

    return rg_group(releases, episodes=[ep for release in releases.values() for ep in release.episodes])


class TestReleasesToGrab:
    """`releases_to_grab` is the one grab-set filter both arrs' pending import update builders share."""

    def test_only_hash_carrying_grabs_are_planned(self) -> None:
        grabbed = make_release(download=True, infohash="h1")
        seadex = {
            "A": rg_group({"u1": grabbed, "u2": make_release(download=True, infohash=None)}),
            "B": rg_group({"u3": make_release(download=False, infohash="h3")}),
        }

        assert releases_to_grab(seadex) == [PlannedGrab("A", "u1", grabbed, "h1")]


class TestGroupsCoveringTheSameFiles:
    """Of the groups marked to grab that cover the same files, one is kept, and a set with only private ones is skipped."""

    def test_interactive_is_noop(self) -> None:
        # A hand-picked run already chose what to grab.
        seadex = {
            "A": rg_group({"u1": make_release(download=True)}),
            "B": rg_group({"u2": make_release(download=True)}),
        }

        make_planner(interactive=True).reduce_overlapping_grabs(seadex)

        assert _grabs(seadex) == {"u1": "grab", "u2": "grab"}

    def test_keeps_first_of_same_files(self) -> None:
        # Two groups of one movie cover the same files.
        seadex = {
            "A": rg_group({"u1": make_release(download=True)}),
            "B": rg_group({"u2": make_release(download=True)}),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"u1": "grab"}

    def test_a_public_group_is_kept_over_a_private_one(self) -> None:
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Pub": rg_group({"u2": make_release(download=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        # A public group kept over a private one says nothing: only a fallback kept over one gets a notice.
        assert skips == PrivateOnlyNotices()

    def test_private_only_skips_and_warns(self) -> None:
        seadex = {"Priv": rg_group({"u1": make_release(download=True, is_public=False)})}

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    def test_private_only_set_with_unrelated_fallback_still_warns(self) -> None:
        # The fallback covers another episode, so it excuses nothing: the private set warns and keeps the title
        # uncached while the fallback is grabbed.
        seadex = {
            "Priv": _sonarr_group(u1=make_release(download=True, is_public=False, size=[100], episodes=[_S1E1])),
            "Fall": _sonarr_group(u2=make_release(download=True, is_fallback=True, size=[200], episodes=[_S1E2])),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    @staticmethod
    def _plan_beside_a_copy_the_arr_has(group: str, *, is_public: bool, other_is_fallback: bool) -> GrabPlan:
        """The plan of `group`'s release on S01E01 and its private release that placed nothing, beside Other's on E01.

        The title's files have no group: E01's is at the size of `group`'s release there, E02's at a size no release
        lists. So the release that placed nothing is grabbed by group and size, while E01 holds `group`'s copy.
        """

        seadex = {
            group: _sonarr_group(
                u1=make_release(size=[100], episodes=[_S1E1], is_public=is_public),
                u2=make_release(size=[900], is_public=False),
            ),
            "Other": _sonarr_group(
                u3=make_release(
                    size=[150], episodes=[placed_episode(season=1, episode=1, size=150)], is_fallback=other_is_fallback
                )
            ),
        }
        title = [sonarr_ep(1, 1, size=100), sonarr_ep(1, 2, size=555)]
        return make_planner().plan(seadex, arr_files(title), title)

    @pytest.mark.parametrize(
        "is_fallback", [pytest.param(True, id="beside-a-fallback"), pytest.param(False, id="beside-a-public-group")]
    )
    def test_a_private_grab_with_no_upgrade_is_skipped_with_the_plain_warning(self, is_fallback: bool) -> None:
        # Nothing says the arr's copy is outdated, so no other group is grabbed in its place: the arr may have that
        # group's files already, as it does here.
        plan = self._plan_beside_a_copy_the_arr_has("Priv", is_public=False, other_is_fallback=is_fallback)

        assert _grabs(plan.seadex_dict) == {}
        assert plan.skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    def test_a_mixed_group_with_no_upgrade_is_kept(self) -> None:
        # The same for a mixed group: it is kept, and the add skips its private release with a warning.
        plan = self._plan_beside_a_copy_the_arr_has("Mixed", is_public=True, other_is_fallback=False)

        assert _grabs(plan.seadex_dict) == {"u2": "grab"}
        assert plan.skips == PrivateOnlyNotices()

    def test_a_fully_public_group_is_kept_over_a_mixed_one_before_it(self) -> None:
        # The mixed group comes first, but its private release would be skipped when added, losing what only it
        # covers: the fully public group is kept.
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": make_release(download=True, size=[100]),
                    "u2": make_release(download=True, is_public=False, size=[200]),
                }
            ),
            "Pub": rg_group({"u3": make_release(download=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u3": "grab"}
        assert skips == PrivateOnlyNotices()

    def test_a_group_whose_private_release_the_arr_has_is_kept_as_fully_public(self) -> None:
        # E01 and E02 have no file, and E03 has the file of Pub's private release at its listed size, so that release
        # isn't marked. Grabbing Pub then skips nothing and brings both missing episodes: it is kept over the mixed
        # group before it, whose private release would be skipped when added.
        seadex = {
            "Mixed": _sonarr_group(
                u1=make_release(size=[100], episodes=[_S1E1]),
                u2=make_release(size=[200], episodes=[_S1E2], is_public=False),
                u3=make_release(size=[300], episodes=[placed_episode(season=1, episode=3, size=300)]),
            ),
            "Pub": _sonarr_group(
                u4=make_release(
                    size=[110, 210],
                    episodes=[
                        placed_episode(season=1, episode=1, size=110),
                        placed_episode(season=1, episode=2, size=210),
                    ],
                ),
                u5=make_release(size=[310], episodes=[placed_episode(season=1, episode=3, size=310)], is_public=False),
            ),
        }
        title = [
            sonarr_ep(1, 1, episode_file_id=0),
            sonarr_ep(1, 2, episode_file_id=0),
            sonarr_ep(1, 3, size=310, release_group="Pub"),
        ]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u4": "grab"}
        assert plan.skips == PrivateOnlyNotices()

    def test_of_mixed_groups_only_the_first_is_kept(self) -> None:
        # With no fully public group the first is kept whole, and the add skips its private release with a warning.
        seadex = {
            "MixedA": rg_group(
                {
                    "u1": make_release(download=True, size=[100]),
                    "u2": make_release(download=True, is_public=False, size=[200]),
                }
            ),
            "MixedB": rg_group(
                {
                    "u3": make_release(download=True, size=[300]),
                    "u4": make_release(download=True, is_public=False, size=[400]),
                }
            ),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"u1": "grab", "u2": "grab"}

    def test_a_private_upgrade_beside_a_fallback_is_skipped_as_outdated(self) -> None:
        # The private grab is an upgrade, so the arr's copy of the preferred release is outdated, and a fallback never
        # replaces that copy: the set warns and is skipped rather than grabbing the fallback.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "Fall": rg_group({"u2": make_release(is_fallback=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _OUTDATED)], outdated=True
        )

    def test_a_fully_public_group_stands_in_for_a_private_upgrade_before_a_mixed_one(self) -> None:
        # Only public releases are grabbed in a private upgrade's place, so a mixed group would leave its private
        # release's files ungrabbed: the fully public group stands in.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "MixedPub": rg_group({"u2": make_release(is_public=False, size=[200]), "u3": make_release(size=[100])}),
            "Pub": rg_group({"u4": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u4": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], _grabbing_instead("Pub"), Severity.INFO)]
        )

    def test_a_public_group_not_marked_stands_in_for_a_private_upgrade(self) -> None:
        # A public group of the set not marked to grab stands in for a private upgrade, named as a public release.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "Pub": rg_group({"u2": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], _grabbing_instead("Pub"), Severity.INFO)]
        )

    def test_a_mixed_group_with_a_private_upgrade_is_kept_when_only_a_fallback_could_stand_in(self) -> None:
        # M's public release, owned and not marked, makes M a public group, and no group is fully public. The only
        # public group not marked to grab is a fallback, which never replaces M's outdated copy: M is kept, and the
        # outdated mark is set with no notice, since the add warns for the private release.
        seadex = {
            "M": rg_group(
                {
                    "u1": make_release(size=[100]),
                    "u2": make_release(download=True, is_public=False, upgrade=True, size=[200]),
                }
            ),
            "F": rg_group({"u3": make_release(is_fallback=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "upgrade"}
        assert skips == PrivateOnlyNotices(outdated=True)

    def test_a_public_group_stands_in_for_a_mixed_groups_private_upgrade(self) -> None:
        # The same mixed group, beside a public group that is no fallback: that group stands in.
        seadex = {
            "M": rg_group(
                {
                    "u1": make_release(size=[100]),
                    "u2": make_release(download=True, is_public=False, upgrade=True, size=[200]),
                }
            ),
            "P": rg_group({"u3": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u3": "grab"}
        reason = "remaining files private-only; grabbing public release P instead"
        assert skips == PrivateOnlyNotices(notices=[PrivateOnlyNotice(["M"], reason, Severity.INFO)])

    def test_a_fallback_kept_over_a_private_group_gets_a_notice(self) -> None:
        # A fallback kept over a private preferred release gets a notice naming it, and nothing is skipped.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Fall": rg_group({"u2": make_release(download=True, is_fallback=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], "private-only; falling back to Fall", Severity.INFO)]
        )

    def test_a_mixed_group_alone_with_a_private_upgrade_is_kept_whole(self) -> None:
        # Alone in its set with a private upgrade, the mixed group has no other public group to stand in: it is kept
        # whole, with no notice.
        seadex = {
            "M": rg_group(
                {
                    "u_pub": make_release(download=True, upgrade=True, size=[100]),
                    "u_priv": make_release(download=True, is_public=False, upgrade=True, size=[200]),
                }
            ),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u_pub": "upgrade", "u_priv": "upgrade"}
        assert skips == PrivateOnlyNotices()

    def test_a_mixed_group_stands_in_with_its_public_releases_only(self) -> None:
        # A mixed group is the only one that can stand in for the private upgrade: its public release is grabbed.
        seadex = {
            "P": rg_group({"p": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "M": rg_group({"m_pub": make_release(size=[100]), "m_priv": make_release(is_public=False, size=[200])}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"m_pub": "grab"}
        assert skips == PrivateOnlyNotices(notices=[PrivateOnlyNotice(["P"], _grabbing_instead("M"), Severity.INFO)])

    def test_groups_that_placed_nothing_are_each_kept(self) -> None:
        # Nothing shows what a group covers when none of its files placed on the title, so a duplicate grab is
        # better than dropping files the title lacks.
        seadex = {
            "A": _sonarr_group(u1=make_release(download=True)),
            "B": _sonarr_group(u2=make_release(download=True)),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"u1": "grab", "u2": "grab"}

    def test_a_private_group_the_arr_has_is_no_skip(self) -> None:
        # Nothing of the set is marked to grab, so there is nothing to warn about and the title can be cached.
        seadex = {"Priv": rg_group({"u1": make_release(is_public=False)})}

        assert _reduce(seadex) == PrivateOnlyNotices()

    def test_a_private_group_never_stands_in_for_a_private_upgrade(self) -> None:
        # Only a public group is grabbed in a private upgrade's place, so beside another private group the set warns.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "Other": rg_group({"u2": make_release(is_public=False)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    def test_an_outdated_skip_survives_a_later_plain_skip(self) -> None:
        # Two sets in one title: an outdated skip, then a plain one. Each adds its groups and its notice, and the
        # second leaves the first's outdated mark.
        seadex = {
            "Priv1": _sonarr_group(
                u1=make_release(download=True, is_public=False, upgrade=True, size=[100], episodes=[_S1E1])
            ),
            "Fall1": _sonarr_group(u2=make_release(is_fallback=True, size=[100], episodes=[_S1E1])),
            "Priv2": _sonarr_group(u3=make_release(download=True, is_public=False, size=[300], episodes=[_S2E1])),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True,
            groups=["Priv1", "Priv2"],
            notices=[PrivateOnlyNotice(["Priv1"], _OUTDATED), PrivateOnlyNotice(["Priv2"], _PRIVATE_ONLY)],
            outdated=True,
        )


class TestSameGroupDuplicateDedup:
    """Of one group's releases marked to grab, a cross-seed of the same files is dropped, told apart by file sizes."""

    @pytest.mark.parametrize(
        ("first", "second", "grabs"),
        [
            pytest.param(
                {"A - S01E01.mkv": 100},
                {"Extras/A - s01e01.mkv": 100},
                {"u1": "grab"},
                id="a-cross-seed-nesting-its-file-in-a-folder",
            ),
            pytest.param(
                {"S01/A - 01.mkv": 100},
                {"S02/A - 01.mkv": 200},
                {"u1": "grab", "u2": "grab"},
                id="the-same-names-at-other-sizes",
            ),
            pytest.param(
                {"A - 01.mkv": 100},
                {"S01/A - 01.mkv": 100, "S02/A - 01.mkv": 100},
                {"u1": "grab", "u2": "grab"},
                id="a-second-file-at-the-same-size",
            ),
            pytest.param({}, {}, {"u1": "grab", "u2": "grab"}, id="neither-lists-its-files"),
        ],
    )
    def test_only_a_cross_seed_of_the_same_files_is_dropped(
        self, first: dict[str, int], second: dict[str, int], grabs: dict[str, Mark]
    ) -> None:
        # Names prove nothing: a cross-seed can nest its file in a folder, and two cours can name their episodes alike.
        # A release that lists no files proves nothing either, so it is never dropped.
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=list(first), size=list(first.values()), download=True),
                    "u2": make_release(files=list(second), size=list(second.values()), download=True),
                }
            ),
        }

        _reduce(seadex)

        assert _grabs(seadex) == grabs

    def test_a_group_standing_in_grabs_one_of_its_cross_seeds(self) -> None:
        # Standing in for a private upgrade marks every public release of the group, two cross-seeds included.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True, size=[100])}),
            "Pub": rg_group({"u2": make_release(size=[100]), "u3": make_release(size=[100])}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], _grabbing_instead("Pub"), Severity.INFO)]
        )


class TestAMovieRelease:
    """A movie's release, decided by its group and its file sizes against Radarr's files."""

    @pytest.mark.parametrize(
        ("sizes", "files", "grabs"),
        [
            pytest.param([100], ArrFiles(), {"u1": "grab"}, id="no-file"),
            pytest.param(
                [100], ArrFiles(by_group={"OldRG": (100,)}), {"u1": "grab"}, id="a-group-seadex-does-not-list"
            ),
            pytest.param([100], ArrFiles(by_group={"EMBER": (100,)}), {}, id="its-group-at-a-listed-size"),
            pytest.param(
                [200], ArrFiles(by_group={"EMBER": (100,)}), {"u1": "upgrade"}, id="its-group-at-another-size"
            ),
            pytest.param(
                [100, 200], ArrFiles(by_group={"EMBER": (100, 999)}), {}, id="its-group-with-one-file-at-another-size"
            ),
            pytest.param(
                [], ArrFiles(by_group={"EMBER": (100,)}), {}, id="its-group-against-a-release-listing-no-files"
            ),
            pytest.param([100, 7], ArrFiles(no_group=(100,)), {}, id="no-group-at-a-listed-size-beside-extras"),
            pytest.param([100], ArrFiles(no_group=(50,)), {"u1": "grab"}, id="no-group-at-another-size"),
            pytest.param([100], ArrFiles(no_group=(100, 60)), {"u1": "grab"}, id="no-group-files-only-partly-listed"),
            pytest.param(
                [], ArrFiles(no_group=(50,)), {"u1": "grab"}, id="no-group-against-a-release-listing-no-files"
            ),
            pytest.param([0, 100], ArrFiles(no_group=(0,)), {"u1": "grab"}, id="an-empty-file-with-no-group"),
        ],
    )
    def test_a_movie_release_is_grabbed_unless_the_arr_has_its_copy(
        self, sizes: list[int], files: ArrFiles, grabs: dict[str, Mark]
    ) -> None:
        # A group matches whatever its case, and its copy is an upgrade only when none of its files is at a listed size.
        # Files with no group are the copy when the release lists all their sizes, since it carries extras the arr never
        # keeps. A release that lists no files, or an empty file, proves nothing.
        seadex = {"Ember": rg_group({"u1": make_release(size=sizes)})}

        plan = make_planner(arr=Arr.RADARR).plan(seadex, files)

        assert _grabs(plan.seadex_dict) == grabs

    @pytest.mark.parametrize("owner", [pytest.param("Ember", id="named"), pytest.param("", id="blank-name")])
    def test_movie_owned_copy_with_no_group_suppresses_the_sibling_grab(self, owner: str) -> None:
        # The file with no group is the owner's by size, so the arr has a preferred copy, as it would had the file kept
        # its group, and another group is not grabbed over it. A blank name matches nothing by name, but owns by size.
        seadex = {
            owner: rg_group({"u1": make_release(size=[100, 7])}),
            "RG": rg_group({"u2": make_release(size=[300])}),
        }

        plan = make_planner(arr=Arr.RADARR).plan(seadex, ArrFiles(no_group=(100,)))

        assert _grabs(plan.seadex_dict) == {}


class TestAReleaseThatPlacedNothing:
    """A Sonarr release the placement put on none of the title's episodes, decided by its group and sizes."""

    @pytest.mark.parametrize(
        ("title", "grabs"),
        [
            pytest.param(
                [sonarr_ep(1, 1, episode_file_id=0), sonarr_ep(1, 2, episode_file_id=0)], {"u1": "grab"}, id="no-file"
            ),
            pytest.param([sonarr_ep(1, 1, size=100), sonarr_ep(1, 2, size=200)], {}, id="no-group-at-its-sizes"),
        ],
    )
    def test_a_release_that_placed_nothing_is_grabbed_unless_the_arr_has_its_copy(
        self, title: list[SonarrEpisode], grabs: dict[str, Mark]
    ) -> None:
        # Its names carry no numbers, so no episode is ever compared: a missing copy is grabbed rather than read as
        # owned. A rename can strip the group from Sonarr's files, and files at its listed sizes are still its copy.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200]))}

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == grabs

    def test_nothing_placed_on_the_title_sibling_copy_no_grab(self) -> None:
        # The title's copy is another preferred group's, so the arr already has a release SeaDex prefers.
        seadex = {
            "RG": _sonarr_group(u1=make_release(size=[300])),
            "Other": _sonarr_group(u2=make_release(size=[100], episodes=[_S1E1])),
        }
        title = [sonarr_ep(1, 1, size=100, release_group="Other")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    @pytest.mark.parametrize(
        ("skip_reason", "grabs"),
        [
            pytest.param(None, {}, id="nothing-placed"),
            pytest.param(SkipReason.AIRED_ONLY, {"u2": "grab"}, id="no-row-for-its-episodes"),
        ],
    )
    def test_a_release_sonarr_has_no_row_for_covers_none_of_the_titles_episodes(
        self, skip_reason: SkipReason | None, grabs: dict[str, Mark]
    ) -> None:
        # The arr's E01 file is Left's at the size Left lists, and its E02 file is Other's. Left's release placed
        # nothing. With no reason it may hold any episode, so Other isn't grabbed over E01. Skipped because Sonarr has
        # no row for its episodes, it holds none of the title's, and Other is grabbed for E01.
        left = make_release(size=[500])
        left.skip_reason = skip_reason
        placed = [placed_episode(season=1, episode=1, size=200), placed_episode(season=1, episode=2, size=300)]
        seadex = {
            "Left": _sonarr_group(u1=left),
            "Other": _sonarr_group(u2=make_release(size=[200, 300], episodes=placed)),
        }
        title = [sonarr_ep(1, 1, size=500, release_group="Left"), sonarr_ep(1, 2, size=300, release_group="Other")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == grabs


class TestAReleaseByEpisode:
    """A Sonarr release decided episode by episode, against the file Sonarr has on each episode it placed on."""

    @pytest.mark.parametrize(
        ("held", "grabs"),
        [
            pytest.param(sonarr_ep(1, 1, episode_file_id=0), {"u1": "grab"}, id="no-file"),
            pytest.param(sonarr_ep(1, 1, size=100, release_group="Era-Raws"), {}, id="its-file"),
            pytest.param(
                sonarr_ep(1, 1, size=999, release_group="Era-Raws"), {"u1": "upgrade"}, id="its-group-at-another-size"
            ),
            pytest.param(
                sonarr_ep(1, 1, size=100, release_group="SubsPlease"), {"u1": "grab"}, id="a-group-seadex-does-not-list"
            ),
            pytest.param(sonarr_ep(1, 1, size=100), {}, id="no-group-at-its-size"),
            pytest.param(sonarr_ep(1, 1, size=555), {"u1": "grab"}, id="no-group-at-another-size"),
            pytest.param(sonarr_ep(1, 1, size=200), {"u1": "grab"}, id="no-group-at-the-size-listed-for-e02"),
        ],
    )
    def test_a_release_is_grabbed_unless_the_arr_has_its_file(
        self, held: SonarrEpisode, grabs: dict[str, Mark]
    ) -> None:
        # Each row is what the arr has on E01, beside E02's file with no group at the size listed there. A rename can
        # strip the group from Sonarr's file, and a file with no group is the release's copy only at the size listed on
        # that episode. Being its copy by size alone, E02's file never hides that the group's file on E01 is outdated.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200], episodes=[_S1E1, _S1E2]))}
        title = [held, sonarr_ep(1, 2, size=200)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == grabs

    def test_a_release_with_one_file_placed_is_compared_by_episode(self) -> None:
        # The release's other file placed nowhere on the title. Sonarr's copy is the group's at another size, an
        # upgrade, though the group and its sizes alone would find the other listed size there and read it as owned.
        placed = placed_episode(season=1, episode=1, size=999)
        seadex = {"RG": _sonarr_group(u1=make_release(size=[100, 999], episodes=[placed]))}
        title = [sonarr_ep(1, 1, size=100, release_group="RG")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_an_empty_file_with_no_group_is_no_copy(self) -> None:
        # An empty file proves nothing, even against a release that lists an empty file there: a failed copy stays
        # grabbable.
        placed = placed_episode(season=1, episode=1, size=0)
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[0], episodes=[placed]))}
        title = [sonarr_ep(1, 1, size=0)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "grab"}

    def test_episode_file_with_no_group_owns_the_episode_for_every_release(self) -> None:
        # Whose file it is doesn't depend on which release asks: the release whose size names it isn't grabbed, and
        # neither is another group's release of the same episode.
        other = make_release(size=[300], episodes=[placed_episode(season=1, episode=1, size=300)])
        seadex = {
            "Era-Raws": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1])),
            "Other": _sonarr_group(u2=other),
        }
        title = [sonarr_ep(1, 1, size=100)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_episode_blank_named_group_never_covers_a_file_with_no_group(self) -> None:
        # A blank group name is no name. A file with no group has none either, and the two must not read as a match
        # that leaves the real group's release ungrabbed.
        seadex = {
            "": _sonarr_group(u1=make_release(size=[500])),
            "Era-Raws": _sonarr_group(u2=make_release(size=[100], episodes=[_S1E1])),
        }
        title = [sonarr_ep(1, 1, size=999)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "grab", "u2": "grab"}


class TestDebugLines:
    """The lines a check logs at debug, naming each episode as Sonarr lists it."""

    def test_at_debug_an_episode_with_no_file_is_still_grabbed(self) -> None:
        # At debug the planner names the group of the file it compared, and an episode with no file has none.
        planner = make_planner(logger=debug_logger("pearlarr-planner-missing-file"))
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1]))}
        title = [sonarr_ep(1, 1, episode_file_id=0)]

        plan = planner.plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "grab"}

    def test_the_debug_lines_name_an_episode_by_the_number_sonarr_lists_it_under(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Sonarr lists the aired S02E01 as S01E04. Its file there has no group and is the release's listed size.
        in_sonarr = [sonarr_ep(1, 4, tvdb_id=9001, size=100)]
        stored = stored_orders({AIRED_ORDER: (OrderRow(2, 1, 9001, None),)})
        view = aired_series(in_sonarr, SonarrSeriesOrders(70, "dvd", None), stored)
        assert isinstance(view, AiredSeries)
        (row,) = view.in_sonarr
        placed = PlacedEpisode(2, 1, 100, printed=PrintedKey.of(row))
        planner = make_planner(logger=debug_logger("pearlarr-planner-printed-numbers"))
        seadex = {"RG": _sonarr_group(u1=make_release(size=[100], episodes=[placed]))}

        with caplog.at_level("DEBUG"):
            planner.plan(seadex_dict=seadex, arr_files=ArrFiles(), title_episodes=[row])

        # Two lines name the episode: the one that says whose file it is, and the one that compares it.
        messages = [r.message for r in caplog.records]
        assert sum("S01E04" in message for message in messages) == 2
        assert not any("S02E01" in message for message in messages)


class TestListedSizeOnTooManyEpisodes:
    """A listed size the group has on more episodes than the release places it on marks the release as an upgrade."""

    @staticmethod
    def _listing(*sizes: int, is_public: bool = True) -> dict[str, GroupReleases]:
        """One Era-Raws release of season 1 episodes 1..n at `sizes`, one placement per size given."""

        episodes = [placed_episode(season=1, episode=n, size=size) for n, size in enumerate(sizes, 1)]
        return {"Era-Raws": _sonarr_group(u1=make_release(size=list(sizes), episodes=episodes, is_public=is_public))}

    @staticmethod
    def _ours(episode: int, size: int) -> SonarrEpisode:
        """Season 1 `episode` with an Era-Raws file at `size`."""

        return sonarr_ep(1, episode, size=size, release_group="Era-Raws")

    def test_one_listed_file_on_two_episode_files_grabs(self) -> None:
        # File 2 imported onto both episodes: episode 1 lacks its own file. Sonarr's rows look the same whether one file
        # sits on both episodes or each holds a copy.
        title = [self._ours(1, 200), self._ours(2, 200)]

        plan = make_planner().plan(self._listing(100, 200), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_a_listed_size_placed_on_no_episode_but_held_grabs(self) -> None:
        # File 999 is listed but placed nowhere, so the arr having it on E02 is one too many.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200, 999], episodes=[_S1E1, _S1E2]))}
        title = [self._ours(1, 100), self._ours(2, 999)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_an_episode_two_listed_files_claim_at_different_sizes_is_left_out(self) -> None:
        # E01 is ambiguous, so size 200 counts once in the arr against its one placement on E02.
        episodes = [_S1E1, placed_episode(season=1, episode=1, size=150), _S1E2]
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 150, 200], episodes=episodes))}
        title = [self._ours(1, 200), self._ours(2, 200)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_one_listed_file_placed_on_two_episodes_and_held_on_both_does_not_grab(self) -> None:
        episodes = [placed_episode(season=1, episode=1, size=300), placed_episode(season=1, episode=2, size=300)]
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[300], episodes=episodes))}
        title = [self._ours(1, 300), self._ours(2, 300)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_size_listed_twice_held_twice_does_not_grab(self) -> None:
        title = [self._ours(1, 300), self._ours(2, 300)]

        plan = make_planner().plan(self._listing(300, 300), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_another_groups_files_at_a_listed_size_are_not_counted(self) -> None:
        # Other's release placed nowhere, so it may hold either episode: its files are not this group's to count.
        seadex = {**self._listing(100, 200), "Other": _sonarr_group(u2=make_release(size=[200, 200]))}
        title = [sonarr_ep(1, 1, size=200, release_group="Other"), sonarr_ep(1, 2, size=200, release_group="Other")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_size_matched_episode_counts(self) -> None:
        # E02's file with no group is this group's by its listed size, so size 200 sits on two episodes.
        title = [self._ours(1, 200), sonarr_ep(1, 2, size=200), self._ours(3, 300)]

        plan = make_planner().plan(self._listing(100, 200, 300), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_a_partial_swap_does_not_grab(self) -> None:
        # Episode 3 at its listed size means not every size differs, and a swap doubles no size.
        title = [self._ours(1, 200), self._ours(2, 100), self._ours(3, 300)]

        plan = make_planner().plan(self._listing(100, 200, 300), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_full_swap_marks_an_upgrade_not_a_plain_grab(self) -> None:
        # Every size differs from its placement, which marks an upgrade though no size sits on too many episodes.
        title = [self._ours(1, 200), self._ours(2, 100)]

        plan = make_planner().plan(self._listing(100, 200), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_a_doubled_unlisted_size_does_not_grab(self) -> None:
        # 999 sits on two episodes, but no release lists it, and E01 at its listed size means not every size differs.
        title = [self._ours(1, 100), self._ours(2, 999), self._ours(3, 999)]

        plan = make_planner().plan(self._listing(100, 200, 300), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_private_doubled_release_is_skipped_rather_than_replaced_by_a_fallback(self) -> None:
        # A fallback never replaces the arr's copy of the preferred release, so the set warns that it is outdated.
        public = make_release(size=[100, 200], episodes=[_S1E1, _S1E2], is_fallback=True)
        seadex = {**self._listing(100, 200, is_public=False), "Fall": _sonarr_group(u2=public)}
        title = [self._ours(1, 200), self._ours(2, 200)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}
        assert plan.skips == PrivateOnlyNotices(
            warned=True, groups=["Era-Raws"], notices=[PrivateOnlyNotice(["Era-Raws"], _OUTDATED)], outdated=True
        )


class TestGroupGuardFacts:
    """What the plan tells the import: which groups' copies the arr has, which are outdated, and whose files lack a group."""

    def test_a_group_with_one_file_at_an_unlisted_size_is_outdated_but_not_grabbed(self) -> None:
        # One episode at an unlisted size is no upgrade of the whole release, but the import must be free to replace
        # that file.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200], episodes=[_S1E1, _S1E2]))}
        title = [
            sonarr_ep(1, 1, size=100, release_group="Era-Raws"),
            sonarr_ep(1, 2, size=999, release_group="Era-Raws"),
        ]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == OwnedAtGrab(stale_groups=("Era-Raws",))

    def test_a_groups_file_on_an_episode_it_lists_nowhere_makes_it_outdated(self) -> None:
        # Y's copy is owned on the one episode Y lists, but the arr also has a Y file on an episode no Y release lists,
        # at a size none lists. Protecting Y would shield the very file X is grabbed to replace.
        x = make_release(
            size=[300, 400],
            episodes=[placed_episode(season=1, episode=1, size=300), placed_episode(season=1, episode=13, size=400)],
        )
        seadex = {"Y": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1])), "X": _sonarr_group(u2=x)}
        title = [sonarr_ep(1, 1, size=100, release_group="Y"), sonarr_ep(1, 13, size=999, release_group="Y")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u2": "grab"}
        assert plan.guards == OwnedAtGrab(entry_groups=("X",), stale_groups=("Y",))

    @pytest.mark.parametrize(
        ("listed", "guards"),
        [
            pytest.param([()], OwnedAtGrab(), id="its-only-release-lists-no-files"),
            pytest.param([(100,), ()], OwnedAtGrab(entry_groups=("Ember",)), id="beside-a-release-listing-the-size"),
        ],
    )
    def test_a_release_that_lists_no_files_says_nothing_of_its_groups_copy(
        self, listed: list[tuple[int, ...]], guards: OwnedAtGrab
    ) -> None:
        # The group's releases placed nothing, and each lists the sizes in `listed`. One that lists none neither
        # protects the arr's copy nor marks it outdated, and takes nothing from a sibling that lists its size.
        releases = {f"u{n}": make_release(size=list(sizes)) for n, sizes in enumerate(listed, 1)}
        seadex = {"Ember": _sonarr_group(**releases)}
        title = [sonarr_ep(1, 1, size=100, release_group="Ember")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == guards

    def test_groups_with_no_files_in_the_arr_stay_protected(self) -> None:
        # A group the arr has no file of is protected, so a copy of it imported while the grab waits is not overwritten.
        other = make_release(size=[300], episodes=[placed_episode(season=1, episode=1, size=300)])
        seadex = {"RG": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1])), "Other": _sonarr_group(u2=other)}
        title = [sonarr_ep(1, 1, episode_file_id=0)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert plan.guards == OwnedAtGrab(entry_groups=("RG", "Other"))

    @pytest.mark.parametrize(
        ("held", "owned_episodes"),
        [
            pytest.param(sonarr_ep(1, 1, size=100, ep_id=101), (SizeMatchedEpisode(101, 100),), id="no-group"),
            pytest.param(sonarr_ep(1, 1, size=100, release_group="Other", ep_id=101), (), id="another-groups-file"),
        ],
    )
    def test_the_plan_names_each_episode_whose_file_with_no_group_is_at_a_listed_size(
        self, held: SonarrEpisode, owned_episodes: tuple[SizeMatchedEpisode, ...]
    ) -> None:
        # The import checks each named file's size again before it trusts it. A file with a group is told by its
        # group, never by its size, so naming it would let it past the group's guard.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1]))}

        plan = make_planner().plan(seadex, arr_files([held]), [held])

        assert plan.guards.owned_episodes == owned_episodes


class TestDroppingAGroupNeverLosesAnEpisode:
    """A dropped group's public release is still grabbed when it brings an episode no kept release does."""

    @staticmethod
    def _mixed(name: str, placed: PlacedEpisode) -> GroupReleases:
        """Group `name`'s public release `{name}_pub` on `placed` and private `{name}_priv` on both, each to grab.

        The private release makes every such group cover S01E01 and S02E01, so they land in one set.
        """

        return _sonarr_group(
            **{
                f"{name}_pub": make_release(download=True, size=[placed.size], episodes=[placed]),
                f"{name}_priv": make_release(
                    download=True, is_public=False, size=[_S1E1.size, _S2E1.size], episodes=[_S1E1, _S2E1]
                ),
            }
        )

    @pytest.mark.parametrize(
        ("order", "grabs"),
        [
            pytest.param("AB", {"a_pub": "grab", "a_priv": "grab", "b_pub": "grab"}, id="a-kept"),
            pytest.param("BA", {"b_pub": "grab", "b_priv": "grab", "a_pub": "grab"}, id="b-kept"),
        ],
    )
    def test_a_dropped_groups_public_release_is_grabbed_for_the_episode_it_alone_brings(
        self, order: str, grabs: dict[str, Mark]
    ) -> None:
        # Neither group is fully public, so the first is kept, and the other's public release still brings the one
        # episode it alone has.
        groups = {"A": self._mixed("a", _S1E1), "B": self._mixed("b", _S2E1)}
        seadex = {name: groups[name] for name in order}

        _reduce(seadex)

        assert _grabs(seadex) == grabs

    def test_a_dropped_release_bringing_fewer_episodes_stays_dropped(self) -> None:
        # The dropped group's public release brings nothing the kept group doesn't: grabbing it would be a duplicate.
        seadex = {
            "A": _sonarr_group(a_pub=make_release(download=True, size=[100, 300], episodes=[_S1E1, _S2E1])),
            "B": self._mixed("b", _S1E1),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"a_pub": "grab"}

    def test_a_dropped_release_bringing_the_same_episodes_stays_dropped(self) -> None:
        # The same when it brings exactly what the kept group does.
        seadex = {
            "A": _sonarr_group(a_pub=make_release(download=True, size=[100], episodes=[_S1E1])),
            "B": _sonarr_group(b_pub=make_release(download=True, size=[100], episodes=[_S1E1])),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"a_pub": "grab"}

    def test_each_dropped_release_is_judged_against_every_release_grabbed_before_it(self) -> None:
        # A is kept with S01E01. B's public release alone brings S02E01, so it is grabbed. C's brings S01E01, which
        # A already does, and stays dropped.
        seadex = {"A": self._mixed("a", _S1E1), "B": self._mixed("b", _S2E1), "C": self._mixed("c", _S1E1)}

        _reduce(seadex)

        assert _grabs(seadex) == {"a_pub": "grab", "a_priv": "grab", "b_pub": "grab"}

    def test_a_dropped_groups_public_release_the_arr_has_stays_ungrabbed(self) -> None:
        # B's public release is the arr's already, so it was never marked to grab, and dropping B leaves it that way
        # though no kept public release covers its episode.
        seadex = {
            "A": self._mixed("a", _S1E1),
            "B": _sonarr_group(
                b_owned=make_release(size=[300], episodes=[_S2E1]),
                b_priv=make_release(download=True, is_public=False, size=[100, 300], episodes=[_S1E1, _S2E1]),
            ),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"a_pub": "grab", "a_priv": "grab"}


class TestTorrentLeftForAHandImport:
    """A torrent left for a hand import is decided by what its window has, never by its numbers."""

    _SIZES = (100, 200, 300)

    @classmethod
    def _seadex(
        cls, skip_reason: SkipReason = SkipReason.MISNUMBERED, **siblings: GroupReleases
    ) -> dict[str, GroupReleases]:
        """The torrent left for a hand import for `skip_reason`, three files at `_SIZES` placed nowhere, beside `siblings`."""

        item = make_release(size=list(cls._SIZES))
        item.skip_reason = skip_reason
        return {"Pack": _sonarr_group(u1=item), **siblings}

    @staticmethod
    def _ours(episode: int, size: int) -> SonarrEpisode:
        """Special `episode` with the torrent's file at `size`."""

        return sonarr_ep(0, episode, size=size, release_group="Pack")

    def test_every_special_holding_a_torrent_file_is_in_place(self) -> None:
        title = [self._ours(n, size) for n, size in enumerate(self._SIZES, 1)]

        plan = make_planner().plan(self._seadex(), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    @pytest.mark.parametrize(
        "skip_reason",
        [
            pytest.param(SkipReason.MISNUMBERED, id="misnumbered"),
            pytest.param(SkipReason.EPISODE_TITLES_CONTRADICT, id="episode-titles-contradict"),
        ],
    )
    def test_an_empty_special_marks_the_torrent_to_grab_as_an_upgrade(self, skip_reason: SkipReason) -> None:
        title = [self._ours(1, 100), self._ours(2, 200), sonarr_ep(0, 3, episode_file_id=0)]

        plan = make_planner().plan(self._seadex(skip_reason), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_files_of_an_unlisted_group_mark_the_torrent_to_grab(self) -> None:
        title = [sonarr_ep(0, n, size=size, release_group="Other") for n, size in enumerate(self._SIZES, 1)]

        plan = make_planner().plan(self._seadex(), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "grab"}

    def test_a_special_a_sibling_release_covers_is_in_place(self) -> None:
        sibling = _sonarr_group(u2=make_release(size=[999], episodes=[placed_episode(season=0, episode=3, size=999)]))
        title = [self._ours(1, 100), self._ours(2, 200), sonarr_ep(0, 3, size=999, release_group="Sib")]

        plan = make_planner().plan(self._seadex(Sib=sibling), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_file_with_no_group_at_a_listed_size_is_the_torrents(self) -> None:
        # A hand import whose names carry no group: the listed size still says whose file each special has.
        title = [sonarr_ep(0, n, size=size) for n, size in enumerate(self._SIZES, 1)]

        plan = make_planner().plan(self._seadex(), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_special_another_release_of_the_group_placed_is_in_place(self) -> None:
        # The group's single-file release covers the third special, so the torrent has nothing left to cover there.
        covered = placed_episode(season=0, episode=3, size=999)
        seadex = self._seadex()
        seadex["Pack"] = _sonarr_group(**seadex["Pack"].releases, u2=make_release(size=[999], episodes=[covered]))
        title = [self._ours(1, 100), self._ours(2, 200), self._ours(3, 999)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}

    def test_a_torrent_file_at_an_unlisted_size_marks_the_torrent_to_grab(self) -> None:
        title = [self._ours(1, 150), self._ours(2, 200), self._ours(3, 300)]

        plan = make_planner().plan(self._seadex(), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_a_file_with_no_group_at_an_unlisted_size_is_not_the_torrents(self) -> None:
        # The plan claims for the torrent no file it can't prove is the torrent's, so the import may replace it.
        title = [self._ours(1, 100), self._ours(2, 200), sonarr_ep(0, 3, size=150, ep_id=103)]

        plan = make_planner().plan(self._seadex(), arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}
        assert plan.guards == OwnedAtGrab(entry_groups=("Pack",))
