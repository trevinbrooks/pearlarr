# pyright: strict
"""Tests for the grab planner: which SeaDex releases a check marks to grab.

`plan` decides each release against what the arr has: nothing to grab where the arr has the group's copy at a listed
size, an upgrade where it has the group's files at other sizes, and a plain grab where it lacks them. Where several
groups cover the same files, one is grabbed, a public one over a private one, and a set with only private releases
is skipped with a notice.

The same-files tests call `reduce_overlapping_grabs` on marks already set: the marks, and which of them are upgrades,
are that step's whole input, and building each from the arr's files would bury it. A group built with `rg_group` and
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
from pearlarr.planner import GrabPlan, GrabPlanner, PrivateOnlyNotice, PrivateOnlyNotices
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


class TestAnyToGrab:
    """`any_to_grab` reports whether any release in the group dict has `to_grab` set."""

    def test_false_when_none_is_to_grab(self) -> None:
        seadex = {"A": rg_group({"u": make_release(download=False)})}
        assert GrabPlanner.any_to_grab(seadex) is False

    def test_true_when_one_is_to_grab(self) -> None:
        seadex = {"A": rg_group({"u": make_release(download=True)})}
        assert GrabPlanner.any_to_grab(seadex) is True


class TestReleasesToGrab:
    """`releases_to_grab` is the one grab-set filter both arrs' pending import update builders share."""

    def test_only_hash_carrying_grabs_are_planned(self) -> None:
        grabbed = make_release(url="u1", download=True, infohash="h1")
        seadex = {
            "A": rg_group({"u1": grabbed, "u2": make_release(url="u2", download=True, infohash=None)}),
            "B": rg_group({"u3": make_release(url="u3", download=False, infohash="h3")}),
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

    def test_prefers_public_keeper(self) -> None:
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
            "Priv": _sonarr_group(u1=make_release(download=True, is_public=False, episodes=[_S1E1])),
            "Fall": _sonarr_group(u2=make_release(download=True, is_fallback=True, episodes=[_S1E2])),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    def test_private_only_set_with_a_fallback_not_to_grab_warns_plainly(self) -> None:
        # The private grab is no upgrade, so nothing says the arr has an outdated copy: the plain warning.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Fall": rg_group({"u2": make_release(is_fallback=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _PRIVATE_ONLY)]
        )

    def test_keeper_prefers_fully_addable_group(self) -> None:
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

    def test_keeper_degrades_to_first_when_all_mixed(self) -> None:
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

    def test_private_only_set_with_fallback_skips_on_size_mismatch(self) -> None:
        # The private grab is an upgrade, so the arr's copy of the preferred release is outdated, and a fallback never
        # replaces that copy: the set warns and is skipped rather than grabbing the fallback.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Fall": rg_group({"u2": make_release(is_fallback=True)}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {}
        assert skips == PrivateOnlyNotices(
            warned=True, groups=["Priv"], notices=[PrivateOnlyNotice(["Priv"], _OUTDATED)], outdated=True
        )

    def test_promotion_prefers_fully_public_group(self) -> None:
        # Only public releases are grabbed in a private upgrade's place, so a mixed group would leave its private
        # release's files ungrabbed: the fully public group stands in.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "MixedPub": rg_group({"u2": make_release(is_public=False, size=[200]), "u3": make_release(size=[100])}),
            "Pub": rg_group({"u4": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u4": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], "private-only; grabbing public release Pub instead", Severity.INFO)]
        )

    def test_promotion_considers_preferred_public_group(self) -> None:
        # A public group of the set not marked to grab stands in for a private upgrade, named as a public release.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Pub": rg_group({"u2": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], "private-only; grabbing public release Pub instead", Severity.INFO)]
        )

    def test_degraded_keeper_skips_when_only_a_fallback_could_stand_in(self) -> None:
        # M's public release, owned and not marked, makes M a public group, and no group is fully addable. The only
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

    def test_degraded_keeper_promotes_preferred_public_on_size_mismatch(self) -> None:
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

    def test_degraded_keeper_without_mismatch_does_not_promote(self) -> None:
        # With no upgrade the grabs may be of files the arr lacks, and the public group not marked may be owned, so
        # nothing stands in and the mixed group is kept.
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": make_release(download=True, size=[100]),
                    "u2": make_release(download=True, is_public=False, size=[200]),
                }
            ),
            "P": rg_group({"u3": make_release()}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u1": "grab", "u2": "grab"}
        assert skips == PrivateOnlyNotices()

    def test_fallback_keeper_over_private_notices(self) -> None:
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

    def test_mixed_group_to_grab_never_promotes_itself(self) -> None:
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

    def test_promotes_mixed_group_not_to_grab_flipping_only_public_releases(self) -> None:
        # A mixed group is the only one that can stand in for the private upgrade: its public release is grabbed.
        seadex = {
            "P": rg_group({"p": make_release(download=True, is_public=False, upgrade=True)}),
            "M": rg_group({"m_pub": make_release(size=[100]), "m_priv": make_release(is_public=False, size=[200])}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"m_pub": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["P"], "private-only; grabbing public release M instead", Severity.INFO)]
        )

    def test_different_files_both_kept(self) -> None:
        seadex = {
            "A": _sonarr_group(u1=make_release(download=True, episodes=[_S1E1])),
            "B": _sonarr_group(u2=make_release(download=True, episodes=[_S1E2])),
        }

        _reduce(seadex)

        assert _grabs(seadex) == {"u1": "grab", "u2": "grab"}

    def test_groups_that_placed_nothing_are_each_kept(self) -> None:
        # Nothing shows what a group covers when none of its files placed on the title, so a duplicate grab is
        # better than dropping files the title lacks.
        seadex = {
            "A": _sonarr_group(u1=make_release(download=True, size=[100])),
            "B": _sonarr_group(u2=make_release(download=True, size=[200])),
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
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
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
            "Priv1": _sonarr_group(u1=make_release(download=True, is_public=False, upgrade=True, episodes=[_S1E1])),
            "Fall1": _sonarr_group(u2=make_release(is_fallback=True, episodes=[_S1E1])),
            "Priv2": _sonarr_group(u3=make_release(download=True, is_public=False, episodes=[_S2E1])),
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

    def test_promoted_duplicates_dedup(self) -> None:
        # Standing in for a private upgrade marks every public release of the group, two cross-seeds included.
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Pub": rg_group({"u2": make_release(size=[100]), "u3": make_release(size=[100])}),
        }

        skips = _reduce(seadex)

        assert _grabs(seadex) == {"u2": "grab"}
        assert skips == PrivateOnlyNotices(
            notices=[PrivateOnlyNotice(["Priv"], "private-only; grabbing public release Pub instead", Severity.INFO)]
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
        # A group matches whatever its case. Files with no group are the release's copy when it lists every one of
        # their sizes: a release carries files the arr never keeps (subtitles, extras), and an empty file proves nothing.
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

    def test_matching_group_radarr_none_size_grabs(self) -> None:
        # What happens today, not a decision: a file whose size Radarr couldn't read leaves its group with no size,
        # which reads as every listed size differing, so the release is grabbed as an upgrade. A release that lists
        # no sizes is the other way round: it keeps the arr's copy.
        seadex = {"RG": rg_group({"u1": make_release(size=[100])})}

        plan = make_planner(arr=Arr.RADARR).plan(seadex, ArrFiles(by_group={"RG": ()}))

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}


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
        ("skip_reason", "grabbed"),
        [
            pytest.param(None, False, id="nothing-placed"),
            pytest.param(SkipReason.AIRED_ONLY, True, id="no-row-for-its-episodes"),
        ],
    )
    def test_a_release_sonarr_has_no_row_for_covers_none_of_the_titles_episodes(
        self, skip_reason: SkipReason | None, grabbed: bool
    ) -> None:
        # The arr's E01 file is Left's, from some other release of the group, and its E02 file is Other's. Left's
        # listed release placed nothing. With no reason it may hold any episode, so Other isn't grabbed over E01.
        # Skipped because Sonarr has no row for its episodes, it holds none of the title's, and Other is grabbed.
        planner = make_planner()
        left = make_release(episodes=[], size=[500], infohash="h1")
        left.skip_reason = skip_reason
        placed = [placed_episode(season=1, episode=1, size=200), placed_episode(season=1, episode=2, size=300)]
        other = make_release(url="https://nyaa.si/view/2", episodes=placed, size=[200, 300], infohash="h2")
        seadex = {"Left": rg_group({"u1": left}), "Other": rg_group({"u2": other}, episodes=placed)}
        result = planner.plan(
            seadex_dict=seadex,
            arr_files=ArrFiles(by_group={"Left": (100,), "Other": (300,)}),
            title_episodes=[
                sonarr_ep(1, 1, size=100, release_group="Left"),
                sonarr_ep(1, 2, size=300, release_group="Other"),
            ],
        )
        assert result.seadex_dict["Other"].releases["u2"].to_grab is grabbed


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
        ],
    )
    def test_a_release_is_grabbed_unless_the_arr_has_its_file(
        self, held: SonarrEpisode, grabs: dict[str, Mark]
    ) -> None:
        # A rename can strip the group from Sonarr's file, and a file at the size the release lists there is still
        # its copy.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1]))}

        plan = make_planner().plan(seadex, arr_files([held]), [held])

        assert _grabs(plan.seadex_dict) == grabs

    def test_a_release_with_one_file_placed_is_compared_by_episode(self) -> None:
        # The release's other file placed nowhere on the title. Sonarr's copy is the group's at another size, an
        # upgrade, though the group and its sizes alone would find the other listed size there and read it as owned.
        placed = placed_episode(season=1, episode=1, size=999)
        seadex = {"RG": _sonarr_group(u1=make_release(size=[100, 999], episodes=[placed]))}
        title = [sonarr_ep(1, 1, size=100, release_group="RG")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

    def test_episode_groupless_zero_size_still_grabs(self) -> None:
        # An empty file proves nothing, even against a release that lists an empty file there: a failed copy stays
        # grabbable.
        placed = placed_episode(season=1, episode=1, size=0)
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[0], episodes=[placed]))}
        title = [sonarr_ep(1, 1, size=0)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "grab"}

    def test_episode_no_group_exact_size_never_masks_an_outdated_release(self) -> None:
        # A file with no group is the release's by its size, so it says nothing about whether the release is outdated.
        # Counting it would hide that the group's file on the other episode is at an unlisted size.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200], episodes=[_S1E1, _S1E2]))}
        title = [sonarr_ep(1, 1, size=100), sonarr_ep(1, 2, size=999, release_group="Era-Raws")]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {"u1": "upgrade"}

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

    def test_debug_logging_path_does_not_crash(self) -> None:
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
        seadex = {"RG": rg_group({"u1": make_release(episodes=[placed], infohash="h1")})}

        with caplog.at_level("DEBUG"):
            planner.plan(seadex_dict=seadex, arr_files=ArrFiles(), title_episodes=[row])

        assert [r.message for r in caplog.records if "S0" in r.message] == [
            "Sonarr file for S01E04 has no release group but matches rg's listed size exactly - reading it as that release",
            "Found SeaDex match to Sonarr for S01E04.",
        ]


class TestListedSizeOnTooManyEpisodes:
    """A listed size the group has on more episodes than the release places it on marks the release as an upgrade."""

    @staticmethod
    def _listing(*sizes: int, is_public: bool = True) -> dict[str, GroupReleases]:
        """One Era-Raws release of season 1 episodes 1..n at `sizes`, one placement per size given."""

        episodes = [placed_episode(season=1, episode=n, size=size) for n, size in enumerate(sizes, 1)]
        item = make_release(size=list(sizes), episodes=episodes, infohash="h1", is_public=is_public)
        return {"Era-Raws": rg_group({"u1": item})}

    @staticmethod
    def _plan(seadex: dict[str, GroupReleases], title_episodes: list[SonarrEpisode]) -> GrabPlan:
        """The planner's result for `seadex` against `title_episodes`, every file with a group reported to the arr."""

        return make_planner().plan(seadex, arr_files(title_episodes), title_episodes)

    @classmethod
    def _decide(cls, seadex: dict[str, GroupReleases], title_episodes: list[SonarrEpisode]) -> Release:
        """The Era-Raws release after the planner decides it against `title_episodes`."""

        return cls._plan(seadex, title_episodes).seadex_dict["Era-Raws"].releases["u1"]

    @staticmethod
    def _ours(episode: int, size: int, file_id: int) -> SonarrEpisode:
        """Season 1 `episode` with Era-Raws' file `file_id` at `size`."""

        return sonarr_ep(1, episode, size=size, release_group="Era-Raws", episode_file_id=file_id)

    def test_one_listed_file_on_two_episode_files_grabs(self) -> None:
        # File 2 imported onto both episodes: episode 1 lacks its own file.
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_one_episode_file_covering_two_episodes_at_a_size_listed_once_grabs(self) -> None:
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 200, 11)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_twelve_file_torrent_with_one_file_on_two_episodes_grabs(self) -> None:
        arr_episodes = [self._ours(n, 700 if n == 2 else 100 * n, 10 + n) for n in range(1, 13)]

        item = self._decide(self._listing(*range(100, 1300, 100)), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_listed_size_placed_on_no_episode_but_held_grabs(self) -> None:
        # File 999 is listed but placed nowhere, so the arr having it on E02 is one too many.
        episodes = [placed_episode(season=1, episode=1, size=100), placed_episode(season=1, episode=2, size=200)]
        seadex = {"Era-Raws": rg_group({"u1": make_release(size=[100, 200, 999], episodes=episodes, infohash="h1")})}

        item = self._decide(seadex, [self._ours(1, 100, 11), self._ours(2, 999, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_an_episode_two_listed_files_claim_at_different_sizes_is_left_out(self) -> None:
        # E01 is ambiguous, so size 200 counts once in the arr against its one placement on E02.
        episodes = [
            placed_episode(season=1, episode=1, size=100),
            placed_episode(season=1, episode=1, size=150),
            placed_episode(season=1, episode=2, size=200),
        ]
        seadex = {"Era-Raws": rg_group({"u1": make_release(size=[100, 150, 200], episodes=episodes, infohash="h1")})}

        item = self._decide(seadex, [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert item.to_grab is False

    def test_a_two_episode_listed_file_held_once_under_both_keys_does_not_grab(self) -> None:
        item = self._decide(self._listing(300, 300), [self._ours(1, 300, 11), self._ours(2, 300, 11)])

        assert item.to_grab is False

    def test_a_size_listed_twice_held_twice_does_not_grab(self) -> None:
        item = self._decide(self._listing(300, 300), [self._ours(1, 300, 11), self._ours(2, 300, 12)])

        assert item.to_grab is False

    def test_a_blanket_covered_group_at_a_listed_size_is_not_counted(self) -> None:
        # Other's release placed nowhere, so it blankets both episodes: its files are not this group's to count.
        seadex = {
            **self._listing(100, 200),
            "Other": rg_group({"u2": make_release(url="https://nyaa.si/view/2", infohash="h2")}),
        }
        item = self._decide(
            seadex,
            [
                sonarr_ep(1, 1, size=200, release_group="Other", episode_file_id=11),
                sonarr_ep(1, 2, size=200, release_group="Other", episode_file_id=12),
            ],
        )

        assert item.to_grab is False

    def test_a_size_matched_episode_counts(self) -> None:
        # E02's file with no group is this group's by its listed size, so size 200 sits on two episodes.
        no_group = sonarr_ep(1, 2, size=200, release_group=None, episode_file_id=12)

        item = self._decide(self._listing(100, 200, 300), [self._ours(1, 200, 11), no_group, self._ours(3, 300, 13)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_partial_swap_does_not_grab(self) -> None:
        # Episode 3 at its listed size keeps the all-sizes-differ check out of it, and a swap doubles no size.
        arr_episodes = [self._ours(1, 200, 11), self._ours(2, 100, 12), self._ours(3, 300, 13)]

        item = self._decide(self._listing(100, 200, 300), arr_episodes)

        assert item.to_grab is False

    def test_a_full_swap_marks_an_upgrade_not_a_plain_grab(self) -> None:
        # Every size differs from its placement, so the existing all-sizes-differ check marks an upgrade. No size is
        # doubled, and this rule would mark an upgrade too: the two agree.
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 100, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_doubled_unlisted_size_does_not_grab(self) -> None:
        # A doubled unlisted size is the outdated-copy check's business, which one listed match stops.
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 999, 12), self._ours(3, 999, 13)]

        item = self._decide(self._listing(100, 200, 300), arr_episodes)

        assert item.to_grab is False

    def test_a_private_groups_doubled_size_reports_private_only(self) -> None:
        result = self._plan(self._listing(100, 200, is_public=False), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert (result.skips.warned, result.seadex_dict["Era-Raws"].releases["u1"].to_grab) == (True, False)

    @staticmethod
    def _beside_public(*, is_fallback: bool) -> dict[str, GroupReleases]:
        """The private Era-Raws release of 100 and 200 beside a public group listing the same files."""

        episodes = [placed_episode(season=1, episode=1, size=100), placed_episode(season=1, episode=2, size=200)]
        public = make_release(
            url="https://nyaa.si/view/2", size=[100, 200], episodes=episodes, infohash="h2", is_fallback=is_fallback
        )
        return {
            **TestListedSizeOnTooManyEpisodes._listing(100, 200, is_public=False),
            "Pub": rg_group({"u2": public}),
        }

    def test_a_private_doubled_release_promotes_a_public_group_listing_the_same_files(self) -> None:
        result = self._plan(self._beside_public(is_fallback=False), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert result.seadex_dict["Pub"].releases["u2"].to_grab is True
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False
        assert result.skips.warned is False

    def test_a_private_doubled_release_is_skipped_rather_than_replaced_by_a_fallback(self) -> None:
        # A fallback never replaces the Arr's copy of the preferred release: the doubled-size skip warns instead.
        result = self._plan(self._beside_public(is_fallback=True), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert result.seadex_dict["Pub"].releases["u2"].to_grab is False
        assert (result.skips.warned, result.skips.outdated) == (True, True)


class TestGroupGuardFacts:
    """What the plan tells the import: which groups' copies the arr has, which are outdated, and whose files lack a group."""

    def test_no_episode_mixed_sizes_exit_the_guard_without_a_grab(self) -> None:
        # One of the movie's files is at a listed size and one isn't: no upgrade of the whole release, so nothing is
        # grabbed, but the copy is partly outdated and the group is not protected.
        seadex = {"Blunt": rg_group({"u1": make_release(size=[100, 200])})}

        plan = make_planner(arr=Arr.RADARR).plan(seadex, ArrFiles(by_group={"Blunt": (100, 999)}))

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == OwnedAtGrab(stale_groups=("Blunt",))

    def test_episode_partly_outdated_group_exits_the_guard_without_a_grab(self) -> None:
        # The same by episode: one episode at an unlisted size is no upgrade of the whole release, but the import
        # must be free to replace that file.
        seadex = {"Era-Raws": _sonarr_group(u1=make_release(size=[100, 200], episodes=[_S1E1, _S1E2]))}
        title = [
            sonarr_ep(1, 1, size=100, release_group="Era-Raws"),
            sonarr_ep(1, 2, size=999, release_group="Era-Raws"),
        ]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == OwnedAtGrab(stale_groups=("Era-Raws",))

    def test_outdated_file_at_an_unlisted_episode_condemns_the_group(self) -> None:
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

    def test_blind_release_never_reads_as_an_upgrade(self) -> None:
        # A release that lists no sizes proves nothing: the arr's copy stands, and with every release of the group
        # blind the group is neither protected nor outdated.
        seadex = {"Ember": rg_group({"u1": make_release(size=[])})}

        plan = make_planner(arr=Arr.RADARR).plan(seadex, ArrFiles(by_group={"Ember": (100,)}))

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == OwnedAtGrab()

    def test_blind_release_beside_a_sized_owned_one_keeps_the_group_protected(self) -> None:
        # A blind cross-seed takes nothing from the protection the sized release earned.
        seadex = {"Ember": rg_group({"u1": make_release(size=[100]), "u2": make_release(size=[])})}

        plan = make_planner(arr=Arr.RADARR).plan(seadex, ArrFiles(by_group={"Ember": (100,)}))

        assert _grabs(plan.seadex_dict) == {}
        assert plan.guards == OwnedAtGrab(entry_groups=("Ember",))

    def test_groups_with_no_files_in_the_arr_stay_protected(self) -> None:
        # A group the arr has no file of is protected, so a copy of it imported while the grab waits is not overwritten.
        kept = make_release(size=[300], episodes=[placed_episode(season=1, episode=1, size=300)])
        seadex = {"RG": _sonarr_group(u1=make_release(size=[100], episodes=[_S1E1])), "Kept": _sonarr_group(u2=kept)}
        title = [sonarr_ep(1, 1, episode_file_id=0)]

        plan = make_planner().plan(seadex, arr_files(title), title)

        assert plan.guards == OwnedAtGrab(entry_groups=("RG", "Kept"))

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


class TestReduceDropRescue:
    """Group-atomic drops must not lose episode coverage (the rescue pass)."""

    E11 = placed_episode(season=1, episode=1, size=100)
    E21 = placed_episode(season=2, episode=1, size=100)

    def _equal_union_pair(self, *, b_first: bool) -> dict[str, GroupReleases]:
        # Two mixed groups whose unions match (the private releases inflate both
        # keys to {S1E1, S2E1}), so they land in ONE same-files set.
        a = rg_group(
            {
                "a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1"),
                "a_priv": make_release(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            episodes=[self.E11, self.E21],
        )
        b = rg_group(
            {
                "b_pub": make_release(download=True, is_public=True, episodes=[self.E21], infohash="b1"),
                "b_priv": make_release(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            episodes=[self.E11, self.E21],
        )
        return {"B": b, "A": a} if b_first else {"A": a, "B": b}

    def test_losers_uncovered_public_release_is_rescued(self) -> None:
        # Neither mixed group is fully addable, so the keeper degrades to the
        # first. The loser's public release carries the set's only addable S2 (or S1)
        # coverage and must be marked to grab again, whichever group wins.
        for b_first in (False, True):
            planner = make_planner()
            seadex = self._equal_union_pair(b_first=b_first)
            planner.reduce_overlapping_grabs(seadex)
            assert seadex["A"].releases["a_pub"].to_grab is True, f"b_first={b_first}"
            assert seadex["B"].releases["b_pub"].to_grab is True, f"b_first={b_first}"
            # Exactly one private release survives (the keeper's). The loser's is dropped.
            priv_flags = [seadex["A"].releases["a_priv"].to_grab, seadex["B"].releases["b_priv"].to_grab]
            assert sorted(priv_flags) == [False, True], f"b_first={b_first}"

    def test_covered_drop_stays_dropped(self) -> None:
        # The loser's public release adds nothing over the keeper's surviving public
        # coverage: it stays dropped (no duplicate grab).
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11, self.E21], infohash="a1")},
                episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="b1")},
                # The union is inflated to match A's key (the R1 mechanic).
                episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_equal_coverage_drop_stays_dropped(self) -> None:
        # MUTATION PIN (_rescue_dropped_coverage): `release_keys <= survivor_keys`
        # flipped to `<` marks to grab again a dropped release whose coverage EQUALS the
        # survivors' - a pure duplicate grab. Equal single-episode coverage on
        # both public groups: the loser must stay dropped.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                episodes=[self.E11],
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="b1")},
                episodes=[self.E11],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_rescue_accumulates_survivor_coverage(self) -> None:
        # MUTATION PIN (_rescue_dropped_coverage): `survivor_keys |= release_keys`
        # degraded to `=` forgets the ORIGINAL survivors after the first rescue.
        # Keeper A covers S1. Dropped B (S2) is rescued. Dropped C (S1) is already
        # covered by A and must stay dropped - under `=` it would be marked to grab again.
        planner = make_planner()
        shared = [self.E11, self.E21]
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                episodes=shared,
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E21], infohash="b1")},
                episodes=shared,
            ),
            "C": rg_group(
                {"c_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="c1")},
                episodes=shared,
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is True  # rescued: only S2 carrier
        assert seadex["C"].releases["c_pub"].to_grab is False  # covered by keeper A

    def test_movie_releases_never_rescued(self) -> None:
        # Movie/no-parse releases have no episode vocabulary, so the rescue can't
        # (and mustn't) reason about their coverage.
        planner = make_planner()
        seadex = {
            "A": rg_group({"a_pub": make_release(download=True, is_public=True, infohash="a1")}),
            "B": rg_group({"b_pub": make_release(download=True, is_public=True, infohash="b1")}),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_owned_public_release_never_resurrected(self) -> None:
        # A release the matcher left not to grab (owned) is not part of this pass's drops, so
        # the rescue never touches it - even when its coverage leaves the set.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {
                    "b_owned": make_release(download=False, is_public=True, episodes=[self.E21], infohash="b2"),
                    "b_priv": make_release(
                        download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None
                    ),
                },
                episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_owned"].to_grab is False
        assert seadex["B"].releases["b_priv"].to_grab is False


class TestTorrentLeftForAHandImport:
    """A torrent left for a hand import is decided by what its window has, never by its numbers."""

    _SIZES = (100, 200, 300)

    @classmethod
    def _seadex(
        cls, skip_reason: SkipReason = SkipReason.MISNUMBERED, **siblings: GroupReleases
    ) -> dict[str, GroupReleases]:
        """The torrent left for a hand import for `skip_reason`, three files at `_SIZES` placed nowhere, beside `siblings`."""

        item = make_release(size=list(cls._SIZES), episodes=[], infohash="h1")
        item.skip_reason = skip_reason
        return {"Pack": rg_group({"u1": item}), **siblings}

    @staticmethod
    def _decide(seadex: dict[str, GroupReleases], title_episodes: list[SonarrEpisode]) -> Release:
        """The torrent's release after the planner decides it against the window `title_episodes`."""

        return make_planner().plan(seadex, arr_files(title_episodes), title_episodes).seadex_dict["Pack"].releases["u1"]

    @staticmethod
    def _ours(episode: int, size: int, file_id: int) -> SonarrEpisode:
        """Special `episode` with the torrent's file `file_id` at `size`."""

        return sonarr_ep(0, episode, size=size, release_group="Pack", episode_file_id=file_id)

    @pytest.mark.parametrize("skip_reason", [SkipReason.MISNUMBERED, SkipReason.EPISODE_TITLES_CONTRADICT])
    def test_every_special_holding_a_torrent_file_is_in_place(self, skip_reason: SkipReason) -> None:
        arr_episodes = [self._ours(n, size, 10 + n) for n, size in enumerate(self._SIZES, 1)]

        item = self._decide(self._seadex(skip_reason), arr_episodes)

        assert item.to_grab is False

    @pytest.mark.parametrize("skip_reason", [SkipReason.MISNUMBERED, SkipReason.EPISODE_TITLES_CONTRADICT])
    def test_an_empty_special_marks_the_torrent_to_grab_as_an_upgrade(self, skip_reason: SkipReason) -> None:
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 200, 12), sonarr_ep(0, 3, episode_file_id=0)]

        item = self._decide(self._seadex(skip_reason), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_files_of_an_unlisted_group_mark_the_torrent_to_grab(self) -> None:
        arr_episodes = [
            sonarr_ep(0, n, size=size, release_group="Other", episode_file_id=10 + n)
            for n, size in enumerate(self._SIZES, 1)
        ]

        item = self._decide(self._seadex(), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, False)

    def test_a_special_a_sibling_release_covers_is_in_place(self) -> None:
        covered = placed_episode(season=0, episode=3, size=999)
        sibling = rg_group({"u2": make_release(url="u2", infohash="h2", episodes=[covered])}, episodes=[covered])
        arr_episodes = [
            self._ours(1, 100, 11),
            self._ours(2, 200, 12),
            sonarr_ep(0, 3, size=999, release_group="Sib", episode_file_id=13),
        ]

        item = self._decide(self._seadex(Sib=sibling), arr_episodes)

        assert item.to_grab is False

    def test_a_file_with_no_group_at_a_listed_size_is_the_torrents(self) -> None:
        # A hand import whose names carry no group: the listed size still says whose file each special has.
        arr_episodes = [sonarr_ep(0, n, size=size, episode_file_id=10 + n) for n, size in enumerate(self._SIZES, 1)]

        item = self._decide(self._seadex(), arr_episodes)

        assert item.to_grab is False

    def test_a_special_another_release_of_the_group_placed_is_in_place(self) -> None:
        # The group's single-file release covers the third special, so the torrent has nothing left to cover there.
        covered = placed_episode(season=0, episode=3, size=999)
        seadex = self._seadex()
        seadex["Pack"] = rg_group(
            {**seadex["Pack"].releases, "u2": make_release(url="u2", infohash="h2", episodes=[covered])},
            episodes=[covered],
        )
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 200, 12), self._ours(3, 999, 13)]

        item = self._decide(seadex, arr_episodes)

        assert item.to_grab is False

    def test_a_torrent_file_at_an_unlisted_size_marks_the_torrent_to_grab(self) -> None:
        arr_episodes = [self._ours(1, 150, 11), self._ours(2, 200, 12), self._ours(3, 300, 13)]

        item = self._decide(self._seadex(), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_no_episode_list_leaves_the_torrent_not_to_grab(self) -> None:
        result = make_planner().plan(self._seadex(), ArrFiles(), None)

        assert result.seadex_dict["Pack"].releases["u1"].to_grab is False
