# pyright: strict
"""Characterization tests for the download-decision engine.

This is the core domain logic extracted into `DownloadPlanner` in Phase 4:
`get_any_to_download`, `reduce_overlapping_downloads` and
`filter_by_release_group`. The tests assert on the resulting per-url
`download` flags and the private-only skip outcome surfaced on the
`PlanResult` / `PrivateOnlySkips` (rather than mutated run state, as before
Phase 4).

The `reduce_overlapping_downloads` branch matrix these tests pin:

    interactive ............................ early return, nothing touched
    per same-files set (no flagged: skip set), public_flagged == 0:
      upgrade_pending + promotion found .... INFO "private-only; grabbing public alternative X"
      upgrade_pending + none + fallback .... WARNING stale reason, stale_held, skipped
      otherwise ............................ WARNING "not supported", skipped
    public_flagged >= 1:
      first fully-addable group ............ keeper, losers unflag (debug only)
      none addable + mismatch + promotion .. INFO "remaining files private-only; grabbing ..."
      promotion refused + fallback rides ... stale_held set SILENTLY, degrade to first
      otherwise ............................ degrade to public_flagged[0]
      fallback keeper over private drops ... INFO "private-only; falling back to X"
    then per set: rescue re-flags just-dropped public urls no survivor covers.
    finally per group: identical non-empty filesets dedup to the first.

Skip state (skipped/groups/notices/stale_held) accumulates
across sets onto ONE result - the multi-set tests pin that accumulation.
"""

import logging

import pytest

from pearlarr.arr_models import ArrFiles, SonarrEpisode
from pearlarr.config import Arr
from pearlarr.output import Severity
from pearlarr.planner import GrabPlan, GrabPlanner
from pearlarr.releases import GroupReleases, PlacementSkip, PlannedGrab, Release, ReleaseEpisode, releases_to_grab

from .builders import arr_releases, make_planner, make_release, rg_group, sonarr_ep


class TestGetAnyToDownload:
    """`get_any_to_download` reports whether any url in the group dict has its `download` flag set."""

    def test_false_when_none_flagged(self) -> None:
        seadex = {"A": rg_group({"u": make_release(download=False)})}
        assert GrabPlanner.any_to_grab(seadex) is False

    def test_true_when_one_flagged(self) -> None:
        seadex = {"A": rg_group({"u": make_release(download=True)})}
        assert GrabPlanner.any_to_grab(seadex) is True


class TestFlaggedUrls:
    """`flagged_urls` is the one grab-set filter both arrs' seed builders share."""

    def test_only_hash_carrying_grabs_ride_the_triples(self) -> None:
        grabbed = make_release(url="u1", download=True, infohash="h1")
        seadex = {
            "A": rg_group({"u1": grabbed, "u2": make_release(url="u2", download=True, infohash=None)}),
            "B": rg_group({"u3": make_release(url="u3", download=False, infohash="h3")}),
        }

        assert releases_to_grab(seadex) == [PlannedGrab("A", "u1", grabbed, "h1")]


class TestReduceOverlappingDownloads:
    """`reduce_overlapping_downloads` picks one keeper per same-files set per the module's branch matrix, accumulating skip state across sets."""

    def test_interactive_is_noop(self) -> None:
        planner = make_planner(interactive=True)
        seadex = {
            "A": rg_group({"u1": make_release(download=True)}),
            "B": rg_group({"u2": make_release(download=True)}),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["B"].releases["u2"].to_grab is True

    def test_keeps_first_of_same_files(self) -> None:
        # No all_episodes -> all treated as the same files -> keep first flagged
        planner = make_planner()
        seadex = {
            "A": rg_group({"u1": make_release(download=True)}),
            "B": rg_group({"u2": make_release(download=True)}),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["B"].releases["u2"].to_grab is False

    def test_prefers_public_keeper(self) -> None:
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Pub": rg_group({"u2": make_release(download=True, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Pub"].releases["u2"].to_grab is True
        assert seadex["Priv"].releases["u1"].to_grab is False
        # A preferred public keeper over a preferred private pick is unremarkable
        # (only a fallback keeper gets an INFO notice).
        assert skips.notices == []

    def test_private_only_skips_and_flags(self) -> None:
        planner = make_planner()
        seadex = {"Priv": rg_group({"u1": make_release(download=True, is_public=False)})}
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert skips.warned is True
        assert skips.groups == ["Priv"]
        # Plain-reason mutant killer: no fallback rides, so no stale hold.
        assert skips.outdated is False
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].reason == "private-only (private releases not supported)"
        assert skips.notices[0].severity is Severity.WARNING

    def test_private_only_set_with_unrelated_fallback_still_warns(self) -> None:
        # The fallback covers DIFFERENT files (own same-files set): it doesn't
        # excuse dropping this set, so the private set still warns and holds
        # the title while the fallback proceeds.
        planner = make_planner()
        seadex = {
            "Priv": rg_group(
                {"u1": make_release(download=True, is_public=False)},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=0)],
            ),
            "Fall": rg_group(
                {"u2": make_release(download=True, is_public=True, is_fallback=True)},
                all_episodes=[ReleaseEpisode(season=1, episode=2, size=0)],
            ),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert seadex["Fall"].releases["u2"].to_grab is True
        assert skips.warned is True
        assert skips.groups == ["Priv"]
        assert len(skips.notices) == 1
        assert skips.notices[0].severity is Severity.WARNING

    def test_private_only_set_with_an_unflagged_fallback_warns_plainly(self) -> None:
        # Same files, an unflagged public fallback, and the private flag is no
        # upgrade: nothing says the Arr holds a stale copy, so the set gets the
        # plain warning and the title stays uncached.
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Fall": rg_group({"u2": make_release(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert seadex["Fall"].releases["u2"].to_grab is False
        assert skips.warned is True
        assert skips.groups == ["Priv"]
        assert skips.outdated is False
        assert [(n.reason, n.severity) for n in skips.notices] == [
            ("private-only (private releases not supported)", Severity.WARNING),
        ]

    def test_keeper_prefers_fully_addable_group(self) -> None:
        # A mixed group (public + flagged private url) is first, but its private
        # url would be refused at add time, losing the coverage only it carries:
        # the fully-public group wins keeper regardless of order.
        planner = make_planner()
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": make_release(download=True, is_public=True),
                    "u2": make_release(download=True, is_public=False),
                },
            ),
            "Pub": rg_group({"u3": make_release(download=True, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Pub"].releases["u3"].to_grab is True
        assert seadex["Mixed"].releases["u1"].to_grab is False
        assert seadex["Mixed"].releases["u2"].to_grab is False
        assert skips.warned is False
        assert skips.notices == []

    def test_keeper_degrades_to_first_when_all_mixed(self) -> None:
        # No fully-addable group in the set: keep the first public group (its
        # private url then hits the add-time WARNING), matching the old order.
        planner = make_planner()
        seadex = {
            "MixedA": rg_group(
                {
                    "u1": make_release(download=True, is_public=True),
                    "u2": make_release(download=True, is_public=False),
                },
            ),
            "MixedB": rg_group(
                {
                    "u3": make_release(download=True, is_public=True),
                    "u4": make_release(download=True, is_public=False),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["MixedA"].releases["u1"].to_grab is True
        assert seadex["MixedA"].releases["u2"].to_grab is True
        assert seadex["MixedB"].releases["u3"].to_grab is False
        assert seadex["MixedB"].releases["u4"].to_grab is False

    def test_private_only_set_with_fallback_holds_on_size_mismatch(self) -> None:
        # The private pick was re-flagged for a SIZE mismatch: the Arr owns the
        # preferred release at a stale size, and a fallback (a cascade loser)
        # never replaces an owned copy - warn and hold instead of promoting.
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Fall": rg_group({"u2": make_release(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Fall"].releases["u2"].to_grab is False
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert skips.warned is True
        assert skips.groups == ["Priv"]
        assert skips.outdated is True
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].severity is Severity.WARNING
        outdated_reason = (
            "private-only; your copy is outdated (its file size no longer matches the release) "
            "and only a fallback covers it"
        )
        assert skips.notices[0].reason == outdated_reason

    def test_promotion_prefers_fully_public_group(self) -> None:
        # Promotion only flips public urls, so a mixed group (a public url
        # sharing its release group with a private pick) would leave its private
        # url's coverage ungrabbed: the fully-public group wins.
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "MixedPub": rg_group(
                {
                    "u2": make_release(download=False, is_public=False),
                    "u3": make_release(download=False, is_public=True),
                },
            ),
            "Pub": rg_group({"u4": make_release(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Pub"].releases["u4"].to_grab is True
        assert seadex["MixedPub"].releases["u2"].to_grab is False
        assert seadex["MixedPub"].releases["u3"].to_grab is False
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert len(skips.notices) == 1
        assert "grabbing public alternative Pub" in skips.notices[0].reason

    def test_promotion_considers_preferred_public_group(self) -> None:
        # The size-mismatch promotion pool is any unflagged public group in the
        # set, not just fallbacks: a PREFERRED public pick with the same coverage
        # is promoted instead of the old warn-and-hold.
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Pub": rg_group({"u2": make_release(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Pub"].releases["u2"].to_grab is True
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert skips.warned is False
        assert skips.groups == []
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].severity is Severity.INFO
        # A preferred group is not a fallback, so the notice must not say so.
        assert skips.notices[0].reason == "private-only; grabbing public alternative Pub"

    def test_degraded_keeper_holds_when_only_a_fallback_could_stand_in(self) -> None:
        # A mixed group's unflagged (owned) public url makes it public_to_grab
        # and no group is fully addable. The only unflagged public alternative
        # is a FALLBACK, which never replaces M's owned stale copy: the keeper
        # degrades to M (the add-time gate then refuses its private url) and the
        # stale bit is set - deliberately with no planner notice.
        planner = make_planner()
        seadex = {
            "M": rg_group(
                {
                    "u1": make_release(download=False, is_public=True),
                    "u2": make_release(download=True, is_public=False, upgrade=True),
                },
            ),
            "F": rg_group({"u3": make_release(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["F"].releases["u3"].to_grab is False
        assert seadex["M"].releases["u1"].to_grab is False
        assert seadex["M"].releases["u2"].to_grab is True
        assert skips.warned is False
        assert skips.outdated is True
        assert skips.notices == []

    def test_degraded_keeper_promotes_preferred_public_on_size_mismatch(self) -> None:
        # Same degraded-keeper shape, but the unflagged alternative is a
        # PREFERRED public group: equal rank still supersedes, so it's promoted.
        planner = make_planner()
        seadex = {
            "M": rg_group(
                {
                    "u1": make_release(download=False, is_public=True),
                    "u2": make_release(download=True, is_public=False, upgrade=True),
                },
            ),
            "P": rg_group({"u3": make_release(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["P"].releases["u3"].to_grab is True
        assert seadex["M"].releases["u1"].to_grab is False
        assert seadex["M"].releases["u2"].to_grab is False
        assert skips.warned is False
        assert skips.outdated is False
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["M"]
        assert skips.notices[0].severity is Severity.INFO
        assert skips.notices[0].reason == "remaining files private-only; grabbing public alternative P"

    def test_degraded_keeper_without_mismatch_does_not_promote(self) -> None:
        # No size mismatch means the flags may be plain missing-content grabs.
        # the unflagged public group could be genuinely owned, so the keeper
        # degrades as before and nothing is promoted.
        planner = make_planner()
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": make_release(download=True, is_public=True),
                    "u2": make_release(download=True, is_public=False),
                },
            ),
            "P": rg_group({"u3": make_release(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Mixed"].releases["u1"].to_grab is True
        assert seadex["Mixed"].releases["u2"].to_grab is True
        assert seadex["P"].releases["u3"].to_grab is False
        assert skips.notices == []

    def test_fallback_keeper_over_private_notices(self) -> None:
        # Same files: the public fallback is kept over the private preferred pick,
        # with an INFO notice naming the keeper (and no skipped flag).
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False)}),
            "Fall": rg_group({"u2": make_release(download=True, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Fall"].releases["u2"].to_grab is True
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert skips.warned is False
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].severity is Severity.INFO
        # The keeper flow is now the sole producer of this wording. Pin it exactly.
        assert skips.notices[0].reason == "private-only; falling back to Fall"

    def test_flagged_mixed_group_never_promotes_itself(self) -> None:
        # MUTATION PIN (_promote_public_alternative): the candidate filter's
        # `not flagged AND public` flipped to `or` admits the flagged mixed group
        # itself, which then "promotes" over itself and unflags everything. With
        # no unflagged public group the keeper must degrade to the mixed group,
        # both its urls staying flagged, with no notice.
        planner = make_planner()
        seadex = {
            "M": rg_group(
                {
                    "u_pub": make_release(download=True, is_public=True),
                    "u_priv": make_release(download=True, is_public=False, upgrade=True),
                },
            ),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["M"].releases["u_pub"].to_grab is True
        assert seadex["M"].releases["u_priv"].to_grab is True
        assert skips.warned is False
        assert skips.notices == []

    def test_promotes_mixed_unflagged_group_flipping_only_public_urls(self) -> None:
        # MUTATION PIN (_promote_public_alternative): the `next(..., candidates[0])`
        # fallback-to-first mutated away would return None and fall to the
        # warn-and-hold. An upgrade-pending set whose ONLY unflagged public group
        # is MIXED must still promote it, flipping just its public url.
        planner = make_planner()
        seadex = {
            "P": rg_group({"p": make_release(download=True, is_public=False, upgrade=True, infohash=None)}),
            "M": rg_group(
                {
                    "m_pub": make_release(download=False, is_public=True, infohash="m1"),
                    "m_priv": make_release(download=False, is_public=False, infohash=None),
                },
            ),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["M"].releases["m_pub"].to_grab is True
        assert seadex["M"].releases["m_priv"].to_grab is False  # promotion is public-only
        assert seadex["P"].releases["p"].to_grab is False
        assert skips.warned is False
        assert len(skips.notices) == 1
        assert skips.notices[0].severity is Severity.INFO
        # M is a preferred group, not a fallback, so the verb must say so.
        assert skips.notices[0].reason == "private-only; grabbing public alternative M"

    def test_different_files_both_kept(self) -> None:
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"u1": make_release(download=True)},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=0)],
            ),
            "B": rg_group(
                {"u2": make_release(download=True)},
                all_episodes=[ReleaseEpisode(season=1, episode=2, size=0)],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["B"].releases["u2"].to_grab is True

    def test_two_private_only_sets_accumulate_groups_and_notices(self) -> None:
        # Two disjoint private-only sets in ONE entry: the skip state must
        # accumulate onto one result - both group names, one WARNING notice per
        # set (in set order) - not overwrite or collapse to the last set.
        planner = make_planner()
        seadex = {
            "P1": rg_group(
                {"u1": make_release(download=True, is_public=False)},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=0)],
            ),
            "P2": rg_group(
                {"u2": make_release(download=True, is_public=False)},
                all_episodes=[ReleaseEpisode(season=2, episode=1, size=0)],
            ),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["P1"].releases["u1"].to_grab is False
        assert seadex["P2"].releases["u2"].to_grab is False
        assert skips.warned is True
        assert skips.groups == ["P1", "P2"]
        assert [n.groups for n in skips.notices] == [["P1"], ["P2"]]
        assert [n.severity for n in skips.notices] == [Severity.WARNING, Severity.WARNING]

    def test_a_stale_hold_survives_a_later_plain_skip(self) -> None:
        # One entry, two sets: a stale hold (set 1), then a plain private-only
        # skip (set 2). The later set doesn't clear the hold, and each set keeps
        # its own reason.
        planner = make_planner()
        e1 = [ReleaseEpisode(season=1, episode=1, size=0)]
        e2 = [ReleaseEpisode(season=2, episode=1, size=0)]
        seadex = {
            "Priv1": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}, all_episodes=e1),
            "Fall1": rg_group({"u2": make_release(download=False, is_public=True, is_fallback=True)}, all_episodes=e1),
            "Priv2": rg_group({"u3": make_release(download=True, is_public=False)}, all_episodes=e2),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        outdated_reason = (
            "private-only; your copy is outdated (its file size no longer matches the release) "
            "and only a fallback covers it"
        )
        assert skips.outdated is True
        assert skips.groups == ["Priv1", "Priv2"]
        assert [n.reason for n in skips.notices] == [outdated_reason, "private-only (private releases not supported)"]

    def test_both_promotion_prefixes_in_one_entry(self) -> None:
        # The two promotion notices are DISTINCT: the no-public-flagged branch
        # says "private-only" (the whole set was private), the degraded-keeper
        # branch says "remaining files private-only" (the set had a public url).
        # One entry exercising both must surface both wordings, nothing skipped.
        planner = make_planner()
        e1 = [ReleaseEpisode(season=1, episode=1, size=0)]
        e2 = [ReleaseEpisode(season=2, episode=1, size=0)]
        seadex = {
            "PrivA": rg_group(
                {"u1": make_release(download=True, is_public=False, upgrade=True)},
                all_episodes=e1,
            ),
            "PubA": rg_group({"u2": make_release(download=False, is_public=True)}, all_episodes=e1),
            "MixedB": rg_group(
                {
                    "u3": make_release(download=False, is_public=True),
                    "u4": make_release(download=True, is_public=False, upgrade=True),
                },
                all_episodes=e2,
            ),
            "PubB": rg_group({"u5": make_release(download=False, is_public=True)}, all_episodes=e2),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["PubA"].releases["u2"].to_grab is True
        assert seadex["PubB"].releases["u5"].to_grab is True
        assert seadex["PrivA"].releases["u1"].to_grab is False
        assert seadex["MixedB"].releases["u3"].to_grab is False
        assert seadex["MixedB"].releases["u4"].to_grab is False
        assert skips.warned is False
        assert [n.reason for n in skips.notices] == [
            "private-only; grabbing public alternative PubA",
            "remaining files private-only; grabbing public alternative PubB",
        ]


class TestSameGroupDuplicateDedup:
    """Within one group, flagged urls carrying the same files dedup to the first.

    An unsized listing never dedups (can't prove identity).
    """

    def test_identical_filesets_keep_first(self) -> None:
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=["A - S01E01.mkv"], size=[100], download=True),
                    "u2": make_release(files=["A - S01E01.mkv"], size=[100], download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is False

    def test_promoted_duplicates_dedup(self) -> None:
        # The promotion branch flips EVERY public url of the promoted group: two
        # cross-seeded copies of one release must still yield a single grab.
        planner = make_planner()
        seadex = {
            "Priv": rg_group({"u1": make_release(download=True, is_public=False, upgrade=True)}),
            "Pub": rg_group(
                {
                    "u2": make_release(files=["P - S01E01.mkv"], size=[100], download=False, is_public=True),
                    "u3": make_release(files=["P - S01E01.mkv"], size=[100], download=False, is_public=True),
                },
            ),
        }
        skips = planner.reduce_overlapping_grabs(seadex)
        assert seadex["Pub"].releases["u2"].to_grab is True
        assert seadex["Pub"].releases["u3"].to_grab is False
        assert seadex["Priv"].releases["u1"].to_grab is False
        assert "grabbing public alternative Pub" in skips.notices[0].reason

    def test_different_filesets_both_kept(self) -> None:
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=["A - S01E01.mkv"], size=[100], download=True),
                    "u2": make_release(files=["A - S02E01.mkv"], size=[200], download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is True

    def test_folder_nested_listing_still_dedups(self) -> None:
        # One listing folds the file under a folder the other lists flat:
        # still the same release cross-seeded, so identity compares sizes.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=["A - S01E01.mkv"], size=[100], download=True),
                    "u2": make_release(files=["Extras/A - s01e01.mkv"], size=[100], download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is False

    def test_same_names_at_different_sizes_never_dedup(self) -> None:
        # Two per-cour listings can name their episodes identically. They are
        # DIFFERENT files, and only their sizes say so - collapsing them would
        # silently drop a whole cour from the plan.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=["S01/A - 01.mkv"], size=[100], download=True),
                    "u2": make_release(files=["S02/A - 01.mkv"], size=[200], download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is True

    def test_a_listing_carrying_more_copies_never_dedups(self) -> None:
        # A two-file batch is not a duplicate of the one-file listing whose
        # size it shares: unflagging it would silently drop half its content.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(files=["A - 01.mkv"], size=[100], download=True),
                    "u2": make_release(files=["S01/A - 01.mkv", "S02/A - 01.mkv"], size=[100, 100], download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is True

    def test_unknown_filesets_both_kept(self) -> None:
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {
                    "u1": make_release(download=True),
                    "u2": make_release(download=True),
                },
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["A"].releases["u2"].to_grab is True

    def test_identical_filesets_across_groups_never_dedup(self) -> None:
        # The dedup's seen-set is rebuilt PER GROUP: two groups in different
        # same-files sets carrying identical file lists both survive (a hoisted
        # cross-group seen-set would silently drop B's coverage).
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"u1": make_release(files=["X - S01E01.mkv"], download=True)},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=0)],
            ),
            "B": rg_group(
                {"u2": make_release(files=["X - S01E01.mkv"], download=True)},
                all_episodes=[ReleaseEpisode(season=2, episode=1, size=0)],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["u1"].to_grab is True
        assert seadex["B"].releases["u2"].to_grab is True


class TestFilterByReleaseGroup:
    """`filter_by_release_group` skips a release whose group and size(s) already match what the arr holds (case-insensitively), downloading otherwise."""

    def test_new_group_no_episodes_downloads(self) -> None:
        planner = make_planner()
        seadex = {"NewRG": rg_group({"u1": make_release(episodes=[], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"OldRG": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["NewRG"].releases["u1"].to_grab is True

    def test_matching_group_sizes_match_no_download(self) -> None:
        planner = make_planner()
        seadex = {"RG": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"RG": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is False

    def test_matching_group_sizes_differ_downloads(self) -> None:
        planner = make_planner()
        seadex = {"RG": rg_group({"u1": make_release(episodes=[], size=[200], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"RG": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is True

    def test_matching_group_case_insensitive_no_download(self) -> None:
        # The no-episode path matches by normalized name like the per-episode
        # path: Radarr's "EMBER" is SeaDex's "Ember", so a size match must not
        # re-download (raw comparison used to re-grab owned content).
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"EMBER": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is False

    def test_matching_group_case_insensitive_size_mismatch_downloads(self) -> None:
        # The same normalized group with DISJOINT sizes must still upgrade.
        # This pins the membership + size-lookup normalization specifically:
        # the overlap-gate normalization alone would read the group as covered
        # and leave download False (a silently missed upgrade).
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[200], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"EMBER": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is True
        assert result.seadex_dict["Ember"].releases["u1"].is_upgrade is True

    def test_nothing_placed_on_the_entry_nothing_on_disk_downloads(self) -> None:
        # The placement put none of the url's files on the entry's episodes (its
        # other seasons' names carry numbers, the entry's own do not): judged like
        # a numberless url, by group and size, so an absent entry is grabbed rather
        # than read as owned because no episode was ever compared.
        planner = make_planner()
        seadex = {"RG": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(0, 21, episode_file_id=0)],
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is True

    def test_nothing_placed_on_the_entry_owned_copy_holds(self) -> None:
        # The entry's copy on disk is this group's at a listed size: held.
        planner = make_planner()
        seadex = {"RG": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"RG": (100,)}),
            ep_list=[sonarr_ep(0, 21, size=100, release_group="RG")],
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is False

    def test_nothing_placed_on_the_entry_sibling_copy_holds(self) -> None:
        # The entry's copy on disk is a sibling recommended group's: nothing to flag.
        planner = make_planner()
        seadex = {
            "RG": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")}),
            "Other": rg_group(
                {
                    "u2": make_release(
                        episodes=[ReleaseEpisode(season=0, episode=21, size=200)], size=[200], infohash="h2"
                    )
                },
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Other": (200,)}),
            ep_list=[sonarr_ep(0, 21, size=200, release_group="Other")],
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is False

    def test_one_placed_record_keeps_the_per_episode_leg(self) -> None:
        # One file placed on the entry, so the per-episode leg decides: the
        # entry's copy is this group's at another size, an upgrade the
        # group-and-size leg (a listed size on disk) would have held.
        planner = make_planner()
        episodes = [ReleaseEpisode(season=1, episode=1, size=999)]
        seadex = {"RG": rg_group({"u1": make_release(episodes=episodes, size=[100, 999], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"RG": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="RG")],
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is True
        assert result.seadex_dict["RG"].releases["u1"].is_upgrade is True

    def test_episode_match_same_rg_and_size_no_download(self) -> None:
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Era-Raws": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False

    def test_episode_different_rg_downloads(self) -> None:
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"SubsPlease": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True

    def test_episode_same_rg_all_sizes_differ_downloads(self) -> None:
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=999)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Era-Raws": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True

    def test_episode_groupless_file_at_exact_size_no_download(self) -> None:
        # A rename can strip the on-disk group tag: a positive exact-size match
        # against the pick's file still identifies the copy - no re-grab.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=100, release_group=None)],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False

    def test_episode_groupless_file_at_other_size_downloads(self) -> None:
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=555, release_group=None)],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True

    def test_episode_groupless_zero_size_still_downloads(self) -> None:
        # 0 == 0 must not read as identity: a failed-copy stub against a
        # zero-size listing entry stays grabbable.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=0)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=0, release_group=None)],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True

    def test_episode_untagged_exact_size_never_masks_a_stale_release(self) -> None:
        # One untagged file identified by size must not vouch for the release's
        # OTHER episodes: its size matched by construction, so counting it in
        # the staleness fold would strand every stale episode around it.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(
                        episodes=[
                            ReleaseEpisode(season=1, episode=1, size=100),
                            ReleaseEpisode(season=1, episode=2, size=200),
                        ],
                        infohash="h1",
                    ),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Era-Raws": (999,)}),
            ep_list=[
                sonarr_ep(1, 1, size=100, release_group=None),
                sonarr_ep(1, 2, size=999, release_group="Era-Raws"),
            ],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True
        assert result.seadex_dict["Era-Raws"].releases["u1"].is_upgrade is True

    def test_episode_untagged_file_owns_the_episode_for_every_pick(self) -> None:
        # Ownership is a property of the FILE: the pick whose size identified it
        # skips its download, and so must the sibling pick covering the same
        # episode - otherwise the entry is re-downloaded from the other pick.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {"u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1")},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=100)],
            ),
            "Other": rg_group(
                {"u2": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=300)], infohash="h2")},
                all_episodes=[ReleaseEpisode(season=1, episode=1, size=300)],
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=100, release_group=None, ep_id=101)],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False
        assert result.seadex_dict["Other"].releases["u2"].to_grab is False
        # The plan reports the identified episode - with the size backing the
        # claim - so the import seeds protect exactly that file.
        assert result.guards.owned_episodes == ((101, 100),)

    def test_episode_blank_named_group_never_covers_an_untagged_file(self) -> None:
        # PIN: a pick whose name normalizes to None (a blank name) is indexed
        # nowhere in the coverage index. An untagged on-disk file's group also
        # normalizes to None, so indexing the blank name would read the file as
        # covered and silently suppress the real pick's grab.
        planner = make_planner()
        seadex = {
            "": rg_group({"u1": make_release(episodes=[], infohash="h1")}),
            "Era-Raws": rg_group(
                {"u2": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h2")},
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=999, release_group=None, ep_id=101)],
        )
        assert result.seadex_dict["Era-Raws"].releases["u2"].to_grab is True

    def test_plan_drops_an_ownership_claim_with_no_real_episode_id(self) -> None:
        # PIN (intent, not a bug): the owned-episodes fold keeps only episodes
        # with a real id - an id-0 claim could never be re-verified or POSTed
        # at import time, so it must not ride the guards.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {"u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1")},
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, size=100, release_group=None, ep_id=0)],
        )
        assert result.guards.owned_episodes == ()

    def test_plan_reports_no_owned_episode_for_a_tagged_file(self) -> None:
        # A file that still carries a group tag is identified by name, never by
        # size - reading it as owned would bypass the group guard entirely.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {"u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1")},
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Other": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Other", ep_id=101)],
        )
        assert result.guards.owned_episodes == ()

    def test_episode_partly_stale_group_exits_the_guard_without_a_download(self) -> None:
        # One episode at an unlisted size is not a whole-release upgrade (that
        # needs every episode), but it does mean the copy on disk is partly
        # stale - the group must exit the never-overwrite set so the import can
        # replace the stale file, and it is a positive stale verdict.
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(
                        episodes=[
                            ReleaseEpisode(season=1, episode=1, size=100),
                            ReleaseEpisode(season=1, episode=2, size=200),
                        ],
                        infohash="h1",
                    ),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Era-Raws": (100, 999)}),
            ep_list=[
                sonarr_ep(1, 1, size=100, release_group="Era-Raws"),
                sonarr_ep(1, 2, size=999, release_group="Era-Raws"),
            ],
        )
        release = result.seadex_dict["Era-Raws"].releases["u1"]
        assert release.to_grab is False
        assert result.guards.entry_groups == ()
        assert result.guards.stale_groups == ("Era-Raws",)

    def test_movie_groupless_file_at_exact_sizes_no_download(self) -> None:
        # The no-episode twin: untagged files key their sizes under None, and
        # every untagged file belonging to this listing reads as owned.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(100,)),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is False

    def test_movie_groupless_file_identified_past_the_listing_extras(self) -> None:
        # A listing carries files the arr never holds (subs, fonts, extras), so
        # containment runs arr -> listing. Requiring every LISTED size on disk
        # would only ever identify a single-file release.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[100, 7], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(100,)),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is False

    def test_movie_with_no_files_at_all_downloads(self) -> None:
        # A movie with no files folds to an empty record, leaving no untagged
        # size to identify anything: an empty multiset must never read as covered.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is True

    def test_movie_zero_size_untagged_file_downloads(self) -> None:
        # A zero-byte failed copy proves nothing, even against a zero-size entry.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[0, 100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(0,)),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is True

    def test_movie_groupless_file_at_other_size_downloads(self) -> None:
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(50,)),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u1"].to_grab is True

    def test_movie_blank_named_listing_never_name_matches_untagged_files(self) -> None:
        # PIN: a pick whose name normalizes to None has no name to match, so it
        # reaches the untagged-ownership check, where a sizeless listing proves
        # nothing and the grab stands. The old null-keyed dict read a blank pick
        # name against untagged files as a NAME match and held the copy unverified.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"": rg_group({"u1": make_release(episodes=[], size=[], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(50,)),
            ep_list=None,
        )
        assert result.seadex_dict[""].releases["u1"].to_grab is True

    def test_movie_blank_named_listing_partial_size_overlap_downloads(self) -> None:
        # PIN: untagged ownership is all-or-nothing. A blank-named pick whose
        # listing covers only PART of the untagged multiset does not own the
        # copy, so the grab stands. The old null-keyed dict name-matched the
        # blank name and held the copy on any size overlap.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"": rg_group({"u1": make_release(episodes=[], size=[50], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(50, 60)),
            ep_list=None,
        )
        assert result.seadex_dict[""].releases["u1"].to_grab is True

    def test_movie_blank_named_listing_grabs_plain_not_upgrade(self) -> None:
        # PIN: a blank name never reaches the name-match arm, so its grab is a
        # plain download, never an upgrade - with no name there is no evidence
        # the on-disk copy is an older cut of the same release.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(50,)),
            ep_list=None,
        )
        item = result.seadex_dict[""].releases["u1"]
        assert item.to_grab is True
        assert item.is_upgrade is False

    def test_movie_blank_named_listing_sets_no_overlap_when_not_owner(self) -> None:
        # PIN: a blank name that does NOT own the untagged copy sets no overlap,
        # so the entry grabs (the movie-level reducer keeps one pick). The old
        # null-keyed dict read blank-vs-untagged as both a name match AND an
        # overlap, holding every pick - the entry was silently skipped.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {
            "": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")}),
            "RG": rg_group({"u2": make_release(episodes=[], size=[300], infohash="h2")}),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(100, 60)),
            ep_list=None,
        )
        assert result.seadex_dict[""].releases["u1"].to_grab is True

    def test_movie_blank_named_owner_holds_siblings_by_overlap(self) -> None:
        # PIN: ownership IS an overlap. When the blank-named pick owns the
        # untagged copy by size, a sibling pick absent from the library holds
        # exactly as it would had the copy kept its tag - nothing grabs.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {
            "": rg_group({"u1": make_release(episodes=[], size=[100, 7], infohash="h1")}),
            "RG": rg_group({"u2": make_release(episodes=[], size=[300], infohash="h2")}),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(100,)),
            ep_list=None,
        )
        assert result.seadex_dict[""].releases["u1"].to_grab is False
        assert result.seadex_dict["RG"].releases["u2"].to_grab is False

    def test_episodes_but_no_ep_list_skips(self) -> None:
        planner = make_planner()
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=None,
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False

    def test_matching_group_radarr_none_size_downloads(self) -> None:
        # A group whose only file has an unreadable size keeps its name with no
        # sizes. That empty set is disjoint from the real SeaDex sizes, so the
        # group is grabbed.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"RG": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"RG": ()}),
            ep_list=None,
        )
        assert result.seadex_dict["RG"].releases["u1"].to_grab is True

    def test_debug_logging_path_does_not_crash(self) -> None:
        # Exercise the debug_on=True branch (its f-strings are otherwise skipped)
        planner = make_planner()
        planner.logger.setLevel(logging.DEBUG)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": make_release(episodes=[ReleaseEpisode(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"SubsPlease": (100,)}),
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is True
        # Reset so a later test doesn't inherit DEBUG from the shared logger
        planner.logger.setLevel(logging.WARNING)


class TestListedSizeOnTooManyEpisodes:
    """A listed size our group holds on more episodes than the listing places it on flags the url as an upgrade."""

    @staticmethod
    def _listing(*sizes: int, is_public: bool = True) -> dict[str, GroupReleases]:
        """One Era-Raws listing of season 1 episodes 1..n at `sizes`, one placement per size given."""

        episodes = [ReleaseEpisode(season=1, episode=n, size=size) for n, size in enumerate(sizes, 1)]
        item = make_release(size=list(sizes), episodes=episodes, infohash="h1", is_public=is_public)
        return {"Era-Raws": rg_group({"u1": item})}

    @staticmethod
    def _plan(seadex: dict[str, GroupReleases], ep_list: list[SonarrEpisode]) -> GrabPlan:
        """The planner's result for `seadex` against `ep_list`, every tagged file reported to the arr."""

        return make_planner().plan(seadex, arr_releases(ep_list), ep_list)

    @classmethod
    def _decide(cls, seadex: dict[str, GroupReleases], ep_list: list[SonarrEpisode]) -> Release:
        """The Era-Raws url after the planner judges it against `ep_list`."""

        return cls._plan(seadex, ep_list).seadex_dict["Era-Raws"].releases["u1"]

    @staticmethod
    def _ours(episode: int, size: int, file_id: int) -> SonarrEpisode:
        """Season 1 `episode` holding Era-Raws' file `file_id` at `size`."""

        return sonarr_ep(1, episode, size=size, release_group="Era-Raws", episode_file_id=file_id)

    def test_one_listed_file_on_two_episode_files_downloads(self) -> None:
        # File 2 imported onto both episodes: episode 1 lacks its own file.
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_one_episode_file_covering_two_episodes_at_a_size_listed_once_downloads(self) -> None:
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 200, 11)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_twelve_file_batch_with_one_file_on_two_episodes_downloads(self) -> None:
        arr_episodes = [self._ours(n, 700 if n == 2 else 100 * n, 10 + n) for n in range(1, 13)]

        item = self._decide(self._listing(*range(100, 1300, 100)), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_listed_size_placed_on_no_episode_but_held_downloads(self) -> None:
        # File 999 is listed but placed nowhere, so holding it on E02 is one hold too many.
        episodes = [ReleaseEpisode(season=1, episode=1, size=100), ReleaseEpisode(season=1, episode=2, size=200)]
        seadex = {"Era-Raws": rg_group({"u1": make_release(size=[100, 200, 999], episodes=episodes, infohash="h1")})}

        item = self._decide(seadex, [self._ours(1, 100, 11), self._ours(2, 999, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_an_episode_two_listed_files_claim_at_different_sizes_is_left_out(self) -> None:
        # E01 is ambiguous, so size 200 counts once held against its one placement on E02.
        episodes = [
            ReleaseEpisode(season=1, episode=1, size=100),
            ReleaseEpisode(season=1, episode=1, size=150),
            ReleaseEpisode(season=1, episode=2, size=200),
        ]
        seadex = {"Era-Raws": rg_group({"u1": make_release(size=[100, 150, 200], episodes=episodes, infohash="h1")})}

        item = self._decide(seadex, [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert item.to_grab is False

    def test_a_two_episode_listed_file_held_once_under_both_keys_does_not_download(self) -> None:
        item = self._decide(self._listing(300, 300), [self._ours(1, 300, 11), self._ours(2, 300, 11)])

        assert item.to_grab is False

    def test_a_size_listed_twice_held_twice_does_not_download(self) -> None:
        item = self._decide(self._listing(300, 300), [self._ours(1, 300, 11), self._ours(2, 300, 12)])

        assert item.to_grab is False

    def test_a_blanket_covered_group_at_a_listed_size_is_not_counted(self) -> None:
        # Other's url placed nowhere, so it blankets both episodes: its files are not ours to count.
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

    def test_a_size_identified_episode_counts(self) -> None:
        # E02's untagged file is ours by its listed size, so size 200 sits on two episodes.
        no_group = sonarr_ep(1, 2, size=200, release_group=None, episode_file_id=12)

        item = self._decide(self._listing(100, 200, 300), [self._ours(1, 200, 11), no_group, self._ours(3, 300, 13)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_partial_swap_does_not_download(self) -> None:
        # Episode 3 in place keeps the all-sizes-differ fold out of it, and a swap doubles no size.
        arr_episodes = [self._ours(1, 200, 11), self._ours(2, 100, 12), self._ours(3, 300, 13)]

        item = self._decide(self._listing(100, 200, 300), arr_episodes)

        assert item.to_grab is False

    def test_a_full_swap_flags_an_upgrade_not_a_plain_download(self) -> None:
        # Every size differs from its placement, so the existing fold flags an upgrade. No size is doubled,
        # and this rule would flag an upgrade too: the two agree.
        item = self._decide(self._listing(100, 200), [self._ours(1, 200, 11), self._ours(2, 100, 12)])

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_a_doubled_unlisted_size_does_not_download(self) -> None:
        # A doubled unlisted size is the stale-copy fold's business, which one listed match holds off.
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 999, 12), self._ours(3, 999, 13)]

        item = self._decide(self._listing(100, 200, 300), arr_episodes)

        assert item.to_grab is False

    def test_a_private_groups_doubled_size_reports_private_only(self) -> None:
        result = self._plan(self._listing(100, 200, is_public=False), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert (result.skips.warned, result.seadex_dict["Era-Raws"].releases["u1"].to_grab) == (True, False)

    @staticmethod
    def _beside_public(*, is_fallback: bool) -> dict[str, GroupReleases]:
        """The private Era-Raws listing of 100 and 200 beside a public group listing the same files."""

        episodes = [ReleaseEpisode(season=1, episode=1, size=100), ReleaseEpisode(season=1, episode=2, size=200)]
        public = make_release(
            url="https://nyaa.si/view/2", size=[100, 200], episodes=episodes, infohash="h2", is_fallback=is_fallback
        )
        return {
            **TestListedSizeOnTooManyEpisodes._listing(100, 200, is_public=False),
            "Pub": rg_group({"u2": public}),
        }

    def test_a_private_doubled_pick_promotes_a_public_group_listing_the_same_files(self) -> None:
        result = self._plan(self._beside_public(is_fallback=False), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert result.seadex_dict["Pub"].releases["u2"].to_grab is True
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False
        assert result.skips.warned is False

    def test_a_private_doubled_pick_is_held_rather_than_replaced_by_a_fallback(self) -> None:
        # A fallback never replaces an owned copy of the preferred pick: the doubled hold warns instead.
        result = self._plan(self._beside_public(is_fallback=True), [self._ours(1, 200, 11), self._ours(2, 200, 12)])

        assert result.seadex_dict["Pub"].releases["u2"].to_grab is False
        assert (result.skips.warned, result.skips.outdated) == (True, True)


class TestGroupVerdicts:
    """The plan's guard verdicts: entry_groups (verified current) and stale_groups."""

    def test_no_episode_mixed_sizes_exit_the_guard_without_a_download(self) -> None:
        # A partial size overlap is not a whole-release upgrade (no grab), but
        # the copy is provably part-stale: reading the total-disjoint check as
        # "current" would protect the stale file forever.
        planner = make_planner()
        seadex = {"Blunt": rg_group({"u1": make_release(episodes=[], size=[100, 200], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Blunt": (100, 999)}),
            ep_list=None,
        )
        assert result.seadex_dict["Blunt"].releases["u1"].to_grab is False
        assert result.guards.entry_groups == ()
        assert result.guards.stale_groups == ("Blunt",)

    def test_stale_leftover_at_an_unlisted_episode_condemns_the_group(self) -> None:
        # Pick Y is current on every episode it lists, but the disk also holds
        # a Y-tagged leftover at an episode NO Y url lists. Blanket protection
        # would shield the very file the pack was grabbed to replace.
        planner = make_planner()
        seadex = {
            "Y": rg_group(
                {
                    "u1": make_release(
                        episodes=[ReleaseEpisode(season=1, episode=1, size=100)], size=[100], infohash="h1"
                    )
                },
            ),
            "X": rg_group(
                {
                    "u2": make_release(
                        episodes=[
                            ReleaseEpisode(season=1, episode=1, size=300),
                            ReleaseEpisode(season=1, episode=13, size=400),
                        ],
                        size=[300, 400],
                        infohash="h2",
                    ),
                },
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Y": (100, 999)}),
            ep_list=[
                sonarr_ep(1, 1, size=100, release_group="Y"),
                sonarr_ep(1, 13, size=999, release_group="Y"),
            ],
        )
        assert result.seadex_dict["X"].releases["u2"].to_grab is True
        assert "Y" not in result.guards.entry_groups
        assert result.guards.stale_groups == ("Y",)

    def test_blind_listing_never_reads_as_an_upgrade(self) -> None:
        # A listing with no sizes proves nothing: the held copy stands (no
        # grab, no upgrade marker), and with every listing blind the group is
        # unverifiable - in NEITHER guard list.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {"Ember": rg_group({"u1": make_release(episodes=[], size=[], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Ember": (100,)}),
            ep_list=None,
        )
        release = result.seadex_dict["Ember"].releases["u1"]
        assert release.to_grab is False
        assert release.is_upgrade is False
        assert result.guards.entry_groups == ()
        assert result.guards.stale_groups == ()

    def test_blind_url_beside_a_sized_current_one_keeps_the_group_protected(self) -> None:
        # One blind cross-seed must not void the protection the sized listing
        # earned: blind urls contribute no evidence, in either direction.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {
            "Ember": rg_group(
                {
                    "u1": make_release(episodes=[], size=[100], infohash="h1"),
                    "u2": make_release(episodes=[], size=[], infohash="h2"),
                },
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(by_group={"Ember": (100,)}),
            ep_list=None,
        )
        assert result.seadex_dict["Ember"].releases["u2"].to_grab is False
        assert result.guards.entry_groups == ("Ember",)
        assert result.guards.stale_groups == ()

    def test_groups_with_nothing_on_disk_stay_protected(self) -> None:
        # Vacuously current: a pick with no on-disk files is protected, so a
        # copy of it imported mid-wait (fresh, recommended) is not overwritten.
        planner = make_planner()
        seadex = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        episodes=[ReleaseEpisode(season=1, episode=1, size=100)], size=[100], infohash="h1"
                    )
                },
            ),
            "Kept": rg_group(
                {
                    "u2": make_release(
                        episodes=[ReleaseEpisode(season=1, episode=1, size=300)], size=[300], infohash="h2"
                    )
                },
            ),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[sonarr_ep(1, 1, episode_file_id=0)],
        )
        assert set(result.guards.entry_groups) == {"RG", "Kept"}
        assert result.guards.stale_groups == ()

    def test_sonarr_untagged_copy_holds_an_unparseable_url(self) -> None:
        # A Sonarr url whose files never parsed reaches the blunt matcher. The
        # untagged on-disk files (a rename stripped the tags) all belong to it
        # by size, so it is held rather than re-grabbed on every check.
        planner = make_planner()
        seadex = {"Era-Raws": rg_group({"u1": make_release(episodes=[], size=[100, 200], infohash="h1")})}
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(),
            ep_list=[
                sonarr_ep(1, 1, size=100, release_group=None),
                sonarr_ep(1, 2, size=200, release_group=None),
            ],
        )
        assert result.seadex_dict["Era-Raws"].releases["u1"].to_grab is False

    def test_movie_untagged_owned_copy_suppresses_the_sibling_pick(self) -> None:
        # The untagged movie file is pick A's by size. Grabbing sibling pick B
        # would replace the recommended copy the entry already owns - the
        # owned copy counts as an overlap exactly as a tagged one would.
        planner = make_planner(arr=Arr.RADARR)
        seadex = {
            "A": rg_group({"u1": make_release(episodes=[], size=[100], infohash="h1")}),
            "B": rg_group({"u2": make_release(episodes=[], size=[200], infohash="h2")}),
        }
        result = planner.plan(
            seadex_dict=seadex,
            arr_releases=ArrFiles(no_group=(100,)),
            ep_list=None,
        )
        assert result.seadex_dict["A"].releases["u1"].to_grab is False
        assert result.seadex_dict["B"].releases["u2"].to_grab is False


class TestReduceDropRescue:
    """Group-atomic drops must not lose episode coverage (the rescue pass)."""

    E11 = ReleaseEpisode(season=1, episode=1, size=100)
    E21 = ReleaseEpisode(season=2, episode=1, size=100)

    def _equal_union_pair(self, *, b_first: bool) -> dict[str, GroupReleases]:
        # Two mixed groups whose unions match (the private batches inflate both
        # keys to {S1E1, S2E1}), so they land in ONE same-files set.
        a = rg_group(
            {
                "a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1"),
                "a_priv": make_release(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            all_episodes=[self.E11, self.E21],
        )
        b = rg_group(
            {
                "b_pub": make_release(download=True, is_public=True, episodes=[self.E21], infohash="b1"),
                "b_priv": make_release(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            all_episodes=[self.E11, self.E21],
        )
        return {"B": b, "A": a} if b_first else {"A": a, "B": b}

    def test_losers_uncovered_public_url_is_rescued(self) -> None:
        # Neither mixed group is fully addable, so the keeper degrades to the
        # first. The loser's public url carries the set's only addable S2 (or S1)
        # coverage and must be re-flagged, whichever group wins.
        for b_first in (False, True):
            planner = make_planner()
            seadex = self._equal_union_pair(b_first=b_first)
            planner.reduce_overlapping_grabs(seadex)
            assert seadex["A"].releases["a_pub"].to_grab is True, f"b_first={b_first}"
            assert seadex["B"].releases["b_pub"].to_grab is True, f"b_first={b_first}"
            # Exactly one private batch survives (the keeper's). The loser's is dropped.
            priv_flags = [seadex["A"].releases["a_priv"].to_grab, seadex["B"].releases["b_priv"].to_grab]
            assert sorted(priv_flags) == [False, True], f"b_first={b_first}"

    def test_covered_drop_stays_dropped(self) -> None:
        # The loser's public url adds nothing over the keeper's surviving public
        # coverage: it stays dropped (no duplicate grab).
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11, self.E21], infohash="a1")},
                all_episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="b1")},
                # The union is inflated to match A's key (the R1 mechanic).
                all_episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_equal_coverage_drop_stays_dropped(self) -> None:
        # MUTATION PIN (_rescue_dropped_coverage): `release_keys <= survivor_keys`
        # flipped to `<` re-flags a dropped url whose coverage EQUALS the
        # survivors' - a pure duplicate grab. Equal single-episode coverage on
        # both public groups: the loser must stay dropped.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                all_episodes=[self.E11],
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="b1")},
                all_episodes=[self.E11],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_rescue_accumulates_survivor_coverage(self) -> None:
        # MUTATION PIN (_rescue_dropped_coverage): `survivor_keys |= release_keys`
        # degraded to `=` forgets the ORIGINAL survivors after the first rescue.
        # Keeper A covers S1. Dropped B (S2) is rescued. Dropped C (S1) is already
        # covered by A and must stay dropped - under `=` it would be re-flagged.
        planner = make_planner()
        shared = [self.E11, self.E21]
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                all_episodes=shared,
            ),
            "B": rg_group(
                {"b_pub": make_release(download=True, is_public=True, episodes=[self.E21], infohash="b1")},
                all_episodes=shared,
            ),
            "C": rg_group(
                {"c_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="c1")},
                all_episodes=shared,
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is True  # rescued: only S2 carrier
        assert seadex["C"].releases["c_pub"].to_grab is False  # covered by keeper A

    def test_movie_urls_never_rescued(self) -> None:
        # Movie/no-parse urls have no episode vocabulary, so the rescue can't
        # (and mustn't) reason about their coverage.
        planner = make_planner()
        seadex = {
            "A": rg_group({"a_pub": make_release(download=True, is_public=True, infohash="a1")}),
            "B": rg_group({"b_pub": make_release(download=True, is_public=True, infohash="b1")}),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_pub"].to_grab is False

    def test_owned_public_url_never_resurrected(self) -> None:
        # A matcher-unflagged (owned) url is not part of this pass's drops, so
        # the rescue never touches it - even when its coverage leaves the set.
        planner = make_planner()
        seadex = {
            "A": rg_group(
                {"a_pub": make_release(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                all_episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {
                    "b_owned": make_release(download=False, is_public=True, episodes=[self.E21], infohash="b2"),
                    "b_priv": make_release(
                        download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None
                    ),
                },
                all_episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_grabs(seadex)
        assert seadex["A"].releases["a_pub"].to_grab is True
        assert seadex["B"].releases["b_owned"].to_grab is False
        assert seadex["B"].releases["b_priv"].to_grab is False


class TestRefusedPack:
    """A pack left for a hand import is judged by what its window holds, never by its numbers."""

    _SIZES = (100, 200, 300)

    @classmethod
    def _seadex(
        cls, skip_reason: PlacementSkip = PlacementSkip.MISNUMBERED, **siblings: GroupReleases
    ) -> dict[str, GroupReleases]:
        """The pack left for a hand import for `hold`, three files at `_SIZES` placed nowhere, beside `siblings`."""

        item = make_release(size=list(cls._SIZES), episodes=[], infohash="h1")
        item.skip_reason = skip_reason
        return {"Pack": rg_group({"u1": item}), **siblings}

    @staticmethod
    def _decide(seadex: dict[str, GroupReleases], ep_list: list[SonarrEpisode]) -> Release:
        """The pack's url after the planner judges it against the window `ep_list`."""

        return make_planner().plan(seadex, arr_releases(ep_list), ep_list).seadex_dict["Pack"].releases["u1"]

    @staticmethod
    def _ours(episode: int, size: int, file_id: int) -> SonarrEpisode:
        """Special `episode` holding the pack's file `file_id` at `size`."""

        return sonarr_ep(0, episode, size=size, release_group="Pack", episode_file_id=file_id)

    @pytest.mark.parametrize("skip_reason", [PlacementSkip.MISNUMBERED, PlacementSkip.TITLE_REFUTED])
    def test_every_special_holding_a_pack_file_is_in_place(self, skip_reason: PlacementSkip) -> None:
        arr_episodes = [self._ours(n, size, 10 + n) for n, size in enumerate(self._SIZES, 1)]

        item = self._decide(self._seadex(skip_reason), arr_episodes)

        assert item.to_grab is False

    @pytest.mark.parametrize("skip_reason", [PlacementSkip.MISNUMBERED, PlacementSkip.TITLE_REFUTED])
    def test_an_empty_special_flags_the_pack_as_an_upgrade(self, skip_reason: PlacementSkip) -> None:
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 200, 12), sonarr_ep(0, 3, episode_file_id=0)]

        item = self._decide(self._seadex(skip_reason), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_files_of_an_unlisted_group_flag_the_pack(self) -> None:
        arr_episodes = [
            sonarr_ep(0, n, size=size, release_group="Other", episode_file_id=10 + n)
            for n, size in enumerate(self._SIZES, 1)
        ]

        item = self._decide(self._seadex(), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, False)

    def test_a_special_a_sibling_pick_covers_is_in_place(self) -> None:
        covered = ReleaseEpisode(season=0, episode=3, size=999)
        sibling = rg_group({"u2": make_release(url="u2", infohash="h2", episodes=[covered])}, all_episodes=[covered])
        arr_episodes = [
            self._ours(1, 100, 11),
            self._ours(2, 200, 12),
            sonarr_ep(0, 3, size=999, release_group="Sib", episode_file_id=13),
        ]

        item = self._decide(self._seadex(Sib=sibling), arr_episodes)

        assert item.to_grab is False

    def test_an_untagged_file_at_a_listed_size_is_the_packs(self) -> None:
        # A hand import whose names carry no group tag: the listed size still says whose file each special holds.
        arr_episodes = [sonarr_ep(0, n, size=size, episode_file_id=10 + n) for n, size in enumerate(self._SIZES, 1)]

        item = self._decide(self._seadex(), arr_episodes)

        assert item.to_grab is False

    def test_a_special_another_url_of_the_group_placed_is_in_place(self) -> None:
        # The group's single-file release covers the third special, so the pack has nothing left to fill there.
        covered = ReleaseEpisode(season=0, episode=3, size=999)
        seadex = self._seadex()
        seadex["Pack"] = rg_group(
            {**seadex["Pack"].releases, "u2": make_release(url="u2", infohash="h2", episodes=[covered])},
            all_episodes=[covered],
        )
        arr_episodes = [self._ours(1, 100, 11), self._ours(2, 200, 12), self._ours(3, 999, 13)]

        item = self._decide(seadex, arr_episodes)

        assert item.to_grab is False

    def test_a_pack_file_at_an_unlisted_size_flags_the_pack(self) -> None:
        arr_episodes = [self._ours(1, 150, 11), self._ours(2, 200, 12), self._ours(3, 300, 13)]

        item = self._decide(self._seadex(), arr_episodes)

        assert (item.to_grab, item.is_upgrade) == (True, True)

    def test_no_episode_list_leaves_the_pack_unflagged(self) -> None:
        result = make_planner().plan(self._seadex(), ArrFiles(), None)

        assert result.seadex_dict["Pack"].releases["u1"].to_grab is False
