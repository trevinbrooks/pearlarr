# pyright: strict
"""Characterization tests for the download-decision engine.

This is the core domain logic extracted into ``DownloadPlanner`` in Phase 4:
``get_any_to_download``, ``reduce_overlapping_downloads``,
``filter_by_torrent_hash`` and ``filter_by_release_group``. The tests assert on
the resulting per-url ``download`` flags, the returned hash list, and the
private-only skip outcome surfaced on the :class:`PlanResult` /
:class:`PublicOnlySkips` (rather than mutated run state, as before Phase 4).
"""

import logging

from seadexarr.modules.config import Arr
from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.seadex_types import EpisodeRecord, SeadexReleaseGroupItem

from .builders import make_planner, rg_group, sonarr_ep, url_item


class TestGetAnyToDownload:
    def test_false_when_none_flagged(self) -> None:
        seadex = {"A": rg_group({"u": url_item(download=False)})}
        assert DownloadPlanner.get_any_to_download(seadex) is False

    def test_true_when_one_flagged(self) -> None:
        seadex = {"A": rg_group({"u": url_item(download=True)})}
        assert DownloadPlanner.get_any_to_download(seadex) is True


class TestReduceOverlappingDownloads:
    def test_interactive_is_noop(self) -> None:
        planner = make_planner(interactive=True)
        seadex = {
            "A": rg_group({"u1": url_item(download=True)}),
            "B": rg_group({"u2": url_item(download=True)}),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["B"].urls["u2"].download is True

    def test_keeps_first_of_same_files(self) -> None:
        # No all_episodes -> all treated as the same files -> keep first flagged
        planner = make_planner(public_only=False)
        seadex = {
            "A": rg_group({"u1": url_item(download=True)}),
            "B": rg_group({"u2": url_item(download=True)}),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["B"].urls["u2"].download is False

    def test_public_only_prefers_public_keeper(self) -> None:
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False)}),
            "Pub": rg_group({"u2": url_item(download=True, is_public=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Pub"].urls["u2"].download is True
        assert seadex["Priv"].urls["u1"].download is False
        # A preferred public keeper over a preferred private pick is unremarkable
        # (only a fallback keeper gets an INFO notice).
        assert skips.notices == []

    def test_public_only_private_only_skips_and_flags(self) -> None:
        planner = make_planner(public_only=True)
        seadex = {"Priv": rg_group({"u1": url_item(download=True, is_public=False)})}
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is True
        assert skips.groups == ["Priv"]
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].reason == "private-only (private releases not allowed)"
        assert skips.notices[0].level == logging.WARNING

    def test_private_only_set_with_unrelated_fallback_still_warns(self) -> None:
        # The fallback covers DIFFERENT files (own same-files set): it doesn't
        # excuse dropping this set, so the private set still warns and holds
        # the title while the fallback proceeds.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group(
                {"u1": url_item(download=True, is_public=False)},
                all_episodes=[EpisodeRecord(season=1, episode=1, size=0)],
            ),
            "Fall": rg_group(
                {"u2": url_item(download=True, is_public=True, is_fallback=True)},
                all_episodes=[EpisodeRecord(season=1, episode=2, size=0)],
            ),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Priv"].urls["u1"].download is False
        assert seadex["Fall"].urls["u2"].download is True
        assert skips.skipped is True
        assert skips.groups == ["Priv"]
        assert len(skips.notices) == 1
        assert skips.notices[0].level == logging.WARNING

    def test_private_only_set_with_owned_fallback_soft_skips(self) -> None:
        # Same files, and the public fallback is unflagged (the Arr already has
        # its release): drop the private pick at INFO with no skipped flag, so
        # the title can still cache as done.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False)}),
            "Fall": rg_group({"u2": url_item(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is False
        assert skips.groups == []
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].level == logging.INFO

    def test_public_only_keeper_prefers_fully_addable_group(self) -> None:
        # A mixed group (public + flagged private url) is first, but its private
        # url would be refused at add time, losing the coverage only it carries:
        # the fully-public group wins keeper regardless of order.
        planner = make_planner(public_only=True)
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": url_item(download=True, is_public=True),
                    "u2": url_item(download=True, is_public=False),
                },
            ),
            "Pub": rg_group({"u3": url_item(download=True, is_public=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Pub"].urls["u3"].download is True
        assert seadex["Mixed"].urls["u1"].download is False
        assert seadex["Mixed"].urls["u2"].download is False
        assert skips.skipped is False
        assert skips.notices == []

    def test_public_only_keeper_degrades_to_first_when_all_mixed(self) -> None:
        # No fully-addable group in the set: keep the first public group (its
        # private url then hits the add-time WARNING), matching the old order.
        planner = make_planner(public_only=True)
        seadex = {
            "MixedA": rg_group(
                {
                    "u1": url_item(download=True, is_public=True),
                    "u2": url_item(download=True, is_public=False),
                },
            ),
            "MixedB": rg_group(
                {
                    "u3": url_item(download=True, is_public=True),
                    "u4": url_item(download=True, is_public=False),
                },
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["MixedA"].urls["u1"].download is True
        assert seadex["MixedA"].urls["u2"].download is True
        assert seadex["MixedB"].urls["u3"].download is False
        assert seadex["MixedB"].urls["u4"].download is False

    def test_private_only_set_with_fallback_promotes_on_size_mismatch(self) -> None:
        # The private pick was re-flagged for a SIZE mismatch (an upgrade is
        # pending): the Arr holds a stale copy, not the fallback's files, so the
        # fallback is promoted and grabbed instead of the soft-skip firing.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False, size_mismatch=True)}),
            "Fall": rg_group({"u2": url_item(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Fall"].urls["u2"].download is True
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is False
        assert skips.groups == []
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].level == logging.INFO
        assert "falling back to Fall" in skips.notices[0].reason

    def test_promotion_prefers_fully_public_fallback_group(self) -> None:
        # Promotion only flips public urls, so a mixed fallback group (a public
        # fallback sharing its release group with a private pick) would leave
        # its private url's coverage ungrabbed: the fully-public group wins.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False, size_mismatch=True)}),
            "MixedFall": rg_group(
                {
                    "u2": url_item(download=False, is_public=False),
                    "u3": url_item(download=False, is_public=True, is_fallback=True),
                },
            ),
            "Fall": rg_group({"u4": url_item(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Fall"].urls["u4"].download is True
        assert seadex["MixedFall"].urls["u2"].download is False
        assert seadex["MixedFall"].urls["u3"].download is False
        assert seadex["Priv"].urls["u1"].download is False
        assert len(skips.notices) == 1
        assert "falling back to Fall" in skips.notices[0].reason

    def test_promotion_considers_preferred_public_group(self) -> None:
        # The size-mismatch promotion pool is any unflagged public group in the
        # set, not just fallbacks: a PREFERRED public pick with the same coverage
        # is promoted instead of the old warn-and-hold.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False, size_mismatch=True)}),
            "Pub": rg_group({"u2": url_item(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Pub"].urls["u2"].download is True
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is False
        assert skips.groups == []
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].level == logging.INFO
        # A preferred group is not a fallback, so the notice must not say so.
        assert skips.notices[0].reason == "private-only; grabbing public alternative Pub"

    def test_degraded_keeper_promotes_unflagged_public_on_size_mismatch(self) -> None:
        # A mixed group's unflagged (owned) public url makes it public_flagged,
        # so the old promotion arm was unreachable: the keeper degraded to the
        # mixed group and its stale private url was refused at add time. With a
        # size mismatch present, the unflagged public group is promoted instead.
        planner = make_planner(public_only=True)
        seadex = {
            "M": rg_group(
                {
                    "u1": url_item(download=False, is_public=True),
                    "u2": url_item(download=True, is_public=False, size_mismatch=True),
                },
            ),
            "F": rg_group({"u3": url_item(download=False, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["F"].urls["u3"].download is True
        assert seadex["M"].urls["u1"].download is False
        assert seadex["M"].urls["u2"].download is False
        assert skips.skipped is False
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["M"]
        assert skips.notices[0].level == logging.INFO
        assert skips.notices[0].reason == "remaining files private-only; falling back to F"

    def test_degraded_keeper_without_mismatch_does_not_promote(self) -> None:
        # No size mismatch means the flags may be plain missing-content grabs;
        # the unflagged public group could be genuinely owned, so the keeper
        # degrades as before and nothing is promoted.
        planner = make_planner(public_only=True)
        seadex = {
            "Mixed": rg_group(
                {
                    "u1": url_item(download=True, is_public=True),
                    "u2": url_item(download=True, is_public=False),
                },
            ),
            "P": rg_group({"u3": url_item(download=False, is_public=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Mixed"].urls["u1"].download is True
        assert seadex["Mixed"].urls["u2"].download is True
        assert seadex["P"].urls["u3"].download is False
        assert skips.notices == []

    def test_fallback_keeper_over_private_notices(self) -> None:
        # Same files: the public fallback is kept over the private preferred pick,
        # with an INFO notice naming the keeper (and no skipped flag).
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False)}),
            "Fall": rg_group({"u2": url_item(download=True, is_public=True, is_fallback=True)}),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Fall"].urls["u2"].download is True
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is False
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].level == logging.INFO
        assert "Fall" in skips.notices[0].reason

    def test_different_files_both_kept(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "A": rg_group(
                {"u1": url_item(download=True)},
                all_episodes=[EpisodeRecord(season=1, episode=1, size=0)],
            ),
            "B": rg_group(
                {"u2": url_item(download=True)},
                all_episodes=[EpisodeRecord(season=1, episode=2, size=0)],
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["B"].urls["u2"].download is True


class TestSameGroupDuplicateDedup:
    """Within one group, flagged urls with identical non-empty filesets dedup to
    the first; unknown filesets never dedup (can't prove identity)."""

    def test_identical_filesets_keep_first(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "A": rg_group(
                {
                    "u1": url_item(files=["A - S01E01.mkv"], download=True),
                    "u2": url_item(files=["A - S01E01.mkv"], download=True),
                },
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["A"].urls["u2"].download is False

    def test_promoted_fallback_duplicates_dedup(self) -> None:
        # The promotion branch flips EVERY public url of the fallback group: two
        # cross-seeded copies of one release must still yield a single grab.
        planner = make_planner(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False, size_mismatch=True)}),
            "Fall": rg_group(
                {
                    "u2": url_item(files=["F - S01E01.mkv"], download=False, is_public=True, is_fallback=True),
                    "u3": url_item(files=["F - S01E01.mkv"], download=False, is_public=True, is_fallback=True),
                },
            ),
        }
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Fall"].urls["u2"].download is True
        assert seadex["Fall"].urls["u3"].download is False
        assert seadex["Priv"].urls["u1"].download is False
        assert "falling back to Fall" in skips.notices[0].reason

    def test_different_filesets_both_kept(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "A": rg_group(
                {
                    "u1": url_item(files=["A - S01E01.mkv"], download=True),
                    "u2": url_item(files=["A - S02E01.mkv"], download=True),
                },
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["A"].urls["u2"].download is True

    def test_unknown_filesets_both_kept(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "A": rg_group(
                {
                    "u1": url_item(download=True),
                    "u2": url_item(download=True),
                },
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["u1"].download is True
        assert seadex["A"].urls["u2"].download is True


class TestFilterByTorrentHash:
    def test_flags_uncached_hashes(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"A": rg_group({"u1": url_item(infohash="h1", download=False)})}
        result = planner.filter_by_torrent_hash(seadex_dict=seadex, cached_hashes=[])
        assert result.seadex_dict["A"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_cached_hash_not_flagged_but_still_listed(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"A": rg_group({"u1": url_item(infohash="h1", download=False)})}
        result = planner.filter_by_torrent_hash(
            seadex_dict=seadex,
            cached_hashes=["h1"],
        )
        assert result.seadex_dict["A"].urls["u1"].download is False
        assert result.torrent_hashes == ["h1"]


class TestFilterByReleaseGroup:
    def test_new_group_no_episodes_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"NewRG": rg_group({"u1": url_item(episodes=[], infohash="h1")})}
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"OldRG": [100]},
            ep_list=None,
        )
        assert result.seadex_dict["NewRG"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_matching_group_sizes_match_no_download(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"RG": rg_group({"u1": url_item(episodes=[], size=[100], infohash="h1")})}
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"RG": [100]},
            ep_list=None,
        )
        assert result.seadex_dict["RG"].urls["u1"].download is False
        assert result.torrent_hashes == []

    def test_matching_group_sizes_differ_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"RG": rg_group({"u1": url_item(episodes=[], size=[200], infohash="h1")})}
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"RG": [100]},
            ep_list=None,
        )
        assert result.seadex_dict["RG"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episode_match_same_rg_and_size_no_download(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"Era-Raws": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is False
        assert result.torrent_hashes == []

    def test_episode_different_rg_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"SubsPlease": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episode_same_rg_all_sizes_differ_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=999)], infohash="h1"),
                }
            ),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"Era-Raws": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episodes_but_no_ep_list_skips(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={},
            ep_list=None,
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is False
        assert result.torrent_hashes == []

    def test_matching_group_radarr_none_size_downloads(self) -> None:
        # Radarr's release dict carries an empty size list when the movie has no
        # file. as_size_list keeps that [], which is disjoint from the real
        # SeaDex sizes, so the group is grabbed.
        planner = make_planner(public_only=False)
        seadex = {"RG": rg_group({"u1": url_item(episodes=[], size=[100], infohash="h1")})}
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.RADARR,
            arr_release_dict={"RG": []},
            ep_list=None,
        )
        assert result.seadex_dict["RG"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_debug_logging_path_does_not_crash(self) -> None:
        # Exercise the debug_on=True branch (its f-strings are otherwise skipped)
        planner = make_planner(public_only=False)
        planner.logger.setLevel(logging.DEBUG)
        seadex = {
            "Era-Raws": rg_group(
                {
                    "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
                }
            ),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr=Arr.SONARR,
            arr_release_dict={"SubsPlease": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]
        # Reset so a later test doesn't inherit DEBUG from the shared logger
        planner.logger.setLevel(logging.WARNING)


class TestReduceDropRescue:
    """Group-atomic drops must not lose episode coverage (the rescue pass)."""

    E11 = EpisodeRecord(season=1, episode=1, size=100)
    E21 = EpisodeRecord(season=2, episode=1, size=100)

    def _equal_union_pair(self, *, b_first: bool) -> dict[str, SeadexReleaseGroupItem]:
        # Two mixed groups whose unions match (the private batches inflate both
        # keys to {S1E1, S2E1}), so they land in ONE same-files set.
        a = rg_group(
            {
                "a_pub": url_item(download=True, is_public=True, episodes=[self.E11], infohash="a1"),
                "a_priv": url_item(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            all_episodes=[self.E11, self.E21],
        )
        b = rg_group(
            {
                "b_pub": url_item(download=True, is_public=True, episodes=[self.E21], infohash="b1"),
                "b_priv": url_item(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
            },
            all_episodes=[self.E11, self.E21],
        )
        return {"B": b, "A": a} if b_first else {"A": a, "B": b}

    def test_losers_uncovered_public_url_is_rescued(self) -> None:
        # Neither mixed group is fully addable, so the keeper degrades to the
        # first; the loser's public url carries the set's only addable S2 (or S1)
        # coverage and must be re-flagged, whichever group wins.
        for b_first in (False, True):
            planner = make_planner(public_only=True)
            seadex = self._equal_union_pair(b_first=b_first)
            planner.reduce_overlapping_downloads(seadex)
            assert seadex["A"].urls["a_pub"].download is True, f"b_first={b_first}"
            assert seadex["B"].urls["b_pub"].download is True, f"b_first={b_first}"
            # Exactly one private batch survives (the keeper's); the loser's is dropped.
            priv_flags = [seadex["A"].urls["a_priv"].download, seadex["B"].urls["b_priv"].download]
            assert sorted(priv_flags) == [False, True], f"b_first={b_first}"

    def test_covered_drop_stays_dropped(self) -> None:
        # The loser's public url adds nothing over the keeper's surviving public
        # coverage: it stays dropped (no duplicate grab).
        planner = make_planner(public_only=True)
        seadex = {
            "A": rg_group(
                {"a_pub": url_item(download=True, is_public=True, episodes=[self.E11, self.E21], infohash="a1")},
                all_episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {"b_pub": url_item(download=True, is_public=True, episodes=[self.E11], infohash="b1")},
                # The union is inflated to match A's key (the R1 mechanic).
                all_episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["a_pub"].download is True
        assert seadex["B"].urls["b_pub"].download is False

    def test_movie_urls_never_rescued(self) -> None:
        # Movie/no-parse urls have no episode vocabulary, so the rescue can't
        # (and mustn't) reason about their coverage.
        planner = make_planner(public_only=True)
        seadex = {
            "A": rg_group({"a_pub": url_item(download=True, is_public=True, infohash="a1")}),
            "B": rg_group({"b_pub": url_item(download=True, is_public=True, infohash="b1")}),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["a_pub"].download is True
        assert seadex["B"].urls["b_pub"].download is False

    def test_owned_public_url_never_resurrected(self) -> None:
        # A matcher-unflagged (owned) url is not part of this pass's drops, so
        # the rescue never touches it - even when its coverage leaves the set.
        planner = make_planner(public_only=True)
        seadex = {
            "A": rg_group(
                {"a_pub": url_item(download=True, is_public=True, episodes=[self.E11], infohash="a1")},
                all_episodes=[self.E11, self.E21],
            ),
            "B": rg_group(
                {
                    "b_owned": url_item(download=False, is_public=True, episodes=[self.E21], infohash="b2"),
                    "b_priv": url_item(download=True, is_public=False, episodes=[self.E11, self.E21], infohash=None),
                },
                all_episodes=[self.E11, self.E21],
            ),
        }
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["A"].urls["a_pub"].download is True
        assert seadex["B"].urls["b_owned"].download is False
        assert seadex["B"].urls["b_priv"].download is False
