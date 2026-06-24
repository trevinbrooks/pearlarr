"""Characterization tests for the download-decision engine.

This is the core domain logic extracted into ``DownloadPlanner`` in Phase 4:
``get_any_to_download``, ``reduce_overlapping_downloads``,
``filter_by_torrent_hash`` and ``filter_by_release_group``. The tests assert on
the resulting per-url ``download`` flags, the returned hash list, and the
private-only skip outcome surfaced on the :class:`PlanResult` /
:class:`PublicOnlySkips` (rather than mutated run state, as before Phase 4).
"""

import logging

from seadexarr.modules.planner import DownloadPlanner
from seadexarr.modules.seadex_types import EpisodeRecord
from tests.builders import make_planner, rg_group, sonarr_ep, url_item


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
        planner.reduce_overlapping_downloads(seadex)
        assert seadex["Pub"].urls["u2"].download is True
        assert seadex["Priv"].urls["u1"].download is False

    def test_public_only_private_only_skips_and_flags(self) -> None:
        planner = make_planner(public_only=True)
        seadex = {"Priv": rg_group({"u1": url_item(download=True, is_public=False)})}
        skips = planner.reduce_overlapping_downloads(seadex)
        assert seadex["Priv"].urls["u1"].download is False
        assert skips.skipped is True
        assert skips.groups == ["Priv"]
        assert len(skips.notices) == 1
        assert skips.notices[0].groups == ["Priv"]
        assert skips.notices[0].reason == "private-only (public_only on)"

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
            seadex_dict=seadex, cached_hashes=["h1"],
        )
        assert result.seadex_dict["A"].urls["u1"].download is False
        assert result.torrent_hashes == ["h1"]


class TestFilterByReleaseGroup:
    def test_new_group_no_episodes_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {"NewRG": rg_group({"u1": url_item(episodes=[], infohash="h1")})}
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
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
            arr="sonarr",
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
            arr="sonarr",
            arr_release_dict={"RG": [100]},
            ep_list=None,
        )
        assert result.seadex_dict["RG"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episode_match_same_rg_and_size_no_download(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
            }),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"Era-Raws": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is False
        assert result.torrent_hashes == []

    def test_episode_different_rg_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
            }),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"SubsPlease": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episode_same_rg_all_sizes_differ_downloads(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=999)], infohash="h1"),
            }),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"Era-Raws": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]

    def test_episodes_but_no_ep_list_skips(self) -> None:
        planner = make_planner(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
            }),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
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
            arr="radarr",
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
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[EpisodeRecord(season=1, episode=1, size=100)], infohash="h1"),
            }),
        }
        result = planner.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"SubsPlease": [100]},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert result.seadex_dict["Era-Raws"].urls["u1"].download is True
        assert result.torrent_hashes == ["h1"]
        # Reset so a later test doesn't inherit DEBUG from the shared logger
        planner.logger.setLevel(logging.WARNING)
