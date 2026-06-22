"""Characterization tests for the download-decision engine.

This is the core domain logic extracted into ``DownloadPlanner`` in Phase 4:
``get_any_to_download``, ``reduce_overlapping_downloads``,
``filter_by_torrent_hash`` and ``filter_by_release_group``. The tests assert on
the resulting per-url ``download`` flags, the returned hash list, and the
public_only skip side effects.
"""

import logging

from tests.builders import make_arr, rg_group, sonarr_ep, url_item


class TestGetAnyToDownload:
    def test_false_when_none_flagged(self) -> None:
        seadex = {"A": rg_group({"u": url_item(download=False)})}
        assert make_arr().get_any_to_download(seadex) is False

    def test_true_when_one_flagged(self) -> None:
        seadex = {"A": rg_group({"u": url_item(download=True)})}
        assert make_arr().get_any_to_download(seadex) is True


class TestReduceOverlappingDownloads:
    def test_interactive_is_noop(self) -> None:
        arr = make_arr(interactive=True)
        seadex = {
            "A": rg_group({"u1": url_item(download=True)}),
            "B": rg_group({"u2": url_item(download=True)}),
        }
        arr.reduce_overlapping_downloads(seadex)
        assert seadex["A"]["urls"]["u1"]["download"] is True
        assert seadex["B"]["urls"]["u2"]["download"] is True

    def test_keeps_first_of_same_files(self) -> None:
        # No all_episodes -> all treated as the same files -> keep first flagged
        arr = make_arr(public_only=False)
        seadex = {
            "A": rg_group({"u1": url_item(download=True)}),
            "B": rg_group({"u2": url_item(download=True)}),
        }
        arr.reduce_overlapping_downloads(seadex)
        assert seadex["A"]["urls"]["u1"]["download"] is True
        assert seadex["B"]["urls"]["u2"]["download"] is False

    def test_public_only_prefers_public_keeper(self) -> None:
        arr = make_arr(public_only=True)
        seadex = {
            "Priv": rg_group({"u1": url_item(download=True, is_public=False)}),
            "Pub": rg_group({"u2": url_item(download=True, is_public=True)}),
        }
        arr.reduce_overlapping_downloads(seadex)
        assert seadex["Pub"]["urls"]["u2"]["download"] is True
        assert seadex["Priv"]["urls"]["u1"]["download"] is False

    def test_public_only_private_only_skips_and_flags(self) -> None:
        arr = make_arr(public_only=True)
        seadex = {"Priv": rg_group({"u1": url_item(download=True, is_public=False)})}
        arr.reduce_overlapping_downloads(seadex)
        assert seadex["Priv"]["urls"]["u1"]["download"] is False
        assert arr.public_only_skipped is True
        assert arr.public_only_groups == ["Priv"]

    def test_different_files_both_kept(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {
            "A": rg_group(
                {"u1": url_item(download=True)},
                all_episodes=[{"season": 1, "episode": 1}],
            ),
            "B": rg_group(
                {"u2": url_item(download=True)},
                all_episodes=[{"season": 1, "episode": 2}],
            ),
        }
        arr.reduce_overlapping_downloads(seadex)
        assert seadex["A"]["urls"]["u1"]["download"] is True
        assert seadex["B"]["urls"]["u2"]["download"] is True


class TestFilterByTorrentHash:
    def test_flags_uncached_hashes(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {"A": rg_group({"u1": url_item(infohash="h1", download=False)})}
        hashes, out = arr.filter_by_torrent_hash(al_id=1, seadex_dict=seadex, arr="sonarr")
        assert out["A"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]

    def test_cached_hash_not_flagged_but_still_listed(self) -> None:
        arr = make_arr(
            public_only=False,
            cache={"anilist_entries": {"sonarr": {"1": {"torrent_hashes": ["h1"]}}}},
        )
        seadex = {"A": rg_group({"u1": url_item(infohash="h1", download=False)})}
        hashes, out = arr.filter_by_torrent_hash(al_id=1, seadex_dict=seadex, arr="sonarr")
        assert out["A"]["urls"]["u1"]["download"] is False
        assert hashes == ["h1"]


class TestFilterByReleaseGroup:
    def test_new_group_no_episodes_downloads(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {"NewRG": rg_group({"u1": url_item(episodes=[], infohash="h1")})}
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"OldRG": {"size": [100]}},
            ep_list=None,
        )
        assert out["NewRG"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]

    def test_matching_group_sizes_match_no_download(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {"RG": rg_group({"u1": url_item(episodes=[], size=[100], infohash="h1")})}
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"RG": {"size": [100]}},
            ep_list=None,
        )
        assert out["RG"]["urls"]["u1"]["download"] is False
        assert hashes == []

    def test_matching_group_sizes_differ_downloads(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {"RG": rg_group({"u1": url_item(episodes=[], size=[200], infohash="h1")})}
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"RG": {"size": [100]}},
            ep_list=None,
        )
        assert out["RG"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]

    def test_episode_match_same_rg_and_size_no_download(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[{"season": 1, "episode": 1, "size": 100}], infohash="h1"),
            }),
        }
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"Era-Raws": {"size": [100]}},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert out["Era-Raws"]["urls"]["u1"]["download"] is False
        assert hashes == []

    def test_episode_different_rg_downloads(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[{"season": 1, "episode": 1, "size": 100}], infohash="h1"),
            }),
        }
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"SubsPlease": {"size": [100]}},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert out["Era-Raws"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]

    def test_episode_same_rg_all_sizes_differ_downloads(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[{"season": 1, "episode": 1, "size": 999}], infohash="h1"),
            }),
        }
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"Era-Raws": {"size": [100]}},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="Era-Raws")],
        )
        assert out["Era-Raws"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]

    def test_episodes_but_no_ep_list_skips(self) -> None:
        arr = make_arr(public_only=False)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[{"season": 1, "episode": 1, "size": 100}], infohash="h1"),
            }),
        }
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={},
            ep_list=None,
        )
        assert out["Era-Raws"]["urls"]["u1"]["download"] is False
        assert hashes == []

    def test_debug_logging_path_does_not_crash(self) -> None:
        # Exercise the debug_on=True branch (its f-strings are otherwise skipped)
        arr = make_arr(public_only=False)
        arr.logger.setLevel(logging.DEBUG)
        seadex = {
            "Era-Raws": rg_group({
                "u1": url_item(episodes=[{"season": 1, "episode": 1, "size": 100}], infohash="h1"),
            }),
        }
        hashes, out = arr.filter_by_release_group(
            seadex_dict=seadex,
            arr="sonarr",
            arr_release_dict={"SubsPlease": {"size": [100]}},
            ep_list=[sonarr_ep(1, 1, size=100, release_group="SubsPlease")],
        )
        assert out["Era-Raws"]["urls"]["u1"]["download"] is True
        assert hashes == ["h1"]
