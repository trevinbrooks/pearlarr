"""Unit tests for ``SonarrSync._build_pending_seeds``.

The seed-construction heart of the wait/import feature: it turns the filtered
SeaDex releases into the durable ``PendingImport`` records the import path later
reads, mapping each grabbed video file to authoritative Sonarr episode ids via the
cached ``/parse`` results and the ``(season, episode) -> id`` index. Built bare
(no live Sonarr) with a seeded in-memory parse cache.
"""

import types

from seadexarr.modules.manual_import import normalize_basename
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import SonarrEpisode

from .builders import make_sonarr_sync, rg_group, url_item


def _strat(parse_cache: dict) -> SonarrSync:
    return make_sonarr_sync(
        cache_store=types.SimpleNamespace(data={"sonarr_parse_cache": parse_cache}),
    )


def _ep(ep_id: int, season: int, episode: int) -> SonarrEpisode:
    return SonarrEpisode.from_api(
        {"id": ep_id, "seasonNumber": season, "episodeNumber": episode},
    )


class TestBuildPendingSeeds:
    def test_seeds_only_download_with_hash(self) -> None:
        ep_list = [_ep(101, 1, 1)]
        parse_cache = {"Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]}}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": url_item(files=["Show - 02.mkv"], size=[2000], infohash="h2", download=False),
                    "u3": url_item(files=["Show - 03.mkv"], size=[3000], infohash=None, download=True),
                },
            ),
        }

        seeds = _strat(parse_cache)._build_pending_seeds(
            seadex_dict=seadex_dict, ep_list=ep_list, sonarr_series_id=7, anilist_title="Show",
        )

        # Only the download+hash url is seeded (no download / no hash are skipped).
        assert set(seeds) == {"h1"}
        seed = seeds["h1"]
        assert seed.series_id == 7
        assert seed.title == "Show"
        assert seed.file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        assert seed.seadex_files == ["Show - 01.mkv"]
        assert seed.seadex_sizes == [1000]
        assert seed.season_number == 1
        # A single-file torrent gets the flat fallback (its one file's ids).
        assert seed.episode_ids == [101]

    def test_multi_file_pack_de_unions_flat_fallback(self) -> None:
        ep_list = [_ep(101, 1, 1), _ep(102, 1, 2)]
        parse_cache = {
            "Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]},
            "Show - 02.mkv": {"episodes": [{"season": 1, "episode": 2}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 01.mkv", "Show - 02.mkv"],
                        size=[1000, 2000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._build_pending_seeds(
            seadex_dict=seadex_dict, ep_list=ep_list, sonarr_series_id=7, anilist_title="Show",
        )

        seed = seeds["h1"]
        assert seed.file_episode_map == {
            normalize_basename("Show - 01.mkv"): [101],
            normalize_basename("Show - 02.mkv"): [102],
        }
        # The cross-file union bug fix: a multi-file pack carries NO flat fallback,
        # so the single-file rule can never stamp the whole season onto one file.
        assert seed.episode_ids == []
        assert seed.season_number == 1

    def test_mixed_seasons_leaves_season_number_none(self) -> None:
        ep_list = [_ep(101, 1, 1), _ep(201, 2, 1)]
        parse_cache = {
            "S01.mkv": {"episodes": [{"season": 1, "episode": 1}]},
            "S02.mkv": {"episodes": [{"season": 2, "episode": 1}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["S01.mkv", "S02.mkv"], size=[1, 2], infohash="h1", download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._build_pending_seeds(
            seadex_dict=seadex_dict, ep_list=ep_list, sonarr_series_id=7, anilist_title="Show",
        )

        assert seeds["h1"].season_number is None

    def test_unparsed_video_still_seeded_for_import_time_repair(self) -> None:
        # No grab-time parse hit -> an empty map, but the seed is STILL persisted
        # (it carries a video file) so the import-time repair can map it later.
        ep_list = [_ep(101, 1, 1)]
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True),
                },
            ),
        }

        seeds = _strat({})._build_pending_seeds(
            seadex_dict=seadex_dict, ep_list=ep_list, sonarr_series_id=7, anilist_title="Show",
        )

        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].seadex_files == ["Show - 01.mkv"]

    def test_no_video_files_is_not_seeded(self) -> None:
        # A release with only non-video files (subs) has nothing to import.
        ep_list = [_ep(101, 1, 1)]
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - 01.ass"], size=[10], infohash="h1", download=True),
                },
            ),
        }

        seeds = _strat({})._build_pending_seeds(
            seadex_dict=seadex_dict, ep_list=ep_list, sonarr_series_id=7, anilist_title="Show",
        )

        assert seeds == {}
