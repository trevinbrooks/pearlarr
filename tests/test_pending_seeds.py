# pyright: strict
# pyright: reportPrivateUsage=false
# The tests assert on the strat's private collaborators (_parse / _reconciler),
# which strict re-flags; the repo disables reportPrivateUsage for tests.
"""Unit tests for ``ImportReconciler.build_pending_seeds`` (via the strat).

The seed-construction heart of the wait/import feature: it turns the filtered
SeaDex releases into the durable ``PendingImport`` records the import path later
reads, mapping each grabbed video file to authoritative Sonarr episode ids via the
cached ``/parse`` results and the ``(season, episode) -> id`` index. Built bare
(no live Sonarr) with a seeded in-memory parse cache.
"""

from seadexarr.modules.manual_import import normalize_basename
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import SonarrEpisode

from .builders import FakeCacheStore, make_config, make_logger, make_sonarr_sync, rg_group, url_item

# One persisted ``/parse`` cache shape: ``filename -> {"episodes": [{season, episode}]}``.
# The seed builder reads ``record["episodes"]`` straight off this (no freshness stamp),
# so the test records carry only that key.
type ParseCache = dict[str, dict[str, list[dict[str, int]]]]


class _FakeSonarrParse:
    """Stands in for the ``SonarrClient`` the parse cache calls.

    ``SonarrParseCache.parse_episodes_from_seadex`` only touches ``sonarr.parse``;
    this scripts that one result so the parse pass populates the shared cache itself.
    """

    def __init__(self, result: list[dict[str, int]] | None) -> None:
        self._result = result

    def parse(self, filename: str) -> list[dict[str, int]] | None:
        del filename
        return self._result


def _strat(parse_cache: ParseCache) -> SonarrSync:
    return make_sonarr_sync(
        cache_store=FakeCacheStore(sonarr_parse=parse_cache),
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

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            sonarr_series_id=7,
            anilist_title="Show",
        )

        # Only the download+hash url is seeded (no download / no hash are skipped).
        assert set(seeds) == {"h1"}
        seed = seeds["h1"]
        assert seed.series_id == 7
        assert seed.title == "Show"
        assert seed.file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        assert seed.seadex_files == ["Show - 01.mkv"]
        assert seed.seadex_sizes == [1000]
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

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            sonarr_series_id=7,
            anilist_title="Show",
        )

        seed = seeds["h1"]
        assert seed.file_episode_map == {
            normalize_basename("Show - 01.mkv"): [101],
            normalize_basename("Show - 02.mkv"): [102],
        }
        # The cross-file union bug fix: a multi-file pack carries NO flat fallback,
        # so the single-file rule can never stamp the whole season onto one file.
        assert seed.episode_ids == []

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

        seeds = _strat({})._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            sonarr_series_id=7,
            anilist_title="Show",
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

        seeds = _strat({})._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            sonarr_series_id=7,
            anilist_title="Show",
        )

        assert seeds == {}


class TestParseWriteVisibleToSeeds:
    """The parse cache (writer) and the seed builder (reader) are now separate
    objects; they must share one ``cache_store`` so a parse write earlier in the
    run is visible to the seed read - the staged-write invariant the split risks."""

    def test_parse_write_feeds_seed_build(self) -> None:
        sonarr = _FakeSonarrParse([{"season": 1, "episode": 1}])
        # No pre-seed: the parse pass must populate the shared cache itself.
        strat = make_sonarr_sync(
            sonarr=sonarr,
            _config=make_config(sleep_time=2),  # sequential: deterministic, no warm pool
            cache_store=FakeCacheStore(),
            logger=make_logger(),
        )
        ep_list = [_ep(101, 1, 1)]
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        # Writer: fills the SHARED cache_store via the parse collaborator.
        strat._parse.parse_episodes_from_seadex(seadex_dict, series_fp="fp")
        # Reader: the seed builder reads that record back out of the same store.
        seeds = strat._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            sonarr_series_id=7,
            anilist_title="Show",
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
