# pyright: strict
# pyright: reportPrivateUsage=false
# The tests assert on the strat's private collaborators (_parse / _reconciler),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Unit tests for `ImportReconciler.build_pending_seeds` (via the strat).

The seed-construction heart of the wait/import feature: it turns the filtered
SeaDex releases into the durable `PendingImport` records the import path later
reads, mapping each grabbed video file to authoritative Sonarr episode ids via the
cached `/parse` results and the `(season, episode) -> id` index. Built bare
(no live Sonarr) with a seeded in-memory parse cache.
"""

from collections.abc import Mapping

from pearlarr.config import Arr
from pearlarr.manual_import import GuardFacts, normalize_basename
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import EpisodeRecord, ParsedEpisode
from pearlarr.sonarr_import import PendingSeedContext
from pearlarr.sonarr_import_plan import EpisodeFileStatus, trusted_groups

from .builders import (
    SEP,
    FakeCacheStore,
    make_config,
    make_sonarr_sync,
    pending_import,
    rg_group,
    sonarr_ep,
    url_item,
)
from .fakes import FakeSonarrClient

# One persisted `/parse` cache shape: `filename -> {"episodes": [{season, episode}]}`,
# plus the optional `full_season` bool the grab-time seed guard reads. The seed
# builder reads both straight off this (no freshness stamp). A covariant Mapping
# so a records-only literal and a full-season one both pass without annotation.
type ParseCache = Mapping[str, Mapping[str, object]]


def _strat(parse_cache: ParseCache) -> SonarrSync:
    return make_sonarr_sync(
        cache_store=FakeCacheStore(sonarr_parse={name: dict(rec) for name, rec in parse_cache.items()}),
    )


class TestBuildPendingSeeds:
    """`build_pending_seeds` seeds a `PendingImport` per download+hash video url.

    Filenames map to episode ids via the parse cache (falling back flat when
    unparsed). Releases with no video files are skipped.
    """

    def test_seeds_only_download_with_hash(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        # Only the download+hash url is seeded (no download / no hash are skipped).
        assert set(seeds) == {"h1"}
        seed = seeds["h1"]
        assert seed.series_id == 7
        assert seed.al_id == 1  # part of the record's PendingKey
        assert seed.title == "Show"
        assert seed.file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        assert seed.seadex_files == ["Show - 01.mkv"]
        # The record's own episode slice, for the wait/notification label.
        assert seed.slice_coverage == "S01 E01"
        # episode_ids is a legacy read-only fallback. New seeds never write it.
        assert seed.episode_ids == []

    def test_seed_copies_the_plan_guard_groups(self) -> None:
        # entry_groups/stale_groups are the PLAN's verdicts, copied through
        # verbatim (see the planner's group-verdict tests for the derivation)
        # so the import guard reads exactly what the grab decision judged.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {"Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]}}
        seadex_dict = {
            "RG": rg_group({"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                guards=GuardFacts(entry_groups=("RG", "Kept"), stale_groups=("Stale",)),
            ),
        )

        assert seeds["h1"].guards.entry_groups == ("RG", "Kept")
        assert seeds["h1"].guards.stale_groups == ("Stale",)

    def test_seed_records_the_listing_sizes(self) -> None:
        # The grabbed url's file sizes ride the record so the import can tell
        # this release's own files from a stale same-group copy.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {"Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]}}
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01.mkv"], size=[1000, 50], infohash="h1", download=True)},
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].release_sizes == [1000, 50]

    def test_seed_marks_targets_already_holding_a_pick(self) -> None:
        # A target already holding another pick's file at grab time was never
        # this torrent's to insert: it lands in preowned_episode_ids so the
        # wait's inserted counts start at zero, not at the pre-existing files.
        ep_list = [
            sonarr_ep(1, 1, ep_id=101, size=500, release_group="Kept"),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
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
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                guards=GuardFacts(entry_groups=("RG", "Kept")),
            ),
        )

        assert seeds["h1"].preowned_episode_ids == [101]

    def test_seed_carries_the_plan_identified_episodes(self) -> None:
        # The plan resolved which untagged files a pick's listed size named
        # (see the planner's owned-episodes tests). The seed persists the
        # (id, size) pairs so the import doesn't copy over the very file the
        # grab just called ours - and can re-verify the claim by size.
        ep_list = [sonarr_ep(1, 1, ep_id=101, size=1000), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parse_cache = {"Show - 02.mkv": {"episodes": [{"season": 1, "episode": 2}]}}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 02.mkv"],
                        size=[2000],
                        infohash="h1",
                        download=True,
                        episodes=[EpisodeRecord(season=1, episode=2, size=2000)],
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                guards=GuardFacts(owned_episodes=((101, 1000),)),
            ),
        )

        assert seeds["h1"].guards.owned_episodes == ((101, 1000),)

    def test_multi_file_pack_de_unions_flat_fallback(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        seed = seeds["h1"]
        assert seed.file_episode_map == {
            normalize_basename("Show - 01.mkv"): [101],
            normalize_basename("Show - 02.mkv"): [102],
        }
        assert seed.slice_coverage == "S01 E01-E02"
        # No seed ever carries the flat fallback (it's legacy read-only), so the
        # old cross-file union bug (a whole season stamped onto one file) is out.
        assert seed.episode_ids == []

    def test_sibling_slice_files_are_excluded_not_intended(self) -> None:
        # A Part 1 entry over a Part 1+2 pack: files parsing cleanly OUTSIDE
        # this entry's set land in excluded_files, so map + excluded account
        # for every file and the record stays determinate (a real progress
        # bar, a deadline that re-anchors per landing file).
        ep_list = [sonarr_ep(3, 1, ep_id=101, episode_file_id=0), sonarr_ep(3, 2, ep_id=102, episode_file_id=0)]
        parse_cache = {
            "Show - S03E01.mkv": {"episodes": [{"season": 3, "episode": 1}]},
            "Show - S03E02.mkv": {"episodes": [{"season": 3, "episode": 2}]},
            "Show - S03E13.mkv": {"episodes": [{"season": 3, "episode": 13}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - S03E01.mkv", "Show - S03E02.mkv", "Show - S03E13.mkv"],
                        size=[1000, 1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        seed = seeds["h1"]
        assert set(seed.file_episode_map) == {
            normalize_basename("Show - S03E01.mkv"),
            normalize_basename("Show - S03E02.mkv"),
        }
        assert seed.excluded_files == [normalize_basename("Show - S03E13.mkv")]

    def test_collision_refused_duplicate_is_excluded(self) -> None:
        # A second claimant on an already-claimed episode (a v2 duplicate): the
        # seed refuses it (first claim wins) AND excludes it - this record will
        # never import it, so completeness may account for it.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {
            "Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]},
            "Show - 01v2.mkv": {"episodes": [{"season": 1, "episode": 1}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 01.mkv", "Show - 01v2.mkv"],
                        size=[1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        seed = seeds["h1"]
        assert seed.file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        assert seed.excluded_files == [normalize_basename("Show - 01v2.mkv")]

    def test_unparsed_and_vetoed_files_stay_possibly_ours(self) -> None:
        # A file with no parse record and a full-season-vetoed zip: neither is
        # KNOWABLY another slice's, so neither is excluded - the record stays
        # indeterminate (conservative) rather than trusting an incomplete map.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {
            "Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]},
            # "Show - Extra.mkv" has no cached parse at all.
            "Show - Zip.mkv": {"episodes": [{"season": 2, "episode": 1}], "full_season": True},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 01.mkv", "Show - Extra.mkv", "Show - Zip.mkv"],
                        size=[1000, 1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        seed = seeds["h1"]
        assert set(seed.file_episode_map) == {normalize_basename("Show - 01.mkv")}
        assert seed.excluded_files == []

    def test_sibling_per_episode_torrents_get_distinct_slice_labels(self) -> None:
        # The live shape that motivated the slice: one entry, one group, one
        # torrent per episode. Identical title·group labels made "which episodes
        # imported?" unanswerable from the wait report / notification.
        ep_list = [sonarr_ep(2, 6, ep_id=101, episode_file_id=0), sonarr_ep(2, 7, ep_id=102, episode_file_id=0)]
        parse_cache = {
            "Show - S02E06.mkv": {"episodes": [{"season": 2, "episode": 6}]},
            "Show - S02E07.mkv": {"episodes": [{"season": 2, "episode": 7}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - S02E06.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": url_item(files=["Show - S02E07.mkv"], size=[1000], infohash="h2", download=True),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].display_label == f"Show{SEP}RG{SEP}S02 E06"
        assert seeds["h2"].display_label == f"Show{SEP}RG{SEP}S02 E07"

    def test_unparsed_video_still_seeded_for_import_time_repair(self) -> None:
        # No grab-time parse hit -> an empty map, but the seed is STILL persisted
        # (it carries a video file) so the import-time repair can map it later.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].seadex_files == ["Show - 01.mkv"]
        # Nothing claimed -> no slice (the label falls back to title · group).
        assert seeds["h1"].slice_coverage is None

    def test_no_video_files_is_not_seeded(self) -> None:
        # A release with only non-video files (subs) has nothing to import.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds == {}


class TestSeedGuards:
    """The grab-time mirror of the import-side borrow gates.

    A full-season parse never seeds, colliding claims resolve first-wins in
    SeaDex file order, and a duplicate leaf seeds once. A refused file is left
    unseeded so import-time assignment places or refuses it under the full
    guard set.
    """

    def test_full_season_parse_never_seeds(self) -> None:
        # An OP/ED whose bare-"S05" name Sonarr matches to the whole season:
        # the parse record carries all 12 pairs, and none of them may seed
        # (the old behavior imported this one file as every episode).
        ep_list = [sonarr_ep(5, e, ep_id=500 + e, episode_file_id=0) for e in range(1, 13)]
        parse_cache = {
            "Show S05 Ending.mkv": {"episodes": [{"season": 5, "episode": e} for e in range(1, 13)]},
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S05 Ending.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        # Still tracked (it carries a video file), just never pre-assigned.
        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].seadex_files == ["Show S05 Ending.mkv"]

    def test_small_full_season_parse_never_seeds(self) -> None:
        # A bare-"S01" OP/ED Sonarr matched to a whole season of only two
        # episodes: the pair count slips under the span cap, so Sonarr's own
        # fullSeason flag on the record is what keeps it from seeding.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parse_cache = {
            "Show S01 Opening.mkv": {
                "episodes": [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}],
                "full_season": True,
            },
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        # Still tracked (it carries a video file), just never pre-assigned.
        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}

    def test_legitimate_double_episode_span_still_seeds(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parse_cache = {
            "Show - 01-02.mkv": {"episodes": [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01-02.mkv"): [101, 102]}

    def test_partially_resolving_span_is_not_seeded(self) -> None:
        # A double-episode file straddling the entry boundary: only episode 1
        # is in this entry's list, so seeding [101] would half-import the file.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {
            "Show - 01-02.mkv": {"episodes": [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {}

    def test_second_claim_of_a_seeded_id_is_not_seeded(self) -> None:
        # "13" and "13v2" both parse to S02E13: the first file in SeaDex order
        # wins, deterministically, and the later claimant is left for
        # import-time assignment (which refuses the second claim of a taken id).
        ep_list = [sonarr_ep(2, 13, ep_id=213, episode_file_id=0)]
        parse_cache = {
            "Show - 13.mkv": {"episodes": [{"season": 2, "episode": 13}]},
            "Show - 13v2.mkv": {"episodes": [{"season": 2, "episode": 13}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 13.mkv", "Show - 13v2.mkv"],
                        size=[1000, 1001],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 13.mkv"): [213]}

    def test_partial_collision_refuses_the_whole_later_file(self) -> None:
        # The later file claims one taken id and one free one: assignment
        # defers the whole file on any collision, so the seed refuses it whole
        # rather than seeding the free half.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parse_cache = {
            "Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]},
            "Show - 01-02.mkv": {"episodes": [{"season": 1, "episode": 1}, {"season": 1, "episode": 2}]},
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["Show - 01.mkv", "Show - 01-02.mkv"],
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}

    def test_duplicate_leaf_names_seed_once(self) -> None:
        # The same basename in two folders collapses in the basename-keyed
        # map: the first occurrence's claim stands and the copy is refused,
        # so the map is deterministic and never double-claims the id.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse_cache = {"Show - 01.mkv": {"episodes": [{"season": 1, "episode": 1}]}}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(
                        files=["FolderA/Show - 01.mkv", "FolderB/Show - 01.mkv"],
                        size=[1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        seeds = _strat(parse_cache)._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        # Both physical files stay tracked (the leaves list is disk truth).
        assert seeds["h1"].seadex_files == ["Show - 01.mkv", "Show - 01.mkv"]


class TestParseWriteVisibleToSeeds:
    """The parse cache (writer) and the seed builder (reader) are now separate objects.

    They must share one `cache_store` so a parse write earlier in the run is
    visible to the seed read - the staged-write invariant the split risks.
    """

    def test_parse_write_feeds_seed_build(self) -> None:
        sonarr = FakeSonarrClient(parse=[ParsedEpisode(season=1, episode=1)])
        # No pre-seed: the parse pass must populate the shared cache itself.
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(sleep_time=2),  # sequential: deterministic, no warm pool
            cache_store=FakeCacheStore(),
        )
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
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
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}

    def test_full_season_flag_flows_from_parse_write_to_seed_refusal(self) -> None:
        # End-to-end through the real write path: a fullSeason parse persists the
        # flag on the record, and the seed builder refuses the file by it.
        sonarr = FakeSonarrClient(
            parse=[ParsedEpisode(season=1, episode=1), ParsedEpisode(season=1, episode=2)],
            parse_full_season=True,
        )
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(sleep_time=2),  # sequential: deterministic, no warm pool
            cache_store=FakeCacheStore(),
        )
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        strat._parse.parse_episodes_from_seadex(seadex_dict, series_fp="fp")
        seeds = strat._reconciler.build_pending_seeds(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show"),
        )

        assert seeds["h1"].file_episode_map == {}


class TestNetInsertedCounts:
    """Preowned targets never count as files this torrent inserted."""

    def test_progress_counts_exclude_preowned_targets(self) -> None:
        # 101 already held another pick's file at grab time (preowned); only
        # 102 is this torrent's to insert. The bar must read 0 of 1, not 1 of 2
        # - a pre-existing file shown as inserted overstates the import.
        sonarr = FakeSonarrClient(
            episodes=[
                sonarr_ep(1, 1, ep_id=101, size=500, release_group="Kept"),
                sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
            ]
        )
        strat = make_sonarr_sync(sonarr=sonarr, cache_store=FakeCacheStore())
        pending = pending_import(
            series_id=7,
            file_episode_map={"Show - 01.mkv": [101], "Show - 02.mkv": [102]},
            episode_ids=[],
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            guards=GuardFacts(entry_groups=("Kept",)),
            preowned_episode_ids=[101],
        )

        progress = strat._reconciler.import_progress(pending)

        assert progress.determinate is True
        assert (progress.done, progress.total) == (0, 1)


class TestTrustedGroups:
    """The per-group trust policy: own group + own entry picks + sibling GRABBED groups only."""

    def test_sibling_entry_picks_never_contaminate(self) -> None:
        # Another entry's record contributes its grabbed group, never its pick
        # list: a group recommended for one season says nothing about another
        # season's episodes.
        sibling = pending_import(
            infohash="s1",
            al_id=999,
            release_group="SibGrab",
            guards=GuardFacts(entry_groups=("SibGrab", "SibPick")),
        )
        own = pending_import(release_group="Ours", guards=GuardFacts(entry_groups=("Ours", "OtherPick")))

        assert set(trusted_groups(own, [sibling])) == {"ours", "otherpick", "sibgrab"}

    def test_sibling_vote_refused_for_a_group_this_plan_judged_stale(self) -> None:
        # A sibling record grabbed group B earlier; THIS entry's plan judged its
        # on-disk B copy stale and excluded it from entry_groups. The sibling's
        # vote must not re-admit B, or the stale files it shields read done and
        # the replacement import never happens.
        sibling = pending_import(infohash="s1", al_id=999, release_group="B")
        own = pending_import(
            release_group="Ours",
            guards=GuardFacts(entry_groups=("Ours",), stale_groups=("B",)),
        )

        assert set(trusted_groups(own, [sibling])) == {"ours"}

    def test_own_group_survives_its_own_stale_verdict(self) -> None:
        # A same-group size upgrade lists its own group stale. The group must
        # stay recommended (it is the identity of the files being imported) -
        # the stale copies are told apart by size instead.
        own = pending_import(release_group="Ours", guards=GuardFacts(stale_groups=("Ours",)), release_sizes=[1000])

        assert trusted_groups(own) == {"ours": frozenset({1000})}

    def test_own_group_without_sizes_is_trusted_by_name(self) -> None:
        # No listed sizes (an older record, or a blind listing) means no size
        # gate: the None value tells the classifier to trust the name alone.
        own = pending_import(release_group="Ours")

        assert trusted_groups(own) == {"ours": None}

    def test_own_group_sizes_union_same_group_siblings(self) -> None:
        # Two records grabbing the same group (a per-cour torrent each) each
        # list their own sizes; a file either record's listing carries is a
        # current copy, so the size gate reads their union.
        sibling = pending_import(infohash="s1", al_id=999, release_group="Ours", release_sizes=[2000])
        own = pending_import(release_group="Ours", release_sizes=[1000])

        assert trusted_groups(own, [sibling])["ours"] == frozenset({1000, 2000})

    def test_seed_statuses_read_sibling_votes_from_the_store(self) -> None:
        # The store round trip behind the pure fold: a sibling record persisted
        # by an earlier grab rehydrates and its group protects the on-disk file.
        sibling = pending_import(infohash="s1", al_id=999, release_group="SibGrab")
        store = FakeCacheStore(pending={str(Arr.SONARR): {sibling.key: sibling.to_json()}})
        sonarr = FakeSonarrClient(episodes=[sonarr_ep(1, 1, ep_id=101, size=500, release_group="SibGrab")])
        strat = make_sonarr_sync(sonarr=sonarr, cache_store=store)
        own = pending_import(release_group="Ours", file_episode_map={"Show - 01.mkv": [101]})

        seed = strat._reconciler._seed_statuses(own, [101])

        assert seed.statuses == {101: EpisodeFileStatus.RECOMMENDED}
