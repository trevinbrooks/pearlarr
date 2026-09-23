# pyright: strict
# pyright: reportPrivateUsage=false
# The tests assert on the strat's private collaborators (_parse / _episodes),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Unit tests for `build_pending_seeds` over the strat's grab-time placement.

The seed-construction heart of the wait/import feature: it turns the filtered
SeaDex releases into the durable `PendingImport` records the import path later
reads, placing each grabbed video file by the same `assign_episode_ids` the
import wait runs. Built bare (no live Sonarr) with a seeded in-memory parse
cache and a warm whole-series episode list.
"""

from collections.abc import Mapping
from datetime import datetime

from pearlarr.cache import UPDATED_AT_STR_FORMAT
from pearlarr.config import Arr
from pearlarr.episode_state import EpisodeFileStatus, trusted_groups
from pearlarr.grab_placement import EntryPlacements, PendingSeedContext, SeedScope, build_pending_seeds
from pearlarr.manual_import import EntryNames, GuardFacts, OwnedEpisode, PendingImport, normalize_basename
from pearlarr.parse_records import to_parse_record
from pearlarr.placement_types import episode_index
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import EpisodeRecord, Json, ParsedFileInfo, SeadexDict, SonarrEpisode

from .builders import (
    SEP,
    FakeCacheStore,
    make_sonarr_sync,
    parsed_info,
    pending_import,
    rg_group,
    sonarr_ep,
    url_item,
)
from .fakes import FakeSonarrClient

# The grab-time parses the seed reads back, keyed by filename. A covariant
# Mapping so a plain literal passes without annotation.
type ParseCache = Mapping[str, ParsedFileInfo]

# The per-entry grab stamp, threaded through the context onto every seed.
_ADDED_AT = "2026-06-24 00:00:00"


def _rows(parses: ParseCache) -> dict[str, dict[str, Json]]:
    """The persisted parse records, stamped now so the seed's freshness window counts them."""

    stamp = datetime.now().strftime(UPDATED_AT_STR_FORMAT)
    return {name: {"fetched_at": stamp, "parse": to_parse_record(info)} for name, info in parses.items()}


def _strat(parses: ParseCache, series: list[SonarrEpisode]) -> SonarrSync:
    """A strat holding the pre-seeded parse records and series 7's whole episode list.

    Every parse resolves through that whole-series map, so an unread list
    (the bare default) refuses every map-dependent verdict.
    """

    return make_sonarr_sync(
        cache_store=FakeCacheStore(sonarr_parse=_rows(parses)),
        ep_list_cache={7: series},
    )


def _build(
    strat: SonarrSync,
    seadex_dict: SeadexDict,
    entry: PendingSeedContext,
    *,
    scope: SeedScope | None = None,
) -> dict[str, PendingImport]:
    """Place the entry's files under `scope` (by default the strat's whole series), then fold the seeds."""

    series = strat._episodes.cached_episodes(entry.series_id) or []
    scope = scope or _scope(series, series)
    placed = EntryPlacements.place(scope, strat._parse.parsed_files(seadex_dict, series_fp=""))
    return build_pending_seeds(seadex_dict, placed, entry)


def _scope(ep_list: list[SonarrEpisode], series: list[SonarrEpisode], names: EntryNames | None = None) -> SeedScope:
    """The grab-time scope: the entry's own list over the whole-series map."""

    return SeedScope(episode_index(ep_list), episode_index(series), names or EntryNames())


class TestBuildPendingSeeds:
    """`build_pending_seeds` seeds a `PendingImport` per download+hash video url.

    Filenames are placed onto episode ids through the parse cache and the
    whole-series map. Releases with no video files are skipped.
    """

    def test_seeds_only_download_with_hash(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": url_item(files=["Show - 02.mkv"], size=[2000], infohash="h2", download=False),
                    "u3": url_item(files=["Show - 03.mkv"], size=[3000], infohash=None, download=True),
                },
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
            scope=_scope(ep_list, ep_list, EntryNames("Show", ("Show", "Shou"))),
        )

        # Only the download+hash url is seeded (no download / no hash are skipped).
        assert set(seeds) == {"h1"}
        seed = seeds["h1"]
        assert seed.series_id == 7
        assert seed.al_id == 1  # part of the record's PendingKey
        assert seed.title == "Show"
        assert seed.names == EntryNames("Show", ("Show", "Shou"))
        assert seed.added_at == _ADDED_AT  # the context stamp, not a fold-side clock read
        assert seed.file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        assert seed.seadex_files == ["Show - 01.mkv"]
        # The record's own episode slice, for the wait/notification label.
        assert seed.slice_coverage == "S01 E01"
        # episode_ids is a legacy read-only fallback. New seeds never write it.
        assert seed.episode_ids == []

    def test_stale_parse_record_is_a_miss(self) -> None:
        # The seed reads the sweep's own rows under the sweep's freshness rule: a
        # row the sweep refused and could not refresh must not place a file here.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        stale = {
            "Show - 01.mkv": {
                "fetched_at": "2020-01-01 00:00:00",
                "parse": to_parse_record(parsed_info(season=1, episodes=(1,))),
            },
        }
        strat = make_sonarr_sync(
            cache_store=FakeCacheStore(sonarr_parse=stale),
            ep_list_cache={7: ep_list},
        )
        seadex_dict = {
            "RG": rg_group({"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        seeds = _build(
            strat,
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {}

    def test_seed_copies_the_plan_guard_groups(self) -> None:
        # entry_groups/stale_groups are the PLAN's verdicts, copied through
        # verbatim (see the planner's group-verdict tests for the derivation)
        # so the import guard reads exactly what the grab decision judged.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group({"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                added_at=_ADDED_AT,
                guards=GuardFacts(entry_groups=("RG", "Kept"), stale_groups=("Stale",)),
            ),
        )

        assert seeds["h1"].guards.entry_groups == ("RG", "Kept")
        assert seeds["h1"].guards.stale_groups == ("Stale",)

    def test_seed_records_the_listing_sizes(self) -> None:
        # The grabbed url's file sizes ride the record so the import can tell
        # this release's own files from a stale same-group copy.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01.mkv"], size=[1000, 50], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 02.mkv": parsed_info(season=1, episodes=(2,)),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                added_at=_ADDED_AT,
                guards=GuardFacts(entry_groups=("RG", "Kept")),
            ),
        )

        assert seeds["h1"].preowned_episode_ids == [101]

    def test_seed_carries_the_plan_identified_episodes(self) -> None:
        # The plan resolved which untagged files a pick's listed size named
        # (see the planner's owned-episodes tests). The seed persists the
        # (id, size) pairs so the import doesn't copy over the very file the
        # grab just called ours, and can re-verify the claim by size.
        ep_list = [sonarr_ep(1, 1, ep_id=101, size=1000), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 02.mkv": parsed_info(season=1, episodes=(2,))}
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(
                al_id=1,
                series_id=7,
                title="Show",
                added_at=_ADDED_AT,
                guards=GuardFacts(owned_episodes=(OwnedEpisode(101, 1000),)),
            ),
        )

        assert seeds["h1"].guards.owned_episodes == ((101, 1000),)

    def test_multi_file_pack_de_unions_flat_fallback(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 02.mkv": parsed_info(season=1, episodes=(2,)),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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

    def test_empty_entry_seeds_nothing_and_excludes_nothing(self) -> None:
        # An entry that resolved no episodes has no scope to place into. An empty
        # index must never read as "no scope" (which places against the live map).
        series = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group({"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        seeds = _build(
            _strat(parses, series),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
            scope=_scope([], series),
        )

        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].excluded_files == []

    def test_unread_series_map_seeds_nothing(self) -> None:
        # D12: the count legs need no map, but a seed is final where an import poll is retried,
        # so the whole placement waits for a served list. Two numberless files over two episodes
        # would otherwise zip in name order.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        files = ["Show - Special A.mkv", "Show - Special B.mkv"]
        parses = {name: parsed_info() for name in files}
        seadex_dict = {"RG": rg_group({"u1": url_item(files=files, size=[1000, 1000], infohash="h1", download=True)})}
        sonarr = FakeSonarrClient()
        sonarr.episodes_return = None
        strat = make_sonarr_sync(
            sonarr=sonarr, cache_store=FakeCacheStore(sonarr_parse=_rows(parses)), ep_list_cache={}
        )

        seeds = _build(
            strat,
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
            scope=_scope(ep_list, []),
        )

        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].excluded_files == []
        assert seeds["h1"].ordered_episode_ids == [101, 102]

    def test_sibling_slice_files_are_excluded_not_intended(self) -> None:
        # A Part 1 entry over a Part 1+2 pack: files resolving in the series map
        # but OUTSIDE this entry's set land in excluded_files, so map + excluded
        # account for every file and the record stays determinate (a real progress
        # bar, a deadline that re-anchors per landing file).
        ep_list = [sonarr_ep(3, 1, ep_id=101, episode_file_id=0), sonarr_ep(3, 2, ep_id=102, episode_file_id=0)]
        series = [*ep_list, sonarr_ep(3, 13, ep_id=113, episode_file_id=0)]
        parses = {
            "Show - S03E01.mkv": parsed_info(season=3, episodes=(1,)),
            "Show - S03E02.mkv": parsed_info(season=3, episodes=(2,)),
            "Show - S03E13.mkv": parsed_info(season=3, episodes=(13,)),
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

        seeds = _build(
            _strat(parses, series),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
            scope=_scope(ep_list, series),
        )

        seed = seeds["h1"]
        assert set(seed.file_episode_map) == {
            normalize_basename("Show - S03E01.mkv"),
            normalize_basename("Show - S03E02.mkv"),
        }
        assert seed.excluded_files == [normalize_basename("Show - S03E13.mkv")]

    def test_file_resolving_nowhere_in_the_series_is_not_excluded(self) -> None:
        # Its key exists nowhere in the series, so nothing proves it another
        # slice's: the record keeps it as possibly ours rather than writing it off.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - S09E09.mkv": parsed_info(season=9, episodes=(9,))}
        seadex_dict = {
            "RG": rg_group({"u1": url_item(files=["Show - S09E09.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].excluded_files == []

    def test_collision_refused_duplicate_is_excluded(self) -> None:
        # Two claimants on one episode (a v2 beside its v1): the seed takes the
        # later version AND excludes the other, so this record will never
        # import it, so completeness may account for it.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 01v2.mkv": parsed_info(season=1, episodes=(1,)),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        seed = seeds["h1"]
        assert seed.file_episode_map == {normalize_basename("Show - 01v2.mkv"): [101]}
        assert seed.excluded_files == [normalize_basename("Show - 01.mkv")]

    def test_unparsed_and_vetoed_files_stay_possibly_ours(self) -> None:
        # A file with no parse record and a full-season-vetoed zip: neither is
        # KNOWABLY another slice's, so neither is excluded - the record stays
        # indeterminate (conservative) rather than trusting an incomplete map.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            # "Show - Extra.mkv" has no cached parse at all.
            "Show - Zip.mkv": parsed_info(matched=((2, 1),), full_season=True),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        seed = seeds["h1"]
        assert set(seed.file_episode_map) == {normalize_basename("Show - 01.mkv")}
        assert seed.excluded_files == []

    def test_sibling_per_episode_torrents_get_distinct_slice_labels(self) -> None:
        # The live shape that motivated the slice: one entry, one group, one
        # torrent per episode. Identical title·group labels made "which episodes
        # imported?" unanswerable from the wait report / notification.
        ep_list = [sonarr_ep(2, 6, ep_id=101, episode_file_id=0), sonarr_ep(2, 7, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - S02E06.mkv": parsed_info(season=2, episodes=(6,)),
            "Show - S02E07.mkv": parsed_info(season=2, episodes=(7,)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - S02E06.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": url_item(files=["Show - S02E07.mkv"], size=[1000], infohash="h2", download=True),
                },
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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

        seeds = _build(
            _strat({}, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}
        assert seeds["h1"].seadex_files == ["Show - 01.mkv"]
        # Nothing claimed -> the slice is the whole window it is verified against.
        assert seeds["h1"].slice_coverage == "S01 E01"

    def test_unclaimed_record_labels_with_the_whole_window(self) -> None:
        # Two urls over one window: the parsed file claims its episode, the
        # unparsed one claims nothing, so its label names every episode it
        # is verified against rather than losing the slice.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - S01E01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": url_item(files=["Show - S01E01.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": url_item(files=["Show - Extra.mkv"], size=[1000], infohash="h2", download=True),
                },
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].display_label == f"Show{SEP}RG{SEP}S01 E01"
        assert seeds["h2"].file_episode_map == {}
        assert seeds["h2"].display_label == f"Show{SEP}RG{SEP}S01 E01-E02"

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

        seeds = _build(
            _strat({}, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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
        # the parse carries all 12 matched pairs, and none of them may seed
        # (the old behavior imported this one file as every episode).
        ep_list = [sonarr_ep(5, e, ep_id=500 + e, episode_file_id=0) for e in range(1, 13)]
        parses = {
            "Show S05 Ending.mkv": parsed_info(
                matched=tuple((5, e) for e in range(1, 13)),
            ),
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S05 Ending.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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
        parses = {
            "Show S01 Opening.mkv": parsed_info(
                matched=((1, 1), (1, 2)),
                full_season=True,
            ),
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        # Still tracked (it carries a video file), just never pre-assigned.
        assert set(seeds) == {"h1"}
        assert seeds["h1"].file_episode_map == {}

    def test_legitimate_double_episode_span_still_seeds(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01-02.mkv"): [101, 102]}

    def test_partially_resolving_span_is_not_seeded(self) -> None:
        # A double-episode file straddling the entry boundary: only episode 1
        # is in this entry's list, so seeding [101] would half-import the file.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        series = [*ep_list, sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            _strat(parses, series),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
            scope=_scope(ep_list, series),
        )

        assert seeds["h1"].file_episode_map == {}

    def test_second_claim_of_a_seeded_id_is_not_seeded(self) -> None:
        # "13" and "13v2" both parse to S02E13: the later version wins,
        # deterministically, and the other claimant is left for import-time
        # assignment (which refuses the second claim of a taken id).
        ep_list = [sonarr_ep(2, 13, ep_id=213, episode_file_id=0)]
        parses = {
            "Show - 13.mkv": parsed_info(season=2, episodes=(13,)),
            "Show - 13v2.mkv": parsed_info(season=2, episodes=(13,)),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 13v2.mkv"): [213]}

    def test_partial_collision_refuses_the_whole_later_file(self) -> None:
        # The later file claims one taken id and one free one: assignment
        # defers the whole file on any collision, so the seed refuses it whole
        # rather than seeding the free half.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2)),
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}

    def test_duplicate_leaf_names_seed_once(self) -> None:
        # The same basename in two folders collapses in the basename-keyed
        # map: the first occurrence's claim stands and the copy is refused,
        # so the map is deterministic and never double-claims the id.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
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

        seeds = _build(
            _strat(parses, ep_list),
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        # Both physical files stay tracked (the leaves list is disk truth).
        assert seeds["h1"].seadex_files == ["Show - 01.mkv", "Show - 01.mkv"]


class TestParseWriteFeedsSeeds:
    """One `/parse` per file name: the gather asks Sonarr, persists the reading, and the placement seeds by it."""

    @staticmethod
    def _cold(parse: ParsedFileInfo, ep_list: list[SonarrEpisode]) -> tuple[SonarrSync, FakeSonarrClient]:
        """A strat whose Sonarr answers every `/parse` with `parse`, its cache cold."""

        sonarr = FakeSonarrClient(parse_fn=lambda _f: parse)
        return make_sonarr_sync(sonarr=sonarr, cache_store=FakeCacheStore(), ep_list_cache={7: ep_list}), sonarr

    def test_parse_write_feeds_seed_build(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse = parsed_info(season=1, episodes=(1,), matched=((1, 1),))
        strat, sonarr = self._cold(parse, ep_list)
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            strat,
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
        )

        assert seeds["h1"].file_episode_map == {normalize_basename("Show - 01.mkv"): [101]}
        # Asked once, and persisted for the next run's gather.
        assert sonarr.parse_calls == ["Show - 01.mkv"]
        assert strat._parse.cache_store.get_sonarr_parse("Show - 01.mkv") is not None

    def test_full_season_flag_flows_from_parse_write_to_seed_refusal(self) -> None:
        # End-to-end through the real write path: a fullSeason parse persists the
        # flag on the record, and the placement refuses the file by it.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parse = parsed_info(
            matched=(
                (1, 1),
                (1, 2),
            ),
            full_season=True,
        )
        strat, _ = self._cold(parse, ep_list)
        seadex_dict = {
            "RG": rg_group(
                {"u1": url_item(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        seeds = _build(
            strat,
            seadex_dict,
            entry=PendingSeedContext(al_id=1, series_id=7, title="Show", added_at=_ADDED_AT),
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

        assert seed.statuses.by_id == {101: EpisodeFileStatus.RECOMMENDED}
