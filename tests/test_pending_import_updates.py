# pyright: strict
# pyright: reportPrivateUsage=false
# The tests assert on the strat's private collaborators (_parse / _episodes),
# which strict re-reports. The repo disables reportPrivateUsage for tests.
"""Unit tests for `build_pending_import_updates` over the strat's grab-time placement.

The file-map-building heart of the wait/import feature: it turns the filtered
SeaDex releases into the `PendingImportUpdate`s the pipeline writes into the durable
`PendingImport`s the import path later reads, placing each grabbed video
file by `place_release`, whose file map the import imports as is. Built standalone (no live
Sonarr) with a pre-filled in-memory parse cache and a warm whole-series episode list.
"""

from collections.abc import Mapping, Sequence

from pearlarr.arr_models import ParsedFileInfo, SonarrEpisode
from pearlarr.boundary import Json
from pearlarr.config import Arr
from pearlarr.episode_state import EpisodeFileStatus, GrabbedGroups, TrustedGroups, trusted_groups
from pearlarr.file_paths import normalize_name
from pearlarr.grab_placement import (
    EntryPlacements,
    PendingImportUpdate,
    TitleFacts,
    build_pending_import_updates,
)
from pearlarr.parse_cache import parse_to_json
from pearlarr.pending_import import OwnedAtGrab, PendingImport, SizeMatchedEpisode, TitleNames
from pearlarr.placement_types import GrabbingTitle, episode_index
from pearlarr.releases import ReleaseEpisode, SeadexDict
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.stamps import now_stamp

from .builders import (
    SEP,
    FakeCacheStore,
    make_grabbing_title,
    make_release,
    make_sonarr_sync,
    new_torrent,
    parsed_info,
    pending_import,
    rg_group,
    sonarr_ep,
    title_facts,
)
from .fakes import FakeSonarrClient

# The grab-time parses the placement reads back, keyed by filename. A covariant
# Mapping so a plain literal passes without annotation.
type ParseCache = Mapping[str, ParsedFileInfo]

# The pipeline's stamp, where a test persists an update as its pending import.
_ADDED_AT = "2026-06-24 00:00:00"


def _rows(parses: ParseCache) -> dict[str, dict[str, Json]]:
    """The persisted parse rows, stamped now so the placement's freshness window counts them."""

    stamp = now_stamp()
    return {name: {"fetched_at": stamp, "parse": parse_to_json(info)} for name, info in parses.items()}


def _strat(parses: ParseCache, series: list[SonarrEpisode]) -> SonarrStrategy:
    """A strat holding the pre-filled parse rows and series 7's whole episode list.

    Every parse resolves through that whole-series index, so an unread list
    (the empty default) rejects every parse that needs the index.
    """

    return make_sonarr_sync(
        cache_store=FakeCacheStore(sonarr_parse=_rows(parses)),
        ep_list_cache={7: series},
    )


def _build(
    strat: SonarrStrategy,
    seadex_dict: SeadexDict,
    facts: TitleFacts,
    *,
    grabbing_title: GrabbingTitle | None = None,
) -> dict[str, PendingImportUpdate]:
    """Place the entry's files under `grabbing_title` (by default the strat's whole series), then build the updates."""

    series = strat._episodes.cached_episodes(facts.series_id) or []
    grabbing_title = grabbing_title or _grabbing_title(series, series)
    files = strat._parse.parsed_files(seadex_dict, series_hash="")
    placed = EntryPlacements.place(grabbing_title, files, dict.fromkeys(files, new_torrent(grabbing_title)))
    return build_pending_import_updates(seadex_dict, placed, facts)


def _record(update: PendingImportUpdate) -> PendingImport:
    """The update stored at the grab stamp, for the views only the pending import has (label, file map episodes)."""

    return update.pending_import_at(_ADDED_AT, fresh=True)


def _grabbing_title(
    ep_list: list[SonarrEpisode], series: list[SonarrEpisode], names: TitleNames | None = None
) -> GrabbingTitle:
    """The grab-time title: title 1's list over the whole-series index."""

    return make_grabbing_title(episode_index(ep_list), episode_index(series), names=names or TitleNames())


def _trust(own: PendingImport, siblings: Sequence[PendingImport] = ()) -> TrustedGroups:
    """The trusted groups of `own`'s one claim: its plan's guards, the pending import's group, the series' siblings."""

    return trusted_groups(own.claims[0].guards, GrabbedGroups(own.torrent_group, [s.torrent_group for s in siblings]))


class TestBuildPendingImportUpdates:
    """`build_pending_import_updates` builds a `PendingImportUpdate` per download+hash video release.

    Filenames are placed onto episode ids through the parse cache and the
    whole-series index. Releases with no video files are skipped.
    """

    def test_builds_updates_only_for_grabs_with_a_hash(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": make_release(files=["Show - 02.mkv"], size=[2000], infohash="h2", download=False),
                    "u3": make_release(files=["Show - 03.mkv"], size=[3000], infohash=None, download=True),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
            grabbing_title=_grabbing_title(ep_list, ep_list, TitleNames("Show", ("Show", "Shou"))),
        )

        # Only the download+hash release gets an update (no download / no hash are skipped).
        assert set(updates) == {"h1"}
        update = updates["h1"]
        claim = update.claim
        assert claim.series_id == 7
        assert claim.al_id == 1  # the title's guard-row key
        assert claim.title == "Show"
        assert claim.names == TitleNames("Show", ("Show", "Shou"))
        assert update.pending_import is None
        pending = update.pending_import_at(_ADDED_AT, fresh=True)
        assert pending.added_at == _ADDED_AT
        assert pending.claims == (claim,)
        assert dict(update.file_map) == {normalize_name("Show - 01.mkv"): (101,)}
        assert update.facts.seadex_files == ("Show - 01.mkv",)
        # The claim's episode coverage, for the wait/notification label, and the window it is checked under.
        assert claim.claimed_coverage == "S01 E01"
        assert claim.ordered_episode_ids == (101,)

    def test_stale_parse_row_is_an_unread_parse(self) -> None:
        # The placement reads the sweep's rows under the sweep's freshness rule: a
        # row the sweep rejected and could not refresh must not place a file here.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        stale = {
            "Show - 01.mkv": {
                "fetched_at": "2020-01-01 00:00:00",
                "parse": parse_to_json(parsed_info(season=1, episodes=(1,))),
            },
        }
        strat = make_sonarr_sync(
            cache_store=FakeCacheStore(sonarr_parse=stale),
            ep_list_cache={7: ep_list},
        )
        seadex_dict = {
            "RG": rg_group({"u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        updates = _build(
            strat,
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {}

    def test_update_copies_the_plan_guard_groups(self) -> None:
        # entry_groups/stale_groups are the PLAN's findings, copied through
        # verbatim (see the planner's group-verdict tests for the derivation)
        # so the import guard reads exactly what the grab decision found.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group({"u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(
                guards=OwnedAtGrab(entry_groups=("RG", "Kept"), stale_groups=("Stale",)),
            ),
        )

        claim = updates["h1"].claim
        assert claim.guards.entry_groups == ("RG", "Kept")
        assert claim.guards.stale_groups == ("Stale",)

    def test_update_records_the_listing_sizes(self) -> None:
        # The grabbed release's file sizes go with the pending import so the import can tell
        # this release's files from an outdated same-group copy.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show - 01.mkv"], size=[1000, 50], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert updates["h1"].facts.release_sizes == (1000, 50)
        assert updates["h1"].facts.sizes_by_name == {"show - 01.mkv": 1000}
        assert _record(updates["h1"]).sizes_by_name == {"show - 01.mkv": 1000}

    def test_update_marks_file_map_episodes_already_holding_a_preferred_release(self) -> None:
        # An episode already with another preferred release's file at grab time was never
        # this torrent's to import: it goes in preowned_episode_ids so the
        # wait's done counts start at zero, not at the pre-existing files.
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
                    "u1": make_release(
                        files=["Show - 01.mkv", "Show - 02.mkv"],
                        size=[1000, 2000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(
                guards=OwnedAtGrab(entry_groups=("RG", "Kept")),
            ),
        )

        assert updates["h1"].claim.preowned_episode_ids == (101,)

    def test_update_carries_the_plan_size_matched_episodes(self) -> None:
        # The plan resolved which no-group files a preferred release's listed size named
        # (see the planner's owned-episodes tests). The update persists the
        # (id, size) pairs so the import doesn't copy over the very file the
        # grab just called this torrent's, and can confirm the match by size again.
        ep_list = [sonarr_ep(1, 1, ep_id=101, size=1000), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 02.mkv": parsed_info(season=1, episodes=(2,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 02.mkv"],
                        size=[2000],
                        infohash="h1",
                        download=True,
                        episodes=[ReleaseEpisode(season=1, episode=2, size=2000)],
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(
                guards=OwnedAtGrab(owned_episodes=(SizeMatchedEpisode(101, 1000),)),
            ),
        )

        assert updates["h1"].claim.guards.owned_episodes == ((101, 1000),)

    def test_multi_file_torrent_maps_each_file_to_its_own_episode(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 02.mkv": parsed_info(season=1, episodes=(2,)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 01.mkv", "Show - 02.mkv"],
                        size=[1000, 2000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        update = updates["h1"]
        assert dict(update.file_map) == {
            normalize_name("Show - 01.mkv"): (101,),
            normalize_name("Show - 02.mkv"): (102,),
        }
        assert update.claim.claimed_coverage == "S01 E01-E02"
        # The file map episodes are the file map's per-file ids alone, so the old cross-file
        # union bug (a whole season put onto one file) is out.
        assert _record(update).file_map_episode_ids() == [101, 102]

    def test_empty_title_places_nothing_and_excludes_nothing(self) -> None:
        # A title that resolved no episodes has no window to place into. An empty
        # index must never read as "no window" (which places against the live index).
        series = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group({"u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)}),
        }

        updates = _build(
            _strat(parses, series),
            seadex_dict,
            facts=title_facts(),
            grabbing_title=_grabbing_title([], series),
        )

        assert dict(updates["h1"].file_map) == {}
        assert updates["h1"].excluded == ()

    def test_unread_series_map_places_nothing(self) -> None:
        # The count steps need no index, but an update is final where an import poll is retried,
        # so the whole placement waits for a served list. Two files with no name number over two episodes
        # would otherwise zip in name order.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        files = ["Show - Special A.mkv", "Show - Special B.mkv"]
        parses = {name: parsed_info() for name in files}
        seadex_dict = {
            "RG": rg_group({"u1": make_release(files=files, size=[1000, 1000], infohash="h1", download=True)})
        }
        sonarr = FakeSonarrClient()
        sonarr.episodes_return = None
        strat = make_sonarr_sync(
            sonarr=sonarr, cache_store=FakeCacheStore(sonarr_parse=_rows(parses)), ep_list_cache={}
        )

        updates = _build(
            strat,
            seadex_dict,
            facts=title_facts(),
            grabbing_title=_grabbing_title(ep_list, []),
        )

        assert dict(updates["h1"].file_map) == {}
        assert updates["h1"].excluded == ()
        assert updates["h1"].claim.ordered_episode_ids == (101, 102)

    def test_sibling_window_files_are_excluded_not_intended(self) -> None:
        # A Part 1 title over a Part 1+2 torrent: files resolving in the series index
        # but OUTSIDE this title's set go in the exclusions, so file map + excluded
        # account for every file and the pending import stays determinate (a real progress
        # bar, an import timeout that re-anchors per arriving file).
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
                    "u1": make_release(
                        files=["Show - S03E01.mkv", "Show - S03E02.mkv", "Show - S03E13.mkv"],
                        size=[1000, 1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, series),
            seadex_dict,
            facts=title_facts(),
            grabbing_title=_grabbing_title(ep_list, series),
        )

        update = updates["h1"]
        assert set(update.file_map) == {
            normalize_name("Show - S03E01.mkv"),
            normalize_name("Show - S03E02.mkv"),
        }
        assert update.excluded == (normalize_name("Show - S03E13.mkv"),)

    def test_file_resolving_nowhere_in_the_series_is_not_excluded(self) -> None:
        # Its key exists nowhere in the series, so nothing proves it another
        # title's: the pending import keeps it as possibly this torrent's rather than writing it off.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - S09E09.mkv": parsed_info(season=9, episodes=(9,))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show - S09E09.mkv"], size=[1000], infohash="h1", download=True)}
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {}
        assert updates["h1"].excluded == ()

    def test_collision_rejected_duplicate_is_excluded(self) -> None:
        # Two claimants on one episode (a v2 beside its v1): the placement takes the
        # later version AND excludes the other, so this pending import will never
        # import it, so completeness may account for it.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 01v2.mkv": parsed_info(season=1, episodes=(1,)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 01.mkv", "Show - 01v2.mkv"],
                        size=[1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        update = updates["h1"]
        assert dict(update.file_map) == {normalize_name("Show - 01v2.mkv"): (101,)}
        assert update.excluded == (normalize_name("Show - 01.mkv"),)

    def test_unparsed_and_rejected_files_stay_possibly_this_torrents(self) -> None:
        # A file with no cached parse and a full-season-rejected zip: neither is
        # KNOWABLY another title's, so neither is excluded - the pending import stays
        # indeterminate (conservative) rather than trusting an incomplete file map.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            # "Show - Extra.mkv" has no cached parse at all.
            "Show - Zip.mkv": parsed_info(matched=((2, 1),), full_season=True),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 01.mkv", "Show - Extra.mkv", "Show - Zip.mkv"],
                        size=[1000, 1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        update = updates["h1"]
        assert set(update.file_map) == {normalize_name("Show - 01.mkv")}
        assert update.excluded == ()

    def test_sibling_per_episode_torrents_get_distinct_coverage_labels(self) -> None:
        # The live shape that motivated the claimed coverage: one title, one group, one
        # torrent per episode. Identical name·group labels made "which episodes
        # imported?" unanswerable from the wait report / notification.
        ep_list = [sonarr_ep(2, 6, ep_id=101, episode_file_id=0), sonarr_ep(2, 7, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - S02E06.mkv": parsed_info(season=2, episodes=(6,)),
            "Show - S02E07.mkv": parsed_info(season=2, episodes=(7,)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(files=["Show - S02E06.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": make_release(files=["Show - S02E07.mkv"], size=[1000], infohash="h2", download=True),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert _record(updates["h1"]).display_label == f"Show{SEP}RG{SEP}S02 E06"
        assert _record(updates["h2"]).display_label == f"Show{SEP}RG{SEP}S02 E07"

    def test_an_unparsed_video_still_builds_an_update_with_an_empty_map(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True),
                },
            ),
        }

        updates = _build(
            _strat({}, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert set(updates) == {"h1"}
        assert dict(updates["h1"].file_map) == {}
        assert updates["h1"].facts.seadex_files == ("Show - 01.mkv",)
        # Nothing claimed -> the claimed coverage is the whole window it is checked against.
        assert updates["h1"].claim.claimed_coverage == "S01 E01"

    def test_unclaimed_pending_import_labels_with_the_whole_window(self) -> None:
        # Two releases over one window: the parsed file claims its episode, the
        # unparsed one claims nothing, so its label names every episode it
        # is checked against rather than losing the claimed coverage.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - S01E01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(files=["Show - S01E01.mkv"], size=[1000], infohash="h1", download=True),
                    "u2": make_release(files=["Show - Extra.mkv"], size=[1000], infohash="h2", download=True),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert _record(updates["h1"]).display_label == f"Show{SEP}RG{SEP}S01 E01"
        assert dict(updates["h2"].file_map) == {}
        assert _record(updates["h2"]).display_label == f"Show{SEP}RG{SEP}S01 E01-E02"

    def test_no_video_files_builds_no_update(self) -> None:
        # A release with only non-video files (subs) has nothing to import.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(files=["Show - 01.ass"], size=[10], infohash="h1", download=True),
                },
            ),
        }

        updates = _build(
            _strat({}, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert updates == {}


class TestFileMapGuards:
    """The grab's file map guards.

    A full-season parse never places, colliding claims resolve first-wins in
    SeaDex file order, and a duplicate file name places once. A rejected file is left
    out of the file map, and the import never places it.
    """

    def test_full_season_parse_never_places(self) -> None:
        # An OP/ED whose plain-"S05" name Sonarr matches to the whole season:
        # the parse carries all 12 matched pairs, and none of them may place
        # (the old behavior imported this one file as every episode).
        ep_list = [sonarr_ep(5, e, ep_id=500 + e, episode_file_id=0) for e in range(1, 13)]
        parses = {
            "Show S05 Ending.mkv": parsed_info(
                matched=tuple((5, e) for e in range(1, 13)),
            ),
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show S05 Ending.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        # Still tracked (it carries a video file), just never placed ahead.
        assert set(updates) == {"h1"}
        assert dict(updates["h1"].file_map) == {}
        assert updates["h1"].facts.seadex_files == ("Show S05 Ending.mkv",)

    def test_small_full_season_parse_never_places(self) -> None:
        # A plain-"S01" OP/ED Sonarr matched to a whole season of only two
        # episodes: the pair count slips under the span cap, so Sonarr's own
        # fullSeason flag on the parse is what keeps it out of the file map.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {
            "Show S01 Opening.mkv": parsed_info(
                matched=((1, 1), (1, 2)),
                full_season=True,
            ),
        }
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        # Still tracked (it carries a video file), just never placed ahead.
        assert set(updates) == {"h1"}
        assert dict(updates["h1"].file_map) == {}

    def test_legitimate_double_episode_span_still_places(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {normalize_name("Show - 01-02.mkv"): (101, 102)}

    def test_partially_resolving_span_is_not_placed(self) -> None:
        # A double-episode file straddling the title boundary: only episode 1
        # is in this title's list, so placing it on [101] would half-import the file.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        series = [*ep_list, sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {"Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2))}
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show - 01-02.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            _strat(parses, series),
            seadex_dict,
            facts=title_facts(),
            grabbing_title=_grabbing_title(ep_list, series),
        )

        assert dict(updates["h1"].file_map) == {}

    def test_second_claim_of_an_id_in_the_file_map_is_not_placed(self) -> None:
        # "13" and "13v2" both parse to S02E13: the later version wins,
        # deterministically, and the other claimant is left out of the file map.
        ep_list = [sonarr_ep(2, 13, ep_id=213, episode_file_id=0)]
        parses = {
            "Show - 13.mkv": parsed_info(season=2, episodes=(13,)),
            "Show - 13v2.mkv": parsed_info(season=2, episodes=(13,)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 13.mkv", "Show - 13v2.mkv"],
                        size=[1000, 1001],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {normalize_name("Show - 13v2.mkv"): (213,)}

    def test_partial_collision_leaves_the_whole_later_file_unplaced(self) -> None:
        # The later file claims one already-placed id and one unplaced one: placement
        # leaves the whole file unplaced on any collision, so it is rejected whole
        # rather than placing the other half.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0), sonarr_ep(1, 2, ep_id=102, episode_file_id=0)]
        parses = {
            "Show - 01.mkv": parsed_info(season=1, episodes=(1,)),
            "Show - 01-02.mkv": parsed_info(season=1, episodes=(1, 2)),
        }
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["Show - 01.mkv", "Show - 01-02.mkv"],
                        size=[1000, 2000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {normalize_name("Show - 01.mkv"): (101,)}

    def test_duplicate_file_names_place_once(self) -> None:
        # The same file name in two folders collapses in the name-keyed
        # file map: the first occurrence's claim stands and the copy is rejected,
        # so the file map is deterministic and never double-claims the id.
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parses = {"Show - 01.mkv": parsed_info(season=1, episodes=(1,))}
        seadex_dict = {
            "RG": rg_group(
                {
                    "u1": make_release(
                        files=["FolderA/Show - 01.mkv", "FolderB/Show - 01.mkv"],
                        size=[1000, 1000],
                        infohash="h1",
                        download=True,
                    ),
                },
            ),
        }

        updates = _build(
            _strat(parses, ep_list),
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {normalize_name("Show - 01.mkv"): (101,)}
        # Both physical files stay tracked (the file name list is the folder's truth).
        assert updates["h1"].facts.seadex_files == ("Show - 01.mkv", "Show - 01.mkv")


class TestParseWriteFeedsUpdates:
    """One `/parse` per file name: the gather asks Sonarr, persists the parse, and the placement places by it."""

    @staticmethod
    def _cold(parse: ParsedFileInfo, ep_list: list[SonarrEpisode]) -> tuple[SonarrStrategy, FakeSonarrClient]:
        """A strat whose Sonarr answers every `/parse` with `parse`, its cache cold."""

        sonarr = FakeSonarrClient(parse_fn=lambda _f: parse)
        return make_sonarr_sync(sonarr=sonarr, cache_store=FakeCacheStore(), ep_list_cache={7: ep_list}), sonarr

    def test_parse_write_feeds_the_update_build(self) -> None:
        ep_list = [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)]
        parse = parsed_info(season=1, episodes=(1,), matched=((1, 1),))
        strat, sonarr = self._cold(parse, ep_list)
        seadex_dict = {
            "RG": rg_group(
                {"u1": make_release(files=["Show - 01.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            strat,
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {normalize_name("Show - 01.mkv"): (101,)}
        # Asked once, and persisted for the next run's gather.
        assert sonarr.parse_calls == ["Show - 01.mkv"]
        assert strat._parse.cache_store.get_sonarr_parse("Show - 01.mkv") is not None

    def test_full_season_flag_flows_from_parse_write_to_rejection(self) -> None:
        # End-to-end through the real write path: a fullSeason parse persists the
        # flag on the cached parse, and the placement rejects the file by it.
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
                {"u1": make_release(files=["Show S01 Opening.mkv"], size=[1000], infohash="h1", download=True)},
            ),
        }

        updates = _build(
            strat,
            seadex_dict,
            facts=title_facts(),
        )

        assert dict(updates["h1"].file_map) == {}


class TestNetInsertedCounts:
    """Preowned episodes never count toward this torrent's done files."""

    def test_progress_counts_exclude_preowned_file_map_episodes(self) -> None:
        # 101 already had another preferred release's file at grab time (preowned); only
        # 102 is this torrent's to import. The bar must read 0 of 1, not 1 of 2
        # - a pre-existing file shown as done overstates the import.
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
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            guards=OwnedAtGrab(entry_groups=("Kept",)),
            preowned_episode_ids=[101],
        )

        progress = strat._reconciler.import_progress(pending)

        assert progress.determinate is True
        assert (progress.done, progress.total) == (0, 1)


class TestTrustedGroups:
    """The trusted groups: the torrent's group + its entry's preferred groups + sibling GRABBED groups only."""

    def test_sibling_entry_preferred_groups_never_contaminate(self) -> None:
        # Another title's pending import contributes its grabbed group, never its preferred
        # list: a group trusted for one season says nothing about another
        # season's episodes.
        sibling = pending_import(
            infohash="s1",
            al_id=999,
            release_group="SibGrab",
            guards=OwnedAtGrab(entry_groups=("SibGrab", "SibPick")),
        )
        own = pending_import(release_group="Ours", guards=OwnedAtGrab(entry_groups=("Ours", "OtherPick")))

        assert set(_trust(own, [sibling])) == {"ours", "otherpick", "sibgrab"}

    def test_sibling_grabbed_group_refused_for_a_group_this_plan_found_outdated(self) -> None:
        # A sibling pending import grabbed group B earlier; THIS title's plan found its
        # B copy in the arr outdated and excluded it from entry_groups. The sibling's
        # group must not re-admit B, or the outdated files it shields read done and
        # the replacement import never happens.
        sibling = pending_import(infohash="s1", al_id=999, release_group="B")
        own = pending_import(
            release_group="Ours",
            guards=OwnedAtGrab(entry_groups=("Ours",), stale_groups=("B",)),
        )

        assert set(_trust(own, [sibling])) == {"ours"}

    def test_this_torrents_group_survives_its_own_outdated_verdict(self) -> None:
        # A same-group size update lists its group as outdated. The group must
        # stay trusted (it is the identity of the files being imported) -
        # the outdated copies are told apart by size instead.
        own = pending_import(release_group="Ours", guards=OwnedAtGrab(stale_groups=("Ours",)), release_sizes=[1000])

        assert _trust(own) == {"ours": frozenset({1000})}

    def test_this_torrents_group_without_sizes_is_trusted_by_name(self) -> None:
        # No listed sizes (an older pending import, or a blind listing) means no size
        # gate: the None value tells the classifier to trust the name alone.
        own = pending_import(release_group="Ours")

        assert _trust(own) == {"ours": None}

    def test_this_torrents_group_sizes_union_same_group_siblings(self) -> None:
        # Two pending imports grabbing the same group (a per-cour torrent each) each
        # list their sizes; a file either pending import's listing carries is an
        # owned copy, so the size gate reads their union.
        sibling = pending_import(infohash="s1", al_id=999, release_group="Ours", release_sizes=[2000])
        own = pending_import(release_group="Ours", release_sizes=[1000])

        assert _trust(own, [sibling])["ours"] == frozenset({1000, 2000})

    def test_file_map_statuses_read_sibling_grabbed_groups_from_the_store(self) -> None:
        # The store round trip behind the pure merge: a sibling pending import persisted
        # by an earlier grab loads and its group protects the arr's file.
        sibling = pending_import(infohash="s1", al_id=999, release_group="SibGrab")
        store = FakeCacheStore(pending={str(Arr.SONARR): {sibling.infohash: sibling.to_json()}})
        sonarr = FakeSonarrClient(episodes=[sonarr_ep(1, 1, ep_id=101, size=500, release_group="SibGrab")])
        strat = make_sonarr_sync(sonarr=sonarr, cache_store=store)
        own = pending_import(release_group="Ours", file_episode_map={"Show - 01.mkv": [101]})

        update = strat._reconciler._read_statuses(own, [101])

        assert update is not None
        assert update.statuses.by_id == {101: EpisodeFileStatus.TRUSTED}
