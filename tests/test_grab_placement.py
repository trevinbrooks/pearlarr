# pyright: strict
"""Grab-time placement: what `place_release` gives the planner, the pending import a release joins, and the updates."""

from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar

import pytest

from pearlarr.arr_models import MatchedEpisode, ParsedFileInfo, SonarrEpisode
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_placement import (
    EntryPlacements,
    FetchedSeadex,
    KnownTorrent,
    ListedFile,
    PendingImportUpdate,
    PlacedGrab,
    PlacedRelease,
    SizeOverName,
    TorrentFetches,
    build_pending_import_update,
    build_pending_import_updates,
    build_title_claim,
    build_update_without_window,
    entry_hashes,
    place_release,
    size_over_name_files,
)
from pearlarr.pending_import import NO_ORDER_ONLY, FileEpisodeMap, PendingImport, TitleNames, TitleWindow
from pearlarr.placement_types import (
    EMPTY_LISTING,
    NO_EVIDENCE,
    BatchPlacements,
    EpisodeIndex,
    FetchedListings,
    GrabbingTitle,
    PlacementVerdict,
    SeadexEvidence,
    episode_index,
)
from pearlarr.releases import (
    NO_ANOTHER_TITLES_COUNTS,
    AnotherTitlesCounts,
    EpisodeKey,
    OrderOnlyKey,
    PlannedGrab,
    ReleaseEpisode,
    SeadexDict,
    SkipReason,
)
from pearlarr.tvdb_orders import IndexOrders, OrderRow

from .builders import (
    ALIASED_SPECIALS_LISTING,
    UNREAD_EVIDENCE,
    by_name,
    known_torrent,
    make_grabbing_title,
    make_release,
    new_torrent,
    parsed_info,
    pending_import,
    pending_import_update,
    rg_group,
    sonarr_ep,
    title_claim,
    title_facts,
    title_window,
    torrent_evidence,
    torrent_facts,
    torrent_listing,
    two_claim_pending_import,
    verdicts_of,
)

_SEASON = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0) for n in range(1, 5)]
_SPECIAL = sonarr_ep(0, 1, ep_id=501, episode_file_id=0)
_SERIES = [
    _SPECIAL,
    *_SEASON,
    sonarr_ep(2, 1, ep_id=201, episode_file_id=0),
    sonarr_ep(2, 2, ep_id=202, episode_file_id=0),
]
_RUN = [f"Show - {n:02d}.mkv" for n in range(1, 5)]
_SEASON_NAMES = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)]
_INDEXES = {7: episode_index(_SERIES)}
_STAMP = "2026-06-24 00:00:00"
_SPECIALS = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)]
_FACTS = torrent_facts(
    is_dual_audio=True, seadex_files=("Show - 01.mkv",), release_sizes=(10,), sizes_by_name={"show - 01.mkv": 10}
)


def _grabbing_title(
    window_episodes: list[SonarrEpisode], names: TitleNames | None = None, *, al_id: int = 1
) -> GrabbingTitle:
    return make_grabbing_title(
        episode_index(window_episodes), episode_index(_SERIES), al_id=al_id, names=names or TitleNames()
    )


def _number_with_no_match(number: int) -> ParsedFileInfo:
    """A `- 01` name Sonarr matched nothing for."""

    return ParsedFileInfo(episode_numbers=(number,))


def _matched(season: int, episode: int) -> ParsedFileInfo:
    return ParsedFileInfo(
        season_number=season,
        episode_numbers=(episode,),
        matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),),
    )


def _season_episode_with_no_match(season: int, episode: int) -> ParsedFileInfo:
    """A spelled-out season and episode Sonarr matched nothing for."""

    return ParsedFileInfo(season_number=season, episode_numbers=(episode,))


def _season_files(count: int = 4) -> list[ListedFile]:
    """The season pack's first `count` files, each matched by Sonarr and sized `10 * n`."""

    return [ListedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(_SEASON_NAMES[:count], start=1)]


def _torrent(pending: PendingImport | None = None, *, listed: Sequence[TitleWindow] | None = ()) -> KnownTorrent:
    """A `KnownTorrent` over the shared series, by default a new torrent that nothing lists.

    `listed=None` means SeaDex couldn't be read (`UNREAD_EVIDENCE`).
    """

    evidence = UNREAD_EVIDENCE if listed is None else torrent_evidence(*listed)
    return known_torrent(pending, indexes=_INDEXES, evidence=evidence)


_NEW = _torrent()


def _stored_pending_import() -> PendingImport:
    """A stored pending import placing the torrent's first two files, its one claim over the season on series 7."""

    return pending_import(
        infohash="h1",
        series_id=7,
        file_episode_map={"show - s01e01.mkv": [101], "show - s01e02.mkv": [102]},
        seadex_files=_SEASON_NAMES,
        ordered_episode_ids=(101, 102, 103, 104),
    )


class TestPlaceRelease:
    """One release's files placed by the import's passes, its episodes the planner's coverage vocabulary."""

    def test_a_sequence_sonarr_read_nothing_of_records_the_window(self) -> None:
        files = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.episodes == tuple(ReleaseEpisode(1, n, 10 * n) for n in range(1, 5))
        assert placement.placements.placed == {normalize_file_name(name): (100 + n,) for n, name in enumerate(_RUN, 1)}
        assert placement.covered_ids == {101, 102, 103, 104}
        assert placement.inputs_read
        assert placement.pending_import is None

    def test_each_placed_episode_intends_its_files_listed_size(self) -> None:
        files = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.intended_sizes == {100 + n: 10 * n for n in range(1, 5)}

    def test_a_whole_series_torrent_records_only_the_titles_window(self) -> None:
        names = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)] + ["Show - S02E01.mkv", "Show - S02E02.mkv"]
        parses = [_matched(1, n) for n in range(1, 5)] + [_matched(2, 1), _matched(2, 2)]
        files = [ListedFile(name, 1, parse) for name, parse in zip(names, parses, strict=True)]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.episodes == tuple(ReleaseEpisode(1, n, 1) for n in range(1, 5))
        assert {p.name for p in placement.placements.excluded} == {"show - s02e01.mkv", "show - s02e02.mkv"}

    def test_a_file_placed_nowhere_records_nothing(self) -> None:
        files = [ListedFile("Show - S02E01.mkv", 1, _matched(2, 1))]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.episodes == ()
        assert verdicts_of(placement.placements) == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_a_sequence_waiting_on_a_parse_records_nothing_and_skips_the_release_as_unread(self) -> None:
        files = [
            ListedFile(name, 1, None if n == 3 else _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.episodes == ()
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read
        assert verdicts_of(placement.placements) == {PlacementVerdict.WAITING_ON_PARSE}

    def test_a_file_beside_a_sequence_waiting_on_a_parse_still_places(self) -> None:
        files = [ListedFile("Show - OVA.mkv", 5, _matched(0, 1))] + [
            ListedFile(name, 1, None if n == 3 else _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _grabbing_title([_SPECIAL, *_SEASON]), _NEW)

        assert placement.episodes == (ReleaseEpisode(0, 1, 5),)
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read

    def test_two_files_sharing_a_file_name_keep_their_own_sizes(self) -> None:
        # The gather already stripped the folders: one file name, placed once, two listed sizes.
        files = [
            ListedFile("Show - S01E01.mkv", 10, _matched(1, 1)),
            ListedFile("Show - S01E01.mkv", 20, _matched(1, 1)),
        ]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.episodes == (ReleaseEpisode(1, 1, 10), ReleaseEpisode(1, 1, 20))
        assert placement.placements.placed == {"show - s01e01.mkv": (101,)}

    def test_an_episode_two_listed_sizes_share_intends_no_size(self) -> None:
        # Either could be the file intended on E01.
        files = [
            ListedFile("Show - S01E01.mkv", 10, _matched(1, 1)),
            ListedFile("Show - S01E01.mkv", 20, _matched(1, 1)),
        ]

        placement = place_release(files, _grabbing_title(_SEASON), _NEW)

        assert placement.intended_sizes == {}

    def test_an_unread_episode_index_places_nothing_and_skips_the_release_as_unread(self) -> None:
        # The series list did not serve, so the grabbing title has no series index and its window can't be ordered.
        files = [ListedFile(name, 1, _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)]
        grabbing_title = make_grabbing_title(episode_index(_SEASON), episode_index([]))

        placement = place_release(files, grabbing_title, known_torrent())

        assert not grabbing_title.can_place
        assert placement.placements.placements == ()
        assert placement.episodes == ()
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read

    def test_no_files_is_an_empty_placement_with_its_inputs_read(self) -> None:
        placement = place_release([], _grabbing_title(_SEASON), _NEW)

        assert placement == PlacedRelease(
            files=(),
            placements=BatchPlacements(()),
            episodes=(),
            covered_ids=frozenset(),
            inputs_read=True,
            skip_reason=None,
            another_titles_counts=NO_ANOTHER_TITLES_COUNTS,
            pending_import=None,
            intended_sizes={},
        )


class TestPlaceSizeMatches:
    """A file whose size matches an episode goes on that episode."""

    _LOST = ListedFile("Show - Lost Tape.mkv", 70, ParsedFileInfo())

    def test_a_file_with_no_number_is_placed_by_its_listed_size(self) -> None:
        torrent = known_torrent(indexes=_INDEXES, evidence=torrent_evidence(episode_by_size={70: 103}))

        placement = place_release([self._LOST, *_season_files(2)], _grabbing_title(_SEASON), torrent)

        assert placement.placements.placed["show - lost tape.mkv"] == (103,)
        assert PlacementVerdict.SIZE_MATCH in verdicts_of(placement.placements)
        assert placement.inputs_read

    def test_a_hash_less_release_is_placed_by_the_series_episodes_by_size(self) -> None:
        # No listing matches a torrent without a hash, but the series' episodes by size still apply to it.
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash=None, download=True)})}
        known = TorrentFetches({}, _INDEXES, FetchedListings({}, {70: 103})).known(seadex_dict)

        placement = place_release([self._LOST, *_season_files(2)], _grabbing_title(_SEASON), known["u1"])

        assert placement.placements.placed["show - lost tape.mkv"] == (103,)
        assert placement.skip_reason is None
        assert placement.inputs_read

    def test_unread_episodes_by_size_skip_the_release(self) -> None:
        # SeaDex is unreachable, so there are no episodes by size. The torrent is still placed by its numbers for
        # coverage, but the release is skipped, since anything the grab puts in the file map is never placed again.
        torrent = known_torrent(indexes=_INDEXES, evidence=FetchedSeadex(EMPTY_LISTING, None))

        placement = place_release([self._LOST, *_season_files(2)], _grabbing_title(_SEASON), torrent)

        assert placement.placements.placed == {"show - s01e01.mkv": (101,), "show - s01e02.mkv": (102,)}
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read

    def test_the_size_over_name_files_are_those_placed_over_their_names(self) -> None:
        # The file with no parsed number isn't an override since it has none. For the file Sonarr matched, the
        # warning shows Sonarr's parse.
        torrent = known_torrent(
            indexes=_INDEXES, evidence=torrent_evidence(episode_by_size={70: 103, 10: 102, 20: 101})
        )
        matched = ListedFile(
            "Show - Second Half.mkv",
            20,
            ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=2),)),
        )
        placement = place_release([self._LOST, _season_files(1)[0], matched], _grabbing_title(_SEASON), torrent)

        assert size_over_name_files(placement, episode_index(_SERIES)) == (
            SizeOverName("show - s01e01.mkv", "S01E01", EpisodeKey(1, 2)),
            SizeOverName("show - second half.mkv", "S01E02", EpisodeKey(1, 1)),
        )


class TestPlaceOntoResident:
    """A listed torrent with a stored pending import: only the file map's leftover is placed, the whole map recorded."""

    def test_the_unplaced_files_are_placed_and_the_whole_file_map_recorded(self) -> None:
        pending = _stored_pending_import()

        placement = place_release(_season_files(), _grabbing_title(_SEASON), _torrent(pending))

        # The file map's two names are not re-placed, and the episodes and claim read the full file map.
        assert placement.placements.placed == {"show - s01e03.mkv": (103,), "show - s01e04.mkv": (104,)}
        assert placement.episodes == tuple(ReleaseEpisode(1, n, 10 * n) for n in range(1, 5))
        assert placement.covered_ids == {101, 102, 103, 104}
        assert placement.inputs_read
        assert placement.pending_import is pending

    def test_an_unread_parse_in_a_file_map_holding_the_whole_torrent_skips_nothing(self) -> None:
        pending = pending_import(
            infohash="h1",
            file_episode_map={normalize_file_name(name): [100 + n] for n, name in enumerate(_SEASON_NAMES, start=1)},
            seadex_files=_SEASON_NAMES,
            ordered_episode_ids=(101, 102, 103, 104),
        )
        files = [ListedFile(_SEASON_NAMES[0], 10, None), *_season_files()[1:]]

        placement = place_release(files, _grabbing_title(_SEASON), _torrent(pending))

        assert placement.skip_reason is None
        assert placement.inputs_read
        assert placement.covered_ids == {101, 102, 103, 104}

    def test_an_unread_parse_in_the_file_map_skips_the_release_while_a_file_is_left_to_place(self) -> None:
        files = [ListedFile(_SEASON_NAMES[0], 10, None), *_season_files()[1:]]

        placement = place_release(files, _grabbing_title(_SEASON), _torrent(_stored_pending_import()))

        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read

    def test_a_file_map_episode_outside_the_titles_window_records_nothing(self) -> None:
        # A sibling title's episode (201) and an id no index holds (999): neither is this title's.
        pending = pending_import(
            infohash="h1",
            file_episode_map={"show - s02e01.mkv": [201], "other - 01.mkv": [999]},
            seadex_files=["Show - S02E01.mkv", "Other - 01.mkv"],
            ordered_episode_ids=(201, 202),
        )
        files = [
            ListedFile("Show - S02E01.mkv", 5, _matched(2, 1)),
            ListedFile("Other - 01.mkv", 6, _number_with_no_match(1)),
        ]

        placement = place_release(files, _grabbing_title(_SEASON, al_id=2), _torrent(pending))

        assert placement.placements.placements == ()
        assert placement.episodes == ()
        assert placement.covered_ids == frozenset()
        assert placement.inputs_read

    def test_a_regrab_places_the_unplaced_file_under_its_new_window(self) -> None:
        # The title's stored claim covers two episodes. Re-listed over the whole season, its new window
        # replaces the stored one, so the third file lands where the stored window could not put it.
        pending = pending_import(
            infohash="h1",
            file_episode_map={"show - s01e01.mkv": [101]},
            seadex_files=_SEASON_NAMES[:3],
            ordered_episode_ids=(101, 102),
        )
        files = _season_files(3)

        stale = place_release(files, _grabbing_title(_SEASON[:2]), _torrent(pending))
        fresh = place_release(files, _grabbing_title(_SEASON), _torrent(pending))

        assert stale.placements.placed == {"show - s01e02.mkv": (102,)}
        assert fresh.placements.placed == {"show - s01e02.mkv": (102,), "show - s01e03.mkv": (103,)}
        assert fresh.covered_ids == {101, 102, 103}

    def test_a_regrab_runs_the_new_window_in_the_stored_ones_place(self) -> None:
        # The title's stored claim is a two-episode window, and the pair Sonarr read nothing of indexes
        # whichever two-episode window runs. Re-listed over the two episodes before it, the new window runs
        # in the stored one's place, never after it, so the pair lands on the new episodes.
        pending = pending_import(
            infohash="h1", file_episode_map={}, seadex_files=_RUN[:2], ordered_episode_ids=(103, 104)
        )
        files = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN[:2], start=1)]

        placement = place_release(files, _grabbing_title(_SEASON[:2]), _torrent(pending))

        assert placement.placements.placed == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert placement.covered_ids == {101, 102}

    def test_a_sibling_claim_on_an_unread_series_places_nothing_and_skips_the_release_as_unread(self) -> None:
        # Title 2's claim places two of the files on series 8, which this run could not read. Its window can't be
        # ordered, so nothing is placed, and the release waits for the read rather than counting as another title's.
        pending = pending_import(
            infohash="h1",
            series_id=8,
            al_id=2,
            file_episode_map={"show - s01e01.mkv": [901], "show - s01e02.mkv": [902]},
            seadex_files=_SEASON_NAMES,
            ordered_episode_ids=(901, 902, 903, 904),
        )
        torrent = _torrent(pending)

        placement = place_release(_season_files(), _grabbing_title(_SEASON), torrent)

        assert torrent.order(_grabbing_title(_SEASON).window) is None
        assert placement.placements.placements == ()
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read
        assert placement.covered_ids == frozenset()
        assert placement.pending_import is pending

    def test_a_stored_pending_import_under_the_titles_unread_series_leaves_the_inputs_unread(self) -> None:
        # The title's series list did not serve, so its window can't be ordered and the title is re-checked.
        # The stored file map is still the pending import's: its ids in the title's window record and count as claimed.
        unread = make_grabbing_title(episode_index(_SEASON), episode_index([]))
        torrent = known_torrent(_stored_pending_import(), evidence=torrent_evidence())

        placement = place_release(_season_files(), unread, torrent)

        assert placement.placements.placements == ()
        assert not placement.inputs_read
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert placement.episodes == (ReleaseEpisode(1, 1, 10), ReleaseEpisode(1, 2, 20))
        assert placement.covered_ids == {101, 102}


_ST1 = ("Alpha Start", "Bravo Turn", "Charlie Rise")
_ST2 = ("Echo Dawn", "Foxtrot Fall", "Golf Night")
# Two titled seasons and a special, season 2 numbered on in absolute order.
_TITLED = [
    *(sonarr_ep(1, n, ep_id=100 + n, absolute=n, title=t, episode_file_id=0) for n, t in enumerate(_ST1, 1)),
    *(sonarr_ep(2, n, ep_id=200 + n, absolute=3 + n, title=t, episode_file_id=0) for n, t in enumerate(_ST2, 1)),
    sonarr_ep(0, 1, ep_id=501, title="Beach Special", episode_file_id=0),
]
_TITLED_INDEX = episode_index(_TITLED)


class TestPlaceUnderAnotherTitlesContradiction:
    """Title 1's torrent, whose numbered sequence episode titles contradict in title 1's window, placed for title 2."""

    @staticmethod
    def _place(files: dict[str, ParsedFileInfo], mapped: FileEpisodeMap, window_episodes: list[int]) -> PlacedRelease:
        """Title 2 places `files` onto title 1's pending import of `mapped` on season 1, in window `window_episodes`."""

        pending = pending_import(
            infohash="h1",
            series_id=7,
            file_episode_map=mapped,
            seadex_files=tuple(files),
            ordered_episode_ids=(101, 102, 103),
        )
        grabbing_title = make_grabbing_title(
            episode_index([ep for ep in _TITLED if ep.id in window_episodes]), _TITLED_INDEX, al_id=2
        )
        listed_files = [ListedFile(name, 1000 + i, info) for i, (name, info) in enumerate(files.items())]
        torrent = known_torrent(pending, indexes={7: _TITLED_INDEX}, evidence=torrent_evidence())
        return place_release(listed_files, grabbing_title, torrent)

    def test_the_sequel_title_places_the_sequence_title_1s_window_contradicts(self) -> None:
        # Sonarr read the sequel's plain numbers as first-season absolutes, and its titled file names a sequel
        # episode.
        first = {f"show - s01e{n:02d}.mkv": _matched(1, n) for n in (1, 2, 3)}
        sequel = {
            name: ParsedFileInfo(
                absolute_episode_numbers=(n,), matched_episodes=(MatchedEpisode(season_number=1, episode_number=n),)
            )
            for n, name in enumerate(("show s2 - 01 - echo dawn.mkv", "show s2 - 02.mkv", "show s2 - 03.mkv"), 1)
        }

        placement = self._place(
            {**first, **sequel}, {name: (100 + n,) for n, name in enumerate(first, 1)}, [201, 202, 203]
        )

        assert placement.skip_reason is None
        assert placement.covered_ids == {201, 202, 203}

    def test_a_title_placing_nothing_is_skipped_as_title_1s_not_as_contradicted(self) -> None:
        # The titled file is the third episode's, under the second's season and episode. None of the files is the
        # special's, and title 1's pending import holds the titled one, so the release is title 1's, not contradicted
        # under title 2. The two plain files are title 1's too: its window contradicted them by episode title, and
        # title 2's never read them as title 2's, so they're no hand import for title 2.
        files = {
            "show - 01.mkv": _matched(1, 1),
            "show - 02 - charlie rise.mkv": _matched(1, 2),
            "show - 03.mkv": _matched(1, 3),
        }

        placement = self._place(files, {"show - 02 - charlie rise.mkv": (103,)}, [501])

        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=3, unplaced=0)


class TestPlaceSkipped:
    """A release the grab skips: a specials torrent the listing contradicts, or a listing the run could not read."""

    @staticmethod
    def _specials(count: int = 3) -> list[ListedFile]:
        """A torrent of the first `count` specials, each read by Sonarr as its number's special."""

        return [ListedFile(f"Show.S00E{n:02d}.mkv", 10 * n, _matched(0, n)) for n in range(1, count + 1)]

    @staticmethod
    def _specials_grabbing_title(*numbers: int) -> GrabbingTitle:
        """Title 1 over the specials `numbers`, within the all-specials series."""

        return make_grabbing_title(episode_index([_SPECIALS[n - 1] for n in numbers]), episode_index(_SPECIALS))

    @staticmethod
    def _listing(listed: tuple[int, ...] | None) -> KnownTorrent:
        """A new torrent listed over `listed` in the specials series. None means SeaDex couldn't be read."""

        evidence = UNREAD_EVIDENCE if listed is None else torrent_evidence(title_window(1, listed))
        return known_torrent(indexes={7: episode_index(_SPECIALS)}, evidence=evidence)

    def test_a_misnumbered_torrent_skips_the_release_and_records_nothing(self) -> None:
        # The entries list the torrent over specials 2, 4 and 6, three as the torrent is wide: its `1` is no listed
        # special, so the torrent numbers by another TVDB state. Nothing places, and the planner sees no coverage.
        placement = place_release(self._specials(), self._specials_grabbing_title(2, 4), self._listing((502, 504, 506)))

        assert verdicts_of(placement.placements) == {PlacementVerdict.MISNUMBERED}
        assert placement.skip_reason is SkipReason.MISNUMBERED
        assert placement.episodes == ()
        assert placement.covered_ids == frozenset()
        assert placement.inputs_read

    def test_a_torrent_skipped_for_a_sequence_whose_numbers_disagree_records_none_of_its_placed_files(self) -> None:
        # The season's absolutes count the special between E02 and E03, and the torrent's numbers go on past the end.
        season = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=0, absolute=1),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0, absolute=2),
            sonarr_ep(1, 3, ep_id=103, episode_file_id=0, absolute=4),
        ]
        series = episode_index([*season, sonarr_ep(0, 1, ep_id=501, episode_file_id=0, absolute=3)])
        files = [
            ListedFile(name, 10 * n, _matched(1, n) if n < 4 else _season_episode_with_no_match(1, n))
            for n, name in enumerate(_SEASON_NAMES, 1)
        ]
        grabbing_title = make_grabbing_title(episode_index(season), series)

        placement = place_release(files, grabbing_title, known_torrent(indexes={7: series}))

        assert placement.skip_reason is SkipReason.MISNUMBERED
        assert placement.placements.placed == {"show - s01e01.mkv": (101,), "show - s01e02.mkv": (102,)}
        assert placement.episodes == ()

    _DVD: ClassVar[tuple[OrderRow, ...]] = (
        OrderRow(1, 1, 101, None),
        OrderRow(1, 2, 102, None),
        OrderRow(1, 3, 501, None),
        OrderRow(1, 4, 103, None),
    )
    """A DVD order that counts the special as the season's third episode, so the aired third is its fourth."""

    @staticmethod
    def _counted_special(dvd: tuple[OrderRow, ...]) -> tuple[list[SonarrEpisode], SonarrEpisode, EpisodeIndex]:
        """A three-episode season whose absolutes count a special between E02 and E03, with `dvd` as its DVD order.

        Returns the season, the special, and the series index, which also has the next season's opener. Each
        episode's TVDB id is its Sonarr id.
        """

        season = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=0, absolute=1, tvdb_id=101),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0, absolute=2, tvdb_id=102),
            sonarr_ep(1, 3, ep_id=103, episode_file_id=0, absolute=4, tvdb_id=103),
        ]
        special = sonarr_ep(0, 1, ep_id=501, episode_file_id=0, absolute=3, tvdb_id=501)
        opener = sonarr_ep(2, 1, ep_id=201, episode_file_id=0, absolute=5, tvdb_id=201)
        return season, special, episode_index([*season, special, opener], IndexOrders({"dvd": dvd}, None))

    def test_a_torrent_numbered_by_the_dvd_order_is_grabbed_for_the_season_and_for_the_special(self) -> None:
        # The torrent's numbers go on past the season's end because it follows the DVD order. The placer rejects
        # the aired parse of its last two files. The order puts the third on the special and the fourth on the aired
        # third episode.
        season, special, series = self._counted_special(self._DVD)
        files = [
            ListedFile(name, 10 * n, _matched(1, n) if n < 4 else _season_episode_with_no_match(1, n))
            for n, name in enumerate(_SEASON_NAMES, 1)
        ]
        listed = torrent_evidence(title_window(1, (101, 102, 103)), title_window(2, (501,)))
        torrent = known_torrent(indexes={7: series}, evidence=listed)

        for_season = place_release(files, make_grabbing_title(episode_index(season), series), torrent)
        for_special = place_release(files, make_grabbing_title(episode_index([special]), series, al_id=2), torrent)

        assert (for_season.skip_reason, for_special.skip_reason) == (None, None)
        assert for_season.covered_ids == {101, 102, 103}
        assert for_season.episodes[-1] == ReleaseEpisode(season=1, episode=3, size=40)
        assert for_special.covered_ids == {501}
        assert for_special.episodes == (ReleaseEpisode(season=0, episode=1, size=30),)

    def test_a_torrent_the_order_places_onto_episodes_no_title_holds_is_left_for_a_hand_import(self) -> None:
        # The torrent's two files are numbered from the special's position on. This DVD order reads the first as the
        # special and the second as the next season's opener, and nothing lists the torrent under either.
        season, _special, series = self._counted_special((*self._DVD[:3], OrderRow(1, 4, 201, None)))
        files = [
            ListedFile(_SEASON_NAMES[2], 30, _matched(1, 3)),
            ListedFile(_SEASON_NAMES[3], 40, _season_episode_with_no_match(1, 4)),
        ]
        grabbing_title = make_grabbing_title(episode_index(season), series)

        placement = place_release(files, grabbing_title, known_torrent(indexes={7: series}))

        assert verdicts_of(placement.placements) == {PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW}
        assert placement.skip_reason is SkipReason.MISNUMBERED
        assert placement.episodes == ()

    @staticmethod
    def _titled_grabbing_title(*numbers: int) -> GrabbingTitle:
        """Title 1 over season 1's episodes `numbers`, within a titled four-episode season."""

        titles = ("Pilot Flight", "The Return", "Festival Night", "Final Bell")
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, title=t) for n, t in enumerate(titles, 1)]
        return make_grabbing_title(episode_index([season[n - 1] for n in numbers]), episode_index(season))

    def test_a_contradicted_torrent_placing_nothing_skips_the_release_and_records_nothing(self) -> None:
        # The titled file is the third episode's, under the plain file's season and episode.
        files = [
            ListedFile("show - 02 - Festival Night [grp].mkv", 10, _season_episode_with_no_match(1, 2)),
            ListedFile("show - 03 [grp].mkv", 20, _season_episode_with_no_match(1, 3)),
        ]

        grabbing_title = self._titled_grabbing_title(1, 2, 3)

        placement = place_release(files, grabbing_title, new_torrent(grabbing_title))

        assert placement.skip_reason is SkipReason.TITLE_REFUTED
        assert placement.episodes == ()

    def test_a_contradicted_torrent_a_sibling_places_part_of_is_skipped_as_contradicted(self) -> None:
        # The OVA lands on title 2's special, but the contradicted files are nobody's, so the release isn't another
        # title's.
        titles = ("Pilot Flight", "The Return", "Festival Night", "Final Bell")
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, title=t) for n, t in enumerate(titles, 1)]
        series = episode_index([sonarr_ep(0, 1, ep_id=501, episode_file_id=0), *season])
        files = [
            ListedFile("show - 02 - Festival Night [grp].mkv", 10, _season_episode_with_no_match(1, 2)),
            ListedFile("show - 03 [grp].mkv", 20, _season_episode_with_no_match(1, 3)),
            ListedFile("show - OVA [grp].mkv", 30, _matched(0, 1)),
        ]
        torrent = known_torrent(indexes={7: series}, evidence=torrent_evidence(title_window(2, (501,))))

        placement = place_release(files, make_grabbing_title(episode_index(season[:3]), series), torrent)

        assert by_name(placement.placements)["show - ova [grp].mkv"] == ((), PlacementVerdict.ANOTHER_TITLES)
        assert placement.skip_reason is SkipReason.TITLE_REFUTED
        assert placement.covered_ids == frozenset()

    _EARLY_KEYS: ClassVar[tuple[str, ...]] = (
        "show - 01 - The Return [grp].mkv",
        "show - 02 - Festival Night [grp].mkv",
        "show - 03 [grp].mkv",
    )
    """A torrent numbered one episode early, as the episode titles show. The untitled third can't be placed."""

    def _place_early_keys(self) -> tuple[PlacedRelease, GrabbingTitle]:
        files = [ListedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(self._EARLY_KEYS, 1)]
        grabbing_title = self._titled_grabbing_title(1, 2, 3)
        return place_release(files, grabbing_title, new_torrent(grabbing_title)), grabbing_title

    def test_a_contradicted_torrent_placing_some_files_is_not_skipped(self) -> None:
        placement, _ = self._place_early_keys()

        assert placement.skip_reason is None
        assert placement.placements.placed == {
            normalize_file_name(self._EARLY_KEYS[0]): (102,),
            normalize_file_name(self._EARLY_KEYS[1]): (103,),
        }
        assert placement.placements.hand_import_files == (normalize_file_name(self._EARLY_KEYS[2]),)
        assert len(placement.episodes) == 2

    def test_the_pending_import_update_keeps_the_file_left_for_a_hand_import(self) -> None:
        placement, grabbing_title = self._place_early_keys()
        release = PlacedGrab("grp", make_release(url="u", infohash="h1"), "h1", placement)

        update = build_pending_import_update(release, grabbing_title, title_facts())

        assert update.hand_import_files == (normalize_file_name(self._EARLY_KEYS[2]),)

    def test_a_torrent_the_listing_agrees_with_places_and_is_not_skipped(self) -> None:
        placement = place_release(self._specials(), self._specials_grabbing_title(1, 2), self._listing((501, 502, 503)))

        assert placement.skip_reason is None
        assert placement.placements.placed == {"show.s00e01.mkv": (501,), "show.s00e02.mkv": (502,)}
        assert verdicts_of(placement.placements) == {PlacementVerdict.EXACT, PlacementVerdict.OUTSIDE_WINDOW}

    def test_a_torrent_its_aliases_point_into_the_listing_places_and_is_not_skipped(self) -> None:
        # The torrent's `1` isn't a listed special, but the titles' aliases point 1, 2 and 3 at 2, 4 and 6. Special 6
        # is title 2's, which has no claim yet, so its file is set aside as title 2's.
        torrent = known_torrent(
            indexes={7: episode_index(_SPECIALS)},
            evidence=torrent_evidence(
                title_window(1, (502, 504)), title_window(2, (506,)), aliases=ALIASED_SPECIALS_LISTING.special_aliases
            ),
        )

        placement = place_release(self._specials(), self._specials_grabbing_title(2, 4), torrent)

        assert placement.skip_reason is None
        assert placement.placements.placed == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (504,)}
        assert by_name(placement.placements)["show.s00e03.mkv"] == ((), PlacementVerdict.ANOTHER_TITLES)
        assert placement.covered_ids == {502, 504}

    _OTHER = episode_index([sonarr_ep(0, n, ep_id=900 + n, episode_file_id=0) for n in (1, 2, 3)])
    """Another series' three specials, which the listing doesn't cover."""

    @classmethod
    def _aliased_against(cls, numbered: tuple[int, ...]) -> KnownTorrent:
        """The aliased torrent, claimed for title 1 on specials 2 and 4 and for title 2 on `numbered` of `_OTHER`."""

        return known_torrent(
            two_claim_pending_import(series_ids=(7, 8), windows=((502, 504), numbered)),
            indexes={7: episode_index(_SPECIALS), 8: cls._OTHER},
            evidence=torrent_evidence(
                *ALIASED_SPECIALS_LISTING.windows, aliases=ALIASED_SPECIALS_LISTING.special_aliases
            ),
        )

    @pytest.mark.parametrize("al_id", [1, 2])
    def test_a_torrent_another_series_places_against_the_aliases_skips_the_release(self, al_id: int) -> None:
        # The stored pending import also claims the torrent for another series: there every file places by its
        # number, against what the aliases say, so the torrent waits for a hand import. Either title skips it: the
        # aliases placed or claimed its files under the first, the second placed them.
        grabbing_title = (
            self._specials_grabbing_title(2, 4)
            if al_id == 1
            else make_grabbing_title(self._OTHER, self._OTHER, al_id=2, series_id=8)
        )

        placement = place_release(self._specials(), grabbing_title, self._aliased_against((901, 902, 903)))

        assert verdicts_of(placement.placements) == {PlacementVerdict.MISNUMBERED}
        assert placement.skip_reason is SkipReason.MISNUMBERED
        assert placement.covered_ids == frozenset()

    def test_a_file_the_aliases_point_at_that_another_series_places_skips_the_release(self) -> None:
        # The aliases point the third file at special 6, outside this window, and the other series places it by
        # its number. The first two still place.
        placement = place_release(self._specials(), self._specials_grabbing_title(2, 4), self._aliased_against((903,)))

        assert placement.skip_reason is SkipReason.MISNUMBERED
        assert placement.placements.placed == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (504,)}

    def test_a_file_misnumbered_under_another_titles_window_skips_nothing(self) -> None:
        # The stored pending import claims season 1 for title 1, where the first file's numbers disagree. Title 2's
        # window reads that file as another season's, and places the other two.
        first = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, absolute=n) for n in (1, 2, 3)]
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0, absolute=3 + n) for n in (1, 2)]
        series = episode_index([*first, *second])
        files = [
            ListedFile("Show - S01E02 (0003).mkv", 10, _matched(1, 2)),
            ListedFile("Show - S02E01 (0004).mkv", 20, _matched(2, 1)),
            ListedFile("Show - S02E02 (0005).mkv", 30, _matched(2, 2)),
        ]
        pending = pending_import(
            infohash="h1", seadex_files=[f.file_name for f in files], ordered_episode_ids=(101, 102, 103)
        )
        grabbing_title = make_grabbing_title(episode_index(second), series, al_id=2)

        placement = place_release(files, grabbing_title, known_torrent(pending, indexes={7: series}))

        assert PlacementVerdict.MISNUMBERED in verdicts_of(placement.placements)
        assert placement.skip_reason is None
        assert placement.episodes == (ReleaseEpisode(2, 1, 20), ReleaseEpisode(2, 2, 30))

    def test_an_unread_listing_skips_the_release_and_places_nothing(self) -> None:
        placement = place_release(self._specials(), self._specials_grabbing_title(1, 2), self._listing(None))

        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert placement.placements.placements == ()
        assert placement.episodes == ()
        assert not placement.inputs_read

    def test_an_unread_parse_under_a_specials_listing_skips_the_release(self) -> None:
        files = [ListedFile(f"Show.S00E{n:02d}.mkv", 10 * n, None if n == 2 else _matched(0, n)) for n in (1, 2, 3)]

        placement = place_release(files, self._specials_grabbing_title(1, 2), self._listing((501, 502, 503)))

        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read

    def test_an_unread_listing_over_a_seasoned_window_places_nothing_and_skips(self) -> None:
        # Another title the listing names could have some of the torrent, so nothing is placed until it's read. The
        # episodes by size weren't read either, so the release is skipped until they are.
        placement = place_release(_season_files(), _grabbing_title(_SEASON), _torrent(listed=None))

        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read
        assert placement.placements.placements == ()

    def test_a_file_less_release_under_an_unread_listing_waits_on_nothing(self) -> None:
        placement = place_release([], self._specials_grabbing_title(1, 2), self._listing(None))

        assert placement.inputs_read
        assert placement.skip_reason is None

    def test_attach_writes_each_releases_skip_reason(self) -> None:
        # The third release's listing is two wide against a three-file torrent: the listing stands down, no skip.
        urls = ("u1", "u2", "u3")
        seadex_dict: SeadexDict = {
            "RG": rg_group({u: make_release(url=u, infohash=f"h{i}", download=True) for i, u in enumerate(urls, 1)})
        }
        placed = EntryPlacements.place(
            self._specials_grabbing_title(2, 4),
            dict.fromkeys(urls, self._specials()),
            {
                "u1": self._listing((502, 504, 506)),
                "u2": self._listing(None),
                "u3": self._listing((502, 504)),
            },
        )

        placed.attach_placements(seadex_dict)

        holds = [seadex_dict["RG"].releases[u].skip_reason for u in urls]
        assert holds == [SkipReason.MISNUMBERED, SkipReason.INPUT_UNREAD, None]
        assert placed.input_unread_groups(seadex_dict) == ("RG",)


class TestPlaceBesideAnotherListedTitle:
    """A release another listed title shares: skipped as that title's only when none of it lands on this title."""

    def test_a_title_placing_some_beside_a_sibling_placing_the_rest_is_not_skipped(self) -> None:
        torrent = _torrent(listed=[title_window(1, (101, 102)), title_window(2, (103, 104))])

        placement = place_release(_season_files(), _grabbing_title(_SEASON[:2]), torrent)

        assert by_name(placement.placements) == {
            "show - s01e01.mkv": ((101,), PlacementVerdict.EXACT),
            "show - s01e02.mkv": ((102,), PlacementVerdict.EXACT),
            "show - s01e03.mkv": ((), PlacementVerdict.ANOTHER_TITLES),
            "show - s01e04.mkv": ((), PlacementVerdict.ANOTHER_TITLES),
        }
        assert placement.skip_reason is None

    def test_a_file_only_a_sibling_left_unplaced_is_the_siblings(self) -> None:
        # Title 1's window reads the first file's numbers as disagreeing, and places the second. Title 2's window
        # read neither as title 2's, so the unplaced file is title 1's to report, not this title's hand import.
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, absolute=n) for n in (1, 2, 3)]
        series = episode_index([_SPECIAL, *season])
        files = [
            ListedFile("Show - S01E02 (0003).mkv", 10, _matched(1, 2)),
            ListedFile("Show - S01E03 (0003).mkv", 20, _matched(1, 3)),
        ]
        grabbing_title = make_grabbing_title(episode_index([_SPECIAL]), series, al_id=2)
        torrent = known_torrent(indexes={7: series}, evidence=torrent_evidence(title_window(1, (101, 102, 103))))

        placement = place_release(files, grabbing_title, torrent)

        assert by_name(placement.placements)["show - s01e02 (0003).mkv"] == ((), PlacementVerdict.ANOTHER_TITLES)
        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=2, unplaced=0)

    def test_attach_writes_each_releases_skip_counts(self) -> None:
        # Title 3's specials window places nothing of the season pack the two season titles take.
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": make_release(url="u1", infohash="h1", download=True)})}
        torrent = _torrent(listed=[title_window(1, (101, 102)), title_window(2, (103, 104))])
        placed = EntryPlacements.place(_grabbing_title([_SPECIAL], al_id=3), {"u1": _season_files()}, {"u1": torrent})

        placed.attach_placements(seadex_dict)

        item = seadex_dict["RG"].releases["u1"]
        assert (item.skip_reason, item.another_titles_counts) == (
            SkipReason.PLACED_ELSEWHERE,
            AnotherTitlesCounts(another_titles=4, unplaced=0),
        )

    def test_a_stored_file_on_an_episode_no_other_window_holds_is_nobodys(self) -> None:
        # The file map puts the torrent's last two files past this title's window, and no other listed title has
        # those episodes, so nothing is another title's and the release isn't skipped.
        pending = self._pending_import_of_3_and_4()

        placement = place_release(
            _season_files()[2:], _grabbing_title(_SEASON[:2]), _torrent(pending, listed=[title_window(1, (101, 102))])
        )

        assert placement.skip_reason is None
        assert placement.another_titles_counts == NO_ANOTHER_TITLES_COUNTS
        assert placement.covered_ids == frozenset()

    def test_a_stored_file_on_an_episode_a_sibling_window_holds_is_the_siblings(self) -> None:
        pending = self._pending_import_of_3_and_4()
        listed = [title_window(1, (101, 102)), title_window(2, (103, 104))]

        placement = place_release(_season_files()[2:], _grabbing_title(_SEASON[:2]), _torrent(pending, listed=listed))

        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=2, unplaced=0)

    def test_a_kept_title_runs_before_an_ignored_one_of_its_kind(self) -> None:
        # Both windows are seasons of four, and the numbered sequence fits either. The ignored title has the lower
        # id, but the kept title's window places first, so the sequence is the kept title's and nothing is skipped.
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0) for n in range(1, 5)]
        series = episode_index([*_SEASON, *second])
        files = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN, 1)]
        grabbing_title = make_grabbing_title(episode_index(_SEASON), series, al_id=2)
        listed = torrent_evidence(title_window(1, (ep.id for ep in second), ignored=True))

        placement = place_release(files, grabbing_title, known_torrent(indexes={7: series}, evidence=listed))

        assert placement.covered_ids == {101, 102, 103, 104}
        assert placement.skip_reason is None

    def test_a_kept_title_runs_before_a_stored_claim_of_an_ignored_one(self) -> None:
        # Title 1 claimed the sequence before it was ignored, and its stored claim can't know that. The listings
        # flag its listed window, so its claim's window runs after the kept title's: the sequence is title 2's and
        # nothing is skipped.
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0) for n in range(1, 5)]
        series = episode_index([*_SEASON, *second])
        files = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN, 1)]
        pending = pending_import(
            infohash="h1", al_id=1, ordered_episode_ids=(101, 102, 103, 104), file_episode_map={}, seadex_files=_RUN
        )
        listed = torrent_evidence(
            title_window(1, (101, 102, 103, 104), ignored=True), title_window(2, (ep.id for ep in second))
        )
        grabbing_title = make_grabbing_title(episode_index(second), series, al_id=2)

        placement = place_release(files, grabbing_title, known_torrent(pending, indexes={7: series}, evidence=listed))

        assert placement.covered_ids == {201, 202, 203, 204}
        assert placement.skip_reason is None

    @staticmethod
    def _pending_import_of_3_and_4() -> PendingImport:
        """A stored pending import of the torrent's last two files, placed on the season's third and fourth episodes."""

        return pending_import(
            infohash="h1",
            series_id=7,
            file_episode_map={"show - s01e03.mkv": [103], "show - s01e04.mkv": [104]},
            seadex_files=_SEASON_NAMES[2:],
            ordered_episode_ids=(101, 102, 103, 104),
        )

    def test_a_sequence_waiting_on_a_parse_under_a_sibling_skips_the_release_as_unread(self) -> None:
        # The numbered sequence fits title 1's window, and the NCOP's parse is unread.
        files = [ListedFile(name, 10 * n, ParsedFileInfo()) for n, name in enumerate(_RUN, 1)]
        files.append(ListedFile("Show - NCOP.mkv", 50, None))
        torrent = _torrent(listed=[title_window(1, (101, 102, 103, 104)), title_window(2, (201, 202))])

        placement = place_release(files, _grabbing_title(_SERIES[-2:], al_id=2), torrent)

        assert by_name(placement.placements) == {
            **{normalize_file_name(name): ((), PlacementVerdict.WAITING_ON_PARSE) for name in _RUN},
            "show - ncop.mkv": ((), PlacementVerdict.EXTRA),
        }
        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert not placement.inputs_read


# neko-raws' The Rolling Girls BD batch (nyaa 743477) and its Sonarr series. SeaDex lists the batch under the
# season's entry (AniList 20815) and the Chibi shorts' (AniList 102077), whose window is the series' 12 specials.
_RG_EPISODES = (
    "King of Rookies",
    "Center of the World",
    "I Wanna Be a Hero",
    "Dream",
    "Disappointing Person",
    "Lightning Speed",
    "Give Me the Stars",
    "After the Rain",
    "Thieves of the Night",
    "No No No",
    "Rose of Passion",
    "The Future Is in Our Hands",
)
_RG_SHORTS = (
    "Chibi #1 Special Move",
    "Chibi #2 Nickname",
    "Chibi #3 I Can't Hear You",
    "Chibi #4 Sorting Duty",
    "Chibi #5 Summoning",
    "Chibi #6 Please Take This Seriously",
    "Chibi #7 If I Were to Become a Best",
    "Chibi #8 Identity",
    "Chibi #9 Letting Your Guard Down",
    "Chibi #10 Gossip",
    "Chibi #11 Fallen Over",
    "Chibi #12 Hero",
)
_RG_SEASON = [
    sonarr_ep(1, n, ep_id=40905 + n, absolute=n, title=title, episode_file_id=0)
    for n, title in enumerate(_RG_EPISODES, 1)
]
_RG_SPECIALS = [
    sonarr_ep(0, n, ep_id=40893 + n, title=title, episode_file_id=0) for n, title in enumerate(_RG_SHORTS, 1)
]
_RG_SERIES = episode_index([*_RG_SPECIALS, *_RG_SEASON])
_RG_MAIN = TitleNames(series="The Rolling Girls", anilist=("The Rolling Girls", "Rolling☆Girls"))
_RG_CHIBI = TitleNames(series="The Rolling Girls", anilist=("Rolling☆Girls: Chibi☆Rolling☆Girls Korokoro Gekijou",))
_RG_SEASON_WINDOW = title_window(20815, (ep.id for ep in _RG_SEASON), series_id=919, names=_RG_MAIN)
_RG_CHIBI_WINDOW = title_window(102077, (ep.id for ep in _RG_SPECIALS), series_id=919, names=_RG_CHIBI)
_RG_EVIDENCE = torrent_evidence(_RG_SEASON_WINDOW, _RG_CHIBI_WINDOW)


def _rg_name(stem: str) -> str:
    return f"[neko-raws]The Rolling Girls{stem}[BD][1080p][Hi10p][FLAC].mkv"


_RG_EPISODE_SIZES = (
    3558150369,
    3973711515,
    3941779309,
    3990822624,
    4203858580,
    4468218825,
    4160475173,
    4450357433,
    3911101136,
    4210091934,
    4411058595,
    4226868601,
)
_RG_SHORT_SIZES = (131743215, 114360566, 86421088, 70253068, 99820463, 143155027)
# Sonarr reads each episode's plain number as an absolute and matches it to the season, and matches nothing for
# the shorts. It reads nothing at all from the OP and ED.
_RG_FILES = (
    *(
        ListedFile(
            _rg_name(f" - {n:02d}"),
            size,
            ParsedFileInfo(
                season_number=0,
                absolute_episode_numbers=(n,),
                matched_episodes=(MatchedEpisode(season_number=1, episode_number=n, id=40905 + n),),
            ),
        )
        for n, size in enumerate(_RG_EPISODE_SIZES, 1)
    ),
    *(
        ListedFile(
            _rg_name(f" Chibi Special - {n:02d}"), size, ParsedFileInfo(season_number=0, absolute_episode_numbers=(n,))
        )
        for n, size in enumerate(_RG_SHORT_SIZES, 1)
    ),
    ListedFile(_rg_name(" - OP"), 305872174, ParsedFileInfo()),
    ListedFile(_rg_name(" - ED"), 541996641, ParsedFileInfo()),
)
_RG_EPISODE_NAMES = [normalize_file_name(f.file_name) for f in _RG_FILES[:12]]
# Under either window the shorts place nowhere, and the OP and ED are extras.
_RG_REST = {
    **{normalize_file_name(f.file_name): ((), PlacementVerdict.NO_EPISODE_FOUND) for f in _RG_FILES[12:18]},
    **{normalize_file_name(f.file_name): ((), PlacementVerdict.EXTRA) for f in _RG_FILES[18:]},
}


class TestSeasonBatchListedOnTwoTitles:
    """The real batch: its 12 episodes are the season's under either title, never the Chibi title's specials."""

    @staticmethod
    def _place(
        al_id: int, pending: PendingImport | None = None, evidence: FetchedSeadex = _RG_EVIDENCE
    ) -> PlacedRelease:
        main = al_id == 20815
        grabbing_title = make_grabbing_title(
            episode_index(_RG_SEASON if main else _RG_SPECIALS),
            _RG_SERIES,
            al_id=al_id,
            series_id=919,
            names=_RG_MAIN if main else _RG_CHIBI,
        )
        return place_release(
            _RG_FILES, grabbing_title, known_torrent(pending, indexes={919: _RG_SERIES}, evidence=evidence)
        )

    def test_under_the_season_title_the_episodes_place_exact_on_the_season(self) -> None:
        placement = self._place(20815)

        assert by_name(placement.placements) == {
            **{
                name: ((ep.id,), PlacementVerdict.EXACT) for name, ep in zip(_RG_EPISODE_NAMES, _RG_SEASON, strict=True)
            },
            **_RG_REST,
        }
        assert placement.skip_reason is None
        assert placement.covered_ids == {ep.id for ep in _RG_SEASON}

    def test_under_the_chibi_title_alone_the_episodes_are_the_seasons_and_the_release_is_skipped(self) -> None:
        # The season's window judges the episodes first, and the season's title has no claim on a pending import yet,
        # so they're set aside as its files. Nothing lands on the Chibi title, so the release is skipped as another
        # title's.
        placement = self._place(102077)

        assert by_name(placement.placements) == {
            **dict.fromkeys(_RG_EPISODE_NAMES, ((), PlacementVerdict.ANOTHER_TITLES)),
            **_RG_REST,
        }
        assert placement.episodes == ()
        assert placement.covered_ids == frozenset()
        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=12, unplaced=6)

    def test_under_the_chibi_title_with_the_season_title_ignored_the_release_is_still_skipped(self) -> None:
        # The season's title is on the ignore list: nobody grabs for it, but its window still sets the episodes
        # aside as the season's, so they never land on the Chibi title's specials. A season window runs before a
        # specials window whether or not its title is ignored, so the placement is the kept title's.
        evidence = torrent_evidence(replace(_RG_SEASON_WINDOW, ignored=True), _RG_CHIBI_WINDOW)

        placement = self._place(102077, evidence=evidence)
        kept = self._place(102077)

        assert placement.placements == kept.placements
        assert (placement.episodes, placement.skip_reason, placement.another_titles_counts) == (
            kept.episodes,
            kept.skip_reason,
            kept.another_titles_counts,
        )

    def test_under_the_chibi_title_beside_the_seasons_pending_import_the_release_is_skipped(self) -> None:
        pending = pending_import(
            infohash="h1",
            al_id=20815,
            series_id=919,
            file_episode_map={name: (ep.id,) for name, ep in zip(_RG_EPISODE_NAMES, _RG_SEASON, strict=True)},
            seadex_files=[f.file_name for f in _RG_FILES],
            ordered_episode_ids=tuple(ep.id for ep in _RG_SEASON),
        )

        placement = self._place(102077, pending)

        # The file map already puts the episodes on the season, so only the rest is placed, and the Chibi title
        # gets none of the episodes.
        assert by_name(placement.placements) == _RG_REST
        assert placement.episodes == ()
        assert placement.covered_ids == frozenset()
        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=12, unplaced=6)


def _absolute_only(number: int, title: str) -> ParsedFileInfo:
    """Sonarr's parse of a name carrying `number` alone: season 0, the absolute, and the parsed series name."""

    return parsed_info(season=0, absolutes=(number,), series_title=title)


# Tsundere's Stray Cats Overrun! BD batch (nyaa 1050453): twelve episodes named by number alone, six SP files, listed
# under the season's entry (AniList 7590) and the specials' (AniList 8599). Sonarr matched the numbers to nothing.
_SC_SPECIALS = [sonarr_ep(0, n, ep_id=32761 + n, episode_file_id=0) for n in range(1, 8)]
_SC_SEASON = [sonarr_ep(1, n, ep_id=32768 + n, episode_file_id=0, absolute=n) for n in range(1, 14)]
_SC_SERIES = episode_index([*_SC_SPECIALS, *_SC_SEASON])
_SC_MAIN = TitleNames("Stray Cats Overrun!", ("Mayoi Neko Overrun!",))
_SC_EXTRAS = TitleNames("Stray Cats Overrun!", ("Mayoi Neko Overrun! Specials",))
_SC_SEASON_WINDOW = title_window(7590, (ep.id for ep in _SC_SEASON), names=_SC_MAIN, episode_count=13)
_SC_SPECIALS_WINDOW = title_window(8599, (ep.id for ep in _SC_SPECIALS[:6]), names=_SC_EXTRAS, episode_count=6)
_SC_EPISODES = [
    f"[Tsundere] Mayoi Neko Overrun! - {n:02d} [BDRip h264 1920x1080 FLAC][746CE979].mkv" for n in range(1, 13)
]
_SC_SPS = [f"[Tsundere] Mayoi Neko Overrun! - SP{n} [BDRip h264 1920x1080 FLAC][B66672E2].mkv" for n in range(1, 7)]
_SC_FILES = [
    *(
        ListedFile(name, 1_000_000_000 + n, _absolute_only(n, "Mayoi Neko Overrun!"))
        for n, name in enumerate(_SC_EPISODES, 1)
    ),
    *(ListedFile(name, 100_000_000 + n, ParsedFileInfo()) for n, name in enumerate(_SC_SPS, 1)),
]


# My Hero Academia (Sonarr 254): the Ikinokore OVA title (AniList 122349) is specials 8 and 9. HorribleSubs' single
# 01 reads S01E01 by the series' numbers, outside the window, so the passes leave it. The window counts its specials
# from one.
_MHA_SERIES = episode_index(
    [
        *(sonarr_ep(0, n, ep_id=9246 + n, episode_file_id=0) for n in range(7, 11)),
        *(sonarr_ep(1, n, ep_id=9270 + n, episode_file_id=0, absolute=n) for n in range(1, 4)),
    ]
)
_MHA_NAMES = TitleNames(
    "My Hero Academia",
    (
        "My Hero Academia: Make It! Do-or-Die Survival Training",
        "Boku no Hero Academia: Ikinokore! Kesshi no Survival Kunren",
    ),
)
_MHA_GRABBING_TITLE = make_grabbing_title(
    episode_index([_MHA_SERIES.by_id[9254], _MHA_SERIES.by_id[9255]]),
    _MHA_SERIES,
    al_id=122349,
    names=_MHA_NAMES,
    episode_count=2,
    alternate_titles=("Boku no Hero Academia",),
)
_MHA_WINDOW = title_window(122349, (9254, 9255), names=_MHA_NAMES, episode_count=2)
_MHA_OVA = "[HorribleSubs] Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren - 01 [1080p].mkv"
_MHA_FILES = (
    ListedFile(
        _MHA_OVA, 1_400_000_000, _absolute_only(1, "Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren")
    ),
)


class TestPlaceWhereEveryNumberingAgrees:
    """An absolute-only file the windows left is placed where every numbering agrees, after the merge."""

    def test_a_single_a_specials_only_entry_lists_places_on_its_first_special(self) -> None:
        torrent = known_torrent(indexes={7: _MHA_SERIES}, evidence=torrent_evidence(_MHA_WINDOW))

        placement = place_release(_MHA_FILES, _MHA_GRABBING_TITLE, torrent)

        assert by_name(placement.placements) == {
            normalize_file_name(_MHA_OVA): ((9254,), PlacementVerdict.NUMBERINGS_AGREE)
        }
        assert placement.covered_ids == {9254}
        assert placement.episodes == (ReleaseEpisode(0, 8, 1_400_000_000),)
        assert placement.skip_reason is None

    def test_a_release_the_file_map_covers_places_nothing_and_records_the_file_map(self) -> None:
        # The file map holds the single where an earlier grab placed it. The numbering step would propose that
        # special for it again, but no file is left to place.
        pending = pending_import(
            infohash="h1",
            al_id=122349,
            file_episode_map={normalize_file_name(_MHA_OVA): [9254]},
            seadex_files=[_MHA_OVA],
            ordered_episode_ids=(9254, 9255),
        )
        torrent = known_torrent(pending, indexes={7: _MHA_SERIES}, evidence=torrent_evidence(_MHA_WINDOW))

        placement = place_release(_MHA_FILES, _MHA_GRABBING_TITLE, torrent)

        assert placement == PlacedRelease(
            files=_MHA_FILES,
            placements=BatchPlacements(()),
            episodes=(ReleaseEpisode(0, 8, 1_400_000_000),),
            covered_ids=frozenset({9254}),
            inputs_read=True,
            skip_reason=None,
            another_titles_counts=NO_ANOTHER_TITLES_COUNTS,
            pending_import=pending,
            intended_sizes={9254: 1_400_000_000},
        )

    def test_a_batch_counted_from_one_places_on_the_seasons_title(self) -> None:
        grabbing_title = make_grabbing_title(
            episode_index(_SC_SEASON), _SC_SERIES, al_id=7590, names=_SC_MAIN, episode_count=13
        )
        torrent = known_torrent(
            indexes={7: _SC_SERIES}, evidence=torrent_evidence(_SC_SEASON_WINDOW, _SC_SPECIALS_WINDOW)
        )

        placement = place_release(_SC_FILES, grabbing_title, torrent)

        assert by_name(placement.placements) == {
            **{
                normalize_file_name(name): ((ep.id,), PlacementVerdict.NUMBERINGS_AGREE)
                for name, ep in zip(_SC_EPISODES, _SC_SEASON, strict=False)
            },
            **dict.fromkeys((normalize_file_name(name) for name in _SC_SPS), ((), PlacementVerdict.NO_EPISODE_FOUND)),
        }
        assert placement.covered_ids == {ep.id for ep in _SC_SEASON[:12]}
        assert placement.skip_reason is None

    def test_under_the_specials_title_the_batch_is_the_seasons_and_the_release_is_skipped(self) -> None:
        # The numbering step places the twelve under the season's window, so for the specials' title they're another
        # title's, as a file the season's window had placed would be: no hand import, and nothing of the release
        # lands here.
        grabbing_title = make_grabbing_title(
            episode_index(_SC_SPECIALS[:6]), _SC_SERIES, al_id=8599, names=_SC_EXTRAS, episode_count=6
        )
        torrent = known_torrent(
            indexes={7: _SC_SERIES}, evidence=torrent_evidence(_SC_SEASON_WINDOW, _SC_SPECIALS_WINDOW)
        )

        placement = place_release(_SC_FILES, grabbing_title, torrent)

        assert {by_name(placement.placements)[normalize_file_name(name)] for name in _SC_EPISODES} == {
            ((), PlacementVerdict.ANOTHER_TITLES)
        }
        assert placement.episodes == ()
        assert placement.skip_reason is SkipReason.PLACED_ELSEWHERE
        assert placement.another_titles_counts == AnotherTitlesCounts(another_titles=12, unplaced=6)


class TestKnownTorrent:
    """The windows a torrent is placed under: every listed title's, every stored claim's, and the grabbing title's."""

    def test_an_unread_listing_orders_nothing(self) -> None:
        assert _torrent(listed=None).order(_grabbing_title(_SEASON).window) is None

    def test_an_unread_listing_gives_no_evidence(self) -> None:
        # Whether the unread listing holds the release is `place_release`'s call: the windows build as if listed
        # nowhere.
        torrent = known_torrent(evidence=UNREAD_EVIDENCE)

        assert torrent.seadex_evidence({"show - 01.mkv": 10}) == NO_EVIDENCE

    def test_the_evidence_gives_each_size_match_its_episode(self) -> None:
        torrent = known_torrent(
            evidence=torrent_evidence(title_window(1, (101, 102)), episode_by_size={10: 101, 30: 103})
        )

        evidence = torrent.seadex_evidence({"show - 01.mkv": 10, "show - 02.mkv": 20})

        assert evidence == SeadexEvidence(torrent_listing(title_window(1, (101, 102))), {"show - 01.mkv": 101})

    def test_the_evidence_carries_the_listings_aliases(self) -> None:
        torrent = known_torrent(evidence=torrent_evidence(title_window(1, (501, 502)), aliases={1: 2}))

        assert torrent.seadex_evidence({}) == SeadexEvidence(
            torrent_listing(title_window(1, (501, 502)), aliases={1: 2})
        )

    def test_the_grabbing_titles_window_replaces_its_claims(self) -> None:
        order = _torrent(two_claim_pending_import()).order(_grabbing_title(_SEASON).window)

        assert order is not None
        assert [w.ordered_episode_ids for w in order.windows] == [(101, 102, 103, 104), (103, 104)]
        assert (order.grabbing_window_at, order.imports) == (0, {0, 1})

    def test_the_grabbing_titles_window_carries_its_anilist_episode_count(self) -> None:
        grabbing_title = make_grabbing_title(episode_index(_SEASON), episode_index(_SERIES), episode_count=4)

        assert grabbing_title.window == title_window(1, (101, 102, 103, 104), episode_count=4)

    def test_a_new_grabbing_titles_window_takes_its_anilist_ids_place(self) -> None:
        order = _torrent(two_claim_pending_import()).order(_grabbing_title(_SEASON, al_id=3).window)

        assert order is not None
        assert [w.al_id for w in order.windows] == [1, 2, 3]
        assert (order.grabbing_window_at, order.imports) == (2, {0, 1, 2})

    def test_a_listed_title_that_never_claimed_runs_but_is_not_imported(self) -> None:
        order = _torrent(listed=[title_window(2, (103, 104))]).order(_grabbing_title(_SEASON).window)

        assert order is not None
        assert [w.al_id for w in order.windows] == [1, 2]
        assert (order.grabbing_window_at, order.imports) == (0, {0})

    def test_a_claim_moving_series_leaves_its_old_series_unread(self) -> None:
        # Title 1's stored claim is on series 7, and this grab claims series 8.
        pending = pending_import(
            infohash="h1",
            al_id=1,
            ordered_episode_ids=(101, 102, 103, 104),
            file_episode_map={},
            seadex_files=_SEASON_NAMES,
        )
        other = episode_index([sonarr_ep(1, n, ep_id=800 + n, episode_file_id=0) for n in range(1, 5)])
        fresh = torrent_evidence(title_window(1, (801, 802, 803, 804), series_id=8))
        torrent = known_torrent(pending, indexes={8: other}, evidence=fresh)

        order = torrent.order(make_grabbing_title(other, other, al_id=1, series_id=8).window)

        assert order is not None
        assert [(w.al_id, w.series_id) for w in order.windows] == [(1, 8)]
        assert order.imports == {0}


class TestTorrentFetches:
    """What the run read of a title's torrents, keyed by url."""

    @staticmethod
    def _dict() -> SeadexDict:
        return {
            "RG": rg_group(
                {
                    "u1": make_release(url="u1", infohash="h1", download=True),
                    "u2": make_release(url="u2", infohash=None, download=True),
                    "u3": make_release(url="u3", infohash="h3", download=True),
                }
            ),
        }

    def test_entry_hashes_leave_out_a_hash_less_release(self) -> None:
        assert entry_hashes(self._dict()) == {"h1", "h3"}

    def test_known_by_url_over_the_stored_pending_imports_and_listings(self) -> None:
        pending = _stored_pending_import()
        listing = torrent_listing(title_window(1, (101,)))
        reads = TorrentFetches({"h1": pending}, _INDEXES, FetchedListings({"h1": listing, "h3": None}, None))

        known = reads.known(self._dict())

        # u2 has no hash, so it's new and unlisted. u3 has no pending import and is listed under an unread window.
        # That unread window means no episodes by size for any release, u1 included.
        assert known == {
            "u1": KnownTorrent(pending, _INDEXES, FetchedSeadex(listing, None)),
            "u2": KnownTorrent(None, _INDEXES, FetchedSeadex(EMPTY_LISTING, None)),
            "u3": KnownTorrent(None, _INDEXES, FetchedSeadex(None, None)),
        }

    def test_a_marked_release_reads_its_pending_import_by_its_cross_seed_and_its_listing_by_seadexs_hash(self) -> None:
        seadex_dict = self._dict()
        seadex_dict["RG"].releases["u1"].cross_seed_hash = "c1"
        pending = _stored_pending_import()
        listing = torrent_listing(title_window(1, (101,)))
        reads = TorrentFetches({"c1": pending}, _INDEXES, FetchedListings({"h1": listing, "h3": None}, None))

        assert entry_hashes(seadex_dict) == {"c1", "h3"}
        assert reads.known(seadex_dict)["u1"] == KnownTorrent(pending, _INDEXES, FetchedSeadex(listing, None))

    def test_an_unreachable_source_leaves_every_hash_unread_and_a_hash_less_release_listed_nowhere(self) -> None:
        known = TorrentFetches({}, _INDEXES, None).known(self._dict())

        assert {url: torrent.fetched for url, torrent in known.items()} == {
            "u1": FetchedSeadex(None, None),
            "u2": FetchedSeadex(EMPTY_LISTING, None),
            "u3": FetchedSeadex(None, None),
        }


class TestEntryPlacements:
    """The per-title merge: episodes onto the release dict, the groups a failed read skipped, and the updates."""

    @staticmethod
    def _entry(names: TitleNames | None = None) -> tuple[EntryPlacements, SeadexDict]:
        grabbing_title = _grabbing_title(_SEASON, names)
        run = [ListedFile(name, 10 * n, _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)]
        listed_files = [
            ListedFile(name, 1, None if n == 1 else _number_with_no_match(n)) for n, name in enumerate(_RUN, start=1)
        ]
        placed = EntryPlacements.place(
            grabbing_title, {"u1": run, "u2": listed_files, "u3": []}, dict.fromkeys(("u1", "u2", "u3"), _NEW)
        )
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {"u1": make_release(url="u1", files=_RUN, size=[10, 20, 30, 40], infohash="h1", download=True)}
            ),
            "Held": rg_group({"u2": make_release(url="u2", files=_RUN, size=[1] * 4, infohash="h2", download=True)}),
            "Subs": rg_group(
                {"u3": make_release(url="u3", files=["Show.ass"], size=[1], infohash="h3", download=True)}
            ),
        }
        return placed, seadex_dict

    def test_attach_writes_each_releases_episodes_and_the_group_union(self) -> None:
        placed, seadex_dict = self._entry()

        placed.attach_placements(seadex_dict)

        sequence_episodes = [ReleaseEpisode(1, n, 10 * n) for n in range(1, 5)]
        assert seadex_dict["RG"].releases["u1"].episodes == sequence_episodes
        assert seadex_dict["RG"].all_episodes == sequence_episodes
        assert seadex_dict["Held"].releases["u2"].episodes == []
        assert seadex_dict["Held"].all_episodes == []
        assert seadex_dict["Subs"].all_episodes == []

    def test_input_unread_groups_name_the_skipped_release(self) -> None:
        placed, seadex_dict = self._entry()

        assert placed.input_unread_groups(seadex_dict) == ("Held",)

    def test_pending_import_updates_include_the_placement_and_carry_the_windows_names(self) -> None:
        names = TitleNames("Show", ("Show", "Shou"))
        placed, seadex_dict = self._entry(names)
        facts = title_facts()

        listed_files = build_pending_import_updates(seadex_dict, placed, facts)

        assert set(listed_files) == {"h1", "h2"}
        assert dict(listed_files["h1"].file_map) == {
            normalize_file_name(name): (100 + n,) for n, name in enumerate(_RUN, 1)
        }
        listed = {normalize_file_name(name): 10 * n for n, name in enumerate(_RUN, 1)}
        assert listed_files["h1"].facts == torrent_facts(
            seadex_files=tuple(_RUN), release_sizes=(10, 20, 30, 40), sizes_by_name=listed
        )
        assert listed_files["h1"].claim.names == names
        assert listed_files["h1"].pending_import is None
        assert dict(listed_files["h2"].file_map) == {}

    def test_a_marked_release_is_stored_under_its_cross_seed(self) -> None:
        placed, seadex_dict = self._entry()
        seadex_dict["RG"].releases["u1"].cross_seed_hash = "c1"

        listed_files = build_pending_import_updates(seadex_dict, placed, title_facts())

        assert set(listed_files) == {"c1", "h2"}
        pending_imports = {key: update.pending_import_at(_STAMP, fresh=True) for key, update in listed_files.items()}
        assert (pending_imports["c1"].infohash, pending_imports["c1"].is_cross_seed) == ("c1", True)
        assert pending_imports["h2"].is_cross_seed is False

    def test_a_release_with_a_stored_pending_import_is_stored_on_it(self) -> None:
        pending = _stored_pending_import()
        placed = EntryPlacements.place(_grabbing_title(_SEASON), {"u1": _season_files()}, {"u1": _torrent(pending)})
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {"u1": make_release(url="u1", files=_SEASON_NAMES, size=[10, 20, 30, 40], infohash="h1", download=True)}
            ),
        }

        listed_files = build_pending_import_updates(seadex_dict, placed, title_facts())

        assert listed_files["h1"].pending_import is pending
        assert dict(listed_files["h1"].file_map) == {"show - s01e03.mkv": (103,), "show - s01e04.mkv": (104,)}


class TestOrderOnlyUpdate:
    """A torrent of numbered names one file longer than its season: the DVD order splits the opener in two halves."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
    _DVD: ClassVar[tuple[OrderRow, ...]] = (
        OrderRow(7, 1, 9001, None),
        OrderRow(7, 2, 9002, None),
        *(OrderRow(7, n + 1, 400 + n, None) for n in range(2, 13)),
        OrderRow(7, 14, 501, None),
        OrderRow(7, 21, 401, None),
    )
    """The DVD order: two halves Sonarr doesn't list, the aired second episode on, then the whole opener."""

    def test_the_update_carries_the_halves_dvd_keys_onto_the_pending_import(self) -> None:
        season = [sonarr_ep(4, n, ep_id=400 + n, episode_file_id=0, tvdb_id=400 + n) for n in range(1, 13)]
        opener = sonarr_ep(5, 1, ep_id=501, episode_file_id=0, tvdb_id=501)
        series = episode_index([*season, opener], IndexOrders({"dvd": self._DVD}, None))
        grabbing_title = make_grabbing_title(episode_index(season), series, names=TitleNames("Show", ("Show Ending",)))
        files = [ListedFile(name, 10 * n, parsed_info()) for n, name in enumerate(self._RUN, 1)]
        placement = place_release(files, grabbing_title, new_torrent(grabbing_title))
        release = PlacedGrab("grp", make_release(url="u", infohash="h1"), "h1", placement)

        update = build_pending_import_update(release, grabbing_title, title_facts())

        assert update.order_only == {
            normalize_file_name(self._RUN[0]): OrderOnlyKey("dvd", EpisodeKey(7, 1)),
            normalize_file_name(self._RUN[1]): OrderOnlyKey("dvd", EpisodeKey(7, 2)),
        }
        assert update.pending_import_at(_STAMP, fresh=True).order_only == update.order_only


class TestPendingImportAt:
    """The pending import an update stores: a new one, or the stored one with the update added."""

    def test_a_new_pending_import_starts_at_the_stamp(self) -> None:
        claim = title_claim()
        update = PendingImportUpdate(
            facts=_FACTS,
            file_map={"show - 01.mkv": (101,)},
            excluded=("show - 02.mkv",),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=claim,
            pending_import=None,
        )

        pending = update.pending_import_at(_STAMP, fresh=True)

        assert update.joined() is None
        assert pending == PendingImport(
            infohash="h1",
            release_group="RG",
            is_dual_audio=True,
            seadex_files=("Show - 01.mkv",),
            added_at=_STAMP,
            file_episode_map={"show - 01.mkv": (101,)},
            claims=(claim,),
            excluded_files=("show - 02.mkv",),
            release_sizes=(10,),
            sizes_by_name={"show - 01.mkv": 10},
        )
        # A new pending import starts at the stamp whether or not the add was new.
        assert update.pending_import_at(_STAMP, fresh=False) == pending

    def test_joined_merges_the_placements_exclusions_and_claim(self) -> None:
        stored = pending_import(
            infohash="h1",
            added_at="2026-01-01 00:00:00",
            file_episode_map={"show - 01.mkv": [101]},
            excluded_files=("show - 09.mkv",),
            preowned_episode_ids=(101,),
            awaiting_cleanup=True,
        )
        claim = title_claim(ordered_episode_ids=(101, 102, 103), preowned_episode_ids=(102,))
        update = PendingImportUpdate(
            facts=_FACTS,
            file_map={"show - 02.mkv": (102,)},
            excluded=("show - 03.mkv",),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=claim,
            pending_import=stored,
        )

        pending = update.pending_import_at(_STAMP, fresh=False)

        assert update.joined() == pending
        assert dict(pending.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert pending.excluded_files == ("show - 09.mkv", "show - 03.mkv")
        # The title's claim is replaced, its first preowned ids kept.
        assert pending.claims == (replace(claim, preowned_episode_ids=(101,)),)
        assert pending.awaiting_cleanup is False
        # Not a new add: the pending import keeps its `added_at`, and the identity stays the stored one's.
        assert pending.added_at == "2026-01-01 00:00:00"
        assert pending.release_group == stored.release_group

    def test_joined_appends_a_new_titles_claim(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        update = pending_import_update("h1", stored=stored, al_id=2)

        pending = update.pending_import_at(_STAMP, fresh=False)

        assert pending.claims == (*stored.claims, update.claim)

    def test_joined_onto_a_pending_import_without_sizes_by_name_learns_them(self) -> None:
        stored = pending_import(infohash="h1")
        update = PendingImportUpdate(
            facts=_FACTS,
            file_map={},
            excluded=(),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=title_claim(),
            pending_import=stored,
        )

        assert update.pending_import_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 10}

    def test_joined_keeps_the_stored_sizes_by_name(self) -> None:
        stored = pending_import(infohash="h1", sizes_by_name={"show - 01.mkv": 99})
        update = PendingImportUpdate(
            facts=_FACTS,
            file_map={},
            excluded=(),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=title_claim(),
            pending_import=stored,
        )

        assert update.pending_import_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 99}

    def test_a_new_pending_import_keeps_the_hand_import_files(self) -> None:
        update = pending_import_update("h1", hand_import_files=("show - 02.mkv",))

        assert update.pending_import_at(_STAMP, fresh=True).hand_import_files == ("show - 02.mkv",)

    def test_joined_merges_the_hand_import_files(self) -> None:
        stored = pending_import(infohash="h1", hand_import_files=("show - 02.mkv",))

        kept = pending_import_update("h1", stored=stored).pending_import_at(_STAMP, fresh=False)
        joined = pending_import_update("h1", stored=stored, hand_import_files=("show - 03.mkv",)).pending_import_at(
            _STAMP, fresh=False
        )

        assert kept.hand_import_files == ("show - 02.mkv",)
        assert joined.hand_import_files == ("show - 02.mkv", "show - 03.mkv")

    def test_joined_merges_the_order_only_keys(self) -> None:
        halves = [normalize_file_name(f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv") for n in (1, 2)]
        keys = {halves[0]: OrderOnlyKey("dvd", EpisodeKey(7, 1)), halves[1]: OrderOnlyKey("dvd", EpisodeKey(7, 2))}
        stored = pending_import(infohash="h1", hand_import_files=halves[:1], order_only={halves[0]: keys[halves[0]]})

        joined = pending_import_update(
            "h1", stored=stored, hand_import_files=(halves[1],), order_only={halves[1]: keys[halves[1]]}
        ).pending_import_at(_STAMP, fresh=False)

        assert joined.order_only == keys

    def test_fresh_restarts_the_stored_pending_import(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        update = pending_import_update("h1", stored=stored, al_id=2)

        pending = update.pending_import_at(_STAMP, fresh=True)

        assert pending.added_at == _STAMP


class TestBuildUpdateWithNoWindow:
    """A Radarr grab's update: the listing's facts, no files or window, an id-less claim on the title."""

    @staticmethod
    def _planned_grab() -> PlannedGrab:
        return PlannedGrab("RG", "https://nyaa.si/view/1", make_release(infohash="h1", is_dual_audio=True), "h1")

    def test_a_new_torrents_update_has_an_id_less_claim(self) -> None:
        facts = title_facts(al_id=9, series_id=0, title="Movie", url="https://releases.moe/9")

        update = build_update_without_window(self._planned_grab(), facts, None)

        assert update.pending_import is None
        assert update.facts == torrent_facts(is_dual_audio=True)
        assert update.file_map == {} and update.excluded == () and update.order_only == {}
        assert update.claim == title_claim(
            al_id=9, series_id=0, title="Movie", url="https://releases.moe/9", guards=facts.guards
        )

    def test_a_listed_torrent_joins_its_stored_pending_import(self) -> None:
        stored = pending_import(infohash="h1", al_id=1, series_id=0)

        pending = build_update_without_window(
            self._planned_grab(), title_facts(al_id=2, series_id=0), stored
        ).pending_import_at(_STAMP, fresh=False)

        assert pending.release_group == stored.release_group
        assert [claim.al_id for claim in pending.claims] == [1, 2]


class TestTitleClaim:
    """The title's claim reads the whole file map inside the title's window."""

    @staticmethod
    def _release(files: list[ListedFile], placed: PlacedRelease) -> PlacedGrab:
        names = [f.file_name for f in files]
        sizes = [f.size for f in files]
        return PlacedGrab(
            "RG", make_release(url="u1", files=names, size=sizes, infohash="h1", download=True), "h1", placed
        )

    def test_ids_come_from_the_whole_file_map_inside_the_titles_window(self) -> None:
        files = _season_files(3)
        placed = place_release(files, _grabbing_title(_SEASON), _torrent(_stored_pending_import()))

        claim = build_title_claim(self._release(files, placed), _grabbing_title(_SEASON), title_facts())

        # This call placed E03 alone, yet the claim also carries the pending import's E01 and E02.
        assert placed.placements.placed == {"show - s01e03.mkv": (103,)}
        assert claim.claimed_coverage == "S01 E01-E03"
        assert claim.ordered_episode_ids == (101, 102, 103, 104)

    def test_preowned_reads_every_claimed_id(self) -> None:
        # E01 (the pending import's, not this call's) already has this torrent's group's file at a listed size.
        window_episodes = [sonarr_ep(1, 1, ep_id=101, size=10, release_group="RG"), *_SEASON[1:]]
        files = _season_files(3)
        placed = place_release(files, _grabbing_title(window_episodes), _torrent(_stored_pending_import()))

        claim = build_title_claim(self._release(files, placed), _grabbing_title(window_episodes), title_facts())

        assert claim.preowned_episode_ids == (101,)

    def test_this_torrents_misplaced_file_is_not_preowned(self) -> None:
        # E01 has this torrent's group's E02 file (a listed size, not E01's): the grab still owes E01 its file.
        window_episodes = [sonarr_ep(1, 1, ep_id=101, size=20, release_group="RG"), *_SEASON[1:]]
        files = _season_files(3)
        placed = place_release(files, _grabbing_title(window_episodes), _NEW)

        claim = build_title_claim(self._release(files, placed), _grabbing_title(window_episodes), title_facts())

        assert claim.preowned_episode_ids == ()
