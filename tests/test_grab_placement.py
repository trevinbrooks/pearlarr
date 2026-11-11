# pyright: strict
"""The grab-time placement: what `place_release` records for the planner, the record a url accretes onto, and the seeds."""

from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar

import pytest

from pearlarr.grab_placement import (
    EntryPlacements,
    IdentityOverride,
    KnownTorrent,
    PendingSeed,
    SeedFile,
    SeedRelease,
    SeedScope,
    TorrentEvidence,
    TorrentReads,
    UrlPlacement,
    build_entry_claim,
    build_pending_seed,
    build_pending_seeds,
    build_unscoped_seed,
    entry_hashes,
    identity_overrides,
    place_release,
)
from pearlarr.manual_import import (
    NO_ORDER_ONLY,
    EntryNames,
    EntryWindow,
    FileEpisodeMap,
    PendingImport,
    normalize_basename,
    normalized_leaf,
)
from pearlarr.placement_types import (
    EMPTY_LISTING,
    NO_EVIDENCE,
    EpisodeAssignment,
    EpisodeIndex,
    ListingEvidence,
    ListingsRead,
    PlacementVerdict,
    episode_index,
)
from pearlarr.seadex_types import (
    NO_ELSEWHERE,
    ElsewhereCounts,
    EpisodeKey,
    EpisodeRecord,
    FlaggedUrl,
    GrabHold,
    MatchedEpisode,
    OrderOnlyKey,
    ParsedFileInfo,
    SeadexDict,
    SonarrEpisode,
)
from pearlarr.tvdb_orders import OrderRow, SeriesOrdering

from .builders import (
    ALIASED_PACK,
    UNREAD_EVIDENCE,
    by_name,
    entry_claim,
    entry_facts,
    entry_window,
    known_torrent,
    new_torrent,
    parsed_info,
    pending_import,
    pending_seed,
    rg_group,
    seed_scope,
    sonarr_ep,
    torrent_evidence,
    torrent_facts,
    torrent_listing,
    two_claim_record,
    url_item,
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
_PACK = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)]
_INDEXES = {7: episode_index(_SERIES)}
_STAMP = "2026-06-24 00:00:00"
_SPECIALS = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)]
_FACTS = torrent_facts(
    is_dual_audio=True, seadex_files=("Show - 01.mkv",), release_sizes=(10,), sizes_by_name={"show - 01.mkv": 10}
)


def _scope(entry: list[SonarrEpisode], names: EntryNames | None = None, *, al_id: int = 1) -> SeedScope:
    return seed_scope(episode_index(entry), episode_index(_SERIES), al_id=al_id, names=names or EntryNames())


def _bare(number: int) -> ParsedFileInfo:
    """A `- 01` name Sonarr matched nothing for."""

    return ParsedFileInfo(episode_numbers=(number,))


def _matched(season: int, episode: int) -> ParsedFileInfo:
    return ParsedFileInfo(
        season_number=season,
        episode_numbers=(episode,),
        matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),),
    )


def _bare_key(season: int, episode: int) -> ParsedFileInfo:
    """A spelled key Sonarr matched nothing for."""

    return ParsedFileInfo(season_number=season, episode_numbers=(episode,))


def _pack(count: int = 4) -> list[SeedFile]:
    """The season pack's first `count` files, each matched by Sonarr and sized `10 * n`."""

    return [SeedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(_PACK[:count], start=1)]


def _torrent(record: PendingImport | None = None, *, listed: Sequence[EntryWindow] | None = ()) -> KnownTorrent:
    """A `KnownTorrent` over the shared series, by default a new torrent that nothing lists.

    `listed=None` means SeaDex couldn't be read (`UNREAD_EVIDENCE`).
    """

    evidence = UNREAD_EVIDENCE if listed is None else torrent_evidence(*listed)
    return known_torrent(record, indexes=_INDEXES, evidence=evidence)


_NEW = _torrent()


def _resident_record() -> PendingImport:
    """A stored record mapping the pack's first two files, its one claim over the season on series 7."""

    return pending_import(
        infohash="h1",
        series_id=7,
        file_episode_map={"show - s01e01.mkv": [101], "show - s01e02.mkv": [102]},
        seadex_files=_PACK,
        ordered_episode_ids=(101, 102, 103, 104),
    )


class TestPlaceRelease:
    """One url's files placed by the import's own passes, its records the planner's coverage vocabulary."""

    def test_a_run_sonarr_read_nothing_of_records_the_window(self) -> None:
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == tuple(EpisodeRecord(1, n, 10 * n) for n in range(1, 5))
        assert placement.assignment.assigned == {normalized_leaf(name): (100 + n,) for n, name in enumerate(_RUN, 1)}
        assert placement.claimed_ids == {101, 102, 103, 104}
        assert placement.inputs_known
        assert placement.stored is None

    def test_each_placed_episode_intends_its_files_listed_size(self) -> None:
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.intended_sizes == {100 + n: 10 * n for n in range(1, 5)}

    def test_a_whole_series_pack_records_only_the_entrys_slice(self) -> None:
        names = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)] + ["Show - S02E01.mkv", "Show - S02E02.mkv"]
        parses = [_matched(1, n) for n in range(1, 5)] + [_matched(2, 1), _matched(2, 2)]
        files = [SeedFile(name, 1, parse) for name, parse in zip(names, parses, strict=True)]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == tuple(EpisodeRecord(1, n, 1) for n in range(1, 5))
        assert {p.name for p in placement.assignment.excluded} == {"show - s02e01.mkv", "show - s02e02.mkv"}

    def test_a_file_placed_nowhere_leaves_no_record(self) -> None:
        files = [SeedFile("Show - S02E01.mkv", 1, _matched(2, 1))]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == ()
        assert verdicts_of(placement.assignment) == {PlacementVerdict.FOREIGN}

    def test_a_held_run_records_nothing_holds_the_url_and_its_inputs_are_not_known(self) -> None:
        files = [SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == ()
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known
        assert verdicts_of(placement.assignment) == {PlacementVerdict.HELD}

    def test_a_file_beside_a_held_run_still_places(self) -> None:
        files = [SeedFile("Show - OVA.mkv", 5, _matched(0, 1))] + [
            SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _scope([_SPECIAL, *_SEASON]), _NEW)

        assert placement.records == (EpisodeRecord(0, 1, 5),)
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_two_files_sharing_a_leaf_keep_their_own_sizes(self) -> None:
        # The gather already stripped the folders: one leaf, placed once, two listed sizes.
        files = [SeedFile("Show - S01E01.mkv", 10, _matched(1, 1)), SeedFile("Show - S01E01.mkv", 20, _matched(1, 1))]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 1, 20))
        assert placement.assignment.assigned == {"show - s01e01.mkv": (101,)}

    def test_an_episode_two_listed_sizes_share_intends_no_size(self) -> None:
        # Either could be the file intended on E01.
        files = [SeedFile("Show - S01E01.mkv", 10, _matched(1, 1)), SeedFile("Show - S01E01.mkv", 20, _matched(1, 1))]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.intended_sizes == {}

    def test_an_unread_series_map_places_nothing_and_holds_the_url_as_unread(self) -> None:
        # The series list did not serve, so the scope has no map and the entry's window can't be ordered.
        files = [SeedFile(name, 1, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        scope = seed_scope(episode_index(_SEASON), episode_index([]))

        placement = place_release(files, scope, known_torrent())

        assert not scope.can_place
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_no_files_is_an_empty_placement_with_its_inputs_known(self) -> None:
        placement = place_release([], _scope(_SEASON), _NEW)

        assert placement == UrlPlacement(
            files=(),
            assignment=EpisodeAssignment(()),
            records=(),
            claimed_ids=frozenset(),
            inputs_known=True,
            hold=None,
            hold_counts=NO_ELSEWHERE,
            stored=None,
            intended_sizes={},
        )


class TestPlaceIdentified:
    """A file whose size matches an episode goes on that episode."""

    _LOST = SeedFile("Show - Lost Tape.mkv", 70, ParsedFileInfo())

    def test_a_numberless_file_is_placed_by_its_listed_size(self) -> None:
        torrent = known_torrent(indexes=_INDEXES, evidence=torrent_evidence(identities={70: 103}))

        placement = place_release([self._LOST, *_pack(2)], _scope(_SEASON), torrent)

        assert placement.assignment.assigned["show - lost tape.mkv"] == (103,)
        assert PlacementVerdict.IDENTIFIED in verdicts_of(placement.assignment)
        assert placement.inputs_known

    def test_a_hash_less_url_is_placed_by_the_series_identities(self) -> None:
        # No listing matches a torrent without a hash, but the series' size identities still apply to it.
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": url_item(url="u1", infohash=None, download=True)})}
        known = TorrentReads({}, _INDEXES, ListingsRead({}, {70: 103})).known(seadex_dict)

        placement = place_release([self._LOST, *_pack(2)], _scope(_SEASON), known["u1"])

        assert placement.assignment.assigned["show - lost tape.mkv"] == (103,)
        assert placement.hold is None
        assert placement.inputs_known

    def test_unread_identities_hold_the_url(self) -> None:
        # SeaDex is down, so there are no size identities. The pack is still placed by its numbers for coverage,
        # but the url is held, since anything the grab seeds is never placed again.
        torrent = known_torrent(indexes=_INDEXES, evidence=TorrentEvidence(EMPTY_LISTING, None))

        placement = place_release([self._LOST, *_pack(2)], _scope(_SEASON), torrent)

        assert placement.assignment.assigned == {"show - s01e01.mkv": (101,), "show - s01e02.mkv": (102,)}
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_the_overrides_are_the_files_placed_over_their_names(self) -> None:
        # The numberless file isn't an override since it has no number. For the file Sonarr matched, the warning
        # shows Sonarr's reading.
        torrent = known_torrent(indexes=_INDEXES, evidence=torrent_evidence(identities={70: 103, 10: 102, 20: 101}))
        matched = SeedFile(
            "Show - Second Half.mkv",
            20,
            ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=2),)),
        )
        placement = place_release([self._LOST, _pack(1)[0], matched], _scope(_SEASON), torrent)

        assert identity_overrides(placement, episode_index(_SERIES)) == (
            IdentityOverride("show - s01e01.mkv", "S01E01", EpisodeKey(1, 2)),
            IdentityOverride("show - second half.mkv", "S01E02", EpisodeKey(1, 1)),
        )


class TestPlaceOntoResident:
    """A listed torrent with a stored record: only the map's leftover is placed, the whole map is recorded."""

    def test_the_leftover_is_placed_and_the_whole_map_recorded(self) -> None:
        record = _resident_record()

        placement = place_release(_pack(), _scope(_SEASON), _torrent(record))

        # The record's two mapped names are not re-placed, and the records and claim read the full map.
        assert placement.assignment.assigned == {"show - s01e03.mkv": (103,), "show - s01e04.mkv": (104,)}
        assert placement.records == tuple(EpisodeRecord(1, n, 10 * n) for n in range(1, 5))
        assert placement.claimed_ids == {101, 102, 103, 104}
        assert placement.inputs_known
        assert placement.stored is record

    def test_a_seeded_parse_miss_on_a_wholly_mapped_torrent_holds_nothing(self) -> None:
        record = pending_import(
            infohash="h1",
            file_episode_map={normalize_basename(name): [100 + n] for n, name in enumerate(_PACK, start=1)},
            seadex_files=_PACK,
            ordered_episode_ids=(101, 102, 103, 104),
        )
        files = [SeedFile(_PACK[0], 10, None), *_pack()[1:]]

        placement = place_release(files, _scope(_SEASON), _torrent(record))

        assert placement.hold is None
        assert placement.inputs_known
        assert placement.claimed_ids == {101, 102, 103, 104}

    def test_a_seeded_parse_miss_holds_the_url_while_a_file_is_left_to_place(self) -> None:
        files = [SeedFile(_PACK[0], 10, None), *_pack()[1:]]

        placement = place_release(files, _scope(_SEASON), _torrent(_resident_record()))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_a_seeded_id_outside_the_entry_records_nothing(self) -> None:
        # A sibling entry's episode (201) and an id no index holds (999): neither is this entry's.
        record = pending_import(
            infohash="h1",
            file_episode_map={"show - s02e01.mkv": [201], "other - 01.mkv": [999]},
            seadex_files=["Show - S02E01.mkv", "Other - 01.mkv"],
            ordered_episode_ids=(201, 202),
        )
        files = [SeedFile("Show - S02E01.mkv", 5, _matched(2, 1)), SeedFile("Other - 01.mkv", 6, _bare(1))]

        placement = place_release(files, _scope(_SEASON, al_id=2), _torrent(record))

        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.claimed_ids == frozenset()
        assert placement.inputs_known

    def test_a_re_flag_places_the_leftover_under_its_fresh_window(self) -> None:
        # The entry's stored claim covers two episodes. Re-listed over the whole season, its fresh window
        # replaces the stale one, so the third file lands where the stale window could not put it.
        record = pending_import(
            infohash="h1",
            file_episode_map={"show - s01e01.mkv": [101]},
            seadex_files=_PACK[:3],
            ordered_episode_ids=(101, 102),
        )
        files = _pack(3)

        stale = place_release(files, _scope(_SEASON[:2]), _torrent(record))
        fresh = place_release(files, _scope(_SEASON), _torrent(record))

        assert stale.assignment.assigned == {"show - s01e02.mkv": (102,)}
        assert fresh.assignment.assigned == {"show - s01e02.mkv": (102,), "show - s01e03.mkv": (103,)}
        assert fresh.claimed_ids == {101, 102, 103}

    def test_a_re_flag_runs_the_fresh_window_in_the_stale_ones_place(self) -> None:
        # The entry's stored claim is a stale two-episode slice, and the pair Sonarr read nothing of indexes
        # whichever two-episode window runs. Re-listed over the two episodes before it, the fresh window runs
        # in the stale one's place, never after it, so the pair lands on the fresh episodes.
        record = pending_import(
            infohash="h1", file_episode_map={}, seadex_files=_RUN[:2], ordered_episode_ids=(103, 104)
        )
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN[:2], start=1)]

        placement = place_release(files, _scope(_SEASON[:2]), _torrent(record))

        assert placement.assignment.assigned == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert placement.claimed_ids == {101, 102}

    def test_a_sibling_claim_on_an_unread_series_places_nothing_and_holds_the_url_as_unread(self) -> None:
        # Entry 2's claim maps two of the files onto series 8, which this run could not read. Its window can't be
        # ordered, so nothing is placed, and the url waits for the read rather than reading as another title's.
        record = pending_import(
            infohash="h1",
            series_id=8,
            al_id=2,
            file_episode_map={"show - s01e01.mkv": [901], "show - s01e02.mkv": [902]},
            seadex_files=_PACK,
            ordered_episode_ids=(901, 902, 903, 904),
        )
        torrent = _torrent(record)

        placement = place_release(_pack(), _scope(_SEASON), torrent)

        assert torrent.order(_scope(_SEASON).window) is None
        assert placement.assignment.placements == ()
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known
        assert placement.claimed_ids == frozenset()
        assert placement.stored is record

    def test_a_stored_record_under_an_unread_own_series_leaves_the_inputs_unknown(self) -> None:
        # The entry's own series list did not serve, so its window can't be ordered and the title is re-checked.
        # The stored map is still the record's: its in-entry ids record and count as claimed.
        unread = seed_scope(episode_index(_SEASON), episode_index([]))
        torrent = known_torrent(_resident_record(), evidence=torrent_evidence())

        placement = place_release(_pack(), unread, torrent)

        assert placement.assignment.placements == ()
        assert not placement.inputs_known
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 2, 20))
        assert placement.claimed_ids == {101, 102}


_ST1 = ("Alpha Start", "Bravo Turn", "Charlie Rise")
_ST2 = ("Echo Dawn", "Foxtrot Fall", "Golf Night")
# Two titled seasons and a special, season 2 numbered on in absolute order.
_TITLED = [
    *(sonarr_ep(1, n, ep_id=100 + n, absolute=n, title=t, episode_file_id=0) for n, t in enumerate(_ST1, 1)),
    *(sonarr_ep(2, n, ep_id=200 + n, absolute=3 + n, title=t, episode_file_id=0) for n, t in enumerate(_ST2, 1)),
    sonarr_ep(0, 1, ep_id=501, title="Beach Special", episode_file_id=0),
]
_TITLED_INDEX = episode_index(_TITLED)


class TestPlaceUnderAnotherEntrysRefutal:
    """A torrent entry 1's record holds, whose run its titles refute under entry 1's window, placed for another entry."""

    @staticmethod
    def _place(files: dict[str, ParsedFileInfo], mapped: FileEpisodeMap, entry: list[int]) -> UrlPlacement:
        """Entry 2 places `files` onto entry 1's record of `mapped` over season 1, its own window `entry`."""

        record = pending_import(
            infohash="h1",
            series_id=7,
            file_episode_map=mapped,
            seadex_files=tuple(files),
            ordered_episode_ids=(101, 102, 103),
        )
        scope = seed_scope(episode_index([ep for ep in _TITLED if ep.id in entry]), _TITLED_INDEX, al_id=2)
        seeds = [SeedFile(name, 1000 + i, info) for i, (name, info) in enumerate(files.items())]
        torrent = known_torrent(record, indexes={7: _TITLED_INDEX}, evidence=torrent_evidence())
        return place_release(seeds, scope, torrent)

    def test_the_sequel_entry_places_the_run_entry_1s_window_refutes(self) -> None:
        # Sonarr read the sequel's bare numbers as first-season absolutes, and its titled member names a sequel
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

        assert placement.hold is None
        assert placement.claimed_ids == {201, 202, 203}

    def test_an_entry_placing_nothing_is_held_as_entry_1s_not_as_refuted(self) -> None:
        # The titled file is the third episode's, under the second's key. None of the files is the special's, and
        # entry 1's record holds the titled one, so the url is entry 1's, not refuted under entry 2. The two plain
        # files are entry 1's too: its window refuted them by title, and entry 2's never read them as its own, so
        # they're no hand import for entry 2.
        files = {
            "show - 01.mkv": _matched(1, 1),
            "show - 02 - charlie rise.mkv": _matched(1, 2),
            "show - 03.mkv": _matched(1, 3),
        }

        placement = self._place(files, {"show - 02 - charlie rise.mkv": (103,)}, [501])

        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=3, unplaced=0)


class TestPlaceHeld:
    """A url the grab holds: a specials pack the listing contradicts, or a listing the run could not read."""

    @staticmethod
    def _specials(count: int = 3) -> list[SeedFile]:
        """A pack of the first `count` specials, each read by Sonarr as its own number's special."""

        return [SeedFile(f"Show.S00E{n:02d}.mkv", 10 * n, _matched(0, n)) for n in range(1, count + 1)]

    @staticmethod
    def _specials_scope(*numbers: int) -> SeedScope:
        """Entry 1 over the specials `numbers`, within the all-specials series."""

        return seed_scope(episode_index([_SPECIALS[n - 1] for n in numbers]), episode_index(_SPECIALS))

    @staticmethod
    def _listing(listed: tuple[int, ...] | None) -> KnownTorrent:
        """A new torrent listed over `listed` in the specials series. None means SeaDex couldn't be read."""

        evidence = UNREAD_EVIDENCE if listed is None else torrent_evidence(entry_window(1, listed))
        return known_torrent(indexes={7: episode_index(_SPECIALS)}, evidence=evidence)

    def test_a_misnumbered_pack_holds_the_url_and_records_nothing(self) -> None:
        # The entries list the pack over specials 2, 4 and 6, three as the pack is wide: its `1` is no listed
        # special, so the pack counts by another TVDB state. Nothing places, and the planner sees no coverage.
        placement = place_release(self._specials(), self._specials_scope(2, 4), self._listing((502, 504, 506)))

        assert verdicts_of(placement.assignment) == {PlacementVerdict.MISNUMBERED}
        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.records == ()
        assert placement.claimed_ids == frozenset()
        assert placement.inputs_known

    def test_a_pack_held_for_a_disputed_run_records_none_of_its_placed_files(self) -> None:
        # The season's absolutes count the special between E02 and E03, and the pack is keyed on past the end.
        season = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=0, absolute=1),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0, absolute=2),
            sonarr_ep(1, 3, ep_id=103, episode_file_id=0, absolute=4),
        ]
        series = episode_index([*season, sonarr_ep(0, 1, ep_id=501, episode_file_id=0, absolute=3)])
        files = [
            SeedFile(name, 10 * n, _matched(1, n) if n < 4 else _bare_key(1, n)) for n, name in enumerate(_PACK, 1)
        ]
        scope = seed_scope(episode_index(season), series)

        placement = place_release(files, scope, known_torrent(indexes={7: series}))

        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.assignment.assigned == {"show - s01e01.mkv": (101,), "show - s01e02.mkv": (102,)}
        assert placement.records == ()

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
        return season, special, episode_index([*season, special, opener], SeriesOrdering({"dvd": dvd}, None))

    def test_a_pack_numbered_by_the_dvd_order_is_grabbed_for_the_season_and_for_the_special(self) -> None:
        # The pack is keyed on past the season's end because it follows the DVD order. The aired reading refuses
        # its last two files. The order puts the third on the special and the fourth on the aired third episode.
        season, special, series = self._counted_special(self._DVD)
        files = [
            SeedFile(name, 10 * n, _matched(1, n) if n < 4 else _bare_key(1, n)) for n, name in enumerate(_PACK, 1)
        ]
        listed = torrent_evidence(entry_window(1, (101, 102, 103)), entry_window(2, (501,)))
        torrent = known_torrent(indexes={7: series}, evidence=listed)

        for_season = place_release(files, seed_scope(episode_index(season), series), torrent)
        for_special = place_release(files, seed_scope(episode_index([special]), series, al_id=2), torrent)

        assert (for_season.hold, for_special.hold) == (None, None)
        assert for_season.claimed_ids == {101, 102, 103}
        assert for_season.records[-1] == EpisodeRecord(season=1, episode=3, size=40)
        assert for_special.claimed_ids == {501}
        assert for_special.records == (EpisodeRecord(season=0, episode=1, size=30),)

    def test_a_pack_the_order_reads_onto_episodes_no_title_holds_is_held_for_a_hand_import(self) -> None:
        # The pack's two files are keyed from the special's position on. This DVD order reads the first as the
        # special and the second as the next season's opener, and nothing lists the torrent under either.
        season, _special, series = self._counted_special((*self._DVD[:3], OrderRow(1, 4, 201, None)))
        files = [SeedFile(_PACK[2], 30, _matched(1, 3)), SeedFile(_PACK[3], 40, _bare_key(1, 4))]
        scope = seed_scope(episode_index(season), series)

        placement = place_release(files, scope, known_torrent(indexes={7: series}))

        assert verdicts_of(placement.assignment) == {PlacementVerdict.OTHER_ORDER_ELSEWHERE}
        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.records == ()

    @staticmethod
    def _titled_scope(*numbers: int) -> SeedScope:
        """Entry 1 over season 1's episodes `numbers`, within a titled four-episode season."""

        titles = ("Pilot Flight", "The Return", "Festival Night", "Final Bell")
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, title=t) for n, t in enumerate(titles, 1)]
        return seed_scope(episode_index([season[n - 1] for n in numbers]), episode_index(season))

    def test_a_pack_its_titles_refute_that_places_nothing_holds_the_url_and_records_nothing(self) -> None:
        # The titled file is the third episode's, under the plain file's key.
        files = [
            SeedFile("show - 02 - Festival Night [grp].mkv", 10, _bare_key(1, 2)),
            SeedFile("show - 03 [grp].mkv", 20, _bare_key(1, 3)),
        ]

        scope = self._titled_scope(1, 2, 3)

        placement = place_release(files, scope, new_torrent(scope))

        assert placement.hold is GrabHold.TITLE_REFUTED
        assert placement.records == ()

    def test_a_refuted_pack_a_sibling_places_part_of_is_held_as_refuted(self) -> None:
        # The OVA lands on entry 2's special, but the refuted files are nobody's, so the url isn't another title's.
        titles = ("Pilot Flight", "The Return", "Festival Night", "Final Bell")
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, title=t) for n, t in enumerate(titles, 1)]
        series = episode_index([sonarr_ep(0, 1, ep_id=501, episode_file_id=0), *season])
        files = [
            SeedFile("show - 02 - Festival Night [grp].mkv", 10, _bare_key(1, 2)),
            SeedFile("show - 03 [grp].mkv", 20, _bare_key(1, 3)),
            SeedFile("show - OVA [grp].mkv", 30, _matched(0, 1)),
        ]
        torrent = known_torrent(indexes={7: series}, evidence=torrent_evidence(entry_window(2, (501,))))

        placement = place_release(files, seed_scope(episode_index(season[:3]), series), torrent)

        assert by_name(placement.assignment)["show - ova [grp].mkv"] == ((), PlacementVerdict.LISTED_ELSEWHERE)
        assert placement.hold is GrabHold.TITLE_REFUTED
        assert placement.claimed_ids == frozenset()

    _EARLY_KEYS: ClassVar[tuple[str, ...]] = (
        "show - 01 - The Return [grp].mkv",
        "show - 02 - Festival Night [grp].mkv",
        "show - 03 [grp].mkv",
    )
    """A pack whose keys are each one episode early, as the titles show. The untitled third can't be placed."""

    def _place_early_keys(self) -> tuple[UrlPlacement, SeedScope]:
        files = [SeedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(self._EARLY_KEYS, 1)]
        scope = self._titled_scope(1, 2, 3)
        return place_release(files, scope, new_torrent(scope)), scope

    def test_a_pack_its_titles_refute_that_places_some_files_is_not_held(self) -> None:
        placement, _ = self._place_early_keys()

        assert placement.hold is None
        assert placement.assignment.assigned == {
            normalize_basename(self._EARLY_KEYS[0]): (102,),
            normalize_basename(self._EARLY_KEYS[1]): (103,),
        }
        assert placement.assignment.hand_import_files == (normalize_basename(self._EARLY_KEYS[2]),)
        assert len(placement.records) == 2

    def test_the_seed_keeps_the_file_left_for_a_hand_import(self) -> None:
        placement, scope = self._place_early_keys()
        release = SeedRelease("grp", url_item(url="u", infohash="h1"), "h1", placement)

        seed = build_pending_seed(release, scope, entry_facts())

        assert seed.hand_import_files == (normalize_basename(self._EARLY_KEYS[2]),)

    def test_a_pack_the_listing_agrees_with_places_and_is_not_held(self) -> None:
        placement = place_release(self._specials(), self._specials_scope(1, 2), self._listing((501, 502, 503)))

        assert placement.hold is None
        assert placement.assignment.assigned == {"show.s00e01.mkv": (501,), "show.s00e02.mkv": (502,)}
        assert verdicts_of(placement.assignment) == {PlacementVerdict.EXACT, PlacementVerdict.FOREIGN}

    def test_a_pack_its_aliases_map_onto_the_listing_places_and_is_not_held(self) -> None:
        # The pack's `1` isn't a listed special, but the entries' aliases map 1, 2 and 3 onto 2, 4 and 6. Special 6
        # is entry 2's, which has no claim yet, so its file is set aside as entry 2's.
        torrent = known_torrent(
            indexes={7: episode_index(_SPECIALS)},
            evidence=torrent_evidence(
                entry_window(1, (502, 504)), entry_window(2, (506,)), aliases=ALIASED_PACK.special_aliases
            ),
        )

        placement = place_release(self._specials(), self._specials_scope(2, 4), torrent)

        assert placement.hold is None
        assert placement.assignment.assigned == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (504,)}
        assert by_name(placement.assignment)["show.s00e03.mkv"] == ((), PlacementVerdict.LISTED_ELSEWHERE)
        assert placement.claimed_ids == {502, 504}

    _OTHER = episode_index([sonarr_ep(0, n, ep_id=900 + n, episode_file_id=0) for n in (1, 2, 3)])
    """Another series' three specials, which the listing doesn't cover."""

    @classmethod
    def _aliased_against(cls, numbered: tuple[int, ...]) -> KnownTorrent:
        """The aliased pack, claimed for entry 1 over specials 2 and 4 and for entry 2 over `numbered` of `_OTHER`."""

        return known_torrent(
            two_claim_record(series_ids=(7, 8), windows=((502, 504), numbered)),
            indexes={7: episode_index(_SPECIALS), 8: cls._OTHER},
            evidence=torrent_evidence(*ALIASED_PACK.entries, aliases=ALIASED_PACK.special_aliases),
        )

    @pytest.mark.parametrize("al_id", [1, 2])
    def test_a_pack_another_series_places_by_its_numbers_against_the_aliases_holds_the_url(self, al_id: int) -> None:
        # The stored record also claims the pack for another series: there every member places by its own number,
        # against what the aliases say, so the pack waits for a hand import. Either entry holds it: the aliases
        # placed or claimed its files under the first, the second placed them.
        scope = self._specials_scope(2, 4) if al_id == 1 else seed_scope(self._OTHER, self._OTHER, al_id=2, series_id=8)

        placement = place_release(self._specials(), scope, self._aliased_against((901, 902, 903)))

        assert verdicts_of(placement.assignment) == {PlacementVerdict.MISNUMBERED}
        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.claimed_ids == frozenset()

    def test_a_file_the_aliases_claim_that_another_series_places_holds_the_url(self) -> None:
        # The aliases claim the third file for special 6, outside this window, and the other series places it by
        # its number. The first two still place.
        placement = place_release(self._specials(), self._specials_scope(2, 4), self._aliased_against((903,)))

        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.assignment.assigned == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (504,)}

    def test_a_file_misnumbered_under_another_entrys_window_holds_nothing(self) -> None:
        # The stored record claims season 1 for entry 1, where the first file's key and count disagree. Entry 2's
        # window reads that file as another season's, and places its own.
        first = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, absolute=n) for n in (1, 2, 3)]
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0, absolute=3 + n) for n in (1, 2)]
        series = episode_index([*first, *second])
        files = [
            SeedFile("Show - S01E02 (0003).mkv", 10, _matched(1, 2)),
            SeedFile("Show - S02E01 (0004).mkv", 20, _matched(2, 1)),
            SeedFile("Show - S02E02 (0005).mkv", 30, _matched(2, 2)),
        ]
        record = pending_import(
            infohash="h1", seadex_files=[f.basename for f in files], ordered_episode_ids=(101, 102, 103)
        )
        scope = seed_scope(episode_index(second), series, al_id=2)

        placement = place_release(files, scope, known_torrent(record, indexes={7: series}))

        assert PlacementVerdict.MISNUMBERED in verdicts_of(placement.assignment)
        assert placement.hold is None
        assert placement.records == (EpisodeRecord(2, 1, 20), EpisodeRecord(2, 2, 30))

    def test_an_unread_listing_holds_the_url_and_places_nothing(self) -> None:
        placement = place_release(self._specials(), self._specials_scope(1, 2), self._listing(None))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert not placement.inputs_known

    def test_a_parse_miss_under_a_specials_listing_holds_the_url(self) -> None:
        files = [SeedFile(f"Show.S00E{n:02d}.mkv", 10 * n, None if n == 2 else _matched(0, n)) for n in (1, 2, 3)]

        placement = place_release(files, self._specials_scope(1, 2), self._listing((501, 502, 503)))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_an_unread_listing_over_a_seasoned_window_places_nothing_and_holds(self) -> None:
        # Another title the listing names could own some of the pack, so nothing is placed until it's read. The
        # size identities weren't read either, so the url is held until they are.
        placement = place_release(_pack(), _scope(_SEASON), _torrent(listed=None))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known
        assert placement.assignment.placements == ()

    def test_a_file_less_url_under_an_unread_listing_waits_on_nothing(self) -> None:
        placement = place_release([], self._specials_scope(1, 2), self._listing(None))

        assert placement.inputs_known
        assert placement.hold is None

    def test_attach_writes_each_urls_hold(self) -> None:
        # The third url's listing is two wide against a three-file pack: the listing stands down, no hold.
        urls = ("u1", "u2", "u3")
        seadex_dict: SeadexDict = {
            "RG": rg_group({u: url_item(url=u, infohash=f"h{i}", download=True) for i, u in enumerate(urls, 1)})
        }
        placed = EntryPlacements.place(
            self._specials_scope(2, 4),
            dict.fromkeys(urls, self._specials()),
            {
                "u1": self._listing((502, 504, 506)),
                "u2": self._listing(None),
                "u3": self._listing((502, 504)),
            },
        )

        placed.attach_placements(seadex_dict)

        holds = [seadex_dict["RG"].urls[u].hold for u in urls]
        assert holds == [GrabHold.MISNUMBERED, GrabHold.INPUT_UNREAD, None]
        assert placed.input_missing_groups(seadex_dict) == ("RG",)


class TestPlaceBesideAnotherListedTitle:
    """A url another listed title shares: held as that title's only when none of it lands on the entry."""

    def test_an_entry_placing_some_beside_a_sibling_placing_the_rest_is_not_held(self) -> None:
        torrent = _torrent(listed=[entry_window(1, (101, 102)), entry_window(2, (103, 104))])

        placement = place_release(_pack(), _scope(_SEASON[:2]), torrent)

        assert by_name(placement.assignment) == {
            "show - s01e01.mkv": ((101,), PlacementVerdict.EXACT),
            "show - s01e02.mkv": ((102,), PlacementVerdict.EXACT),
            "show - s01e03.mkv": ((), PlacementVerdict.LISTED_ELSEWHERE),
            "show - s01e04.mkv": ((), PlacementVerdict.LISTED_ELSEWHERE),
        }
        assert placement.hold is None

    def test_a_file_only_a_sibling_refused_is_the_siblings(self) -> None:
        # Entry 1's window reads the first file's key and count as disagreeing, and places the second. Entry 2's
        # window read neither as its own, so the refused file is entry 1's to report, not this entry's hand import.
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, absolute=n) for n in (1, 2, 3)]
        series = episode_index([_SPECIAL, *season])
        files = [
            SeedFile("Show - S01E02 (0003).mkv", 10, _matched(1, 2)),
            SeedFile("Show - S01E03 (0003).mkv", 20, _matched(1, 3)),
        ]
        scope = seed_scope(episode_index([_SPECIAL]), series, al_id=2)
        torrent = known_torrent(indexes={7: series}, evidence=torrent_evidence(entry_window(1, (101, 102, 103))))

        placement = place_release(files, scope, torrent)

        assert by_name(placement.assignment)["show - s01e02 (0003).mkv"] == ((), PlacementVerdict.LISTED_ELSEWHERE)
        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=2, unplaced=0)

    def test_attach_writes_each_urls_hold_counts(self) -> None:
        # Entry 3's specials window places nothing of the season pack the two season entries take.
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": url_item(url="u1", infohash="h1", download=True)})}
        torrent = _torrent(listed=[entry_window(1, (101, 102)), entry_window(2, (103, 104))])
        placed = EntryPlacements.place(_scope([_SPECIAL], al_id=3), {"u1": _pack()}, {"u1": torrent})

        placed.attach_placements(seadex_dict)

        item = seadex_dict["RG"].urls["u1"]
        assert (item.hold, item.hold_counts) == (GrabHold.PLACED_ELSEWHERE, ElsewhereCounts(elsewhere=4, unplaced=0))

    def test_a_stored_file_on_an_episode_no_other_window_holds_is_nobodys(self) -> None:
        # The record maps the pack's last two files past this entry's window, and no other listed title holds
        # those episodes, so nothing is another title's and the url isn't held.
        record = self._record_mapping_3_and_4()

        placement = place_release(
            _pack()[2:], _scope(_SEASON[:2]), _torrent(record, listed=[entry_window(1, (101, 102))])
        )

        assert placement.hold is None
        assert placement.hold_counts == NO_ELSEWHERE
        assert placement.claimed_ids == frozenset()

    def test_a_stored_file_on_an_episode_a_sibling_window_holds_is_the_siblings(self) -> None:
        record = self._record_mapping_3_and_4()
        listed = [entry_window(1, (101, 102)), entry_window(2, (103, 104))]

        placement = place_release(_pack()[2:], _scope(_SEASON[:2]), _torrent(record, listed=listed))

        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=2, unplaced=0)

    def test_a_kept_title_runs_before_an_ignored_one_of_its_kind(self) -> None:
        # Both windows are seasons of four, and the bare run fits either. The ignored title has the lower id, but
        # the kept title's window places first, so the run is the kept title's and nothing is held.
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0) for n in range(1, 5)]
        series = episode_index([*_SEASON, *second])
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, 1)]
        scope = seed_scope(episode_index(_SEASON), series, al_id=2)
        listed = torrent_evidence(entry_window(1, (ep.id for ep in second), ignored=True))

        placement = place_release(files, scope, known_torrent(indexes={7: series}, evidence=listed))

        assert placement.claimed_ids == {101, 102, 103, 104}
        assert placement.hold is None

    def test_a_kept_title_runs_before_a_stale_claim_of_an_ignored_one(self) -> None:
        # Entry 1 claimed the run before it was ignored, and its stored claim can't know that. The read flags its
        # listed window, so its claim's window runs after the kept title's: the run is entry 2's and nothing is held.
        second = [sonarr_ep(2, n, ep_id=200 + n, episode_file_id=0) for n in range(1, 5)]
        series = episode_index([*_SEASON, *second])
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, 1)]
        record = pending_import(
            infohash="h1", al_id=1, ordered_episode_ids=(101, 102, 103, 104), file_episode_map={}, seadex_files=_RUN
        )
        listed = torrent_evidence(
            entry_window(1, (101, 102, 103, 104), ignored=True), entry_window(2, (ep.id for ep in second))
        )
        scope = seed_scope(episode_index(second), series, al_id=2)

        placement = place_release(files, scope, known_torrent(record, indexes={7: series}, evidence=listed))

        assert placement.claimed_ids == {201, 202, 203, 204}
        assert placement.hold is None

    @staticmethod
    def _record_mapping_3_and_4() -> PendingImport:
        """A stored record of the pack's last two files, mapped onto the season's third and fourth episodes."""

        return pending_import(
            infohash="h1",
            series_id=7,
            file_episode_map={"show - s01e03.mkv": [103], "show - s01e04.mkv": [104]},
            seadex_files=_PACK[2:],
            ordered_episode_ids=(101, 102, 103, 104),
        )

    def test_a_run_a_sibling_held_holds_the_url_as_unread(self) -> None:
        # The run fits entry 1's window; the NCOP's parse is unread.
        files = [SeedFile(name, 10 * n, ParsedFileInfo()) for n, name in enumerate(_RUN, 1)]
        files.append(SeedFile("Show - NCOP.mkv", 50, None))
        torrent = _torrent(listed=[entry_window(1, (101, 102, 103, 104)), entry_window(2, (201, 202))])

        placement = place_release(files, _scope(_SERIES[-2:], al_id=2), torrent)

        assert by_name(placement.assignment) == {
            **{normalize_basename(name): ((), PlacementVerdict.HELD) for name in _RUN},
            "show - ncop.mkv": ((), PlacementVerdict.EXTRA),
        }
        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known


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
_RG_MAIN = EntryNames(series="The Rolling Girls", anilist=("The Rolling Girls", "Rolling☆Girls"))
_RG_CHIBI = EntryNames(series="The Rolling Girls", anilist=("Rolling☆Girls: Chibi☆Rolling☆Girls Korokoro Gekijou",))
_RG_SEASON_WINDOW = entry_window(20815, (ep.id for ep in _RG_SEASON), series_id=919, names=_RG_MAIN)
_RG_CHIBI_WINDOW = entry_window(102077, (ep.id for ep in _RG_SPECIALS), series_id=919, names=_RG_CHIBI)
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
# Sonarr reads each episode's bare number as an absolute and matches it to the season, and matches nothing for
# the shorts. It reads nothing at all from the OP and ED.
_RG_FILES = (
    *(
        SeedFile(
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
        SeedFile(
            _rg_name(f" Chibi Special - {n:02d}"), size, ParsedFileInfo(season_number=0, absolute_episode_numbers=(n,))
        )
        for n, size in enumerate(_RG_SHORT_SIZES, 1)
    ),
    SeedFile(_rg_name(" - OP"), 305872174, ParsedFileInfo()),
    SeedFile(_rg_name(" - ED"), 541996641, ParsedFileInfo()),
)
_RG_EPISODE_NAMES = [normalize_basename(f.basename) for f in _RG_FILES[:12]]
# Under either window the shorts place nowhere, and the OP and ED are extras.
_RG_REST = {
    **{normalize_basename(f.basename): ((), PlacementVerdict.SKIPPED) for f in _RG_FILES[12:18]},
    **{normalize_basename(f.basename): ((), PlacementVerdict.EXTRA) for f in _RG_FILES[18:]},
}


class TestRollingGirlsBatch:
    """The real batch: its 12 episodes are the season's under either entry, never the Chibi entry's specials."""

    @staticmethod
    def _place(
        al_id: int, record: PendingImport | None = None, evidence: TorrentEvidence = _RG_EVIDENCE
    ) -> UrlPlacement:
        main = al_id == 20815
        scope = seed_scope(
            episode_index(_RG_SEASON if main else _RG_SPECIALS),
            _RG_SERIES,
            al_id=al_id,
            series_id=919,
            names=_RG_MAIN if main else _RG_CHIBI,
        )
        return place_release(_RG_FILES, scope, known_torrent(record, indexes={919: _RG_SERIES}, evidence=evidence))

    def test_under_the_season_entry_the_episodes_place_exact_on_the_season(self) -> None:
        placement = self._place(20815)

        assert by_name(placement.assignment) == {
            **{
                name: ((ep.id,), PlacementVerdict.EXACT) for name, ep in zip(_RG_EPISODE_NAMES, _RG_SEASON, strict=True)
            },
            **_RG_REST,
        }
        assert placement.hold is None
        assert placement.claimed_ids == {ep.id for ep in _RG_SEASON}

    def test_under_the_chibi_entry_alone_the_episodes_are_the_seasons_and_the_url_is_held(self) -> None:
        # The season's window decides the episodes first, and the season's entry has no claim on a record yet, so
        # they're set aside as its files. Nothing lands on the Chibi entry, so the url is held as another title's.
        placement = self._place(102077)

        assert by_name(placement.assignment) == {
            **dict.fromkeys(_RG_EPISODE_NAMES, ((), PlacementVerdict.LISTED_ELSEWHERE)),
            **_RG_REST,
        }
        assert placement.records == ()
        assert placement.claimed_ids == frozenset()
        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=12, unplaced=6)

    def test_under_the_chibi_entry_with_the_season_entry_ignored_the_url_is_still_held(self) -> None:
        # The season's title is on the ignore list: nobody grabs for it, but its window still sets the episodes
        # aside as its own, so they never land on the Chibi entry's specials. A season window runs before a specials
        # window whether or not its title is ignored, so the placement is the kept title's.
        evidence = torrent_evidence(replace(_RG_SEASON_WINDOW, ignored=True), _RG_CHIBI_WINDOW)

        placement = self._place(102077, evidence=evidence)
        kept = self._place(102077)

        assert placement.assignment == kept.assignment
        assert (placement.records, placement.hold, placement.hold_counts) == (kept.records, kept.hold, kept.hold_counts)

    def test_under_the_chibi_entry_beside_the_seasons_record_the_url_is_held(self) -> None:
        record = pending_import(
            infohash="h1",
            al_id=20815,
            series_id=919,
            file_episode_map={name: (ep.id,) for name, ep in zip(_RG_EPISODE_NAMES, _RG_SEASON, strict=True)},
            seadex_files=[f.basename for f in _RG_FILES],
            ordered_episode_ids=tuple(ep.id for ep in _RG_SEASON),
        )

        placement = self._place(102077, record)

        # The record already maps the episodes onto the season, so only the rest is placed, and the Chibi entry
        # gets none of the episodes.
        assert by_name(placement.assignment) == _RG_REST
        assert placement.records == ()
        assert placement.claimed_ids == frozenset()
        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=12, unplaced=6)


def _lone(number: int, title: str) -> ParsedFileInfo:
    """Sonarr's parse of a name carrying `number` alone: season 0, the absolute, and the series title it lifted."""

    return parsed_info(season=0, absolutes=(number,), series_title=title)


# Tsundere's Stray Cats Overrun! BD batch (nyaa 1050453): twelve episodes named by number alone, six SP files, listed
# under the season's entry (AniList 7590) and the specials' (AniList 8599). Sonarr matched the numbers to nothing.
_SC_SPECIALS = [sonarr_ep(0, n, ep_id=32761 + n, episode_file_id=0) for n in range(1, 8)]
_SC_SEASON = [sonarr_ep(1, n, ep_id=32768 + n, episode_file_id=0, absolute=n) for n in range(1, 14)]
_SC_SERIES = episode_index([*_SC_SPECIALS, *_SC_SEASON])
_SC_MAIN = EntryNames("Stray Cats Overrun!", ("Mayoi Neko Overrun!",))
_SC_EXTRAS = EntryNames("Stray Cats Overrun!", ("Mayoi Neko Overrun! Specials",))
_SC_SEASON_WINDOW = entry_window(7590, (ep.id for ep in _SC_SEASON), names=_SC_MAIN, episode_count=13)
_SC_SPECIALS_WINDOW = entry_window(8599, (ep.id for ep in _SC_SPECIALS[:6]), names=_SC_EXTRAS, episode_count=6)
_SC_EPISODES = [
    f"[Tsundere] Mayoi Neko Overrun! - {n:02d} [BDRip h264 1920x1080 FLAC][746CE979].mkv" for n in range(1, 13)
]
_SC_SPS = [f"[Tsundere] Mayoi Neko Overrun! - SP{n} [BDRip h264 1920x1080 FLAC][B66672E2].mkv" for n in range(1, 7)]
_SC_FILES = [
    *(SeedFile(name, 1_000_000_000 + n, _lone(n, "Mayoi Neko Overrun!")) for n, name in enumerate(_SC_EPISODES, 1)),
    *(SeedFile(name, 100_000_000 + n, ParsedFileInfo()) for n, name in enumerate(_SC_SPS, 1)),
]


class TestPlaceWhereEveryNumberingAgrees:
    """A file named by number alone that the windows left is placed where every numbering agrees, after the merge."""

    def test_a_single_a_specials_only_entry_lists_places_on_its_first_special(self) -> None:
        # My Hero Academia (Sonarr 254): the Ikinokore OVA entry (AniList 122349) is specials 8 and 9. HorribleSubs'
        # 01 reads S01E01 by the series' numbers, outside the window, so the passes skip it; the window counts its
        # specials from one.
        series = episode_index(
            [
                *(sonarr_ep(0, n, ep_id=9246 + n, episode_file_id=0) for n in range(7, 11)),
                *(sonarr_ep(1, n, ep_id=9270 + n, episode_file_id=0, absolute=n) for n in range(1, 4)),
            ]
        )
        names = EntryNames(
            "My Hero Academia",
            (
                "My Hero Academia: Make It! Do-or-Die Survival Training",
                "Boku no Hero Academia: Ikinokore! Kesshi no Survival Kunren",
            ),
        )
        window = entry_window(122349, (9254, 9255), names=names, episode_count=2)
        scope = seed_scope(
            episode_index([series.by_id[9254], series.by_id[9255]]),
            series,
            al_id=122349,
            names=names,
            episode_count=2,
            alternate_titles=("Boku no Hero Academia",),
        )
        name = "[HorribleSubs] Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren - 01 [1080p].mkv"
        files = [
            SeedFile(name, 1_400_000_000, _lone(1, "Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren"))
        ]

        placement = place_release(files, scope, known_torrent(indexes={7: series}, evidence=torrent_evidence(window)))

        assert by_name(placement.assignment) == {normalize_basename(name): ((9254,), PlacementVerdict.NUMBERINGS_AGREE)}
        assert placement.claimed_ids == {9254}
        assert placement.records == (EpisodeRecord(0, 8, 1_400_000_000),)
        assert placement.hold is None

    def test_a_batch_counted_from_one_places_on_the_seasons_entry(self) -> None:
        scope = seed_scope(episode_index(_SC_SEASON), _SC_SERIES, al_id=7590, names=_SC_MAIN, episode_count=13)
        torrent = known_torrent(
            indexes={7: _SC_SERIES}, evidence=torrent_evidence(_SC_SEASON_WINDOW, _SC_SPECIALS_WINDOW)
        )

        placement = place_release(_SC_FILES, scope, torrent)

        assert by_name(placement.assignment) == {
            **{
                normalize_basename(name): ((ep.id,), PlacementVerdict.NUMBERINGS_AGREE)
                for name, ep in zip(_SC_EPISODES, _SC_SEASON, strict=False)
            },
            **dict.fromkeys((normalize_basename(name) for name in _SC_SPS), ((), PlacementVerdict.SKIPPED)),
        }
        assert placement.claimed_ids == {ep.id for ep in _SC_SEASON[:12]}
        assert placement.hold is None

    def test_under_the_specials_entry_the_batch_is_the_seasons_and_the_url_is_held(self) -> None:
        # The fill places the twelve under the season's window, so for the specials' entry they're another title's,
        # as a file the season's window had placed would be: no hand import, and nothing of the url lands here.
        scope = seed_scope(episode_index(_SC_SPECIALS[:6]), _SC_SERIES, al_id=8599, names=_SC_EXTRAS, episode_count=6)
        torrent = known_torrent(
            indexes={7: _SC_SERIES}, evidence=torrent_evidence(_SC_SEASON_WINDOW, _SC_SPECIALS_WINDOW)
        )

        placement = place_release(_SC_FILES, scope, torrent)

        assert {by_name(placement.assignment)[normalize_basename(name)] for name in _SC_EPISODES} == {
            ((), PlacementVerdict.LISTED_ELSEWHERE)
        }
        assert placement.records == ()
        assert placement.hold is GrabHold.PLACED_ELSEWHERE
        assert placement.hold_counts == ElsewhereCounts(elsewhere=12, unplaced=6)


class TestKnownTorrent:
    """The windows a torrent is placed under: every listed entry's, every stored claim's, and the entry's own."""

    def test_an_unread_listing_orders_nothing(self) -> None:
        assert _torrent(listed=None).order(_scope(_SEASON).window) is None

    def test_an_unread_listing_gives_no_evidence(self) -> None:
        # Whether the unread listing holds the url is `place_release`'s call: the windows build as if listed nowhere.
        torrent = known_torrent(evidence=UNREAD_EVIDENCE)

        assert torrent.listing_evidence({"show - 01.mkv": 10}) == NO_EVIDENCE

    def test_the_evidence_maps_matched_sizes_to_episodes(self) -> None:
        torrent = known_torrent(evidence=torrent_evidence(entry_window(1, (101, 102)), identities={10: 101, 30: 103}))

        evidence = torrent.listing_evidence({"show - 01.mkv": 10, "show - 02.mkv": 20})

        assert evidence == ListingEvidence(torrent_listing(entry_window(1, (101, 102))), {"show - 01.mkv": 101})

    def test_the_evidence_carries_the_listings_aliases(self) -> None:
        torrent = known_torrent(evidence=torrent_evidence(entry_window(1, (501, 502)), aliases={1: 2}))

        assert torrent.listing_evidence({}) == ListingEvidence(
            torrent_listing(entry_window(1, (501, 502)), aliases={1: 2})
        )

    def test_the_entrys_own_window_replaces_its_claims(self) -> None:
        order = _torrent(two_claim_record()).order(_scope(_SEASON).window)

        assert order is not None
        assert [w.ordered_episode_ids for w in order.windows] == [(101, 102, 103, 104), (103, 104)]
        assert (order.at(1), order.imports) == (0, {0, 1})

    def test_the_entrys_own_window_carries_its_anilist_episode_count(self) -> None:
        scope = seed_scope(episode_index(_SEASON), episode_index(_SERIES), episode_count=4)

        assert scope.window == entry_window(1, (101, 102, 103, 104), episode_count=4)

    def test_a_new_entrys_own_window_takes_its_anilist_ids_place(self) -> None:
        order = _torrent(two_claim_record()).order(_scope(_SEASON, al_id=3).window)

        assert order is not None
        assert [w.al_id for w in order.windows] == [1, 2, 3]
        assert (order.at(3), order.imports) == (2, {0, 1, 2})

    def test_a_listed_entry_that_never_claimed_runs_but_is_not_imported(self) -> None:
        order = _torrent(listed=[entry_window(2, (103, 104))]).order(_scope(_SEASON).window)

        assert order is not None
        assert [w.al_id for w in order.windows] == [1, 2]
        assert (order.at(1), order.imports) == (0, {0})

    def test_a_claim_moving_series_leaves_its_old_series_unread(self) -> None:
        # Entry 1's stored claim is on series 7; this grab claims series 8.
        record = pending_import(
            infohash="h1",
            al_id=1,
            ordered_episode_ids=(101, 102, 103, 104),
            file_episode_map={},
            seadex_files=_PACK,
        )
        other = episode_index([sonarr_ep(1, n, ep_id=800 + n, episode_file_id=0) for n in range(1, 5)])
        fresh = torrent_evidence(entry_window(1, (801, 802, 803, 804), series_id=8))
        torrent = known_torrent(record, indexes={8: other}, evidence=fresh)

        order = torrent.order(seed_scope(other, other, al_id=1, series_id=8).window)

        assert order is not None
        assert [(w.al_id, w.series_id) for w in order.windows] == [(1, 8)]
        assert order.imports == {0}


class TestTorrentReads:
    """What the run read of an entry's torrents, keyed by url."""

    @staticmethod
    def _dict() -> SeadexDict:
        return {
            "RG": rg_group(
                {
                    "u1": url_item(url="u1", infohash="h1", download=True),
                    "u2": url_item(url="u2", infohash=None, download=True),
                    "u3": url_item(url="u3", infohash="h3", download=True),
                }
            ),
        }

    def test_entry_hashes_skip_a_hash_less_url(self) -> None:
        assert entry_hashes(self._dict()) == {"h1", "h3"}

    def test_known_by_url_over_the_stored_records_and_listings(self) -> None:
        record = _resident_record()
        listing = torrent_listing(entry_window(1, (101,)))
        reads = TorrentReads({"h1": record}, _INDEXES, ListingsRead({"h1": listing, "h3": None}, None))

        known = reads.known(self._dict())

        # u2 has no hash, so it's new and unlisted. u3 has no record and is listed under an unread window. That
        # unread window means no size identities for any url, u1 included.
        assert known == {
            "u1": KnownTorrent(record, _INDEXES, TorrentEvidence(listing, None)),
            "u2": KnownTorrent(None, _INDEXES, TorrentEvidence(EMPTY_LISTING, None)),
            "u3": KnownTorrent(None, _INDEXES, TorrentEvidence(None, None)),
        }

    def test_a_marked_url_reads_its_record_by_its_cross_seed_and_its_listing_by_seadexs_hash(self) -> None:
        seadex_dict = self._dict()
        seadex_dict["RG"].urls["u1"].cross_seed_hash = "c1"
        record = _resident_record()
        listing = torrent_listing(entry_window(1, (101,)))
        reads = TorrentReads({"c1": record}, _INDEXES, ListingsRead({"h1": listing, "h3": None}, None))

        assert entry_hashes(seadex_dict) == {"c1", "h3"}
        assert reads.known(seadex_dict)["u1"] == KnownTorrent(record, _INDEXES, TorrentEvidence(listing, None))

    def test_an_outage_leaves_every_hash_unread_and_a_hash_less_url_listed_nowhere(self) -> None:
        known = TorrentReads({}, _INDEXES, None).known(self._dict())

        assert {url: torrent.evidence for url, torrent in known.items()} == {
            "u1": TorrentEvidence(None, None),
            "u2": TorrentEvidence(EMPTY_LISTING, None),
            "u3": TorrentEvidence(None, None),
        }


class TestEntryPlacements:
    """The per-entry fold: records onto the release dict, the groups a failed read held, and the seeds."""

    @staticmethod
    def _entry(names: EntryNames | None = None) -> tuple[EntryPlacements, SeadexDict]:
        scope = _scope(_SEASON, names)
        run = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        held = [SeedFile(name, 1, None if n == 1 else _bare(n)) for n, name in enumerate(_RUN, start=1)]
        placed = EntryPlacements.place(
            scope, {"u1": run, "u2": held, "u3": []}, dict.fromkeys(("u1", "u2", "u3"), _NEW)
        )
        seadex_dict: SeadexDict = {
            "RG": rg_group({"u1": url_item(url="u1", files=_RUN, size=[10, 20, 30, 40], infohash="h1", download=True)}),
            "Held": rg_group({"u2": url_item(url="u2", files=_RUN, size=[1] * 4, infohash="h2", download=True)}),
            "Subs": rg_group({"u3": url_item(url="u3", files=["Show.ass"], size=[1], infohash="h3", download=True)}),
        }
        return placed, seadex_dict

    def test_attach_writes_each_urls_records_and_the_group_union(self) -> None:
        placed, seadex_dict = self._entry()

        placed.attach_placements(seadex_dict)

        run_records = [EpisodeRecord(1, n, 10 * n) for n in range(1, 5)]
        assert seadex_dict["RG"].urls["u1"].episodes == run_records
        assert seadex_dict["RG"].all_episodes == run_records
        assert seadex_dict["Held"].urls["u2"].episodes == []
        assert seadex_dict["Held"].all_episodes == []
        assert seadex_dict["Subs"].all_episodes == []

    def test_input_missing_groups_name_the_held_release(self) -> None:
        placed, seadex_dict = self._entry()

        assert placed.input_missing_groups(seadex_dict) == ("Held",)

    def test_seeds_fold_the_placement_and_carry_the_scopes_names(self) -> None:
        names = EntryNames("Show", ("Show", "Shou"))
        placed, seadex_dict = self._entry(names)
        entry = entry_facts()

        seeds = build_pending_seeds(seadex_dict, placed, entry)

        assert set(seeds) == {"h1", "h2"}
        assert dict(seeds["h1"].placements) == {normalize_basename(name): (100 + n,) for n, name in enumerate(_RUN, 1)}
        listed = {normalize_basename(name): 10 * n for n, name in enumerate(_RUN, 1)}
        assert seeds["h1"].facts == torrent_facts(
            seadex_files=tuple(_RUN), release_sizes=(10, 20, 30, 40), sizes_by_name=listed
        )
        assert seeds["h1"].claim.names == names
        assert seeds["h1"].stored is None
        assert dict(seeds["h2"].placements) == {}

    def test_a_marked_url_seeds_under_its_cross_seed(self) -> None:
        placed, seadex_dict = self._entry()
        seadex_dict["RG"].urls["u1"].cross_seed_hash = "c1"

        seeds = build_pending_seeds(seadex_dict, placed, entry_facts())

        assert set(seeds) == {"c1", "h2"}
        records = {key: seed.record_at(_STAMP, fresh=True) for key, seed in seeds.items()}
        assert (records["c1"].infohash, records["c1"].cross_seed) == ("c1", True)
        assert records["h2"].cross_seed is False

    def test_a_url_with_a_stored_record_seeds_onto_it(self) -> None:
        record = _resident_record()
        placed = EntryPlacements.place(_scope(_SEASON), {"u1": _pack()}, {"u1": _torrent(record)})
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {"u1": url_item(url="u1", files=_PACK, size=[10, 20, 30, 40], infohash="h1", download=True)}
            ),
        }

        seeds = build_pending_seeds(seadex_dict, placed, entry_facts())

        assert seeds["h1"].stored is record
        assert dict(seeds["h1"].placements) == {"show - s01e03.mkv": (103,), "show - s01e04.mkv": (104,)}


class TestOrderOnlySeed:
    """A bare-numbered pack one file longer than its season: the DVD order splits the opener in two halves."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
    _DVD: ClassVar[tuple[OrderRow, ...]] = (
        OrderRow(7, 1, 9001, None),
        OrderRow(7, 2, 9002, None),
        *(OrderRow(7, n + 1, 400 + n, None) for n in range(2, 13)),
        OrderRow(7, 14, 501, None),
        OrderRow(7, 21, 401, None),
    )
    """The DVD order: two halves Sonarr doesn't list, the aired second episode on, then the whole opener."""

    def test_the_seed_carries_the_halves_dvd_keys_onto_the_record(self) -> None:
        season = [sonarr_ep(4, n, ep_id=400 + n, episode_file_id=0, tvdb_id=400 + n) for n in range(1, 13)]
        opener = sonarr_ep(5, 1, ep_id=501, episode_file_id=0, tvdb_id=501)
        series = episode_index([*season, opener], SeriesOrdering({"dvd": self._DVD}, None))
        scope = seed_scope(episode_index(season), series, names=EntryNames("Show", ("Show Ending",)))
        files = [SeedFile(name, 10 * n, parsed_info()) for n, name in enumerate(self._RUN, 1)]
        placement = place_release(files, scope, new_torrent(scope))
        release = SeedRelease("grp", url_item(url="u", infohash="h1"), "h1", placement)

        seed = build_pending_seed(release, scope, entry_facts())

        assert seed.order_only == {
            normalize_basename(self._RUN[0]): OrderOnlyKey("dvd", EpisodeKey(7, 1)),
            normalize_basename(self._RUN[1]): OrderOnlyKey("dvd", EpisodeKey(7, 2)),
        }
        assert seed.record_at(_STAMP, fresh=True).order_only == seed.order_only


class TestPendingSeedRecordAt:
    """The record a seed persists: a new one, or an accretion onto the stored one."""

    def test_a_new_record_starts_at_the_stamp(self) -> None:
        claim = entry_claim()
        seed = PendingSeed(
            facts=_FACTS,
            placements={"show - 01.mkv": (101,)},
            excluded=("show - 02.mkv",),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=claim,
            stored=None,
        )

        record = seed.record_at(_STAMP, fresh=True)

        assert seed.joined() is None
        assert record == PendingImport(
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
        # A new record starts at the stamp whether or not the add was fresh.
        assert seed.record_at(_STAMP, fresh=False) == record

    def test_accreted_folds_the_placements_exclusions_and_claim(self) -> None:
        stored = pending_import(
            infohash="h1",
            added_at="2026-01-01 00:00:00",
            file_episode_map={"show - 01.mkv": [101]},
            excluded_files=("show - 09.mkv",),
            preowned_episode_ids=(101,),
            awaiting_cleanup=True,
        )
        claim = entry_claim(ordered_episode_ids=(101, 102, 103), preowned_episode_ids=(102,))
        seed = PendingSeed(
            facts=_FACTS,
            placements={"show - 02.mkv": (102,)},
            excluded=("show - 03.mkv",),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=claim,
            stored=stored,
        )

        record = seed.record_at(_STAMP, fresh=False)

        assert seed.joined() == record
        assert dict(record.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert record.excluded_files == ("show - 09.mkv", "show - 03.mkv")
        # The entry's own claim is replaced, its first preowned ids kept.
        assert record.claims == (replace(claim, preowned_episode_ids=(101,)),)
        assert record.awaiting_cleanup is False
        # Not a fresh add: the record keeps its `added_at`, and the identity stays the stored record's.
        assert record.added_at == "2026-01-01 00:00:00"
        assert record.release_group == stored.release_group

    def test_accreted_appends_a_new_entrys_claim(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        seed = pending_seed("h1", stored=stored, al_id=2)

        record = seed.record_at(_STAMP, fresh=False)

        assert record.claims == (*stored.claims, seed.claim)

    def test_accreted_onto_a_record_without_sizes_by_name_learns_them(self) -> None:
        stored = pending_import(infohash="h1")
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=entry_claim(),
            stored=stored,
        )

        assert seed.record_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 10}

    def test_accreted_keeps_the_stored_sizes_by_name(self) -> None:
        stored = pending_import(infohash="h1", sizes_by_name={"show - 01.mkv": 99})
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            hand_import_files=(),
            order_only=NO_ORDER_ONLY,
            claim=entry_claim(),
            stored=stored,
        )

        assert seed.record_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 99}

    def test_a_new_record_keeps_the_hand_import_files(self) -> None:
        seed = pending_seed("h1", hand_import_files=("show - 02.mkv",))

        assert seed.record_at(_STAMP, fresh=True).hand_import_files == ("show - 02.mkv",)

    def test_accreted_joins_the_hand_import_files(self) -> None:
        stored = pending_import(infohash="h1", hand_import_files=("show - 02.mkv",))

        kept = pending_seed("h1", stored=stored).record_at(_STAMP, fresh=False)
        joined = pending_seed("h1", stored=stored, hand_import_files=("show - 03.mkv",)).record_at(_STAMP, fresh=False)

        assert kept.hand_import_files == ("show - 02.mkv",)
        assert joined.hand_import_files == ("show - 02.mkv", "show - 03.mkv")

    def test_accreted_joins_the_order_only_keys(self) -> None:
        halves = [normalize_basename(f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv") for n in (1, 2)]
        keys = {halves[0]: OrderOnlyKey("dvd", EpisodeKey(7, 1)), halves[1]: OrderOnlyKey("dvd", EpisodeKey(7, 2))}
        stored = pending_import(infohash="h1", hand_import_files=halves[:1], order_only={halves[0]: keys[halves[0]]})

        joined = pending_seed(
            "h1", stored=stored, hand_import_files=(halves[1],), order_only={halves[1]: keys[halves[1]]}
        ).record_at(_STAMP, fresh=False)

        assert joined.order_only == keys

    def test_fresh_restarts_the_stored_record(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        seed = pending_seed("h1", stored=stored, al_id=2)

        record = seed.record_at(_STAMP, fresh=True)

        assert record.added_at == _STAMP


class TestBuildUnscopedSeed:
    """A Radarr grab's seed: the listing's facts, no files or window, an id-less claim on the entry."""

    @staticmethod
    def _flagged() -> FlaggedUrl:
        return FlaggedUrl("RG", "https://nyaa.si/view/1", url_item(infohash="h1", is_dual_audio=True), "h1")

    def test_a_new_torrent_seeds_an_id_less_claim(self) -> None:
        facts = entry_facts(al_id=9, series_id=0, title="Movie", url="https://releases.moe/9")

        seed = build_unscoped_seed(self._flagged(), facts, None)

        assert seed.stored is None
        assert seed.facts == torrent_facts(is_dual_audio=True)
        assert seed.placements == {} and seed.excluded == () and seed.order_only == {}
        assert seed.claim == entry_claim(
            al_id=9, series_id=0, title="Movie", url="https://releases.moe/9", guards=facts.guards
        )

    def test_a_listed_torrent_accretes_its_stored_record(self) -> None:
        stored = pending_import(infohash="h1", al_id=1, series_id=0)

        record = build_unscoped_seed(self._flagged(), entry_facts(al_id=2, series_id=0), stored).record_at(
            _STAMP, fresh=False
        )

        assert record.release_group == stored.release_group
        assert [claim.al_id for claim in record.claims] == [1, 2]


class TestEntryClaim:
    """The entry's claim reads the whole map inside the entry."""

    @staticmethod
    def _release(files: list[SeedFile], placed: UrlPlacement) -> SeedRelease:
        names = [f.basename for f in files]
        sizes = [f.size for f in files]
        return SeedRelease(
            "RG", url_item(url="u1", files=names, size=sizes, infohash="h1", download=True), "h1", placed
        )

    def test_ids_come_from_the_whole_map_inside_the_entry(self) -> None:
        files = _pack(3)
        placed = place_release(files, _scope(_SEASON), _torrent(_resident_record()))

        claim = build_entry_claim(self._release(files, placed), _scope(_SEASON), entry_facts())

        # This call placed E03 alone, yet the slice also carries the record's E01 and E02.
        assert placed.assignment.assigned == {"show - s01e03.mkv": (103,)}
        assert claim.slice_coverage == "S01 E01-E03"
        assert claim.ordered_episode_ids == (101, 102, 103, 104)

    def test_preowned_reads_every_claimed_id(self) -> None:
        # E01 (the record's, not this call's) already holds the own group's file at a listed size.
        entry = [sonarr_ep(1, 1, ep_id=101, size=10, release_group="RG"), *_SEASON[1:]]
        files = _pack(3)
        placed = place_release(files, _scope(entry), _torrent(_resident_record()))

        claim = build_entry_claim(self._release(files, placed), _scope(entry), entry_facts())

        assert claim.preowned_episode_ids == (101,)

    def test_a_misplaced_own_file_is_not_preowned(self) -> None:
        # E01 holds the own group's E02 file (a listed size, not E01's): the grab still owes E01 its file.
        entry = [sonarr_ep(1, 1, ep_id=101, size=20, release_group="RG"), *_SEASON[1:]]
        files = _pack(3)
        placed = place_release(files, _scope(entry), _NEW)

        claim = build_entry_claim(self._release(files, placed), _scope(entry), entry_facts())

        assert claim.preowned_episode_ids == ()
