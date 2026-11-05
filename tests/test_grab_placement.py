# pyright: strict
"""The grab-time placement: what `place_release` records for the planner, the record a url accretes onto, and the seeds."""

from collections.abc import Sequence
from dataclasses import replace

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
    build_pending_seeds,
    build_unscoped_seed,
    entry_hashes,
    identity_overrides,
    place_release,
)
from pearlarr.manual_import import (
    EMPTY_LISTING,
    NO_IDENTIFIED,
    EntryNames,
    EntryWindow,
    FileEpisodeMap,
    PendingImport,
    normalize_basename,
    normalized_leaf,
)
from pearlarr.placement_types import (
    NO_EVIDENCE,
    EpisodeAssignment,
    ListingEvidence,
    ListingsRead,
    PlacementVerdict,
    episode_index,
)
from pearlarr.seadex_types import (
    EpisodeKey,
    EpisodeRecord,
    FlaggedUrl,
    GrabHold,
    MatchedEpisode,
    ParsedFileInfo,
    SeadexDict,
    SonarrEpisode,
)

from .builders import (
    ALIASED_PACK,
    UNREAD_EVIDENCE,
    entry_claim,
    entry_facts,
    entry_window,
    known_torrent,
    pending_import,
    rg_group,
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
    return SeedScope(al_id, episode_index(entry), episode_index(_SERIES), names or EntryNames())


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


def _resident_record(series_id: int = 7) -> PendingImport:
    """A stored record mapping the pack's first two files, its one claim over the season on `series_id`."""

    return pending_import(
        infohash="h1",
        series_id=series_id,
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

    def test_a_held_run_records_nothing_and_its_inputs_are_not_known(self) -> None:
        # One parse request failed: the run is held for import time, and the title re-checks next run.
        files = [SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), _NEW)

        assert placement.records == ()
        assert not placement.inputs_known
        assert verdicts_of(placement.assignment) == {PlacementVerdict.HELD}

    def test_a_file_beside_a_held_run_still_places(self) -> None:
        files = [SeedFile("Show - OVA.mkv", 5, _matched(0, 1))] + [
            SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _scope([_SPECIAL, *_SEASON]), _NEW)

        assert placement.records == (EpisodeRecord(0, 1, 5),)
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

    def test_a_scope_that_cannot_place_records_nothing(self) -> None:
        # An unread series map: the seed waits for import time, and the planner judges by group and size.
        files = [SeedFile(name, 1, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        scope = SeedScope(1, episode_index(_SEASON), episode_index([]), EntryNames())

        placement = place_release(files, scope, _NEW)

        assert not scope.can_place
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.inputs_known

    def test_no_files_is_an_empty_placement_with_its_inputs_known(self) -> None:
        placement = place_release([], _scope(_SEASON), _NEW)

        assert placement == UrlPlacement(
            files=(),
            assignment=EpisodeAssignment(()),
            records=(),
            claimed_ids=frozenset(),
            inputs_known=True,
            hold=None,
            stored=None,
            intended_sizes={},
            identified=NO_IDENTIFIED,
        )


class TestPlaceIdentified:
    """A file whose size matches an episode goes on that episode, and the grab remembers it for the import."""

    _LOST = SeedFile("Show - Lost Tape.mkv", 70, ParsedFileInfo())

    def test_a_numberless_file_is_placed_by_its_listed_size(self) -> None:
        torrent = known_torrent(indexes=_INDEXES, evidence=torrent_evidence(identities={70: 103}))

        placement = place_release([self._LOST, *_pack(2)], _scope(_SEASON), torrent)

        assert placement.assignment.assigned["show - lost tape.mkv"] == (103,)
        assert PlacementVerdict.IDENTIFIED in verdicts_of(placement.assignment)
        assert placement.identified == {"show - lost tape.mkv": 103}
        assert placement.inputs_known

    def test_a_hash_less_url_is_placed_by_the_series_identities(self) -> None:
        # No listing matches a torrent without a hash, but the series' size identities still apply to it.
        seadex_dict: SeadexDict = {"RG": rg_group({"u1": url_item(url="u1", infohash=None, download=True)})}
        known = TorrentReads({}, _INDEXES, ListingsRead({}, {70: 103})).known(seadex_dict)

        placement = place_release([self._LOST, *_pack(2)], _scope(_SEASON), known["u1"])

        assert placement.identified == {"show - lost tape.mkv": 103}
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

    def test_an_unread_claim_series_places_nothing_and_its_inputs_are_not_known(self) -> None:
        # A stored claim on a series this run could not read: the leftover waits for the import poll.
        record = _resident_record(series_id=8)
        torrent = _torrent(record)

        placement = place_release(_pack(), _scope(_SEASON), torrent)

        assert torrent.windows(1, _scope(_SEASON).target(NO_EVIDENCE)) is None
        assert placement.assignment.placements == ()
        assert not placement.inputs_known
        # The stored map is still the record's: its in-entry ids record and count as claimed.
        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 2, 20))
        assert placement.claimed_ids == {101, 102}
        assert placement.stored is record

    def test_a_stored_record_under_an_unread_own_series_leaves_the_inputs_unknown(self) -> None:
        # The entry's own series list did not serve: a coarse verdict would be final, so the title is re-checked.
        unread = SeedScope(1, episode_index(_SEASON), episode_index([]), EntryNames())

        placement = place_release(_pack(), unread, _torrent(_resident_record()))

        assert placement.assignment.placements == ()
        assert not placement.inputs_known


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
        scope = SeedScope(2, episode_index([ep for ep in _TITLED if ep.id in entry]), _TITLED_INDEX, EntryNames())
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

    def test_an_entry_placing_nothing_is_grabbed_despite_entry_1s_refutal(self) -> None:
        # The titled file is the third episode's, under the second's key. None of the files is the special's.
        files = {
            "show - 01.mkv": _matched(1, 1),
            "show - 02 - charlie rise.mkv": _matched(1, 2),
            "show - 03.mkv": _matched(1, 3),
        }

        placement = self._place(files, {"show - 02 - charlie rise.mkv": (103,)}, [501])

        assert placement.hold is None


class TestPlaceHeld:
    """A url the grab holds: a specials pack the listing contradicts, or a listing the run could not read."""

    @staticmethod
    def _specials(count: int = 3) -> list[SeedFile]:
        """A pack of the first `count` specials, each read by Sonarr as its own number's special."""

        return [SeedFile(f"Show.S00E{n:02d}.mkv", 10 * n, _matched(0, n)) for n in range(1, count + 1)]

    @staticmethod
    def _specials_scope(*numbers: int) -> SeedScope:
        """Entry 1 over the specials `numbers`, within the all-specials series."""

        return SeedScope(1, episode_index([_SPECIALS[n - 1] for n in numbers]), episode_index(_SPECIALS), EntryNames())

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
        scope = SeedScope(1, episode_index(season), series, EntryNames())

        placement = place_release(files, scope, known_torrent(indexes={7: series}))

        assert placement.hold is GrabHold.MISNUMBERED
        assert placement.assignment.assigned == {"show - s01e01.mkv": (101,), "show - s01e02.mkv": (102,)}
        assert placement.records == ()

    @staticmethod
    def _titled_scope(*numbers: int) -> SeedScope:
        """Entry 1 over season 1's episodes `numbers`, within a titled four-episode season."""

        titles = ("Pilot Flight", "The Return", "Festival Night", "Final Bell")
        season = [sonarr_ep(1, n, ep_id=100 + n, episode_file_id=0, title=t) for n, t in enumerate(titles, 1)]
        return SeedScope(1, episode_index([season[n - 1] for n in numbers]), episode_index(season), EntryNames())

    def test_a_pack_its_titles_refute_that_places_nothing_holds_the_url_and_records_nothing(self) -> None:
        # The titled file is the third episode's, under the plain file's key.
        files = [
            SeedFile("show - 02 - Festival Night [grp].mkv", 10, _bare_key(1, 2)),
            SeedFile("show - 03 [grp].mkv", 20, _bare_key(1, 3)),
        ]

        placement = place_release(files, self._titled_scope(1, 2, 3), known_torrent())

        assert placement.hold is GrabHold.TITLE_REFUTED
        assert placement.records == ()

    def test_a_pack_its_titles_refute_that_places_some_files_is_not_held(self) -> None:
        # Each key is one episode early, as the titles show.
        names = ("show - 01 - The Return [grp].mkv", "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv")
        files = [SeedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(names, 1)]

        placement = place_release(files, self._titled_scope(1, 2, 3), known_torrent())

        assert placement.hold is None
        assert placement.assignment.assigned == {
            normalize_basename(names[0]): (102,),
            normalize_basename(names[1]): (103,),
        }
        assert placement.assignment.hand_import_files == (normalize_basename(names[2]),)
        assert len(placement.records) == 2

    def test_a_pack_the_listing_agrees_with_places_and_is_not_held(self) -> None:
        placement = place_release(self._specials(), self._specials_scope(1, 2), self._listing((501, 502, 503)))

        assert placement.hold is None
        assert placement.assignment.assigned == {"show.s00e01.mkv": (501,), "show.s00e02.mkv": (502,)}
        assert verdicts_of(placement.assignment) == {PlacementVerdict.EXACT, PlacementVerdict.FOREIGN}

    def test_a_pack_its_aliases_map_onto_the_listing_places_and_is_not_held(self) -> None:
        # The pack's `1` isn't a listed special, but the entries' aliases map 1, 2 and 3 onto 2, 4 and 6.
        torrent = known_torrent(
            indexes={7: episode_index(_SPECIALS)},
            evidence=torrent_evidence(*ALIASED_PACK.entries, aliases=ALIASED_PACK.special_aliases),
        )

        placement = place_release(self._specials(), self._specials_scope(2, 4), torrent)

        assert placement.hold is None
        assert placement.assignment.assigned == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (504,)}
        assert verdicts_of(placement.assignment) == {PlacementVerdict.ALTERNATE, PlacementVerdict.ALIASED_ELSEWHERE}
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
        scope = self._specials_scope(2, 4) if al_id == 1 else SeedScope(2, self._OTHER, self._OTHER, EntryNames())

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
        scope = SeedScope(2, episode_index(second), series, EntryNames())

        placement = place_release(files, scope, known_torrent(record, indexes={7: series}))

        assert PlacementVerdict.MISNUMBERED in verdicts_of(placement.assignment)
        assert placement.hold is None
        assert placement.records == (EpisodeRecord(2, 1, 20), EpisodeRecord(2, 2, 30))

    def test_an_unread_listing_holds_the_url_and_places_nothing(self) -> None:
        # No verdict could judge the pack, and import time (no listing) would place it by its numbers.
        placement = place_release(self._specials(), self._specials_scope(1, 2), self._listing(None))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert not placement.inputs_known

    def test_a_parse_miss_under_a_specials_listing_holds_the_url(self) -> None:
        # One parse unread: the pack cannot be judged against its listing, and import time (no listing) would
        # place it by its numbers. Held until the parse reads.
        files = [SeedFile(f"Show.S00E{n:02d}.mkv", 10 * n, None if n == 2 else _matched(0, n)) for n in (1, 2, 3)]

        placement = place_release(files, self._specials_scope(1, 2), self._listing((501, 502, 503)))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known

    def test_a_parse_miss_under_a_seasoned_listing_is_no_hold(self) -> None:
        # A season's numbering is stable: no listing judges the pack, so an unread parse holds nothing.
        files = [SeedFile(_PACK[0], 10, None), *_pack()[1:]]

        placement = place_release(files, _scope(_SEASON), _torrent(listed=[entry_window(1, (101, 102, 103, 104))]))

        assert placement.hold is None
        assert not placement.inputs_known

    def test_an_unread_listing_over_a_seasoned_window_places_by_number_and_holds(self) -> None:
        # A season pack isn't checked against a listing, so it's placed as if unlisted. The size identities
        # weren't read, though, so the url is held until they are.
        placement = place_release(_pack(), _scope(_SEASON), _torrent(listed=None))

        assert placement.hold is GrabHold.INPUT_UNREAD
        assert not placement.inputs_known
        assert placement.assignment.assigned == {
            normalize_basename(name): (100 + n,) for n, name in enumerate(_PACK, 1)
        }

    def test_a_parse_miss_under_a_listing_leaving_a_window_special_out_is_no_hold(self) -> None:
        # The listing does not cover the window, so it judges nothing: an unread parse holds nothing either.
        files = [SeedFile(f"Show.S00E{n:02d}.mkv", 10 * n, None if n == 2 else _matched(0, n)) for n in (1, 2, 3)]

        placement = place_release(files, self._specials_scope(1, 2), self._listing((502, 504, 506)))

        assert placement.hold is None
        assert not placement.inputs_known

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


class TestKnownTorrent:
    """The windows a torrent is placed under: the stored claims' with the entry's own in its claim's place."""

    def test_windows_need_every_claims_series_read(self) -> None:
        record = pending_import(claims=(entry_claim(al_id=1, series_id=7), entry_claim(al_id=2, series_id=8)))
        own = _scope(_SEASON).target(NO_EVIDENCE)

        assert _torrent(record).windows(1, own) is None
        both = KnownTorrent(record, {**_INDEXES, 8: _INDEXES[7]}, TorrentEvidence(EMPTY_LISTING, {}))
        assert both.windows(1, own) is not None

    def test_an_unread_listing_gives_no_evidence(self) -> None:
        # Whether the unread listing holds the url is `place_release`'s call: the windows build as if listed nowhere.
        torrent = known_torrent(evidence=UNREAD_EVIDENCE)

        assert torrent.listing_evidence({"show - 01.mkv": 10}) == NO_EVIDENCE

    def test_the_evidence_maps_matched_sizes_to_episodes(self) -> None:
        torrent = known_torrent(evidence=torrent_evidence(entry_window(1, (101, 102)), identities={10: 101, 30: 103}))

        evidence = torrent.listing_evidence({"show - 01.mkv": 10, "show - 02.mkv": 20})

        assert evidence == ListingEvidence(torrent_listing(entry_window(1, (101, 102))), {"show - 01.mkv": 101})

    def test_the_evidence_merges_the_stored_records_names(self) -> None:
        # Without a fresh read the stored names are used as is. With one, a file the two disagree on is dropped.
        record = pending_import(identified={"show - 02.mkv": 102, "show - 03.mkv": 103})
        sizes = {"show - 01.mkv": 10, "show - 03.mkv": 30}
        read = known_torrent(record, evidence=torrent_evidence(identities={10: 101, 30: 104}))
        unread = known_torrent(record, evidence=UNREAD_EVIDENCE)

        assert read.listing_evidence(sizes).identified == {"show - 01.mkv": 101, "show - 02.mkv": 102}
        assert unread.listing_evidence(sizes).identified == {"show - 02.mkv": 102, "show - 03.mkv": 103}

    def test_the_evidence_carries_the_listings_aliases(self) -> None:
        torrent = known_torrent(evidence=torrent_evidence(entry_window(1, (501, 502)), aliases={1: 2}))

        assert torrent.listing_evidence({}) == ListingEvidence(
            torrent_listing(entry_window(1, (501, 502)), aliases={1: 2})
        )

    def test_every_window_carries_the_own_windows_listing(self) -> None:
        own = _scope(_SEASON).target(
            ListingEvidence(torrent_listing(entry_window(1, (101, 102))), {"show - 01.mkv": 101})
        )

        windows = _torrent(two_claim_record()).windows(1, own)

        assert windows is not None
        assert [w.listing for w in windows] == [own.listing, own.listing]

    def test_windows_replace_the_entrys_own_claim_in_place(self) -> None:
        own = _scope(_SEASON).target(NO_EVIDENCE)

        windows = _torrent(two_claim_record()).windows(1, own)

        assert windows is not None
        assert [tuple(w.resolved) for w in windows] == [(101, 102, 103, 104), (103, 104)]

    def test_windows_append_a_new_entrys_own_last(self) -> None:
        own = _scope(_SEASON, al_id=3).target(NO_EVIDENCE)

        windows = _torrent(two_claim_record()).windows(3, own)

        assert windows is not None
        assert [tuple(w.resolved) for w in windows] == [(101, 102), (103, 104), (101, 102, 103, 104)]


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

        # The subs-only url seeds nothing; the held one is seeded with an empty map for import time.
        assert set(seeds) == {"h1", "h2"}
        assert dict(seeds["h1"].placements) == {normalize_basename(name): (100 + n,) for n, name in enumerate(_RUN, 1)}
        listed = {normalize_basename(name): 10 * n for n, name in enumerate(_RUN, 1)}
        assert seeds["h1"].facts == torrent_facts(
            seadex_files=tuple(_RUN), release_sizes=(10, 20, 30, 40), sizes_by_name=listed
        )
        assert seeds["h1"].claim.names == names
        assert not seeds["h1"].accreted
        assert dict(seeds["h2"].placements) == {}

    def test_a_url_with_a_stored_record_seeds_onto_it(self) -> None:
        record = _resident_record()
        placed = EntryPlacements.place(_scope(_SEASON), {"u1": _pack()}, {"u1": _torrent(record)})
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {"u1": url_item(url="u1", files=_PACK, size=[10, 20, 30, 40], infohash="h1", download=True)}
            ),
        }

        seeds = build_pending_seeds(seadex_dict, placed, entry_facts())

        assert seeds["h1"].accreted
        assert seeds["h1"].stored is record
        assert dict(seeds["h1"].placements) == {"show - s01e03.mkv": (103,), "show - s01e04.mkv": (104,)}


class TestPendingSeedRecordAt:
    """The one record constructor: a birth, or an accretion onto the stored record."""

    def test_born_stamps_the_birth_and_the_claim(self) -> None:
        claim = entry_claim(claimed_at="")
        seed = PendingSeed(
            facts=_FACTS,
            placements={"show - 01.mkv": (101,)},
            excluded=("show - 02.mkv",),
            identified=NO_IDENTIFIED,
            claim=claim,
            stored=None,
        )

        record = seed.record_at(_STAMP, fresh=True)

        assert not seed.accreted
        assert record == PendingImport(
            infohash="h1",
            release_group="RG",
            is_dual_audio=True,
            seadex_files=("Show - 01.mkv",),
            added_at=_STAMP,
            file_episode_map={"show - 01.mkv": (101,)},
            claims=(replace(claim, claimed_at=_STAMP),),
            excluded_files=("show - 02.mkv",),
            release_sizes=(10,),
            sizes_by_name={"show - 01.mkv": 10},
        )
        # A birth is stamped whether or not the add was fresh.
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
        claim = entry_claim(claimed_at="", ordered_episode_ids=(101, 102, 103), preowned_episode_ids=(102,))
        seed = PendingSeed(
            facts=_FACTS,
            placements={"show - 02.mkv": (102,)},
            excluded=("show - 03.mkv",),
            identified=NO_IDENTIFIED,
            claim=claim,
            stored=stored,
        )

        record = seed.record_at(_STAMP, fresh=False)

        assert seed.accreted
        assert dict(record.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}
        assert record.excluded_files == ("show - 09.mkv", "show - 03.mkv")
        # The entry's own claim is replaced under its first clock, its first preowned ids kept.
        assert record.claims == (replace(claim, claimed_at=stored.added_at, preowned_episode_ids=(101,)),)
        assert record.awaiting_cleanup is False
        # Not a fresh add: the birth keeps its clock, and the identity stays the stored record's.
        assert record.added_at == "2026-01-01 00:00:00"
        assert record.release_group == stored.release_group

    def test_accreted_appends_a_new_entrys_claim(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        claim = entry_claim(al_id=2, claimed_at="")
        seed = PendingSeed(
            facts=_FACTS, placements={}, excluded=(), identified=NO_IDENTIFIED, claim=claim, stored=stored
        )

        record = seed.record_at(_STAMP, fresh=False)

        assert record.claims == (*stored.claims, replace(claim, claimed_at=_STAMP))
        assert [c.claimed_at for c in record.claims] == ["2026-01-01 00:00:00", _STAMP]

    def test_accreted_onto_a_record_without_sizes_by_name_learns_them(self) -> None:
        stored = pending_import(infohash="h1")
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            identified=NO_IDENTIFIED,
            claim=entry_claim(claimed_at=""),
            stored=stored,
        )

        assert seed.record_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 10}

    def test_accreted_keeps_the_stored_sizes_by_name(self) -> None:
        stored = pending_import(infohash="h1", sizes_by_name={"show - 01.mkv": 99})
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            identified=NO_IDENTIFIED,
            claim=entry_claim(claimed_at=""),
            stored=stored,
        )

        assert seed.record_at(_STAMP, fresh=False).sizes_by_name == {"show - 01.mkv": 99}

    def test_born_keeps_the_identified_names(self) -> None:
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            identified={"show - 01.mkv": 101},
            claim=entry_claim(claimed_at=""),
            stored=None,
        )

        assert seed.record_at(_STAMP, fresh=True).identified == {"show - 01.mkv": 101}

    def test_accreted_merges_the_identified_names(self) -> None:
        stored = pending_import(infohash="h1", identified={"show - 01.mkv": 101})
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            identified={"show - 02.mkv": 102},
            claim=entry_claim(claimed_at=""),
            stored=stored,
        )

        assert seed.record_at(_STAMP, fresh=False).identified == {"show - 01.mkv": 101, "show - 02.mkv": 102}

    def test_fresh_restamps_every_clock(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        seed = PendingSeed(
            facts=_FACTS,
            placements={},
            excluded=(),
            identified=NO_IDENTIFIED,
            claim=entry_claim(al_id=2, claimed_at=""),
            stored=stored,
        )

        record = seed.record_at(_STAMP, fresh=True)

        assert record.added_at == _STAMP
        assert [c.claimed_at for c in record.claims] == [_STAMP, _STAMP]


class TestBuildUnscopedSeed:
    """A Radarr grab's seed: the listing's facts, no files or window, an id-less claim on the entry."""

    @staticmethod
    def _flagged() -> FlaggedUrl:
        return FlaggedUrl("RG", "https://nyaa.si/view/1", url_item(infohash="h1", is_dual_audio=True), "h1")

    def test_a_new_torrent_seeds_an_id_less_claim(self) -> None:
        facts = entry_facts(al_id=9, series_id=0, title="Movie", url="https://releases.moe/9")

        seed = build_unscoped_seed(self._flagged(), facts, None)

        assert not seed.accreted
        assert seed.facts == torrent_facts(is_dual_audio=True)
        assert seed.placements == {} and seed.excluded == ()
        assert seed.claim == entry_claim(
            al_id=9, series_id=0, title="Movie", url="https://releases.moe/9", claimed_at="", guards=facts.guards
        )

    def test_a_listed_torrent_accretes_its_stored_record(self) -> None:
        stored = pending_import(infohash="h1", al_id=1, series_id=0)

        record = build_unscoped_seed(self._flagged(), entry_facts(al_id=2, series_id=0), stored).record_at(
            _STAMP, fresh=False
        )

        assert record.release_group == stored.release_group
        assert [claim.al_id for claim in record.claims] == [1, 2]


class TestEntryClaim:
    """The entry's claim reads the whole map inside the entry, its clock blank until the record is stamped."""

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
        assert claim.claimed_at == ""

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
