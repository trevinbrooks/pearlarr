# pyright: strict
"""The grab-time placement: what `place_release` records for the planner, the record a url accretes onto, and the seeds."""

from dataclasses import replace

from pearlarr.grab_placement import (
    NO_RESIDENTS,
    EntryPlacements,
    PendingSeed,
    ResidentScope,
    SeedFile,
    SeedRelease,
    SeedScope,
    TorrentFacts,
    UrlPlacement,
    build_entry_claim,
    build_pending_seeds,
    build_unscoped_seed,
    place_release,
    resident_scopes,
)
from pearlarr.manual_import import EntryNames, PendingImport, normalize_basename, normalized_leaf
from pearlarr.placement_types import EpisodeAssignment, PlacementVerdict, episode_index
from pearlarr.seadex_types import EpisodeRecord, FlaggedUrl, MatchedEpisode, ParsedFileInfo, SeadexDict, SonarrEpisode

from .builders import entry_claim, entry_facts, pending_import, rg_group, sonarr_ep, two_claim_record, url_item

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
_FACTS = TorrentFacts("h1", "RG", True, ("Show - 01.mkv",), (10,))


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


def _pack(count: int = 4) -> list[SeedFile]:
    """The season pack's first `count` files, each matched by Sonarr and sized `10 * n`."""

    return [SeedFile(name, 10 * n, _matched(1, n)) for n, name in enumerate(_PACK[:count], start=1)]


def _resident_record(series_id: int = 7) -> PendingImport:
    """A stored record mapping the pack's first two files, its one claim over the season on `series_id`."""

    return pending_import(
        infohash="h1",
        series_id=series_id,
        file_episode_map={"show - s01e01.mkv": [101], "show - s01e02.mkv": [102]},
        seadex_files=_PACK,
        ordered_episode_ids=(101, 102, 103, 104),
    )


def _verdicts(placement: UrlPlacement) -> set[PlacementVerdict]:
    return {p.verdict for p in placement.assignment.placements}


class TestPlaceRelease:
    """One url's files placed by the import's own passes, its records the planner's coverage vocabulary."""

    def test_a_run_sonarr_read_nothing_of_records_the_window(self) -> None:
        files = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), None)

        assert placement.records == tuple(EpisodeRecord(1, n, 10 * n) for n in range(1, 5))
        assert placement.assignment.assigned == {normalized_leaf(name): [100 + n] for n, name in enumerate(_RUN, 1)}
        assert placement.claimed_ids == {101, 102, 103, 104}
        assert placement.inputs_known
        assert placement.stored is None

    def test_a_whole_series_pack_records_only_the_entrys_slice(self) -> None:
        names = [f"Show - S01E{n:02d}.mkv" for n in range(1, 5)] + ["Show - S02E01.mkv", "Show - S02E02.mkv"]
        parses = [_matched(1, n) for n in range(1, 5)] + [_matched(2, 1), _matched(2, 2)]
        files = [SeedFile(name, 1, parse) for name, parse in zip(names, parses, strict=True)]

        placement = place_release(files, _scope(_SEASON), None)

        assert placement.records == tuple(EpisodeRecord(1, n, 1) for n in range(1, 5))
        assert {p.name for p in placement.assignment.excluded} == {"show - s02e01.mkv", "show - s02e02.mkv"}

    def test_a_file_placed_nowhere_leaves_no_record(self) -> None:
        files = [SeedFile("Show - S02E01.mkv", 1, _matched(2, 1))]

        placement = place_release(files, _scope(_SEASON), None)

        assert placement.records == ()
        assert _verdicts(placement) == {PlacementVerdict.FOREIGN}

    def test_a_held_run_records_nothing_and_its_inputs_are_not_known(self) -> None:
        # One parse request failed: the run is held for import time, and the title re-checks next run.
        files = [SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)]

        placement = place_release(files, _scope(_SEASON), None)

        assert placement.records == ()
        assert not placement.inputs_known
        assert _verdicts(placement) == {PlacementVerdict.HELD}

    def test_a_file_beside_a_held_run_still_places(self) -> None:
        files = [SeedFile("Show - OVA.mkv", 5, _matched(0, 1))] + [
            SeedFile(name, 1, None if n == 3 else _bare(n)) for n, name in enumerate(_RUN, start=1)
        ]

        placement = place_release(files, _scope([_SPECIAL, *_SEASON]), None)

        assert placement.records == (EpisodeRecord(0, 1, 5),)
        assert not placement.inputs_known

    def test_two_files_sharing_a_leaf_keep_their_own_sizes(self) -> None:
        # The gather already stripped the folders: one leaf, placed once, two listed sizes.
        files = [SeedFile("Show - S01E01.mkv", 10, _matched(1, 1)), SeedFile("Show - S01E01.mkv", 20, _matched(1, 1))]

        placement = place_release(files, _scope(_SEASON), None)

        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 1, 20))
        assert placement.assignment.assigned == {"show - s01e01.mkv": [101]}

    def test_a_scope_that_cannot_place_records_nothing(self) -> None:
        # An unread series map: the seed waits for import time, and the planner judges by group and size.
        files = [SeedFile(name, 1, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        scope = SeedScope(1, episode_index(_SEASON), episode_index([]), EntryNames())

        placement = place_release(files, scope, None)

        assert not scope.can_place
        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.inputs_known

    def test_no_files_is_an_empty_placement_with_its_inputs_known(self) -> None:
        placement = place_release([], _scope(_SEASON), None)

        assert placement == UrlPlacement((), EpisodeAssignment(()), (), frozenset(), True, None)


class TestPlaceOntoResident:
    """A listed torrent with a stored record: only the map's leftover is placed, the whole map is recorded."""

    def test_the_leftover_is_placed_and_the_whole_map_recorded(self) -> None:
        record = _resident_record()

        placement = place_release(_pack(), _scope(_SEASON), ResidentScope(record, _INDEXES))

        # The record's two mapped names are not re-placed, and the records and claim read the full map.
        assert placement.assignment.assigned == {"show - s01e03.mkv": [103], "show - s01e04.mkv": [104]}
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

        placement = place_release(files, _scope(_SEASON, al_id=2), ResidentScope(record, _INDEXES))

        assert placement.assignment.placements == ()
        assert placement.records == ()
        assert placement.claimed_ids == frozenset()
        assert placement.inputs_known

    def test_a_re_flag_places_the_leftover_under_its_fresh_window(self) -> None:
        # The entry's stored claim covers two episodes; re-listed over the whole season, its fresh window
        # replaces the stale one in place, so the third file lands where the stale window could not put it.
        record = pending_import(
            infohash="h1",
            file_episode_map={"show - s01e01.mkv": [101]},
            seadex_files=_PACK[:3],
            ordered_episode_ids=(101, 102),
        )
        files = _pack(3)

        stale = place_release(files, _scope(_SEASON[:2]), ResidentScope(record, _INDEXES))
        fresh = place_release(files, _scope(_SEASON), ResidentScope(record, _INDEXES))

        assert stale.assignment.assigned == {"show - s01e02.mkv": [102]}
        assert fresh.assignment.assigned == {"show - s01e02.mkv": [102], "show - s01e03.mkv": [103]}
        assert fresh.claimed_ids == {101, 102, 103}

    def test_an_unread_claim_series_places_nothing_and_its_inputs_are_not_known(self) -> None:
        # A stored claim on a series this run could not read: the leftover waits for the import poll.
        record = _resident_record(series_id=8)
        resident = ResidentScope(record, _INDEXES)

        placement = place_release(_pack(), _scope(_SEASON), resident)

        assert not resident.can_place
        assert placement.assignment.placements == ()
        assert not placement.inputs_known
        # The stored map is still the record's: its in-entry ids record and count as claimed.
        assert placement.records == (EpisodeRecord(1, 1, 10), EpisodeRecord(1, 2, 20))
        assert placement.claimed_ids == {101, 102}
        assert placement.stored is record


class TestResidentScope:
    """The stored claims' windows, the entry's own in place of its stored claim."""

    def test_can_place_needs_every_claims_series_read(self) -> None:
        record = pending_import(claims=(entry_claim(al_id=1, series_id=7), entry_claim(al_id=2, series_id=8)))

        assert not ResidentScope(record, _INDEXES).can_place
        assert ResidentScope(record, {**_INDEXES, 8: _INDEXES[7]}).can_place

    def test_windows_replace_the_entrys_own_claim_in_place(self) -> None:
        own = _scope(_SEASON).target()

        windows = ResidentScope(two_claim_record(), _INDEXES).windows(1, own)

        assert windows[0] is own
        assert [tuple(w.resolved) for w in windows] == [(101, 102, 103, 104), (103, 104)]

    def test_windows_append_a_new_entrys_own_last(self) -> None:
        own = _scope(_SEASON, al_id=3).target()

        windows = ResidentScope(two_claim_record(), _INDEXES).windows(3, own)

        assert windows[-1] is own
        assert [tuple(w.resolved) for w in windows] == [(101, 102), (103, 104), (101, 102, 103, 104)]


class TestResidentScopes:
    """The stored records among an entry's urls, keyed by url."""

    def test_keyed_by_url_over_the_stored_hashes(self) -> None:
        record = _resident_record()
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {
                    "u1": url_item(url="u1", infohash="h1", download=True),
                    "u2": url_item(url="u2", infohash=None, download=True),
                    "u3": url_item(url="u3", infohash="h3", download=True),
                }
            ),
        }

        residents = resident_scopes(seadex_dict, {"h1": record}, _INDEXES)

        # A url without a hash, or whose hash has no stored record, is a new torrent.
        assert set(residents) == {"u1"}
        assert residents["u1"] == ResidentScope(record, _INDEXES)


class TestEntryPlacements:
    """The per-entry fold: records onto the release dict, the groups a failed read held, and the seeds."""

    @staticmethod
    def _entry(names: EntryNames | None = None) -> tuple[EntryPlacements, SeadexDict]:
        scope = _scope(_SEASON, names)
        run = [SeedFile(name, 10 * n, _bare(n)) for n, name in enumerate(_RUN, start=1)]
        held = [SeedFile(name, 1, None if n == 1 else _bare(n)) for n, name in enumerate(_RUN, start=1)]
        placed = EntryPlacements.place(scope, {"u1": run, "u2": held, "u3": []}, NO_RESIDENTS)
        seadex_dict: SeadexDict = {
            "RG": rg_group({"u1": url_item(url="u1", files=_RUN, size=[10, 20, 30, 40], infohash="h1", download=True)}),
            "Held": rg_group({"u2": url_item(url="u2", files=_RUN, size=[1] * 4, infohash="h2", download=True)}),
            "Subs": rg_group({"u3": url_item(url="u3", files=["Show.ass"], size=[1], infohash="h3", download=True)}),
        }
        return placed, seadex_dict

    def test_attach_writes_each_urls_records_and_the_group_union(self) -> None:
        placed, seadex_dict = self._entry()

        placed.attach_records(seadex_dict)

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
        assert dict(seeds["h1"].placements) == {normalize_basename(name): [100 + n] for n, name in enumerate(_RUN, 1)}
        assert seeds["h1"].facts == TorrentFacts("h1", "RG", False, tuple(_RUN), (10, 20, 30, 40))
        assert seeds["h1"].claim.names == names
        assert not seeds["h1"].accreted
        assert dict(seeds["h2"].placements) == {}

    def test_a_url_with_a_stored_record_seeds_onto_it(self) -> None:
        record = _resident_record()
        placed = EntryPlacements.place(_scope(_SEASON), {"u1": _pack()}, {"u1": ResidentScope(record, _INDEXES)})
        seadex_dict: SeadexDict = {
            "RG": rg_group(
                {"u1": url_item(url="u1", files=_PACK, size=[10, 20, 30, 40], infohash="h1", download=True)}
            ),
        }

        seeds = build_pending_seeds(seadex_dict, placed, entry_facts())

        assert seeds["h1"].accreted
        assert seeds["h1"].stored is record
        assert dict(seeds["h1"].placements) == {"show - s01e03.mkv": [103], "show - s01e04.mkv": [104]}


class TestPendingSeedRecordAt:
    """The one record constructor: a birth, or an accretion onto the stored record."""

    def test_born_stamps_the_birth_and_the_claim(self) -> None:
        claim = entry_claim(claimed_at="")
        seed = PendingSeed(
            facts=_FACTS, placements={"show - 01.mkv": [101]}, excluded=("show - 02.mkv",), claim=claim, stored=None
        )

        record = seed.record_at(_STAMP, fresh=True)

        assert not seed.accreted
        assert record == PendingImport(
            infohash="h1",
            release_group="RG",
            is_dual_audio=True,
            seadex_files=("Show - 01.mkv",),
            added_at=_STAMP,
            file_episode_map={"show - 01.mkv": [101]},
            claims=(replace(claim, claimed_at=_STAMP),),
            excluded_files=("show - 02.mkv",),
            release_sizes=(10,),
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
            facts=_FACTS, placements={"show - 02.mkv": [102]}, excluded=("show - 03.mkv",), claim=claim, stored=stored
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
        seed = PendingSeed(facts=_FACTS, placements={}, excluded=(), claim=claim, stored=stored)

        record = seed.record_at(_STAMP, fresh=False)

        assert record.claims == (*stored.claims, replace(claim, claimed_at=_STAMP))
        assert [c.claimed_at for c in record.claims] == ["2026-01-01 00:00:00", _STAMP]

    def test_fresh_restamps_every_clock(self) -> None:
        stored = pending_import(infohash="h1", added_at="2026-01-01 00:00:00")
        seed = PendingSeed(
            facts=_FACTS, placements={}, excluded=(), claim=entry_claim(al_id=2, claimed_at=""), stored=stored
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
        assert seed.facts == TorrentFacts(
            infohash="h1", release_group="RG", is_dual_audio=True, seadex_files=(), release_sizes=()
        )
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
        placed = place_release(files, _scope(_SEASON), ResidentScope(_resident_record(), _INDEXES))

        claim = build_entry_claim(self._release(files, placed), _scope(_SEASON), entry_facts())

        # This call placed E03 alone, yet the slice also carries the record's E01 and E02.
        assert placed.assignment.assigned == {"show - s01e03.mkv": [103]}
        assert claim.slice_coverage == "S01 E01-E03"
        assert claim.ordered_episode_ids == (101, 102, 103, 104)
        assert claim.claimed_at == ""

    def test_preowned_reads_every_claimed_id(self) -> None:
        # E01 (the record's, not this call's) already holds the own group's file at a listed size.
        entry = [sonarr_ep(1, 1, ep_id=101, size=10, release_group="RG"), *_SEASON[1:]]
        files = _pack(3)
        placed = place_release(files, _scope(entry), ResidentScope(_resident_record(), _INDEXES))

        claim = build_entry_claim(self._release(files, placed), _scope(entry), entry_facts())

        assert claim.preowned_episode_ids == (101,)
