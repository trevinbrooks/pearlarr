# pyright: strict
"""Direct tests for the pending import record: building it, its placements, claims and windows, and its JSON form."""

import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from pearlarr.file_paths import normalize_basename
from pearlarr.pending_import import (
    NO_ORDER_ONLY,
    GuardFacts,
    OwnedEpisode,
    PendingImport,
    TitleClaim,
    TitleNames,
    TitleWindow,
    added_at_of,
    joined_map,
    sizes_by_episode,
)
from pearlarr.releases import EpisodeKey, OrderOnlyKey

from .builders import SEP, claim_al_ids, entry_claim, entry_window, pending_import, sonarr_ep

HALVES = tuple(normalize_basename(f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv") for n in (1, 2))
"""The two halves of a split opener, which only the DVD order numbers."""

_HALF_KEYS = (OrderOnlyKey("dvd", EpisodeKey(7, 1)), OrderOnlyKey("dvd", EpisodeKey(7, 2)))
"""The halves' DVD-order keys."""


class TestPendingImportBuild:
    """Built from whatever the caller passed, the record folds its map and exclusions and holds them read-only."""

    def test_the_map_folds_its_keys_and_drops_zero_ids(self) -> None:
        pending = pending_import(
            file_episode_map={"Show - 01.MKV": [101], "Show - 02.mkv": [0], "Show - 03.mkv": [0, 103]}
        )

        assert dict(pending.file_episode_map) == {"show - 01.mkv": (101,), "show - 03.mkv": (103,)}

    def test_two_spellings_of_one_name_keep_the_first(self) -> None:
        pending = pending_import(file_episode_map={"Show - 01.mkv": [101], "show - 01.mkv": [111]})

        assert dict(pending.file_episode_map) == {"show - 01.mkv": (101,)}

    def test_a_zero_only_first_spelling_yields_to_the_one_with_ids(self) -> None:
        pending = pending_import(file_episode_map={"Show - 01.mkv": [0], "show - 01.mkv": [111]})

        assert dict(pending.file_episode_map) == {"show - 01.mkv": (111,)}

    def test_the_exclusions_fold_and_dedupe_in_order(self) -> None:
        pending = pending_import(excluded_files=["SHOW - 03.MKV", " Show - 02.mkv ", "show - 03.mkv"])

        assert pending.excluded_files == ("show - 03.mkv", "show - 02.mkv")

    def test_the_sizes_fold_their_keys_and_drop_a_name_whose_spellings_disagree(self) -> None:
        pending = pending_import(
            sizes_by_name={"Sub/Show - 01.MKV": 100, "Show - 02.mkv": 200, "show - 02.mkv": 201},
        )

        assert pending.sizes_by_name == {"show - 01.mkv": 100}

    def test_the_fold_ranks_the_map_over_the_hand_import_files_over_the_exclusions(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101]},
            excluded_files=["show - 01.mkv", "SHOW - 02.MKV", "show - 03.mkv", "show - 05.mkv"],
            hand_import_files=["SHOW - 04.MKV", "Show - 01.mkv", "show - 02.mkv", "Sub/Show - 03.mkv", "show - 04.mkv"],
        )

        assert pending.hand_import_files == ("show - 04.mkv", "show - 02.mkv", "show - 03.mkv")
        assert pending.excluded_files == ("show - 05.mkv",)

    def test_the_order_only_keys_fold_onto_the_hand_import_files(self) -> None:
        pending = pending_import(
            file_episode_map={HALVES[0]: [401]},
            hand_import_files=list(HALVES),
            order_only={HALVES[0]: _HALF_KEYS[0], HALVES[1].upper(): _HALF_KEYS[1]},
        )

        assert pending.order_only == {HALVES[1]: _HALF_KEYS[1]}

    def test_the_map_is_detached_and_read_only(self) -> None:
        # The record copies the caller's map at construction and wraps it read-only, so neither the
        # caller's dict nor a holder of the record can mutate the map behind it. Built through `replace`: the
        # builder copies the map itself, which would pass this pin on its own.
        seed: dict[str, tuple[int, ...]] = {"show - 01.mkv": (101,)}
        pending = replace(pending_import(), file_episode_map=seed)
        seed["show - 02.mkv"] = (102,)
        mutable = cast("dict[str, tuple[int, ...]]", pending.file_episode_map)

        with pytest.raises(TypeError):
            mutable["show - 02.mkv"] = (102,)
        assert dict(pending.file_episode_map) == {"show - 01.mkv": (101,)}


_NO_OP_CHANGES: dict[str, Callable[[PendingImport], PendingImport]] = {
    "no placements": lambda p: p.with_placements({}),
    "a placement of a held name": lambda p: p.with_placements({"show - 01.mkv": (111,)}),
    "no exclusions": lambda p: p.with_exclusions(()),
    "an exclusion already held": lambda p: p.with_exclusions(["SHOW - 02.MKV"]),
    "no sizes onto none": lambda p: p.with_sizes_by_name({}),
    "no hand-import files": lambda p: p.with_hand_import_files((), NO_ORDER_ONLY),
    "a hand-import file already held": lambda p: p.with_hand_import_files(["SHOW - 03.MKV"], NO_ORDER_ONLY),
    "the stored claim again": lambda p: p.with_claim(p.claims[0]),
}
"""Changes that leave a record equal: it holds `show - 01.mkv` on 101, excludes `show - 02.mkv`, and leaves
`show - 03.mkv` for a hand import."""


class TestPendingImportPlacements:
    """`with_placements` and `with_exclusions` grow the record, and `unplaced_names` is what a rebuild may fill.

    Every `with_*` hands the record back on a no-op.
    """

    def test_with_exclusions_appends_after_the_stored_ones(self) -> None:
        # The recorded order is the reading order: what the record already held, then this poll's additions.
        pending = pending_import(excluded_files=["show - 01.mkv"])

        healed = pending.with_exclusions(["show - 03.mkv", "show - 02.mkv"])

        assert healed.excluded_files == ("show - 01.mkv", "show - 03.mkv", "show - 02.mkv")

    @pytest.mark.parametrize("change", _NO_OP_CHANGES.values(), ids=list(_NO_OP_CHANGES))
    def test_a_change_that_leaves_the_record_equal_returns_the_record(
        self, change: Callable[[PendingImport], PendingImport]
    ) -> None:
        # One rule for every `with_*`: the record seam gates its save on it.
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101]},
            excluded_files=["show - 02.mkv"],
            hand_import_files=["show - 03.mkv"],
        )

        assert change(pending) is pending

    def test_with_placements_keeps_the_stored_ids_of_a_name_it_holds(self) -> None:
        pending = pending_import(file_episode_map={"show - 01.mkv": [101]})

        healed = pending.with_placements({"show - 01.mkv": (111,), "show - 02.mkv": (102,)})

        assert dict(healed.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}

    def test_with_hand_import_files_keeps_the_stored_key_of_a_name_it_keys(self) -> None:
        pending = pending_import(hand_import_files=[HALVES[0]], order_only={HALVES[0]: _HALF_KEYS[0]})

        healed = pending.with_hand_import_files(
            [HALVES[0].upper(), HALVES[1]], {HALVES[0].upper(): _HALF_KEYS[1], HALVES[1]: _HALF_KEYS[1]}
        )

        assert healed.order_only == {HALVES[0]: _HALF_KEYS[0], HALVES[1]: _HALF_KEYS[1]}

    def test_with_placements_lifts_a_placed_name_out_of_the_exclusions(self) -> None:
        # A name an earlier poll excluded that a later import placed is a placement, never both.
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", "Show - 02.mkv"],
            excluded_files=["show - 02.mkv", "show - 03.mkv"],
        )

        healed = pending.with_placements({"show - 02.mkv": (102,)})

        assert healed.excluded_files == ("show - 03.mkv",)
        assert dict(healed.file_episode_map) == {"show - 01 [1080p].mkv": (101,), "show - 02.mkv": (102,)}

    def test_with_placements_lifts_a_placed_name_out_of_the_hand_import_files(self) -> None:
        pending = pending_import(hand_import_files=["show - 02.mkv", "show - 03.mkv"])

        healed = pending.with_placements({"show - 02.mkv": (102,)})

        assert healed.hand_import_files == ("show - 03.mkv",)

    def test_unplaced_names_is_the_listing_minus_map_and_exclusions(self) -> None:
        pending = pending_import(
            file_episode_map={"Show - 01.mkv": [101]},
            seadex_files=["Show - 01.mkv", "Show - 02.mkv", "Show - 03.mkv"],
            excluded_files=["show - 03.mkv"],
        )
        assert pending.unplaced_names() == {"show - 02.mkv"}

    @pytest.mark.parametrize(
        ("file_episode_map", "excluded_files"),
        [
            ({}, []),
            ({"Show - 01.mkv": [101]}, []),
            ({"Show - 01.mkv": [101]}, ["show - 02.mkv"]),
            ({"Show - 01.mkv": [101], "Show - 02.mkv": [102]}, []),
        ],
    )
    def test_seed_coverage_accounted_matches_unplaced_names(
        self,
        file_episode_map: dict[str, list[int]],
        excluded_files: list[str],
    ) -> None:
        # The two reads must not drift: accounted means nothing is left unplaced.
        pending = pending_import(
            file_episode_map=file_episode_map,
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            excluded_files=excluded_files,
        )
        assert pending.seed_coverage().accounted == (not pending.unplaced_names())


class TestSizesByEpisode:
    """`sizes_by_episode`: each mapped episode id to its file's listed size."""

    def test_sizes_each_episode_a_listed_file_is_mapped_to(self) -> None:
        sizes = sizes_by_episode({"a.mkv": (1,), "b.mkv": (2, 3), "c.mkv": (4,)}, {"a.mkv": 100, "b.mkv": 200})

        assert sizes == {1: 100, 2: 200, 3: 200}

    def test_an_id_two_files_size_differently_is_dropped(self) -> None:
        assert sizes_by_episode({"a.mkv": (1, 2), "b.mkv": (2,)}, {"a.mkv": 100, "b.mkv": 200}) == {1: 100}


class TestJoinedMap:
    """`joined_map`: a base map, then the added map's new names."""

    def test_adds_the_new_names_after_the_base_entries(self) -> None:
        joined = joined_map({"b.mkv": (2,), "a.mkv": (1,)}, {"c.mkv": (3,)})

        assert list(joined.items()) == [("b.mkv", (2,)), ("a.mkv", (1,)), ("c.mkv", (3,))]

    def test_a_shared_name_keeps_the_base_ids(self) -> None:
        assert joined_map({"a.mkv": (1,)}, {"a.mkv": (9,), "b.mkv": (2,)}) == {"a.mkv": (1,), "b.mkv": (2,)}


class TestListedSizes:
    """`sizes_by_name`: detached and read-only, learned once by `with_sizes_by_name`."""

    def test_detached_from_the_callers_map(self) -> None:
        sizes = {"a.mkv": 100}
        pending = pending_import(sizes_by_name=sizes)
        sizes["a.mkv"] = 1

        assert pending.sizes_by_name == {"a.mkv": 100}

    def test_read_only(self) -> None:
        sizes = cast("dict[str, int]", pending_import(sizes_by_name={"a.mkv": 100}).sizes_by_name)

        with pytest.raises(TypeError):
            sizes["a.mkv"] = 1

    def test_with_sizes_by_name_fills_an_empty_record(self) -> None:
        assert pending_import().with_sizes_by_name({"a.mkv": 100}).sizes_by_name == {"a.mkv": 100}

    def test_with_sizes_by_name_keeps_a_recorded_map(self) -> None:
        pending = pending_import(sizes_by_name={"a.mkv": 100})

        assert pending.with_sizes_by_name({"a.mkv": 1}) is pending


class TestPendingImportClaims:
    """The claim reads over a record: every entry's series, window, guards, and ids, in claim order."""

    def test_series_ids_are_distinct_in_claim_order(self) -> None:
        pending = pending_import(
            claims=(
                entry_claim(al_id=3, series_id=8),
                entry_claim(al_id=1, series_id=7),
                entry_claim(al_id=2, series_id=8),
            ),
        )

        assert pending.series_ids == (8, 7)

    def test_a_claim_persists_as_the_spelled_out_schema(self) -> None:
        # The persisted shape is pinned: a new field never enters a row without a migration.
        claim = entry_claim(
            coverage="S1",
            url="https://example.invalid/1",
            ordered_episode_ids=(101, 102),
            names=TitleNames("Show", ("Show!",)),
            preowned_episode_ids=(101,),
            slice_coverage="S1 E1-2",
        )

        assert json.dumps(claim.to_json()) == (
            '{"al_id": 1, "series_id": 7, "title": "Show", "coverage": "S1", "url": "https://example.invalid/1", '
            '"ordered_episode_ids": [101, 102], "names": {"series": "Show", "anilist": ["Show!"]}, '
            '"preowned_episode_ids": [101], "slice_coverage": "S1 E1-2"}'
        )
        assert TitleClaim.from_json(claim.to_json(), guards={}) == claim

    def test_from_json_skips_a_stored_claimed_at(self) -> None:
        claim = entry_claim()

        assert TitleClaim.from_json({**claim.to_json(), "claimed_at": "2026-06-24 00:00:00"}, guards={}) == claim

    def test_a_claims_window_is_its_entry_series_ids_and_names(self) -> None:
        names = TitleNames("Show", ("Show!",))
        claim = entry_claim(al_id=3, series_id=8, ordered_episode_ids=(102, 101), names=names)

        assert claim.window == entry_window(3, (102, 101), series_id=8, names=names)

    def test_claim_for_is_the_first_claim_on_the_series(self) -> None:
        first, second = entry_claim(al_id=1, series_id=8), entry_claim(al_id=2, series_id=8)
        pending = pending_import(claims=(first, second))

        assert pending.claim_for(8) is first
        with pytest.raises(LookupError, match="series 9"):
            pending.claim_for(9)

    def test_guards_for_merges_the_series_claims_evidence(self) -> None:
        # Groups union in claim order, owned episodes concatenate, and the owned-size read is
        # last-wins per episode. Another series' claim contributes nothing.
        pending = pending_import(
            claims=(
                entry_claim(
                    al_id=1,
                    guards=GuardFacts(
                        entry_groups=("A", "B"), stale_groups=("S",), owned_episodes=(OwnedEpisode(11, 700),)
                    ),
                ),
                entry_claim(
                    al_id=2,
                    series_id=8,
                    guards=GuardFacts(
                        entry_groups=("Z",), stale_groups=("Y",), owned_episodes=(OwnedEpisode(11, 999),)
                    ),
                ),
                entry_claim(
                    al_id=3,
                    guards=GuardFacts(
                        entry_groups=("B", "C"),
                        stale_groups=("S", "T"),
                        owned_episodes=(OwnedEpisode(11, 710), OwnedEpisode(12, 720)),
                    ),
                ),
            ),
        )

        merged = pending.guards_for(7)

        assert merged == GuardFacts(
            entry_groups=("A", "B", "C"),
            stale_groups=("S", "T"),
            owned_episodes=(OwnedEpisode(11, 700), OwnedEpisode(11, 710), OwnedEpisode(12, 720)),
        )
        assert merged.owned_sizes == {11: 710, 12: 720}
        assert pending.guards_for(9) == GuardFacts()

    def test_resolved_ids_union_the_claims_windows_in_claim_order(self) -> None:
        pending = pending_import(
            file_episode_map={"a.mkv": [9]},
            claims=(
                entry_claim(al_id=1, ordered_episode_ids=[3, 1]),
                entry_claim(al_id=2, series_id=8, ordered_episode_ids=[1, 2]),
            ),
        )

        assert pending.resolved_ids() == [3, 1, 2]

    def test_resolved_ids_fall_back_to_the_seeds_sorted_when_every_claim_is_unscoped(self) -> None:
        # The map's ids alone, a stray zero dropped: the specials shape with no window to scope against.
        pending = pending_import(
            file_episode_map={"a.mkv": [7, 0], "b.mkv": [5]},
            claims=(entry_claim(al_id=1), entry_claim(al_id=2, series_id=8)),
        )

        assert pending.resolved_ids() == [5, 7]
        assert pending_import(file_episode_map={}).resolved_ids() == []

    def test_with_claim_appends_a_new_entry(self) -> None:
        pending = pending_import()
        added = entry_claim(al_id=2, series_id=8)

        assert pending.with_claim(added).claims == (*pending.claims, added)
        assert len(pending.claims) == 1

    def test_with_claim_replaces_the_entrys_claim_in_place_keeping_its_preowned_ids(self) -> None:
        # A re-flag refreshes the window, but the net-out stays the FIRST claim's.
        stored = entry_claim(al_id=1, ordered_episode_ids=[101], preowned_episode_ids=[101])
        other = entry_claim(al_id=2, series_id=8)
        pending = pending_import(claims=(stored, other))
        fresh = entry_claim(al_id=1, ordered_episode_ids=[101, 102], preowned_episode_ids=[102])

        refreshed = pending.with_claim(fresh)

        assert refreshed.claims == (replace(fresh, preowned_episode_ids=(101,)), other)
        assert pending.claims == (stored, other)

    def test_with_claim_clears_the_cleanup_flag(self) -> None:
        # A claim joining (or re-flagging) means new files to import: the record is active again, so
        # its owed effects re-run at the eventual retire instead of the heal pass retiring it.
        flagged = pending_import(awaiting_cleanup=True)

        assert flagged.with_claim(entry_claim(al_id=2, series_id=8)).awaiting_cleanup is False
        assert flagged.with_claim(entry_claim(ordered_episode_ids=[101])).awaiting_cleanup is False


class TestDisplayLabel:
    """`display_label`: every claim's title, the group, then the distinct slices. The infohash names a titleless record."""

    def test_one_claim_is_title_group_slice(self) -> None:
        # The group tells apart a series that grabbed several torrents, the slice one group's
        # per-episode siblings.
        pending = pending_import(release_group="Era-Raws", slice_coverage="S02 E06")

        assert pending.display_label == f"Show{SEP}Era-Raws{SEP}S02 E06"

    def test_two_claims_join_their_titles_and_slices(self) -> None:
        pending = pending_import(
            release_group="Era-Raws",
            claims=(
                entry_claim(al_id=1, title="A", slice_coverage="S01 E01"),
                entry_claim(al_id=2, series_id=8, title="B", slice_coverage="S02 E01"),
            ),
        )

        assert pending.display_label == f"A & B{SEP}Era-Raws{SEP}S01 E01, S02 E01"

    def test_a_repeated_title_or_slice_shows_once(self) -> None:
        pending = pending_import(
            release_group="",
            claims=(
                entry_claim(al_id=1, title="A", slice_coverage="S01 E01"),
                entry_claim(al_id=2, title="A", slice_coverage="S01 E01"),
            ),
        )

        assert pending.display_label == f"A{SEP}S01 E01"

    def test_a_groupless_record_shows_the_bare_title(self) -> None:
        assert pending_import(release_group="").display_label == "Show"

    def test_a_titleless_record_falls_back_to_its_infohash(self) -> None:
        assert pending_import(infohash="h", release_group="", title=None).display_label == "h"
        assert pending_import(infohash="h", release_group="Era-Raws", title=None).display_label == f"h{SEP}Era-Raws"


class TestEntryWindow:
    """A window built from Sonarr's episodes."""

    def test_a_window_takes_each_episode_once_in_sonarrs_order(self) -> None:
        episodes = [sonarr_ep(1, 3, ep_id=103), sonarr_ep(1, 1, ep_id=101), sonarr_ep(1, 3, ep_id=103)]

        window = TitleWindow.of(al_id=5, series_id=7, ep_list=episodes, names=TitleNames("Show"))

        assert window == entry_window(5, (103, 101), series_id=7, names=TitleNames("Show"))


def _guarded_claim() -> TitleClaim:
    """A scoped claim on series 55 carrying every serialized field plus a guard row."""

    return entry_claim(
        al_id=990,
        series_id=55,
        title="Some Show",
        coverage="S02 E01-E12",
        url="https://releases.moe/1",
        ordered_episode_ids=[11, 12],
        names=TitleNames("Some Show", ("Some Show", "Aru Show")),
        preowned_episode_ids=[11],
        slice_coverage="S02 E01-E02",
        guards=GuardFacts(entry_groups=("Era-Raws", "OtherPick"), owned_episodes=(OwnedEpisode(11, 700),)),
    )


class TestPendingImportRoundTrip:
    """`PendingImport`'s JSON round trip: the claims ride the blob, the guards their own row, unknown keys are ignored."""

    def test_to_json_from_json_round_trip(self) -> None:
        claim = _guarded_claim()
        pending = PendingImport(
            infohash="abc123",
            release_group="Era-Raws",
            is_dual_audio=True,
            seadex_files=("ep1.mkv", "ep2.mkv"),
            added_at="2026-06-24 12:00:00",
            file_episode_map={"ep1.mkv": (11,), "ep2.mkv": (12,)},
            claims=(claim,),
            excluded_files=("other-slice.mkv",),
            release_sizes=(700, 710),
            awaiting_cleanup=True,
            sizes_by_name={"ep1.mkv": 700, "ep2.mkv": 710},
            hand_import_files=("ep3.mkv",),
            order_only={"ep3.mkv": _HALF_KEYS[0]},
            cross_seed=True,
        )

        raw = pending.to_json()

        # Guard evidence is entry-level (its own guard_facts row): neither the record nor a claim
        # carries a copy, so a bare rehydrate comes back guard-empty.
        assert "guards" not in raw
        claims_raw: list[dict[str, object]] = raw["claims"]
        assert all("guards" not in claim_raw for claim_raw in claims_raw)
        bare = PendingImport.from_json(raw, guards={})
        assert bare == replace(pending, claims=(replace(claim, guards=GuardFacts()),))
        # The caller-supplied rows (the read seams' join) hydrate the claim back whole.
        assert PendingImport.from_json(raw, guards={claim.al_id: claim.guards}) == pending

    def test_from_json_rehydrates_each_claims_guards_by_al_id(self) -> None:
        first, second = GuardFacts(entry_groups=("A",)), GuardFacts(stale_groups=("B",))
        pending = pending_import(
            claims=(entry_claim(al_id=1, guards=first), entry_claim(al_id=2, series_id=8, guards=second)),
        )
        rows = {2: second, 1: first, 3: GuardFacts(entry_groups=("Unclaimed",))}

        rebuilt = PendingImport.from_json(pending.to_json(), guards=rows)

        assert [claim.guards for claim in rebuilt.claims] == [first, second]
        assert rebuilt == pending

    def test_healed_map_round_trips_and_flips_coverage(self) -> None:
        # The placements live in the same blob field as the seed, so a healed
        # record rehydrates mapped on the next run.
        pending = pending_import(file_episode_map={}, seadex_files=["Show - 01 [1080p].mkv"], ordered_episode_ids=[101])

        healed = pending.with_placements({"show - 01 [1080p].mkv": (101,)})

        assert PendingImport.from_json(healed.to_json(), guards={}) == healed
        assert (pending.seed_coverage().mapped, healed.seed_coverage().mapped) == (False, True)

    def test_from_json_ignores_a_blob_guards_key(self) -> None:
        # A frozen grab-time copy in the blob (the divergence the guard_facts row exists to kill)
        # must never resurrect, on the record or inside a claim.
        stale = {"entry_groups": ["Stale"]}
        raw = {"infohash": "h", "guards": stale, "claims": [{"al_id": 1, "series_id": 1, "guards": stale}]}

        assert PendingImport.from_json(raw, guards={}).claims[0].guards == GuardFacts()

    def test_from_json_tolerates_missing_keys(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h"}, guards={})
        assert rebuilt.infohash == "h"
        assert dict(rebuilt.file_episode_map) == {}
        assert rebuilt.claims == ()
        assert rebuilt.added_at == ""
        # Pre-excluded_files records rehydrate empty (completeness stays
        # conservative for them).
        assert rebuilt.excluded_files == ()
        assert rebuilt.release_sizes == ()
        # A record predating the cleanup flag owes nothing.
        assert rebuilt.awaiting_cleanup is False
        assert rebuilt.hand_import_files == ()
        assert rebuilt.order_only == {}

    def test_a_row_without_hand_import_files_leaves_its_unplaced_listed_files_for_a_hand_import(self) -> None:
        record = pending_import(
            file_episode_map={"show - 01 [1080p].mkv": [101]},
            excluded_files=("Show - 03 [1080p].mkv",),
            seadex_files=("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv", "Show - 03 [1080p].mkv"),
            hand_import_files=("show - 02 [1080p].mkv",),
        )
        raw = record.to_json()
        del raw["hand_import_files"]
        raw["identified"] = dict[str, int]()

        rebuilt = PendingImport.from_json(raw, guards={})

        assert rebuilt.hand_import_files == ("show - 02 [1080p].mkv",)
        assert rebuilt.excluded_files == ("show - 03 [1080p].mkv",)
        assert dict(rebuilt.file_episode_map) == {"show - 01 [1080p].mkv": (101,)}

    def test_a_row_with_an_empty_hand_import_files_key_reads_none(self) -> None:
        raw = pending_import(seadex_files=("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv")).to_json()
        raw["hand_import_files"] = list[str]()

        assert PendingImport.from_json(raw, guards={}).hand_import_files == ()

    def test_a_malformed_order_only_entry_reads_as_absent(self) -> None:
        raw = pending_import(hand_import_files=HALVES, order_only=dict(zip(HALVES, _HALF_KEYS, strict=True))).to_json()
        raw["order_only"][HALVES[1]] = {"order": "dvd", "season": 7}

        assert PendingImport.from_json(raw, guards={}).order_only == {HALVES[0]: _HALF_KEYS[0]}

    def test_a_record_predating_sizes_by_name_lists_no_sizes(self) -> None:
        # Its own group is then judged by listed size alone.
        rebuilt = PendingImport.from_json({"infohash": "h", "file_episode_map": {"a.mkv": [1]}}, guards={})

        assert rebuilt.sizes_by_name == {}

    def test_a_claim_tolerates_missing_keys(self) -> None:
        # A claim missing every optional key rehydrates unscoped, unnamed, guard-empty,
        # with coverage and url None, and a missing al_id lands under the 0 sentinel.
        rebuilt = PendingImport.from_json({"infohash": "h", "claims": [{"series_id": 1}]}, guards={})

        assert rebuilt.claims == (
            TitleClaim(
                al_id=0,
                series_id=1,
                title=None,
                coverage=None,
                url=None,
                ordered_episode_ids=(),
                names=TitleNames(),
                preowned_episode_ids=(),
                slice_coverage=None,
            ),
        )

    def test_old_record_with_unknown_keys_rehydrates(self) -> None:
        # Back-compat: a record persisted with since-removed keys still loads
        # (from_json reads only the known keys and ignores the rest).
        raw = {
            "infohash": "h",
            "series_id": 1,
            "episode_ids": [1],
            "seadex_sizes": [1000],
            "title": "T",
            "claims": [{"al_id": 1, "series_id": 1, "key": "h"}],
        }

        rebuilt = PendingImport.from_json(raw, guards={})

        assert rebuilt.infohash == "h"
        assert claim_al_ids(rebuilt) == (1,)

    def test_added_at_of_reads_the_raw_row(self) -> None:
        # The keys-only readers age a row without rehydrating it: a missing or non-string stamp is unset.
        assert added_at_of({"added_at": "2026-06-24 00:00:00"}) == "2026-06-24 00:00:00"
        assert added_at_of({}) == ""
        assert added_at_of({"added_at": 20260624}) == ""
