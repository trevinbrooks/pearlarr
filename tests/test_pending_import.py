# pyright: strict
"""Direct tests for the pending import: building it, its placements, claims and windows, and its JSON form."""

import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from pearlarr.file_paths import normalize_file_name
from pearlarr.pending_import import (
    NO_ORDER_ONLY,
    OwnedAtGrab,
    PendingImport,
    SizeMatchedEpisode,
    TitleClaim,
    TitleNames,
    TitleWindow,
    added_at_of,
    joined_file_map,
    sizes_by_episode,
)
from pearlarr.releases import EpisodeKey, OrderOnlyKey

from .builders import SEP, claim_anilist_ids, pending_import, sonarr_ep, title_claim, title_window

HALVES = tuple(normalize_file_name(f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv") for n in (1, 2))
"""The two halves of a split opener, which only the DVD order numbers."""

_HALF_KEYS = (OrderOnlyKey("dvd", EpisodeKey(7, 1)), OrderOnlyKey("dvd", EpisodeKey(7, 2)))
"""The halves' DVD-order keys."""


class TestPendingImportBuild:
    """Built from whatever the caller passed, the pending import normalizes its file map and exclusions, read-only."""

    def test_the_map_normalizes_its_keys_and_drops_zero_ids(self) -> None:
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

    def test_the_exclusions_normalize_and_dedupe_in_order(self) -> None:
        pending = pending_import(excluded_files=["SHOW - 03.MKV", " Show - 02.mkv ", "show - 03.mkv"])

        assert pending.excluded_files == ("show - 03.mkv", "show - 02.mkv")

    def test_the_sizes_normalize_their_keys_and_drop_a_name_whose_spellings_disagree(self) -> None:
        pending = pending_import(
            sizes_by_name={"Sub/Show - 01.MKV": 100, "Show - 02.mkv": 200, "show - 02.mkv": 201},
        )

        assert pending.sizes_by_name == {"show - 01.mkv": 100}

    def test_the_build_ranks_the_map_over_the_hand_import_files_over_the_exclusions(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01.mkv": [101]},
            excluded_files=["show - 01.mkv", "SHOW - 02.MKV", "show - 03.mkv", "show - 05.mkv"],
            hand_import_files=["SHOW - 04.MKV", "Show - 01.mkv", "show - 02.mkv", "Sub/Show - 03.mkv", "show - 04.mkv"],
        )

        assert pending.hand_import_files == ("show - 04.mkv", "show - 02.mkv", "show - 03.mkv")
        assert pending.excluded_files == ("show - 05.mkv",)

    def test_the_order_only_keys_reduce_to_the_hand_import_files(self) -> None:
        pending = pending_import(
            file_episode_map={HALVES[0]: [401]},
            hand_import_files=list(HALVES),
            order_only={HALVES[0]: _HALF_KEYS[0], HALVES[1].upper(): _HALF_KEYS[1]},
        )

        assert pending.order_only == {HALVES[1]: _HALF_KEYS[1]}

    def test_the_map_is_detached_and_read_only(self) -> None:
        # The pending import copies the caller's file map at construction and wraps it read-only, so neither the
        # caller's dict nor a holder of the pending import can mutate the file map behind it. Built through
        # `replace`: the builder copies the file map itself, which would pass this pin on its own.
        file_map: dict[str, tuple[int, ...]] = {"show - 01.mkv": (101,)}
        pending = replace(pending_import(), file_episode_map=file_map)
        file_map["show - 02.mkv"] = (102,)
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
"""Changes that leave a pending import equal: it has `show - 01.mkv` on 101, excludes `show - 02.mkv`, and leaves
`show - 03.mkv` for a hand import."""


class TestPendingImportPlacements:
    """`with_placements` and `with_exclusions` grow the pending import, `unplaced_names` is what a rebuild may place.

    Every `with_*` hands the pending import back on a no-op.
    """

    def test_with_exclusions_appends_after_the_stored_ones(self) -> None:
        # The recorded order is the order of arrival: what the pending import already had, then this poll's additions.
        pending = pending_import(excluded_files=["show - 01.mkv"])

        healed = pending.with_exclusions(["show - 03.mkv", "show - 02.mkv"])

        assert healed.excluded_files == ("show - 01.mkv", "show - 03.mkv", "show - 02.mkv")

    @pytest.mark.parametrize("change", _NO_OP_CHANGES.values(), ids=list(_NO_OP_CHANGES))
    def test_a_change_that_leaves_the_pending_import_equal_returns_it(
        self, change: Callable[[PendingImport], PendingImport]
    ) -> None:
        # One rule for every `with_*`: the pending import store gates its save on it.
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
    def test_file_map_coverage_accounted_matches_unplaced_names(
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
        assert pending.file_map_coverage().accounted == (not pending.unplaced_names())


class TestSizesByEpisode:
    """`sizes_by_episode`: each file map episode id to its file's listed size."""

    def test_sizes_each_episode_a_listed_file_is_mapped_to(self) -> None:
        sizes = sizes_by_episode({"a.mkv": (1,), "b.mkv": (2, 3), "c.mkv": (4,)}, {"a.mkv": 100, "b.mkv": 200})

        assert sizes == {1: 100, 2: 200, 3: 200}

    def test_an_id_two_files_size_differently_is_dropped(self) -> None:
        assert sizes_by_episode({"a.mkv": (1, 2), "b.mkv": (2,)}, {"a.mkv": 100, "b.mkv": 200}) == {1: 100}


class TestJoinedMap:
    """`joined_file_map`: a base file map, then the added file map's new names."""

    def test_adds_the_new_names_after_the_base_entries(self) -> None:
        joined = joined_file_map({"b.mkv": (2,), "a.mkv": (1,)}, {"c.mkv": (3,)})

        assert list(joined.items()) == [("b.mkv", (2,)), ("a.mkv", (1,)), ("c.mkv", (3,))]

    def test_a_shared_name_keeps_the_base_ids(self) -> None:
        assert joined_file_map({"a.mkv": (1,)}, {"a.mkv": (9,), "b.mkv": (2,)}) == {"a.mkv": (1,), "b.mkv": (2,)}


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

    def test_with_sizes_by_name_fills_an_empty_pending_import(self) -> None:
        assert pending_import().with_sizes_by_name({"a.mkv": 100}).sizes_by_name == {"a.mkv": 100}

    def test_with_sizes_by_name_keeps_a_recorded_map(self) -> None:
        pending = pending_import(sizes_by_name={"a.mkv": 100})

        assert pending.with_sizes_by_name({"a.mkv": 1}) is pending


class TestPendingImportClaims:
    """The claim reads over a pending import: every title's series, window, guards, and ids, in claim order."""

    def test_series_ids_are_distinct_in_claim_order(self) -> None:
        pending = pending_import(
            claims=(
                title_claim(anilist_id=3, series_id=8),
                title_claim(anilist_id=1, series_id=7),
                title_claim(anilist_id=2, series_id=8),
            ),
        )

        assert pending.series_ids == (8, 7)

    def test_a_claim_persists_as_the_spelled_out_schema(self) -> None:
        # The persisted shape is pinned: a new field never enters a row without a migration.
        claim = title_claim(
            window_coverage="S1",
            url="https://example.invalid/1",
            ordered_episode_ids=(101, 102),
            names=TitleNames("Show", ("Show!",)),
            preowned_episode_ids=(101,),
            claimed_coverage="S1 E1-2",
        )

        assert json.dumps(claim.to_json()) == (
            '{"anilist_id": 1, "series_id": 7, "title": "Show", "coverage": "S1", "url": "https://example.invalid/1", '
            '"ordered_episode_ids": [101, 102], "names": {"series": "Show", "anilist": ["Show!"]}, '
            '"preowned_episode_ids": [101], "slice_coverage": "S1 E1-2"}'
        )
        assert TitleClaim.from_json(claim.to_json(), guards={}) == claim

    def test_from_json_skips_a_stored_claimed_at(self) -> None:
        claim = title_claim()

        assert TitleClaim.from_json({**claim.to_json(), "claimed_at": "2026-06-24 00:00:00"}, guards={}) == claim

    def test_a_claims_window_is_its_title_series_ids_and_names(self) -> None:
        names = TitleNames("Show", ("Show!",))
        claim = title_claim(anilist_id=3, series_id=8, ordered_episode_ids=(102, 101), names=names)

        assert claim.window == title_window(3, (102, 101), series_id=8, names=names)

    def test_claim_for_is_the_first_claim_on_the_series(self) -> None:
        first, second = title_claim(anilist_id=1, series_id=8), title_claim(anilist_id=2, series_id=8)
        pending = pending_import(claims=(first, second))

        assert pending.claim_for(8) is first
        with pytest.raises(LookupError, match="series 9"):
            pending.claim_for(9)

    def test_guards_for_merges_the_series_claims_evidence(self) -> None:
        # Groups union in claim order, owned episodes concatenate, and the owned-size read is
        # last-wins per episode. Another series' claim contributes nothing.
        pending = pending_import(
            claims=(
                title_claim(
                    anilist_id=1,
                    guards=OwnedAtGrab(
                        entry_groups=("A", "B"), stale_groups=("S",), owned_episodes=(SizeMatchedEpisode(11, 700),)
                    ),
                ),
                title_claim(
                    anilist_id=2,
                    series_id=8,
                    guards=OwnedAtGrab(
                        entry_groups=("Z",), stale_groups=("Y",), owned_episodes=(SizeMatchedEpisode(11, 999),)
                    ),
                ),
                title_claim(
                    anilist_id=3,
                    guards=OwnedAtGrab(
                        entry_groups=("B", "C"),
                        stale_groups=("S", "T"),
                        owned_episodes=(SizeMatchedEpisode(11, 710), SizeMatchedEpisode(12, 720)),
                    ),
                ),
            ),
        )

        merged = pending.guards_for(7)

        assert merged == OwnedAtGrab(
            entry_groups=("A", "B", "C"),
            stale_groups=("S", "T"),
            owned_episodes=(SizeMatchedEpisode(11, 700), SizeMatchedEpisode(11, 710), SizeMatchedEpisode(12, 720)),
        )
        assert merged.owned_sizes == {11: 710, 12: 720}
        assert pending.guards_for(9) == OwnedAtGrab()

    def test_resolved_ids_union_the_claims_windows_in_claim_order(self) -> None:
        pending = pending_import(
            file_episode_map={"a.mkv": [9]},
            claims=(
                title_claim(anilist_id=1, ordered_episode_ids=[3, 1]),
                title_claim(anilist_id=2, series_id=8, ordered_episode_ids=[1, 2]),
            ),
        )

        assert pending.window_episode_ids() == [3, 1, 2]

    def test_window_episode_ids_fall_back_to_the_file_map_episodes_sorted_when_every_claim_has_no_window(self) -> None:
        # The file map's ids alone, a zero id dropped: the specials shape with no window to check against.
        pending = pending_import(
            file_episode_map={"a.mkv": [7, 0], "b.mkv": [5]},
            claims=(title_claim(anilist_id=1), title_claim(anilist_id=2, series_id=8)),
        )

        assert pending.window_episode_ids() == [5, 7]
        assert pending_import(file_episode_map={}).window_episode_ids() == []

    def test_with_claim_appends_a_new_claim(self) -> None:
        pending = pending_import()
        added = title_claim(anilist_id=2, series_id=8)

        assert pending.with_claim(added).claims == (*pending.claims, added)
        assert len(pending.claims) == 1

    def test_with_claim_replaces_the_titles_claim_in_place_keeping_its_preowned_ids(self) -> None:
        # A second grab plan refreshes the window, but the ids the wait bar leaves out stay the FIRST claim's.
        stored = title_claim(anilist_id=1, ordered_episode_ids=[101], preowned_episode_ids=[101])
        other = title_claim(anilist_id=2, series_id=8)
        pending = pending_import(claims=(stored, other))
        fresh = title_claim(anilist_id=1, ordered_episode_ids=[101, 102], preowned_episode_ids=[102])

        refreshed = pending.with_claim(fresh)

        assert refreshed.claims == (replace(fresh, preowned_episode_ids=(101,)), other)
        assert pending.claims == (stored, other)

    def test_with_claim_clears_awaiting_cleanup(self) -> None:
        # A claim joining (or planned again) means new files to import: the pending import is active again, so
        # its owed effects re-run at the eventual cleanup instead of the pass that finishes cleanups cleaning it up.
        flagged = pending_import(awaiting_cleanup=True)

        assert flagged.with_claim(title_claim(anilist_id=2, series_id=8)).awaiting_cleanup is False
        assert flagged.with_claim(title_claim(ordered_episode_ids=[101])).awaiting_cleanup is False


class TestDisplayLabel:
    """`display_label`: each claim's name, the group, then distinct claimed coverages. The hash names a nameless one."""

    def test_one_claim_is_title_group_coverage(self) -> None:
        # The group tells apart a series that grabbed several torrents, the claimed coverage one group's
        # per-episode siblings.
        pending = pending_import(release_group="Era-Raws", claimed_coverage="S02 E06")

        assert pending.display_label == f"Show{SEP}Era-Raws{SEP}S02 E06"

    def test_two_claims_join_their_titles_and_slices(self) -> None:
        pending = pending_import(
            release_group="Era-Raws",
            claims=(
                title_claim(anilist_id=1, title="A", claimed_coverage="S01 E01"),
                title_claim(anilist_id=2, series_id=8, title="B", claimed_coverage="S02 E01"),
            ),
        )

        assert pending.display_label == f"A & B{SEP}Era-Raws{SEP}S01 E01, S02 E01"

    def test_a_repeated_title_or_coverage_shows_once(self) -> None:
        pending = pending_import(
            release_group="",
            claims=(
                title_claim(anilist_id=1, title="A", claimed_coverage="S01 E01"),
                title_claim(anilist_id=2, title="A", claimed_coverage="S01 E01"),
            ),
        )

        assert pending.display_label == f"A{SEP}S01 E01"

    def test_a_groupless_pending_import_shows_the_bare_title(self) -> None:
        assert pending_import(release_group="").display_label == "Show"

    def test_a_titleless_pending_import_falls_back_to_its_infohash(self) -> None:
        assert pending_import(infohash="h", release_group="", title=None).display_label == "h"
        assert pending_import(infohash="h", release_group="Era-Raws", title=None).display_label == f"h{SEP}Era-Raws"


class TestTitleWindow:
    """A window built from Sonarr's episodes."""

    def test_a_window_takes_each_episode_once_in_sonarrs_order(self) -> None:
        episodes = [sonarr_ep(1, 3, ep_id=103), sonarr_ep(1, 1, ep_id=101), sonarr_ep(1, 3, ep_id=103)]

        window = TitleWindow.of(anilist_id=5, series_id=7, episodes=episodes, names=TitleNames("Show"))

        assert window == title_window(5, (103, 101), series_id=7, names=TitleNames("Show"))


def _guarded_claim() -> TitleClaim:
    """A claim with a window on series 55 carrying every serialized field plus a guard row."""

    return title_claim(
        anilist_id=990,
        series_id=55,
        title="Some Show",
        window_coverage="S02 E01-E12",
        url="https://releases.moe/1",
        ordered_episode_ids=[11, 12],
        names=TitleNames("Some Show", ("Some Show", "Aru Show")),
        preowned_episode_ids=[11],
        claimed_coverage="S02 E01-E02",
        guards=OwnedAtGrab(entry_groups=("Era-Raws", "OtherPick"), owned_episodes=(SizeMatchedEpisode(11, 700),)),
    )


class TestPendingImportRoundTrip:
    """`PendingImport`'s JSON round trip: claims go in the blob, guards in a separate row, unknown keys are ignored."""

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
            is_cross_seed=True,
        )

        raw = pending.to_json()

        # Guard evidence is title-level (a separate guard_facts row): neither the pending import nor a claim
        # carries a copy, so a plain load comes back guard-empty.
        assert "guards" not in raw
        claims_raw: list[dict[str, object]] = raw["claims"]
        assert all("guards" not in claim_raw for claim_raw in claims_raw)
        bare = PendingImport.from_json(raw, guards={})
        assert bare == replace(pending, claims=(replace(claim, guards=OwnedAtGrab()),))
        # The caller-supplied rows (the store reads' join) load the claim back whole.
        assert PendingImport.from_json(raw, guards={claim.anilist_id: claim.guards}) == pending

    def test_from_json_loads_each_claims_guards_by_anilist_id(self) -> None:
        first, second = OwnedAtGrab(entry_groups=("A",)), OwnedAtGrab(stale_groups=("B",))
        pending = pending_import(
            claims=(title_claim(anilist_id=1, guards=first), title_claim(anilist_id=2, series_id=8, guards=second)),
        )
        rows = {2: second, 1: first, 3: OwnedAtGrab(entry_groups=("Unclaimed",))}

        rebuilt = PendingImport.from_json(pending.to_json(), guards=rows)

        assert [claim.guards for claim in rebuilt.claims] == [first, second]
        assert rebuilt == pending

    def test_a_map_with_placements_round_trips_and_flips_coverage(self) -> None:
        # The placements live in the same blob field as the grab's, so an updated
        # pending import loads with them placed on the next run.
        pending = pending_import(file_episode_map={}, seadex_files=["Show - 01 [1080p].mkv"], ordered_episode_ids=[101])

        healed = pending.with_placements({"show - 01 [1080p].mkv": (101,)})

        assert PendingImport.from_json(healed.to_json(), guards={}) == healed
        assert (pending.file_map_coverage().file_map, healed.file_map_coverage().file_map) == (False, True)

    def test_from_json_ignores_a_blob_guards_key(self) -> None:
        # A grab-time copy saved in the blob (the divergence the guard_facts row exists to kill)
        # must never resurrect, on the pending import or inside a claim.
        stale = {"entry_groups": ["Stale"]}
        raw = {"infohash": "h", "guards": stale, "claims": [{"anilist_id": 1, "series_id": 1, "guards": stale}]}

        assert PendingImport.from_json(raw, guards={}).claims[0].guards == OwnedAtGrab()

    def test_from_json_tolerates_missing_keys(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h"}, guards={})
        assert rebuilt.infohash == "h"
        assert dict(rebuilt.file_episode_map) == {}
        assert rebuilt.claims == ()
        assert rebuilt.added_at == ""
        # Pending imports from before excluded_files load empty (completeness stays
        # conservative for them).
        assert rebuilt.excluded_files == ()
        assert rebuilt.release_sizes == ()
        # A pending import predating `awaiting_cleanup` owes nothing.
        assert rebuilt.awaiting_cleanup is False
        assert rebuilt.hand_import_files == ()
        assert rebuilt.order_only == {}

    def test_a_row_without_hand_import_files_leaves_its_unplaced_listed_files_for_a_hand_import(self) -> None:
        pending = pending_import(
            file_episode_map={"show - 01 [1080p].mkv": [101]},
            excluded_files=("Show - 03 [1080p].mkv",),
            seadex_files=("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv", "Show - 03 [1080p].mkv"),
            hand_import_files=("show - 02 [1080p].mkv",),
        )
        raw = pending.to_json()
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

    def test_a_pending_import_predating_sizes_by_name_lists_no_sizes(self) -> None:
        # This torrent's group is then checked by listed size alone.
        rebuilt = PendingImport.from_json({"infohash": "h", "file_episode_map": {"a.mkv": [1]}}, guards={})

        assert rebuilt.sizes_by_name == {}

    def test_a_claim_tolerates_missing_keys(self) -> None:
        # A claim lacking every optional key loads with no window, unnamed, guard-empty,
        # with coverage and url None, and an absent anilist_id goes under the 0 sentinel.
        rebuilt = PendingImport.from_json({"infohash": "h", "claims": [{"series_id": 1}]}, guards={})

        assert rebuilt.claims == (
            TitleClaim(
                anilist_id=0,
                series_id=1,
                title=None,
                window_coverage=None,
                url=None,
                ordered_episode_ids=(),
                names=TitleNames(),
                preowned_episode_ids=(),
                claimed_coverage=None,
            ),
        )

    def test_old_pending_import_with_unknown_keys_loads(self) -> None:
        # Back-compat: a pending import persisted with since-removed keys still loads
        # (from_json reads only the known keys and ignores the rest).
        raw = {
            "infohash": "h",
            "series_id": 1,
            "episode_ids": [1],
            "seadex_sizes": [1000],
            "title": "T",
            "claims": [{"anilist_id": 1, "series_id": 1, "key": "h"}],
        }

        rebuilt = PendingImport.from_json(raw, guards={})

        assert rebuilt.infohash == "h"
        assert claim_anilist_ids(rebuilt) == (1,)

    def test_added_at_of_reads_the_raw_row(self) -> None:
        # The keys-only readers age a row without loading it: an absent or non-string stamp is unset.
        assert added_at_of({"added_at": "2026-06-24 00:00:00"}) == "2026-06-24 00:00:00"
        assert added_at_of({}) == ""
        assert added_at_of({"added_at": 20260624}) == ""
