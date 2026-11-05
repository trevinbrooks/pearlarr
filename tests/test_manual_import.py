# pyright: strict
"""The pure manual-import vocabulary: normalizers, `PendingImport`, the wait states, telemetry, path translation.

The planning modules' tests sit beside this file, one per module: `test_placement_types`, `test_placer`,
`test_episode_state`, `test_import_files`, `test_probe_verdicts`, `test_import_quality`.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from typing import cast

import pytest

from pearlarr.manual_import import (
    EMPTY_LISTING,
    LEAVE_PROBE,
    NO_IDENTIFIED,
    Deferral,
    EntryClaim,
    EntryNames,
    EntryWindow,
    GuardFacts,
    ImportProbe,
    OwnedEpisode,
    PendingImport,
    PendingState,
    TorrentListing,
    TorrentTelemetry,
    WaitOutcome,
    added_at_of,
    classify_pending,
    joined_map,
    newest_claimed_at_of,
    normalize_basename,
    normalize_group,
    normalize_rg,
    normalized_leaf,
    path_leaf,
    sanitize_torrent_telemetry,
    sizes_by_episode,
    translate_download_path,
)
from pearlarr.seadex_types import RemotePathMapping

from .builders import SEP, claim_al_ids, entry_claim, entry_window, pending_import, sonarr_ep, torrent_listing


class TestNormalize:
    """`normalize_basename`/`normalize_group` casefold and strip for stable comparison.

    Interior dashes in a group name are never stripped, only wrapping ones.
    """

    def test_nfc_nfd_match(self) -> None:
        # Same text, NFC (composed) vs NFD (decomposed) "é". Both fold equal.
        nfc = "Café - 01.mkv"
        nfd = "Café - 01.mkv"
        assert normalize_basename(nfc) == normalize_basename(nfd)

    def test_strips_and_casefolds(self) -> None:
        assert normalize_basename("  Show - 01.MKV  ") == "show - 01.mkv"

    def test_folding_a_folded_key_gives_the_same_key(self) -> None:
        # MUTATION PIN: casefold expands ß to "ss" and leaves the mark after it hanging off the second s,
        # uncomposed, until the NFC after the casefold composes it. A stored record folds its keys again on
        # every load, so without that NFC the second pass would re-key it.
        name = "Der Weiß\u0323e Ritter - 01.mkv"
        once = normalize_basename(name)
        assert once == "der weis\u1e63e ritter - 01.mkv"
        assert normalize_basename(once) == once

    def test_canonically_equal_names_fold_alike_when_casefold_turns_a_mark_into_a_letter(self) -> None:
        # MUTATION PIN: casefold turns U+0345 (the Greek iota below) into a letter, so the NFC before it is what
        # makes these two spellings of one name fold alike. With only the NFC after it, they fold apart.
        assert normalize_basename("\u1fb4 - 01.mkv") == normalize_basename("\u03b1\u0345\u0301 - 01.mkv")

    def test_windows_paths_and_trailing_separators_still_fold_to_a_leaf(self) -> None:
        # MUTATION PIN: a Windows arr's paths reach pearlarr verbatim, where posixpath.basename
        # sees no separator at all and would key the whole path.
        assert normalized_leaf("C:\\downloads\\Show\\Show - 01.mkv") == "show - 01.mkv"
        # And a directory entry never folds to the empty leaf, which every other
        # unnamed thing would then collide with.
        assert normalized_leaf("Show/NC/") == "nc"

    def test_path_leaf_folds_separators_but_preserves_the_name(self) -> None:
        # The parser-input twin of normalized_leaf: same separator folding, but
        # case/unicode intact - Sonarr's /parse and the video gate must see the
        # real filename, never a whole Windows path or a folded one.
        assert path_leaf("C:\\downloads\\Show\\Show - 01 [1080p].MKV") == "Show - 01 [1080p].MKV"
        assert path_leaf("Show/NC/") == "NC"

    def test_path_leaf_keeps_a_leaf_that_looks_like_a_drive(self) -> None:
        # MUTATION PIN: catches a path_leaf on os.path only on the Windows CI leg, where os.path is ntpath and
        # reads the leading "K:" as a drive, so the grab's key for the file never matches the one Sonarr's path
        # gives.
        assert path_leaf("K: Return of Kings - 01.mkv") == "K: Return of Kings - 01.mkv"

    def test_group_casefold(self) -> None:
        assert normalize_group("SubGroup") == normalize_group("subgroup")

    def test_group_strip_only_removes_wrapping_dashes(self) -> None:
        # MUTATION PIN: strip("-") widened to a multi-char strip set would eat
        # the X off an X-edged group. Only wrapping dashes (and whitespace) go,
        # and interior dashes always stay.
        assert normalize_group("Xrays-") == "xrays"
        assert normalize_group("X-Raws") == "x-raws"

    def test_group_dash_wrapped_agrees_with_planner(self) -> None:
        # normalize_group is the single source of truth normalize_rg delegates to.
        # A dash-wrapped group must compare equal on both ends or a release the
        # planner grabbed gets re-imported over by the overwrite guard.
        assert normalize_group("-Aergia-") == "aergia"
        assert normalize_group("-Aergia-") == normalize_rg("-Aergia-")
        assert normalize_group("Aergia") == normalize_rg("-Aergia-")


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

    def test_the_by_name_maps_fold_their_keys_and_drop_a_name_whose_spellings_disagree(self) -> None:
        pending = pending_import(
            sizes_by_name={"Sub/Show - 01.MKV": 100, "Show - 02.mkv": 200, "show - 02.mkv": 201},
            identified={"Show - 01.MKV": 101},
        )

        assert pending.sizes_by_name == {"show - 01.mkv": 100}
        assert pending.identified == {"show - 01.mkv": 101}

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
    "no identified names": lambda p: p.with_identified({}),
    "the identified name again": lambda p: p.with_identified({"show - 01.mkv": 101}),
    "the stored claim again": lambda p: p.with_claim(p.claims[0]),
}
"""Changes that leave a record equal: it holds `show - 01.mkv` on 101, identified by size, and excludes
`show - 02.mkv`."""


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
            identified={"show - 01.mkv": 101},
        )

        assert change(pending) is pending

    def test_with_placements_keeps_the_stored_ids_of_a_name_it_holds(self) -> None:
        pending = pending_import(file_episode_map={"show - 01.mkv": [101]})

        healed = pending.with_placements({"show - 01.mkv": (111,), "show - 02.mkv": (102,)})

        assert dict(healed.file_episode_map) == {"show - 01.mkv": (101,), "show - 02.mkv": (102,)}

    def test_with_placements_lifts_a_placed_name_out_of_the_exclusions(self) -> None:
        # A name an earlier poll excluded that a later import placed is a placement, never both.
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", "Show - 02.mkv"],
            excluded_files=["show - 02.mkv", "show - 03.mkv"],
        )

        healed = pending.with_placements({"show - 02.mkv": (102,)})

        assert healed.excluded_files == ("show - 03.mkv",)
        assert dict(healed.file_episode_map) == {"show - 01 [1080p].mkv": (101,), "show - 02.mkv": (102,)}

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


class TestIdentifiedNames:
    """`identified`: files the grab placed by size, merged across grabs, with conflicting names dropped."""

    def test_a_record_predating_them_identifies_nothing(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h", "file_episode_map": {}}, guards={})

        assert rebuilt.identified == NO_IDENTIFIED

    def test_detached_and_read_only(self) -> None:
        names = {"a.mkv": 1}
        identified = cast("dict[str, int]", pending_import(identified=names).identified)
        names["a.mkv"] = 2

        assert identified == {"a.mkv": 1}
        with pytest.raises(TypeError):
            identified["a.mkv"] = 2

    def test_with_identified_merges_the_names(self) -> None:
        joined = pending_import(identified={"a.mkv": 1}).with_identified({"b.mkv": 2})

        assert joined.identified == {"a.mkv": 1, "b.mkv": 2}

    def test_a_name_two_grabs_identify_differently_is_dropped(self) -> None:
        pending = pending_import(identified={"a.mkv": 1, "b.mkv": 2})

        assert pending.with_identified({"a.mkv": 3}).identified == {"b.mkv": 2}


class TestWithListing:
    """`with_listing`: a series' newest read replaces its stored windows, and an empty read keeps them all."""

    _TWO_SERIES = torrent_listing(
        entry_window(1, (101,)), entry_window(2, (102,)), entry_window(3, (801,), series_id=8)
    )
    """Entries 1 and 2 listed on series 7, entry 3 on series 8."""

    def test_a_read_of_another_series_keeps_the_stored_windows(self) -> None:
        pending = pending_import(listing=torrent_listing(entry_window(1, (101,)), entry_window(2, (102,))))

        joined = pending.with_listing(torrent_listing(entry_window(3, (801,), series_id=8)))

        assert joined.listing == self._TWO_SERIES

    def test_a_re_read_of_a_series_replaces_its_windows_and_keeps_the_others(self) -> None:
        pending = pending_import(listing=self._TWO_SERIES)

        joined = pending.with_listing(torrent_listing(entry_window(4, (103,))))

        assert joined.listing == torrent_listing(entry_window(3, (801,), series_id=8), entry_window(4, (103,)))

    def test_the_aliases_join_and_the_new_read_wins_a_number_both_pair(self) -> None:
        pending = pending_import(listing=torrent_listing(entry_window(1, (501,)), aliases={1: 2, 2: 4}))

        joined = pending.with_listing(torrent_listing(entry_window(3, (801,), series_id=8), aliases={2: 6, 3: 8}))

        assert joined.listing.special_aliases == {1: 2, 2: 6, 3: 8}

    def test_an_empty_listing_keeps_the_stored_one(self) -> None:
        pending = pending_import(listing=torrent_listing(entry_window(1, (501,))))

        assert pending.with_listing(EMPTY_LISTING) is pending


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
            names=EntryNames("Show", ("Show!",)),
            preowned_episode_ids=(101,),
            slice_coverage="S1 E1-2",
        )

        assert json.dumps(claim.to_json()) == (
            '{"al_id": 1, "series_id": 7, "title": "Show", "coverage": "S1", "url": "https://example.invalid/1", '
            '"ordered_episode_ids": [101, 102], "names": {"series": "Show", "anilist": ["Show!"]}, '
            '"preowned_episode_ids": [101], "slice_coverage": "S1 E1-2", "claimed_at": "2026-06-24 00:00:00"}'
        )
        assert EntryClaim.from_json(claim.to_json(), guards={}) == claim

    def test_a_claims_window_is_its_entry_series_ids_and_names(self) -> None:
        names = EntryNames("Show", ("Show!",))
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

    def test_with_claim_replaces_the_entrys_claim_in_place_keeping_its_clock_and_preowned_ids(self) -> None:
        # A re-flag refreshes the window, but the clock stays the FIRST claim's (the TTL never
        # restarts on a re-flag) and so does the net-out.
        stored = entry_claim(al_id=1, ordered_episode_ids=[101], preowned_episode_ids=[101])
        other = entry_claim(al_id=2, series_id=8)
        pending = pending_import(claims=(stored, other))
        fresh = entry_claim(
            al_id=1,
            ordered_episode_ids=[101, 102],
            preowned_episode_ids=[102],
            claimed_at="2026-06-25 00:00:00",
        )

        refreshed = pending.with_claim(fresh)

        assert refreshed.claims == (replace(fresh, preowned_episode_ids=(101,), claimed_at=stored.claimed_at), other)
        assert pending.claims == (stored, other)

    def test_with_claim_clears_the_cleanup_flag(self) -> None:
        # A claim joining (or re-flagging) means new files to import: the record is active again, so
        # its owed effects re-run at the eventual retire instead of the heal pass retiring it.
        flagged = pending_import(awaiting_cleanup=True)

        assert flagged.with_claim(entry_claim(al_id=2, series_id=8)).awaiting_cleanup is False
        assert flagged.with_claim(entry_claim(ordered_episode_ids=[101])).awaiting_cleanup is False

    def test_newest_claimed_at_of_reads_the_raw_row(self) -> None:
        # The prune's clock off the stored dict, no rehydration: junk and a missing key are skipped,
        # and a row with no parseable claim stamp reads None.
        raw = pending_import(
            claims=(
                entry_claim(al_id=1, claimed_at="2026-06-24 00:00:00"),
                entry_claim(al_id=2, series_id=8, claimed_at="junk"),
                entry_claim(al_id=3, claimed_at="2026-06-26 00:00:00"),
            ),
        ).to_json()

        assert newest_claimed_at_of(raw) == datetime(2026, 6, 26)
        assert newest_claimed_at_of({"claims": [{"al_id": 1}]}) is None
        assert newest_claimed_at_of({}) is None


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


class TestTorrentListing:
    """The listing's windows and aliases: built from Sonarr's episodes, copied from the caller, and persisted."""

    def test_a_window_takes_each_episode_once_in_sonarrs_order(self) -> None:
        episodes = [sonarr_ep(1, 3, ep_id=103), sonarr_ep(1, 1, ep_id=101), sonarr_ep(1, 3, ep_id=103)]

        window = EntryWindow.of(al_id=5, series_id=7, ep_list=episodes, names=EntryNames("Show"))

        assert window == entry_window(5, (103, 101), series_id=7, names=EntryNames("Show"))

    def test_a_window_persists_as_the_spelled_out_schema(self) -> None:
        window = entry_window(5, (103, 101), series_id=7, names=EntryNames("Show", ("Show!",)))

        text = json.dumps(window.to_json())

        assert text == (
            '{"al_id": 5, "series_id": 7, "ordered_episode_ids": [103, 101], '
            '"names": {"series": "Show", "anilist": ["Show!"]}}'
        )
        assert EntryWindow.from_json(json.loads(text)) == window

    def test_a_listing_persists_its_aliases_as_pairs(self) -> None:
        # A JSON object would hand the aliases' TMDB numbers back as strings, so they ride as pairs.
        listing = torrent_listing(entry_window(1, (501,)), aliases={13: 12})

        text = json.dumps(listing.to_json())

        assert text == (
            '{"entries": [{"al_id": 1, "series_id": 7, "ordered_episode_ids": [501], '
            '"names": {"series": "", "anilist": []}}], "special_aliases": [[13, 12]]}'
        )
        assert TorrentListing.from_json(json.loads(text)) == listing

    def test_two_windows_under_one_anilist_id_keep_the_first(self) -> None:
        first, second, other = entry_window(1, (501,)), entry_window(1, (502,)), entry_window(2, (503,))

        assert torrent_listing(other, first, second).entries == (first, other)

    def test_a_change_to_the_callers_dict_leaves_the_listing_as_it_was(self) -> None:
        aliases = {13: 12}
        listing = torrent_listing(entry_window(1, (501,)), aliases=aliases)

        aliases[14] = 13

        assert dict(listing.special_aliases) == {13: 12}


def _guarded_claim() -> EntryClaim:
    """A scoped claim on series 55 carrying every serialized field plus a guard row."""

    return entry_claim(
        al_id=990,
        series_id=55,
        title="Some Show",
        coverage="S02 E01-E12",
        url="https://releases.moe/1",
        ordered_episode_ids=[11, 12],
        names=EntryNames("Some Show", ("Some Show", "Aru Show")),
        preowned_episode_ids=[11],
        slice_coverage="S02 E01-E02",
        claimed_at="2026-06-24 12:00:00",
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
            identified={"ep2.mkv": 12},
            listing=torrent_listing(claim.window, entry_window(991, (13,), series_id=55), aliases={1: 2}),
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
        # A record predating the listing holds none.
        assert rebuilt.listing == EMPTY_LISTING

    def test_a_record_predating_sizes_by_name_lists_no_sizes(self) -> None:
        # Its own group is then judged by listed size alone.
        rebuilt = PendingImport.from_json({"infohash": "h", "file_episode_map": {"a.mkv": [1]}}, guards={})

        assert rebuilt.sizes_by_name == {}

    def test_a_claim_tolerates_missing_keys(self) -> None:
        # A claim missing every optional key rehydrates unscoped, unnamed, unstamped, guard-empty,
        # with coverage and url None, and a missing al_id lands under the 0 sentinel.
        rebuilt = PendingImport.from_json({"infohash": "h", "claims": [{"series_id": 1}]}, guards={})

        assert rebuilt.claims == (
            EntryClaim(
                al_id=0,
                series_id=1,
                title=None,
                coverage=None,
                url=None,
                ordered_episode_ids=(),
                names=EntryNames(),
                preowned_episode_ids=(),
                slice_coverage=None,
                claimed_at="",
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


class TestPendingStateAndProbe:
    """The shared carried-over status vocabulary + the import probe value object."""

    def test_pending_state_members(self) -> None:
        assert {s.name for s in PendingState} == {
            "QUEUED",
            "DOWNLOADED",
            "IMPORTED",
            "ERRORED",
            "MISSING",
        }

    def test_pending_state_is_its_string(self) -> None:
        assert PendingState.DOWNLOADED == "downloaded"
        assert PendingState.QUEUED == "queued"

    def test_imported_probe_claims_the_files_without_a_command(self) -> None:
        probe = ImportProbe.imported()
        assert probe.files_present is True
        assert probe.command_issued is False

    def test_waiting_probe_carries_the_attempt_flags(self) -> None:
        probe = ImportProbe.waiting(command_issued=True, deferral=Deferral.IMPORT)
        assert probe.files_present is False
        assert probe.command_issued is True
        assert probe.deferral is Deferral.IMPORT
        assert probe.deferred is True
        assert probe.attempted is True

    def test_waiting_probe_defaults_to_no_deferral(self) -> None:
        probe = ImportProbe.waiting()
        assert probe.deferral is Deferral.NONE
        assert probe.deferred is False

    def test_leave_probe_records_no_attempt(self) -> None:
        assert LEAVE_PROBE.attempted is False
        assert LEAVE_PROBE.files_present is False
        assert LEAVE_PROBE.command_issued is False

    def test_a_probes_placements_are_detached_and_read_only(self) -> None:
        placed: dict[str, tuple[int, ...]] = {"show - 01.mkv": (101,)}
        probe = ImportProbe(files_present=False, command_issued=True, placements=placed)
        placed["show - 02.mkv"] = (102,)
        mutable = cast("dict[str, tuple[int, ...]]", probe.placements)

        with pytest.raises(TypeError):
            mutable["show - 02.mkv"] = (102,)
        assert dict(probe.placements) == {"show - 01.mkv": (101,)}


class TestClassifyPending:
    """`classify_pending` folds a poll's outcome + the files-present flag into a state."""

    def test_missing(self) -> None:
        assert classify_pending(WaitOutcome.MISSING, False) is PendingState.MISSING

    def test_errored(self) -> None:
        assert classify_pending(WaitOutcome.ERRORED, False) is PendingState.ERRORED

    def test_still_downloading_is_queued(self) -> None:
        assert classify_pending(None, False) is PendingState.QUEUED

    def test_complete_and_files_present_is_imported(self) -> None:
        assert classify_pending(WaitOutcome.COMPLETE, True) is PendingState.IMPORTED

    def test_complete_without_files_is_importing(self) -> None:
        # The copy is still in flight -> downloaded, never imported, until the
        # files are verified present.
        assert classify_pending(WaitOutcome.COMPLETE, False) is PendingState.DOWNLOADED


class TestSanitizeTorrentTelemetry:
    """MUTATION PIN: pins the pure telemetry sanitizer's clamps and sentinel folds.

    Covers the numeric-string `_as_float` path too - a cluster of ~10 surviving mutants.
    """

    @pytest.mark.parametrize(
        ("progress", "dlspeed", "eta", "completed", "size", "expected"),
        [
            # All-None getattr reads fold to the empty telemetry.
            (None, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # A clean row passes through untouched.
            (0.64, 3_200_000, 130, 1_800, 2_900, TorrentTelemetry(0.64, 3_200_000, 130, 1_800, 2_900)),
            # NaN progress (float and string) folds to 0.0, not a poisoned bar.
            (float("nan"), 100, 130, 50, 200, TorrentTelemetry(0.0, 100, 130, 50, 200)),
            ("nan", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Numeric-string progress parses. Junk folds to 0.0.
            ("0.75", None, None, None, None, TorrentTelemetry(0.75, None, None, None, None)),
            ("fast", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Progress clamps to [0, 1] on both ends.
            (1.5, None, None, None, None, TorrentTelemetry(1.0, None, None, None, None)),
            (-0.25, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Idle (0) and negative speeds read as "no speed", never a 0 B/s row.
            (0.5, 0, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, -5, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            # qBittorrent's 8_640_000 "infinite" eta and a 0/negative eta are unknown.
            # The last finite second still renders.
            (0.5, 100, 8_640_000, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 0, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 8_639_999, None, None, TorrentTelemetry(0.5, 100, 8_639_999, None, None)),
            # Zero/negative byte counts are unknown, not empty-progress readings.
            (0.5, None, None, 0, 0, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, None, None, -3, -1, TorrentTelemetry(0.5, None, None, None, None)),
            # An over-count clamps done to the total, never a >100% bar.
            (0.5, None, None, 500, 200, TorrentTelemetry(0.5, None, None, 200, 200)),
            # Bytes done without a known total still renders.
            (0.5, None, None, 100, None, TorrentTelemetry(0.5, None, None, 100, None)),
        ],
    )
    def test_edge_inputs(
        self,
        progress: object,
        dlspeed: object,
        eta: object,
        completed: object,
        size: object,
        expected: TorrentTelemetry,
    ) -> None:
        row = SimpleNamespace(progress=progress, dlspeed=dlspeed, eta=eta, completed=completed, size=size)

        assert sanitize_torrent_telemetry(row) == expected

    def test_attrless_row_folds_to_the_zero_reading(self) -> None:
        # The fields are read best-effort off the row: a row missing them all is the empty telemetry.
        assert sanitize_torrent_telemetry(object()) == TorrentTelemetry(0.0, None, None, None, None)


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in WaitOutcome} == {"COMPLETE", "ERRORED", "MISSING"}


def _mapping(remote: str, local: str, *, host: str | None = None) -> RemotePathMapping:
    """One remote path mapping from the raw API field names."""

    return RemotePathMapping.model_validate({"host": host, "remotePath": remote, "localPath": local})


class TestTranslateDownloadPath:
    """The remote-path translation behind the folder-scan fallback."""

    def test_no_mappings_is_a_no_op(self) -> None:
        assert translate_download_path("/d/folder", [], "qbit") == "/d/folder"

    def test_prefix_translates_and_suffix_survives(self) -> None:
        # The live incident mapping: trailing slash on both stored paths.
        mappings = [_mapping("/home/u/torrents/4k-tv/", "/remote/torrents/4k-tv/")]
        assert (
            translate_download_path("/home/u/torrents/4k-tv/Show S01", mappings, None)
            == "/remote/torrents/4k-tv/Show S01"
        )

    def test_exact_match_translates_to_local_root(self) -> None:
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads", mappings, None) == "/data"

    def test_separator_boundary_is_respected(self) -> None:
        # /downloads must NOT prefix-match /downloads-x/f.
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads-x/f", mappings, None) == "/downloads-x/f"

    def test_trailing_slash_tolerated_on_either_side(self) -> None:
        assert translate_download_path("/d/f", [_mapping("/d/", "/l")], None) == "/l/f"
        assert translate_download_path("/d/f", [_mapping("/d", "/l/")], None) == "/l/f"

    def test_suffix_case_is_preserved(self) -> None:
        # Compare case-insensitively but never fold the suffix: POSIX targets
        # are case-sensitive.
        mappings = [_mapping("/Downloads", "/data")]
        assert translate_download_path("/downloads/Show S01/Ep.MKV", mappings, None) == "/data/Show S01/Ep.MKV"

    def test_windows_backslash_remote_path(self) -> None:
        mappings = [_mapping("C:\\torrents\\", "/data/torrents")]
        assert translate_download_path("C:\\torrents\\Show\\ep.mkv", mappings, None) == "/data/torrents/Show/ep.mkv"

    def test_longest_prefix_wins(self) -> None:
        mappings = [
            _mapping("/d", "/short"),
            _mapping("/d/tv", "/long"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, None) == "/long/Show"

    def test_host_equality_tiebreaks_equal_prefixes(self) -> None:
        mappings = [
            _mapping("/d", "/other-client", host="other"),
            _mapping("/d", "/ours", host="qbit.local"),
        ]
        assert translate_download_path("/d/Show", mappings, "QBIT.LOCAL") == "/ours/Show"

    def test_host_mismatch_never_excludes(self) -> None:
        # Sonarr's host is the download-client host as SONARR knows it -
        # routinely a different string from our qBittorrent host.
        mappings = [_mapping("/d", "/data", host="sonarr-view-of-qbit")]
        assert translate_download_path("/d/Show", mappings, "localhost") == "/data/Show"

    def test_longer_prefix_beats_host_match(self) -> None:
        mappings = [
            _mapping("/d", "/host-matched", host="qbit"),
            _mapping("/d/tv", "/longer", host="other"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, "qbit") == "/longer/Show"

    def test_mapping_missing_either_path_is_skipped(self) -> None:
        mappings = [_mapping("", "/data"), _mapping("/d", "")]
        assert translate_download_path("/d/Show", mappings, None) == "/d/Show"

    def test_single_file_content_path_translates(self) -> None:
        # A single-FILE torrent's content_path is the file itself.
        mappings = [_mapping("/d/", "/data/")]
        assert translate_download_path("/d/Show - 01.mkv", mappings, None) == "/data/Show - 01.mkv"
