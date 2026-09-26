# pyright: strict
"""A torrent placed under several windows: each judges the whole torrent, the verdicts merge by window order."""

from collections.abc import Iterable, Mapping
from dataclasses import replace

import pytest

from pearlarr.manual_import import EntryNames
from pearlarr.placement_types import (
    NO_EVIDENCE,
    EpisodeIndex,
    ListingEvidence,
    PlacementBatch,
    PlacementVerdict,
    TargetScope,
)
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo
from pearlarr.window_placement import WindowedAssignment, assign_across_windows, place_leftover, windows_of

from .builders import blind, by_name, entry_claim, numbered_names, parsed_info, place, series_index

_SERIES = series_index(
    {
        **{EpisodeKey(0, n): 500 + n for n in (1, 2, 3)},
        **{EpisodeKey(1, n): 600 + n for n in range(1, 7)},
        **{EpisodeKey(2, n): 700 + n for n in range(1, 7)},
    }
)
_SPECIALS = [501, 502, 503]
_FIRST_COUR = [601, 602, 603]
_SECOND_COUR = [604, 605, 606]
# A series numbering its one season 2, so a first-season key resolves to nothing under it.
_OTHER_SERIES = series_index({EpisodeKey(2, n): 200 + n for n in (1, 2, 3)})
# A series of six specials, room for a listing a three-wide pack's numbers miss.
_MANY_SPECIALS = series_index({EpisodeKey(0, n): 500 + n for n in range(1, 7)})


def _window(resolved: list[int], *titles: str, used: Iterable[int] = (), series: EpisodeIndex = _SERIES) -> TargetScope:
    return TargetScope(resolved, series, used=frozenset(used), names=EntryNames("Show", titles))


def _batch(parsed: Mapping[str, ParsedFileInfo | None]) -> PlacementBatch:
    return PlacementBatch(list(parsed), parsed)


def _placed_under(result: WindowedAssignment) -> dict[str, tuple[tuple[int, ...], int]]:
    """Name -> (ids, window index) for the files a window placed."""

    return {v.placement.name: (v.placement.ids, v.window_index) for v in result.verdicts if v.window_index is not None}


class TestOneWindow:
    """The composition under one window is the placer, verdict for verdict."""

    def test_one_window_is_the_single_window_placer(self) -> None:
        # The identity covers a placed run, a foreign file, and a skipped one alike.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "keyed.mkv": parsed_info(season=1, episodes=(1,)), "loose.mkv": parsed_info()}
        window = _window(_SPECIALS, used=[0])
        alone = place(parsed, window)

        result = assign_across_windows(_batch(parsed), (window,))

        assert result.merged == alone
        assert {p.verdict for p in alone.placements} == {
            PlacementVerdict.RELEASE_RUN,
            PlacementVerdict.FOREIGN,
            PlacementVerdict.SKIPPED,
        }
        assert [v.window_index for v in result.verdicts] == [0 if p.verdict.placed else None for p in alone.placements]

    def test_no_windows_skip_every_file(self) -> None:
        result = assign_across_windows(_batch(blind(["a.mkv", "b.mkv"])), ())

        assert by_name(result.merged) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.SKIPPED))


class TestSeveralWindows:
    """Every window judges the whole torrent, and window order decides what several windows would take."""

    def test_a_season_pack_splits_across_two_cour_windows(self) -> None:
        run = numbered_names("show", 6)

        result = assign_across_windows(_batch(blind(run)), (_window(_FIRST_COUR), _window(_SECOND_COUR)))

        assert _placed_under(result) == {name: ((601 + i,), i // 3) for i, name in enumerate(run)}
        assert result.assigned_under(0) == {name: [601 + i] for i, name in enumerate(run[:3])}
        assert result.assigned_under(1) == {name: [604 + i] for i, name in enumerate(run[3:])}

    def test_a_franchise_pack_places_each_run_under_the_window_its_title_names(self) -> None:
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS, "Show Alpha"), _window(_FIRST_COUR, "Show Beta"))

        result = assign_across_windows(_batch(blind([*alpha, *beta])), windows)

        assert _placed_under(result) == {
            **{name: ((501 + i,), 0) for i, name in enumerate(alpha)},
            **{name: ((601 + i,), 1) for i, name in enumerate(beta)},
        }

    def test_a_later_windows_placement_does_not_re_judge_an_earlier_one(self) -> None:
        # A blind pair under the first window stays a tie once the second takes its named run: every
        # window judges the whole torrent, since a remainder re-read as a fresh run (a `1..6` whose
        # last three another window placed reads as `1..3`) could index a window the whole run never could.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS), _window(_FIRST_COUR, "Show Beta"))

        result = assign_across_windows(_batch(blind([*alpha, *beta])), windows)

        assert _placed_under(result) == {name: ((601 + i,), 1) for i, name in enumerate(beta)}
        assert result.merged.skipped == tuple(alpha)

    def test_windows_on_different_series_place_their_own_files(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), "b.mkv": parsed_info(season=2, episodes=(1,))}
        windows = (_window(_FIRST_COUR), _window([201, 202, 203], series=_OTHER_SERIES))

        result = assign_across_windows(_batch(parsed), windows)

        assert _placed_under(result) == {"a.mkv": ((601,), 0), "b.mkv": ((201,), 1)}

    def test_an_id_placed_under_one_window_is_used_under_an_overlapping_one(self) -> None:
        # The run takes the first three episodes under the first window. Alone, the second window's
        # ordered zip would put the blind file onto the first of them: two files never share an episode
        # across windows, so the ids the first window placed are seeds under the second.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "b.mkv": parsed_info()}
        wider = [*_FIRST_COUR, 604]
        assert place(parsed, _window(wider)).assigned["b.mkv"] == [601]

        result = assign_across_windows(_batch(parsed), (_window(_FIRST_COUR), _window(wider)))

        assert _placed_under(result) == {name: ((601 + i,), 0) for i, name in enumerate(run)}
        assert by_name(result.merged)["b.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_a_window_untouched_by_earlier_placements_keeps_its_count_legs(self) -> None:
        # A run from anywhere fits the second window's width: nothing placed under the first is a seed there.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = assign_across_windows(_batch(parsed), (_window([601]), _window([701, 702, 703])))

        assert _placed_under(result) == {"a.mkv": ((601,), 0), **{name: ((701 + i,), 1) for i, name in enumerate(run)}}

    def test_an_id_placed_under_one_window_narrows_an_overlapping_one(self) -> None:
        # The id the first window placed is a seed under the second, whose window is then the three
        # episodes left, and a run numbered as those fits it. Four wide, the run would fit nothing.
        run = numbered_names("sp", 4)[1:]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = assign_across_windows(_batch(parsed), (_window([601]), _window([601, 602, 603, 604])))

        assert _placed_under(result) == {"a.mkv": ((601,), 0), **{name: ((602 + i,), 1) for i, name in enumerate(run)}}

    def test_a_placed_verdict_wins_over_every_claim(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,))}

        result = assign_across_windows(_batch(parsed), (_window([501]), _window([601])))

        assert _placed_under(result) == {"a.mkv": ((601,), 1)}
        assert result.merged.excluded == ()

    @pytest.mark.parametrize(
        ("windows", "verdict"),
        [
            pytest.param(
                (_window([501]), _window([201], series=_OTHER_SERIES)),
                PlacementVerdict.SKIPPED,
                id="foreign then skipped",
            ),
            pytest.param(
                (_window([201], series=_OTHER_SERIES), _window([601])),
                PlacementVerdict.DUPLICATE,
                id="skipped then duplicate",
            ),
            pytest.param((_window([501]), _window([601])), PlacementVerdict.DUPLICATE, id="foreign then duplicate"),
            pytest.param((_window([601]), _window([501])), PlacementVerdict.DUPLICATE, id="duplicate then foreign"),
        ],
    )
    def test_claims_merge_to_the_most_specific(
        self, windows: tuple[TargetScope, ...], verdict: PlacementVerdict
    ) -> None:
        # Two files keyed alike: the second is a duplicate wherever the first places, foreign where the
        # key resolves outside, and skipped where it resolves to nothing.
        parsed = {name: parsed_info(season=1, episodes=(1,)) for name in ("a.mkv", "d.mkv")}

        result = assign_across_windows(_batch(parsed), windows)

        assert by_name(result.merged)["d.mkv"] == ((), verdict)

    def test_an_extra_under_one_window_yields_to_another_windows_claim(self) -> None:
        # The opening's word is a title word of the second entry, which reads the file as possibly its own.
        parsed = {"show op.mkv": parsed_info()}
        windows = (_window([501]), _window([601], "Show Op", used=[601]))

        result = assign_across_windows(_batch(parsed), windows)

        assert by_name(result.merged)["show op.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_an_unscoped_window_seeds_every_earlier_placement(self) -> None:
        # The unscoped window resolves any key against the whole series: an id another window placed
        # is used there too, so the keyed file cannot take it a second time.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "d.mkv": parsed_info(season=1, episodes=(3, 4))}

        result = assign_across_windows(_batch(parsed), (_window([601, 602, 603, 604], used=[604]), _window([])))

        assert _placed_under(result) == {name: ((601 + i,), 0) for i, name in enumerate(run)}
        assert by_name(result.merged)["d.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_an_earlier_windows_held_file_outranks_a_later_windows_placement(self) -> None:
        # The first member's key would place it under the second window's exact pass: the hold stands.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), run[0]: parsed_info(season=1, episodes=(1,)), "x.mkv": None}

        result = assign_across_windows(_batch(parsed), (_window(_SPECIALS), _window([601])))

        assert _placed_under(result) == {}
        assert {by_name(result.merged)[name] for name in run} == {((), PlacementVerdict.HELD)}

    def test_an_earlier_windows_misnumbered_pack_outranks_a_later_windows_placement(self) -> None:
        # Under the first window's listing (specials 2, 4 and 6, three as the pack is wide) the pack's `1` is
        # no listed special: misnumbered, a human's. The second window, unlisted, would place it by number.
        pack = {f"Show.S00E{n:02d}.mkv": parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n in (1, 2, 3)}
        listing = ListingEvidence(frozenset({502, 504, 506}))
        listed = replace(_window([502, 504], series=_MANY_SPECIALS), listing=listing)

        result = assign_across_windows(_batch(pack), (listed, _window([501, 502, 503], series=_MANY_SPECIALS)))

        assert _placed_under(result) == {}
        assert {verdict for _ids, verdict in by_name(result.merged).values()} == {PlacementVerdict.MISNUMBERED}

    def test_the_parse_flag_is_the_torrents(self) -> None:
        # An unknown parse the first window never held (its window is one slot) still holds the run
        # under the second: the flag counts every file of the torrent under every window.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "x.mkv": None}

        result = assign_across_windows(_batch(parsed), (_window([501]), _window(_FIRST_COUR)))

        assert _placed_under(result) == {}
        assert {by_name(result.merged)[name] for name in run} == {((), PlacementVerdict.HELD)}

    def test_two_numberless_files_over_two_one_slot_windows_place_nowhere(self) -> None:
        result = assign_across_windows(_batch(blind(["a.mkv", "b.mkv"])), (_window([501]), _window([502])))

        assert by_name(result.merged) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.SKIPPED))


class TestWindowsOf:
    """`windows_of`: one window per claim in order, over its series' index, with the claim's names and the listing."""

    def test_one_window_per_claim_over_its_series_index(self) -> None:
        names = EntryNames("Show", ("Show Alpha",))
        claims = (
            entry_claim(al_id=1, series_id=7, ordered_episode_ids=[601, 602], names=names),
            entry_claim(al_id=2, series_id=8, ordered_episode_ids=[201]),
        )

        listing = ListingEvidence(frozenset({601}), {"a.mkv": 601})

        windows = windows_of(claims, {7: _SERIES, 8: _OTHER_SERIES}, listing)

        assert windows == (
            TargetScope([601, 602], _SERIES, names=names, listing=listing),
            TargetScope([201], _OTHER_SERIES, names=EntryNames(), listing=listing),
        )

    def test_an_unscoped_claim_gives_an_unscoped_window(self) -> None:
        (window,) = windows_of((entry_claim(series_id=7),), {7: _SERIES}, NO_EVIDENCE)

        assert window == TargetScope([], _SERIES)
        assert window.unscoped


class TestPlaceLeftover:
    """`place_leftover`: the mapped names stay put, their ids are used, and the whole batch's parses ride along."""

    def test_the_mapped_names_are_not_re_placed(self) -> None:
        run = numbered_names("show", 3)

        result = place_leftover({run[0]: [601]}, _batch(blind(run)), (_window(_FIRST_COUR),))

        assert [v.placement.name for v in result.verdicts] == run[1:]
        assert _placed_under(result) == {run[1]: ((602,), 0), run[2]: ((603,), 0)}

    def test_the_parse_overlay_keeps_the_whole_batchs_parses(self) -> None:
        # The mapped sharer's parse still feeds the shared-absolute tell: over the leftover alone, the
        # v2 would take the spare id.
        v1, v2 = "e - 12.mkv", "e - 12v2.mkv"
        parsed = {v1: parsed_info(absolutes=(12,)), v2: parsed_info(absolutes=(12,))}
        series = series_index({EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587})
        alone = assign_across_windows(_batch({v2: parsed[v2]}), (_window([2586, 2587], used=[2586], series=series),))
        assert _placed_under(alone) == {v2: ((2587,), 0)}

        result = place_leftover({v1: [2586]}, _batch(parsed), (_window([2586, 2587], series=series),))

        assert by_name(result.merged) == {v2: ((), PlacementVerdict.SKIPPED)}

    def test_a_mapped_id_is_used_under_a_scoped_window_only_inside_its_ids(self) -> None:
        # One slot, one blind file: the slot is gone when the map holds it, open when the map's id is
        # another window's.
        batch = _batch(blind(["a.mkv", "b.mkv"]))

        inside = place_leftover({"a.mkv": [601]}, batch, (_window([601]),))
        outside = place_leftover({"a.mkv": [999]}, batch, (_window([601]),))

        assert by_name(inside.merged) == {"b.mkv": ((), PlacementVerdict.SKIPPED)}
        assert _placed_under(outside) == {"b.mkv": ((601,), 0)}

    def test_every_mapped_id_is_used_under_an_unscoped_window(self) -> None:
        # The unscoped window resolves any key against the whole series, so an id mapped anywhere is
        # taken there: the keyed file cannot land on it a second time.
        keyed = {"b.mkv": parsed_info(season=1, episodes=(1,))}

        free = place_leftover({}, _batch(keyed), (_window([]),))
        used = place_leftover({"a.mkv": [601]}, _batch({**blind(["a.mkv"]), **keyed}), (_window([]),))

        assert _placed_under(free) == {"b.mkv": ((601,), 0)}
        assert by_name(used.merged) == {"b.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_two_windows_on_two_series_place_each_file_under_the_first_that_resolves_it(self) -> None:
        # The season-2 key resolves outside the first series' window and inside the second's. The mapped
        # file holds the first-season key's episode, so that file stays loud.
        parsed = {
            "s.mkv": parsed_info(),
            "a.mkv": parsed_info(season=1, episodes=(1,)),
            "b.mkv": parsed_info(season=2, episodes=(1,)),
        }
        windows = (_window(_FIRST_COUR), _window([201, 202, 203], series=_OTHER_SERIES))

        held = place_leftover({"s.mkv": [601]}, _batch(parsed), windows)
        free = place_leftover({"s.mkv": [602]}, _batch(parsed), windows)

        assert _placed_under(held) == {"b.mkv": ((201,), 1)}
        assert by_name(held.merged)["a.mkv"] == ((), PlacementVerdict.SKIPPED)
        assert _placed_under(free) == {"a.mkv": ((601,), 0), "b.mkv": ((201,), 1)}

    def test_one_window_and_an_empty_map_is_the_cross_window_placement(self) -> None:
        run = numbered_names("sp", 3)
        batch = _batch({**blind(run), "keyed.mkv": parsed_info(season=1, episodes=(1,)), "x.mkv": None})
        window = _window(_SPECIALS)

        assert place_leftover({}, batch, (window,)) == assign_across_windows(batch, (window,))
