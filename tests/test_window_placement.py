# pyright: strict
"""A torrent placed under several windows: each judges the whole torrent, the verdicts merge by window order."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import pytest

from pearlarr.manual_import import EntryClaim, EntryNames, EntryWindow, FileEpisodeMap
from pearlarr.numberings import NumberingReading
from pearlarr.placement_types import (
    NO_EVIDENCE,
    NO_SEEDED,
    EpisodeIndex,
    ListingEvidence,
    PlacementVerdict,
    TargetScope,
)
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo
from pearlarr.window_placement import (
    WindowedAssignment,
    WindowOrder,
    ordered_windows,
    place_across_windows,
    windows_of,
)

from .builders import (
    ALIASED_PACK,
    batch,
    blind,
    by_name,
    entry_claim,
    entry_window,
    numbered_names,
    parsed_info,
    place,
    series_index,
    specials_pack,
    torrent_listing,
    verdicts_of,
)

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
# Another series of three specials, whose window the aliased pack's listing never covers.
_OTHER_SPECIALS = series_index({EpisodeKey(0, n): 900 + n for n in (1, 2, 3)})
# `ALIASED_PACK` as the evidence every window carries: the pack's 1, 2 and 3 aliased onto specials 2, 4 and 6.
_ALIASED = ListingEvidence(ALIASED_PACK)
# `_SERIES` as Sonarr series 7, the series the window and claim builders default to.
_SERIES_7: Mapping[int, EpisodeIndex] = {7: _SERIES}


def _window(resolved: list[int], *titles: str, used: Iterable[int] = (), series: EpisodeIndex = _SERIES) -> TargetScope:
    return TargetScope(resolved, series, used=frozenset(used), names=EntryNames("Show", titles))


def _listed_window(listing: ListingEvidence, ids: list[int], series: EpisodeIndex = _MANY_SPECIALS) -> TargetScope:
    """One window of `ids` over `series`, carrying `listing`."""

    return replace(_window(ids, series=series), listing=listing)


def _listed_windows(listing: ListingEvidence, *windows: list[int]) -> tuple[TargetScope, ...]:
    """One window per id list over the many-specials series, each carrying `listing`."""

    return tuple(_listed_window(listing, ids) for ids in windows)


def _place_across(parsed: Mapping[str, ParsedFileInfo | None], windows: Sequence[TargetScope]) -> WindowedAssignment:
    """Every parsed name placed under `windows`, with no stored map."""

    return place_across_windows(batch(parsed), windows)


def _disputed_torrent() -> tuple[list[str], dict[str, ParsedFileInfo | None], tuple[TargetScope, ...]]:
    """A pack of seven keyed on past the season's end, under two cours whose special's absolute sits between them.

    The first cour places three, the second refuses the last four as misnumbered.
    """

    series = series_index(
        {**{EpisodeKey(1, n): 600 + n for n in range(1, 7)}, EpisodeKey(0, 1): 501},
        absolutes={601: 1, 602: 2, 603: 3, 501: 4, 604: 5, 605: 6, 606: 7},
    )
    pack = [f"Show - S01E{n:02d}.mkv" for n in range(1, 8)]
    parsed: dict[str, ParsedFileInfo | None] = {
        name: parsed_info(season=1, episodes=(n,), matched=((1, n),) if n < 7 else ()) for n, name in enumerate(pack, 1)
    }
    return pack, parsed, (_window(_FIRST_COUR, series=series), _window(_SECOND_COUR, series=series))


def _disputed_pack() -> tuple[list[str], WindowedAssignment]:
    """`_disputed_torrent` placed: its pack and the assignment."""

    pack, parsed, windows = _disputed_torrent()
    return pack, _place_across(parsed, windows)


class TestOneWindow:
    """The composition under one window is the placer, verdict for verdict."""

    def test_one_window_is_the_single_window_placer(self) -> None:
        # The identity covers a placed run, a foreign file, and a skipped one alike.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "keyed.mkv": parsed_info(season=1, episodes=(1,)), "loose.mkv": parsed_info()}
        window = _window(_SPECIALS, used=[0])
        alone = place(parsed, window)

        result = _place_across(parsed, (window,))

        assert result.merged == alone
        assert verdicts_of(alone) == {
            PlacementVerdict.RELEASE_RUN,
            PlacementVerdict.FOREIGN,
            PlacementVerdict.SKIPPED,
        }

    def test_no_windows_skip_every_file(self) -> None:
        result = _place_across(blind(["a.mkv", "b.mkv"]), ())

        assert by_name(result.merged) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.SKIPPED))


class TestSeveralWindows:
    """Every window judges the whole torrent, and window order decides what several windows would take."""

    def test_a_season_pack_splits_across_two_cour_windows(self) -> None:
        run = numbered_names("show", 6)

        result = _place_across(blind(run), (_window(_FIRST_COUR), _window(_SECOND_COUR)))

        assert result.merged.assigned == {name: (601 + i,) for i, name in enumerate(run)}

    def test_a_franchise_pack_places_each_run_under_the_window_its_title_names(self) -> None:
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS, "Show Alpha"), _window(_FIRST_COUR, "Show Beta"))

        result = _place_across(blind([*alpha, *beta]), windows)

        assert result.merged.assigned == {
            **{name: (501 + i,) for i, name in enumerate(alpha)},
            **{name: (601 + i,) for i, name in enumerate(beta)},
        }

    def test_a_later_windows_placement_does_not_re_judge_an_earlier_one(self) -> None:
        # A blind pair under the first window stays a tie once the second takes its named run: every
        # window judges the whole torrent, since a remainder re-read as a fresh run (a `1..6` whose
        # last three another window placed reads as `1..3`) could index a window the whole run never could.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS), _window(_FIRST_COUR, "Show Beta"))

        result = _place_across(blind([*alpha, *beta]), windows)

        assert result.merged.assigned == {name: (601 + i,) for i, name in enumerate(beta)}
        assert result.merged.skipped == tuple(alpha)

    def test_windows_on_different_series_place_their_own_files(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), "b.mkv": parsed_info(season=2, episodes=(1,))}
        windows = (_window(_FIRST_COUR), _window([201, 202, 203], series=_OTHER_SERIES))

        result = _place_across(parsed, windows)

        assert result.merged.assigned == {"a.mkv": (601,), "b.mkv": (201,)}

    def test_an_id_placed_under_one_window_is_used_under_an_overlapping_one(self) -> None:
        # The run takes the first three episodes under the first window. Alone, the second window's
        # ordered zip would put the blind file onto the first of them: two files never share an episode
        # across windows, so the ids the first window placed are used under the second.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "b.mkv": parsed_info()}
        wider = [*_FIRST_COUR, 604]
        assert place(parsed, _window(wider)).assigned["b.mkv"] == (601,)

        result = _place_across(parsed, (_window(_FIRST_COUR), _window(wider)))

        assert result.merged.assigned == {name: (601 + i,) for i, name in enumerate(run)}
        assert by_name(result.merged)["b.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_a_window_untouched_by_earlier_placements_keeps_its_count_legs(self) -> None:
        # A run from anywhere fits the second window's width: nothing placed under the first is used there.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = _place_across(parsed, (_window([601]), _window([701, 702, 703])))

        assert result.merged.assigned == {"a.mkv": (601,), **{name: (701 + i,) for i, name in enumerate(run)}}

    def test_an_id_placed_under_one_window_narrows_an_overlapping_one(self) -> None:
        # The id the first window placed is used under the second, whose window is then the three
        # episodes left, and a run numbered as those fits it. Four wide, the run would fit nothing.
        run = numbered_names("sp", 4)[1:]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = _place_across(parsed, (_window([601]), _window([601, 602, 603, 604])))

        assert result.merged.assigned == {"a.mkv": (601,), **{name: (602 + i,) for i, name in enumerate(run)}}

    def test_a_placed_verdict_wins_over_every_claim(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,))}

        result = _place_across(parsed, (_window([501]), _window([601])))

        assert result.merged.assigned == {"a.mkv": (601,)}
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

        result = _place_across(parsed, windows)

        assert by_name(result.merged)["d.mkv"] == ((), verdict)

    def test_an_extra_under_one_window_yields_to_another_windows_claim(self) -> None:
        # The opening's word is a title word of the second entry, which reads the file as possibly its own.
        parsed = {"show op.mkv": parsed_info()}
        windows = (_window([501]), _window([601], "Show Op", used=[601]))

        result = _place_across(parsed, windows)

        assert by_name(result.merged)["show op.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_every_earlier_placement_is_used_under_an_unscoped_window(self) -> None:
        # The unscoped window resolves any key against the whole series: an id another window placed
        # is used there too, so the keyed file cannot take it a second time.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "d.mkv": parsed_info(season=1, episodes=(3, 4))}

        result = _place_across(parsed, (_window([601, 602, 603, 604], used=[604]), _window([])))

        assert result.merged.assigned == {name: (601 + i,) for i, name in enumerate(run)}
        assert by_name(result.merged)["d.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_an_earlier_windows_held_file_outranks_a_later_windows_placement(self) -> None:
        # The first member's key would place it under the second window's exact pass: the hold stands.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), run[0]: parsed_info(season=1, episodes=(1,)), "x.mkv": None}

        result = _place_across(parsed, (_window(_SPECIALS), _window([601])))

        assert result.merged.assigned == {}
        assert {by_name(result.merged)[name] for name in run} == {((), PlacementVerdict.HELD)}

    def test_a_run_disputed_past_the_cour_boundary_places_under_the_first_cour_and_is_refused_under_the_second(
        self,
    ) -> None:
        pack, result = _disputed_pack()

        assert result.merged.assigned == {name: (600 + n,) for n, name in enumerate(pack[:3], 1)}
        assert {by_name(result.merged)[name][1] for name in pack[3:]} == {PlacementVerdict.MISNUMBERED}

    def test_an_earlier_windows_misnumbered_pack_outranks_a_later_windows_placement(self) -> None:
        # Under the first window's listing (specials 2, 4 and 6, three as the pack is wide) the pack's `1` is
        # no listed special: misnumbered, a human's. The second window, unlisted, would place it by number.
        pack = specials_pack(1, 2, 3)
        listed = _listed_window(ListingEvidence(torrent_listing(*ALIASED_PACK.entries)), [502, 504])

        result = _place_across(pack, (listed, _window([501, 502, 503], series=_MANY_SPECIALS)))

        assert result.merged.assigned == {}
        assert verdicts_of(result.merged) == {PlacementVerdict.MISNUMBERED}

    def test_an_earlier_windows_title_refutal_yields_to_a_later_windows_placement(self) -> None:
        # Sonarr read the second season's bare numbers as first-season absolutes, and its titled member names a
        # second-season episode.
        titles = ("Alpha Start", "Bravo Turn", "Charlie Rise", "Echo Dawn", "Foxtrot Fall", "Golf Night")
        series = series_index(
            {**{EpisodeKey(1, n): 600 + n for n in (1, 2, 3)}, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3)}},
            absolutes={601: 1, 602: 2, 603: 3, 701: 4, 702: 5, 703: 6},
            titles=dict(zip((601, 602, 603, 701, 702, 703), titles, strict=True)),
        )
        sequel = ["show s2 - 01 - echo dawn.mkv", "show s2 - 02.mkv", "show s2 - 03.mkv"]
        parsed = {name: parsed_info(absolutes=(n,), matched=((1, n),)) for n, name in enumerate(sequel, 1)}

        result = _place_across(
            parsed, (_window([601, 602, 603], series=series), _window([701, 702, 703], series=series))
        )

        assert result.merged.assigned == {name: (700 + n,) for n, name in enumerate(sequel, 1)}

    def test_an_aliased_pack_places_each_member_under_the_window_holding_its_special(self) -> None:
        # Each window maps the whole pack. The member outside the first window is `ALIASED_ELSEWHERE` there and
        # placed under the second.
        windows = _listed_windows(_ALIASED, [502, 504], [506])

        result = _place_across(specials_pack(1, 2, 3), windows)

        assert result.merged.assigned == {
            "Show.S00E01.mkv": (502,),
            "Show.S00E02.mkv": (504,),
            "Show.S00E03.mkv": (506,),
        }
        assert verdicts_of(result.merged) == {PlacementVerdict.ALTERNATE}

    def test_a_window_sharing_a_special_the_first_placed_still_maps_the_pack_through_its_aliases(self) -> None:
        # The second window also admits special 2, which the first window placed. It still maps the whole pack,
        # skips that member, and places the other two by their aliases rather than their own numbers.
        listing = ListingEvidence(torrent_listing(entry_window(1, (502, 505, 506)), aliases={1: 2, 5: 6, 6: 5}))
        windows = _listed_windows(listing, [502], [502, 505, 506])

        result = _place_across(specials_pack(1, 5, 6), windows)

        assert [window.assigned for window in result.windows] == [
            {"Show.S00E01.mkv": (502,)},
            {"Show.S00E05.mkv": (506,), "Show.S00E06.mkv": (505,)},
        ]
        assert result.merged.assigned == {
            "Show.S00E01.mkv": (502,),
            "Show.S00E05.mkv": (506,),
            "Show.S00E06.mkv": (505,),
        }
        assert verdicts_of(result.merged) == {PlacementVerdict.ALTERNATE}

    @pytest.mark.parametrize("judged_first", [True, False], ids=["judged then unjudged", "unjudged then judged"])
    def test_a_member_the_aliases_and_another_windows_numbers_place_differently_is_misnumbered(
        self, judged_first: bool
    ) -> None:
        # The first series' window maps the pack through its aliases (one member's special lies outside it). The
        # other series' window, which the listing doesn't cover, places every member by its own number. The two
        # disagree on every member, and neither may win by coming first.
        judged = _listed_window(_ALIASED, [502, 504])
        unjudged = _listed_window(_ALIASED, [901, 902, 903], series=_OTHER_SPECIALS)
        windows = (judged, unjudged) if judged_first else (unjudged, judged)

        result = _place_across(specials_pack(1, 2, 3), windows)

        assert result.merged.assigned == {}
        assert verdicts_of(result.merged) == {PlacementVerdict.MISNUMBERED}

    def test_a_member_whose_aliased_special_no_window_holds_stays_aliased_elsewhere_and_skipped(self) -> None:
        # No window admits special 6, so the third member is left as the aliases' claim, never placed by number.
        window = _listed_window(_ALIASED, [502, 504])

        result = _place_across(specials_pack(1, 2, 3), (window,))

        assert by_name(result.merged)["Show.S00E03.mkv"] == ((), PlacementVerdict.ALIASED_ELSEWHERE)
        assert result.merged.skipped == ("Show.S00E03.mkv",)

    @pytest.mark.parametrize("judged_first", [True, False], ids=["judged then foreign", "foreign then judged"])
    def test_a_member_aliased_in_one_window_and_foreign_in_another_stays_aliased_elsewhere(
        self, judged_first: bool
    ) -> None:
        # The other series' window holds two of its specials, so the third member reads as foreign there. The
        # aliases' claim outranks that whichever window comes first: the file is skipped, never excluded.
        judged = _listed_window(_ALIASED, [502, 504])
        other = _listed_window(_ALIASED, [901, 902], series=_OTHER_SPECIALS)
        windows = (judged, other) if judged_first else (other, judged)

        result = _place_across(specials_pack(1, 2, 3), windows)

        assert by_name(result.merged)["Show.S00E03.mkv"] == ((), PlacementVerdict.ALIASED_ELSEWHERE)
        assert result.merged.excluded == ()

    def test_the_parse_flag_is_the_torrents(self) -> None:
        # An unknown parse the first window never held (its window is one slot) still holds the run
        # under the second: the flag counts every file of the torrent under every window.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "x.mkv": None}

        result = _place_across(parsed, (_window([501]), _window(_FIRST_COUR)))

        assert result.merged.assigned == {}
        assert {by_name(result.merged)[name] for name in run} == {((), PlacementVerdict.HELD)}

    def test_two_numberless_files_over_two_one_slot_windows_place_nowhere(self) -> None:
        result = _place_across(blind(["a.mkv", "b.mkv"]), (_window([501]), _window([502])))

        assert by_name(result.merged) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.SKIPPED))


def _order(own: EntryWindow, listed: Sequence[EntryWindow], claims: Sequence[EntryClaim] = ()) -> WindowOrder:
    """`ordered_windows` over series 7, every window on it, for the grab whose entry's window is `own`."""

    order = ordered_windows(torrent_listing(*listed), claims, own, _SERIES_7)
    assert order is not None
    return order


class TestOrderedWindows:
    """`ordered_windows`: one window per AniList id, season windows before specials windows, and which import."""

    def test_season_windows_run_before_specials_windows_each_in_anilist_id_order(self) -> None:
        listed = (
            entry_window(1, _SPECIALS[:1]),
            entry_window(4, _SECOND_COUR),
            entry_window(2, _SPECIALS[1:]),
            entry_window(3, _FIRST_COUR),
        )

        order = _order(listed[3], listed)

        assert [w.al_id for w in order.windows] == [3, 4, 1, 2]

    def test_an_ignored_title_runs_after_the_kept_ones_of_its_kind(self) -> None:
        listed = (
            entry_window(1, _FIRST_COUR, ignored=True),
            entry_window(2, _SECOND_COUR),
            entry_window(3, _SPECIALS[:1], ignored=True),
            entry_window(4, _SPECIALS[1:]),
        )

        order = _order(listed[1], listed)

        assert [w.al_id for w in order.windows] == [2, 1, 4, 3]

    def test_ids_outside_are_every_other_windows_ids(self) -> None:
        listed = (entry_window(1, _FIRST_COUR), entry_window(2, _SECOND_COUR), entry_window(3, _SPECIALS))

        order = _order(listed[0], listed)

        assert order.ids_outside(2) == frozenset(_FIRST_COUR + _SPECIALS)

    def test_the_grabbing_entrys_window_wins_its_slot(self) -> None:
        own = entry_window(1, _FIRST_COUR)

        order = _order(own, (entry_window(1, _FIRST_COUR[:2]),), (entry_claim(al_id=1),))

        assert order.windows == (own,)

    def test_a_claimed_entrys_window_is_its_claims_and_it_imports(self) -> None:
        own = entry_window(1, _FIRST_COUR)
        claim = entry_claim(al_id=2, ordered_episode_ids=_SECOND_COUR[:2])

        order = _order(own, (entry_window(2, _SECOND_COUR),), (claim,))

        assert (order.windows, order.imports) == ((own, claim.window), {0, 1})

    def test_a_claims_window_takes_its_listed_windows_ignored_flag(self) -> None:
        # Entry 1 claimed the torrent before it was ignored. Its claim's window still imports, but runs after the
        # kept season window, where its listed window would.
        claim = entry_claim(al_id=1, ordered_episode_ids=_FIRST_COUR[:2])
        listed = (entry_window(1, _FIRST_COUR, ignored=True), entry_window(2, _SECOND_COUR))

        order = _order(listed[1], listed, (claim,))

        assert order.windows == (listed[1], replace(claim.window, ignored=True))
        assert order.imports == {0, 1}

    def test_a_claims_window_takes_the_flag_off_an_empty_listed_window(self) -> None:
        # The listed window came back empty, so it doesn't run, but its flag still reaches the claim's window.
        claim = entry_claim(al_id=1, ordered_episode_ids=_FIRST_COUR[:2])
        listed = (entry_window(1, (), ignored=True), entry_window(2, _SECOND_COUR))

        order = _order(listed[1], listed, (claim,))

        assert order.windows == (listed[1], replace(claim.window, ignored=True))

    def test_a_claims_window_takes_its_listed_windows_episode_count(self) -> None:
        # A claim doesn't keep AniList's count, so its window reads it off the entry's listed window.
        own = entry_window(1, _FIRST_COUR)
        claim = entry_claim(al_id=2, ordered_episode_ids=_SECOND_COUR[:2])

        order = _order(own, (entry_window(2, _SECOND_COUR, episode_count=3),), (claim,))

        assert order.windows == (own, replace(claim.window, episode_count=3))

    def test_a_listed_entry_never_claimed_runs_its_listed_window_and_does_not_import(self) -> None:
        listed = entry_window(2, _SECOND_COUR)

        order = _order(entry_window(1, _FIRST_COUR), (listed,))

        assert order.windows[1] == listed
        assert (order.at(1), order.imports) == (0, {0})

    def test_a_claim_its_entry_no_longer_lists_keeps_its_window(self) -> None:
        # Unchanged, so with no ignored flag and no episode count: the listing has neither for it.
        claim = entry_claim(al_id=2, ordered_episode_ids=_SECOND_COUR)
        listed = entry_window(1, _FIRST_COUR, episode_count=3)

        order = _order(listed, (listed,), (claim,))

        assert order.windows == (listed, claim.window)
        assert order.imports == {0, 1}

    def test_the_own_window_and_the_claims_import_wherever_they_sort(self) -> None:
        listed = (entry_window(1, _FIRST_COUR),)
        claim = entry_claim(al_id=2, ordered_episode_ids=_SECOND_COUR)

        order = _order(entry_window(3, _SPECIALS), listed, (claim,))

        assert [w.al_id for w in order.windows] == [1, 2, 3]
        assert (order.at(3), order.imports) == (2, {1, 2})

    def test_a_listed_window_with_no_ids_is_left_out(self) -> None:
        # It would be unscoped, admitting every id of the series.
        order = _order(entry_window(1, _FIRST_COUR), (entry_window(2, ()),))

        assert [w.al_id for w in order.windows] == [1]

    def test_a_claim_with_no_ids_still_runs_with_the_specials_windows(self) -> None:
        # A record an older version wrote can hold such a claim, which admits any id.
        claim = entry_claim(al_id=1)
        listed = entry_window(2, _FIRST_COUR)

        order = _order(listed, (listed,), (claim,))

        assert order.windows == (listed, claim.window)
        assert order.imports == {0, 1}

    def test_a_window_on_a_series_left_unread_orders_nothing(self) -> None:
        listed = torrent_listing(entry_window(2, (201,), series_id=8))

        assert ordered_windows(listed, (), entry_window(1, _FIRST_COUR), _SERIES_7) is None


class TestDecidedBy:
    """Which window decided each file, and what that means for the record the windows place for."""

    def test_each_file_names_the_first_window_to_place_or_refuse_it(self) -> None:
        run = numbered_names("show", 6)

        result = _place_across(
            {**blind(run), "loose.mkv": parsed_info()}, (_window(_FIRST_COUR), _window(_SECOND_COUR))
        )

        assert result.decided_by == {**dict.fromkeys(run[:3], 0), **dict.fromkeys(run[3:], 1)}

    def test_the_record_sets_aside_a_file_a_window_it_doesnt_import_placed(self) -> None:
        run = numbered_names("show", 6)
        result = _place_across(blind(run), (_window(_FIRST_COUR), _window(_SECOND_COUR)))

        assignment = result.for_record({0})

        assert assignment.assigned == {name: (601 + i,) for i, name in enumerate(run[:3])}
        assert {by_name(assignment)[name] for name in run[3:]} == {((), PlacementVerdict.LISTED_ELSEWHERE)}

    def test_the_record_sets_aside_a_hand_import_no_window_it_imports_was_party_to(self) -> None:
        # The second cour refused the last four as misnumbered, and the first cour read them as another slice's.
        pack, result = _disputed_pack()

        assignment = result.for_record({0})

        assert {by_name(assignment)[name][1] for name in pack[3:]} == {PlacementVerdict.LISTED_ELSEWHERE}

    def test_the_record_keeps_a_hand_import_a_window_it_imports_was_party_to(self) -> None:
        pack, result = _disputed_pack()

        for imports in ({1}, {0, 1}):
            assignment = result.for_record(imports)

            assert {by_name(assignment)[name][1] for name in pack[3:]} == {PlacementVerdict.MISNUMBERED}

    def test_the_record_keeps_a_disputed_file_a_window_it_imports_placed_by_number(self) -> None:
        # The aliases claim the pack's 1 for the second window while the first places it by number: a dispute is
        # misnumbered, and the first window, which placed it, is party to it.
        pack = specials_pack(1, 2, 3)
        result = _place_across(pack, _listed_windows(_ALIASED, [501, 503, 505], [502, 504, 506]))

        assignment = result.for_record({0})

        assert by_name(assignment)[next(iter(pack))][1] is PlacementVerdict.MISNUMBERED

    def test_a_file_another_window_placed_or_refused_is_listed_elsewhere_for_this_one(self) -> None:
        pack, result = _disputed_pack()

        assert (result.listed_elsewhere_under(0), result.listed_elsewhere_under(1)) == (
            frozenset(pack[3:]),
            frozenset(pack[:3]),
        )

    def test_a_file_another_window_held_is_not_listed_elsewhere(self) -> None:
        # No window has read a held file yet.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "x.mkv": None}

        result = _place_across(parsed, (_window(_SPECIALS), _window([601])))

        assert {by_name(result.merged)[name][1] for name in run} == {PlacementVerdict.HELD}
        assert result.listed_elsewhere_under(1) == frozenset()


_GAP = "show - 05.mkv"
"""A name whose bare 5 the shared series resolves to nothing: every window skips it."""
_PLACED = "show - 01.mkv"
"""A name whose bare 1 Sonarr matched to S01E01: the exact pass places it under the first window."""


def _filled(
    parsed: Mapping[str, ParsedFileInfo | None],
    windows: Sequence[TargetScope],
    proposals: Mapping[str, tuple[int, ...]],
    *,
    seeded: FileEpisodeMap = NO_SEEDED,
) -> tuple[WindowedAssignment, WindowedAssignment]:
    """The torrent placed under `windows`, before and after the reader's `proposals` fill it."""

    torrent = batch(parsed, seeded=seeded)
    before = place_across_windows(torrent, windows)
    return before, before.filled(torrent, NumberingReading(proposals))


def _gap_and_placed() -> dict[str, ParsedFileInfo | None]:
    return {
        _GAP: parsed_info(season=0, absolutes=(5,)),
        _PLACED: parsed_info(season=0, absolutes=(1,), matched=((1, 1),)),
    }


class TestFilled:
    """The reader's proposals fill the gaps the windows left, and touch nothing else."""

    def test_a_gap_is_filled_and_the_window_holding_its_ids_decides_it(self) -> None:
        before, after = _filled(_gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,)})

        assert by_name(after.merged) == {
            _GAP: ((605,), PlacementVerdict.NUMBERINGS_AGREE),
            _PLACED: ((601,), PlacementVerdict.EXACT),
        }
        assert after.decided_by == {_PLACED: 0, _GAP: 1}
        assert after.windows == before.windows

    def test_a_file_a_window_placed_keeps_its_placement(self) -> None:
        _, after = _filled(_gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_PLACED: (602,)})

        assert by_name(after.merged)[_PLACED] == ((601,), PlacementVerdict.EXACT)

    def test_a_file_a_window_refused_keeps_its_refusal(self) -> None:
        pack, parsed, windows = _disputed_torrent()

        _, after = _filled(parsed, windows, {pack[6]: (606,)})

        assert by_name(after.merged)[pack[6]] == ((), PlacementVerdict.MISNUMBERED)

    def test_a_file_a_window_claimed_by_alias_is_not_filled(self) -> None:
        # The aliases claim the pack's 3 for a window that holds no special 6, so it merges to skipped; the first
        # window was still party to it, and the second holds the proposed id.
        windows = (_listed_window(_ALIASED, [502, 504]), _listed_window(_ALIASED, [201], series=_OTHER_SERIES))

        _, after = _filled(specials_pack(1, 2, 3), windows, {"Show.S00E03.mkv": (201,)})

        assert by_name(after.merged)["Show.S00E03.mkv"] == ((), PlacementVerdict.SKIPPED)

    def test_a_file_a_window_read_inside_itself_is_not_filled(self) -> None:
        # The OVA's absolute reads to S01E02 inside the window; it lost that episode to a refused file. The passes
        # judged it by its own numbers, so the reader's proposal would reverse them.
        index = series_index(
            {EpisodeKey(1, 1): 601, EpisodeKey(1, 2): 602, EpisodeKey(1, 3): 603, EpisodeKey(5, 2): 702},
            absolutes={601: 1, 602: 2, 603: 3, 702: 102},
        )
        ova = "Show - 02 (OVA) [grp].mkv"
        parsed = {
            "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv": parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            ova: parsed_info(absolutes=(2,), matched=((1, 2),)),
        }

        _, after = _filled(parsed, (_window(_FIRST_COUR, series=index),), {ova: (603,)})

        assert by_name(after.merged)[ova] == ((), PlacementVerdict.SKIPPED)

    def test_an_id_a_window_placed_is_not_filled_again(self) -> None:
        _, after = _filled(_gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (601,)})

        assert by_name(after.merged)[_GAP] == ((), PlacementVerdict.SKIPPED)

    def test_an_id_the_stored_map_holds_is_not_filled(self) -> None:
        _, after = _filled(
            {_GAP: parsed_info(season=0, absolutes=(5,))},
            (_window(_FIRST_COUR), _window(_SECOND_COUR)),
            {_GAP: (605,)},
            seeded={"old.mkv": (605,)},
        )

        assert by_name(after.merged)[_GAP] == ((), PlacementVerdict.SKIPPED)

    def test_an_id_an_earlier_fill_took_is_not_filled_again(self) -> None:
        other = "show - 05 [other].mkv"
        parsed = {_GAP: parsed_info(season=0, absolutes=(5,)), other: parsed_info(season=0, absolutes=(5,))}

        _, after = _filled(parsed, (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,), other: (605,)})

        assert by_name(after.merged) == {
            _GAP: ((605,), PlacementVerdict.NUMBERINGS_AGREE),
            other: ((), PlacementVerdict.SKIPPED),
        }

    def test_ids_no_one_window_holds_are_not_filled(self) -> None:
        # A two-number name read across the cour boundary: each id has a window, the pair has none.
        name = "show - 03-04.mkv"

        _, after = _filled(
            {name: parsed_info(season=0, absolutes=(3, 4))},
            (_window(_FIRST_COUR), _window(_SECOND_COUR)),
            {name: (603, 604)},
        )

        assert by_name(after.merged)[name] == ((), PlacementVerdict.SKIPPED)

    def test_a_name_titled_as_an_episode_outside_the_proposed_ids_is_not_filled(self) -> None:
        titled = series_index({EpisodeKey(1, n): 600 + n for n in range(1, 7)}, titles={604: "Final Bell"})
        name = "show - 05 - Final Bell [grp].mkv"

        _, after = _filled(
            {name: parsed_info(season=0, absolutes=(5,))}, (_window(_FIRST_COUR, series=titled),), {name: (601,)}
        )

        assert by_name(after.merged)[name] == ((), PlacementVerdict.SKIPPED)

    def test_a_fill_into_a_window_the_record_does_not_import_is_another_titles(self) -> None:
        _, after = _filled(_gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,)})

        assert by_name(after.for_record({0}))[_GAP] == ((), PlacementVerdict.LISTED_ELSEWHERE)
        assert by_name(after.for_record({1}))[_GAP] == ((605,), PlacementVerdict.NUMBERINGS_AGREE)


class TestWindowsOf:
    """`windows_of`: one scope per entry window in order, over its series' index, with its names, count and listing."""

    def test_one_scope_per_window_over_its_series_index(self) -> None:
        names = EntryNames("Show", ("Show Alpha",))
        entries = (
            entry_window(1, (601, 602), series_id=7, names=names, episode_count=2),
            entry_window(2, (201,), series_id=8),
        )
        listing = ListingEvidence(torrent_listing(entry_window(1, (601,))), {"a.mkv": 601})

        windows = windows_of(entries, {7: _SERIES, 8: _OTHER_SERIES}, listing)

        assert windows == (
            TargetScope([601, 602], _SERIES, names=names, listing=listing, episode_count=2),
            TargetScope([201], _OTHER_SERIES, names=EntryNames(), listing=listing),
        )

    def test_a_claim_with_no_ids_gives_an_unscoped_window(self) -> None:
        (window,) = windows_of((entry_claim(series_id=7).window,), {7: _SERIES}, NO_EVIDENCE)

        assert window == TargetScope([], _SERIES)
        assert window.unscoped


class TestStoredMap:
    """A stored map: its names stay put and its ids are used under every window that admits them."""

    def test_the_mapped_names_are_not_re_placed(self) -> None:
        run = numbered_names("show", 3)

        result = place_across_windows(batch(blind(run), seeded={run[0]: (601,)}), (_window(_FIRST_COUR),))

        assert [p.name for p in result.merged.placements] == run[1:]
        assert result.merged.assigned == {run[1]: (602,), run[2]: (603,)}

    def test_a_mapped_id_takes_its_slot(self) -> None:
        # One slot, one blind file: the slot is gone when the map holds it.
        result = place_across_windows(batch(blind(["a.mkv", "b.mkv"]), seeded={"a.mkv": (601,)}), (_window([601]),))

        assert by_name(result.merged) == {"b.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_every_mapped_id_is_used_under_an_unscoped_window(self) -> None:
        # The unscoped window resolves any key against the whole series, so an id mapped anywhere is
        # taken there: the keyed file cannot land on it a second time.
        keyed = {"b.mkv": parsed_info(season=1, episodes=(1,))}

        free = _place_across(keyed, (_window([]),))
        used = place_across_windows(batch({**blind(["a.mkv"]), **keyed}, seeded={"a.mkv": (601,)}), (_window([]),))

        assert free.merged.assigned == {"b.mkv": (601,)}
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

        held = place_across_windows(batch(parsed, seeded={"s.mkv": (601,)}), windows)
        free = place_across_windows(batch(parsed, seeded={"s.mkv": (602,)}), windows)

        assert held.merged.assigned == {"b.mkv": (201,)}
        assert by_name(held.merged)["a.mkv"] == ((), PlacementVerdict.SKIPPED)
        assert free.merged.assigned == {"a.mkv": (601,), "b.mkv": (201,)}

    def test_a_later_entry_places_the_rest_of_a_pack_an_earlier_entry_partly_mapped(self) -> None:
        # The already-mapped files still count as part of the pack, so the new window maps the whole pack through
        # its aliases.
        seeded = {"Show.S00E01.mkv": (502,), "Show.S00E02.mkv": (504,)}
        stored, own = _listed_windows(_ALIASED, [502, 504], [506])

        result = place_across_windows(batch(specials_pack(1, 2, 3), seeded=seeded), (stored, own))

        assert result.merged.assigned == {"Show.S00E03.mkv": (506,)}
        assert by_name(result.merged)["Show.S00E03.mkv"][1] is PlacementVerdict.ALTERNATE

    def test_a_pack_an_earlier_entry_mapped_by_its_own_numbers_is_not_placed_by_the_aliases(self) -> None:
        # The stored record holds file 2 on special 2, its own number, where its alias says special 4. Every window
        # sees that seed, so none places the rest of the pack by the aliases.
        seeded = {"Show.S00E02.mkv": (502,)}
        stored, own = _listed_windows(_ALIASED, [502, 504], [506])

        result = place_across_windows(batch(specials_pack(1, 2, 3), seeded=seeded), (stored, own))

        assert result.merged.assigned == {}
        assert PlacementVerdict.ALTERNATE not in verdicts_of(result.merged)
