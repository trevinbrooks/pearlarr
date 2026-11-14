# pyright: strict
"""A torrent placed under several windows: each judges the whole torrent, the verdicts merge by window order."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import pytest

from pearlarr.arr_models import ParsedFileInfo
from pearlarr.numberings import NumberingProposals
from pearlarr.pending_import import FileEpisodeMap, TitleClaim, TitleNames, TitleWindow
from pearlarr.placement_types import (
    NO_EVIDENCE,
    NO_FILE_MAP,
    EpisodeIndex,
    PlacementTarget,
    PlacementVerdict,
    SeadexEvidence,
)
from pearlarr.releases import EpisodeKey
from pearlarr.window_placement import (
    WindowedPlacements,
    WindowOrder,
    ordered_windows,
    place_across_windows,
    targets_of,
)

from .builders import (
    ALIASED_SPECIALS_LISTING,
    batch,
    blind,
    by_name,
    final_windows,
    merged_of,
    numbered_names,
    parsed_info,
    parsed_specials,
    place,
    series_index,
    title_claim,
    title_window,
    torrent_listing,
    under_window,
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
# A series numbering its one season 2, so a season 1 episode resolves to nothing under it.
_OTHER_SERIES = series_index({EpisodeKey(2, n): 200 + n for n in (1, 2, 3)})
# A series of six specials, room for a listing a three-wide batch's numbers miss.
_MANY_SPECIALS = series_index({EpisodeKey(0, n): 500 + n for n in range(1, 7)})
# Another series of three specials, whose window the aliased batch's listing never covers.
_OTHER_SPECIALS = series_index({EpisodeKey(0, n): 900 + n for n in (1, 2, 3)})
# `ALIASED_SPECIALS_LISTING` as the evidence every window carries: the batch's 1, 2 and 3 aliased onto
# specials 2, 4 and 6.
_ALIASED = SeadexEvidence(ALIASED_SPECIALS_LISTING)
# `_SERIES` as Sonarr series 7, the series the window and claim builders default to.
_SERIES_7: Mapping[int, EpisodeIndex] = {7: _SERIES}


def _window(
    resolved: list[int], *titles: str, used: Iterable[int] = (), series: EpisodeIndex = _SERIES
) -> PlacementTarget:
    return PlacementTarget(resolved, series, used=frozenset(used), title_names=TitleNames("Show", titles))


def _listed_window(listing: SeadexEvidence, ids: list[int], series: EpisodeIndex = _MANY_SPECIALS) -> PlacementTarget:
    """One window of `ids` over `series`, carrying `listing`."""

    return replace(_window(ids, series=series), seadex=listing)


def _listed_windows(listing: SeadexEvidence, *windows: list[int]) -> tuple[PlacementTarget, ...]:
    """One window per id list over the many-specials series, each carrying `listing`."""

    return tuple(_listed_window(listing, ids) for ids in windows)


def _place_across(
    parsed: Mapping[str, ParsedFileInfo | None], windows: Sequence[PlacementTarget]
) -> WindowedPlacements:
    """Every parsed name placed under `windows`, with no file map."""

    return place_across_windows(batch(parsed), windows)


def _numbers_disagree_torrent() -> tuple[list[str], dict[str, ParsedFileInfo | None], tuple[PlacementTarget, ...]]:
    """A torrent of seven, S01E01 on past the season's end, under two cours whose special's absolute sits between them.

    The first cour places three, the second rejects the last four as misnumbered.
    """

    series = series_index(
        {**{EpisodeKey(1, n): 600 + n for n in range(1, 7)}, EpisodeKey(0, 1): 501},
        absolutes={601: 1, 602: 2, 603: 3, 501: 4, 604: 5, 605: 6, 606: 7},
    )
    names = [f"Show - S01E{n:02d}.mkv" for n in range(1, 8)]
    parsed: dict[str, ParsedFileInfo | None] = {
        name: parsed_info(season=1, episodes=(n,), matched=((1, n),) if n < 7 else ())
        for n, name in enumerate(names, 1)
    }
    return names, parsed, (_window(_FIRST_COUR, series=series), _window(_SECOND_COUR, series=series))


def _numbers_disagree_placed() -> tuple[list[str], WindowedPlacements]:
    """`_numbers_disagree_torrent` placed: its torrent's names and the placements."""

    names, parsed, windows = _numbers_disagree_torrent()
    return names, _place_across(parsed, windows)


class TestOneWindow:
    """The composition under one window is the placer, verdict for verdict."""

    def test_one_window_is_the_single_window_placer(self) -> None:
        # The identity covers a placed numbered sequence, a file outside the window, and an unplaced one alike.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "keyed.mkv": parsed_info(season=1, episodes=(1,)), "loose.mkv": parsed_info()}
        window = _window(_SPECIALS, used=[0])
        alone = place(parsed, window)

        result = _place_across(parsed, (window,))

        assert merged_of(result) == alone
        assert verdicts_of(alone) == {
            PlacementVerdict.SEQUENCE_OVER_PARSE,
            PlacementVerdict.OUTSIDE_WINDOW,
            PlacementVerdict.NO_EPISODE_FOUND,
        }

    def test_no_windows_leave_every_file_unplaced(self) -> None:
        result = _place_across(blind(["a.mkv", "b.mkv"]), ())

        assert by_name(merged_of(result)) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.NO_EPISODE_FOUND))


class TestSeveralWindows:
    """Every window judges the whole torrent, and window order decides what several windows would place."""

    def test_a_season_torrent_splits_across_two_cour_windows(self) -> None:
        run = numbered_names("show", 6)

        result = _place_across(blind(run), (_window(_FIRST_COUR), _window(_SECOND_COUR)))

        assert merged_of(result).placed == {name: (601 + i,) for i, name in enumerate(run)}

    def test_a_franchise_torrent_places_each_sequence_under_the_window_its_title_names(self) -> None:
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS, "Show Alpha"), _window(_FIRST_COUR, "Show Beta"))

        result = _place_across(blind([*alpha, *beta]), windows)

        assert merged_of(result).placed == {
            **{name: (501 + i,) for i, name in enumerate(alpha)},
            **{name: (601 + i,) for i, name in enumerate(beta)},
        }

    def test_a_later_windows_placement_does_not_re_judge_an_earlier_one(self) -> None:
        # A blind pair under the first window stays a tie once the second places the sequence its title names: every
        # window judges the whole torrent, since a remainder re-read as a fresh sequence (a `1..6` whose last three
        # another window placed reads as `1..3`) could index a window the whole sequence never could.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        windows = (_window(_SPECIALS), _window(_FIRST_COUR, "Show Beta"))

        result = _place_across(blind([*alpha, *beta]), windows)

        assert merged_of(result).placed == {name: (601 + i,) for i, name in enumerate(beta)}
        assert merged_of(result).unplaced == tuple(alpha)

    def test_windows_on_different_series_place_their_own_files(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), "b.mkv": parsed_info(season=2, episodes=(1,))}
        windows = (_window(_FIRST_COUR), _window([201, 202, 203], series=_OTHER_SERIES))

        result = _place_across(parsed, windows)

        assert merged_of(result).placed == {"a.mkv": (601,), "b.mkv": (201,)}

    def test_an_id_placed_under_one_window_is_used_under_an_overlapping_one(self) -> None:
        # The sequence is placed on the first three episodes under the first window. Alone, the second window's
        # pairing in order would put the blind file onto the first of them: two files never share an episode
        # across windows, so the ids the first window placed are used under the second.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "b.mkv": parsed_info()}
        wider = [*_FIRST_COUR, 604]
        assert place(parsed, _window(wider)).placed["b.mkv"] == (601,)

        result = _place_across(parsed, (_window(_FIRST_COUR), _window(wider)))

        assert merged_of(result).placed == {name: (601 + i,) for i, name in enumerate(run)}
        assert by_name(merged_of(result))["b.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_a_window_no_earlier_placement_used_keeps_its_name_number_passes(self) -> None:
        # A numbered sequence from anywhere fits the second window's width: nothing placed under the first is used
        # there.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = _place_across(parsed, (_window([601]), _window([701, 702, 703])))

        assert merged_of(result).placed == {"a.mkv": (601,), **{name: (701 + i,) for i, name in enumerate(run)}}

    def test_an_id_placed_under_one_window_narrows_an_overlapping_one(self) -> None:
        # The id the first window placed is used under the second, whose window is then the three
        # episodes left, and a sequence numbered as those fits it. Four wide, the sequence would fit nothing.
        run = numbered_names("sp", 4)[1:]
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,)), **blind(run)}

        result = _place_across(parsed, (_window([601]), _window([601, 602, 603, 604])))

        assert merged_of(result).placed == {"a.mkv": (601,), **{name: (602 + i,) for i, name in enumerate(run)}}

    def test_a_placed_verdict_wins_over_every_provisional_one(self) -> None:
        parsed = {"a.mkv": parsed_info(season=1, episodes=(1,))}

        result = _place_across(parsed, (_window([501]), _window([601])))

        assert merged_of(result).placed == {"a.mkv": (601,)}
        assert merged_of(result).excluded == ()

    @pytest.mark.parametrize(
        ("windows", "verdict"),
        [
            pytest.param(
                (_window([501]), _window([201], series=_OTHER_SERIES)),
                PlacementVerdict.NO_EPISODE_FOUND,
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
    def test_provisional_verdicts_merge_to_the_most_specific(
        self, windows: tuple[PlacementTarget, ...], verdict: PlacementVerdict
    ) -> None:
        # Two files with the same season and episode: the second is a duplicate wherever the first places, outside
        # the window where those resolve outside it, and `NO_EPISODE_FOUND` where they resolve to nothing.
        parsed = {name: parsed_info(season=1, episodes=(1,)) for name in ("a.mkv", "d.mkv")}

        result = _place_across(parsed, windows)

        assert by_name(merged_of(result))["d.mkv"] == ((), verdict)

    def test_an_extra_under_one_window_yields_to_another_windows_provisional_verdict(self) -> None:
        # The opening's word is a title word of the second title, which reads the file as possibly its own.
        parsed = {"show op.mkv": parsed_info()}
        windows = (_window([501]), _window([601], "Show Op", used=[601]))

        result = _place_across(parsed, windows)

        assert by_name(merged_of(result))["show op.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_every_earlier_placement_is_used_under_a_target_with_no_window(self) -> None:
        # The target with no window resolves any season and episode against the whole series: an id another window
        # placed is used there too, so the file with that season and episode cannot be placed on it a second time.
        run = numbered_names("show", 3)
        parsed = {**blind(run), "d.mkv": parsed_info(season=1, episodes=(3, 4))}

        result = _place_across(parsed, (_window([601, 602, 603, 604], used=[604]), _window([])))

        assert merged_of(result).placed == {name: (601 + i,) for i, name in enumerate(run)}
        assert by_name(merged_of(result))["d.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_an_earlier_windows_file_waiting_on_its_parse_outranks_a_later_windows_placement(self) -> None:
        # The first member's parsed season and episode would place it under the second window's exact pass: it still
        # waits on the parse.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), run[0]: parsed_info(season=1, episodes=(1,)), "x.mkv": None}

        result = _place_across(parsed, (_window(_SPECIALS), _window([601])))

        assert merged_of(result).placed == {}
        assert {by_name(merged_of(result))[name] for name in run} == {((), PlacementVerdict.WAITING_ON_PARSE)}

    def test_a_sequence_whose_numbers_disagree_places_under_the_first_cour_and_is_misnumbered_under_the_second(
        self,
    ) -> None:
        names, result = _numbers_disagree_placed()

        assert merged_of(result).placed == {name: (600 + n,) for n, name in enumerate(names[:3], 1)}
        assert {by_name(merged_of(result))[name][1] for name in names[3:]} == {PlacementVerdict.MISNUMBERED}

    def test_an_earlier_windows_misnumbered_torrent_outranks_a_later_windows_placement(self) -> None:
        # Under the first window's listing (specials 2, 4 and 6, three as the batch is wide) the batch's `1` is
        # no listed special: misnumbered, a human's. The second window, unlisted, would place it by number.
        parsed = parsed_specials(1, 2, 3)
        listed = _listed_window(SeadexEvidence(torrent_listing(*ALIASED_SPECIALS_LISTING.windows)), [502, 504])

        result = _place_across(parsed, (listed, _window([501, 502, 503], series=_MANY_SPECIALS)))

        assert merged_of(result).placed == {}
        assert verdicts_of(merged_of(result)) == {PlacementVerdict.MISNUMBERED}

    def test_an_earlier_windows_title_contradiction_yields_to_a_later_windows_placement(self) -> None:
        # Sonarr parsed the second season's name numbers as first-season absolutes, and its titled member names a
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

        assert merged_of(result).placed == {name: (700 + n,) for n, name in enumerate(sequel, 1)}

    def test_an_aliased_torrent_places_each_member_under_the_window_holding_its_special(self) -> None:
        # Each window judges the whole batch through the aliases. The member outside the first window is
        # `ALIAS_OUTSIDE_WINDOW` there and placed under the second.
        windows = _listed_windows(_ALIASED, [502, 504], [506])

        result = _place_across(parsed_specials(1, 2, 3), windows)

        assert merged_of(result).placed == {
            "Show.S00E01.mkv": (502,),
            "Show.S00E02.mkv": (504,),
            "Show.S00E03.mkv": (506,),
        }
        assert verdicts_of(merged_of(result)) == {PlacementVerdict.SPECIAL_ALIAS}

    def test_a_window_sharing_a_special_the_first_placed_still_judges_the_torrent_through_its_aliases(self) -> None:
        # The second window also admits special 2, which the first window placed. It still judges the whole batch
        # through the aliases, leaves that member unplaced, and places the other two by their aliases rather than
        # their own numbers.
        listing = SeadexEvidence(torrent_listing(title_window(1, (502, 505, 506)), aliases={1: 2, 5: 6, 6: 5}))
        windows = _listed_windows(listing, [502], [502, 505, 506])

        result = _place_across(parsed_specials(1, 5, 6), windows)

        assert [under_window(result, at).placed for at in (0, 1)] == [
            {"Show.S00E01.mkv": (502,)},
            {"Show.S00E05.mkv": (506,), "Show.S00E06.mkv": (505,)},
        ]
        assert merged_of(result).placed == {
            "Show.S00E01.mkv": (502,),
            "Show.S00E05.mkv": (506,),
            "Show.S00E06.mkv": (505,),
        }
        assert verdicts_of(merged_of(result)) == {PlacementVerdict.SPECIAL_ALIAS}

    @pytest.mark.parametrize("judged_first", [True, False], ids=["judged then unjudged", "unjudged then judged"])
    def test_a_member_the_aliases_and_another_windows_numbers_place_differently_is_misnumbered(
        self, judged_first: bool
    ) -> None:
        # The first series' window judges the batch through its aliases (one member's special lies outside it). The
        # other series' window, which the listing doesn't cover, places every member by its own number. The two
        # disagree on every member, and neither may win by coming first.
        judged = _listed_window(_ALIASED, [502, 504])
        unjudged = _listed_window(_ALIASED, [901, 902, 903], series=_OTHER_SPECIALS)
        windows = (judged, unjudged) if judged_first else (unjudged, judged)

        result = _place_across(parsed_specials(1, 2, 3), windows)

        assert merged_of(result).placed == {}
        assert verdicts_of(merged_of(result)) == {PlacementVerdict.MISNUMBERED}

    def test_a_member_whose_aliased_special_no_window_holds_stays_aliased_for_another_window_and_unplaced(self) -> None:
        # No window admits special 6, so the third member is left with the aliases' provisional verdict, never placed
        # by number.
        window = _listed_window(_ALIASED, [502, 504])

        result = _place_across(parsed_specials(1, 2, 3), (window,))

        assert by_name(merged_of(result))["Show.S00E03.mkv"] == ((), PlacementVerdict.ALIAS_OUTSIDE_WINDOW)
        assert merged_of(result).unplaced == ("Show.S00E03.mkv",)

    @pytest.mark.parametrize("judged_first", [True, False], ids=["judged then foreign", "foreign then judged"])
    def test_a_member_aliased_in_one_window_and_outside_another_stays_aliased_for_another_window(
        self, judged_first: bool
    ) -> None:
        # The other series' window holds two of its specials, so the third member reads as outside the window there.
        # The aliases' provisional verdict outranks that whichever window comes first: the file is unplaced, never
        # excluded.
        judged = _listed_window(_ALIASED, [502, 504])
        other = _listed_window(_ALIASED, [901, 902], series=_OTHER_SPECIALS)
        windows = (judged, other) if judged_first else (other, judged)

        result = _place_across(parsed_specials(1, 2, 3), windows)

        assert by_name(merged_of(result))["Show.S00E03.mkv"] == ((), PlacementVerdict.ALIAS_OUTSIDE_WINDOW)
        assert merged_of(result).excluded == ()

    def test_the_parse_flag_is_the_torrents(self) -> None:
        # An unknown parse that left nothing waiting under the first window (its window is one slot) still leaves the
        # sequence `WAITING_ON_PARSE` under the second: the flag counts every file of the torrent under every window.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "x.mkv": None}

        result = _place_across(parsed, (_window([501]), _window(_FIRST_COUR)))

        assert merged_of(result).placed == {}
        assert {by_name(merged_of(result))[name] for name in run} == {((), PlacementVerdict.WAITING_ON_PARSE)}

    def test_two_files_with_no_numbers_over_two_one_slot_windows_place_nowhere(self) -> None:
        result = _place_across(blind(["a.mkv", "b.mkv"]), (_window([501]), _window([502])))

        assert by_name(merged_of(result)) == dict.fromkeys(("a.mkv", "b.mkv"), ((), PlacementVerdict.NO_EPISODE_FOUND))


def _order(own: TitleWindow, listed: Sequence[TitleWindow], claims: Sequence[TitleClaim] = ()) -> WindowOrder:
    """`ordered_windows` over series 7, every window on it, for the grab whose title's window is `own`."""

    order = ordered_windows(torrent_listing(*listed), claims, own, _SERIES_7)
    assert order is not None
    return order


class TestOrderedWindows:
    """`ordered_windows`: one window per AniList id, season windows before specials windows, and which import."""

    def test_season_windows_run_before_specials_windows_each_in_anilist_id_order(self) -> None:
        listed = (
            title_window(1, _SPECIALS[:1]),
            title_window(4, _SECOND_COUR),
            title_window(2, _SPECIALS[1:]),
            title_window(3, _FIRST_COUR),
        )

        order = _order(listed[3], listed)

        assert [w.al_id for w in order.windows] == [3, 4, 1, 2]

    def test_an_ignored_title_runs_after_the_kept_ones_of_its_kind(self) -> None:
        listed = (
            title_window(1, _FIRST_COUR, ignored=True),
            title_window(2, _SECOND_COUR),
            title_window(3, _SPECIALS[:1], ignored=True),
            title_window(4, _SPECIALS[1:]),
        )

        order = _order(listed[1], listed)

        assert [w.al_id for w in order.windows] == [2, 1, 4, 3]

    def test_other_ids_are_every_other_windows_ids(self) -> None:
        listed = (title_window(1, _FIRST_COUR), title_window(2, _SECOND_COUR), title_window(3, _SPECIALS))

        order = _order(listed[1], listed)

        assert order.other_ids == frozenset(_FIRST_COUR + _SPECIALS)

    def test_the_grabbing_titles_window_wins_its_slot(self) -> None:
        own = title_window(1, _FIRST_COUR)

        order = _order(own, (title_window(1, _FIRST_COUR[:2]),), (title_claim(al_id=1),))

        assert order.windows == (own,)

    def test_a_claimed_titles_window_is_its_claims_and_it_imports(self) -> None:
        own = title_window(1, _FIRST_COUR)
        claim = title_claim(al_id=2, ordered_episode_ids=_SECOND_COUR[:2])

        order = _order(own, (title_window(2, _SECOND_COUR),), (claim,))

        assert (order.windows, order.imports) == ((own, claim.window), {0, 1})

    def test_a_claims_window_takes_its_listed_windows_ignored_flag(self) -> None:
        # Title 1 claimed the torrent before it was ignored. Its claim's window still imports, but runs after the
        # kept season window, where its listed window would.
        claim = title_claim(al_id=1, ordered_episode_ids=_FIRST_COUR[:2])
        listed = (title_window(1, _FIRST_COUR, ignored=True), title_window(2, _SECOND_COUR))

        order = _order(listed[1], listed, (claim,))

        assert order.windows == (listed[1], replace(claim.window, ignored=True))
        assert order.imports == {0, 1}

    def test_a_claims_window_takes_the_flag_off_an_empty_listed_window(self) -> None:
        # The listed window came back empty, so it doesn't run, but its flag still reaches the claim's window.
        claim = title_claim(al_id=1, ordered_episode_ids=_FIRST_COUR[:2])
        listed = (title_window(1, (), ignored=True), title_window(2, _SECOND_COUR))

        order = _order(listed[1], listed, (claim,))

        assert order.windows == (listed[1], replace(claim.window, ignored=True))

    def test_a_claims_window_takes_its_listed_windows_episode_count(self) -> None:
        # A claim doesn't keep AniList's count, so its window reads it off the title's listed window.
        own = title_window(1, _FIRST_COUR)
        claim = title_claim(al_id=2, ordered_episode_ids=_SECOND_COUR[:2])

        order = _order(own, (title_window(2, _SECOND_COUR, episode_count=3),), (claim,))

        assert order.windows == (own, replace(claim.window, episode_count=3))

    def test_a_listed_title_never_claimed_runs_its_listed_window_and_does_not_import(self) -> None:
        listed = title_window(2, _SECOND_COUR)

        order = _order(title_window(1, _FIRST_COUR), (listed,))

        assert order.windows[1] == listed
        assert (order.grabbing_window_at, order.imports) == (0, {0})

    def test_a_claim_its_entry_no_longer_lists_keeps_its_window(self) -> None:
        # Unchanged, so with no ignored flag and no episode count: the listing has neither for it.
        claim = title_claim(al_id=2, ordered_episode_ids=_SECOND_COUR)
        listed = title_window(1, _FIRST_COUR, episode_count=3)

        order = _order(listed, (listed,), (claim,))

        assert order.windows == (listed, claim.window)
        assert order.imports == {0, 1}

    def test_the_own_window_and_the_claims_import_wherever_they_sort(self) -> None:
        listed = (title_window(1, _FIRST_COUR),)
        claim = title_claim(al_id=2, ordered_episode_ids=_SECOND_COUR)

        order = _order(title_window(3, _SPECIALS), listed, (claim,))

        assert [w.al_id for w in order.windows] == [1, 2, 3]
        assert (order.grabbing_window_at, order.imports) == (2, {1, 2})

    def test_a_listed_window_with_no_ids_is_left_out(self) -> None:
        # It would mean no window, admitting every id of the series.
        order = _order(title_window(1, _FIRST_COUR), (title_window(2, ()),))

        assert [w.al_id for w in order.windows] == [1]

    def test_a_claim_with_no_ids_still_runs_with_the_specials_windows(self) -> None:
        # A pending import an older version wrote can hold such a claim, which admits any id.
        claim = title_claim(al_id=1)
        listed = title_window(2, _FIRST_COUR)

        order = _order(listed, (listed,), (claim,))

        assert order.windows == (listed, claim.window)
        assert order.imports == {0, 1}

    def test_a_window_on_a_series_left_unread_orders_nothing(self) -> None:
        listed = torrent_listing(title_window(2, (201,), series_id=8))

        assert ordered_windows(listed, (), title_window(1, _FIRST_COUR), _SERIES_7) is None


class TestDecidedBy:
    """Which window gave each file its final verdict, and what that means for the pending import they place for."""

    def test_each_file_names_the_first_window_to_give_it_a_final_verdict(self) -> None:
        run = numbered_names("show", 6)

        result = _place_across(
            {**blind(run), "loose.mkv": parsed_info()}, (_window(_FIRST_COUR), _window(_SECOND_COUR))
        )

        assert final_windows(result) == {**dict.fromkeys(run[:3], 0), **dict.fromkeys(run[3:], 1)}

    def test_the_pending_import_sets_aside_a_file_a_window_it_doesnt_import_placed(self) -> None:
        run = numbered_names("show", 6)
        result = _place_across(blind(run), (_window(_FIRST_COUR), _window(_SECOND_COUR)))

        placements = result.seen_by(0, {0}).placements

        assert placements.placed == {name: (601 + i,) for i, name in enumerate(run[:3])}
        assert {by_name(placements)[name] for name in run[3:]} == {((), PlacementVerdict.ANOTHER_TITLES)}

    def test_the_pending_import_sets_aside_a_hand_import_no_window_it_imports_was_involved_in(self) -> None:
        # The second cour rejected the last four as misnumbered, and the first cour read them as outside the window.
        names, result = _numbers_disagree_placed()

        placements = result.seen_by(0, {0}).placements

        assert {by_name(placements)[name][1] for name in names[3:]} == {PlacementVerdict.ANOTHER_TITLES}

    def test_the_pending_import_keeps_a_hand_import_a_window_it_imports_was_involved_in(self) -> None:
        names, result = _numbers_disagree_placed()

        for imports in ({1}, {0, 1}):
            placements = result.seen_by(1, imports).placements

            assert {by_name(placements)[name][1] for name in names[3:]} == {PlacementVerdict.MISNUMBERED}

    def test_the_pending_import_keeps_a_file_whose_numbers_disagree_a_window_it_imports_placed_by_number(self) -> None:
        # The aliases point the batch's 1 at the second window while the first places it by number: a file whose
        # aliases and numbers disagree is misnumbered, and the first window, which placed it, is involved in it.
        parsed = parsed_specials(1, 2, 3)
        result = _place_across(parsed, _listed_windows(_ALIASED, [501, 503, 505], [502, 504, 506]))

        placements = result.seen_by(0, {0}).placements

        assert by_name(placements)[next(iter(parsed))][1] is PlacementVerdict.MISNUMBERED

    def test_a_file_another_window_gave_a_final_verdict_is_another_titles_for_this_one(self) -> None:
        names, result = _numbers_disagree_placed()

        assert (result.seen_by(0, {0}).another_titles, result.seen_by(1, {1}).another_titles) == (
            frozenset(names[3:]),
            frozenset(names[:3]),
        )

    def test_a_file_another_window_left_waiting_on_its_parse_is_not_another_titles(self) -> None:
        # No window has read a file waiting on its parse yet.
        run = numbered_names("sp", 3)
        parsed = {**blind(run), "x.mkv": None}

        result = _place_across(parsed, (_window(_SPECIALS), _window([601])))

        assert {by_name(merged_of(result))[name][1] for name in run} == {PlacementVerdict.WAITING_ON_PARSE}
        assert result.seen_by(1, {1}).another_titles == frozenset()


_GAP = "show - 05.mkv"
"""An absolute-only name whose 5 the shared series resolves to nothing: every window leaves it `NO_EPISODE_FOUND`."""
_PLACED = "show - 01.mkv"
"""An absolute-only name whose 1 Sonarr matched to S01E01: the exact pass places it under the first window."""


def _before_and_after_proposals(
    parsed: Mapping[str, ParsedFileInfo | None],
    windows: Sequence[PlacementTarget],
    proposals: Mapping[str, tuple[int, ...]],
    *,
    file_map: FileEpisodeMap = NO_FILE_MAP,
) -> tuple[WindowedPlacements, WindowedPlacements]:
    """The torrent placed under `windows`, before and after the numbering step places its `proposals`."""

    torrent = batch(parsed, file_map=file_map)
    before = place_across_windows(torrent, windows)
    return before, before.with_proposals(torrent, NumberingProposals(proposals))


def _gap_and_placed() -> dict[str, ParsedFileInfo | None]:
    return {
        _GAP: parsed_info(season=0, absolutes=(5,)),
        _PLACED: parsed_info(season=0, absolutes=(1,), matched=((1, 1),)),
    }


class TestNumberingStep:
    """The numbering step's proposals place files in the gaps the windows left, and touch nothing else."""

    def test_a_file_in_a_gap_is_placed_by_the_window_holding_its_ids(self) -> None:
        before, after = _before_and_after_proposals(
            _gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,)}
        )

        assert by_name(merged_of(after)) == {
            _GAP: ((605,), PlacementVerdict.NUMBERINGS_AGREE),
            _PLACED: ((601,), PlacementVerdict.EXACT),
        }
        assert final_windows(after) == {_PLACED: 0, _GAP: 1}
        assert [under_window(after, at) for at in (0, 1)] == [under_window(before, at) for at in (0, 1)]

    def test_a_file_a_window_placed_keeps_its_placement(self) -> None:
        _, after = _before_and_after_proposals(
            _gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_PLACED: (602,)}
        )

        assert by_name(merged_of(after))[_PLACED] == ((601,), PlacementVerdict.EXACT)

    def test_a_file_a_window_left_final_unplaced_stays_final_unplaced(self) -> None:
        names, parsed, windows = _numbers_disagree_torrent()

        _, after = _before_and_after_proposals(parsed, windows, {names[6]: (606,)})

        assert by_name(merged_of(after))[names[6]] == ((), PlacementVerdict.MISNUMBERED)

    def test_a_file_aliased_outside_a_window_is_not_placed(self) -> None:
        # The aliases give the batch's 3 a provisional verdict for a window that holds no special 6, so it merges to
        # `NO_EPISODE_FOUND`. The first window was still involved in it, and the second holds the proposed id.
        windows = (_listed_window(_ALIASED, [502, 504]), _listed_window(_ALIASED, [201], series=_OTHER_SERIES))

        _, after = _before_and_after_proposals(parsed_specials(1, 2, 3), windows, {"Show.S00E03.mkv": (201,)})

        assert by_name(merged_of(after))["Show.S00E03.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_a_file_a_window_read_inside_itself_is_not_placed(self) -> None:
        # The OVA's absolute reads to S01E02 inside the window, and it lost that episode to a file left for a hand
        # import. The passes judged it by its own numbers, so the numbering step's proposal would reverse them.
        index = series_index(
            {EpisodeKey(1, 1): 601, EpisodeKey(1, 2): 602, EpisodeKey(1, 3): 603, EpisodeKey(5, 2): 702},
            absolutes={601: 1, 602: 2, 603: 3, 702: 102},
        )
        ova = "Show - 02 (OVA) [grp].mkv"
        parsed = {
            "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv": parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            ova: parsed_info(absolutes=(2,), matched=((1, 2),)),
        }

        _, after = _before_and_after_proposals(parsed, (_window(_FIRST_COUR, series=index),), {ova: (603,)})

        assert by_name(merged_of(after))[ova] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_an_id_a_window_used_is_not_used_again(self) -> None:
        _, after = _before_and_after_proposals(
            _gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (601,)}
        )

        assert by_name(merged_of(after))[_GAP] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_an_id_in_the_file_map_is_not_used_again(self) -> None:
        _, after = _before_and_after_proposals(
            {_GAP: parsed_info(season=0, absolutes=(5,))},
            (_window(_FIRST_COUR), _window(_SECOND_COUR)),
            {_GAP: (605,)},
            file_map={"old.mkv": (605,)},
        )

        assert by_name(merged_of(after))[_GAP] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_an_id_the_numbering_step_already_used_is_not_used_again(self) -> None:
        other = "show - 05 [other].mkv"
        parsed = {_GAP: parsed_info(season=0, absolutes=(5,)), other: parsed_info(season=0, absolutes=(5,))}

        _, after = _before_and_after_proposals(
            parsed, (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,), other: (605,)}
        )

        assert by_name(merged_of(after)) == {
            _GAP: ((605,), PlacementVerdict.NUMBERINGS_AGREE),
            other: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_ids_no_one_window_holds_are_not_used(self) -> None:
        # A two-number name read across the cour boundary: each id has a window, the pair has none.
        name = "show - 03-04.mkv"

        _, after = _before_and_after_proposals(
            {name: parsed_info(season=0, absolutes=(3, 4))},
            (_window(_FIRST_COUR), _window(_SECOND_COUR)),
            {name: (603, 604)},
        )

        assert by_name(merged_of(after))[name] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_a_name_titled_as_an_episode_outside_the_proposed_ids_is_not_placed(self) -> None:
        titled = series_index({EpisodeKey(1, n): 600 + n for n in range(1, 7)}, titles={604: "Final Bell"})
        name = "show - 05 - Final Bell [grp].mkv"

        _, after = _before_and_after_proposals(
            {name: parsed_info(season=0, absolutes=(5,))}, (_window(_FIRST_COUR, series=titled),), {name: (601,)}
        )

        assert by_name(merged_of(after))[name] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_a_file_placed_into_a_window_the_pending_import_does_not_import_is_another_titles(self) -> None:
        _, after = _before_and_after_proposals(
            _gap_and_placed(), (_window(_FIRST_COUR), _window(_SECOND_COUR)), {_GAP: (605,)}
        )

        assert by_name(after.seen_by(0, {0}).placements)[_GAP] == ((), PlacementVerdict.ANOTHER_TITLES)
        assert by_name(after.seen_by(1, {1}).placements)[_GAP] == ((605,), PlacementVerdict.NUMBERINGS_AGREE)


class TestTargetsOf:
    """`targets_of`: one target per title window in order, over its series' index, with its names, count and listing."""

    def test_one_target_per_window_over_its_series_index(self) -> None:
        names = TitleNames("Show", ("Show Alpha",))
        windows = (
            title_window(1, (601, 602), series_id=7, names=names, episode_count=2),
            title_window(2, (201,), series_id=8),
        )
        listing = SeadexEvidence(torrent_listing(title_window(1, (601,))), {"a.mkv": 601})

        scopes = targets_of(windows, {7: _SERIES, 8: _OTHER_SERIES}, listing)

        assert scopes == (
            PlacementTarget([601, 602], _SERIES, title_names=names, seadex=listing, episode_count=2),
            PlacementTarget([201], _OTHER_SERIES, title_names=TitleNames(), seadex=listing),
        )

    def test_a_claim_with_no_ids_gives_a_target_with_no_window(self) -> None:
        (target,) = targets_of((title_claim(series_id=7).window,), {7: _SERIES}, NO_EVIDENCE)

        assert target == PlacementTarget([], _SERIES)
        assert target.no_window


class TestFileMap:
    """A file map: its names stay put and its ids are used under every window that admits them."""

    def test_the_names_in_the_file_map_are_not_re_placed(self) -> None:
        run = numbered_names("show", 3)

        result = place_across_windows(batch(blind(run), file_map={run[0]: (601,)}), (_window(_FIRST_COUR),))

        assert [p.name for p in merged_of(result).placements] == run[1:]
        assert merged_of(result).placed == {run[1]: (602,), run[2]: (603,)}

    def test_an_id_in_the_file_map_takes_its_slot(self) -> None:
        # One slot, one blind file: the slot is gone when the file map holds it.
        result = place_across_windows(batch(blind(["a.mkv", "b.mkv"]), file_map={"a.mkv": (601,)}), (_window([601]),))

        assert by_name(merged_of(result)) == {"b.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_every_id_in_the_file_map_is_used_under_a_target_with_no_window(self) -> None:
        # The target with no window resolves any season and episode against the whole series, so any id in the file
        # map is used there: the file with that season and episode cannot land on it a second time.
        parsed = {"b.mkv": parsed_info(season=1, episodes=(1,))}

        free = _place_across(parsed, (_window([]),))
        used = place_across_windows(batch({**blind(["a.mkv"]), **parsed}, file_map={"a.mkv": (601,)}), (_window([]),))

        assert merged_of(free).placed == {"b.mkv": (601,)}
        assert by_name(merged_of(used)) == {"b.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_two_windows_on_two_series_place_each_file_under_the_first_that_resolves_it(self) -> None:
        # The second season's file resolves outside the first series' window and inside the second's. The file in
        # the file map is placed on the first-season file's episode, so that file stays loud.
        parsed = {
            "s.mkv": parsed_info(),
            "a.mkv": parsed_info(season=1, episodes=(1,)),
            "b.mkv": parsed_info(season=2, episodes=(1,)),
        }
        windows = (_window(_FIRST_COUR), _window([201, 202, 203], series=_OTHER_SERIES))

        with_file_map = place_across_windows(batch(parsed, file_map={"s.mkv": (601,)}), windows)
        free = place_across_windows(batch(parsed, file_map={"s.mkv": (602,)}), windows)

        assert merged_of(with_file_map).placed == {"b.mkv": (201,)}
        assert by_name(merged_of(with_file_map))["a.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert merged_of(free).placed == {"a.mkv": (601,), "b.mkv": (201,)}

    def test_a_later_title_places_the_rest_of_a_torrent_an_earlier_title_partly_placed(self) -> None:
        # The files already in the file map still count as part of the batch, so the new window judges the whole
        # batch through its aliases.
        file_map = {"Show.S00E01.mkv": (502,), "Show.S00E02.mkv": (504,)}
        stored, own = _listed_windows(_ALIASED, [502, 504], [506])

        result = place_across_windows(batch(parsed_specials(1, 2, 3), file_map=file_map), (stored, own))

        assert merged_of(result).placed == {"Show.S00E03.mkv": (506,)}
        assert by_name(merged_of(result))["Show.S00E03.mkv"][1] is PlacementVerdict.SPECIAL_ALIAS

    def test_a_torrent_an_earlier_title_placed_by_its_own_numbers_is_not_placed_by_the_aliases(self) -> None:
        # The pending import's file map puts file 2 on special 2, its own number, where its alias says special 4.
        # Every window sees that file map, so none places the rest of the batch by the aliases.
        file_map = {"Show.S00E02.mkv": (502,)}
        stored, own = _listed_windows(_ALIASED, [502, 504], [506])

        result = place_across_windows(batch(parsed_specials(1, 2, 3), file_map=file_map), (stored, own))

        assert merged_of(result).placed == {}
        assert PlacementVerdict.SPECIAL_ALIAS not in verdicts_of(merged_of(result))
