# pyright: strict
"""The placer's passes through `assign_episode_ids`, over synthetic parses and series maps.

The grab-time path through `place_release` is pinned by `test_grab_placement` and `test_pending_seeds`.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import pytest

from pearlarr.manual_import import EntryNames
from pearlarr.placement_types import EpisodeAssignment, Placement, PlacementVerdict, TargetScope
from pearlarr.seadex_types import EpisodeKey, MatchedEpisode, ParsedFileInfo

from .builders import blind, by_name, numbered_names, parsed_info, place, series_index

_GONE = "gone.mkv"
"""A name outside the batch: parsed blind it keeps the numberless zip out, unparsed (None) it holds the batch."""


def _zip_blocked(parsed: Mapping[str, ParsedFileInfo | None]) -> dict[str, ParsedFileInfo | None]:
    """The parses plus a blind name outside the batch, so the numberless zip never places what a pass refused."""

    return {**parsed, _GONE: parsed_info()}


class TestBorrowedPairPlacement:
    """The reading limits every pass shares: borrow cap, all-or-nothing resolution, duplicate collapse.

    Sonarr's series-matched pairs are borrowed only by a name carrying no
    numbers of its own, and only inside the record's set.
    """

    _SCOPE: ClassVar[TargetScope] = TargetScope(
        [11, 12, 13], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12, EpisodeKey(1, 3): 13})
    )

    def test_a_pair_inside_the_entry_places_exactly(self) -> None:
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        assert by_name(place(parsed, self._SCOPE)) == {"span.mkv": ((11, 12), PlacementVerdict.EXACT)}

    def test_a_pair_resolving_nowhere_in_the_series_is_skipped(self) -> None:
        # The series map knows nothing of it, so it is possibly ours, never proved another slice's.
        parsed = {"span.mkv": parsed_info(matched=((9, 9),))}

        assert by_name(place(parsed, self._SCOPE)) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_partially_resolving_span_refuses_the_whole_file(self) -> None:
        # One pair in the map, one out: placing the resolved half would half-import a multi-episode file.
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (9, 9)))}

        assert by_name(place(parsed, self._SCOPE)) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_full_season_match_never_borrows(self) -> None:
        # A bare "S01" extra matches the WHOLE season: none of its pairs place, alone or beside the episodes.
        pack = parsed_info(matched=((1, 1), (1, 2)), full_season=True)
        beside = {
            "extras.mkv": pack,
            "ep-01.mkv": parsed_info(season=0, absolutes=(1,), matched=((1, 1),)),
            "ep-02.mkv": parsed_info(season=0, absolutes=(2,), matched=((1, 2),)),
        }
        pair = TargetScope([11, 12], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12}))

        assert by_name(place({"pack.mkv": pack}, self._SCOPE)) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}
        assert by_name(place(beside, pair)) == {
            "extras.mkv": ((), PlacementVerdict.SKIPPED),
            "ep-01.mkv": ((11,), PlacementVerdict.EXACT),
            "ep-02.mkv": ((12,), PlacementVerdict.EXACT),
        }

    def test_a_full_season_name_key_is_refused_too(self) -> None:
        # The flag vetoes the name's own key too, not just the borrow.
        parsed = {"pack.mkv": parsed_info(season=1, episodes=(1,), full_season=True)}

        assert by_name(place(parsed, self._SCOPE)) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_wide_span_the_name_claims_itself_places_whole(self) -> None:
        # The cap is a BORROW limit. An explicit "E01-E04" range resolving every key inside the
        # set is a complete reading of the name's own claim, so width never refuses it.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed = {"span.mkv": parsed_info(season=1, episodes=(1, 2, 3, 4))}

        assert by_name(place(parsed, scope)) == {"span.mkv": ((11, 12, 13, 14), PlacementVerdict.EXACT)}

    def test_a_borrowed_span_just_over_the_cap_is_refused(self) -> None:
        # Four distinct pairs a NUMBERLESS name only borrowed is the season-pack shape sans flag.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed = {"pack.mkv": parsed_info(matched=tuple((1, e) for e in range(1, 5)))}

        assert by_name(place(parsed, scope)) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_triple_span_still_places(self) -> None:
        # The cap boundary: three episodes is still one file's claim.
        parsed = {"triple.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3)))}

        assert by_name(place(parsed, self._SCOPE)) == {"triple.mkv": ((11, 12, 13), PlacementVerdict.EXACT)}

    def test_duplicate_pairs_collapse_to_one_claim(self) -> None:
        # Junk wire repeats are one claim, so the file places rather than reading as a wide span.
        parsed = {"one.mkv": parsed_info(matched=((1, 1), (1, 1)))}
        one_key = TargetScope([11, 12], series_index({EpisodeKey(1, 1): 11}))

        assert by_name(place(parsed, self._SCOPE)) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}
        assert by_name(place(parsed, one_key)) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}


class TestForeignClassification:
    """`FOREIGN` needs a COMPLETE reading landing entirely outside the record's set.

    Anything less (a partial span, a veto, no reading at all) is possibly ours
    and stays `SKIPPED`, so the count legs may still place it.
    """

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(3, 1): 101, EpisodeKey(3, 12): 112, EpisodeKey(3, 13): 113}

    def test_a_clean_reading_fully_outside_is_another_slice(self) -> None:
        # In the window's season or another: the over-grab guard, identity must land INSIDE the set.
        same_season = {"other.mkv": parsed_info(season=3, episodes=(13,))}
        other_season = {"x.mkv": parsed_info(season=1, episodes=(1,))}
        specials = TargetScope([8030], series_index({EpisodeKey(0, 1): 8030, EpisodeKey(1, 1): 8033}))

        assert by_name(place(same_season, TargetScope([101], series_index(self._MAP)))) == {
            "other.mkv": ((), PlacementVerdict.FOREIGN)
        }
        assert by_name(place(other_season, specials)) == {"x.mkv": ((), PlacementVerdict.FOREIGN)}

    def test_a_reading_inside_the_entry_is_ours(self) -> None:
        parsed = {"mine.mkv": parsed_info(season=3, episodes=(1,))}

        assert by_name(place(parsed, TargetScope([101], series_index(self._MAP)))) == {
            "mine.mkv": ((101,), PlacementVerdict.EXACT)
        }

    def test_a_partially_resolving_span_stays_possibly_ours(self) -> None:
        # A double-episode with one key off the map may be partly ours, and is not provably bogus either:
        # the single-file arm refuses it whether or not the window's one id is on the map.
        parsed = {"d.mkv": parsed_info(season=3, episodes=(12, 99))}
        off_map = {"d.mkv": parsed_info(season=1, episodes=(5, 99))}

        assert by_name(place(parsed, TargetScope([101], series_index(self._MAP)))) == {
            "d.mkv": ((), PlacementVerdict.SKIPPED)
        }
        assert by_name(place(off_map, TargetScope([900], series_index({EpisodeKey(1, 5): 505})))) == {
            "d.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_a_vetoed_full_season_reading_is_still_ours(self) -> None:
        # Sonarr reads a bare "S0X" as the whole season: a missing episode token, not several episodes,
        # so the one leftover id takes the file, whether or not the season is on the map.
        unmapped = {"pack.mkv": parsed_info(season=2, episodes=(1,), full_season=True)}
        mapped = {"show S2 - OVA.mkv": parsed_info(season=2, full_season=True)}

        assert by_name(place(unmapped, TargetScope([101], series_index(self._MAP)))) == {
            "pack.mkv": ((101,), PlacementVerdict.SINGLE)
        }
        assert by_name(place(mapped, TargetScope([900], series_index({EpisodeKey(2, 1): 501})))) == {
            "show S2 - OVA.mkv": ((900,), PlacementVerdict.SINGLE)
        }

    def test_a_vetoed_wide_span_stays_possibly_ours(self) -> None:
        parsed = {"pack.mkv": parsed_info(matched=tuple((9, n) for n in range(1, 11)))}

        assert by_name(place(parsed, TargetScope([101], series_index(self._MAP)))) == {
            "pack.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_no_reading_at_all_stays_possibly_ours(self) -> None:
        # Two leftover ids keep the degenerate single-file arm out, so the classification is what is pinned.
        parsed = {"blank.mkv": parsed_info()}

        assert by_name(place(parsed, TargetScope([101, 112], series_index(self._MAP)))) == {
            "blank.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_an_empty_series_map_refuses_the_verdict(self) -> None:
        # Every key misses an unserved map, so nothing may be called another slice's.
        parsed = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert by_name(place(parsed, TargetScope([101, 112], series_index({})))) == {
            "other.mkv": ((), PlacementVerdict.SKIPPED)
        }


# Two specials, a four-episode first season, and a three-episode second, titled and numbered through.
_MAP: dict[EpisodeKey, int] = {
    EpisodeKey(0, 1): 501,
    EpisodeKey(0, 2): 502,
    **{EpisodeKey(1, n): 600 + n for n in range(1, 5)},
    **{EpisodeKey(2, n): 700 + n for n in range(1, 4)},
}
_TITLES: dict[int, str] = {
    501: "Beach Day",
    502: "Two Part: Cruel World",
    601: "Pilot Flight",
    602: "The Return",
    603: "Festival Night",
    604: "Final Bell",
    701: "New Dawn",
}
_SERIES = series_index(
    _MAP, absolutes={**{600 + n: n for n in range(1, 5)}, **{700 + n: 4 + n for n in range(1, 4)}}, titles=_TITLES
)
_FIRST_THREE = [601, 602, 603]


def _scope(
    resolved: Sequence[int],
    *,
    used: Sequence[int] = (),
    series: str = "Show",
    titles: Mapping[int, str] | None = None,
) -> TargetScope:
    """A scope over the shared series map, retitled wholesale when `titles` is given."""

    index = _SERIES if titles is None else series_index(_MAP, titles=titles)
    return TargetScope(list(resolved), index, used=frozenset(used), names=EntryNames(series))


class TestAssignEpisodeTitle:
    """The titled pass: a name whose title is one episode of the scope goes there, whatever its number said."""

    def test_a_shifted_match_yields_to_the_title(self) -> None:
        # Sonarr's absolute match landed one episode early. The title is the third episode's.
        name = "show - 03 - Festival Night [grp].mkv"

        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 2),))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_key_the_series_never_had_yields_to_the_title(self) -> None:
        # A CRC tag read as an episode key resolves nowhere. The title still names the episode.
        name = "show - the return [E8F03223].mkv"

        placed = by_name(place({name: parsed_info(episodes=(8,))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((602,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_head_alone_names_the_episode(self) -> None:
        # The group wrote the title up to the colon, as groups often do with a long title.
        name = "show - Cruel World Part [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _scope([501, 502], titles={502: "Cruel World Part: Two Lives"})))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_as_written_outranks_another_episodes_head(self) -> None:
        # A movie's title is the head of its epilogue's: the file named with it is the movie.
        name = "show - Cruel World Part [grp].mkv"
        titles = {501: "Cruel World Part: Epilogue Drama", 502: "Show: Cruel World Part"}

        placed = by_name(place({name: parsed_info()}, _scope([501, 502], titles=titles)))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_head_several_episodes_share_names_none(self) -> None:
        name = "show - Lost Girls [grp].mkv"
        titles = {501: "Lost Girls: Wall Goodbye", 502: "Lost Girls: Cruel World"}

        placed = by_name(place({name: parsed_info()}, _scope([501, 502], titles=titles)))

        assert placed == {name: ((), PlacementVerdict.SKIPPED)}

    def test_a_subtitle_alone_names_the_episode(self) -> None:
        # The name carries only the words past the title's colon, and its own number says the other special.
        name = "show S00E01 - Cruel World [grp].mkv"

        placed = by_name(place({name: parsed_info(season=0, episodes=(1,))}, _scope([501, 502])))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_two_files_carry_names_neither(self) -> None:
        # A recap repeats the title: the title names one file, and nothing says which.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - 01 - Beach Day [grp].mkv": parsed_info(),
            "show recap - 01 - Beach Day [grp].mkv": parsed_info(),
        }

        placed = by_name(place(parsed, _scope([501])))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.SKIPPED))

    def test_a_title_inside_a_double_is_no_contradiction(self) -> None:
        name = "show - 01 - Pilot Flight [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(1, 2))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((601, 602), PlacementVerdict.EXACT)}

    def test_a_double_the_title_contradicts_is_refused(self) -> None:
        # Placed as read, the two-episode file would hold an episode its title says it is not.
        name = "show - 02 - Pilot Flight [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2, 3))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((), PlacementVerdict.SKIPPED)}

    @pytest.mark.parametrize(
        ("used", "verdict"),
        [
            pytest.param((), PlacementVerdict.SKIPPED, id="an episode still open"),
            pytest.param((501,), PlacementVerdict.FOREIGN, id="the window full"),
        ],
    )
    def test_a_file_titled_as_another_slices_episode_is_foreign_only_once_nothing_is_left_for_it(
        self, used: Sequence[int], verdict: PlacementVerdict
    ) -> None:
        # The title alone never excludes a file the entry may still need.
        name = "show - pilot flight [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _scope([501], used=used)))

        assert placed == {name: ((), verdict)}

    def test_a_title_and_a_reading_both_outside_agree_the_file_is_another_slices(self) -> None:
        # Sonarr read it outside the scope too, onto a different episode than the title: neither says it is ours.
        name = "show - pilot flight [grp].mkv"

        placed = by_name(place({name: parsed_info(matched=((2, 1),))}, _scope([501])))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}

    def test_a_titled_file_takes_its_episode_from_its_keyed_neighbour(self) -> None:
        # The file's title is the next episode's, which its neighbour holds by key: the group numbered the run
        # one off, so the neighbour's key is as wrong and stays loud.
        titled, neighbour = "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            titled: parsed_info(season=1, episodes=(2,)),
            neighbour: parsed_info(season=1, episodes=(3,)),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {titled: ((603,), PlacementVerdict.EPISODE_TITLE), neighbour: ((), PlacementVerdict.SKIPPED)}

    def test_the_higher_version_of_a_titled_file_takes_the_title(self) -> None:
        # Both versions carry the title, so the title stays theirs. The v2 places, the v1 is its duplicate.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {v1: ((), PlacementVerdict.DUPLICATE), v2: ((602,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_seeded_versions_title_proves_the_leftover_its_duplicate(self) -> None:
        # The v2 was seeded on the title's episode. The v1 left over is titled as the same episode.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = by_name(place(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[v1]))

        assert placed == {v1: ((), PlacementVerdict.DUPLICATE)}


class TestAssignRefutedRun:
    """A run a member's title refutes: its count was wrong about every member's reading in the scope."""

    _RUN: ClassVar[list[str]] = numbered_names("show", 4, ("The Return", "Festival Night", "", ""))
    _PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        name: parsed_info(absolutes=(n,), matched=((1, n),)) for n, name in enumerate(_RUN, start=1)
    }

    def test_the_titled_members_place_and_the_rest_stay_loud(self) -> None:
        # Sonarr matched each file one episode early. The two titled files place by title, the untitled one
        # inside the scope is skipped (its match is as wrong), and the one read past the scope is the other slice's.
        placed = by_name(place(self._PARSED, _scope(_FIRST_THREE)))

        assert placed == {
            self._RUN[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            self._RUN[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            self._RUN[2]: ((), PlacementVerdict.SKIPPED),
            self._RUN[3]: ((), PlacementVerdict.FOREIGN),
        }

    def test_a_run_keyed_by_the_group_is_refuted_like_a_matched_one(self) -> None:
        # The group's own keys are one off, as its titles show: the untitled member's key is as wrong.
        run = numbered_names("show", 3, ("The Return", "Festival Night", ""))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n, name in enumerate(run, start=1)
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {
            run[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            run[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            run[2]: ((), PlacementVerdict.SKIPPED),
        }

    def test_a_member_its_own_title_confirms_keeps_its_key(self) -> None:
        # The numbering strays from the second file on: the first, titled as its own episode, stays placed.
        run = numbered_names("show", 3, ("Pilot Flight", "Festival Night", "Final Bell"))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n, name in enumerate(run, start=1)
        }

        placed = by_name(place(parsed, _scope([*_FIRST_THREE, 604])))

        assert placed == {
            run[0]: ((601,), PlacementVerdict.EXACT),
            run[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            run[2]: ((604,), PlacementVerdict.EPISODE_TITLE),
        }

    def test_an_import_poll_judges_the_leftover_as_the_grab_did(self) -> None:
        # The titled files were seeded. The poll places only the leftover, over the whole torrent's evidence,
        # so the untitled file's shifted match still never lands on the seed's empty episode.
        placed = by_name(place(self._PARSED, _scope(_FIRST_THREE, used=[602, 603]), to_place=self._RUN[2:]))

        assert placed == {
            self._RUN[2]: ((), PlacementVerdict.SKIPPED),
            self._RUN[3]: ((), PlacementVerdict.FOREIGN),
        }


class TestAssignRefutedZips:
    """A zip a title contradicts places nothing by count: the pair it refutes says the count is off."""

    def test_a_release_run_a_title_contradicts_places_only_the_titled_file(self) -> None:
        run = numbered_names("show", 3, ("The Return", "", ""))

        placed = by_name(place(blind(run), _scope(_FIRST_THREE)))

        assert placed == {
            run[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            run[1]: ((), PlacementVerdict.SKIPPED),
            run[2]: ((), PlacementVerdict.SKIPPED),
        }

    @pytest.mark.parametrize("absolutes", [True, False], ids=["absolute zip", "ordered zip"])
    def test_a_zip_a_title_contradicts_places_nothing(self, absolutes: bool) -> None:
        # The files zip 1:1 by count, but the first is titled as the next season's episode.
        names = ("show new dawn.mkv", "show b.mkv", "show c.mkv")
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(absolutes=(n,)) if absolutes else parsed_info() for n, name in enumerate(names, 1)
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.SKIPPED))


class TestAssignOverlappingClaims:
    """Two open files reading one episode differently: a span among them is refused, a same-span pair is not."""

    def test_a_span_holding_an_episode_another_file_reads_alone_is_refused(self) -> None:
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {span: ((), PlacementVerdict.SKIPPED), single: ((602,), PlacementVerdict.EXACT)}

    def test_an_import_poll_refuses_the_span_the_grab_did(self) -> None:
        # The single was seeded. The span still holds an episode it read differently, and one nothing has.
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[span]))

        assert placed == {span: ((), PlacementVerdict.SKIPPED)}

    def test_a_vetoed_wide_match_holds_no_episode(self) -> None:
        # A recap Sonarr matched to four episodes is past the borrow cap: its claims dispute nothing.
        span, recap = "show S01E01-E02 [grp].mkv", "show recap [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            recap: parsed_info(matched=((1, 1), (1, 2), (1, 3), (1, 4))),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {span: ((601, 602), PlacementVerdict.EXACT), recap: ((), PlacementVerdict.SKIPPED)}

    def test_a_span_only_partly_taken_is_no_duplicate(self) -> None:
        # A seed titled as the span's first episode holds it. The span's second episode has nothing else.
        span, seed = "show S01E02-E03 [grp].mkv", "show - the return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {span: parsed_info(season=1, episodes=(2, 3)), seed: parsed_info()}

        placed = by_name(place(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[span]))

        assert placed == {span: ((), PlacementVerdict.SKIPPED)}

    def test_two_versions_of_one_span_keep_their_reading(self) -> None:
        first, second = "show S01E01-E02 [grp].mkv", "show S01E01-E02 v2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1, 2)),
            second: parsed_info(season=1, episodes=(1, 2)),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE)))

        assert placed == {first: ((), PlacementVerdict.DUPLICATE), second: ((601, 602), PlacementVerdict.EXACT)}


class TestAssignExtras:
    """An extras file is set aside before any pass, and never counted by one."""

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("[grp] show - NCOP1 [bd].mkv", id="a creditless opening"),
            pytest.param("show ED2.mkv", id="an ending"),
            pytest.param("show - PV01.mkv", id="a preview"),
            pytest.param("show - CM 01.mkv", id="a commercial"),
            pytest.param("[BD Menu 1.1] show [2012].mkv", id="a menu"),
            pytest.param("show - creditless opening.mkv", id="the word creditless"),
            pytest.param("show previews.mkv", id="a plural"),
            pytest.param("[NCOP] show.mkv", id="a leading extras bracket"),
        ],
    )
    def test_an_extras_word_sets_the_file_aside(self, name: str) -> None:
        # Alone with one open episode, the file would otherwise be the single-file placement.
        assert by_name(place({name: parsed_info()}, _scope([601]))) == {name: ((), PlacementVerdict.EXTRA)}

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("[CMS] show - special.mkv", id="the release group's tag"),
            pytest.param("[CMS-Raws] show - special.mkv", id="a hyphenated tag"),
            pytest.param("show - special [CMS].mkv", id="a trailing tag"),
            pytest.param("show - special [ED38807C].mkv", id="a CRC tag"),
            pytest.param("show - black ops.mkv", id="a short form's plural"),
        ],
    )
    def test_a_tag_is_no_extras_word(self, name: str) -> None:
        assert by_name(place({name: parsed_info()}, _scope([601]))) == {name: ((601,), PlacementVerdict.SINGLE)}

    def test_a_file_sonarr_matched_by_its_own_key_is_no_extra(self) -> None:
        name = "show - S01E02 - the trailer park.mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_match_by_number_alone_leaves_an_extra_aside(self) -> None:
        # Sonarr read the menu's number as the first episode's: the name still says what the file is.
        name = "[BD Menu 01] show.mkv"

        placed = by_name(place({name: parsed_info(absolutes=(1,), matched=((1, 1),))}, _scope(_FIRST_THREE)))

        assert placed == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_title_word_of_the_entry_is_no_extras_word(self) -> None:
        name = "show op - special.mkv"

        assert by_name(place({name: parsed_info()}, _scope([601], series="Show Op"))) == {
            name: ((601,), PlacementVerdict.SINGLE)
        }

    def test_a_file_an_episode_title_names_is_never_an_extra(self) -> None:
        name = "show - preview party.mkv"

        placed = by_name(place({name: parsed_info()}, _scope([601], titles={601: "Preview Party"})))

        assert placed == {name: ((601,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_menu_named_after_the_episode_is_still_an_extra(self) -> None:
        # The title shields its own words only: the extras word stands outside them.
        name = "[bd menu 01] show - festival night.mkv"

        assert by_name(place({name: parsed_info()}, _scope([603]))) == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_title_two_files_share_still_shields_them_from_the_extras_words(self) -> None:
        # A recap repeats a title holding an extras word: neither file is titled, and neither is an extra.
        first, recap = "show - 01 - preview party [grp].mkv", "show - 02 - preview party [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1,)),
            recap: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _scope(_FIRST_THREE, titles={601: "Preview Party"})))

        assert placed == {first: ((601,), PlacementVerdict.EXACT), recap: ((602,), PlacementVerdict.EXACT)}

    def test_the_assignment_lists_the_extra_among_the_excluded(self) -> None:
        name = "show - PV.mkv"

        result = place({name: parsed_info()}, _scope([601]))

        assert result.excluded == (Placement(name, (), PlacementVerdict.EXTRA),)
        assert result.skipped == ()


class TestAssignAliasReading:
    """A name Sonarr read under a series alias: its key yields to the match moving it, numbers intact, into a season."""

    _SEQUEL: ClassVar[list[int]] = [701, 702, 703]

    def test_a_match_moving_the_key_into_the_scopes_season_places_it(self) -> None:
        # The sequel numbers itself season one. Sonarr matched the name into the season our map holds it under.
        name = "show flat S01E02 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((2, 2),))}, _scope(self._SEQUEL)))

        assert placed == {name: ((702,), PlacementVerdict.EXACT)}

    def test_a_match_changing_the_number_is_no_alias(self) -> None:
        name = "show flat S01E02 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((2, 1),))}, _scope(self._SEQUEL)))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}

    def test_a_match_onto_the_specials_is_no_alias(self) -> None:
        # A first-season key matched to a special keeps naming the season's episode: another slice's.
        name = "show S01E01 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(1,), matched=((0, 1),))}, _scope([501, 502])))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}


class TestAssignExactSeason:
    """The exact pass: a correctly named file Sonarr just couldn't match to the series."""

    def test_specials_assigned_by_exact_season_episode(self) -> None:
        parsed = {
            "s00e01.mkv": parsed_info(season=0, episodes=(1,)),
            "s00e02.mkv": parsed_info(season=0, episodes=(2,)),
        }
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032, EpisodeKey(1, 1): 8033}

        result = place(parsed, TargetScope([8030, 8031, 8032], series_index(ep_id_map)))

        assert result.assigned == {"s00e01.mkv": [8030], "s00e02.mkv": [8031]}
        assert (result.skipped, result.excluded) == ((), ())

    def test_empty_resolved_set_places_correctly_named_specials(self) -> None:
        # With no resolved set the exact pass falls back to the live series map,
        # so a correctly named file lands on its real episode instead of sticking forever.
        parsed = {
            "s00e01.mkv": parsed_info(season=0, episodes=(1,)),
            "s00e02.mkv": parsed_info(season=0, episodes=(2,)),
        }
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032}

        result = place(parsed, TargetScope([], series_index(ep_id_map)))

        assert result.assigned == {"s00e01.mkv": [8030], "s00e02.mkv": [8031]}
        assert (result.skipped, result.excluded) == ((), ())


class TestAssignAbsolute:
    """The absolute zip: absolute numbers only ORDER the files onto the resolved set, never decide identity."""

    def test_mis_numbered_specials_map_positionally(self) -> None:
        # Files named "01".."05" that are really S00E05..E09: "01" takes the first resolved episode.
        files = [f"{n:02d}.mkv" for n in range(1, 6)]
        parsed = {name: parsed_info(absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, TargetScope([8034, 8035, 8036, 8037, 8038], series_index({})))

        assert result.skipped == ()
        assert result.assigned == {
            "01.mkv": [8034],
            "02.mkv": [8035],
            "03.mkv": [8036],
            "04.mkv": [8037],
            "05.mkv": [8038],
        }

    def test_continuous_absolute_batch_spans_seasons(self) -> None:
        # The only multi-season pack trusted: a continuous 1..4 onto a season-sorted set.
        parsed = {f"e{i}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 5)}

        result = place(parsed, TargetScope([501, 502, 601, 602], series_index({})))

        assert result.assigned == {
            "e1.mkv": [501],
            "e2.mkv": [502],
            "e3.mkv": [601],
            "e4.mkv": [602],
        }

    def test_no_signal_file_refuses_the_positional_leg(self) -> None:
        # A file whose parse yields nothing could be a hiccuped real episode, so the whole zip is refused.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "c.mkv": parsed_info(),
        }

        result = place(parsed, TargetScope([501, 502], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_a_menu_file_neither_takes_a_slot_nor_refuses_the_zip(self) -> None:
        # A menu is never an episode: set aside by name before the count passes run.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "menu.mkv": parsed_info(),
        }

        result = place(parsed, TargetScope([501, 502], series_index({})))

        assert result.assigned == {"a.mkv": [501], "b.mkv": [502]}

    def test_absolute_ova_pack_maps_onto_resolved_set(self) -> None:
        # Thirteen season-0 absolute-only OVA files onto the entry's S00E16..E28, count-matched 13:13.
        files = [f"{n:02d}.mkv" for n in range(1, 14)]
        parsed = {name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, TargetScope(list(range(2090, 2103)), series_index({})))

        assert result.skipped == ()
        assert result.assigned == {f"{n:02d}.mkv": [2089 + n] for n in range(1, 14)}

    def test_mixed_exact_then_leftover_absolute(self) -> None:
        # The exact pass places the named file. The absolute file maps onto the one leftover id.
        parsed = {
            "s01e01.mkv": parsed_info(season=1, episodes=(1,)),
            "extra.mkv": parsed_info(absolutes=(2,)),
        }

        result = place(parsed, TargetScope([8033, 8044], series_index({EpisodeKey(1, 1): 8033})))

        assert result.assigned == {"s01e01.mkv": [8033], "extra.mkv": [8044]}
        assert result.skipped == ()


class TestAssignMatchedPairs:
    """The exact pass's matched-pairs fallback: Sonarr's series-matched `(season, episode)` for absolute-only names."""

    def test_multi_entry_batch_places_exactly_inside_the_set(self) -> None:
        # A batch spanning two entries plus a special, the record covering the second only:
        # in-set files place exactly and the two the map resolves OUTSIDE are another slice's.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep-11.mkv": parsed_info(season=0, absolutes=(11,), matched=((1, 11),)),
            "ep-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "ep-13.mkv": parsed_info(season=0, absolutes=(13,), matched=((1, 13),)),
            "sp-17.5.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587, EpisodeKey(0, 1): 2574}

        result = place(parsed, TargetScope([2586, 2587], series_index(ep_id_map)))

        assert result.assigned == {"ep-12.mkv": [2586], "ep-13.mkv": [2587]}
        assert result.skipped == ()
        assert {p.name: p.verdict for p in result.excluded} == {
            "ep-11.mkv": PlacementVerdict.FOREIGN,
            "sp-17.5.mkv": PlacementVerdict.FOREIGN,
        }

    def test_name_parsed_pair_beats_matched_pair(self) -> None:
        # A name that carries its own (season, episode) never defers to Sonarr's matched resolution.
        parsed = {"x.mkv": parsed_info(season=2, episodes=(5,), matched=((9, 9),))}
        ep_id_map = {EpisodeKey(2, 5): 400, EpisodeKey(9, 9): 999}

        result = place(parsed, TargetScope([400, 999], series_index(ep_id_map)))

        assert result.assigned == {"x.mkv": [400]}

    def test_matched_pairs_never_apply_unscoped(self) -> None:
        # With NO resolved set the live-map fallback trusts only a name-parsed pair.
        # Sonarr's series match must not decide identity on its own.
        parsed = {"x.mkv": parsed_info(season=0, absolutes=(3,), matched=((1, 3),))}

        result = place(parsed, TargetScope([], series_index({EpisodeKey(1, 3): 300})))

        assert result.assigned == {}
        assert result.skipped == ("x.mkv",)

    def test_partially_in_set_matched_span_is_skipped(self) -> None:
        # A matched span reaching outside the resolved set is refused whole, as a name-parsed one is.
        parsed = {"span.mkv": parsed_info(season=0, matched=((1, 1), (1, 3)))}
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 3): 503}

        result = place(parsed, TargetScope([501, 502], series_index(ep_id_map)))

        assert result.assigned == {}
        assert result.skipped == ("span.mkv",)

    def test_an_out_of_set_match_on_the_sole_file_is_foreign(self) -> None:
        # One numberless file, one leftover id, and Sonarr's title match claims an out-of-set
        # episode: a complete reading outside the entry is another slice's, never the leftover's.
        parsed = {"only.mkv": parsed_info(matched=((1, 5),))}

        result = place(parsed, TargetScope([900], series_index({EpisodeKey(1, 5): 555})))

        assert result.assigned == {}
        assert [p.verdict for p in result.excluded] == [PlacementVerdict.FOREIGN]

    @pytest.mark.parametrize(
        ("matched_id", "expected"),
        [
            # A disagreeing id is refused as possibly ours, never proven another slice's.
            pytest.param(999, ((), PlacementVerdict.SKIPPED), id="another series' id is refused"),
            pytest.param(501, ((501,), PlacementVerdict.EXACT), id="an agreeing id places"),
        ],
    )
    def test_sonarr_s_matched_id_must_agree_with_the_map(
        self, matched_id: int, expected: tuple[tuple[int, ...], PlacementVerdict]
    ) -> None:
        # Sonarr may match another series whose numbers coincide with ours: its episode id then disagrees.
        info = ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=matched_id),))

        result = place({"x.mkv": info}, TargetScope([501, 502], series_index({EpisodeKey(1, 1): 501})))

        assert by_name(result) == {"x.mkv": expected}

    def test_mixed_id_duplicate_claims_place_once(self) -> None:
        # (s,e,None) and (s,e,id) survive the triple dedup as two claims, yet name the episode once.
        # Two resolved ids keep the single-file arm out, so this pins the exact pass itself.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
            ),
        )

        result = place({"x.mkv": info}, TargetScope([501, 502], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {"x.mkv": [501]}

    def test_wrong_id_match_cannot_veto_the_single_file_fallback(self) -> None:
        # A disagreeing id refuses the CLAIM, but one numberless file and one leftover id still place.
        info = ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),))

        result = place({"only.mkv": info}, TargetScope([501], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {"only.mkv": [501]}

    def test_junk_duplicates_beyond_the_cap_still_collapse_and_place(self) -> None:
        # The cap counts DISTINCT claims: four wire duplicates of one pair are one claim, not a season pack.
        parsed = {"x.mkv": parsed_info(matched=((1, 1), (1, 1), (1, 1), (1, 1)))}

        result = place(parsed, TargetScope([501], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {"x.mkv": [501]}

    def test_mixed_id_duplicate_of_a_triple_span_still_places(self) -> None:
        # An id-bearing junk duplicate of one pair can't inflate a triple past the cap.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
                MatchedEpisode(season_number=1, episode_number=2),
                MatchedEpisode(season_number=1, episode_number=3),
            ),
        )
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502, EpisodeKey(1, 3): 503}

        result = place({"x.mkv": info}, TargetScope([501, 502, 503], series_index(ep_id_map)))

        assert result.assigned == {"x.mkv": [501, 502, 503]}

    def test_partially_resolved_double_absolute_never_half_imports(self) -> None:
        # A "12-13" file whose match resolved only E12: the borrowed span doesn't cover the absolutes.
        parsed = {"d.mkv": parsed_info(season=0, absolutes=(12, 13), matched=((1, 12),))}

        result = place(parsed, TargetScope([2586, 2587], series_index({EpisodeKey(1, 12): 2586})))

        assert result.assigned == {}
        assert result.skipped == ("d.mkv",)

    def test_fully_resolved_double_absolute_places_both(self) -> None:
        parsed = {"d.mkv": parsed_info(season=0, absolutes=(12, 13), matched=((1, 12), (1, 13)))}
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = place(parsed, TargetScope([2586, 2587], series_index(ep_id_map)))

        assert result.assigned == {"d.mkv": [2586, 2587]}
        assert result.skipped == ()

    def test_matched_span_never_half_imports_via_the_single_file_fallback(self) -> None:
        # Sonarr says the file spans E01+E02: the single-file arm honors that count even where it ignores identity.
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        result = place(parsed, TargetScope([501], series_index({EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502})))

        assert result.assigned == {}
        assert result.skipped == ("span.mkv",)

    def test_full_season_file_never_takes_the_spare_id(self) -> None:
        # The exact pass quarantines the season-pack shape. The single-file arm must not hand it the spare id.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras-s01.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3), (1, 4)), full_season=True),
            "e01.mkv": parsed_info(season=1, episodes=(1,)),
            "e02.mkv": parsed_info(season=1, episodes=(2,)),
            "e03.mkv": parsed_info(season=1, episodes=(3,)),
        }
        ep_id_map = {EpisodeKey(1, n): 500 + n for n in range(1, 5)}

        result = place(parsed, TargetScope([501, 502, 503, 504], series_index(ep_id_map)))

        assert result.assigned == {"e01.mkv": [501], "e02.mkv": [502], "e03.mkv": [503]}
        assert result.skipped == ("extras-s01.mkv",)


class TestAssignExactPrecedence:
    """Two open files resolving to one episode: `_Reading.rank` decides, and batch order only breaks a tie."""

    def test_a_corroborated_own_key_beats_a_borrowed_pair(self) -> None:
        # The "17.5 (S00E01)" special names itself. The "- 17" beside it only
        # borrowed the same pair from a match TVDB's interleaving shifted.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep - 17.mkv": parsed_info(season=0, absolutes=(17,), matched=((0, 1),)),
            "ep - 17.5 (S00E01).mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {EpisodeKey(0, 1): 2574, EpisodeKey(1, 17): 2591}

        result = place(parsed, TargetScope([2574], series_index(ep_id_map)))

        assert result.assigned == {"ep - 17.5 (S00E01).mkv": [2574]}
        assert by_name(result)["ep - 17.mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_a_borrowed_pair_beats_an_own_key_sonarr_never_matched(self) -> None:
        # A "- Bonus" whose CRC tag parsed as E8 carries a key Sonarr could not
        # match to the series. The "- 08" it collides with is the episode.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep - Bonus [E8F03223].mkv": parsed_info(season=1, episodes=(8,)),
            "ep - 08.mkv": parsed_info(season=0, absolutes=(8,), matched=((1, 8),)),
        }

        result = place(parsed, TargetScope([508], series_index({EpisodeKey(1, 8): 508})))

        assert result.assigned == {"ep - 08.mkv": [508]}
        assert by_name(result)["ep - Bonus [E8F03223].mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_batch_order_breaks_a_tie_within_a_rank(self) -> None:
        # Two own keys of one rank for one episode, corroborated or not: the first stays, the second is a duplicate.
        corroborated: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            ("a - S01E08.mkv", "b - S01E08.mkv"), parsed_info(season=1, episodes=(8,), matched=((1, 8),))
        )
        own_keys: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            ("a.mkv", "b.mkv"), parsed_info(season=0, episodes=(1,))
        )
        specials = series_index({EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(1, 1): 601})

        assert by_name(place(corroborated, TargetScope([508], series_index({EpisodeKey(1, 8): 508})))) == {
            "a - S01E08.mkv": ((508,), PlacementVerdict.EXACT),
            "b - S01E08.mkv": ((), PlacementVerdict.DUPLICATE),
        }
        assert by_name(place(own_keys, TargetScope([501, 502], specials))) == {
            "a.mkv": ((501,), PlacementVerdict.EXACT),
            "b.mkv": ((), PlacementVerdict.DUPLICATE),
        }


class TestAssignBesideForeignFiles:
    """A file whose own key reads outside the entry is another slice's: it never blocks a count pass."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 501,
        EpisodeKey(1, 2): 502,
        EpisodeKey(0, 1): 901,
        EpisodeKey(2, 1): 601,
        EpisodeKey(2, 2): 602,
    }

    def test_one_numberless_leftover_beside_foreign_files_places_single(self) -> None:
        # The OVA's entry resolves one id. The season files around it name
        # another season, so the OVA is the only contender.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - S01E01.mkv": parsed_info(season=1, episodes=(1,)),
            "show - S01E02.mkv": parsed_info(season=1, episodes=(2,)),
            "show - OVA.mkv": parsed_info(),
        }

        result = place(parsed, TargetScope([901], series_index(self._MAP)))

        assert result.assigned == {"show - OVA.mkv": [901]}
        assert [p.verdict for p in result.excluded] == [PlacementVerdict.FOREIGN, PlacementVerdict.FOREIGN]

    def test_a_borrowed_pair_outside_yields_to_an_open_file(self) -> None:
        # Sonarr matched the special elsewhere: its opinion steps aside for the
        # one file the listing leaves open, which takes the one id.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - special.mkv": parsed_info(matched=((1, 5),)),
            "show - OVA.mkv": parsed_info(),
        }
        ep_id_map = {EpisodeKey(1, 5): 555, EpisodeKey(0, 1): 901}

        result = place(parsed, TargetScope([901], series_index(ep_id_map)))

        assert result.assigned == {"show - OVA.mkv": [901]}
        assert [p.verdict for p in result.excluded] == [PlacementVerdict.FOREIGN]

    def test_the_absolute_zip_counts_only_the_contenders(self) -> None:
        # Two absolute-only leftovers zip onto the two-id window although a
        # first-season file rides in the same batch.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - S01E01.mkv": parsed_info(season=1, episodes=(1,)),
            "show - 14.mkv": parsed_info(season=0, absolutes=(14,)),
            "show - 13.mkv": parsed_info(season=0, absolutes=(13,)),
        }

        result = place(parsed, TargetScope([601, 602], series_index(self._MAP)))

        assert result.assigned == {"show - 13.mkv": [601], "show - 14.mkv": [602]}
        assert by_name(result)["show - S01E01.mkv"] == ((), PlacementVerdict.FOREIGN)


class TestDuplicateEvidence:
    """A collision is a duplicate only when the id's holder reads there too. Otherwise it is a skip to report."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}
    _KEYED = "show - S01E12 [1080p].mkv"

    @classmethod
    def _place(cls, seeded_parse: ParsedFileInfo | None) -> EpisodeAssignment:
        # The seed holds 2586 through "seeded.mkv". The keyed on-disk file resolves there by name.
        parsed = {"seeded.mkv": seeded_parse, cls._KEYED: parsed_info(season=1, episodes=(12,))}
        return place(parsed, TargetScope([2586, 2587], series_index(cls._MAP), used=frozenset({2586})), [cls._KEYED])

    def test_a_seeded_holder_reading_the_same_episode_proves_the_duplicate(self) -> None:
        result = self._place(parsed_info(season=1, episodes=(12,)))

        assert by_name(result) == {self._KEYED: ((), PlacementVerdict.DUPLICATE)}

    def test_a_positionally_seeded_holder_leaves_a_skip_the_caller_reports(self) -> None:
        # The seed zipped a numberless file onto 2586. A file naming that episode outright disagrees
        # with it, and a disagreement is reported, never persisted as an exclusion.
        result = self._place(parsed_info())

        assert by_name(result) == {self._KEYED: ((), PlacementVerdict.SKIPPED)}
        assert result.excluded == ()


class TestAssignScopeGate:
    """The scope is the whole resolved set: a fully seeded one stays enforced, and only an empty one is unscoped."""

    def test_fully_seeded_scope_never_unlocks_the_live_map(self) -> None:
        # Every resolved id is used, yet a correctly named out-of-scope file is refused, not placed on the live map.
        parsed = {"x.mkv": parsed_info(season=1, episodes=(1,))}

        result = place(parsed, TargetScope([8044], series_index({EpisodeKey(1, 1): 8033}), used=frozenset({8044})))

        assert result.assigned == {}
        assert by_name(result) == {"x.mkv": ((), PlacementVerdict.FOREIGN)}

    def test_empty_resolved_set_skips_absolute_only_files(self) -> None:
        # With no resolved set the absolute zip has nothing to index into. Absolutes never decide identity alone.
        files = [f"{n:02d}.mkv" for n in range(1, 4)]
        parsed = {name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, TargetScope([], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)


class TestAbsoluteDuplicateTell:
    """The absolute zip needs a clean 1:1 count and no absolute shared across the batch's parses, seeded or not."""

    def test_count_mismatch_skips(self) -> None:
        parsed = {"a.mkv": parsed_info(absolutes=(1,)), "b.mkv": parsed_info(absolutes=(2,))}

        result = place(parsed, TargetScope([1, 2, 3], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv"]

    def test_hiccuped_episode_parse_refuses_the_leg(self) -> None:
        # A None parse may be a real episode the parse hiccuped on. The next poll re-parses it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "c.mkv": None,
        }

        result = place(parsed, TargetScope([1, 2, 3], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_multi_absolute_file_vetoes_the_leg(self) -> None:
        # A file spanning two absolutes ("01-02") can't be placed positionally.
        parsed: dict[str, ParsedFileInfo | None] = {
            "span.mkv": parsed_info(absolutes=(1, 2)),
            "c.mkv": parsed_info(absolutes=(3,)),
        }

        result = place(parsed, TargetScope([1, 2, 3], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["c.mkv", "span.mkv"]

    def test_out_of_set_absolute_cannot_fill_in_for_a_hiccuped_episode(self) -> None:
        # An out-of-entry sibling (absolute 11, matched out of set) must not fill the count for a hiccuped E12.
        parsed: dict[str, ParsedFileInfo | None] = {
            "s-11.mkv": parsed_info(season=0, absolutes=(11,), matched=((1, 11),)),
            "e-12.mkv": None,
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586}

        result = place(parsed, TargetScope([2586], series_index(ep_id_map)))

        assert result.assigned == {}
        assert by_name(result) == {
            "s-11.mkv": ((), PlacementVerdict.FOREIGN),
            "e-12.mkv": ((), PlacementVerdict.SKIPPED),
        }

    def test_the_earlier_version_of_a_placed_file_is_the_duplicate(self) -> None:
        # The exact pass places the "- 12v2" (a later version outranks its earlier one). The "- 12"
        # shares absolute 12, so the batch-wide duplicate tell refuses the absolute zip for it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = place(parsed, TargetScope([2586, 2587], series_index(ep_id_map)))

        assert result.assigned == {"e-12v2.mkv": [2586]}
        assert by_name(result)["e-12.mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_seeded_sharer_still_vetoes_the_positional_leg(self) -> None:
        # The v1 was placed on an earlier poll, and its parse still reaches the duplicate tell. The map is
        # unserved for the pair, so nothing resolves and the count is the only thing that can decide.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = place(parsed, TargetScope([2587], series_index({})), ["e-12v2.mkv"])
        alone = place({"e-12v2.mkv": parsed["e-12v2.mkv"]}, TargetScope([2587], series_index({})))

        assert result.assigned == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.SKIPPED)}
        # The control: drop the sharer and the very same file takes the spare id.
        assert alone.assigned == {"e-12v2.mkv": [2587]}

    @pytest.mark.parametrize(
        ("name", "sharer"),
        [
            pytest.param("e-12.mkv", None, id="a blipped parse"),
            pytest.param("e-s01e12.mkv", parsed_info(season=1, episodes=(12,), offline=True), id="an offline parse"),
        ],
    )
    def test_a_sharer_parse_blind_to_absolutes_refuses_the_positional_leg(
        self, name: str, sharer: ParsedFileInfo | None
    ) -> None:
        # A parse the caller couldn't get, or the offline stand-in that knows nothing of absolutes,
        # may be hiding the duplicate: the zip fails closed.
        parsed = {name: sharer, "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),))}

        result = place(parsed, TargetScope([2587], series_index({})), ["e-12v2.mkv"])

        assert result.assigned == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_junk_duplicate_absolute_within_one_parse_does_not_veto(self) -> None:
        # One parse repeating its own absolute is wire junk, not a restart tell: the leftover still places.
        parsed: dict[str, ParsedFileInfo | None] = {
            "seeded-12.mkv": parsed_info(season=0, absolutes=(12, 12)),
            "left-13.mkv": parsed_info(season=0, absolutes=(13,)),
        }

        result = place(parsed, TargetScope([507], series_index({})), ["left-13.mkv"])

        assert result.assigned == {"left-13.mkv": [507]}
        assert result.skipped == ()

    def test_multi_absolute_seeded_sharer_still_vetoes(self) -> None:
        # Every absolute of every parse is counted, so a seeded "12-13" span shows the leftover v2's duplicate.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12-13.mkv": parsed_info(season=0, absolutes=(12, 13)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = place(parsed, TargetScope([2588], series_index({})), ["e-12v2.mkv"])

        assert result.assigned == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_two_sub_series_restart_is_refused(self) -> None:
        # Two sub-series whose numbering both restart at 1 share absolutes: the tell of a season-boundary
        # scramble, so the whole absolute zip is refused rather than mis-assigned.
        main = {f"main-{i:02d}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 4)}
        spinoff = {f"spinoff-{i:02d}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 4)}
        parsed = {**main, **spinoff}

        result = place(parsed, TargetScope([501, 502, 503, 601, 602, 603], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(parsed)


class TestAssignBogusKeyDowngrade:
    """A name key that exists nowhere in the series is noise, not identity.

    The downgrade only ever feeds the 1:1 single-file fallback. A key that
    resolves ANYWHERE in the series map stays real evidence.
    """

    def test_movie_year_bogus_key_places_the_sole_resolved_episode(self) -> None:
        # "Title.2020" parses S20E20, a key the series doesn't have: one file and one id make it numberless.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, TargetScope([900], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {"movie.mkv": [900]}
        assert result.skipped == ()

    def test_resolving_key_is_never_downgraded(self) -> None:
        # The same key EXISTS in the series, outside our set: real evidence that names the file another slice's.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, TargetScope([900], series_index({EpisodeKey(20, 20): 555})))

        assert result.assigned == {}
        assert by_name(result) == {"movie.mkv": ((), PlacementVerdict.FOREIGN)}

    def test_bogus_key_with_absolutes_is_not_downgraded(self) -> None:
        # Absolutes are real signal beside a bogus key, and the multi-absolute span keeps the zip refused too.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,), absolutes=(20, 21))}

        result = place(parsed, TargetScope([900], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {}
        assert result.skipped == ("movie.mkv",)

    def test_bogus_key_with_single_matched_pair_still_places(self) -> None:
        # The name parses a nonexistent S02E00 and Sonarr matched one pair: one pair never vetoes the fallback.
        parsed = {"sp.mkv": parsed_info(season=2, episodes=(0,), matched=((1, 5),))}

        result = place(parsed, TargetScope([900], series_index({EpisodeKey(1, 5): 555})))

        assert result.assigned == {"sp.mkv": [900]}
        assert result.skipped == ()


class TestAssignNumberlessZip:
    """The numberless zip and its single-file arm: with no number left, name order is the only signal."""

    def test_single_numberless_file_single_target_is_placed(self) -> None:
        # One leftover file, one leftover id, and Sonarr saw the name and found no number: it's that one.
        result = place({"only.mkv": ParsedFileInfo()}, TargetScope([900], series_index({})))

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == ()

    def test_single_none_parse_single_target_is_refused(self) -> None:
        # A None parse is no evidence (a blipped v2's absolute may hide behind it). An unparseable
        # name comes back as an all-empty parse, not None, and places as above.
        result = place({"only.mkv": None}, TargetScope([900], series_index({})))

        assert result.assigned == {}
        assert result.skipped == ("only.mkv",)

    def test_numberless_batch_zips_in_name_order(self) -> None:
        # Name order maps onto airing order regardless of the on-disk listing order.
        result = place(blind(["sp2.mkv", "sp1.mkv", "sp3.mkv"]), TargetScope([901, 902, 903], series_index({})))

        assert result.assigned == {"sp1.mkv": [901], "sp2.mkv": [902], "sp3.mkv": [903]}
        assert result.skipped == ()

    def test_zip_orders_digits_naturally(self) -> None:
        # "sp10" sorts after "sp2": lexical order would hand sp10 the second id.
        result = place(blind(["sp1.mkv", "sp2.mkv", "sp10.mkv"]), TargetScope([901, 902, 903], series_index({})))

        assert result.assigned == {"sp1.mkv": [901], "sp2.mkv": [902], "sp10.mkv": [903]}

    def test_mixed_batch_never_zips(self) -> None:
        # The exact pass placing one file leaves more parses than leftovers: numberless extras never fill episodes.
        parsed = {
            "e01.mkv": parsed_info(season=1, episodes=(1,)),
            "interview.mkv": parsed_info(),
            "making of.mkv": parsed_info(),
        }

        result = place(parsed, TargetScope([501, 502, 503], series_index({EpisodeKey(1, 1): 501})))

        assert result.assigned == {"e01.mkv": [501]}
        assert sorted(result.skipped) == ["interview.mkv", "making of.mkv"]

    def test_seeded_sibling_parse_kills_the_zip(self) -> None:
        # A parse for a file outside the batch proves a prior placement, so the whole zip is refused.
        parsed = blind(["seeded.mkv", "sp1.mkv", "sp2.mkv"])

        result = place(parsed, TargetScope([901, 902], series_index({})), ["sp1.mkv", "sp2.mkv"])

        assert result.assigned == {}
        assert sorted(result.skipped) == ["sp1.mkv", "sp2.mkv"]

    def test_count_mismatch_refuses_both_ways(self) -> None:
        # Two files onto one id, and one file onto two ids: nothing places off a non-1:1 count.
        surplus_files = place(blind(["sp1.mkv", "sp2.mkv"]), TargetScope([901], series_index({})))
        surplus_ids = place(blind(["sp1.mkv"]), TargetScope([901, 902], series_index({})))

        assert surplus_files.assigned == {}
        assert sorted(surplus_files.skipped) == ["sp1.mkv", "sp2.mkv"]
        assert surplus_ids.assigned == {}
        assert surplus_ids.skipped == ("sp1.mkv",)

    @pytest.mark.parametrize(
        "third",
        [
            pytest.param(None, id="a parse the caller couldn't get"),
            pytest.param(parsed_info(offline=True), id="the offline stand-in"),
        ],
    )
    def test_a_parse_that_is_no_real_numberless_read_refuses_the_zip(self, third: ParsedFileInfo | None) -> None:
        parsed = {**blind(["sp1.mkv", "sp2.mkv"]), "sp3.mkv": third}

        result = place(parsed, TargetScope([901, 902, 903], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["sp1.mkv", "sp2.mkv", "sp3.mkv"]

    def test_bogus_key_member_refuses_the_zip(self) -> None:
        # The bogus-key downgrade is 1:1 only: two movies can share one bogus key.
        parsed = {**blind(["sp1.mkv", "sp2.mkv"]), "movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, TargetScope([901, 902, 903], series_index({})))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["movie.mkv", "sp1.mkv", "sp2.mkv"]


class TestAssignTitledSingle:
    """The titled single-file arm: a one-episode window, several numberless leftovers, and the one a title names."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 901,
        EpisodeKey(2, 1): 601,
        EpisodeKey(2, 2): 602,
    }

    @classmethod
    def _place(cls, parsed: dict[str, ParsedFileInfo | None], *titles: str, series: str = "Show") -> EpisodeAssignment:
        scope = TargetScope([901], series_index(cls._MAP), names=EntryNames(series, titles))
        return place(parsed, scope)

    def test_the_leftover_a_title_names_is_the_episode(self) -> None:
        movie, live = "show - the movie [grp].mkv", "show - live [grp].mkv"

        result = self._place({movie: parsed_info(), live: parsed_info()}, "Show: The Movie")

        assert by_name(result) == {movie: ((901,), PlacementVerdict.TITLED), live: ((), PlacementVerdict.SKIPPED)}

    def test_an_extras_file_set_aside_leaves_the_other_the_single_file(self) -> None:
        # A trailer is never an episode, so the movie is the batch's one file for the one episode.
        movie, trailer = "show - the movie [grp].mkv", "show - trailer [grp].mkv"

        result = self._place({movie: parsed_info(), trailer: parsed_info()})

        assert by_name(result) == {movie: ((901,), PlacementVerdict.SINGLE), trailer: ((), PlacementVerdict.EXTRA)}

    def test_case_and_accents_fold_away(self) -> None:
        deja, live = "Show - Deja Vu [grp].mkv", "show - live [grp].mkv"

        result = self._place({deja: parsed_info(), live: parsed_info()}, "Show: D\u00e9j\u00e0 Vu")

        assert result.assigned == {deja: [901]}

    def test_a_title_that_is_the_series_names_nothing(self) -> None:
        # The entry is the series itself: no word of its title tells one file from another.
        parsed: dict[str, ParsedFileInfo | None] = {
            "long show name - sunny day [grp].mkv": parsed_info(),
            "long show name - prologue [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Long Show Name", "Nagai Show", series="Long Show Name")

        assert result.assigned == {}

    def test_a_descriptor_in_a_tag_never_names_the_series_title(self) -> None:
        # Tags drop away, leaving the bare series name, which the title's own leftover words never match.
        parsed: dict[str, ParsedFileInfo | None] = {
            "[grp] show (director's cut) (show II date to date) [bd].mkv": parsed_info(),
            "[grp] show - live [bd].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: Date to Date")

        assert result.assigned == {}

    def test_half_the_leftover_words_name_the_file(self) -> None:
        # The season words drop with the series title. "sunny day" is half the combined leftover.
        sunny, prologue = "show - sunny day [grp].mkv", "show - prologue [grp].mkv"

        result = self._place({sunny: parsed_info(), prologue: parsed_info()}, "Show 2nd Season: Sunny Day")

        assert result.assigned == {sunny: [901]}

    def test_an_extras_file_is_never_the_titled_one(self) -> None:
        # The opening's words match the title best, but an opening is never the episode.
        opening = "show II - picture in picture OP [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            opening: parsed_info(),
            "show II - live [grp].mkv": parsed_info(),
            "show II - making of [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show II: Picture in Picture")

        assert result.assigned == {}
        assert by_name(result)[opening] == ((), PlacementVerdict.EXTRA)

    def test_a_file_sharing_only_a_title_s_opening_words_is_not_named(self) -> None:
        # The romaji title opens with the franchise name, which the English series title never sheds.
        # A file carrying those words and none after them scores half the leftover yet names the franchise.
        parsed: dict[str, ParsedFileInfo | None] = {
            "[grp] nagai show ni [bd].mkv": parsed_info(),
            "[grp] long show - ova [bd].mkv": parsed_info(),
        }

        result = self._place(
            parsed, "Long Show Two: The Big Finale", "Nagai Show Ni: Ookii Ketsumatsu", series="Long Show"
        )

        assert result.assigned == {}

    def test_refusing_the_best_never_promotes_the_runner_up(self) -> None:
        # The best match shares only the title's opening words. The file carrying the title's last word
        # scores less and must not become the pick by the refusal.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - alpha beta gamma [grp].mkv": parsed_info(),
            "show - alpha ova [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: Alpha Beta Gamma OVA")

        assert result.assigned == {}

    def test_the_romaji_title_names_the_file_the_english_one_does_not(self) -> None:
        movie, special = "show movie endymion no kiseki [grp].mkv", "show movie special [grp].mkv"

        result = self._place(
            {movie: parsed_info(), special: parsed_info()}, "Show: The Miracle of Endymion", "Show: Endymion no Kiseki"
        )

        assert result.assigned == {movie: [901]}

    def test_two_leftovers_a_title_names_alike_place_nothing(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - the movie [grp].mkv": parsed_info(),
            "show - the movie [alt].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: The Movie")

        assert result.assigned == {}

    def test_without_titles_several_leftovers_stay(self) -> None:
        result = self._place({"show - the movie [grp].mkv": parsed_info(), "show - live [grp].mkv": parsed_info()})

        assert result.assigned == {}

    def test_an_unknown_parse_in_the_batch_holds_the_titled_leg(self) -> None:
        # The sibling's parse blipped this poll: it may be the episode's own numbered file, so nothing is titled yet.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - 12 [grp].mkv": None,
            "show - the movie [grp].mkv": parsed_info(),
            "show - trailer [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: The Movie")

        assert result.assigned == {}

    def test_a_full_season_match_of_another_season_is_no_span(self) -> None:
        # "show S2 - OVA": a season token and no episode, matched to every S2 episode. S2 is
        # not the entry's, so the file is one numberless leftover, and the sole one takes the window.
        name = "show S2 - OVA [grp].mkv"

        result = self._place({name: parsed_info(season=2, matched=((2, 1), (2, 2)), full_season=True)})

        assert by_name(result) == {name: ((901,), PlacementVerdict.SINGLE)}

    def test_a_full_season_match_reaching_into_the_entry_spans(self) -> None:
        # The same read over the entry's own season is the extras-of-this-season shape: several episodes.
        name = "show S2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=2, matched=((2, 1), (2, 2)), full_season=True)
        }

        result = place(parsed, TargetScope([601], series_index(self._MAP)))

        assert result.assigned == {}


def _numbered(template: str) -> list[str]:
    """Three names from `template`, numbered 1 to 3."""
    return [template.format(n=n) for n in (1, 2, 3)]


class TestReleaseNumberForms:
    """Which name shapes carry a release number, read through the run pass that consumes them.

    A `1..N` run over a 3-wide window places. A shape carrying no number forms
    no run at all, so nothing indexes the window.
    """

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
    }

    @classmethod
    def _place(cls, names: Sequence[str], *, blocked: bool = False) -> EpisodeAssignment:
        """Run three blind names against a 3-wide specials window.

        `blocked` adds a fourth parse so the numberless zip can never place a
        refused run instead, leaving the run pass the only thing under test.
        """

        parsed = blind(names)
        return place(
            _zip_blocked(parsed) if blocked else parsed, TargetScope([501, 502, 503], series_index(cls._MAP)), names
        )

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(_numbered("show - 0{n} - title [tag].mkv"), id="the middle form between dashes"),
            pytest.param(_numbered("show 0{n}v2 [tag].mkv"), id="the trailing form past a version suffix"),
            # Both forms drop the separator before the number, so all three read prefix "show".
            pytest.param(
                ["show - 01 - title [tag].mkv", "show - 02.mkv", "show - 03 - other.mkv"], id="both forms in one run"
            ),
            pytest.param(_numbered("show_-_0{n}_[bd].mkv"), id="underscores as spaces"),
            # One strip would leave "[a] (b)" behind and the trailing form would miss the number.
            pytest.param(_numbered("show - 0{n} [a] (b) [c].mkv"), id="nested trailing tags to a fixpoint"),
            # A titled keyed name carries no middle or trailing number, so its key is the count.
            pytest.param(_numbered("[grp] show - S07E0{n} - a titled episode.mkv"), id="the keyed form"),
            # The middle form would split "S01E01 - 01" into three prefixes. The key keeps them one run.
            pytest.param(_numbered("show S01E0{n} - 0{n} - title.mkv"), id="the key outranks a middle number"),
            # "S0101" parses as a whole season 101 though the release meant episode 01 of season 1.
            # The group's own digits ride behind, so the packed form is read before the trailing one.
            pytest.param(_numbered("show.S010{n}.1080p.Blu-ray.x265-grp067.mkv"), id="the packed form"),
            pytest.param(_numbered("show - S2 - 0{n} [1{n}] - title [tag].mkv"), id="past an absolute bracket"),
            pytest.param(_numbered("show - Episode 0{n} - title [tag].mkv"), id="past an episode word"),
            pytest.param(_numbered("show - Ep. 0{n} - title [tag].mkv"), id="past an abbreviated episode word"),
        ],
    )
    def test_a_name_shape_carrying_a_release_number_forms_the_run(self, names: list[str]) -> None:
        assert by_name(self._place(names)) == {
            name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(names)
        }

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(["show 2019.mkv", "show 2020.mkv", "show 2021.mkv"], id="a four-digit year"),
            # The tag strip takes the whole bracket off, so a CRC's digits never become a count.
            pytest.param(
                [f"show part {word} [ABCD123{n}].mkv" for n, word in enumerate(("one", "two", "three"), 1)],
                id="digits only inside a tail tag",
            ),
            pytest.param([f"0{n} - show.mkv" for n in (1, 2, 3)], id="a leading number"),
        ],
    )
    def test_a_name_shape_carrying_no_release_number_forms_no_run(self, names: list[str]) -> None:
        assert sorted(self._place(names, blocked=True).skipped) == sorted(names)

    def test_a_numbered_extras_run_is_set_aside(self) -> None:
        # Previews count previews: a "PV 01..03" beside three unreadable specials never takes their window.
        names = _numbered("show - PV 0{n} [tag].mkv")

        result = self._place(names, blocked=True)

        assert {by_name(result)[name] for name in names} == {((), PlacementVerdict.EXTRA)}

    @pytest.mark.parametrize(
        "template",
        ["show - 0{n}{v} [tag].mkv", "show - 0{n}{v} - title [tag].mkv"],
        ids=["trailing", "before the title"],
    )
    def test_a_later_version_displaces_the_earlier_one_in_the_run(self, template: str) -> None:
        # Two names share a number: the higher `vN` is the member and the other its duplicate once the run places.
        names = [template.format(n=n, v=v) for n, v in ((1, ""), (2, ""), (2, "v2"), (3, ""))]

        result = self._place(names)

        assert result.assigned == {names[0]: [501], names[2]: [502], names[3]: [503]}
        assert by_name(result)[names[1]] == ((), PlacementVerdict.DUPLICATE)

    def test_equal_versions_of_one_number_break_the_run(self) -> None:
        names = ["show - 01 [tag].mkv", "show - 02 [tag].mkv", "show - 02 [other].mkv", "show - 03 [tag].mkv"]

        assert self._place(names).assigned == {}


class TestAssignReleaseRun:
    """The release-run pass: the batch's one `1..N` run indexes a one-season window Sonarr read incoherently."""

    _WINDOW: ClassVar[list[int]] = [501, 502, 503]
    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
    }
    # A whole first season beside the specials, for the franchise-pack shapes.
    _SEASONED: ClassVar[dict[EpisodeKey, int]] = {**_MAP, **{EpisodeKey(1, n): 600 + n for n in (3, 4, 5, 6)}}
    # Six specials beside the six-episode season, for the specials-window shapes.
    _WIDE_SPECIALS: ClassVar[dict[EpisodeKey, int]] = {**_SEASONED, **{EpisodeKey(0, n): 500 + n for n in (4, 5, 6)}}
    _RUN: ClassVar[list[str]] = numbered_names("sp", 3)

    @classmethod
    def _scope(cls) -> TargetScope:
        return TargetScope(cls._WINDOW, series_index(cls._MAP))

    @classmethod
    def _ran(cls) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
        return {name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(cls._RUN)}

    def test_a_reading_of_nothing_lets_the_run_index_the_window(self) -> None:
        # Sonarr read no number from any member, so the release's own 1..N is all there is.
        assert by_name(place(blind(self._RUN), self._scope())) == self._ran()

    def test_matched_pairs_outside_the_window_do_not_stand(self) -> None:
        # Every member matched the same out-of-window episode: incoherent, so the run wins.
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(matched=((1, 1),)) for name in self._RUN}

        assert by_name(place(parsed, self._scope())) == self._ran()

    def test_a_members_own_key_outside_a_specials_window_is_overridden(self) -> None:
        # A TVDB-shifted special names another season. Over a season-0 window the run stands.
        parsed = blind(self._RUN)
        parsed[self._RUN[0]] = parsed_info(season=1, episodes=(1,))

        assert by_name(place(parsed, self._scope())) == self._ran()

    def test_bogus_keys_do_not_stand(self) -> None:
        # Keys that exist nowhere in the series are parse artifacts, never a reading.
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(season=20, episodes=(20,)) for name in self._RUN}

        assert by_name(place(parsed, self._scope())) == self._ran()

    def test_a_coherent_permuted_reading_stands(self) -> None:
        # Every member reads one distinct id inside the window, so Sonarr's
        # reading decides placement even though it permutes the run's order.
        parsed: dict[str, ParsedFileInfo | None] = {
            self._RUN[0]: parsed_info(season=0, episodes=(3,)),
            self._RUN[1]: parsed_info(season=0, episodes=(1,)),
            self._RUN[2]: parsed_info(season=0, episodes=(2,)),
        }

        assert by_name(place(parsed, self._scope())) == {
            self._RUN[0]: ((503,), PlacementVerdict.EXACT),
            self._RUN[1]: ((501,), PlacementVerdict.EXACT),
            self._RUN[2]: ((502,), PlacementVerdict.EXACT),
        }

    def test_two_runs_of_one_width_refuse_and_suppress_the_numbered_run(self) -> None:
        # Which run owns the window is unknowable, and the later blind pass must not guess either.
        names = [*numbered_names("a", 3), *numbered_names("b", 3)]
        result = place(blind(names), self._scope())

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(names)

    def test_a_width_one_window_refuses(self) -> None:
        # A one-id window is no run: the single-file arm is what places it.
        result = place({self._RUN[0]: parsed_info()}, TargetScope([501], series_index(self._MAP)))

        assert by_name(result) == {self._RUN[0]: ((501,), PlacementVerdict.SINGLE)}

    def test_a_gappy_window_refuses(self) -> None:
        # Episodes 1, 2, 4 are not a run's worth of consecutive slots.
        gappy = {EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(0, 4): 504, EpisodeKey(1, 1): 601}
        parsed = blind(self._RUN)
        parsed["pack.mkv"] = parsed_info(season=1, episodes=(1,))

        result = place(parsed, TargetScope([601, 501, 502, 504], series_index(gappy)))

        assert result.assigned == {"pack.mkv": [601]}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_two_season_window_refuses(self) -> None:
        # The extra parse keeps the numberless zip out, so the refusal is what is pinned.
        parsed = _zip_blocked(blind(self._RUN))

        result = place(parsed, TargetScope([501, 502, 601], series_index(self._MAP)), self._RUN)

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_member_whose_lone_absolute_disagrees_leaves_the_run(self) -> None:
        # Its "02" was not the release's count, so no full-width run is left to fit.
        parsed = blind(self._RUN)
        parsed[self._RUN[1]] = parsed_info(absolutes=(9,))

        result = place(parsed, self._scope())

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_full_season_read_never_refuses(self) -> None:
        # A member read as a whole season ("S0101" is season 101, "S1 - 02" is
        # season 1) has no episode token, so the run still indexes the window.
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=101, full_season=True) for name in self._RUN
        }
        parsed[self._RUN[1]] = parsed_info(season=1, full_season=True)

        assert by_name(place(parsed, self._scope())) == self._ran()

    def test_a_run_numbered_as_the_windows_episodes_indexes_it(self) -> None:
        # A split cour's second half counts on from the first: 12..14 over S01E12..E14.
        scope = TargetScope([612, 613, 614], series_index({EpisodeKey(1, n): 600 + n for n in (12, 13, 14)}))
        run = [f"sp - {n} [grp].mkv" for n in (12, 13, 14)]
        parsed = blind(run)

        result = place(parsed, scope)

        assert by_name(result) == {name: ((612 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(run)}

    def test_a_run_from_anywhere_indexes_a_window_sonarr_read_nothing_of(self) -> None:
        # A release numbering the whole series: 14..16 over a 3-wide window, no member read.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=7, episodes=(n,)) for n, name in zip((14, 15, 16), run, strict=True)
        }

        result = place(parsed, self._scope())

        assert by_name(result) == {name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(run)}

    def test_a_run_from_anywhere_needs_every_member_unread(self) -> None:
        # One member Sonarr did read says the numbering is not the series' own: the tier stands down.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = blind(run)
        parsed[run[0]] = parsed_info(matched=((1, 1),))

        result = place(_zip_blocked(parsed), self._scope(), run)

        assert by_name(result) == {
            run[0]: ((), PlacementVerdict.FOREIGN),
            run[1]: ((), PlacementVerdict.SKIPPED),
            run[2]: ((), PlacementVerdict.SKIPPED),
        }

    def test_a_one_to_n_run_outranks_a_run_from_anywhere(self) -> None:
        # Both fit the width. The release's own 1..N is the count, the other run is extras.
        offset = [f"extra - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = blind((*self._RUN, *offset))

        result = place(parsed, self._scope())

        assert by_name(result) == {**self._ran(), **dict.fromkeys(offset, ((), PlacementVerdict.SKIPPED))}

    def test_two_runs_from_anywhere_are_ambiguous(self) -> None:
        run_a = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        run_b = [f"extra - {n} [grp].mkv" for n in (20, 21, 22)]
        parsed = _zip_blocked(blind((*run_a, *run_b)))

        result = place(parsed, self._scope(), [*run_a, *run_b])

        assert result.assigned == {}

    def test_several_fitting_runs_leave_the_one_a_title_names(self) -> None:
        # A franchise pack: two 1..3 runs fit the window, and the entry's own title picks one.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        parsed = blind((*alpha, *beta))

        result = place(
            parsed, TargetScope(self._WINDOW, series_index(self._MAP), names=EntryNames("Show", ("Show Beta",)))
        )

        assert by_name(result) == {
            **{name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(beta)},
            **dict.fromkeys(alpha, ((), PlacementVerdict.SKIPPED)),
        }

    def test_a_title_that_is_the_series_names_no_run(self) -> None:
        # The entry is the series itself, so its title decides nothing and the pack stays ambiguous.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        parsed = blind((*alpha, *beta))

        result = place(parsed, TargetScope(self._WINDOW, series_index(self._MAP), names=EntryNames("Show", ("Show",))))

        assert result.assigned == {}

    def test_a_run_sonarr_read_whole_elsewhere_stands_aside(self) -> None:
        # A franchise pack: the AniList title names the base run, but Sonarr read that run whole into
        # season 1, so the unread sequel run takes the window instead.
        base = numbered_names("show name", 3)
        sequel = numbered_names("show name season two", 3)
        parsed: dict[str, ParsedFileInfo | None] = {
            **{name: parsed_info(absolutes=(i,), matched=((1, i),)) for i, name in enumerate(base, 1)},
            **blind(sequel),
        }
        scope = TargetScope(self._WINDOW, series_index(self._SEASONED), names=EntryNames("Show", ("Show Name",)))

        result = place(parsed, scope)

        assert by_name(result) == {
            **{name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(sequel)},
            **dict.fromkeys(base, ((), PlacementVerdict.FOREIGN)),
        }

    def test_a_run_numbered_as_the_window_outranks_a_run_from_one(self) -> None:
        # A two-cour pack for the second cour: Sonarr read the first cour whole into the first half.
        first = numbered_names("show part 1", 3)
        second = [f"show part 2 - 0{i} [grp].mkv" for i in (4, 5, 6)]
        parsed: dict[str, ParsedFileInfo | None] = {
            **{name: parsed_info(absolutes=(i,), matched=((1, i),)) for i, name in enumerate(first, 1)},
            **blind(second),
        }
        window = [604, 605, 606]

        result = place(parsed, TargetScope(window, series_index(self._SEASONED)))

        assert by_name(result) == {
            **{name: ((window[i],), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(second)},
            **dict.fromkeys(first, ((), PlacementVerdict.FOREIGN)),
        }

    def test_a_refused_title_pick_keeps_the_numbered_run_down(self) -> None:
        # Several runs fit, the title picks one, and a member's own key refuses it: the losing blind run
        # must not index the window through the numbered-run pass.
        picked = numbered_names("show wrath", 3)
        other = numbered_names("show revival", 3)
        parsed = blind((*picked, *other))
        parsed[picked[1]] = parsed_info(season=1, episodes=(2,))
        scope = TargetScope([604, 605, 606], series_index(self._SEASONED), names=EntryNames("Show", ("Show: Wrath",)))

        result = place(parsed, scope)

        assert result.assigned == {}

    def test_an_empty_series_map_refuses(self) -> None:
        # With no map there is no window to index, and the extra parse keeps the zip out.
        parsed = _zip_blocked(blind(self._RUN))

        result = place(parsed, TargetScope(self._WINDOW, series_index({})), self._RUN)

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_an_unknown_parse_holds_the_members(self) -> None:
        # One unreadable name anywhere in the batch, and no pass may place the run.
        parsed = blind(self._RUN)
        parsed[_GONE] = None

        result = place(parsed, self._scope(), self._RUN)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.HELD}

    def test_a_held_run_holds_its_superseded_versions_too(self) -> None:
        # The earlier version is the run's duplicate once it places, so it waits with the members
        # rather than taking the episode by its own exact key.
        names = [*self._RUN, "sp - 02v2 [grp].mkv"]
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((0, i),)) for i, name in enumerate(self._RUN, 1)
        }
        parsed[names[3]] = parsed_info(matched=((0, 2),))
        parsed[_GONE] = None

        result = place(parsed, self._scope(), names)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.HELD}

    def test_a_members_own_key_in_another_regular_season_stands_the_run_down(self) -> None:
        # Over a REGULAR-season window a member naming another season is
        # evidence the torrent is mislisted, so nothing is indexed onto it.
        regular = {EpisodeKey(2, 1): 701, EpisodeKey(2, 2): 702, EpisodeKey(2, 3): 703, EpisodeKey(1, 1): 601}
        parsed = blind(self._RUN)
        parsed[self._RUN[0]] = parsed_info(season=1, episodes=(1,))

        result = place(parsed, TargetScope([701, 702, 703], series_index(regular)))

        assert result.assigned == {}
        assert by_name(result)[self._RUN[0]] == ((), PlacementVerdict.FOREIGN)
        assert sorted(result.skipped) == sorted(self._RUN[1:])

    def test_a_non_member_reading_inside_the_window_refuses_the_run(self) -> None:
        # Another file owns one of the slots, so the run does not own the window whole.
        parsed = blind(self._RUN)
        parsed["extra.mkv"] = parsed_info(season=0, episodes=(2,))

        result = place(parsed, self._scope())

        assert result.assigned == {"extra.mkv": [502]}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_member_sonarr_matched_to_two_episodes_stays_in_the_run(self) -> None:
        # Sonarr's scene map reading one member as a double episode is the incoherence
        # the run overrides (measured: every such pack was N files for N episodes).
        parsed = blind(self._RUN)
        parsed[self._RUN[1]] = parsed_info(matched=((0, 2), (0, 3)))

        result = place(parsed, self._scope())

        assert result.assigned == dict(zip(self._RUN, ([501], [502], [503]), strict=True))
        assert {p.verdict for p in result.placements} == {PlacementVerdict.RELEASE_RUN}

    def test_a_name_the_parses_never_covered_holds_like_a_miss(self) -> None:
        # A batch whose parses skip a name to place is as unknown as one carrying a None.
        parsed = blind(self._RUN[:2])

        result = place(parsed, self._scope(), self._RUN)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.HELD}

    @pytest.mark.parametrize(
        ("title", "placed"),
        [
            pytest.param("Show: Wrath", True, id="names the fit"),
            pytest.param("Show: Revival", False, id="names the other"),
        ],
    )
    def test_a_title_keeps_the_fit_it_names_and_refuses_one_it_does_not(self, title: str, placed: bool) -> None:
        # One run fits the window and a wider one does not. The title naming the fit keeps it. Naming the
        # other refuses the fit (the named run is never promoted onto a window it does not fit) and the
        # numbered run stands down with it.
        fit = numbered_names("show wrath", 3)
        other = numbered_names("show revival", 4)
        parsed = blind((*fit, *other))
        scope = TargetScope(self._WINDOW, series_index(self._SEASONED), names=EntryNames("Show", (title,)))

        result = place(parsed, scope)

        assert result.assigned == ({name: [501 + i] for i, name in enumerate(fit)} if placed else {})

    @pytest.mark.parametrize("title", ["Show 3rd Season", "Show Third Season", "Show Season 3", "Show III", "Show 3"])
    def test_a_season_counted_any_way_names_the_run_counted_that_way(self, title: str) -> None:
        # AniList counts a season as it likes and a release as it likes: both fold to the plain number.
        second = numbered_names("show s2", 3)
        third = numbered_names("show s3", 3)
        parsed = blind((*second, *third))
        scope = TargetScope(self._WINDOW, series_index(self._SEASONED), names=EntryNames("Show", (title,)))

        result = place(parsed, scope)

        assert result.assigned == {name: [501 + i] for i, name in enumerate(third)}

    def test_a_whole_season_run_over_a_slice_window_places_the_slice(self) -> None:
        # A season pack listed on a cour's entry: the run counts the whole season, the window is two of
        # its episodes, and the members past the slice are the other cour's.
        run = numbered_names("show", 6)
        parsed = blind(run)

        result = place(parsed, TargetScope([603, 604], series_index(self._SEASONED)))

        assert by_name(result) == {
            run[2]: ((603,), PlacementVerdict.RELEASE_RUN),
            run[3]: ((604,), PlacementVerdict.RELEASE_RUN),
            **dict.fromkeys((run[0], run[1], run[4], run[5]), ((), PlacementVerdict.FOREIGN)),
        }

    def test_a_coherent_reading_of_the_slice_stands(self) -> None:
        run = numbered_names("show", 6)
        parsed = blind(run)
        parsed[run[2]] = parsed_info(season=1, episodes=(3,))
        parsed[run[3]] = parsed_info(season=1, episodes=(4,))

        result = place(parsed, TargetScope([603, 604], series_index(self._SEASONED)))

        assert {name: by_name(result)[name] for name in (run[2], run[3])} == {
            run[2]: ((603,), PlacementVerdict.EXACT),
            run[3]: ((604,), PlacementVerdict.EXACT),
        }

    def test_a_specials_window_takes_no_covering_run(self) -> None:
        # A `1..6` run Sonarr read nothing from, on an entry of two of the six specials: neither its count
        # nor a slice indexes the specials.
        run = numbered_names("show", 6)
        parsed = blind(run)

        result = place(parsed, TargetScope([501, 502], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.SKIPPED))

    @pytest.mark.parametrize(
        ("read", "verdict"),
        [pytest.param(6, PlacementVerdict.FOREIGN, id="whole"), pytest.param(3, PlacementVerdict.SKIPPED, id="half")],
    )
    def test_a_run_sonarr_read_into_another_season_never_covers(self, read: int, verdict: PlacementVerdict) -> None:
        # A shorts run as wide as the season, which Sonarr read into the specials: read whole, it is
        # theirs, and read only in part, it is nowhere.
        run = numbered_names("show mini", 6)
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((0, i),)) if i <= read else parsed_info() for i, name in enumerate(run, 1)
        }

        result = place(parsed, TargetScope([604, 605, 606], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), verdict))

    def test_a_run_read_across_the_season_and_the_specials_is_nowhere(self) -> None:
        # Sonarr read the first file into the season and the rest onto the specials: no one season
        # claims the run, so its files are neither the entry's nor foreign.
        run = numbered_names("show", 6)
        reads = ((1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=(pair,)) for name, pair in zip(run, reads, strict=True)
        }

        result = place(parsed, TargetScope([604, 605, 606], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.SKIPPED))

    def test_a_runs_own_lower_version_inside_the_window_does_not_refuse_it(self) -> None:
        # The displaced `- 02` reads as the second episode, as its `- 02v2` does: a member's own version
        # is no rival file, so the run still indexes the third episode Sonarr read nothing for.
        run = numbered_names("show", 6)
        later = "show - 02v2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((1, i),)) if i != 3 else parsed_info() for i, name in enumerate(run, 1)
        }
        parsed[later] = parsed_info(matched=((1, 2),))
        window = [601, 602, 603, 604, 605, 606]

        result = place(parsed, TargetScope(window, series_index(self._SEASONED)))

        members = [run[0], later, *run[2:]]
        assert by_name(result) == {
            **{name: ((ep_id,), PlacementVerdict.RELEASE_RUN) for name, ep_id in zip(members, window, strict=True)},
            run[1]: ((), PlacementVerdict.DUPLICATE),
        }

    @pytest.mark.parametrize("beside", [False, True], ids=["alone", "beside another run"])
    def test_a_run_read_partly_into_another_season_places_nothing_by_its_reads(self, beside: bool) -> None:
        # A bare sequel pack Sonarr read by absolute number, two files into the first season and the rest
        # from the sequel's start: the read that lands in the entry is as disputed as the rest.
        two_seasons = {
            **{EpisodeKey(1, n): 600 + n for n in (1, 2)},
            **{EpisodeKey(2, n): 700 + n for n in range(1, 7)},
        }
        run = numbered_names("show", 6)
        other = numbered_names("show alt", 6) if beside else []
        reads = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (2, 4))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=(pair,)) for name, pair in zip(run, reads, strict=True)
        }
        parsed.update(blind(other))

        result = place(parsed, TargetScope([704, 705, 706], series_index(two_seasons)))

        assert by_name(result) == dict.fromkeys((*run, *other), ((), PlacementVerdict.SKIPPED))

    def test_a_refused_pick_stands_the_numbered_run_down(self) -> None:
        # A keyed file inside the window refuses the season's run and takes its episode: the two blind
        # extras that now fit what is left do not fill it.
        run = numbered_names("show", 6)
        keyed = "show S01E04 [grp].mkv"
        extras = numbered_names("show extra", 2)
        parsed = blind((*run, *extras))
        parsed[keyed] = parsed_info(season=1, episodes=(4,), matched=((1, 4),))

        result = place(parsed, TargetScope([604, 605, 606], series_index(self._SEASONED)))

        assert by_name(result) == {
            keyed: ((604,), PlacementVerdict.EXACT),
            **dict.fromkeys((*run, *extras), ((), PlacementVerdict.SKIPPED)),
        }

    def test_a_season_with_a_numbering_gap_takes_no_covering_run(self) -> None:
        # TVDB skips the third episode, so the season counts five and a 1..5 run fits its count while its
        # numbers count on past the gap: the slice is not the run's fourth and fifth.
        gapped = {key: ep_id for key, ep_id in self._SEASONED.items() if key != EpisodeKey(1, 3)}
        run = numbered_names("show", 5)
        parsed = blind(run)

        result = place(parsed, TargetScope([604, 605, 606], series_index(gapped)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.SKIPPED))

    def test_a_seed_owning_part_of_the_scope_keeps_a_run_from_anywhere_out(self) -> None:
        # Three files fit a three-wide leftover by chance once a seed holds the rest of the scope:
        # the count only speaks for the whole scope.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = _zip_blocked(blind(run))
        scope = TargetScope(
            [601, 602, 603, 604, 605, 606], series_index(self._SEASONED), used=frozenset({601, 602, 603})
        )

        result = place(parsed, scope, run)

        assert result.assigned == {}


class TestAssignAbsoluteWindow:
    """The release-run pass over a window spanning seasons: the absolute numbering orders it when every slot has one."""

    _RUN: ClassVar[list[str]] = numbered_names("show", 3)
    _WINDOW: ClassVar[list[int]] = [501, 601, 602]
    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(1, 1): 601, EpisodeKey(0, 1): 501, EpisodeKey(1, 2): 602}
    """S01E01, then the special TVDB interleaved after it, then S01E02."""

    @classmethod
    def _place(cls, *absolutes: int | None, blocked: bool = False) -> EpisodeAssignment:
        """Sonarr's shifted reading over the slots carrying `absolutes`: the third file collides with the second."""

        carried = {ep_id: n for ep_id, n in zip(cls._MAP.values(), absolutes, strict=True) if n is not None}
        parsed: dict[str, ParsedFileInfo | None] = {
            cls._RUN[0]: parsed_info(matched=((1, 1),)),
            cls._RUN[1]: parsed_info(matched=((1, 2),)),
            cls._RUN[2]: parsed_info(matched=((1, 2),)),
        }
        scope = TargetScope(cls._WINDOW, series_index(cls._MAP, absolutes=carried))
        return place(_zip_blocked(parsed) if blocked else parsed, scope, cls._RUN)

    def test_the_run_follows_the_absolute_order_across_seasons(self) -> None:
        # An entry holding a special TVDB interleaved: "02" is absolute 2, the special, whatever season it sits in.
        result = self._place(1, 2, 3)

        assert result.assigned == {self._RUN[0]: [601], self._RUN[1]: [501], self._RUN[2]: [602]}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.RELEASE_RUN}

    @pytest.mark.parametrize("absolutes", [(1, None, 3), (1, 2, 4)])
    def test_without_an_absolute_order_the_reading_stands(self, absolutes: tuple[int | None, ...]) -> None:
        # A slot with no absolute, or a gap between them, leaves the run nothing to index by: the reading
        # stands, and the collided third file is the one leftover onto the one leftover slot.
        result = self._place(*absolutes, blocked=True)

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((501,), PlacementVerdict.SINGLE),
        }


class TestAssignRunTitleEvidence:
    """The release-run pass among several fitting runs: the members' episode titles pick one, or refuse the pick."""

    _WINDOW: ClassVar[list[int]] = [501, 502, 503]
    _TITLES: ClassVar[dict[int, str]] = {
        501: "Beach Day",
        502: "Hot Springs",
        503: "Festival Night",
        601: "Pilot Flight",
        602: "The Return",
    }

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
    }
    # Two 1..3 runs fitting a franchise pack's specials window: one titled as the window's episodes, one as
    # another season's.
    _INSIDE: ClassVar[list[str]] = numbered_names(
        "show a", 3, ("Beach Day 1080p", "Hot Springs 1080p", "Festival Night 1080p")
    )
    _OUTSIDE: ClassVar[list[str]] = numbered_names("show b", 3, ("Pilot Flight 1080p", "The Return 1080p", "Bonus"))

    @classmethod
    def _place(
        cls, *runs: list[str], retitled: Mapping[int, str] | None = None, blocked: bool = False
    ) -> EpisodeAssignment:
        names = [name for run in runs for name in run]
        parsed = blind(names)
        series = series_index(cls._MAP, titles={**cls._TITLES, **(retitled or {})})
        return place(_zip_blocked(parsed) if blocked else parsed, TargetScope(cls._WINDOW, series), names)

    def test_the_run_whose_members_name_the_window_is_picked(self) -> None:
        result = self._place(self._INSIDE, self._OUTSIDE)

        assert result.assigned == {name: [501 + i] for i, name in enumerate(self._INSIDE)}

    def test_the_other_runs_titled_members_are_foreign_once_the_window_is_full(self) -> None:
        result = self._place(self._INSIDE, self._OUTSIDE)

        assert [by_name(result)[name] for name in self._OUTSIDE] == [
            ((), PlacementVerdict.FOREIGN),
            ((), PlacementVerdict.FOREIGN),
            ((), PlacementVerdict.SKIPPED),
        ]

    def test_one_titled_member_places_only_itself(self) -> None:
        # One title is too little to pick a run among two, and enough to place its own file.
        inside = numbered_names("show a", 3, ("Beach Day", "", ""))

        result = self._place(inside, numbered_names("show b", 3))

        assert result.assigned == {inside[0]: [501]}
        assert by_name(result)[inside[0]] == ((501,), PlacementVerdict.EPISODE_TITLE)

    @pytest.mark.parametrize("blocked", [True, False], ids=["mixed batch", "pristine batch"])
    def test_members_naming_only_other_episodes_refuse_the_run(self, blocked: bool) -> None:
        # The one fitting run is titled as another season's episodes: nothing places, and neither the
        # numbered run nor the ordered zip of a pristine batch fills the window behind the refusal.
        result = self._place(numbered_names("show", 3, ("Pilot Flight", "The Return", "Bonus")), blocked=blocked)

        assert result.assigned == {}

    def test_a_short_subtitle_is_no_run_evidence(self) -> None:
        # The word past the colon leads two members' tails, and says too little to count against the run.
        run = numbered_names("show", 3, ("Home Alone", "Home Again", ""))

        result = self._place(run, retitled={601: "Chapter 4: Home"})

        assert result.assigned == {name: [501 + i] for i, name in enumerate(run)}

    def test_one_word_titles_still_refuse_the_run(self) -> None:
        # A title too short to place a file is still evidence against a run titled as another season's.
        run = numbered_names("show", 3, ("Pilot", "Return", "Bonus"))

        result = self._place(run, retitled={601: "Pilot", 602: "Return"}, blocked=True)

        assert result.assigned == {}

    def test_a_title_shared_by_both_sides_counts_for_neither(self) -> None:
        # A recap season repeats the titles: a member naming an episode on each side is no evidence.
        run = numbered_names("show", 3, ("Beach Day", "Hot Springs", "Bonus"))

        result = self._place(run, retitled={601: "Beach Day", 602: "Hot Springs"}, blocked=True)

        assert result.assigned == {name: [501 + i] for i, name in enumerate(run)}


class TestRereadSeasonRun:
    """A `1..N` run Sonarr matched into the one N-episode season and its specials is that season's own numbering."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(0, 1): 501, **{EpisodeKey(1, n): 600 + n for n in (1, 2, 3, 4)}}
    _SHIFTED: ClassVar[tuple[tuple[int, int], ...]] = ((1, 1), (1, 2), (0, 1), (1, 3), (1, 4))
    """Sonarr's reading after the special TVDB interleaved after the second episode shifted it."""
    _TWO_OF_FOUR: ClassVar[dict[EpisodeKey, int]] = {
        **{EpisodeKey(0, n): 500 + n for n in (1, 2, 3, 4)},
        **{EpisodeKey(1, n): 600 + n for n in (1, 2, 3, 4)},
    }
    """Four specials beside a four-episode season, the tie's shape."""
    _TIED: ClassVar[tuple[tuple[int, int], ...]] = ((1, 1), (0, 1), (1, 2), (0, 2))
    """Sonarr's reading of a four-file run, as many onto the specials as into the season."""

    @staticmethod
    def _reads(run: list[str], pairs: tuple[tuple[int, int], ...]) -> dict[str, ParsedFileInfo | None]:
        return {name: parsed_info(matched=(pair,)) for name, pair in zip(run, pairs[: len(run)], strict=True)}

    @classmethod
    def _place(
        cls, run: list[str], window: list[int], series: dict[EpisodeKey, int] | None = None
    ) -> EpisodeAssignment:
        parsed = cls._reads(run, cls._SHIFTED)
        return place(parsed, TargetScope(window, series_index(series or cls._MAP)))

    def test_the_specials_entry_gets_none_of_the_seasons_run(self) -> None:
        # Four files for a four-episode season: the third is the third episode, not the special Sonarr read.
        result = self._place(numbered_names("show", 4), [501])

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.FOREIGN}

    def test_the_seasons_entry_reads_the_run_as_numbered(self) -> None:
        run = numbered_names("show", 4)

        result = self._place(run, [601, 602, 603, 604])

        assert result.assigned == {name: [601 + i] for i, name in enumerate(run)}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.EXACT}

    def test_a_lower_version_of_a_member_is_re_read_with_it(self) -> None:
        # Sonarr shifted the `- 03` and its `- 03v2` onto the special alike: the lower version is re-read
        # as the third episode with its member, so the specials entry places neither.
        run = numbered_names("show", 4)
        later = "show - 03v2 [grp].mkv"
        parsed = self._reads(run, self._SHIFTED)
        parsed[later] = parsed_info(matched=((0, 1),))

        result = place(parsed, TargetScope([501], series_index(self._MAP)))

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.FOREIGN}

    def test_a_ties_possible_episode_keeps_another_run_out_of_the_window(self) -> None:
        # Under the tie the first file may be the season's first episode, so two blind extras that fit
        # the entry's first two episodes do not take them.
        run = numbered_names("show", 4)
        extras = numbered_names("show extra", 2)
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(extras))

        result = place(parsed, TargetScope([601, 602], series_index(self._TWO_OF_FOUR)))

        assert by_name(result) == dict.fromkeys((*run, *extras), ((), PlacementVerdict.SKIPPED))

    def test_a_title_never_places_a_ties_file_alone(self) -> None:
        # Over the one special's entry the AniList title names the run's second file: under the tie it
        # is the second episode or the second special, never the first special.
        series = {key: ep_id for key, ep_id in self._TWO_OF_FOUR.items() if key.episode <= 2}
        run = numbered_names("show", 2)
        parsed = self._reads(run, self._TIED)
        scope = TargetScope([501], series_index(series), names=EntryNames("Show", ("Show 2",)))

        result = place(parsed, scope)

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.SKIPPED))

    def test_a_title_naming_files_that_form_no_run_refuses_the_tied_pick(self) -> None:
        # The sequel's entry lists its title, which names three loose sequel files and not the tied run:
        # the run is refused there rather than indexed by count.
        run = numbered_names("show", 4)
        loose = [f"show two - {n} [grp].mkv" for n in ("05", "06", "08")]
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(loose))
        series = {**self._TWO_OF_FOUR, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}
        scope = TargetScope([701, 702, 703, 704], series_index(series), names=EntryNames("Show", ("Show Two",)))

        result = place(parsed, scope)

        assert by_name(result) == dict.fromkeys((*run, *loose), ((), PlacementVerdict.SKIPPED))

    def test_a_ties_run_stands_aside_from_an_entry_neither_numbering_reaches(self) -> None:
        # The tie is between the first season and the specials, so over the second season's entry the
        # tied run is elsewhere and the blind sequel run indexes it.
        run = numbered_names("show", 4)
        sequel = numbered_names("show 2", 4)
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(sequel))
        series = {**self._TWO_OF_FOUR, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}

        result = place(parsed, TargetScope([701, 702, 703, 704], series_index(series)))

        assert by_name(result) == {
            **dict.fromkeys(run, ((), PlacementVerdict.SKIPPED)),
            **{name: ((701 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(sequel)},
        }

    def test_a_run_one_wider_than_the_season_carried_the_special(self) -> None:
        # Five files for four episodes and one special: Sonarr's reading stands.
        run = numbered_names("show", 5)

        result = self._place(run, [501])

        assert result.assigned == {run[2]: [501]}

    def test_another_season_of_the_same_width_does_not_block_the_reread(self) -> None:
        # Sonarr's reads name the season the run belongs to, so a second four-episode season is no rival.
        two_seasons = {**self._MAP, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}

        result = self._place(numbered_names("show", 4), [501], series=two_seasons)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.FOREIGN}

    @pytest.mark.parametrize(
        ("specials", "verdict"),
        [
            pytest.param(2, PlacementVerdict.FOREIGN, id="fewer specials"),
            pytest.param(4, PlacementVerdict.SKIPPED, id="as many specials"),
        ],
    )
    def test_as_many_onto_the_specials_as_into_the_season_is_read_by_the_specials_count(
        self, specials: int, verdict: PlacementVerdict
    ) -> None:
        # Two of four members onto two specials: the season's, unless a specials release could be as
        # wide as the run, which is a tie nothing places over the specials entry.
        series = {key: ep_id for key, ep_id in self._TWO_OF_FOUR.items() if key.season or key.episode <= specials}
        run = numbered_names("show", 4)
        parsed = self._reads(run, self._TIED)

        result = place(parsed, TargetScope([501, 502], series_index(series)))

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {verdict}


class TestAssignNumberedRun:
    """The numbered-run pass: one `1..N` run indexes a contiguous one-season window among files Sonarr read blind."""

    _WINDOW: ClassVar[list[int]] = [10370, 10371, 10372]
    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 21): 10370,
        EpisodeKey(0, 22): 10371,
        EpisodeKey(0, 23): 10372,
        EpisodeKey(0, 25): 10374,
        EpisodeKey(1, 1): 10384,
    }
    _MAIN: ClassVar[str] = "show s01e01 [bd].mkv"

    @staticmethod
    def _run(stem: str, numbers: range) -> list[str]:
        return [f"[grp] show {stem}{i} [bd 1080p x264 10bit flac].mkv" for i in numbers]

    @classmethod
    def _batch(cls, run: list[str]) -> dict[str, ParsedFileInfo | None]:
        """The season-pack file plus a blind run, in file order."""

        parsed: dict[str, ParsedFileInfo | None] = {cls._MAIN: parsed_info(season=1, episodes=(1,))}
        parsed.update(blind(run))
        return parsed

    @classmethod
    def _scope(cls) -> TargetScope:
        return TargetScope([10384, *cls._WINDOW], series_index(cls._MAP))

    def test_a_run_beside_a_season_pack_places(self) -> None:
        # The mixed batch the ordered zip refuses: the pack takes its own key and the
        # run indexes what is left, which the two-season window keeps the release-run pass out of.
        run = self._run("extra ", range(1, 4))
        result = place(self._batch(run), self._scope())

        assert result.assigned == {self._MAIN: [10384], **{name: [10370 + i] for i, name in enumerate(run)}}
        assert (result.skipped, result.excluded) == ((), ())
        assert {p.verdict for p in result.placements if p.name in run} == {PlacementVerdict.NUMBERED_RUN}

    def test_a_later_version_displaces_the_earlier_one_in_a_blind_run(self) -> None:
        # The blind run keeps the higher `vN` and leaves the lower as its duplicate, as the release run does.
        run = self._run("extra ", range(1, 4))
        later = "[grp] show extra 2v2 [bd 1080p x264 10bit flac].mkv"
        parsed = self._batch([*run, later])

        result = place(parsed, self._scope())

        assert result.assigned == {self._MAIN: [10384], run[0]: [10370], later: [10371], run[2]: [10372]}
        assert by_name(result)[run[1]] == ((), PlacementVerdict.DUPLICATE)

    def test_a_season_zero_absolute_run_still_counts_as_blind(self) -> None:
        # Absolutes alone resolve nothing, so a season-0 run reads as blind and indexes the window.
        main = "[grp] show s2 - 01 [bd].mkv"
        run = [f"[grp] ova - 0{i} [bd].mkv" for i in (1, 2, 3)]
        parsed: dict[str, ParsedFileInfo | None] = {main: parsed_info(season=1, episodes=(1,), absolutes=(1,))}
        parsed.update({name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(run)})

        result = place(parsed, self._scope())

        assert [list(p.ids) for p in result.placements] == [[10384], [10370], [10371], [10372]]

    @pytest.mark.parametrize(
        ("run", "window"),
        [
            pytest.param(_run("a ", range(1, 4)) + _run("b ", range(1, 4)), _WINDOW, id="two runs of one width"),
            # Only a 1..N run indexes a window: a 2..4 run says nothing about where it starts.
            pytest.param(_run("extra ", range(2, 5)), _WINDOW, id="a run not starting at one"),
            pytest.param(_run("extra ", range(1, 6)), _WINDOW, id="a run wider than the window"),
            pytest.param(_run("extra ", range(1, 4)), [10370, 10371, 10374], id="a gappy window"),
        ],
    )
    def test_a_run_that_cannot_index_the_window_refuses(self, run: list[str], window: list[int]) -> None:
        result = place(self._batch(run), TargetScope([10384, *window], series_index(self._MAP)))

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)

    def test_an_unknown_parse_refuses(self) -> None:
        # An unreadable name anywhere holds every count pass closed, the blind one too.
        run = self._run("extra ", range(1, 4))
        parsed = self._batch(run)
        parsed[_GONE] = None

        result = place(parsed, self._scope(), [self._MAIN, *run])

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)


class TestPlacementClassification:
    """What `finish()` calls the leftovers, and what a leftover's classification does NOT buy it earlier."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(1, 1): 601,
    }

    def test_a_file_bound_for_foreign_still_counts_against_the_ordered_zip(self) -> None:
        # Exclusion is decided LAST, so a foreign leaf is an open leftover while the
        # count passes run: the numberless pair beside it never zips.
        parsed: dict[str, ParsedFileInfo | None] = {
            "one.mkv": parsed_info(),
            "two.mkv": parsed_info(),
            "far.mkv": parsed_info(season=1, episodes=(1,)),
        }

        result = place(parsed, TargetScope([501, 502], series_index(self._MAP)))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["one.mkv", "two.mkv"]
        assert by_name(result)["far.mkv"] == ((), PlacementVerdict.FOREIGN)

    def test_a_reading_that_resolves_nowhere_stays_countable(self) -> None:
        # Nothing proved it another slice's, so the absolute zip may still place it.
        parsed: dict[str, ParsedFileInfo | None] = {"x.mkv": parsed_info(season=9, episodes=(9,), absolutes=(9,))}

        result = place(parsed, TargetScope([501], series_index(self._MAP)))

        assert by_name(result) == {"x.mkv": ((501,), PlacementVerdict.ABSOLUTE)}

    def test_the_bogus_key_single_arm_refuses_on_an_empty_map(self) -> None:
        # Over an unserved map every key "misses", so no key may be called bogus.
        parsed: dict[str, ParsedFileInfo | None] = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, TargetScope([501], series_index({})))

        assert by_name(result) == {"movie.mkv": ((), PlacementVerdict.SKIPPED)}
