# pyright: strict
"""Placement through `assign_episode_ids`: borrowed pairs, foreign files, titles, extras, the alias reading.

The passes themselves are pinned by the fixture-driven `test_manual_import_fixtures`, and the grab-time
path through `place_release` by `test_grab_placement` and `test_pending_seeds`.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import pytest

from pearlarr.manual_import import EntryNames
from pearlarr.placement_types import Placement, PlacementBatch, PlacementVerdict, TargetScope
from pearlarr.placer import assign_episode_ids
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo

from .builders import numbered_names, parsed_info, series_index


def _verdicts(
    parsed: dict[str, ParsedFileInfo | None], scope: TargetScope, to_place: Sequence[str] | None = None
) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
    """Place `to_place` (every parsed name when None) under the scope, keyed name -> (ids, verdict)."""

    result = assign_episode_ids(PlacementBatch(list(parsed) if to_place is None else list(to_place), parsed), scope)
    return {p.name: (p.ids, p.verdict) for p in result.placements}


class TestBorrowedPairPlacement:
    """The reading limits every pass shares: borrow cap, all-or-nothing resolution, duplicate collapse.

    Sonarr's series-matched pairs are borrowed only by a name carrying no
    numbers of its own, and only inside the record's set.
    """

    _SCOPE: ClassVar[TargetScope] = TargetScope(
        [11, 12, 13], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12, EpisodeKey(1, 3): 13})
    )

    def test_a_pair_inside_the_entry_places_exactly(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((11, 12), PlacementVerdict.EXACT)}

    def test_a_pair_resolving_nowhere_in_the_series_is_skipped(self) -> None:
        # D11: the series map knows nothing of it, so it is possibly ours, never proved another slice's.
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((9, 9),))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_partially_resolving_span_refuses_the_whole_file(self) -> None:
        # One pair in the map, one out: placing the resolved half would half-import a multi-episode file.
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((1, 1), (9, 9)))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_full_season_match_never_borrows(self) -> None:
        # D6: a bare "S01" extra matches the WHOLE season, so none of its pairs may place.
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(matched=((1, 1), (1, 2)), full_season=True)}

        assert _verdicts(parsed, self._SCOPE) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_full_season_name_key_is_refused_too(self) -> None:
        # D6 again, on the other claim source: the flag vetoes the name's own key, not just the borrow.
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(season=1, episodes=(1,), full_season=True)}

        assert _verdicts(parsed, self._SCOPE) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_wide_span_the_name_claims_itself_places_whole(self) -> None:
        # The cap is a BORROW limit. An explicit "E01-E04" range resolving every key inside the
        # set is a complete reading of the name's own claim, so width never refuses it.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(season=1, episodes=(1, 2, 3, 4))}

        assert _verdicts(parsed, scope) == {"span.mkv": ((11, 12, 13, 14), PlacementVerdict.EXACT)}

    def test_a_borrowed_span_just_over_the_cap_is_refused(self) -> None:
        # Four distinct pairs a NUMBERLESS name only borrowed is the season-pack shape sans flag.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(matched=tuple((1, e) for e in range(1, 5)))}

        assert _verdicts(parsed, scope) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_triple_span_still_places(self) -> None:
        # The cap boundary: three episodes is still one file's claim.
        parsed: dict[str, ParsedFileInfo | None] = {"triple.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3)))}

        assert _verdicts(parsed, self._SCOPE) == {"triple.mkv": ((11, 12, 13), PlacementVerdict.EXACT)}

    def test_duplicate_pairs_collapse_to_one_claim(self) -> None:
        # Junk wire repeats are one claim, so the file places rather than reading as a wide span.
        parsed: dict[str, ParsedFileInfo | None] = {"one.mkv": parsed_info(matched=((1, 1), (1, 1)))}

        assert _verdicts(parsed, self._SCOPE) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}


class TestForeignClassification:
    """D11: `FOREIGN` needs a COMPLETE reading landing entirely outside the record's set.

    Anything less (a partial span, a veto, no reading at all) is possibly ours
    and stays `SKIPPED`, so the count legs may still place it.
    """

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(3, 1): 101, EpisodeKey(3, 12): 112, EpisodeKey(3, 13): 113}

    def test_a_clean_reading_fully_outside_is_another_slice(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "other.mkv": ((), PlacementVerdict.FOREIGN)
        }

    def test_a_reading_inside_the_entry_is_ours(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"mine.mkv": parsed_info(season=3, episodes=(1,))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "mine.mkv": ((101,), PlacementVerdict.EXACT)
        }

    def test_a_partially_resolving_span_stays_possibly_ours(self) -> None:
        # A boundary double-episode with one pair off the map may be partly ours.
        parsed: dict[str, ParsedFileInfo | None] = {"d.mkv": parsed_info(season=3, episodes=(12, 99))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "d.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_a_vetoed_full_season_reading_is_still_ours(self) -> None:
        # Sonarr reads a bare "S0X" as the whole season: a missing episode token,
        # not several episodes, so the one leftover id takes the file.
        parsed: dict[str, ParsedFileInfo | None] = {
            "pack.mkv": parsed_info(season=2, episodes=(1,), full_season=True),
        }

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "pack.mkv": ((101,), PlacementVerdict.SINGLE)
        }

    def test_a_vetoed_wide_span_stays_possibly_ours(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {
            "pack.mkv": parsed_info(matched=tuple((9, n) for n in range(1, 11)))
        }

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "pack.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_no_reading_at_all_stays_possibly_ours(self) -> None:
        # Two leftover ids keep the degenerate single-file arm out, so the classification is what is pinned.
        parsed: dict[str, ParsedFileInfo | None] = {"blank.mkv": parsed_info()}

        assert _verdicts(parsed, TargetScope([101, 112], series_index(self._MAP))) == {
            "blank.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_an_empty_series_map_refuses_the_verdict(self) -> None:
        # D8: every key misses an unserved map, so nothing may be called another slice's.
        parsed: dict[str, ParsedFileInfo | None] = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert _verdicts(parsed, TargetScope([101, 112], series_index({}))) == {
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

        placed = _verdicts({name: parsed_info(absolutes=(3,), matched=((1, 2),))}, _scope(_FIRST_THREE))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_key_the_series_never_had_yields_to_the_title(self) -> None:
        # A CRC tag read as an episode key resolves nowhere. The title still names the episode.
        name = "show - the return [E8F03223].mkv"

        placed = _verdicts({name: parsed_info(episodes=(8,))}, _scope(_FIRST_THREE))

        assert placed == {name: ((602,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_head_alone_names_the_episode(self) -> None:
        # The group wrote the title up to the colon, as groups often do with a long title.
        name = "show - Cruel World Part [grp].mkv"

        placed = _verdicts({name: parsed_info()}, _scope([501, 502], titles={502: "Cruel World Part: Two Lives"}))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_as_written_outranks_another_episodes_head(self) -> None:
        # A movie's title is the head of its epilogue's: the file named with it is the movie.
        name = "show - Cruel World Part [grp].mkv"
        titles = {501: "Cruel World Part: Epilogue Drama", 502: "Show: Cruel World Part"}

        placed = _verdicts({name: parsed_info()}, _scope([501, 502], titles=titles))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_head_several_episodes_share_names_none(self) -> None:
        name = "show - Lost Girls [grp].mkv"
        titles = {501: "Lost Girls: Wall Goodbye", 502: "Lost Girls: Cruel World"}

        placed = _verdicts({name: parsed_info()}, _scope([501, 502], titles=titles))

        assert placed == {name: ((), PlacementVerdict.SKIPPED)}

    def test_a_subtitle_alone_names_the_episode(self) -> None:
        # The name carries only the words past the title's colon, and its own number says the other special.
        name = "show S00E01 - Cruel World [grp].mkv"

        placed = _verdicts({name: parsed_info(season=0, episodes=(1,))}, _scope([501, 502]))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_two_files_carry_names_neither(self) -> None:
        # A recap repeats the title: the title names one file, and nothing says which.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - 01 - Beach Day [grp].mkv": parsed_info(),
            "show recap - 01 - Beach Day [grp].mkv": parsed_info(),
        }

        placed = _verdicts(parsed, _scope([501]))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.SKIPPED))

    def test_a_title_inside_a_double_is_no_contradiction(self) -> None:
        name = "show - 01 - Pilot Flight [grp].mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(1, 2))}, _scope(_FIRST_THREE))

        assert placed == {name: ((601, 602), PlacementVerdict.EXACT)}

    def test_a_double_the_title_contradicts_is_refused(self) -> None:
        # Placed as read, the two-episode file would hold an episode its title says it is not.
        name = "show - 02 - Pilot Flight [grp].mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(2, 3))}, _scope(_FIRST_THREE))

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

        placed = _verdicts({name: parsed_info()}, _scope([501], used=used))

        assert placed == {name: ((), verdict)}

    def test_a_title_and_a_reading_both_outside_agree_the_file_is_another_slices(self) -> None:
        # Sonarr read it outside the scope too, onto a different episode than the title: neither says it is ours.
        name = "show - pilot flight [grp].mkv"

        placed = _verdicts({name: parsed_info(matched=((2, 1),))}, _scope([501]))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}

    def test_a_titled_file_takes_its_episode_from_its_keyed_neighbour(self) -> None:
        # The file's title is the next episode's, which its neighbour holds by key: the group numbered the run
        # one off, so the neighbour's key is as wrong and stays loud.
        titled, neighbour = "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            titled: parsed_info(season=1, episodes=(2,)),
            neighbour: parsed_info(season=1, episodes=(3,)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == {titled: ((603,), PlacementVerdict.EPISODE_TITLE), neighbour: ((), PlacementVerdict.SKIPPED)}

    def test_the_higher_version_of_a_titled_file_takes_the_title(self) -> None:
        # Both versions carry the title, so the title stays theirs. The v2 places, the v1 is its duplicate.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == {v1: ((), PlacementVerdict.DUPLICATE), v2: ((602,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_seeded_versions_title_proves_the_leftover_its_duplicate(self) -> None:
        # The v2 was seeded on the title's episode. The v1 left over is titled as the same episode.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = _verdicts(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[v1])

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
        placed = _verdicts(self._PARSED, _scope(_FIRST_THREE))

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

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

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

        placed = _verdicts(parsed, _scope([*_FIRST_THREE, 604]))

        assert placed == {
            run[0]: ((601,), PlacementVerdict.EXACT),
            run[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            run[2]: ((604,), PlacementVerdict.EPISODE_TITLE),
        }

    def test_an_import_poll_judges_the_leftover_as_the_grab_did(self) -> None:
        # The titled files were seeded. The poll places only the leftover, over the whole torrent's evidence,
        # so the untitled file's shifted match still never lands on the seed's empty episode.
        placed = _verdicts(self._PARSED, _scope(_FIRST_THREE, used=[602, 603]), to_place=self._RUN[2:])

        assert placed == {
            self._RUN[2]: ((), PlacementVerdict.SKIPPED),
            self._RUN[3]: ((), PlacementVerdict.FOREIGN),
        }


class TestAssignRefutedZips:
    """A zip a title contradicts places nothing by count: the pair it refutes says the count is off."""

    def test_a_release_run_a_title_contradicts_places_only_the_titled_file(self) -> None:
        run = numbered_names("show", 3, ("The Return", "", ""))

        placed = _verdicts({name: parsed_info() for name in run}, _scope(_FIRST_THREE))

        assert placed == {
            run[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            run[1]: ((), PlacementVerdict.SKIPPED),
            run[2]: ((), PlacementVerdict.SKIPPED),
        }

    def test_an_absolute_zip_a_title_contradicts_places_nothing(self) -> None:
        # The absolutes zip 1:1, but the first file is titled as the next season's episode.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show new dawn.mkv": parsed_info(absolutes=(1,)),
            "show b.mkv": parsed_info(absolutes=(2,)),
            "show c.mkv": parsed_info(absolutes=(3,)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.SKIPPED))

    def test_an_ordered_zip_a_title_contradicts_places_nothing(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {
            "show new dawn.mkv": parsed_info(),
            "show b.mkv": parsed_info(),
            "show c.mkv": parsed_info(),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.SKIPPED))


class TestAssignOverlappingClaims:
    """Two open files reading one episode differently: a span among them is refused, a same-span pair is not."""

    def test_a_span_holding_an_episode_another_file_reads_alone_is_refused(self) -> None:
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == {span: ((), PlacementVerdict.SKIPPED), single: ((602,), PlacementVerdict.EXACT)}

    def test_an_import_poll_refuses_the_span_the_grab_did(self) -> None:
        # The single was seeded. The span still holds an episode it read differently, and one nothing has.
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[span])

        assert placed == {span: ((), PlacementVerdict.SKIPPED)}

    def test_a_vetoed_wide_match_holds_no_episode(self) -> None:
        # A recap Sonarr matched to four episodes is past the borrow cap: its claims dispute nothing.
        span, recap = "show S01E01-E02 [grp].mkv", "show recap [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            recap: parsed_info(matched=((1, 1), (1, 2), (1, 3), (1, 4))),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

        assert placed == {span: ((601, 602), PlacementVerdict.EXACT), recap: ((), PlacementVerdict.SKIPPED)}

    def test_a_span_only_partly_taken_is_no_duplicate(self) -> None:
        # A seed titled as the span's first episode holds it. The span's second episode has nothing else.
        span, seed = "show S01E02-E03 [grp].mkv", "show - the return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {span: parsed_info(season=1, episodes=(2, 3)), seed: parsed_info()}

        placed = _verdicts(parsed, _scope(_FIRST_THREE, used=[602]), to_place=[span])

        assert placed == {span: ((), PlacementVerdict.SKIPPED)}

    def test_two_versions_of_one_span_keep_their_reading(self) -> None:
        first, second = "show S01E01-E02 [grp].mkv", "show S01E01-E02 v2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1, 2)),
            second: parsed_info(season=1, episodes=(1, 2)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE))

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
        assert _verdicts({name: parsed_info()}, _scope([601])) == {name: ((), PlacementVerdict.EXTRA)}

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
        assert _verdicts({name: parsed_info()}, _scope([601])) == {name: ((601,), PlacementVerdict.SINGLE)}

    def test_a_file_sonarr_matched_by_its_own_key_is_no_extra(self) -> None:
        name = "show - S01E02 - the trailer park.mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, _scope(_FIRST_THREE))

        assert placed == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_match_by_number_alone_leaves_an_extra_aside(self) -> None:
        # Sonarr read the menu's number as the first episode's: the name still says what the file is.
        name = "[BD Menu 01] show.mkv"

        placed = _verdicts({name: parsed_info(absolutes=(1,), matched=((1, 1),))}, _scope(_FIRST_THREE))

        assert placed == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_title_word_of_the_entry_is_no_extras_word(self) -> None:
        name = "show op - special.mkv"

        assert _verdicts({name: parsed_info()}, _scope([601], series="Show Op")) == {
            name: ((601,), PlacementVerdict.SINGLE)
        }

    def test_a_file_an_episode_title_names_is_never_an_extra(self) -> None:
        name = "show - preview party.mkv"

        placed = _verdicts({name: parsed_info()}, _scope([601], titles={601: "Preview Party"}))

        assert placed == {name: ((601,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_menu_named_after_the_episode_is_still_an_extra(self) -> None:
        # The title shields its own words only: the extras word stands outside them.
        name = "[bd menu 01] show - festival night.mkv"

        assert _verdicts({name: parsed_info()}, _scope([603])) == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_title_two_files_share_still_shields_them_from_the_extras_words(self) -> None:
        # A recap repeats a title holding an extras word: neither file is titled, and neither is an extra.
        first, recap = "show - 01 - preview party [grp].mkv", "show - 02 - preview party [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1,)),
            recap: parsed_info(season=1, episodes=(2,)),
        }

        placed = _verdicts(parsed, _scope(_FIRST_THREE, titles={601: "Preview Party"}))

        assert placed == {first: ((601,), PlacementVerdict.EXACT), recap: ((602,), PlacementVerdict.EXACT)}

    def test_the_assignment_lists_the_extra_among_the_excluded(self) -> None:
        name = "show - PV.mkv"

        result = assign_episode_ids(PlacementBatch([name], {name: parsed_info()}), _scope([601]))

        assert result.excluded == (Placement(name, (), PlacementVerdict.EXTRA),)
        assert result.skipped == ()


class TestAssignAliasReading:
    """A name Sonarr read under a series alias: its key yields to the match moving it, numbers intact, into a season."""

    _SEQUEL: ClassVar[list[int]] = [701, 702, 703]

    def test_a_match_moving_the_key_into_the_scopes_season_places_it(self) -> None:
        # The sequel numbers itself season one. Sonarr matched the name into the season our map holds it under.
        name = "show flat S01E02 [grp].mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(2,), matched=((2, 2),))}, _scope(self._SEQUEL))

        assert placed == {name: ((702,), PlacementVerdict.EXACT)}

    def test_a_match_changing_the_number_is_no_alias(self) -> None:
        name = "show flat S01E02 [grp].mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(2,), matched=((2, 1),))}, _scope(self._SEQUEL))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}

    def test_a_match_onto_the_specials_is_no_alias(self) -> None:
        # A first-season key matched to a special keeps naming the season's episode: another slice's.
        name = "show S01E01 [grp].mkv"

        placed = _verdicts({name: parsed_info(season=1, episodes=(1,), matched=((0, 1),))}, _scope([501, 502]))

        assert placed == {name: ((), PlacementVerdict.FOREIGN)}
