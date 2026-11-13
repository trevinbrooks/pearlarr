# pyright: strict
"""The placer's passes through `place_batch`, over synthetic parses and episode indexes.

The grab-time path through `place_release` is pinned by `test_grab_placement` and `test_pending_import_updates`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import ClassVar

import pytest

from pearlarr.arr_models import MatchedEpisode, ParsedFileInfo
from pearlarr.pending_import import TitleNames
from pearlarr.placement_types import (
    EMPTY_LISTING,
    NO_SIZE_MATCHES,
    BatchPlacements,
    EpisodeIndex,
    Placement,
    PlacementTarget,
    PlacementVerdict,
    SeadexEvidence,
    SizeMatches,
    TorrentListing,
)
from pearlarr.placer import NameFacts, place_batch
from pearlarr.releases import EpisodeKey

from .builders import (
    batch,
    blind,
    by_name,
    numbered_names,
    parsed_info,
    parsed_specials,
    place,
    series_index,
    title_window,
    torrent_listing,
    verdicts_of,
)

_GONE = "gone.mkv"
"""A name outside the batch: parsed blind it keeps pairing in name order out, unread (None) it makes the batch wait."""


def _pair_in_order_blocked(parsed: Mapping[str, ParsedFileInfo | None]) -> dict[str, ParsedFileInfo | None]:
    """The parses plus a blind name outside the batch, so pairing in name order never places what a pass would not."""

    return {**parsed, _GONE: parsed_info()}


class TestFromMatchedPlacement:
    """The parsed-episode limits every pass shares: the matched span cap, all-or-nothing resolution, duplicate collapse.

    Sonarr's matched episodes are taken only by a name carrying no
    numbers of its own, and only inside the window.
    """

    _TARGET: ClassVar[PlacementTarget] = PlacementTarget(
        [11, 12, 13], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12, EpisodeKey(1, 3): 13})
    )

    def test_a_pair_inside_the_window_places_exactly(self) -> None:
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        assert by_name(place(parsed, self._TARGET)) == {"span.mkv": ((11, 12), PlacementVerdict.EXACT)}

    def test_a_pair_resolving_nowhere_in_the_series_is_unplaced(self) -> None:
        # The episode index knows nothing of it, so it is possibly the window's, never proved outside it.
        parsed = {"span.mkv": parsed_info(matched=((9, 9),))}

        assert by_name(place(parsed, self._TARGET)) == {"span.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_partially_resolving_span_leaves_the_whole_file_unplaced(self) -> None:
        # One season and episode in the index, one not: placing the resolved half would half-import a multi-episode
        # file.
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (9, 9)))}

        assert by_name(place(parsed, self._TARGET)) == {"span.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_full_season_match_never_places(self) -> None:
        # A "S01" extra with no episode matches the WHOLE season: none of its matched episodes place, alone or beside
        # the episodes.
        full_season = parsed_info(matched=((1, 1), (1, 2)), full_season=True)
        beside = {
            "extras.mkv": full_season,
            "ep-01.mkv": parsed_info(season=0, absolutes=(1,), matched=((1, 1),)),
            "ep-02.mkv": parsed_info(season=0, absolutes=(2,), matched=((1, 2),)),
        }
        pair = PlacementTarget([11, 12], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12}))

        assert by_name(place({"pack.mkv": full_season}, self._TARGET)) == {
            "pack.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }
        assert by_name(place(beside, pair)) == {
            "extras.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND),
            "ep-01.mkv": ((11,), PlacementVerdict.EXACT),
            "ep-02.mkv": ((12,), PlacementVerdict.EXACT),
        }

    def test_the_season_and_episode_in_a_full_season_name_is_rejected_too(self) -> None:
        # The flag rejects the name's season and episode too, not just the matched episodes.
        parsed = {"pack.mkv": parsed_info(season=1, episodes=(1,), full_season=True)}

        assert by_name(place(parsed, self._TARGET)) == {"pack.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_wide_span_the_name_spells_out_places_whole(self) -> None:
        # The cap is a FROM-MATCHED limit. A spelled-out "E01-E04" range resolving every season and episode inside the
        # window is a complete parse of what the name spells out, so width never rejects it.
        target = PlacementTarget([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed = {"span.mkv": parsed_info(season=1, episodes=(1, 2, 3, 4))}

        assert by_name(place(parsed, target)) == {"span.mkv": ((11, 12, 13, 14), PlacementVerdict.EXACT)}

    def test_a_from_matched_span_just_over_the_cap_is_rejected(self) -> None:
        # Four distinct matched episodes a name with no numbers only took from the match is the season-pack shape sans
        # flag.
        target = PlacementTarget([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed = {"pack.mkv": parsed_info(matched=tuple((1, e) for e in range(1, 5)))}

        assert by_name(place(parsed, target)) == {"pack.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_triple_span_still_places(self) -> None:
        # The cap boundary: three episodes is still one file's span.
        parsed = {"triple.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3)))}

        assert by_name(place(parsed, self._TARGET)) == {"triple.mkv": ((11, 12, 13), PlacementVerdict.EXACT)}

    def test_duplicate_pairs_collapse_to_one(self) -> None:
        # Junk wire repeats are one season and episode, so the file places rather than reading as a wide span.
        parsed = {"one.mkv": parsed_info(matched=((1, 1), (1, 1)))}
        one_key = PlacementTarget([11, 12], series_index({EpisodeKey(1, 1): 11}))

        assert by_name(place(parsed, self._TARGET)) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}
        assert by_name(place(parsed, one_key)) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}


class TestOutsideTheWindow:
    """`OUTSIDE_WINDOW` needs COMPLETE parsed episodes landing entirely outside the window.

    Anything less (a partial span, a rejection, no parsed episodes at all) is possibly the window's
    and stays `NO_EPISODE_FOUND`, so the pairing passes may still place it.
    """

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(3, 1): 101,
        EpisodeKey(3, 12): 112,
        EpisodeKey(3, 13): 113,
    }

    def test_a_clean_parse_fully_outside_is_outside_the_window(self) -> None:
        # In the window's season or another: the over-grab guard, identity must land INSIDE the window.
        same_season = {"other.mkv": parsed_info(season=3, episodes=(13,))}
        other_season = {"x.mkv": parsed_info(season=1, episodes=(1,))}
        specials = PlacementTarget([8030], series_index({EpisodeKey(0, 1): 8030, EpisodeKey(1, 1): 8033}))

        assert by_name(place(same_season, PlacementTarget([101], series_index(self._ID_BY_KEY)))) == {
            "other.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW)
        }
        assert by_name(place(other_season, specials)) == {"x.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_a_parse_inside_the_window_places(self) -> None:
        parsed = {"mine.mkv": parsed_info(season=3, episodes=(1,))}

        assert by_name(place(parsed, PlacementTarget([101], series_index(self._ID_BY_KEY)))) == {
            "mine.mkv": ((101,), PlacementVerdict.EXACT)
        }

    def test_a_partially_resolving_span_stays_possibly_the_windows(self) -> None:
        # A double-episode with one season and episode off the index may be partly the window's, and is not provably
        # bogus either: the single-file pass doesn't place it whether or not the window's one id is in the index.
        parsed = {"d.mkv": parsed_info(season=3, episodes=(12, 99))}
        off_map = {"d.mkv": parsed_info(season=1, episodes=(5, 99))}

        assert by_name(place(parsed, PlacementTarget([101], series_index(self._ID_BY_KEY)))) == {
            "d.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }
        assert by_name(place(off_map, PlacementTarget([900], series_index({EpisodeKey(1, 5): 505})))) == {
            "d.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }

    def test_a_rejected_full_season_parse_is_still_the_windows(self) -> None:
        # Sonarr reads "S0X" alone as the whole season: no episode token, not several episodes,
        # so the one unused id takes the file, whether or not the season is in the index.
        unmapped = {"pack.mkv": parsed_info(season=2, episodes=(1,), full_season=True)}
        mapped = {"show S2 - OVA.mkv": parsed_info(season=2, full_season=True)}

        assert by_name(place(unmapped, PlacementTarget([101], series_index(self._ID_BY_KEY)))) == {
            "pack.mkv": ((101,), PlacementVerdict.SINGLE)
        }
        assert by_name(place(mapped, PlacementTarget([900], series_index({EpisodeKey(2, 1): 501})))) == {
            "show S2 - OVA.mkv": ((900,), PlacementVerdict.SINGLE)
        }

    def test_a_rejected_wide_span_stays_possibly_the_windows(self) -> None:
        parsed = {"pack.mkv": parsed_info(matched=tuple((9, n) for n in range(1, 11)))}

        assert by_name(place(parsed, PlacementTarget([101], series_index(self._ID_BY_KEY)))) == {
            "pack.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }

    def test_no_parsed_episodes_stays_possibly_the_windows(self) -> None:
        # Two unused ids keep the degenerate single-file pass out, so the classification is what is pinned.
        parsed = {"blank.mkv": parsed_info()}

        assert by_name(place(parsed, PlacementTarget([101, 112], series_index(self._ID_BY_KEY)))) == {
            "blank.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }

    def test_an_empty_episode_index_blocks_the_verdict(self) -> None:
        # No season and episode resolves in an unserved index, so nothing may be called outside the window.
        parsed = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert by_name(place(parsed, PlacementTarget([101, 112], series_index({})))) == {
            "other.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)
        }


# Two specials, a four-episode first season, and a three-episode second, titled and numbered through.
_ID_BY_KEY: dict[EpisodeKey, int] = {
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
    _ID_BY_KEY, absolutes={**{600 + n: n for n in range(1, 5)}, **{700 + n: 4 + n for n in range(1, 4)}}, titles=_TITLES
)
_FIRST_THREE = [601, 602, 603]


def _target(
    resolved: Sequence[int],
    *,
    used: Sequence[int] = (),
    series: str = "Show",
    titles: Mapping[int, str] | None = None,
) -> PlacementTarget:
    """A target over the shared episode index, retitled wholesale when `titles` is given."""

    index = _SERIES if titles is None else series_index(_ID_BY_KEY, titles=titles)
    return PlacementTarget(list(resolved), index, used=frozenset(used), title_names=TitleNames(series))


class TestPlaceEpisodeTitle:
    """The episode-title pass: a name whose title is one episode of the window goes there, whatever its number said."""

    def test_a_shifted_match_yields_to_the_title(self) -> None:
        # Sonarr's absolute match landed one episode early. The title is the third episode's.
        name = "show - 03 - Festival Night [grp].mkv"

        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 2),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_with_an_ampersand_matches_a_name_spelling_it_and(self) -> None:
        name = "show - 03 - Sun and Moon [grp].mkv"
        titles = {**_TITLES, 603: "Sun & Moon"}

        placed = by_name(
            place({name: parsed_info(absolutes=(3,), matched=((1, 2),))}, _target(_FIRST_THREE, titles=titles))
        )

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_season_and_episode_the_series_never_had_yields_to_the_title(self) -> None:
        # A CRC tag read as a season and episode resolves nowhere. The title still names the episode.
        name = "show - the return [E8F03223].mkv"

        placed = by_name(place({name: parsed_info(episodes=(8,))}, _target(_FIRST_THREE)))

        assert placed == {name: ((602,), PlacementVerdict.EPISODE_TITLE)}

    @pytest.mark.parametrize("names", [TitleNames("Show"), TitleNames("Show", ("Show",))])
    def test_a_title_opening_with_a_number_names_the_episode(self, names: TitleNames) -> None:
        name = "show - 20 Years Worth of Wishes [grp].mkv"
        index = series_index(_ID_BY_KEY, titles={603: "20 Years' Worth of Wishes"})

        placed = by_name(place({name: parsed_info()}, PlacementTarget(_FIRST_THREE, index, title_names=names)))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_head_alone_names_the_episode(self) -> None:
        # The group wrote the title up to the colon, as groups often do with a long title.
        name = "show - Cruel World Part [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _target([501, 502], titles={502: "Cruel World Part: Two Lives"})))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_title_as_written_outranks_another_episodes_head(self) -> None:
        # A movie's title is the head of its epilogue's: the file named with it is the movie.
        name = "show - Cruel World Part [grp].mkv"
        titles = {501: "Cruel World Part: Epilogue Drama", 502: "Show: Cruel World Part"}

        placed = by_name(place({name: parsed_info()}, _target([501, 502], titles=titles)))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_head_several_episodes_share_names_none(self) -> None:
        name = "show - Lost Girls [grp].mkv"
        titles = {501: "Lost Girls: Wall Goodbye", 502: "Lost Girls: Cruel World"}

        placed = by_name(place({name: parsed_info()}, _target([501, 502], titles=titles)))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_subtitle_alone_names_the_episode(self) -> None:
        # The name carries only the words past the title's colon, and its own number says the other special.
        name = "show S00E01 - Cruel World [grp].mkv"

        placed = by_name(place({name: parsed_info(season=0, episodes=(1,))}, _target([501, 502])))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    @pytest.mark.parametrize(
        ("title", "name"),
        [
            pytest.param("Cruel World - Two Lives", "show - Cruel World [grp].mkv", id="the head before a spaced dash"),
            pytest.param(
                "Cruel World - Two Lives", "show - Two Lives [grp].mkv", id="the subtitle after a spaced dash"
            ),
            pytest.param("Cruel World – Two Lives", "show - Two Lives [grp].mkv", id="an en dash"),
            pytest.param(
                "Cruel World -Two Lives-", "show - Cruel World [grp].mkv", id="the head before a wrapped segment"
            ),
            pytest.param("Cruel World -Two Lives-", "show - Two Lives [grp].mkv", id="the wrapped segment"),
        ],
    )
    def test_a_title_part_a_dash_sets_off_names_the_episode(self, title: str, name: str) -> None:
        placed = by_name(place({name: parsed_info()}, _target([501, 502], titles={502: title})))

        assert placed == {name: ((502,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_hyphenated_word_is_no_subtitle(self) -> None:
        name = "show - Flopping Again [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _target([501, 502], titles={502: "Flip-Flopping Again"})))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_title_two_files_carry_names_neither(self) -> None:
        # A recap repeats the title: the title names one file, and nothing says which.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - 01 - Beach Day [grp].mkv": parsed_info(),
            "show recap - 01 - Beach Day [grp].mkv": parsed_info(),
        }

        placed = by_name(place(parsed, _target([501])))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_title_inside_a_double_is_no_contradiction(self) -> None:
        name = "show - 01 - Pilot Flight [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(1, 2))}, _target(_FIRST_THREE)))

        assert placed == {name: ((601, 602), PlacementVerdict.EXACT)}

    def test_a_double_the_title_contradicts_is_rejected(self) -> None:
        # Placed as read, the two-episode file would hold an episode its title says it is not.
        name = "show - 02 - Pilot Flight [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2, 3))}, _target(_FIRST_THREE)))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    @pytest.mark.parametrize(
        ("used", "verdict"),
        [
            pytest.param((), PlacementVerdict.NO_EPISODE_FOUND, id="an episode still open"),
            pytest.param((501,), PlacementVerdict.OUTSIDE_WINDOW, id="the window full"),
        ],
    )
    def test_a_file_titled_outside_the_window_is_outside_it_only_once_nothing_is_left_for_it(
        self, used: Sequence[int], verdict: PlacementVerdict
    ) -> None:
        # The episode title alone never excludes a file the title may still need.
        name = "show - pilot flight [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _target([501], used=used)))

        assert placed == {name: ((), verdict)}

    def test_a_title_and_a_parse_both_outside_agree_the_file_is_outside_the_window(self) -> None:
        # Sonarr read it outside the window too, onto a different episode than the title: neither says it is the
        # window's.
        name = "show - pilot flight [grp].mkv"

        placed = by_name(place({name: parsed_info(matched=((2, 1),))}, _target([501])))

        assert placed == {name: ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_a_file_titled_outside_stays_unplaced_while_its_episode_is_unused(self) -> None:
        # Sonarr read it onto the unused special, and its title is that of an episode outside the window.
        name = "show - pilot flight [grp].mkv"

        placed = by_name(place({name: parsed_info(matched=((0, 1),))}, _target([501])))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_file_titled_outside_is_outside_the_window_once_its_episode_is_used(self) -> None:
        # The same file beside the plain one Sonarr read onto the special.
        plain, titled = "show - 01 [grp].mkv", "show - pilot flight [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            plain: parsed_info(matched=((0, 1),)),
            titled: parsed_info(matched=((0, 1),)),
        }

        placed = by_name(place(parsed, _target([501])))

        assert placed == {plain: ((501,), PlacementVerdict.EXACT), titled: ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_a_lone_title_whose_episode_a_season_episode_neighbour_reads_is_unplaced(self) -> None:
        # The file's title is the next episode's, which its neighbour reads by season and episode: the group numbered
        # the sequence one off, or the title is wrong, and one title cannot say which. Both stay loud, the neighbour
        # as a member of the sequence its title contradicted.
        titled, neighbour = "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            titled: parsed_info(season=1, episodes=(2,)),
            neighbour: parsed_info(season=1, episodes=(3,)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {
            titled: ((), PlacementVerdict.NO_EPISODE_FOUND),
            neighbour: ((), PlacementVerdict.TITLE_REFUTED),
        }

    def test_the_higher_version_of_a_titled_file_takes_the_title(self) -> None:
        # Both versions carry the title, so the title stays theirs. The v2 places, the v1 is its duplicate.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {v1: ((), PlacementVerdict.DUPLICATE), v2: ((602,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_file_map_versions_title_proves_the_unplaced_file_its_duplicate(self) -> None:
        # The v2 is in the file map on the title's episode. The v1 left unplaced carries the same episode's title.
        v1, v2 = "show - 02 - The Return [grp].mkv", "show - 02v2 - The Return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            (v1, v2), parsed_info(absolutes=(2,), matched=((1, 3),))
        )

        placed = by_name(place(parsed, _target(_FIRST_THREE), file_map={v2: (602,)}))

        assert placed == {v1: ((), PlacementVerdict.DUPLICATE)}

    def test_a_name_number_a_title_word_follows_yields_to_the_title(self) -> None:
        # No dash between the name number and the title, and Sonarr's absolute match landed one early.
        name = "show - 03 Festival Night [grp].mkv"

        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 2),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_whole_name_that_is_a_title_with_a_number_names_the_episode(self) -> None:
        # "17 Times 2" has the name number 2 by its shape. Whole, it is the third episode's title.
        name = "show - 17 Times 2 [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _target(_FIRST_THREE, titles={603: "17 Times 2"})))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    @pytest.mark.parametrize(
        ("series", "name", "title"),
        [
            ("Show", "show - 11 - 1 = [grp].mkv", "11 - 1 ="),
            ("Show", "show Season 2 - 11 - 1 = [grp].mkv", "11 - 1 ="),
            ("Show", "show - 1 2 3 [grp].mkv", "1 2 3"),
            ("Show", "show - 2 - 1 = [grp].mkv", "2 - 1 ="),
            ("Fox x Owl", "fox x owl - 12 x 12 [grp].mkv", "12 x 12"),
            ("Show No. 8", "show no 08 - 11 - 1 = [grp].mkv", "11 - 1 ="),
        ],
    )
    def test_a_title_of_numbers_read_whole_names_the_episode(self, series: str, name: str, title: str) -> None:
        # The name number is one of the title's numbers. A season tag, or the series name's own number, is no part
        # of it. "2 - 1 =" spells S02E01, a season and episode the name doesn't carry: its name number isn't
        # zero-padded.
        placed = by_name(place({name: parsed_info()}, _target(_FIRST_THREE, series=series, titles={603: title})))

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    @pytest.mark.parametrize("tag", ["S1", "Season 1", "1st Season", "1", "I"])
    def test_a_names_season_tag_and_name_number_spell_no_title(self, tag: str) -> None:
        # The third episode is titled "1.2", which the season and the name number spell.
        name = f"show {tag} - 02 [grp].mkv"
        parsed = {name: parsed_info(absolutes=(2,), matched=((1, 2),))}

        placed = by_name(place(parsed, _target(_FIRST_THREE, titles={**_TITLES, 603: "1.2"})))

        assert placed == {name: ((602,), PlacementVerdict.EXACT)}

    @pytest.mark.parametrize(
        "name",
        [
            "show 1 - 02 [grp].mkv",
            "show.1 - 02 [grp].mkv",
            "show-1-02 [grp].mkv",
            "show: 1 - 02 [grp].mkv",
            "show 1st - 02 [grp].mkv",
            "show.1.02 [grp].mkv",
            "show - 1 - 02 [grp].mkv",
            "show 1 ep 02 [grp].mkv",
            "show 1 Episode 02 [grp].mkv",
        ],
    )
    def test_a_season_and_padded_name_number_spell_no_title_without_sonarrs_season_and_episode(self, name: str) -> None:
        # The third episode is titled "1.2". Sonarr reads absolute 2 but matches no episode, and the season and the
        # zero-padded name number still spell no title, however the name sets them apart.
        placed = by_name(
            place({name: parsed_info(absolutes=(2,))}, _target(_FIRST_THREE, titles={**_TITLES, 603: "1.2"}))
        )

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_torrent_by_season_and_padded_name_number_places_by_its_numbers(self) -> None:
        # The third episode is titled "1.2", which the second file's numbers spell. Sonarr reads absolute numbers.
        names = [f"show - 1 - {n:02d} [grp].mkv" for n in (1, 2, 3)]
        parsed = {name: parsed_info(absolutes=(n,)) for n, name in enumerate(names, 1)}

        placed = by_name(place(parsed, _target(_FIRST_THREE, titles={**_TITLES, 603: "1.2"})))

        assert placed == {
            name: ((ep_id,), PlacementVerdict.ABSOLUTE) for name, ep_id in zip(names, _FIRST_THREE, strict=True)
        }

    def test_a_plain_season_and_unpadded_name_number_spell_no_title(self) -> None:
        # Sonarr reads nothing here, but a number right behind a word and just before the name number is the season.
        name = "show 1 - 2 [grp].mkv"

        placed = by_name(place({name: parsed_info()}, _target(_FIRST_THREE, titles={**_TITLES, 603: "1.2"})))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    @pytest.mark.parametrize(
        ("name", "parsed"),
        [
            ("show - 1 - 02 [grp].mkv", parsed_info(absolutes=(2,), matched=((1, 2),))),
            ("show.1.02 [grp].mkv", parsed_info(absolutes=(2,), matched=((1, 2),))),
            ("show-1-2 [grp].mkv", parsed_info(season=1, episodes=(2,), matched=((1, 2),))),
        ],
    )
    def test_a_title_of_numbers_spelling_sonarrs_season_and_episode_is_no_title(
        self, name: str, parsed: ParsedFileInfo
    ) -> None:
        # Sonarr reads S01E02, the season and episode the third episode's title "1.2" spells, from absolute 2 or from
        # the name's unpadded pair, which by itself spells out no season and episode.
        placed = by_name(place({name: parsed}, _target(_FIRST_THREE, titles={**_TITLES, 603: "1.2"})))

        assert placed == {name: ((602,), PlacementVerdict.EXACT)}

    @pytest.mark.parametrize("window", [[*_FIRST_THREE, 604, 701, 702, 703], [701, 702, 703]])
    def test_a_title_spelling_its_own_episodes_season_and_episode_names_it(self, window: list[int]) -> None:
        # S02E03 is titled "2.3", the name's plain season and name number. Sonarr reads absolute 3, S01E03.
        name = "show 2 - 03 [grp].mkv"
        parsed = {name: parsed_info(absolutes=(3,), matched=((1, 3),))}

        placed = by_name(place(parsed, _target(window, titles={**_TITLES, 703: "2.3"})))

        assert placed == {name: ((703,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_name_titled_both_ways_as_different_episodes_names_neither(self) -> None:
        # Name number then title, or a title opening with a number.
        name = "show - 02 Lost Girls [grp].mkv"
        titles = {602: "Lost Girls", 603: "02 Lost Girls"}

        placed = by_name(place({name: parsed_info()}, _target(_FIRST_THREE, titles=titles)))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}


class TestRejectParsedByTitleNumber:
    """A name number a title word follows is the title's number unless a title confirms it or an episode carries it."""

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("show - 03 Festival Night [grp].mkv", id="the title confirms the count"),
            pytest.param("show - 03 Festival Night WEB-DL 1080p [grp].mkv", id="a source word is quality junk"),
            pytest.param("show - 03 Finale [grp].mkv", id="one word says too little"),
            pytest.param("show.03&04.1080p.bluray.x264-grp.mkv", id="release junk behind a double spells no title"),
        ],
    )
    def test_a_name_number_sonarr_matched_stands_when_its_words_confirm_it_or_say_too_little(self, name: str) -> None:
        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 3),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((603,), PlacementVerdict.EXACT)}

    def test_a_name_number_its_own_title_nearly_confirms_stands(self) -> None:
        # The name carries episode 3's title a word off ("on" for "of").
        name = "show - 03 Festival on the Night [grp].mkv"
        target = _target(_FIRST_THREE, titles={**_TITLES, 603: "Festival of the Night"})

        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 3),))}, target))

        assert placed == {name: ((603,), PlacementVerdict.EXACT)}

    def test_a_name_number_sonarr_matched_that_no_title_confirms_is_misnumbered(self) -> None:
        # Episode 3 under a title Sonarr lacks, or an extra.
        name = "show - 03 Days of Summer [grp].mkv"

        placed = by_name(place({name: parsed_info(absolutes=(3,), matched=((1, 3),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_name_number_glued_onto_season_0_that_no_title_confirms_is_misnumbered(self) -> None:
        # Sonarr glues "0002" into S00E02, and no title confirms it.
        name = "show 0002 Days of Summer [grp].mkv"

        placed = by_name(place({name: parsed_info(season=0, episodes=(2,), matched=((0, 2),))}, _target([501, 502])))

        assert placed == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_an_unplaced_file_numbered_by_its_title_is_not_paired_in_order_onto_the_last_episode(self) -> None:
        # Two files are matched. The third's 100 numbers nothing in the series, and its words name no episode.
        parsed = {
            "show - 01 [grp].mkv": parsed_info(absolutes=(1,), matched=((1, 1),)),
            "show - 02 [grp].mkv": parsed_info(absolutes=(2,), matched=((1, 2),)),
            "show - 100 Days Later [grp].mkv": parsed_info(absolutes=(100,)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {
            "show - 01 [grp].mkv": ((601,), PlacementVerdict.EXACT),
            "show - 02 [grp].mkv": ((602,), PlacementVerdict.EXACT),
            "show - 100 Days Later [grp].mkv": ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_an_unmatched_torrent_titled_its_own_way_still_pairs_in_order(self) -> None:
        # Sonarr matched nothing, and every name number is an episode of the series.
        names = [
            "show - 01 First Flight [grp].mkv",
            "show - 02 Coming Back [grp].mkv",
            "show - 03 Night Fair [grp].mkv",
        ]
        parsed = {name: parsed_info(absolutes=(n,)) for n, name in enumerate(names, 1)}

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {name: ((600 + n,), PlacementVerdict.ABSOLUTE) for n, name in enumerate(names, 1)}


class TestPlaceContradictedSequence:
    """A sequence a member's title contradicts: its numbering was wrong about every member's parse in the window."""

    _RUN: ClassVar[list[str]] = numbered_names("show", 4, ("The Return", "Festival Night", "", ""))
    _PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        name: parsed_info(absolutes=(n,), matched=((1, n),)) for n, name in enumerate(_RUN, start=1)
    }

    def test_the_titled_members_place_and_the_rest_stay_loud(self) -> None:
        # Sonarr matched each file one episode early. The two titled files place by title, the untitled one
        # inside the window is contradicted (its match is as wrong), and the one read past the window is outside it.
        placed = by_name(place(self._PARSED, _target(_FIRST_THREE)))

        assert placed == {
            self._RUN[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            self._RUN[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            self._RUN[2]: ((), PlacementVerdict.TITLE_REFUTED),
            self._RUN[3]: ((), PlacementVerdict.OUTSIDE_WINDOW),
        }

    def test_a_season_episode_sequence_is_contradicted_like_a_matched_one(self) -> None:
        # The group's seasons and episodes are one off, as its titles show: the untitled member's is as wrong.
        run = numbered_names("show", 3, ("The Return", "Festival Night", ""))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n, name in enumerate(run, start=1)
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {
            run[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            run[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            run[2]: ((), PlacementVerdict.TITLE_REFUTED),
        }

    def test_a_member_its_own_title_confirms_keeps_its_season_and_episode(self) -> None:
        # The numbering strays from the second file on: the first, carrying its own episode's title, stays placed.
        run = numbered_names("show", 3, ("Pilot Flight", "Festival Night", "Final Bell"))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n, name in enumerate(run, start=1)
        }

        placed = by_name(place(parsed, _target([*_FIRST_THREE, 604])))

        assert placed == {
            run[0]: ((601,), PlacementVerdict.EXACT),
            run[1]: ((603,), PlacementVerdict.EPISODE_TITLE),
            run[2]: ((604,), PlacementVerdict.EPISODE_TITLE),
        }

    def test_an_import_poll_judges_the_unplaced_files_as_the_grab_did(self) -> None:
        # The titled files are in the file map. The poll places only the unplaced files, over the whole torrent's
        # evidence, so the untitled file's shifted match still never lands on the file map's empty episode.
        file_map = {self._RUN[0]: (602,), self._RUN[1]: (603,)}

        placed = by_name(place(self._PARSED, _target(_FIRST_THREE), file_map=file_map))

        assert placed == {
            self._RUN[2]: ((), PlacementVerdict.TITLE_REFUTED),
            self._RUN[3]: ((), PlacementVerdict.OUTSIDE_WINDOW),
        }


class TestPlaceContradictedPairInOrder:
    """A pairing in order a title contradicts places nothing: the pair it contradicts says the numbering is off."""

    def test_a_sequence_a_title_contradicts_places_only_the_titled_file(self) -> None:
        run = numbered_names("show", 3, ("The Return", "", ""))

        placed = by_name(place(blind(run), _target(_FIRST_THREE)))

        assert placed == {
            run[0]: ((602,), PlacementVerdict.EPISODE_TITLE),
            run[1]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[2]: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    @pytest.mark.parametrize("absolutes", [True, False], ids=["absolute zip", "ordered zip"])
    def test_a_pair_in_order_a_title_contradicts_places_nothing(self, absolutes: bool) -> None:
        # The files pair 1:1 in order, but the first carries the title of the next season's episode.
        names = ("show new dawn.mkv", "show b.mkv", "show c.mkv")
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(absolutes=(n,)) if absolutes else parsed_info() for n, name in enumerate(names, 1)
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == dict.fromkeys(parsed, ((), PlacementVerdict.NO_EPISODE_FOUND))


class TestPlaceOverlappingClaims:
    """Two unplaced files reading one episode differently: a span among them is rejected, a same-span pair is not."""

    def test_a_span_holding_an_episode_another_file_reads_alone_is_rejected(self) -> None:
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {span: ((), PlacementVerdict.NO_EPISODE_FOUND), single: ((602,), PlacementVerdict.EXACT)}

    def test_an_import_poll_rejects_the_span_the_grab_did(self) -> None:
        # The single is in the file map. The span still holds an episode it read differently, and one nothing has.
        span, single = "show S01E01-E02 [grp].mkv", "show S01E02 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            single: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE), file_map={single: (602,)}))

        assert placed == {span: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_rejected_wide_match_contests_no_episode(self) -> None:
        # A recap Sonarr matched to four episodes is past the matched span cap: its parsed episodes contest nothing.
        span, recap = "show S01E01-E02 [grp].mkv", "show recap [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            span: parsed_info(season=1, episodes=(1, 2)),
            recap: parsed_info(matched=((1, 1), (1, 2), (1, 3), (1, 4))),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {span: ((601, 602), PlacementVerdict.EXACT), recap: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_span_only_partly_used_is_no_duplicate(self) -> None:
        # A file in the file map carries the title of the span's first episode and is on it. The span's second
        # episode has nothing else.
        span, seed = "show S01E02-E03 [grp].mkv", "show - the return [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {span: parsed_info(season=1, episodes=(2, 3)), seed: parsed_info()}

        placed = by_name(place(parsed, _target(_FIRST_THREE), file_map={seed: (602,)}))

        assert placed == {span: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_two_versions_of_one_span_keep_their_parse(self) -> None:
        first, second = "show S01E01-E02 [grp].mkv", "show S01E01-E02 v2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1, 2)),
            second: parsed_info(season=1, episodes=(1, 2)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE)))

        assert placed == {first: ((), PlacementVerdict.DUPLICATE), second: ((601, 602), PlacementVerdict.EXACT)}


class TestPlaceExtras:
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
        # Alone with one unused episode, the file would otherwise be the single-file placement.
        assert by_name(place({name: parsed_info()}, _target([601]))) == {name: ((), PlacementVerdict.EXTRA)}

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
        assert by_name(place({name: parsed_info()}, _target([601]))) == {name: ((601,), PlacementVerdict.SINGLE)}

    def test_a_file_sonarr_matched_by_the_season_and_episode_in_its_name_is_no_extra(self) -> None:
        name = "show - S01E02 - the trailer park.mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_match_by_number_alone_leaves_an_extra_aside(self) -> None:
        # Sonarr read the menu's number as the first episode's: the name still says what the file is.
        name = "[BD Menu 01] show.mkv"

        placed = by_name(place({name: parsed_info(absolutes=(1,), matched=((1, 1),))}, _target(_FIRST_THREE)))

        assert placed == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_word_of_the_titles_names_is_no_extras_word(self) -> None:
        name = "show op - special.mkv"

        assert by_name(place({name: parsed_info()}, _target([601], series="Show Op"))) == {
            name: ((601,), PlacementVerdict.SINGLE)
        }

    def test_a_file_an_episode_title_names_is_never_an_extra(self) -> None:
        name = "show - preview party.mkv"

        placed = by_name(place({name: parsed_info()}, _target([601], titles={601: "Preview Party"})))

        assert placed == {name: ((601,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_menu_named_after_the_episode_is_still_an_extra(self) -> None:
        # The title shields its own words only: the extras word stands outside them.
        name = "[bd menu 01] show - festival night.mkv"

        assert by_name(place({name: parsed_info()}, _target([603]))) == {name: ((), PlacementVerdict.EXTRA)}

    def test_a_title_two_files_share_still_shields_them_from_the_extras_words(self) -> None:
        # A recap repeats a title holding an extras word: neither file is titled, and neither is an extra.
        first, recap = "show - 01 - preview party [grp].mkv", "show - 02 - preview party [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            first: parsed_info(season=1, episodes=(1,)),
            recap: parsed_info(season=1, episodes=(2,)),
        }

        placed = by_name(place(parsed, _target(_FIRST_THREE, titles={601: "Preview Party"})))

        assert placed == {first: ((601,), PlacementVerdict.EXACT), recap: ((602,), PlacementVerdict.EXACT)}

    def test_the_batch_placements_list_the_extra_among_the_excluded(self) -> None:
        name = "show - PV.mkv"

        result = place({name: parsed_info()}, _target([601]))

        assert result.excluded == (Placement(name, (), PlacementVerdict.EXTRA),)
        assert result.unplaced == ()


class TestPlaceAliasParse:
    """A name Sonarr read under a series alias: its season and episode yield to the season shift into a season."""

    _SEQUEL: ClassVar[list[int]] = [701, 702, 703]

    def test_a_season_shift_into_the_windows_season_places_it(self) -> None:
        # The sequel numbers itself season one. Sonarr matched the name into the season the mapping holds it under.
        name = "show flat S01E02 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((2, 2),))}, _target(self._SEQUEL)))

        assert placed == {name: ((702,), PlacementVerdict.EXACT)}

    def test_a_match_changing_the_number_is_no_season_shift(self) -> None:
        name = "show flat S01E02 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(2,), matched=((2, 1),))}, _target(self._SEQUEL)))

        assert placed == {name: ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_a_match_onto_the_specials_is_no_season_shift(self) -> None:
        # A spelled-out first-season episode matched to a special keeps naming the season's episode: outside the
        # window.
        name = "show S01E01 [grp].mkv"

        placed = by_name(place({name: parsed_info(season=1, episodes=(1,), matched=((0, 1),))}, _target([501, 502])))

        assert placed == {name: ((), PlacementVerdict.OUTSIDE_WINDOW)}


class TestPlaceExactSeason:
    """The exact pass: a correctly named file Sonarr just couldn't match to the series."""

    def test_specials_placed_by_exact_season_episode(self) -> None:
        parsed = {
            "s00e01.mkv": parsed_info(season=0, episodes=(1,)),
            "s00e02.mkv": parsed_info(season=0, episodes=(2,)),
        }
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032, EpisodeKey(1, 1): 8033}

        result = place(parsed, PlacementTarget([8030, 8031, 8032], series_index(ep_id_map)))

        assert result.placed == {"s00e01.mkv": (8030,), "s00e02.mkv": (8031,)}
        assert (result.unplaced, result.excluded) == ((), ())

    def test_no_window_places_correctly_named_specials(self) -> None:
        # With no window the exact pass falls back to the live episode index,
        # so a correctly named file lands on its real episode instead of sticking forever.
        parsed = {
            "s00e01.mkv": parsed_info(season=0, episodes=(1,)),
            "s00e02.mkv": parsed_info(season=0, episodes=(2,)),
        }
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032}

        result = place(parsed, PlacementTarget([], series_index(ep_id_map)))

        assert result.placed == {"s00e01.mkv": (8030,), "s00e02.mkv": (8031,)}
        assert (result.unplaced, result.excluded) == ((), ())


class TestPlaceAbsolute:
    """Pairing in absolute order: absolute numbers only ORDER the files onto the window, never decide identity."""

    def test_mis_numbered_specials_place_by_position(self) -> None:
        # Files named "01".."05" that are really S00E05..E09: "01" takes the first window episode.
        files = [f"{n:02d}.mkv" for n in range(1, 6)]
        parsed = {name: parsed_info(absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, PlacementTarget([8034, 8035, 8036, 8037, 8038], series_index({})))

        assert result.unplaced == ()
        assert result.placed == {
            "01.mkv": (8034,),
            "02.mkv": (8035,),
            "03.mkv": (8036,),
            "04.mkv": (8037,),
            "05.mkv": (8038,),
        }

    def test_continuous_absolute_batch_spans_seasons(self) -> None:
        # The only multi-season torrent trusted: a continuous 1..4 onto a season-sorted window.
        parsed = {f"e{i}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 5)}

        result = place(parsed, PlacementTarget([501, 502, 601, 602], series_index({})))

        assert result.placed == {
            "e1.mkv": (501,),
            "e2.mkv": (502,),
            "e3.mkv": (601,),
            "e4.mkv": (602,),
        }

    def test_a_file_with_no_signal_blocks_pair_in_order(self) -> None:
        # A file whose parse yields nothing could be a hiccuped real episode, so the whole pairing is blocked.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "c.mkv": parsed_info(),
        }

        result = place(parsed, PlacementTarget([501, 502], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_a_menu_file_neither_takes_a_slot_nor_blocks_pair_in_order(self) -> None:
        # A menu is never an episode: set aside by name before the pairing passes run.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "menu.mkv": parsed_info(),
        }

        result = place(parsed, PlacementTarget([501, 502], series_index({})))

        assert result.placed == {"a.mkv": (501,), "b.mkv": (502,)}

    def test_absolute_ova_torrent_places_onto_the_window(self) -> None:
        # Thirteen season-0 absolute-only OVA files onto the title's S00E16..E28, count-matched 13:13.
        files = [f"{n:02d}.mkv" for n in range(1, 14)]
        parsed = {name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, PlacementTarget(list(range(2090, 2103)), series_index({})))

        assert result.unplaced == ()
        assert result.placed == {f"{n:02d}.mkv": (2089 + n,) for n in range(1, 14)}

    def test_mixed_exact_then_absolute_onto_the_unused_id(self) -> None:
        # The exact pass places the named file. The absolute file goes onto the one unused id.
        parsed = {
            "s01e01.mkv": parsed_info(season=1, episodes=(1,)),
            "extra.mkv": parsed_info(absolutes=(2,)),
        }

        result = place(parsed, PlacementTarget([8033, 8044], series_index({EpisodeKey(1, 1): 8033})))

        assert result.placed == {"s01e01.mkv": (8033,), "extra.mkv": (8044,)}
        assert result.unplaced == ()


class TestPlaceFromMatched:
    """The exact pass's fallback to matched episodes: Sonarr's `(season, episode)` for absolute-only names."""

    def test_multi_title_batch_places_exactly_inside_the_window(self) -> None:
        # A batch spanning two titles plus a special, the window covering the second only:
        # in-window files place exactly and the two the index resolves OUTSIDE are outside the window.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep-11.mkv": parsed_info(season=0, absolutes=(11,), matched=((1, 11),)),
            "ep-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "ep-13.mkv": parsed_info(season=0, absolutes=(13,), matched=((1, 13),)),
            "sp-17.5.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587, EpisodeKey(0, 1): 2574}

        result = place(parsed, PlacementTarget([2586, 2587], series_index(ep_id_map)))

        assert result.placed == {"ep-12.mkv": (2586,), "ep-13.mkv": (2587,)}
        assert result.unplaced == ()
        assert {p.name: p.verdict for p in result.excluded} == {
            "ep-11.mkv": PlacementVerdict.OUTSIDE_WINDOW,
            "sp-17.5.mkv": PlacementVerdict.OUTSIDE_WINDOW,
        }

    def test_name_parsed_pair_beats_matched_episode(self) -> None:
        # A name that carries its own (season, episode) never defers to Sonarr's matched resolution.
        parsed = {"x.mkv": parsed_info(season=2, episodes=(5,), matched=((9, 9),))}
        ep_id_map = {EpisodeKey(2, 5): 400, EpisodeKey(9, 9): 999}

        result = place(parsed, PlacementTarget([400, 999], series_index(ep_id_map)))

        assert result.placed == {"x.mkv": (400,)}

    def test_matched_episodes_never_apply_with_no_window(self) -> None:
        # With NO window the live-index fallback trusts only a season and episode parsed from the name.
        # Sonarr's series match must not decide identity on its own.
        parsed = {"x.mkv": parsed_info(season=0, absolutes=(3,), matched=((1, 3),))}

        result = place(parsed, PlacementTarget([], series_index({EpisodeKey(1, 3): 300})))

        assert result.placed == {}
        assert result.unplaced == ("x.mkv",)

    def test_partially_in_window_matched_span_is_unplaced(self) -> None:
        # A matched span reaching outside the window is rejected whole, as a name-parsed one is.
        parsed = {"span.mkv": parsed_info(season=0, matched=((1, 1), (1, 3)))}
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 3): 503}

        result = place(parsed, PlacementTarget([501, 502], series_index(ep_id_map)))

        assert result.placed == {}
        assert result.unplaced == ("span.mkv",)

    def test_an_out_of_window_match_on_the_sole_file_is_outside_the_window(self) -> None:
        # One file with no parsed number, one unused id, and Sonarr's title match names an out-of-window
        # episode: a complete parse outside the window is outside it, never the unused id's.
        parsed = {"only.mkv": parsed_info(matched=((1, 5),))}

        result = place(parsed, PlacementTarget([900], series_index({EpisodeKey(1, 5): 555})))

        assert result.placed == {}
        assert [p.verdict for p in result.excluded] == [PlacementVerdict.OUTSIDE_WINDOW]

    @pytest.mark.parametrize(
        ("matched_id", "expected"),
        [
            # A disagreeing id is rejected as possibly the window's, never proven outside it.
            pytest.param(999, ((), PlacementVerdict.NO_EPISODE_FOUND), id="another series' id is refused"),
            pytest.param(501, ((501,), PlacementVerdict.EXACT), id="an agreeing id places"),
        ],
    )
    def test_sonarr_s_matched_id_must_agree_with_the_index(
        self, matched_id: int, expected: tuple[tuple[int, ...], PlacementVerdict]
    ) -> None:
        # Sonarr may match another series whose numbers coincide with this one's: its episode id then disagrees.
        info = ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=matched_id),))

        result = place({"x.mkv": info}, PlacementTarget([501, 502], series_index({EpisodeKey(1, 1): 501})))

        assert by_name(result) == {"x.mkv": expected}

    def test_mixed_id_duplicate_matched_episodes_place_once(self) -> None:
        # (s,e,None) and (s,e,id) survive the triple dedup as two, yet name the episode once.
        # Two window ids keep the single-file pass out, so this pins the exact pass itself.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
            ),
        )

        result = place({"x.mkv": info}, PlacementTarget([501, 502], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {"x.mkv": (501,)}

    def test_wrong_id_match_cannot_block_the_single_file_fallback(self) -> None:
        # A disagreeing id rejects the MATCH, but one file with no parsed number and one unused id still place.
        info = ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),))

        result = place({"only.mkv": info}, PlacementTarget([501], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {"only.mkv": (501,)}

    def test_junk_duplicates_beyond_the_cap_still_collapse_and_place(self) -> None:
        # The cap counts DISTINCT matched episodes: four wire duplicates of one are one, not a season pack.
        parsed = {"x.mkv": parsed_info(matched=((1, 1), (1, 1), (1, 1), (1, 1)))}

        result = place(parsed, PlacementTarget([501], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {"x.mkv": (501,)}

    def test_mixed_id_duplicate_of_a_triple_span_still_places(self) -> None:
        # An id-bearing junk duplicate of one matched episode can't inflate a triple past the cap.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
                MatchedEpisode(season_number=1, episode_number=2),
                MatchedEpisode(season_number=1, episode_number=3),
            ),
        )
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502, EpisodeKey(1, 3): 503}

        result = place({"x.mkv": info}, PlacementTarget([501, 502, 503], series_index(ep_id_map)))

        assert result.placed == {"x.mkv": (501, 502, 503)}

    def test_partially_resolved_double_absolute_never_half_imports(self) -> None:
        # A "12-13" file whose match resolved only E12: the span from the match doesn't cover the absolutes.
        parsed = {"d.mkv": parsed_info(season=0, absolutes=(12, 13), matched=((1, 12),))}

        result = place(parsed, PlacementTarget([2586, 2587], series_index({EpisodeKey(1, 12): 2586})))

        assert result.placed == {}
        assert result.unplaced == ("d.mkv",)

    def test_fully_resolved_double_absolute_places_both(self) -> None:
        parsed = {"d.mkv": parsed_info(season=0, absolutes=(12, 13), matched=((1, 12), (1, 13)))}
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = place(parsed, PlacementTarget([2586, 2587], series_index(ep_id_map)))

        assert result.placed == {"d.mkv": (2586, 2587)}
        assert result.unplaced == ()

    def test_matched_span_never_half_imports_via_the_single_file_fallback(self) -> None:
        # Sonarr says the file spans E01+E02: the single-file pass honors that count even where it ignores identity.
        parsed = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        result = place(parsed, PlacementTarget([501], series_index({EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502})))

        assert result.placed == {}
        assert result.unplaced == ("span.mkv",)

    def test_full_season_file_never_takes_the_spare_id(self) -> None:
        # The exact pass quarantines the season-pack shape. The single-file pass must not hand it the spare id.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras-s01.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3), (1, 4)), full_season=True),
            "e01.mkv": parsed_info(season=1, episodes=(1,)),
            "e02.mkv": parsed_info(season=1, episodes=(2,)),
            "e03.mkv": parsed_info(season=1, episodes=(3,)),
        }
        ep_id_map = {EpisodeKey(1, n): 500 + n for n in range(1, 5)}

        result = place(parsed, PlacementTarget([501, 502, 503, 504], series_index(ep_id_map)))

        assert result.placed == {"e01.mkv": (501,), "e02.mkv": (502,), "e03.mkv": (503,)}
        assert result.unplaced == ("extras-s01.mkv",)


class TestPlaceExactPrecedence:
    """Two unplaced files resolving to one episode: `ParsedEpisodes.rank` picks, and batch order only breaks a tie."""

    def test_a_season_and_episode_in_the_name_sonarr_agrees_with_beats_a_pair_from_matched(self) -> None:
        # The "17.5 (S00E01)" special names itself. The "- 17" beside it only
        # took the same season and episode from a match TVDB's interleaving shifted.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep - 17.mkv": parsed_info(season=0, absolutes=(17,), matched=((0, 1),)),
            "ep - 17.5 (S00E01).mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {EpisodeKey(0, 1): 2574, EpisodeKey(1, 17): 2591}

        result = place(parsed, PlacementTarget([2574], series_index(ep_id_map)))

        assert result.placed == {"ep - 17.5 (S00E01).mkv": (2574,)}
        assert by_name(result)["ep - 17.mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_a_pair_from_matched_beats_a_season_and_episode_in_the_name_sonarr_never_matched(self) -> None:
        # A "- Bonus" whose CRC tag parsed as E8 carries a season and episode Sonarr could not
        # match to the series. The "- 08" it collides with is the episode.
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep - Bonus [E8F03223].mkv": parsed_info(season=1, episodes=(8,)),
            "ep - 08.mkv": parsed_info(season=0, absolutes=(8,), matched=((1, 8),)),
        }

        result = place(parsed, PlacementTarget([508], series_index({EpisodeKey(1, 8): 508})))

        assert result.placed == {"ep - 08.mkv": (508,)}
        assert by_name(result)["ep - Bonus [E8F03223].mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_batch_order_breaks_a_tie_within_a_rank(self) -> None:
        # Two names spelling out one episode at one rank, Sonarr agreeing or not: the first stays, the second is a
        # duplicate.
        corroborated: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            ("a - S01E08.mkv", "b - S01E08.mkv"), parsed_info(season=1, episodes=(8,), matched=((1, 8),))
        )
        own_keys: dict[str, ParsedFileInfo | None] = dict.fromkeys(
            ("a.mkv", "b.mkv"), parsed_info(season=0, episodes=(1,))
        )
        specials = series_index({EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(1, 1): 601})

        assert by_name(place(corroborated, PlacementTarget([508], series_index({EpisodeKey(1, 8): 508})))) == {
            "a - S01E08.mkv": ((508,), PlacementVerdict.EXACT),
            "b - S01E08.mkv": ((), PlacementVerdict.DUPLICATE),
        }
        assert by_name(place(own_keys, PlacementTarget([501, 502], specials))) == {
            "a.mkv": ((501,), PlacementVerdict.EXACT),
            "b.mkv": ((), PlacementVerdict.DUPLICATE),
        }


class TestPlaceBesideFilesOutsideTheWindow:
    """A file whose season and episode read outside the window is outside it: it never blocks a pairing pass."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 501,
        EpisodeKey(1, 2): 502,
        EpisodeKey(0, 1): 901,
        EpisodeKey(2, 1): 601,
        EpisodeKey(2, 2): 602,
    }

    def test_one_unplaced_file_with_no_number_beside_files_outside_the_window_places_single(self) -> None:
        # The OVA's title resolves one id. The season files around it name
        # another season, so the OVA is the only contender.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - S01E01.mkv": parsed_info(season=1, episodes=(1,)),
            "show - S01E02.mkv": parsed_info(season=1, episodes=(2,)),
            "show - OVA.mkv": parsed_info(),
        }

        result = place(parsed, PlacementTarget([901], series_index(self._ID_BY_KEY)))

        assert result.placed == {"show - OVA.mkv": (901,)}
        assert [p.verdict for p in result.excluded] == [
            PlacementVerdict.OUTSIDE_WINDOW,
            PlacementVerdict.OUTSIDE_WINDOW,
        ]

    def test_a_from_matched_pair_outside_yields_to_an_unplaced_file(self) -> None:
        # Sonarr matched the special outside the window: its opinion steps aside for the
        # one file the listing leaves unplaced, which takes the one id.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - special.mkv": parsed_info(matched=((1, 5),)),
            "show - OVA.mkv": parsed_info(),
        }
        ep_id_map = {EpisodeKey(1, 5): 555, EpisodeKey(0, 1): 901}

        result = place(parsed, PlacementTarget([901], series_index(ep_id_map)))

        assert result.placed == {"show - OVA.mkv": (901,)}
        assert [p.verdict for p in result.excluded] == [PlacementVerdict.OUTSIDE_WINDOW]

    def test_pairing_in_absolute_order_counts_only_the_contenders(self) -> None:
        # Two absolute-only unplaced files pair in order onto the two-id window although a
        # first-season file is in the same batch.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - S01E01.mkv": parsed_info(season=1, episodes=(1,)),
            "show - 14.mkv": parsed_info(season=0, absolutes=(14,)),
            "show - 13.mkv": parsed_info(season=0, absolutes=(13,)),
        }

        result = place(parsed, PlacementTarget([601, 602], series_index(self._ID_BY_KEY)))

        assert result.placed == {"show - 13.mkv": (601,), "show - 14.mkv": (602,)}
        assert by_name(result)["show - S01E01.mkv"] == ((), PlacementVerdict.OUTSIDE_WINDOW)


class TestDuplicateEvidence:
    """A collision is a duplicate only when the file on the id reads there too. Otherwise it is unplaced, to report."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}
    _SPELLED_NAME = "show - S01E12 [1080p].mkv"

    @classmethod
    def _place(cls, file_map_parse: ParsedFileInfo | None) -> BatchPlacements:
        # The file map puts "seeded.mkv" on 2586. The file spelling out its season and episode resolves there by name.
        parsed = {"seeded.mkv": file_map_parse, cls._SPELLED_NAME: parsed_info(season=1, episodes=(12,))}
        return place(
            parsed, PlacementTarget([2586, 2587], series_index(cls._ID_BY_KEY)), file_map={"seeded.mkv": (2586,)}
        )

    def test_a_file_in_the_file_map_reading_the_same_episode_proves_the_duplicate(self) -> None:
        result = self._place(parsed_info(season=1, episodes=(12,)))

        assert by_name(result) == {self._SPELLED_NAME: ((), PlacementVerdict.DUPLICATE)}

    def test_a_file_map_file_paired_in_order_leaves_an_unplaced_file_the_caller_reports(self) -> None:
        # The file map has a file with no parsed number on 2586, paired in order. A file naming that episode outright
        # disagrees with it, and a disagreement is reported, never persisted as an exclusion.
        result = self._place(parsed_info())

        assert by_name(result) == {self._SPELLED_NAME: ((), PlacementVerdict.NO_EPISODE_FOUND)}
        assert result.excluded == ()


class TestPlaceWindowGate:
    """The window is all its episode ids: a fully used one stays enforced, and only an empty one is no window."""

    def test_a_window_an_earlier_window_fully_used_never_unlocks_the_live_index(self) -> None:
        # An earlier window used every window id, yet a correctly named file outside the window is outside it, not
        # placed on the live index.
        parsed = {"x.mkv": parsed_info(season=1, episodes=(1,))}

        result = place(parsed, PlacementTarget([8044], series_index({EpisodeKey(1, 1): 8033}), used=frozenset({8044})))

        assert result.placed == {}
        assert by_name(result) == {"x.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_no_window_leaves_absolute_only_files_unplaced(self) -> None:
        # With no window, pairing in absolute order has nothing to index into. Absolutes never decide identity alone.
        files = [f"{n:02d}.mkv" for n in range(1, 4)]
        parsed = {name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = place(parsed, PlacementTarget([], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(files)


class TestAbsoluteDuplicateTell:
    """Pairing in absolute order needs a clean 1:1 count and no absolute two parses share, the file map's too."""

    def test_count_mismatch_places_nothing(self) -> None:
        parsed = {"a.mkv": parsed_info(absolutes=(1,)), "b.mkv": parsed_info(absolutes=(2,))}

        result = place(parsed, PlacementTarget([1, 2, 3], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["a.mkv", "b.mkv"]

    def test_a_hiccuped_episode_parse_blocks_the_pass(self) -> None:
        # A None parse may be a real episode the parse hiccuped on. The next poll re-parses it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": parsed_info(absolutes=(1,)),
            "b.mkv": parsed_info(absolutes=(2,)),
            "c.mkv": None,
        }

        result = place(parsed, PlacementTarget([1, 2, 3], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_a_multi_absolute_file_blocks_the_pass(self) -> None:
        # A file spanning two absolutes ("01-02") can't be placed positionally.
        parsed: dict[str, ParsedFileInfo | None] = {
            "span.mkv": parsed_info(absolutes=(1, 2)),
            "c.mkv": parsed_info(absolutes=(3,)),
        }

        result = place(parsed, PlacementTarget([1, 2, 3], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["c.mkv", "span.mkv"]

    def test_out_of_window_absolute_cannot_stand_in_for_a_hiccuped_episode(self) -> None:
        # A sibling outside the window (absolute 11, matched outside it) must not make up the count for a hiccuped E12.
        parsed: dict[str, ParsedFileInfo | None] = {
            "s-11.mkv": parsed_info(season=0, absolutes=(11,), matched=((1, 11),)),
            "e-12.mkv": None,
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586}

        result = place(parsed, PlacementTarget([2586], series_index(ep_id_map)))

        assert result.placed == {}
        assert by_name(result) == {
            "s-11.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW),
            "e-12.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_the_earlier_version_of_a_placed_file_is_the_duplicate(self) -> None:
        # The exact pass places the "- 12v2" (a later version outranks its earlier one). The "- 12"
        # shares absolute 12, so the batch-wide duplicate tell blocks pairing in absolute order for it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = place(parsed, PlacementTarget([2586, 2587], series_index(ep_id_map)))

        assert result.placed == {"e-12v2.mkv": (2586,)}
        assert by_name(result)["e-12.mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_a_sharer_in_the_file_map_still_blocks_pair_in_order(self) -> None:
        # The v1 was placed on an earlier poll, and its parse still reaches the duplicate tell. The index is
        # unserved for the matched episode, so nothing resolves and pairing in order is the only thing that can
        # place it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = place(parsed, PlacementTarget([2587], series_index({})), file_map={"e-12.mkv": (2586,)})
        alone = place({"e-12v2.mkv": parsed["e-12v2.mkv"]}, PlacementTarget([2587], series_index({})))

        assert result.placed == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}
        # The control: drop the sharer and the very same file takes the spare id.
        assert alone.placed == {"e-12v2.mkv": (2587,)}

    def test_a_sharer_parse_the_caller_couldnt_get_blocks_pair_in_order(self) -> None:
        parsed = {"e-12.mkv": None, "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),))}

        result = place(parsed, PlacementTarget([2587], series_index({})), file_map={"e-12.mkv": (2586,)})

        assert result.placed == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_junk_duplicate_absolute_within_one_parse_does_not_block(self) -> None:
        # One parse repeating its own absolute is wire junk, not a restart tell: the other file still places.
        parsed: dict[str, ParsedFileInfo | None] = {
            "seeded-12.mkv": parsed_info(season=0, absolutes=(12, 12)),
            "left-13.mkv": parsed_info(season=0, absolutes=(13,)),
        }

        result = place(parsed, PlacementTarget([507], series_index({})), file_map={"seeded-12.mkv": (506,)})

        assert result.placed == {"left-13.mkv": (507,)}
        assert result.unplaced == ()

    def test_multi_absolute_sharer_in_the_file_map_still_blocks(self) -> None:
        # Every absolute of every parse is counted, so a "12-13" span in the file map shows the unplaced v2's duplicate.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12-13.mkv": parsed_info(season=0, absolutes=(12, 13)),
            "e-12v2.mkv": parsed_info(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = place(parsed, PlacementTarget([2588], series_index({})), file_map={"e-12-13.mkv": (2586, 2587)})

        assert result.placed == {}
        assert by_name(result) == {"e-12v2.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_two_sub_series_restart_blocks_pair_in_order(self) -> None:
        # Two sub-series whose numbering both restart at 1 share absolutes: the tell of a season-boundary
        # scramble, so pairing in absolute order is blocked whole rather than misplacing files.
        main = {f"main-{i:02d}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 4)}
        spinoff = {f"spinoff-{i:02d}.mkv": parsed_info(absolutes=(i,)) for i in range(1, 4)}
        parsed = {**main, **spinoff}

        result = place(parsed, PlacementTarget([501, 502, 503, 601, 602, 603], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(parsed)


class TestPlaceBogusSeasonEpisodeDowngrade:
    """A season and episode in the name that exist nowhere in the series are noise, not identity.

    The downgrade only ever feeds the 1:1 single-file fallback. A season and episode that
    resolve ANYWHERE in the episode index stay real evidence.
    """

    def test_movie_year_bogus_season_episode_places_the_sole_window_episode(self) -> None:
        # "Title.2020" parses S20E20, which the series doesn't have: one file and one id count it as no parsed
        # number.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, PlacementTarget([900], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {"movie.mkv": (900,)}
        assert result.unplaced == ()

    def test_a_resolving_season_and_episode_is_never_downgraded(self) -> None:
        # The same season and episode EXISTS in the series, outside the window: real evidence that puts the file
        # outside it.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, PlacementTarget([900], series_index({EpisodeKey(20, 20): 555})))

        assert result.placed == {}
        assert by_name(result) == {"movie.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_bogus_season_episode_with_absolutes_is_not_downgraded(self) -> None:
        # Absolutes are real signal beside a bogus season and episode, and the multi-absolute span keeps pairing in
        # order blocked too.
        parsed = {"movie.mkv": parsed_info(season=20, episodes=(20,), absolutes=(20, 21))}

        result = place(parsed, PlacementTarget([900], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {}
        assert result.unplaced == ("movie.mkv",)

    def test_bogus_season_episode_with_one_matched_episode_still_places(self) -> None:
        # The name parses a nonexistent S02E00 and Sonarr matched one episode: one matched episode never blocks the
        # fallback.
        parsed = {"sp.mkv": parsed_info(season=2, episodes=(0,), matched=((1, 5),))}

        result = place(parsed, PlacementTarget([900], series_index({EpisodeKey(1, 5): 555})))

        assert result.placed == {"sp.mkv": (900,)}
        assert result.unplaced == ()


class TestPlaceNoNumbersInOrder:
    """Pairing in name order and the single-file pass: with no number left, name order is the only signal."""

    def test_single_file_with_no_number_single_target_is_placed(self) -> None:
        # One unplaced file, one unused id, and Sonarr saw the name and found no number: it's that one.
        result = place({"only.mkv": ParsedFileInfo()}, PlacementTarget([900], series_index({})))

        assert result.placed == {"only.mkv": (900,)}
        assert result.unplaced == ()

    def test_single_none_parse_single_target_is_unplaced(self) -> None:
        # A None parse is no evidence (a blipped v2's absolute may hide behind it). An unparseable
        # name comes back as an all-empty parse, not None, and places as above.
        result = place({"only.mkv": None}, PlacementTarget([900], series_index({})))

        assert result.placed == {}
        assert result.unplaced == ("only.mkv",)

    def test_batch_with_no_numbers_pairs_in_name_order(self) -> None:
        # Name order pairs with airing order regardless of the order the folder lists the files in.
        result = place(blind(["sp2.mkv", "sp1.mkv", "sp3.mkv"]), PlacementTarget([901, 902, 903], series_index({})))

        assert result.placed == {"sp1.mkv": (901,), "sp2.mkv": (902,), "sp3.mkv": (903,)}
        assert result.unplaced == ()

    def test_pair_in_order_orders_digits_naturally(self) -> None:
        # "sp10" sorts after "sp2": lexical order would hand sp10 the second id.
        result = place(blind(["sp1.mkv", "sp2.mkv", "sp10.mkv"]), PlacementTarget([901, 902, 903], series_index({})))

        assert result.placed == {"sp1.mkv": (901,), "sp2.mkv": (902,), "sp10.mkv": (903,)}

    def test_mixed_batch_never_pairs_in_order(self) -> None:
        # The exact pass placing one file leaves more parses than unplaced files: extras with no parsed number never
        # take episodes.
        parsed = {
            "e01.mkv": parsed_info(season=1, episodes=(1,)),
            "interview.mkv": parsed_info(),
            "making of.mkv": parsed_info(),
        }

        result = place(parsed, PlacementTarget([501, 502, 503], series_index({EpisodeKey(1, 1): 501})))

        assert result.placed == {"e01.mkv": (501,)}
        assert sorted(result.unplaced) == ["interview.mkv", "making of.mkv"]

    def test_a_file_map_siblings_parse_kills_pair_in_order(self) -> None:
        # The parse of a file in the file map proves a prior placement, so pairing in order is blocked.
        parsed = blind(["seeded.mkv", "sp1.mkv", "sp2.mkv"])

        result = place(parsed, PlacementTarget([901, 902], series_index({})), file_map={"seeded.mkv": (900,)})

        assert result.placed == {}
        assert sorted(result.unplaced) == ["sp1.mkv", "sp2.mkv"]

    def test_count_mismatch_blocks_both_ways(self) -> None:
        # Two files onto one id, and one file onto two ids: nothing places off a non-1:1 count.
        surplus_files = place(blind(["sp1.mkv", "sp2.mkv"]), PlacementTarget([901], series_index({})))
        surplus_ids = place(blind(["sp1.mkv"]), PlacementTarget([901, 902], series_index({})))

        assert surplus_files.placed == {}
        assert sorted(surplus_files.unplaced) == ["sp1.mkv", "sp2.mkv"]
        assert surplus_ids.placed == {}
        assert surplus_ids.unplaced == ("sp1.mkv",)

    def test_a_parse_the_caller_couldnt_get_blocks_pair_in_order(self) -> None:
        parsed = {**blind(["sp1.mkv", "sp2.mkv"]), "sp3.mkv": None}

        result = place(parsed, PlacementTarget([901, 902, 903], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["sp1.mkv", "sp2.mkv", "sp3.mkv"]

    def test_bogus_season_episode_member_blocks_pair_in_order(self) -> None:
        # The downgrade of a bogus season and episode is 1:1 only: two movies can share one.
        parsed = {**blind(["sp1.mkv", "sp2.mkv"]), "movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, PlacementTarget([901, 902, 903], series_index({})))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["movie.mkv", "sp1.mkv", "sp2.mkv"]


class TestPlaceTitledSingle:
    """The titled single-file pass: a one-episode window, several files with no parsed number, the one a title names."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 901,
        EpisodeKey(2, 1): 601,
        EpisodeKey(2, 2): 602,
    }

    @classmethod
    def _place(cls, parsed: dict[str, ParsedFileInfo | None], *titles: str, series: str = "Show") -> BatchPlacements:
        target = PlacementTarget([901], series_index(cls._ID_BY_KEY), title_names=TitleNames(series, titles))
        return place(parsed, target)

    def test_the_unplaced_file_a_title_names_is_the_episode(self) -> None:
        movie, live = "show - the movie [grp].mkv", "show - live [grp].mkv"

        result = self._place({movie: parsed_info(), live: parsed_info()}, "Show: The Movie")

        assert by_name(result) == {
            movie: ((901,), PlacementVerdict.TITLE_NAME),
            live: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_an_extras_file_set_aside_leaves_the_other_the_single_file(self) -> None:
        # A trailer is never an episode, so the movie is the batch's one file for the one episode.
        movie, trailer = "show - the movie [grp].mkv", "show - trailer [grp].mkv"

        result = self._place({movie: parsed_info(), trailer: parsed_info()})

        assert by_name(result) == {movie: ((901,), PlacementVerdict.SINGLE), trailer: ((), PlacementVerdict.EXTRA)}

    def test_case_and_accents_are_ignored(self) -> None:
        deja, live = "Show - Deja Vu [grp].mkv", "show - live [grp].mkv"

        result = self._place({deja: parsed_info(), live: parsed_info()}, "Show: D\u00e9j\u00e0 Vu")

        assert result.placed == {deja: (901,)}

    def test_a_title_that_is_the_series_names_nothing(self) -> None:
        # The title is the series itself: no word of its name tells one file from another.
        parsed: dict[str, ParsedFileInfo | None] = {
            "long show name - sunny day [grp].mkv": parsed_info(),
            "long show name - prologue [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Long Show Name", "Nagai Show", series="Long Show Name")

        assert result.placed == {}

    def test_a_descriptor_in_a_tag_never_names_the_title(self) -> None:
        # Tags drop away, leaving only the series name, which the title's own leftover words never match.
        parsed: dict[str, ParsedFileInfo | None] = {
            "[grp] show (director's cut) (show II date to date) [bd].mkv": parsed_info(),
            "[grp] show - live [bd].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: Date to Date")

        assert result.placed == {}

    def test_half_the_leftover_words_name_the_file(self) -> None:
        # The season words drop with the series name. "sunny day" is half the combined leftover.
        sunny, prologue = "show - sunny day [grp].mkv", "show - prologue [grp].mkv"

        result = self._place({sunny: parsed_info(), prologue: parsed_info()}, "Show 2nd Season: Sunny Day")

        assert result.placed == {sunny: (901,)}

    def test_an_extras_file_is_never_the_titled_one(self) -> None:
        # The opening's words match the title best, but an opening is never the episode.
        opening = "show II - picture in picture OP [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            opening: parsed_info(),
            "show II - live [grp].mkv": parsed_info(),
            "show II - making of [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show II: Picture in Picture")

        assert result.placed == {}
        assert by_name(result)[opening] == ((), PlacementVerdict.EXTRA)

    def test_a_file_sharing_only_a_title_s_opening_words_is_not_named(self) -> None:
        # The romaji title opens with the franchise name, which the English series name never sheds.
        # A file carrying those words and none after them scores half the leftover yet names the franchise.
        parsed: dict[str, ParsedFileInfo | None] = {
            "[grp] nagai show ni [bd].mkv": parsed_info(),
            "[grp] long show - ova [bd].mkv": parsed_info(),
        }

        result = self._place(
            parsed, "Long Show Two: The Big Finale", "Nagai Show Ni: Ookii Ketsumatsu", series="Long Show"
        )

        assert result.placed == {}

    def test_rejecting_the_best_never_promotes_the_next_in_line(self) -> None:
        # The best match shares only the title's opening words. The file carrying the title's last word
        # scores less and must not become the pick by the rejection.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - alpha beta gamma [grp].mkv": parsed_info(),
            "show - alpha ova [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: Alpha Beta Gamma OVA")

        assert result.placed == {}

    def test_the_romaji_title_names_the_file_the_english_one_does_not(self) -> None:
        movie, special = "show movie endymion no kiseki [grp].mkv", "show movie special [grp].mkv"

        result = self._place(
            {movie: parsed_info(), special: parsed_info()}, "Show: The Miracle of Endymion", "Show: Endymion no Kiseki"
        )

        assert result.placed == {movie: (901,)}

    def test_two_unplaced_files_a_title_names_alike_place_nothing(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - the movie [grp].mkv": parsed_info(),
            "show - the movie [alt].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: The Movie")

        assert result.placed == {}

    def test_without_titles_several_unplaced_files_stay(self) -> None:
        result = self._place({"show - the movie [grp].mkv": parsed_info(), "show - live [grp].mkv": parsed_info()})

        assert result.placed == {}

    def test_an_unknown_parse_in_the_batch_blocks_the_title_pass(self) -> None:
        # The sibling's parse blipped this poll: it may be the episode's own numbered file, so nothing is titled yet.
        parsed: dict[str, ParsedFileInfo | None] = {
            "show - 12 [grp].mkv": None,
            "show - the movie [grp].mkv": parsed_info(),
            "show - trailer [grp].mkv": parsed_info(),
        }

        result = self._place(parsed, "Show: The Movie")

        assert result.placed == {}

    def test_a_full_season_match_of_another_season_is_no_span(self) -> None:
        # "show S2 - OVA": a season token and no episode, matched to every S2 episode. S2 is
        # not the title's, so the file is one unplaced file with no parsed number, and the sole one takes the window.
        name = "show S2 - OVA [grp].mkv"

        result = self._place({name: parsed_info(season=2, matched=((2, 1), (2, 2)), full_season=True)})

        assert by_name(result) == {name: ((901,), PlacementVerdict.SINGLE)}

    def test_a_full_season_match_reaching_into_the_window_spans(self) -> None:
        # The same read over the title's own season is the extras-of-this-season shape: several episodes.
        name = "show S2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=2, matched=((2, 1), (2, 2)), full_season=True)
        }

        result = place(parsed, PlacementTarget([601], series_index(self._ID_BY_KEY)))

        assert result.placed == {}


def _numbered(template: str) -> list[str]:
    """Three names from `template`, numbered 1 to 3."""
    return [template.format(n=n) for n in (1, 2, 3)]


class TestNameNumberForms:
    """Which name shapes carry a name number, read through the sequence pass that consumes them.

    A `1..N` sequence over a 3-wide window places. A shape carrying no number forms
    no sequence at all, so nothing indexes the window.
    """

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
    }

    @classmethod
    def _place(cls, names: Sequence[str], *, blocked: bool = False) -> BatchPlacements:
        """Run three blind names against a 3-wide specials window.

        `blocked` adds a fourth parse so pairing in name order can never place a
        blocked sequence instead, leaving the sequence pass the only thing under test.
        """

        parsed = blind(names)
        return place(
            _pair_in_order_blocked(parsed) if blocked else parsed,
            PlacementTarget([501, 502, 503], series_index(cls._ID_BY_KEY)),
            names,
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
            # A titled name spelling out season and episode carries no middle or trailing number, so its episode is the
            # name number.
            pytest.param(_numbered("[grp] show - S07E0{n} - a titled episode.mkv"), id="the keyed form"),
            # The middle form would split "S01E01 - 01" into three prefixes. The season and episode keep them one
            # sequence.
            pytest.param(_numbered("show S01E0{n} - 0{n} - title.mkv"), id="the key outranks a middle number"),
            # "S0101" parses as a whole season 101 though the release meant episode 01 of season 1.
            # The group's own digits come after it, so the packed form is read before the trailing one.
            pytest.param(_numbered("show.S010{n}.1080p.Blu-ray.x265-grp067.mkv"), id="the packed form"),
            pytest.param(_numbered("show - S2 - 0{n} [1{n}] - title [tag].mkv"), id="past an absolute bracket"),
            pytest.param(_numbered("show - Episode 0{n} - title [tag].mkv"), id="past an episode word"),
            pytest.param(_numbered("show - Ep. 0{n} - title [tag].mkv"), id="past an abbreviated episode word"),
            # A scene-dotted name number, its "vN" behind a dot of its own, between the title and the quality words.
            pytest.param(_numbered("show.00{n}.v4.480p.DVD.FLAC2.0.x264-grp.mkv"), id="the dotted form"),
            # A series past its thousandth episode pads its name number to four digits behind a zero.
            pytest.param(
                _numbered("show - 000{n} (BD 1080p) [grp][ABCD123{n}].mkv"), id="a zero-padded trailing count"
            ),
        ],
    )
    def test_a_name_shape_carrying_a_name_number_forms_the_sequence(self, names: list[str]) -> None:
        assert by_name(self._place(names)) == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(names)
        }

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(["show 2019.mkv", "show 2020.mkv", "show 2021.mkv"], id="a four-digit year"),
            pytest.param(["show.2019.1080p.mkv", "show.2020.1080p.mkv", "show.2021.1080p.mkv"], id="a dotted year"),
            # The tag strip takes the whole bracket off, so a CRC's digits never become a name number.
            pytest.param(
                [f"show part {word} [ABCD123{n}].mkv" for n, word in enumerate(("one", "two", "three"), 1)],
                id="digits only inside a tail tag",
            ),
            pytest.param([f"0{n} - show.mkv" for n in (1, 2, 3)], id="a leading number"),
        ],
    )
    def test_a_name_shape_carrying_no_name_number_forms_no_sequence(self, names: list[str]) -> None:
        assert sorted(self._place(names, blocked=True).unplaced) == sorted(names)

    def test_a_numbered_extras_sequence_is_set_aside(self) -> None:
        # Previews number previews: a "PV 01..03" beside three unreadable specials never takes their window.
        names = _numbered("show - PV 0{n} [tag].mkv")

        result = self._place(names, blocked=True)

        assert {by_name(result)[name] for name in names} == {((), PlacementVerdict.EXTRA)}

    @pytest.mark.parametrize(
        "template",
        ["show - 0{n}{v} [tag].mkv", "show - 0{n}{v} - title [tag].mkv"],
        ids=["trailing", "before the title"],
    )
    def test_a_later_version_displaces_the_earlier_one_in_the_sequence(self, template: str) -> None:
        # Two names share a number: the higher `vN` is the member and the other its duplicate once the sequence
        # places.
        names = [template.format(n=n, v=v) for n, v in ((1, ""), (2, ""), (2, "v2"), (3, ""))]

        result = self._place(names)

        assert result.placed == {names[0]: (501,), names[2]: (502,), names[3]: (503,)}
        assert by_name(result)[names[1]] == ((), PlacementVerdict.DUPLICATE)

    def test_equal_versions_of_one_number_break_the_sequence(self) -> None:
        names = ["show - 01 [tag].mkv", "show - 02 [tag].mkv", "show - 02 [other].mkv", "show - 03 [tag].mkv"]

        assert self._place(names).placed == {}


class TestPlaceSequenceOverParse:
    """Sequence over parse: the batch's one `1..N` sequence indexes a one-season window Sonarr read incoherently."""

    _WINDOW: ClassVar[list[int]] = [501, 502, 503]
    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
    }
    # A whole first season beside the specials, for the franchise torrent shapes.
    _SEASONED: ClassVar[dict[EpisodeKey, int]] = {**_ID_BY_KEY, **{EpisodeKey(1, n): 600 + n for n in (3, 4, 5, 6)}}
    # Six specials beside the six-episode season, for the specials-window shapes.
    _WIDE_SPECIALS: ClassVar[dict[EpisodeKey, int]] = {**_SEASONED, **{EpisodeKey(0, n): 500 + n for n in (4, 5, 6)}}
    _RUN: ClassVar[list[str]] = numbered_names("sp", 3)

    @classmethod
    def _target(cls) -> PlacementTarget:
        return PlacementTarget(cls._WINDOW, series_index(cls._ID_BY_KEY))

    @classmethod
    def _ran(cls) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
        return {name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(cls._RUN)}

    def test_a_parse_of_nothing_lets_the_sequence_index_the_window(self) -> None:
        # Sonarr read no number from any member, so the release's own 1..N is all there is.
        assert by_name(place(blind(self._RUN), self._target())) == self._ran()

    def test_matched_episodes_outside_the_window_do_not_stand(self) -> None:
        # Every member matched the same out-of-window episode: incoherent, so the sequence wins.
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(matched=((1, 1),)) for name in self._RUN}

        assert by_name(place(parsed, self._target())) == self._ran()

    def test_the_season_and_episode_in_a_members_name_outside_a_specials_window_is_overridden(self) -> None:
        # A TVDB-shifted special names another season. Over a season-0 window the sequence stands.
        parsed = blind(self._RUN)
        parsed[self._RUN[0]] = parsed_info(season=1, episodes=(1,))

        assert by_name(place(parsed, self._target())) == self._ran()

    def test_bogus_seasons_and_episodes_do_not_stand(self) -> None:
        # Seasons and episodes that exist nowhere in the series are parse artifacts, never parsed episodes.
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(season=20, episodes=(20,)) for name in self._RUN}

        assert by_name(place(parsed, self._target())) == self._ran()

    _S_AND_NUMBER_NAMES: ClassVar[list[str]] = [f"[grp] Show S{n} (BD 1080p).mkv" for n in (1, 2, 3)]

    @classmethod
    def _beside_the_season(cls, specials: Sequence[str]) -> dict[str, ParsedFileInfo | None]:
        """Sonarr reads a lone "S1" as a whole first season, and the "- 01".."- 06" beside it as its episodes."""

        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(season=1, full_season=True) for name in specials}
        for n, name in enumerate(numbered_names("[grp] Show", 6), 1):
            parsed[name] = parsed_info(season=1, episodes=(n,))
        return parsed

    def test_a_torrents_s_and_number_set_indexes_the_specials_window(self) -> None:
        # Zog42 numbers specials "S1".."S3" beside "- 01".."- 06": the set is the specials' own 1..N.
        target = PlacementTarget(self._WINDOW, series_index(self._SEASONED))
        placed = by_name(place(self._beside_the_season(self._S_AND_NUMBER_NAMES), target))

        assert {name: placed[name] for name in self._S_AND_NUMBER_NAMES} == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(self._S_AND_NUMBER_NAMES)
        }

    def test_one_s_and_number_beside_the_season_is_a_season_token(self) -> None:
        target = PlacementTarget(self._WINDOW, series_index(self._SEASONED))
        placed = by_name(place(self._beside_the_season(self._S_AND_NUMBER_NAMES[:1]), target))

        assert placed[self._S_AND_NUMBER_NAMES[0]][1] is not PlacementVerdict.SEQUENCE_OVER_PARSE
        assert placed[self._S_AND_NUMBER_NAMES[0]][0] == ()

    _TITLED: ClassVar[list[str]] = [
        f"Wisp.100.Lantern.Keepers.of.the.Moor.{n:02d}.1080p.WEB.H264-grp.mkv" for n in (1, 2, 3)
    ]

    @pytest.mark.parametrize(
        "names",
        [
            TitleNames("Wisp 100: Lantern Keepers of the Moor"),
            TitleNames("", ("Wisp 100: Lantern Keepers of the Moor",)),
            TitleNames("Wisp 100: Lantern Keepers of the Moor", ("Wisp 100: Lantern Keepers of the Moor",)),
        ],
    )
    def test_the_titles_name_lets_its_numbered_torrent_place(self, names: TitleNames) -> None:
        # Sonarr read nothing from the dotted names.
        target = PlacementTarget(self._WINDOW, series_index(self._ID_BY_KEY), title_names=names)

        assert by_name(place(blind(self._TITLED), target)) == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(self._TITLED)
        }

    @pytest.mark.parametrize("names", [TitleNames("Show 2"), TitleNames("Show", ("Show Season 2", "Show 2"))])
    def test_a_name_number_equal_to_the_titles_season_digit_keeps_the_sequence(self, names: TitleNames) -> None:
        # A second season's torrent numbers "- 1".."- 3" unpadded.
        run = [f"[grp] Show - {n} (1080p).mkv" for n in (1, 2, 3)]
        target = PlacementTarget(self._WINDOW, series_index(self._ID_BY_KEY), title_names=names)

        assert by_name(place(blind(run), target)) == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(run)
        }

    @pytest.mark.parametrize("names", [TitleNames(), TitleNames("Wisp 100"), TitleNames("", ("Wispish 100",))])
    def test_without_the_title_the_numbered_torrent_is_not_read_as_a_sequence(self, names: TitleNames) -> None:
        target = PlacementTarget(self._WINDOW, series_index(self._ID_BY_KEY), title_names=names)

        assert PlacementVerdict.SEQUENCE_OVER_PARSE not in verdicts_of(place(blind(self._TITLED), target))

    def test_sonarr_numbering_by_the_title_does_not_contradict_the_torrent(self) -> None:
        # Sonarr read the title's 100 as each file's absolute, an episode the series lacks.
        parsed = {name: parsed_info(season=0, absolutes=(100,)) for name in self._TITLED}
        titled = PlacementTarget(
            self._WINDOW, series_index(self._ID_BY_KEY), title_names=TitleNames("Wisp 100: Lantern Keepers of the Moor")
        )

        assert by_name(place(parsed, titled)) == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(self._TITLED)
        }
        assert PlacementVerdict.SEQUENCE_OVER_PARSE not in verdicts_of(place(parsed, self._target()))

    def test_a_coherent_permuted_parse_stands(self) -> None:
        # Every member reads one distinct id inside the window, so Sonarr's
        # parse decides placement even though it permutes the sequence's order.
        parsed: dict[str, ParsedFileInfo | None] = {
            self._RUN[0]: parsed_info(season=0, episodes=(3,)),
            self._RUN[1]: parsed_info(season=0, episodes=(1,)),
            self._RUN[2]: parsed_info(season=0, episodes=(2,)),
        }

        assert by_name(place(parsed, self._target())) == {
            self._RUN[0]: ((503,), PlacementVerdict.EXACT),
            self._RUN[1]: ((501,), PlacementVerdict.EXACT),
            self._RUN[2]: ((502,), PlacementVerdict.EXACT),
        }

    def test_two_sequences_of_one_width_block_and_suppress_the_numbered_sequence(self) -> None:
        # Which sequence owns the window is unknowable, and the later blind pass must not guess either.
        names = [*numbered_names("a", 3), *numbered_names("b", 3)]
        result = place(blind(names), self._target())

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(names)

    def test_a_width_one_window_blocks_the_pass(self) -> None:
        # A one-id window is no sequence: the single-file pass is what places it.
        result = place({self._RUN[0]: parsed_info()}, PlacementTarget([501], series_index(self._ID_BY_KEY)))

        assert by_name(result) == {self._RUN[0]: ((501,), PlacementVerdict.SINGLE)}

    def test_specials_numbered_apart_that_are_not_the_whole_title_stay_unplaced(self) -> None:
        # The title also holds S01E01, so specials 1, 2 and 4 aren't its whole order, though they aired in turn.
        gappy = {EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(0, 4): 504, EpisodeKey(1, 1): 601}
        aired = {501: "2016-01-01", 502: "2016-02-01", 504: "2016-04-01"}
        parsed = blind(self._RUN)
        parsed["pack.mkv"] = parsed_info(season=1, episodes=(1,))

        result = place(parsed, PlacementTarget([601, 501, 502, 504], series_index(gappy, aired=aired)))

        assert result.placed == {"pack.mkv": (601,)}
        assert sorted(result.unplaced) == sorted(self._RUN)

    # A title's three files beside two shorts of another title: more files than the title has episodes, so only
    # the title's own sequence can place them.
    _TRIO: ClassVar[list[str]] = numbered_names("sp", 3)
    _BESIDE_SHORTS: ClassVar[list[str]] = [*_TRIO, *numbered_names("short", 2)]

    @staticmethod
    def _apart(season: int, aired: Mapping[int, str | None] | None = None) -> EpisodeIndex:
        """Six episodes of one season: the first, third and sixth aired three months apart, unless `aired` says."""

        ids = {EpisodeKey(season, n): 500 + n for n in range(1, 7)}
        return series_index(ids, aired={501: "2016-03-01", 503: "2016-06-01", 506: "2016-09-01"} | dict(aired or {}))

    @pytest.mark.parametrize(
        ("resolved", "episode_count"),
        [
            pytest.param([501, 503, 506], None, id="anilist-gives-no-count"),
            pytest.param([501, 503, 506], 3, id="anilist-counts-the-scopes-episodes"),
            pytest.param([506, 501, 503], None, id="a-scope-out-of-number-order"),
        ],
    )
    def test_a_titles_specials_numbered_apart_take_its_sequence(
        self, resolved: list[int], episode_count: int | None
    ) -> None:
        # The title is specials 1, 3 and 6, aired months apart.
        target = PlacementTarget(resolved, self._apart(0), episode_count=episode_count)

        result = place(blind(self._BESIDE_SHORTS), target)

        placed = by_name(result)
        assert result.placed == {self._TRIO[0]: (501,), self._TRIO[1]: (503,), self._TRIO[2]: (506,)}
        assert {placed[name][1] for name in self._TRIO} == {PlacementVerdict.SEQUENCE_OVER_PARSE}

    @pytest.mark.parametrize(
        "aired",
        [
            pytest.param({506: "2016-06-01"}, id="the-last-tied"),
            pytest.param({506: "2016-05-01"}, id="the-last-earlier"),
            pytest.param({506: None}, id="the-last-undated"),
            pytest.param({501: None}, id="the-first-undated"),
        ],
    )
    def test_specials_numbered_apart_that_did_not_air_in_turn_place_nothing(self, aired: dict[int, str | None]) -> None:
        # Nothing but the air dates says the higher numbers came out later.
        result = place(blind(self._BESIDE_SHORTS), PlacementTarget([501, 503, 506], self._apart(0, aired)))

        assert result.placed == {}

    @pytest.mark.parametrize("episode_count", [4, 2], ids=["an-episode-sonarr-lacks", "fewer-than-the-scope"])
    def test_specials_numbered_apart_place_nothing_when_anilist_counts_another_number_of_episodes(
        self, episode_count: int
    ) -> None:
        # With a count of four, the title has an episode Sonarr lacks, and nothing says where in its order it falls.
        target = PlacementTarget([501, 503, 506], self._apart(0), episode_count=episode_count)

        assert place(blind(self._BESIDE_SHORTS), target).placed == {}

    def test_specials_numbered_apart_take_a_sequence_spelled_out_as_its_own_season(self) -> None:
        # The release numbers the title as a first season of its own. Over specials that season and episode doesn't
        # block the sequence.
        ids = {EpisodeKey(0, n): 500 + n for n in range(1, 7)} | {EpisodeKey(1, n): 600 + n for n in range(1, 4)}
        index = series_index(ids, aired={501: "2016-03-01", 503: "2016-06-01", 506: "2016-09-01"})
        spelled_names = [f"Spin.S01E{n:02d}.mkv" for n in range(1, 4)]
        parsed = blind(numbered_names("short", 2))
        parsed.update({name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(spelled_names, start=1)})

        result = place(parsed, PlacementTarget([501, 503, 506], index))

        assert result.placed == {spelled_names[0]: (501,), spelled_names[1]: (503,), spelled_names[2]: (506,)}

    def test_a_seasons_episodes_numbered_apart_place_nothing(self) -> None:
        # Episodes 1, 3 and 6 of a season: the sequence would have to count the episodes between them.
        result = place(blind(self._BESIDE_SHORTS), PlacementTarget([501, 503, 506], self._apart(1)))

        assert result.placed == {}

    def test_specials_numbered_apart_with_consecutive_absolutes_follow_the_absolutes(self) -> None:
        # Special 1 is absolute 14 and special 3 is absolute 13, so the torrent's 13 and 14 go where the absolutes
        # say.
        ids = {EpisodeKey(0, n): 500 + n for n in (1, 2, 3)}
        index = series_index(ids, absolutes={501: 14, 503: 13}, aired={501: "2016-03-01", 503: "2016-06-01"})
        names = ["show - 13 [grp].mkv", "show - 14 [grp].mkv"]
        parsed = {names[0]: parsed_info(absolutes=(13,)), names[1]: parsed_info(absolutes=(14,))}

        result = place(parsed, PlacementTarget([501, 503], index))

        assert result.placed == {names[0]: (503,), names[1]: (501,)}

    def test_a_two_season_window_blocks_the_pass(self) -> None:
        # The extra parse keeps pairing in name order out, so the block is what is pinned.
        parsed = _pair_in_order_blocked(blind(self._RUN))

        result = place(parsed, PlacementTarget([501, 502, 601], series_index(self._ID_BY_KEY)), self._RUN)

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(self._RUN)

    def test_a_member_whose_absolute_only_number_disagrees_leaves_the_sequence(self) -> None:
        # Its "02" was not the release's name number, so no full-width sequence is left to fit.
        parsed = blind(self._RUN)
        parsed[self._RUN[1]] = parsed_info(absolutes=(9,))

        result = place(parsed, self._target())

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(self._RUN)

    def test_a_member_whose_own_numbers_name_several_episodes_blocks_the_sequence(self) -> None:
        # Its name parsed as absolutes 1 and 2, two episodes where the sequence needs one.
        parsed = blind(self._RUN)
        parsed[self._RUN[0]] = parsed_info(absolutes=(1, 2))

        assert by_name(place(parsed, self._target())) == dict.fromkeys(
            self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND)
        )

    def test_a_full_season_read_never_blocks(self) -> None:
        # A member read as a whole season ("S0101" is season 101, "S1 - 02" is
        # season 1) has no episode token, so the sequence still indexes the window.
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=101, full_season=True) for name in self._RUN
        }
        parsed[self._RUN[1]] = parsed_info(season=1, full_season=True)

        assert by_name(place(parsed, self._target())) == self._ran()

    def test_a_sequence_numbered_as_the_windows_episodes_indexes_it(self) -> None:
        # A split cour's second half numbers on from the first: 12..14 over S01E12..E14.
        target = PlacementTarget([612, 613, 614], series_index({EpisodeKey(1, n): 600 + n for n in (12, 13, 14)}))
        run = [f"sp - {n} [grp].mkv" for n in (12, 13, 14)]
        parsed = blind(run)

        result = place(parsed, target)

        assert by_name(result) == {
            name: ((612 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(run)
        }

    def test_a_sequence_from_anywhere_indexes_a_window_sonarr_read_nothing_of(self) -> None:
        # A release numbering the whole series: 14..16 over a 3-wide window, no member read.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(season=7, episodes=(n,)) for n, name in zip((14, 15, 16), run, strict=True)
        }

        result = place(parsed, self._target())

        assert by_name(result) == {
            name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(run)
        }

    def test_a_sequence_from_anywhere_needs_every_member_read_blind(self) -> None:
        # One member Sonarr did read says the numbering is not the series' own: the sequence from anywhere stands
        # down.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = blind(run)
        parsed[run[0]] = parsed_info(matched=((1, 1),))

        result = place(_pair_in_order_blocked(parsed), self._target(), run)

        assert by_name(result) == {
            run[0]: ((), PlacementVerdict.OUTSIDE_WINDOW),
            run[1]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[2]: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_a_whole_season_sequence_read_across_two_seasons_still_blocks_a_sequence_from_anywhere(self) -> None:
        # Sonarr matched the 1..4 sequence onto two other seasons, so it lands nowhere: rejected, it still outranks the
        # sequence from anywhere, and nothing places. Read whole into one other season, it would be OUTSIDE_WINDOW and
        # the sequence from anywhere would place.
        index = series_index({EpisodeKey(s, e): s * 100 + e for s in (1, 2, 3) for e in range(1, 5)})
        covering = numbered_names("A", 4)
        anywhere = [f"B - {n:02d} [grp].mkv" for n in (5, 6)]
        reads = ((1, 1), (1, 2), (3, 1), (3, 2))
        parsed = {
            **{name: parsed_info(matched=(pair,)) for name, pair in zip(covering, reads, strict=True)},
            **blind(anywhere),
        }

        result = place(parsed, PlacementTarget([201, 202], index))

        assert by_name(result) == dict.fromkeys((*covering, *anywhere), ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_one_to_n_sequence_outranks_a_sequence_from_anywhere(self) -> None:
        # Both fit the width. The release's own 1..N is the numbering, the other sequence is extras.
        offset = [f"extra - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = blind((*self._RUN, *offset))

        result = place(parsed, self._target())

        assert by_name(result) == {**self._ran(), **dict.fromkeys(offset, ((), PlacementVerdict.NO_EPISODE_FOUND))}

    def test_two_sequences_from_anywhere_are_ambiguous(self) -> None:
        run_a = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        run_b = [f"extra - {n} [grp].mkv" for n in (20, 21, 22)]
        parsed = _pair_in_order_blocked(blind((*run_a, *run_b)))

        result = place(parsed, self._target(), [*run_a, *run_b])

        assert result.placed == {}

    def test_several_fitting_sequences_leave_the_one_a_title_names(self) -> None:
        # A franchise torrent: two 1..3 sequences fit the window, and the title's own names pick one.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        parsed = blind((*alpha, *beta))

        result = place(
            parsed,
            PlacementTarget(
                self._WINDOW, series_index(self._ID_BY_KEY), title_names=TitleNames("Show", ("Show Beta",))
            ),
        )

        assert by_name(result) == {
            **{name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(beta)},
            **dict.fromkeys(alpha, ((), PlacementVerdict.NO_EPISODE_FOUND)),
        }

    def test_a_title_that_is_the_series_names_no_sequence(self) -> None:
        # The title is the series itself, so its name decides nothing and the torrent stays ambiguous.
        alpha = numbered_names("show alpha", 3)
        beta = numbered_names("show beta", 3)
        parsed = blind((*alpha, *beta))

        result = place(
            parsed,
            PlacementTarget(self._WINDOW, series_index(self._ID_BY_KEY), title_names=TitleNames("Show", ("Show",))),
        )

        assert result.placed == {}

    def test_a_sequence_sonarr_read_whole_outside_the_window_stands_aside(self) -> None:
        # A franchise torrent: the AniList name names the base sequence, but Sonarr read that sequence whole into
        # season 1, so the blind sequel sequence takes the window instead.
        base = numbered_names("show name", 3)
        sequel = numbered_names("show name season two", 3)
        parsed: dict[str, ParsedFileInfo | None] = {
            **{name: parsed_info(absolutes=(i,), matched=((1, i),)) for i, name in enumerate(base, 1)},
            **blind(sequel),
        }
        target = PlacementTarget(
            self._WINDOW, series_index(self._SEASONED), title_names=TitleNames("Show", ("Show Name",))
        )

        result = place(parsed, target)

        assert by_name(result) == {
            **{name: ((501 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(sequel)},
            **dict.fromkeys(base, ((), PlacementVerdict.OUTSIDE_WINDOW)),
        }

    def test_a_sequence_numbered_as_the_window_outranks_a_sequence_from_one(self) -> None:
        # A two-cour torrent for the second cour: Sonarr read the first cour whole into the first half.
        first = numbered_names("show part 1", 3)
        second = [f"show part 2 - 0{i} [grp].mkv" for i in (4, 5, 6)]
        parsed: dict[str, ParsedFileInfo | None] = {
            **{name: parsed_info(absolutes=(i,), matched=((1, i),)) for i, name in enumerate(first, 1)},
            **blind(second),
        }
        window = [604, 605, 606]

        result = place(parsed, PlacementTarget(window, series_index(self._SEASONED)))

        assert by_name(result) == {
            **{name: ((window[i],), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(second)},
            **dict.fromkeys(first, ((), PlacementVerdict.OUTSIDE_WINDOW)),
        }

    def test_a_blocked_title_pick_keeps_the_numbered_sequence_down(self) -> None:
        # Several sequences fit, the title picks one, and a member's season and episode block it: the losing blind
        # sequence must not index the window through the numbered sequence pass.
        picked = numbered_names("show wrath", 3)
        other = numbered_names("show revival", 3)
        parsed = blind((*picked, *other))
        parsed[picked[1]] = parsed_info(season=1, episodes=(2,))
        target = PlacementTarget(
            [604, 605, 606], series_index(self._SEASONED), title_names=TitleNames("Show", ("Show: Wrath",))
        )

        result = place(parsed, target)

        assert result.placed == {}

    def test_an_empty_episode_index_blocks_the_pass(self) -> None:
        # With no index there is no window to index, and the extra parse keeps pairing in name order out.
        parsed = _pair_in_order_blocked(blind(self._RUN))

        result = place(parsed, PlacementTarget(self._WINDOW, series_index({})), self._RUN)

        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(self._RUN)

    def test_an_unknown_parse_keeps_the_members_waiting(self) -> None:
        # One unreadable name anywhere in the batch, and no pass may place the sequence.
        parsed = blind(self._RUN)
        parsed[_GONE] = None

        result = place(parsed, self._target(), self._RUN)

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.WAITING_ON_PARSE}

    def test_a_waiting_sequence_keeps_its_superseded_versions_waiting_too(self) -> None:
        # The earlier version is the sequence's duplicate once it places, so it waits with the members
        # rather than taking the episode by its own exact season and episode.
        names = [*self._RUN, "sp - 02v2 [grp].mkv"]
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((0, i),)) for i, name in enumerate(self._RUN, 1)
        }
        parsed[names[3]] = parsed_info(matched=((0, 2),))
        parsed[_GONE] = None

        result = place(parsed, self._target(), names)

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.WAITING_ON_PARSE}

    def test_the_season_and_episode_in_a_members_name_in_another_regular_season_stands_the_sequence_down(self) -> None:
        # Over a REGULAR-season window a member naming another season is
        # evidence the torrent is mislisted, so nothing is indexed onto it.
        regular = {EpisodeKey(2, 1): 701, EpisodeKey(2, 2): 702, EpisodeKey(2, 3): 703, EpisodeKey(1, 1): 601}
        parsed = blind(self._RUN)
        parsed[self._RUN[0]] = parsed_info(season=1, episodes=(1,))

        result = place(parsed, PlacementTarget([701, 702, 703], series_index(regular)))

        assert result.placed == {}
        assert by_name(result)[self._RUN[0]] == ((), PlacementVerdict.OUTSIDE_WINDOW)
        assert sorted(result.unplaced) == sorted(self._RUN[1:])

    def test_a_non_member_parse_inside_the_window_blocks_the_sequence(self) -> None:
        # Another file owns one of the slots, so the sequence does not own the window whole.
        parsed = blind(self._RUN)
        parsed["extra.mkv"] = parsed_info(season=0, episodes=(2,))

        result = place(parsed, self._target())

        assert result.placed == {"extra.mkv": (502,)}
        assert sorted(result.unplaced) == sorted(self._RUN)

    def test_a_member_sonarr_matched_to_two_episodes_stays_in_the_sequence(self) -> None:
        # Sonarr's scene mapping reading one member as a double episode is the incoherence
        # the sequence overrides (measured: every such torrent was N files for N episodes).
        parsed = blind(self._RUN)
        parsed[self._RUN[1]] = parsed_info(matched=((0, 2), (0, 3)))

        result = place(parsed, self._target())

        assert result.placed == dict(zip(self._RUN, ((501,), (502,), (503,)), strict=True))
        assert {p.verdict for p in result.placements} == {PlacementVerdict.SEQUENCE_OVER_PARSE}

    def test_a_name_the_parses_never_covered_waits_like_an_unread_parse(self) -> None:
        # No caller builds this (both parse every name they place), but a name to place with no parse entry still
        # counts as unknown, like a None.
        parsed = blind(self._RUN[:2])

        result = place(parsed, self._target(), self._RUN)

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.WAITING_ON_PARSE}

    @pytest.mark.parametrize(
        ("title", "placed"),
        [
            pytest.param("Show: Wrath", True, id="names the fit"),
            pytest.param("Show: Revival", False, id="names the other"),
        ],
    )
    def test_a_title_keeps_the_fit_it_names_and_blocks_one_it_does_not(self, title: str, placed: bool) -> None:
        # One sequence fits the window and a wider one does not. The title naming the fit keeps it. Naming the
        # other blocks the fit (the named sequence is never promoted onto a window it does not fit) and the
        # numbered sequence pass stands down with it.
        fit = numbered_names("show wrath", 3)
        other = numbered_names("show revival", 4)
        parsed = blind((*fit, *other))
        target = PlacementTarget(self._WINDOW, series_index(self._SEASONED), title_names=TitleNames("Show", (title,)))

        result = place(parsed, target)

        assert result.placed == ({name: (501 + i,) for i, name in enumerate(fit)} if placed else {})

    @pytest.mark.parametrize("title", ["Show 3rd Season", "Show Third Season", "Show Season 3", "Show III", "Show 3"])
    def test_a_season_numbered_any_way_names_the_sequence_numbered_that_way(self, title: str) -> None:
        # AniList numbers a season as it likes and a release as it likes: both reduce to the plain number.
        second = numbered_names("show s2", 3)
        third = numbered_names("show s3", 3)
        parsed = blind((*second, *third))
        target = PlacementTarget(self._WINDOW, series_index(self._SEASONED), title_names=TitleNames("Show", (title,)))

        result = place(parsed, target)

        assert result.placed == {name: (501 + i,) for i, name in enumerate(third)}

    def test_a_whole_season_sequence_over_a_part_season_window_places_the_window(self) -> None:
        # A season pack listed on a cour's entry: the sequence numbers the whole season, the window is two of
        # its episodes, and the members past the window are the other cour's.
        run = numbered_names("show", 6)
        parsed = blind(run)

        result = place(parsed, PlacementTarget([603, 604], series_index(self._SEASONED)))

        assert by_name(result) == {
            run[2]: ((603,), PlacementVerdict.SEQUENCE_OVER_PARSE),
            run[3]: ((604,), PlacementVerdict.SEQUENCE_OVER_PARSE),
            **dict.fromkeys((run[0], run[1], run[4], run[5]), ((), PlacementVerdict.OUTSIDE_WINDOW)),
        }

    def test_a_coherent_parse_of_the_window_stands(self) -> None:
        run = numbered_names("show", 6)
        parsed = blind(run)
        parsed[run[2]] = parsed_info(season=1, episodes=(3,))
        parsed[run[3]] = parsed_info(season=1, episodes=(4,))

        result = place(parsed, PlacementTarget([603, 604], series_index(self._SEASONED)))

        assert {name: by_name(result)[name] for name in (run[2], run[3])} == {
            run[2]: ((603,), PlacementVerdict.EXACT),
            run[3]: ((604,), PlacementVerdict.EXACT),
        }

    def test_a_specials_window_takes_no_whole_season_sequence(self) -> None:
        # A `1..6` sequence Sonarr read nothing from, on a title of two of the six specials: neither its numbering
        # nor a part of it indexes the specials.
        run = numbered_names("show", 6)
        parsed = blind(run)

        result = place(parsed, PlacementTarget([501, 502], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.NO_EPISODE_FOUND))

    @pytest.mark.parametrize(
        ("read", "verdict"),
        [
            pytest.param(6, PlacementVerdict.OUTSIDE_WINDOW, id="whole"),
            pytest.param(3, PlacementVerdict.NO_EPISODE_FOUND, id="half"),
        ],
    )
    def test_a_sequence_sonarr_read_into_another_season_is_no_whole_season_sequence(
        self, read: int, verdict: PlacementVerdict
    ) -> None:
        # A shorts sequence as wide as the season, which Sonarr read into the specials: read whole, it is
        # theirs, and read only in part, it is nowhere.
        run = numbered_names("show mini", 6)
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((0, i),)) if i <= read else parsed_info() for i, name in enumerate(run, 1)
        }

        result = place(parsed, PlacementTarget([604, 605, 606], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), verdict))

    def test_a_sequence_read_across_the_season_and_the_specials_is_nowhere(self) -> None:
        # Sonarr read the first file into the season and the rest onto the specials: no one season
        # takes the sequence, so its files are neither the title's nor outside the window.
        run = numbered_names("show", 6)
        reads = ((1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=(pair,)) for name, pair in zip(run, reads, strict=True)
        }

        result = place(parsed, PlacementTarget([604, 605, 606], series_index(self._WIDE_SPECIALS)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_sequences_own_lower_version_inside_the_window_does_not_block_it(self) -> None:
        # The displaced `- 02` reads as the second episode, as its `- 02v2` does: a member's own version
        # is no rival file, so the sequence still indexes the third episode Sonarr read nothing for.
        run = numbered_names("show", 6)
        later = "show - 02v2 [grp].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(matched=((1, i),)) if i != 3 else parsed_info() for i, name in enumerate(run, 1)
        }
        parsed[later] = parsed_info(matched=((1, 2),))
        window = [601, 602, 603, 604, 605, 606]

        result = place(parsed, PlacementTarget(window, series_index(self._SEASONED)))

        members = [run[0], later, *run[2:]]
        assert by_name(result) == {
            **{
                name: ((ep_id,), PlacementVerdict.SEQUENCE_OVER_PARSE)
                for name, ep_id in zip(members, window, strict=True)
            },
            run[1]: ((), PlacementVerdict.DUPLICATE),
        }

    @pytest.mark.parametrize("beside", [False, True], ids=["alone", "beside another run"])
    def test_a_sequence_read_partly_into_another_season_places_nothing_by_its_reads(self, beside: bool) -> None:
        # A plain sequel torrent Sonarr read by absolute number, two files into the first season and the rest
        # from the sequel's start: the read that lands in the window is as doubtful as the rest.
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

        result = place(parsed, PlacementTarget([704, 705, 706], series_index(two_seasons)))

        assert by_name(result) == dict.fromkeys((*run, *other), ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_blocked_pick_stands_the_numbered_sequence_down(self) -> None:
        # A file spelling out its season and episode inside the window blocks the season's sequence and takes its
        # episode: the two blind extras that now fit what is left do not take it.
        run = numbered_names("show", 6)
        spelled_name = "show S01E04 [grp].mkv"
        extras = numbered_names("show extra", 2)
        parsed = blind((*run, *extras))
        parsed[spelled_name] = parsed_info(season=1, episodes=(4,), matched=((1, 4),))

        result = place(parsed, PlacementTarget([604, 605, 606], series_index(self._SEASONED)))

        assert by_name(result) == {
            spelled_name: ((604,), PlacementVerdict.EXACT),
            **dict.fromkeys((*run, *extras), ((), PlacementVerdict.NO_EPISODE_FOUND)),
        }

    def test_a_season_with_a_numbering_gap_takes_no_whole_season_sequence(self) -> None:
        # TVDB skips the third episode, so the season counts five and a 1..5 sequence fits its count while its
        # numbers go on past the gap: the window is not the sequence's fourth and fifth.
        gapped = {key: ep_id for key, ep_id in self._SEASONED.items() if key != EpisodeKey(1, 3)}
        run = numbered_names("show", 5)
        parsed = blind(run)

        result = place(parsed, PlacementTarget([604, 605, 606], series_index(gapped)))

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_an_earlier_window_owning_part_of_the_target_keeps_a_sequence_from_anywhere_out(self) -> None:
        # Three files fit three unused ids by chance once an earlier window used the rest of the target:
        # the numbering only speaks for the whole target.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        parsed = _pair_in_order_blocked(blind(run))
        target = PlacementTarget(
            [601, 602, 603, 604, 605, 606], series_index(self._SEASONED), used=frozenset({601, 602, 603})
        )

        result = place(parsed, target, run)

        assert result.placed == {}

    def test_a_file_map_owning_part_of_the_target_keeps_a_sequence_from_anywhere_out(self) -> None:
        # The same three slots left over, used by the file map instead of an earlier window: without the guard
        # the specials would take 604-606 as a sequence.
        run = [f"sp - {n} [grp].mkv" for n in (14, 15, 16)]
        mapped = {
            f"show - {n:02d} [grp].mkv": parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n in (1, 2, 3)
        }
        parsed = _pair_in_order_blocked({**mapped, **blind(run)})
        target = PlacementTarget([601, 602, 603, 604, 605, 606], series_index(self._SEASONED))

        result = place(parsed, target, run, file_map={name: (600 + n,) for n, name in enumerate(mapped, 1)})

        assert result.placed == {}


class TestPlaceAbsoluteWindow:
    """Sequence over parse on a window spanning seasons: the absolute numbering orders it when every slot has one."""

    _RUN: ClassVar[list[str]] = numbered_names("show", 3)
    _WINDOW: ClassVar[list[int]] = [501, 601, 602]
    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(1, 1): 601, EpisodeKey(0, 1): 501, EpisodeKey(1, 2): 602}
    """S01E01, then the special TVDB interleaved after it, then S01E02."""

    @classmethod
    def _place(cls, *absolutes: int | None, blocked: bool = False) -> BatchPlacements:
        """Sonarr's shifted parse over the slots carrying `absolutes`: the third file collides with the second."""

        carried = {ep_id: n for ep_id, n in zip(cls._ID_BY_KEY.values(), absolutes, strict=True) if n is not None}
        parsed: dict[str, ParsedFileInfo | None] = {
            cls._RUN[0]: parsed_info(matched=((1, 1),)),
            cls._RUN[1]: parsed_info(matched=((1, 2),)),
            cls._RUN[2]: parsed_info(matched=((1, 2),)),
        }
        target = PlacementTarget(cls._WINDOW, series_index(cls._ID_BY_KEY, absolutes=carried))
        return place(_pair_in_order_blocked(parsed) if blocked else parsed, target, cls._RUN)

    def test_the_sequence_follows_the_absolute_order_across_seasons(self) -> None:
        # A title holding a special TVDB interleaved: "02" is absolute 2, the special, whatever season it sits in.
        result = self._place(1, 2, 3)

        assert result.placed == {self._RUN[0]: (601,), self._RUN[1]: (501,), self._RUN[2]: (602,)}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.SEQUENCE_OVER_PARSE}

    @pytest.mark.parametrize("absolutes", [(1, None, 3), (1, 2, 4)])
    def test_without_an_absolute_order_the_parse_stands(self, absolutes: tuple[int | None, ...]) -> None:
        # A slot with no absolute, or a gap between them, leaves the sequence nothing to index by: the parse
        # stands, and the collided third file is the one unplaced file onto the one unused slot.
        result = self._place(*absolutes, blocked=True)

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((501,), PlacementVerdict.SINGLE),
        }


class TestPlaceSequenceTitleEvidence:
    """Sequence over parse among several fitting sequences: the members' episode titles pick one, or block the pick."""

    _WINDOW: ClassVar[list[int]] = [501, 502, 503]
    _TITLES: ClassVar[dict[int, str]] = {
        501: "Beach Day",
        502: "Hot Springs",
        503: "Festival Night",
        601: "Pilot Flight",
        602: "The Return",
    }

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
    }
    # Two 1..3 sequences fitting a franchise torrent's specials window: one titled as the window's episodes, one as
    # another season's.
    _INSIDE: ClassVar[list[str]] = numbered_names(
        "show a", 3, ("Beach Day 1080p", "Hot Springs 1080p", "Festival Night 1080p")
    )
    _OUTSIDE: ClassVar[list[str]] = numbered_names("show b", 3, ("Pilot Flight 1080p", "The Return 1080p", "Bonus"))

    @classmethod
    def _place(
        cls, *runs: list[str], retitled: Mapping[int, str] | None = None, blocked: bool = False
    ) -> BatchPlacements:
        names = [name for run in runs for name in run]
        parsed = blind(names)
        series = series_index(cls._ID_BY_KEY, titles={**cls._TITLES, **(retitled or {})})
        return place(_pair_in_order_blocked(parsed) if blocked else parsed, PlacementTarget(cls._WINDOW, series), names)

    def test_the_sequence_whose_members_name_the_window_is_picked(self) -> None:
        result = self._place(self._INSIDE, self._OUTSIDE)

        assert result.placed == {name: (501 + i,) for i, name in enumerate(self._INSIDE)}

    def test_the_other_sequences_titled_members_are_outside_the_window_once_it_is_full(self) -> None:
        result = self._place(self._INSIDE, self._OUTSIDE)

        assert [by_name(result)[name] for name in self._OUTSIDE] == [
            ((), PlacementVerdict.OUTSIDE_WINDOW),
            ((), PlacementVerdict.OUTSIDE_WINDOW),
            ((), PlacementVerdict.NO_EPISODE_FOUND),
        ]

    def test_one_titled_member_places_only_itself(self) -> None:
        # One title is too little to pick a sequence among two, and enough to place its own file.
        inside = numbered_names("show a", 3, ("Beach Day", "", ""))

        result = self._place(inside, numbered_names("show b", 3))

        assert result.placed == {inside[0]: (501,)}
        assert by_name(result)[inside[0]] == ((501,), PlacementVerdict.EPISODE_TITLE)

    @pytest.mark.parametrize("blocked", [True, False], ids=["mixed batch", "pristine batch"])
    def test_members_naming_only_other_episodes_block_the_sequence(self, blocked: bool) -> None:
        # The one fitting sequence is titled as another season's episodes: nothing places, and neither the numbered
        # sequence pass nor pairing in name order on a batch with nothing judged takes the window behind the block.
        result = self._place(numbered_names("show", 3, ("Pilot Flight", "The Return", "Bonus")), blocked=blocked)

        assert result.placed == {}

    def test_a_short_subtitle_is_no_episode_title_match(self) -> None:
        # The word past the colon leads two members' words after the number, and says too little to count against
        # the sequence.
        run = numbered_names("show", 3, ("Home Alone", "Home Again", ""))

        result = self._place(run, retitled={601: "Chapter 4: Home"})

        assert result.placed == {name: (501 + i,) for i, name in enumerate(run)}

    def test_one_word_titles_still_block_the_sequence(self) -> None:
        # A title too short to place a file is still evidence against a sequence titled as another season's.
        run = numbered_names("show", 3, ("Pilot", "Return", "Bonus"))

        result = self._place(run, retitled={601: "Pilot", 602: "Return"}, blocked=True)

        assert result.placed == {}

    def test_a_title_shared_by_both_sides_counts_for_neither(self) -> None:
        # A recap season repeats the titles: a member naming an episode on each side is no evidence.
        run = numbered_names("show", 3, ("Beach Day", "Hot Springs", "Bonus"))

        result = self._place(run, retitled={601: "Beach Day", 602: "Hot Springs"}, blocked=True)

        assert result.placed == {name: (501 + i,) for i, name in enumerate(run)}


class TestRereadSeasonSequence:
    """A `1..N` sequence Sonarr matched into the one N-episode season and its specials is the season's own numbering."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        **{EpisodeKey(1, n): 600 + n for n in (1, 2, 3, 4)},
    }
    _SHIFTED: ClassVar[tuple[tuple[int, int], ...]] = ((1, 1), (1, 2), (0, 1), (1, 3), (1, 4))
    """Sonarr's parse after the special TVDB interleaved after the second episode shifted it."""
    _TWO_OF_FOUR: ClassVar[dict[EpisodeKey, int]] = {
        **{EpisodeKey(0, n): 500 + n for n in (1, 2, 3, 4)},
        **{EpisodeKey(1, n): 600 + n for n in (1, 2, 3, 4)},
    }
    """Four specials beside a four-episode season, where a sequence can fit a season and the specials equally."""
    _TIED: ClassVar[tuple[tuple[int, int], ...]] = ((1, 1), (0, 1), (1, 2), (0, 2))
    """Sonarr's parse of a four-file sequence, as many onto the specials as into the season."""

    @staticmethod
    def _reads(run: list[str], pairs: tuple[tuple[int, int], ...]) -> dict[str, ParsedFileInfo | None]:
        return {name: parsed_info(matched=(pair,)) for name, pair in zip(run, pairs[: len(run)], strict=True)}

    @classmethod
    def _place(cls, run: list[str], window: list[int], series: dict[EpisodeKey, int] | None = None) -> BatchPlacements:
        parsed = cls._reads(run, cls._SHIFTED)
        return place(parsed, PlacementTarget(window, series_index(series or cls._ID_BY_KEY)))

    def test_the_specials_window_gets_none_of_the_seasons_sequence(self) -> None:
        # Four files for a four-episode season: the third is the third episode, not the special Sonarr read.
        result = self._place(numbered_names("show", 4), [501])

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_the_seasons_window_reads_the_sequence_as_numbered(self) -> None:
        run = numbered_names("show", 4)

        result = self._place(run, [601, 602, 603, 604])

        assert result.placed == {name: (601 + i,) for i, name in enumerate(run)}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.EXACT}

    def test_a_lower_version_of_a_member_is_re_read_with_it(self) -> None:
        # Sonarr shifted the `- 03` and its `- 03v2` onto the special alike: the lower version is re-read
        # as the third episode with its member, so the specials window places neither.
        run = numbered_names("show", 4)
        later = "show - 03v2 [grp].mkv"
        parsed = self._reads(run, self._SHIFTED)
        parsed[later] = parsed_info(matched=((0, 1),))

        result = place(parsed, PlacementTarget([501], series_index(self._ID_BY_KEY)))

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_a_possible_episode_of_a_sequence_fitting_both_keeps_another_sequence_out_of_the_window(self) -> None:
        # The sequence fits the season and the specials equally, so the first file may be the season's first episode,
        # and two blind extras that fit the window's first two episodes do not take them.
        run = numbered_names("show", 4)
        extras = numbered_names("show extra", 2)
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(extras))

        result = place(parsed, PlacementTarget([601, 602], series_index(self._TWO_OF_FOUR)))

        assert by_name(result) == dict.fromkeys((*run, *extras), ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_title_never_places_a_file_fitting_both_alone(self) -> None:
        # Over the one special's window the AniList name names the sequence's second file: fitting both equally, it
        # is the second episode or the second special, never the first special.
        series = {key: ep_id for key, ep_id in self._TWO_OF_FOUR.items() if key.episode <= 2}
        run = numbered_names("show", 2)
        parsed = self._reads(run, self._TIED)
        target = PlacementTarget([501], series_index(series), title_names=TitleNames("Show", ("Show 2",)))

        result = place(parsed, target)

        assert by_name(result) == dict.fromkeys(run, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_title_naming_files_that_form_no_sequence_blocks_the_sequence_fitting_both(self) -> None:
        # The sequel's window carries its AniList name, which names three loose sequel files and not the sequence
        # fitting both: the sequence is blocked there rather than indexed by its name numbers.
        run = numbered_names("show", 4)
        loose = [f"show two - {n} [grp].mkv" for n in ("05", "06", "08")]
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(loose))
        series = {**self._TWO_OF_FOUR, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}
        target = PlacementTarget(
            [701, 702, 703, 704], series_index(series), title_names=TitleNames("Show", ("Show Two",))
        )

        result = place(parsed, target)

        assert by_name(result) == dict.fromkeys((*run, *loose), ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_sequence_fitting_both_stands_aside_from_a_window_neither_numbering_reaches(self) -> None:
        # The sequence fits the first season and the specials equally, so over the second season's window it is
        # outside the window and the blind sequel sequence indexes it.
        run = numbered_names("show", 4)
        sequel = numbered_names("show 2", 4)
        parsed = self._reads(run, self._TIED)
        parsed.update(blind(sequel))
        series = {**self._TWO_OF_FOUR, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}

        result = place(parsed, PlacementTarget([701, 702, 703, 704], series_index(series)))

        assert by_name(result) == {
            **dict.fromkeys(run, ((), PlacementVerdict.NO_EPISODE_FOUND)),
            **{name: ((701 + i,), PlacementVerdict.SEQUENCE_OVER_PARSE) for i, name in enumerate(sequel)},
        }

    def test_a_sequence_one_wider_than_the_season_carried_the_special(self) -> None:
        # Five files for four episodes and one special: Sonarr's parse stands.
        run = numbered_names("show", 5)

        result = self._place(run, [501])

        assert result.placed == {run[2]: (501,)}

    def test_another_season_of_the_same_width_does_not_block_the_reread(self) -> None:
        # Sonarr's reads name the season the sequence belongs to, so a second four-episode season is no rival.
        two_seasons = {**self._ID_BY_KEY, **{EpisodeKey(2, n): 700 + n for n in (1, 2, 3, 4)}}

        result = self._place(numbered_names("show", 4), [501], series=two_seasons)

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.OUTSIDE_WINDOW}

    @pytest.mark.parametrize(
        ("specials", "verdict"),
        [
            pytest.param(2, PlacementVerdict.OUTSIDE_WINDOW, id="fewer specials"),
            pytest.param(4, PlacementVerdict.NO_EPISODE_FOUND, id="as many specials"),
        ],
    )
    def test_as_many_onto_the_specials_as_into_the_season_is_read_by_the_specials_count(
        self, specials: int, verdict: PlacementVerdict
    ) -> None:
        # Two of four members onto two specials: the season's, unless a specials release could be as
        # wide as the sequence, which then fits both equally and nothing places over the specials window.
        series = {key: ep_id for key, ep_id in self._TWO_OF_FOUR.items() if key.season or key.episode <= specials}
        run = numbered_names("show", 4)
        parsed = self._reads(run, self._TIED)

        result = place(parsed, PlacementTarget([501, 502], series_index(series)))

        assert result.placed == {}
        assert {p.verdict for p in result.placements} == {verdict}


# Twenty-five specials and one seasoned episode, the series the specials torrent tests place into.
_SPECIALS_ID_BY_KEY: dict[EpisodeKey, int] = {
    **{EpisodeKey(0, n): 500 + n for n in range(1, 26)},
    EpisodeKey(1, 1): 601,
}
_SPECIALS_INDEX = series_index(_SPECIALS_ID_BY_KEY)


def _listing(*ids: int) -> TorrentListing:
    """A torrent one entry lists, over the specials `ids`."""

    return torrent_listing(title_window(1, ids))


# The eight specials the entries list (7, 12, 13, 15, 17, 20, 22 and 23), and a torrent as wide numbered another way.
_EIGHT_WINDOW = title_window(1, (507, 512, 513, 515, 517, 520, 522, 523))
_EIGHT_LISTING = torrent_listing(_EIGHT_WINDOW)
_TORRENT_NUMBERS = (7, 13, 14, 15, 16, 17, 18, 19)
# A listing of three specials, room for a three-wide torrent numbered 1, 2 and 3 to miss it.
_THREE_LISTING = _listing(502, 504, 506)


def _specials_placement_target(
    resolved: Sequence[int], listing: TorrentListing, size_matches: SizeMatches = NO_SIZE_MATCHES
) -> PlacementTarget:
    """A window of `resolved` over the specials series, carrying `listing` and the `size_matches`."""

    evidence = SeadexEvidence(listing, size_matches)
    return PlacementTarget(list(resolved), _SPECIALS_INDEX, title_names=TitleNames("Show"), seadex=evidence)


class TestMisnumberedSpecials:
    """A specials torrent numbered by another TVDB state than the windows listing it is set aside whole, not reordered.

    The torrent is one `S00Exx` sequence, each file read as its own number's special, as wide as the listing, and some
    number is no listed special's. The listing has no special aliases here. Every shape the pass cannot judge
    places by number as before.
    """

    _TITLED: ClassVar[EpisodeIndex] = series_index(
        _SPECIALS_ID_BY_KEY, titles={501: "Pilot Flight", 502: "Distress Call", 503: "Final Bell"}
    )

    def test_a_torrent_the_listing_contradicts_is_set_aside_whole(self) -> None:
        # The torrent's 14, 16, 18 and 19 are no listed special, so its numbering is another TVDB state's. Its 7, 13 and
        # 15 are not placed either.
        parsed = parsed_specials(*_TORRENT_NUMBERS)

        result = place(parsed, _specials_placement_target([507, 512, 513], _EIGHT_LISTING))

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}
        assert result.placed == {}
        assert sorted(result.unplaced) == sorted(parsed)

    def test_one_stray_number_leaves_the_torrent_misnumbered_and_never_reorders_it(self) -> None:
        # Seven of eight numbers are listed: placing them and the eighth by its position would be a guess that the
        # listing's order is the torrent's. The whole torrent is a human's.
        result = place(
            parsed_specials(*range(1, 9)),
            _specials_placement_target([501, 502, 503], _listing(*range(501, 508), 520)),
        )

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_an_extra_beside_the_torrent_is_set_aside_on_its_own(self) -> None:
        parsed = {**parsed_specials(1, 2, 3), "Show.S00.Trailer.mkv": parsed_info()}

        placed = by_name(place(parsed, _specials_placement_target([502, 504], _THREE_LISTING)))

        assert placed["Show.S00.Trailer.mkv"] == ((), PlacementVerdict.EXTRA)
        assert {placed[name][1] for name in parsed_specials(1, 2, 3)} == {PlacementVerdict.MISNUMBERED}

    def test_a_torrent_the_listing_agrees_with_places_by_number(self) -> None:
        placed = by_name(
            place(parsed_specials(1, 2, 3), _specials_placement_target([501, 502, 503], _listing(501, 502, 503)))
        )

        assert placed == {
            "Show.S00E01.mkv": ((501,), PlacementVerdict.EXACT),
            "Show.S00E02.mkv": ((502,), PlacementVerdict.EXACT),
            "Show.S00E03.mkv": ((503,), PlacementVerdict.EXACT),
        }

    def test_a_listing_with_a_seasoned_episode_is_no_specials_listing(self) -> None:
        # A season's numbering is stable: a number outside it is a mislisting, never a shifted numbering.
        placed = by_name(
            place(parsed_specials(1, 2, 3), _specials_placement_target([501, 502], _listing(501, 502, 601)))
        )

        assert placed == {
            "Show.S00E01.mkv": ((501,), PlacementVerdict.EXACT),
            "Show.S00E02.mkv": ((502,), PlacementVerdict.EXACT),
            "Show.S00E03.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW),
        }

    def test_a_listing_of_another_width_stands_down(self) -> None:
        placed = by_name(
            place(parsed_specials(1, 2, 3), _specials_placement_target([501, 502], _listing(501, 502, 504, 505)))
        )

        assert placed["Show.S00E01.mkv"] == ((501,), PlacementVerdict.EXACT)
        assert placed["Show.S00E03.mkv"] == ((), PlacementVerdict.OUTSIDE_WINDOW)

    def test_a_window_outside_the_listing_stands_down(self) -> None:
        result = place(parsed_specials(1, 2, 3), _specials_placement_target([501, 509], _THREE_LISTING))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_a_target_with_no_window_stands_down(self) -> None:
        result = place(parsed_specials(1, 2, 3), _specials_placement_target([], _THREE_LISTING))

        assert result.placed == {"Show.S00E01.mkv": (501,), "Show.S00E02.mkv": (502,), "Show.S00E03.mkv": (503,)}

    def test_a_window_with_a_file_map_episode_stands_down(self) -> None:
        # The file map already places a file, so the window has a used id and the torrent isn't set aside.
        result = place(
            parsed_specials(1, 2, 3),
            _specials_placement_target([504], _THREE_LISTING),
            file_map={"Show.S00E02.mkv": (504,)},
        )

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_a_file_map_episode_in_another_window_leaves_the_torrent_misnumbered(self) -> None:
        # The file map puts a file on a special outside this window, so nothing in this window is used yet and
        # the torrent is still set aside.
        result = place(
            parsed_specials(1, 2, 3),
            _specials_placement_target([504], _THREE_LISTING),
            file_map={"Show.S00E02.mkv": (502,)},
        )

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_title_contradicting_the_numbering_keeps_the_listing_out(self) -> None:
        # The first file is titled as special 2, which the second reads by number.
        parsed = {
            **parsed_specials(2, 3),
            "Show.S00E01 - Distress Call.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }

        placed = by_name(
            place(parsed, replace(_specials_placement_target([502, 504], _THREE_LISTING), index=self._TITLED))
        )

        assert placed["Show.S00E01 - Distress Call.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert PlacementVerdict.MISNUMBERED not in {verdict for _ids, verdict in placed.values()}

    def test_a_duplicate_number_stands_down(self) -> None:
        parsed = {
            **parsed_specials(1, 3),
            "Show.S00E01.BluRay.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }

        result = place(parsed, _specials_placement_target([501, 503], _listing(501, 503, 505)))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)
        assert result.placed["Show.S00E03.mkv"] == (503,)

    def test_a_multi_episode_special_stands_down(self) -> None:
        parsed = {
            **parsed_specials(1),
            "Show.S00E02-E03.mkv": parsed_info(season=0, episodes=(2, 3), matched=((0, 2), (0, 3))),
        }

        result = place(parsed, _specials_placement_target([502, 504], _listing(502, 504)))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_an_unread_parse_stands_down(self) -> None:
        parsed = {**parsed_specials(1, 3), "Show.S00E02.mkv": None}

        result = place(parsed, _specials_placement_target([502, 504], _THREE_LISTING))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_a_second_prefix_stands_down(self) -> None:
        parsed = {**parsed_specials(1, 3), "Other.S00E02.mkv": parsed_info(season=0, episodes=(2,), matched=((0, 2),))}

        result = place(parsed, _specials_placement_target([502, 504], _THREE_LISTING))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_a_file_with_no_number_beside_the_torrent_stands_down(self) -> None:
        # The torrent is not one sequence: a file with no name number is beside it, and the listing judges a torrent
        # only when it is one sequence.
        parsed = {**parsed_specials(1, 2, 3), "Show - Bonus.mkv": parsed_info()}

        result = place(parsed, _specials_placement_target([502, 504], _THREE_LISTING))

        assert PlacementVerdict.MISNUMBERED not in verdicts_of(result)

    def test_one_title_confirming_its_number_keeps_the_torrent_misnumbered(self) -> None:
        # The first file is titled as special 1, the number it carries. That confirms one file: a special
        # inserted after it would still shift the rest, so the listing keeps judging the torrent.
        parsed = {
            **parsed_specials(2, 3),
            "Show.S00E01 - Pilot Flight.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }

        result = place(parsed, replace(_specials_placement_target([502, 504], _THREE_LISTING), index=self._TITLED))

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_title_on_every_member_confirming_its_number_stands_down(self) -> None:
        # Every file is titled as the special it is numbered: the torrent is numbered by this TVDB state.
        titles = {1: "Pilot Flight", 2: "Distress Call", 3: "Final Bell"}
        parsed = {
            f"Show.S00E{n:02d} - {title}.mkv": parsed_info(season=0, episodes=(n,), matched=((0, n),))
            for n, title in titles.items()
        }

        placed = by_name(
            place(parsed, replace(_specials_placement_target([502, 504], _THREE_LISTING), index=self._TITLED))
        )

        assert placed["Show.S00E02 - Distress Call.mkv"][0] == (502,)
        assert PlacementVerdict.MISNUMBERED not in {verdict for _ids, verdict in placed.values()}

    def test_a_superseded_version_is_misnumbered_with_its_torrent(self) -> None:
        parsed = {
            **parsed_specials(1, 2, 3),
            "Show.S00E01v2.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }

        result = place(parsed, _specials_placement_target([502, 504], _THREE_LISTING))

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}
        assert sorted(result.unplaced) == sorted(parsed)

    def test_a_number_the_series_lacks_leaves_the_torrent_misnumbered(self) -> None:
        result = place(parsed_specials(1, 2, 26), _specials_placement_target([501, 502], _listing(501, 502, 503)))

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_no_listing_places_by_number(self) -> None:
        placed = by_name(place(parsed_specials(1, 2, 3), _specials_placement_target([501, 502], EMPTY_LISTING)))

        assert placed == {
            "Show.S00E01.mkv": ((501,), PlacementVerdict.EXACT),
            "Show.S00E02.mkv": ((502,), PlacementVerdict.EXACT),
            "Show.S00E03.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW),
        }

    def test_a_member_its_size_matches_stays_in_the_misnumbered_torrent(self) -> None:
        # The misnumbered specials pass runs first, so a size match never pulls one special out of a torrent set
        # aside whole.
        parsed = parsed_specials(*_TORRENT_NUMBERS)
        target = _specials_placement_target([507, 512, 513], _EIGHT_LISTING, {"Show.S00E14.mkv": 512})

        assert verdicts_of(place(parsed, target)) == {PlacementVerdict.MISNUMBERED}


class TestAlternateNumbering:
    """A misnumbered specials torrent is placed through the listing's special aliases when they cover it one to one.

    Each member goes on the special its alias points at, or is `ALIAS_OUTSIDE_WINDOW` when that special is outside the
    window. Anything short of one to one onto the listing, or a title or size match against an alias, sets
    the torrent aside whole as it would without aliases. A file in the file map off its alias keeps the aliases out of
    a window the file map uses.
    """

    # Points the torrent's numbers at the eight listed specials.
    _ALIASES: ClassVar[dict[int, int]] = {7: 7, 13: 12, 14: 13, 15: 15, 16: 17, 17: 20, 18: 22, 19: 23}
    _LISTING: ClassVar[TorrentListing] = torrent_listing(_EIGHT_WINDOW, aliases=_ALIASES)

    def test_each_member_goes_on_its_aliased_special_or_stays_aliased_outside_the_window(self) -> None:
        placed = by_name(
            place(parsed_specials(*_TORRENT_NUMBERS), _specials_placement_target([507, 512, 513], self._LISTING))
        )

        assert placed == {
            "Show.S00E07.mkv": ((507,), PlacementVerdict.SPECIAL_ALIAS),
            "Show.S00E13.mkv": ((512,), PlacementVerdict.SPECIAL_ALIAS),
            "Show.S00E14.mkv": ((513,), PlacementVerdict.SPECIAL_ALIAS),
            **{f"Show.S00E{n}.mkv": ((), PlacementVerdict.ALIAS_OUTSIDE_WINDOW) for n in (15, 16, 17, 18, 19)},
        }

    def test_a_torrent_the_listing_agrees_with_keeps_its_numbers(self) -> None:
        # Every number is already a listed special, so the aliases aren't used.
        placed = by_name(
            place(
                parsed_specials(7, 12, 13, 15, 17, 20, 22, 23),
                _specials_placement_target([507, 512, 513], self._LISTING),
            )
        )

        assert {name: placed[name] for name in ("Show.S00E12.mkv", "Show.S00E13.mkv")} == {
            "Show.S00E12.mkv": ((512,), PlacementVerdict.EXACT),
            "Show.S00E13.mkv": ((513,), PlacementVerdict.EXACT),
        }

    @pytest.mark.parametrize(
        "aliases",
        [
            pytest.param({7: 7, 13: 12, 14: 13, 15: 15, 16: 17, 17: 20, 18: 22}, id="a number without an alias"),
            pytest.param({**_ALIASES, 19: 30}, id="an alias onto a special the series lacks"),
            pytest.param({**_ALIASES, 19: 24}, id="an alias outside the listing"),
            pytest.param({**_ALIASES, 19: 22}, id="two numbers onto one special"),
        ],
    )
    def test_aliases_not_one_to_one_onto_the_listing_leave_the_torrent_misnumbered(
        self, aliases: dict[int, int]
    ) -> None:
        result = place(
            parsed_specials(*_TORRENT_NUMBERS),
            _specials_placement_target([507, 512, 513], torrent_listing(_EIGHT_WINDOW, aliases=aliases)),
        )

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_title_naming_another_special_than_its_alias_leaves_the_torrent_misnumbered(self) -> None:
        # File 13 is titled as special 13 but its alias says special 12, so the torrent is left for a hand import.
        parsed = {
            **parsed_specials(*(n for n in _TORRENT_NUMBERS if n != 13)),
            "Show.S00E13 - Final Bell.mkv": parsed_info(season=0, episodes=(13,), matched=((0, 13),)),
        }
        titled = replace(
            _specials_placement_target([507, 512, 513], self._LISTING),
            index=series_index(_SPECIALS_ID_BY_KEY, titles={513: "Final Bell"}),
        )

        result = place(parsed, titled)

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_superseded_version_is_a_duplicate(self) -> None:
        parsed = {
            **parsed_specials(*_TORRENT_NUMBERS),
            "Show.S00E07v2.mkv": parsed_info(season=0, episodes=(7,), matched=((0, 7),)),
        }

        placed = by_name(place(parsed, _specials_placement_target([507, 512, 513], self._LISTING)))

        assert placed["Show.S00E07v2.mkv"] == ((507,), PlacementVerdict.SPECIAL_ALIAS)
        assert placed["Show.S00E07.mkv"] == ((), PlacementVerdict.DUPLICATE)

    def test_a_member_whose_special_is_already_used_is_unplaced(self) -> None:
        # An earlier window already used special 7. The aliases still cover the whole torrent, and only that member
        # is unplaced.
        target = replace(_specials_placement_target([507, 512, 513], self._LISTING), used=frozenset({507}))

        placed = by_name(place(parsed_specials(*_TORRENT_NUMBERS), target))

        assert placed["Show.S00E07.mkv"] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert placed["Show.S00E13.mkv"] == ((512,), PlacementVerdict.SPECIAL_ALIAS)
        assert placed["Show.S00E14.mkv"] == ((513,), PlacementVerdict.SPECIAL_ALIAS)
        assert {placed[f"Show.S00E{n}.mkv"][1] for n in (15, 16, 17, 18, 19)} == {PlacementVerdict.ALIAS_OUTSIDE_WINDOW}

    def test_a_size_match_on_another_special_than_its_alias_leaves_the_torrent_misnumbered(self) -> None:
        # File 13's size matches special 13 but its alias says special 12, so the torrent is left for a hand import.
        target = _specials_placement_target([507, 512, 513], self._LISTING, {"Show.S00E13.mkv": 513})

        result = place(parsed_specials(*_TORRENT_NUMBERS), target)

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_superseded_versions_size_match_against_its_alias_leaves_the_torrent_misnumbered(self) -> None:
        # The v1 of file 13 is the size of special 13, where its alias says special 12. The v2 supersedes it in
        # the torrent, but the v1's size still speaks for the number, so the torrent is left for a hand import.
        parsed = {
            **parsed_specials(*_TORRENT_NUMBERS),
            "Show.S00E13v2.mkv": parsed_info(season=0, episodes=(13,), matched=((0, 13),)),
        }
        target = _specials_placement_target([507, 512, 513], self._LISTING, {"Show.S00E13.mkv": 513})

        result = place(parsed, target)

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_a_superseded_version_in_the_file_map_on_its_own_number_keeps_the_aliases_out(self) -> None:
        # The file map puts the v1 of file 13 on special 13, its own number. The v2 is left to place, and
        # the file map's disagreement with the aliases keeps the rest of the torrent off them too.
        parsed = {
            **parsed_specials(*_TORRENT_NUMBERS),
            "Show.S00E13v2.mkv": parsed_info(season=0, episodes=(13,), matched=((0, 13),)),
        }
        file_map = {"Show.S00E13.mkv": (513,)}

        result = place(parsed, _specials_placement_target([507, 512, 513], self._LISTING), file_map=file_map)

        assert not verdicts_of(result) & {PlacementVerdict.SPECIAL_ALIAS, PlacementVerdict.MISNUMBERED}

    def test_the_file_map_on_its_own_number_contradicts_the_aliases_and_the_torrent_keeps_its_numbers(self) -> None:
        # The file map puts file 13 on special 13, its own number, where its alias says special 12. The aliases
        # are wrong for this torrent, so nothing is placed by them, and the window the file map uses isn't set aside
        # either.
        parsed = parsed_specials(*_TORRENT_NUMBERS)
        file_map = {"Show.S00E13.mkv": (513,)}

        result = place(parsed, _specials_placement_target([507, 512, 513], self._LISTING), file_map=file_map)

        assert not verdicts_of(result) & {PlacementVerdict.SPECIAL_ALIAS, PlacementVerdict.MISNUMBERED}

    def test_a_file_in_the_file_map_spanning_its_alias_and_another_special_keeps_the_aliases_out(self) -> None:
        # The file map puts file 13 on two specials, its alias among them. A member on more than its one
        # aliased special is off it, so the torrent keeps its numbers.
        parsed = parsed_specials(*_TORRENT_NUMBERS)
        file_map = {"Show.S00E13.mkv": (512, 513)}

        result = place(parsed, _specials_placement_target([507, 512, 513], self._LISTING), file_map=file_map)

        assert not verdicts_of(result) & {PlacementVerdict.SPECIAL_ALIAS, PlacementVerdict.MISNUMBERED}

    def test_a_file_in_the_file_map_on_its_aliased_special_agrees_and_the_rest_places_by_the_aliases(self) -> None:
        # The file map puts file 13 on special 12, its alias: the rest of the torrent follows the aliases.
        parsed = parsed_specials(*_TORRENT_NUMBERS)
        file_map = {"Show.S00E13.mkv": (512,)}

        placed = by_name(place(parsed, _specials_placement_target([507, 512, 513], self._LISTING), file_map=file_map))

        assert placed["Show.S00E07.mkv"] == ((507,), PlacementVerdict.SPECIAL_ALIAS)
        assert placed["Show.S00E14.mkv"] == ((513,), PlacementVerdict.SPECIAL_ALIAS)
        assert {placed[f"Show.S00E{n}.mkv"][1] for n in (15, 16, 17, 18, 19)} == {PlacementVerdict.ALIAS_OUTSIDE_WINDOW}


class TestSizeMatch:
    """A file the same size as a single-file SeaDex release for one episode goes on that episode, over its name."""

    _SPECIALS: ClassVar[dict[str, ParsedFileInfo | None]] = {
        "Show.S00E01.mkv": parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        "Show.S00E02.mkv": parsed_info(season=0, episodes=(2,), matched=((0, 2),)),
    }
    _SWAPPED: ClassVar[dict[str, int]] = {"Show.S00E01.mkv": 502, "Show.S00E02.mkv": 501}
    _PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        f"Show.S01E{n:02d}.mkv": parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n in range(1, 5)
    }
    _SEASON: ClassVar[list[int]] = [601, 602, 603, 604]

    @staticmethod
    def _target(
        resolved: Sequence[int], size_matches: Mapping[str, int], *, used: Sequence[int] = ()
    ) -> PlacementTarget:
        listing = SeadexEvidence(EMPTY_LISTING, size_matches)
        return PlacementTarget(
            list(resolved), _SERIES, used=frozenset(used), title_names=TitleNames("Show"), seadex=listing
        )

    def test_a_file_with_no_number_is_placed_by_its_size(self) -> None:
        name = "Show - Special [grp].mkv"

        placed = by_name(place({name: parsed_info()}, self._target([501, 502], {name: 502})))

        assert placed == {name: ((502,), PlacementVerdict.SIZE_MATCH)}

    def test_the_size_outranks_the_name(self) -> None:
        # Each special is named as the other. Both are placed by size, and the caller warns for each.
        placed = by_name(place(self._SPECIALS, self._target([501, 502], self._SWAPPED)))

        assert placed == {
            "Show.S00E01.mkv": ((502,), PlacementVerdict.SIZE_OVER_NAME),
            "Show.S00E02.mkv": ((501,), PlacementVerdict.SIZE_OVER_NAME),
        }

    def test_a_season_episode_name_is_overridden_though_its_name_number_matches_the_special(self) -> None:
        # Its size says S00E02 and its season and episode say S01E02. The name number 2 is just its episode again, so
        # it doesn't count as agreeing.
        name = "Show.S01E02.mkv"

        placed = by_name(place({name: self._PARSED[name]}, self._target([501, 502], {name: 502})))

        assert placed == {name: ((502,), PlacementVerdict.SIZE_OVER_NAME)}

    def test_a_size_matching_the_other_window_is_outside_the_window(self) -> None:
        # One special per window. The file named as this window's special belongs to the other window by size, so
        # it isn't placed here.
        placed = by_name(place(self._SPECIALS, self._target([502], self._SWAPPED)))

        assert placed == {
            "Show.S00E01.mkv": ((502,), PlacementVerdict.SIZE_OVER_NAME),
            "Show.S00E02.mkv": ((), PlacementVerdict.OUTSIDE_WINDOW),
        }

    def test_a_size_matching_another_series_leaves_the_file_to_its_name(self) -> None:
        name = "Show.S01E01.mkv"

        placed = by_name(place({name: self._PARSED[name]}, self._target([601], {name: 999})))

        assert placed == {name: ((601,), PlacementVerdict.EXACT)}

    def test_an_agreeing_size_leaves_a_placing_torrent_as_it_was(self) -> None:
        placed = by_name(place(self._PARSED, self._target(self._SEASON, {"Show.S01E02.mkv": 602})))

        assert placed == {f"Show.S01E{n:02d}.mkv": ((600 + n,), PlacementVerdict.EXACT) for n in range(1, 5)}

    def test_a_size_one_batch_file_disagrees_with_moves_that_member_alone(self) -> None:
        # The second file's size puts it on episode 3, which the third file also reads as. Being placed by size
        # doesn't make the third file a duplicate, so it's unplaced and reported, and episode 2 stays unused.
        placed = by_name(place(self._PARSED, self._target(self._SEASON, {"Show.S01E02.mkv": 603})))

        assert placed == {
            "Show.S01E01.mkv": ((601,), PlacementVerdict.EXACT),
            "Show.S01E02.mkv": ((603,), PlacementVerdict.SIZE_OVER_NAME),
            "Show.S01E03.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND),
            "Show.S01E04.mkv": ((604,), PlacementVerdict.EXACT),
        }

    def test_a_size_disagreeing_with_a_sequences_number_bars_the_sequence(self) -> None:
        # The last member's size puts it on episode 1, so the sequence's numbering is off. Nothing places the rest by
        # pairing or by sequence, so each one is left unplaced for a hand check.
        run = numbered_names("show", 4)
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(absolutes=(n,)) for n, name in enumerate(run, 1)}

        placed = by_name(place(parsed, self._target(self._SEASON, {run[3]: 601})))

        assert placed == {
            run[0]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[1]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[2]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[3]: ((601,), PlacementVerdict.SIZE_OVER_NAME),
        }

    def test_a_size_disagreeing_with_a_matched_number_bars_the_name_number_only(self) -> None:
        # Sonarr matched each member, so the others keep their matches. The first member is unplaced, since the
        # last one used its episode by size.
        run = numbered_names("show", 4)
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(absolutes=(n,), matched=((1, n),)) for n, name in enumerate(run, 1)
        }

        placed = by_name(place(parsed, self._target(self._SEASON, {run[3]: 601})))

        assert placed == {
            run[0]: ((), PlacementVerdict.NO_EPISODE_FOUND),
            run[1]: ((602,), PlacementVerdict.EXACT),
            run[2]: ((603,), PlacementVerdict.EXACT),
            run[3]: ((601,), PlacementVerdict.SIZE_OVER_NAME),
        }

    def test_a_size_agreeing_with_an_unresolved_number_leaves_the_sequence_to_place(self) -> None:
        # Sonarr resolved nothing, and the last member's size matches the episode its number points at. Nothing
        # disagrees, so the sequence places by its numbers and the last member by size.
        run = numbered_names("show", 4)
        parsed: dict[str, ParsedFileInfo | None] = {name: parsed_info(absolutes=(n,)) for n, name in enumerate(run, 1)}

        placed = by_name(place(parsed, self._target(self._SEASON, {run[3]: 604})))

        assert placed == {
            run[0]: ((601,), PlacementVerdict.SEQUENCE_OVER_PARSE),
            run[1]: ((602,), PlacementVerdict.SEQUENCE_OVER_PARSE),
            run[2]: ((603,), PlacementVerdict.SEQUENCE_OVER_PARSE),
            run[3]: ((604,), PlacementVerdict.SIZE_MATCH),
        }

    def test_with_no_window_a_size_matching_another_series_leaves_the_file_to_its_name(self) -> None:
        # A target with no window admits any id: only the series check keeps the file off another series' episode.
        name = "Show.S01E01.mkv"
        target = PlacementTarget(
            [], _SERIES, title_names=TitleNames("Show"), seadex=SeadexEvidence(EMPTY_LISTING, {name: 999})
        )

        assert by_name(place({name: self._PARSED[name]}, target)) == {name: ((601,), PlacementVerdict.EXACT)}

    def test_a_size_agreeing_with_a_rejected_parse_places_it(self) -> None:
        # The titles contradict the sequence's numbering, so the last member's parse is rejected. Its size points at
        # the same episode, though, and the rejection was only against the numbers, so it's placed.
        run = numbered_names("show", 4, ("The Return", "Festival Night", "", ""))
        parsed: dict[str, ParsedFileInfo | None] = {
            name: parsed_info(absolutes=(n,), matched=((1, n),)) for n, name in enumerate(run, start=1)
        }

        placed = by_name(place(parsed, self._target(self._SEASON, {run[3]: 604})))

        assert placed[run[3]] == ((604,), PlacementVerdict.SIZE_MATCH)

    def test_a_title_naming_another_episode_keeps_the_file_for_the_later_passes(self) -> None:
        name = "show - 01 - Festival Night [grp].mkv"

        placed = by_name(
            place({name: parsed_info(absolutes=(1,), matched=((1, 1),))}, self._target(_FIRST_THREE, {name: 601}))
        )

        assert placed == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}

    def test_a_size_matching_a_used_episode_sets_the_file_aside(self) -> None:
        # The file with no parsed number would otherwise take the one unused id, but its size says it's the used
        # episode.
        name = "Show - Special [grp].mkv"

        placed = by_name(place({name: parsed_info()}, self._target([501, 502], {name: 502}, used=[502])))

        assert placed == {name: ((), PlacementVerdict.NO_EPISODE_FOUND)}


class TestSeasonEpisodeRange:
    """A `SxxEyy-zz` range Sonarr read as its first episode is the whole range to every pass."""

    _NAME = "show s01e02-03 [grp].mkv"
    # Sonarr reads the range's end as an absolute.
    _SHORT: ClassVar[dict[str, ParsedFileInfo | None]] = {
        _NAME: parsed_info(season=1, episodes=(2,), absolutes=(3,), matched=((1, 2),))
    }

    def test_a_range_read_short_places_on_the_whole_range(self) -> None:
        placed = by_name(place(self._SHORT, _target([602, 603])))

        assert placed == {self._NAME: ((602, 603), PlacementVerdict.EXACT)}

    def test_a_range_partly_used_never_pairs_in_order_onto_its_unused_episode(self) -> None:
        # Pairing in absolute order sees the widened parse, which no longer reads the range's end as an absolute.
        placed = by_name(place(self._SHORT, _target([602, 603], used=[603])))

        assert placed == {self._NAME: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_range_reaching_past_the_window_is_unplaced(self) -> None:
        # The window holds only the range's first episode: the file is never placed on that episode alone.
        placed = by_name(place(self._SHORT, _target([602])))

        assert placed == {self._NAME: ((), PlacementVerdict.NO_EPISODE_FOUND)}

    def test_a_range_sonarr_matched_places_whole_despite_an_extras_word(self) -> None:
        # Sonarr matched the range's first episode, so "preview" in the name doesn't make the file an extra.
        name = "show s01e02-03 preview [grp].mkv"
        parsed = {name: parsed_info(season=1, episodes=(2,), absolutes=(3,), matched=((1, 2),))}

        placed = by_name(place(parsed, _target([602, 603])))

        assert placed == {name: ((602, 603), PlacementVerdict.EXACT)}

    def test_a_widened_range_keeps_sonarrs_season_shift(self) -> None:
        # A sequel numbered as its own season 1 that Sonarr matched into season 2: the whole range follows the shift.
        parsed = {self._NAME: parsed_info(season=1, episodes=(2,), absolutes=(3,), matched=((2, 2),))}

        placed = by_name(place(parsed, _target([702, 703])))

        assert placed == {self._NAME: ((702, 703), PlacementVerdict.EXACT)}


class TestPlaceNumberedSequence:
    """The numbered sequence pass: one `1..N` sequence indexes the sequence's window among files Sonarr read blind."""

    _WINDOW: ClassVar[list[int]] = [10370, 10371, 10372]
    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
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
        """The season-pack file plus a blind sequence, in file order."""

        parsed: dict[str, ParsedFileInfo | None] = {cls._MAIN: parsed_info(season=1, episodes=(1,))}
        parsed.update(blind(run))
        return parsed

    @classmethod
    def _target(cls) -> PlacementTarget:
        return PlacementTarget([10384, *cls._WINDOW], series_index(cls._ID_BY_KEY))

    def test_a_sequence_beside_a_season_pack_places(self) -> None:
        # The mixed batch that blocks pairing in name order: the season-pack file takes the season and episode in its
        # name and the sequence indexes what is left, which the two-season window keeps sequence over parse out of.
        run = self._run("extra ", range(1, 4))
        result = place(self._batch(run), self._target())

        assert result.placed == {self._MAIN: (10384,), **{name: (10370 + i,) for i, name in enumerate(run)}}
        assert (result.unplaced, result.excluded) == ((), ())
        assert {p.verdict for p in result.placements if p.name in run} == {PlacementVerdict.SEQUENCE}

    def test_a_later_version_displaces_the_earlier_one_in_a_blind_sequence(self) -> None:
        # The blind sequence keeps the higher `vN` and leaves the lower as its duplicate, as sequence over parse does.
        run = self._run("extra ", range(1, 4))
        later = "[grp] show extra 2v2 [bd 1080p x264 10bit flac].mkv"
        parsed = self._batch([*run, later])

        result = place(parsed, self._target())

        assert result.placed == {self._MAIN: (10384,), run[0]: (10370,), later: (10371,), run[2]: (10372,)}
        assert by_name(result)[run[1]] == ((), PlacementVerdict.DUPLICATE)

    def test_a_season_zero_absolute_sequence_still_counts_as_blind(self) -> None:
        # Absolutes alone resolve nothing, so a season-0 sequence reads as blind and indexes the window.
        main = "[grp] show s2 - 01 [bd].mkv"
        run = [f"[grp] ova - 0{i} [bd].mkv" for i in (1, 2, 3)]
        parsed: dict[str, ParsedFileInfo | None] = {main: parsed_info(season=1, episodes=(1,), absolutes=(1,))}
        parsed.update({name: parsed_info(season=0, absolutes=(i + 1,)) for i, name in enumerate(run)})

        result = place(parsed, self._target())

        assert [list(p.ids) for p in result.placements] == [[10384], [10370], [10371], [10372]]

    @pytest.mark.parametrize(
        ("run", "window"),
        [
            pytest.param(_run("a ", range(1, 4)) + _run("b ", range(1, 4)), _WINDOW, id="two runs of one width"),
            # Only a 1..N sequence indexes a window: a 2..4 sequence says nothing about where it starts.
            pytest.param(_run("extra ", range(2, 5)), _WINDOW, id="a run not starting at one"),
            pytest.param(_run("extra ", range(1, 6)), _WINDOW, id="a run wider than the window"),
        ],
    )
    def test_a_sequence_that_cannot_index_the_window_blocks_the_pass(self, run: list[str], window: list[int]) -> None:
        result = place(self._batch(run), PlacementTarget([10384, *window], series_index(self._ID_BY_KEY)))

        assert result.placed == {self._MAIN: (10384,)}
        assert sorted(result.unplaced) == sorted(run)

    def test_an_unknown_parse_blocks_the_pass(self) -> None:
        # An unreadable name anywhere blocks every name number pass, the blind one too.
        run = self._run("extra ", range(1, 4))
        parsed = self._batch(run)
        parsed[_GONE] = None

        result = place(parsed, self._target(), [self._MAIN, *run])

        assert result.placed == {self._MAIN: (10384,)}
        assert sorted(result.unplaced) == sorted(run)


class TestPlacementClassification:
    """What `_pass_classify` calls each unplaced file, and what its classification does NOT buy it earlier."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(1, 1): 601,
    }

    def test_a_file_bound_outside_the_window_still_counts_against_pair_in_order(self) -> None:
        # Exclusion is decided LAST, so a file outside the window is still unplaced while the
        # pairing passes run: the pair with no parsed numbers beside it is never paired in order.
        parsed: dict[str, ParsedFileInfo | None] = {
            "one.mkv": parsed_info(),
            "two.mkv": parsed_info(),
            "far.mkv": parsed_info(season=1, episodes=(1,)),
        }

        result = place(parsed, PlacementTarget([501, 502], series_index(self._ID_BY_KEY)))

        assert result.placed == {}
        assert sorted(result.unplaced) == ["one.mkv", "two.mkv"]
        assert by_name(result)["far.mkv"] == ((), PlacementVerdict.OUTSIDE_WINDOW)

    def test_a_file_whose_parse_resolves_nowhere_may_still_pair_in_absolute_order(self) -> None:
        # Nothing proved it outside the window, so pairing in absolute order may still place it.
        parsed: dict[str, ParsedFileInfo | None] = {"x.mkv": parsed_info(season=9, episodes=(9,), absolutes=(9,))}

        result = place(parsed, PlacementTarget([501], series_index(self._ID_BY_KEY)))

        assert by_name(result) == {"x.mkv": ((501,), PlacementVerdict.ABSOLUTE)}

    def test_the_bogus_season_episode_single_branch_blocks_on_an_empty_index(self) -> None:
        # Over an unserved episode index every season and episode "misses", so none may be called bogus.
        parsed: dict[str, ParsedFileInfo | None] = {"movie.mkv": parsed_info(season=20, episodes=(20,))}

        result = place(parsed, PlacementTarget([501], series_index({})))

        assert by_name(result) == {"movie.mkv": ((), PlacementVerdict.NO_EPISODE_FOUND)}


class TestNumbersDisagree:
    """A name with two numberings is right about at most one of them: the file is never placed by either numbering."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
        EpisodeKey(1, 3): 603,
        EpisodeKey(0, 1): 501,
        EpisodeKey(5, 2): 702,
    }
    _ABSOLUTES: ClassVar[dict[int, int]] = {601: 1, 602: 2, 603: 3, 702: 102}
    """The season's absolutes are 1 to 3, and one far episode carries absolute 102."""

    @classmethod
    def _target(cls, *resolved: int, absolutes: Mapping[int, int] | None = None) -> PlacementTarget:
        index = series_index(cls._ID_BY_KEY, absolutes=cls._ABSOLUTES if absolutes is None else absolutes)
        return PlacementTarget(list(resolved), index, title_names=TitleNames("Show"))

    def test_a_bracketed_number_naming_another_episode_rejects_the_season_and_episode(self) -> None:
        # "(0003)" is episode 3's absolute number, and the season and episode say episode 2.
        name = "[grp] Show S01E02 (0003) (WEB 1080p) [ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_bracketed_number_agreeing_with_the_season_and_episode_places(self) -> None:
        name = "[grp] Show S01E02 (0002) (WEB 1080p) [ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_bracketed_number_no_episode_carries_leaves_the_season_and_episode_alone(self) -> None:
        # A bracketed number that is no absolute of the series (a bitrate, a year).
        name = "[grp] Show S01E02 (2019) [ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_bracketed_number_naming_another_episode_disagrees_with_a_parse_from_matched(self) -> None:
        # Sonarr matched the "02" as episode 2 and gave the file that season and episode. "[0102]" numbers it as
        # episode 102.
        name = "[0102] Show - 02 (WEB 1080p) [ABCD1234].mkv"

        result = place({name: parsed_info(absolutes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_bracketed_number_agreeing_with_a_parse_from_matched_places(self) -> None:
        name = "[0002] Show - 02 (WEB 1080p) [ABCD1234].mkv"

        result = place({name: parsed_info(absolutes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_glued_number_that_is_another_episodes_absolute_rejects_the_guessed_season_and_episode(self) -> None:
        # Sonarr read "102" as S01E02 and matched it, and the series has an episode 102 too.
        name = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_glued_number_of_a_short_series_is_the_season_and_episode(self) -> None:
        # No episode 102 exists.
        name = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"
        absolutes = {601: 1, 602: 2, 603: 3}

        result = place(
            {name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))},
            self._target(601, 602, 603, absolutes=absolutes),
        )

        assert by_name(result) == {name: ((602,), PlacementVerdict.EXACT)}

    def test_a_guessed_season_and_episode_sonarr_never_matched_is_rejected(self) -> None:
        # "0102 v2" read as S01E03 with no series match behind it, and the series has an episode 102.
        name = "Show - 0102 v2 (WEB 1080p) [grp][ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(3,))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_guessed_season_and_episode_beside_a_number_no_episode_carries_stands(self) -> None:
        # No episode 536 exists, and Sonarr matched no series.
        name = "Show - 0536 v2 (WEB 1080p) [grp][ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(3,))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((603,), PlacementVerdict.EXACT)}

    def test_a_title_number_behind_a_glued_number_still_disagrees_with_the_season_and_episode(self) -> None:
        # Sonarr read "102" as S01E02 and matched it, with the title's "100" ahead of it.
        name = "Show.102.The.100.Year.Plan.1080p.WEB-grp.mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_spelled_out_season_and_episode_sonarr_never_matched_still_places(self) -> None:
        # The name says S01E03 itself, and Sonarr matched no series.
        name = "Show - S01E03 (WEB 1080p) [grp][ABCD1234].mkv"

        result = place({name: parsed_info(season=1, episodes=(3,))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((603,), PlacementVerdict.EXACT)}

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("Show.S01.Ep03.1080p.WEB.H.264-grp.mkv", id="a codec number"),
            pytest.param("Show 100 Season 1 Episode 03 [grp].mkv", id="a title number"),
        ],
    )
    def test_a_codec_or_title_number_beside_an_unmatched_season_and_episode_does_not_disagree(self, name: str) -> None:
        # The only other numbers are "264" and the title's "100".
        result = place({name: parsed_info(season=1, episodes=(3,))}, self._target(601, 602, 603))

        assert by_name(result) == {name: ((603,), PlacementVerdict.EXACT)}

    @pytest.mark.parametrize(
        ("name", "episodes", "ids"),
        [
            pytest.param("Show S01E03v2 2017 1080p [grp].mkv", (3,), (603,), id="a versioned key"),
            pytest.param("Show S01E02E03 2017 1080p [grp].mkv", (2, 3), (602, 603), id="a two-episode key"),
        ],
    )
    def test_a_spelled_out_season_and_episode_in_any_form_sonarr_never_matched_still_places(
        self, name: str, episodes: tuple[int, ...], ids: tuple[int, ...]
    ) -> None:
        result = place({name: parsed_info(season=1, episodes=episodes)}, self._target(601, 602, 603))

        assert by_name(result) == {name: (ids, PlacementVerdict.EXACT)}

    def test_numbers_that_disagree_over_episodes_the_window_lacks_are_left_to_their_window(self) -> None:
        # S01E02 or episode 102, and neither is in the specials window.
        name = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(501))

        assert by_name(result) == {name: ((), PlacementVerdict.OUTSIDE_WINDOW)}

    def test_numbers_that_disagree_over_an_episode_the_window_holds_by_the_other_number_are_misnumbered(self) -> None:
        # The season and episode say S01E02, outside the window, and the name number says episode 102, inside it.
        name = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"

        result = place({name: parsed_info(season=1, episodes=(2,), matched=((1, 2),))}, self._target(702))

        assert by_name(result) == {name: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_lower_version_of_a_number_that_disagrees_is_a_copy(self) -> None:
        # The top version at 102 has numbers that disagree (S01E02 or episode 102), and the other cut of 102 is a
        # copy of it.
        top = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"
        copy = "Show.102.Canon.Cut.v2.480p.DVD.FLAC2.0.x264-grp.mkv"
        parsed = {name: parsed_info(season=1, episodes=(2,), matched=((1, 2),)) for name in (top, copy)}

        result = place(parsed, self._target(601, 602, 603))

        assert by_name(result) == {top: ((), PlacementVerdict.MISNUMBERED), copy: ((), PlacementVerdict.DUPLICATE)}

    def test_a_member_whose_numbers_disagree_is_still_placed_by_its_sequence(self) -> None:
        # Sonarr read "0003" as S00E03 (a season and episode the series lacks) and the first two as absolutes.
        names = [f"Show - 000{n} (BD 1080p) [grp][ABCD123{n}].mkv" for n in (1, 2, 3)]
        parsed = {
            names[0]: parsed_info(absolutes=(1,), matched=((1, 1),)),
            names[1]: parsed_info(absolutes=(2,), matched=((1, 2),)),
            names[2]: parsed_info(season=0, episodes=(3,)),
        }

        result = place(parsed, self._target(601, 602, 603))

        assert by_name(result) == {
            name: ((600 + n,), PlacementVerdict.SEQUENCE_OVER_PARSE) for n, name in enumerate(names, 1)
        }


class TestNextInLine:
    """A rejection gives no episode to the file that would have lost it to the rejected parse."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
        EpisodeKey(1, 3): 603,
        EpisodeKey(5, 2): 702,
    }
    _ABSOLUTES: ClassVar[dict[int, int]] = {601: 1, 602: 2, 603: 3, 702: 102}
    _NUMBERS_DISAGREE: ClassVar[str] = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"
    """Read as S01E02 from its glued number, which is also episode 102's absolute number: rejected."""

    @classmethod
    def _target(cls, titles: Mapping[int, str] | None = None) -> PlacementTarget:
        index = series_index(cls._ID_BY_KEY, absolutes=cls._ABSOLUTES, titles=titles)
        return PlacementTarget([601, 602, 603], index, title_names=TitleNames("Show"))

    def test_the_next_in_line_never_takes_the_episode_the_rejected_file_would_have_won(self) -> None:
        # Sonarr matched the OVA to S01E02 by its absolute number, the episode the other file's season and episode
        # point at.
        ova = "Show - 02 (OVA) [grp].mkv"
        parsed = {
            self._NUMBERS_DISAGREE: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            ova: parsed_info(absolutes=(2,), matched=((1, 2),)),
        }

        result = place(parsed, self._target())

        assert by_name(result) == {
            self._NUMBERS_DISAGREE: ((), PlacementVerdict.MISNUMBERED),
            ova: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_the_next_in_line_is_not_placed_on_its_lost_episode_by_its_title(self) -> None:
        titled = "Show - 02 - Second Episode Title [grp].mkv"
        parsed = {
            self._NUMBERS_DISAGREE: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            titled: parsed_info(absolutes=(2,), matched=((1, 2),)),
        }

        result = place(parsed, self._target(titles={602: "Second Episode Title"}))

        assert by_name(result) == {
            self._NUMBERS_DISAGREE: ((), PlacementVerdict.MISNUMBERED),
            titled: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_the_next_in_line_is_not_placed_on_its_lost_episode_by_an_anilist_name(self) -> None:
        # One id unused, and the next in line, with no parsed number, is the file an AniList name matches.
        titled = "Show - The Special Title [grp].mkv"
        parsed = {
            self._NUMBERS_DISAGREE: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            titled: parsed_info(matched=((1, 2),)),
        }
        index = series_index(self._ID_BY_KEY, absolutes=self._ABSOLUTES)

        result = place(
            parsed, PlacementTarget([602], index, title_names=TitleNames("Show", ("Show The Special Title",)))
        )

        assert by_name(result) == {
            self._NUMBERS_DISAGREE: ((), PlacementVerdict.MISNUMBERED),
            titled: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_a_file_ranked_ahead_of_the_rejected_one_keeps_its_episode(self) -> None:
        # Equal rank and version, and the spelled-out season and episode comes first in the batch.
        spelled_name = "[grp] Show S01E02.mkv"
        disagreeing_name = "Show.102.480p.DVD.FLAC2.0.x264-grp.mkv"
        parsed = {
            spelled_name: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            disagreeing_name: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
        }

        result = place(parsed, self._target())

        assert by_name(result) == {
            spelled_name: ((602,), PlacementVerdict.EXACT),
            disagreeing_name: ((), PlacementVerdict.MISNUMBERED),
        }

    def test_the_next_in_line_neither_blocks_the_rejected_files_sequence_nor_outlives_it(self) -> None:
        # The first name's bracketed number names episode 102, the season and episode sequence of three fits the
        # window, and the OVA is matched to S01E01.
        names = ["[grp] Show S01E01 (0102).mkv", "[grp] Show S01E02 (0002).mkv", "[grp] Show S01E03 (0003).mkv"]
        ova = "Show - 01 (OVA) [grp].mkv"
        parsed = {
            ova: parsed_info(absolutes=(1,), matched=((1, 1),)),
            **{name: parsed_info(season=1, episodes=(n,), matched=((1, n),)) for n, name in enumerate(names, 1)},
        }

        result = place(parsed, self._target())

        assert by_name(result) == {
            ova: ((), PlacementVerdict.DUPLICATE),
            **{name: ((600 + n,), PlacementVerdict.SEQUENCE_OVER_PARSE) for n, name in enumerate(names, 1)},
        }

    def test_the_next_in_line_inside_the_rejected_files_sequence_is_still_placed_by_its_name_number(self) -> None:
        # A torrent numbered 001 to 103: Sonarr read the first hundred as absolutes and the rest as glued seasons
        # and episodes.
        names = {n: f"Show.{n:03d}.v4.480p.DVD.FLAC2.0.x264-grp.mkv" for n in range(1, 104)}
        parsed = {
            name: parsed_info(absolutes=(n,), matched=((1, n),))
            if n <= 100
            else parsed_info(season=1, episodes=(n - 100,), matched=((1, n - 100),))
            for n, name in names.items()
        }
        index = series_index({EpisodeKey(1, n): 1000 + n for n in names}, absolutes={1000 + n: n for n in names})

        result = place(parsed, PlacementTarget(list(range(1001, 1104)), index, title_names=TitleNames("Show")))

        assert by_name(result) == {
            name: ((1000 + n,), PlacementVerdict.SEQUENCE_OVER_PARSE) for n, name in names.items()
        }

    def test_a_rejected_lower_copy_still_keeps_the_top_version_off_its_episode(self) -> None:
        # Only the lower copy carries the disagreeing season and episode, and a season and episode in the name
        # outranks the top version's matched episode. It is still in the running when the next in line are read, so
        # the top version loses the episode to it before the copy is set aside.
        top = "Show.102.v4.480p.DVD.FLAC2.0.x264-grp.mkv"
        copy = "Show.102.Canon.Cut.v2.480p.DVD.FLAC2.0.x264-grp.mkv"
        parsed = {
            top: parsed_info(matched=((1, 2),)),
            copy: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
        }

        result = place(parsed, self._target())

        assert by_name(result) == {
            top: ((), PlacementVerdict.NO_EPISODE_FOUND),
            copy: ((), PlacementVerdict.DUPLICATE),
        }


class TestInterleavedSpecial:
    """A season and episode sequence past its season's end, where TVDB numbers a special inside the season's absolutes.

    The release numbered the special as an episode, or appended an extra: the seasons and episodes go past the end
    the same. The members before the special agree either way. From the special on, the two numberings name
    different episodes.
    """

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
        EpisodeKey(1, 3): 603,
        EpisodeKey(0, 1): 501,
    }
    _RUN: ClassVar[list[str]] = [f"[grp] Show - S01E0{n}.mkv" for n in (1, 2, 3, 4)]

    @classmethod
    def _place(cls, absolutes: Mapping[int, int], ids: Sequence[int] = (601, 602, 603)) -> BatchPlacements:
        parsed = {
            cls._RUN[0]: parsed_info(season=1, episodes=(1,), matched=((1, 1),)),
            cls._RUN[1]: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            cls._RUN[2]: parsed_info(season=1, episodes=(3,), matched=((1, 3),)),
            cls._RUN[3]: parsed_info(season=1, episodes=(4,)),
        }
        target = PlacementTarget(
            list(ids), series_index(cls._ID_BY_KEY, absolutes=absolutes), title_names=TitleNames("Show")
        )
        return place(parsed, target)

    def test_the_members_from_the_special_on_are_misnumbered(self) -> None:
        # The special is absolute 3, between S01E02 and S01E03: "S01E03" is the special or the aired third
        # episode, and "S01E04" is the aired third episode or nothing.
        result = self._place({601: 1, 602: 2, 501: 3, 603: 4})

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((), PlacementVerdict.MISNUMBERED),
            self._RUN[3]: ((), PlacementVerdict.MISNUMBERED),
        }

    def test_a_window_holding_the_special_places_no_member_from_the_special_on_by_position(self) -> None:
        # The sequence fits the window in absolute order, its third member on the special.
        result = self._place({601: 1, 602: 2, 501: 3, 603: 4}, (601, 602, 501, 603))

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((), PlacementVerdict.MISNUMBERED),
            self._RUN[3]: ((), PlacementVerdict.MISNUMBERED),
        }

    def test_a_special_outside_the_seasons_numbering_leaves_the_sequence_alone(self) -> None:
        # The special has no absolute number.
        result = self._place({601: 1, 602: 2, 603: 3})

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((603,), PlacementVerdict.EXACT),
            self._RUN[3]: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_a_sequence_numbered_on_past_a_double_length_special_is_misnumbered_from_the_special_on(self) -> None:
        # The release numbers the special between S01E02 and S01E03 as "S01E03-E04" and the aired third episode as
        # "S01E05".
        names = [f"[grp] Show - S01E{key}.mkv" for key in ("01", "02", "03-E04", "05")]
        parsed = {
            names[0]: parsed_info(season=1, episodes=(1,), matched=((1, 1),)),
            names[1]: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            names[2]: parsed_info(season=1, episodes=(3, 4), matched=((1, 3),)),
            names[3]: parsed_info(season=1, episodes=(5,)),
        }
        index = series_index(self._ID_BY_KEY, absolutes={601: 1, 602: 2, 501: 3, 603: 4})

        result = place(parsed, PlacementTarget([601, 602, 603], index, title_names=TitleNames("Show")))

        assert by_name(result) == {
            names[0]: ((601,), PlacementVerdict.EXACT),
            names[1]: ((602,), PlacementVerdict.EXACT),
            names[2]: ((), PlacementVerdict.MISNUMBERED),
            names[3]: ((), PlacementVerdict.MISNUMBERED),
        }

    def test_a_sequence_numbered_within_its_season_is_placed_as_read(self) -> None:
        # Two of three episodes, numbered past the special's position but not past the season's end.
        names = [f"[grp] Show - S01E0{n}.mkv" for n in (2, 3)]
        parsed = {
            names[0]: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            names[1]: parsed_info(season=1, episodes=(3,), matched=((1, 3),)),
        }
        index = series_index(self._ID_BY_KEY, absolutes={601: 1, 602: 2, 501: 3, 603: 4})

        result = place(parsed, PlacementTarget([601, 602, 603], index, title_names=TitleNames("Show")))

        assert by_name(result) == {
            names[0]: ((602,), PlacementVerdict.EXACT),
            names[1]: ((603,), PlacementVerdict.EXACT),
        }

    def test_the_windows_the_sequence_spreads_over_reject_it_and_another_seasons_reads_it_as_it_was(self) -> None:
        # From the special on, the members are S01E03 and the special, or the special and nothing. The season's
        # window and the special's window each hold one of those, and the second season's window holds none.
        parsed = {
            self._RUN[0]: parsed_info(season=1, episodes=(1,), matched=((1, 1),)),
            self._RUN[1]: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
            self._RUN[2]: parsed_info(season=1, episodes=(3,), matched=((1, 3),)),
            self._RUN[3]: parsed_info(season=1, episodes=(4,)),
        }
        index = series_index({**self._ID_BY_KEY, EpisodeKey(2, 1): 701}, absolutes={601: 1, 602: 2, 501: 3, 603: 4})

        verdicts = {
            resolved: {name: v for name, (_, v) in by_name(place(parsed, target)).items() if name in self._RUN[2:]}
            for resolved in (603, 501, 701)
            for target in (PlacementTarget([resolved], index, title_names=TitleNames("Show")),)
        }

        assert verdicts == {
            603: {self._RUN[2]: PlacementVerdict.MISNUMBERED, self._RUN[3]: PlacementVerdict.MISNUMBERED},
            501: {self._RUN[2]: PlacementVerdict.MISNUMBERED, self._RUN[3]: PlacementVerdict.MISNUMBERED},
            701: {self._RUN[2]: PlacementVerdict.OUTSIDE_WINDOW, self._RUN[3]: PlacementVerdict.NO_EPISODE_FOUND},
        }


class TestPlacerFacts:
    """`PlacerResult.facts`: every name as the passes read it before the first pass, file map and gone names too."""

    def test_every_torrent_name_gets_the_facts_the_passes_judge_by(self) -> None:
        # The titled file carries episode 2's title, and the extra is an NCOP. The file in the file map is numbered by
        # its own title's number, which no title confirms. The gone name reads inside the window on its matched
        # episode.
        titled, extra, counted = (
            "show - 02 The Return [grp].mkv",
            "[grp] show - NCOP1 [bd].mkv",
            "show - 03 Days of Summer [grp].mkv",
        )
        parsed = {
            titled: parsed_info(absolutes=(2,), matched=((1, 2),)),
            extra: parsed_info(),
            counted: parsed_info(absolutes=(3,), matched=((1, 3),)),
            _GONE: parsed_info(absolutes=(1,), matched=((1, 1),)),
        }

        facts = place_batch(batch(parsed, [titled, extra], file_map={counted: (603,)}), _target(_FIRST_THREE)).facts

        assert facts == {
            titled: NameFacts(
                episode_by_title=602, parses_inside_window=True, extra=False, parsed_by_title_number=False
            ),
            extra: NameFacts(
                episode_by_title=None, parses_inside_window=False, extra=True, parsed_by_title_number=False
            ),
            counted: NameFacts(
                episode_by_title=None, parses_inside_window=True, extra=False, parsed_by_title_number=True
            ),
            _GONE: NameFacts(
                episode_by_title=None, parses_inside_window=True, extra=False, parsed_by_title_number=False
            ),
        }

    def test_a_parse_outside_the_window_does_not_read_inside(self) -> None:
        parsed = {"show - 05 [grp].mkv": parsed_info(absolutes=(5,), matched=((2, 1),))}

        facts = place_batch(batch(parsed), _target(_FIRST_THREE)).facts

        assert facts["show - 05 [grp].mkv"].parses_inside_window is False

    def test_a_parse_a_pass_rejects_still_reads_inside(self) -> None:
        # Sonarr matched the name to episode 2, inside the window, but it carries episode 3's title: the contradiction
        # rejects the parse, and the episode-title pass places the file.
        name = "show - 03 - Festival Night [grp].mkv"
        parsed = {name: parsed_info(absolutes=(3,), matched=((1, 2),))}

        result = place_batch(batch(parsed), _target(_FIRST_THREE))

        assert by_name(result.placements) == {name: ((603,), PlacementVerdict.EPISODE_TITLE)}
        assert result.facts[name].parses_inside_window is True

    def test_a_parse_the_season_re_read_replaces_keeps_its_first_read(self) -> None:
        # A special TVDB interleaved after the second episode shifted Sonarr's match: the third file onto the special,
        # outside the window. The season re-read reads it as the season's third episode, where the exact pass places
        # it.
        run = numbered_names("show", 4)
        matched = ((1, 1), (1, 2), (0, 1), (1, 3))
        parsed = {name: parsed_info(matched=(pair,)) for name, pair in zip(run, matched, strict=True)}

        result = place_batch(batch(parsed), _target([601, 602, 603, 604]))

        assert by_name(result.placements)[run[2]] == ((603,), PlacementVerdict.EXACT)
        assert result.facts[run[2]].parses_inside_window is False
