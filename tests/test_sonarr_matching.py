# pyright: strict
"""The copy of Sonarr's matching: the title key, the alias a name is read by, and the episodes a parse matches."""

from collections.abc import Sequence

import pytest

from pearlarr.aired_view import AiredSeries, stand_in
from pearlarr.arr_models import AlternateTitle, MatchedEpisode, SonarrEpisode
from pearlarr.sonarr_accents import UNACCENTED
from pearlarr.sonarr_matching import AiredMatcher, NoAnswer, sonarr_title_key
from pearlarr.tvdb_orders import AIRED_ORDER, EpisodeDetails, OrderRow

from .builders import parsed_info, sonarr_ep

_TITLE = "Series Title"
_JOINING_WORDS = ("the", "and", "or", "an", "of", "a")


def _alias(
    title: str = _TITLE, *, scene: int | None = None, tvdb: int | None = None, origin: str = "", comment: str = ""
) -> AlternateTitle:
    """An alternate title, field for field: its scene season, its TVDB season, its origin and its comment."""

    return AlternateTitle(
        title=title, scene_season_number=scene, season_number=tvdb, scene_origin=origin, comment=comment
    )


def _matcher(rows: Sequence[SonarrEpisode], *aliases: AlternateTitle) -> AiredMatcher:
    return AiredMatcher.of(AiredSeries(order=AIRED_ORDER, in_sonarr=tuple(rows), stand_ins=()), aliases)


def _ids(answer: tuple[MatchedEpisode, ...] | NoAnswer) -> list[int | None]:
    assert not isinstance(answer, NoAnswer)
    return [episode.id for episode in answer]


# --- sonarr_title_key: Sonarr's own cases ------------------------------------


@pytest.mark.parametrize(
    ("title", "key"),
    [
        ("Series", "series"),
        ("Series (2009)", "series2009"),
        ("Series.2010", "series2010"),
        ("Series_and_Title_Sonarr", "seriestitlesonarr"),
        ("CaPitAl", "capital"),
        ("peri.od", "period"),
        ("this.^&%^**$%@#$!That", "thisthat"),
        ("test/test", "testtest"),
        ("90210", "90210"),
        ("24", "24"),
        ("Test: Something à Deux", "testsomethingdeux"),
        ("Parler à", "parlera"),
        ("Ríkið", "rikid"),
        ("The Series", "theseries"),
        ("The Series Show With Sonarr Dev", "theseriesshowwithsonarrdev"),
        ("The.Series.Show", "theseriesshow"),
        ("Series Title A", "seriestitlea"),
        ("3%", "3percent"),
        ("Series Top & 100% Coding Developers", "seriestop100percentcodingdevelopers"),
        ("Series Title What's Your W@%rking Deal?!", "seriestitlewhatsyourwrkingdeal"),
    ],
)
def test_title_key_cleans_a_title_as_sonarr_does(title: str, key: str) -> None:
    assert sonarr_title_key(title) == key


@pytest.mark.parametrize("word", _JOINING_WORDS)
def test_title_key_drops_a_joining_word_between_words(word: str) -> None:
    for shape in ("word.{}.word", "word {} word", "word-{}-word"):
        assert sonarr_title_key(shape.format(word)) == "wordword"


@pytest.mark.parametrize("word", _JOINING_WORDS)
def test_title_key_keeps_a_joining_word_that_ends_the_title(word: str) -> None:
    for shape in ("word.word.{}", "word-word-{}", "word-word {}"):
        assert sonarr_title_key(shape.format(word)) == f"wordword{word}"


@pytest.mark.parametrize("word", _JOINING_WORDS)
def test_title_key_keeps_a_joining_word_that_starts_the_title(word: str) -> None:
    for shape in ("{}.word.word", "{}-word-word", "{} word word"):
        assert sonarr_title_key(shape.format(word)) == f"{word}wordword"


@pytest.mark.parametrize("word", _JOINING_WORDS)
def test_title_key_keeps_a_joining_word_inside_a_longer_word(word: str) -> None:
    for shape in ("word.{}word", "word {}word", "word-{}word", "word{}.word", "word{}-word"):
        assert sonarr_title_key(shape.format(word)) == f"word{word}word"


# --- sonarr_title_key: where Python and .NET part ------------------------------


@pytest.mark.parametrize(("title", "key"), [("x 3% y", "x3percenty"), ("x3% y", "x3y")])
def test_title_key_spells_percent_after_a_number_that_starts_at_a_word_boundary(title: str, key: str) -> None:
    assert sonarr_title_key(title) == key


def test_title_key_keeps_only_what_dotnet_counts_as_a_word_character() -> None:
    # A fraction is a number to Python's isalnum, and no word character to .NET.
    assert sonarr_title_key("Series Title ½") == "seriestitle"


def test_title_key_strips_accents_by_sonarrs_table_and_no_other_way() -> None:
    assert sonarr_title_key("Serieß Tiłle þ") == "seriesstilleth"
    # The table has no combining accent, and maps a small capital to the capital.
    assert sonarr_title_key("Séries ᴀ") == "sériesA"


def test_the_accent_table_holds_the_entries_a_lowercased_string_can_hold() -> None:
    assert len(UNACCENTED) == 488
    assert all(len(lowered := chr(code).lower()) != 1 or lowered == chr(code) for code in UNACCENTED)


# --- AiredMatcher: the alias a name is read by ---------------------------------------


def test_a_name_is_read_by_the_aliases_under_its_title_key() -> None:
    matcher = _matcher([sonarr_ep(1, 2, ep_id=12), sonarr_ep(3, 2, ep_id=32)], _alias("Series: The Title!", scene=3))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(2,), series_title="series.title"))) == [32]
    assert _ids(matcher.matches(parsed_info(season=1, episodes=(2,), series_title="Other Title"))) == [12]


@pytest.mark.parametrize(
    ("title", "ep_id"),
    [
        ("Series Title!!", 15),
        ("Series Title`!!", 25),
        ("Series Title!!!", 35),
        ("Series Title!!!!", 35),
        ("Series Title !!", 15),
    ],
)
def test_of_aliases_under_one_key_the_one_titled_as_the_name_or_else_closest_to_it_reads_it(
    title: str, ep_id: int
) -> None:
    # Sonarr's own cases, with an absolute number read in each alias's season.
    rows = [sonarr_ep(1, 5, ep_id=15), sonarr_ep(2, 5, ep_id=25), sonarr_ep(3, 5, ep_id=35)]
    aliases = (
        _alias("Series Title!!", scene=1),
        _alias("Series Title`!!", scene=2),
        _alias("Series Title!!!", scene=3),
    )

    assert _ids(_matcher(rows, *aliases).matches(parsed_info(season=0, absolutes=(5,), series_title=title))) == [ep_id]


def test_with_no_exact_title_the_alias_reached_by_dropping_characters_beats_one_reached_by_adding() -> None:
    rows = [sonarr_ep(2, 5, ep_id=25), sonarr_ep(3, 5, ep_id=35)]
    matcher = _matcher(rows, _alias("Series Title!", scene=2), _alias("SeriesTitle", scene=3))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(5,), series_title=_TITLE))) == [35]


def test_of_aliases_titled_as_the_name_the_one_with_the_latest_tvdb_season_comes_first() -> None:
    # A season of -1 comes before none, as .NET sorts a missing number lowest.
    rows = [sonarr_ep(1, 2, ep_id=12), sonarr_ep(2, 2, ep_id=22)]
    matcher = _matcher(rows, _alias(scene=2), _alias(tvdb=-1))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(2,), series_title=_TITLE))) == [12]


def test_of_aliases_as_close_to_the_name_as_each_other_the_one_with_the_latest_tvdb_season_wins() -> None:
    rows = [sonarr_ep(1, 2, ep_id=12), sonarr_ep(2, 2, ep_id=22)]
    matcher = _matcher(rows, _alias("Series-Title", scene=2), _alias("Series_Title", tvdb=-1))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(2,), series_title=_TITLE))) == [12]


def test_an_absolute_number_is_read_in_the_season_of_the_first_alias_that_maps_no_season() -> None:
    rows = [sonarr_ep(2, 5, ep_id=25), sonarr_ep(4, 5, ep_id=45)]
    matcher = _matcher(rows, _alias(scene=2, tvdb=3), _alias(scene=4))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(5,), series_title=_TITLE))) == [45]


def test_a_name_with_a_season_is_read_by_the_latest_mapping_alias_that_starts_by_then() -> None:
    # Sonarr's own case: of two aliases started by season 4, the later one.
    rows = [sonarr_ep(2, 1, ep_id=21), sonarr_ep(3, 1, ep_id=31), sonarr_ep(4, 1, ep_id=41), sonarr_ep(5, 1, ep_id=51)]
    matcher = _matcher(rows, _alias(scene=2, tvdb=3), _alias(scene=3, tvdb=3))

    assert _ids(matcher.matches(parsed_info(season=4, episodes=(1,), series_title=_TITLE))) == [41]
    assert _ids(matcher.matches(parsed_info(season=2, episodes=(1,), series_title=_TITLE))) == [31]


def test_a_name_whose_season_is_before_every_mapping_alias_is_read_by_none() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11), sonarr_ep(9, 1, ep_id=91)]
    matcher = _matcher(rows, _alias(scene=2, tvdb=3), _alias(scene=9))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE))) == [11]


def test_an_alias_with_a_tvdb_season_of_minus_one_maps_no_season() -> None:
    # Were it a mapping alias, no alias would give the absolute number a season, and TVDB's would be read.
    rows = [sonarr_ep(1, 5, ep_id=15, absolute=5), sonarr_ep(2, 5, ep_id=25, absolute=17)]
    matcher = _matcher(rows, _alias(scene=2, tvdb=-1))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(5,), series_title=_TITLE))) == [25]


def test_an_alias_with_a_tvdb_season_below_zero_leaves_the_names_season_as_it_is() -> None:
    # Sonarr's own case.
    matcher = _matcher([sonarr_ep(1, 1, ep_id=11)], _alias(scene=1, tvdb=-1))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE))) == [11]


# --- AiredMatcher: a season and episode ------------------------------------------


def test_a_season_and_episode_name_matches_tvdbs_numbers_in_the_names_order() -> None:
    matcher = _matcher([sonarr_ep(1, 1, ep_id=11), sonarr_ep(1, 2, ep_id=12), sonarr_ep(1, 3, ep_id=13)])

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(3, 9, 1)))) == [13, 11]


def test_the_numbers_releases_give_an_episode_are_read_before_tvdbs() -> None:
    rows = [
        sonarr_ep(1, 1, ep_id=11),
        sonarr_ep(1, 2, ep_id=12),
        sonarr_ep(2, 1, ep_id=21, scene_season=1, scene_episode=1),
    ]

    # Releases number S02E01 as S01E01. They number nothing S01E02, so TVDB's S01E02 it is.
    assert _ids(_matcher(rows).matches(parsed_info(season=1, episodes=(1, 2)))) == [21, 12]


def test_two_episodes_releases_number_alike_both_match() -> None:
    rows = [
        sonarr_ep(1, 1, ep_id=11, scene_season=1, scene_episode=1),
        sonarr_ep(1, 2, ep_id=12, scene_season=1, scene_episode=1),
    ]

    assert _ids(_matcher(rows).matches(parsed_info(season=1, episodes=(1,)))) == [11, 12]


def test_an_alias_of_tvdb_origin_has_the_name_read_by_tvdbs_numbers_alone() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11), sonarr_ep(2, 1, ep_id=21, scene_season=1, scene_episode=1)]
    matcher = _matcher(rows, _alias(tvdb=-1, origin="tvdb"))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE))) == [11]


def test_an_alias_that_maps_a_season_shifts_the_names_season_by_the_gap() -> None:
    rows = [sonarr_ep(2, 1, ep_id=21), sonarr_ep(5, 1, ep_id=51), sonarr_ep(6, 1, ep_id=61)]
    matcher = _matcher(rows, _alias(scene=2, tvdb=5))

    assert _ids(matcher.matches(parsed_info(season=2, episodes=(1,), series_title=_TITLE))) == [51]
    assert _ids(matcher.matches(parsed_info(season=3, episodes=(1,), series_title=_TITLE))) == [61]


def test_an_alias_with_a_scene_season_alone_turns_a_first_season_into_its_own() -> None:
    rows = [sonarr_ep(1, 2, ep_id=12), sonarr_ep(3, 2, ep_id=32)]
    matcher = _matcher(rows, _alias(scene=3))

    assert _ids(matcher.matches(parsed_info(season=1, episodes=(2,), series_title=_TITLE))) == [32]


def test_a_name_with_no_season_matches_nothing() -> None:
    matcher = _matcher([sonarr_ep(1, 1, ep_id=11)])

    assert matcher.matches(parsed_info(season=-1, episodes=(1,))) == ()
    assert matcher.matches(parsed_info(episodes=(1,))) == ()


def test_a_special_name_matches_only_in_season_zero() -> None:
    matcher = _matcher([sonarr_ep(0, 1, ep_id=1), sonarr_ep(1, 1, ep_id=11)])

    assert _ids(matcher.matches(parsed_info(season=0, episodes=(1,), special=True))) == [1]
    assert matcher.matches(parsed_info(season=1, episodes=(1,), special=True)) == ()


def test_a_row_with_no_numbers_is_never_matched() -> None:
    matcher = _matcher([sonarr_ep(None, None, ep_id=5, absolute=1)])

    assert matcher.matches(parsed_info(season=0, absolutes=(1,))) == ()


def test_a_stand_in_matches_like_one_of_sonarrs_rows() -> None:
    missing = stand_in(OrderRow(1, 2, 902, 2), EpisodeDetails(title="Second", air_date=None, runtime=24))
    view = AiredSeries(order="dvd", in_sonarr=(sonarr_ep(1, 1, ep_id=11),), stand_ins=(missing,))

    assert _ids(AiredMatcher.of(view, ()).matches(parsed_info(season=1, episodes=(1, 2)))) == [11, -902]


# --- AiredMatcher: a whole season -------------------------------------------------


def test_a_full_season_name_matches_the_season_as_releases_number_it_else_as_tvdb_does() -> None:
    rows = [
        sonarr_ep(1, 1, ep_id=11, scene_season=2, scene_episode=1),
        sonarr_ep(1, 2, ep_id=12, scene_season=2, scene_episode=2),
        sonarr_ep(2, 1, ep_id=21),
        sonarr_ep(3, 1, ep_id=31),
    ]
    matcher = _matcher(rows)

    assert _ids(matcher.matches(parsed_info(season=2, full_season=True))) == [11, 12]
    assert _ids(matcher.matches(parsed_info(season=3, full_season=True))) == [31]


def test_an_alias_of_tvdb_origin_has_a_full_season_read_by_tvdbs_seasons_alone() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11, scene_season=2, scene_episode=1), sonarr_ep(2, 1, ep_id=21)]
    matcher = _matcher(rows, _alias(tvdb=-1, origin="tvdb"))

    assert _ids(matcher.matches(parsed_info(season=2, full_season=True, series_title=_TITLE))) == [21]


# --- AiredMatcher: absolute numbers -----------------------------------------------


def test_an_absolute_number_is_the_one_episode_releases_number_so_series_wide() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11, absolute=1), sonarr_ep(1, 2, ep_id=12, absolute=2, scene_absolute=1)]

    assert _ids(_matcher(rows).matches(parsed_info(season=0, absolutes=(1,)))) == [12]


def test_an_absolute_number_releases_give_two_episodes_is_tvdbs_absolute_number() -> None:
    rows = [
        sonarr_ep(1, 1, ep_id=11, absolute=1),
        sonarr_ep(1, 2, ep_id=12, absolute=2, scene_absolute=1),
        sonarr_ep(1, 3, ep_id=13, absolute=3, scene_absolute=1),
    ]

    assert _ids(_matcher(rows).matches(parsed_info(season=0, absolutes=(1,)))) == [11]


def test_an_alias_with_a_season_has_an_absolute_number_read_as_an_episode_of_it() -> None:
    rows = [sonarr_ep(1, 3, ep_id=13, absolute=3), sonarr_ep(2, 3, ep_id=23, absolute=15)]
    matcher = _matcher(rows, _alias(scene=2))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(3,), series_title=_TITLE))) == [23]


def test_an_alias_for_every_season_sends_an_absolute_number_straight_to_tvdbs() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11, absolute=1), sonarr_ep(1, 2, ep_id=12, absolute=2, scene_absolute=1)]
    matcher = _matcher(rows, _alias(scene=-1))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(1,), series_title=_TITLE))) == [11]


def test_only_a_season_past_the_first_with_no_episode_number_has_an_absolute_number_read_within_it() -> None:
    rows = [
        sonarr_ep(1, 1, ep_id=11, absolute=13),
        sonarr_ep(1, 13, ep_id=113, absolute=25),
        sonarr_ep(2, 13, ep_id=213, absolute=49),
    ]
    matcher = _matcher(rows)

    assert _ids(matcher.matches(parsed_info(season=2, absolutes=(13,)))) == [213]
    assert _ids(matcher.matches(parsed_info(season=1, absolutes=(13,)))) == [11]


def test_a_season_the_name_gives_with_an_episode_number_has_no_absolute_number_read_within_it() -> None:
    # Sonarr's own case.
    rows = [sonarr_ep(1, 1, ep_id=11, absolute=1), sonarr_ep(2, 1, ep_id=21, absolute=13)]

    assert _ids(_matcher(rows).matches(parsed_info(season=2, episodes=(1,), absolutes=(1,)))) == [11]


def test_an_alias_of_tvdb_origin_has_an_absolute_number_read_by_tvdbs_absolute_numbers_alone() -> None:
    rows = [
        sonarr_ep(1, 13, ep_id=113, absolute=13),
        sonarr_ep(1, 14, ep_id=114, absolute=14, scene_season=1, scene_episode=13, scene_absolute=13),
        sonarr_ep(2, 13, ep_id=213, absolute=27),
    ]
    matcher = _matcher(rows, _alias(tvdb=-1, origin="tvdb"))

    assert _ids(matcher.matches(parsed_info(season=0, absolutes=(13,), series_title=_TITLE))) == [113]
    assert _ids(matcher.matches(parsed_info(season=2, absolutes=(13,), series_title=_TITLE))) == [113]


def test_a_special_absolute_number_is_the_special_of_that_number_else_tvdbs_absolute_number() -> None:
    numbered = [sonarr_ep(0, 2, ep_id=2), sonarr_ep(1, 2, ep_id=12, absolute=2)]
    name = parsed_info(season=0, absolutes=(2,), special=True)

    assert _ids(_matcher(numbered).matches(name)) == [2]
    assert _ids(_matcher(numbered[1:]).matches(name)) == [12]


def test_a_special_absolute_number_is_read_by_tvdbs_special_numbers_alone() -> None:
    # Sonarr's own case.
    rows = [sonarr_ep(0, 1, ep_id=1), sonarr_ep(0, 2, ep_id=2, scene_season=0, scene_episode=1, scene_absolute=0)]

    assert _ids(_matcher(rows).matches(parsed_info(season=0, absolutes=(1,), special=True))) == [1]


# --- AiredMatcher: episode 0 of a season --------------------------------------------


def test_an_episode_zero_releases_number_so_is_then_looked_up_by_tvdbs_numbers_alone() -> None:
    by_releases = sonarr_ep(0, 1, ep_id=1, title="Extra", scene_season=2, scene_episode=0)
    name = parsed_info(season=2, episodes=(0,))

    assert _matcher([by_releases]).matches(name) == ()
    assert _ids(_matcher([by_releases, sonarr_ep(2, 0, ep_id=20)]).matches(name)) == [20]


def test_an_episode_zero_has_no_answer_when_a_special_has_a_title_to_search_for() -> None:
    matcher = _matcher([sonarr_ep(0, 1, ep_id=1, title="Extra"), sonarr_ep(2, 0, ep_id=20)])

    assert matcher.matches(parsed_info(season=2, episodes=(0,))) is NoAnswer.SPECIAL_BY_TITLE


def test_an_episode_zero_is_matched_by_number_when_no_special_has_a_title() -> None:
    matcher = _matcher([sonarr_ep(0, 1, ep_id=1), sonarr_ep(2, 0, ep_id=20)])

    assert _ids(matcher.matches(parsed_info(season=2, episodes=(0,)))) == [20]


# --- AiredMatcher: no answer ---------------------------------------------------------


def test_two_episodes_on_one_absolute_number_are_sonarrs_failure() -> None:
    matcher = _matcher([sonarr_ep(1, 1, ep_id=11, absolute=1), sonarr_ep(2, 1, ep_id=21, absolute=1)])

    assert matcher.matches(parsed_info(season=0, absolutes=(1,))) is NoAnswer.SONARR_FAILS


def test_a_commented_alias_that_changes_the_match_leaves_no_answer() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11), sonarr_ep(2, 1, ep_id=21, scene_season=1, scene_episode=1)]
    matcher = _matcher(rows, _alias(tvdb=-1, origin="tvdb", comment="Some releases"))

    assert matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE)) is NoAnswer.LIMITED_ALIAS


def test_a_commented_alias_that_changes_nothing_leaves_the_answer() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11), sonarr_ep(2, 1, ep_id=21, scene_season=1, scene_episode=1)]
    matcher = _matcher(rows, _alias(tvdb=-1, origin="tvdb", comment="Some releases"))

    assert _ids(matcher.matches(parsed_info(season=2, episodes=(1,), series_title=_TITLE))) == [21]


def test_a_commented_alias_is_also_tried_as_the_only_alias_left() -> None:
    # Listed after a plain alias it never reads the name, unless its pattern passes and Sonarr drops the plain one.
    rows = [sonarr_ep(1, 1, ep_id=11), sonarr_ep(2, 1, ep_id=21, scene_season=1, scene_episode=1)]
    matcher = _matcher(rows, _alias(tvdb=2), _alias(tvdb=-1, origin="tvdb", comment="Some releases"))

    assert matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE)) is NoAnswer.LIMITED_ALIAS


def test_more_commented_aliases_than_are_tried_leave_no_answer() -> None:
    titles = ("Series Title", "Series-Title", "Series.Title", "Series: Title", "Series Title!")
    matcher = _matcher([sonarr_ep(1, 1, ep_id=11)], *(_alias(title, comment="Some releases") for title in titles))

    assert matcher.matches(parsed_info(season=1, episodes=(1,), series_title=_TITLE)) is NoAnswer.LIMITED_ALIAS
