# pyright: strict
"""Tests for a Sonarr title's check on a series by the order Sonarr numbers it by.

On a series Sonarr numbers by another order than the aired one, the check goes by the aired episodes, places a file
by what Sonarr's parses say of the series, and prints each episode by the number Sonarr lists. A series in aired
order says nothing of its numbers.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import pytest
from seadex import TorrentRecord

from pearlarr.arr_models import MatchedEpisode, ParsedFileInfo, SonarrEpisode, SonarrSeries
from pearlarr.file_paths import normalize_file_name
from pearlarr.mappings import TitleMapping
from pearlarr.output import Severity
from pearlarr.releases import EpisodeKey, PrintedKey, SeadexDict, SkipReason
from pearlarr.tvdb_orders import AIRED_ORDER, OrderRow
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    make_config,
    make_entry_record,
    make_release,
    make_run_deps,
    make_torrent_record,
    parsed_info,
    rg_group,
    sonarr_ep,
    stored_orders,
    stored_scene_mappings,
)
from .fakes import FakeSonarrClient, diagnostic_messages, install_recording_hub
from .title_check_builders import (
    DVD_E1,
    EP1,
    EP2,
    GRAB_FILES,
    GRAB_URL,
    SERIES_7,
    SERIES_7_DVD,
    FakeRunServices,
    aired_orders,
    anilist_counting,
    mapping_one_series,
    sonarr_checking,
    sonarr_grab_run,
    sonarr_listing,
)

_LISTED_EP1 = DVD_E1.renumbered(season=1, episode=1, absolute=1)
"""Series 7's first DVD row as its view holds it, under its aired number."""


def _check_title_in_another_order(
    rows: list[SonarrEpisode], *, grab: bool = True, ignore_unmonitored: bool = False
) -> FakeRunServices:
    """Drive a Sonarr `check_title` for title 5, all of series 7, over Sonarr's `rows` of it by its DVD order.

    The series is as `sonarr_listing` stores it. The entry lists one torrent of `GRAB_FILES`, which Sonarr parses
    as the season's first two episodes, and the plan grabs it unless `grab` is off.
    """

    release = make_release(url=GRAB_URL, infohash="h1", files=list(GRAB_FILES), size=[1000] * len(GRAB_FILES))
    run = FakeRunServices(
        prologue_entry=make_entry_record(),
        anilist_title="Show",
        seadex_dict={"NAN0": rg_group({GRAB_URL: release})},
        plan_marks=frozenset({GRAB_URL}) if grab else frozenset(),
        import_wait_mode=ImportWaitMode.BLOCKING,
    )
    sonarr = sonarr_listing(run.cache_store, SERIES_7_DVD, rows)
    sonarr.parse_fn = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(GRAB_FILES, start=1)}.get
    deps = make_run_deps(config=make_config(ignore_unmonitored=ignore_unmonitored), cache_store=run.cache_store)
    sonarr_checking(run, sonarr, deps).check_title(SERIES_7_DVD, 5, TitleMapping(anilist_id=5))
    return run


class TestSonarrCheckTitleOnASeriesInAnotherOrder:
    """A title of a series Sonarr numbers by another order is checked by its aired episodes.

    What Sonarr holds is read off the episodes it has a row for, and the title's window keeps every aired episode.
    """

    def test_a_title_whose_every_episode_in_sonarr_is_unmonitored_stops_as_unmonitored(self) -> None:
        unmonitored = DVD_E1.model_copy(update={"monitored": False})

        run = _check_title_in_another_order([unmonitored], ignore_unmonitored=True)

        assert (run.unmonitored_titles, run.log_al_title_calls) == (["Show"], [])

    def test_a_cached_rows_coverage_backfill_is_of_the_episodes_sonarr_has(self) -> None:
        run = _check_title_in_another_order([DVD_E1])

        (coverage,) = run.cached_skip_coverages
        assert coverage() == "S01 E01"

    def test_the_plan_is_made_over_the_episodes_sonarr_has(self) -> None:
        run = _check_title_in_another_order([DVD_E1])

        assert run.planned_title_episodes == [(_LISTED_EP1,)]

    @pytest.mark.parametrize(
        ("rows", "grab", "skip_reason", "left_out"),
        [
            pytest.param([DVD_E1], True, None, ["S01 E02 (1 of 2 files, NAN0)"], id="one-of-its-files"),
            pytest.param(
                [sonarr_ep(1, 3, ep_id=103, tvdb_id=9003, episode_file_id=0)],
                True,
                SkipReason.AIRED_ONLY,
                ["S01 E01-E02 (2 of 2 files, NAN0)"],
                id="its-every-file",
            ),
            pytest.param([DVD_E1], False, None, [], id="a-release-the-plan-does-not-grab"),
        ],
    )
    def test_a_grabs_files_on_episodes_sonarr_has_no_row_for_are_reported_in_one_row(
        self, rows: list[SonarrEpisode], grab: bool, skip_reason: SkipReason | None, left_out: list[str]
    ) -> None:
        run = _check_title_in_another_order(rows, grab=grab)

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[GRAB_URL].skip_reason is skip_reason
        assert run.left_out == left_out

    @pytest.mark.parametrize(
        ("orders_stored", "not_stored"),
        [
            pytest.param(False, "aired episodes", id="aired-episodes"),
            pytest.param(True, "scene numbers from TheXEM", id="scene-numbers"),
        ],
    )
    def test_a_series_with_no_aired_view_yet_is_left_unchecked_with_a_warning_of_what_is_not_stored(
        self, orders_stored: bool, not_stored: str
    ) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        if orders_stored:
            run.cache_store.put_tvdb_orders(SERIES_7_DVD.tvdbId, aired_orders())
        recording = install_recording_hub()
        strat = sonarr_checking(run, FakeSonarrClient(all_series=[SERIES_7_DVD]))

        strat.check_title(SERIES_7_DVD, 5, TitleMapping(anilist_id=5))

        (warning,) = diagnostic_messages(recording, Severity.WARNING)
        assert warning == (
            f"Show not checked - Sonarr numbers its series by the DVD order and the series' {not_stored} are not "
            "stored yet - checking again next run"
        )
        assert (run.no_episodes_titles, run.log_al_title_calls, run.pauses_owed) == ([], [], 1)
        assert (run.grab_requests, run.no_releases_calls, run.nothing_to_grab_calls) == ([], [], [])


class TestSonarrCheckTitleOnASeriesInAiredOrder:
    """A title of a series in aired order says nothing of its numbers, and nothing when its series can't be read."""

    def test_the_title_block_says_nothing_of_its_numbers(self) -> None:
        run = sonarr_grab_run([EP1, EP2])

        # The check ran through to its grab, past the title block and the row for files left out.
        assert len(run.grab_requests) == 1
        assert (run.numbers_orders, run.left_out) == ([], [])

    def test_a_series_whose_episodes_could_not_be_read_is_left_unchecked_quietly(self) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record())
        recording = install_recording_hub()
        sonarr = FakeSonarrClient(all_series=[SERIES_7])
        sonarr.episodes_return = None
        strat = sonarr_checking(run, sonarr)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))

        assert diagnostic_messages(recording) == []
        assert (run.no_episodes_titles, run.log_al_title_calls, run.pauses_owed) == ([], [], 0)
        assert (run.grab_requests, run.no_releases_calls) == ([], [])


@dataclass(frozen=True, slots=True, kw_only=True)
class _SeriesInSonarr:
    """Series 7 as Sonarr has it, and how Sonarr parses each file of the one torrent SeaDex lists for it."""

    series: SonarrSeries
    episodes: list[SonarrEpisode]
    parses: Mapping[str, ParsedFileInfo]
    orders: Mapping[str, tuple[OrderRow, ...]] | None = None
    """The TVDB orders stored for the series, with TheXEM read as mapping none of its episodes. None when nothing
    is stored."""


def _series_seven(season_type: str, alternate_title: dict[str, object]) -> SonarrSeries:
    """Sonarr's row of series 7 as the wire sends it, numbered by `season_type`, with its one alternate title."""

    return SonarrSeries.model_validate(
        {"id": 7, "title": "Show", "tvdbId": 70, "seasonType": season_type, "alternateTitles": [alternate_title]}
    )


def _sequel_in_dvd_order() -> _SeriesInSonarr:
    """A sequel's four files numbered as a first season, on a series Sonarr numbers by its DVD order.

    TVDB's aired order has two seasons, of three and four episodes. The DVD order is one season of the first six, so
    Sonarr lists the aired S02E01 to S02E03 as S01E04 to S01E06 and has no row for the aired S02E04. It reads
    "Show Zoku S01E0n" by the second season's name and matches nothing. An aired Sonarr matches the second season's
    episodes.
    """

    aired = (
        *(OrderRow(1, n, 9000 + n, n) for n in (1, 2, 3)),
        *(OrderRow(2, n, 9003 + n, 3 + n) for n in (1, 2, 3, 4)),
    )
    dvd = tuple(OrderRow(1, n, 9000 + n, None) for n in range(1, 7))
    names = [f"Show Zoku S01E{n:02d}.mkv" for n in (1, 2, 3, 4)]
    return _SeriesInSonarr(
        series=_series_seven("dvd", {"title": "Show Zoku", "sceneSeasonNumber": 2}),
        episodes=[sonarr_ep(1, n, ep_id=10 + n, tvdb_id=9000 + n, episode_file_id=0) for n in range(1, 7)],
        parses={
            name: ParsedFileInfo(season_number=1, episode_numbers=(n,), series_title="Show Zoku", series_id=7)
            for n, name in enumerate(names, start=1)
        },
        orders={"official": aired, "dvd": dvd},
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Title:
    """A title of series 7's second season, starting `offset` episodes into it.

    AniList counts its `episodes`, and has no answer for it when None. SeaDex lists one release of it, a lone file of
    `file_size` bytes, and has no entry for it when None.
    """

    offset: int = 0
    episodes: int | None = None
    file_size: int | None = None


def _check_second_season(library: _SeriesInSonarr, titles: Mapping[int, _Title] | None = None) -> FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 of series 7's second season through a real `SonarrStrategy`.

    The season's titles are `titles`, or title 5 alone over the whole season when None. The library read, the parse
    cache, each parse's matched episodes and the series' listings are the real ones. The plan grabs the torrent, whose
    nth file is n thousand bytes.
    """

    titles = titles or {5: _Title()}
    names = list(library.parses)
    sizes = [1000 * n for n in range(1, len(names) + 1)]
    release = make_release(url=GRAB_URL, infohash="h1", files=names, size=sizes)
    seadex: SeadexDict = {"NAN0": rg_group({GRAB_URL: release})}
    cache = FakeCacheStore()
    if library.orders is not None:
        cache.put_tvdb_orders(library.series.tvdbId, stored_orders(library.orders))
        cache.put_scene_mappings(library.series.tvdbId, stored_scene_mappings())
    mappings = {n: TitleMapping(anilist_id=n, tvdb_season=2, tvdb_epoffset=title.offset) for n, title in titles.items()}
    run = FakeRunServices(
        anilist_ids=mappings,
        prologue_entry=make_entry_record(),
        seadex_dict=seadex,
        plan_marks=frozenset({GRAB_URL}),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=cache,
    )
    entries = {
        n: make_entry_record(anilist_id=n, torrents=(_lone_file(n, title.file_size),))
        for n, title in titles.items()
        if title.file_size is not None
    }
    deps = mapping_one_series(
        make_run_deps(cache_store=cache, seadex=FakeSeaDexSource(entries)), tvdb_id=library.series.tvdbId, anilist_id=5
    )
    for n, title in titles.items():
        if title.episodes is not None:
            deps.anilist.answers[n] = anilist_counting(n, title.episodes)
    sonarr = FakeSonarrClient(all_series=[library.series], episodes=library.episodes, parse_fn=library.parses.get)
    sonarr_checking(run, sonarr, deps).check_title(library.series, 5, mappings[5])
    return run


def _lone_file(anilist_id: int, size: int) -> TorrentRecord:
    """A release of title `anilist_id` whose one file is `size` bytes."""

    return make_torrent_record(infohash=f"{anilist_id:040x}", file_names=("Show Zoku.mkv",), file_size=size)


class TestSonarrCheckTitleOverSonarrsParses:
    """A title's check places a file by what the library read says of its series: the order and the other names."""

    def test_a_sequels_files_numbered_as_a_first_season_land_on_the_aired_second_season(self) -> None:
        library = _sequel_in_dvd_order()
        first, second, third, fourth = (normalize_file_name(name) for name in library.parses)

        run = _check_second_season(library)

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert update.file_map == {first: (14,), second: (15,), third: (16,)}
        assert update.excluded == (fourth,)
        assert run.left_out == ["S02 E04 (1 of 4 files, NAN0)"]
        assert [(ep.aired_key, ep.printed) for ep in req.seadex_dict["NAN0"].releases[GRAB_URL].episodes] == [
            (EpisodeKey(2, n), PrintedKey(EpisodeKey(1, 3 + n))) for n in (1, 2, 3)
        ]

    def test_a_file_named_by_an_alternate_title_of_the_series_lands_by_its_number_in_the_season(self) -> None:
        # The group numbers the second season from one under the series' other name, and Sonarr matched the file
        # into the first season. The number is read within the season only because the run knows the name.
        name = "[grp] Shou - 01.mkv"
        library = _SeriesInSonarr(
            series=_series_seven(AIRED_ORDER, {"title": "Shou"}),
            episodes=[
                *(sonarr_ep(1, n, ep_id=10 + n, absolute=n, episode_file_id=0) for n in (1, 2, 3, 4)),
                *(sonarr_ep(2, n, ep_id=14 + n, absolute=4 + n, episode_file_id=0) for n in (1, 2, 3)),
            ],
            parses={
                name: ParsedFileInfo(
                    season_number=0,
                    absolute_episode_numbers=(1,),
                    series_title="Shou",
                    matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=11),),
                    series_id=7,
                )
            },
        )

        run = _check_second_season(library)

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == {normalize_file_name(name): (15,)}


class TestSonarrCheckTitlePrintsSonarrsNumbers:
    """On a series Sonarr numbers by another order, a title's lines print each episode by the number Sonarr lists."""

    def test_the_titles_rows_and_stored_coverage_are_in_sonarrs_numbers_under_the_named_order(self) -> None:
        run = _check_second_season(_sequel_in_dvd_order())

        (req,) = run.grab_requests
        claim = req.pending_import_updates["h1"].claim
        assert (run.title_block_coverages, run.missing_episodes) == (["S01 E04-E06"], ["S01 E04-E06"])
        assert (req.update.get("coverage"), claim.window_coverage, claim.claimed_coverage) == ("S01 E04-E06",) * 3
        assert run.numbers_orders == ["dvd"]

    def test_a_file_placed_by_size_over_its_name_is_warned_of_by_sonarrs_number(self) -> None:
        # Title 5 is the aired S02E02 alone, which Sonarr lists as S01E05, and SeaDex lists its one release at the
        # first file's size.
        recording = install_recording_hub()

        _check_second_season(_sequel_in_dvd_order(), {5: _Title(offset=1, episodes=1, file_size=1000)})

        (warning,) = diagnostic_messages(recording, Severity.WARNING)
        assert warning == (
            f"{GRAB_URL}: show zoku s01e01.mkv goes on S01E05 because its size matches SeaDex's S01E05 release "
            "(its name reads as S01E01)"
        )

    def test_a_second_file_placed_by_size_adds_a_more_files_note_to_the_warning(self) -> None:
        # Two files placed by size in one grab take a title with no episode count whose season also holds two
        # one-episode titles. Titles 6 and 7 are the aired S02E01 and S02E02 alone, and SeaDex lists their releases at
        # the second and first files' sizes.
        recording = install_recording_hub()
        one_episode = {6: _Title(episodes=1, file_size=2000), 7: _Title(offset=1, episodes=1, file_size=1000)}

        _check_second_season(_sequel_in_dvd_order(), {5: _Title(), **one_episode})

        (warning,) = diagnostic_messages(recording, Severity.WARNING)
        assert warning.endswith("(its name reads as S01E01) (1 more file in the debug log)")
