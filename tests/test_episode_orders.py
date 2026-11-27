# pyright: strict
# pyright: reportPrivateUsage=false
# The tests read the strategy's private collaborators (_episodes / _parse / _reads), which strict re-flags.
# The repo disables reportPrivateUsage for tests.
"""Each series' TVDB orders through a run's reads and the Sonarr strategy: what is fetched, what grab and import get.

The grab is served a file's parse too, matched as an aired Sonarr matches it.

A test of the reads alone builds them alone (`_reads`). One that asks for a title's episodes, a fetch, the parse
cache, the import or the run's start builds the strategy (`_sync`), whose collaborators share one read of each
series. A whole run is driven through `drive_arr_run` (`_run`).
"""

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from typing import override

import pytest
from seadex import EntryRecord

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_models import (
    AlternateTitle,
    MatchedEpisode,
    ParsedFileInfo,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
)
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.mappings import AnimeIdsByName, TitleMapping
from pearlarr.output import BootStepFinished, Severity, TitleHeader, TitleRow
from pearlarr.output.events import TitleState
from pearlarr.parse_cache import parse_to_json
from pearlarr.pending_import import PendingImport, TvdbIds
from pearlarr.releases import EpisodeKey, SeadexDict
from pearlarr.scene_numbers import SceneMapping
from pearlarr.series_reads import SeriesReads
from pearlarr.skyhook_client import SkyHookReply
from pearlarr.sonarr_episodes import SonarrEpisodes, sonarr_series_hash
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import OrderRow, SeriesNumbering
from pearlarr.wait_types import NO_PROGRESS, ImportProgress, ImportWaitMode

from .builders import (
    PENDING_IMPORT_ANILIST_ID,
    PENDING_IMPORT_STAMP,
    FakeCacheStore,
    FakeSeaDexSource,
    anilist_body,
    drive_arr_run,
    make_config,
    make_entry_record,
    make_parse_cache,
    make_pending_follower,
    make_release,
    make_run_deps,
    make_series_reads,
    make_sonarr_episodes,
    pending_import,
    rg_group,
    sonarr_ep,
    stored_orders,
    stored_scene_mappings,
    two_claim_pending_import,
)
from .fakes import (
    FakeSkyHookClient,
    FakeSonarrClient,
    FakeXemClient,
    RecordingHub,
    diagnostic_messages,
    install_recording_hub,
)
from .title_check_builders import FakeRunServices, sonarr_checking
from .wait_fakes import FakeQbit

_DVD = (OrderRow(1, 1, 9002, None), OrderRow(1, 2, 9001, None))
"""The dvd order of a two-episode season that swaps its episodes."""

_ORDERS = {"official": (OrderRow(1, 1, 9001, 1), OrderRow(1, 2, 9002, 2)), "dvd": _DVD}
"""A stored answer's orders: official and `_DVD`."""

_LONGER_AIRED = {"official": (*_ORDERS["official"], OrderRow(1, 3, 9003, 3)), "dvd": _DVD}
"""A stored answer whose official order has a third episode `_DVD` lacks."""

_ROWS = [sonarr_ep(1, 1, ep_id=11, tvdb_id=9001), sonarr_ep(1, 2, ep_id=12, tvdb_id=9002)]
"""Sonarr's rows of the season, numbered as aired."""

_DVD_ROWS = [sonarr_ep(1, 1, ep_id=11, tvdb_id=9002), sonarr_ep(1, 2, ep_id=12, tvdb_id=9001)]
"""Sonarr's rows of the season, numbered by `_DVD`."""

_SEASON_ONE = TitleMapping(anilist_id=500, tvdb_season=1)
"""A title mapped to the whole of season 1."""


def _series(
    series_id: int, tvdb_id: int, *, season_type: str | None = "official", counts: Mapping[str, int] | None = None
) -> SonarrSeries:
    """A Sonarr series row as the wire sends it, listing `counts` as v5 does (none, as on v4, when None).

    With `season_type=None` the row has no `seasonType` key, as a Sonarr that sends no order has it.
    """

    raw: dict[str, object] = {"id": series_id, "title": f"Show {series_id}", "tvdbId": tvdb_id}
    if season_type is not None:
        raw["seasonType"] = season_type
    if counts is not None:
        raw["seasonTypes"] = [{"type": name, "episodeCount": count} for name, count in counts.items()]
    return SonarrSeries.model_validate(raw)


_MAPPED: AnimeIdsByName = {"Show 1": {"anilist_id": 500, "tvdb_id": 50}}
"""The AniList mapping of the strategies and episode readers here: title 500 is of TVDB series 50, and no other series has one."""


def _reads(
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
) -> SeriesReads:
    """A run's reads after its library read of `series`, with no strategy around them.

    The series and their rows are as `_sync` has them. TVDB orders are fetched through `client`, and a series has
    the scene mappings `cache` holds.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ROWS))
    return make_series_reads(sonarr, cache, library=series, skyhook_client=client)


def _episodes(
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
) -> SonarrEpisodes:
    """A run's episode reader after its library read of `series`, built as the strategy builds it.

    It takes what `_reads` takes, and its reads have the series, rows, orders and scene mappings `_reads` gives.
    The titles are mapped as `_MAPPED` has them.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ROWS))
    sonarr.all_series_return = series
    deps = make_run_deps(cache_store=cache, skyhook_client=client, anime_ids=_MAPPED)
    episodes = make_sonarr_episodes(deps, sonarr=sonarr)
    episodes.refresh_library()
    return episodes


def _sync(
    series: list[SonarrItem],
    cache: FakeCacheStore,
    *,
    sonarr: FakeSonarrClient | None = None,
    waits: ImportWaitMode = ImportWaitMode.OFF,
) -> SonarrStrategy:
    """The Sonarr strategy after the Sonarr run's library fetch of `series`, with SkyHook holding no series.

    The titles are mapped as `_MAPPED` has them. Every series' episodes are `_ROWS` unless `sonarr` serves others.
    Its AniList knows no title, a series has the scene mappings `cache` holds, and the run waits on imports as
    `waits` says.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ROWS))
    sonarr.all_series_return = series
    deps = make_run_deps(cache_store=cache, anime_ids=_MAPPED)
    services = ArrRunServices(deps, Arr.SONARR)
    services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=waits))
    strat = SonarrStrategy(deps, services, sonarr_client=sonarr)
    strat.get_items()
    return strat


def _stored(orders: Mapping[str, tuple[OrderRow, ...]]) -> FakeCacheStore:
    """A cache holding `orders` for TVDB series 50, and TheXEM's answer for it, which maps none of its episodes."""

    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(orders))
    cache.put_scene_mappings(50, stored_scene_mappings())
    return cache


def _numbered(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, EpisodeKey]]:
    """Each episode's id and its `(season, episode)`, in the order given."""

    return [(ep.id, EpisodeKey(ep.season_number or 0, ep.episode_number or 0)) for ep in episodes]


@pytest.mark.parametrize(
    ("series_id", "season_type", "other_orders"),
    [
        pytest.param(1, "official", {"dvd"}, id="official"),
        # A null or junk `seasonType` reads as "", which names no order.
        pytest.param(1, "", set[str](), id="no-order"),
        # The library read has series 1 alone.
        pytest.param(2, "official", set[str](), id="not-in-the-library"),
    ],
)
def test_a_series_not_numbered_by_another_order_is_served_sonarrs_rows_and_given_other_orders_only_if_it_names_one(
    series_id: int, season_type: str, other_orders: set[str]
) -> None:
    # The stored aired order has a third episode Sonarr has no row for, and none is stood in for it.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    reads = _reads([_series(1, 50, season_type=season_type)], cache)

    view = reads.aired_series(series_id)
    index = reads.series_index(series_id)

    assert view == AiredSeries(in_sonarr=tuple(_ROWS), stand_ins=())
    assert index is not None
    assert (set(index.by_id), set(index.orders)) == ({11, 12}, other_orders)


def test_a_series_numbered_by_another_order_is_served_as_aired_with_its_stored_orders_read_onto_it() -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], _stored(_LONGER_AIRED), sonarr=sonarr)

    view = reads.aired_series(1)
    index = reads.series_index(1)

    assert isinstance(view, AiredSeries)
    assert _numbered(view.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert index is not None
    assert index.id_by_key == {(1, 1): 12, (1, 2): 11, (1, 3): -9003}
    assert {name: order.id_by_key for name, order in index.orders.items()} == {"dvd": {(1, 1): 11, (1, 2): 12}}


def test_a_titles_episodes_of_a_series_numbered_by_another_order_are_its_aired_ones() -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    episodes = _episodes([_series(1, 50, season_type="dvd")], _stored(_LONGER_AIRED), sonarr=sonarr)

    title = episodes.title_episodes(1, 500, _SEASON_ONE)

    assert not isinstance(title, NoView)
    assert _numbered(title.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert _numbered(title.in_sonarr) == [(12, (1, 1)), (11, (1, 2))]
    assert title.index().by_id[-9003].absolute_episode_number == 3


def test_a_series_numbered_by_another_order_with_nothing_stored_has_no_view() -> None:
    reads = _reads([_series(1, 50, season_type="dvd")], FakeCacheStore())

    assert reads.aired_series(1) is NoView.ORDERS_UNREAD
    assert reads.series_index(1) is None


def _scene(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, tuple[int | None, int | None, int | None]]]:
    """Each episode's id and its scene season, episode and absolute number, in the order given."""

    return [
        (ep.id, (ep.scene_season_number, ep.scene_episode_number, ep.scene_absolute_episode_number)) for ep in episodes
    ]


_SCENE_ROWS = [
    sonarr_ep(1, 1, ep_id=11, tvdb_id=9002, scene_season=9, scene_episode=1, scene_absolute=91),
    sonarr_ep(1, 2, ep_id=12, tvdb_id=9001, scene_season=9, scene_episode=2, scene_absolute=92),
]
"""`_DVD_ROWS`, each with scene numbers Sonarr left on it from before it numbered the series by `_DVD`."""


def test_a_series_numbered_by_another_order_whose_scene_mappings_were_never_read_has_no_view() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    reads = _reads([_series(1, 50, season_type="dvd")], cache)

    assert reads.aired_series(1) is NoView.SCENE_UNREAD
    assert reads.series_index(1) is None


def test_a_series_numbered_by_another_order_that_thexem_has_no_mappings_for_has_a_view_without_scene_numbers() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_SCENE_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _scene(view.aired) == [(12, (None, None, None)), (11, (None, None, None)), (-9003, (None, None, None))]


def test_a_series_numbered_by_another_order_carries_the_scene_numbers_its_mappings_give_its_aired_episodes() -> None:
    # TheXEM maps the first aired episode, and the two after it are numbered on from it. The third is a stand-in.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings((SceneMapping(1, 1, 2, 1, 13),)))
    sonarr = FakeSonarrClient(episodes=list(_SCENE_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _scene(view.aired) == [(12, (2, 1, 13)), (11, (2, 2, 14)), (-9003, (2, 3, 15))]


def test_a_series_in_aired_order_keeps_the_scene_numbers_sonarr_sent_whatever_thexem_maps() -> None:
    rows = [sonarr_ep(1, 1, ep_id=11, scene_season=9, scene_episode=1, scene_absolute=91)]
    cache = FakeCacheStore()
    cache.put_scene_mappings(50, stored_scene_mappings((SceneMapping(1, 1, 2, 1, 13),)))
    reads = _reads([_series(1, 50)], cache, sonarr=FakeSonarrClient(episodes=rows))

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _scene(view.aired) == [(11, (9, 1, 91))]


def test_a_new_read_of_a_series_replaces_its_view_and_index() -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    before = reads.series_index(1)
    third = sonarr_ep(1, 3, ep_id=13)
    sonarr.episodes_return = [*_ROWS, third]

    reads.read_episodes(1)

    view = reads.aired_series(1)
    index = reads.series_index(1)
    assert view == AiredSeries(in_sonarr=(*_ROWS, third), stand_ins=())
    assert before is not None and index is not None
    assert (set(before.by_id), set(index.by_id)) == ({11, 12}, {11, 12, 13})


def test_a_series_the_library_read_has_in_another_order_is_renumbered_whatever_numbers_its_rows_carry() -> None:
    # Sonarr's rows are numbered as aired, not by the dvd order the library read names.
    reads = _reads([_series(1, 50, season_type="dvd")], _stored(_LONGER_AIRED))

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _numbered(view.aired) == [(11, (1, 1)), (12, (1, 2)), (-9003, (1, 3))]


@pytest.mark.parametrize(
    "library",
    [pytest.param([_series(1, 50)], id="in-the-library"), pytest.param([_series(2, 60)], id="not-in-the-library")],
)
def test_a_series_sonarr_sends_no_episodes_of_has_none(library: list[SonarrItem]) -> None:
    recording = install_recording_hub()
    reads = _reads(library, FakeCacheStore(), sonarr=FakeSonarrClient(episodes=[]))

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=(), stand_ins=())
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_the_series_hash_of_a_run_is_of_the_series_with_an_anilist_mapping() -> None:
    sonarr = FakeSonarrClient(all_series=[_series(1, 50), _series(2, 60)])
    deps = make_run_deps(anime_ids=_MAPPED)
    reads = SeriesReads(sonarr, deps.skyhook, deps.xem)

    make_sonarr_episodes(deps, sonarr=sonarr, reads=reads).refresh_library()

    assert reads.series_hash == sonarr_series_hash([1])


def test_the_scene_mappings_of_a_pending_imports_series_are_read_for_the_one_in_another_order() -> None:
    # One pending import claimed on both series. Only series 2, of TVDB series 60, is in another order.
    pending = two_claim_pending_import(series_ids=(1, 2))
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat = _sync([_series(1, 50), _series(2, 60, season_type="dvd")], cache, waits=ImportWaitMode.BLOCKING)
    strat.start_of_run([])

    read = strat.prefetch_scene_numbers([])

    assert read == 1
    assert (cache.get_scene_mappings(50), cache.get_scene_mappings(60) is not None) == (None, True)


_OUT_OF_STEP = (
    "Show 1: Sonarr's episodes of the series are not numbered by the order Sonarr names for it - holding off on "
    "the series until they are - refresh it in Sonarr"
)
"""The warning for series 1 when its rows aren't numbered by its order as SkyHook gave it this run."""

_UNCONFIRMED = (
    "Show 1: Sonarr's episodes of the series may not be numbered by the order Sonarr names for it, and SkyHook "
    "could not be reached to make sure - holding off on the series until it can"
)
"""The warning for series 1 when its rows aren't numbered by its stored order and SkyHook gave no new one."""

_SWAPPED_SINCE = {"official": _DVD}
"""TVDB series 50's orders once TVDB has swapped the two aired episodes: the aired order numbers them as `_DVD`."""


def _skyhook(
    orders: Mapping[str, tuple[OrderRow, ...]], *, failing: frozenset[tuple[int, str]] = frozenset()
) -> FakeSkyHookClient:
    """A SkyHook that has TVDB series 50 with `orders`, each reply as the wire sends it, and the `failing` fetches."""

    listed = [{"type": name, "episodeCount": len(rows)} for name, rows in orders.items()]
    return FakeSkyHookClient(
        {
            (50, name): SkyHookReply.model_validate(
                {
                    "tvdbId": 50,
                    "status": "Ended",
                    "seasonTypes": listed,
                    "episodes": [
                        {
                            "tvdbId": row.tvdb_id,
                            "seasonNumber": row.season,
                            "episodeNumber": row.episode,
                            "absoluteEpisodeNumber": row.absolute,
                        }
                        for row in rows
                    ],
                }
            )
            for name, rows in orders.items()
        },
        failing=failing,
    )


def _aired_over_dvd_rows(client: FakeSkyHookClient) -> SeriesReads:
    """A run's reads after a library read of series 1 as aired, on a Sonarr whose rows of it are `_DVD_ROWS`.

    An earlier run stored `_ORDERS` for TVDB series 50. SkyHook is `client`.
    """

    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    return _reads([_series(1, 50)], cache, client, sonarr=FakeSonarrClient(episodes=list(_DVD_ROWS)))


@pytest.mark.parametrize(
    ("client", "warning"),
    [
        pytest.param(_skyhook(_ORDERS), _OUT_OF_STEP, id="still-out-of-step"),
        # SkyHook doesn't have the series any more, so the new answer has no aired order to clear the rows by.
        pytest.param(FakeSkyHookClient(), _OUT_OF_STEP, id="no-rows-for-the-order"),
        pytest.param(FakeSkyHookClient(failing=frozenset({(50, "official")})), _UNCONFIRMED, id="skyhook-unreachable"),
    ],
)
def test_an_aired_series_over_rows_numbered_by_another_order_has_no_view_with_a_warning(
    client: FakeSkyHookClient, warning: str
) -> None:
    recording = install_recording_hub()
    reads = _aired_over_dvd_rows(client)

    view = reads.aired_series(1)

    assert view is NoView.READ_FAILED
    assert diagnostic_messages(recording, Severity.WARNING) == [warning]


def test_rows_out_of_step_with_an_answer_from_an_earlier_run_are_tested_again_by_one_fetched_now() -> None:
    # TVDB swapped the two episodes since the answer was stored, and Sonarr's rows already follow it.
    recording = install_recording_hub()
    reads = _aired_over_dvd_rows(_skyhook(_SWAPPED_SINCE))

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_rows_of_a_series_with_no_stored_answer_are_read_as_they_come() -> None:
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=FakeSonarrClient(episodes=list(_DVD_ROWS)))

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())


def test_a_series_with_no_view_for_the_rest_of_the_run_is_unread_wherever_a_release_needs_it() -> None:
    # Series 1's rows are out of step with its stored orders. A torrent with a title of it has no index of it to be
    # placed by, and a file name Sonarr matched to it has no aired match.
    reads = _aired_over_dvd_rows(_skyhook(_ORDERS))

    assert (reads.series_index(1), reads.aired_parse(ParsedFileInfo(series_id=1))) == (None, NoView.READ_FAILED)


def test_the_import_is_not_given_rows_out_of_step_with_the_order_the_library_read_names() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], cache, _skyhook(_ORDERS), sonarr=sonarr)
    before = reads.read_episodes(1)
    sonarr.episodes_return = list(_DVD_ROWS)

    refused = reads.read_episodes(1)

    assert before == _ROWS
    assert refused is None
    assert reads.last_known_episodes(1) == _ROWS


_ON_THE_FIRST = pending_import(
    infohash="h",
    series_id=1,
    file_episode_map={"show - 01 [1080p].mkv": (11,)},
    ordered_episode_ids=[11],
    tvdb_ids={11: 9001},
)
"""A pending import on series 1, its one file on episode id 11, which was TVDB episode 9001 when it was stored."""


def test_the_import_and_the_check_of_a_run_hold_off_on_rows_out_of_step_with_one_warning() -> None:
    recording = install_recording_hub()
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    run = FakeRunServices(prologue_entry=make_entry_record(), cache_store=cache)
    sonarr = FakeSonarrClient(all_series=[_series(1, 50)], episodes=list(_DVD_ROWS))
    deps = make_run_deps(cache_store=cache, skyhook_client=_skyhook(_ORDERS), anime_ids=_MAPPED)
    strat = sonarr_checking(run, sonarr, deps)

    progress = strat.import_progress(_ON_THE_FIRST)
    strat.check_title(_series(1, 50), 500, _SEASON_ONE)

    assert (progress, run.log_al_title_calls) == (NO_PROGRESS, [])
    assert diagnostic_messages(recording, Severity.WARNING) == [_OUT_OF_STEP]


def test_a_series_with_no_view_for_the_rest_of_the_run_keeps_none_once_the_import_reads_its_rows_in_step() -> None:
    # Sonarr renumbered the series as aired part way through the run. Its titles checked from here on are left for
    # the next run, which reads the series afresh.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50)], cache, _skyhook(_ORDERS), sonarr=sonarr)
    reads.aired_series(1)
    sonarr.episodes_return = list(_ROWS)

    in_step = reads.read_episodes(1)

    assert (in_step, reads.aired_series(1)) == (_ROWS, NoView.READ_FAILED)


def test_a_sonarr_that_sends_no_order_has_its_rows_taken_as_they_come() -> None:
    # The stored aired order numbers the two episodes the other way round. Nothing is tested against it.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50, season_type=None)], cache, _skyhook(_ORDERS), sonarr=sonarr)

    rows = reads.read_episodes(1)
    view = reads.aired_series(1)

    assert rows == _DVD_ROWS
    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())


def test_the_import_and_the_follower_read_a_series_with_an_aired_view_as_sonarr_numbers_it() -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], _stored(_ORDERS), sonarr=sonarr)

    assert (reads.read_episodes(1), reads.viewed_episodes(1)) == (_DVD_ROWS, _DVD_ROWS)


def test_the_import_goes_ahead_on_a_series_in_another_order_that_has_no_view() -> None:
    # No orders are stored for the series. The import posts Sonarr's own episode ids, so it needs no view.
    strat = _sync([_series(1, 50, season_type="dvd")], FakeCacheStore())

    assert strat.import_progress(_ON_THE_FIRST) == ImportProgress(0, 1, determinate=True)


def test_a_series_with_no_view_has_no_rows_a_view_is_built_from() -> None:
    # Sonarr numbers the series by its dvd order, and no TVDB orders are stored for it.
    reads = _reads([_series(1, 50, season_type="dvd")], FakeCacheStore(), sonarr=FakeSonarrClient(episodes=_DVD_ROWS))

    assert reads.viewed_episodes(1) is None
    assert reads.last_known_episodes(1) == _DVD_ROWS


@pytest.mark.parametrize(
    ("series_id", "unread"),
    [
        pytest.param(1, False, id="stored"),
        pytest.param(2, True, id="no-row"),
        pytest.param(3, False, id="no-tvdb-id"),
        pytest.param(4, False, id="not-in-the-library"),
    ],
)
def test_a_library_series_with_no_stored_answer_is_unread(series_id: int, unread: bool) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    episodes = _episodes([_series(1, 50), _series(2, 60), _series(3, 0)], cache)

    assert episodes.metadata_unread([], [series_id]) is unread


@pytest.mark.parametrize(
    ("season_type", "unread"),
    [pytest.param("dvd", True, id="another-order"), pytest.param("official", False, id="aired-order")],
)
def test_a_series_whose_scene_mappings_were_never_read_is_unread_only_in_another_order(
    season_type: str, unread: bool
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    episodes = _episodes([_series(1, 50, season_type=season_type)], cache)

    assert episodes.metadata_unread([], [1]) is unread


def test_a_series_in_another_order_is_read_once_thexem_was_read_for_it_even_with_no_mappings() -> None:
    episodes = _episodes([_series(1, 50, season_type="dvd")], _stored(_ORDERS))

    assert episodes.metadata_unread([], [1]) is False


@pytest.mark.parametrize(
    ("season_type", "client", "unread"),
    [
        pytest.param("dvd", FakeSkyHookClient(failing=frozenset({(50, "official")})), True, id="another-order"),
        pytest.param("official", FakeSkyHookClient(failing=frozenset({(50, "official")})), False, id="aired-order"),
        pytest.param("dvd", _skyhook(_ORDERS), False, id="another-order-and-fetched"),
    ],
)
def test_a_series_in_another_order_is_unread_while_a_fetch_its_stored_orders_were_due_is_missed(
    season_type: str, client: FakeSkyHookClient, unread: bool
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, fresh_for=timedelta(days=-1)))
    cache.put_scene_mappings(50, stored_scene_mappings())
    episodes = _episodes([_series(1, 50, season_type=season_type)], cache, client)

    episodes.prefetch_orders([1])

    assert (episodes.metadata_unread([], [1]), episodes.title_unread(500, 1)) == (unread, unread)


def test_orders_are_fetched_once_per_tvdb_series() -> None:
    # Two Sonarr series of one TVDB series, one with no TVDB id, and one the Sonarr run's library fetch lacks.
    cache = FakeCacheStore()
    episodes = _episodes([_series(1, 50), _series(2, 50), _series(3, 0)], cache)

    assert episodes.prefetch_orders([1, 2, 3, 9]) == 1
    assert cache.stats().tvdb_orders == 1


def test_a_fresh_stored_answer_is_fetched_again_once_sonarr_lists_another_episode_count() -> None:
    # The stored answer is fresh, so only the count Sonarr's series row carries can make it due.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2}))
    episodes = _episodes([_series(1, 50, counts={"official": 3})], cache)

    assert episodes.prefetch_orders([1]) == 1


@pytest.mark.parametrize(
    ("series_id", "mode", "fetched"),
    [
        pytest.param(1, ImportWaitMode.BLOCKING, 1, id="waiting"),
        pytest.param(2, ImportWaitMode.BLOCKING, 1, id="waiting-on-a-series-with-no-anilist-mapping"),
        pytest.param(1, ImportWaitMode.OFF, 0, id="off"),
    ],
)
def test_a_pending_imports_series_is_fetched_while_imports_are_waited_on(
    series_id: int, mode: ImportWaitMode, fetched: int
) -> None:
    # Nothing in the scan needs a series, but grab and import read the pending import's series anyway.
    pending = pending_import(series_id=series_id)
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat = _sync([_series(1, 50), _series(2, 60)], cache, waits=mode)
    strat.start_of_run([])

    assert strat.prefetch_orders([]) == fetched


_TWO_SEASONS = (
    OrderRow(0, 1, 9100, None),
    *(OrderRow(1, n, 9000 + n, n) for n in (1, 2, 3)),
    *(OrderRow(2, n, 9003 + n, 3 + n) for n in (1, 2, 3, 4)),
)
"""The aired order of a series with a special and two seasons, of three and four episodes."""

_AS_ONE_SEASON = (OrderRow(0, 1, 9100, None), *(OrderRow(1, n, 9000 + n, n) for n in range(1, 7)))
"""Its dvd order: the special, and the first six episodes as one season. It lacks the seventh."""

_ONE_SEASON_ROWS = [
    sonarr_ep(0, 1, ep_id=20, tvdb_id=9100, title="Recap"),
    *(sonarr_ep(1, n, ep_id=10 + n, tvdb_id=9000 + n, absolute=n) for n in range(1, 7)),
]
"""Sonarr's rows of that series, numbered by `_AS_ONE_SEASON`."""

_ZOKU = AlternateTitle(title="Show Zoku", scene_season_number=2)
"""The series' name for its second season."""


def _matched(season: int, episode: int, ep_id: int) -> MatchedEpisode:
    return MatchedEpisode(season_number=season, episode_number=episode, id=ep_id)


def _stored_as_one_season(aired: tuple[OrderRow, ...] = _TWO_SEASONS) -> FakeCacheStore:
    """`_stored` with TVDB series 50's orders: `aired`, and `_AS_ONE_SEASON` as its dvd order."""

    return _stored({"official": aired, "dvd": _AS_ONE_SEASON})


def _in_one_season(*alternate_titles: AlternateTitle) -> SonarrSeries:
    """Series 1 of TVDB series 50, numbered by its dvd order, with its other names in Sonarr."""

    return _series(1, 50, season_type="dvd").model_copy(update={"alternateTitles": alternate_titles})


def _reads_in_one_season(sonarr: FakeSonarrClient | None = None, series: SonarrSeries | None = None) -> SeriesReads:
    """A run's reads after a library read of series 1, numbered by `_AS_ONE_SEASON`, before anything reads its rows.

    Sonarr's rows are `_ONE_SEASON_ROWS` unless `sonarr` serves others, and the series has no alternate title
    unless `series` has.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    return _reads([series or _in_one_season()], _stored_as_one_season(), sonarr=sonarr)


def _parse_cache_in_one_season(sonarr: FakeSonarrClient, series: SonarrSeries | None = None) -> SonarrParseCache:
    """The parse cache over the library read and the stored orders `_reads_in_one_season` has, Sonarr being `sonarr`."""

    return make_parse_cache(sonarr, _stored_as_one_season(), library=[series or _in_one_season()])


def _parsing(parses: Mapping[str, ParsedFileInfo]) -> FakeSonarrClient:
    """A Sonarr with `_ONE_SEASON_ROWS` that answers each file name with its parse."""

    return FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS), parse_fn=parses.get)


def _listing(*names: str) -> SeadexDict:
    """One release listing the files, each of 100 bytes."""

    url = "https://nyaa.si/1"
    return {"Group": rg_group({url: make_release(url=url, files=list(names), size=[100] * len(names))})}


_FIFTH = "Show - 05.mkv"

_FIFTH_BY_SONARR = ParsedFileInfo(
    season_number=0,
    absolute_episode_numbers=(5,),
    series_title="Show",
    matched_episodes=(_matched(1, 5, 15),),
    series_id=1,
)
"""Sonarr's own parse of `_FIFTH`: the fifth episode of its one season."""


def test_a_name_with_only_an_absolute_number_is_matched_to_the_aired_episode_with_that_number() -> None:
    # Sonarr counts the fifth episode as S01E05. As aired it is the second season's second.
    reads = _reads_in_one_season()

    assert reads.aired_parse(_FIFTH_BY_SONARR) == _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))


def test_a_name_with_a_season_and_episode_is_matched_to_the_aired_episode_numbered_so() -> None:
    # Sonarr's order has no second season, so it matched the name to nothing.
    reads = _reads_in_one_season()
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 2, 15),))


def test_an_episode_sonarr_matched_by_its_own_order_is_not_passed_on() -> None:
    # S01E05 is an episode of Sonarr's one season. The aired first season has three.
    reads = _reads_in_one_season()
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(5,),
        matched_episodes=(_matched(1, 5, 15),),
        series_id=1,
    )

    assert reads.aired_parse(info) == info.with_matched(())


def test_a_name_is_matched_to_an_aired_episode_sonarrs_order_has_no_row_for() -> None:
    reads = _reads_in_one_season()
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 4, -9007),))


def test_a_name_under_an_alternate_title_of_a_season_is_matched_within_that_season() -> None:
    # Read by the series' own name, the third episode is S01E03.
    reads = _reads_in_one_season(series=_in_one_season(_ZOKU))
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(3,), series_title="Show Zoku", series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 3, 16),))


@pytest.mark.parametrize(
    "info",
    [
        pytest.param(_FIFTH_BY_SONARR.model_copy(update={"series_id": 2}), id="aired-order"),
        pytest.param(_FIFTH_BY_SONARR.model_copy(update={"series_id": 9}), id="not-in-the-library"),
        pytest.param(_FIFTH_BY_SONARR.model_copy(update={"series_id": None, "matched_episodes": ()}), id="no-series"),
    ],
)
def test_a_parse_not_of_a_series_in_another_order_comes_back_as_it_is(info: ParsedFileInfo) -> None:
    # Of series 1, which is in another order, the parse would be matched again as aired.
    reads = _reads([_in_one_season(), _series(2, 60)], _stored_as_one_season())

    assert reads.aired_parse(info) == info


def test_a_new_read_of_a_series_matches_a_name_to_its_new_rows() -> None:
    # An import poll re-reads the series after Sonarr gains a row for the seventh episode, where the stand-in was.
    sonarr = FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    reads = _reads_in_one_season(sonarr)
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_id=1)
    before = reads.aired_parse(info)
    sonarr.episodes_return = [*_ONE_SEASON_ROWS, sonarr_ep(1, 7, ep_id=17, tvdb_id=9007)]

    reads.read_episodes(1)

    assert before == info.with_matched((_matched(2, 4, -9007),))
    assert reads.aired_parse(info) == info.with_matched((_matched(2, 4, 17),))


def test_a_listed_file_of_a_series_in_another_order_carries_its_aired_matches() -> None:
    # Nothing in the run has read the series yet.
    parse_cache = _parse_cache_in_one_season(_parsing({_FIFTH: _FIFTH_BY_SONARR}))

    ((file,),) = parse_cache.parsed_files(_listing(_FIFTH)).values()

    assert file.parse == _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))


def test_the_parse_cache_keeps_sonarrs_own_answer_for_a_series_in_another_order() -> None:
    cache = _stored_as_one_season()
    parse_cache = make_parse_cache(_parsing({_FIFTH: _FIFTH_BY_SONARR}), cache, library=[_in_one_season()])

    parse_cache.parsed_files(_listing(_FIFTH))

    row = cache.get_sonarr_parse(_FIFTH)
    assert row is not None
    assert row["parse"] == parse_to_json(_FIFTH_BY_SONARR)


def test_a_cached_parse_is_matched_as_aired_again_without_asking_sonarr() -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR})
    parse_cache = _parse_cache_in_one_season(sonarr)
    parse_cache.parsed_files(_listing(_FIFTH))

    ((again,),) = parse_cache.parsed_files(_listing(_FIFTH)).values()

    assert again.parse == _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))
    assert sonarr.parse_calls == [_FIFTH]


_FIFTH_ON_NO_EPISODE = _FIFTH_BY_SONARR.with_matched(())
"""Sonarr's parse of `_FIFTH` when it matches the name to the series and to no episode of it."""


def _asked_to_parse(cache: FakeCacheStore, series: SonarrSeries, by_sonarr: ParsedFileInfo) -> list[str]:
    """The names one run asks Sonarr to parse for a release listing `_FIFTH`, over the parses `cache` holds.

    The run's library read has `series` alone, and Sonarr answers `_FIFTH` with `by_sonarr`.
    """

    sonarr = _parsing({_FIFTH: by_sonarr})
    make_parse_cache(sonarr, cache, library=[series]).parsed_files(_listing(_FIFTH))
    return sonarr.parse_calls


# A parse with no matched episode is stored under a freshness rule of its own, so each change runs over both kinds.
@pytest.mark.parametrize(
    "by_sonarr",
    [
        pytest.param(_FIFTH_BY_SONARR, id="matched-episodes"),
        pytest.param(_FIFTH_ON_NO_EPISODE, id="no-matched-episode"),
    ],
)
@pytest.mark.parametrize(
    ("before", "now", "asked"),
    [
        pytest.param(_in_one_season(), _series(1, 50), [_FIFTH], id="another-order-then-aired"),
        pytest.param(_series(1, 50), _in_one_season(), [_FIFTH], id="aired-then-another-order"),
        pytest.param(_in_one_season(), _in_one_season(), [], id="another-order-on-both-runs"),
    ],
)
def test_a_stored_parse_is_asked_for_again_once_its_series_is_in_another_order_than_it_was_stored_under(
    before: SonarrSeries, now: SonarrSeries, by_sonarr: ParsedFileInfo, asked: list[str]
) -> None:
    cache = _stored_as_one_season()
    _asked_to_parse(cache, before, by_sonarr)

    assert _asked_to_parse(cache, now, by_sonarr) == asked


_SIXTH = "Show - 06.mkv"


def _with_two_sixth_episodes() -> SonarrParseCache:
    """`_parse_cache_in_one_season` where TVDB's aired order counts the special as a sixth episode too.

    Two aired episodes carry the number, and Sonarr answers a name it reads by such a number with an error.
    """

    by_sonarr = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(6,),
        series_title="Show",
        matched_episodes=(_matched(1, 6, 16),),
        series_id=1,
    )
    cache = _stored_as_one_season((OrderRow(0, 1, 9100, 6), *_TWO_SEASONS[1:]))
    return make_parse_cache(_parsing({_SIXTH: by_sonarr}), cache, library=[_in_one_season()])


def test_a_name_an_aired_sonarr_answers_with_an_error_is_an_unread_parse() -> None:
    parse_cache = _with_two_sixth_episodes()

    ((file,),) = parse_cache.parsed_files(_listing(_SIXTH)).values()

    assert file.parse is None


def test_a_name_an_alternate_title_may_not_count_for_has_no_matched_episodes() -> None:
    # Sonarr reads only some release names by the alternate title. By it the third episode is S02E03, without it
    # S01E03.
    limited = _ZOKU.model_copy(update={"comment": "Some releases only"})
    name = "Show Zoku - 03.mkv"
    by_sonarr = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(3,),
        series_title="Show Zoku",
        matched_episodes=(_matched(1, 3, 13),),
        series_id=1,
    )
    parse_cache = _parse_cache_in_one_season(_parsing({name: by_sonarr}), _in_one_season(limited))

    ((file,),) = parse_cache.parsed_files(_listing(name)).values()

    assert file.parse == by_sonarr.with_matched(())


def test_a_name_sonarr_searches_for_a_specials_title_has_no_matched_episodes() -> None:
    # Sonarr looks for a special's title in the name of an episode 0 of a season, and the parse doesn't keep the name.
    name = "Show S01E00 Recap.mkv"
    by_sonarr = ParsedFileInfo(
        season_number=1,
        episode_numbers=(0,),
        matched_episodes=(_matched(0, 1, 20),),
        series_id=1,
    )
    parse_cache = _parse_cache_in_one_season(_parsing({name: by_sonarr}))

    ((file,),) = parse_cache.parsed_files(_listing(name)).values()

    assert file.parse == by_sonarr.with_matched(())


def test_a_listed_file_of_a_series_with_no_aired_view_yet_has_no_matched_episodes() -> None:
    # No orders are stored for the series.
    parse_cache = make_parse_cache(_parsing({_FIFTH: _FIFTH_BY_SONARR}), FakeCacheStore(), library=[_in_one_season()])

    ((file,),) = parse_cache.parsed_files(_listing(_FIFTH)).values()

    assert file.parse == _FIFTH_BY_SONARR.with_matched(())


def test_the_listed_files_of_a_series_whose_episodes_fail_to_read_are_unread_parses_after_one_try() -> None:
    # A failed read isn't kept, and Sonarr's client warns on each one. The second file is left unread without a try.
    of_series_one = ParsedFileInfo(series_id=1)
    sonarr = _parsing({_FIFTH: of_series_one, _SIXTH: of_series_one})
    sonarr.episodes_return = None
    parse_cache = _parse_cache_in_one_season(sonarr)

    ((fifth, sixth),) = parse_cache.parsed_files(_listing(_FIFTH, _SIXTH)).values()

    assert (fifth.parse, sixth.parse) == (None, None)
    assert sonarr.episodes_calls == [1]


_WITHOUT_TVDB_IDS = pending_import(
    infohash="h", series_id=1, file_episode_map={"show - 01 [1080p].mkv": (11,)}, tvdb_ids=None
)
"""A pending import on series 1 stored without TVDB ids, its one file on episode id 11."""


class _SeriesTwoUnread(FakeSonarrClient):
    """A `FakeSonarrClient` whose read of series 2's episodes fails."""

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        return None if series_id == 2 else super().episodes(series_id, quiet=quiet)


def _at_run_start(
    series: list[SonarrItem],
    pending: PendingImport,
    sonarr: FakeSonarrClient | None = None,
    *,
    waits: ImportWaitMode = ImportWaitMode.BLOCKING,
) -> tuple[SonarrStrategy, FakeCacheStore]:
    """The strategy (`_sync`) of a run that waits on imports as `waits` says, `pending` stored and its title cached.

    TVDB series 50 has `_ORDERS` stored, and TheXEM was read for it and maps none of its episodes. No other series
    has either.
    """

    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    cache.put_scene_mappings(50, stored_scene_mappings())
    cache.update_cached_title(Arr.SONARR, PENDING_IMPORT_ANILIST_ID, {"updated_at": PENDING_IMPORT_STAMP})
    return _sync(series, cache, sonarr=sonarr, waits=waits), cache


def _title_cached(cache: FakeCacheStore) -> bool:
    """Whether the claiming title has a cache stamp."""

    cached = cache.get_cached_title(Arr.SONARR, PENDING_IMPORT_ANILIST_ID)
    return cached is not None and cached.updated_at is not None


def test_a_pending_import_stored_without_tvdb_ids_on_a_series_in_another_order_is_dropped_with_one_warning() -> None:
    recording = install_recording_hub()
    strat, cache = _at_run_start([_series(1, 50, season_type="dvd")], _WITHOUT_TVDB_IDS)

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {}
    assert not _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Pending import Show · SubGroup dropped: its episodes were matched before pearlarr could read the order "
            "Sonarr numbers its series by - checking its titles again (episodes it already imported may be on the "
            "wrong episodes)"
        )
    ]


def test_a_pending_import_stored_without_tvdb_ids_that_awaits_cleanup_is_left_alone() -> None:
    recording = install_recording_hub()
    pending = replace(_WITHOUT_TVDB_IDS, awaiting_cleanup=True)
    strat, cache = _at_run_start(
        [_series(1, 50, season_type="dvd")], pending, FakeSonarrClient(episodes=list(_DVD_ROWS))
    )

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}
    assert _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_a_pending_import_stored_without_tvdb_ids_takes_them_from_the_rows_of_its_aired_series() -> None:
    recording = install_recording_hub()
    strat, cache = _at_run_start([_series(1, 50)], _WITHOUT_TVDB_IDS)

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {
        "h": replace(_WITHOUT_TVDB_IDS, tvdb_ids=TvdbIds({11: 9001})).to_json()
    }
    assert _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == []


@pytest.mark.parametrize(
    ("series", "rows"),
    [
        pytest.param(_series(1, 50), None, id="a-failed-read"),
        pytest.param(_series(1, 50), [], id="no-rows"),
        pytest.param(_series(1, 70, season_type="dvd"), _DVD_ROWS, id="another-order-with-no-view"),
    ],
)
def test_a_pending_import_stored_without_tvdb_ids_is_left_as_stored_when_its_series_has_no_aired_rows_to_read(
    series: SonarrSeries, rows: list[SonarrEpisode] | None
) -> None:
    sonarr = FakeSonarrClient()
    sonarr.episodes_return = rows
    strat, cache = _at_run_start([series], _WITHOUT_TVDB_IDS, sonarr)

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": _WITHOUT_TVDB_IDS.to_json()}
    assert _title_cached(cache)


def test_a_two_series_pending_import_stored_without_tvdb_ids_waits_for_a_read_of_both_series() -> None:
    # Every id it holds is on a row Sonarr sends, so a read of both series would give it its TVDB ids.
    pending = two_claim_pending_import(
        series_ids=(1, 2),
        windows=((11,), (12,)),
        infohash="h",
        file_episode_map={"show - 01 [1080p].mkv": (11,)},
        tvdb_ids=None,
    )
    strat, cache = _at_run_start([_series(1, 50), _series(2, 60)], pending, _SeriesTwoUnread(episodes=list(_ROWS)))

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}


def test_a_pending_import_stored_without_tvdb_ids_holding_an_id_sonarr_sends_no_row_for_is_dropped_with_one_warning() -> (
    None
):
    recording = install_recording_hub()
    pending = replace(_WITHOUT_TVDB_IDS, file_episode_map={"show - 01 [1080p].mkv": (13,)})
    strat, cache = _at_run_start([_series(1, 50)], pending)

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {}
    assert not _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Pending import Show · SubGroup dropped: Sonarr no longer has an episode it was matched to - checking "
            "its titles again"
        )
    ]


def test_a_pending_import_stored_without_tvdb_ids_is_left_as_stored_while_imports_are_not_waited_on() -> None:
    strat, cache = _at_run_start([_series(1, 50)], _WITHOUT_TVDB_IDS, waits=ImportWaitMode.OFF)

    strat.start_of_run([])

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": _WITHOUT_TVDB_IDS.to_json()}


def test_a_stored_pending_import_is_not_followed_under_the_rows_of_a_series_the_run_holds() -> None:
    # Sonarr is part way through renumbering the series by its dvd order: the one row it sent is out of step with
    # the aired order, and TVDB episode 9001 is on none. Followed under it, the file on that episode would leave
    # the file map.
    pending = pending_import(
        infohash="h",
        series_id=1,
        file_episode_map={"show - 01 [1080p].mkv": (11,), "show - 02 [1080p].mkv": (12,)},
        tvdb_ids={11: 9001, 12: 9002},
    )
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=[_DVD_ROWS[0]])
    reads = _reads([_series(1, 50)], cache, _skyhook(_ORDERS), sonarr=sonarr)
    follower = make_pending_follower(cache, reads)
    # A title's check asks for the index of every claimed series first, which is where the run comes to hold this one.
    assert reads.series_index(1) is None

    followed = follower.followed({"h": pending})

    assert followed == {}
    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}


_SEASON_TWO_AIRED = {"official": (*_ORDERS["official"], OrderRow(2, 1, 9003, 3)), "dvd": _DVD}
"""A stored answer whose official order has a second season `_DVD` has no episode of."""

_SEASON_TITLES = {500: "First", 600: "Second", 700: "Third"}
"""The AniList titles of seasons 1, 2 and 3 of TVDB series 50.

Of `_SEASON_TWO_AIRED`'s orders, both have First's episodes, only the aired one has Second's, and neither has Third's.
"""

_NO_EPISODE_IN_THE_ORDER = (
    "Sonarr's DVD order of the series has none of Second's episodes - not checked again until the series' order, "
    "its episode count or the title's SeaDex entry changes"
)


def _entry(anilist_id: int) -> EntryRecord:
    """The title's SeaDex entry, which lists no release."""

    return make_entry_record(anilist_id=anilist_id, url=f"https://releases.moe/{anilist_id}")


def _library_row(season_type: str | None, total: int | None, counts: Mapping[str, int] | None = None) -> SonarrSeries:
    """Series 1 of TVDB series 50 as a library read sends it, with `total` episodes by its `statistics`.

    With `season_type=None` the row has no `seasonType` key, and with `total=None` no `statistics`. It lists the
    episode count of each order in `counts`, and no order without them.
    """

    raw: dict[str, object] = {"id": 1, "title": "Show", "tvdbId": 50}
    if season_type is not None:
        raw["seasonType"] = season_type
    if total is not None:
        raw["statistics"] = {"seasonCount": 1, "episodeCount": total, "totalEpisodeCount": total}
    if counts is not None:
        raw["seasonTypes"] = [{"type": name, "episodeCount": count} for name, count in counts.items()]
    return SonarrSeries.model_validate(raw)


def _run(
    cache: FakeCacheStore,
    library: SonarrSeries,
    xem: FakeXemClient | None = None,
    *,
    titles: Mapping[int, str] = _SEASON_TITLES,
) -> RecordingHub:
    """One whole Sonarr run over `cache` (`_run_deps`) mapping `titles`, and what it printed.

    Sonarr lists `library` alone, with the season's two rows numbered by the order it names. TVDB series 50 has
    `_SEASON_TWO_AIRED` stored, written over what `cache` held on every call. TheXEM is `xem`, one that maps no
    episode of any series when left out.
    """

    cache.put_tvdb_orders(50, stored_orders(_SEASON_TWO_AIRED))
    rows = _DVD_ROWS if library.seasonType == "dvd" else _ROWS
    sonarr = FakeSonarrClient(all_series=[library], episodes=list(rows))
    return _run_over(sonarr, _run_deps(cache, xem=xem, titles=titles))


def _run_deps(
    cache: FakeCacheStore,
    *,
    xem: FakeXemClient | None = None,
    skyhook: FakeSkyHookClient | None = None,
    titles: Mapping[int, str] = _SEASON_TITLES,
) -> ArrRunDeps:
    """What a whole run over `cache` is built from, with activity detection off.

    qBittorrent is configured, so it is a real run, which saves what it stores for the next. Each of `titles`, by
    AniList id, has its SeaDex entry (`_entry`), its AniList answer and its season of TVDB series 50, in order from
    season 1. TheXEM is `xem` and SkyHook is `skyhook`, each one that has no series when left out.
    """

    seadex = FakeSeaDexSource({anilist_id: _entry(anilist_id) for anilist_id in titles})
    config = make_config(url="http://sonarr", api_key="key", sleep_time=0, detect_arr_activity=False)
    seasons: AnimeIdsByName = {
        name: {"anilist_id": anilist_id, "tvdb_id": 50, "tvdb_season": season}
        for season, (anilist_id, name) in enumerate(titles.items(), start=1)
    }
    deps = make_run_deps(
        config=config,
        cache_store=cache,
        qbit=FakeQbit().client(),
        seadex=seadex,
        xem_client=xem,
        skyhook_client=skyhook,
        anime_ids=seasons,
    )
    for anilist_id, name in titles.items():
        deps.anilist.answers[anilist_id] = anilist_body(anilist_id, name)
    return deps


def _run_over(sonarr: FakeSonarrClient, deps: ArrRunDeps) -> RecordingHub:
    """One whole Sonarr run on `sonarr` with `deps`, and what it printed."""

    recording = install_recording_hub()
    drive_arr_run(deps, lambda services: SonarrStrategy(deps, services, sonarr_client=sonarr))
    return recording


def _checked(recording: RecordingHub) -> list[int | None]:
    """The AniList ids of the titles the run opened a block to check, leaving out the ones it found unchanged."""

    return [header.anilist_id for header in recording.of_type(TitleHeader) if header.state is TitleState.CHECKING]


def _no_episode_rows(recording: RecordingHub) -> list[str]:
    """The titles the run printed a no-episodes row for."""

    return [row.label for row in recording.of_type(TitleRow) if row.state is TitleState.NO_EPISODES]


@pytest.mark.parametrize(
    ("library", "cached"),
    [
        pytest.param(_library_row("dvd", 2), ("dvd", 2), id="another-order"),
        pytest.param(_library_row("official", 2), ("official", 2), id="aired"),
        pytest.param(_library_row(None, None), ("official", None), id="no-order-and-no-count-sent"),
    ],
)
def test_a_checked_title_is_cached_with_the_order_and_episode_count_of_its_series(
    library: SonarrSeries, cached: tuple[str, int | None]
) -> None:
    cache = FakeCacheStore()

    _run(cache, library)

    first = cache.get_cached_title(Arr.SONARR, 500)
    assert first is not None
    assert (first.season_type, first.series_row_count) == cached


@pytest.mark.parametrize(
    ("before", "now"),
    [
        pytest.param(_library_row("official", 2), _library_row("dvd", 2), id="another-order"),
        pytest.param(_library_row("dvd", 2), _library_row("official", 2), id="aired-again"),
        pytest.param(_library_row("dvd", 2), _library_row("dvd", 3), id="another-count-under-another-order"),
    ],
)
def test_a_cached_title_is_checked_again_once_its_series_is_numbered_another_way(
    before: SonarrSeries, now: SonarrSeries
) -> None:
    cache = FakeCacheStore()
    _run(cache, before)

    recording = _run(cache, now)

    assert _checked(recording) == [500]


def test_a_series_whose_titles_are_all_cached_is_checked_again_once_it_is_numbered_another_way() -> None:
    # With no title of the series left uncached, only the start-of-run hook's mark of First has the run read the
    # series' episodes and scene numbers before the scan. Marked any later, the check waits a run for them.
    cache = FakeCacheStore()
    _run(cache, _library_row("official", 2), titles={500: "First"})

    recording = _run(cache, _library_row("dvd", 2), titles={500: "First"})

    assert _checked(recording) == [500]


@pytest.mark.parametrize(
    ("before", "now"),
    [
        pytest.param(_library_row("dvd", 2), _library_row("dvd", 2), id="nothing-changed"),
        # A Sonarr that sends no order numbers every series as aired, whose count is not compared.
        pytest.param(_library_row(None, 2), _library_row(None, 3), id="a-sonarr-that-sends-no-order"),
    ],
)
def test_a_cached_title_is_left_unchanged_while_its_series_is_numbered_the_same_way(
    before: SonarrSeries, now: SonarrSeries
) -> None:
    cache = FakeCacheStore()
    _run(cache, before)

    recording = _run(cache, now)

    assert _checked(recording) == []


_WITH_A_THIRD = {
    "official": (*_ORDERS["official"], OrderRow(1, 3, 9003, 3)),
    "dvd": (*_DVD, OrderRow(1, 3, 9003, None)),
}
"""TVDB series 50's orders once it has a third episode, which both number alike."""


def _dvd_sonarr(total: int) -> FakeSonarrClient:
    """A Sonarr with series 1 in the DVD order at `total` episodes, listing that many under each of its orders."""

    third = sonarr_ep(1, 3, ep_id=13, tvdb_id=9003)
    library = _library_row("dvd", total, {"official": total, "dvd": total})
    return FakeSonarrClient(all_series=[library], episodes=[*_DVD_ROWS, third][:total])


def test_a_title_checked_on_orders_skyhook_could_not_refresh_is_checked_again_once_it_answers() -> None:
    # The title is cached while its series has two episodes. Sonarr then has a third, and the counts it lists
    # make the stored orders, which lack that episode, due a fetch.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2, "dvd": 2}))
    _run_over(_dvd_sonarr(2), _run_deps(cache))
    unreachable = FakeSkyHookClient(failing=frozenset({(50, "official")}))
    _run_over(_dvd_sonarr(3), _run_deps(cache, skyhook=unreachable))

    recording = _run_over(_dvd_sonarr(3), _run_deps(cache, skyhook=_skyhook(_WITH_A_THIRD)))

    first = cache.get_cached_title(Arr.SONARR, 500)
    assert _checked(recording) == [500]
    assert first is not None
    assert first.coverage == "S01 E01-E03"


@pytest.mark.parametrize(
    ("library", "checked"),
    [
        pytest.param(_library_row("official", 2), [], id="aired"),
        pytest.param(_library_row("dvd", 2), [500], id="another-order"),
    ],
)
def test_a_title_cached_with_no_order_is_checked_again_only_when_its_series_is_in_another_order(
    library: SonarrSeries, checked: list[int]
) -> None:
    # The row as a run stored it before a title's cache kept its series' order and episode count.
    cache = FakeCacheStore()
    cache.update_cached_title(Arr.SONARR, 500, {"updated_at": _entry(500).updated_at})

    recording = _run(cache, library)

    assert _checked(recording) == checked


def test_a_title_sonarrs_order_has_no_episode_of_warns_once_and_is_cached() -> None:
    cache = FakeCacheStore()

    recording = _run(cache, _library_row("dvd", 2))

    second = cache.get_cached_title(Arr.SONARR, 600)
    assert diagnostic_messages(recording, Severity.WARNING) == [_NO_EPISODE_IN_THE_ORDER]
    assert second is not None
    assert (second.updated_at is not None, second.coverage) == (True, "")
    assert second.numbering == SeriesNumbering("dvd", 2)


def test_a_title_sonarrs_order_has_no_episode_of_is_not_warned_of_again_with_nothing_changed() -> None:
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2))

    recording = _run(cache, _library_row("dvd", 2))

    assert diagnostic_messages(recording, Severity.WARNING) == []
    assert _no_episode_rows(recording) == ["Third"]


def test_a_title_sonarrs_order_has_no_episode_of_is_checked_again_when_the_series_episode_count_changes() -> None:
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2))

    recording = _run(cache, _library_row("dvd", 3))

    assert diagnostic_messages(recording, Severity.WARNING) == [_NO_EPISODE_IN_THE_ORDER]


def test_a_title_sonarrs_order_has_no_episode_of_is_checked_again_while_its_series_due_orders_are_unread() -> None:
    # The stored orders are past their fresh-until, and SkyHook is unreachable for the run.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_SEASON_TWO_AIRED, fresh_for=timedelta(days=-1)))
    sonarr = FakeSonarrClient(all_series=[_library_row("dvd", 2)], episodes=list(_DVD_ROWS))
    unreachable = FakeSkyHookClient(failing=frozenset({(50, "official")}))

    recording = _run_over(sonarr, _run_deps(cache, skyhook=unreachable))

    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Sonarr's DVD order of the series has none of Second's episodes - checking again next run"
    ]
    assert cache.get_cached_title(Arr.SONARR, 600) is None


def test_a_title_with_no_aired_episode_prints_its_row_again_on_the_next_run_and_is_not_cached() -> None:
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2))

    recording = _run(cache, _library_row("dvd", 2))

    assert _no_episode_rows(recording) == ["Third"]
    assert cache.get_cached_title(Arr.SONARR, 700) is None


_THEXEM_STEP = "Fetching TheXEM scene numbers"


def _boot_steps(recording: RecordingHub) -> dict[str, str | None]:
    """The detail each boot step of the run ended on, by the step's label, in the order the steps ran."""

    return {step.label: step.detail for step in recording.of_type(BootStepFinished)}


def test_a_run_reads_the_scene_mappings_of_its_series_in_another_order_after_the_tvdb_orders() -> None:
    recording = _run(FakeCacheStore(), _library_row("dvd", 2))

    steps = _boot_steps(recording)
    assert list(steps)[-2:] == ["Fetching TVDB orders", _THEXEM_STEP]
    assert steps[_THEXEM_STEP] == "1 series"


def test_a_run_over_a_library_all_in_aired_order_has_no_thexem_boot_step() -> None:
    recording = _run(FakeCacheStore(), _library_row("official", 2))

    assert _THEXEM_STEP not in _boot_steps(recording)


def test_an_unreachable_thexem_shows_on_its_own_boot_step_and_not_on_the_tvdb_orders_one() -> None:
    recording = _run(FakeCacheStore(), _library_row("dvd", 2), FakeXemClient(failing=frozenset({50})))

    steps = _boot_steps(recording)
    assert (steps["Fetching TVDB orders"], steps[_THEXEM_STEP]) == ("cached", "unreachable")


def _selection_stored(cache: FakeCacheStore) -> bool:
    """Whether a run stored its selection settings in `cache`, which then differ from any others."""

    return cache.selection_changed(Arr.SONARR, "some-other-digest")


def test_a_run_that_found_thexem_unreachable_stores_the_selection_hash_and_leaves_its_titles_for_the_next_run() -> None:
    # The unread series' titles are not cached, so the next run checks them whatever this one committed.
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2), FakeXemClient(failing=frozenset({50})))
    hash_stored = _selection_stored(cache)

    recording = _run(cache, _library_row("dvd", 2))

    assert hash_stored is True
    assert _checked(recording) == [500]


@pytest.mark.parametrize(
    ("failing", "note", "stored"),
    [
        pytest.param(frozenset({(50, "official")}), "unreachable", False, id="outage"),
        pytest.param(frozenset[tuple[int, str]](), "1 series", True, id="healthy"),
    ],
)
def test_a_run_stores_the_selection_hash_only_once_skyhook_answered_and_its_boot_step_says_which(
    failing: frozenset[tuple[int, str]], note: str, stored: bool
) -> None:
    # Sonarr lists a third episode the stored orders lack, so they are due a fetch. A run that read the series by the
    # old ones leaves the hash unstored, so the next run checks again what a settings change named.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2, "dvd": 2}))

    recording = _run_over(_dvd_sonarr(3), _run_deps(cache, skyhook=_skyhook(_WITH_A_THIRD, failing=failing)))

    assert _boot_steps(recording)["Fetching TVDB orders"] == note
    assert _selection_stored(cache) is stored
