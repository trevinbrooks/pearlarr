# pyright: strict
# pyright: reportPrivateUsage=false
# The tests read the strategy's private collaborators (_episodes / _parse / _reads / _reconciler),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Each series' TVDB orders through a run's reads and the Sonarr strategy: what is fetched, what grab and import get.

The grab is served a file's parse too, matched as an aired Sonarr matches it.

A test of the reads alone builds them alone (`_reads`). One that asks for a title's episodes, the parse cache, the
import or a whole run builds the strategy (`_sync`).
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import date, timedelta
from typing import Any, override

import pytest
from seadex import EntryRecord

from pearlarr.aired_view import AiredSeries, NoView, stand_in
from pearlarr.arr_http import NoResource
from pearlarr.arr_models import (
    AlternateTitle,
    MatchedEpisode,
    ParsedFileInfo,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
)
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.boot_flow import BootFlow
from pearlarr.config import Arr
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import MappingResolver, MappingSources, TitleMapping
from pearlarr.output import BootStepFinished, Severity, TitleHeader, TitleRow
from pearlarr.output.events import TitleState
from pearlarr.parse_cache import parse_to_json
from pearlarr.pending_follow import PendingFollower
from pearlarr.pending_import import PendingImport, TvdbIds
from pearlarr.releases import EpisodeKey, SeadexDict
from pearlarr.scene_numbers import SceneMapping
from pearlarr.series_reads import SeriesReads
from pearlarr.skyhook_client import SkyHookReply
from pearlarr.sonarr_episodes import TitleEpisodes, _dated, sonarr_series_hash
from pearlarr.sonarr_matching import AiredMatcher
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import EpisodeDetails, OrderRow, SeriesNumbering
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    PENDING_IMPORT_ANILIST_ID,
    PENDING_IMPORT_STAMP,
    FakeCacheStore,
    FakeSeaDexSource,
    anilist_body,
    every_series,
    known_torrent,
    make_config,
    make_entry_record,
    make_pending_follower,
    make_release,
    make_run_deps,
    make_series_reads,
    make_sonarr_sync,
    pending_import,
    rg_group,
    sonarr_ep,
    stand_in_ep,
    stored_orders,
    stored_scene_mappings,
    title_window,
    torrent_evidence,
    two_claim_pending_import,
)
from .fakes import (
    FakeSkyHookClient,
    FakeSonarrClient,
    FakeXemClient,
    HookedSonarrClient,
    RecordingHub,
    debug_logger,
    diagnostic_messages,
    install_recording_hub,
)

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


def _reads(
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
) -> SeriesReads:
    """A run's reads after its library read of `series`, with no strategy around them.

    The series, their rows and what Sonarr answers of one alone are as `_sync` has them. TVDB orders are fetched
    through `client`, and a series has the scene mappings `cache` holds.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ROWS))
    sonarr.all_series_return = series
    return make_series_reads(sonarr, cache, library=series, skyhook_client=client)


def _sync(
    monkeypatch: pytest.MonkeyPatch,
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
    xem: FakeXemClient | None = None,
) -> SonarrStrategy:
    """The Sonarr strategy after the Sonarr run's library fetch of `series`, its orders fetched through `client`.

    Every series has an AniList mapping, and Sonarr answers a read of one as the library lists it unless `sonarr`
    is scripted otherwise. Every series' episodes are `_ROWS` unless `sonarr` serves others. Its AniList knows no
    title. Scene mappings are read through `xem`, and none is read here: a series has the ones `cache` holds.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ROWS))
    sonarr.all_series_return = series
    strat = make_sonarr_sync(sonarr=sonarr, cache_store=cache, skyhook_client=client, xem_client=xem)
    monkeypatch.setattr(strat._episodes, "mapped_series", every_series)
    strat.get_items()
    return strat


def _numbered(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, EpisodeKey]]:
    """Each episode's id and its `(season, episode)`, in the order given."""

    return [(ep.id, EpisodeKey(ep.season_number or 0, ep.episode_number or 0)) for ep in episodes]


@pytest.mark.parametrize(
    ("series_id", "season_type", "order", "other_orders"),
    [
        pytest.param(1, "official", "official", {"dvd"}, id="official"),
        # A null or junk `seasonType` names no order.
        pytest.param(1, "", "", set[str](), id="no-order"),
        # The library's one series carries no `seasonType`, so a series it lacks is read as its rows come.
        pytest.param(2, None, "", set[str](), id="not-in-the-library"),
    ],
)
def test_a_series_not_numbered_by_another_order_is_served_sonarrs_rows_as_sent(
    series_id: int, season_type: str | None, order: str, other_orders: set[str]
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    reads = _reads([_series(1, 50, season_type=season_type)], cache)

    view = reads.aired_series(series_id)
    index = reads.series_index(series_id)

    assert isinstance(view, AiredSeries)
    assert (reads.library_orders(series_id).season_type, view.aired) == (order, tuple(_ROWS))
    assert index is not None
    assert (set(index.by_id), set(index.orders)) == ({11, 12}, other_orders)


def test_a_series_numbered_by_another_order_is_served_as_aired_with_its_stored_orders_read_onto_it() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    view = reads.aired_series(1)
    index = reads.series_index(1)

    assert isinstance(view, AiredSeries)
    assert _numbered(view.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert index is not None
    assert index.id_by_key == {(1, 1): 12, (1, 2): 11, (1, 3): -9003}
    assert {name: order.id_by_key for name, order in index.orders.items()} == {"dvd": {(1, 1): 11, (1, 2): 12}}


def test_a_titles_episodes_of_a_series_numbered_by_another_order_are_its_aired_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    title = strat._episodes.title_episodes(1, 500, _SEASON_ONE)

    assert not isinstance(title, NoView)
    assert _numbered(title.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert _numbered(title.in_sonarr) == [(12, (1, 1)), (11, (1, 2))]
    assert title.index().by_id[-9003] == stand_in_ep(1, 3, tvdb_id=9003).model_copy(
        update={"absolute_episode_number": 3}
    )


def test_a_dated_special_sonarr_has_no_row_for_is_named_by_its_aired_number() -> None:
    row = sonarr_ep(0, 4, ep_id=4, air_date="2019-02-20")
    special = stand_in(OrderRow(0, 5, 905, None), EpisodeDetails(title="", air_date=date(2020, 8, 29), runtime=95))

    assert _dated([row, special]) == "S00E04 (aired 2019-02-20), aired S00E05 (on 2020-08-29)"


def test_a_series_numbered_by_another_order_with_nothing_stored_has_no_view(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], FakeCacheStore(), sonarr=sonarr)

    assert strat._reads.aired_series(1) is NoView.ORDERS_UNREAD
    assert strat._reads.series_index(1) is None
    assert strat._episodes.title_episodes(1, 500, _SEASON_ONE) is NoView.ORDERS_UNREAD
    assert strat._series_indexes({1}) == {}


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


def test_a_series_numbered_by_another_order_whose_scene_mappings_were_never_read_has_no_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    assert strat._reads.aired_series(1) is NoView.SCENE_UNREAD
    assert strat._reads.series_index(1) is None
    assert strat._episodes.title_episodes(1, 500, _SEASON_ONE) is NoView.SCENE_UNREAD


def test_a_series_numbered_by_another_order_that_thexem_has_no_mappings_for_has_a_view_without_scene_numbers() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
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
    rows = [sonarr_ep(1, 1, ep_id=11, tvdb_id=9001, scene_season=9, scene_episode=1, scene_absolute=91)]
    cache = FakeCacheStore()
    cache.put_scene_mappings(50, stored_scene_mappings((SceneMapping(1, 1, 2, 1, 13),)))
    reads = _reads([_series(1, 50)], cache, sonarr=FakeSonarrClient(episodes=rows))

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _scene(view.aired) == [(11, (9, 1, 91))]


def test_a_series_index_is_built_once_for_a_read_of_its_rows() -> None:
    reads = _reads([_series(1, 50)], FakeCacheStore())

    first = reads.series_index(1)

    assert first is not None
    assert reads.series_index(1) is first


def test_a_new_read_of_a_series_replaces_its_view_and_index() -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    before = reads.series_index(1)
    third = sonarr_ep(1, 3, ep_id=13, tvdb_id=9003)
    sonarr.episodes_return = [*_ROWS, third]

    reads.read_episodes(1)

    view = reads.aired_series(1)
    index = reads.series_index(1)
    assert before is not None
    assert set(before.by_id) == {11, 12}
    assert isinstance(view, AiredSeries)
    assert view.aired == (*_ROWS, third)
    assert index is not None
    assert set(index.by_id) == {11, 12, 13}


def test_a_sonarr_that_sends_no_order_is_never_asked_for_a_series_alone() -> None:
    # No series of the library read carries `seasonType`, so each is read as aired, as its rows come.
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    library: list[SonarrItem] = [_series(1, 50, season_type=None), _series(2, 60, season_type=None)]
    reads = _reads(library, FakeCacheStore(), sonarr=sonarr)

    views = [reads.aired_series(series_id) for series_id in (1, 2)]

    assert views == [AiredSeries(in_sonarr=tuple(_ROWS), stand_ins=())] * 2
    assert sonarr.series_calls == []


def test_a_series_of_an_all_aired_library_is_read_alone_when_sonarr_sends_its_order() -> None:
    # Every series says `official`. That the key was sent, whatever its value, is what has each one read again.
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50), _series(2, 60)], FakeCacheStore(), sonarr=sonarr)

    reads.aired_series(1)
    reads.aired_series(2)

    assert sonarr.series_calls == [1, 2]


def test_a_series_both_reads_say_is_aired_is_read_alone_once_for_a_read_of_its_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), sonarr=sonarr)

    first = strat._episodes.title_episodes(1, 500, _SEASON_ONE)
    second = strat._episodes.title_episodes(1, 501, TitleMapping(anilist_id=501, tvdb_season=1))

    assert first == TitleEpisodes(aired=tuple(_ROWS))
    assert second == first
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([1], [1])


_ORDER_CHANGED = (
    "Show 1 not checked - the order Sonarr numbers the series by changed during this run - checking its titles "
    "again next run"
)
"""The warning for series 1 once Sonarr names another order for it than this run's library read did."""


def test_a_series_whose_order_changed_since_the_library_read_has_no_view_for_the_rest_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sonarr names the dvd order only once it has sent the rows, so a read made before them would miss it.
    recording = install_recording_hub()
    sonarr = HookedSonarrClient(episodes=list(_DVD_ROWS))
    sonarr.on_episodes = lambda: sonarr.series_return.update({1: _series(1, 50, season_type="dvd")})
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), sonarr=sonarr)

    views = [
        strat._reads.aired_series(1),
        strat._episodes.title_episodes(1, 500, _SEASON_ONE),
        strat._reads.aired_series(1),
    ]

    assert views == [NoView.READ_FAILED] * 3
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([1], [1])
    assert strat._reads.last_known_episodes(1) == _DVD_ROWS
    assert diagnostic_messages(recording, Severity.WARNING) == [_ORDER_CHANGED]


def test_a_series_the_library_read_has_in_another_order_is_renumbered_whatever_sonarr_says_of_it_later() -> None:
    # Sonarr went to the aired order after the library read: its rows and a read of the series alone both say so.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)
    sonarr.series_return[1] = _series(1, 50)

    view = reads.aired_series(1)

    assert isinstance(view, AiredSeries)
    assert _numbered(view.aired) == [(11, (1, 1)), (12, (1, 2)), (-9003, (1, 3))]
    assert sonarr.series_calls == []


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(NoResource.FAILED, id="failed"),
        # A reply with no `seasonType`, from a Sonarr that sent one with the library, says nothing of the order.
        pytest.param(_series(1, 50, season_type=None), id="no-order-key"),
    ],
)
def test_a_failed_read_of_a_series_alone_gives_no_view_and_the_next_call_asks_again(
    answer: SonarrItem | NoResource,
) -> None:
    recording = install_recording_hub()
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    sonarr.series_return[1] = answer

    failed = reads.aired_series(1)
    del sonarr.series_return[1]
    view = reads.aired_series(1)

    assert failed is NoView.READ_FAILED
    assert view == AiredSeries(in_sonarr=tuple(_ROWS), stand_ins=())
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([1], [1, 1])
    assert diagnostic_messages(recording, Severity.WARNING) == []


@pytest.mark.parametrize(
    "library",
    [pytest.param([_series(1, 50)], id="in-the-library"), pytest.param([_series(2, 60)], id="not-in-the-library")],
)
def test_a_series_sonarr_answers_it_does_not_have_has_no_episodes(library: list[SonarrItem]) -> None:
    recording = install_recording_hub()
    sonarr = FakeSonarrClient(episodes=[])
    reads = _reads(library, FakeCacheStore(), sonarr=sonarr)
    sonarr.series_return[1] = NoResource.GONE

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=(), stand_ins=())
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_a_series_the_library_read_lacks_that_sonarr_has_by_now_has_no_view_for_the_rest_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Series 2 came to Sonarr after the library read, so the run knows neither its order nor its TVDB series.
    recording = install_recording_hub()
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    sonarr.series_return[2] = _series(2, 60)

    views = [strat._reads.aired_series(2), strat._reads.aired_series(2)]

    assert views == [NoView.READ_FAILED] * 2
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([2], [2])
    # Nothing the read said was kept: the run still has no TVDB series to fetch orders for.
    assert strat._episodes.prefetch_orders([2]) == 0
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Show 2 not checked - the series was not in Sonarr when this run read its library - checking its "
            "titles next run"
        )
    ]


def test_a_series_with_no_view_for_the_rest_of_the_run_is_unread_wherever_a_release_needs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torrent SeaDex lists under a title of series 1 too is grabbed by a title of series 2.
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60)], FakeCacheStore(), sonarr=sonarr)
    sonarr.series_return[1] = _series(1, 50, season_type="dvd")
    strat._reads.aired_series(1)
    named_as_series_one = ParsedFileInfo(
        season_number=1,
        episode_numbers=(2,),
        series_title="Show 1",
        matched_episodes=(_matched(1, 2, 12),),
        series_id=1,
    )

    indexes = strat._series_indexes({1, 2})
    listed = known_torrent(indexes=indexes, evidence=torrent_evidence(title_window(500, (11, 12), series_id=1)))

    assert set(indexes) == {2}
    assert listed.order(title_window(600, (11, 12), series_id=2)) is None
    assert strat._reads.aired_parse(named_as_series_one) is NoView.READ_FAILED


def test_a_view_built_after_the_imports_read_of_a_series_reads_it_alone_once_more() -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=sonarr)

    reads.read_episodes(1)
    asked_by_the_import = list(sonarr.series_calls)
    reads.aired_series(1)
    reads.aired_series(1)

    assert asked_by_the_import == [1]
    assert sonarr.series_calls == [1, 1]


def _with_an_unmapped_series(
    cache: FakeCacheStore, sonarr: FakeSonarrClient
) -> tuple[SonarrStrategy, list[SonarrItem]]:
    """The strategy and the items of its library read, where only TVDB series 50 has an AniList mapping.

    Sonarr lists series 1 of TVDB series 50, and series 2 of TVDB series 60, numbered by its dvd order.
    """

    sonarr.all_series_return = [_series(1, 50), _series(2, 60, season_type="dvd")]
    mapped = MappingSources(anime={"Show 1": {"anilist_id": 500, "tvdb_id": 50}}, anidb=False, anibridge=False)
    deps = make_run_deps(cache_store=cache)
    resolver = MappingResolver(refresh_days=1, ignore_anilist_ids=set(), web=deps.web, sources=mapped)
    deps = replace(deps, resolver=resolver)
    strat = SonarrStrategy(deps, ArrRunServices(deps, Arr.SONARR), sonarr_client=sonarr)
    return strat, strat.get_items()


def test_a_series_with_no_anilist_mapping_is_read_by_the_order_the_library_read_gives_it() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(60, stored_orders(_LONGER_AIRED))
    cache.put_scene_mappings(60, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))

    strat, items = _with_an_unmapped_series(cache, sonarr)

    view = strat._reads.aired_series(2)
    assert [item.id for item in items] == [1]
    assert strat._reads.series_hash == sonarr_series_hash([1])
    assert sonarr.all_series_calls == 1
    assert isinstance(view, AiredSeries)
    assert _numbered(view.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]


def test_the_orders_of_a_pending_imports_series_with_no_anilist_mapping_are_fetched() -> None:
    pending = pending_import(infohash="h", series_id=2)
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    # Nothing in the scan needs a series, so the pending import's is the only one asked for.
    strat, _ = _with_an_unmapped_series(cache, FakeSonarrClient(episodes=list(_DVD_ROWS)))
    strat._services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING))
    strat.start_of_run([])

    fetched = strat.prefetch_orders([])

    assert fetched == 1
    assert cache.get_tvdb_orders(60) is not None


def test_the_scene_mappings_of_a_pending_imports_series_are_read_for_the_one_in_another_order() -> None:
    # One pending import claimed on both series. Only series 2, of TVDB series 60, is in another order.
    pending = two_claim_pending_import(series_ids=(1, 2), infohash="h")
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat, _ = _with_an_unmapped_series(cache, FakeSonarrClient(episodes=list(_DVD_ROWS)))
    strat._services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING))
    strat.start_of_run([])

    read = strat.prefetch_scene_numbers([])

    assert read == 1
    assert (cache.get_scene_mappings(50), cache.get_scene_mappings(60) is not None) == (None, True)


def test_a_series_with_no_anilist_mapping_and_no_stored_answer_is_unread() -> None:
    strat, _ = _with_an_unmapped_series(FakeCacheStore(), FakeSonarrClient(episodes=list(_DVD_ROWS)))

    assert strat._episodes.metadata_unread([], [2]) is True


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


def _skyhook(orders: Mapping[str, tuple[OrderRow, ...]]) -> FakeSkyHookClient:
    """A SkyHook that has TVDB series 50 with `orders`, each reply as the wire sends it."""

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
        }
    )


def _aired_over_dvd_rows(client: FakeSkyHookClient, *, stored: bool = True) -> SeriesReads:
    """A run's reads after a library read of series 1 as aired, on a Sonarr whose rows of it are `_DVD_ROWS`.

    An earlier run stored `_ORDERS` for TVDB series 50, unless `stored` is off. SkyHook is `client`.
    """

    cache = FakeCacheStore()
    if stored:
        cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    return _reads([_series(1, 50)], cache, client, sonarr=sonarr)


@pytest.mark.parametrize(
    ("client", "warning"),
    [
        pytest.param(_skyhook(_ORDERS), _OUT_OF_STEP, id="still-out-of-step"),
        # SkyHook doesn't have the series any more, so the new answer has no aired order to clear the rows by.
        pytest.param(FakeSkyHookClient(), _OUT_OF_STEP, id="no-rows-for-the-order"),
        pytest.param(FakeSkyHookClient(failing=frozenset({(50, "official")})), _UNCONFIRMED, id="skyhook-unreachable"),
    ],
)
def test_a_series_both_reads_say_is_aired_over_rows_numbered_by_another_order_has_no_view_for_the_rest_of_the_run(
    client: FakeSkyHookClient, warning: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = install_recording_hub()
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], cache, client, sonarr=sonarr)

    views = [
        strat._reads.aired_series(1),
        strat._episodes.title_episodes(1, 500, _SEASON_ONE),
        strat._reads.aired_series(1),
    ]

    assert views == [NoView.READ_FAILED] * 3
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([1], [1])
    assert diagnostic_messages(recording, Severity.WARNING) == [warning]


def test_rows_out_of_step_with_an_answer_from_an_earlier_run_are_tested_again_by_one_fetched_now() -> None:
    # TVDB swapped the two episodes since the answer was stored, and Sonarr's rows already follow it.
    recording = install_recording_hub()
    client = _skyhook(_SWAPPED_SINCE)
    reads = _aired_over_dvd_rows(client)

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())
    assert client.fetch_calls == [(50, "official")]
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_rows_out_of_step_with_an_answer_this_run_wrote_are_not_fetched_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _skyhook(_ORDERS)
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), client, sonarr=sonarr)
    strat._episodes.prefetch_orders([1])
    asked_at_the_prefetch = list(client.fetch_calls)

    view = strat._reads.aired_series(1)

    assert view is NoView.READ_FAILED
    assert client.fetch_calls == asked_at_the_prefetch == [(50, "official"), (50, "dvd")]


def test_rows_of_a_series_with_no_stored_answer_are_read_as_they_come() -> None:
    client = _skyhook(_ORDERS)
    reads = _aired_over_dvd_rows(client, stored=False)

    view = reads.aired_series(1)

    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())
    assert client.fetch_calls == []


def test_a_parse_of_a_series_whose_rows_are_out_of_step_has_no_answer() -> None:
    reads = _aired_over_dvd_rows(_skyhook(_ORDERS))
    reads.aired_series(1)
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(2,),
        series_title="Show 1",
        matched_episodes=(_matched(1, 2, 12),),
        series_id=1,
    )

    assert reads.aired_parse(info) is NoView.READ_FAILED


def test_the_import_reads_rows_numbered_by_the_order_sonarr_names_once_it_has_sent_them() -> None:
    # Sonarr went to the dvd order after the library read. It names that order only once it has sent the rows,
    # so a read of the series made before them, or the library read's order, would refuse these rows.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    sonarr = HookedSonarrClient(episodes=list(_DVD_ROWS))
    sonarr.on_episodes = lambda: sonarr.series_return.update({1: _series(1, 50, season_type="dvd")})
    reads = _reads([_series(1, 50)], cache, sonarr=sonarr)

    rows = reads.read_episodes(1)

    assert rows == _DVD_ROWS
    assert reads.last_known_episodes(1) == _DVD_ROWS
    assert (sonarr.episodes_calls, sonarr.series_calls) == ([1], [1])


def test_the_import_is_not_given_rows_out_of_step_with_the_order_sonarr_names() -> None:
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


def test_rows_out_of_step_are_warned_of_once_for_the_import_and_the_check_together() -> None:
    recording = install_recording_hub()
    reads = _aired_over_dvd_rows(_skyhook(_ORDERS))

    answers = [reads.read_episodes(1), reads.read_episodes(1), reads.aired_series(1)]

    assert answers == [None, None, NoView.READ_FAILED]
    assert diagnostic_messages(recording, Severity.WARNING) == [_OUT_OF_STEP]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(NoResource.FAILED, id="failed"),
        pytest.param(NoResource.GONE, id="gone"),
        pytest.param(_series(1, 50, season_type=None), id="no-order-key"),
    ],
)
def test_the_import_is_given_no_rows_of_a_series_it_cannot_read_alone(answer: SonarrItem | NoResource) -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    reads = _reads([_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    sonarr.series_return[1] = answer

    assert reads.read_episodes(1) is None
    assert reads.last_known_episodes(1) == []


def test_a_sonarr_that_sends_no_order_has_its_rows_taken_as_they_come() -> None:
    # The stored aired order numbers the two episodes the other way round. Nothing is tested against it.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    client = _skyhook(_ORDERS)
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    reads = _reads([_series(1, 50, season_type=None)], cache, client, sonarr=sonarr)

    rows = reads.read_episodes(1)
    view = reads.aired_series(1)

    assert rows == _DVD_ROWS
    assert view == AiredSeries(in_sonarr=tuple(_DVD_ROWS), stand_ins=())
    assert (sonarr.series_calls, client.fetch_calls) == ([], [])


@pytest.mark.parametrize(
    "stored",
    [pytest.param(True, id="aired-view"), pytest.param(False, id="no-view")],
)
def test_the_import_reads_a_series_numbered_by_another_order_as_sonarr_numbers_it(
    stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    if stored:
        cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
        cache.put_scene_mappings(50, stored_scene_mappings())
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    statuses = strat._reconciler._read(pending_import(series_id=1))

    assert statuses is not None
    assert statuses.snapshot.by_series[1].episodes.key_by_id == {11: (1, 1), 12: (1, 2)}


@pytest.mark.parametrize(
    ("series_id", "unread"),
    [
        pytest.param(1, False, id="stored"),
        pytest.param(2, True, id="no-row"),
        pytest.param(3, False, id="no-tvdb-id"),
        pytest.param(4, False, id="not-in-the-library"),
    ],
)
def test_a_library_series_with_no_stored_answer_is_unread(
    series_id: int, unread: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60), _series(3, 0)], cache)

    assert strat._episodes.metadata_unread([], [series_id]) is unread


@pytest.mark.parametrize(
    ("season_type", "read", "unread"),
    [
        pytest.param("dvd", False, True, id="another-order-never-read"),
        pytest.param("dvd", True, False, id="another-order-read-with-no-mappings"),
        pytest.param("official", False, False, id="aired-order"),
    ],
)
def test_a_series_in_another_order_is_unread_until_its_scene_mappings_were_read(
    season_type: str, read: bool, unread: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    if read:
        cache.put_scene_mappings(50, stored_scene_mappings())
    strat = _sync(monkeypatch, [_series(1, 50, season_type=season_type)], cache)

    assert strat._episodes.metadata_unread([], [1]) is unread


def test_orders_are_fetched_once_per_tvdb_series(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two Sonarr series of one TVDB series, one with no TVDB id, and one the Sonarr run's library fetch lacks.
    cache = FakeCacheStore()
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 50), _series(3, 0)], cache)

    assert strat._episodes.prefetch_orders([1, 2, 3, 9]) == 1
    assert cache.stats().tvdb_orders == 1


def test_sonarr_v5_counts_are_sent_with_the_fetch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # The stored answer is fresh, so only the count Sonarr's series row carries can make it due.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2}))
    strat = _sync(monkeypatch, [_series(1, 50, counts={"official": 3})], cache)

    assert strat._episodes.prefetch_orders([1]) == 1


@pytest.mark.parametrize(
    ("mode", "fetched"),
    [pytest.param(ImportWaitMode.BLOCKING, 1, id="waiting"), pytest.param(ImportWaitMode.OFF, 0, id="off")],
)
def test_a_pending_imports_series_is_fetched_while_imports_are_waited_on(
    mode: ImportWaitMode, fetched: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing in the scan needs series 2, but grab and import read the pending import's series anyway.
    pending = pending_import(infohash="h", series_id=2)
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat = _sync(monkeypatch, [_series(2, 60)], cache)
    strat._services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=mode))
    strat.start_of_run([])

    assert strat.prefetch_orders([]) == fetched


class _CountingPendingReads(FakeCacheStore):
    """A store counting the reads of every stored pending import."""

    pending_reads = 0

    @override
    def get_pending_import_rows(self, arr: Arr) -> dict[str, dict[str, Any]]:
        self.pending_reads += 1
        return super().get_pending_import_rows(arr)


def test_the_stored_pending_imports_are_loaded_once_for_the_start_of_a_run_and_its_prefetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = pending_import(infohash="h", series_id=2)
    cache = _CountingPendingReads(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat = _sync(monkeypatch, [_series(2, 60, season_type="dvd")], cache)
    strat._services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING))
    cache.pending_reads = 0

    strat.start_of_run([])
    strat.prefetch_orders([])
    strat.prefetch_scene_numbers([])

    assert cache.pending_reads == 1


def test_an_unreachable_source_leaves_a_new_series_without_orders_and_serves_the_stored_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Series 1's stored answer is due but SkyHook is unreachable: the stale answer still serves the grab.
    # Series 2 was never fetched, so its index has no orders.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, fresh_for=timedelta(days=-1)))
    client = FakeSkyHookClient(failing=frozenset({(50, "official"), (60, "official")}))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60)], cache, client)

    assert strat._episodes.prefetch_orders([1, 2]) == 0

    indexes = strat._series_indexes({1, 2})
    assert indexes[1].orders["dvd"].id_by_key == {(1, 1): 12, (1, 2): 11}
    assert indexes[2].orders == {}


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

_URL = "https://nyaa.si/1"


def _matched(season: int, episode: int, ep_id: int) -> MatchedEpisode:
    return MatchedEpisode(season_number=season, episode_number=episode, id=ep_id)


def _stored_as_one_season(aired: tuple[OrderRow, ...] = _TWO_SEASONS) -> FakeCacheStore:
    """A cache with TVDB series 50's orders stored: `aired`, and `_AS_ONE_SEASON` as its dvd order.

    TheXEM was read for the series, and maps none of its episodes.
    """

    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders({"official": aired, "dvd": _AS_ONE_SEASON}))
    cache.put_scene_mappings(50, stored_scene_mappings())
    return cache


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


def _synced_in_one_season(
    monkeypatch: pytest.MonkeyPatch, sonarr: FakeSonarrClient | None = None, series: SonarrSeries | None = None
) -> SonarrStrategy:
    """The strategy over the library read, the rows and the stored orders `_reads_in_one_season` has."""

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    return _sync(monkeypatch, [series or _in_one_season()], _stored_as_one_season(), sonarr=sonarr)


def _parsing(parses: Mapping[str, ParsedFileInfo]) -> FakeSonarrClient:
    """A Sonarr with `_ONE_SEASON_ROWS` that answers each file name with its parse."""

    return FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS), parse_fn=parses.get)


def _listing(*names: str) -> SeadexDict:
    """One release listing the files, each of 100 bytes."""

    return {"Group": rg_group({_URL: make_release(url=_URL, files=list(names), size=[100] * len(names))})}


def test_a_name_with_only_an_absolute_number_is_matched_to_the_aired_episode_with_that_number() -> None:
    # Sonarr counts the fifth episode as S01E05. As aired it is the second season's second.
    reads = _reads_in_one_season()
    info = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(5,),
        series_title="Show",
        matched_episodes=(_matched(1, 5, 15),),
        series_id=1,
    )

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 2, 15),))


def test_a_name_with_a_season_and_episode_is_matched_to_the_aired_episode_numbered_so() -> None:
    # Sonarr's order has no second season, so it matched the name to nothing.
    reads = _reads_in_one_season()
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 2, 15),))


def test_an_episode_sonarr_matched_by_its_own_order_is_not_passed_on() -> None:
    # S01E05 is an episode of Sonarr's one season. The aired first season has three.
    reads = _reads_in_one_season()
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(5,),
        series_title="Show",
        matched_episodes=(_matched(1, 5, 15),),
        series_id=1,
    )

    assert reads.aired_parse(info) == info.with_matched(())


def test_a_name_is_matched_to_an_aired_episode_sonarrs_order_has_no_row_for() -> None:
    reads = _reads_in_one_season()
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_title="Show", series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 4, -9007),))


def test_a_name_under_an_alternate_title_of_a_season_is_matched_within_that_season() -> None:
    # Read by the series' own name, the third episode is S01E03.
    reads = _reads_in_one_season(series=_in_one_season(_ZOKU))
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(3,), series_title="Show Zoku", series_id=1)

    assert reads.aired_parse(info) == info.with_matched((_matched(2, 3, 16),))


@pytest.mark.parametrize(
    "series_id",
    [
        pytest.param(2, id="aired-order"),
        pytest.param(9, id="not-in-the-library"),
        pytest.param(None, id="no-series"),
    ],
)
def test_a_parse_not_of_a_series_in_another_order_comes_back_as_it_is(series_id: int | None) -> None:
    reads = _reads([_in_one_season(), _series(2, 60)], _stored_as_one_season())
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(2,),
        series_title="Show 2",
        matched_episodes=(_matched(1, 2, 12),) if series_id is not None else (),
        series_id=series_id,
    )

    assert reads.aired_parse(info) is info


def test_a_parse_reads_the_same_whatever_the_run_read_of_its_series_before(monkeypatch: pytest.MonkeyPatch) -> None:
    # One run has read nothing of series 1, one checked a title of it, and one re-read it in an import poll after.
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)
    unread = _synced_in_one_season(monkeypatch)
    checked = _synced_in_one_season(monkeypatch)
    checked._episodes.title_episodes(1, 500, _SEASON_ONE)
    polled = _synced_in_one_season(monkeypatch)
    polled._episodes.title_episodes(1, 500, _SEASON_ONE)
    polled._reads.read_episodes(1)

    answers = [strat._reads.aired_parse(info) for strat in (unread, checked, polled)]

    assert answers == [info.with_matched((_matched(2, 2, 15),))] * 3


def test_a_series_matcher_is_built_once_for_a_read_of_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[AiredSeries] = []
    build = AiredMatcher.of

    def recording(series: AiredSeries, aliases: Iterable[AlternateTitle]) -> AiredMatcher:
        built.append(series)
        return build(series, aliases)

    monkeypatch.setattr(AiredMatcher, "of", recording)
    reads = _reads_in_one_season()
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)

    reads.aired_parse(info)
    reads.aired_parse(info)

    assert built == [reads.aired_series(1)]


def test_a_new_read_of_a_series_matches_a_name_to_its_new_rows() -> None:
    # An import poll re-reads the series after Sonarr gains a row for the seventh episode, where the stand-in was.
    sonarr = FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    reads = _reads_in_one_season(sonarr)
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_title="Show", series_id=1)
    before = reads.aired_parse(info)
    sonarr.episodes_return = [*_ONE_SEASON_ROWS, sonarr_ep(1, 7, ep_id=17, tvdb_id=9007)]

    reads.read_episodes(1)

    assert before == info.with_matched((_matched(2, 4, -9007),))
    assert reads.aired_parse(info) == info.with_matched((_matched(2, 4, 17),))


_FIFTH = "Show - 05.mkv"

_FIFTH_BY_SONARR = ParsedFileInfo(
    season_number=0,
    absolute_episode_numbers=(5,),
    series_title="Show",
    matched_episodes=(_matched(1, 5, 15),),
    series_id=1,
)
"""Sonarr's own parse of `_FIFTH`: the fifth episode of its one season."""


def test_a_listed_file_of_a_series_in_another_order_carries_its_aired_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing in the run has read the series yet.
    strat = _synced_in_one_season(monkeypatch, _parsing({_FIFTH: _FIFTH_BY_SONARR}))

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))),)}


def test_the_parse_cache_keeps_sonarrs_own_answer_for_a_series_in_another_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strat = _synced_in_one_season(monkeypatch, _parsing({_FIFTH: _FIFTH_BY_SONARR}))

    strat._parse.parsed_files(_listing(_FIFTH))

    row = strat._parse.cache_store.get_sonarr_parse(_FIFTH)
    assert row is not None
    assert row["parse"] == parse_to_json(_FIFTH_BY_SONARR)


def test_a_cached_parse_is_matched_as_aired_again_without_asking_sonarr(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR})
    strat = _synced_in_one_season(monkeypatch, sonarr)
    strat._parse.parsed_files(_listing(_FIFTH))

    ((again,),) = strat._parse.parsed_files(_listing(_FIFTH)).values()

    assert again.parse == _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))
    assert sonarr.parse_calls == [_FIFTH]


_FIFTH_ON_NO_EPISODE = _FIFTH_BY_SONARR.with_matched(())
"""Sonarr's parse of `_FIFTH` when it matches the name to the series and to no episode of it."""

_EITHER_PARSE = pytest.mark.parametrize(
    "by_sonarr",
    [
        pytest.param(_FIFTH_BY_SONARR, id="matched-episodes"),
        pytest.param(_FIFTH_ON_NO_EPISODE, id="no-matched-episode"),
    ],
)


def _asked_to_parse(
    monkeypatch: pytest.MonkeyPatch, cache: FakeCacheStore, series: SonarrSeries, by_sonarr: ParsedFileInfo
) -> list[str]:
    """The names one run asks Sonarr to parse for a release listing `_FIFTH`, over the parses `cache` holds.

    The run's library read has `series` alone, and Sonarr answers `_FIFTH` with `by_sonarr`.
    """

    sonarr = _parsing({_FIFTH: by_sonarr})
    strat = _sync(monkeypatch, [series], cache, sonarr=sonarr)
    strat._parse.parsed_files(_listing(_FIFTH))
    return sonarr.parse_calls


@_EITHER_PARSE
def test_a_parse_stored_under_another_order_is_asked_for_again_once_its_series_is_aired(
    monkeypatch: pytest.MonkeyPatch, by_sonarr: ParsedFileInfo
) -> None:
    cache = _stored_as_one_season()
    _asked_to_parse(monkeypatch, cache, _in_one_season(), by_sonarr)

    assert _asked_to_parse(monkeypatch, cache, _series(1, 50), by_sonarr) == [_FIFTH]


def test_a_parse_stored_with_no_order_is_asked_for_again_once_its_series_is_in_another_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _stored_as_one_season()
    _asked_to_parse(monkeypatch, cache, _series(1, 50), _FIFTH_BY_SONARR)

    assert _asked_to_parse(monkeypatch, cache, _in_one_season(), _FIFTH_BY_SONARR) == [_FIFTH]


@_EITHER_PARSE
def test_a_parse_stored_under_another_order_is_not_asked_for_again_while_its_series_stays_in_it(
    monkeypatch: pytest.MonkeyPatch, by_sonarr: ParsedFileInfo
) -> None:
    cache = _stored_as_one_season()
    _asked_to_parse(monkeypatch, cache, _in_one_season(), by_sonarr)

    assert _asked_to_parse(monkeypatch, cache, _in_one_season(), by_sonarr) == []


_SIXTH = "Show - 06.mkv"


def _with_two_sixth_episodes(monkeypatch: pytest.MonkeyPatch) -> SonarrStrategy:
    """`_synced_in_one_season` where TVDB's aired order counts the special as a sixth episode too.

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
    return _sync(monkeypatch, [_in_one_season()], cache, sonarr=_parsing({_SIXTH: by_sonarr}))


def test_a_name_an_aired_sonarr_answers_with_an_error_is_an_unread_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _with_two_sixth_episodes(monkeypatch)

    files = strat._parse.parsed_files(_listing(_SIXTH))

    assert files == {_URL: (ListedFile(_SIXTH, 100, None),)}


def test_the_debug_log_names_a_file_an_aired_sonarr_answers_with_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    strat = _with_two_sixth_episodes(monkeypatch)
    strat._parse.logger = debug_logger("pearlarr-aired-parse")

    with caplog.at_level("DEBUG"):
        strat._parse.parsed_files(_listing(_SIXTH))

    assert [record.message for record in caplog.records] == [
        f"No parse for {_SIXTH}: two aired episodes of its series share a number it is matched by"
    ]


def test_a_name_an_alternate_title_may_not_count_for_has_no_matched_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
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
    strat = _synced_in_one_season(monkeypatch, _parsing({name: by_sonarr}), _in_one_season(limited))

    files = strat._parse.parsed_files(_listing(name))

    assert files == {_URL: (ListedFile(name, 100, by_sonarr.with_matched(())),)}


def test_a_name_sonarr_searches_for_a_specials_title_has_no_matched_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sonarr looks for a special's title in the name of an episode 0 of a season, and the parse doesn't keep the name.
    name = "Show S01E00 Recap.mkv"
    by_sonarr = ParsedFileInfo(
        season_number=1,
        episode_numbers=(0,),
        series_title="Show",
        matched_episodes=(_matched(0, 1, 20),),
        series_id=1,
    )
    strat = _synced_in_one_season(monkeypatch, _parsing({name: by_sonarr}))

    files = strat._parse.parsed_files(_listing(name))

    assert files == {_URL: (ListedFile(name, 100, by_sonarr.with_matched(())),)}


@pytest.mark.parametrize(
    "orders",
    [
        pytest.param(None, id="aired-order-not-stored"),
        pytest.param({"official": _TWO_SEASONS, "dvd": _AS_ONE_SEASON}, id="scene-mappings-never-read"),
    ],
)
def test_a_listed_file_of_a_series_with_no_aired_view_yet_has_no_matched_episodes(
    orders: Mapping[str, tuple[OrderRow, ...]] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    if orders is not None:
        cache.put_tvdb_orders(50, stored_orders(orders))
    strat = _sync(monkeypatch, [_in_one_season()], cache, sonarr=_parsing({_FIFTH: _FIFTH_BY_SONARR}))

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, _FIFTH_BY_SONARR.with_matched(())),)}


def test_a_listed_file_of_a_series_whose_episodes_fail_to_read_is_an_unread_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR})
    sonarr.episodes_return = None
    strat = _synced_in_one_season(monkeypatch, sonarr)

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, None),)}


def test_a_series_whose_episodes_fail_to_read_is_asked_for_once_however_many_files_it_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR, _SIXTH: _FIFTH_BY_SONARR})
    sonarr.episodes_return = None
    strat = _synced_in_one_season(monkeypatch, sonarr)
    asked_before = len(sonarr.episodes_calls)

    files = strat._parse.parsed_files(_listing(_FIFTH, _SIXTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, None), ListedFile(_SIXTH, 100, None))}
    assert sonarr.episodes_calls[asked_before:] == [1]


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
    series: list[SonarrItem], pending: PendingImport, sonarr: FakeSonarrClient | None = None
) -> tuple[PendingFollower, FakeCacheStore]:
    """The follower of a run after its library read of `series`, `pending` stored and its title cached.

    TVDB series 50 has `_ORDERS` stored, and TheXEM was read for it and maps none of its episodes.
    """

    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    cache.put_scene_mappings(50, stored_scene_mappings())
    cache.update_cached_title(Arr.SONARR, PENDING_IMPORT_ANILIST_ID, {"updated_at": PENDING_IMPORT_STAMP})
    return make_pending_follower(cache, _reads(series, cache, sonarr=sonarr)), cache


def _title_cached(cache: FakeCacheStore) -> bool:
    """Whether the claiming title has a cache stamp."""

    cached = cache.get_cached_title(Arr.SONARR, PENDING_IMPORT_ANILIST_ID)
    return cached is not None and cached.updated_at is not None


def test_a_pending_import_stored_without_tvdb_ids_on_a_series_in_another_order_is_dropped_with_one_warning() -> None:
    recording = install_recording_hub()
    series: list[SonarrItem] = [_series(1, 50, season_type="dvd")]
    follower, cache = _at_run_start(series, _WITHOUT_TVDB_IDS, FakeSonarrClient(episodes=list(_DVD_ROWS)))

    follower.begin_run(waits_on_imports=True)

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
    series: list[SonarrItem] = [_series(1, 50, season_type="dvd")]
    follower, cache = _at_run_start(series, pending, FakeSonarrClient(episodes=list(_DVD_ROWS)))

    follower.begin_run(waits_on_imports=True)

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}
    assert _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == []


def test_a_pending_import_stored_without_tvdb_ids_takes_them_from_the_rows_of_its_aired_series() -> None:
    recording = install_recording_hub()
    series: list[SonarrItem] = [_series(1, 50)]
    follower, cache = _at_run_start(series, _WITHOUT_TVDB_IDS)

    follower.begin_run(waits_on_imports=True)

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
        pytest.param(_series(1, 60, season_type="dvd"), _DVD_ROWS, id="another-order-with-no-orders-stored"),
    ],
)
def test_a_pending_import_stored_without_tvdb_ids_is_left_as_stored_when_its_series_has_no_aired_rows_to_read(
    series: SonarrSeries, rows: list[SonarrEpisode] | None
) -> None:
    sonarr = FakeSonarrClient()
    sonarr.episodes_return = rows
    follower, cache = _at_run_start([series], _WITHOUT_TVDB_IDS, sonarr)

    follower.begin_run(waits_on_imports=True)

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": _WITHOUT_TVDB_IDS.to_json()}
    assert _title_cached(cache)


def test_a_pending_import_stored_without_tvdb_ids_is_left_as_stored_while_its_series_scene_mappings_are_unread() -> (
    None
):
    # TVDB series 60 has its orders stored, and TheXEM was never read for it.
    series: list[SonarrItem] = [_series(1, 60, season_type="dvd")]
    follower, cache = _at_run_start(series, _WITHOUT_TVDB_IDS, FakeSonarrClient(episodes=list(_DVD_ROWS)))
    cache.put_tvdb_orders(60, stored_orders(_ORDERS))

    follower.begin_run(waits_on_imports=True)

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": _WITHOUT_TVDB_IDS.to_json()}
    assert _title_cached(cache)


def test_a_two_series_pending_import_stored_without_tvdb_ids_waits_for_a_read_of_both_series() -> None:
    pending = two_claim_pending_import(
        series_ids=(1, 2),
        windows=((11,), (12,)),
        infohash="h",
        file_episode_map={"show - 01 [1080p].mkv": (11,)},
        tvdb_ids=None,
    )
    series: list[SonarrItem] = [_series(1, 50), _series(2, 60)]
    follower, cache = _at_run_start(series, pending, _SeriesTwoUnread(episodes=list(_ROWS)))

    follower.begin_run(waits_on_imports=True)

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}


def test_a_pending_import_stored_without_tvdb_ids_holding_an_id_sonarr_sends_no_row_for_is_dropped_with_one_warning() -> (
    None
):
    recording = install_recording_hub()
    pending = replace(_WITHOUT_TVDB_IDS, file_episode_map={"show - 01 [1080p].mkv": (13,)})
    series: list[SonarrItem] = [_series(1, 50)]
    follower, cache = _at_run_start(series, pending)

    follower.begin_run(waits_on_imports=True)

    assert cache.get_pending_import_rows(Arr.SONARR) == {}
    assert not _title_cached(cache)
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Pending import Show · SubGroup dropped: Sonarr no longer has an episode it was matched to - checking "
            "its titles again"
        )
    ]


def test_a_pending_import_stored_without_tvdb_ids_is_left_as_stored_while_imports_are_not_waited_on() -> None:
    series: list[SonarrItem] = [_series(1, 50)]
    follower, cache = _at_run_start(series, _WITHOUT_TVDB_IDS)

    follower.begin_run(waits_on_imports=False)

    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": _WITHOUT_TVDB_IDS.to_json()}


def test_a_stored_pending_import_is_not_followed_under_the_rows_of_a_series_the_run_holds() -> None:
    # Sonarr went to the dvd order after the library read and is part way through renumbering: TVDB episode 9002
    # is on none of the rows it sent. Followed under them, the file on that episode would leave the file map.
    pending = pending_import(
        infohash="h",
        series_id=1,
        file_episode_map={"show - 01 [1080p].mkv": (11,), "show - 02 [1080p].mkv": (12,)},
        tvdb_ids={11: 9001, 12: 9002},
    )
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    sonarr = FakeSonarrClient(episodes=[_ROWS[0]])
    reads = _reads([_series(1, 50)], cache, sonarr=sonarr)
    sonarr.series_return[1] = _series(1, 50, season_type="dvd")
    follower = make_pending_follower(cache, reads)
    # A title's check asks for the index of every claimed series first, which is where the run comes to hold this one.
    assert reads.series_index(1) is None

    followed = follower.followed({"h": pending})

    assert followed == {}
    assert cache.get_pending_import_rows(Arr.SONARR) == {"h": pending.to_json()}


def test_the_rows_a_series_view_is_built_from_are_as_sonarr_numbers_them() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    cache.put_scene_mappings(50, stored_scene_mappings())
    reads = _reads([_series(1, 50, season_type="dvd")], cache, sonarr=FakeSonarrClient(episodes=list(_DVD_ROWS)))

    assert reads.viewed_episodes(1) == _DVD_ROWS


def test_a_series_with_no_view_has_no_rows_a_view_is_built_from() -> None:
    # Sonarr numbers the series by its dvd order, and no TVDB orders are stored for it.
    reads = _reads([_series(1, 50, season_type="dvd")], FakeCacheStore(), sonarr=FakeSonarrClient(episodes=_DVD_ROWS))

    assert reads.viewed_episodes(1) is None
    assert reads.last_known_episodes(1) == _DVD_ROWS


_SEASON_TWO_AIRED = {"official": (*_ORDERS["official"], OrderRow(2, 1, 9003, 3)), "dvd": _DVD}
"""A stored answer whose official order has a second season `_DVD` has no episode of."""

_SEASON_TITLES = {500: "First", 600: "Second", 700: "Third"}
"""The AniList titles of seasons 1, 2 and 3 of TVDB series 50. Neither stored order has a third season."""

_NO_EPISODE_IN_THE_ORDER = (
    "Sonarr's DVD order of the series has none of Second's episodes - not checked again until the series' order, "
    "its episode count or the title's SeaDex entry changes"
)


def _entry(anilist_id: int) -> EntryRecord:
    """The title's SeaDex entry, which lists no release."""

    return make_entry_record(anilist_id=anilist_id, url=f"https://releases.moe/{anilist_id}")


def _library_row(season_type: str | None, total: int | None) -> SonarrSeries:
    """Series 1 of TVDB series 50 as a library read sends it, with `total` episodes by its `statistics`.

    With `season_type=None` the row has no `seasonType` key, and with `total=None` no `statistics`.
    """

    raw: dict[str, object] = {"id": 1, "title": "Show", "tvdbId": 50}
    if season_type is not None:
        raw["seasonType"] = season_type
    if total is not None:
        raw["statistics"] = {"seasonCount": 1, "episodeCount": total, "totalEpisodeCount": total}
    return SonarrSeries.model_validate(raw)


def _run(cache: FakeCacheStore, library: SonarrSeries, xem: FakeXemClient | None = None) -> RecordingHub:
    """One whole Sonarr run over `cache` with activity detection off, and what it printed.

    Sonarr lists `library` alone, with the season's two rows numbered by the order it names. Each of
    `_SEASON_TITLES` has its SeaDex entry (`_entry`), and TVDB series 50 has `_SEASON_TWO_AIRED` stored. TheXEM
    is `xem`, one that maps no episode of any series when left out.
    """

    cache.put_tvdb_orders(50, stored_orders(_SEASON_TWO_AIRED))
    rows = _DVD_ROWS if library.seasonType == "dvd" else _ROWS
    sonarr = FakeSonarrClient(all_series=[library], episodes=list(rows))
    seadex = FakeSeaDexSource({anilist_id: _entry(anilist_id) for anilist_id in _SEASON_TITLES})
    config = make_config(url="http://sonarr", api_key="key", sleep_time=0, detect_arr_activity=False)
    deps = make_run_deps(config=config, cache_store=cache, seadex=seadex, xem_client=xem)
    seasons = MappingSources(
        anime={
            name: {"anilist_id": anilist_id, "tvdb_id": 50, "tvdb_season": season}
            for season, (anilist_id, name) in enumerate(_SEASON_TITLES.items(), start=1)
        },
        anidb=False,
        anibridge=False,
    )
    deps = replace(
        deps, resolver=MappingResolver(refresh_days=1, ignore_anilist_ids=set(), web=deps.web, sources=seasons)
    )
    for anilist_id, name in _SEASON_TITLES.items():
        deps.anilist.answers[anilist_id] = anilist_body(anilist_id, name)
    services = ArrRunServices(deps, Arr.SONARR)
    recording = install_recording_hub()
    ArrRunLoop(deps, services).run_arr(
        SonarrStrategy(deps, services, sonarr_client=sonarr), item_id=None, dry_run=False, boot=BootFlow()
    )
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
    cache = FakeCacheStore()
    entry = _entry(500)
    cache.update_cached_title(
        Arr.SONARR, 500, {"updated_at": entry.updated_at, "name": "First", "url": entry.url, "coverage": "S01"}
    )

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
    assert "Second" not in _no_episode_rows(recording)


def test_a_title_sonarrs_order_has_no_episode_of_is_checked_again_when_the_series_episode_count_changes() -> None:
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2))

    recording = _run(cache, _library_row("dvd", 3))

    assert diagnostic_messages(recording, Severity.WARNING) == [_NO_EPISODE_IN_THE_ORDER]


def test_a_title_with_no_aired_episode_prints_its_row_on_every_run_and_is_not_cached() -> None:
    cache = FakeCacheStore()

    printed = [_no_episode_rows(_run(cache, _library_row("dvd", 2))) for _ in range(2)]

    assert ["Third" in rows for rows in printed] == [True, True]
    assert cache.get_cached_title(Arr.SONARR, 700) is None


_THEXEM_STEP = "Fetching TheXEM scene numbers"


def _boot_steps(recording: RecordingHub) -> dict[str, str | None]:
    """The detail each boot step of the run ended on, by the step's label, in the order the steps ran."""

    return {step.label: step.detail for step in recording.of_type(BootStepFinished)}


def test_a_run_reads_the_scene_mappings_of_its_series_in_another_order_after_the_tvdb_orders() -> None:
    xem = FakeXemClient()

    recording = _run(FakeCacheStore(), _library_row("dvd", 2), xem)

    steps = _boot_steps(recording)
    assert xem.asked == [50]
    assert list(steps)[-2:] == ["Fetching TVDB orders", _THEXEM_STEP]
    assert steps[_THEXEM_STEP] == "1 series"


def test_a_run_over_a_library_all_in_aired_order_asks_thexem_nothing_and_has_no_boot_step_for_it() -> None:
    xem = FakeXemClient()

    recording = _run(FakeCacheStore(), _library_row("official", 2), xem)

    assert xem.asked == []
    assert _THEXEM_STEP not in _boot_steps(recording)


def test_an_unreachable_thexem_shows_on_its_own_boot_step_and_not_on_the_tvdb_orders_one() -> None:
    recording = _run(FakeCacheStore(), _library_row("dvd", 2), FakeXemClient(failing=frozenset({50})))

    steps = _boot_steps(recording)
    assert (steps["Fetching TVDB orders"], steps[_THEXEM_STEP]) == ("cached", "unreachable")


def test_a_run_that_found_thexem_unreachable_stores_the_selection_hash_and_leaves_its_titles_for_the_next_run() -> None:
    # The unread series' titles are not cached, so the next run checks them whatever this one committed.
    cache = FakeCacheStore()
    _run(cache, _library_row("dvd", 2), FakeXemClient(failing=frozenset({50})))
    hash_stored = cache.selection_changed(Arr.SONARR, "some-other-digest")

    recording = _run(cache, _library_row("dvd", 2))

    assert hash_stored is True
    assert _checked(recording) == [500]
