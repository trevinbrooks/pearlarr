# pyright: strict
# pyright: reportPrivateUsage=false
# The tests read the strategy's private collaborators (_episodes / _reconciler),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Each series' TVDB orders through the Sonarr strategy: what it fetches, and what grab and import are served."""

from collections.abc import Mapping
from datetime import timedelta

import pytest

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_models import SonarrEpisode, SonarrItem, SonarrSeries
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.releases import EpisodeKey
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import OrderRow
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    FakeCacheStore,
    ScriptedAniListClient,
    make_anilist_gateway,
    make_sonarr_sync,
    pending_import,
    sonarr_ep,
    stand_in_ep,
    stored_orders,
)
from .fakes import FakeSkyHookClient, FakeSonarrClient

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
    series_id: int, tvdb_id: int, *, season_type: str = "official", counts: Mapping[str, int] | None = None
) -> SonarrSeries:
    """A Sonarr series row as the wire sends it, listing `counts` as v5 does (none, as on v4, when None)."""

    raw: dict[str, object] = {
        "id": series_id,
        "title": f"Show {series_id}",
        "tvdbId": tvdb_id,
        "seasonType": season_type,
    }
    if counts is not None:
        raw["seasonTypes"] = [{"type": name, "episodeCount": count} for name, count in counts.items()]
    return SonarrSeries.model_validate(raw)


def _sync(
    monkeypatch: pytest.MonkeyPatch,
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
) -> SonarrStrategy:
    """The Sonarr strategy after the Sonarr run's library fetch of `series`, its orders fetched through `client`.

    Every series' episodes are `_ROWS` unless `sonarr` serves others. Its AniList knows no title and is never
    reached over the network.
    """

    strat = make_sonarr_sync(
        sonarr=sonarr or FakeSonarrClient(episodes=list(_ROWS)), cache_store=cache, skyhook_client=client
    )

    def _library() -> list[SonarrItem]:
        return series

    monkeypatch.setattr(strat._episodes, "get_all_sonarr_series", _library)
    monkeypatch.setattr(strat._episodes, "_anilist", make_anilist_gateway(ScriptedAniListClient(title=None)))
    strat.get_items()
    return strat


def _numbered(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, EpisodeKey]]:
    """Each episode's id and its `(season, episode)`, in the order given."""

    return [(ep.id, EpisodeKey(ep.season_number or 0, ep.episode_number or 0)) for ep in episodes]


@pytest.mark.parametrize(
    ("series_id", "season_type", "other_orders"),
    [
        pytest.param(1, "official", {"dvd"}, id="official"),
        # A null or junk `seasonType` names no order.
        pytest.param(1, "", set[str](), id="no-order"),
        pytest.param(2, "", set[str](), id="not-in-the-library"),
    ],
)
def test_a_series_not_numbered_by_another_order_is_served_sonarrs_rows_as_sent(
    series_id: int, season_type: str, other_orders: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    strat = _sync(monkeypatch, [_series(1, 50, season_type=season_type)], cache)

    view = strat._episodes.aired_series(series_id)
    index = strat._episodes.series_index(series_id)

    assert isinstance(view, AiredSeries)
    assert (view.order, view.aired) == (season_type, tuple(_ROWS))
    assert index is not None
    assert (set(index.by_id), set(index.orders)) == ({11, 12}, other_orders)


def test_a_series_numbered_by_another_order_is_served_as_aired_with_its_stored_orders_read_onto_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    view = strat._episodes.aired_series(1)
    index = strat._episodes.series_index(1)

    assert isinstance(view, AiredSeries)
    assert view.order == "dvd"
    assert _numbered(view.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert index is not None
    assert index.id_by_key == {(1, 1): 12, (1, 2): 11, (1, 3): -9003}
    assert {name: order.id_by_key for name, order in index.orders.items()} == {"dvd": {(1, 1): 11, (1, 2): 12}}


def test_a_titles_episodes_of_a_series_numbered_by_another_order_are_its_aired_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    title = strat._episodes.title_episodes(1, 500, _SEASON_ONE)

    assert not isinstance(title, NoView)
    assert title.order == "dvd"
    assert _numbered(title.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert _numbered(title.in_sonarr) == [(12, (1, 1)), (11, (1, 2))]
    assert title.index().by_id[-9003] == stand_in_ep(1, 3, tvdb_id=9003).model_copy(
        update={"absolute_episode_number": 3}
    )


def test_a_series_numbered_by_another_order_with_nothing_stored_has_no_view(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], FakeCacheStore(), sonarr=sonarr)

    assert strat._episodes.aired_series(1) is NoView.ORDERS_UNREAD
    assert strat._episodes.series_index(1) is None
    assert strat._episodes.title_episodes(1, 500, _SEASON_ONE) is NoView.ORDERS_UNREAD
    assert strat._series_indexes({1}) == {}


def test_a_series_index_is_built_once_for_a_read_of_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore())

    first = strat._episodes.series_index(1)

    assert first is not None
    assert strat._episodes.series_index(1) is first


def test_a_new_read_of_a_series_replaces_its_view_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    before = strat._episodes.series_index(1)
    third = sonarr_ep(1, 3, ep_id=13, tvdb_id=9003)
    sonarr.episodes_return = [*_ROWS, third]

    strat._episodes.read_episodes(1)

    view = strat._episodes.aired_series(1)
    index = strat._episodes.series_index(1)
    assert before is not None
    assert set(before.by_id) == {11, 12}
    assert isinstance(view, AiredSeries)
    assert view.aired == (*_ROWS, third)
    assert index is not None
    assert set(index.by_id) == {11, 12, 13}


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
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    statuses = strat._reconciler._read_statuses(pending_import(series_id=1), [11])

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

    assert strat._episodes.orders_unread([series_id]) is unread


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

    assert strat.prefetch_orders([]) == fetched


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
