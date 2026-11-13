# pyright: strict
# pyright: reportPrivateUsage=false
# The tests read the strategy's private collaborators (_episodes / _reconciler),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Each series' TVDB orders through the Sonarr strategy: what it fetches, and what grab and import are served."""

from collections.abc import Mapping
from datetime import timedelta

import pytest

from pearlarr.arr_models import SonarrItem, SonarrSeries
from pearlarr.config import Arr
from pearlarr.reporter import RunContext
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.tvdb_orders import NO_ORDERING, OrderRow, SeriesOrdering
from pearlarr.wait_types import ImportWaitMode

from .builders import FakeCacheStore, make_sonarr_sync, pending_import, sonarr_ep, stored_orders
from .fakes import FakeSkyHookClient, FakeSonarrClient

_DVD = (OrderRow(1, 1, 9002, None), OrderRow(1, 2, 9001, None))
"""The dvd order of a two-episode season that swaps its episodes."""

_ORDERS = {"official": (OrderRow(1, 1, 9001, 1), OrderRow(1, 2, 9002, 2)), "dvd": _DVD}
"""A stored record's orders: official and `_DVD`."""


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
) -> SonarrSync:
    """The Sonarr strategy after the run's library read of `series`, its orders fetched through `client`."""

    strat = make_sonarr_sync(
        sonarr=FakeSonarrClient(
            episodes=[sonarr_ep(1, 1, ep_id=11, tvdb_id=9001), sonarr_ep(1, 2, ep_id=12, tvdb_id=9002)]
        ),
        cache_store=cache,
        skyhook_client=client,
    )

    def _library() -> list[SonarrItem]:
        return series

    monkeypatch.setattr(strat._episodes, "get_all_sonarr_series", _library)
    strat.get_items()
    return strat


@pytest.mark.parametrize(
    ("season_type", "expected"),
    [
        pytest.param("official", SeriesOrdering({"dvd": _DVD}, None), id="official"),
        # The index isn't translated yet, so another order can't be read onto a series numbered by one.
        pytest.param("dvd", NO_ORDERING, id="dvd-ordered"),
    ],
)
def test_a_series_is_served_its_stored_orders_only_when_sonarr_numbers_it_by_official(
    season_type: str, expected: SeriesOrdering, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type=season_type)], cache)

    assert strat._episodes.orders_for(1) == expected
    assert strat._episodes.orders_for(2) == NO_ORDERING


@pytest.mark.parametrize(
    ("season_type", "aired"),
    [
        pytest.param("official", True, id="official"),
        pytest.param("dvd", False, id="dvd-ordered"),
        # A null or junk `seasonType` names no order.
        pytest.param("", False, id="no-order"),
    ],
)
def test_a_series_numbers_aired_only_when_sonarr_numbers_it_by_official(
    season_type: str, aired: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    strat = _sync(monkeypatch, [_series(1, 50, season_type=season_type)], FakeCacheStore())

    assert strat._episodes.numbers_aired(1) is aired
    assert strat._episodes.numbers_aired(2) is False


@pytest.mark.parametrize(
    ("series_id", "unread"),
    [
        pytest.param(1, False, id="stored"),
        pytest.param(2, True, id="no-row"),
        pytest.param(3, False, id="no-tvdb-id"),
        pytest.param(4, False, id="not-in-the-library"),
    ],
)
def test_a_library_series_with_no_stored_record_is_unread(
    series_id: int, unread: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60), _series(3, 0)], cache)

    assert strat._episodes.orders_unread([series_id]) is unread


def test_orders_are_fetched_once_per_tvdb_series(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two Sonarr series of one TVDB series, one with no TVDB id, and one this run's library read lacks.
    cache = FakeCacheStore()
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 50), _series(3, 0)], cache)

    assert strat._episodes.prefetch_orders([1, 2, 3, 9]) == 1
    assert cache.stats().tvdb_orders == 1


def test_sonarr_v5_counts_ride_the_fetch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # The record is fresh, so only the count Sonarr's series row carries can make it due.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2}))
    strat = _sync(monkeypatch, [_series(1, 50, counts={"official": 3})], cache)

    assert strat._episodes.prefetch_orders([1]) == 1


@pytest.mark.parametrize(
    ("mode", "fetched"),
    [pytest.param(ImportWaitMode.BLOCKING, 1, id="waiting"), pytest.param(ImportWaitMode.OFF, 0, id="off")],
)
def test_a_stored_records_series_is_fetched_while_imports_are_waited_on(
    mode: ImportWaitMode, fetched: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing in the scan needs series 2, but grab and import read the record's series anyway.
    record = pending_import(infohash="h", series_id=2)
    cache = FakeCacheStore(pending={str(Arr.SONARR): {record.infohash: record.to_json()}})
    strat = _sync(monkeypatch, [_series(2, 60)], cache)
    strat._services.begin_run(RunContext(arr=Arr.SONARR, import_wait_mode=mode))

    assert strat.prefetch_orders([]) == fetched


def test_an_outage_leaves_a_new_series_without_orders_and_serves_the_stored_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Series 1's record is due but SkyHook is down: the stale record still serves grab and import. Series 2 was
    # never fetched, so its index has no orders.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, fresh_for=timedelta(days=-1)))
    client = FakeSkyHookClient(failing=frozenset({(50, "official"), (60, "official")}))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60)], cache, client)

    assert strat._episodes.prefetch_orders([1, 2]) == 0

    indexes = strat._series_indexes({1, 2})
    assert indexes[1].orders["dvd"].id_by_key == {(1, 1): 12, (1, 2): 11}
    assert indexes[2].orders == {}
    seed = strat._reconciler._read_seed(pending_import(series_id=1), [11])
    assert seed is not None
    assert seed.snapshot.by_series[1].episodes.orders["dvd"].keys == {(1, 1), (1, 2)}
