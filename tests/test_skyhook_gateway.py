# pyright: strict
"""The TVDB order cache: which series refetch, what a fetch stores, and what the readers are served."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pearlarr.cache import CacheStore
from pearlarr.seadex_types import SkyHookReply
from pearlarr.skyhook_gateway import ACTIVE_FRESH_DAYS, QUIET_FRESH_DAYS, OrderRequest, SkyHookGateway
from pearlarr.stamps import parse_stamp, stamp_of
from pearlarr.tvdb_orders import NO_ORDERS, OrderEpisode, StoredOrder, StoredOrders

from .builders import FakeCacheStore
from .fakes import FakeSkyHookClient
from .http_mock import skyhook_fixture

_SERIES = 112961


def _fixture(name: str) -> SkyHookReply:
    return SkyHookReply.model_validate(skyhook_fixture(name))


_OFFICIAL = _fixture("tvdb_112961_official.json")
_DVD = _fixture("tvdb_112961_dvd.json")
_REPLIES = {(_SERIES, "official"): _OFFICIAL, (_SERIES, "dvd"): _DVD}


def _gateway(
    replies: Mapping[tuple[int, str], SkyHookReply] = _REPLIES,
    *,
    cache: FakeCacheStore | None = None,
    failing: frozenset[tuple[int, str]] = frozenset(),
) -> SkyHookGateway:
    return SkyHookGateway(client=FakeSkyHookClient(replies, failing=failing), cache_store=cache or FakeCacheStore())


def _stored(*, fresh_for: timedelta, counts: Mapping[str, int | None] | None = None) -> StoredOrders:
    """A record of the series fetched a day ago, its official and dvd orders counted by `counts`."""

    counts = counts or {}
    now = datetime.now()
    row = (OrderEpisode(1, 1, 2273831, None),)
    return StoredOrders(
        fetched_at=stamp_of(now - timedelta(days=1)),
        fresh_until=stamp_of(now + fresh_for),
        orders={name: StoredOrder(row, counts.get(name)) for name in ("official", "dvd")},
    )


def _fetches(gateway: SkyHookGateway, counts: Mapping[str, int] | None = None) -> int:
    return gateway.prefetch([OrderRequest(_SERIES, counts)], preview=True)


def test_a_first_fill_stores_every_listed_order_official_with_its_absolutes() -> None:
    gateway = _gateway()

    assert _fetches(gateway, {"official": 4, "dvd": 4}) == 1

    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert stored.orders == {
        "official": StoredOrder(
            (
                OrderEpisode(0, 1, 2273871, None),
                OrderEpisode(1, 1, 2273831, 1),
                OrderEpisode(1, 2, 2273841, 2),
                OrderEpisode(1, 3, 2273851, 3),
                OrderEpisode(1, 4, 2273861, 4),
            ),
            4,
        ),
        "dvd": StoredOrder(
            (
                OrderEpisode(1, 1, 2273831, None),
                OrderEpisode(1, 1, 2273851, None),
                OrderEpisode(1, 2, 2273841, None),
                OrderEpisode(1, 2, 2273861, None),
            ),
            4,
        ),
    }


def _recent(reply: SkyHookReply) -> SkyHookReply:
    """The reply with one more episode, aired five days ago."""

    assert reply.episodes is not None
    aired = datetime.now(UTC) - timedelta(days=5)
    newest = reply.episodes[-1].model_copy(update={"episode_number": 5, "tvdb_id": 1, "air_date_utc": aired})
    return reply.model_copy(update={"episodes": (*reply.episodes, newest)})


@pytest.mark.parametrize(
    ("official", "fresh_days"),
    [
        pytest.param(_OFFICIAL, QUIET_FRESH_DAYS, id="ended-and-quiet"),
        pytest.param(_OFFICIAL.model_copy(update={"status": "Continuing"}), ACTIVE_FRESH_DAYS, id="continuing"),
        pytest.param(_recent(_OFFICIAL), ACTIVE_FRESH_DAYS, id="ended-but-aired-lately"),
    ],
)
def test_a_record_stays_fresh_longer_once_the_series_is_quiet(official: SkyHookReply, fresh_days: int) -> None:
    gateway = _gateway({(_SERIES, "official"): official, (_SERIES, "dvd"): _DVD})

    _fetches(gateway)

    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert parse_stamp(stored.fresh_until) - parse_stamp(stored.fetched_at) == timedelta(days=fresh_days)


@pytest.mark.parametrize(
    ("fresh_for", "fetched"),
    [pytest.param(timedelta(days=-1), 1, id="past-fresh-until"), pytest.param(timedelta(days=1), 0, id="fresh")],
)
def test_a_record_refetches_once_its_fresh_until_passes(fresh_for: timedelta, fetched: int) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(_SERIES, _stored(fresh_for=fresh_for))

    assert _fetches(_gateway(cache=cache)) == fetched


@pytest.mark.parametrize(
    ("counts", "fetched"),
    [
        pytest.param({"official": 4, "dvd": 5}, 1, id="count-moved"),
        pytest.param({"official": 4}, 1, id="order-dropped"),
        pytest.param({"official": 4, "dvd": 4, "alternate": 4}, 1, id="order-appeared"),
        # The stored record has one row per order: what Sonarr counted at fetch time is the bar, not our rows.
        pytest.param({"official": 4, "dvd": 4}, 0, id="count-as-stored"),
    ],
)
def test_a_fresh_record_refetches_when_sonarr_counts_move(counts: dict[str, int], fetched: int) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(_SERIES, _stored(fresh_for=timedelta(days=1), counts={"official": 4, "dvd": 4}))

    assert _fetches(_gateway(cache=cache), counts) == fetched


def test_a_series_skyhook_does_not_know_is_stored_without_orders_and_waits() -> None:
    gateway = _gateway({})

    assert _fetches(gateway) == 1
    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert stored.orders == {}
    assert parse_stamp(stored.fresh_until) - parse_stamp(stored.fetched_at) == timedelta(days=ACTIVE_FRESH_DAYS)
    # Every order Sonarr lists would read as new against no stored order, so the counts don't refetch it.
    assert _fetches(gateway, {"official": 4, "dvd": 4}) == 0


def test_an_outage_mid_series_keeps_the_stale_record_and_serves_it() -> None:
    cache = FakeCacheStore()
    stale = _stored(fresh_for=timedelta(days=-1))
    cache.put_tvdb_orders(_SERIES, stale)
    gateway = _gateway(cache=cache, failing=frozenset({(_SERIES, "dvd")}))

    assert _fetches(gateway) == 0

    assert gateway.outage is True
    assert gateway.orders(_SERIES) == stale


def test_other_orders_leave_out_official_and_empty_orders() -> None:
    cache = FakeCacheStore()
    row = (OrderEpisode(1, 1, 2273831, None),)
    record = StoredOrders(
        fetched_at=stamp_of(datetime.now()),
        fresh_until=stamp_of(datetime.now()),
        orders={"official": StoredOrder(row, None), "dvd": StoredOrder(row, None), "alternate": StoredOrder((), None)},
    )
    cache.put_tvdb_orders(_SERIES, record)
    gateway = _gateway(cache=cache)

    assert gateway.other_orders(_SERIES) == {"dvd": row}
    assert gateway.orders(_SERIES) == record
    assert gateway.other_orders(1) is NO_ORDERS


@pytest.mark.parametrize(
    ("preview", "kept"), [pytest.param(False, True, id="real"), pytest.param(True, False, id="preview")]
)
def test_each_record_is_saved_as_it_is_written_unless_previewing(tmp_path: Path, preview: bool, kept: bool) -> None:
    path = str(tmp_path / "cache.db")
    store = CacheStore.load(path, config_checksum="x")
    SkyHookGateway(client=FakeSkyHookClient(_REPLIES), cache_store=store).prefetch(
        [OrderRequest(_SERIES, None)], preview=preview
    )
    # Closing without the end-of-run save is how an interrupted run ends.
    store.close()

    reopened = CacheStore.load(path, config_checksum="x")
    assert (reopened.get_tvdb_orders(_SERIES) is not None) is kept
    reopened.close()
