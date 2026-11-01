# pyright: strict
"""The TVDB order cache: which series refetch, what a fetch stores, and what the readers are served."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pearlarr.cache import CacheStore
from pearlarr.seadex_types import SkyHookReply
from pearlarr.skyhook_gateway import SkyHookGateway
from pearlarr.stamps import parse_stamp
from pearlarr.tvdb_orders import NO_ORDERS, OFFICIAL_ORDER, OrderRow, SeriesOrderFacts

from .builders import FakeCacheStore, stored_orders
from .fakes import FakeSkyHookClient
from .http_mock import skyhook_fixture

_SERIES = 112961


def _fixture(name: str) -> SkyHookReply:
    return SkyHookReply.model_validate(skyhook_fixture(name))


_OFFICIAL = _fixture("tvdb_112961_official.json")
_DVD = _fixture("tvdb_112961_dvd.json")
_REPLIES = {(_SERIES, "official"): _OFFICIAL, (_SERIES, "dvd"): _DVD}
_ROW = (OrderRow(1, 1, 2273831, None),)
_STORED = {"official": _ROW, "dvd": _ROW}
"""A stored record's orders, one row each."""


def _gateway(
    replies: Mapping[tuple[int, str], SkyHookReply] = _REPLIES,
    *,
    cache: FakeCacheStore | None = None,
    failing: frozenset[tuple[int, str]] = frozenset(),
) -> SkyHookGateway:
    return SkyHookGateway(client=FakeSkyHookClient(replies, failing=failing), cache_store=cache or FakeCacheStore())


def _fetches(gateway: SkyHookGateway, counts: Mapping[str, int] | None = None) -> int:
    return gateway.prefetch([SeriesOrderFacts(_SERIES, OFFICIAL_ORDER, counts)], preview=True)


def _fresh_days(gateway: SkyHookGateway) -> float:
    stored = gateway.orders(_SERIES)
    assert stored is not None
    return (parse_stamp(stored.fresh_until) - parse_stamp(stored.fetched_at)) / timedelta(days=1)


def test_a_first_fill_stores_every_listed_order_official_with_its_absolutes() -> None:
    gateway = _gateway()

    assert _fetches(gateway, {"official": 4, "dvd": 4}) == 1

    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert stored.orders == {
        "official": (
            OrderRow(0, 1, 2273871, None),
            OrderRow(1, 1, 2273831, 1),
            OrderRow(1, 2, 2273841, 2),
            OrderRow(1, 3, 2273851, 3),
            OrderRow(1, 4, 2273861, 4),
        ),
        "dvd": (
            OrderRow(1, 1, 2273831, None),
            OrderRow(1, 1, 2273851, None),
            OrderRow(1, 2, 2273841, None),
            OrderRow(1, 2, 2273861, None),
        ),
    }
    assert stored.sonarr_counts == {"official": 4, "dvd": 4}


def _recent(reply: SkyHookReply) -> SkyHookReply:
    """The reply with one more episode, aired five days ago."""

    aired = datetime.now(UTC) - timedelta(days=5)
    newest = reply.episodes[-1].model_copy(update={"episode_number": 5, "tvdb_id": 1, "air_date_utc": aired})
    return reply.model_copy(update={"episodes": (*reply.episodes, newest)})


@pytest.mark.parametrize(
    ("official", "fresh_days"),
    [
        pytest.param(_OFFICIAL, 30, id="ended-and-quiet"),
        pytest.param(_OFFICIAL.model_copy(update={"status": "Continuing"}), 7, id="continuing"),
        pytest.param(_recent(_OFFICIAL), 7, id="ended-but-aired-lately"),
    ],
)
def test_a_record_stays_fresh_longer_once_the_series_is_quiet(official: SkyHookReply, fresh_days: int) -> None:
    gateway = _gateway({(_SERIES, "official"): official, (_SERIES, "dvd"): _DVD})

    _fetches(gateway)

    assert _fresh_days(gateway) == fresh_days


@pytest.mark.parametrize(
    ("fresh_for", "fetched"),
    [pytest.param(timedelta(days=-1), 1, id="past-fresh-until"), pytest.param(timedelta(days=1), 0, id="fresh")],
)
def test_a_record_refetches_once_its_fresh_until_passes(fresh_for: timedelta, fetched: int) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(_SERIES, stored_orders(_STORED, fresh_for=fresh_for))

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
    cache.put_tvdb_orders(_SERIES, stored_orders(_STORED, sonarr_counts={"official": 4, "dvd": 4}))

    assert _fetches(_gateway(cache=cache), counts) == fetched


def test_an_order_only_sonarr_lists_is_kept_in_the_counts_and_refetches_nothing() -> None:
    # Sonarr's last refresh still lists an alternate order SkyHook no longer sends.
    gateway = _gateway()
    counts = {"official": 4, "dvd": 4, "alternate": 3}

    assert [_fetches(gateway, counts) for _ in range(2)] == [1, 0]

    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert set(stored.orders) == {"official", "dvd"}
    assert stored.sonarr_counts == counts


def test_a_series_skyhook_does_not_have_is_stored_without_orders_for_a_week() -> None:
    gateway = _gateway({})

    assert [_fetches(gateway, {"official": 4}) for _ in range(2)] == [1, 0]

    stored = gateway.orders(_SERIES)
    assert stored is not None
    assert stored.orders == {}
    assert gateway.has_record(_SERIES) is True
    assert _fresh_days(gateway) == 7


def test_an_outage_mid_series_keeps_the_stale_record_and_serves_it() -> None:
    cache = FakeCacheStore()
    stale = stored_orders(_STORED, fresh_for=timedelta(days=-1))
    cache.put_tvdb_orders(_SERIES, stale)
    gateway = _gateway(cache=cache, failing=frozenset({(_SERIES, "dvd")}))

    assert _fetches(gateway) == 0

    assert gateway.outage is True
    assert gateway.orders(_SERIES) == stale


def test_other_orders_leave_out_official_and_empty_orders() -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(_SERIES, stored_orders({**_STORED, "alternate": ()}))
    gateway = _gateway(cache=cache)

    assert gateway.other_orders(_SERIES) == {"dvd": _ROW}
    assert gateway.other_orders(1) is NO_ORDERS


@pytest.mark.parametrize(
    ("preview", "kept"), [pytest.param(False, True, id="real"), pytest.param(True, False, id="preview")]
)
def test_each_record_is_saved_as_it_is_written_unless_previewing(tmp_path: Path, preview: bool, kept: bool) -> None:
    path = str(tmp_path / "cache.db")
    store = CacheStore.load(path, config_checksum="x")
    SkyHookGateway(client=FakeSkyHookClient(_REPLIES), cache_store=store).prefetch(
        [SeriesOrderFacts(_SERIES, OFFICIAL_ORDER, None)], preview=preview
    )
    # Closing without the end-of-run save is how an interrupted run ends.
    store.close()

    reopened = CacheStore.load(path, config_checksum="x")
    assert (reopened.get_tvdb_orders(_SERIES) is not None) is kept
    reopened.close()
