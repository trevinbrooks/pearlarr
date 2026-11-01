"""SkyHook order gateway: which series' TVDB orders to refetch, and the stored record per series."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import NamedTuple

from .cache import AbstractCacheStore
from .seadex_types import ProgressSink, SkyHookReply
from .skyhook_client import AbstractSkyHookClient, SkyHookMiss
from .stamps import parse_stamp_or_none, stamp_of
from .tvdb_orders import NO_ORDERS, OFFICIAL_ORDER, OrderEpisode, SeriesOrders, StoredOrder, StoredOrders

QUIET_AFTER_DAYS = 30
"""An ended series whose newest episode aired longer ago than this is quiet."""

QUIET_FRESH_DAYS = 30
"""How long a quiet series' record stays fresh."""

ACTIVE_FRESH_DAYS = 7
"""How long any other series' record stays fresh, a series SkyHook doesn't know included."""


class OrderRequest(NamedTuple):
    """One series the run reads orders for."""

    tvdb_id: int
    sonarr_counts: Mapping[str, int] | None
    """Sonarr v5's episode count per order it lists for the series. None on v4."""


class SkyHookGateway:
    """The TVDB order cache over the SkyHook client: refetches due series, serves stored records at any age."""

    def __init__(self, *, client: AbstractSkyHookClient, cache_store: AbstractCacheStore) -> None:
        self._client = client
        self._cache = cache_store

    @property
    def outage(self) -> bool:
        """True once SkyHook has been declared unreachable for this run."""

        return self._client.outage

    def prefetch(
        self,
        requests: Sequence[OrderRequest],
        *,
        preview: bool,
        progress: ProgressSink | None = None,
    ) -> int:
        """Refetch every due series, one at a time, writing each one's record whole. Returns how many were written.

        Each record is saved as it is written, unless `preview`, so an interrupted first fill keeps what it fetched.
        On an outage the series being fetched keeps its old record, and the rest are left for the next run.
        """

        now = datetime.now()
        due = [request for request in requests if self._due(request, now)]
        written = 0
        for done, request in enumerate(due, start=1):
            record = self._fetch_series(request, now)
            if record is None:
                break
            self._cache.put_tvdb_orders(request.tvdb_id, record)
            self._cache.save(preview=preview)
            written += 1
            if progress is not None:
                progress.progress(done / len(due), f"{done}/{len(due)}")
        return written

    def orders(self, tvdb_id: int) -> StoredOrders | None:
        """The series' stored record at any age, or None when it has never been fetched."""

        return self._cache.get_tvdb_orders(tvdb_id)

    def other_orders(self, tvdb_id: int) -> SeriesOrders:
        """Every stored order of the series except official, empty orders left out."""

        stored = self._cache.get_tvdb_orders(tvdb_id)
        if stored is None:
            return NO_ORDERS
        return MappingProxyType(
            {name: order.episodes for name, order in stored.orders.items() if name != OFFICIAL_ORDER and order.episodes}
        )

    def known(self, tvdb_id: int) -> bool:
        """Whether the series has a stored record."""

        return self._cache.get_tvdb_orders(tvdb_id) is not None

    def _due(self, request: OrderRequest, now: datetime) -> bool:
        """Whether the series needs a fetch: no record, a record past its `fresh_until`, or Sonarr's counts moved."""

        stored = self._cache.get_tvdb_orders(request.tvdb_id)
        if stored is None:
            return True
        fresh_until = parse_stamp_or_none(stored.fresh_until)
        if fresh_until is None or fresh_until <= now:
            return True
        # A record of a series SkyHook doesn't know has no orders to compare, so it waits for its fresh_until.
        if not stored.orders or request.sonarr_counts is None:
            return False
        return dict(request.sonarr_counts) != stored.sonarr_counts

    def _fetch_series(self, request: OrderRequest, now: datetime) -> StoredOrders | None:
        """The series' new record: official first, then each other order it lists. None on an outage."""

        counts = request.sonarr_counts or {}
        official = self._client.fetch(request.tvdb_id, OFFICIAL_ORDER)
        if official is SkyHookMiss.OUTAGE:
            return None
        if official is SkyHookMiss.UNKNOWN:
            return _record({}, now, quiet=False)
        orders = {OFFICIAL_ORDER: StoredOrder(_rows(official), counts.get(OFFICIAL_ORDER))}
        for listed in official.season_types:
            if listed.type in orders:
                continue
            reply = self._client.fetch(request.tvdb_id, listed.type)
            if reply is SkyHookMiss.OUTAGE:
                return None
            rows = () if reply is SkyHookMiss.UNKNOWN else _rows(reply)
            orders[listed.type] = StoredOrder(rows, counts.get(listed.type))
        return _record(orders, now, quiet=_is_quiet(official, now))


def _rows(reply: SkyHookReply) -> tuple[OrderEpisode, ...]:
    """One reply's rows as order episodes."""

    return tuple(
        OrderEpisode(row.season_number, row.episode_number, row.tvdb_id, row.absolute_episode_number)
        for row in reply.episodes or ()
    )


def _is_quiet(official: SkyHookReply, now: datetime) -> bool:
    """Whether the series has ended and its newest episode aired more than `QUIET_AFTER_DAYS` ago."""

    aired = [row.air_date_utc for row in official.episodes or () if row.air_date_utc is not None]
    cutoff = now.astimezone(UTC) - timedelta(days=QUIET_AFTER_DAYS)
    return official.status == "Ended" and bool(aired) and max(aired) < cutoff


def _record(orders: Mapping[str, StoredOrder], now: datetime, *, quiet: bool) -> StoredOrders:
    """The record stamped now, fresh for the quiet or active window."""

    fresh_for = timedelta(days=QUIET_FRESH_DAYS if quiet else ACTIVE_FRESH_DAYS)
    return StoredOrders(fetched_at=stamp_of(now), fresh_until=stamp_of(now + fresh_for), orders=orders)
