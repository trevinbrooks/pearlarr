"""SkyHook order gateway: which series' TVDB orders to refetch, and the stored record per series."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from .cache import AbstractCacheStore
from .progress import ProgressSink
from .skyhook_client import AbstractSkyHookClient, SkyHookNoReply, SkyHookReply
from .stamps import parse_stamp_or_none, stamp_of
from .tvdb_orders import AIRED_ORDER, NO_ORDERS, CachedOrders, OrderRow, OrdersByName, SonarrSeriesOrders

QUIET_AFTER_DAYS = 30
"""An ended series whose newest episode aired longer ago than this is quiet."""

QUIET_FRESH_DAYS = 30
"""How long a quiet series' record stays fresh."""

ACTIVE_FRESH_DAYS = 7
"""How long any other series' record stays fresh, a series SkyHook doesn't have included."""


class SkyHookGateway:
    """The TVDB order cache over the SkyHook client: refetches due series, serves stored records at any age."""

    def __init__(self, *, client: AbstractSkyHookClient, cache_store: AbstractCacheStore) -> None:
        """Bind the client and the cache (network-free). Records read are kept for the run."""

        self._client = client
        self._cache = cache_store
        self._answers: dict[int, CachedOrders | None] = {}

    @property
    def unreachable(self) -> bool:
        """True once SkyHook has been declared unreachable for this run."""

        return self._client.unreachable

    def prefetch(
        self,
        series: Sequence[SonarrSeriesOrders],
        *,
        dry_run: bool,
        progress: ProgressSink | None = None,
    ) -> int:
        """Refetch every due series, one at a time, writing each one's record whole. Returns how many were written.

        Each record is saved as it is written, unless `preview`, so an interrupted first fill keeps what it fetched.
        On an outage the series being fetched keeps its old record, and the rest are left for the next run.
        """

        now = datetime.now()
        due = [facts for facts in series if self._due(facts, now)]
        written = 0
        for done, facts in enumerate(due, start=1):
            answer = self._fetch_series(facts, now)
            if answer is None:
                break
            self._cache.put_tvdb_orders(facts.tvdb_id, answer)
            self._answers[facts.tvdb_id] = answer
            self._cache.save(dry_run=dry_run)
            written += 1
            if progress is not None:
                progress.progress(done / len(due), f"{done}/{len(due)}")
        return written

    def cached_orders(self, tvdb_id: int) -> CachedOrders | None:
        """The series' stored record at any age, or None when it has never been fetched."""

        if tvdb_id not in self._answers:
            self._answers[tvdb_id] = self._cache.get_tvdb_orders(tvdb_id)
        return self._answers[tvdb_id]

    def other_orders(self, tvdb_id: int) -> OrdersByName:
        """Every stored order of the series except official, empty orders left out."""

        stored = self.cached_orders(tvdb_id)
        if stored is None:
            return NO_ORDERS
        return MappingProxyType({name: rows for name, rows in stored.orders.items() if name != AIRED_ORDER and rows})

    def has_cached_answer(self, tvdb_id: int) -> bool:
        """Whether the series has a stored record, even one with no orders."""

        return self.cached_orders(tvdb_id) is not None

    def _due(self, facts: SonarrSeriesOrders, now: datetime) -> bool:
        """Whether the series needs a fetch: no record, a record past its `fresh_until`, or Sonarr's counts moved."""

        stored = self.cached_orders(facts.tvdb_id)
        if stored is None:
            return True
        fresh_until = parse_stamp_or_none(stored.fresh_until)
        if fresh_until is None or fresh_until <= now:
            return True
        # Against the counts Sonarr listed at fetch time, not the orders SkyHook sent: an order only Sonarr lists
        # would otherwise read as new on every run.
        return facts.counts is not None and dict(facts.counts) != stored.sonarr_counts

    def _fetch_series(self, facts: SonarrSeriesOrders, now: datetime) -> CachedOrders | None:
        """The series' new record: official first, then each other order it lists. None on an outage."""

        aired_reply = self._client.fetch(facts.tvdb_id, AIRED_ORDER)
        if aired_reply is SkyHookNoReply.UNREACHABLE:
            return None
        orders: dict[str, tuple[OrderRow, ...]] = {}
        quiet = False
        if aired_reply is not SkyHookNoReply.NOT_FOUND:
            orders[AIRED_ORDER] = _rows(aired_reply)
            quiet = _is_quiet(aired_reply, now)
            for listed in aired_reply.season_types:
                if listed.type in orders:
                    continue
                reply = self._client.fetch(facts.tvdb_id, listed.type)
                if reply is SkyHookNoReply.UNREACHABLE:
                    return None
                orders[listed.type] = () if reply is SkyHookNoReply.NOT_FOUND else _rows(reply)
        fresh_for = timedelta(days=QUIET_FRESH_DAYS if quiet else ACTIVE_FRESH_DAYS)
        return CachedOrders(
            fetched_at=stamp_of(now),
            fresh_until=stamp_of(now + fresh_for),
            orders=orders,
            sonarr_counts=facts.counts,
        )


def _rows(reply: SkyHookReply) -> tuple[OrderRow, ...]:
    """One reply's rows as order rows."""

    return tuple(
        OrderRow(row.season_number, row.episode_number, row.tvdb_id, row.absolute_episode_number)
        for row in reply.episodes
    )


def _is_quiet(aired_reply: SkyHookReply, now: datetime) -> bool:
    """Whether the series has ended and its newest episode aired more than `QUIET_AFTER_DAYS` ago."""

    aired = [row.air_date_utc for row in aired_reply.episodes if row.air_date_utc is not None]
    cutoff = now.astimezone(UTC) - timedelta(days=QUIET_AFTER_DAYS)
    return aired_reply.ended and bool(aired) and max(aired) < cutoff
