"""TVDB episode orders: the records the SkyHook cache stores per series and the view placement reads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, Self

OFFICIAL_ORDER = "official"
"""SkyHook's and Sonarr's name for TVDB's aired order, the order Sonarr numbers a series by default."""


class OrderEpisode(NamedTuple):
    """One episode as one TVDB order numbers it."""

    season: int
    episode: int
    tvdb_id: int
    """TVDB's episode id, the same in every order."""
    absolute: int | None
    """TVDB's absolute number. Only the official order carries one."""


type SeriesOrders = Mapping[str, Sequence[OrderEpisode]]
"""Order type name -> that order's episodes."""

NO_ORDERS: SeriesOrders = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class StoredOrder:
    """One order as stored: its episodes and the count Sonarr listed for it when it was fetched."""

    episodes: tuple[OrderEpisode, ...]
    """Empty when SkyHook lists the order but sent no rows for it, or sent rows that failed validation."""

    sonarr_count: int | None
    """Sonarr v5's `seasonTypes[].episodeCount` for this order at fetch time. None on v4, or when Sonarr didn't list
    the order."""


@dataclass(frozen=True, slots=True)
class StoredOrders:
    """One series' cached orders, replaced whole on each fetch."""

    fetched_at: str
    """When the record was fetched, in `stamps.UPDATED_AT_STR_FORMAT`."""

    fresh_until: str
    """When the record is due for a refetch, in `stamps.UPDATED_AT_STR_FORMAT`."""

    orders: Mapping[str, StoredOrder]
    """Order type name (`official`, `dvd`, `alternate`, ...) -> the order. Empty when SkyHook doesn't know the
    series."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", MappingProxyType(dict(self.orders)))

    @property
    def sonarr_counts(self) -> dict[str, int]:
        """The counts Sonarr listed at fetch time, by order type (the orders it didn't list are left out)."""

        return {name: order.sonarr_count for name, order in self.orders.items() if order.sonarr_count is not None}

    def to_json(self) -> dict[str, Any]:
        """The record as the cache stores it, each episode a `[season, episode, tvdb_id, absolute]` array."""

        return {
            "fetched_at": self.fetched_at,
            "fresh_until": self.fresh_until,
            "orders": {
                name: {"sonarr_count": order.sonarr_count, "episodes": [list(ep) for ep in order.episodes]}
                for name, order in self.orders.items()
            },
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self | None:
        """Rebuild a stored record, or None when it doesn't have the stored shape (the series then refetches)."""

        try:
            orders = {
                name: StoredOrder(
                    episodes=tuple(OrderEpisode(*row) for row in order["episodes"]),
                    sonarr_count=order["sonarr_count"],
                )
                for name, order in raw["orders"].items()
            }
            return cls(fetched_at=raw["fetched_at"], fresh_until=raw["fresh_until"], orders=orders)
        except (KeyError, TypeError, AttributeError):
            return None


@dataclass(frozen=True, slots=True)
class SeriesOrderFacts:
    """What one Sonarr series row says about its orders."""

    tvdb_id: int
    season_type: str
    """The order Sonarr numbers the series by (`seasonType`)."""

    counts: Mapping[str, int] | None
    """Sonarr v5's episode count per listed order. None on v4, which doesn't list them."""


class OrderTranslation(NamedTuple):
    """How to read a series Sonarr numbers by a non-official order back into aired numbering."""

    order_name: str
    """The order the series is set to."""

    aired: tuple[OrderEpisode, ...]
    """The stored official order (empty when it isn't stored)."""


class SeriesOrdering(NamedTuple):
    """The orders one series' episode index is built with."""

    others: SeriesOrders
    """Every stored order except official, empty orders left out."""

    translation: OrderTranslation | None
    """None when Sonarr's rows are already in aired order."""


NO_ORDERING = SeriesOrdering(NO_ORDERS, None)
