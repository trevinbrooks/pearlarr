"""TVDB episode orders: the records the SkyHook cache stores per series and the view placement reads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, Self

AIRED_ORDER = "official"
"""SkyHook's and Sonarr's name for TVDB's aired order, the order Sonarr numbers a series by default."""

_DISPLAY_NAMES: Mapping[str, str] = MappingProxyType(
    {"dvd": "DVD", "altdvd": "alternate DVD", "alttwo": "second alternate"}
)


def order_display_name(order: str) -> str:
    """An order's type name as a message prints it (`dvd` as "DVD"). A type with no entry prints as stored."""

    return _DISPLAY_NAMES.get(order, order)


class OrderRow(NamedTuple):
    """One episode as one TVDB order numbers it."""

    season: int
    episode: int
    tvdb_id: int
    """TVDB's episode id, the same in every order."""
    absolute: int | None
    """TVDB's absolute number. Only the official order carries one."""


type OrdersByName = Mapping[str, Sequence[OrderRow]]
"""Order type name -> that order's rows."""

NO_ORDERS: OrdersByName = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class CachedOrders:
    """One series' cached orders, replaced whole on each fetch."""

    fetched_at: str
    """When the record was fetched, in `stamps.STAMP_FORMAT`."""

    fresh_until: str
    """When the record is due for a refetch, in `stamps.STAMP_FORMAT`."""

    orders: Mapping[str, tuple[OrderRow, ...]]
    """Order type name (`official`, `dvd`, `alternate`, ...) -> its rows. An order is empty when SkyHook lists it but
    sent no usable rows for it, and the map is empty when SkyHook doesn't have the series."""

    sonarr_counts: Mapping[str, int] | None
    """Sonarr's episode count per order it listed at fetch time, every listed order kept, SkyHook's or not. None
    when Sonarr listed no orders (always on v4)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", MappingProxyType(dict(self.orders)))
        if self.sonarr_counts is not None:
            object.__setattr__(self, "sonarr_counts", MappingProxyType(dict(self.sonarr_counts)))

    def to_json(self) -> dict[str, Any]:
        """The record as the cache stores it, each row a `[season, episode, tvdb_id, absolute]` array."""

        return {
            "fetched_at": self.fetched_at,
            "fresh_until": self.fresh_until,
            "orders": {name: [list(row) for row in rows] for name, rows in self.orders.items()},
            "sonarr_counts": None if self.sonarr_counts is None else dict(self.sonarr_counts),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self | None:
        """Rebuild a stored record, or None when it doesn't have the stored shape (the series then refetches)."""

        try:
            orders = {name: tuple(OrderRow(*row) for row in rows) for name, rows in raw["orders"].items()}
            counts = raw["sonarr_counts"]
            return cls(
                fetched_at=raw["fetched_at"],
                fresh_until=raw["fresh_until"],
                orders=orders,
                sonarr_counts=None if counts is None else dict(counts),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


@dataclass(frozen=True, slots=True)
class SonarrSeriesOrders:
    """What one Sonarr series row says about its orders."""

    tvdb_id: int
    season_type: str
    """The order Sonarr numbers the series by (`seasonType`)."""

    counts: Mapping[str, int] | None
    """Sonarr's episode count per listed order. None when Sonarr lists no orders (always on v4)."""


class OrderTranslation(NamedTuple):
    """How to read a series Sonarr numbers by a non-official order back into aired numbering."""

    order_name: str
    """The order the series is set to."""

    aired: tuple[OrderRow, ...]
    """The stored official order (empty when it isn't stored)."""


class IndexOrders(NamedTuple):
    """The orders one series' episode index is built with."""

    others: OrdersByName
    """Every stored order except official, empty orders left out."""

    translation: OrderTranslation | None
    """None when Sonarr's rows are already in aired order."""


NO_INDEX_ORDERS = IndexOrders(NO_ORDERS, None)
