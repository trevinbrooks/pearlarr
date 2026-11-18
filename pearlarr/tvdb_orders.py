"""TVDB episode orders: the answers the SkyHook cache stores per series and the view placement reads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, NamedTuple, Self

AIRED_ORDER = "official"
"""SkyHook's and Sonarr's name for TVDB's aired order, the order Sonarr numbers a series by default."""

_DISPLAY_NAMES: Mapping[str, str] = MappingProxyType(
    {"dvd": "DVD", "altdvd": "alternate DVD", "alttwo": "second alternate"}
)


def order_display_name(order: str) -> str:
    """An order's type name as a message prints it (`dvd` as "DVD"). A type with no display name prints as stored."""

    return _DISPLAY_NAMES.get(order, order)


def is_another_order(order: str) -> bool:
    """Whether the type name is of an order other than the aired one. False for "", which names no order."""

    return order not in ("", AIRED_ORDER)


class OrderRow(NamedTuple):
    """One episode as one TVDB order numbers it."""

    season: int
    episode: int
    tvdb_id: int
    """TVDB's episode id, the same in every order."""
    absolute: int | None
    """TVDB's absolute number. Only the official order carries one."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeDetails:
    """What TVDB says of one episode beside its numbers, the same in every order."""

    title: str
    """Empty when TVDB has no title for the episode."""

    air_date: date | None
    """The day the episode aired, None when TVDB has no date for it."""

    runtime: int | None
    """Minutes."""

    def to_json(self) -> list[Any]:
        """The details as the cache stores them, a `[title, air_date, runtime]` array with the day in ISO form."""

        return [self.title, None if self.air_date is None else self.air_date.isoformat(), self.runtime]

    @classmethod
    def from_json(cls, raw: list[Any]) -> Self:
        """Rebuild stored details. Raises `ValueError` or `TypeError` when `raw` doesn't have the stored shape."""

        title, air_date, runtime = raw
        return cls(title=title, air_date=None if air_date is None else date.fromisoformat(air_date), runtime=runtime)


type OrdersByName = Mapping[str, Sequence[OrderRow]]
"""Order type name -> that order's rows."""

NO_ORDERS: OrdersByName = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class AiredOrder:
    """A stored answer's aired order with what TVDB says of each of its episodes."""

    rows: tuple[OrderRow, ...]
    """Empty when SkyHook sent no usable rows for the aired order."""

    details: Mapping[int, EpisodeDetails]
    """TVDB episode id -> the episode's details, one for every row."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CachedOrders:
    """One series' cached orders, replaced whole on each fetch."""

    fetched_at: str
    """When the answer was fetched, in `stamps.STAMP_FORMAT`."""

    fresh_until: str
    """When the answer is due for a refetch, in `stamps.STAMP_FORMAT`."""

    orders: Mapping[str, tuple[OrderRow, ...]]
    """Order type name (`official`, `dvd`, `alternate`, ...) -> its rows. An order is empty when SkyHook lists it but
    sent no usable rows for it, and `orders` is empty when SkyHook doesn't have the series."""

    sonarr_counts: Mapping[str, int] | None
    """Sonarr's episode count per order it listed at fetch time, every listed order kept, SkyHook's or not. None
    when Sonarr listed no orders (always on v4)."""

    details: Mapping[int, EpisodeDetails] | None
    """TVDB episode id -> the episode's details, for every row of the official order: an answer built with a row
    that has none raises `ValueError`. None on an answer stored before details were kept."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", MappingProxyType(dict(self.orders)))
        if self.sonarr_counts is not None:
            object.__setattr__(self, "sonarr_counts", MappingProxyType(dict(self.sonarr_counts)))
        if self.details is not None:
            object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
            # The aired view reads each aired row's details by its TVDB episode id (`aired_view.aired_series`), so an
            # answer that lacks one is refused.
            without = [row.tvdb_id for row in self.orders.get(AIRED_ORDER, ()) if row.tvdb_id not in self.details]
            if without:
                raise ValueError(f"the aired order's TVDB episodes {without} have no details")

    @property
    def aired(self) -> AiredOrder | None:
        """The aired order and its episodes' details, what numbering a series as aired reads.

        None on an answer stored before details were kept. A series Sonarr numbers by another order has no aired
        view from such an answer, and is due a fetch for one.
        """

        if self.details is None:
            return None
        return AiredOrder(self.orders.get(AIRED_ORDER, ()), self.details)

    def to_json(self) -> dict[str, Any]:
        """The answer as the cache stores it, each row a `[season, episode, tvdb_id, absolute]` array.

        `details` is keyed by the TVDB episode id as a string, since a JSON object has no other keys.
        """

        details = self.details
        return {
            "fetched_at": self.fetched_at,
            "fresh_until": self.fresh_until,
            "orders": {name: [list(row) for row in rows] for name, rows in self.orders.items()},
            "sonarr_counts": None if self.sonarr_counts is None else dict(self.sonarr_counts),
            "details": None if details is None else {str(tvdb_id): row.to_json() for tvdb_id, row in details.items()},
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self | None:
        """Rebuild a stored answer, or None when it doesn't have the stored shape (the series then refetches).

        An answer that keeps details and has none for a row of its official order doesn't have it. An answer stored
        before details were kept has no `details` key and rebuilds with `details` None.
        """

        try:
            orders = {name: tuple(OrderRow(*row) for row in rows) for name, rows in raw["orders"].items()}
            counts = raw["sonarr_counts"]
            details = raw.get("details")
            return cls(
                fetched_at=raw["fetched_at"],
                fresh_until=raw["fresh_until"],
                orders=orders,
                sonarr_counts=None if counts is None else dict(counts),
                details=None
                if details is None
                else {int(tvdb_id): EpisodeDetails.from_json(row) for tvdb_id, row in details.items()},
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


@dataclass(frozen=True, slots=True)
class SeriesNumbering:
    """How Sonarr numbered a series at one read: by which order, and over how many episodes.

    A title is cached with the numbering its series had when it was checked, and is checked again once the series
    is numbered another way (`differs_from`). Build one with `of`.
    """

    order: str
    """The order's type name, "" for the aired order."""

    row_count: int | None
    """Sonarr's count of the series' episodes. None for the aired order, and when Sonarr sent no count."""

    @classmethod
    def of(cls, order: str | None, row_count: int | None) -> Self:
        """The numbering under `order` as Sonarr or a cached title names it, None being a title cached with no order.

        Every name of the aired order reads as one numbering with no count. A series numbered as aired gains
        episodes as they are announced, and a count kept for it would check its titles again each time.
        """

        if order is None or not is_another_order(order):
            return cls("", None)
        return cls(order, row_count)

    def differs_from(self, other: Self) -> bool:
        """Whether the two are of different orders, or of one order with different counts.

        A count missing on either side tells nothing, so the two then differ only by their orders.
        """

        if self.order != other.order:
            return True
        return self.row_count is not None and other.row_count is not None and self.row_count != other.row_count


@dataclass(frozen=True, slots=True)
class SonarrSeriesOrders:
    """What one Sonarr series row says about its orders."""

    tvdb_id: int
    season_type: str
    """The order Sonarr numbers the series by (`seasonType`)."""

    counts: Mapping[str, int] | None
    """Sonarr's episode count per listed order. None when Sonarr lists no orders (always on v4)."""

    row_count: int | None = None
    """Sonarr's count of the series' episodes (`statistics.totalEpisodeCount`). None when Sonarr sent none."""

    @property
    def in_another_order(self) -> bool:
        """Whether Sonarr names an order other than the aired one for the series. False when it names none."""

        return is_another_order(self.season_type)

    @property
    def numbering(self) -> SeriesNumbering:
        """How Sonarr numbers the series as of this row."""

        return SeriesNumbering.of(self.season_type, self.row_count)
