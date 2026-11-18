"""What a run's read of Sonarr's library says of each series, kept for the whole run."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from .arr_models import AlternateTitle, SonarrItem
from .tvdb_orders import SonarrSeriesOrders


def series_orders(series: SonarrItem) -> SonarrSeriesOrders:
    """What one Sonarr series resource says about its orders, from the library read or a read of the series alone."""

    return SonarrSeriesOrders(
        tvdb_id=series.tvdbId,
        season_type=series.seasonType,
        counts={listed.type: listed.episode_count for listed in series.seasonTypes} if series.seasonTypes else None,
        row_count=series.totalEpisodeCount,
    )


@dataclass(frozen=True, slots=True)
class LibrarySeries:
    """What this run's library read says of one series."""

    orders: SonarrSeriesOrders
    """Its TVDB id and the order Sonarr numbers it by."""

    alternate_titles: tuple[AlternateTitle, ...] = ()
    """Its other names in Sonarr, in Sonarr's order."""


NOT_IN_LIBRARY = LibrarySeries(SonarrSeriesOrders(tvdb_id=0, season_type="", counts=None))
"""A series this run's library read doesn't have: it names no order and has no other names."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LibraryRead:
    """One read of Sonarr's library. A run makes one and never changes it, so every view is built under its order.

    Build one with `of`.
    """

    by_id: Mapping[int, LibrarySeries]
    """What the read says of each series in Sonarr, with an AniList mapping or without."""

    sends_order: bool
    """Whether Sonarr sent an order with any series of the read (`SonarrItem.order_sent`). One that sends none
    numbers every series as aired, and no series is read a second time for it."""

    series_hash: str
    """The hash of the ids of the series with an AniList mapping (`sonarr_series_hash`). It pins the `/parse` cache
    rows with no matched episodes to the library state, so they are parsed again once Sonarr has the series."""

    in_another_order: Mapping[int, str] = field(init=False)
    """Series id -> its order, for each series the read has in an order other than the aired one. A cached `/parse`
    row is good only under the order its series was in when it was stored."""

    def __post_init__(self) -> None:
        in_another_order = {
            series_id: series.orders.season_type
            for series_id, series in self.by_id.items()
            if series.orders.in_another_order
        }
        object.__setattr__(self, "in_another_order", in_another_order)

    @classmethod
    def of(cls, in_sonarr: Sequence[SonarrItem], series_hash: str) -> Self:
        """The read of every series Sonarr sent, with the hash of the ones that have an AniList mapping."""

        return cls(
            by_id={series.id: LibrarySeries(series_orders(series), series.alternateTitles) for series in in_sonarr},
            sends_order=any(series.order_sent for series in in_sonarr),
            series_hash=series_hash,
        )

    def series(self, series_id: int) -> LibrarySeries:
        """What the read says of the series, `NOT_IN_LIBRARY` for one it doesn't have."""

        return self.by_id.get(series_id, NOT_IN_LIBRARY)


NO_LIBRARY = LibraryRead(by_id={}, sends_order=False, series_hash="")
"""What stands for the library read until a run makes one: no series, and no order sent."""
