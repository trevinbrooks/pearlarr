"""Whether Sonarr's rows of a series can be read by the order the run's library read names for it.

For the check and the import alike, rows count only while they are numbered by that order, as far as the series'
stored TVDB orders can tell (`aired_view.numbered_by`).

The order is the library read's for the whole run. When a series' order is changed in Sonarr after that, its rows
fail the test once Sonarr has renumbered them, wherever the two orders number an episode differently, and the next
run reads the new order.

Two kinds of rows are never tested. A Sonarr that sends no order numbers every series as aired, so its rows are
taken as they come. And a check never asks about a series the library read has in another order: its view is
renumbered by TVDB episode id, which is right whatever numbers its rows carry.
"""

from enum import Enum, auto

from . import aired_view as _aired_view
from .arr_models import SonarrEpisode
from .library_read import LibraryRead
from .output import hub_warn
from .skyhook_gateway import SkyHookGateway
from .tvdb_orders import AIRED_ORDER, SonarrSeriesOrders


class _Numbering(Enum):
    """Whether a series' rows are numbered by the order the library read names for it (`RowOrderGuard._numbering`)."""

    IN_STEP = auto()
    """They are, or nothing stored can tell."""

    OUT_OF_STEP = auto()
    """They are not, by an answer SkyHook gave this run."""

    UNCONFIRMED = auto()
    """They are not by the stored answer, and SkyHook couldn't be reached for a new one."""


class RowOrderGuard:
    """Tests a run's rows against the order the library read names for their series, and keeps the series it holds.

    One is built per library read, so its holds and the warnings it gave end with the run.
    """

    def __init__(self, skyhook: SkyHookGateway, library: LibraryRead) -> None:
        """Bind the stored TVDB orders and the run's library read."""

        self._skyhook = skyhook
        self._library = library

        # The series with no aired view for the rest of the run (`hold`).
        self._held: set[int] = set()

        # The series this run warned of rows out of step with their order (`numbered_as_named`).
        self._out_of_step: set[int] = set()

    def holds(self, series_id: int) -> bool:
        """Whether the series has no aired view for the rest of the run (`hold`)."""

        return series_id in self._held

    def hold(self, series_id: int) -> None:
        """Leave the series with no aired view for the rest of the run.

        Its titles are not cached, so the next run checks them.
        """

        self._held.add(series_id)

    def numbered_as_named(self, series_id: int, rows: list[SonarrEpisode]) -> bool:
        """Whether `rows` are numbered by the order the library read names for the series (`_numbering`).

        Rows that aren't are warned of once per series per run, whichever of the check and the import asks first.
        """

        if not self._library.sends_order:
            return True
        series = self._library.series(series_id)
        numbering = self._numbering(rows, series.orders)
        if numbering is _Numbering.IN_STEP:
            return True
        if series_id not in self._out_of_step:
            self._out_of_step.add(series_id)
            name = series.title or f"Series {series_id}"
            if numbering is _Numbering.OUT_OF_STEP:
                hub_warn(
                    f"{name}: Sonarr's episodes of the series are not numbered by the order Sonarr names for it - "
                    "holding off on the series until they are - refresh it in Sonarr"
                )
            else:
                hub_warn(
                    f"{name}: Sonarr's episodes of the series may not be numbered by the order Sonarr names for it, "
                    "and SkyHook could not be reached to make sure - holding off on the series until it can"
                )
        return False

    def _numbering(self, rows: list[SonarrEpisode], facts: SonarrSeriesOrders) -> _Numbering:
        """Whether `rows` are numbered by the order `facts` names, by the series' stored TVDB orders.

        This guards against rows Sonarr has not renumbered under an order it already names. Read as they are, or
        imported onto, they would stand for other episodes than that order gives them.

        With no stored answer, or no rows stored for the order, nothing can tell and the rows pass. Rows that fail
        are tested once more by an answer fetched again, since the stored one may be the side that is behind. An
        answer with no rows for the order doesn't clear them.
        """

        order = facts.season_type or AIRED_ORDER
        stored = self._skyhook.cached_orders(facts.tvdb_id)
        listed = None if stored is None else stored.orders.get(order)
        if not listed or _aired_view.numbered_by(rows, listed):
            return _Numbering.IN_STEP
        fresh = self._skyhook.refetch(facts)
        if fresh is None:
            return _Numbering.UNCONFIRMED
        listed = fresh.orders.get(order)
        if listed and _aired_view.numbered_by(rows, listed):
            return _Numbering.IN_STEP
        return _Numbering.OUT_OF_STEP
