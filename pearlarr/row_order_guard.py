"""Whether Sonarr's rows of a series can be read by the order Sonarr names for it.

Rows are read as already aired only when Sonarr says twice that the series is: in the library read, and in a read
of the series alone made after its rows. For the check and the import alike, rows count only while they are
numbered by the order Sonarr names for the series, as far as its stored TVDB orders can tell
(`aired_view.numbered_by`).

The two go by different orders:

- A title's check (`RowOrderGuard.aired_rows`) goes by the library read's order, the one every view of the run is
  built under. A series whose order changed since that read is held for the run.
- An import (`RowOrderGuard.numbered_as_named`) goes by the order the read of the series alone names: an import
  follows a change of order Sonarr finished during the run, which the library read's order would hold back until
  the next one.

Two kinds of rows are never tested. A Sonarr that sends no order numbers every series as aired, so its rows are
taken as they come. And a check never asks about a series the library read has in another order: its view is
renumbered by TVDB episode id, which is right whatever numbers its rows carry.
"""

from enum import Enum, auto

from . import aired_view as _aired_view
from .arr_http import NoResource
from .arr_models import SonarrEpisode, SonarrItem
from .library_read import LibraryRead, series_orders
from .output import hub_warn
from .skyhook_gateway import SkyHookGateway
from .sonarr_client import AbstractSonarrClient
from .tvdb_orders import AIRED_ORDER, SonarrSeriesOrders


class AiredRows(Enum):
    """Whether rows the library read takes as aired can be read as aired (`RowOrderGuard.aired_rows`)."""

    CONFIRMED = auto()
    """They can."""

    GONE = auto()
    """Sonarr answers that it doesn't have the series, so it has no episodes."""

    UNREAD = auto()
    """Not on this call. Either the read of the series alone failed, and the next call asks again, or the series is
    held for the run (`RowOrderGuard.holds`)."""


class _Numbering(Enum):
    """Whether a series' rows are numbered by the order Sonarr names for it (`RowOrderGuard._numbering`)."""

    IN_STEP = auto()
    """They are, or nothing stored can tell."""

    OUT_OF_STEP = auto()
    """They are not, by an answer SkyHook gave this run."""

    UNCONFIRMED = auto()
    """They are not by the stored answer, and SkyHook couldn't be reached for a new one."""


def _name_of(series_id: int, series: SonarrItem) -> str:
    """The series' name for a warning: its Sonarr title, or its id when Sonarr sent none."""

    return series.title or f"Series {series_id}"


class RowOrderGuard:
    """Tests a run's rows against the order Sonarr names for their series, and holds the series that fail a check.

    One is built per library read, so its holds and the warnings it gave end with the run.
    """

    def __init__(self, sonarr: AbstractSonarrClient, skyhook: SkyHookGateway, library: LibraryRead) -> None:
        """Bind the client a series is read alone through, the stored TVDB orders, and the run's library read."""

        self._sonarr = sonarr
        self._skyhook = skyhook
        self._library = library

        # The series with no aired view for the rest of the run (`_hold`).
        self._held: set[int] = set()

        # The series this run warned of rows out of step with their order (`_in_step`).
        self._out_of_step: set[int] = set()

    def holds(self, series_id: int) -> bool:
        """Whether the series has no aired view for the rest of the run (`_hold`)."""

        return series_id in self._held

    def aired_rows(self, series_id: int, rows: list[SonarrEpisode]) -> AiredRows:
        """Whether rows the library read takes as aired can be read so, once a read of the series alone agrees.

        The read is made after the rows: an order Sonarr changed before it sent them shows in it. The two reads
        agree when neither names another order. Counts and the TVDB id are not compared.

        A series whose order changed, and one the library read doesn't have, is held for the run (`_hold`). So is
        one whose rows aren't numbered by the aired order both reads name (`_in_step`). A failed read, a reply
        without the order among them, leaves the rows unread for this call only.
        """

        if not self._library.sends_order:
            return AiredRows.CONFIRMED
        current = self._sonarr.series(series_id)
        if current is NoResource.GONE:
            return AiredRows.GONE
        if isinstance(current, NoResource) or not current.order_sent:
            return AiredRows.UNREAD
        name = _name_of(series_id, current)
        if series_id not in self._library.by_id:
            self._hold(
                series_id,
                f"{name} not checked - the series was not in Sonarr when this run read its library - checking its "
                "titles next run",
            )
            return AiredRows.UNREAD
        if series_orders(current).in_another_order:
            self._hold(
                series_id,
                f"{name} not checked - the order Sonarr numbers the series by changed during this run - checking "
                "its titles again next run",
            )
            return AiredRows.UNREAD
        if not self._in_step(series_id, current, rows):
            # Warned of there, once for the check and the import together.
            self._held.add(series_id)
            return AiredRows.UNREAD
        return AiredRows.CONFIRMED

    def numbered_as_named(self, series_id: int, rows: list[SonarrEpisode]) -> bool:
        """Whether a read of the series alone, made after `rows`, names an order they are numbered by (`_in_step`).

        The order is that read's and not the library read's: an import follows a change of order Sonarr finished
        during the run, which the library read's order would hold back until the next one.

        False when the read fails, a reply without the order among them, and when Sonarr answers that it has no
        such series.
        """

        if not self._library.sends_order:
            return True
        current = self._sonarr.series(series_id)
        if isinstance(current, NoResource) or not current.order_sent:
            return False
        return self._in_step(series_id, current, rows)

    def _hold(self, series_id: int, warning: str) -> None:
        """Leave the series without an aired view for the rest of the run, with one warning.

        Nothing the run holds of it changes. Its titles are not cached, so the next run checks them.
        """

        self._held.add(series_id)
        hub_warn(warning)

    def _in_step(self, series_id: int, current: SonarrItem, rows: list[SonarrEpisode]) -> bool:
        """Whether `rows` are numbered by the order `current`, a read of the series alone, names (`_numbering`).

        Rows that aren't are warned of once per series per run, whichever of the check and the import asks first.
        """

        numbering = self._numbering(rows, series_orders(current))
        if numbering is _Numbering.IN_STEP:
            return True
        if series_id not in self._out_of_step:
            self._out_of_step.add(series_id)
            name = _name_of(series_id, current)
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
