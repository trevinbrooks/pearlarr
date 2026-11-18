"""What a run read of Sonarr's series: its library read, each series' rows, and what is built from the rows.

A title's check reads every series as aired (`aired_view`), whatever order Sonarr numbers it by, the episodes Sonarr
matched a file name to included (`aired_parse`). The import reads Sonarr's own rows. A series in another order takes
its scene numbers from TheXEM's mappings (`scene_numbers`), since Sonarr keeps none for it.

Whether a series' rows can be read at all is asked of `row_order_guard`, which holds every rule for it.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never, override

from . import aired_view as _aired_view
from .aired_view import AiredSeries, NoView
from .arr_models import AlternateTitle, ParsedFileInfo, SonarrEpisode
from .library_read import NO_LIBRARY, LibraryRead
from .placement_types import EpisodeIndex, episode_index
from .releases import season_episode_key
from .row_order_guard import AiredRows, RowOrderGuard
from .scene_numbers import scene_numbers
from .skyhook_gateway import SkyHookGateway
from .sonarr_client import AbstractSonarrClient
from .sonarr_matching import AiredMatcher, NoAnswer
from .tvdb_orders import NO_ORDERS, OrdersByName, SonarrSeriesOrders
from .xem_gateway import XemGateway

_GONE = AiredSeries(in_sonarr=(), stand_ins=())
"""The view of a series Sonarr answers it doesn't have: no episodes."""


@dataclass(slots=True)
class _SeriesRead:
    """What this run last read of one series, and what it built from that read."""

    rows: list[SonarrEpisode]
    """The series' episodes as Sonarr sent them."""

    view: AiredSeries | NoView | None = None
    """The aired view of `rows`. None until one is built, and never `NoView.READ_FAILED`, which is asked for again."""

    index: EpisodeIndex | None = None
    """The view's placement index. None until it is first asked for, and for a series with no view."""

    matcher: AiredMatcher | None = None
    """Matches a parsed file name to the view's episodes. None until a parse of a series in another order needs it."""


class AiredParseSource(ABC):
    """What the parse cache asks of a run's series: what its rows are good under, and a parse's aired matches.

    `SeriesReads` is the real one.
    """

    @property
    @abstractmethod
    def series_hash(self) -> str: ...

    @property
    @abstractmethod
    def series_in_another_order(self) -> Mapping[int, str]: ...

    @abstractmethod
    def aired_parse(self, info: ParsedFileInfo) -> ParsedFileInfo | NoAnswer | NoView: ...


class SeriesRows(ABC):
    """A series' rows as Sonarr numbers them, which an import reads its files off and follows a pending import by.

    `SeriesReads` is the real one.
    """

    @abstractmethod
    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None: ...

    @abstractmethod
    def last_known_episodes(self, series_id: int) -> list[SonarrEpisode]: ...


class SeriesReads(AiredParseSource, SeriesRows):
    """A run's library read and its per-series episode cache, with each series' aired view, index and matcher.

    Built once in `SonarrStrategy` and shared by the title check, the parse cache and the import, so each of them
    reads a series off the same rows. The TVDB orders come off the SkyHook gateway, read once per series per run,
    and the scene mappings of a series in another order off the TheXEM gateway.
    """

    def __init__(self, sonarr: AbstractSonarrClient, skyhook: SkyHookGateway, xem: XemGateway) -> None:
        """Bind the strategy's Sonarr client and the two gateways. No run has started, so no series is known."""

        self._sonarr = sonarr
        self._skyhook = skyhook
        self._xem = xem
        self._library = NO_LIBRARY
        self._guard = RowOrderGuard(sonarr, skyhook, NO_LIBRARY)

        # Each series' rows and what is built from them, kept for the run so the several AniList ids of a
        # multi-season series share one fetch. Reset at the arr run's start (`begin_run`).
        self._series: dict[int, _SeriesRead] = {}

    def begin_run(self, library: LibraryRead) -> None:
        """Start a run over its library read. Every series' rows go, and so does every hold on a series."""

        self._library = library
        self._guard = RowOrderGuard(self._sonarr, self._skyhook, library)
        self._series = {}

    @property
    def library(self) -> LibraryRead:
        """This run's library read, the same for the whole run."""

        return self._library

    @property
    @override
    def series_hash(self) -> str:
        """The current run's series-id hash (read by the parse cache)."""

        return self._library.series_hash

    @property
    @override
    def series_in_another_order(self) -> Mapping[int, str]:
        """Each series the library read has in an order other than the aired one, with that order.

        Read by the parse cache. The same for the whole run.
        """

        return self._library.in_another_order

    def library_orders(self, series_id: int) -> SonarrSeriesOrders:
        """What this run's library read says of the series' orders, the same for the whole run.

        A series the library read doesn't have names no order and has no count.
        """

        return self._library.series(series_id).orders

    def alternate_titles(self, series_id: int) -> tuple[AlternateTitle, ...]:
        """The series' other names in Sonarr, in Sonarr's order. None for a series the library read doesn't have."""

        return self._library.series(series_id).alternate_titles

    def aired_series(self, series_id: int) -> AiredSeries | NoView:
        """The series' episodes as aired (`aired_view.aired_series`), or why it has no view.

        Sonarr's rows are fetched when the run holds none, and the view is built once per read of them. A series
        held for the run (`RowOrderGuard.holds`) has none, and Sonarr is asked nothing more about it.
        """

        if self._guard.holds(series_id):
            return NoView.READ_FAILED
        read = self._read(series_id)
        if read is None:
            return NoView.READ_FAILED
        if read.view is None:
            view = self._view_of(series_id, read.rows)
            if view is NoView.READ_FAILED:
                # Not kept with the rows, so the next call asks Sonarr again.
                return view
            read.view = view
        return read.view

    def _view_of(self, series_id: int, rows: list[SonarrEpisode]) -> AiredSeries | NoView:
        """The aired view of rows just read, under the order the library read gives the series.

        A series in another order is renumbered by TVDB episode id, which is right whatever numbers its rows
        carry, and then given its scene numbers (`_scene_numbered`). Rows taken as already aired are right only if
        the series is still in aired order, so the guard confirms them first (`RowOrderGuard.aired_rows`).
        """

        facts = self._library.series(series_id).orders
        if facts.in_another_order:
            view = _aired_view.aired_series(rows, facts, self._skyhook.cached_orders(facts.tvdb_id))
            return view if isinstance(view, NoView) else self._scene_numbered(view, facts.tvdb_id)
        confirmed = self._guard.aired_rows(series_id, rows)
        match confirmed:
            case AiredRows.CONFIRMED:
                return _aired_view.aired_series(rows, facts, None)
            case AiredRows.GONE:
                return _GONE
            case AiredRows.UNREAD:
                return NoView.READ_FAILED
        assert_never(confirmed)

    def _scene_numbered(self, view: AiredSeries, tvdb_id: int) -> AiredSeries | NoView:
        """The view of a series in another order with the scene numbers TheXEM's stored mappings give its episodes.

        Sonarr keeps no scene numbers for such a series, and the view's builder clears the ones its rows carry.
        Mappings never read give no view: a file named by its scene numbers would be matched by the aired ones.
        Mappings read at any age give one, and so does a series TheXEM has none for.
        """

        mappings = self._xem.scene_mappings(tvdb_id)
        if mappings is None:
            return NoView.SCENE_UNREAD
        aired = (season_episode_key(ep.season_number, ep.episode_number) for ep in view.aired)
        return view.with_scene_numbers(scene_numbers(aired, mappings))

    def series_index(self, series_id: int) -> EpisodeIndex | None:
        """The whole series' placement index, numbered as aired, with its other TVDB orders read onto it.

        Stand-ins included. Sonarr's rows the aired order lacks are held apart (`EpisodeIndex.own_only`). Built once
        per read of the series' rows. None when the series has no view.
        """

        view = self.aired_series(series_id)
        if isinstance(view, NoView):
            return None
        read = self._series[series_id]
        if read.index is None:
            read.index = episode_index(view.aired, self._other_orders(series_id), view.own_only)
        return read.index

    def _other_orders(self, series_id: int) -> OrdersByName:
        """The series' stored orders other than the aired one. None of them for a series that names no order."""

        facts = self._library.series(series_id).orders
        return self._skyhook.other_orders(facts.tvdb_id) if facts.season_type else NO_ORDERS

    @override
    def aired_parse(self, info: ParsedFileInfo) -> ParsedFileInfo | NoAnswer | NoView:
        """The parse with the matched episodes an aired Sonarr gives it, or why its answer can't be known.

        Only a parse Sonarr matched to a series it numbers by another order changes. Sonarr matched that name by
        its own order's numbers, so its matched episodes are never passed on: they are matched again over the
        series' aired view (`aired_series`, which fetches a series the run hasn't read), by the series' alternate
        titles (`AiredMatcher`), and one may be a stand-in. A series with no aired view answers why it has none,
        and so does a parse of a series held for the run (`RowOrderGuard.holds`).
        """

        series_id = info.series_id
        if series_id is None:
            return info
        if self._guard.holds(series_id):
            # Asked first: Sonarr may have matched the name by an order the library read doesn't know of.
            return NoView.READ_FAILED
        series = self._library.series(series_id)
        if not series.orders.in_another_order:
            return info
        view = self.aired_series(series_id)
        if isinstance(view, NoView):
            return view
        read = self._series[series_id]
        if read.matcher is None:
            read.matcher = AiredMatcher.of(view, series.alternate_titles)
        matched = read.matcher.matches(info)
        return matched if isinstance(matched, NoAnswer) else info.with_matched(matched)

    def _read(self, series_id: int) -> _SeriesRead | None:
        """What the run holds of the series, its rows fetched when it holds none. None when the fetch failed."""

        read = self._series.get(series_id)
        if read is None:
            rows = self._sonarr.episodes(series_id)
            if rows is None:
                return None
            read = self._series[series_id] = _SeriesRead(rows)
        return read

    def store(self, series_id: int, rows: list[SonarrEpisode]) -> None:
        """Replace what the run holds of the series with a new read. What was built from the older rows goes too."""

        self._series[series_id] = _SeriesRead(rows)

    @override
    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """The series' episodes as Sonarr numbers them, read now, refreshing the per-run cache.

        None when the read failed. None too when the rows aren't numbered by the order Sonarr names for the series
        once it has sent them (`RowOrderGuard.numbered_as_named`). Such rows are not kept, so the run goes on
        holding the ones it had.
        """

        fetched = self._sonarr.episodes(series_id)
        if fetched is None or not self._guard.numbered_as_named(series_id, fetched):
            return None
        self.store(series_id, fetched)
        return fetched

    @override
    def last_known_episodes(self, series_id: int) -> list[SonarrEpisode]:
        """The series' last successfully read episodes this run, as Sonarr numbers them, or empty."""

        read = self._series.get(series_id)
        return [] if read is None else read.rows
