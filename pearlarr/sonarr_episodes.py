"""Sonarr episode-domain collaborator: series enumeration + episode resolution.

`SonarrEpisodes` owns the per-run episode cache (and the series-id hash that pins
the `/parse` cache rows with no matched episodes) and the logic that turns a (series,
AniList id, mapping) into the relevant episode list: season / AniBridge
filtering, AniDB remaps, offset slicing, the specials chosen by air date, the
existing files as `ArrFiles`, and the concurrent prefetch warm. It also keeps each
series' TVDB order facts and alternate titles. A title's check reads every series as
aired (`aired_view`), whatever order Sonarr numbers it by, the episodes Sonarr matched a
file name to included (`aired_parse`). The import reads Sonarr's own rows.

Rows are read as already aired only when Sonarr says twice that the series is: in the
library read, and in a read of the series alone made after its rows. For the check and
the import alike, rows count only while they are numbered by the order Sonarr names for
the series, as far as its stored TVDB orders can tell (`aired_view.numbered_by`).
"""

import concurrent.futures
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import assert_never, override

from . import aired_view as _aired_view
from .aired_view import AiredSeries, NoView, is_stand_in
from .anibridge import TvdbMappings, season_holds
from .arr_http import NoResource
from .arr_models import AlternateTitle, ArrFiles, ParsedFileInfo, SonarrEpisode, SonarrItem
from .arr_run_services import ArrRunDeps, ArrRunServices
from .config import AppConfig
from .library_filter import IdField, build_id_filters, collect_anime_items
from .mappings import ExternalIds, MappingKind, MappingSource, TitleMapping
from .output import Severity, hub_note, hub_warn
from .placement_types import EpisodeIndex, episode_index
from .progress import ProgressCallback
from .releases import season_episode_key
from .sonarr_client import AbstractSonarrClient
from .sonarr_matching import AiredMatcher, NoAnswer
from .specials_by_date import specials_by_date
from .tvdb_orders import AIRED_ORDER, NO_ORDERS, OrdersByName, SonarrSeriesOrders

# Bounded concurrency for the episode/parse network fan-out. Only used when
# advanced.sleep_time == 0. Must stay <= the session's pool_maxsize (ArrRunDeps.build).
# Episode fetches are I/O-bound, but a typical Sonarr (often behind a reverse
# proxy) saturates around ~10-12 concurrent, so larger values don't help.
SONARR_FETCH_WORKERS = 12


def fetch_workers(config: AppConfig) -> int:
    """Concurrency for the episode / parse network fan-out.

    Sequential (1) whenever a rate-limit throttle is requested
    (`advanced.sleep_time > 0`), so the bounded pool never bypasses the configured
    throttle. Otherwise, the bounded pool size. Shared by the episode
    prefetch and the parse warm, so it lives at module scope, owned by neither.
    """

    if config.advanced.sleep_time > 0:
        return 1
    return SONARR_FETCH_WORKERS


def sonarr_series_hash(series_ids: Iterable[int]) -> str:
    """Stable hash of the current Sonarr series-id set.

    Invalidates the `/parse` cache rows with no matched episodes: such a parse most often means
    the file's series isn't in Sonarr, so a new series id flips the hash and
    re-parses the affected rows. `hashlib` (not `hash()`) for stability
    across processes.

    Args:
        series_ids: Current Sonarr series ids (sorted/de-duped).
    """

    joined = ",".join(str(i) for i in sorted(set(series_ids)))
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def check_ep_by_anime_ids(
    ep: SonarrEpisode,
    tvdb_season: int,
) -> bool:
    """Check whether to include an episode by Anime ID style.

    A `tvdb_season` of -1 matches every season except specials (season 0).
    """

    if tvdb_season == -1:
        return ep.season_number != 0  # -1 = anything but specials
    return ep.season_number == tvdb_season


def check_ep_by_anibridge(
    ep: SonarrEpisode,
    tvdb_mappings: TvdbMappings,
) -> bool:
    """Check whether a Sonarr episode is covered by an AniBridge mapping.

    `tvdb_mappings` maps a season to its inclusive `(start, end)` TVDB episode
    ranges, and `season_holds` checks the episode against its season's.
    """

    ep_season = ep.season_number if ep.season_number is not None else -1
    ep_episode = ep.episode_number if ep.episode_number is not None else -1

    ranges = tvdb_mappings.get(ep_season)

    # Season isn't part of this mapping at all
    if ranges is None:
        return False

    return season_holds(ranges, ep_episode)


def _dated(episodes: Iterable[SonarrEpisode]) -> str:
    """The episodes for a log line, each as its `S00E01` label and air date, comma-joined."""

    return ", ".join(
        f"{season_episode_key(ep.season_number, ep.episode_number).label} (aired {ep.air_date})" for ep in episodes
    )


@dataclass(frozen=True, slots=True)
class _LibrarySeries:
    """What this run's library read says of one series."""

    orders: SonarrSeriesOrders
    """Its TVDB id and the order Sonarr numbers it by."""

    alternate_titles: tuple[AlternateTitle, ...] = ()
    """Its other names in Sonarr, in Sonarr's order."""


_NOT_IN_LIBRARY = _LibrarySeries(SonarrSeriesOrders(tvdb_id=0, season_type="", counts=None))
"""A series this run's library read doesn't have: it names no order and has no other names."""

_GONE = AiredSeries(order="", in_sonarr=(), stand_ins=())
"""The view of a series Sonarr answers it doesn't have: no episodes, under no order."""


class _Numbering(Enum):
    """Whether a series' rows are numbered by the order Sonarr names for it (`SonarrEpisodes._numbering`)."""

    IN_STEP = auto()
    """They are, or nothing stored can tell."""

    OUT_OF_STEP = auto()
    """They are not, by an answer SkyHook gave this run."""

    UNCONFIRMED = auto()
    """They are not by the stored answer, and SkyHook couldn't be reached for a new one."""


def _name_of(series_id: int, series: SonarrItem) -> str:
    """The series' name for a warning: its Sonarr title, or its id when Sonarr sent none."""

    return series.title or f"Series {series_id}"


def series_orders(series: SonarrItem) -> SonarrSeriesOrders:
    """What one Sonarr series resource says about its orders, from the library read or a read of the series alone."""

    return SonarrSeriesOrders(
        tvdb_id=series.tvdbId,
        season_type=series.seasonType,
        counts={listed.type: listed.episode_count for listed in series.seasonTypes} if series.seasonTypes else None,
        row_count=series.totalEpisodeCount,
    )


@dataclass(frozen=True, slots=True)
class TitleEpisodes:
    """One title's episodes of its series, numbered as aired."""

    order: str
    """The order Sonarr numbers the series by (`AiredSeries.order`)."""

    aired: tuple[SonarrEpisode, ...]
    """Every aired episode of the title, a stand-in for each one Sonarr has no row for."""

    in_sonarr: tuple[SonarrEpisode, ...] = field(init=False)
    """The ones Sonarr has a row for."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_sonarr", tuple(ep for ep in self.aired if not is_stand_in(ep.id)))

    def index(self) -> EpisodeIndex:
        """The title's placement index, stand-ins included."""

        return episode_index(self.aired)


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


class EpisodeSource(ABC):
    """The episode reads a title's check, a series' listings and the parse cache ask for, every series numbered as aired.

    `SonarrEpisodes` is the real one. `title_episodes` raises `ValueError` for a mapping it can't resolve.
    """

    @property
    @abstractmethod
    def series_hash(self) -> str: ...

    @property
    @abstractmethod
    def series_in_another_order(self) -> Mapping[int, str]: ...

    @abstractmethod
    def title_episodes(self, series_id: int, anilist_id: int, mapping: TitleMapping) -> TitleEpisodes | NoView: ...

    @abstractmethod
    def series_index(self, series_id: int) -> EpisodeIndex | None: ...

    @abstractmethod
    def library_orders(self, series_id: int) -> SonarrSeriesOrders: ...

    @abstractmethod
    def alternate_titles(self, series_id: int) -> tuple[AlternateTitle, ...]: ...

    @abstractmethod
    def aired_parse(self, info: ParsedFileInfo) -> ParsedFileInfo | NoAnswer: ...

    @abstractmethod
    def sonarr_files(self, episodes: Iterable[SonarrEpisode]) -> ArrFiles: ...

    @abstractmethod
    def episode_count(self, anilist_id: int) -> int | None: ...

    @abstractmethod
    def metadata_unread(self, anilist_ids: Iterable[int], series_ids: Iterable[int] = ()) -> bool: ...


class SeriesRows(ABC):
    """A series' rows as Sonarr numbers them, which an import reads its files off and follows a pending import by.

    `SonarrEpisodes` is the real one.
    """

    @abstractmethod
    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None: ...

    @abstractmethod
    def last_known_episodes(self, series_id: int) -> list[SonarrEpisode]: ...


class SonarrEpisodes(EpisodeSource, SeriesRows):
    """Owns the per-run episode cache + the (series, anilist_id, mapping) -> episodes logic, and each series' orders.

    Constructed once per run in `SonarrStrategy` from the shared
    `ArrRunDeps`, the strategy's Sonarr client, and the arr run
    services (kept as `self._services` for the prefetch needs-check gate).
    The TVDB orders come off `deps.skyhook`, read once per series per run.
    """

    def __init__(self, deps: ArrRunDeps, sonarr: AbstractSonarrClient, services: ArrRunServices) -> None:
        """Bind the shared collaborators the episode logic reads.

        Args:
            deps: The shared collaborators (config/mappings/AniList
                gateway/SkyHook order gateway are unpacked off it).
            sonarr: The strategy's Sonarr client (built once,
                reused so a multi-season series fetches its episodes once).
            services: The arr run services. `prefetch` calls into it to
                resolve AniList ids and the needs-check gate.
        """

        self.sonarr = sonarr
        self._services = services
        self._config = deps.config
        self._resolver = deps.resolver
        self.anibridge = deps.resolver.anibridge
        # Episode counts / formats resolve through the gateway (shared with the
        # arr run loop), so its warm cache and retry narration apply here too.
        self._anilist = deps.anilist
        self._skyhook = deps.skyhook

        # Per-run cache of the raw Sonarr episode fetch and what is built from it, keyed by series id. A
        # multi-season series maps to several AniList ids, each of which would
        # otherwise re-fetch the same whole-series episode list. Cache it for the
        # run so the network round-trip happens once per series. Reset at the arr run's start
        # (refresh_library, driven by the strategy's get_items hook).
        self._series: dict[int, _SeriesRead] = {}

        # Hash of the current Sonarr series-id set, recomputed each run in
        # refresh_library. Pins the `/parse` cache rows with no matched episodes to the library
        # state so they are parsed again once Sonarr has the series.
        self._series_hash: str = ""

        # What the library read says of each series in Sonarr, with an AniList mapping or without. Rebuilt whole in
        # refresh_library and never changed inside a run, so a view is always built under the library read's order.
        self._library: dict[int, _LibrarySeries] = {}

        # Series id -> its order, for each series the library read has in an order other than the aired one.
        # Rebuilt in refresh_library with `_library`. A cached `/parse` row is good only under the order its
        # series was in when it was stored.
        self._in_another_order: Mapping[int, str] = {}

        # Whether Sonarr sent an order with any series of the library read (`SonarrItem.order_sent`). One that
        # sends none numbers every series as aired, and no series is read a second time for it.
        self._sends_order: bool = False

        # The series with no aired view for the rest of the run (`_hold`). Reset in refresh_library, with the rows.
        self._held: set[int] = set()

        # The series this run warned of rows out of step with their order (`_in_step`). Reset in refresh_library.
        self._out_of_step: set[int] = set()

        # The AniList ids whose choice of specials by air date this run already logged. Reset in refresh_library.
        self._picks_logged: set[int] = set()

    @property
    @override
    def series_hash(self) -> str:
        """The current run's series-id hash (read by the parse cache)."""

        return self._series_hash

    @property
    @override
    def series_in_another_order(self) -> Mapping[int, str]:
        """Each series the library read has in an order other than the aired one, with that order.

        Read by the parse cache. The same for the whole run.
        """

        return self._in_another_order

    def refresh_library(self) -> list[SonarrItem]:
        """Enumerate the AniList-mapped Sonarr series. The arr run's start reset point.

        Drops the per-run episode cache and rehashes the series-id set so a
        fresh run always re-reads the current library. Called once per run from the
        strategy's `get_items` hook.

        Sonarr is asked for its series once. The returned series and their hash are the mapped ones. What Sonarr
        says of every series is kept for the run, since a pending import may claim one with no mapping.
        """

        self._series = {}
        self._held = set()
        self._out_of_step = set()
        self._picks_logged = set()
        in_sonarr = self.sonarr.all_series()
        self._library = {s.id: _LibrarySeries(series_orders(s), s.alternateTitles) for s in in_sonarr}
        orders = ((series_id, series.orders.numbering.order) for series_id, series in self._library.items())
        self._in_another_order = {series_id: order for series_id, order in orders if order}
        self._sends_order = any(s.order_sent for s in in_sonarr)
        series = self.mapped_series(in_sonarr)
        # Hash the current series-id set once per run to invalidate the parse cache
        # rows with no matched episodes (see sonarr_series_hash).
        self._series_hash = sonarr_series_hash(s.id for s in series)
        return series

    def mapped_series(self, in_sonarr: list[SonarrItem]) -> list[SonarrItem]:
        """The series among Sonarr's with AniList mapping info, sorted by name."""

        fields = (IdField("tvdb_id", "tvdbId"), IdField("imdb_id", "imdbId"))
        return collect_anime_items(in_sonarr, build_id_filters(fields, self._resolver, self.anibridge))

    def prefetch(self, items: list[SonarrItem], *, progress: ProgressCallback | None = None) -> int:
        """Warm the per-series episode lists CONCURRENTLY before the scan loop.

        One sequential `/api/v3/episode` round-trip per processed series is the
        dominant sweep cost (~30s here). Fetching them up front over a bounded
        thread pool collapses that to a few seconds while keeping every list
        FRESH - the grab/skip decision reads each episode's existing file, so the
        lists are deliberately not persisted across runs. Only the network fetch
        runs in the pool. The per-run cache is populated on the
        main thread (the cache and mappings are not thread-safe), and a series'
        aired view is built there on first ask. A series whose
        fetch fails is left unwarmed and re-fetched (and logged) by
        `title_episodes` on the main thread during the loop.

        Only series the per-title loop would actually check are warmed: a series
        whose every title the loop short-circuits (no SeaDex entry, or cached
        and unchanged - `title_needs_check`) is left out, since `title_episodes`
        would never run for it. This keeps a warm run from re-fetching the episode
        lists the loop's unchanged-title short-circuit already obviates.

        Args:
            items: The run's series list (already narrowed for
                a single-series run).
            progress: Boot step fed per-series
                fraction + "done/total" detail as each fetch completes. None
                outside the boot.

        Returns:
            How many series were warmed (attempted) - the needs-check subset,
            for the caller's boot step detail. A series whose fetch returned None
            still counts.
        """

        series_ids = self.titles_needing_check(items)
        if not series_ids:
            return 0

        def warm(series_id: int) -> tuple[int, list[SonarrEpisode] | None]:
            # quiet: a transient failure here isn't logged from a worker. title_episodes
            # retries and logs it on the main thread if it still fails. A RAISE (not
            # just a None return) becomes None too, so one bad series can't abort
            # the whole concurrent sweep (the per-series failure handling the docstring describes).
            try:
                return series_id, self.sonarr.episodes(series_id, quiet=True)
            except Exception:
                return series_id, None

        # submit + as_completed (not pool.map): advance the bar as each series
        # FINISHES, so a slow series doesn't freeze the bar then jump. max_workers=1
        # (sleep_time throttle) still runs serially. Results are consumed on the main
        # thread, so the cache write and the progress drive are both single-threaded.
        total = len(series_ids)
        workers = min(fetch_workers(self._config), total)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(warm, sid) for sid in series_ids]
            try:
                for fut in concurrent.futures.as_completed(futures):
                    sid, eps = fut.result()
                    if eps is not None:
                        self._store(sid, eps)
                    done += 1
                    if progress is not None:
                        progress.progress(done / total, f"{done}/{total}")
            except BaseException:
                # A stop (SIGTERM, Ctrl-C) drops the fetches not yet started, which the `with` exit would otherwise run.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return total

    def titles_needing_check(self, items: Iterable[SonarrItem]) -> list[int]:
        """The ids of the series the per-title loop will check, in item order, each once.

        Leaves out a series whose every title the loop short-circuits on (`title_needs_check`), and an unmonitored
        one under `ignore_unmonitored`.
        """

        services = self._services
        series_ids: list[int] = []
        seen: set[int] = set()
        for item in items:
            if item.id in seen:
                continue
            if not item.monitored and self._config.sonarr.ignore_unmonitored:
                continue
            anilist_ids = services.title_mappings(ExternalIds.of_series(item), log_ignored=False)
            if not anilist_ids:
                continue
            if not any(services.title_needs_check(aid) for aid in anilist_ids):
                continue
            seen.add(item.id)
            series_ids.append(item.id)
        return series_ids

    def prefetch_orders(self, series_ids: Iterable[int], *, progress: ProgressCallback | None = None) -> int:
        """Refetch the due TVDB orders of the series (see `SkyHookGateway.prefetch`). Returns how many were written.

        A series is asked for by the TVDB id the library read gives it, with an AniList mapping or without: a
        pending import may claim one that has none. A series absent from the library read is left out, since there
        is no TVDB id to ask for.
        """

        in_library = (self._library[sid].orders for sid in series_ids if sid in self._library)
        by_tvdb = {orders.tvdb_id: orders for orders in in_library if orders.tvdb_id > 0}
        return self._skyhook.prefetch(list(by_tvdb.values()), dry_run=self._services.is_dry_run(), progress=progress)

    @override
    def episode_count(self, anilist_id: int) -> int | None:
        """AniList's episode count for the title, None when AniList gives none."""

        return self._anilist.episode_count(anilist_id)

    @override
    def metadata_unread(self, anilist_ids: Iterable[int], series_ids: Iterable[int] = ()) -> bool:
        """Whether a check that reads these titles and these series' orders is missing one of their answers.

        A title is unread when AniList left it so (`AniListGateway.air_dates_unread`). A series is unread when it has
        no stored orders, as when SkyHook was unreachable before its first fetch. That counts for a series with no
        AniList mapping too, which a pending import may claim. A series this run's library read doesn't have is
        never unread: nothing could fetch it.
        """

        if any(self._anilist.air_dates_unread(anilist_id) for anilist_id in anilist_ids):
            return True
        return any(
            (series := self._library.get(sid)) is not None
            and series.orders.tvdb_id > 0
            and not self._skyhook.has_cached_answer(series.orders.tvdb_id)
            for sid in series_ids
        )

    def aired_series(self, series_id: int) -> AiredSeries | NoView:
        """The series' episodes as aired (`aired_view.aired_series`), or why it has no view.

        Sonarr's rows are fetched when the run holds none, and the view is built once per read of them. A series
        held for the run (`_hold`) has none, and Sonarr is asked nothing more about it.
        """

        if series_id in self._held:
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
        carry. Rows taken as already aired are right only if the series is still in aired order, so on a Sonarr
        that sends an order they are confirmed first (`_confirmed_aired`).
        """

        facts = self._library_series(series_id).orders
        if facts.in_another_order:
            return _aired_view.aired_series(rows, facts, self._skyhook.cached_orders(facts.tvdb_id))
        if not self._sends_order:
            return _aired_view.aired_series(rows, facts, None)
        return self._confirmed_aired(series_id, rows)

    def _confirmed_aired(self, series_id: int, rows: list[SonarrEpisode]) -> AiredSeries | NoView:
        """The view of rows the library read takes as aired, once a read of the series alone agrees.

        The read is made after the rows: an order Sonarr changed before it sent them shows in it. The two reads
        agree when neither names another order. Counts and the TVDB id are not compared.

        A series whose order changed, and one the library read doesn't have, is held for the run (`_hold`). So is
        one whose rows aren't numbered by the aired order both reads name (`_in_step`). A failed read, a reply
        without the order among them, gives no view for this call only. A series Sonarr answers it doesn't have
        has no episodes.
        """

        current = self.sonarr.series(series_id)
        if current is NoResource.GONE:
            return _GONE
        if isinstance(current, NoResource) or not current.order_sent:
            return NoView.READ_FAILED
        name = _name_of(series_id, current)
        library = self._library.get(series_id)
        if library is None:
            self._hold(
                series_id,
                f"{name} not checked - the series was not in Sonarr when this run read its library - checking its "
                "titles next run",
            )
            return NoView.READ_FAILED
        if series_orders(current).in_another_order:
            self._hold(
                series_id,
                f"{name} not checked - the order Sonarr numbers the series by changed during this run - checking "
                "its titles again next run",
            )
            return NoView.READ_FAILED
        if not self._in_step(series_id, current, rows):
            # Warned of there, once for the check and the import together.
            self._held.add(series_id)
            return NoView.READ_FAILED
        return _aired_view.aired_series(rows, library.orders, None)

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

    @override
    def series_index(self, series_id: int) -> EpisodeIndex | None:
        """The whole series' placement index, numbered as aired, with its other TVDB orders read onto it.

        Stand-ins included. Built once per read of the series' rows. None when the series has no view.
        """

        view = self.aired_series(series_id)
        if isinstance(view, NoView):
            return None
        read = self._series[series_id]
        if read.index is None:
            read.index = episode_index(view.aired, self._other_orders(series_id))
        return read.index

    def _other_orders(self, series_id: int) -> OrdersByName:
        """The series' stored orders other than the aired one. None of them for a series that names no order."""

        facts = self._library_series(series_id).orders
        return self._skyhook.other_orders(facts.tvdb_id) if facts.season_type else NO_ORDERS

    def _library_series(self, series_id: int) -> _LibrarySeries:
        """What the library read says of the series, `_NOT_IN_LIBRARY` for one it doesn't have."""

        return self._library.get(series_id, _NOT_IN_LIBRARY)

    @override
    def library_orders(self, series_id: int) -> SonarrSeriesOrders:
        """What this run's library read says of the series' orders, the same for the whole run.

        A series the library read doesn't have names no order and has no count.
        """

        return self._library_series(series_id).orders

    @override
    def alternate_titles(self, series_id: int) -> tuple[AlternateTitle, ...]:
        """The series' other names in Sonarr, in Sonarr's order. None for a series the library read doesn't have."""

        return self._library_series(series_id).alternate_titles

    @override
    def aired_parse(self, info: ParsedFileInfo) -> ParsedFileInfo | NoAnswer:
        """The parse with the matched episodes an aired Sonarr gives it, or why its answer can't be known.

        Only a parse Sonarr matched to a series it numbers by another order changes. Sonarr matched that name by
        its own order's numbers, so its matched episodes are never passed on: they are matched again over the
        series' aired view (`aired_series`, which fetches a series the run hasn't read), by the series' alternate
        titles (`AiredMatcher`), and one may be a stand-in. A parse of a series held for the run (`_hold`) has no
        answer.
        """

        series_id = info.series_id
        if series_id is None:
            return info
        if series_id in self._held:
            # Asked first: Sonarr may have matched the name by an order the library read doesn't know of.
            return NoAnswer.EPISODES_UNREAD
        series = self._library_series(series_id)
        if not series.orders.in_another_order:
            return info
        view = self.aired_series(series_id)
        match view:
            case NoView.READ_FAILED:
                return NoAnswer.EPISODES_UNREAD
            case NoView.ORDERS_UNREAD:
                return NoAnswer.ORDERS_UNREAD
            case AiredSeries():
                read = self._series[series_id]
                if read.matcher is None:
                    read.matcher = AiredMatcher.of(view, series.alternate_titles)
                matched = read.matcher.matches(info)
                return matched if isinstance(matched, NoAnswer) else info.with_matched(matched)
        assert_never(view)

    def _read(self, series_id: int) -> _SeriesRead | None:
        """What the run holds of the series, its rows fetched when it holds none. None when the fetch failed."""

        read = self._series.get(series_id)
        if read is None:
            rows = self.sonarr.episodes(series_id)
            if rows is None:
                return None
            read = self._store(series_id, rows)
        return read

    def _store(self, series_id: int, rows: list[SonarrEpisode]) -> _SeriesRead:
        """Replace what the run holds of the series with a new read. What was built from the older rows goes too."""

        read = self._series[series_id] = _SeriesRead(rows)
        return read

    @override
    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """The series' episodes as Sonarr numbers them, read now, refreshing the per-run cache.

        None when the read failed. None too, on a Sonarr that sends an order, when the rows aren't numbered by the
        order Sonarr names for the series once it has sent them (`_numbered_as_named`). Such rows are not kept, so
        the run goes on holding the ones it had.
        """

        fetched = self.sonarr.episodes(series_id)
        if fetched is None or (self._sends_order and not self._numbered_as_named(series_id, fetched)):
            return None
        self._store(series_id, fetched)
        return fetched

    def _numbered_as_named(self, series_id: int, rows: list[SonarrEpisode]) -> bool:
        """Whether a read of the series alone, made after `rows`, names an order they are numbered by (`_in_step`).

        The order is that read's and not the library read's: an import follows a change of order Sonarr finished
        during the run, which the library read's order would hold back until the next one.

        False when the read fails, a reply without the order among them, and when Sonarr answers that it has no
        such series.
        """

        current = self.sonarr.series(series_id)
        if isinstance(current, NoResource) or not current.order_sent:
            return False
        return self._in_step(series_id, current, rows)

    @override
    def last_known_episodes(self, series_id: int) -> list[SonarrEpisode]:
        """The series' last successfully read episodes this run, as Sonarr numbers them, or empty."""

        read = self._series.get(series_id)
        return [] if read is None else read.rows

    @override
    def title_episodes(self, series_id: int, anilist_id: int, mapping: TitleMapping) -> TitleEpisodes | NoView:
        """The episodes of the series an AniList mapping covers, numbered as aired, or why the series has no view.

        They are the mapping's episodes, unless those are specials that aired off AniList's dates and
        `specials_by_date` chooses others of the series.
        """

        tvdb_season = mapping.tvdb_season

        if anilist_id == -1:
            raise ValueError("AniList ID not defined!")

        anidb_id = mapping.anidb_id

        mode = mapping.kind

        # Get all the episodes for the whole series. The fetch is per-series (not
        # per-AniList-id), so a multi-season series resolving to several ids would
        # otherwise re-request the identical list. Cache it per series for the run
        # and only do the per-id filtering below on the shared, read-only view.
        view = self.aired_series(series_id)
        if isinstance(view, NoView):
            return view
        series_episodes = view.aired

        # Filter down here by various things. Resolve the include test once by
        # mapping kind rather than re-branching on a string for every episode. The
        # comprehension preserves the series' order, exactly as the append loop did.
        if mode is MappingKind.SEASON_OFFSET:
            if mapping.source is MappingSource.ANIBRIDGE:
                # AniBridge mapping without TVDB mappings (imdb/tmdb-resolved, so no tvdb season
                # ranges): tvdb_season=-1 would otherwise grab the wrong episodes.
                # Return none. check_title surfaces NO_EPISODES + the AniBridge warning.
                return TitleEpisodes(order=view.order, aired=())
            mapped = [ep for ep in series_episodes if check_ep_by_anime_ids(ep=ep, tvdb_season=tvdb_season)]
        else:
            tvdb_mappings = mapping.tvdb_mappings or {}
            if not tvdb_mappings:
                # AniBridge attached no usable per-season ranges, so nothing resolves.
                # Return empty. check_title surfaces the visible NO_EPISODES row.
                return TitleEpisodes(order=view.order, aired=())
            mapped = [ep for ep in series_episodes if check_ep_by_anibridge(ep=ep, tvdb_mappings=tvdb_mappings)]

        # For OVAs and movies, the offsets can often be wrong, so if we have specific mappings
        # then take that into account here
        media_format = self._anilist.media_format(anilist_id)

        # Potentially pull out a bunch of mappings from AniDB. These should
        # be for anything not marked as TV, and specials as marked by
        # being in Season 0. The resolver owns the AniDB parse: it returns the
        # season's {tvdb_ep: anidb_ep} dict ({} when none) and raises on an
        # ambiguous id.
        anidb_episode_by_tvdb: dict[int, dict[int, int]] = {}
        if self._resolver.has_anidb and anidb_id is not None and (media_format != "TV" or tvdb_season == 0):
            anidb_episode_by_tvdb = self._resolver.anidb_episode_by_tvdb(anidb_id, tvdb_season)

        # Prefer the AniDB mapping dict over any offsets
        if len(anidb_episode_by_tvdb) > 0:
            in_anidb: list[SonarrEpisode] = []

            # See if we have the mapping for each episode
            for ep in mapped:
                season_number = ep.season_number
                episode_number = ep.episode_number
                if season_number is None or episode_number is None:
                    continue

                anidb_mapping_dict_entry = anidb_episode_by_tvdb.get(
                    season_number,
                    {},
                ).get(episode_number, None)
                if anidb_mapping_dict_entry is not None:
                    in_anidb.append(ep)

            # These episodes are read-only from here on (coverage,
            # sonarr_files, and the planner only read them), so we
            # return references into the shared cache rather than cloning.
            mapped = in_anidb

        # No AniDB mapping: anime-id mappings still need the offset slice, while
        # AniBridge mappings are already fully filtered above (no-op).
        elif mode is MappingKind.SEASON_OFFSET:
            mapped = self._apply_anime_id_offsets(
                season_episodes=mapped,
                anilist_id=anilist_id,
                mapping=mapping,
            )

        return TitleEpisodes(order=view.order, aired=tuple(self._picked_by_date(anilist_id, mapped, series_episodes)))

    def _picked_by_date(
        self, anilist_id: int, mapping_episodes: list[SonarrEpisode], series_episodes: Sequence[SonarrEpisode]
    ) -> list[SonarrEpisode]:
        """The title's episodes: `mapping_episodes`, or the specials of `series_episodes` AniList's dates choose over them.

        A choice is logged once per run, though a run resolves a title several times.
        """

        airing = self._anilist.airing(anilist_id)
        picked = specials_by_date(mapping_episodes, series_episodes, airing)
        if picked is None:
            return mapping_episodes
        if anilist_id not in self._picks_logged:
            self._picks_logged.add(anilist_id)
            dates = f"{airing.start}" if airing.end in (None, airing.start) else f"{airing.start} to {airing.end}"
            hub_note(
                f"AniList {anilist_id} maps to {_dated(mapping_episodes)} but is dated {dates} - using {_dated(picked)}",
                severity=Severity.DEBUG,
            )
        return picked

    def _apply_anime_id_offsets(
        self,
        season_episodes: list[SonarrEpisode],
        anilist_id: int,
        mapping: TitleMapping,
    ) -> list[SonarrEpisode]:
        """Slice an anime-id episode list down by its TVDB offset / AniList count.

        Args:
            season_episodes: Season-filtered episodes to slice.
            anilist_id: AniList ID, used to resolve the expected episode count.
            mapping: The title's mapping (read for `tvdb_epoffset` and `tvdb_season`.
                A `tvdb_season` of -1 means single-season offset slice).
        """

        # Slice the list to get the correct episodes, so any potential offsets
        ep_offset = mapping.tvdb_epoffset
        episode_count = self.episode_count(anilist_id)

        # If we don't get a number of episodes, use them all
        if episode_count is None:
            episode_count = len(season_episodes) - ep_offset

        # Check that we're including this by the episode number. This only
        # works for single-seasons, so be careful!
        if mapping.tvdb_season != -1:
            return [ep for ep in season_episodes if 1 <= (ep.episode_number or 0) - ep_offset <= episode_count]

        return season_episodes[ep_offset : episode_count + ep_offset]

    @override
    def sonarr_files(self, episodes: Iterable[SonarrEpisode]) -> ArrFiles:
        """Convert the episodes' existing files into an `ArrFiles`."""

        # keep_no_group=False: episode files with no group go with the episode list
        # into the planner's size matching instead (pinned).
        return ArrFiles.from_files(
            [f for ep in episodes if ep.episode_file_id != 0 and (f := ep.episode_file) is not None],
            keep_no_group=False,
        )
