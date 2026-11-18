"""Sonarr episode-domain collaborator: series enumeration + episode resolution.

`SonarrEpisodes` owns the per-run episode cache (and the series-id hash
that pins the `/parse` cache rows with no series) and the logic that turns a (series,
AniList id, mapping) into the relevant episode list: season / AniBridge
filtering, AniDB remaps, offset slicing, the specials chosen by air date, the
existing files as `ArrFiles`, and the concurrent prefetch warm. It also keeps each
series' TVDB order facts. A title's check reads every series as aired (`aired_view`),
whatever order Sonarr numbers it by, and the import reads Sonarr's own rows.
"""

import concurrent.futures
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import override

from . import aired_view as _aired_view
from .aired_view import AiredSeries, NoView, is_stand_in
from .anibridge import TvdbMappings, season_holds
from .arr_models import ArrFiles, SonarrEpisode, SonarrItem
from .arr_run_services import ArrRunDeps, ArrRunServices
from .config import AppConfig
from .library_filter import IdField, build_id_filters, collect_anime_items
from .mappings import ExternalIds, MappingKind, MappingSource, TitleMapping
from .output import Severity, hub_note
from .placement_types import EpisodeIndex, episode_index
from .progress import ProgressCallback
from .releases import season_episode_key
from .sonarr_client import AbstractSonarrClient
from .specials_by_date import specials_by_date
from .tvdb_orders import NO_ORDERS, OrdersByName, SonarrSeriesOrders

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

    Invalidates the `/parse` cache rows with no series: a parse with no series almost always means
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


_NO_FACTS = SonarrSeriesOrders(tvdb_id=0, season_type="", counts=None)
"""The facts of a series this run's library read doesn't have. It names no order and is read as aired."""


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
    """The aired view of `rows`. None until it is first asked for."""

    index: EpisodeIndex | None = None
    """The view's placement index. None until it is first asked for, and for a series with no view."""


class EpisodeSource(ABC):
    """The episode reads a title's check and a series' listings ask for, every series numbered as aired.

    `SonarrEpisodes` is the real one. `title_episodes` raises `ValueError` for a mapping it can't resolve.
    """

    @property
    @abstractmethod
    def series_hash(self) -> str: ...

    @abstractmethod
    def title_episodes(self, series_id: int, anilist_id: int, mapping: TitleMapping) -> TitleEpisodes | NoView: ...

    @abstractmethod
    def series_index(self, series_id: int) -> EpisodeIndex | None: ...

    @abstractmethod
    def sonarr_files(self, episodes: Iterable[SonarrEpisode]) -> ArrFiles: ...

    @abstractmethod
    def episode_count(self, anilist_id: int) -> int | None: ...

    @abstractmethod
    def any_air_dates_unread(self, anilist_ids: Iterable[int]) -> bool: ...

    @abstractmethod
    def orders_unread(self, series_ids: Iterable[int]) -> bool: ...


class SonarrEpisodes(EpisodeSource):
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
        # refresh_library. Pins the `/parse` cache rows with no series to the library
        # state so they are parsed again once Sonarr has the series.
        self._series_hash: str = ""

        # Each series row's order facts, rebuilt whole in refresh_library.
        self._order_facts: dict[int, SonarrSeriesOrders] = {}

        # The AniList ids whose choice of specials by air date this run already logged. Reset in refresh_library.
        self._picks_logged: set[int] = set()

    @property
    @override
    def series_hash(self) -> str:
        """The current run's series-id hash (read by the parse cache)."""

        return self._series_hash

    def refresh_library(self) -> list[SonarrItem]:
        """Enumerate the AniList-mapped Sonarr series. The arr run's start reset point.

        Drops the per-run episode cache and rehashes the series-id set so a
        fresh run always re-reads the current library. Called once per run from the
        strategy's `get_items` hook.
        """

        self._series = {}
        self._picks_logged = set()
        series = self.get_all_sonarr_series()
        # Hash the current series-id set once per run to invalidate the parse cache
        # rows with no series (see sonarr_series_hash).
        self._series_hash = sonarr_series_hash(s.id for s in series)
        self._order_facts = {
            s.id: SonarrSeriesOrders(
                tvdb_id=s.tvdbId,
                season_type=s.seasonType,
                counts={listed.type: listed.episode_count for listed in s.seasonTypes} if s.seasonTypes else None,
            )
            for s in series
        }
        return series

    def get_all_sonarr_series(self) -> list[SonarrItem]:
        """Get all series in Sonarr with AniList mapping info."""

        fields = (IdField("tvdb_id", "tvdbId"), IdField("imdb_id", "imdbId"))
        return collect_anime_items(
            self.sonarr.all_series,
            build_id_filters(fields, self._resolver, self.anibridge),
        )

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

        A series absent from this run's library read is left out: there is no TVDB id to ask for.
        """

        by_tvdb = {
            facts.tvdb_id: facts
            for sid in series_ids
            if (facts := self._order_facts.get(sid)) is not None and facts.tvdb_id > 0
        }
        return self._skyhook.prefetch(list(by_tvdb.values()), dry_run=self._services.is_dry_run(), progress=progress)

    @override
    def episode_count(self, anilist_id: int) -> int | None:
        """AniList's episode count for the title, None when AniList gives none."""

        return self._anilist.episode_count(anilist_id)

    @override
    def any_air_dates_unread(self, anilist_ids: Iterable[int]) -> bool:
        """Whether AniList left any of the titles unread (`AniListGateway.air_dates_unread`)."""

        return any(self._anilist.air_dates_unread(anilist_id) for anilist_id in anilist_ids)

    @override
    def orders_unread(self, series_ids: Iterable[int]) -> bool:
        """Whether any of the series has no stored orders, as when SkyHook was unreachable before its first fetch.

        A series this run's library read doesn't have is never unread: nothing could fetch it.
        """

        return any(
            (facts := self._order_facts.get(sid)) is not None
            and facts.tvdb_id > 0
            and not self._skyhook.has_cached_answer(facts.tvdb_id)
            for sid in series_ids
        )

    def aired_series(self, series_id: int) -> AiredSeries | NoView:
        """The series' episodes as aired (`aired_view.aired_series`), or why it has no view.

        Sonarr's rows are fetched when the run holds none, and the view is built once per read of them.
        """

        read = self._read(series_id)
        if read is None:
            return NoView.READ_FAILED
        if read.view is None:
            facts = self._order_facts.get(series_id, _NO_FACTS)
            stored = self._skyhook.cached_orders(facts.tvdb_id) if facts.in_another_order else None
            read.view = _aired_view.aired_series(read.rows, facts, stored)
        return read.view

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

        facts = self._order_facts.get(series_id, _NO_FACTS)
        return self._skyhook.other_orders(facts.tvdb_id) if facts.season_type else NO_ORDERS

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

    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """The series' episodes as Sonarr numbers them, read now, refreshing the per-run cache.

        None when the read failed.
        """

        fetched = self.sonarr.episodes(series_id)
        if fetched is not None:
            self._store(series_id, fetched)
        return fetched

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
