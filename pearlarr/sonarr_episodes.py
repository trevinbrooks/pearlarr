"""Sonarr episode-domain collaborator: series enumeration + episode resolution.

`SonarrEpisodes` enumerates the library and turns a (series, AniList id, mapping) into
the relevant episode list: season / AniBridge filtering, AniDB remaps, offset slicing,
the specials chosen by air date, the existing files as `ArrFiles`, and the concurrent
prefetch warm. It reads every series as aired off the run's reads (`series_reads`),
which hold the library read and each series' rows.
"""

import concurrent.futures
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import override

from .aired_view import NoView
from .anibridge import TvdbMappings, season_holds
from .arr_http import fetch_workers
from .arr_models import ArrFiles, SonarrEpisode, SonarrItem
from .arr_run_services import ArrRunDeps, ArrRunServices
from .episode_ids import is_stand_in
from .library_filter import IdField, build_id_filters, collect_anime_items
from .library_read import LibraryRead
from .mappings import ExternalIds, MappingKind, MappingSource, TitleMapping
from .output import Severity, hub_note
from .placement_types import EpisodeIndex, episode_index
from .progress import ProgressCallback
from .releases import PrintedKey
from .series_reads import SeriesReads
from .sonarr_client import AbstractSonarrClient
from .specials_by_date import specials_by_date
from .tvdb_orders import SonarrSeriesOrders


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
    """The episodes for a log line, each as the number it prints by (`PrintedKey.of`) and its air date, comma-joined."""

    dated: list[str] = []
    for ep in episodes:
        printed = PrintedKey.of(ep)
        # A number marked as aired says the word itself.
        date = f"on {ep.air_date}" if printed.aired else f"aired {ep.air_date}"
        dated.append(f"{printed.label} ({date})")
    return ", ".join(dated)


@dataclass(frozen=True, slots=True)
class TitleEpisodes:
    """One title's episodes of its series, numbered as aired."""

    aired: tuple[SonarrEpisode, ...]
    """Every aired episode of the title, a stand-in for each one Sonarr has no row for."""

    in_sonarr: tuple[SonarrEpisode, ...] = field(init=False)
    """The ones Sonarr has a row for."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_sonarr", tuple(ep for ep in self.aired if not is_stand_in(ep.id)))

    def index(self) -> EpisodeIndex:
        """The title's placement index, stand-ins included."""

        return episode_index(self.aired)


class EpisodeSource(ABC):
    """A title's episodes of its series, numbered as aired, and AniList's count of them, as a series' listings ask.

    `SonarrEpisodes` is the real one. `title_episodes` raises `ValueError` for a mapping it can't resolve.
    """

    @abstractmethod
    def title_episodes(self, series_id: int, anilist_id: int, mapping: TitleMapping) -> TitleEpisodes | NoView: ...

    @abstractmethod
    def episode_count(self, anilist_id: int) -> int | None: ...


class SonarrEpisodes(EpisodeSource):
    """Enumerates the library, warms the run's reads, and owns the (series, anilist_id, mapping) -> episodes logic.

    Constructed once per run in `SonarrStrategy` from the shared
    `ArrRunDeps`, the strategy's Sonarr client, the arr run
    services (kept as `self._services` for the prefetch needs-check gate), and the run's reads.
    """

    def __init__(
        self, deps: ArrRunDeps, sonarr: AbstractSonarrClient, services: ArrRunServices, reads: SeriesReads
    ) -> None:
        """Bind the shared collaborators the episode logic reads.

        Four, one past the usual: `reads` comes in built, since the parse cache and the import read series off the
        same one.

        Args:
            deps: The shared collaborators (config/mappings/AniList
                gateway/SkyHook order gateway/TheXEM mapping gateway are unpacked off it).
            sonarr: The strategy's Sonarr client (built once,
                reused so a multi-season series fetches its episodes once).
            services: The arr run services. `prefetch` calls into it to
                resolve AniList ids and the needs-check gate.
            reads: The run's reads. Each series' rows and aired view come off it, and `refresh_library`
                starts its run.
        """

        self.sonarr = sonarr
        self._services = services
        self._reads = reads
        self._config = deps.config
        self._resolver = deps.resolver
        self.anibridge = deps.resolver.anibridge
        # Episode counts / formats resolve through the gateway (shared with the
        # arr run loop), so its warm cache and retry narration apply here too.
        self._anilist = deps.anilist
        self._skyhook = deps.skyhook
        self._xem = deps.xem

        # The AniList ids whose choice of specials by air date this run already logged. Reset in refresh_library.
        self._picks_logged: set[int] = set()

    def refresh_library(self) -> list[SonarrItem]:
        """Enumerate the AniList-mapped Sonarr series. The arr run's start reset point.

        Starts the run's reads over the library just read (`SeriesReads.begin_run`), which drops the per-run
        episode cache, so a fresh run always re-reads the current library. Called once per run from the
        strategy's `get_items` hook.

        Sonarr is asked for its series once. The returned series and their hash are the mapped ones. What Sonarr
        says of every series is kept for the run, since a pending import may claim one with no mapping.
        """

        in_sonarr = self.sonarr.all_series()
        series = self.mapped_series(in_sonarr)
        # Hash the mapped series-id set once per run to invalidate the parse cache
        # rows with no matched episodes (see sonarr_series_hash).
        self._reads.begin_run(LibraryRead.of(in_sonarr, sonarr_series_hash(s.id for s in series)))
        self._picks_logged = set()
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
                        self._reads.store(sid, eps)
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

        return self._skyhook.prefetch(
            self._orders_by_tvdb(series_ids), dry_run=self._services.is_dry_run(), progress=progress
        )

    def prefetch_scene_numbers(self, series_ids: Iterable[int], *, progress: ProgressCallback | None = None) -> int:
        """Read the due scene mappings of the series in another order (see `XemGateway.prefetch`).

        Returns how many were written. The series are the ones `prefetch_orders` asks SkyHook for, less those in
        aired order: their scene numbers are read off Sonarr's rows, so TheXEM is never asked for them.
        """

        return self._xem.prefetch(
            self._orders_by_tvdb(series_ids), dry_run=self._services.is_dry_run(), progress=progress
        )

    def _orders_by_tvdb(self, series_ids: Iterable[int]) -> list[SonarrSeriesOrders]:
        """The library read's order facts of the series it has a TVDB id for, once per TVDB id."""

        library = self._reads.library.by_id
        in_library = (library[sid].orders for sid in series_ids if sid in library)
        by_tvdb = {orders.tvdb_id: orders for orders in in_library if orders.tvdb_id > 0}
        return list(by_tvdb.values())

    @override
    def episode_count(self, anilist_id: int) -> int | None:
        """AniList's episode count for the title, None when AniList gives none."""

        return self._anilist.episode_count(anilist_id)

    def metadata_unread(self, anilist_ids: Iterable[int], series_ids: Iterable[int] = ()) -> bool:
        """Whether a check that reads these titles and these series' orders is missing one of their answers.

        A title is unread when AniList left it so (`AniListGateway.air_dates_unread`). A series is unread when it has
        no stored orders, as when SkyHook was unreachable before its first fetch. One Sonarr numbers by another
        order is unread too when its scene mappings were never read, as when TheXEM was, and when its stored orders
        are behind for the run (`_orders_behind`). That counts for a series with no AniList mapping too, which a
        pending import may claim. A series this run's library read doesn't have is never unread: nothing could
        fetch it.
        """

        if any(self._anilist.air_dates_unread(anilist_id) for anilist_id in anilist_ids):
            return True
        library = self._reads.library.by_id
        return any(
            (series := library.get(sid)) is not None
            and (tvdb_id := series.orders.tvdb_id) > 0
            and (
                not self._skyhook.has_cached_answer(tvdb_id)
                or (series.orders.in_another_order and self._xem.scene_mappings(tvdb_id) is None)
                or self._orders_behind(series.orders)
            )
            for sid in series_ids
        )

    def title_unread(self, anilist_id: int, series_id: int) -> bool:
        """Whether a check that reads the title's episodes alone, and places nothing, is missing one of their answers.

        That is the title's AniList answer (`AniListGateway.air_dates_unread`), or newer orders of its series
        (`_orders_behind`). What else `metadata_unread` asks of a series is for a check that places files.
        """

        if self._anilist.air_dates_unread(anilist_id):
            return True
        series = self._reads.library.by_id.get(series_id)
        return series is not None and self._orders_behind(series.orders)

    def _orders_behind(self, orders: SonarrSeriesOrders) -> bool:
        """Whether the series is numbered by another order and its stored orders were due a fetch this run missed.

        Its aired view is then built from the old answer (`SkyHookGateway.fetch_missed`). An episode Sonarr has
        since gained is in no title's window, so a title cached on that view would not be checked again for it.
        """

        return orders.in_another_order and self._skyhook.fetch_missed(orders.tvdb_id)

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
        view = self._reads.aired_series(series_id)
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
                return TitleEpisodes(aired=())
            mapped = [ep for ep in series_episodes if check_ep_by_anime_ids(ep=ep, tvdb_season=tvdb_season)]
        else:
            tvdb_mappings = mapping.tvdb_mappings or {}
            if not tvdb_mappings:
                # AniBridge attached no usable per-season ranges, so nothing resolves.
                # Return empty. check_title surfaces the visible NO_EPISODES row.
                return TitleEpisodes(aired=())
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

        return TitleEpisodes(aired=tuple(self._picked_by_date(anilist_id, mapped, series_episodes)))

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

    def sonarr_files(self, episodes: Iterable[SonarrEpisode]) -> ArrFiles:
        """Convert the episodes' existing files into an `ArrFiles`."""

        # keep_no_group=False: episode files with no group go with the episode list
        # into the planner's size matching instead (pinned).
        return ArrFiles.from_files(
            [f for ep in episodes if ep.episode_file_id != 0 and (f := ep.episode_file) is not None],
            keep_no_group=False,
        )
