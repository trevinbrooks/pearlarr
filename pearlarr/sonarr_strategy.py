"""The Sonarr strategy: series/episode coverage and per-AniList-id processing over the services hub."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import override

from . import coverage as _coverage
from .arr_models import HistoryRecord, SonarrItem
from .arr_run_services import ArrRunDeps, ArrRunServices, bind_arr_http
from .arr_strategy import ArrStrategy
from .config import Arr
from .grab_pipeline import NO_PENDING_IMPORT_UPDATES, GrabRequest
from .grab_placement import (
    EntryPlacements,
    PendingImportUpdate,
    SizeOverName,
    TitleFacts,
    TorrentFetches,
    build_pending_import_updates,
    entry_hashes,
    size_over_name_files,
)
from .mappings import ExternalIds, MappingSource, TitleMapping
from .output import hub_warn
from .output.console_text import count_noun, pluralize
from .pending_import import PendingImport
from .placement_types import EpisodeIndex, GrabbingTitle, Placement, episode_index
from .planner import get_episode_keys
from .progress import ProgressCallback
from .releases import PlacementSkip, SeadexDict, ignore_tag_set, releases_to_grab
from .series_listings import SeriesListings
from .sonarr_client import AbstractSonarrClient, SonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import import ImportExecutor, ImportReconciler
from .sonarr_parse import SonarrParseCache
from .wait_types import AttemptKind, AttemptResult, EffectStatus, ImportProgress, ImportWaitMode


def releases_overlap(seadex_dict: SeadexDict) -> bool:
    """See if SeaDex releases have overlapping episodes."""

    # Shares get_episode_keys with get_same_files_groups (planner) but
    # deliberately differs on unparsed releases: here an unparsed release is
    # assumed to overlap (we can't prove it doesn't), whereas get_same_files_groups
    # keeps it separate (so we never drop content we couldn't verify). Keep both
    # consistent if the coverage semantics change.
    episode_sets: dict[str, set[tuple[int | None, int | None]]] = {}
    for rg, group_releases in seadex_dict.items():
        all_episodes = group_releases.all_episodes or []
        episode_sets[rg] = get_episode_keys(all_episodes)

    release_groups: list[str] = list(episode_sets.keys())
    for i, rg1 in enumerate(release_groups):
        for rg2 in release_groups[i + 1 :]:
            # If either release hasn't been parsed, then we can't rule out an
            # overlap, so assume they overlap
            if len(episode_sets[rg1]) == 0 or len(episode_sets[rg2]) == 0:
                return True

            # Otherwise they overlap if they share any episode
            if episode_sets[rg1] & episode_sets[rg2]:
                return True

    return False


def _file_and_why(placement: Placement) -> str:
    """One unplaced file for the debug line: its name and its verdict's words, with the order's key when it has one."""

    key = "" if placement.order_only is None else f": {placement.order_only.label}"
    return f"{placement.name} ({placement.verdict}{key})"


def _override_line(url: str, override: SizeOverName) -> str:
    """The warning for one file placed by size over its name."""

    label = override.by_size.label
    return (
        f"{url}: {override.name} goes on {label} because its size matches SeaDex's {label} release "
        f"(its name reads as {override.parsed_as})"
    )


class SonarrStrategy(ArrStrategy[SonarrItem]):
    """Sonarr sync strategy: owns the Sonarr REST client + episode domain logic.

    See `ArrSync` for the shared DI/hook-wiring regime.
    """

    def __init__(
        self,
        deps: ArrRunDeps,
        services: ArrRunServices,
        *,
        sonarr_client: AbstractSonarrClient | None = None,
    ) -> None:
        """Stand up the Sonarr client from the injected shared collaborators.

        Args:
            deps: The shared collaborators. The config this strategy reads
                directly is unpacked off it, and it's handed to the Sonarr
                collaborators for the cache/AniList gateway/log formatter they
                read.
            services: The services hub the per-id hooks call into.
            sonarr_client: A pre-built client to use instead of constructing
                the real `SonarrClient` (which needs the connection keys).
                None builds the real one.
        """

        self._services = services
        self._config = deps.config
        self.logger = deps.logger

        # Set up Sonarr. An injected client (tests) is used as-is. Otherwise the
        # connection keys are required only now (when a Sonarr run runs) and the
        # real client is built over the run's shared httpx client (parse fires
        # one request per file, so its keep-alive removes a per-file handshake).
        if sonarr_client is not None:
            self.sonarr: AbstractSonarrClient = sonarr_client
        else:
            # A None deps.arr_http means the keys are missing - the fallback
            # bind raises require_connection's error here, the same point as before.
            base = deps.arr_http or bind_arr_http(Arr.SONARR, self._config, deps.http)
            self.sonarr = SonarrClient(
                # Streak re-warns pace with the wait view's digest cadence.
                http=replace(base, rewarn_s=self._config.imports.digest_interval),
                logger=self.logger,
            )

        # Episode-domain collaborator: owns the per-run episode cache + series-id
        # fingerprint and the (series, al_id, mapping) -> episodes resolution. Built
        # from the shared deps + this client. The strategy delegates get_items /
        # prefetch_episodes / prefetch_orders to it, reads its series_hash for the parse
        # cache, and builds every episode index with its orders_for.
        self._episodes = SonarrEpisodes(deps, self.sonarr, self._services)

        # Reads every SeaDex entry of the series once per run, for each entry's episode window and the size
        # identities files are placed by. Window names come from the title ladder an entry's own run uses.
        self._listings = SeriesListings(
            deps.seadex,
            self._episodes,
            ignore_tag_set(self._config.seadex.ignore_tags),
            lambda al_id: self._services.resolve_title(al_id).names,
        )

        # Parse-cache collaborator: grab-time `/parse` of SeaDex filenames + the
        # durable, freshness-checked parse cache (read-through the shared cache_store).
        # The run's series fingerprint is threaded per call.
        self._parse = SonarrParseCache(deps, self.sonarr)

        # Import-execution collaborator: builds + POSTs the authoritative manual
        # import from the record's map, owns the per-run quality/language
        # caches + the once-per-record rescan, and exposes the queue/command reads
        # import_completed consults. Built from the shared deps + this client.
        # Its caches reset in get_items, the run-start hook.
        self._executor = ImportExecutor(deps, self.sonarr)

        # Import-reconcile collaborator: the import_completed decision + the
        # grab-time pending-seed build. Composes the episode collaborator + the
        # executor. The import_completed / check_title hooks delegate to it.
        self._reconciler = ImportReconciler(self._services.pending_imports, self._episodes, self._executor)

    # --- ArrStrategy hooks ------------------------------------------------------

    @override
    def get_items(self) -> list[SonarrItem]:
        """Every Sonarr series with AniList mapping info.

        Also the run-start hook: reset the per-run import scratch here, and let the
        episode collaborator reset its own cache + re-fingerprint the series-id set
        as it enumerates (this is called once, before the per-item loop).
        """

        self._executor.reset()
        self._listings.reset()
        return self._episodes.refresh_library()

    @override
    def filter_to_single(self, items: list[SonarrItem], item_id: int) -> list[SonarrItem]:
        """Narrow the series list to a single TVDB ID."""

        filtered = [s for s in items if s.tvdbId == item_id]
        if len(filtered) == 0:
            hub_warn(f"No anime series with TVDB ID {item_id} found in Sonarr - check the --series-id value")
        return filtered

    @override
    def item_mappings(
        self,
        item: SonarrItem,
        log_ignored: bool = True,
    ) -> dict[int, TitleMapping]:
        """Resolve AniList ids for a Sonarr series (by TVDB / IMDb id)."""

        return self._services.title_mappings(
            ExternalIds.of_series(item),
            log_ignored=log_ignored,
        )

    @property
    @override
    def warms_episodes(self) -> bool:
        return True

    @override
    def prefetch_episodes(self, items: list[SonarrItem], *, progress: ProgressCallback | None = None) -> int:
        """Warm the per-series episode lists before the scan loop.

        Delegates to the episode collaborator's concurrent prefetch. Returns how
        many series it warmed (the needs-scan subset), for the caller's ledger.
        """

        return self._episodes.prefetch(items, progress=progress)

    @override
    def prefetch_orders(self, items: list[SonarrItem], *, progress: ProgressCallback | None = None) -> int:
        """Refetch the due TVDB orders of the series the scan reads and of every stored record's series.

        A record's series is read at grab and import whether or not its own item needs a scan.
        """

        series_ids = self._episodes.titles_needing_check(items)
        if self._services.import_wait_mode is not ImportWaitMode.OFF:
            pending_imports = self._services.pending_imports
            series_ids.extend(
                sid for pending in pending_imports.load(pending_imports.rows()).values() for sid in pending.series_ids
            )
        return self._episodes.prefetch_orders(series_ids, progress=progress)

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Sonarr history since `date` (delegates to the client)."""

        return self.sonarr.history_since(date)

    @override
    def check_title(
        self,
        item: SonarrItem,
        al_id: int,
        mapping: TitleMapping,
    ) -> None:
        """Process one AniList id for a Sonarr series.

        The middle is the episode-aware part: resolve the relevant episode list,
        its coverage and release groups, parse the SeaDex file lists into
        episodes, then hand off to the shared grab/cache tail.
        """

        services = self._services

        sd_entry = services.begin_check(al_id)
        if sd_entry is None:
            return
        sd_url = sd_entry.url
        sonarr_series_id = item.id

        # Skip if already cached. The one-time backfill on a legacy record adds
        # the URL and the season/episode coverage. The coverage needs the episode
        # list, so it's resolved lazily, only when the backfill actually runs.
        if services.is_unchanged(
            al_id,
            sd_entry,
            lambda: _coverage.coverage_string(
                _coverage.episodes_from_ep_list(
                    self._episodes.mapping_episodes(
                        sonarr_series_id=sonarr_series_id,
                        al_id=al_id,
                        mapping=mapping,
                    ),
                ),
            ),
        ):
            return

        # Resolved now, logged once the episode coverage is known.
        title = services.resolve_title(al_id)

        # Setup info for cache
        update = services.new_cached_title_update(title, sd_entry)

        # Get the episode list for all relevant episodes
        ep_list = self._episodes.mapping_episodes(
            sonarr_series_id=sonarr_series_id,
            al_id=al_id,
            mapping=mapping,
        )

        if ep_list is None:
            return

        if not ep_list:
            # Resolved zero episodes (season not in Sonarr, offset past the end, or
            # AniBridge with no ranges): skip, don't mislabel "unmonitored" or grab orphans.
            services.log_no_episodes(title.name)
            if mapping.source is MappingSource.ANIBRIDGE and not mapping.tvdb_mappings:
                # Surface any AniBridge no-usable-ranges case LOUDLY (distinct from a
                # Sonarr-library gap): a WARNING under the skip row naming the cause.
                # Keys off source, so it covers BOTH an empty-{} tvdb entry (mode
                # ANIBRIDGE) and a degraded imdb/tmdb-resolved entry (mode ANIME_IDS),
                # while a legit Kometa whole-series entry (source ANIME_IDS) stays quiet.
                hub_warn(f"AniBridge has no usable season ranges for {title.name} - skipping")
            services.owe_pause()
            return

        # If all episodes are unmonitored, then skip if ignore_unmonitored is switched on
        if self._config.sonarr.ignore_unmonitored and not any(ep.monitored for ep in ep_list):
            services.log_anilist_item_unmonitored(
                item_title=title.name,
            )
            services.owe_pause()
            return

        # Now that we have the episodes, log the active entry with its
        # season/episode coverage + URL, and remember them for the cache so
        # future cached runs can show the same detail
        coverage = _coverage.coverage_string(
            _coverage.episodes_from_ep_list(ep_list),
        )
        services.open_title_block(
            title=title.name,
            sd_entry=sd_entry,
            coverage=coverage,
        )
        update["coverage"] = coverage
        update["url"] = sd_url

        sonarr_releases = self._episodes.get_sonarr_releases(ep_list=ep_list)

        self.logger.debug(
            f"Sonarr release {pluralize(sonarr_releases.group_count(), 'group')}: {sonarr_releases.groups_label()}"
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = services.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            # Nothing is placed here, so only the entry whose window gave the coverage counts.
            services.report_no_releases(al_id, update, metadata_unread=self._episodes.any_air_dates_unread([al_id]))
            return

        self.logger.debug(f"SeaDex: {', '.join(seadex_dict)}")

        # Place every listed file now: the import imports this map and places nothing, so the grab is judged by
        # it. Placed under every listing entry's window and every stored claim's, a stored torrent as its record's
        # leftover. The series maps are the whole-series lists (a per-run cache hit each).
        waits_on_imports = services.import_wait_mode is not ImportWaitMode.OFF
        hashes = entry_hashes(seadex_dict)
        stored: dict[str, PendingImport] = (
            services.pending_imports.stored_pending_imports(hashes) if waits_on_imports else {}
        )
        series_ids = {sonarr_series_id, *(sid for pending in stored.values() for sid in pending.series_ids)}
        indexes = self._series_indexes(series_ids)
        resolved = self._services.mappings_with_ignored(ExternalIds.of_series(item))
        listings = self._listings.read(sonarr_series_id, resolved)
        grabbing_title = GrabbingTitle(
            al_id=al_id,
            series_id=sonarr_series_id,
            title_index=episode_index(ep_list, self._episodes.orders_for(sonarr_series_id)),
            series_index=indexes.get(sonarr_series_id, episode_index([])),
            title_names=title.names,
            episode_count=self._episodes.episode_count(al_id),
            alternate_titles=tuple(alt.title for alt in item.alternateTitles),
            aired_order=self._episodes.aired_order(sonarr_series_id),
        )
        placed = EntryPlacements.place(
            grabbing_title,
            self._parse.parsed_files(seadex_dict, series_hash=self._episodes.series_hash),
            TorrentFetches(stored, indexes, listings).known(seadex_dict),
        )
        placed.attach_placements(seadex_dict)
        self._log_placements(placed)

        # If we're in interactive mode and there are multiple equivalent options here, then select
        if self._config.advanced.interactive and len(seadex_dict) > 1 and releases_overlap(seadex_dict=seadex_dict):
            seadex_dict = services.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )
            # Every token was invalid: skip WITHOUT caching (grab_and_cache would
            # cache the title as done and suppress it forever) so it re-prompts
            # next run.
            if len(seadex_dict) == 0:
                services.report_invalid_selection()
                return

        # Filter downloads by whether the episodes in each torrent match the release
        # group we have in Sonarr
        plan = services.filter_seadex_downloads(
            seadex_dict=seadex_dict,
            arr_releases=sonarr_releases,
            ep_list=ep_list,
        )
        seadex_dict = plan.seadex_dict
        self._warn_overrides(placed, seadex_dict)

        # Build the per-torrent seeds the engine persists at the add site: one per release marked for download
        # (download + hash), carrying our own (basename -> Sonarr episode ids) map so the later manual import never
        # trusts Sonarr's blind parse. Gated on the engine's RESOLVED mode (cli > config), not the raw config, so
        # a CLI override agrees with the engine's persist/reconcile/blocking gates.
        pending_import_updates: Mapping[str, PendingImportUpdate] = NO_PENDING_IMPORT_UPDATES
        if waits_on_imports:
            pending_import_updates = build_pending_import_updates(
                seadex_dict,
                placed,
                TitleFacts(
                    al_id=al_id,
                    series_id=sonarr_series_id,
                    title=title.name,
                    coverage=coverage,
                    url=sd_url,
                    guards=plan.guards,
                ),
            )

        services.grab_and_cache(
            GrabRequest(
                al_id=al_id,
                arr_title=item.title,
                title_name=title.name,
                entry=sd_entry,
                seadex_dict=seadex_dict,
                update=update,
                replaced_groups=sonarr_releases.groups(),
                coverage=coverage,
                pending_import_updates=pending_import_updates,
                input_unread_groups=placed.input_unread_groups(seadex_dict),
                # Every entry of the series: the placement reads the window of each one SeaDex has a record for.
                metadata_unread=(
                    self._episodes.any_air_dates_unread(resolved.kept_and_ignored)
                    or self._episodes.orders_unread(series_ids)
                ),
            ),
        )

    def _series_indexes(self, series_ids: Iterable[int]) -> dict[int, EpisodeIndex]:
        """One index per series whose whole list served this run (a cold read fetches once); an unread one is absent."""

        return {
            sid: episode_index(episodes, self._episodes.orders_for(sid))
            for sid in series_ids
            if (episodes := self._episodes.cached_episodes(sid)) is not None
        }

    def _warn_overrides(self, placed: EntryPlacements, seadex_dict: SeadexDict) -> None:
        """Warn once per url this run grabs if any of its files were placed by size over their names.

        The warning shows the first such file, and the rest go to the debug log. `seadex_dict` must be the
        download plan's, so urls the run doesn't grab never warn.
        """

        for planned in releases_to_grab(seadex_dict):
            if overrides := size_over_name_files(placed.by_url[planned.url], placed.grabbing_title.series_index):
                first, *rest = overrides
                more = f" ({count_noun(len(rest), 'more file')} in the debug log)" if rest else ""
                hub_warn(f"{_override_line(planned.url, first)}{more}")
                for override in rest:
                    self.logger.debug(_override_line(planned.url, override))

    def _log_placements(self, placed: EntryPlacements) -> None:
        """One debug line per url with video files: its coverage, then each file set aside or held for a hand import."""

        for url, placement in placed.by_url.items():
            if not placement.files:
                continue
            # The hold and the failed read explain an empty coverage on a listed release. A url held on the read
            # names it once, in its hold.
            assignment = placement.placements
            aside = ", ".join(_file_and_why(p) for p in assignment.excluded)
            by_hand = ", ".join(_file_and_why(p) for p in assignment.placements if p.verdict.left_for_hand_import)
            not_grabbed_note = "" if placement.skip_reason is None else f" (not grabbed: {placement.skip_reason})"
            note_read_failed = not placement.inputs_read and placement.skip_reason is not PlacementSkip.INPUT_UNREAD
            self.logger.debug(
                f"{url}: placed {_coverage.coverage_string(list(placement.episodes)) or 'nothing'}"
                f"{' (a read failed)' if note_read_failed else ''}{not_grabbed_note}"
                f"{f'; set aside: {aside}' if aside else ''}"
                f"{f'; left for a hand import: {by_hand}' if by_hand else ''}"
            )

    @override
    def pending_import_series_id(self, item: SonarrItem) -> int | None:
        """The Sonarr series id whose carried-over pending records this item owns.

        The engine's per-item snapshot hook keys off this. A Sonarr series owns
        its pending records by `series_id`, which is the Sonarr series id.
        """

        return item.id

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt_kind: AttemptKind,
    ) -> AttemptResult:
        """One reconcile/import poll for a completed download (delegated).

        The @abstractmethod hook stays here so the ABC instantiates. The reconcile
        decision lives on `ImportReconciler`.
        """

        return self._reconciler.import_completed(pending, content_path, attempt_kind)

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap read-only files-landed count for the wait bar (delegated)."""

        return self._reconciler.import_progress(pending)

    @override
    def remove_from_queue(self, pending: PendingImport) -> EffectStatus:
        """Dismiss the torrent's leftover Sonarr queue entry (see `ImportExecutor.close_tracked`)."""

        return self._executor.remove_from_queue(pending)

    @property
    @override
    def supports_blocking_wait(self) -> bool:
        """Sonarr owns the interleaved end-of-run wait/import monitor cockpit."""

        return True
