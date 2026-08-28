"""The Radarr strategy: movie matching and per-AniList-id processing over the services hub."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import override

from .arr_activity import IMPORT_EVENTS, format_history_date
from .cache import CacheRecord, now_stamp, parse_stamp
from .config import Arr
from .grab_pipeline import GrabRequest
from .log import pluralize
from .manual_import import (
    NO_PROGRESS,
    AttemptKind,
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
)
from .mappings import ExternalIds, MappingEntry
from .output import hub_warn
from .protocols import ArrSync
from .radarr_client import AbstractRadarrClient, RadarrClient, collect_anime_movies
from .run_services import RunDeps, RunServices, bind_arr_http
from .seadex_types import ArrReleases, HistoryRecord, ProgressSink, RadarrItem, flagged_urls

# Clock-skew cushion subtracted from the oldest pending record's grab time before the history query.
# The added_at stamps are converted to UTC first, so this absorbs only genuine NTP drift, never a timezone
# gap. A window starting after a real import event would miss its evidence and strand the record until TTL.
_HISTORY_SKEW_HOURS = 2


@dataclass(frozen=True, slots=True)
class _ImportEvidence:
    """One end-of-run check's Radarr import-history evidence, fetched once and memoized."""

    imported_hashes: frozenset[str]
    readable: bool
    """False when the history read failed: the check then never moves a torrent."""


class RadarrSync(ArrSync[RadarrItem]):
    """Radarr sync strategy: owns the Radarr REST client and the movie domain logic."""

    def __init__(
        self,
        deps: RunDeps,
        services: RunServices,
        radarr_client: AbstractRadarrClient | None = None,
    ) -> None:
        """Stand up the Radarr client from the injected shared collaborators."""

        self._services = services
        self._config = deps.config
        self.logger = deps.logger
        self.cache_store = deps.cache_store
        # The check's Radarr import history, memoized. Reset at run start (get_items) so it can't stale.
        self._evidence: _ImportEvidence | None = None
        # Two id sources for collect_anime_movies: the resolver's Anime-IDs candidate sets
        # (from SQL), and the AniBridge view's own.
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge

        # An injected client (tests) is used as-is. Otherwise the connection keys are required
        # only now, when a Radarr run actually runs.
        if radarr_client is not None:
            self.radarr: AbstractRadarrClient = radarr_client
        else:
            # A None deps.arr_http means the keys are missing, so the fallback bind raises here.
            self.radarr = RadarrClient(http=deps.arr_http or bind_arr_http(Arr.RADARR, self._config, deps.http))

    # --- ArrSync hooks ------------------------------------------------------

    @override
    def get_items(self) -> list[RadarrItem]:
        """Every Radarr movie that has an associated AniList ID."""

        self._evidence = None
        return self.get_all_radarr_movies()

    @override
    def filter_to_single(
        self,
        items: list[RadarrItem],
        item_id: int,
    ) -> list[RadarrItem]:
        """Narrow the movie list to a single TMDB ID."""

        filtered = [m for m in items if m.tmdbId == item_id]
        if len(filtered) == 0:
            hub_warn(f"No anime movie with TMDB ID {item_id} found in Radarr - check the --movie-id value")
        return filtered

    @override
    def item_anilist_ids(
        self,
        item: RadarrItem,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        """Resolve AniList ids for a Radarr movie (by TMDB / IMDb id)."""

        return self._services.get_anilist_ids(
            ExternalIds(tmdb=item.tmdbId, imdb=item.imdbId),
            log_ignored=log_ignored,
        )

    @property
    @override
    def warms_episodes(self) -> bool:
        return False

    @override
    def prefetch_episodes(self, items: list[RadarrItem], *, progress: ProgressSink | None = None) -> int:
        """No-op: movies have no episode lists to warm. Returns 0 (warmed none)."""

        del items, progress
        return 0

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        """Radarr history since `date` (delegates to the client)."""

        return self.radarr.history_since(date)

    @override
    def process_al_id(
        self,
        item: RadarrItem,
        al_id: int,
        mapping: MappingEntry,
    ) -> bool:
        """Process one AniList id for a Radarr movie."""

        run = self._services

        sd_entry = run.al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url

        # Movies have no episode coverage, so the backfill is just the URL.
        if run.cached_entry_skip(al_id, sd_entry, lambda: ""):
            return False

        anilist_title = run.get_anilist_title(al_id=al_id)
        run.log_al_title(anilist_title=anilist_title, sd_entry=sd_entry)

        cache_details: CacheRecord = {
            "name": anilist_title,
            "updated_at": sd_entry.updated_at,
            "torrent_hashes": [],
            "url": sd_url,
            "coverage": "",
        }

        radarr_releases = self.get_radarr_releases(
            radarr_movie_id=item.id,
        )

        self.logger.debug(
            f"Radarr release {pluralize(radarr_releases.group_count(), 'group')}: {radarr_releases.groups_label()}"
        )

        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(al_id, cache_details)

        self.logger.debug(f"SeaDex: {', '.join(seadex_dict)}")

        if self._config.advanced.interactive and len(seadex_dict) > 1:
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )
            # Every token was invalid: skip WITHOUT caching, since grab_and_cache would cache the
            # title as done and suppress it forever. It re-prompts next run.
            if len(seadex_dict) == 0:
                return run.invalid_selection_skip()

        plan = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr_releases=radarr_releases,
        )
        torrent_hashes, seadex_dict = plan.torrent_hashes, plan.seadex_dict

        # Seed a pending record per grabbed torrent so the engine's gate persists it: the category move then
        # defers until Radarr imports the movie, and for a torrent shared with a Sonarr grab until both arrs clear.
        pending_seeds: dict[str, PendingImport] | None = None
        if run.import_wait_mode is not ImportWaitMode.OFF:
            added_at = now_stamp()
            # No guard fields: Radarr's import path reads nothing but the infohash.
            pending_seeds = {
                infohash: PendingImport(
                    infohash=infohash,
                    al_id=al_id,
                    title=anilist_title,
                    release_group=srg,
                    url=sd_url,
                    added_at=added_at,
                    series_id=0,
                    file_episode_map={},
                    episode_ids=[],
                    is_dual_audio=False,
                    seadex_files=[],
                    coverage=None,
                    ordered_episode_ids=[],
                )
                for srg, _url_item, infohash in flagged_urls(seadex_dict)
            }

        return run.grab_and_cache(
            GrabRequest(
                al_id=al_id,
                item_title=item.title,
                anilist_title=anilist_title,
                entry=sd_entry,
                seadex_dict=seadex_dict,
                torrent_hashes=torrent_hashes,
                cache_details=cache_details,
                # Every edition's tagged group, not just the first file's.
                replaced_groups=radarr_releases.replaced_groups(),
                pending_seeds=pending_seeds,
            ),
        )

    @override
    def pending_import_series_id(self, item: RadarrItem) -> int | None:
        """None: a Radarr record is not keyed by series."""

        del item
        return None

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt: AttemptKind,
    ) -> ImportProbe:
        """Reconcile one completed Radarr download against Radarr's import history.

        Radarr imports its own completed downloads, so this only reads evidence and never drives an import.
        """

        del content_path, attempt
        evidence = self._import_evidence()
        # An outage is no evidence, so wait. Never move a torrent on a missing read.
        imported = evidence.readable and pending.infohash.casefold() in evidence.imported_hashes
        return ImportProbe(
            ImportReadiness.IMPORTED if imported else ImportReadiness.RETRY,
            files_present=imported,
            command_issued=False,
        )

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Indeterminate zero: a Radarr record reaches no bar."""

        del pending
        return NO_PROGRESS

    @property
    @override
    def supports_blocking_monitor(self) -> bool:
        """No waiting monitor: Radarr records run the one-cycle check off import history."""

        return False

    def _import_evidence(self) -> _ImportEvidence:
        """The check's Radarr import history, fetched once then memoized (reset at run start)."""

        if self._evidence is None:
            self._evidence = self._fetch_import_evidence()
        return self._evidence

    def _fetch_import_evidence(self) -> _ImportEvidence:
        """Query Radarr history since the oldest pending grab and index the import events."""

        records = self.radarr.history_since(format_history_date(self._history_query_start()))
        if records is None:
            return _ImportEvidence(frozenset(), readable=False)
        imported = frozenset(
            record.download_id.casefold()
            for record in records
            if record.download_id and record.event_type.casefold() in IMPORT_EVENTS
        )
        return _ImportEvidence(imported, readable=True)

    def _history_query_start(self) -> datetime:
        """Aware-UTC lower bound for the import-history query: oldest pending grab, minus skew.

        The `added_at` stamps are local-naive, so `.astimezone(UTC)` reads them as local and converts.
        """

        floor = datetime.now(UTC) - timedelta(days=self._config.imports.pending_max_age_days)
        stamps: list[datetime] = []
        for raw in self.cache_store.get_pending(Arr.RADARR).values():
            try:
                stamps.append(parse_stamp(PendingImport.from_json(raw).added_at).astimezone(UTC))
            except (TypeError, ValueError):
                continue
        oldest = min(stamps) if stamps else floor
        return oldest - timedelta(hours=_HISTORY_SKEW_HOURS)

    # --- Radarr domain logic ------------------------------------------------

    def get_all_radarr_movies(self) -> list[RadarrItem]:
        """Get all movies in Radarr that have an associated AniList ID."""

        return collect_anime_movies(
            self.radarr,
            self._mappings,
            self.anibridge,
        )

    def get_radarr_releases(
        self,
        radarr_movie_id: int,
    ) -> ArrReleases:
        """Fold the movie's existing files into an `ArrReleases`."""

        # A movie can carry several files (an upgrade or a multi-edition), so every size is kept.
        return ArrReleases.from_files(
            self.radarr.movie_files(radarr_movie_id),
            keep_untagged=True,
        )
