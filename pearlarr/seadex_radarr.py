"""The Radarr strategy: movie matching and per-AniList-id processing over the services hub."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import override

from .arr_activity import IMPORT_EVENTS, format_history_date
from .cache import UPDATED_AT_STR_FORMAT, CacheRecord
from .config import Arr
from .grab_pipeline import GrabRequest, clean_replaced_groups
from .log import pluralize
from .manual_import import ImportProbe, ImportProgress, ImportReadiness, ImportWaitMode, PendingImport
from .mappings import ExternalIds, MappingEntry
from .output import hub_warn
from .protocols import ArrSync
from .radarr_client import AbstractRadarrClient, collect_anime_movies, make_radarr_client
from .run_services import RunDeps, RunServices
from .seadex_types import ArrReleaseDict, HistoryRecord, ProgressSink, RadarrItem

# Clock-skew cushion subtracted from the oldest pending record's grab time before
# querying Radarr import history. The added_at stamps are converted local-naive ->
# UTC first, so this only has to absorb genuine NTP drift between us and Radarr,
# never a timezone gap - a query window that started after a real import event
# would miss its evidence and strand the record until TTL.
_HISTORY_SKEW_HOURS = 2


@dataclass(frozen=True, slots=True)
class _ImportEvidence:
    """One reconcile pass's Radarr import-history evidence, fetched once and memoized.

    `imported_hashes` are the casefolded `downloadId`s of the import events in the
    query window. `readable` is False when `history_since` failed (a Radarr
    outage), so the reconcile waits and NEVER moves a torrent on missing evidence.
    """

    imported_hashes: frozenset[str]
    readable: bool


class RadarrSync(ArrSync[RadarrItem]):
    """Radarr sync strategy: owns the Radarr REST client + movie domain logic.

    See `ArrSync` for the shared DI/hook-wiring regime.
    """

    def __init__(
        self,
        deps: RunDeps,
        services: RunServices,
        radarr_client: AbstractRadarrClient | None = None,
    ) -> None:
        """Stand up the Radarr client from the injected shared collaborators.

        Args:
            deps: The shared collaborators. The config/mappings
                this strategy needs are read off it.
            services: The services hub the per-id hooks call into.
            radarr_client: A pre-built client to use instead of constructing
                the real `RadarrClient` (which needs the connection keys).
                None builds the real one.
        """

        self._services = services
        self._config = deps.config
        self.logger = deps.logger
        # Read directly for the reconcile's oldest-pending lookup + import-history match.
        self.cache_store = deps.cache_store
        # The reconcile pass's Radarr import history, fetched once then memoized.
        # None = not yet fetched; reset at run start (get_items) so it can't stale.
        self._evidence: _ImportEvidence | None = None
        # The resolver supplies the Anime-IDs candidate id-sets (from SQL) that
        # `collect_anime_movies` filters with. The AniBridge view supplies its own.
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge

        # An injected client (tests) is used as-is. Otherwise the connection keys
        # are required only now, when a Radarr run actually runs.
        if radarr_client is not None:
            self.radarr: AbstractRadarrClient = radarr_client
        else:
            radarr_url, radarr_api_key = self._config.require_connection(Arr.RADARR)
            self.radarr = make_radarr_client(
                url=radarr_url,
                api_key=radarr_api_key,
                http=deps.http,
            )

    # --- ArrSync hooks ------------------------------------------------------

    @override
    def get_items(self) -> list[RadarrItem]:
        """Every Radarr movie that has an associated AniList ID.

        Also the run-start hook: clears the per-run import-history memo so a stale
        window never carries into this run's reconcile.
        """

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
        """Process one AniList id for a Radarr movie.

        A movie is a single file, so the middle is simply: resolve the Radarr
        release group, pull the SeaDex releases, filter them, then hand off to
        the shared grab/cache tail. `mapping` is unused (movies need no episode
        mapping) but is accepted to match the shared hook signature.
        """

        run = self._services

        sd_entry = run.al_id_prologue(al_id)
        if sd_entry is None:
            return False
        sd_url = sd_entry.url

        # Skip if already cached. Movies have no episode coverage, so the
        # one-time backfill on a legacy record is just the URL.
        if run.cached_entry_skip(al_id, sd_entry, lambda: ""):
            return False

        # Resolve the AniList title, then log the active entry (a movie has no
        # episode coverage, so the line carries just the URL)
        anilist_title = run.get_anilist_title(al_id=al_id)
        run.log_al_title(anilist_title=anilist_title, sd_entry=sd_entry)

        # Setup info for cache (URL so cached runs can link to SeaDex - movies have
        # no episode coverage)
        cache_details: CacheRecord = {
            "name": anilist_title,
            "updated_at": sd_entry.updated_at,
            "torrent_hashes": [],
            "url": sd_url,
            "coverage": "",
        }

        radarr_release_dict = self.get_radarr_release_dict(
            radarr_movie_id=item.id,
        )
        radarr_release_groups = list(radarr_release_dict)

        self.logger.debug(
            f"Radarr release {pluralize(len(radarr_release_groups), 'group')}: {', '.join(rg or '(none)' for rg in radarr_release_groups)}"
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(al_id, cache_details)

        self.logger.debug(f"SeaDex: {', '.join(seadex_dict)}")

        # If we're in interactive mode and there are multiple options here, then select
        if self._config.advanced.interactive and len(seadex_dict) > 1:
            seadex_dict = run.filter_seadex_interactive(
                seadex_dict=seadex_dict,
                sd_entry=sd_entry,
            )
            # Every token was invalid: skip WITHOUT caching (grab_and_cache would
            # cache the title as done and suppress it forever) so it re-prompts
            # next run.
            if len(seadex_dict) == 0:
                return run.invalid_selection_skip()

        torrent_hashes, seadex_dict = run.filter_seadex_downloads(
            al_id=al_id,
            seadex_dict=seadex_dict,
            arr_release_dict=radarr_release_dict,
        )

        # Seed a pending-import record per grabbed torrent so the engine's
        # data-driven gate persists it. The post-import category move then defers
        # until Radarr's own completed-download handling imports the movie - and,
        # for a torrent shared with a Sonarr grab, until both arrs' records clear.
        # Only the Sonarr-domain fields stay empty (a Radarr record tracks no
        # episode mapping). Gated on the resolved wait mode so an off run seeds
        # nothing (the pipeline's own gate would skip them regardless).
        pending_seeds: dict[str, PendingImport] | None = None
        if run.import_wait_mode is not ImportWaitMode.OFF:
            added_at = datetime.now().strftime(UPDATED_AT_STR_FORMAT)
            pending_seeds = {
                url_item.infohash: PendingImport(
                    infohash=url_item.infohash,
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
                for srg, srg_item in seadex_dict.items()
                for url_item in srg_item.urls.values()
                if url_item.download and url_item.infohash
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
                # Every edition's group (not just the first file's). The helper drops
                # the {None:[None]} placeholder and any real-file null key.
                replaced_groups=clean_replaced_groups(radarr_release_dict),
                pending_seeds=pending_seeds,
            ),
        )

    @override
    def pending_import_series_id(self, item: RadarrItem) -> int | None:
        """No series id: a Radarr record is not keyed by series, so no inline snapshot.

        Radarr DOES record pending imports now, but they carry `series_id=0` and
        reconcile in `_finalize_run` (never the per-item snapshot). Returning
        `None` short-circuits the engine's per-item snapshot hook for every movie.
        """

        del item
        return None

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Reconcile one completed Radarr download against Radarr's import history.

        Radarr imports its own completed downloads (completed-download handling),
        so this never drives an import - it only reads evidence. The history
        window is fetched ONCE per reconcile pass (memoized, reset at run start)
        and matched by casefolded `downloadId`:

          * a matching import event -> `files_present` (the reconcile drops the
            record and runs the gated category move),
          * no event yet -> leave the record pending (the category stays deferred),
          * a history outage (`history_since` -> None) -> leave pending, never move.

        `content_path` / `force` / `at_deadline` are unused - there is no local
        import to drive or defer, only the yes/no history check.
        """

        del content_path, force, at_deadline
        evidence = self._import_evidence()
        if not evidence.readable:
            # Outage: no evidence, so wait - never move a torrent on a missing read.
            return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)
        imported = pending.infohash.casefold() in evidence.imported_hashes
        return ImportProbe(
            ImportReadiness.IMPORTED if imported else ImportReadiness.LEAVE,
            files_present=imported,
            command_issued=False,
        )

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Indeterminate zero: Radarr never enters the blocking monitor (the only caller).

        Radarr records reconcile off import history, never through the wait
        cockpit's fast-lane bar, so this returns the safe "no bar, promote
        nothing" value to satisfy the `ArrSync` contract.
        """

        del pending
        return ImportProgress(0, 0, determinate=False)

    @property
    @override
    def supports_blocking_monitor(self) -> bool:
        """No blocking monitor: Radarr records reconcile off import history in every mode."""

        return False

    def _import_evidence(self) -> _ImportEvidence:
        """The reconcile pass's Radarr import history, fetched once then memoized.

        Reset to None at run start (`get_items`), so a stale window never carries
        into the next run. Radarr runs exactly one reconcile pass per run, so this
        one memo IS the per-pass fetch the reconcile needs.
        """

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

        The `added_at` stamps are local-naive grab times; `.astimezone(UTC)` reads
        them as local and converts, so the skew cushion only covers real clock
        drift. No pending record outlives its TTL, so that age is the floor when
        no stamp parses.
        """

        floor = datetime.now(UTC) - timedelta(days=self._config.imports.pending_max_age_days)
        stamps: list[datetime] = []
        for raw in self.cache_store.get_pending(Arr.RADARR).values():
            try:
                stamps.append(
                    datetime.strptime(PendingImport.from_json(raw).added_at, UPDATED_AT_STR_FORMAT).astimezone(UTC)
                )
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

    def get_radarr_release_dict(
        self,
        radarr_movie_id: int,
    ) -> ArrReleaseDict:
        """Get a dictionary of useful info for a Radarr movie."""

        # Accumulate sizes per release group (a movie can carry several files - an
        # upgrade or a multi-edition). All of them are already present, so the planner dedups
        # against each. Mirrors Sonarr's per-episode accumulation rather than
        # collapsing to one file or hard-erroring on >1 group.
        radarr_release_dict: ArrReleaseDict = {}
        for mf in self.radarr.movie_files(radarr_movie_id):
            radarr_release_dict.setdefault(mf.release_group, []).append(mf.size)

        # No files: a single unknown-group placeholder keeps the shape uniform
        if not radarr_release_dict:
            radarr_release_dict = {None: [None]}

        return radarr_release_dict
