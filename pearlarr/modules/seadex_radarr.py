"""The Radarr strategy: movie matching and per-AniList-id processing over the services hub."""

from typing import override

from .cache import CacheRecord
from .config import Arr
from .grab_pipeline import GrabRequest
from .log import indent_string
from .manual_import import ImportProbe, ImportProgress, ImportReadiness, PendingImport
from .mappings import ExternalIds, MappingEntry
from .output import hub_warn
from .protocols import ArrSync
from .radarr_client import AbstractRadarrClient, collect_anime_movies, make_radarr_client
from .run_services import RunDeps, RunServices
from .seadex_types import ArrReleaseDict, HistoryRecord, ProgressSink, RadarrItem


class RadarrSync(ArrSync[RadarrItem]):
    """Radarr sync strategy: owns the Radarr REST client + movie domain logic.

    Implements the `ArrSync` hooks the run machinery drives.
    The composition root injects the shared `RunDeps` (used
    to stand up the client) and the `RunServices` hub
    (held as `self._services`); the per-id hooks call the shared pipeline
    through it.
    """

    def __init__(
        self,
        deps: RunDeps,
        services: RunServices,
        radarr_client: AbstractRadarrClient | None = None,
    ) -> None:
        """Stand up the Radarr client from the injected shared collaborators.

        Args:
            deps: The shared collaborators; the config/mappings
                this strategy needs are read off it.
            services: The services hub the per-id hooks call into.
            radarr_client: Injectable replacement for the real `RadarrClient`,
                which needs the connection keys; None builds the real one.
                Tests inject a scripted fake through this typed seam, so the
                real `__init__` runs without keys.
        """

        self._services = services
        self._config = deps.config
        self.logger = deps.logger
        # The resolver supplies the Anime-IDs candidate id-sets (from SQL) that
        # `collect_anime_movies` filters with; the AniBridge view supplies its own.
        self._mappings = deps.mappings
        self.anibridge = deps.mappings.anibridge

        # An injected client (tests) is used as-is; otherwise the connection keys
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
        """Every Radarr movie that has an associated AniList ID."""

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
            hub_warn(f"No anime movie with TMDB ID {item_id} found in Radarr")
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

        # Setup info for cache (URL so cached runs can link to SeaDex; movies have
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
            indent_string(
                f"Radarr release group(s): {', '.join(rg or '(none)' for rg in radarr_release_groups)}",
            ),
        )

        # Produce a dictionary of info from the SeaDex request
        seadex_dict = run.get_seadex_dict(sd_entry=sd_entry)

        if len(seadex_dict) == 0:
            return run.no_releases_skip(al_id, cache_details)

        self.logger.debug(
            indent_string(
                f"SeaDex: {', '.join(seadex_dict)}",
            ),
        )

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

        return run.grab_and_cache(
            GrabRequest(
                al_id=al_id,
                item_title=item.title,
                anilist_title=anilist_title,
                entry=sd_entry,
                seadex_dict=seadex_dict,
                torrent_hashes=torrent_hashes,
                cache_details=cache_details,
                # The full ordered group list, mirroring Sonarr - the notifier
                # renders every edition's group, not just the first file's.
                release_group=radarr_release_groups,
            ),
        )

    @override
    def pending_import_series_id(self, item: RadarrItem) -> int | None:
        """No-op: Radarr movies record no pending imports (out of scope).

        Returns `None` so the engine's per-item snapshot hook short-circuits for
        every Radarr movie - there are no carried-over import records to report.
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
        """No-op: the series-pinned manual import is Sonarr-only (out of scope).

        Radarr never records a pending import (`grab_and_cache` passes no
        `pending_seeds`), so this is never reached in practice; it returns a
        `LEAVE` probe (the safest terminal value - never drops a record, no files
        verified) to satisfy the `ArrSync` contract.
        """

        del pending, content_path, force, at_deadline
        return ImportProbe(ImportReadiness.LEAVE, files_present=False, command_issued=False)

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """No-op: Radarr records no pending imports, so this is never reached.

        Returns an indeterminate zero (the safest value - shows no bar, promotes
        nothing) to satisfy the `ArrSync` contract.
        """

        del pending
        return ImportProgress(0, 0, determinate=False)

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
        # upgrade or a multi-edition); the user has all of them, so the planner dedups
        # against each. Mirrors Sonarr's per-episode accumulation rather than collapsing
        # to one file (which dropped the other sizes) or hard-erroring on >1 group
        # (which skipped the movie every run).
        radarr_release_dict: ArrReleaseDict = {}
        for mf in self.radarr.movie_files(radarr_movie_id):
            radarr_release_dict.setdefault(mf.release_group, []).append(mf.size)

        # No files: a single unknown-group placeholder keeps the shape uniform
        if not radarr_release_dict:
            radarr_release_dict = {None: [None]}

        return radarr_release_dict
