"""Import-time subsystem: decide a download's state, then build/POST the import.

Two collaborators:

* `ImportExecutor` owns the mechanical "consume" side: throttled download
  rescan, queue/command reads, and - the heart of it - turning OUR resolved
  `basename -> episode ids` map (built by
  `FileEpisodeMapper`) into a Sonarr ManualImport payload
  and POSTing it, with per-run quality/language caches.
* `ImportReconciler` owns the *decision*: one `import_completed` poll (what
  state the download is in, when to step in vs. defer to Sonarr) and the
  grab-time `PendingImport` seed build. It composes the episode collaborator +
  the executor. The strategy's `import_completed` / `process_al_id` hooks are
  thin delegators onto it.
"""

import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from .cache import UPDATED_AT_STR_FORMAT
from .config import Arr
from .log import count_noun, pluralize
from .manual_import import (
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    PendingImport,
    normalize_basename,
    normalize_group,
)
from .output import hub_note, hub_warn
from .run_services import RunDeps
from .seadex_types import (
    CommandResource,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    QualityDefinition,
    QualitySource,
    RemotePathMapping,
    SeadexDict,
    SonarrEpisode,
)
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import_plan import (
    ContentPaths,
    DownloadHistoryVerdict,
    EpisodeFileStatus,
    EpisodeSnapshot,
    ImportAction,
    ImportDecision,
    ParsedQuality,
    QueueVerdict,
    all_targets_done,
    build_episode_id_map,
    classify_download_history,
    classify_queue,
    derive_languages,
    episode_file_statuses,
    episode_ids_for_parsed,
    manual_import_in_flight,
    parse_quality_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
    targets_needing_import,
    translate_download_path,
)
from .sonarr_mapper import FileEpisodeMapper
from .sonarr_parse import parsed_episodes, video_file_entries

# RefreshMonitoredDownloads is quick (Sonarr re-scans its clients). Poll its
# command status up to this many times, sleeping this long between, before
# proceeding regardless. Waiting means the queue we read next reflects the
# rescan. The bound means a stuck command never blocks the run.
_REFRESH_COMMAND_MAX_POLLS = 30
_REFRESH_COMMAND_POLL_S = 1
_COMMAND_TERMINAL_STATES = frozenset({"completed", "failed", "aborted", "cancelled"})


def _hostname(url: str | None) -> str | None:
    """The bare hostname of a configured URL (tolerates a scheme-less `host:port`)."""

    if not url:
        return None
    return urlsplit(url).hostname or urlsplit(f"//{url}").hostname


class _CandidateScan(NamedTuple):
    """One poll's candidate fetch plus the entry shape it implies.

    The shape is uniform across every file of the resulting command - which is
    also what keeps the in-flight guard's no-downloadId arm applicable (its
    `file_hashes` gate requires ALL entries of a command to omit the id).
    """

    candidates: list[ManualImportCandidate]
    omit_download_id: bool
    """True only for a dead-tracked folder scan: the entries must not carry a
    downloadId (Sonarr's Execute tail NREs on the poisoned tracked download)."""


class _EntryContext(NamedTuple):
    """Per-command context for building file entries (uniform across the command)."""

    pending: PendingImport
    content_path: str
    omit_download_id: bool


class ImportExecutor:
    """Builds/POSTs the manual-import payload + owns the per-run import caches.

    Constructed once per run in `SonarrSync` from the shared
    deps, the strategy's Sonarr client, and the strategy's
    `FileEpisodeMapper`. `import_completed` decides the
    download's state and, when it's time to step in, calls
    `run_manual_import`. The executor also exposes the throttled rescan and
    the queue/command reads that decision consults.
    """

    def __init__(self, deps: RunDeps, sonarr: AbstractSonarrClient, mapper: FileEpisodeMapper) -> None:
        """Bind the Sonarr client, config/logger, and the file->episode mapper.

        Args:
            deps: The shared collaborators. The config + logger are read
                off it.
            sonarr: The strategy's Sonarr client.
            mapper: The strategy's import-time mapper (its
                `candidate_files` / `assign` build the authoritative map).
        """

        self.sonarr = sonarr
        self._config = deps.config
        self.logger = deps.logger
        self._mapper = mapper

        # Per-run caches of the Sonarr quality-definition / language lists, used
        # to resolve a quality name / language names into the manual-import
        # payload objects. Fetched lazily on the first import and then reused for
        # the rest of the run so repeated imports don't re-hit the endpoints.
        # None means "not yet fetched" (cleared in reset, the run-start hook).
        self._quality_defs_cache: list[QualityDefinition] | None = None
        self._languages_cache: list[Language] | None = None

        # Infohashes for which we've already warned that some on-disk files could
        # not be placed in the resolved set, so the loud "left these for you" line
        # is logged once a run rather than every poll until the record clears.
        self._warned_unplaceable: set[str] = set()

        # Whether the unmatched-default_quality warning fired this run. The seam
        # runs once per FILE, so without this the typo would warn on every import.
        self._warned_default_quality = False

        # Monotonic time of the last RefreshMonitoredDownloads we asked Sonarr for,
        # used to throttle the rescan: the blocking pass calls import_completed
        # every poll and may walk several torrents back-to-back, so we re-issue the
        # (global) refresh at most once per imports.poll_interval rather than on
        # every call. None means "not refreshed yet this run" (reset in reset).
        self._last_refresh_monotonic: float | None = None

        # Folder-scan fallback scratch (all per-run, cleared in reset):
        # infohashes pinned to folder mode after a NONEMPTY folder scan, the
        # memoized history verdicts (VERDICTS only - a probe failure is never
        # stored, so the next fallback activation re-probes), the lazy
        # remote-path-mapping list (None = not fetched yet), and the computed
        # content_path translations the in-flight guard reads back.
        self._folder_pinned: set[str] = set()
        self._history_verdicts: dict[str, DownloadHistoryVerdict] = {}
        self._path_mappings: list[RemotePathMapping] | None = None
        self._translated_paths: dict[str, str] = {}

        # Dead-tracked downloads whose folder scan came back EMPTY (the one
        # genuinely stuck fallback shape), warned once a run - not every poll.
        self._warned_empty_folder: set[str] = set()

    def reset(self) -> None:
        """Drop the per-run import scratch (run-start, via get_items)."""

        self._quality_defs_cache = None
        self._languages_cache = None
        self._warned_unplaceable = set()
        self._warned_default_quality = False
        self._last_refresh_monotonic = None
        self._folder_pinned = set()
        self._history_verdicts = {}
        self._path_mappings = None
        self._translated_paths = {}
        self._warned_empty_folder = set()

    def refresh_downloads(self) -> None:
        """Queue RefreshMonitoredDownloads (throttled) and wait for it, best-effort.

        RefreshMonitoredDownloads is global and the blocking pass polls often (and
        may walk several torrents back-to-back), so it's re-issued at most once per
        `imports.poll_interval`. Waiting for the command to finish means the queue
        read that follows reflects the rescan. The poll bound means a stuck command
        can never block the run, and a failure to queue/confirm just leaves the
        next queue read slightly stale (a later poll corrects it).
        """

        now = time.monotonic()
        interval = self._config.imports.poll_interval
        if self._last_refresh_monotonic is not None and now - self._last_refresh_monotonic < interval:
            return
        self._last_refresh_monotonic = now

        cmd_id = self.sonarr.refresh_monitored_downloads()
        if cmd_id is None:
            return
        self.logger.debug("Asked Sonarr to rescan its downloads")

        for _ in range(_REFRESH_COMMAND_MAX_POLLS):
            command = self.sonarr.command_status(cmd_id)
            state = command.status or ""
            if state.casefold() in _COMMAND_TERMINAL_STATES:
                return
            time.sleep(_REFRESH_COMMAND_POLL_S)

    def queue_states(self, infohash: str) -> list[str]:
        """This download's queue-record states, for `classify_queue`.

        Matches records to the torrent by `downloadId` (case-insensitively,
        Sonarr stores the infohash uppercased) and keeps each record's
        `trackedDownloadState` - the one signal the verdict depends on. Records
        with no tracked state are dropped. An empty result means Sonarr isn't
        tracking the download.
        """

        target = infohash.casefold()
        states: list[str] = []
        for record in self.sonarr.queue():
            dl_id = record.download_id
            if dl_id is None or dl_id.casefold() != target:
                continue
            if record.state:
                states.append(record.state)
        return states

    def list_commands(self) -> list[CommandResource]:
        """The current Sonarr command list, for the in-flight ManualImport guard.

        A thin pass-through to `SonarrClient.list_commands` (mirrors
        `queue_states`' delegation to `self.sonarr`). Fetched fresh
        every poll - never cached - since an in-flight command's status changes as
        Sonarr finishes the import.
        """

        return self.sonarr.list_commands()

    def content_paths(self, content_path: str) -> ContentPaths:
        """The raw + Sonarr-visible path pair the in-flight guard matches against.

        Read-only over the memoized translations (never fetches): before the
        first fallback activation both views are the raw path, which is the
        pre-fallback status quo the episode-id guard arm already covers.
        """

        return ContentPaths(
            raw=content_path,
            sonarr_visible=self._translated_paths.get(content_path, content_path),
        )

    def _remote_path_mappings(self) -> list[RemotePathMapping]:
        """Sonarr's remote path mappings, fetched once per run on first fallback use.

        A failed fetch caches [] - the client's warning fires once, translation
        degrades to a no-op for the rest of the run (bounded: an untranslatable
        folder scans empty, so nothing pins and the downloadId scan is retried
        next poll - the next run re-fetches).
        """

        if self._path_mappings is None:
            self._path_mappings = self.sonarr.remote_path_mappings() or []
        return self._path_mappings

    def _sonarr_visible_path(self, content_path: str) -> str:
        """Translate (and memoize) a download path into Sonarr's filesystem view."""

        translated = self._translated_paths.get(content_path)
        if translated is None:
            translated = translate_download_path(
                content_path,
                self._remote_path_mappings(),
                _hostname(self._config.qbittorrent.host),
            )
            self._translated_paths[content_path] = translated
        return translated

    def _probe_history(self, pending: PendingImport) -> DownloadHistoryVerdict | None:
        """Classify a download's Sonarr history (memoized), or None on probe failure.

        Verdicts are memoized for the run. A FAILED probe is never stored, so
        the next fallback activation re-probes instead of locking the record
        into one branch. The first dead-tracked classification notes the state
        on the hub, once per record per run (the verdict memo gates it).
        """

        verdict = self._history_verdicts.get(pending.infohash)
        if verdict is not None:
            return verdict
        page = self.sonarr.history_for_download(download_id=pending.infohash)
        if page is None:
            return None
        verdict = classify_download_history(page.records)
        self._history_verdicts[pending.infohash] = verdict
        if verdict.dead_tracked:
            when = f" on {verdict.date.split('T')[0]}" if verdict.date else ""
            hub_note(
                f"{pending.display_label}: Sonarr recorded this download as {verdict.event}{when} "
                "and won't serve it by id - importing from its folder instead"
            )
        return verdict

    def _scan_candidates(self, pending: PendingImport, content_path: str) -> _CandidateScan | None:
        """The candidates for one poll: the downloadId scan, with the folder fallback.

        DESIGN PRINCIPLE: the folder FALLBACK is what kills the dead loop (a
        download whose prior imported/failed/ignored history makes the
        downloadId scan 500 forever). The history probe only decides how much
        noise the import allocates. Every probe misclassification degrades to
        bounded one-command noise - never a loop - which is why probe failure
        defaults to the status-quo entry shape (keep the downloadId).

        The fallback trigger is deliberately loose (ANY downloadId-scan
        failure): the folder scan is equal-fidelity, so a false activation
        costs one extra GET. A downloadId scan answering `[]` is NOT a trigger
        (Sonarr answered "no files visible yet" - the existing retry semantics
        apply). Returns None only when the active scan(s) failed outright.

        NOISE OWNERSHIP: the downloadId scan fails silently (its client warn
        would brand every dead-tracked poll with a misleading "will retry"
        right before the fallback handles it), and this method surfaces the one
        line each outcome deserves: the dead-tracked note on the first probe, a
        once-per-run warning when a dead-tracked download's folder scan finds
        NOTHING (the genuinely stuck shape - untranslated path, deleted files),
        and the folder-scan client warn when the fallback fetch itself fails.
        """

        if pending.infohash not in self._folder_pinned:
            candidates = self.sonarr.manual_import_candidates(pending=pending)
            if candidates is not None:
                return _CandidateScan(candidates, omit_download_id=False)
            self.logger.debug(
                f"{content_path}: downloadId scan failed for {pending.display_label} - trying the folder fallback"
            )

        verdict = self._probe_history(pending)
        folder = self._sonarr_visible_path(content_path)
        folder_candidates = self.sonarr.manual_import_candidates_by_folder(
            folder=folder,
            title=pending.display_label,
        )
        if folder_candidates is None:
            return None
        if folder_candidates:
            # INVARIANT: pin folder mode on NONEMPTY success only - 200 `[]` is
            # exactly what an invisible/untranslated folder returns, and pinning
            # on it would wedge the record in a mode that can never see files
            # while the recoverable downloadId scan goes unretried.
            self._folder_pinned.add(pending.infohash)
        elif verdict is not None and verdict.dead_tracked and pending.infohash not in self._warned_empty_folder:
            # Dead-tracked + empty folder = silent retries until the record
            # expires. Say so once. A clean/unknown verdict self-heals by id.
            self._warned_empty_folder.add(pending.infohash)
            hub_warn(
                f"{pending.display_label}: Sonarr won't serve this download by id and "
                f"a scan of its folder found no files ({folder}) - will retry"
            )
        return _CandidateScan(
            folder_candidates,
            omit_download_id=verdict is not None and verdict.dead_tracked,
        )

    def run_manual_import(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        snapshot: EpisodeSnapshot,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Drive our authoritative series-pinned manual import for one download.

        Scans for candidates by downloadId - falling back to a folder scan of
        the (remote-path-translated) `content_path` when Sonarr won't serve the
        download by id (see `_scan_candidates`) - then repairs our
        file->episode map from the actual on-disk files (re-parsing
        whatever the seed didn't cover, mapped through OUR `(season, episode) ->
        id` index - never Sonarr's candidate episode assignment), then imports
        EXACTLY the files our map intends: each file's episodes that don't already
        hold a recommended release (so a recommended file is never overwritten) and
        no file outside our map (so an episode our mapping gave to another preferred
        torrent is never imported here). An intended file Sonarr can't see yet is
        retried, never silently skipped.

        Returns an `ImportProbe`. A manual-import command's copy is async, so
        accepting the command is NOT `files_present` - the probe reads
        `RETRY` + `command_issued` until a later poll verifies the episode files
        actually landed. `files_present` is set only when every intended episode
        already holds a recommended file (nothing left to copy).

        Args:
            pending: The durable record for the completed torrent.
            content_path: The qBittorrent `content_path` to import from,
                also the label for this download's log lines.
            snapshot: The same-poll episode index + normalized
                recommended-group guard set.
            at_deadline: The final attempt - a still-missing intended file
                is terminal, so warn loudly. Otherwise it's an expected early-poll
                gap and only logged at debug.
        """

        scan = self._scan_candidates(pending, content_path)
        if scan is None:
            # Transient (timeout / non-200). The folder-scan client warned. Ask again.
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        candidates_by_basename = self._mapper.candidate_files(scan.candidates)
        ep_id_map = build_episode_id_map(list(snapshot.episodes_by_id.values()))
        authoritative_map, unplaceable = self._mapper.assign(
            pending,
            candidates_by_basename,
            ep_id_map,
        )
        if unplaceable:
            self._warn_unplaceable_files(pending, unplaceable)

        if not authoritative_map:
            self.logger.debug(f"{content_path}: no mappable files for {pending.display_label} yet")
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Done-check against the COMPLETE (repaired) intended set, from the files.
        target_ids = sorted({i for ids in authoritative_map.values() for i in ids})
        statuses = episode_file_statuses(target_ids, snapshot)
        if all_targets_done(statuses):
            self.logger.debug(f"{content_path}: already imported (recommended files present)")
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        needing = targets_needing_import(statuses)
        decisions = plan_import_files(authoritative_map, candidates_by_basename, needing)

        entry_context = _EntryContext(pending, content_path, scan.omit_download_id)
        files: list[ManualImportFile] = []
        missing: list[str] = []
        for decision in decisions:
            match decision.action:
                case ImportAction.MISSING:
                    missing.append(decision.basename)
                case ImportAction.IMPORT:
                    files.append(self._build_file_entry(decision, entry_context))
                case _:
                    # SAMPLE / ALREADY / SKIP_DONE -> nothing to import for this
                    # file. Surface the distinct vocabulary at debug.
                    self.logger.debug(f"{decision.action.name}: {decision.basename}")

        if missing:
            # Intended files our map covers but Sonarr can't see yet. An early poll
            # finding them absent is expected (the copy hasn't landed), so it's only
            # noisy at the deadline, where a still-missing file is terminal: warn
            # loudly only then, debug otherwise. Either way the record is retried,
            # never dropped silently.
            message = (
                f"{content_path}: {count_noun(len(missing), 'intended file')} "
                f"not visible to Sonarr for {pending.display_label} - will retry"
            )
            if at_deadline:
                hub_warn(message)
            else:
                self.logger.debug(message)

        if not files:
            # Nothing to queue this poll: retry if files are merely missing, else
            # everything intended is already satisfied (already/sample/skip_done).
            if missing:
                return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        import_mode = self._config.imports.mode
        if scan.omit_download_id and import_mode == "auto":
            # The untracked Execute branch with Auto resolves to MOVE (no
            # DownloadClientItem to report CanMoveFiles), which would rip the
            # files out from under the seeding torrent. An explicitly configured
            # move/copy is honored as set.
            import_mode = "copy"
        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=import_mode,
        )
        if cmd_id is None:
            self.logger.debug(f"{content_path}: Sonarr rejected the import command; will retry")
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # The command was accepted, but its copy is async - the episode files may
        # not have landed yet (a remote-mount copy isn't instant). Do NOT declare
        # the files imported on command acceptance: report RETRY + command_issued,
        # so the next monitor cycle flips to files_present once they appear.
        self.logger.debug(f"{content_path}: queued {count_noun(len(files), 'file')} for import (command {cmd_id})")
        return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=True)

    def _warn_unplaceable_files(
        self,
        pending: PendingImport,
        unplaceable: list[str],
    ) -> None:
        """Warn (once a run per download) about on-disk files we couldn't place.

        These are files Sonarr sees but our resolved mapping can't confidently
        assign (ambiguous numbering, an extra that slipped the skip list, a pack
        that doesn't line up 1:1). We import what we can and leave these - surfacing
        them loudly so they're never silently dropped.
        """

        if pending.infohash in self._warned_unplaceable:
            return
        self._warned_unplaceable.add(pending.infohash)
        label = pending.display_label
        coverage = f" ({pending.coverage})" if pending.coverage else ""
        hub_warn(
            f"{label}{coverage}: {count_noun(len(unplaceable), 'file')} could not be matched "
            f"to an episode and {pluralize(len(unplaceable), 'was', 'were')} not imported"
        )

    def _import_language_objects(self, pending: PendingImport) -> list[Language]:
        """Resolve the import language objects for a record (lazily cached)."""

        if self._languages_cache is None:
            self._languages_cache = self.sonarr.languages()
        lang_names = derive_languages(
            pending.is_dual_audio,
            self._config.imports.languages_dual,
            self._config.imports.languages_single,
        )
        return resolve_language_objects(lang_names, self._languages_cache)

    def _quality_definitions(self) -> list[QualityDefinition]:
        """The Sonarr quality definitions (lazily fetched + cached for the run)."""

        if self._quality_defs_cache is None:
            self._quality_defs_cache = self.sonarr.quality_definitions()
        return self._quality_defs_cache

    def _build_file_entry(
        self,
        decision: ImportDecision,
        context: _EntryContext,
    ) -> ManualImportFile:
        """Build one ManualImport file payload from a planned `import` decision.

        The episode ids come straight from our authoritative map (never Sonarr's
        parse). The quality is decided per axis with precedence Sonarr's parse ->
        our filename parse -> the configured default, and always emits a real
        quality (never an omitted key), warning only when it resolves to Unknown.
        The language objects + quality definitions are read from this run's caches.

        Only `import` decisions reach here, so `decision.path` is the on-disk
        candidate path (always set). The `or decision.basename` keeps the
        payload `path` a non-null `str` for the type.
        """

        pending = context.pending
        content_path = context.content_path
        lang_objs = self._import_language_objects(pending)
        quality_defs = self._quality_definitions()

        path = decision.path or decision.basename
        base = os.path.basename(path)
        sonarr_axes = quality_axes_from_model(decision.quality)
        our_axes = parse_quality_from_filename(base)
        default_name = self._config.imports.default_quality
        default_axes = quality_axes_from_name(default_name, quality_defs)
        # A configured default that matches no definition contributes nothing.
        # surface the (likely) typo once per run instead of staying silent.
        if default_name and default_axes == ParsedQuality() and not self._warned_default_quality:
            self._warned_default_quality = True
            hub_warn(f"imports.default_quality '{default_name}' matches no Sonarr quality definition - ignoring it")
        quality = resolve_quality(
            sonarr_axes,
            our_axes,
            default_axes,
            quality_defs,
            decision.quality,
        )
        # A resolved-but-source-less quality is the synthesized Unknown (an empty
        # nested quality already folded to None at the parse boundary).
        resolved = quality.quality
        if resolved is None or QualitySource.parse(resolved.source) is None:
            hub_warn(
                f"{content_path}: could not confidently resolve quality for {base} - importing as Unknown "
                "(re-grab risk)"
            )
        entry = ManualImportFile(
            path=path,
            seriesId=pending.series_id,
            episodeIds=decision.episode_ids,
            releaseGroup=pending.release_group,
            languages=lang_objs,
            quality=quality,
        )
        if context.omit_download_id:
            # Dead-tracked shape: the downloadId stays UNSET so the
            # exclude_unset wire dump omits the key entirely (never null) and
            # Sonarr's Execute takes the untracked branch - the tracked tail
            # dereferences the poisoned download's null ImportItem.
            return entry
        # model_copy(update=...) marks downloadId as set, so it reaches the wire.
        return entry.model_copy(update={"downloadId": pending.infohash})


class _SeedStatuses(NamedTuple):
    """The seed-gated import state both reconcile consumers read.

    A same-poll episode snapshot (fresh index + recommended overwrite-guard
    groups) and the per-target file statuses pinned to the seed set.
    """

    snapshot: EpisodeSnapshot
    statuses: dict[int, EpisodeFileStatus]


@dataclass(frozen=True, slots=True)
class PendingSeedContext:
    """The per-entry values every seed built for one AniList entry carries.

    One instance per `process_al_id` call, threaded whole into
    `ImportReconciler.build_pending_seeds` (instead of five loose params) and
    copied onto each `PendingImport` the entry produces.
    """

    al_id: int
    """The AniList entry id - part of each record's `PendingKey`."""
    series_id: int
    """The Sonarr series id the entry's files belong to."""
    title: str
    """The AniList display title (logging only)."""
    coverage: str | None = None
    """The entry's season/episode coverage at grab time (logging only)."""
    url: str | None = None
    """The SeaDex entry URL at grab time (logging only)."""


class ImportReconciler:
    """Decides a completed download's state and builds the grab-time seeds.

    Constructed once per run in `SonarrSync` from the shared
    deps, the episode collaborator, and the `ImportExecutor`. The strategy's
    `import_completed` / `process_al_id` hooks delegate here: `import_completed`
    drives one reconcile poll (deferring the payload mechanics to the executor), and
    `build_pending_seeds` produces the authoritative `PendingImport` records the
    engine persists at the add site.
    """

    def __init__(self, deps: RunDeps, episodes: SonarrEpisodes, executor: ImportExecutor) -> None:
        """Bind the cache/logger off the deps + the composed collaborators.

        Args:
            deps: The shared collaborators. The cache store + logger are
                read off it.
            episodes: The strategy's episode collaborator (its
                `episodes_for_series` is the source of truth for "imported").
            executor: The strategy's import executor (rescan,
                queue/command reads, and the manual-import POST).
        """

        self._episodes = episodes
        self._executor = executor
        self.cache_store = deps.cache_store
        self.logger = deps.logger

    def build_pending_seeds(
        self,
        *,
        seadex_dict: SeadexDict,
        ep_list: list[SonarrEpisode],
        entry: PendingSeedContext,
    ) -> dict[str, PendingImport]:
        """Build `infohash -> PendingImport` for every release marked to grab.

        For each downloadable url with a hash, seed our authoritative
        `normalized basename -> episode ids` map from the cached `/parse`
        results and the `(season, episode) -> id` index. The map is best-effort
        at grab time (the series may not be fully in Sonarr yet). It self-heals at
        import time, when the files are on disk and the series exists, so a record
        is seeded for every grabbed torrent that carries at least one video file -
        not only the ones already fully mapped.

        Seeds honor the used-once discipline assignment enforces: the first
        file in SeaDex order claiming an episode id wins, and a later claimant
        (a v2, or a duplicate leaf from a second folder) is left unseeded for
        import-time assignment, which defers the colliding file the same way.

        Args:
            seadex_dict: The filtered releases. `url_item.download`
                marks the ones the engine will add.
            ep_list: The relevant Sonarr episodes (carry ids).
            entry: The per-entry context (al_id, series id, title,
                coverage, url) copied onto every seed. The coverage/url ride the
                record so a carried-over record can render its inline
                `files`/`link` lines next run without re-deriving them.

        Returns:
            Seeds keyed by infohash (empty when nothing downloadable carries a
            video file). Per-entry: a sibling entry sharing a torrent builds its
            own seed with its own `al_id`, so the records coexist in the store.
        """

        ep_id_map = build_episode_id_map(ep_list)
        # The resolved episode ids for this entry, in season order - persisted onto
        # every record so import-time assignment maps files into OUR set (the same
        # mapping the add flow resolved) instead of re-deriving identity from
        # Sonarr's title parse.
        ordered_episode_ids = [ep.id for ep in ep_list if ep.id]
        # Per-file parse records are read straight from the cache facade
        # (`get_sonarr_parse`): each is the persisted parse entry
        # `{"fetched_at": str, "episodes": [...]}` written by
        # `parse_episodes_from_seadex` in the same run (staged writes are visible
        # to reads on the same connection).
        added_at = datetime.now().strftime(UPDATED_AT_STR_FORMAT)

        pending_seeds: dict[str, PendingImport] = {}

        for srg, srg_item in seadex_dict.items():
            for url_item in srg_item.urls.values():
                if not (url_item.download and url_item.infohash):
                    continue

                # The video files this torrent should import (subs / fonts / NCED
                # dropped).
                video_files = [base for _, base in video_file_entries(url_item.files)]

                # No importable video files at all -> nothing to track.
                if not video_files:
                    continue

                # Best-effort grab-time mapping, keyed by NORMALIZED basename so it
                # matches the on-disk leaves at import time (NFC/NFD-safe).
                file_episode_map: dict[str, list[int]] = {}
                claimed: set[int] = set()
                for base in video_files:
                    record = self.cache_store.get_sonarr_parse(base)
                    if not record:
                        continue
                    file_ids = episode_ids_for_parsed(parsed_episodes(record), ep_id_map)
                    # First claim in file order wins: assignment defers a later
                    # file whose ids collide, so the seed refuses it the same way.
                    if file_ids and not any(i in claimed for i in file_ids):
                        file_episode_map[normalize_basename(base)] = file_ids
                        claimed.update(file_ids)

                pending_seeds[url_item.infohash] = PendingImport(
                    infohash=url_item.infohash,
                    series_id=entry.series_id,
                    al_id=entry.al_id,
                    file_episode_map=file_episode_map,
                    # episode_ids is a legacy read-only fallback: never seeded (any
                    # value here would only duplicate the map, which readers dedupe).
                    episode_ids=[],
                    release_group=srg,
                    is_dual_audio=url_item.is_dual_audio,
                    seadex_files=video_files,
                    title=entry.title,
                    added_at=added_at,
                    coverage=entry.coverage,
                    url=entry.url,
                    ordered_episode_ids=ordered_episode_ids,
                )

        return pending_seeds

    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """One reconcile/import poll for a completed download.

        Reads the current episode files and Sonarr's (refreshed) queue as the
        source of truth - never the cache:

          * every intended episode already holds the recommended release ->
            `IMPORTED` + `files_present` (drop the record).
          * Sonarr is genuinely importing right now -> `RETRY` (don't race it).
          * a clean `importPending` -> `RETRY` until `force` (the engine
            forces on the snapshot/reconcile passes and on the final in-bound
            monitor poll, so a download Sonarr will never import - e.g. Completed
            Download Handling off, which parks it in `importPending` forever -
            is still imported rather than waited on indefinitely).
          * otherwise (`importBlocked` / `failed` / not tracked / forced clean
            pending) -> drive our authoritative series-pinned manual import.

        Args:
            pending: The durable record for the completed torrent.
            content_path: The qBittorrent `content_path` to import from.
            force: Stop deferring to Sonarr on a clean `importPending`.
            at_deadline: The final attempt - a still-missing intended file
                is terminal, so warn loudly (off the deadline it's debug).
        """

        label = pending.display_label

        # Rescan (throttled) so the queue we read reflects the finished torrent.
        self._executor.refresh_downloads()

        # "Files inserted" bar counts, pinned to the seed set so the denominator
        # never rescales mid-import. Determinate only when the seed map covers every
        # intended file. An incomplete map reports 0/0 so the importing row stays
        # indeterminate (a partial seed must never show a misleading bar) and only
        # the manual import's repaired done-check below can finish it.
        seeded_targets = self._pending_target_ids(pending)
        seed_complete = bool(seeded_targets) and self._seed_map_is_complete(pending)
        seed = self._seed_statuses(pending, seeded_targets if seed_complete else [])
        done = self._recommended_count(seed.statuses)
        total = len(seeded_targets) if seed_complete else 0

        def probe(readiness: ImportReadiness, *, files_present: bool, command_issued: bool) -> ImportProbe:
            return ImportProbe(
                readiness,
                files_present=files_present,
                command_issued=command_issued,
                imported_count=done,
                target_count=total,
            )

        # Fast path: when our grab-time map already covers every video file, the
        # done-check is trustworthy without scanning the folder. An incomplete map
        # falls through to the manual import, which repairs it from the on-disk
        # files and re-checks against the complete set.
        if seed_complete and all_targets_done(seed.statuses):
            self.logger.debug(f"{label}: already imported (recommended files present)")
            return probe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        verdict = classify_queue(self._executor.queue_states(pending.infohash))
        if verdict is QueueVerdict.WAIT:
            self.logger.debug(f"{label}: Sonarr is importing; waiting")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)
        if verdict is QueueVerdict.PENDING_CLEAN and not force:
            self.logger.debug(f"{label}: Sonarr has it pending; waiting")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # A ManualImport we (or a prior run) already POSTed may still be running
        # server-side after Sonarr dropped the torrent from the regular queue - so
        # the queue reads "empty -> step in" and we'd stack a duplicate every poll.
        # NOT gated on `force`: the carried-over reconcile path always forces, and
        # that is exactly the path that loops. An in-flight command must suppress a
        # re-issue regardless (`force` overrides Sonarr's clean-pending deferral,
        # a different state). A false positive only waits (bounded by the deadline).
        if manual_import_in_flight(
            self._executor.list_commands(),
            pending.infohash,
            self._executor.content_paths(content_path),
            set(seeded_targets),
        ):
            self.logger.debug(f"{label}: a ManualImport is already in flight; waiting")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # STEP_IN, an empty queue, or a forced clean-pending: drive our import.
        result = self._executor.run_manual_import(
            pending,
            content_path,
            snapshot=seed.snapshot,
            at_deadline=at_deadline,
        )
        return replace(result, imported_count=done, target_count=total)

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Cheap, read-only "files inserted" count for one record (the Tier-2 bar).

        Reads ONLY the fresh episode files - never the throttled refresh, the queue,
        or qBittorrent - and counts the seed targets that now hold the recommended
        release. `determinate` is False (and the counts 0) unless the seed map is
        whole, so a partial seed never shows a misleading bar or promotes early. That
        decision is left to the heavy poll's repaired done-check.
        """

        seeded_targets = self._pending_target_ids(pending)
        if not seeded_targets or not self._seed_map_is_complete(pending):
            return ImportProgress(0, 0, determinate=False)
        seed = self._seed_statuses(pending, seeded_targets)
        return ImportProgress(self._recommended_count(seed.statuses), len(seeded_targets), determinate=True)

    def _seed_statuses(self, pending: PendingImport, targets: list[int]) -> _SeedStatuses:
        """Fetch the series' episodes FRESH and classify `targets` against them.

        Episode files are the source of truth for "already imported". Callers gate
        the bar on seed completeness by passing `[]` (empty statuses). The episode
        index + recommended groups are still fetched for the manual import.
        """

        episodes = self._episodes.episodes_for_series(pending.series_id)
        snapshot = EpisodeSnapshot(
            episodes_by_id={ep.id: ep for ep in episodes if ep.id},
            recommended_groups=self._recommended_groups(pending.series_id, pending.release_group),
        )
        return _SeedStatuses(snapshot, episode_file_statuses(targets, snapshot))

    @staticmethod
    def _recommended_count(statuses: dict[int, EpisodeFileStatus]) -> int:
        """How many target episodes already hold the recommended release (bar numerator)."""

        return sum(1 for status in statuses.values() if status is EpisodeFileStatus.RECOMMENDED)

    def _series_pending_records(self, series_id: int) -> list[dict[str, Any]]:
        """Raw durable pending records for one series (any release group).

        Each record is the genuinely-open cache JSON form of a
        `PendingImport` (`to_json`/`from_json`), so it is typed
        `dict[str, Any]`.
        """

        # `get_pending_for_series` returns a fresh snapshot `{infohash -> record}`
        # already filtered to this series in SQL (so a record dropped earlier this run
        # is absent). The `record ->> 'series_id'` match only returns JSON objects,
        # so every value is a typed record - no defensive isinstance/widen needed.
        return list(self.cache_store.get_pending_for_series(Arr.SONARR, series_id).values())

    def _recommended_groups(self, series_id: int, this_group: str) -> set[str]:
        """Normalized recommended groups for the series (the overwrite-guard set).

        The union of this torrent's group and the group of every other pending
        record we grabbed for the same series, so an episode our mapping assigned
        to another preferred torrent is never overwritten by this one.
        """

        groups: set[str] = set()
        if this_group:
            groups.add(normalize_group(this_group))
        for raw in self._series_pending_records(series_id):
            group = raw.get("release_group")
            if group:
                groups.add(normalize_group(group))
        return groups

    @staticmethod
    def _pending_target_ids(pending: PendingImport) -> list[int]:
        """Our intended episode ids for a record (map values + single-file fallback)."""

        ids: list[int] = []
        seen: set[int] = set()
        for file_ids in pending.file_episode_map.values():
            for ep_id in file_ids:
                if ep_id and ep_id not in seen:
                    seen.add(ep_id)
                    ids.append(ep_id)
        for ep_id in pending.episode_ids:
            if ep_id and ep_id not in seen:
                seen.add(ep_id)
                ids.append(ep_id)
        return ids

    @staticmethod
    def _seed_map_is_complete(pending: PendingImport) -> bool:
        """Whether the grab-time map already covers every video file we grabbed."""

        return bool(pending.seadex_files) and len(pending.file_episode_map) >= len(
            pending.seadex_files,
        )
