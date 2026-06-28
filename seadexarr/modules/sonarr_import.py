"""Import-time execution: build and POST the authoritative manual import.

Extracted from :class:`~.seadex_sonarr.SonarrSync`. ``ImportExecutor`` owns the
mechanical "consume" side of one completed download: throttled download rescan,
queue/command reads, and - the heart of it - turning OUR resolved
``basename -> episode ids`` map (built by :class:`~.sonarr_mapper.FileEpisodeMapper`)
into a Sonarr ManualImport payload and POSTing it, with per-run quality/language
caches. The reconcile *decision* (what state the download is in, when to step in)
stays on the strategy's ``import_completed`` for now.
"""

import os
import time

from .log import indent_string
from .manual_import import (
    ImportAction,
    ImportDecision,
    ImportProbe,
    ImportReadiness,
    PendingImport,
    QueueRecordView,
    all_targets_done,
    build_episode_id_map,
    derive_languages,
    episode_file_statuses,
    parse_quality_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
    targets_needing_import,
)
from .seadex_arr import RunDeps
from .seadex_types import (
    CommandResource,
    Language,
    ManualImportFile,
    QualityDefinition,
    QualitySource,
    SonarrEpisode,
)
from .sonarr_client import SonarrClient
from .sonarr_mapper import FileEpisodeMapper

# RefreshMonitoredDownloads is quick (Sonarr re-scans its clients); poll its
# command status up to this many times, sleeping this long between, before
# proceeding regardless. Waiting means the queue we read next reflects the
# rescan; the bound means a stuck command never blocks the run.
_REFRESH_COMMAND_MAX_POLLS = 30
_REFRESH_COMMAND_POLL_S = 1
_COMMAND_TERMINAL_STATES = frozenset({"completed", "failed", "aborted", "cancelled"})


class ImportExecutor:
    """Builds/POSTs the manual-import payload + owns the per-run import caches.

    Constructed once per run in :class:`~.seadex_sonarr.SonarrSync` from the shared
    deps, the strategy's Sonarr client, and the strategy's
    :class:`~.sonarr_mapper.FileEpisodeMapper`. ``import_completed`` decides the
    download's state and, when it's time to step in, calls
    :meth:`run_manual_import`; the executor also exposes the throttled rescan and
    the queue/command reads that decision consults.
    """

    def __init__(self, deps: RunDeps, sonarr: SonarrClient, mapper: FileEpisodeMapper) -> None:
        """Bind the Sonarr client, config/logger, and the file->episode mapper.

        Args:
            deps (RunDeps): The shared collaborators; the config + logger are read
                off it.
            sonarr (SonarrClient): The strategy's Sonarr client.
            mapper (FileEpisodeMapper): The strategy's import-time mapper (its
                ``candidate_files`` / ``assign`` build the authoritative map).
        """

        self.sonarr = sonarr
        self._config = deps.config
        self.logger = deps.logger
        self._mapper = mapper

        # Per-run caches of the Sonarr quality-definition / language lists, used
        # to resolve a quality name / language names into the manual-import
        # payload objects. Fetched lazily on the first import and then reused for
        # the rest of the run so repeated imports don't re-hit the endpoints;
        # None means "not yet fetched" (cleared in reset, the run-start hook).
        self._quality_defs_cache: list[QualityDefinition] | None = None
        self._languages_cache: list[Language] | None = None

        # Infohashes for which we've already warned that some on-disk files could
        # not be placed in the resolved set, so the loud "left these for you" line
        # is logged once a run rather than every poll until the record clears.
        self._warned_unplaceable: set[str] = set()

        # Monotonic time of the last RefreshMonitoredDownloads we asked Sonarr for,
        # used to throttle the rescan: the blocking pass calls import_completed
        # every poll and may walk several torrents back-to-back, so we re-issue the
        # (global) refresh at most once per import_poll_interval rather than on
        # every call. None means "not refreshed yet this run" (reset in reset).
        self._last_refresh_monotonic: float | None = None

    def reset(self) -> None:
        """Drop the per-run import scratch (run-start, via get_items)."""

        self._quality_defs_cache = None
        self._languages_cache = None
        self._warned_unplaceable = set()
        self._last_refresh_monotonic = None

    def refresh_downloads(self) -> None:
        """Queue RefreshMonitoredDownloads (throttled) and wait for it, best-effort.

        RefreshMonitoredDownloads is global and the blocking pass polls often (and
        may walk several torrents back-to-back), so it's re-issued at most once per
        ``import_poll_interval``. Waiting for the command to finish means the queue
        read that follows reflects the rescan; the poll bound means a stuck command
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
        self.logger.debug(indent_string("Asked Sonarr to rescan its downloads"))

        for _ in range(_REFRESH_COMMAND_MAX_POLLS):
            command = self.sonarr.command_status(cmd_id)
            state = command.status or ""
            if state.casefold() in _COMMAND_TERMINAL_STATES:
                return
            time.sleep(_REFRESH_COMMAND_POLL_S)

    def queue_record_views(self, infohash: str) -> tuple[str, list[QueueRecordView]]:
        """Reduce this download's queue records to what :func:`classify_queue` needs.

        Matches records to the torrent by ``downloadId`` (case-insensitively;
        Sonarr stores the infohash uppercased) and keeps the state, status, and
        whether status messages are present - the three signals that tell a healthy
        pending item from a stuck/blocked one. Records with no tracked state are
        dropped; an empty result means Sonarr isn't tracking the download.

        Args:
            infohash (str): The torrent infohash (the download id).
        """

        target = infohash.casefold()
        views: list[QueueRecordView] = []
        download_id = ""
        for record in self.sonarr.queue():
            dl_id = record.download_id
            if dl_id is None or dl_id.casefold() != target:
                continue
            if not record.state:
                continue
            download_id = dl_id
            views.append(
                QueueRecordView(
                    state=record.state,
                    status=record.status or "",
                    has_messages=record.has_messages,
                ),
            )
        return download_id if download_id else infohash, views

    def list_commands(self) -> list[CommandResource]:
        """The current Sonarr command list, for the in-flight ManualImport guard.

        A thin pass-through to :meth:`SonarrClient.list_commands` (mirrors
        :meth:`queue_record_views`' delegation to ``self.sonarr``). Fetched fresh
        every poll - never cached - since an in-flight command's status changes as
        Sonarr finishes the import.
        """

        return self.sonarr.list_commands()

    def run_manual_import(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        episodes_by_id: dict[int, SonarrEpisode],
        recommended_groups: set[str],
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Drive our authoritative series-pinned manual import for one download.

        Scans ``content_path`` for candidates (pinned to ``pending.series_id``),
        repairs our file->episode map from the actual on-disk files (re-parsing
        whatever the seed didn't cover, mapped through OUR ``(season, episode) ->
        id`` index - never Sonarr's candidate episode assignment), then imports
        EXACTLY the files our map intends: each file's episodes that don't already
        hold a recommended release (so a recommended file is never overwritten) and
        no file outside our map (so an episode our mapping gave to another preferred
        torrent is never imported here). An intended file Sonarr can't see yet is
        retried, never silently skipped.

        Returns an :class:`ImportProbe`. A manual-import command's copy is async, so
        accepting the command is NOT ``files_present`` - the probe reads
        ``RETRY`` + ``command_issued`` until a later poll verifies the episode files
        actually landed. ``files_present`` is set only when every intended episode
        already holds a recommended file (nothing left to copy).

        Args:
            pending (PendingImport): The durable record for the completed torrent.
            content_path (str): The qBittorrent ``content_path`` to import from;
                also the label for this download's log lines.
            episodes_by_id (dict[int, SonarrEpisode]): Current series episodes by id.
            recommended_groups (set[str]): Normalized recommended-group guard set.
            at_deadline (bool): The final attempt - a still-missing intended file
                is terminal, so warn loudly; otherwise it's an expected early-poll
                gap and only logged at debug.
        """

        candidates = self.sonarr.manual_import_candidates(
            pending=pending,
            filter_existing_files=False,
        )
        if candidates is None:
            # Transient (timeout / non-200); the client already warned. Ask again.
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        candidates_by_basename = self._mapper.candidate_files(candidates)
        ep_id_map = build_episode_id_map(list(episodes_by_id.values()))
        authoritative_map, unplaceable = self._mapper.assign(
            pending,
            candidates_by_basename,
            ep_id_map,
        )
        if unplaceable:
            self._warn_unplaceable_files(pending, unplaceable)

        if not authoritative_map:
            self.logger.debug(
                indent_string(f"{content_path}: no mappable files for {pending.title} yet"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Done-check against the COMPLETE (repaired) intended set, from the files.
        target_ids = sorted({i for ids in authoritative_map.values() for i in ids})
        statuses = episode_file_statuses(target_ids, episodes_by_id, recommended_groups)
        if all_targets_done(statuses):
            self.logger.debug(
                indent_string(f"{content_path}: already imported (recommended files present)"),
            )
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        needing = targets_needing_import(statuses)
        decisions = plan_import_files(authoritative_map, candidates_by_basename, needing)

        files: list[ManualImportFile] = []
        missing: list[str] = []
        for decision in decisions:
            match decision.action:
                case ImportAction.MISSING:
                    missing.append(decision.basename)
                case ImportAction.IMPORT:
                    files.append(self._build_file_entry(decision, pending, content_path))
                case _:
                    # SAMPLE / ALREADY / SKIP_DONE -> nothing to import for this file.
                    continue

        if missing:
            # Intended files our map covers but Sonarr can't see yet. An early poll
            # finding them absent is expected (the copy hasn't landed), so it's only
            # noisy at the deadline, where a still-missing file is terminal: warn
            # loudly only then, debug otherwise. Either way the record is retried,
            # never dropped silently.
            message = indent_string(
                f"{content_path}: {len(missing)} intended file(s) not visible to Sonarr for {pending.title}; will retry",
            )
            if at_deadline:
                self.logger.warning(message)
            else:
                self.logger.debug(message)

        if not files:
            # Nothing to queue this poll: retry if files are merely missing, else
            # everything intended is already satisfied (already/sample/skip_done).
            if missing:
                return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=self._config.imports.mode,
        )
        if cmd_id is None:
            self.logger.debug(
                indent_string(f"{content_path}: Sonarr rejected the import command; will retry"),
            )
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # The command was accepted, but its copy is async - the episode files may
        # not have landed yet (a remote-mount copy isn't instant). Do NOT declare
        # the files imported on command acceptance: report RETRY + command_issued,
        # so the next monitor cycle flips to files_present once they appear.
        self.logger.debug(
            indent_string(f"{content_path}: queued {len(files)} file(s) for import (command {cmd_id})"),
        )
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
        label = pending.title or pending.infohash
        coverage = f" ({pending.coverage})" if pending.coverage else ""
        self.logger.warning(
            indent_string(
                f"{label}{coverage}: {len(unplaceable)} file(s) could not be mapped "
                f"to a resolved episode and were left unimported",
            ),
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
        pending: PendingImport,
        content_path: str,
    ) -> ManualImportFile:
        """Build one ManualImport file payload from a planned ``import`` decision.

        The episode ids come straight from our authoritative map (never Sonarr's
        parse); the quality is decided per axis with precedence Sonarr's parse ->
        our filename parse -> the configured default, and always emits a real
        quality (never an omitted key), warning only when it resolves to Unknown.
        The language objects + quality definitions are read from this run's caches.

        Only ``import`` decisions reach here, so ``decision.path`` is the on-disk
        candidate path (always set); the ``or decision.basename`` keeps the
        payload ``path`` a non-null ``str`` for the type.
        """

        lang_objs = self._import_language_objects(pending)
        quality_defs = self._quality_definitions()

        path = decision.path or decision.basename
        entry: ManualImportFile = {
            "path": path,
            "seriesId": pending.series_id,
            "episodeIds": decision.episode_ids,
            "releaseGroup": pending.release_group,
            "downloadId": pending.infohash,
            "languages": lang_objs,
        }

        base = os.path.basename(path)
        sonarr_axes = quality_axes_from_model(decision.quality)
        our_axes = parse_quality_from_filename(base)
        default_axes = quality_axes_from_name(self._config.imports.default_quality, quality_defs)
        quality = resolve_quality(
            sonarr_axes,
            our_axes,
            default_axes,
            quality_defs,
            decision.quality,
        )
        entry["quality"] = quality
        resolved = quality.get("quality") or {}
        if QualitySource.parse(resolved.get("source")) is None:
            self.logger.warning(
                indent_string(
                    f"{content_path}: could not confidently resolve quality for {base}; importing as Unknown (re-grab risk)",
                ),
            )
        return entry
