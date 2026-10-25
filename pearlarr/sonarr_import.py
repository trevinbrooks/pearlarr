"""Import-time subsystem: decide a download's state, then build/POST the import."""

import logging
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

from .arr_http import DeleteOutcome
from .episode_state import EpisodeFileStatus, EpisodeSnapshot, GroupVotes, RecordSnapshot, TargetStatuses
from .import_files import CandidateFile, ImportAction, ImportDecision, plan_import_files
from .import_quality import (
    ParsedQuality,
    derive_languages,
    parse_quality_from_filename,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
)
from .log import count_noun, pluralize
from .manual_import import (
    NO_PROGRESS,
    AttemptKind,
    Deferral,
    EffectStatus,
    FileEpisodeMap,
    ImportProbe,
    ImportProgress,
    PendingImport,
    normalized_leaf,
    path_leaf,
    translate_download_path,
)
from .output import hub_note, hub_warn
from .pending_records import PendingRecords
from .placement_types import episode_index
from .probe_verdicts import (
    CommandBlock,
    ContentPaths,
    DownloadHistoryVerdict,
    DownloadMatch,
    HistoryImport,
    QueueVerdict,
    classify_commands,
    classify_download_history,
    classify_queue,
    disk_commands_changed_since,
    placements_from_history,
    rescanned_since_add,
    sonarr_restarted,
)
from .run_services import RunDeps
from .seadex_types import (
    CommandResource,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    QualityDefinition,
    QualitySource,
    QueueRecord,
    RemotePathMapping,
    SonarrEpisode,
)
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_mapper import FileEpisodeMapper

# How many times the rescan's status is polled, `_RESCAN_COMMAND_POLL_S` apart.
_RESCAN_COMMAND_MAX_POLLS = 30
_RESCAN_COMMAND_POLL_S = 1
_COMMAND_TERMINAL_STATES = frozenset({"completed", "failed", "aborted", "cancelled"})
# A kept by-id scan expires after this many poll intervals.
_KEPT_SCAN_POLLS = 2


def _hostname(url: str | None) -> str | None:
    """The bare hostname of a configured URL (tolerates a scheme-less `host:port`)."""

    if not url:
        return None
    return urlsplit(url).hostname or urlsplit(f"//{url}").hostname


@dataclass(frozen=True, slots=True)
class _CandidateScan:
    """One poll's candidate fetch plus the history verdict behind the entry shape it implies."""

    candidates: list[ManualImportCandidate]
    history: DownloadHistoryVerdict | None
    """The fallback's history verdict. None when the by-id scan served or the probe failed, and both read clean."""
    import_rows: tuple[HistoryImport, ...] = ()
    """The dead-tracking import's rows that stand in for an empty folder scan."""
    by_id: bool = False
    """Whether the by-id scan answered with files."""

    @property
    def omit_download_id(self) -> bool:
        """ALL entries must omit the downloadId: Sonarr's Execute tail NREs on the poisoned download."""

        return self.history is not None and self.history.dead_tracked

    @property
    def dead_tracked_empty(self) -> bool:
        """Sonarr won't serve the download by id and its folder scan found nothing."""

        return self.omit_download_id and not self.candidates


@dataclass(frozen=True, slots=True)
class PollReading:
    """What a step-in poll read after its guard reads cleared, handed to the import."""

    scan: _CandidateScan
    """The by-id scan, or the folder fallback behind it."""
    snapshot: RecordSnapshot
    """The snapshot of every series the record spans, read after the baseline."""
    baseline: list[CommandResource]
    """The command list read once the poll's candidates were in hand."""
    at_deadline: bool
    """Whether this is the readiness-deadline attempt."""


@dataclass(frozen=True, slots=True)
class _ImportContext:
    """One poll's import context, bound once after the scan and read by every step past it."""

    pending: PendingImport
    """The record this poll imports."""
    content_path: str
    """The download's content path as qBittorrent reports it."""
    scan: _CandidateScan
    """This poll's candidate fetch plus the history verdict behind it."""
    snapshot: RecordSnapshot
    """The same-poll snapshot of every series the record spans, which the done-check reads."""
    baseline: list[CommandResource]
    """The command list read once the poll's candidates were in hand."""
    candidates_by_basename: dict[str, CandidateFile]
    """The scan's candidates keyed by normalized basename, built once for the planner and the file entries."""
    at_deadline: bool
    """Whether this is the readiness-deadline attempt, the one that warns."""

    def posting_series(self, ep_id: int) -> int | None:
        """The series a file posts under: its route's, else a one-series record's."""

        if (series_id := self.snapshot.route(ep_id).series_id) is not None:
            return series_id
        series_ids = self.pending.series_ids
        return series_ids[0] if len(series_ids) == 1 else None


@dataclass
class _ScanScratch:
    """The scanner's per-run fallback scratch, reassigned wholesale by `reset`."""

    folder_pinned: dict[str, bool] = field(default_factory=dict[str, bool])
    """The downloads pinned to their folder -> whether the downloadId scan before the pin failed."""
    history_verdicts: dict[str, DownloadHistoryVerdict] = field(default_factory=dict[str, DownloadHistoryVerdict])
    """The dead verdicts, with their import rows. A clean verdict or a probe failure re-probes next activation."""
    path_mappings: list[RemotePathMapping] | None = None
    """Sonarr's remote path mappings. None = unfetched."""
    translated_paths: dict[str, str] = field(default_factory=dict[str, str])
    """Download path -> its Sonarr-visible path."""


class CandidateScanner:
    """Per-run candidate scans: the downloadId scan with the folder fallback."""

    def __init__(self, sonarr: AbstractSonarrClient, qbit_host: str | None, logger: logging.Logger) -> None:
        self.sonarr = sonarr
        self._qbit_host = qbit_host
        self.logger = logger
        self._scratch = _ScanScratch()

    def reset(self) -> None:
        """Drop the per-run fallback scratch (run-start, via the executor)."""

        self._scratch = _ScanScratch()

    def content_paths(self, content_path: str) -> ContentPaths:
        """The raw + Sonarr-visible path pair the in-flight guard matches against.

        Read-only over the memoized translations, never fetches. Both views are the raw path until a fallback activates.
        """

        return ContentPaths(
            raw=content_path,
            sonarr_visible=self._scratch.translated_paths.get(content_path, content_path),
        )

    def _remote_path_mappings(self) -> list[RemotePathMapping]:
        """Sonarr's remote path mappings, fetched once per run on first fallback use.

        A failed fetch caches [] for the run: translation degrades to a no-op, so nothing pins and the scan is retried.
        """

        if self._scratch.path_mappings is None:
            self._scratch.path_mappings = self.sonarr.remote_path_mappings() or []
        return self._scratch.path_mappings

    def _sonarr_visible_path(self, content_path: str) -> str:
        """Translate (and memoize) a download path into Sonarr's filesystem view."""

        translated = self._scratch.translated_paths.get(content_path)
        if translated is None:
            translated = translate_download_path(
                content_path,
                self._remote_path_mappings(),
                self._qbit_host,
            )
            self._scratch.translated_paths[content_path] = translated
        return translated

    def _probe_history(self, pending: PendingImport) -> DownloadHistoryVerdict | None:
        """Classify a download's Sonarr history (a dead verdict memoized), or None on probe failure."""

        verdict = self._scratch.history_verdicts.get(pending.infohash)
        if verdict is not None:
            return verdict
        page = self.sonarr.history_for_download(download_id=pending.infohash)
        if page is None:
            return None
        verdict = classify_download_history(page)
        if verdict.dead_tracked:
            self._scratch.history_verdicts[pending.infohash] = verdict
            when = f" on {verdict.date.split('T')[0]}" if verdict.date else ""
            self.logger.debug(
                f"{pending.display_label}: Sonarr recorded this download as {verdict.event}{when} "
                "and won't serve it by id - importing from its folder instead"
            )
        return verdict

    def _dead_tracked(self, pending: PendingImport) -> bool:
        """Whether Sonarr's history says it won't serve the download by id."""

        verdict = self._probe_history(pending)
        return verdict is not None and verdict.dead_tracked

    def scan(self, pending: PendingImport, content_path: str, *, untracked: bool) -> _CandidateScan | None:
        """The candidates for one poll: the downloadId scan, with the folder fallback. None when the scan failed."""

        by_id_failed = self._scratch.folder_pinned.get(pending.infohash)
        if by_id_failed is None:
            candidates = self.sonarr.manual_import_candidates(pending=pending)
            by_id_failed = candidates is None
            if candidates is None:
                self.logger.debug(
                    f"{content_path}: downloadId scan failed for {pending.display_label} - trying the folder fallback"
                )
            elif candidates or not (untracked and self._dead_tracked(pending)):
                return _CandidateScan(candidates, history=None, by_id=bool(candidates))

        verdict = self._probe_history(pending)
        folder = self._sonarr_visible_path(content_path)
        folder_candidates = self.sonarr.manual_import_candidates_by_folder(
            folder=folder,
            title=pending.display_label,
        )
        if folder_candidates is None:
            return None
        if folder_candidates:
            # INVARIANT: pin folder mode on NONEMPTY success only. A 200 `[]` is exactly what an invisible or
            # untranslated folder returns, and pinning on it wedges the record blind to files, downloadId unretried.
            self._scratch.folder_pinned[pending.infohash] = by_id_failed
            return _CandidateScan(folder_candidates, history=verdict)
        # History stands in for an empty folder only after a failed downloadId scan.
        rows = verdict.import_rows if verdict is not None and by_id_failed else ()
        return _CandidateScan([], history=verdict, import_rows=rows)


@dataclass(frozen=True, slots=True)
class _KeptScan:
    """A by-id scan kept from a blocked poll for the record's next one."""

    content_path: str
    """The content path the scan was taken under."""
    scan: _CandidateScan
    """The scan itself."""
    kept_at_monotonic: float
    """The executor clock's reading when the scan was kept."""


@dataclass
class _ImportScratch:
    """The executor's per-run import scratch, reassigned wholesale by `reset`. A None cache = not yet fetched."""

    quality_defs: list[QualityDefinition] | None = None
    languages: list[Language] | None = None
    warned_unplaceable: set[str] = field(default_factory=set[str])
    warned_overridden: set[str] = field(default_factory=set[str])
    warned_empty_folder: set[str] = field(default_factory=set[str])
    warned_default_quality: bool = False
    last_rescan_monotonic: float | None = None
    rescanned: set[str] = field(default_factory=set[str])
    """Infohashes whose rescan completed this run."""
    kept_scans: dict[str, _KeptScan] = field(default_factory=dict[str, _KeptScan])
    """By-id scans kept from a blocked poll, by infohash."""
    issued_command_ids: set[int] = field(default_factory=set[int])
    completed_download_handling: bool | None = None


class ImportExecutor:
    """Builds/POSTs the manual-import payload + owns the per-run import caches."""

    def __init__(self, deps: RunDeps, sonarr: AbstractSonarrClient, mapper: FileEpisodeMapper) -> None:
        """Bind the Sonarr client, config/logger, mapper, and the candidate scanner."""

        self.sonarr = sonarr
        self._config = deps.config
        self.logger = deps.logger
        self._clock = deps.clock
        self._mapper = mapper
        self.scanner = CandidateScanner(sonarr, _hostname(deps.config.qbittorrent.host), deps.logger)
        self._scratch = _ImportScratch()

    def reset(self) -> None:
        """Drop the per-run import scratch (run-start, via get_items)."""

        self._scratch = _ImportScratch()
        self.scanner.reset()

    def completed_download_handling_enabled(self) -> bool:
        """Whether Sonarr imports its own completed downloads: with it off, it parks them forever."""

        if self._scratch.completed_download_handling is None:
            self._scratch.completed_download_handling = (
                self.sonarr.download_client_config().enable_completed_download_handling
            )
        return self._scratch.completed_download_handling

    def _is_own_command(self, command_id: int) -> bool:
        """Whether we POSTed this ManualImport command id this run."""

        return command_id in self._scratch.issued_command_ids

    def rescan_once(self, infohash: str) -> bool:
        """Rescan Sonarr's downloads for a record with no completed rescan on file, never within a poll interval.

        The interval runs from the last rescan's end. True when the rescan completed.
        """

        if infohash in self._scratch.rescanned:
            return False
        last = self._scratch.last_rescan_monotonic
        if last is not None and self._clock.now() - last < self._config.imports.poll_interval:
            return False
        try:
            completed = self._rescan()
        finally:
            self._scratch.last_rescan_monotonic = self._clock.now()
        if completed:
            self._scratch.rescanned.add(infohash)
        return completed

    def forget_rescans(self) -> None:
        """Drop this run's rescan markers."""

        self._scratch.rescanned.clear()

    def keep_scan(self, infohash: str, content_path: str, scan: _CandidateScan) -> None:
        """Keep a by-id scan for the record's next poll."""

        if scan.by_id:
            self._scratch.kept_scans[infohash] = _KeptScan(content_path, scan, self._clock.now())

    def take_kept_scan(self, infohash: str, content_path: str) -> _CandidateScan | None:
        """The by-id scan kept for this poll, taken once. None when there is none, it expired, or the path moved."""

        kept = self._scratch.kept_scans.pop(infohash, None)
        if kept is None or kept.content_path != content_path:
            return None
        if self._clock.now() - kept.kept_at_monotonic > _KEPT_SCAN_POLLS * self._config.imports.poll_interval:
            return None
        return kept.scan

    def _rescan(self) -> bool:
        """POST the rescan and wait for it. True when Sonarr reported it complete."""

        cmd_id = self.sonarr.refresh_monitored_downloads()
        if cmd_id is None:
            return False
        self.logger.debug("Asked Sonarr to rescan its downloads")

        for _ in range(_RESCAN_COMMAND_MAX_POLLS):
            command = self.sonarr.command_status(cmd_id)
            state = (command.status or "").casefold()
            if state in _COMMAND_TERMINAL_STATES:
                return state == "completed"
            if state == "orphaned":
                # A Sonarr restart cut the rescan short.
                cmd_id = self.sonarr.refresh_monitored_downloads()
                if cmd_id is None:
                    return False
                self.logger.debug("Sonarr restarted during the rescan - asked it to rescan again")
            self._clock.sleep(_RESCAN_COMMAND_POLL_S)
        return False

    def queue_records(self, infohash: str) -> list[QueueRecord] | None:
        """This download's queue records, matched case-insensitively (Sonarr stores the hash uppercased).

        None when the queue read failed.
        """

        queue = self.sonarr.queue()
        if queue is None:
            return None
        target = infohash.casefold()
        return [
            record for record in queue if record.download_id is not None and record.download_id.casefold() == target
        ]

    def close_tracked(self, pending: PendingImport) -> EffectStatus:
        """Dismiss Sonarr's leftover queue entry, leaving an unknown-series one alone (dismissing it 500s)."""

        rows = self.queue_records(pending.infohash)
        if rows is None:
            # The read already retried inside the transport, so no in-run RETRY.
            self.logger.debug(f"{pending.display_label}: queue read failed - the cleanup will retry next run")
            return EffectStatus.FAILED
        queue_id = next((row.id for row in rows if row.id and row.series_id), None)
        if queue_id is None:
            if any(row.id for row in rows):
                self.logger.debug(f"{pending.display_label}: unknown-series queue entry - left for Sonarr to drop")
            else:
                self.logger.debug(f"{pending.display_label}: no leftover Sonarr queue entry to close")
            return EffectStatus.SKIPPED
        match self.sonarr.queue_delete(queue_id):
            case DeleteOutcome.OK:
                hub_note(f"Removed the imported download {pending.display_label} from Sonarr's queue")
                return EffectStatus.DONE
            case DeleteOutcome.GONE:
                self.logger.debug(f"{pending.display_label}: queue entry {queue_id} already gone - nothing to remove")
                return EffectStatus.DONE
            case DeleteOutcome.FAILED:
                # The client's warning names no download (coalesced template). This one does.
                self.logger.debug(f"{pending.display_label}: queue entry {queue_id} not removed - will retry")
                return EffectStatus.RETRY

    def list_commands(self) -> list[CommandResource] | None:
        """The current Sonarr command list, never cached. None when the read failed."""

        return self.sonarr.list_commands()

    def command_block(
        self, commands: list[CommandResource], pending: PendingImport, content_path: str
    ) -> CommandBlock | None:
        """The command block that stops a step-in for the download now, logged, else None."""

        match = DownloadMatch(pending.infohash, self.scanner.content_paths(content_path), set(pending.target_ids()))
        block = classify_commands(commands, match, self._is_own_command)
        if block is not None:
            self.logger.debug(f"{pending.display_label}: {block.value}; waiting")
        return block

    def run_manual_import(self, pending: PendingImport, content_path: str, reading: PollReading) -> ImportProbe:
        """Import EXACTLY the files our map intends, never over a recommended file that isn't a copy of our own.

        The probe carries this poll's fresh placements for the record seam to persist.
        """

        scan = reading.scan
        context = _ImportContext(
            pending=pending,
            content_path=content_path,
            scan=scan,
            snapshot=reading.snapshot,
            baseline=reading.baseline,
            candidates_by_basename=self._mapper.candidate_files(scan.candidates),
            at_deadline=reading.at_deadline,
        )
        assignment = self._mapper.assign(pending, context.candidates_by_basename, context.snapshot.indexes)
        recovered: FileEpisodeMap = {}
        if scan.import_rows and not pending.seed_coverage().mapped:
            # An empty scan means the mapper placed nothing, so the record's own map is the assignment.
            recovered = placements_from_history(scan.import_rows, pending)
            self.logger.debug(
                f"{content_path}: Sonarr's history placed {count_noun(len(recovered), 'file')} "
                f"for {pending.display_label} that no scan saw"
            )
        placed = assignment.placed
        excluded = assignment.excluded
        authoritative = {**assignment.assigned, **recovered}
        on_record = set(pending.excluded_files)
        if new_exclusions := [p for p in excluded if p.name not in on_record]:
            verdicts = ", ".join(f"{p.name} ({p.verdict})" for p in new_exclusions)
            self.logger.debug(f"{pending.display_label}: never this record's to import: {verdicts}")
        unplaced = assignment.unplaced
        if unplaced and assignment.settled and not authoritative:
            if context.snapshot.statuses(pending.resolved_ids(), pending.file_episode_map).all_done():
                # A hand import placed what no poll could: every intended episode holds a recommended file.
                self.logger.debug(f"{content_path}: already imported (recommended files present)")
                probe = ImportProbe.imported()
            else:
                # Nothing to verify or import, and the same files come back next poll: an outcome, not a wait.
                # The skips name what a hand import must place (every one an on-disk key, since the mapper
                # judges only what it read from the scan). Only exclusions fall back to naming those.
                unmatched = assignment.skipped or unplaced
                names = tuple(path_leaf(context.candidates_by_basename[name].path) for name in unmatched)
                probe = ImportProbe.unmatched(names)
        else:
            if unplaced and not assignment.settled and not context.at_deadline:
                # A parse or episode-index miss may be hiding the match, so the skip is no verdict yet. Ask again next poll.
                self.logger.debug(
                    f"{pending.display_label}: {count_noun(len(unplaced), 'file')} unplaced behind a missing input"
                )
            elif skips := self._reportable_skips(pending, assignment.skipped):
                # From the deadline on a miss that never healed is reported like any other skip.
                self._warn_unplaceable_files(pending, skips)
            if overridden := assignment.overridden:
                self._warn_overridden_files(pending, overridden)
            for name, ids in {**placed, **recovered}.items():
                self.logger.debug(f"{pending.display_label}: placed {name} -> {ids}")
            probe = self._verify_or_import(context, authoritative)
        # The mapper's placements and exclusions are kept whatever the branch (the file was on disk this poll).
        # History's are kept only once they verified, so a row that never verifies is re-derived next run, not pinned.
        placements = {**placed, **recovered} if probe.files_present else placed
        return replace(probe, placements=placements, exclusions=tuple(p.name for p in excluded))

    def _verify_or_import(self, context: _ImportContext, authoritative_map: FileEpisodeMap) -> ImportProbe:
        """Verify the mapped files, or plan and POST the import for the ones still needed."""

        pending = context.pending
        content_path = context.content_path
        # Done-check against the intended set: the seeded map plus what the mapper placed this poll, or what
        # Sonarr's history says it imported when no scan saw the files. `all_done()` needs a non-empty `by_id`.
        target_ids = sorted({i for ids in authoritative_map.values() for i in ids})
        statuses = context.snapshot.statuses(target_ids, authoritative_map)
        if statuses.all_done():
            self.logger.debug(f"{content_path}: already imported (recommended files present)")
            return ImportProbe.imported()

        if not authoritative_map:
            if context.scan.dead_tracked_empty:
                self._note_dead_tracked_empty(context)
            self.logger.debug(f"{content_path}: no mappable files for {pending.display_label} yet")
            return ImportProbe.waiting()

        decisions = plan_import_files(authoritative_map, context.candidates_by_basename, statuses)

        files: list[ManualImportFile] = []
        missing: list[str] = []
        unrouted: list[str] = []
        for decision in decisions:
            match decision.action:
                case ImportAction.MISSING:
                    missing.append(decision.basename)
                case ImportAction.IMPORT:
                    series_id = context.posting_series(decision.episode_ids[0]) if decision.episode_ids else None
                    if series_id is None:
                        unrouted.append(decision.basename)
                    else:
                        files.append(self._build_file_entry(decision, context, series_id))
                case _:
                    self.logger.debug(f"{decision.action.name}: {decision.basename}")

        if unrouted:
            # No claim's window and no series index places the file: it waits, and never verifies as imported.
            message = (
                f"{content_path}: {count_noun(len(unrouted), 'intended file')} resolve to no series "
                f"{pending.display_label} claims ({', '.join(unrouted)}) - will retry"
            )
            if context.at_deadline:
                hub_warn(message)
            else:
                self.logger.debug(message)

        if missing:
            # Absent files are expected on an early poll, so only the deadline attempt warns.
            reason = ""
            if context.scan.dead_tracked_empty:
                folder = self.scanner.content_paths(content_path).sonarr_visible
                reason = f" (it won't serve the download by id and a scan of {folder} found no files)"
            message = (
                f"{content_path}: {count_noun(len(missing), 'intended file')} "
                f"not visible to Sonarr for {pending.display_label}{reason} - will retry"
            )
            if context.at_deadline:
                hub_warn(message)
            else:
                self.logger.debug(message)

        if not files:
            if missing or unrouted:
                return ImportProbe.waiting()
            return ImportProbe.imported()

        import_mode = self._config.imports.mode
        if context.scan.omit_download_id and import_mode == "auto":
            # Untracked Execute with Auto resolves to MOVE (no DownloadClientItem to report CanMoveFiles),
            # ripping files from the seeding torrent. An explicitly configured move/copy is honored as set.
            import_mode = "copy"
        if blocked := self._guard_before_post(context):
            return blocked
        # One command for every series the record spans: Sonarr's ManualImportCommand resolves the series per file.
        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=import_mode,
        )
        if cmd_id is None:
            self.logger.debug(f"{content_path}: Sonarr rejected the import command; will retry")
            return ImportProbe.waiting()

        # Sonarr's copy is async, so acceptance is not `files_present`. Our fresh copy is a credited wait.
        self._scratch.issued_command_ids.add(cmd_id)
        self.logger.debug(f"{content_path}: queued {count_noun(len(files), 'file')} for import (command {cmd_id})")
        return ImportProbe.waiting(command_issued=True, deferral=Deferral.ISSUED)

    def _guard_before_post(self, context: _ImportContext) -> ImportProbe | None:
        """The wait when the command list read now blocks the POST, else None."""

        commands = self.list_commands()
        if commands is None:
            self.logger.debug(f"{context.pending.display_label}: command list read failed; retrying")
            return ImportProbe.waiting()
        block = self.command_block(commands, context.pending, context.content_path)
        if block is None:
            if sonarr_restarted(commands):
                block = CommandBlock.SONARR_RESTARTED
            elif disk_commands_changed_since(context.baseline, commands):
                block = CommandBlock.DISK_COMMAND_CHANGED
            if block is not None:
                self.logger.debug(f"{context.pending.display_label}: {block.value}; waiting")
        return None if block is None else block.waiting()

    def _note_dead_tracked_empty(self, context: _ImportContext) -> None:
        """Surface a dead-tracked, empty-folder poll for an unmapped record.

        Debug until the deadline, then one warning a run.
        """

        pending = context.pending
        # Read-only over the memo the fallback filled on this same poll, never a fetch.
        folder = self.scanner.content_paths(context.content_path).sonarr_visible
        # An empty folder right after completion is expected (Sonarr's mount lags), so only the deadline
        # attempt says so, once a run: dead-tracked plus still empty means silent retries until expiry.
        message = (
            f"{pending.display_label}: Sonarr won't serve this download by id and "
            f"a scan of its folder found no files ({folder}) - will retry"
        )
        if context.at_deadline and pending.infohash not in self._scratch.warned_empty_folder:
            self._scratch.warned_empty_folder.add(pending.infohash)
            hub_warn(message)
        else:
            self.logger.debug(message)

    def _reportable_skips(self, pending: PendingImport, skipped: tuple[str, ...]) -> tuple[str, ...]:
        """The skipped names no recorded exclusion accounts for.

        A file excluded at grab time or on an earlier poll (a sibling's slice, a refused duplicate) was
        reported then and stays quiet.
        """

        excluded = {normalized_leaf(name) for name in pending.excluded_files}
        reportable = tuple(name for name in skipped if name not in excluded)
        if len(reportable) < len(skipped):
            self.logger.debug(
                f"{pending.display_label}: {count_noun(len(skipped) - len(reportable), 'file')} "
                "excluded on record, not reported"
            )
        return reportable

    def _warn_unplaceable_files(self, pending: PendingImport, reportable: tuple[str, ...]) -> None:
        """Warn (once a run per download) about on-disk files we couldn't place.

        We import what we can and leave the rest, surfaced loudly so nothing is silently dropped.
        """

        if pending.infohash in self._scratch.warned_unplaceable:
            return
        self._scratch.warned_unplaceable.add(pending.infohash)
        hub_warn(
            f"{pending.display_label}: {count_noun(len(reportable), 'file')} could not be matched "
            f"to an episode and {pluralize(len(reportable), 'was', 'were')} not imported"
        )

    def _warn_overridden_files(self, pending: PendingImport, names: tuple[str, ...]) -> None:
        """Warn once per run per download about files placed by size over their names."""

        if pending.infohash in self._scratch.warned_overridden:
            return
        self._scratch.warned_overridden.add(pending.infohash)
        hub_warn(
            f"{pending.display_label}: {', '.join(names)} placed by file size instead of by "
            f"{pluralize(len(names), 'its name', 'their names')}"
        )

    def _import_language_objects(self, pending: PendingImport) -> list[Language]:
        """Resolve the import language objects for a record (lazily cached)."""

        if self._scratch.languages is None:
            self._scratch.languages = self.sonarr.languages()
        lang_names = derive_languages(
            pending.is_dual_audio,
            self._config.imports.languages_dual,
            self._config.imports.languages_single,
        )
        return resolve_language_objects(lang_names, self._scratch.languages)

    def _quality_definitions(self) -> list[QualityDefinition]:
        """The Sonarr quality definitions (lazily fetched + cached for the run)."""

        if self._scratch.quality_defs is None:
            self._scratch.quality_defs = self.sonarr.quality_definitions()
        return self._scratch.quality_defs

    def _build_file_entry(
        self,
        decision: ImportDecision,
        context: _ImportContext,
        series_id: int,
    ) -> ManualImportFile:
        """Build one ManualImport file payload, always with a real quality (an omitted key NREs in Sonarr)."""

        pending = context.pending
        content_path = context.content_path
        lang_objs = self._import_language_objects(pending)
        quality_defs = self._quality_definitions()

        path = decision.path or decision.basename
        base = path_leaf(path)
        sonarr_axes = quality_axes_from_model(decision.quality)
        our_axes = parse_quality_from_filename(base)
        default_name = self._config.imports.default_quality
        default_axes = quality_axes_from_name(default_name, quality_defs)
        # The quality seam runs once per FILE, so the flag keeps a config typo to one warning per run.
        if default_name and default_axes == ParsedQuality() and not self._scratch.warned_default_quality:
            self._scratch.warned_default_quality = True
            hub_warn(f"imports.default_quality '{default_name}' matches no Sonarr quality definition - ignoring it")
        quality = resolve_quality(
            sonarr_axes,
            our_axes,
            default_axes,
            quality_defs,
            decision.quality,
        )
        # A resolved-but-source-less quality is the synthesized Unknown (an empty nested quality already folded
        # to None at the parse boundary).
        resolved = quality.quality
        if resolved is None or QualitySource.parse(resolved.source) is None:
            hub_warn(
                f"{content_path}: could not confidently resolve quality for {base} - importing as Unknown "
                "(re-grab risk)"
            )
        entry = ManualImportFile(
            path=path,
            seriesId=series_id,
            episodeIds=list(decision.episode_ids),
            releaseGroup=pending.release_group,
            languages=lang_objs,
            quality=quality,
        )
        if context.scan.omit_download_id:
            # Left UNSET so the exclude_unset dump omits the key entirely (never null) and Execute takes the
            # untracked branch. The tracked tail dereferences the poisoned download's null ImportItem.
            return entry
        # model_copy marks the key set so it reaches the wire. Uppercased because Sonarr matches its tracked
        # downloads case-sensitively, and an unmatched import leaves the queue record open to a re-import.
        return entry.model_copy(update={"downloadId": pending.infohash.upper()})


@dataclass(frozen=True, slots=True)
class _SeedStatuses:
    """A record snapshot, the per-target file statuses over the seed set, and the bar counts."""

    snapshot: RecordSnapshot
    """The snapshot of every series the record spans that the statuses were judged over."""
    statuses: TargetStatuses
    """The file status of every seeded target under its claim's guards."""
    done: int
    """Recommended files present, net of the preowned targets."""
    total: int
    """Seeded targets, net of the preowned ones."""

    def with_counts(self, probe: ImportProbe) -> ImportProbe:
        """`probe` carrying this seed's bar counts."""

        return replace(probe, imported_count=self.done, target_count=self.total)


@dataclass(frozen=True, slots=True)
class _SonarrView:
    """Sonarr's view of a download this poll: its queue verdict and the command list read with it."""

    verdict: QueueVerdict
    """The queue verdict over the download's rows."""
    commands: list[CommandResource]
    """The command list read just before the queue."""
    untracked: bool
    """Whether the download has no queue row and Sonarr lists a completed rescan from after its add."""

    @property
    def needs_rescan(self) -> bool:
        """A download with no row or a row in motion, without the untracked proof."""

        return self.verdict in (QueueVerdict.WAIT, QueueVerdict.NOT_IN_QUEUE) and not self.untracked


@dataclass(frozen=True, slots=True)
class _Poll:
    """One import poll's inputs and its seed set."""

    pending: PendingImport
    """The record this poll imports."""
    content_path: str
    """The download's content path as qBittorrent reports it."""
    at_deadline: bool
    """Whether this is the readiness-deadline attempt."""
    targets: list[int]
    """The seeded target ids when the seed accounts for the record, else empty."""
    complete: bool
    """Whether the seed map covers every intended file."""

    @classmethod
    def of(cls, pending: PendingImport, content_path: str, attempt: AttemptKind) -> "_Poll":
        """The poll record for `pending`, its seed set gated by the record's coverage."""

        targets = pending.target_ids()
        coverage = pending.seed_coverage()
        accounted = bool(targets) and coverage.accounted
        return cls(
            pending, content_path, attempt.at_deadline, targets if accounted else [], accounted and coverage.mapped
        )

    @property
    def label(self) -> str:
        """The record's display label."""

        return self.pending.display_label


class ImportReconciler:
    """Decides a completed download's state and builds the grab-time seeds."""

    def __init__(self, records: PendingRecords, episodes: SonarrEpisodes, executor: ImportExecutor) -> None:
        """Bind the run's record seam and the composed collaborators (the logger is the executor's)."""

        self._records = records
        self._episodes = episodes
        self._executor = executor
        self.logger = executor.logger

    def import_completed(self, pending: PendingImport, content_path: str, attempt: AttemptKind) -> ImportProbe:
        """One import poll. Episode files and Sonarr's queue are the truth, never the cache."""

        poll = _Poll.of(pending, content_path, attempt)
        # Bar counts over the seed set. An unaccounted record reports 0/0 (indeterminate).
        fresh = self._read_seed(poll.pending, poll.targets)
        if fresh is not None and (imported := self._already_imported(poll, fresh)):
            return imported
        seed = fresh if fresh is not None else self._last_known_seed(poll.pending, poll.targets)

        view = self._sonarr_view(poll)
        if view is None:
            # Fail closed: an unreadable read waits, never reads as idle or untracked.
            self.logger.debug(f"{poll.label}: Sonarr read failed; retrying")
            return seed.with_counts(ImportProbe.waiting())
        if view.verdict is QueueVerdict.WAIT:
            self.logger.debug(f"{poll.label}: Sonarr still has it in motion; waiting")
            return seed.with_counts(ImportProbe.waiting())
        if view.verdict is QueueVerdict.IMPORTING:
            self.logger.debug(f"{poll.label}: Sonarr is importing it; waiting")
            return seed.with_counts(ImportProbe.waiting(deferral=Deferral.IMPORT))

        # The guard holds every attempt kind, the deadline one included.
        if block := self._executor.command_block(view.commands, poll.pending, poll.content_path):
            return seed.with_counts(block.waiting())
        if (
            view.verdict is QueueVerdict.PENDING_CLEAN
            and not poll.at_deadline
            and self._executor.completed_download_handling_enabled()
        ):
            self.logger.debug(f"{poll.label}: Sonarr has it pending and will import it; waiting")
            return seed.with_counts(ImportProbe.waiting())

        taken = self._executor.take_kept_scan(poll.pending.infohash, poll.content_path)
        kept = None if view.verdict is QueueVerdict.NOT_IN_QUEUE else taken
        scan = (
            kept
            if kept is not None
            else self._executor.scanner.scan(poll.pending, poll.content_path, untracked=view.untracked)
        )
        if scan is None:
            return seed.with_counts(ImportProbe.waiting())
        probe = self._step_in(poll, scan, seed)
        if kept is None and probe.deferral is Deferral.BUSY:
            self._executor.keep_scan(poll.pending.infohash, poll.content_path, scan)
        return probe

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Read-only "files inserted" count: this path can promote, so it never trusts exclusions."""

        seeded_targets = pending.target_ids()
        if not seeded_targets or not pending.seed_coverage().mapped:
            return NO_PROGRESS
        seed = self._read_seed(pending, seeded_targets)
        return NO_PROGRESS if seed is None else ImportProgress(seed.done, seed.total, determinate=True)

    def _sonarr_view(self, poll: _Poll) -> _SonarrView | None:
        """Sonarr's view this poll, read again after the rescan a lagging view may get. None when unreadable."""

        view = self._read_sonarr(poll.pending)
        if view is None or not view.needs_rescan:
            return view
        if sonarr_restarted(view.commands):
            self._executor.forget_rescans()
        if self._executor.rescan_once(poll.pending.infohash):
            view = self._read_sonarr(poll.pending)
        return view

    def _read_sonarr(self, pending: PendingImport) -> _SonarrView | None:
        """One read of the command list, then the queue. None when either read failed."""

        commands = self._executor.list_commands()
        if commands is None:
            return None
        rows = self._executor.queue_records(pending.infohash)
        if rows is None:
            return None
        verdict = classify_queue(rows)
        untracked = verdict is QueueVerdict.NOT_IN_QUEUE and rescanned_since_add(commands, pending.added_at)
        return _SonarrView(verdict, commands, untracked)

    def _step_in(self, poll: _Poll, scan: _CandidateScan, seed: _SeedStatuses) -> ImportProbe:
        """The poll past the scan: the guard baseline, the files read again, the hand-off to the import."""

        baseline = self._executor.list_commands()
        if baseline is None:
            self.logger.debug(f"{poll.label}: command list read failed; retrying")
            return seed.with_counts(ImportProbe.waiting())
        fresh = self._read_seed(poll.pending, poll.targets)
        if fresh is None:
            self.logger.debug(f"{poll.label}: episode read failed; retrying")
            return seed.with_counts(ImportProbe.waiting())
        if imported := self._already_imported(poll, fresh):
            return imported
        if block := self._executor.command_block(baseline, poll.pending, poll.content_path):
            return fresh.with_counts(block.waiting())
        reading = PollReading(scan, fresh.snapshot, baseline, at_deadline=poll.at_deadline)
        return fresh.with_counts(self._executor.run_manual_import(poll.pending, poll.content_path, reading))

    def _already_imported(self, poll: _Poll, seed: _SeedStatuses) -> ImportProbe | None:
        """The IMPORTED probe when the seed map is whole and every recommended file is present, else None."""

        if not poll.complete or not seed.statuses.all_done():
            return None
        self.logger.debug(f"{poll.label}: already imported (recommended files present)")
        return seed.with_counts(ImportProbe.imported())

    def _read_seed(self, pending: PendingImport, targets: list[int]) -> _SeedStatuses | None:
        """The seed from each claimed series' episodes read now. None when any read failed."""

        lists: dict[int, list[SonarrEpisode]] = {}
        for series_id in pending.series_ids:
            ep_list = self._episodes.read_episodes(series_id)
            if ep_list is None:
                return None
            lists[series_id] = ep_list
        return self._seed_from(pending, targets, lists)

    def _last_known_seed(self, pending: PendingImport, targets: list[int]) -> _SeedStatuses:
        """The seed from each claimed series' last successfully read episodes."""

        lists = {sid: self._episodes.last_known_episodes(sid) for sid in pending.series_ids}
        return self._seed_from(pending, targets, lists)

    def _seed_from(
        self, pending: PendingImport, targets: list[int], lists: dict[int, list[SonarrEpisode]]
    ) -> _SeedStatuses:
        """Classify `targets` under each claim's guards over the given episode lists, with the bar counts."""

        guards = self._records.guards()
        votes: dict[int, GroupVotes] = {}
        by_series: dict[int, EpisodeSnapshot] = {}
        for series_id in pending.series_ids:
            index = episode_index(lists[series_id])
            # Unfiltered: a cleanup-flagged sibling's files are on disk, so dropping it would loosen the guard.
            siblings = self._records.for_series(series_id, guards).values()
            votes[series_id] = GroupVotes(pending.own_group, tuple(record.own_group for record in siblings))
            by_series[series_id] = EpisodeSnapshot.guarded(index, pending.guards_for(series_id), votes[series_id])
        by_claim = {
            claim.al_id: EpisodeSnapshot.guarded(
                by_series[claim.series_id].episodes, claim.guards, votes[claim.series_id]
            )
            for claim in pending.claims
        }
        snapshot = RecordSnapshot(pending, by_series, by_claim)
        statuses = snapshot.statuses(targets, pending.file_episode_map)
        preowned = sum(1 for ep_id in targets if snapshot.preowned(ep_id))
        recommended = sum(1 for status in statuses.by_id.values() if status is EpisodeFileStatus.RECOMMENDED)
        return _SeedStatuses(snapshot, statuses, max(0, recommended - preowned), len(targets) - preowned)
