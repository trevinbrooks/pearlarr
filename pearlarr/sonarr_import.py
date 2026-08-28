"""Import-time subsystem: decide a download's state, then build/POST the import."""

import logging
from dataclasses import dataclass, field, replace
from typing import NamedTuple
from urllib.parse import urlsplit

from .config import Arr
from .log import count_noun, pluralize
from .manual_import import (
    NO_PROGRESS,
    AttemptKind,
    ImportProbe,
    ImportProgress,
    ImportReadiness,
    PendingImport,
    path_leaf,
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
    QueueRecord,
    RemotePathMapping,
    SeadexDict,
    SonarrEpisode,
    flagged_urls,
)
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import SonarrEpisodes
from .sonarr_import_plan import (
    ContentPaths,
    DownloadHistoryVerdict,
    DownloadMatch,
    EpisodeFileStatus,
    EpisodeSnapshot,
    ImportAction,
    ImportDecision,
    ParsedQuality,
    PendingSeedContext,
    QueueVerdict,
    SeedFile,
    SeedRelease,
    TargetStatuses,
    build_pending_seed,
    classify_commands,
    classify_download_history,
    classify_queue,
    derive_languages,
    episode_index,
    parse_quality_from_filename,
    plan_import_files,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_language_objects,
    resolve_quality,
    sonarr_process_pass_running,
    translate_download_path,
    trusted_groups,
)
from .sonarr_mapper import FileEpisodeMapper
from .sonarr_parse import parsed_episodes, parsed_full_season, video_file_entries

# RefreshMonitoredDownloads is quick. Poll its status this many times (sleeping between) so the queue we read
# next reflects the rescan, then proceed regardless so a stuck command never blocks the run.
_REFRESH_COMMAND_MAX_POLLS = 30
_REFRESH_COMMAND_POLL_S = 1
_COMMAND_TERMINAL_STATES = frozenset({"completed", "failed", "aborted", "cancelled"})


def _hostname(url: str | None) -> str | None:
    """The bare hostname of a configured URL (tolerates a scheme-less `host:port`)."""

    if not url:
        return None
    return urlsplit(url).hostname or urlsplit(f"//{url}").hostname


class _CandidateScan(NamedTuple):
    """One poll's candidate fetch plus the entry shape it implies."""

    candidates: list[ManualImportCandidate]
    omit_download_id: bool
    """ALL entries must omit the downloadId: Sonarr's Execute tail NREs on the poisoned download."""


class _EntryContext(NamedTuple):
    """Per-command context for building file entries."""

    pending: PendingImport
    content_path: str
    omit_download_id: bool


@dataclass
class _ScanScratch:
    """The scanner's per-run fallback scratch, reassigned wholesale by `reset`.

    `path_mappings` None = unfetched, and `history_verdicts` memoizes VERDICTS
    only, so a probe failure re-probes on the next activation.
    """

    folder_pinned: set[str] = field(default_factory=set[str])
    history_verdicts: dict[str, DownloadHistoryVerdict] = field(default_factory=dict[str, DownloadHistoryVerdict])
    path_mappings: list[RemotePathMapping] | None = None
    translated_paths: dict[str, str] = field(default_factory=dict[str, str])
    warned_empty_folder: set[str] = field(default_factory=set[str])


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
        """Classify a download's Sonarr history (memoized), or None on probe failure."""

        verdict = self._scratch.history_verdicts.get(pending.infohash)
        if verdict is not None:
            return verdict
        page = self.sonarr.history_for_download(download_id=pending.infohash)
        if page is None:
            return None
        verdict = classify_download_history(page.records)
        self._scratch.history_verdicts[pending.infohash] = verdict
        if verdict.dead_tracked:
            when = f" on {verdict.date.split('T')[0]}" if verdict.date else ""
            self.logger.debug(
                f"{pending.display_label}: Sonarr recorded this download as {verdict.event}{when} "
                "and won't serve it by id - importing from its folder instead"
            )
        return verdict

    def scan(self, pending: PendingImport, content_path: str) -> _CandidateScan | None:
        """The candidates for one poll: the downloadId scan, with the folder fallback. None when the scan failed."""

        if pending.infohash not in self._scratch.folder_pinned:
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
            # INVARIANT: pin folder mode on NONEMPTY success only. A 200 `[]` is exactly what an invisible or
            # untranslated folder returns, and pinning on it wedges the record blind to files, downloadId unretried.
            self._scratch.folder_pinned.add(pending.infohash)
        elif verdict is not None and verdict.dead_tracked and pending.infohash not in self._scratch.warned_empty_folder:
            # Dead-tracked plus an empty folder means silent retries until the record expires. Say so once.
            self._scratch.warned_empty_folder.add(pending.infohash)
            hub_warn(
                f"{pending.display_label}: Sonarr won't serve this download by id and "
                f"a scan of its folder found no files ({folder}) - will retry"
            )
        return _CandidateScan(
            folder_candidates,
            omit_download_id=verdict is not None and verdict.dead_tracked,
        )


@dataclass
class _ImportScratch:
    """The executor's per-run import scratch, reassigned wholesale by `reset`. A None cache = not yet fetched."""

    quality_defs: list[QualityDefinition] | None = None
    languages: list[Language] | None = None
    warned_unplaceable: set[str] = field(default_factory=set[str])
    warned_default_quality: bool = False
    last_refresh_monotonic: float | None = None
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

    def is_own_command(self, command_id: int) -> bool:
        """Whether we POSTed this ManualImport command id this run."""

        return command_id in self._scratch.issued_command_ids

    def refresh_downloads(self) -> None:
        """Queue RefreshMonitoredDownloads (throttled), waiting for it and the follow-up ProcessMonitoredDownloads."""

        # The rescan is GLOBAL and the blocking pass walks several torrents back-to-back, so it is re-issued
        # at most once per poll interval.
        now = self._clock.now()
        interval = self._config.imports.poll_interval
        if self._scratch.last_refresh_monotonic is not None and now - self._scratch.last_refresh_monotonic < interval:
            return
        self._scratch.last_refresh_monotonic = now

        cmd_id = self.sonarr.refresh_monitored_downloads()
        if cmd_id is None:
            return
        self.logger.debug("Asked Sonarr to rescan its downloads")

        for _ in range(_REFRESH_COMMAND_MAX_POLLS):
            command = self.sonarr.command_status(cmd_id)
            state = command.status or ""
            if state.casefold() in _COMMAND_TERMINAL_STATES:
                break
            self._clock.sleep(_REFRESH_COMMAND_POLL_S)
        else:
            return

        for _ in range(_REFRESH_COMMAND_MAX_POLLS):
            if not sonarr_process_pass_running(self.list_commands()):
                return
            self._clock.sleep(_REFRESH_COMMAND_POLL_S)

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

    def close_tracked(self, pending: PendingImport) -> None:
        """Dismiss Sonarr's leftover queue entry, leaving an unknown-series one alone (dismissing it 500s)."""

        rows = self.queue_records(pending.infohash)
        if rows is None:
            self.logger.debug(f"{pending.display_label}: queue read failed - leaving any leftover entry")
            return
        queue_id = next((row.id for row in rows if row.id and row.series_id), None)
        if queue_id is None:
            if any(row.id for row in rows):
                self.logger.debug(f"{pending.display_label}: unknown-series queue entry - left for Sonarr to drop")
            else:
                self.logger.debug(f"{pending.display_label}: no leftover Sonarr queue entry to close")
            return
        if self.sonarr.queue_delete(queue_id):
            hub_note(f"Removed the imported download {pending.display_label} from Sonarr's queue")
        else:
            # The client's warning names no download (coalesced template). This one does.
            self.logger.debug(f"{pending.display_label}: queue entry {queue_id} not removed")

    def list_commands(self) -> list[CommandResource]:
        """The current Sonarr command list, never cached (an in-flight command's status changes)."""

        return self.sonarr.list_commands()

    def run_manual_import(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        snapshot: EpisodeSnapshot,
        at_deadline: bool = False,
    ) -> ImportProbe:
        """Import EXACTLY the files our map intends, never over an episode already holding a recommended file."""

        scan = self.scanner.scan(pending, content_path)
        if scan is None:
            # Transient (timeout or non-200). The folder-scan client already warned. Ask again.
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        candidates_by_basename = self._mapper.candidate_files(scan.candidates)
        assignment = self._mapper.assign(pending, candidates_by_basename, snapshot.episodes.id_by_key)
        if assignment.skipped:
            self._warn_unplaceable_files(pending, assignment.skipped)

        authoritative_map = assignment.assigned
        if not authoritative_map:
            self.logger.debug(f"{content_path}: no mappable files for {pending.display_label} yet")
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Done-check against the COMPLETE (repaired) intended set, derived from the on-disk files.
        target_ids = sorted({i for ids in authoritative_map.values() for i in ids})
        statuses = snapshot.statuses(target_ids)
        if statuses.all_done():
            self.logger.debug(f"{content_path}: already imported (recommended files present)")
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        decisions = plan_import_files(authoritative_map, candidates_by_basename, statuses.needing_import())

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
                    self.logger.debug(f"{decision.action.name}: {decision.basename}")

        if missing:
            # Absent files are expected on an early poll, so only the deadline attempt warns.
            message = (
                f"{content_path}: {count_noun(len(missing), 'intended file')} "
                f"not visible to Sonarr for {pending.display_label} - will retry"
            )
            if at_deadline:
                hub_warn(message)
            else:
                self.logger.debug(message)

        if not files:
            if missing:
                return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)
            return ImportProbe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        import_mode = self._config.imports.mode
        if scan.omit_download_id and import_mode == "auto":
            # Untracked Execute with Auto resolves to MOVE (no DownloadClientItem to report CanMoveFiles),
            # ripping files from the seeding torrent. An explicitly configured move/copy is honored as set.
            import_mode = "copy"
        cmd_id = self.sonarr.manual_import_execute(
            files=files,
            import_mode=import_mode,
        )
        if cmd_id is None:
            self.logger.debug(f"{content_path}: Sonarr rejected the import command; will retry")
            return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Sonarr's copy is async, so acceptance is not `files_present`.
        self._scratch.issued_command_ids.add(cmd_id)
        self.logger.debug(f"{content_path}: queued {count_noun(len(files), 'file')} for import (command {cmd_id})")
        return ImportProbe(ImportReadiness.RETRY, files_present=False, command_issued=True)

    def _warn_unplaceable_files(
        self,
        pending: PendingImport,
        unplaceable: list[str],
    ) -> None:
        """Warn (once a run per download) about on-disk files we couldn't place.

        We import what we can and leave the rest, surfaced loudly so nothing is silently dropped.
        """

        if pending.infohash in self._scratch.warned_unplaceable:
            return
        self._scratch.warned_unplaceable.add(pending.infohash)
        label = pending.display_label
        coverage = f" ({pending.coverage})" if pending.coverage else ""
        hub_warn(
            f"{label}{coverage}: {count_noun(len(unplaceable), 'file')} could not be matched "
            f"to an episode and {pluralize(len(unplaceable), 'was', 'were')} not imported"
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
        context: _EntryContext,
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
            seriesId=pending.series_id,
            episodeIds=decision.episode_ids,
            releaseGroup=pending.release_group,
            languages=lang_objs,
            quality=quality,
        )
        if context.omit_download_id:
            # Left UNSET so the exclude_unset dump omits the key entirely (never null) and Execute takes the
            # untracked branch. The tracked tail dereferences the poisoned download's null ImportItem.
            return entry
        # model_copy marks the key set so it reaches the wire. Uppercased because Sonarr matches its tracked
        # downloads case-sensitively, and an unmatched import leaves the queue record open to a re-import.
        return entry.model_copy(update={"downloadId": pending.infohash.upper()})


class _SeedStatuses(NamedTuple):
    """A same-poll episode snapshot plus the per-target file statuses pinned to the seed set."""

    snapshot: EpisodeSnapshot
    statuses: TargetStatuses


class ImportReconciler:
    """Decides a completed download's state and builds the grab-time seeds."""

    def __init__(self, deps: RunDeps, episodes: SonarrEpisodes, executor: ImportExecutor) -> None:
        """Bind the cache/logger off the deps + the composed collaborators."""

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
        """Build `infohash -> PendingImport` per grabbed torrent: a best-effort map that self-heals at import."""

        flagged = flagged_urls(seadex_dict)
        if not flagged:
            return {}

        # One index for the whole entry: its ordered ids ride every record, so import-time assignment maps files
        # into OUR set instead of re-deriving identity from Sonarr's title parse.
        index = episode_index(ep_list)

        pending_seeds: dict[str, PendingImport] = {}
        for srg, url_item, infohash in flagged:
            video_files = [base for _, base in video_file_entries(url_item.files)]
            if not video_files:
                continue
            release = SeedRelease(
                release_group=srg,
                url_item=url_item,
                infohash=infohash,
                files=tuple(self._seed_file(base) for base in video_files),
            )
            seed = build_pending_seed(release, index, entry)
            if seed.excluded_files:
                self.logger.debug(
                    f"{entry.title}: not counted toward completeness "
                    f"(other slice / duplicate): {', '.join(seed.excluded_files)}",
                )
            pending_seeds[infohash] = seed

        return pending_seeds

    def _seed_file(self, base: str) -> SeedFile:
        """One video file with its grab-time parse (staged SQLite writes are visible on the same connection)."""

        record = self.cache_store.get_sonarr_parse(base)
        if not record:
            return SeedFile(base, episodes=None)
        return SeedFile(base, tuple(parsed_episodes(record)), parsed_full_season(record))

    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt: AttemptKind,
    ) -> ImportProbe:
        """One import poll. Episode files and Sonarr's refreshed queue are the truth, never the cache."""

        label = pending.display_label

        # Rescan first so the queue read below reflects the finished torrent.
        self._executor.refresh_downloads()

        # Bar counts, pinned to the seed set so the denominator never rescales mid-import. An unaccounted
        # record reports 0/0 (indeterminate): a partial seed must never show a misleading bar.
        seeded_targets = pending.target_ids()
        coverage = pending.seed_coverage()
        accounted = bool(seeded_targets) and coverage.accounted
        seed_complete = accounted and coverage.mapped
        gated_targets = seeded_targets if accounted else []
        seed = self._seed_statuses(pending, gated_targets)
        # The done-check below still reads the raw statuses (a preowned target is done, just not ours to claim).
        done, total = self._net_counts(pending, gated_targets, seed.statuses)

        def probe(
            readiness: ImportReadiness,
            *,
            files_present: bool,
            command_issued: bool,
            deferred: bool = False,
        ) -> ImportProbe:
            return ImportProbe(
                readiness,
                files_present=files_present,
                command_issued=command_issued,
                imported_count=done,
                target_count=total,
                deferred=deferred,
            )

        # Only a complete map makes the done-check trustworthy without a folder scan.
        if seed_complete and seed.statuses.all_done():
            self.logger.debug(f"{label}: already imported (recommended files present)")
            return probe(ImportReadiness.IMPORTED, files_present=True, command_issued=False)

        queue_rows = self._executor.queue_records(pending.infohash)
        if queue_rows is None:
            # Fail closed: an unreadable queue must never read as untracked (stepping in races Sonarr's
            # import). RETRY defers via the poll loop until the ready deadline graduates the record.
            self.logger.debug(f"{label}: queue read failed; retrying")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)
        verdict = classify_queue(queue_rows)
        if verdict is QueueVerdict.WAIT:
            self.logger.debug(f"{label}: Sonarr is importing; waiting")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)
        if (
            verdict is QueueVerdict.PENDING_CLEAN
            and not attempt.at_deadline
            and self._executor.completed_download_handling_enabled()
        ):
            self.logger.debug(f"{label}: Sonarr has it pending and will import it; waiting")
            return probe(ImportReadiness.RETRY, files_present=False, command_issued=False)

        # Never gated on the attempt kind: our own import in flight is never raced.
        verdict = classify_commands(
            self._executor.list_commands(),
            DownloadMatch(
                pending.infohash,
                self._executor.scanner.content_paths(content_path),
                set(seeded_targets),
            ),
            self._executor.is_own_command,
        )
        if verdict.block:
            self.logger.debug(f"{label}: {verdict.block.value}; waiting")
            return probe(
                ImportReadiness.RETRY,
                files_present=False,
                command_issued=verdict.command_issued,
                deferred=verdict.deferred,
            )

        result = self._executor.run_manual_import(
            pending,
            content_path,
            snapshot=seed.snapshot,
            at_deadline=attempt.at_deadline,
        )
        return replace(result, imported_count=done, target_count=total)

    def import_progress(self, pending: PendingImport) -> ImportProgress:
        """Read-only "files inserted" count: this path can promote, so it never trusts exclusions."""

        seeded_targets = pending.target_ids()
        if not seeded_targets or not pending.seed_coverage().mapped:
            return NO_PROGRESS
        seed = self._seed_statuses(pending, seeded_targets)
        done, total = self._net_counts(pending, seeded_targets, seed.statuses)
        return ImportProgress(done, total, determinate=True)

    def _seed_statuses(self, pending: PendingImport, targets: list[int]) -> _SeedStatuses:
        """Fetch the series' episodes FRESH and classify `targets` against them (`[]` still builds the snapshot)."""

        episodes = self._episodes.episodes_for_series(pending.series_id)
        snapshot = EpisodeSnapshot(
            episodes=episode_index(episodes),
            trusted=trusted_groups(pending, self._series_pending_records(pending.series_id)),
            owned_episode_sizes=pending.guards.owned_sizes,
        )
        return _SeedStatuses(snapshot, snapshot.statuses(targets))

    @staticmethod
    def _net_counts(
        pending: PendingImport,
        targets: list[int],
        statuses: TargetStatuses,
    ) -> tuple[int, int]:
        """The bar's `(done, total)`, net of the grab-time preowned targets (empty `targets` nets to 0/0)."""

        preowned = len(set(pending.preowned_episode_ids) & set(targets))
        recommended = sum(1 for status in statuses.by_id.values() if status is EpisodeFileStatus.RECOMMENDED)
        return max(0, recommended - preowned), len(targets) - preowned

    def _series_pending_records(self, series_id: int) -> list[PendingImport]:
        """The series' durable pending records (any release group), rehydrated.

        A fresh snapshot already filtered in SQL, so a record dropped earlier this run is absent.
        """

        guard_rows = self.cache_store.get_guards(Arr.SONARR)
        return [
            PendingImport.from_json(raw, guards=guard_rows.get(key.al_id))
            for key, raw in self.cache_store.get_pending_for_series(Arr.SONARR, series_id).items()
        ]
