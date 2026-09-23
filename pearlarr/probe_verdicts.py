"""Pure verdicts over the import probes' Sonarr reads: queue, in-flight import, commands, history rows, paths."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

from .manual_import import Deferral, PendingImport, fold_path_separators, normalized_leaf
from .seadex_types import CommandResource, HistoryPage, QueueRecord, RemotePathMapping


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download THIS poll.

    Derived purely from the queue records sharing a `downloadId` (a season pack
    has one record per episode), reading `trackedDownloadState` plus (for
    pending records) `trackedDownloadStatus`. "Already imported" is NOT decided
    here (a successful import is removed from the queue). The caller reads the
    episode files.
    """

    WAIT = auto()
    """Something is in motion (downloading / queued / ...). Let Sonarr finish so we never race it. The wait
    is this record's own: Sonarr's view lags the finished torrent, so the clock keeps running."""

    IMPORTING = auto()
    """Sonarr is copying this download's files right now. Wait, and the wait is Sonarr's, not this record
    stalling (the monitor credits it)."""

    PENDING_CLEAN = auto()
    """A clean (status `ok`) `importPending` record: Sonarr parsed it and is about to import it. See
    `classify_queue` rule 3."""

    STEP_IN = auto()
    """Sonarr can't / won't progress it (`importBlocked` / `failed` / `failedPending` / `ignored`, or an
    `importPending` record flagged warning/error), or it isn't tracking the download at all (empty). Drive
    our authoritative manual import."""


# trackedDownloadState values (camelCase from Sonarr, compared case-folded) that
# mean Sonarr is genuinely working the download right now: wait rather than race
# it. `importing` is the copy itself (a credited wait). `queued`/`delay`/`paused`
# are QueueStatus-ish transients Sonarr may surface in the same field. Treat
# them as "still working" too.
_QUEUE_IMPORTING_STATES = frozenset({"importing"})
_QUEUE_IN_MOTION_STATES = _QUEUE_IMPORTING_STATES | {"downloading", "queued", "delay", "paused"}
_QUEUE_STEP_IN_STATES = frozenset(
    {"importblocked", "failed", "failedpending", "ignored"},
)
# trackedDownloadStatus values marking a pending record Sonarr tried and failed
# to import (e.g. the file wasn't visible on its mount yet). It does not
# reliably retry those, so deferring would just burn the readiness deadline.
_QUEUE_FLAGGED_STATUSES = frozenset({"warning", "error"})


def classify_queue(records: list[QueueRecord]) -> QueueVerdict:
    """Reduce a download's queue records to a single verdict for this poll.

    Side-effect free so the decision can be unit-tested without any HTTP.
    Priority, highest first:

      1. anything in motion, never raced: an `importing` record -> `IMPORTING`
         (Sonarr is copying its files, a credited wait), any other in-motion
         state (downloading / queued / ...) -> `WAIT` (re-evaluate next poll).
      2. any troubled record -> `STEP_IN`: `importBlocked` / `failed` /
         `failedPending` / `ignored`, or an `importPending` record whose
         `trackedDownloadStatus` is warning/error (Sonarr's own import attempt
         failed and it does not reliably retry, so it is Sonarr's flag for
         "handle this yourself").
      3. a CLEAN `importPending` (status `ok`, or none reported) ->
         `PENDING_CLEAN`: Sonarr is about to import it, so let it settle rather
         than step in. Stepping in would race Sonarr's own import.
      4. otherwise (empty because Sonarr isn't tracking it, all `imported`, or an
         unknown state) -> `STEP_IN`.

    `records` are every queue record matching the download (matched by the
    caller). A record with no tracked state contributes nothing. The verdict is
    the action this poll, BEFORE the episode-file "already imported" check the
    caller layers on top.
    """

    importing = False
    in_motion = False
    troubled = False
    clean_pending = False
    for record in records:
        state = (record.state or "").casefold()
        if state in _QUEUE_IMPORTING_STATES:
            importing = True
        elif state in _QUEUE_IN_MOTION_STATES:
            in_motion = True
        elif state in _QUEUE_STEP_IN_STATES:
            troubled = True
        elif state == "importpending":
            # Invariant: only a CLEAN importPending defers to Sonarr. A
            # warning/error-flagged one is a failed Sonarr import attempt it
            # won't reliably retry, so waiting would burn the readiness deadline.
            if (record.status or "").casefold() in _QUEUE_FLAGGED_STATUSES:
                troubled = True
            else:
                clean_pending = True

    if importing:
        return QueueVerdict.IMPORTING
    if in_motion:
        return QueueVerdict.WAIT
    if troubled:
        return QueueVerdict.STEP_IN
    if clean_pending:
        return QueueVerdict.PENDING_CLEAN
    return QueueVerdict.STEP_IN


# A command counts as a ManualImport only under this name (Sonarr's command
# `name`, compared case-folded), and only these statuses mean it is still
# running. A terminal command (completed / failed / aborted / cancelled /
# orphaned) is no longer in flight, so it never wedges a re-import.
_MANUAL_IMPORT_COMMAND_NAME = "manualimport"
_COMMAND_IN_FLIGHT_STATES = frozenset({"queued", "started"})


def _norm_path(path: str) -> str:
    """Normalize a path for a pure (no-disk) prefix compare: separators folded, casefolded."""

    return fold_path_separators(path).casefold()


class ContentPaths(NamedTuple):
    """One download's import folder in both filesystem views the guard must match.

    A dead-tracked folder import POSTs the TRANSLATED (Sonarr-visible) path, so
    a command read back carries that form while the durable record carries the
    raw qBittorrent one: the guard needs both prefixes.
    """

    raw: str
    """The qBittorrent `content_path`."""

    sonarr_visible: str
    """The remote-path-mapped view (equal to `raw` when no translation was
    computed or none applies)."""


class InFlightImport(NamedTuple):
    """The in-flight ManualImport `manual_import_in_flight` matched, plus HOW.

    Only a provable (`by_download_id`) or own-issued match may be credited back
    to the ready deadline. An unproven one stays a plain deadline-bounded wait.
    """

    command: CommandResource
    """The matched, still-running ManualImport."""

    by_download_id: bool
    """True for the primary `download_id` match. False for the path/episode fallback."""


class DownloadMatch(NamedTuple):
    """The identity facets a running command is matched against for one download."""

    infohash: str
    """The durable download id (survives restarts)."""

    content_paths: ContentPaths
    """The import folder in both filesystem views, for the no-download-id fallback."""

    target_ep_ids: set[int]
    """Our intended episode ids, for the same fallback."""


def manual_import_in_flight(commands: list[CommandResource], match: DownloadMatch) -> InFlightImport | None:
    """The queued/started ManualImport covering THIS download, or None.

    Pure (mirrors `classify_queue`). Sonarr copies asynchronously and drops the
    torrent from the queue meanwhile, so the queue alone reads "empty -> step
    in" and we'd stack a duplicate every poll. Matching the durable infohash
    (case-insensitive) closes that loop and survives a process restart. A
    no-download-id folder import falls back to a `content_paths`-prefix or
    `target_ep_ids` overlap, deliberately broad, since a false positive only
    makes us wait (deadline-bounded, never credited) while a miss re-opens the
    duplicate-import loop.
    """

    target_hash = match.infohash.casefold()
    paths = match.content_paths
    target_ep_ids = match.target_ep_ids
    content_prefixes = {_norm_path(paths.raw), _norm_path(paths.sonarr_visible)}
    for command in commands:
        name = (command.name or "").casefold()
        status = (command.status or "").casefold()
        if name != _MANUAL_IMPORT_COMMAND_NAME or status not in _COMMAND_IN_FLIGHT_STATES:
            continue
        file_hashes = {f.download_id.casefold() for f in command.files if f.download_id is not None}
        if target_hash in file_hashes:
            return InFlightImport(command, by_download_id=True)
        # Fallback only for a command whose files carry no download id at all (a
        # folder / season-pack import). A command that DOES carry download ids but
        # for a different torrent must not be swept up by a path/episode overlap.
        if file_hashes:
            continue
        for file in command.files:
            if file.path is not None and any(_norm_path(file.path).startswith(p) for p in content_prefixes):
                return InFlightImport(command, by_download_id=False)
            if any(ep_id in target_ep_ids for ep_id in file.episode_ids):
                return InFlightImport(command, by_download_id=False)
    return None


# The monitored-download pass a completed RefreshMonitoredDownloads immediately
# starts, the one the rescan settles (see `sonarr_process_pass_running`).
_SONARR_PROCESS_PASS_NAMES = frozenset({"processmonitoreddownloads"})

# Sonarr's disk-access commands (its RequiresDiskAccess scheduling class),
# compared case-folded like the command names above: the import passes
# (completed-download handling, the legacy folder scan), the rename/move/delete
# sweeps, and ManualImport itself. A started one blocks every queued one.
_SONARR_DISK_COMMAND_NAMES = _SONARR_PROCESS_PASS_NAMES | {
    "downloadedepisodesscan",
    "manualimport",
    "renamefiles",
    "renameseries",
    "moveseries",
    "bulkmoveseries",
    "deleteseriesfiles",
}


def started_disk_commands(commands: list[CommandResource]) -> list[CommandResource]:
    """The Sonarr disk-access commands executing right now.

    A ManualImport POSTed while one is `started` queues behind it and replays
    minutes stale, re-copying files an intervening pass already placed. Only
    `started` blocks: a queued pass is near-permanently present during a wait,
    so blocking on it would starve the step-in entirely. The list form lets
    `classify_commands` tell an own running command from a foreign one.
    """

    return _started(commands, _SONARR_DISK_COMMAND_NAMES)


class CommandBlock(Enum):
    """Why the running-command snapshot blocks a step-in this poll (values = log phrases).

    Every block is a credited wait (the monitor pauses the ready clock). Only
    `OWN_IMPORT` also claims the import as this record's.
    """

    OWN_IMPORT = "our ManualImport is in flight"
    IN_FLIGHT_IMPORT = "a ManualImport is already in flight"
    DISK_COMMAND = "Sonarr is running a disk command"

    @property
    def claims_import(self) -> bool:
        """Whether the in-flight import is provably ours (the probe reads `command_issued`)."""

        return self is CommandBlock.OWN_IMPORT

    @property
    def deferral(self) -> Deferral:
        """The wait's reason for the probe: an import in flight, or a busy Sonarr."""

        return Deferral.BUSY if self is CommandBlock.DISK_COMMAND else Deferral.IMPORT


def classify_commands(
    commands: list[CommandResource],
    match: DownloadMatch,
    is_own: Callable[[int], bool],
) -> CommandBlock | None:
    """Reduce one `/api/v3/command` snapshot to the block holding a step-in, or None when clear.

    Pure (mirrors `classify_queue`). `is_own` is the executor's this-run
    issued-id memory. An in-flight import covering this download outranks the
    broad disk guard, and it is ours by the download-id match (survives
    restarts) or an issued id. An unproven or foreign command still blocks
    (never race it) and the wait is credited either way, so ownership only
    decides whether the probe claims the import.
    """

    in_flight = manual_import_in_flight(commands, match)
    if in_flight:
        owned = in_flight.by_download_id or is_own(in_flight.command.id)
        return CommandBlock.OWN_IMPORT if owned else CommandBlock.IN_FLIGHT_IMPORT
    if started_disk_commands(commands):
        return CommandBlock.DISK_COMMAND
    return None


def sonarr_process_pass_running(commands: list[CommandResource]) -> bool:
    """Whether Sonarr's monitored-download pass is executing right now.

    A completed RefreshMonitoredDownloads immediately starts one, so the rescan
    settles it briefly (see `ImportExecutor.refresh_downloads`). Checked apart
    from the broader disk-command set because only this pass is self-inflicted
    every poll.
    """

    return bool(_started(commands, _SONARR_PROCESS_PASS_NAMES))


def _started(commands: list[CommandResource], names: frozenset[str]) -> list[CommandResource]:
    """The named commands currently `started`, casefolded on both axes."""

    return [
        command
        for command in commands
        if (command.name or "").casefold() in names and (command.status or "").casefold() == "started"
    ]


# The episode-history events that map a re-appeared download to a queue-hidden
# tracked state (Imported / Failed / Ignored), states Sonarr never runs its
# completed-download Check on, so `manualimport?downloadId=` NREs (HTTP 500)
# forever. Keyed by casefolded eventType, valued by the human label the hub
# note renders. `grabbed` (or none of the four) means genuinely Downloading.
_DEAD_TRACKED_HISTORY_EVENTS = {
    "downloadfolderimported": "imported",
    "downloadfailed": "failed",
    "downloadignored": "ignored",
}
_GRABBED_HISTORY_EVENT = "grabbed"
_IMPORTED_HISTORY_EVENT = "downloadfolderimported"


class HistoryImport(NamedTuple):
    """One file Sonarr's history says it imported for a download, as one row per episode."""

    path: str
    """The imported file's download-folder path, `data.droppedPath`."""
    episode_id: int
    """The episode the row landed on."""
    series_id: int
    """The row's `seriesId`, so a record never claims another series' import."""


@dataclass(frozen=True, slots=True)
class DownloadHistoryVerdict:
    """What a download's newest relevant Sonarr history event says about its state.

    An imported verdict also carries the rows of that import.
    """

    dead_tracked: bool
    """True when Sonarr's history maps the download to a queue-hidden state it
    will never serve by id. Import from its folder instead."""

    event: str | None = None
    """The dead-tracked event label (`imported`/`failed`/`ignored`), for the
    fallback's debug note. None when clean."""

    date: str | None = None
    """The dead-tracked event's raw ISO date, for the fallback's debug note."""

    import_rows: tuple[HistoryImport, ...] = ()
    """The rows of the import that dead-tracked the download, one per file and episode, for
    rebuilding a never-scanned record's map. Empty for a failed or ignored verdict, for a clean
    one, and when the page cannot bound the import."""


def classify_download_history(page: HistoryPage) -> DownloadHistoryVerdict:
    """Classify a download's tracked state from its newest relevant history event.

    The episode-history mirror of Sonarr's own `GetStateFromHistory`
    latest-event rule: walk the page's records (page 1, date-DESCENDING, as
    `history_for_download` returns them) and decide on the first event among
    `grabbed` / `downloadFolderImported` / `downloadFailed` /
    `downloadIgnored`, skipping every other type (e.g. `episodeFileDeleted`).
    Imported/failed/ignored -> dead-tracked. `grabbed` -> clean (a hash Sonarr
    itself re-grabbed after an old failure is genuinely Downloading). None of
    the four found -> clean. Probing only for prior imports would misroute
    re-grabs of previously-FAILED/IGNORED hashes: the same NREs apply there.
    An imported verdict carries its cycle's import rows.
    """

    for index, record in enumerate(page.records):
        event = record.event_type.casefold()
        if event == _GRABBED_HISTORY_EVENT:
            return DownloadHistoryVerdict(dead_tracked=False)
        label = _DEAD_TRACKED_HISTORY_EVENTS.get(event)
        if label is not None:
            rows = _import_rows(page, index) if event == _IMPORTED_HISTORY_EVENT else ()
            return DownloadHistoryVerdict(dead_tracked=True, event=label, date=record.date, import_rows=rows)
    return DownloadHistoryVerdict(dead_tracked=False)


def _import_rows(page: HistoryPage, start: int) -> tuple[HistoryImport, ...]:
    """The rows of the import cycle starting at `start`, or none when the page cannot bound it.

    Every `downloadFolderImported` row down to the next `grabbed`, skipping rows with no path or
    episode id. Empty when the page ends before that grab and is cut, since the cycle may go on.
    """

    rows: list[HistoryImport] = []
    for record in page.records[start:]:
        event = record.event_type.casefold()
        if event == _GRABBED_HISTORY_EVENT:
            return tuple(rows)
        if event == _IMPORTED_HISTORY_EVENT and record.dropped_path and record.episode_id:
            rows.append(HistoryImport(record.dropped_path, record.episode_id, record.item_id))
    if page.total_records > len(page.records):
        return ()
    return tuple(rows)


def placements_from_history(imports: Sequence[HistoryImport], pending: PendingImport) -> dict[str, list[int]]:
    """Sonarr's import rows as placements for the record's unplaced listing files, normalized leaf -> sorted ids.

    Our seed stays authoritative, and a sibling's slice never lands on this record: by name
    (`excluded_files`) and by id (the resolved set, when the record carries one).
    """

    wanted = pending.unplaced_names()
    scope = set(pending.ordered_episode_ids)
    grouped: dict[str, set[int]] = {}
    for row in imports:
        if row.series_id != pending.series_id:
            continue
        name = normalized_leaf(row.path)
        if name not in wanted or (scope and row.episode_id not in scope):
            continue
        grouped.setdefault(name, set()).add(row.episode_id)
    return {name: sorted(ids) for name, ids in grouped.items()}


def _path_segments(path: str) -> list[str]:
    """Split a path into non-empty segments for the boundary-aware compare (separators folded)."""

    return [segment for segment in fold_path_separators(path).split("/") if segment]


def translate_download_path(
    content_path: str,
    mappings: Sequence[RemotePathMapping],
    qbit_host: str | None,
) -> str:
    """Map a download-client path into Sonarr's filesystem view.

    Longest-`remotePath`-prefix match is the PRIMARY rule. Host equality (our
    `qbit_host`, folded here) only tiebreaks equally-long prefixes, and host
    inequality never excludes a mapping: Sonarr's `host` is the download-client
    host as SONARR configured it, routinely a different string from our
    qBittorrent host (localhost vs container name vs IP). Matching is per path
    segment, so it is separator-boundary-aware (`/downloads` never matches
    `/downloads-x/f`), tolerant of trailing slashes and Windows backslashes on
    either side, and case-insensitive, while the suffix keeps its ORIGINAL case
    (POSIX targets are case-sensitive). No match returns the path untranslated
    (the same-filesystem no-op). `content_path` is qBittorrent's, a folder or a
    single file.
    """

    content_segments = _path_segments(content_path)
    folded = [segment.casefold() for segment in content_segments]
    target_host = qbit_host.casefold() if qbit_host else None

    best_rank: tuple[int, bool] | None = None
    best_local = ""
    for mapping in mappings:
        if not mapping.remote_path or not mapping.local_path:
            continue
        remote = [segment.casefold() for segment in _path_segments(mapping.remote_path)]
        if not remote or folded[: len(remote)] != remote:
            continue
        host_matches = target_host is not None and (mapping.host or "").casefold() == target_host
        rank = (len(remote), host_matches)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_local = mapping.local_path

    if best_rank is None:
        return content_path
    base = best_local.rstrip("/\\") or "/"
    suffix = content_segments[best_rank[0] :]
    if not suffix:
        return base
    joined = "/".join(suffix)
    return f"/{joined}" if base == "/" else f"{base}/{joined}"
