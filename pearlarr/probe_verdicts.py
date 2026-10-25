"""Pure verdicts over the import probes' Sonarr reads: queue, in-flight import, commands, history rows."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from typing import NamedTuple

from .manual_import import Deferral, ImportProbe, PendingImport, fold_path_separators, normalized_leaf
from .seadex_types import CommandResource, HistoryPage, QueueRecord
from .stamps import parse_stamp_aware


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a download THIS poll."""

    WAIT = auto()
    """A record in motion (downloading, queued, delay, paused)."""

    IMPORTING = auto()
    """An `importing` record: Sonarr's pass has the download in hand, copying it or re-checking it."""

    PENDING_CLEAN = auto()
    """A clean (status `ok`) `importPending` record."""

    STEP_IN = auto()
    """A record Sonarr won't progress: `importBlocked`, `failed`, `failedPending`, `ignored`, or `importPending`
    flagged warning or error."""

    NOT_IN_QUEUE = auto()
    """No queue record: Sonarr isn't tracking the download, or hasn't rescanned since it appeared."""


# trackedDownloadState values compared case-folded: `importing`, and the transients Sonarr surfaces in the same field.
_QUEUE_IMPORTING_STATES = frozenset({"importing"})
_QUEUE_IN_MOTION_STATES = _QUEUE_IMPORTING_STATES | {"downloading", "queued", "delay", "paused"}
_QUEUE_STEP_IN_STATES = frozenset(
    {"importblocked", "failed", "failedpending", "ignored"},
)
# The `status` values that mark an importPending record as a failed Sonarr attempt.
_QUEUE_FLAGGED_STATUSES = frozenset({"warning", "error"})


def classify_queue(records: Sequence[QueueRecord]) -> QueueVerdict:
    """Reduce a download's queue records to one verdict.

    1. no records -> `NOT_IN_QUEUE`
    2. any record `importing` -> `IMPORTING`
    3. else any record in motion -> `WAIT`
    4. else any record `importBlocked`, `failed`, `failedPending`, `ignored`, or `importPending` with a
       warning or error status -> `STEP_IN`
    5. else a clean `importPending` -> `PENDING_CLEAN`, otherwise -> `STEP_IN`
    """

    if not records:
        return QueueVerdict.NOT_IN_QUEUE
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
_MANUAL_IMPORT_COMMAND_NAMES = frozenset({"manualimport"})
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


def manual_import_in_flight(commands: Sequence[CommandResource], match: DownloadMatch) -> InFlightImport | None:
    """The queued or started ManualImport covering this download, or None.

    Matched by its files' download id (case-insensitive), or, for a command whose files carry no download id, by a
    `content_paths` prefix or a `target_ep_ids` overlap.
    """

    target_hash = match.infohash.casefold()
    paths = match.content_paths
    target_ep_ids = match.target_ep_ids
    content_prefixes = {_norm_path(paths.raw), _norm_path(paths.sonarr_visible)}
    for command in _with_status(commands, _MANUAL_IMPORT_COMMAND_NAMES, _COMMAND_IN_FLIGHT_STATES):
        file_hashes = {f.download_id.casefold() for f in command.files if f.download_id is not None}
        if target_hash in file_hashes:
            return InFlightImport(command, by_download_id=True)
        # A command whose files carry download ids for another torrent is not this download's.
        if file_hashes:
            continue
        for file in command.files:
            if file.path is not None and any(_norm_path(file.path).startswith(p) for p in content_prefixes):
                return InFlightImport(command, by_download_id=False)
            if any(ep_id in target_ep_ids for ep_id in file.episode_ids):
                return InFlightImport(command, by_download_id=False)
    return None


_RESCAN_COMMAND_NAMES = frozenset({"refreshmonitoreddownloads"})
_SONARR_DISK_COMMAND_NAMES = frozenset(
    {
        "processmonitoreddownloads",
        "downloadedepisodesscan",
        "manualimport",
        "renamefiles",
        "renameseries",
        "moveseries",
        "bulkmoveseries",
        "deleteseriesfiles",
    }
)
_COMPLETED = frozenset({"completed"})
# A completed rescan counts as after the add only when it started this much later.
_RESCAN_PROOF_MARGIN = timedelta(seconds=60)


def in_flight_disk_commands(commands: Sequence[CommandResource]) -> list[CommandResource]:
    """The Sonarr disk-access commands queued or executing now."""

    return _with_status(commands, _SONARR_DISK_COMMAND_NAMES, _COMMAND_IN_FLIGHT_STATES)


def disk_commands_changed_since(
    before: Sequence[CommandResource], after: Sequence[CommandResource]
) -> list[CommandResource]:
    """The disk-access commands in `after` whose `(id, status)` pair `before` did not list."""

    listed = {(command.id, (command.status or "").casefold()) for command in before}
    return [
        command
        for command in after
        if (command.name or "").casefold() in _SONARR_DISK_COMMAND_NAMES
        and (command.id, (command.status or "").casefold()) not in listed
    ]


def _completed_rescans(commands: Sequence[CommandResource]) -> list[CommandResource]:
    """The completed RefreshMonitoredDownloads commands Sonarr lists."""

    return _with_status(commands, _RESCAN_COMMAND_NAMES, _COMPLETED)


def sonarr_restarted(commands: Sequence[CommandResource]) -> bool:
    """Whether the list holds no completed rescan at all."""

    return not _completed_rescans(commands)


def rescanned_since_add(commands: Sequence[CommandResource], added_at: str) -> bool:
    """Whether a completed rescan started at least the margin after the `added_at` stamp."""

    added = parse_stamp_aware(added_at)
    if added is None:
        return False
    floor = added + _RESCAN_PROOF_MARGIN
    return any(command.started is not None and command.started >= floor for command in _completed_rescans(commands))


def _with_status(
    commands: Sequence[CommandResource], names: frozenset[str], statuses: frozenset[str]
) -> list[CommandResource]:
    """The named commands in one of `statuses`, casefolded on both axes."""

    return [
        command
        for command in commands
        if (command.name or "").casefold() in names and (command.status or "").casefold() in statuses
    ]


class CommandBlock(Enum):
    """Why the command list blocks a step-in this poll (values are the log phrases)."""

    OWN_IMPORT = "our ManualImport is in flight"
    IN_FLIGHT_IMPORT = "a ManualImport is already in flight"
    DISK_COMMAND = "Sonarr has a disk command queued or running"
    DISK_COMMAND_CHANGED = "a Sonarr disk command ran during the step-in"
    SONARR_RESTARTED = "Sonarr lists no completed rescan (just restarted)"

    @property
    def claims_import(self) -> bool:
        """Whether the in-flight import is provably ours (the probe reads `command_issued`)."""

        return self is CommandBlock.OWN_IMPORT

    @property
    def deferral(self) -> Deferral:
        """The wait's reason for the probe: an import in flight, a busy Sonarr, or none after a restart."""

        if self in _IMPORT_BLOCKS:
            return Deferral.IMPORT
        return Deferral.NONE if self is CommandBlock.SONARR_RESTARTED else Deferral.BUSY

    def waiting(self) -> ImportProbe:
        """The wait this block calls for."""

        return ImportProbe.waiting(command_issued=self.claims_import, deferral=self.deferral)


_IMPORT_BLOCKS = frozenset({CommandBlock.OWN_IMPORT, CommandBlock.IN_FLIGHT_IMPORT})


def classify_commands(
    commands: Sequence[CommandResource],
    match: DownloadMatch,
    is_own: Callable[[int], bool],
) -> CommandBlock | None:
    """Reduce one `/api/v3/command` snapshot to the block holding a step-in, or None when clear.

    `is_own` says whether a command id is one this run POSTed.
    """

    in_flight = manual_import_in_flight(commands, match)
    if in_flight:
        owned = in_flight.by_download_id or is_own(in_flight.command.id)
        return CommandBlock.OWN_IMPORT if owned else CommandBlock.IN_FLIGHT_IMPORT
    if in_flight_disk_commands(commands):
        return CommandBlock.DISK_COMMAND
    return None


# The episode-history events that map a re-appeared download to a queue-hidden
# tracked state (Imported / Failed / Ignored), states Sonarr never runs its
# completed-download Check on, so `manualimport?downloadId=` never serves them
# (a 500 on v4, `[]` on v5). Keyed by casefolded eventType, valued by the label
# the debug note renders. `grabbed` (or none of the four) means genuinely Downloading.
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
    re-grabs of previously-FAILED/IGNORED hashes, which the by-id scan can't
    serve either. A partly imported download reads dead too. An imported
    verdict carries its cycle's import rows.
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

    Our seed stays authoritative, and a row lands only where some claim admits it: by name
    (`excluded_files`) and by id (a claim on the row's series whose window holds the episode).
    """

    wanted = pending.unplaced_names()
    grouped: dict[str, set[int]] = {}
    for row in imports:
        name = normalized_leaf(row.path)
        if name not in wanted:
            continue
        if not any(c.series_id == row.series_id and c.admits(row.episode_id) for c in pending.claims):
            continue
        grouped.setdefault(name, set()).add(row.episode_id)
    return {name: sorted(ids) for name, ids in grouped.items()}
