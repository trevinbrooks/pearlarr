"""Pure decision logic for Sonarr manual imports.

The deterministic planning vocabulary the import subsystem shares: the queue
verdict (`classify_queue`) and the in-flight ManualImport guard, the
`(season, episode) -> id` index and the episode-file status / never-overwrite
checks, the file -> episode assignment (`assign_episode_ids`), the grab-time
seed fold (`build_pending_seed`), the per-file import plan
(`plan_import_files`), and the layered quality/language resolution.

Side-effect free like `manual_import`, from which it imports the wait/outcome
vocabulary and the normalizers.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum, auto
from itertools import takewhile
from types import MappingProxyType
from typing import NamedTuple, Self

from .coverage import coverage_string, episodes_from_ep_list
from .manual_import import (
    Deferral,
    EntryNames,
    GuardFacts,
    PendingImport,
    fold_path_separators,
    normalize_group,
    normalize_rg,
    normalized_leaf,
)
from .seadex_types import (
    CommandResource,
    EpisodeKey,
    EpisodeRecord,
    HistoryPage,
    Language,
    ParsedFileInfo,
    Quality,
    QualityDefinition,
    QualityModel,
    QualitySource,
    QueueRecord,
    RemotePathMapping,
    Revision,
    SeadexDict,
    SeadexUrlItem,
    SonarrEpisode,
    flagged_urls,
    index_episodes_by_key,
    season_episode_key,
)


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download THIS poll.

    Derived purely from the queue records sharing a `downloadId` (a season pack
    has one record per episode), reading `trackedDownloadState` plus - for
    pending records - `trackedDownloadStatus`. "Already imported" is NOT decided
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
# mean Sonarr is genuinely working the download right now - wait rather than race
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
         than step in - stepping in would race Sonarr's own import.
      4. otherwise (empty because Sonarr isn't tracking it, all `imported`, or an
         unknown state) -> `STEP_IN`.

    Args:
        records: Every queue record matching the download (matched by the
            caller). A record with no tracked state contributes nothing.

    Returns:
        The action this poll, BEFORE the episode-file "already imported" check
        the caller layers on top.
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
            # Invariant: only a CLEAN importPending defers to Sonarr - a
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
# running - a terminal command (completed / failed / aborted / cancelled /
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
    raw qBittorrent one - the guard needs both prefixes.
    """

    raw: str
    """The qBittorrent `content_path`."""

    sonarr_visible: str
    """The remote-path-mapped view (equal to `raw` when no translation was
    computed or none applies)."""


class InFlightImport(NamedTuple):
    """The in-flight ManualImport `manual_import_in_flight` matched, plus HOW.

    Only a provable (`by_download_id`) or own-issued match may be credited back
    to the ready deadline; an unproven one stays a plain deadline-bounded wait.
    """

    command: CommandResource
    """The matched, still-running ManualImport."""

    by_download_id: bool
    """True for the primary `download_id` match; False for the path/episode fallback."""


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
    in" and we'd stack a duplicate every poll; matching the durable infohash
    (case-insensitive) closes that loop and survives a process restart. A
    no-download-id folder import falls back to a `content_paths`-prefix or
    `target_ep_ids` overlap - deliberately broad, since a false positive only
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
# starts - the one the rescan settles (see `sonarr_process_pass_running`).
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
    `started` blocks - a queued pass is near-permanently present during a wait,
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

    Pure (mirrors `classify_queue`); `is_own` is the executor's this-run
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
# tracked state (Imported / Failed / Ignored) - states Sonarr never runs its
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
    will never serve by id - import from its folder instead."""

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
    re-grabs of previously-FAILED/IGNORED hashes - the same NREs apply there.
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

    Longest-`remotePath`-prefix match is the PRIMARY rule. Host equality only
    tiebreaks equally-long prefixes, and host inequality never excludes a
    mapping (Sonarr's `host` is the download-client host as SONARR configured
    it - routinely a different string from our qBittorrent host: localhost vs
    container name vs IP). Matching is per path segment, so it is
    separator-boundary-aware (`/downloads` never matches `/downloads-x/f`),
    tolerant of trailing slashes and Windows backslashes on either side, and
    case-insensitive - while the suffix keeps its ORIGINAL case (POSIX targets
    are case-sensitive). No match returns the path untranslated (the
    same-filesystem no-op).

    Args:
        content_path: The qBittorrent `content_path` (a folder or a single
            file).
        mappings: Sonarr's remote path mappings.
        qbit_host: Our qBittorrent hostname (casefolded upstream or not -
            folded here), for the tiebreak only.

    Returns:
        The Sonarr-visible path, or `content_path` unchanged.
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


@dataclass(frozen=True, slots=True)
class EpisodeIndex:
    """The import family's one episode index, built once per episode fetch.

    Both facets detach and wrap read-only at construction; `episode_index` is
    the one builder. Episodes with a falsy id (0) are dropped from EVERY facet
    before keying - a 0 id can never be POSTed to Sonarr, and a real-id twin
    behind one must still win its `(season, episode)` key. The planner's
    `index_episodes_by_key` deliberately KEEPS them (a 0-id file is still
    identity evidence) - do not unify the two.
    """

    by_id: Mapping[int, SonarrEpisode]
    """Episode id -> episode, in the fetch's (season) order - `list(by_id)` is
    the resolved set the add flow persists onto each seed."""

    id_by_key: Mapping[EpisodeKey, int]
    """`(season, episode)` -> episode id (missing numbers collapse to
    `SONARR_MISSING_KEY`, first record wins - via `index_episodes_by_key`)."""

    def __post_init__(self) -> None:
        # Detach from the caller's dicts, then wrap read-only.
        object.__setattr__(self, "by_id", MappingProxyType(dict(self.by_id)))
        object.__setattr__(self, "id_by_key", MappingProxyType(dict(self.id_by_key)))


def episode_index(ep_list: Iterable[SonarrEpisode]) -> EpisodeIndex:
    """Fold one `/api/v3/episode` fetch into the `EpisodeIndex` facets."""

    with_ids = [ep for ep in ep_list if ep.id]
    return EpisodeIndex(
        by_id={ep.id: ep for ep in with_ids},
        id_by_key={key: ep.id for key, ep in index_episodes_by_key(with_ids).items()},
    )


class EpisodeFileStatus(Enum):
    """How an intended target episode's CURRENT Sonarr file relates to ours.

    One read of the episode list drives both invariants - never overwrite a
    recommended file, never skip an episode we intended to import.
    """

    ABSENT = auto()
    """No file yet. Import ours."""

    RECOMMENDED = auto()
    """Already holds a file from a recommended group (ours, another torrent we grabbed for this series, or
    a group the entry's SeaDex picks carried at grab time) - or an untagged file still at the exact size
    the grab-time identification recorded. It is done - do NOT overwrite it."""

    OTHER_GROUP = auto()
    """Holds a file from a non-recommended group - or our own group at a size no current listing carries
    (a stale copy this grab replaces). Import ours over it (the operator's intended replacement)."""

    UNKNOWN_GROUP = auto()
    """Holds a file with no parseable group that no recorded size identifies either. Import ours rather
    than trust an unidentifiable file as recommended."""


@dataclass(frozen=True, slots=True)
class TargetStatuses:
    """Each intended target episode's file status, with the two folds every consumer reads."""

    by_id: Mapping[int, EpisodeFileStatus]
    """One status per de-duplicated target id."""

    def all_done(self) -> bool:
        """True only when EVERY intended target already holds a recommended file.

        The "already imported / drop the record" signal. An UNKNOWN_GROUP or
        OTHER_GROUP file is NOT done (we still intend to import ours), so a
        present-but-unidentifiable file never makes us drop a record prematurely.
        """

        return bool(self.by_id) and all(s is EpisodeFileStatus.RECOMMENDED for s in self.by_id.values())

    def needing_import(self) -> set[int]:
        """The never-skip set: every intended id NOT already a recommended file.

        ABSENT / OTHER_GROUP / UNKNOWN_GROUP all need our import. Only
        RECOMMENDED is excluded (it is done and must not be overwritten).
        """

        return {ep_id for ep_id, status in self.by_id.items() if status is not EpisodeFileStatus.RECOMMENDED}


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus what counts as already ours.

    The episode index and the trust policy are gathered together, so consumers never mix state from two
    different polls.
    """

    episodes: EpisodeIndex
    """The fresh episode index."""

    trusted: Mapping[str, frozenset[int] | None]
    """The per-group trust policy (see `trusted_groups`): normalized group -> the sizes that verify
    its files, or None to trust it by name alone. A group absent here is not recommended - its files
    are replaced."""

    owned_episode_sizes: Mapping[int, int] = MappingProxyType({})
    """Episode id -> the untagged file size the grab-time identification recorded. The claim is honored
    only while the file still sits at that size; anything else untagged classifies as unidentifiable."""

    def statuses(self, target_ep_ids: list[int]) -> TargetStatuses:
        """Classify each intended target episode by its current on-disk file.

        Pure: reads only this snapshot's episode index and per-group trust
        policy (keyed via `normalize_group`). "Already imported" is decided
        HERE from the episode files - not from the queue, since Sonarr drops
        an imported item from its queue almost immediately.
        """

        statuses: dict[int, EpisodeFileStatus] = {}
        for ep_id in target_ep_ids:
            if ep_id in statuses:
                continue
            ep = self.episodes.by_id.get(ep_id)
            if ep is None or not ep.episode_file_id:
                statuses[ep_id] = EpisodeFileStatus.ABSENT
                continue
            group = ep.episode_file.release_group if ep.episode_file else None
            size = ep.episode_file.size if ep.episode_file else None
            if not group:
                # An untagged file still at the exact size the grab-time
                # identification recorded is a recommended copy; anything else
                # untagged (a different file landed meanwhile, or no readable
                # file record at all) stays unidentifiable.
                statuses[ep_id] = (
                    EpisodeFileStatus.RECOMMENDED
                    if size is not None and size == self.owned_episode_sizes.get(ep_id)
                    else EpisodeFileStatus.UNKNOWN_GROUP
                )
                continue
            norm = normalize_group(group)
            if norm not in self.trusted:
                statuses[ep_id] = EpisodeFileStatus.OTHER_GROUP
                continue
            verify_sizes = self.trusted[norm]
            if verify_sizes is not None and size not in verify_sizes:
                # A trusted group at a size no current listing carries: the stale
                # copy this grab replaces, not our just-imported file.
                statuses[ep_id] = EpisodeFileStatus.OTHER_GROUP
            else:
                statuses[ep_id] = EpisodeFileStatus.RECOMMENDED
        return TargetStatuses(statuses)


def trusted_groups(
    pending: PendingImport,
    series_records: Sequence[PendingImport] = (),
) -> dict[str, frozenset[int] | None]:
    """One record's per-group trust policy: group -> verifying sizes, or None for trust-by-name.

    The one home of the overwrite-guard composition, for grab time (no
    `series_records`) and import time (the series' pending records - this
    record's own row may ride along; its votes are no-ops) alike. The entry's
    verified-current pick groups and the series' other grabbed groups are
    trusted by name; a sibling's group is refused when THIS record's plan
    judged it stale on disk (the copies being replaced must not ride back into
    protection on a sibling's vote). The record's OWN group joins last and
    unconditionally - it is the identity of the files being imported - but at
    the sizes its current listings carry (unioned across same-group records),
    so a stale same-group copy is told apart by size and replaced. No listed
    sizes means no size gate (the legacy trust-by-name behavior).
    """

    stale = {norm for g in pending.guards.stale_groups if (norm := normalize_rg(g))}
    trusted: dict[str, frozenset[int] | None] = {
        norm: None for g in pending.guards.entry_groups if (norm := normalize_rg(g))
    }
    own = normalize_rg(pending.release_group)
    own_sizes = set(pending.release_sizes)
    for record in series_records:
        norm = normalize_rg(record.release_group)
        if norm is None:
            continue
        if norm == own:
            own_sizes.update(record.release_sizes)
        if norm not in stale:
            trusted.setdefault(norm, None)
    if own:
        trusted[own] = frozenset(own_sizes) or None
    return trusted


@dataclass(frozen=True, slots=True)
class PendingSeedContext:
    """The per-entry values every seed built for one AniList entry carries.

    One instance per `process_al_id` call, threaded whole into
    `build_pending_seeds` (instead of loose params) and copied onto each
    `PendingImport` the entry produces.
    """

    al_id: int
    """The AniList entry id - part of each record's `PendingKey`."""
    series_id: int
    """The Sonarr series id the entry's files belong to."""
    title: str
    """The AniList display title (logging only)."""
    added_at: str
    """When the entry's seeds were written (`UPDATED_AT_STR_FORMAT`), stamped once per entry at the
    impure edge and copied onto each record for the TTL drop."""
    coverage: str | None = None
    """The entry's season/episode coverage at grab time (logging only)."""
    url: str | None = None
    """The SeaDex entry URL at grab time (logging only)."""
    guards: GuardFacts = field(default_factory=GuardFacts)
    """The plan's overwrite-guard evidence, copied onto every seed unchanged
    (see `GuardFacts`)."""


_SXXEXX: re.Pattern[str] = re.compile(r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})")


def parse_se_from_filename(name: str) -> ParsedFileInfo | None:
    """Offline `SxxExx` fallback for when Sonarr's `/parse` is unreachable.

    Pure + regex-only: pulls a single `SxxExx` out of a leaf and returns it as a
    `ParsedFileInfo` (season + episode). Returns None when the name carries
    no `SxxExx` (an absolute-numbered or unparseable leaf) - those are left to
    Sonarr's parse or the absolute-index leg, never guessed from a bare number.
    Marked `offline` because the regex knows nothing about absolute numbers: a
    dual-numbered name ("S01E12 - 12") parsed here would otherwise launder its
    lost absolute into a "known" parse and blind the positional leg's tell.
    """

    m = _SXXEXX.search(name)
    if not m:
        return None
    return ParsedFileInfo(
        season_number=int(m.group(1)),
        episode_numbers=(int(m.group(2)),),
        offline=True,
    )


class _EpisodeClaim(NamedTuple):
    """One identity claim a file's parse makes: a `(season, episode)` pair, plus Sonarr's id when borrowed."""

    season: int | None
    episode: int
    claimed_id: int | None
    """Sonarr's own episode id from a borrowed matched pair - it must agree
    with our map's id. None for a name-parsed claim (no id to cross-check)."""


class PlacementVerdict(StrEnum):
    """How one file left `assign_episode_ids`: placed by which pass, excluded, or unplaced."""

    EXACT = "exact"
    """Its own `(season, episode)`, or Sonarr's matched pair, resolved inside the scope."""
    RELEASE_RUN = "release run"
    """A `1..N` run's own numbering indexed the window over Sonarr's incoherent reading."""
    ABSOLUTE = "absolute"
    """The clean absolute zip."""
    SINGLE = "single file"
    """One numberless leftover onto one leftover episode."""
    ORDERED = "ordered"
    """The pristine numberless batch, zipped in natural name order."""
    NUMBERED_RUN = "numbered run"
    """A `1..N` run among the files Sonarr could not read at all."""
    TITLED = "titled"
    """The one numberless leftover an entry title names, onto one leftover episode."""
    FOREIGN = "other slice"
    """Resolves cleanly, entirely outside this record's set: never this record's to import."""
    DUPLICATE = "duplicate"
    """Resolves inside the set onto an episode another file already holds."""
    HELD = "held"
    """A release-run member whose batch has an unknown parse: no other pass may place it (re-asked)."""
    SKIPPED = "skipped"
    """Nothing placed it and nothing proved it foreign."""

    @property
    def placed(self) -> bool:
        """Whether the verdict carries episode ids."""

        return self in _PLACED

    @property
    def excluded(self) -> bool:
        """Whether the file is knowably never this record's to import."""

        return self in _EXCLUDED


_PLACED = frozenset(
    {
        PlacementVerdict.EXACT,
        PlacementVerdict.RELEASE_RUN,
        PlacementVerdict.ABSOLUTE,
        PlacementVerdict.SINGLE,
        PlacementVerdict.ORDERED,
        PlacementVerdict.NUMBERED_RUN,
        PlacementVerdict.TITLED,
    }
)
_EXCLUDED = frozenset({PlacementVerdict.FOREIGN, PlacementVerdict.DUPLICATE})


class Placement(NamedTuple):
    """One file's verdict, with its episode ids when placed."""

    name: str
    """The normalized basename."""
    ids: tuple[int, ...]
    """The episode ids, in claim order (empty unless `verdict.placed`)."""
    verdict: PlacementVerdict
    """How the file left the passes."""


class EpisodeAssignment(NamedTuple):
    """The outcome of assigning a torrent's on-disk files to resolved episode ids, one verdict per file."""

    placements: tuple[Placement, ...]
    """One per distinct name in `PlacementBatch.to_place`, batch order."""

    @property
    def assigned(self) -> dict[str, list[int]]:
        """Normalized basename -> episode ids for every placed file (each id in the scope, used once)."""

        return {p.name: list(p.ids) for p in self.placements if p.verdict.placed}

    @property
    def skipped(self) -> tuple[str, ...]:
        """The files nothing placed and nothing excluded: the caller warns, never guesses."""

        return tuple(p.name for p in self.placements if not p.verdict.placed and not p.verdict.excluded)

    @property
    def excluded(self) -> tuple[Placement, ...]:
        """The files this record knowably never imports (another slice's, a refused duplicate), with their verdicts."""

        return tuple(p for p in self.placements if p.verdict.excluded)

    @property
    def unplaced(self) -> tuple[str, ...]:
        """Every file without ids: the skips plus the exclusions."""

        return tuple(p.name for p in self.placements if not p.verdict.placed)


class PlacementBatch(NamedTuple):
    """A torrent's leftover on-disk files to place, with the WHOLE batch's parses.

    `parsed` may cover MORE files than `to_place`: seeded and gone files feed
    the absolute leg's shared-absolute tell, but only `to_place` is ever placed.
    """

    to_place: Sequence[str]
    """Normalized basenames in SeaDex order (the order only fixes deterministic output)."""

    parsed: Mapping[str, ParsedFileInfo | None]
    """Series-agnostic parse per file (None when Sonarr's parse was unavailable
    and no SxxExx fell out of the name)."""

    @property
    def all_parses_known(self) -> bool:
        """Every parse came from Sonarr this run: no transport miss (None) and no offline `SxxExx` stand-in.

        Seeded and gone names ride the batch parsed by name, so a miss on any of them counts, as
        does a name to place the parses never covered.
        """

        names = dict.fromkeys([*self.to_place, *self.parsed])
        return all((info := self.parsed.get(name)) is not None and not info.offline for name in names)


class TargetScope(NamedTuple):
    """The episode set a placement batch may assign into.

    `resolved` is the FULL resolved set and `used` the ids seeds already own,
    kept separate so a fully seeded record stays scope-enforced while an EMPTY
    `resolved` means no scope at all (`unscoped`): the exact leg then places
    name-parsed pairs against the live series map instead of sticking forever.
    """

    resolved: Sequence[int]
    """The entry's resolved episode ids, season order - seeded included. A stray
    zero id still makes the scope real; it is never placed."""

    series: EpisodeIndex
    """ALL the series' episodes: the `(season, episode)` map every key resolves
    through (membership in `resolved` does the scoping, so an exact parse
    outside our entry is rejected), plus the titles and absolute numbers the
    run evidence reads."""

    used: frozenset[int] = frozenset()
    """Ids a seed already owns - never handed to a leftover file."""

    names: EntryNames = EntryNames()
    """The series and AniList titles: when several runs or numberless files could take the window, the one
    an AniList title names does."""

    @property
    def id_by_key(self) -> Mapping[EpisodeKey, int]:
        """`(season, episode)` -> id over the whole series."""

        return self.series.id_by_key

    @property
    def unscoped(self) -> bool:
        """No resolved set to scope against at all (never just fully seeded)."""

        return not self.resolved


class SeedFile(NamedTuple):
    """One listed video file, its size and its cached grab-time `/parse`, as the placement reads it."""

    basename: str
    """The raw SeaDex basename (normalized only where the placement keys the batch)."""

    size: int
    """The listed size, carried onto the records the planner compares by size."""

    parse: ParsedFileInfo | None
    """Sonarr's parse of the name, or None when no fresh parse record exists (an unknown parse holds
    every count leg closed, exactly as it does at import time)."""


class SeedScope(NamedTuple):
    """What one entry's files are placed against: its own index, the whole-series map, and its names."""

    entry: EpisodeIndex
    """The entry's episodes: the resolved set, and what the slice and preowned reads cover."""

    series: EpisodeIndex
    """The WHOLE series (empty when the series list could not be read, which places nothing)."""

    names: EntryNames
    """The series and AniList titles the placement breaks ties by, persisted onto every seed."""

    @property
    def can_place(self) -> bool:
        """Whether the scope is real enough to place against: an entry index and a served series map.

        An empty entry index must never read as "no scope" (the unscoped arm places against the live
        map), and an unread map places nothing: a grab-time verdict is final where an import poll is retried.
        """

        return bool(self.entry.by_id and self.series.id_by_key)

    def target(self) -> TargetScope:
        """The placement scope the grab shares with import time: the entry's ids, nothing used yet."""

        return TargetScope(list(self.entry.by_id), self.series, names=self.names)


class UrlPlacement(NamedTuple):
    """One url's files placed at grab time: the verdicts, and the records the planner judges coverage by."""

    files: tuple[SeedFile, ...]
    """The importable video files in SeaDex order (subs / fonts / NCED already dropped)."""

    assignment: EpisodeAssignment
    """The placement verdicts, one per distinct normalized leaf."""

    records: tuple[EpisodeRecord, ...]
    """One `(season, episode, size)` per placed file and episode, in file order. A file nothing placed
    leaves none, so a url whose files place nowhere blankets the planner's coverage."""

    parses_known: bool
    """Every parse came from Sonarr this run (`PlacementBatch.all_parses_known`). False means a request
    failed and a run may be held, so the title is re-checked next run."""


def place_release(files: Sequence[SeedFile], scope: SeedScope) -> UrlPlacement:
    """Place one url's files by the same `assign_episode_ids` the import wait runs.

    Pure. The batch is keyed by NORMALIZED leaf so it matches the on-disk leaves at import time
    (NFC/NFD-safe): two raw files sharing a leaf share its verdict and keep their own sizes.
    """

    to_place = list(dict.fromkeys(normalized_leaf(f.basename) for f in files))
    parsed: dict[str, ParsedFileInfo | None] = {}
    for f in files:
        parsed.setdefault(normalized_leaf(f.basename), f.parse)
    batch = PlacementBatch(to_place, parsed)
    assignment = assign_episode_ids(batch, scope.target()) if scope.can_place else EpisodeAssignment(())
    assigned = assignment.assigned
    records: list[EpisodeRecord] = []
    for f in files:
        # Every placed id is in the entry's resolved set, so the index read cannot miss.
        for ep_id in assigned.get(normalized_leaf(f.basename), []):
            episode = scope.entry.by_id[ep_id]
            # An episode Sonarr sent unnumbered keys into no coverage, so it records nothing (a blanket).
            if episode.season_number is None or episode.episode_number is None:
                continue
            records.append(EpisodeRecord(season=episode.season_number, episode=episode.episode_number, size=f.size))
    return UrlPlacement(tuple(files), assignment, tuple(records), batch.all_parses_known)


class EntryPlacements(NamedTuple):
    """Every url of one entry placed, keyed as `SeadexReleaseGroupItem.urls` keys them."""

    scope: SeedScope
    """The scope every url was placed under (the seeds persist its names)."""

    by_url: Mapping[str, UrlPlacement]
    """One placement per url, every url present (an empty one for a url with no video file)."""

    @classmethod
    def place(cls, scope: SeedScope, files_by_url: Mapping[str, Sequence[SeedFile]]) -> "EntryPlacements":
        """Place each url's gathered files under one scope."""

        return cls(scope, {url: place_release(files, scope) for url, files in files_by_url.items()})

    def attach_records(self, seadex_dict: SeadexDict) -> None:
        """Write each url's placed records onto its item, and their union onto its group, for the planner."""

        for rg_item in seadex_dict.values():
            all_episodes: list[EpisodeRecord] = []
            for url, url_item in rg_item.urls.items():
                url_item.episodes = list(self.by_url[url].records)
                all_episodes.extend(url_item.episodes)
            rg_item.all_episodes = all_episodes

    def parse_failed_groups(self, seadex_dict: SeadexDict) -> tuple[str, ...]:
        """The groups with a url whose placement waits on a parse Sonarr failed to serve."""

        return tuple(
            group for group, item in seadex_dict.items() if not all(self.by_url[url].parses_known for url in item.urls)
        )


class SeedRelease(NamedTuple):
    """One flagged release's seed inputs: the url record and its placement (I/O done)."""

    release_group: str
    """The SeaDex release group (authoritative)."""

    url_item: SeadexUrlItem
    """The flagged url record whole: the fold reads its sizes and dual-audio flag."""

    infohash: str
    """The qBittorrent tracking key (the url item's hash, already narrowed to `str`)."""

    placed: UrlPlacement
    """The url's files placed at grab time (`place_release`)."""


def build_pending_seed(
    release: SeedRelease,
    scope: SeedScope,
    entry: PendingSeedContext,
) -> PendingImport:
    """Fold one flagged release into its durable `PendingImport` seed.

    Pure: consumes the placement riding `release`, the entry's scope, and the
    per-entry context. The files were placed by the same `assign_episode_ids`
    the import wait runs, so a seed is an import-time placement made early: a
    placed file is seeded, an excluded one (another slice's, a refused
    duplicate) is recorded as never this record's to import, and anything
    held or skipped is left for import time, where the parses are re-read.
    """

    placed = release.placed
    file_episode_map = placed.assignment.assigned
    excluded_files = [placement.name for placement in placed.assignment.excluded]
    claimed = {ep_id for ids in file_episode_map.values() for ep_id in ids}

    # This record's own slice of the entry, so sibling per-episode records label
    # distinctly: the episodes its files claimed, else every episode it is verified against.
    index = scope.entry
    slice_eps = [ep for ep in index.by_id.values() if not claimed or ep.id in claimed]
    seed = PendingImport(
        infohash=release.infohash,
        series_id=entry.series_id,
        al_id=entry.al_id,
        file_episode_map=file_episode_map,
        # episode_ids is a legacy read-only fallback: never seeded (any
        # value here would only duplicate the map, which readers dedupe).
        episode_ids=[],
        release_group=release.release_group,
        is_dual_audio=release.url_item.is_dual_audio,
        seadex_files=[f.basename for f in placed.files],
        title=entry.title,
        added_at=entry.added_at,
        coverage=entry.coverage,
        url=entry.url,
        ordered_episode_ids=list(index.by_id),
        slice_coverage=coverage_string(episodes_from_ep_list(slice_eps)) or None,
        excluded_files=excluded_files,
        guards=entry.guards,
        release_sizes=list(release.url_item.size),
        names=scope.names,
    )
    # Targets that already hold a recommended file at grab time were
    # never this torrent's to insert: classify them against the record's
    # own trust slice (no sibling votes yet) so the wait's inserted
    # counts start at 0.
    grab_snapshot = EpisodeSnapshot(
        episodes=index,
        trusted=trusted_groups(seed),
        owned_episode_sizes=seed.guards.owned_sizes,
    )
    preowned = [
        ep_id
        for ep_id, status in grab_snapshot.statuses(sorted(claimed)).by_id.items()
        if status is EpisodeFileStatus.RECOMMENDED
    ]
    return replace(seed, preowned_episode_ids=preowned)


def build_pending_seeds(
    seadex_dict: SeadexDict,
    placed: EntryPlacements,
    entry: PendingSeedContext,
) -> dict[str, PendingImport]:
    """Fold every flagged url carrying an infohash and a video file into its seed, keyed by infohash. Pure."""

    seeds: dict[str, PendingImport] = {}
    for flagged in flagged_urls(seadex_dict):
        placement = placed.by_url[flagged.url]
        if not placement.files:
            continue
        release = SeedRelease(flagged.group, flagged.item, flagged.infohash, placement)
        seeds[flagged.infohash] = build_pending_seed(release, placed.scope, entry)
    return seeds


# A borrowed span (Sonarr's matched pairs) plausibly covers a double or triple
# episode, never more. A name's own explicit range is its claim at any width.
_MATCHED_SPAN_CAP = 3


class _Reading(NamedTuple):
    """One file's identity reading: what its parse's claims resolve to in OUR map, and how far to trust it.

    The single reading every pass consults, so the seed and the import wait
    can never disagree on what a name says.
    """

    resolved: tuple[int, ...]
    """Every id a claim resolved to (inside the scope or not), claim order, deduped."""

    inside: tuple[int, ...]
    """The subset inside the resolved set (all of `resolved` when the scope is `unscoped`)."""

    complete: bool
    """At least one claim, and every claim resolved (a partial span is never half-placed)."""

    borrowed: bool
    """The claims are Sonarr's matched pairs (the name carried no `(season, episode)` of its own)."""

    vetoed: bool
    """A full-season parse, a borrowed span past `_MATCHED_SPAN_CAP` or short of the name's own absolutes,
    or a run's reads that dispute its count (a tie, or a covering run read into another season): the
    claims are real but never placed on their own, and never prove the file foreign."""

    corroborated: bool
    """Sonarr's series match names every claimed pair (a borrowed reading always is). An own key Sonarr
    could not match to the series is a parse it did not believe either (a CRC tag read as "E8")."""

    tied: tuple[int, ...] | None = None
    """The episodes a tie's name may be, under the season's and the specials' numbering: spoken for, so
    no other run takes a window holding one, and a run of them stands aside only from a scope holding
    none."""

    @property
    def outside(self) -> bool:
        """A trusted reading that resolves wholly outside the scope: the file is another slice's."""

        return self.complete and not self.vetoed and not self.inside

    @property
    def rank(self) -> tuple[bool, bool]:
        """Exact-pass precedence, lowest first: a corroborated own key, a borrowed pair, an unmatched own key."""

        return (not self.corroborated, self.borrowed)


_NO_READING = _Reading((), (), complete=False, borrowed=False, vetoed=False, corroborated=False)


def _read(info: ParsedFileInfo | None, scope: TargetScope, resolved_set: frozenset[int]) -> _Reading:
    """Read one parse against the scope.

    The name's own `(season, episode)` keys are the claims. A name with none
    borrows Sonarr's series-MATCHED pairs, but ONLY under scope enforcement:
    membership in `resolved_set` is what keeps Sonarr's series match from
    deciding identity on its own, so matched pairs never apply `unscoped`. A
    borrowed pair's own episode id must AGREE with our map's id for the same
    numbers, or a wrong-series title match whose numbers coincide with ours
    would resolve. Junk duplicate pairs collapse to one claim. A missing
    season collapses to `SONARR_MISSING_KEY`, matching `EpisodeIndex.id_by_key`.
    """

    if info is None:
        return _NO_READING
    claims: list[_EpisodeClaim] = [_EpisodeClaim(info.season_number, episode, None) for episode in info.episode_numbers]
    borrowed = False
    if not claims and not scope.unscoped and not info.full_season:
        claims = [
            _EpisodeClaim(matched.season_number, matched.episode_number, matched.id)
            for matched in info.matched_episodes
        ]
        borrowed = True
    claims = list(dict.fromkeys(claims))
    if not claims:
        return _NO_READING
    # Only a borrowed span is capped (DISTINCT pairs): Sonarr matches a bare
    # "S01" name to the WHOLE season, so a wide match is the season-pack shape
    # sans flag, while the name's own "E11-E16" is an explicit claim. A borrowed
    # span must also COVER the name's own absolutes, or a "12-13" file whose
    # match resolved only E12 would half-import.
    pairs = {(claim.season, claim.episode) for claim in claims}
    absolutes = set(info.absolute_episode_numbers)
    vetoed = info.full_season or (
        borrowed and (len(pairs) > _MATCHED_SPAN_CAP or (bool(absolutes) and len(pairs) != len(absolutes)))
    )
    matched = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
    corroborated = borrowed or pairs <= matched
    resolved: list[int] = []
    complete = True
    for claim in claims:
        ep_id = scope.id_by_key.get(season_episode_key(claim.season, claim.episode))
        if ep_id and claim.claimed_id in (None, ep_id):
            resolved.append(ep_id)
        else:
            complete = False
    # The triple dedup keeps (s,e,None) and (s,e,id) apart. Collapse the
    # resolved ids so one episode never reaches the wire twice.
    ids = tuple(dict.fromkeys(resolved))
    inside = ids if scope.unscoped else tuple(i for i in ids if i in resolved_set)
    return _Reading(ids, inside, complete=complete, borrowed=borrowed, vetoed=vetoed, corroborated=corroborated)


def _has_no_signal(info: ParsedFileInfo | None) -> bool:
    """Whether a file's NAME carries no usable episode number at all (parse miss).

    Deliberately blind to `matched_episodes`: the degenerate single-file
    fallback keys on this, and an out-of-set Sonarr match must not veto the
    placement OUR resolution intends (Sonarr informs identity, never decides -
    in either direction). Cardinality is `_spans_multiple`'s question.
    """

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def _signal_is_bogus(info: ParsedFileInfo, ep_id_map: Mapping[EpisodeKey, int]) -> bool:
    """Whether the name's numbers provably describe no episode of this series.

    A movie year read as SxxEyy ("Chronicle.2020" parsing S20E20) is a parse
    artifact, not identity: when EVERY name-parsed key misses the WHOLE series
    map and the name carries no absolutes, the signal is noise and the file
    counts as numberless. A key that resolves anywhere in the series is real
    evidence and is never downgraded. Only meaningful over a served map: an
    empty map makes every key "miss", so the caller gates on `map_known`.
    """

    if not info.episode_numbers or info.absolute_episode_numbers:
        return False
    return all(not ep_id_map.get(season_episode_key(info.season_number, episode)) for episode in info.episode_numbers)


def _claims_several(info: ParsedFileInfo) -> bool:
    """Whether the NAME claims more than one episode, so placing the file as one would half-import.

    Only the name's own numbers count. A multi-pair series match is a scene
    map for another numbering, and a full-season read of a file name is a
    missing episode token ("S2 - OVA", "S0101"), never a claim of several.
    """

    return len(set(info.episode_numbers)) > 1 or len(set(info.absolute_episode_numbers)) > 1


def _natural_key(name: str) -> str:
    """Digit-aware sort key ("sp10" sorts after "sp2"): zero-pad digit runs."""

    return re.sub(r"\d+", lambda match: match.group().zfill(12), name)


_BRACKETED = re.compile(r"[\[(][^\[\]()]*[\])]")
_TRAILING_TAG = re.compile(rf"\s*{_BRACKETED.pattern}$")
_NON_WORD = re.compile(r"[^0-9a-z]+")
_TRAILING_VERSION = re.compile(r"v(\d+)$")
# The episode of an "S02E01" key (a keyed run is judged by its keys), else
# the LAST " - NN - " (an "Episode" word may lead the number, a "vN" and
# the absolute in brackets may trail it, and a title follows), else the
# episode of a packed "S0101" (a season the series lacks, read as the
# release's count), else a trailing 1-3 digit integer not glued to more
# digits (a year or CRC tail is no release number).
_KEYED_NUMBER = re.compile(r"^(.*?[Ss]\d{1,2}[Ee])(\d{1,3})(?!\d)")
_MIDDLE_NUMBER = re.compile(
    r"^(.*) - (?:[Ee]pisode |[Ee]p\.? )?(\d{1,3})(?:v(?P<version>\d+))?(?: \[\d{1,3}\])?(?= - )"
)
_PACKED_NUMBER = re.compile(r"^(.*?(?:^|[\s._-])[Ss]\d{2})(\d{2})(?=[\s._-]|$)")
_TRAILING_NUMBER = re.compile(r"^(.*?)(?<!\d)(\d{1,3})$")
_NUMBER_FORMS = (_KEYED_NUMBER, _MIDDLE_NUMBER, _PACKED_NUMBER, _TRAILING_NUMBER)
# The text after a release number is the episode's title when this separates them.
_TITLE_SEPARATOR = " - "
# A numbered extras run (menus, previews, commercials) never indexes an
# episode window, however well its width fits.
_EXTRAS_RUN_TOKENS = frozenset({"pv", "cm", "menu", "trailer", "preview", "teaser", "promo", "op", "ed"})


class _RunMember(NamedTuple):
    """A file's numbered-run membership, read purely from its name."""

    prefix: str
    """The text before the release number: the grouping key."""
    number: int
    tail: str
    """The text after the number when a title separator follows it (the episode's title), else empty."""
    version: int
    """The `vN` after the number or trailing (1 when none): of two names sharing a number, the higher is the member."""


def _stem(name: str) -> str:
    """A name without its extension, trailing bracketed tags, and trailing `vN` (to a fixpoint), underscores as spaces."""

    return _stem_version(name)[0]


def _stem_version(name: str) -> tuple[str, int]:
    """`_stem` plus the highest trailing `vN` it shed (1 when none)."""

    stem = (name.rsplit(".", 1)[0] if "." in name else name).replace("_", " ")
    version = 1
    while True:
        trimmed = _TRAILING_TAG.sub("", stem).rstrip(" .-")
        if (found := _TRAILING_VERSION.search(trimmed)) is not None:
            version = max(version, int(found.group(1)))
            trimmed = trimmed[: found.start()].rstrip(" .-")
        if trimmed == stem:
            return stem, version
        stem = trimmed


def _run_member(name: str) -> _RunMember | None:
    """The release's own number in a name, read purely from the text.

    The stem's separator before the number is dropped, so
    "show_-_07v2_[bd 1080p].mkv" reads as ("show", 7). None when no form fits
    or the number counts extras ("show - PV 01").
    """

    stem, version = _stem_version(name)
    match = next((found for form in _NUMBER_FORMS if (found := form.match(stem)) is not None), None)
    if match is None:
        return None
    prefix = match.group(1).rstrip(" .-")
    words = [word for word in _NON_WORD.split(prefix.casefold()) if word]
    if words and words[-1] in _EXTRAS_RUN_TOKENS:
        return None
    rest = stem[match.end() :]
    tail = rest.removeprefix(_TITLE_SEPARATOR) if rest.startswith(_TITLE_SEPARATOR) else ""
    if (middle := match.groupdict().get("version")) is not None:
        version = max(version, int(middle))
    return _RunMember(prefix, int(match.group(2)), tail, version)


def _version(name: str) -> int:
    """The name's `vN` (1 when none), trailing or after its release number."""

    member = _run_member(name)
    return member.version if member is not None else _stem_version(name)[1]


def _extras_named(name: str) -> bool:
    """Whether the name carries an extras token anywhere: a preview or an opening is never the episode."""

    return not _EXTRAS_RUN_TOKENS.isdisjoint(_NON_WORD.split(name.casefold()))


# A title names a candidate when they share at least half their leftover words.
_MIN_TITLE_OVERLAP = 0.5
# Episode titles decide between runs, or against one, from this many members on.
_MIN_TITLE_HITS = 2
# The words that count a season, folded to its plain number so "2nd Season",
# "Season 2", "S2" and "II" agree. Lone "I", "V" and "X" stay words.
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|[nr]d|th)$")
_SEASON_TOKEN = re.compile(r"^s(\d{1,2})$")
_COUNT_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "vi": "6",
    "vii": "7",
    "viii": "8",
}


def _count_word(word: str) -> str | None:
    """The plain number a word counts with ("02", "2nd", "second", "s2", "ii"), else None."""

    if word.isdigit():
        return str(int(word))
    if (found := _ORDINAL.match(word) or _SEASON_TOKEN.match(word)) is not None:
        return str(int(found.group(1)))
    return _COUNT_WORDS.get(word)


def _word_list(text: str) -> list[str]:
    """The words of a title or name in order: case and accents folded, bracketed groups dropped, seasons counted plainly."""

    folded = unicodedata.normalize("NFKD", _BRACKETED.sub(" ", text)).encode("ascii", "ignore").decode().casefold()
    words = [_count_word(word) or word for word in _NON_WORD.split(folded) if word]
    # "Season" beside its number says nothing the number does not.
    return [
        word
        for index, word in enumerate(words)
        if word != "season"
        or not any(words[near].isdigit() for near in (index - 1, index + 1) if 0 <= near < len(words))
    ]


def _words(text: str) -> frozenset[str]:
    """The distinct words of a title or name."""

    return frozenset(_word_list(text))


class _Leftover(NamedTuple):
    """An AniList title's words beyond the series title, in title order, and as a set."""

    ordered: tuple[str, ...]
    words: frozenset[str]

    @classmethod
    def of(cls, title: str, ground: frozenset[str]) -> "_Leftover":
        ordered = tuple(word for word in _word_list(title) if word not in ground)
        return cls(ordered, frozenset(ordered))

    def overlap(self, rest: frozenset[str]) -> float:
        """The share of the combined leftover words a candidate's leftover has in common with this title."""

        return len(rest & self.words) / len(rest | self.words)

    def opening_only(self, rest: frozenset[str]) -> bool:
        """Whether a candidate shares nothing past this title's opening run of words.

        A title opens with the franchise name in its own language, which the
        series title's words cannot shed, and names the entry after it.
        """

        opening = frozenset(takewhile(rest.__contains__, self.ordered))
        return opening != self.words and opening == rest & self.words


class _Naming(NamedTuple):
    """Every candidate scored against the entry's AniList titles, both sides shed of the series title's words."""

    leftovers: tuple[_Leftover, ...]
    rests: tuple[frozenset[str], ...]
    scores: tuple[float, ...]
    """Each candidate's best overlap with a title."""

    @classmethod
    def of(cls, candidates: Sequence[str], names: EntryNames) -> "_Naming | None":
        """Score the candidates, or None when nothing can name them: no ground, no titles, or a title that IS the series."""

        ground = _words(names.series)
        leftovers = tuple(_Leftover.of(title, ground) for title in names.anilist)
        if not ground or not leftovers or not all(leftover.words for leftover in leftovers):
            return None
        rests = tuple(_words(candidate) - ground for candidate in candidates)
        scores = tuple(max(leftover.overlap(rest) for leftover in leftovers) for rest in rests)
        return cls(leftovers, rests, scores)

    @property
    def winners(self) -> frozenset[int]:
        """The candidates at the top score, when it reaches `_MIN_TITLE_OVERLAP`."""

        top = max(self.scores, default=0.0)
        if top < _MIN_TITLE_OVERLAP:
            return frozenset()
        return frozenset(index for index, score in enumerate(self.scores) if score == top)

    def named(self, index: int) -> bool:
        """Whether a winner shares more than the opening words of some title it scores best against."""

        rest = self.rests[index]
        return any(
            leftover.overlap(rest) == self.scores[index] and not leftover.opening_only(rest)
            for leftover in self.leftovers
        )


def _titled(candidates: Sequence[str], names: EntryNames) -> int | None:
    """The index of the one candidate an AniList title names, else None.

    The unique best sharing at least `_MIN_TITLE_OVERLAP` of the combined
    leftover words wins, unless it shares only the opening words of every
    title it scores best against. The check refuses the winner and never
    promotes a runner-up.
    """

    naming = _Naming.of(candidates, names)
    if naming is None or len(naming.winners) != 1:
        return None
    index = next(iter(naming.winners))
    return index if naming.named(index) else None


def _named(candidates: Sequence[str], names: EntryNames) -> frozenset[int]:
    """The indices of the candidates an AniList title names best (ties included), empty when it names none."""

    naming = _Naming.of(candidates, names)
    return frozenset() if naming is None else naming.winners


def _consecutive(numbers: Sequence[int]) -> bool:
    """Whether the numbers count up by one from the first."""

    return list(numbers) == list(range(numbers[0], numbers[0] + len(numbers)))


class _Numbered(NamedTuple):
    """One run member: its release number, its name, and the title text after the number (empty when none)."""

    number: int
    name: str
    tail: str


class _Run(NamedTuple):
    """One prefix's numbered members, number order."""

    prefix: str
    members: tuple[_Numbered, ...]
    superseded: tuple[_Numbered, ...] = ()
    """The lower versions a member's higher `vN` displaced: duplicates once the run is placed."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)

    @property
    def whole(self) -> tuple[_Numbered, ...]:
        """The members and the lower versions they displaced."""

        return (*self.members, *self.superseded)

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(member.number for member in self.members)

    def consecutive(self, width: int) -> bool:
        """Whether the run is exactly `width` consecutive numbers, wherever it starts."""

        return len(self.members) == width and _consecutive(self.numbers)


def _runs(names: Iterable[str], parsed: Mapping[str, ParsedFileInfo | None]) -> list[_Run]:
    """Every numbered run among `names`, grouped by prefix.

    A name whose parse carries exactly one absolute that disagrees with its
    release number is no member. Of several names sharing a number the
    highest `vN` is the member and the rest are superseded (equal versions
    all stay, which breaks the run).
    """

    members: dict[str, dict[int, list[tuple[int, _Numbered]]]] = {}
    for name in names:
        member = _run_member(name)
        if member is None:
            continue
        info = parsed.get(name)
        absolutes: set[int] = set(info.absolute_episode_numbers) if info is not None else set()
        if len(absolutes) == 1 and member.number not in absolutes:
            continue
        versions = members.setdefault(member.prefix, {}).setdefault(member.number, [])
        versions.append((member.version, _Numbered(member.number, name, member.tail)))
    runs: list[_Run] = []
    for prefix, by_number in members.items():
        kept: list[_Numbered] = []
        superseded: list[_Numbered] = []
        for versions in by_number.values():
            top = max(version for version, _ in versions)
            kept.extend(numbered for version, numbered in versions if version == top)
            superseded.extend(numbered for version, numbered in versions if version < top)
        runs.append(_Run(prefix, tuple(sorted(kept)), tuple(superseded)))
    return runs


def _fitting(runs: Iterable[_Run], numbers: Iterable[int]) -> list[_Run]:
    """The runs numbered exactly `numbers`."""

    expected = tuple(numbers)
    return [run for run in runs if run.numbers == expected]


def _from_one(runs: Iterable[_Run], width: int) -> list[_Run]:
    """The runs numbered `1..width`."""

    return _fitting(runs, range(1, width + 1))


class _RunWindow(NamedTuple):
    """The leftover ids as consecutive episodes in airing order."""

    season: int | None
    """The one season the ids belong to, or None when the series' absolute numbering orders several."""
    ids: tuple[int, ...]
    numbers: tuple[int, ...]
    """The ids' episode numbers (absolute numbers across seasons), the same order."""


class _TitledEpisode(NamedTuple):
    """One series episode with a title, as folded words."""

    ep_id: int
    words: tuple[str, ...]


class _TitleEvidence(NamedTuple):
    """How many of a run's tails name an episode inside the window, and how many one outside it."""

    inside: int
    outside: int

    @property
    def selects(self) -> bool:
        """Enough tails name the window, and more than name anything else."""

        return self.inside >= _MIN_TITLE_HITS and self.inside > self.outside

    @property
    def vetoes(self) -> bool:
        """The tails name other episodes and none of the window's."""

        return self.outside >= _MIN_TITLE_HITS and not self.inside


@dataclass(slots=True)
class _Placer:
    """One `assign_episode_ids` call's state: the readings, the verdicts so far, and the ids they used."""

    batch: PlacementBatch
    scope: TargetScope
    readings: dict[str, _Reading]
    """One reading per distinct name to place, batch order."""
    key_by_id: Mapping[int, EpisodeKey]
    """The series map inverted once (it never changes; only `used` does)."""
    resolved: frozenset[int]
    """The scope's real ids (a stray zero is never one)."""
    evidenced: frozenset[int]
    """The ids some file outside `to_place` (a seeded or gone name) resolves to by its own reading."""
    season_counts: Mapping[int, int]
    """Episodes per season in the series map."""
    absolute_of: Mapping[int, int]
    """Episode id -> the series' absolute number, for the ids that carry one."""
    titled: tuple[_TitledEpisode, ...]
    """The series' episodes that carry a title."""
    verdicts: dict[str, Placement] = field(default_factory=dict[str, Placement])
    used: set[int] = field(default_factory=set[int])
    count_legs_barred: bool = False
    """The release-run pass found several runs, or refused the one it found: no count leg (the numbered run,
    the absolute and ordered zips) places what it would not."""

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name once against the scope."""

        # A stray zero id can never be placed, but it still keeps the scope real
        # (only an EMPTY resolved set unlocks the live-map fallback).
        resolved_set = frozenset(i for i in scope.resolved if i)
        readings = {name: _read(batch.parsed.get(name), scope, resolved_set) for name in dict.fromkeys(batch.to_place)}
        key_by_id = {ep_id: key for key, ep_id in scope.id_by_key.items()}
        evidenced = frozenset(
            ep_id
            for name, info in batch.parsed.items()
            if name not in readings
            for ep_id in _read(info, scope, resolved_set).resolved
        )
        episodes = scope.series.by_id
        state = cls(
            batch,
            scope,
            readings,
            key_by_id,
            resolved_set,
            evidenced,
            season_counts=Counter(key.season for key in scope.id_by_key),
            absolute_of={
                ep_id: ep.absolute_episode_number
                for ep_id, ep in episodes.items()
                if ep.absolute_episode_number is not None
            },
            titled=tuple(
                _TitledEpisode(ep_id, tuple(words)) for ep_id, ep in episodes.items() if (words := _word_list(ep.title))
            ),
            used=set(scope.used),
        )
        state.reread_season_runs()
        return state

    @property
    def map_known(self) -> bool:
        """Whether the series map was served (every map-dependent verdict refuses on an empty one)."""

        return bool(self.scope.id_by_key)

    def reread_season_runs(self) -> None:
        """Re-read a `1..N` run Sonarr matched into one season of exactly N episodes as that season's own numbering.

        TVDB interleaves specials into the absolute numbering, so Sonarr's
        match of a season-only release drifts onto a special after each one.
        The count tells the shapes apart: a release that carried the special
        would number N + 1. Only a run of names without keys of their own,
        read by Sonarr into that one season and its specials, is re-read:
        more into the season, or as many when the specials are not N either.
        As many onto the specials when they count N too is a tie: the reads
        are vetoed, the episodes either numbering gives are `tied`, and only
        the run's count over an entry's window places it. A lower version of
        a member reads as the member does.
        """

        if not self.map_known:
            return
        for run in _runs(list(self.readings), self.batch.parsed):
            width = len(run.members)
            if run.numbers != tuple(range(1, width + 1)):
                continue
            infos = [self.batch.parsed.get(name) for name in run.names]
            if any(info is None or info.episode_numbers for info in infos):
                continue
            readings = [self.readings[name] for name in run.names]
            if any(len(r.resolved) > 1 for r in readings):
                continue
            read = [self.key_by_id[r.resolved[0]].season for r in readings if r.resolved]
            seasons = {season for season in read if season != 0}
            if len(seasons) != 1:
                continue
            season = seasons.pop()
            if self.season_counts.get(season) != width:
                continue
            into_season = read.count(season)
            onto_specials = read.count(0)
            if not onto_specials or into_season < onto_specials:
                continue
            if into_season == onto_specials and self.season_counts.get(0) == width:
                self.tie(run, season)
            else:
                self.reread(run, season)

    def tie(self, run: _Run, season: int) -> None:
        """Veto the run's reads and record the episodes each name may be under either numbering."""

        for numbered in run.whole:
            keys = (EpisodeKey(season, numbered.number), EpisodeKey(0, numbered.number))
            ids = tuple(i for key in keys if (i := self.scope.id_by_key.get(key)))
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True, tied=ids)

    def reread(self, run: _Run, season: int) -> None:
        """Read every name of the run as the season's episode of its number."""

        for numbered in run.whole:
            ep_id = self.scope.id_by_key.get(EpisodeKey(season, numbered.number))
            if ep_id:
                inside = (ep_id,) if ep_id in self.resolved else ()
                self.readings[numbered.name] = _Reading(
                    (ep_id,), inside, complete=True, borrowed=True, vetoed=False, corroborated=True
                )

    def veto(self, run: _Run) -> None:
        """Veto every reading of the run, its displaced versions included."""

        for numbered in run.whole:
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True)

    def window(self) -> list[int]:
        """The leftover ids: the scope's resolved set minus every id used so far, scope order."""

        return [i for i in self.scope.resolved if i and i not in self.used]

    def run_window(self) -> _RunWindow | None:
        """The window as consecutive episodes in airing order, else None.

        One season's episode numbers, else the series' absolute numbers when
        every slot carries one (an entry holding a special TVDB interleaved,
        or two seasons' cours). Never trusts the scope's order: the ids are
        re-sorted by their keys.
        """

        window = self.window()
        if len(window) < 2:
            return None
        keyed = sorted((self.key_by_id[ep_id], ep_id) for ep_id in window if ep_id in self.key_by_id)
        if len(keyed) != len(window):
            return None
        seasons = {key.season for key, _ in keyed}
        episodes = [key.episode for key, _ in keyed]
        if len(seasons) == 1 and _consecutive(episodes):
            return _RunWindow(seasons.pop(), tuple(ep_id for _, ep_id in keyed), tuple(episodes))
        absolute = sorted((self.absolute_of[ep_id], ep_id) for ep_id in window if ep_id in self.absolute_of)
        if len(absolute) != len(window) or not _consecutive([number for number, _ in absolute]):
            return None
        return _RunWindow(None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute))

    def remaining(self) -> list[str]:
        """The names without a verdict yet, batch order."""

        return [name for name in self.readings if name not in self.verdicts]

    def spans_multiple(self, info: ParsedFileInfo) -> bool:
        """Whether the file plausibly holds more than one episode: the name's claim, or a multi-pair series match.

        Cardinality, not identity: a file Sonarr matched to two episodes, placed as one, is a structural
        half-import. A full-season read (a name with no episode token) counts only when that season
        reaches into the entry, never when it names another season.
        """

        if _claims_several(info):
            return True
        pairs = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
        if len(pairs) <= 1:
            return False
        if not info.full_season:
            return True
        return any(
            self.scope.id_by_key.get(season_episode_key(season, episode)) in self.resolved for season, episode in pairs
        )

    def settled_elsewhere(self, run: _Run) -> bool:
        """Whether Sonarr read the run whole into distinct episodes outside the scope: another season's, or the other cour's.

        A tie's run is elsewhere when neither numbering reaches the scope.
        """

        readings = [self.readings[name] for name in run.names]
        if all(r.tied is not None for r in readings):
            return not any(i in self.resolved for r in readings for i in r.tied or ())
        if not all(r.outside and r.corroborated and len(r.resolved) == 1 for r in readings):
            return False
        return len({r.resolved[0] for r in readings}) == len(readings)

    def covering(self, runs: Iterable[_Run], window: _RunWindow) -> list[_Run]:
        """The `1..N` runs for the whole season a strict slice window belongs to: N members over fewer slots."""

        if window.season is None:
            return []
        count = self.season_counts.get(window.season, 0)
        return _from_one(runs, count) if count > len(window.ids) else []

    def covering_refused(self, window: _RunWindow) -> bool:
        """Whether a season run's count says nothing about the window.

        A specials window (a season run as wide as the specials count is a
        coincidence), or a season whose numbering has a gap (the run counts
        on past it).
        """

        season = window.season
        if not season:  # the specials (an absolute window has no covering run)
            return True
        count = self.season_counts.get(season, 0)
        return any(EpisodeKey(season, n) not in self.scope.id_by_key for n in range(1, count + 1))

    def seasons_read(self, run: _Run) -> set[int]:
        """The seasons Sonarr read the run's members into."""

        return {self.key_by_id[ep_id].season for name in run.names for ep_id in self.readings[name].resolved}

    def read_elsewhere(self, run: _Run, season: int | None) -> bool:
        """Whether Sonarr read a member into another season than the window's: the reads dispute the run's count."""

        return any(read != season for read in self.seasons_read(run))

    def title_evidence(self, run: _Run, window_set: frozenset[int]) -> _TitleEvidence:
        """Count the run's tails naming an episode inside the window against those naming one outside it.

        A tail names an episode when the episode's title opens it (the rest
        is quality junk). A tail naming episodes on both sides counts for
        neither.
        """

        inside = outside = 0
        for member in run.members:
            words = tuple(_word_list(member.tail))
            if not words:
                continue
            sides = {
                episode.ep_id in window_set for episode in self.titled if words[: len(episode.words)] == episode.words
            }
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return _TitleEvidence(inside, outside)

    def place(self, name: str, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and take its ids out of the window."""

        self.verdicts[name] = Placement(name, tuple(ids), verdict)
        self.used.update(ids)

    def set_aside(self, name: str, verdict: PlacementVerdict) -> None:
        """Record a verdict that carries no ids (held, excluded, or skipped)."""

        self.verdicts[name] = Placement(name, (), verdict)

    def set_aside_each(self, numbered: Iterable[_Numbered], verdict: PlacementVerdict) -> None:
        """Record one id-less verdict for each numbered name."""

        for one in numbered:
            self.set_aside(one.name, verdict)

    def finish(self) -> EpisodeAssignment:
        """Classify what is still open, then fold the verdicts in batch order."""

        # An id is a proven duplicate's when its holder reads there too: placed this batch (no pass places
        # where a keyed open file resolves without evidence), or a seeded name whose own parse resolves
        # there. A seed that landed a file positionally is a disagreement the caller reports, not a verdict.
        proven = (self.used - set(self.scope.used)) | self.evidenced
        for name in self.remaining():
            reading = self.readings[name]
            if reading.complete and not reading.vetoed and reading.inside == reading.resolved:
                # Resolves cleanly inside the set: the exact pass left it only because its episode is taken.
                taken = [ep_id for ep_id in reading.inside if ep_id in self.used]
                duplicate = bool(taken) and all(ep_id in proven for ep_id in taken)
                self.set_aside(name, PlacementVerdict.DUPLICATE if duplicate else PlacementVerdict.SKIPPED)
            elif reading.outside and self.map_known:
                self.set_aside(name, PlacementVerdict.FOREIGN)
            else:
                self.set_aside(name, PlacementVerdict.SKIPPED)
        return EpisodeAssignment(tuple(self.verdicts[name] for name in self.readings))


def _pass_release_run(state: _Placer) -> None:
    """Judge the release's own numbering against Sonarr's reading of its members.

    Stands down unless the window is consecutive episodes (one season's, or
    the series' absolutes across several) and one run is left to index it.
    The candidates, by tier: runs numbered `1..N` for its width N, runs
    numbered exactly as its episodes (a split cour's second half), runs
    numbered `1..N` for the whole season a slice window belongs to, and N
    consecutive numbers from anywhere when no member's reading is complete
    and no seed owns part of the scope (a release numbering the whole
    series across its seasons). Among several (a franchise pack), a run
    Sonarr read whole elsewhere stands aside for the rest, then the episode
    titles the members carry name one, then an AniList title does, then the
    highest tier holds. Several left is none, and the other count legs
    stand down too. With any parse in the batch unknown the members are HELD (no
    later pass may place what this one could not judge). Refuses (the exact
    pass proceeds) on `_run_refused`, on a covering run whose count says
    nothing about the window, on a covering run Sonarr read into another
    season, or on `_pick_refused`. Every refusal stands the other count
    legs down too, and a disputed covering run Sonarr did not read whole into
    one other season is vetoed before any pick. A coherent reading (every
    member one distinct id inside the window) stands. Otherwise Sonarr's reading is incoherent (a
    TVDB special shifted its match, the pairs point outside, the keys are
    bogus, or it read nothing) and the run indexes the window. A
    whole-season run's members past a slice window are the other slice's.
    """

    if not state.map_known:
        return
    window = state.run_window()
    if window is None:
        return
    width = len(window.ids)
    window_set = frozenset(window.ids)
    runs = _runs(state.remaining(), state.batch.parsed)
    covering = state.covering(runs, window)
    # Reads into another season dispute a covering run's count, pick or not: read whole into one other
    # season, it is that season's (its members foreign), else its files are nowhere.
    disputed = [run for run in covering if state.read_elsewhere(run, window.season)]
    for run in disputed:
        if not (state.settled_elsewhere(run) and len(state.seasons_read(run)) == 1):
            state.veto(run)
    tiers = (
        _from_one(runs, width),
        _fitting(runs, window.numbers),
        covering,
        # A window a seed already took part of fits a run from anywhere by chance, never by count.
        [
            run
            for run in runs
            if not state.used and run.consecutive(width) and not any(state.readings[n].complete for n in run.names)
        ],
    )
    candidates = list(dict.fromkeys(run for tier in tiers for run in tier))
    state.count_legs_barred = len(candidates) > 1
    if state.count_legs_barred:
        candidates = [run for run in candidates if not state.settled_elsewhere(run)] or candidates
    if (
        len(candidates) > 1
        and len(selected := [r for r in candidates if state.title_evidence(r, window_set).selects]) == 1
    ):
        candidates = selected
    if len(candidates) > 1 and (named := _titled([run.prefix for run in candidates], state.scope.names)) is not None:
        candidates = [candidates[named]]
    if len(candidates) > 1:
        candidates = next(kept for tier in tiers if (kept := [run for run in tier if run in candidates]))
    if len(candidates) != 1:
        return
    run = candidates[0]
    if not state.batch.all_parses_known:
        state.set_aside_each(run.whole, PlacementVerdict.HELD)
        return
    covers = run in covering
    refused = (
        _run_refused(state, run, window)
        or (covers and state.covering_refused(window))
        or run in disputed
        or _pick_refused(state, run, runs, window_set)
    )
    if refused:
        state.count_legs_barred = True
        return
    # A whole-season run over a slice window places the slice's numbers; the rest is the other slice's.
    members = tuple(m for m in run.members if m.number in window.numbers) if covers else run.members
    readings = [state.readings[member.name] for member in members]
    coherent = all(
        r.complete and not r.vetoed and len(r.resolved) == 1 and r.resolved[0] in window_set for r in readings
    ) and len({r.resolved[0] for r in readings}) == len(readings)
    if coherent:
        return
    for member, ep_id in zip(members, window.ids, strict=True):
        state.place(member.name, [ep_id], PlacementVerdict.RELEASE_RUN)
    for member in run.members:
        if member not in members:
            state.set_aside(member.name, PlacementVerdict.FOREIGN)
    state.set_aside_each(run.superseded, PlacementVerdict.DUPLICATE)


def _run_refused(state: _Placer, run: _Run, window: _RunWindow) -> bool:
    """Whether the batch proves the run does not own the window whole."""

    window_set = set(window.ids)
    for name in run.names:
        info = state.batch.parsed.get(name)
        # Sonarr matching a member to several episodes is the incoherence the
        # run overrides, and the listing's count backs the run.
        if info is None or _claims_several(info):
            return True
        reading = state.readings[name]
        keyed_outside = not reading.borrowed and reading.complete and not any(i in window_set for i in reading.resolved)
        # A member named for another season of the series overrides only a
        # specials window: a torrent mislisted on a sequel entry never imports onto it.
        if keyed_outside and window.season != 0:
            return True
    members_set = {numbered.name for numbered in run.whole}
    for name in state.remaining():
        if name in members_set:
            continue
        reading = state.readings[name]
        if any(i in window_set for i in reading.tied or ()):
            return True
        if not reading.complete or reading.vetoed:
            continue
        if reading.resolved and all(i in window_set for i in reading.resolved):
            return True
    return False


def _pick_refused(state: _Placer, run: _Run, runs: Sequence[_Run], window_set: frozenset[int]) -> bool:
    """Whether the names put the window's files elsewhere.

    The pick's episode titles name only other episodes, or an AniList title
    names other runs of the batch (ones Sonarr did not read whole elsewhere)
    and not the pick. Either refuses, never promotes.
    """

    if state.title_evidence(run, window_set).vetoes:
        return True
    open_runs = [candidate for candidate in runs if not state.settled_elsewhere(candidate)]
    named = _named([candidate.prefix for candidate in open_runs], state.scope.names)
    return bool(named) and (run not in open_runs or open_runs.index(run) not in named)


def _pass_exact(state: _Placer) -> None:
    """Place every open file whose reading resolves cleanly inside the scope onto unused ids.

    Two files resolving to one episode are judged by `_Reading.rank`, batch
    order within a rank: a "17.5 (S00E01)" beats the "- 17" whose match
    Sonarr shifted onto the same special, while a "- 08" beats an "- ED"
    whose CRC tag parsed as its key. Within that, a "- 09v2" beats its
    "- 09". The loser is left over (a duplicate).
    """

    for name in sorted(state.remaining(), key=lambda name: (state.readings[name].rank, -_version(name))):
        reading = state.readings[name]
        if not reading.complete or reading.vetoed or reading.inside != reading.resolved:
            continue
        if any(i in state.used for i in reading.inside):
            continue
        state.place(name, reading.inside, PlacementVerdict.EXACT)


def _pass_counted(state: _Placer) -> None:
    """The count legs over the open files and the window: absolute zip, else single file, else ordered zip."""

    # A file reading wholly outside the scope is another slice's: it neither
    # takes a leftover id nor blocks the count for the files that could.
    open_names = [name for name in state.remaining() if not state.readings[name].outside]
    window = state.window()
    if not open_names or not window:
        return
    parsed = state.batch.parsed

    abs_by_file: dict[str, int] = {}
    for name in open_names:
        info = parsed.get(name)
        if info is not None and len(info.absolute_episode_numbers) == 1:
            abs_by_file[name] = info.absolute_episode_numbers[0]
    # The restart-numbering tell is a BATCH property, counting every absolute
    # of every parse supplied - seeded files included, or a v1 placed on an
    # earlier poll would hide its v2 from this leg. Deduped per parse: the
    # tell is two FILES sharing an absolute, not junk repeats within one.
    batch_absolutes = [
        number
        for info in parsed.values()
        if info is not None
        for number in dict.fromkeys(info.absolute_episode_numbers)
    ]
    # A parse the caller couldn't get (None), or the offline regex stand-in
    # for one (blind to absolutes: "S01E12 - 12" would launder its lost 12),
    # may be hiding a duplicate - the tell's input is incomplete, so the leg
    # fails CLOSED, the same posture a hiccuped leftover gets from the count.
    if (
        abs_by_file
        and not state.count_legs_barred
        and state.batch.all_parses_known
        and len(abs_by_file) == len(open_names)  # every leftover has one absolute
        and len(abs_by_file) == len(window)  # 1:1 with the leftover ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restart numbering)
    ):
        for name, _abs in sorted(abs_by_file.items(), key=lambda kv: kv[1]):
            state.place(name, [window.pop(0)], PlacementVerdict.ABSOLUTE)
        return

    if len(window) == 1:
        # Degenerate positional: one leftover episode, and a leftover file
        # Sonarr SAW and found no number in, or only a provably-bogus key
        # that exists nowhere in the series (a None parse is no evidence at
        # all, and multi-episode evidence would half-import, so both refuse).
        # The sole such file is that episode. Among several, once every parse
        # is known, the one an AniList title names is, never an extra. A tie's
        # file is neither: its number is the season's or the specials'.
        numberless = [
            name
            for name in open_names
            if (info := parsed.get(name)) is not None
            and (_has_no_signal(info) or (state.map_known and _signal_is_bogus(info, state.scope.id_by_key)))
            and not state.spans_multiple(info)
            and state.readings[name].tied is None
        ]
        if len(open_names) == 1 and numberless:
            state.place(numberless[0], [window[0]], PlacementVerdict.SINGLE)
            return
        if state.batch.all_parses_known:
            episodic = [name for name in numberless if not _extras_named(name)]
            if (named := _titled([_stem(name) for name in episodic], state.scope.names)) is not None:
                state.place(episodic[named], [window[0]], PlacementVerdict.TITLED)
                return

    if (
        len(open_names) > 1
        and not state.count_legs_barred
        and len(open_names) == len(window)
        and len(open_names) == len(parsed)
        and not state.verdicts
        and not state.scope.used
        and all(
            (info := parsed.get(name)) is not None
            and not info.offline
            and _has_no_signal(info)
            and not state.spans_multiple(info)
            for name in open_names
        )
    ):
        # Pristine numberless batch: the parse-map equality proves NOTHING in
        # the batch was placed, held, or seeded (a mixed batch never zips, so
        # an extra can never fill a missing episode's slot), counts match 1:1,
        # and every parse is a real numberless one. Order is the only signal
        # left: zip name order onto airing order (the "Special 1..N" shape).
        for name, ep_id in zip(sorted(open_names, key=_natural_key), window, strict=True):
            state.place(name, [ep_id], PlacementVerdict.ORDERED)


def _pass_numbered_run(state: _Placer) -> None:
    """Index a consecutive window by the one `1..N` run among the files Sonarr could not read at all.

    Unlike the ordered zip this survives a MIXED batch (a specials run beside
    a placed season pack). Blind means the reading resolved nothing, the name
    carries no `(season, episode)`, and the match spans no episodes: a file
    Sonarr placed anywhere in the series merely fell outside our scope, and a
    positional run must never re-home it.
    """

    if not state.map_known or not state.batch.all_parses_known or state.count_legs_barred:
        return
    window = state.run_window()
    if window is None:
        return
    parsed = state.batch.parsed
    blind = [
        name
        for name in state.remaining()
        if not state.readings[name].resolved
        and (info := parsed.get(name)) is not None
        and not info.offline
        and not info.episode_numbers
        and not state.spans_multiple(info)
    ]
    fits = _from_one(_runs(blind, parsed), len(window.ids))
    if len(fits) != 1:
        return
    run = fits[0]
    for name, ep_id in zip(run.names, window.ids, strict=True):
        state.place(name, [ep_id], PlacementVerdict.NUMBERED_RUN)
    state.set_aside_each(run.superseded, PlacementVerdict.DUPLICATE)


def assign_episode_ids(
    batch: PlacementBatch,
    scope: TargetScope,
) -> EpisodeAssignment:
    """Map a torrent's files to OUR resolved episode ids. Names never override.

    The resolved set (`scope.resolved`, season-sorted, lifted from the
    add-flow `ep_list`) is authoritative. A release's own numbering is only ever
    used to *index into* it, never to decide identity. One reading per file
    (`_read`), then passes in strict precedence over one state, each placing
    into the ids the earlier ones left:

    1. **Release run:** the batch's one run fitting a consecutive window
       (one season's episodes, or the series' absolutes across several:
       numbered `1..N`, or as the window's own episodes, or `1..N` for the
       whole season a slice window belongs to, or N consecutive numbers
       Sonarr read nothing of) indexes it when Sonarr's reading of the
       members is incoherent (see `_pass_release_run`). A `1..N` run Sonarr
       matched into the one season of exactly N episodes, a few members
       shifted onto its specials, is first re-read as that season's own
       numbering. Members are HELD, not placed, while any parse in the batch
       is unknown.
    2. **Exact (season, episode):** a file whose reading resolves cleanly inside
       the resolved set is placed there (a name Sonarr just couldn't match, a
       per-season multi-season pack, an absolute-only name borrowing Sonarr's
       matched pair under the same in-set scoping). With NO resolved set
       (`scope.unscoped`) the name-parsed keys place against the live series
       map directly, so a correctly-named file still imports rather than sticking.
    3. **Absolute index:** the leftovers zip onto the leftover ids by absolute
       number, ONLY when every leftover carries a single absolute, the counts
       match 1:1, every parse in the batch is known, and no two files ANYWHERE
       in the batch share an absolute (the restart-numbering tell, counted over
       seeded files too).
    4. **Single file:** one leftover file onto one leftover id, when the name
       carries no number at all (or only a provably-bogus key, one missing the
       WHOLE series map) and Sonarr's matched evidence spans no episodes.
    5. **Ordered zip:** a pristine numberless batch (every parse known, real,
       numberless, single-span, covering EXACTLY the leftover files) zips
       natural name order onto the leftover ids 1:1 (the "Special 1..N" shape).
    6. **Numbered run:** the one `1..N` run among the files Sonarr read nothing
       from indexes a consecutive window of that width, mixed batch or not.
    7. **Classify:** what is left resolves inside the set onto a taken episode
       (`DUPLICATE`), or cleanly and entirely outside it (`FOREIGN`), or is
       simply `SKIPPED`. The caller warns on skips and records exclusions,
       never guesses.

    Args:
        batch: The files to place plus the WHOLE batch's parses (the parses
            may cover more files than are placed, so the shared-absolute tell
            scans every parse supplied so an already-seeded file still exposes
            a duplicate).
        scope: The full resolved set, the seed-claimed ids, and the series
            map. An empty resolved set means no scope at all (the seed-claimed
            ids keep a fully seeded record from masquerading as one). An empty
            series map refuses every map-dependent verdict.

    Returns:
        One `Placement` per distinct file, batch order.
    """

    state = _Placer.start(batch, scope)
    _pass_release_run(state)
    _pass_exact(state)
    _pass_counted(state)
    _pass_numbered_run(state)
    return state.finish()


@dataclass(frozen=True)
class CandidateFile:
    """An on-disk manual-import candidate, reduced to what planning needs.

    Built by the strategy from one raw ManualImportResource.
    """

    basename: str
    """The normalized match key against our authoritative map."""

    path: str
    """What we POST."""

    quality: QualityModel | None
    """Reused if our own quality parse comes up empty."""

    is_sample: bool
    """Folds Sonarr's per-file sample rejection into the plan."""

    is_already_imported: bool
    """Folds Sonarr's per-file already-imported rejection into the plan."""


class ImportAction(StrEnum):
    """What `plan_import_files` decided for one entry in OUR map.

    A `StrEnum` (so each member IS its rendered word, matching the
    `PendingState` / `QueueVerdict` / `EpisodeFileStatus`
    style) - the consumer branches on a typed value instead of a magic string.
    Only `IMPORT` and `MISSING` drive behavior. The three "nothing to import
    for this file" members are kept distinct purely for reporting.
    """

    IMPORT = "import"
    """POST a manual import for this file."""

    SKIP_DONE = "skip_done"
    """Not needed (every target already holds a recommended file), with no Sonarr rejection."""

    SAMPLE = "sample"
    """A sample (never our intended file)."""

    ALREADY = "already"
    """Not needed, and Sonarr flagged an already-imported rejection."""

    MISSING = "missing"
    """Our map intends this file but it isn't on disk (surfaced, never silently skipped)."""


@dataclass(frozen=True)
class ImportDecision:
    """One decision per entry in OUR authoritative map (the source of truth).

    Candidates only supply the on-disk `path` and rejection flags (folded into `action`).
    """

    basename: str
    action: ImportAction
    path: str | None
    """The on-disk path, supplied by the matched candidate."""

    quality: QualityModel | None
    episode_ids: list[int]
    """The episode assignment, strictly from our map, never the candidate's own parse."""


def plan_import_files(
    authoritative_map: dict[str, list[int]],
    candidates_by_basename: dict[str, CandidateFile],
    needing_import: set[int],
) -> list[ImportDecision]:
    """Decide, per intended file, whether/how to import it - strictly from our map.

    Iterates OUR map (never the candidates): a file Sonarr found that isn't in our
    map is never imported, and a file our map intends that isn't on disk is
    surfaced as `missing` (never silently skipped). For a present file both
    invariants are honored via `needing_import` (the non-recommended target
    set): a file whose every episode already holds a recommended release is
    `skip_done` (not overwritten). Otherwise it is imported for exactly its
    needing-import episodes.

    `needing_import` (derived from the EPISODE FILES via
    `EpisodeSnapshot.statuses`) - not Sonarr's per-candidate already-imported
    rejection - is authoritative for whether we still want a file. Sonarr raises
    that rejection whenever the episode already holds *any* file on disk, including
    a non-recommended or unidentifiable-group one we flagged as still-needing
    replacement. Honoring it as a skip there is the grab-then-skip bug (we grab a
    missing-group replacement, then Sonarr's "already imported" makes us skip
    importing it). So `is_already_imported` only yields `already` when NONE of
    the file's episodes still need us (every target already holds a recommended
    file - Sonarr and our episode-file check agree). When a target still needs us
    we import over it, as the never-skip invariant requires. `is_sample` still
    wins (a sample is never our intended file).

    Args:
        authoritative_map: normalized basename -> our ids.
        candidates_by_basename: on-disk files by key.
        needing_import: episode ids still needing our file (from
            `targets_needing_import`).

    Returns:
        One decision per map entry, in map order.
    """

    decisions: list[ImportDecision] = []
    for basename, ep_ids in authoritative_map.items():
        candidate = candidates_by_basename.get(basename)
        if candidate is None:
            decisions.append(ImportDecision(basename, ImportAction.MISSING, None, None, ep_ids))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(basename, ImportAction.SAMPLE, candidate.path, None, []))
            continue
        import_ids = [i for i in ep_ids if i in needing_import]
        if not import_ids:
            # Nothing of ours still needs this file. Sonarr's already-imported
            # rejection and our episode-file done-check agree here, so report the
            # more specific `ALREADY` when Sonarr flagged it, else `SKIP_DONE`.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(
                ImportDecision(basename, action, candidate.path, None, ep_ids),
            )
            continue
        # A target still needs our file: import it over whatever is there, even
        # when Sonarr raised an already-imported rejection (that on-disk file is
        # the non-recommended / unidentifiable one we grabbed to replace).
        decisions.append(
            ImportDecision(
                basename,
                ImportAction.IMPORT,
                candidate.path,
                candidate.quality,
                import_ids,
            ),
        )
    return decisions


# Filename source tokens -> QualitySource, ordered most-specific first so a
# "BluRay Remux" name resolves to BLURAY_RAW (not BLURAY), "BD" counts as BluRay,
# and "WEB-DL" wins over a bare "WEB". A token that matches nothing leaves the
# source axis undetermined (None) - it is NEVER defaulted to WEB here. The
# configured default fills it.
_SOURCE_PATTERNS: list[tuple[re.Pattern[str], QualitySource]] = [
    (re.compile(r"remux", re.IGNORECASE), QualitySource.BLURAY_RAW),
    (re.compile(r"blu-?ray|\bbd\b", re.IGNORECASE), QualitySource.BLURAY),
    (re.compile(r"web-?dl", re.IGNORECASE), QualitySource.WEB),
    (re.compile(r"webrip", re.IGNORECASE), QualitySource.WEBRIP),
    (re.compile(r"hdtv", re.IGNORECASE), QualitySource.TELEVISION),
    (re.compile(r"\bdvd\b", re.IGNORECASE), QualitySource.DVD),
    (re.compile(r"\bweb\b", re.IGNORECASE), QualitySource.WEB),
]

_RESOLUTION_PATTERN: re.Pattern[str] = re.compile(r"(2160|1080|720|480)p", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedQuality:
    """Quality as two independent axes: `source` and `resolution`.

    Either axis is `None` when it could not be authoritatively determined, which
    is what lets the quality decision layer the axes across Sonarr's parse, our
    filename parse, and the configured default (each fills only the axes the
    higher-precedence layers left `None`). The resulting `(source, resolution)`
    pair is matched against Sonarr's quality definitions to pick the real quality.
    """

    source: QualitySource | None = None
    resolution: int | None = None


def parse_quality_from_filename(filename: str) -> ParsedQuality:
    """Best-effort `(source, resolution)` parse of a SeaDex filename.

    Detects a resolution (`2160`/`1080`/`720`/`480`) and a source
    (Remux, BluRay, WEB-DL, WEBRip, WEB, HDTV, DVD), case-insensitively and
    independently. Either axis is `None` when not found - notably an
    unrecognized source is left `None` (NOT defaulted to WEB), so the configured
    default can fill it rather than the file being silently mislabeled.

    Args:
        filename: The SeaDex filename (or full path, only the text matched).

    Returns:
        The parsed axes. Either may be `None`.
    """

    res_match = _RESOLUTION_PATTERN.search(filename)
    resolution = int(res_match.group(1)) if res_match is not None else None

    source: QualitySource | None = None
    for pattern, candidate in _SOURCE_PATTERNS:
        if pattern.search(filename):
            source = candidate
            break
    return ParsedQuality(source=source, resolution=resolution)


def quality_axes_from_model(model: QualityModel | None) -> ParsedQuality:
    """The `(source, resolution)` axes of a Sonarr `QualityModel`.

    Reads the canonical schema path `model.quality.source` /
    `model.quality.resolution` (every field defaults to None on the partial
    models the helpers build). An `"unknown"` source or a `0`/absent
    resolution maps to `None` (undetermined), so an unparsed candidate
    cleanly yields `ParsedQuality()` and falls through to the next
    precedence layer.

    Args:
        model: A candidate's in-context quality model.

    Returns:
        The structured axes Sonarr determined, each possibly None.
    """

    if model is None:
        return ParsedQuality()
    quality = model.quality  # an empty/null wire quality already folded to None
    if quality is None:
        return ParsedQuality()
    resolution = quality.resolution
    if resolution is None or resolution <= 0:
        resolution = None
    return ParsedQuality(source=QualitySource.parse(quality.source), resolution=resolution)


def quality_axes_from_name(
    name: str | None,
    quality_defs: list[QualityDefinition],
) -> ParsedQuality:
    """The `(source, resolution)` axes of a configured default quality NAME.

    Resolves the configured `imports.default_quality` (a Sonarr quality name like
    `"Bluray-2160p"`) to its structured axes by matching it, case-insensitively,
    against the `/api/v3/qualitydefinition` list - so the default contributes a
    real `(source, resolution)` the decision fills gaps from. An unset name, or
    one that matches no definition, yields `ParsedQuality()` (no default).

    Args:
        name: The configured default quality name, if any.
        quality_defs: The `/api/v3/qualitydefinition` list.

    Returns:
        The default's axes, or empty when unset/unmatched.
    """

    if not name:
        return ParsedQuality()
    target = name.casefold()
    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        def_name = quality.name
        if def_name is not None and def_name.casefold() == target:
            return quality_axes_from_model(QualityModel(quality=quality))
    return ParsedQuality()


def derive_languages(
    is_dual_audio: bool,
    dual: list[str],
    single: list[str],
) -> list[str]:
    """Pick the import language list: `dual` when dual-audio, else `single`."""

    return dual if is_dual_audio else single


def _find_definition(
    source: QualitySource,
    resolution: int,
    quality_defs: list[QualityDefinition],
) -> Quality | None:
    """The nested `Quality` whose `(source, resolution)` matches, or None.

    Scans the `/api/v3/qualitydefinition` list for the definition whose nested
    quality has the given structured source and resolution. `(source, resolution)`
    is unique across Sonarr's standard definitions (the only near-collision, Raw-HD
    vs HDTV-1080p, differs by source), so the pair identifies the quality without
    ever matching on its display name.
    """

    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        if quality.resolution == resolution and QualitySource.parse(quality.source) is source:
            return quality
    return None


# A RAW source degrades to its base when no remux/raw definition exists at that
# resolution (try the raw definition first, then this base).
_RAW_DOWNGRADE: dict[QualitySource, QualitySource] = {
    QualitySource.BLURAY_RAW: QualitySource.BLURAY,
    QualitySource.TELEVISION_RAW: QualitySource.TELEVISION,
}


def _candidate_revision(candidate_model: QualityModel | None) -> Revision:
    """The candidate's revision (proper/repack), or a fresh `version 1` default."""

    if candidate_model is not None and candidate_model.revision is not None:
        return candidate_model.revision
    return Revision(version=1, real=0, isRepack=False)


def resolve_quality(
    sonarr: ParsedQuality,
    ours: ParsedQuality,
    default: ParsedQuality,
    quality_defs: list[QualityDefinition],
    candidate_model: QualityModel | None,
) -> QualityModel:
    """Resolve the final manual-import `QualityModel` - never omitted.

    The source and resolution axes are decided independently, each taking the
    first authoritative value in precedence order: Sonarr's parse, then our
    filename parse, then the configured default. When both axes are determined the
    quality definition matching the `(source, resolution)` pair is emitted, so
    the payload always carries a quality Sonarr actually defines (a valid id+name).
    A determined `BLURAY_RAW`/`TELEVISION_RAW` with no matching remux/raw
    definition at that resolution gracefully downgrades to `BLURAY`/`TELEVISION`
    rather than failing.

    Crucially this never returns `None` and the caller never omits the quality:
    omitting it is exactly what made Sonarr crash in
    `FileNameBuilder.AddQualityTokens`. When nothing resolves, Sonarr's own
    candidate model (valid by construction) is re-emitted verbatim. Only if the
    candidate carries no quality at all is an explicit `Unknown` synthesized.

    Args:
        sonarr: Axes from Sonarr's candidate parse (highest).
        ours: Axes from our filename parse.
        default: Axes from the configured default quality.
        quality_defs: The `/api/v3/qualitydefinition` list to match against.
        candidate_model: Sonarr's in-context model, the
            last-resort verbatim fallback.

    Returns:
        The quality to POST. Never omitted.
    """

    # Invariant: the import payload always carries a quality key - omitting it
    # crashes Sonarr in FileNameBuilder.AddQualityTokens (observed on Sonarr 4.x).
    source = sonarr.source or ours.source or default.source
    resolution = sonarr.resolution or ours.resolution or default.resolution
    revision = _candidate_revision(candidate_model)

    if source is not None and resolution is not None:
        quality = _find_definition(source, resolution, quality_defs)
        base = _RAW_DOWNGRADE.get(source)
        if quality is None and base is not None:
            quality = _find_definition(base, resolution, quality_defs)
        if quality is not None:
            return QualityModel(quality=quality, revision=revision)

    # No confident match: re-emit Sonarr's own candidate (valid by construction)
    # rather than omit the quality, else synthesize an explicit Unknown. An
    # EMPTY candidate quality already folded to None at the parse boundary, so
    # this None test guards the already-folded empty quality.
    if candidate_model is not None and candidate_model.quality is not None:
        return candidate_model
    unknown = Quality(id=0, name="Unknown", source="unknown", resolution=0)
    return QualityModel(quality=unknown, revision=revision)


def resolve_language_objects(
    names: list[str],
    lang_defs: list[Language],
) -> list[Language]:
    """Resolve configured language names to Sonarr `{id, name}` objects.

    Case-insensitive match against the `/api/v3/language` list, in request
    order. A name with no match is dropped rather than failing the import.
    """

    by_name: dict[str, Language] = {
        name.casefold(): definition for definition in lang_defs if (name := definition.name) is not None
    }
    resolved: list[Language] = []
    for name in names:
        definition = by_name.get(name.casefold())
        if definition is not None:
            # Re-built fresh with BOTH fields set, so the exclude_unset write
            # dump always carries them (a null id included).
            resolved.append(Language(id=definition.id, name=definition.name))
    return resolved
