"""The wait pass's enums and result types: the wait mode, outcomes and states, the attempt result, download progress."""

import math
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from types import MappingProxyType

from .boundary import coerce_int
from .pending_import import FileEpisodeMap
from .releases import EpisodeKey, OrderOnlyKey


class ImportWaitMode(StrEnum):
    """Controls if and when the manual-import wait/import runs."""

    OFF = "off"
    """Disabled: no waiting, no pending-import records, no manual import."""

    DEFERRED = "deferred"
    """Never wait on a download: record this run's grabs and import earlier runs' finished downloads in one
    pass at the end of the run."""

    BLOCKING = "blocking"
    """Same as `hybrid`."""

    HYBRID = "hybrid"
    """The default: wait at the end of the run for downloads to finish, then import - this run's grabs and
    any download still pending from an earlier run."""


class TorrentStatus(Enum):
    """The result of waiting on a torrent's completion in qBittorrent."""

    COMPLETE = auto()
    """Import now."""

    ERRORED = auto()
    """Leave the pending import stored, retried each run until the torrent recovers or leaves qBittorrent."""

    GONE = auto()
    """The torrent is gone from qBittorrent, so the pending import should be dropped."""


class AttemptKind(Enum):
    """Which kind of attempt a poll makes (see `ImportCompleter.import_completed`)."""

    ORDINARY = auto()
    """An ordinary attempt: a clean `importPending` is Sonarr's while its completed download handling is on."""

    FINAL = auto()
    """The attempt past the stall bound: auto imports past a clean `importPending`, and a result with no
    pause finishes the row."""

    @property
    def at_deadline(self) -> bool:
        """The final attempt for the pending import."""

        return self is AttemptKind.FINAL


class EffectStatus(Enum):
    """One post-import cleanup effect's result (the cleanup contract, never rendered)."""

    DONE = auto()
    """Ran, or the goal was verifiably already met."""

    SKIPPED = auto()
    """Not applicable or left to a sibling: nothing outstanding for THIS pending import."""

    RETRY = auto()
    """Transient, the manager may re-attempt in-run."""

    FAILED = auto()
    """Given up this run, left awaiting cleanup for a later run."""


class CleanupEffect(StrEnum):
    """One post-import cleanup effect. Each member IS its noun in the awaiting-cleanup warning."""

    CATEGORY_MOVE = "category move"
    QUEUE_REMOVAL = "queue removal"


class PendingImportState(StrEnum):
    """The current status of one carried-over pending import, for reporting.

    A `StrEnum` (so each member IS its rendered word) shared by the inline
    report row, the WaitView live region, and the arr run summary
    counters, so one vocabulary describes a carried-over pending import everywhere.
    """

    DOWNLOADING = "queued"
    """Still downloading."""

    DOWNLOADED = "downloaded"
    """The download finished and will be imported by the wait pass."""

    IMPORTED = "imported"
    """Every episode is done. The pending import is dropped, or stays stored only for its post-import
    cleanup."""

    ERRORED = "errored"
    """The download errored in qBittorrent. Left for a later run."""

    GONE = "missing"
    """The torrent is gone from qBittorrent. The pending import is dropped."""


def classify_pending_import(
    wait_outcome: "TorrentStatus | None",
    all_done: bool,
) -> PendingImportState:
    """Reduce a poll's torrent status to a single carried-over `PendingImportState`.

    Args:
        wait_outcome: The torrent's status this
            poll, or `None` while it is still downloading.
        all_done: Whether every intended episode is done
            in Sonarr (the only signal that promotes to `IMPORTED`).
    """

    if wait_outcome is TorrentStatus.GONE:
        return PendingImportState.GONE
    if wait_outcome is TorrentStatus.ERRORED:
        return PendingImportState.ERRORED
    if wait_outcome is None:
        return PendingImportState.DOWNLOADING
    if all_done:
        return PendingImportState.IMPORTED
    return PendingImportState.DOWNLOADED


class PauseReason(Enum):
    """Why a poll waited on Sonarr's work rather than this pending import's (the wait pass pauses for all but NONE)."""

    NONE = auto()
    """Not paused: the wait is this pending import's own and burns its import clock."""

    ISSUED = auto()
    """The auto import's ManualImport command was accepted this poll (paused for once per pending import)."""

    IMPORT = auto()
    """An import is in flight: an auto import from an earlier poll, an unproven one, or Sonarr's own `importing`
    queue record."""

    BUSY = auto()
    """A Sonarr disk command for something else holds the line."""


@dataclass(frozen=True, slots=True)
class HandImportFile:
    """One of the files in a download's folder left for a hand import."""

    name: str
    """Its file name."""

    order_only: OrderOnlyKey | None = None
    """Its key, when only another TVDB order numbers it (`PendingImport.order_only`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LeftForHandImport:
    """The download folder's files left for a hand import, and the pending import's window episodes still needed."""

    files: tuple[HandImportFile, ...] = ()

    missing: tuple[EpisodeKey, ...] = ()
    """The pending import's window episodes still needed that have no file (`StillNeeded.missing`)."""

    untrusted: tuple[EpisodeKey, ...] = ()
    """The pending import's window episodes still needed that have a file (`StillNeeded.untrusted`)."""

    @property
    def names(self) -> tuple[str, ...]:
        """The files' names."""

        return tuple(file.name for file in self.files)

    @property
    def explained_by_order(self) -> bool:
        """Whether the order explains the pending import: files, each with an order-only key, and a needed episode."""

        all_order_only = bool(self.files) and all(file.order_only is not None for file in self.files)
        return all_order_only and bool(self.missing or self.untrusted)


NO_HAND_IMPORT_FILES = LeftForHandImport()
"""No file left for a hand import."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptResult:
    """The result of one attempt (`import_completed`).

    Lets the wait pass tell `imported` (every intended episode is
    done) from `importing` (an import command was accepted but the copy is
    still running).
    """

    all_done: bool
    """Whether every intended episode is done in Sonarr."""

    command_issued: bool
    """Whether a ManualImport command covering this download was accepted."""

    attempted: bool = True
    """False only when no attempt ran (it raised, or no strategy is bound), so the pending import stays as it is."""

    imported_count: int = 0
    """How many of the intended episodes already have a trusted file"""

    file_map_episode_count: int = 0
    """The file map's episodes, 0 means the file map is incomplete (indeterminate)."""

    pause_reason: PauseReason = PauseReason.NONE
    """Why this poll waited on Sonarr's work instead of this pending import's."""

    placements: FileEpisodeMap = field(default_factory=dict[str, tuple[int, ...]])
    """The files Sonarr's history placed for a download no candidate fetch saw (normalized file name -> ids), once
    they are done, copied and wrapped read-only for the pending import store to persist. Empty otherwise."""

    unplaced_files: tuple[str, ...] = ()
    """The unplaced files in the download's folder (one per distinct file name), when that is all the poll
    found. Non-empty only from `nothing_placed()`: `all_done` and `command_issued` False, `pause_reason` NONE."""

    left_for_hand_import: LeftForHandImport = NO_HAND_IMPORT_FILES
    """The files in the download's folder the grab left for a hand import (`PendingImport.hand_import_files`), once
    every placed file is in and the hand import isn't. Its names are non-empty only from `needs_hand_import()`:
    `all_done` and `command_issued` False."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", MappingProxyType(dict(self.placements)))

    @property
    def paused(self) -> bool:
        """Whether this poll waited on Sonarr's work (the wait pass doesn't count the interval against the timeout)."""

        return self.pause_reason is not PauseReason.NONE

    @classmethod
    def imported(cls) -> "AttemptResult":
        """Every intended episode is done, with no auto import command in flight."""

        return cls(all_done=True, command_issued=False)

    @classmethod
    def waiting(cls, *, command_issued: bool = False, pause_reason: PauseReason = PauseReason.NONE) -> "AttemptResult":
        """Not imported yet: poll again until the import timeout."""

        return cls(all_done=False, command_issued=command_issued, pause_reason=pause_reason)

    @classmethod
    def nothing_placed(cls, names: tuple[str, ...]) -> "AttemptResult":
        """Nothing to import or confirm: no file in the folder is placed on an episode."""

        return cls(all_done=False, command_issued=False, unplaced_files=names)

    @classmethod
    def needs_hand_import(cls, left: LeftForHandImport) -> "AttemptResult":
        """Every placed file is in, but files the grab left for a hand import keep the pending import stored."""

        return cls(all_done=False, command_issued=False, left_for_hand_import=left)


NO_ATTEMPT = AttemptResult(attempted=False, all_done=False, command_issued=False)
"""The fail-open attempt result: no attempt ran, so leave the pending import for a later run."""


@dataclass(frozen=True, slots=True)
class ImportProgress:
    """A cheap, read-only count of the done episodes for the wait region's import bar."""

    done: int
    total: int
    determinate: bool
    """True only when the persisted file map covers every intended file, so `done`/`total` are the true full
    set. When False the importing row stays indeterminate (spinner only) and must NOT promote."""

    @property
    def all_done(self) -> bool:
        """Every intended episode is done."""

        return self.determinate and 0 < self.total <= self.done


NO_PROGRESS = ImportProgress(0, 0, determinate=False)
"""The indeterminate zero count a failed or strategy-less poll reports."""


class OutcomeKind(Enum):
    """The wait view's outcome line glyph + color."""

    IMPORTED = ("✔", "ok", "green")
    """The torrent imported."""

    LEFT = ("⚠", "~", "yellow")
    """Left for a later run (a download timeout, an import that isn't done yet, or files only a
    hand import can place). Not a failure, just unfinished."""

    FAILED = ("✖", "x", "bold red")
    """The download errored or vanished from qBittorrent."""

    IN_PROGRESS = ("·", "-", "grey50")
    """Left for the next run on purpose - the deferred wait pass saw it and nothing is wrong."""

    glyph: str
    """The unicode glyph (`✔`/`⚠`/`✖`/`·`)."""

    ascii_glyph: str
    """The ASCII fallback, for dumb terminals / legacy Windows, where `✔` can't be encoded."""

    style: str
    """The rich style its outcome line is colored with."""

    def __init__(self, glyph: str, ascii_glyph: str, style: str) -> None:
        self.glyph = glyph
        self.ascii_glyph = ascii_glyph
        self.style = style

    def glyph_for(self, *, use_unicode: bool) -> str:
        """The outcome line glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.glyph if use_unicode else self.ascii_glyph


class Outcome(Enum):
    """A pending import's outcome in the wait pass, with its rendering vocab."""

    IMPORTED = ("imported", "imported", OutcomeKind.IMPORTED, True)
    MISSING = ("gone", "gone from qBittorrent; title checked again next run", OutcomeKind.FAILED, True)
    DOWNLOAD_ERRORED = ("errored", "download errored; left pending", OutcomeKind.FAILED, False)
    DOWNLOAD_TIMED_OUT = ("timed out", "download timed out; left pending", OutcomeKind.LEFT, False)
    NO_CONTENT_PATH = (
        "no path",
        "complete but no content path reported; left pending",
        OutcomeKind.LEFT,
        False,
    )
    STILL_IMPORTING = ("unfinished", "still importing; left pending", OutcomeKind.LEFT, False)
    SONARR_BUSY = ("sonarr busy", "Sonarr busy with a disk command; left pending", OutcomeKind.LEFT, False)
    NOT_READY = ("not ready", "import not ready; left pending", OutcomeKind.LEFT, False)
    UNMATCHED = ("unmatched", "no file matched an episode; import by hand in Sonarr", OutcomeKind.LEFT, False)
    PARTIAL = (
        "partial",
        "imported in part; import the held-back files by hand in Sonarr",
        OutcomeKind.LEFT,
        False,
    )
    ATTEMPT_FAILED = ("failed", "import attempt failed; left pending", OutcomeKind.LEFT, False)
    NOT_CHECKED = ("not checked", "qBittorrent unreachable; checked again next run", OutcomeKind.LEFT, False)
    IMPORT_IN_PROGRESS = ("in progress", "import in progress; checked again next run", OutcomeKind.IN_PROGRESS, False)
    AWAITING_IMPORT = ("awaiting", "awaiting import; checked again next run", OutcomeKind.IN_PROGRESS, False)
    STILL_DOWNLOADING = ("downloading", "still downloading; checked again next run", OutcomeKind.IN_PROGRESS, False)

    word: str
    """The short outcome line token (every one fits `STATE_WIDTH` = 11)."""

    detail: str
    """The longer human phrase the run report / notification use."""

    kind: OutcomeKind
    """The `OutcomeKind` driving glyph + color + tally."""

    dropped: bool
    """Whether the wait pass removes the pending import from the store on this outcome.
    True for `IMPORTED` (every episode done) and `MISSING` (gone from qBittorrent)."""

    def __init__(
        self,
        word: str,
        detail: str,
        kind: OutcomeKind,
        dropped: bool,
    ) -> None:
        self.word = word
        self.detail = detail
        self.kind = kind
        self.dropped = dropped

    @property
    def style(self) -> str:
        """The rich style for this outcome's line (from its kind)."""

        return self.kind.style

    def glyph(self, *, use_unicode: bool) -> str:
        """The leading outcome line glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.kind.glyph_for(use_unicode=use_unicode)


# Each non-dropped Outcome's tally bucket. Dropped outcomes (IMPORTED, MISSING) leave the store,
# so a wait pass notes every pending import it keeps stored through exactly this table.
PENDING_IMPORT_STATE_FOR_OUTCOME: dict[Outcome, PendingImportState] = {
    Outcome.STILL_DOWNLOADING: PendingImportState.DOWNLOADING,
    Outcome.DOWNLOAD_TIMED_OUT: PendingImportState.DOWNLOADING,
    Outcome.NOT_CHECKED: PendingImportState.DOWNLOADING,
    Outcome.AWAITING_IMPORT: PendingImportState.DOWNLOADED,
    Outcome.IMPORT_IN_PROGRESS: PendingImportState.DOWNLOADED,
    Outcome.STILL_IMPORTING: PendingImportState.DOWNLOADED,
    Outcome.SONARR_BUSY: PendingImportState.DOWNLOADED,
    Outcome.NOT_READY: PendingImportState.DOWNLOADED,
    Outcome.UNMATCHED: PendingImportState.DOWNLOADED,
    Outcome.PARTIAL: PendingImportState.DOWNLOADED,
    Outcome.ATTEMPT_FAILED: PendingImportState.DOWNLOADED,
    Outcome.NO_CONTENT_PATH: PendingImportState.DOWNLOADED,
    Outcome.DOWNLOAD_ERRORED: PendingImportState.ERRORED,
}


# qBittorrent reports a torrent with no meaningful ETA as 8_640_000 seconds
# (100 days), its "infinite" sentinel. Treat it (and anything at/above it) as
# "unknown" rather than rendering a nonsense countdown.
_QBIT_ETA_INFINITE = 8_640_000


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """One info row's sanitized live download progress: download fraction, speed, ETA, and byte counts."""

    progress: float
    speed_bps: int | None
    eta_s: int | None
    bytes_done: int | None
    bytes_total: int | None


_NO_DOWNLOAD_PROGRESS = DownloadProgress(0.0, None, None, None, None)
"""The unread placeholder (no client, or a transient qBittorrent error)."""


@dataclass(frozen=True, slots=True)
class TorrentPoll:
    """One qBittorrent completion poll, with live download progress."""

    outcome: TorrentStatus | None
    """The torrent's status this poll, or None while still downloading (or on a transient qB error)."""

    content_path: str | None
    """The completed download's path (COMPLETE only)."""

    telemetry: DownloadProgress = _NO_DOWNLOAD_PROGRESS
    """The poll's sanitized live download progress (zeroed when nothing was read)."""

    observed: bool = True
    """False when qBittorrent could not actually be read (no client / a transient error)."""

    @property
    def ready_path(self) -> str | None:
        """The completed download's importable path: COMPLETE with a reported content path, else None."""

        if self.outcome is TorrentStatus.COMPLETE and self.content_path:
            return self.content_path
        return None


def download_progress_of(row: object) -> DownloadProgress:
    """Convert one qBittorrent info row to sanitized download progress (fields read best-effort off the row)."""

    frac = _as_float(getattr(row, "progress", None))
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))

    raw_speed = coerce_int(getattr(row, "dlspeed", None))
    speed_bps = raw_speed if raw_speed is not None and raw_speed > 0 else None

    raw_eta = coerce_int(getattr(row, "eta", None))
    eta_s = raw_eta if raw_eta is not None and 0 < raw_eta < _QBIT_ETA_INFINITE else None

    raw_total = coerce_int(getattr(row, "size", None))
    bytes_total = raw_total if raw_total is not None and raw_total > 0 else None
    raw_done = coerce_int(getattr(row, "completed", None))
    bytes_done = raw_done if raw_done is not None and raw_done > 0 else None
    if bytes_done is not None and bytes_total is not None:
        bytes_done = min(bytes_done, bytes_total)
    return DownloadProgress(frac, speed_bps, eta_s, bytes_done, bytes_total)


def torrent_finished(row: object) -> bool:
    """Whether a qBittorrent info row's download is done: complete by its state, or at full progress."""

    complete = getattr(getattr(row, "state_enum", None), "is_complete", False)
    return complete is True or download_progress_of(row).progress >= 1.0


def torrent_errored(row: object) -> bool:
    """Whether a qBittorrent info row is in an error state, missing files included."""

    return getattr(getattr(row, "state_enum", None), "is_errored", False) is True


def _as_float(value: object) -> float | None:
    """Best-effort float, or None for a non-numeric / NaN value."""

    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return None if math.isnan(parsed) else parsed
    return None
