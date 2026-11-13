"""The wait pass's enums and result types: the wait mode, poll outcomes and states, the probe, and torrent telemetry."""

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
    """Leave the record pending, retried each run until the torrent recovers or leaves qBittorrent."""

    GONE = auto()
    """The torrent is gone from qBittorrent, so the record should be dropped."""


class AttemptKind(Enum):
    """Which kind of import attempt a poll is (see `ImportCompleter.import_completed`)."""

    ORDINARY = auto()
    """An ordinary poll: a clean `importPending` is Sonarr's while its completed download handling is on."""

    FINAL = auto()
    """The attempt past the stall bound: steps in past a clean `importPending`, and an uncredited
    result graduates the row."""

    @property
    def at_deadline(self) -> bool:
        """The final attempt for the record."""

        return self is AttemptKind.FINAL


class EffectStatus(Enum):
    """One post-import cleanup effect's result (the retire contract, never rendered)."""

    DONE = auto()
    """Ran, or the goal was verifiably already met."""

    SKIPPED = auto()
    """Not applicable or deferred to a sibling: nothing outstanding for THIS record."""

    RETRY = auto()
    """Transient, the manager may re-attempt in-run."""

    FAILED = auto()
    """Given up this run, kept for a later run."""


class CleanupEffect(StrEnum):
    """One post-import cleanup effect. Each member IS its keep-warn noun."""

    CATEGORY_MOVE = "category move"
    QUEUE_REMOVAL = "queue removal"


class PendingImportState(StrEnum):
    """The current status of one carried-over pending import, for reporting.

    A `StrEnum` (so each member IS its rendered word) shared by the inline
    snapshot ledger row, the WaitView live region, and the end-of-run scoreboard
    counters, so one vocabulary describes a carried-over record everywhere.
    """

    DOWNLOADING = "queued"
    """Still downloading."""

    DOWNLOADED = "downloaded"
    """The download finished and will be imported by the end-of-run pass."""

    IMPORTED = "imported"
    """The episode files are verified present. The record is dropped, or kept only for its post-import
    cleanup."""

    ERRORED = "errored"
    """The download errored in qBittorrent. Left for a later run."""

    GONE = "missing"
    """The torrent is gone from qBittorrent. The record is dropped."""


def classify_pending(
    wait_outcome: "TorrentStatus | None",
    all_done: bool,
) -> PendingImportState:
    """Map a poll's outcome to a single carried-over `PendingState`.

    Args:
        wait_outcome: The torrent's terminal outcome this
            poll, or `None` while it is still downloading.
        all_done: Whether every intended episode file is verified
            present in Sonarr (the only signal that promotes to `IMPORTED`).
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
    """Why a poll waited on Sonarr's work rather than this record's (the monitor credits every reason but NONE)."""

    NONE = auto()
    """Not deferred: the wait is this record's own and burns its ready clock."""

    ISSUED = auto()
    """Our ManualImport was accepted this poll (credited once per record)."""

    IMPORT = auto()
    """An import is in flight: ours from an earlier poll, an unproven one, or Sonarr's own `importing` row."""

    BUSY = auto()
    """A foreign disk command holds the line."""


@dataclass(frozen=True, slots=True)
class HeldFile:
    """One of a download's on-disk files left for a hand import."""

    name: str
    """Its on-disk leaf."""

    order_only: OrderOnlyKey | None = None
    """Its key, when only another TVDB order numbers it (`PendingImport.order_only`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LeftForHandImport:
    """A download's on-disk files left for a hand import, and the record's window episodes still unfilled."""

    files: tuple[HeldFile, ...] = ()

    missing: tuple[EpisodeKey, ...] = ()
    """The record's unfilled window episodes that hold no file (`UnfilledTargets.missing`)."""

    untrusted: tuple[EpisodeKey, ...] = ()
    """The record's unfilled window episodes that hold a file (`UnfilledTargets.not_recommended`)."""

    @property
    def names(self) -> tuple[str, ...]:
        """The files' on-disk leaves."""

        return tuple(file.name for file in self.files)

    @property
    def explained_by_order(self) -> bool:
        """Whether the order explains the record: it has files, each with an order-only key, and an unfilled episode."""

        keyed = bool(self.files) and all(file.order_only is not None for file in self.files)
        return keyed and bool(self.missing or self.untrusted)


NO_HELD_BACK = LeftForHandImport()
"""No file left for a hand import."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptResult:
    """The outcome of one `import_completed` poll.

    Lets the engine tell `imported` (every intended episode file is verified
    present) from `importing` (an import command was accepted but the copy is
    still running).
    """

    all_done: bool
    """Whether every intended episode file is verified present in Sonarr."""

    command_issued: bool
    """Whether a manual-import command covering this download was accepted."""

    attempted: bool = True
    """False only when no attempt ran (it raised, or no strategy is bound), so the record stays pending."""

    imported_count: int = 0
    """How many of the intended episodes already hold the recommended file"""

    target_count: int = 0
    """The intended episodes we mapped, 0 means the seed map is incomplete (indeterminate)."""

    deferral: PauseReason = PauseReason.NONE
    """Why this poll waited on Sonarr's work instead of this record's."""

    placements: FileEpisodeMap = field(default_factory=dict[str, tuple[int, ...]])
    """The files Sonarr's history placed for a download no scan saw (normalized basename -> ids), once they
    verified, copied and wrapped read-only for the record seam to persist. Empty otherwise."""

    unmatched_files: tuple[str, ...] = ()
    """The download's on-disk files no episode claimed (one per distinct basename), when that is all the poll
    found. Non-empty only from `unmatched()`: `files_present` and `command_issued` False, `deferral` NONE."""

    left_for_hand_import: LeftForHandImport = NO_HELD_BACK
    """The download's on-disk files the grab left for a hand import (`PendingImport.hand_import_files`), once every
    placed file is in and the hand import isn't. Its names are non-empty only from `hand_import()`: `files_present`
    and `command_issued` False."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", MappingProxyType(dict(self.placements)))

    @property
    def deferred(self) -> bool:
        """Whether this poll waited on Sonarr's work (the monitor credits the interval back)."""

        return self.deferral is not PauseReason.NONE

    @classmethod
    def imported(cls) -> "AttemptResult":
        """The intended files are verified present, with no import command of ours in flight."""

        return cls(all_done=True, command_issued=False)

    @classmethod
    def waiting(cls, *, command_issued: bool = False, deferral: PauseReason = PauseReason.NONE) -> "AttemptResult":
        """Not imported yet: poll again until the readiness deadline."""

        return cls(all_done=False, command_issued=command_issued, deferral=deferral)

    @classmethod
    def unmatched(cls, names: tuple[str, ...]) -> "AttemptResult":
        """Nothing to import or verify: every file on disk matched no episode."""

        return cls(all_done=False, command_issued=False, unmatched_files=names)

    @classmethod
    def hand_import(cls, held: LeftForHandImport) -> "AttemptResult":
        """Every placed file is in, but files the grab left for a hand import keep the record pending."""

        return cls(all_done=False, command_issued=False, left_for_hand_import=held)


LEAVE_PROBE = AttemptResult(attempted=False, all_done=False, command_issued=False)
"""The fail-open probe: no attempt ran, so leave the record pending for a later run."""


@dataclass(frozen=True, slots=True)
class ImportProgress:
    """A cheap, read-only files-landed count for the wait cockpit's import bar."""

    done: int
    total: int
    determinate: bool
    """True only when the persisted seed map covers every intended file, so `done`/`total` are the true full
    set. When False the importing row stays indeterminate (spinner only) and must NOT promote."""

    @property
    def all_done(self) -> bool:
        """Every intended file verified present."""

        return self.determinate and 0 < self.total <= self.done


NO_PROGRESS = ImportProgress(0, 0, determinate=False)
"""The indeterminate zero reading a failed or strategy-less poll reports."""


class OutcomeKind(Enum):
    """The wait view's ledger glyph + color."""

    IMPORTED = ("✔", "ok", "green")
    """The torrent imported."""

    LEFT = ("⚠", "~", "yellow")
    """Left pending for a later run (a download timeout, an import that hasn't landed yet, or files only a
    human can place). Not a failure, just unfinished."""

    FAILED = ("✖", "x", "bold red")
    """The download errored or vanished from qBittorrent."""

    IN_PROGRESS = ("·", "-", "grey50")
    """Left for the next run on purpose - the one-cycle check saw it and nothing is wrong."""

    glyph: str
    """The unicode glyph (`✔`/`⚠`/`✖`/`·`)."""

    ascii_glyph: str
    """The ASCII fallback, for dumb terminals / legacy Windows, where `✔` can't be encoded."""

    style: str
    """The rich style its ledger row is colored with."""

    def __init__(self, glyph: str, ascii_glyph: str, style: str) -> None:
        self.glyph = glyph
        self.ascii_glyph = ascii_glyph
        self.style = style

    def glyph_for(self, *, use_unicode: bool) -> str:
        """The ledger glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.glyph if use_unicode else self.ascii_glyph


class Outcome(Enum):
    """A torrent's terminal result in the wait pass, with its rendering vocab."""

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
    """The short ledger token (every one fits `STATE_WIDTH` = 11)."""

    detail: str
    """The longer human phrase the run report / notification use."""

    category: OutcomeKind
    """The `OutcomeCategory` driving glyph + color + tally."""

    dropped: bool
    """Whether the engine removes the record from the durable store on this outcome.
    True for `IMPORTED` (files verified present) and `MISSING` (gone from qBittorrent)."""

    def __init__(
        self,
        word: str,
        detail: str,
        category: OutcomeKind,
        dropped: bool,
    ) -> None:
        self.word = word
        self.detail = detail
        self.category = category
        self.dropped = dropped

    @property
    def style(self) -> str:
        """The rich style for this outcome's ledger row (from its category)."""

        return self.category.style

    def glyph(self, *, use_unicode: bool) -> str:
        """The leading ledger glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.category.glyph_for(use_unicode=use_unicode)


# Each non-dropped Outcome's tally bucket. Dropped outcomes (IMPORTED, MISSING) leave the store,
# so a wait pass folds every record it leaves resident through exactly this table.
PENDING_STATE_FOR_OUTCOME: dict[Outcome, PendingImportState] = {
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
class TorrentTelemetry:
    """One info row's sanitized live telemetry: download fraction, speed, ETA, and byte counts."""

    progress: float
    speed_bps: int | None
    eta_s: int | None
    bytes_done: int | None
    bytes_total: int | None


_ZERO_TELEMETRY = TorrentTelemetry(0.0, None, None, None, None)
"""The unread placeholder (no client, or a transient qBittorrent error)."""


@dataclass(frozen=True, slots=True)
class TorrentPoll:
    """One qBittorrent completion poll, with live download telemetry."""

    outcome: TorrentStatus | None
    """The terminal outcome this poll, or None while still downloading (or on a transient qB error)."""

    content_path: str | None
    """The completed download's path (COMPLETE only)."""

    telemetry: TorrentTelemetry = _ZERO_TELEMETRY
    """The poll's sanitized live reading (zeroed when nothing was read)."""

    observed: bool = True
    """False when qBittorrent could not actually be read (no client / a transient error)."""

    @property
    def ready_path(self) -> str | None:
        """The completed download's importable path: COMPLETE with a reported content path, else None."""

        if self.outcome is TorrentStatus.COMPLETE and self.content_path:
            return self.content_path
        return None


def sanitize_torrent_telemetry(row: object) -> TorrentTelemetry:
    """Fold one qBittorrent info row into sanitized telemetry (fields read best-effort off the row)."""

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
    return TorrentTelemetry(frac, speed_bps, eta_s, bytes_done, bytes_total)


def torrent_finished(row: object) -> bool:
    """Whether a qBittorrent info row's download is done: complete by its state, or at full progress."""

    complete = getattr(getattr(row, "state_enum", None), "is_complete", False)
    return complete is True or sanitize_torrent_telemetry(row).progress >= 1.0


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
