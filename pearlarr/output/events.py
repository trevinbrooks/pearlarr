"""The closed output-event vocabulary: frozen facts only, no rich render types.

Producers state WHAT happened, never how it looks: no style strings, no pre-formatted display strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, assert_never

from .trace import CapturedTrace
from ..config import Arr
from ..log import EntryState
from ..manual_import import Outcome, OutcomeCategory

if TYPE_CHECKING:
    from ..reporter import RunStats


class Severity(IntEnum):
    """Event severity."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Accent(Enum):
    """Semantic emphasis."""

    PLAIN = auto()
    DIM = auto()
    GOOD = auto()
    CAUTION = auto()
    BAD = auto()
    ACCENT = auto()
    FOCUS = auto()
    # An informational "nothing to do" status.
    NOTE = auto()


@dataclass(frozen=True, slots=True)
class StyledValue:
    """A plain string plus one semantic emphasis."""

    text: str
    accent: Accent = Accent.PLAIN


class ScopeKind(Enum):
    """The open-node kinds."""

    BOOT_SECTION = auto()
    BOOT_STEP = auto()
    RUN = auto()
    ITEM = auto()
    ENTRY = auto()
    WAIT_REGION = auto()


@dataclass(frozen=True, slots=True)
class ScopeId:
    """A minted scope identity."""

    kind: ScopeKind
    serial: int


class PlacedBy(Enum):
    """How a diagnostic's rendered position was assigned."""

    AMBIENT = auto()  # position-free. Any rendered position is the frontier's guess
    HANDLE = auto()  # demoted from a known (closed) scope. Attribution is exact


# --- run / cycle lifecycle -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunStarted:
    """The process banner facts."""

    version: str
    """May be an empty string."""
    data_dir: str


@dataclass(frozen=True, slots=True)
class CycleStarted:
    """One scheduled-mode cycle begins (1-based)."""

    number: int


@dataclass(frozen=True, slots=True)
class NextRunScheduled:
    """When the next scheduled cycle fires."""

    at: datetime


# --- scope boundaries ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScopeOpened:
    """A handle-backed scope opened."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class ScopeClosed:
    """A handle-backed scope closed."""

    scope: ScopeId


# --- boot cockpit -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootStepStarted:
    """A boot step started."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class BootStepProgressed:
    """A boot step's live progress."""

    scope: ScopeId
    fraction: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class BootStepSlow:
    """A one-time heads-up that a step is slow."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class BootStepFinished:
    """A boot step finished."""

    scope: ScopeId
    label: str
    outcome: OutcomeCategory
    detail: str | None
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class BootReady:
    """The boot capstone."""

    elapsed_s: float


# --- scan -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanStarted:
    """Opens the per-arr run node."""

    arr: Arr
    total: int


@dataclass(frozen=True, slots=True)
class ItemStarted:
    """Opens an item node."""

    arr: Arr
    index: int
    total: int
    title: str


@dataclass(frozen=True, slots=True)
class EntryHeader:
    """One entry block's head."""

    state: EntryState
    title: str
    al_id: int | None = None
    coverage: str | None = None
    url: str | None = None
    incomplete: bool = False
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class EntryDetail:
    """A labeled line inside an entry block."""

    label: str
    value: StyledValue
    severity: Severity = Severity.INFO
    tail: str | None = None
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """A one-line ledger row with no block body."""

    state: EntryState
    label: str
    accent: Accent = Accent.DIM
    scope: ScopeId | None = None


class SkipReason(Enum):
    """Why a release was skipped at add time."""

    PRIVATE_ONLY = auto()
    UNSUPPORTED_TRACKER = auto()
    TRACKER_NOT_SELECTED = auto()

    @property
    def severity(self) -> Severity:
        """The line's severity: a tracker the user did not select is INFO, not a warning."""

        return Severity.INFO if self is SkipReason.TRACKER_NOT_SELECTED else Severity.WARNING


@dataclass(frozen=True, slots=True)
class ReleaseSkipped:
    """One release skipped at add time."""

    group: str
    tracker: str
    reason: SkipReason
    url: str | None = None
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class GrabFailed:
    """A contained transient grab failure."""

    group: str
    url: str
    error: str
    scope: ScopeId | None = None


class GrabStatus(Enum):
    """The grab action's disposition."""

    ADDING = auto()
    WOULD_ADD = auto()
    ALREADY_DOWNLOADING = auto()
    """SeaDex's pick is already in qBittorrent from an earlier run."""


@dataclass(frozen=True, slots=True)
class RecommendedGroup:
    """A recommended release group with its SeaDex tags."""

    name: str
    tags: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        """The "name [tag, tag]" form, or the bare name when untagged."""

        return f"{self.name} [{', '.join(self.tags)}]" if self.tags else self.name


@dataclass(frozen=True, slots=True)
class ReleaseName:
    """A torrent name and its release group."""

    name: str
    group: str

    @property
    def display(self) -> str:
        """The torrent name, falling back to its group when name-less (never "None")."""

        return self.name or self.group


@dataclass(frozen=True, slots=True)
class GrabAction:
    """The whole per-title action block as one atomic fact."""

    status: GrabStatus
    groups: tuple[RecommendedGroup, ...]
    added: tuple[ReleaseName, ...]
    downloading: tuple[ReleaseName, ...]
    waiting_to_import: bool = False
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class CapReached:
    """The `max_torrents` cap was reached."""

    cap: int


@dataclass(frozen=True, slots=True)
class ScanFinished:
    """The scan-close boundary."""

    arr: Arr


# --- summary ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrabFact:
    """One grab for the summary's "added" block."""

    title: str | None
    coverage: str | None
    url: str | None
    name: str | None
    group: str


class NeedsActionCause(Enum):
    """Why a title needs attention. Member names must equal `reporter.NeedsActionKind`'s (mapped by name)."""

    PRIVATE_ONLY = auto()
    PRIVATE_ONLY_NO_FALLBACK = auto()
    PRIVATE_ONLY_STALE = auto()
    UNSUPPORTED_TRACKER = auto()
    GRAB_FAILED = auto()


@dataclass(frozen=True, slots=True)
class NeedsActionFact:
    """One user-actionable skip for the summary."""

    title: str | None
    coverage: str | None
    group: str
    url: str | None
    reason: str
    cause: NeedsActionCause


@dataclass(frozen=True, slots=True)
class RunTally:
    """`RunStats` frozen at summary time."""

    checked: int
    added: tuple[GrabFact, ...]
    up_to_date: int
    cached: int
    no_seadex_entry: int
    seadex_unreachable: int
    no_releases: int
    no_mappings: int
    needs_action: tuple[NeedsActionFact, ...]
    unmonitored: int
    queued: int
    downloaded: int
    imported: int

    @classmethod
    def from_stats(cls, stats: RunStats) -> RunTally:
        return cls(
            checked=stats.checked,
            added=tuple(
                GrabFact(title=g.title, coverage=g.coverage, url=g.url, name=g.name, group=g.group) for g in stats.added
            ),
            up_to_date=stats.up_to_date,
            cached=stats.cached,
            no_seadex_entry=stats.no_seadex_entry,
            seadex_unreachable=stats.seadex_unreachable,
            no_releases=stats.no_releases,
            no_mappings=stats.no_mappings,
            needs_action=tuple(
                NeedsActionFact(
                    title=n.title,
                    coverage=n.coverage,
                    group=n.group,
                    url=n.url,
                    reason=n.reason,
                    cause=NeedsActionCause[n.kind.name],
                )
                for n in stats.needs_action
            ),
            unmonitored=stats.unmonitored,
            queued=stats.queued,
            downloaded=stats.downloaded,
            imported=stats.imported,
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The whole end-of-run scoreboard as one value."""

    arr: Arr
    dry_run_note: str | None
    """The dry-run note, None on a real run."""
    added_count: int
    tally: RunTally
    wait_mode_on: bool
    warnings: int
    errors: int
    elapsed_s: float | None
    tip: NeedsActionCause | None

    @property
    def dry_run(self) -> bool:
        return self.dry_run_note is not None


@dataclass(frozen=True, slots=True)
class RunSummaryReady:
    """The end-of-run summary is ready."""

    summary: RunSummary


# --- wait pass ------------------------------------------------------------------


def clamp01(value: float) -> float:
    """Clamp a progress fraction into [0, 1]."""

    return max(0.0, min(1.0, value))


class Phase(Enum):
    """The lifecycle phase of one torrent in the wait pass."""

    QUEUED = auto()
    """Still downloading (or not yet polled)."""
    DOWNLOADING = auto()
    """Downloading with live telemetry."""
    IMPORTING = auto()
    """The download finished and an import is in flight."""
    TERMINAL = auto()
    """A terminal `Outcome` was reached. These GRADUATE to scrollback and leave the live region."""


# Speed samples a downloading row keeps for its sparkline, one per heavy poll
# (~4 minutes at the default 30s cadence). The producer bounds TorrentView.speed_history to this window.
SPARK_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class TorrentView:
    """One torrent's state for a single frame. `outcome` is non-None iff `phase` is `TERMINAL`."""

    key: str
    label: str
    phase: Phase = Phase.QUEUED
    fraction: float = 0.0
    speed_bps: int | None = None
    eta_s: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    phase_elapsed_s: float = 0.0
    command_issued: bool = False
    import_done: int | None = None
    """"Files inserted" bar: both set = determinate, both None = indeterminate."""
    import_total: int | None = None
    """On a TERMINAL imported row, the ledger's final files count (`phase_elapsed_s` freezes as its wait clock)."""
    speed_history: tuple[int, ...] = ()
    """Speed samples (bytes/s, stalled = 0), newest last, bounded to `SPARK_SAMPLES`."""
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class WaitSnapshot:
    """An immutable description of the whole wait pass, one value per poll cycle."""

    torrents: tuple[TorrentView, ...]
    elapsed_s: float = 0.0

    def counts(self) -> dict[Phase, int]:
        """Count of torrents in each phase (every phase present, 0 by default)."""

        tally: dict[Phase, int] = dict.fromkeys(Phase, 0)
        for torrent in self.torrents:
            tally[torrent.phase] += 1
        return tally

    def done(self) -> int:
        """How many torrents have reached a terminal outcome."""

        return sum(1 for t in self.torrents if t.phase is Phase.TERMINAL)

    def total(self) -> int:
        """How many torrents the pass is (or was) waiting on."""

        return len(self.torrents)

    def overall_fraction(self) -> float:
        """An aggregate 0-1 progress for the header bar. A terminal or importing row counts as complete."""

        if not self.torrents:
            return 0.0
        total = 0.0
        for torrent in self.torrents:
            if torrent.phase in (Phase.TERMINAL, Phase.IMPORTING):
                total += 1.0
            else:
                total += clamp01(torrent.fraction)
        return total / len(self.torrents)


class WaitKind(Enum):
    """Which end-of-run pass a wait region narrates, with its rendering vocab."""

    MONITOR = ("Waiting on {n} download{s} to complete and import...", "wait complete", "waiting", "Wait")
    """The blocking/hybrid monitor: waits for downloads to finish, then imports."""

    CHECK = ("Checking {n} carried-over download{s}...", "check complete", "checking", "Check")
    """The one-cycle check: polls once, imports the finished ones, never waits."""

    start_template: str
    """The digest's opening sentence, with `{n}` count and `{s}` plural-suffix slots."""

    tally_head: str
    """The closing summary's lead phrase."""

    live_verb: str
    """The cockpit header's progress verb, also the structured log's start word."""

    interrupt_noun: str
    """The Ctrl-C notice's pass name."""

    def __init__(self, start_template: str, tally_head: str, live_verb: str, interrupt_noun: str) -> None:
        self.start_template = start_template
        self.tally_head = tally_head
        self.live_verb = live_verb
        self.interrupt_noun = interrupt_noun


@dataclass(frozen=True, slots=True)
class WaitStarted:
    """The wait pass opened."""

    total: int
    pulse_s: float
    """The renderer's pulse throttle interval (`max(poll_s, digest_interval)`)."""
    kind: WaitKind
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class WaitProgress:
    """The engine's per-poll snapshot."""

    snapshot: WaitSnapshot
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class TorrentGraduated:
    """One torrent reached a terminal outcome."""

    label: str
    outcome: Outcome
    files: int | None
    waited_s: float
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class WaitFinished:
    """The wait pass closed, with its tally."""

    imported: int
    deferred: int
    failed: int
    elapsed_s: float
    pending: int
    """Rows the one-cycle check left for the next run on purpose."""
    kind: WaitKind
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The run-close boundary."""

    arr: Arr


# --- diagnostics ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A position-free problem or notice."""

    severity: Severity
    message: str
    origin: str = "app"
    once_key: str | None = None
    trace: CapturedTrace | None = None
    placed_by: PlacedBy = PlacedBy.AMBIENT
    file_only: bool = False
    """Routes hub-containment notes past the console surfaces to the file sink alone."""


# --- json value model -------------------------------------------------------------

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObj = dict[str, JsonValue]


# --- cli command facts ------------------------------------------------------------
# Emitted only by a subcommand, never during a run.


@dataclass(frozen=True, slots=True)
class PathsShown:
    """The `paths` command's resolved data directory and files."""

    data_dir: str
    config: str
    cache: str
    mappings_db: str
    log_dir: str


@dataclass(frozen=True, slots=True)
class StarterConfigWritten:
    """`config init` wrote a starter template."""

    path: str


@dataclass(frozen=True, slots=True)
class ConfigValidated:
    """`config validate` succeeded.

    `migration_notes` None = current schema, non-None = migrated in memory. Empty missing-keys = configured.
    """

    path: str
    migration_notes: tuple[str, ...] | None
    sonarr_missing_keys: tuple[str, ...]
    radarr_missing_keys: tuple[str, ...]
    qbit_configured: bool


@dataclass(frozen=True, slots=True)
class ConfigUpToDate:
    """`config migrate` had nothing to do."""

    path: str


@dataclass(frozen=True, slots=True)
class ConfigMigrated:
    """`config migrate` rewrote the file, saving the previous one."""

    path: str
    backup_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EffectiveConfigShown:
    """`config show` output. `config` is the already-redacted effective dump."""

    path: str
    config: JsonObj


@dataclass(frozen=True, slots=True)
class CacheBackedUp:
    """`cache backup` wrote a fresh snapshot."""

    backup_path: str


@dataclass(frozen=True, slots=True)
class CacheRestored:
    """`cache restore` restored the database."""

    backup_path: str


@dataclass(frozen=True, slots=True)
class CacheRemoved:
    """`cache remove` deleted the database."""

    path: str


@dataclass(frozen=True, slots=True)
class CacheStatsReported:
    """`cache stats`: per-block row counts and the on-disk size."""

    entries: int
    torrent_hashes: int
    anilist_meta: int
    sonarr_parse: int
    pending_imports: int
    guard_facts: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CacheIntegrityReported:
    """`cache check`'s success arm: SQLite's integrity-check result."""

    result: str


# --- the closed union --------------------------------------------------------------

type Event = (
    RunStarted
    | CycleStarted
    | NextRunScheduled
    | ScopeOpened
    | ScopeClosed
    | BootStepStarted
    | BootStepProgressed
    | BootStepSlow
    | BootStepFinished
    | BootReady
    | ScanStarted
    | ItemStarted
    | EntryHeader
    | EntryDetail
    | LedgerRow
    | ReleaseSkipped
    | GrabFailed
    | GrabAction
    | CapReached
    | ScanFinished
    | RunSummaryReady
    | WaitStarted
    | WaitProgress
    | TorrentGraduated
    | WaitFinished
    | RunFinished
    | Diagnostic
    | PathsShown
    | StarterConfigWritten
    | ConfigValidated
    | ConfigUpToDate
    | ConfigMigrated
    | EffectiveConfigShown
    | CacheBackedUp
    | CacheRestored
    | CacheRemoved
    | CacheStatsReported
    | CacheIntegrityReported
)


def _category_severity(category: OutcomeCategory) -> Severity:
    if category is OutcomeCategory.FAILED:
        return Severity.ERROR
    if category is OutcomeCategory.DEFERRED:
        return Severity.WARNING
    return Severity.INFO


def severity_of(event: Event) -> Severity:
    """The severity an event tallies as (drives SeverityCounts + sink level floors)."""

    match event:
        case Diagnostic(severity=severity):
            return severity
        case EntryDetail(severity=severity):
            return severity
        case ReleaseSkipped(reason=reason):
            return reason.severity
        case GrabFailed():
            return Severity.WARNING
        case BootStepFinished():
            # INFO regardless of outcome: a failed/deferred step's caller logs the problem
            # itself, so an outcome-based tally would double-count it.
            return Severity.INFO
        case TorrentGraduated(outcome=outcome):
            # Category-based. wait_graduation_line carries the same level.
            return _category_severity(outcome.category)
        case (
            RunStarted()
            | CycleStarted()
            | NextRunScheduled()
            | ScopeOpened()
            | ScopeClosed()
            | BootStepStarted()
            | BootStepProgressed()
            | BootStepSlow()
            | BootReady()
            | ScanStarted()
            | ItemStarted()
            | EntryHeader()
            | LedgerRow()
            | GrabAction()
            | CapReached()
            | ScanFinished()
            | RunSummaryReady()
            | WaitStarted()
            | WaitProgress()
            | WaitFinished()
            | RunFinished()
            | PathsShown()
            | StarterConfigWritten()
            | ConfigValidated()
            | ConfigUpToDate()
            | ConfigMigrated()
            | EffectiveConfigShown()
            | CacheBackedUp()
            | CacheRestored()
            | CacheRemoved()
            | CacheStatsReported()
            | CacheIntegrityReported()
        ):
            return Severity.INFO
    assert_never(event)
