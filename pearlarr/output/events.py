"""The closed output-event vocabulary: frozen facts only, no rich render types.

Producers state WHAT happened. Renderers own every look decision (glyphs, widths,
indents, styles). Emphasis rides the semantic `Accent`/`Span`/
`StyledValue` value model - never a rich style string, never a pre-formatted
display string. `EntryState` (log.py) and `Outcome`/`OutcomeCategory`
(manual_import.py) are reused, not mirrored, so the vocabulary can't drift from the
domain enums. `Outcome` deliberately stays in manual_import: moving it INTO
this module would cycle - manual_import -> output.events -> config
-> manual_import (config imports `ImportWaitMode`) - and reuse already prevents
the drift that relocation would have bought.
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
    """Event severity. Values mirror stdlib logging so thresholds compare directly."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Accent(Enum):
    """Semantic emphasis. The rich renderer maps it to theme styles, text sinks drop it."""

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
    """A plain string plus one semantic emphasis, unstyled (renderers map the accent)."""

    text: str
    accent: Accent = Accent.PLAIN


class ScopeKind(Enum):
    """The open-node kinds the breadcrumb fold tracks (depths in breadcrumbs.py)."""

    BOOT_SECTION = auto()
    BOOT_STEP = auto()
    RUN = auto()
    ITEM = auto()
    ENTRY = auto()
    WAIT_REGION = auto()


@dataclass(frozen=True, slots=True)
class ScopeId:
    """A minted scope identity. Events carrying one have compile-checked position."""

    kind: ScopeKind
    serial: int


class PlacedBy(Enum):
    """How a diagnostic's rendered position was assigned - the record admits a guess."""

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
    """The scheduled loop's footer fact: when the next cycle fires."""

    at: datetime


# --- scope boundaries (the breadcrumb fold's inputs, see breadcrumbs.py) ----------


@dataclass(frozen=True, slots=True)
class ScopeOpened:
    """A handle-backed scope opened. `label` is the only home of its display name."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class ScopeClosed:
    """A handle-backed scope closed. The fold also closes anything nested deeper."""

    scope: ScopeId


# --- boot cockpit -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootStepStarted:
    """Also the step node's open in the fold's transition table."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class BootStepProgressed:
    """Ephemeral: live surfaces only. Text/json sinks drop it."""

    scope: ScopeId
    fraction: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class BootStepSlow:
    """One-time heads-up on a step's first progress report. Text sinks map it 1:1."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class BootStepFinished:
    """Also the step node's close in the fold's transition table."""

    scope: ScopeId
    label: str
    outcome: OutcomeCategory
    detail: str | None
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class BootReady:
    """The boot capstone. The producer suppresses it on a failed section."""

    elapsed_s: float


# --- scan -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanStarted:
    """Opens the per-arr run node (a boundary event, no handle ceremony)."""

    arr: Arr
    total: int


@dataclass(frozen=True, slots=True)
class ItemStarted:
    """Opens an item node, deterministically closing the previous item/entry."""

    arr: Arr
    index: int
    total: int
    title: str


@dataclass(frozen=True, slots=True)
class EntryHeader:
    """One entry block's head, carried whole (the header-at-open commit rule).

    The focal/dim distinction is renderer policy keyed on `state` - no
    emphasis flag here (Accent is the emphasis mechanism where one is needed).
    """

    state: EntryState
    title: str
    al_id: int | None = None
    coverage: str | None = None
    url: str | None = None
    incomplete: bool = False
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class EntryDetail:
    """A labeled line inside an entry block ("status", "missing episodes", ...)."""

    label: str
    value: StyledValue
    severity: Severity = Severity.INFO
    tail: str | None = None
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """A self-contained one-line ledger row (state + label) with no block body."""

    state: EntryState
    label: str
    accent: Accent = Accent.DIM
    scope: ScopeId | None = None


class SkipReason(Enum):
    """Why a release was skipped at add time. `severity` picks the line's level."""

    PRIVATE_ONLY = auto()
    UNSUPPORTED_TRACKER = auto()
    TRACKER_NOT_SELECTED = auto()

    @property
    def severity(self) -> Severity:
        """TRACKER_NOT_SELECTED is a deliberate choice, so it stays INFO."""

        return Severity.INFO if self is SkipReason.TRACKER_NOT_SELECTED else Severity.WARNING


@dataclass(frozen=True, slots=True)
class ReleaseSkipped:
    """One release skipped at add time, with the group/tracker facts the line renders."""

    group: str
    tracker: str
    reason: SkipReason
    url: str | None = None
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class GrabFailed:
    """A contained transient grab failure. The title retries next run."""

    group: str
    url: str
    error: str
    scope: ScopeId | None = None


class GrabStatus(Enum):
    """The grab action's disposition: a real add, a preview would-add, or already downloading."""

    ADDING = auto()
    WOULD_ADD = auto()
    ALREADY_DOWNLOADING = auto()


@dataclass(frozen=True, slots=True)
class RecommendedGroup:
    """A recommended release group with its SeaDex tags, carried separately.

    `display` is the one place they're joined for output.
    """

    name: str
    tags: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        """The "name [tag, tag]" form, or the bare name when untagged."""

        return f"{self.name} [{', '.join(self.tags)}]" if self.tags else self.name


@dataclass(frozen=True, slots=True)
class ReleaseName:
    """A torrent name + its release group - the group_highlight data, unstyled."""

    name: str
    group: str

    @property
    def display(self) -> str:
        """The torrent name, falling back to its group when name-less (never "None")."""

        return self.name or self.group


@dataclass(frozen=True, slots=True)
class GrabAction:
    """The whole per-title action block as one atomic fact.

    A dry run IS `GrabStatus.WOULD_ADD` - no separate flag to drift from it.
    """

    status: GrabStatus
    groups: tuple[RecommendedGroup, ...]
    added: tuple[ReleaseName, ...]
    downloading: tuple[ReleaseName, ...]
    waiting_to_import: bool = False
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class CapReached:
    """The `max_torrents` cap was reached. The run adds nothing further."""

    cap: int


@dataclass(frozen=True, slots=True)
class ScanFinished:
    """The scan-close boundary.

    Emitted atop `_finalize_run`, before the reconcile, so reconcile
    diagnostics render at run level.
    """

    arr: Arr


# --- summary ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrabFact:
    """One grab for the summary's "added" block - the owned twin of `reporter.GrabRecord`.

    Records map into facts at the reporter chokepoint.
    """

    title: str | None
    coverage: str | None
    url: str | None
    name: str | None
    group: str


class NeedsActionCause(Enum):
    """Why a title needs attention - the owned twin of `reporter.NeedsActionKind`.

    Member names are test-pinned equal so the name-based mapping can't drift.
    """

    PRIVATE_ONLY = auto()
    PRIVATE_ONLY_NO_FALLBACK = auto()
    PRIVATE_ONLY_STALE = auto()
    UNSUPPORTED_TRACKER = auto()
    GRAB_FAILED = auto()


@dataclass(frozen=True, slots=True)
class NeedsActionFact:
    """One user-actionable skip for the summary (owned twin of NeedsActionRecord)."""

    title: str | None
    coverage: str | None
    group: str
    url: str | None
    reason: str
    cause: NeedsActionCause


@dataclass(frozen=True, slots=True)
class RunTally:
    """`RunStats` frozen at summary time.

    `from_stats` is the single conversion site (never two hand-maintained
    field lists, a fields-parity test pins it).
    """

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
    importing: int
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
            importing=stats.importing,
            imported=stats.imported,
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The whole end-of-run scoreboard as one value. The tally rides embedded whole.

    A noteless "DRY RUN" title is unrepresentable.
    """

    arr: Arr
    dry_run_note: str | None
    """The dry-run marker AND its human note in one field (None = a real run)."""
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
    """Atomic (no SummaryScope). Also a boundary closing item/entry nodes."""

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
    """The download finished and an import is in flight (indeterminate)."""
    TERMINAL = auto()
    """A terminal `Outcome` was reached. These GRADUATE to scrollback and leave the live region."""


# Speed samples a downloading row keeps for its sparkline (one per heavy poll,
# so the default 30s cadence holds the last ~4 minutes). The producer bounds
# TorrentView.speed_history to this window.
SPARK_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class TorrentView:
    """One torrent's state for a single frame - the engine's per-poll snapshot row.

    Immutable so a snapshot is a value: the engine rebuilds the row each cycle
    (`dataclasses.replace` off the prior one) and the renderers draw it. Telemetry
    fields are already sanitized (`manual_import.TorrentProbe`). `outcome` is
    non-None iff `phase` is `TERMINAL`.
    """

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
    """"Files inserted" bar for an IMPORTING row: both set -> a determinate done/total bar. Both
    `None` -> indeterminate (just the "importing" word)."""
    import_total: int | None = None
    """On a TERMINAL imported row, `import_done`/`import_total` carry the final files count for
    the ledger, and `phase_elapsed_s` freezes as the ledger's wait clock (the wait region's
    between-poll ticking skips TERMINAL rows, so it can't drift)."""
    speed_history: tuple[int, ...] = ()
    """Speed samples (bytes/s, stalled -> 0), one per heavy poll, newest last - the sparkline
    showing slow-but-moving vs wedged. Bounded by the producer to the sparkline window
    (`SPARK_SAMPLES` above)."""
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class WaitSnapshot:
    """An immutable description of the whole wait pass at one poll cycle.

    The single value the engine pushes per poll cycle. The wait views/renderers
    are pure functions of it. Derived aggregates are pure functions of the
    snapshot, independent of rendering.
    """

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
        """An aggregate 0-1 progress for the header bar (download-completion based).

        Terminal and importing rows count as a finished download (1.0). A still
        downloading/queued row contributes its download fraction. Guards /0.
        """

        if not self.torrents:
            return 0.0
        total = 0.0
        for torrent in self.torrents:
            if torrent.phase in (Phase.TERMINAL, Phase.IMPORTING):
                total += 1.0
            else:
                total += clamp01(torrent.fraction)
        return total / len(self.torrents)


@dataclass(frozen=True, slots=True)
class WaitStarted:
    """The wait pass opened: how many torrents it watches, and the pulse cadence."""

    total: int
    pulse_s: float
    """The renderer's pulse throttle interval (max(poll_s, digest_interval)). The producer
    computes it. No default: the producer must supply it."""
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class WaitProgress:
    """The engine's pure per-poll snapshot.

    The json surface drops it. The text sinks render only a throttled "still
    waiting" pulse from it (mode-independent, like every text line - the file
    is the same whether stdout was a TTY or a pipe).
    """

    snapshot: WaitSnapshot
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class TorrentGraduated:
    """A durable wait-ledger line: one torrent reached a terminal outcome."""

    label: str
    outcome: Outcome
    files: int | None
    waited_s: float
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class WaitFinished:
    """The wait pass closed, with its imported/deferred/failed tally."""

    imported: int
    deferred: int
    failed: int
    elapsed_s: float
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The run-close boundary.

    Emitted at the end of `_finalize_run`, and by the unwind teardown only
    when the leg dies before that tail close. The fold treats it idempotently
    (defense in depth).
    """

    arr: Arr


# --- diagnostics ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A position-free problem/notice: our one-liners (hub_note) + adopted stdlib records."""

    severity: Severity
    message: str
    origin: str = "app"
    once_key: str | None = None
    trace: CapturedTrace | None = None
    placed_by: PlacedBy = PlacedBy.AMBIENT
    file_only: bool = False
    """Routes hub-containment notes past the console surfaces to the file sink alone."""


# --- json value model (the wire types the json surface and cli facts share) --------

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObj = dict[str, JsonValue]


# --- cli command facts ------------------------------------------------------------
# Emitted only by a subcommand's --json / human seat, never during a run. The run
# renderers ignore them. Error/refusal arms reuse Diagnostic (via hub_error).


@dataclass(frozen=True, slots=True)
class PathsShown:
    """The `paths` command's resolved data directory and the files within it."""

    data_dir: str
    config: str
    cache: str
    mappings_db: str
    log_dir: str


@dataclass(frozen=True, slots=True)
class StarterConfigWritten:
    """`config init` wrote a starter template to `path`."""

    path: str


@dataclass(frozen=True, slots=True)
class ConfigValidated:
    """`config validate` succeeded: the run-shaping facts it reports.

    `migration_notes` None = the file is already at the current schema. Non-None
    (possibly empty) = an older schema was migrated in memory at load. An empty
    missing-keys tuple = that arr is configured.
    """

    path: str
    migration_notes: tuple[str, ...] | None
    sonarr_missing_keys: tuple[str, ...]
    radarr_missing_keys: tuple[str, ...]
    qbit_configured: bool


@dataclass(frozen=True, slots=True)
class ConfigUpToDate:
    """`config migrate` had nothing to do: the file is already at the current schema."""

    path: str


@dataclass(frozen=True, slots=True)
class ConfigMigrated:
    """`config migrate` rewrote the file, saving the previous one as `backup_path`."""

    path: str
    backup_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EffectiveConfigShown:
    """`config show` output: `config` is the ALREADY-REDACTED effective dump."""

    path: str
    config: JsonObj


@dataclass(frozen=True, slots=True)
class CacheBackedUp:
    """`cache backup` wrote a fresh snapshot to `backup_path`."""

    backup_path: str


@dataclass(frozen=True, slots=True)
class CacheRestored:
    """`cache restore` restored the database from `backup_path`."""

    backup_path: str


@dataclass(frozen=True, slots=True)
class CacheRemoved:
    """`cache remove` deleted the database at `path`."""

    path: str


@dataclass(frozen=True, slots=True)
class CacheStatsReported:
    """`cache stats`: per-block row counts and the on-disk size in bytes."""

    entries: int
    torrent_hashes: int
    anilist_meta: int
    sonarr_parse: int
    pending_imports: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CacheIntegrityReported:
    """`cache check`'s success arm: the SQLite integrity-check result string."""

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
            # INFO regardless of outcome: a failed/deferred step's caller logs the
            # problem itself, so an outcome-based tally would double-count it.
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
