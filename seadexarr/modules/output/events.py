"""The closed output-event vocabulary: frozen facts only, no rich render types.

Producers state WHAT happened; renderers own every look decision (glyphs, widths,
indents, styles). Emphasis rides the semantic :class:`Accent`/:class:`Span`/
:class:`StyledValue` value model — never a rich style string, never a pre-formatted
display string. ``EntryState`` (log.py) and ``Outcome``/``OutcomeCategory``
(manual_import.py) are reused, not mirrored, so the vocabulary can't drift from the
domain enums. PR7 relocation caution (verified): moving ``Outcome`` INTO this module
would cycle — manual_import -> output.events -> config -> manual_import (config
imports ``ImportWaitMode``) — so PR7 must first move ``ImportWaitMode`` out of
config's import, or leave ``Outcome`` where it is.
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
    from ..wait_view import WaitSnapshot


class Severity(IntEnum):
    """Event severity; values mirror stdlib logging so thresholds compare directly."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Accent(Enum):
    """Semantic emphasis; the rich renderer maps it to theme styles, text sinks drop it."""

    PLAIN = auto()
    DIM = auto()
    GOOD = auto()
    CAUTION = auto()
    BAD = auto()
    ACCENT = auto()
    FOCUS = auto()


@dataclass(frozen=True, slots=True)
class Span:
    """An accented [start:end) slice over a value's plain text (e.g. a release group)."""

    start: int
    end: int
    accent: Accent


@dataclass(frozen=True, slots=True)
class StyledValue:
    """A plain string plus semantic emphasis — the group_highlight data, unstyled."""

    text: str
    accent: Accent = Accent.PLAIN
    spans: tuple[Span, ...] = ()


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
    """A minted scope identity; events carrying one have compile-checked position."""

    kind: ScopeKind
    serial: int


class PlacedBy(Enum):
    """How a diagnostic's rendered position was assigned — the record admits a guess."""

    AMBIENT = auto()  # position-free; any rendered position is the frontier's guess
    HANDLE = auto()  # demoted from a known (closed) scope; attribution is exact


# --- run / cycle lifecycle -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunStarted:
    """The process banner facts (version may be "")."""

    version: str
    data_dir: str


@dataclass(frozen=True, slots=True)
class CycleStarted:
    """One scheduled-mode cycle begins (1-based)."""

    number: int


@dataclass(frozen=True, slots=True)
class NextRunScheduled:
    at: datetime


# --- scope boundaries (the breadcrumb fold's inputs; see breadcrumbs.py) ----------


@dataclass(frozen=True, slots=True)
class ScopeOpened:
    """A handle-backed scope opened; ``label`` is the only home of its display name."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class ScopeClosed:
    scope: ScopeId


# --- boot cockpit -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootStepStarted:
    """Also the step node's open in the fold's transition table."""

    scope: ScopeId
    label: str


@dataclass(frozen=True, slots=True)
class BootStepProgressed:
    """Ephemeral: live surfaces only; text/json sinks drop it."""

    scope: ScopeId
    fraction: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class BootStepSlow:
    """One-time heads-up on a step's first progress report (S6); text sinks map it 1:1."""

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
    """The boot capstone; the producer suppresses it on a failed section."""

    elapsed_s: float


# --- scan -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanStarted:
    """Opens the per-arr run node (B6 boundary; no handle ceremony)."""

    arr: Arr
    total: int


@dataclass(frozen=True, slots=True)
class ItemStarted:
    """Opens an item node, deterministically closing the previous item/entry (B6)."""

    arr: Arr
    index: int
    total: int
    title: str


@dataclass(frozen=True, slots=True)
class EntryHeader:
    """One entry block's head, carried whole (header-at-open commit rule, B5).

    The focal/dim distinction is renderer policy keyed on ``state`` — no
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
    PRIVATE_ONLY = auto()
    UNSUPPORTED_TRACKER = auto()
    TRACKER_NOT_SELECTED = auto()

    @property
    def severity(self) -> Severity:
        """TRACKER_NOT_SELECTED is the user's own choice, so it stays INFO."""

        return Severity.INFO if self is SkipReason.TRACKER_NOT_SELECTED else Severity.WARNING


@dataclass(frozen=True, slots=True)
class ReleaseSkipped:
    group: str
    tracker: str
    reason: SkipReason
    url: str | None = None
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class GrabFailed:
    """A contained transient grab failure; the title retries next run."""

    group: str
    url: str
    error: str
    scope: ScopeId | None = None


class GrabStatus(Enum):
    ADDING = auto()
    WOULD_ADD = auto()
    ALREADY_DOWNLOADING = auto()


@dataclass(frozen=True, slots=True)
class RecommendedGroup:
    """A recommended release group with its SeaDex tags, carried separately (never
    pre-joined into "Group [tag, tag]" display strings)."""

    name: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReleaseName:
    """A torrent name + its release group — the group_highlight data, unstyled."""

    name: str
    group: str


@dataclass(frozen=True, slots=True)
class GrabAction:
    """The whole per-title action block as one atomic fact.

    A dry run IS ``GrabStatus.WOULD_ADD`` — no separate flag to drift from it.
    """

    status: GrabStatus
    groups: tuple[RecommendedGroup, ...]
    added: tuple[ReleaseName, ...]
    downloading: tuple[ReleaseName, ...]
    waiting_to_import: bool = False
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class CapReached:
    cap: int


@dataclass(frozen=True, slots=True)
class ScanFinished:
    """The scan-close boundary (B4.2): emitted atop _finalize_run, before the
    reconcile, so reconcile diagnostics render at run level."""

    arr: Arr


# --- summary ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrabFact:
    """One grab for the summary's "added" block (owned twin of reporter.GrabRecord;
    PR4 maps records into facts at the reporter chokepoint)."""

    title: str | None
    coverage: str | None
    url: str | None
    name: str | None
    group: str


class NeedsActionCause(Enum):
    """Why a title needs the user (owned twin of reporter.NeedsActionKind; member
    names are test-pinned equal so the PR4 name-based mapping can't drift)."""

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
    """RunStats frozen at summary time; :meth:`from_stats` is the single conversion
    site (S10: never two hand-maintained field lists; a fields-parity test pins it)."""

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
    """The whole end-of-run scoreboard as one value; the tally rides embedded whole."""

    arr: Arr
    dry_run: bool
    dry_run_note: str | None
    added_count: int
    tally: RunTally
    wait_mode_on: bool
    warnings: int
    errors: int
    elapsed_s: float | None
    tip: NeedsActionCause | None


@dataclass(frozen=True, slots=True)
class RunSummaryReady:
    """Atomic (S2: no SummaryScope); also a B6 boundary closing item/entry nodes."""

    summary: RunSummary


# --- wait pass ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaitStarted:
    total: int
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class WaitProgress:
    """Ephemeral: carries the engine's pure per-poll snapshot; text/json sinks drop it.

    ``WaitSnapshot`` stays a TYPE_CHECKING import (annotations are stringified);
    PR5 revisits its home when wait_view's machinery moves into the renderer.
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
    imported: int
    deferred: int
    failed: int
    elapsed_s: float
    scope: ScopeId | None = None


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The run-close boundary (B4.3): emitted at the end of _finalize_run and
    defensively by the unwind teardown (B3); the fold treats it idempotently."""

    arr: Arr


# --- diagnostics ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A position-free problem/notice: our stragglers + adopted stdlib records.

    ``file_only`` routes hub-containment notes (and, later, bridge-adopted
    third-party DEBUG/INFO) past the console surfaces to the file sink alone.
    """

    severity: Severity
    message: str
    origin: str = "app"
    once_key: str | None = None
    trace: CapturedTrace | None = None
    placed_by: PlacedBy = PlacedBy.AMBIENT
    file_only: bool = False


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
        case BootStepFinished(outcome=outcome):
            return _category_severity(outcome)
        case TorrentGraduated(outcome=outcome):
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
        ):
            return Severity.INFO
    assert_never(event)
