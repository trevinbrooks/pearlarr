"""The run reporter: per-run context and stats, and the typed emission path the run logs through."""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, Literal

import qbittorrentapi
from seadex import EntryRecord

from .anilist_gateway import AniListGateway
from .cache import AbstractCacheStore
from .config import Arr
from .log import EntryState
from .manual_import import ImportWaitMode, PendingImport, PendingKey, PendingState
from .output import (
    Accent,
    CapReached,
    CountsMark,
    CountsSource,
    Emit,
    EntryDetail,
    EntryFact,
    EntryHeader,
    EntryScope,
    GrabAction,
    GrabFailed,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    NeedsActionCause,
    NeedsActionFact,
    RecommendedGroup,
    ReleaseName,
    ReleaseSkipped,
    RunFinished,
    RunSummary,
    RunSummaryReady,
    RunTally,
    ScanFinished,
    ScanStarted,
    ScopeFactory,
    Severity,
    SeverityCounts,
    StyledValue,
)
from .seadex_types import SeadexDict
from .torrents import AddOutcome, ReleaseOutcome

if TYPE_CHECKING:
    # Annotation-only (PerTitleState.absorb_skips). planner doesn't import us.
    from .planner import PrivateOnlySkips


@dataclass(frozen=True, slots=True)
class GrabRecord:
    """One grab, recorded for the end-of-run summary's "added" detail block."""

    title: str | None
    coverage: str | None
    url: str | None
    name: str | None
    group: str


class NeedsActionKind(Enum):
    """Why a title landed in "needs action". The summary's tips key off this, never the display reason text."""

    PRIVATE_ONLY = auto()
    """The warn-mode skip: a private-only recommended release with no fallback attempted."""
    PRIVATE_ONLY_NO_FALLBACK = auto()
    """Fallback mode that couldn't or wouldn't fall back."""
    PRIVATE_ONLY_STALE = auto()
    """Fallback mode refusing to replace an owned stale copy of the preferred private release."""
    UNSUPPORTED_TRACKER = auto()
    """A recommended release is on a tracker we have no parser for."""
    GRAB_FAILED = auto()
    """A contained transient failure (tracker/qBittorrent down)."""


@dataclass(frozen=True, slots=True)
class NeedsActionRecord:
    """One skip needing attention, recorded for the summary's "needs action" block."""

    title: str | None
    coverage: str | None
    group: str
    url: str | None
    reason: str
    """The human display text."""
    kind: NeedsActionKind
    """The closed classification the reporter's guidance gates on."""


@dataclass
class RunStats:
    """The per-run tally rendered by the end-of-run summary."""

    checked: int = 0
    added: list[GrabRecord] = field(default_factory=list[GrabRecord])
    up_to_date: int = 0
    cached: int = 0
    no_seadex_entry: int = 0
    seadex_unreachable: int = 0
    """Lookups skipped this run, counted apart from `no_seadex_entry`."""
    no_releases: int = 0
    no_mappings: int = 0
    needs_action: list[NeedsActionRecord] = field(
        default_factory=list[NeedsActionRecord],
    )
    unmonitored: int = 0
    queued: int = 0
    """Carried-over `QUEUED` records (a this-run grab stays in `added`)."""
    downloaded: int = 0
    """Carried-over `DOWNLOADED` records (a this-run grab stays in `added`)."""
    imported: int = 0
    """Carried-over `IMPORTED` records (a this-run grab stays in `added`)."""


@dataclass
class PerTitleState:
    """Per-title scratch flags, reset at the top of each title."""

    private_only_skipped: bool = False
    """A private-only release forced a skip, so the title must not be cached as done."""
    private_only_groups: list[str] = field(default_factory=list[str])
    """Group names of the private-only skip, riding along for the run summary's "needs action" list."""
    private_only_stale_held: bool = False
    """A stale owned private pick is held because only a fallback covers it."""
    fallback_covered: bool = False
    """The Arr already owns a public fallback's files."""
    unsupported_tracker_skipped: bool = False
    """The tracker has no parser, so the title must not be cached as done."""
    unsupported_tracker_groups: list[str] = field(default_factory=list[str])
    """Group names of the unsupported-tracker skip, riding along for the summary."""
    unsupported_tracker_hashes: list[str] = field(default_factory=list[str])
    """Hashes held out of the cached hash set, so the release is re-considered once a parser lands."""
    current_title: str | None = None
    """Title of the entry currently being processed, so grabs and the summary can attribute what they grab."""
    current_url: str | None = None
    """SeaDex URL of the entry currently being processed, so grabs and the summary can link what they grab."""
    current_coverage: str | None = None
    """Coverage of the entry currently being processed, so grabs and the summary can attribute what they grab."""

    def absorb_skips(self, skips: "PrivateOnlySkips") -> None:
        """Fold a planner private-only skip result onto this title's flags."""

        self.private_only_skipped |= skips.skipped
        self.private_only_groups.extend(skips.groups)
        self.private_only_stale_held |= skips.stale_held
        self.fallback_covered |= skips.fallback_covered


@dataclass
class RunContext:
    """Per-run state."""

    arr: Arr
    dry_run: bool = False
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF
    """The run's resolved wait mode (cli > config > default). `OFF` makes every pending-import path a no-op."""
    stats: RunStats = field(default_factory=RunStats)
    torrents_added: int = 0
    per_title: PerTitleState = field(default_factory=PerTitleState)
    """Per-title scratch flags, reassigned fresh at the top of each title so none leak into the next."""
    started_monotonic: float | None = None
    """Run clock (monotonic, so an NTP or DST step cannot move it)."""
    counts_mark: CountsMark = field(default_factory=lambda: SeverityCounts().bound_mark())
    """Stamped at run start and diffed for the summary's issues row (an unstamped ctx diffs to zero)."""
    pending_imports: list[PendingImport] = field(
        default_factory=list[PendingImport],
    )
    """Records written THIS run. The durable copies live in `cache_store`."""
    reacquired_keys: set[PendingKey] = field(default_factory=set[PendingKey])
    """Store-resident records re-seen in qBittorrent this run (`ALREADY_ADDED`), skipped by the snapshot and
    the tally. Never also a `pending_imports` record."""
    pending_states: dict[PendingKey, PendingState] = field(
        default_factory=dict[PendingKey, PendingState],
    )
    """Observed status of each carried-over record, keyed per record. Never a this-run grab, which stays
    `added`."""


def is_preview(ctx: RunContext, qbit: qbittorrentapi.Client | None) -> bool:
    """A run is a no-op preview when a dry run was requested OR qBittorrent is not configured."""

    return ctx.dry_run or qbit is None


# The first cause present wins. PRIVATE_ONLY can't co-occur with the fallback-mode kinds
# (private_releases is run-wide), so this only breaks ties between no-fallback and stale.
_TIP_PRECEDENCE: tuple[NeedsActionCause, ...] = (
    NeedsActionCause.PRIVATE_ONLY,
    NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    NeedsActionCause.PRIVATE_ONLY_STALE,
)


def _summary_tip(needs: tuple[NeedsActionFact, ...]) -> NeedsActionCause | None:
    """The cause whose guidance tip the summary shows, or None (renderer maps text)."""

    for cause in _TIP_PRECEDENCE:
        if any(fact.cause is cause for fact in needs):
            return cause
    return None


class RunReporter:
    """Owns the producer surface: each method emits a typed output event."""

    def __init__(
        self,
        *,
        emit: Emit,
        counts: CountsSource,
        cache_store: AbstractCacheStore,
        anilist: AniListGateway,
    ) -> None:
        self._emit = emit
        self._scopes = ScopeFactory(emit)
        self._counts = counts
        self.cache_store = cache_store
        self.anilist = anilist
        # The entry block currently open, None between entries. Boundaries and sibling rows close it first.
        self._entry: EntryScope | None = None

    # --- entry-scope lifecycle + emit helpers --------------------------------

    def _close_entry(self) -> None:
        """Close the open entry scope, if any (idempotent)."""

        if self._entry is not None:
            self._entry.close()
            self._entry = None

    def _open_entry(self, header: EntryHeader) -> None:
        """Close any open entry, then open a fresh one carrying its header."""

        self._close_entry()
        self._entry = self._scopes.entry(header)

    def _block(self, header: EntryHeader) -> None:
        """Emit a self-contained entry block: open its scope, then close it."""

        self._open_entry(header)
        self._close_entry()

    def _post(self, fact: EntryFact) -> None:
        """Post an entry fact on the open scope, else emit it scope-free.

        The scope-free arm is load-bearing, not defensive: the titled-row paths post details after `_ledger`
        closed the entry.
        """

        if self._entry is not None:
            self._entry.post(fact)
        else:
            self._emit(fact)

    def detail(self, label: str, value: StyledValue, *, severity: Severity = Severity.INFO) -> None:
        """The ONE entry-detail path: routes through the open scope, else scope-free."""

        self._post(EntryDetail(label=label, value=value, severity=severity))

    def post(self, fact: ReleaseSkipped | GrabFailed) -> None:
        """The ONE path for the typed release-level facts: via the open entry scope, else scope-free."""

        self._post(fact)

    def _ledger(self, state: EntryState, label: str) -> None:
        """Close any open entry, then emit a scope-free (col-0) ledger row."""

        self._close_entry()
        self._emit(LedgerRow(state, label))

    # --- run / item boundaries -----------------------------------------------

    def log_arr_start(self, arr: Arr, n_items: int) -> None:
        """Announce the start of the run (the per-arr scan-open boundary)."""

        self._close_entry()
        self._emit(ScanStarted(arr=arr, total=n_items))

    def log_arr_item_start(
        self,
        arr: Arr,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> None:
        """Announce the start of one Arr item (closes the previous item/entry)."""

        self._close_entry()
        self._emit(ItemStarted(arr=arr, index=n_item, total=n_items, title=item_title))

    # The two close boundaries carry no `log_` prefix: they state a boundary rather than report one, and no
    # renderer draws a line for either.
    def scan_finished(self, arr: Arr) -> None:
        """Close the scan and its open entry (the per-arr scan-close boundary)."""

        self._close_entry()
        self._emit(ScanFinished(arr=arr))

    def run_finished(self, arr: Arr) -> None:
        """Close the run (the leg-close boundary). bootstrap emits it on unwind."""

        self._close_entry()
        self._emit(RunFinished(arr=arr))

    # --- self-contained ledger rows ------------------------------------------

    def log_entry_status(self, state: EntryState, label: str) -> None:
        """Emit a one-line entry status as a self-contained (col-0) ledger row."""

        self._ledger(state, label)

    def log_arr_item_unmonitored(self, ctx: RunContext, item_title: str) -> None:
        """Report skipping an unmonitored item (bumps the tally, emits its row)."""

        ctx.stats.unmonitored += 1
        self._ledger(EntryState.UNMONITORED, item_title)

    def log_no_anilist_mappings(self, ctx: RunContext, title: str) -> None:
        """Report a title with no AniList mappings (bumps the tally, emits its row)."""

        ctx.stats.no_mappings += 1
        self._ledger(EntryState.NO_MAPPING, title)

    def log_ignored_anilist_id(self, al_id: int) -> None:
        """Report an AniList ID skipped via the ignore list."""

        self._ledger(EntryState.IGNORED, f"AniList #{al_id}")

    def log_no_sd_entry(self, ctx: RunContext, al_id: int) -> None:
        """Report an id with no SeaDex entry (bumps the tally, emits a titled row)."""

        ctx.stats.no_seadex_entry += 1
        self._log_titled_entry(EntryState.NO_ENTRY, al_id)

    def log_seadex_outage_skip(self, ctx: RunContext, al_id: int) -> None:
        """Report a title whose SeaDex lookup was skipped (SeaDex unreachable)."""

        ctx.stats.seadex_unreachable += 1
        # Prefer the cached name: an AniList lookup in a compound outage pays retry backoff per title.
        entry = self.cache_store.get_entry(ctx.arr, al_id)
        self._log_titled_entry(EntryState.SKIPPED, al_id, name=entry.name if entry is not None else None)
        # Scope-free (the titled row closed the entry): the reason rides col-0.
        self.detail("status", StyledValue("lookup skipped (SeaDex unreachable)", Accent.DIM))

    def _log_titled_entry(self, state: EntryState, al_id: int, *, name: str | None = None) -> None:
        """A ledger row for an id with no SeaDex entry block to show."""

        title = name if name is not None else self.anilist.title(al_id)
        self._ledger(state, title or f"AniList #{al_id}")
        # Only repeat the id when the ledger shows a title.
        if title:
            self.detail("anilist", StyledValue(str(al_id)))

    # --- entry-block headers -------------------------------------------------

    def log_al_title(
        self,
        ctx: RunContext,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> None:
        """Open the active-entry block: a "checking" header plus coverage/URL.

        `coverage` is a one-line range (e.g. "S04 E01-E12"). None or "" renders the URL only.
        """

        # Remembered so add_torrent and the summary can attribute what they grab, and show the files we
        # mapped from the Arr even when a release's own file list can't be parsed.
        ctx.per_title.current_title = anilist_title
        ctx.per_title.current_url = sd_entry.url
        ctx.per_title.current_coverage = coverage

        self._open_entry(
            EntryHeader(
                EntryState.CHECKING,
                anilist_title,
                al_id=sd_entry.anilist_id,
                coverage=coverage,
                url=sd_entry.url,
                incomplete=sd_entry.is_incomplete,
            ),
        )

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        # The builder keys row style on state, so a wider type would let a "cached" row render undimmed.
        state: Literal[EntryState.UNCHANGED, EntryState.IN_RADARR] = EntryState.UNCHANGED,
    ) -> None:
        """Emit a cached entry's self-contained block, read from the cache record.

        The explicit `arr` is the one the entry is cached under, which may not be the running arr.
        """

        ctx.stats.cached += 1

        entry = self.cache_store.get_entry(arr, al_id)
        title = entry.name if entry is not None else None
        if title is None:
            # None-gated: an empty stored name must NOT trigger a lookup.
            title = self.anilist.title(al_id)
        if title is None:
            title = "(unknown title)"

        # A complete block: self-close so a later diagnostic attributes to the open item, not to this row.
        self._block(
            EntryHeader(
                state,
                title,
                al_id=al_id,
                coverage=entry.coverage if entry is not None else None,
                url=entry.url if entry is not None else None,
            ),
        )

    # MISSING and ERRORED have no ledger vocabulary, so they render nothing inline.
    _PENDING_ENTRY_STATES: ClassVar[dict[PendingState, EntryState]] = {
        PendingState.QUEUED: EntryState.QUEUED,
        PendingState.DOWNLOADED: EntryState.DOWNLOADED,
        PendingState.IMPORTED: EntryState.IMPORTED,
    }

    def log_pending_snapshot(self, state: PendingState, pending: PendingImport) -> bool:
        """Emit a carried-over pending record's self-contained block inline in the series block.

        Bumps no counter: the engine owns the drop and count bookkeeping.
        """

        entry_state = self._PENDING_ENTRY_STATES.get(state)
        if entry_state is None:
            return False
        # Row style is renderer policy keyed on state, so the producer passes no style.
        self._block(
            EntryHeader(
                entry_state,
                pending.display_label,
                coverage=pending.coverage,
                url=pending.url,
            ),
        )
        return True

    # --- entry-block details -------------------------------------------------

    def log_no_seadex_releases(self, ctx: RunContext) -> None:
        """Report no suitable SeaDex releases (a status detail on the open entry)."""

        ctx.stats.no_releases += 1
        self.detail("status", StyledValue("no suitable releases on SeaDex", Accent.DIM))

    def log_seadex_action(
        self,
        seadex_dict: SeadexDict,
        results: list[ReleaseOutcome],
        dry_run: bool = False,
        monitor_active: bool = False,
    ) -> bool:
        """Post the action block for a title that differs from SeaDex's pick, after the adding has run."""

        # Nothing grabbed and nothing already present (every release skipped): leave the status to the
        # inline "skipped" warning.
        if not results and not dry_run:
            return False

        # A hashless/private release has no name, and the builder falls back to its group.
        added: list[ReleaseName] = []
        downloading: list[ReleaseName] = []
        for r in results:
            if r.added:
                added.append(ReleaseName(r.name or "", r.group))
            elif r.outcome is AddOutcome.ALREADY_ADDED:
                downloading.append(ReleaseName(r.name or "", r.group))

        if dry_run:
            status = GrabStatus.WOULD_ADD
        elif added:
            status = GrabStatus.ADDING
        else:
            status = GrabStatus.ALREADY_DOWNLOADING

        # Tag is a StrEnum, so sorting is by value.
        groups = tuple(
            RecommendedGroup(name=srg, tags=tuple(sorted(map(str, srg_item.tags))))
            for srg, srg_item in seadex_dict.items()
            if any(u.download for u in srg_item.urls.values())
        )

        self._post(
            GrabAction(
                status=status,
                groups=groups,
                added=tuple(added),
                downloading=tuple(downloading),
                waiting_to_import=monitor_active,
            ),
        )
        return True

    def log_max_torrents_added(self, cap: int) -> None:
        """Report hitting the per-run torrent cap (advanced.max_torrents_to_add)."""

        # Close the entry first: the scan breaks here and _finalize_run's check runs before the summary, so
        # a still-open entry frontier would misplace its diagnostics under the capped title.
        self._close_entry()
        self._emit(CapReached(cap=cap))

    # --- summary boundary ----------------------------------------------------

    def counts_mark(self) -> CountsMark:
        """The counts mark run start stamps. `log_run_summary` diffs against it."""

        return self._counts().bound_mark()

    def log_run_summary(self, ctx: RunContext, *, preview: bool, has_client: bool) -> None:
        """Emit the end-of-run scoreboard."""

        self._close_entry()

        # The mark carries the counter it was stamped on, so this diff can never read a different hub.
        since = ctx.counts_mark.since()

        dry_run_note = None
        if preview:
            dry_run_note = "nothing grabbed" if has_client else "qBittorrent not configured; nothing grabbed"

        tally = RunTally.from_stats(ctx.stats)
        elapsed_s = (time.monotonic() - ctx.started_monotonic) if ctx.started_monotonic is not None else None

        self._emit(
            RunSummaryReady(
                summary=RunSummary(
                    arr=ctx.arr,
                    dry_run_note=dry_run_note,
                    added_count=ctx.torrents_added,
                    tally=tally,
                    wait_mode_on=ctx.import_wait_mode is not ImportWaitMode.OFF,
                    warnings=since.warning,
                    # The errors property sums ERROR and CRITICAL.
                    errors=since.errors,
                    elapsed_s=elapsed_s,
                    tip=_summary_tip(tally.needs_action),
                ),
            ),
        )
