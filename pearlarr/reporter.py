"""The run reporter: the typed emission path the run logs through."""

import time
from typing import ClassVar

from seadex import EntryRecord

from .anilist_gateway import AniListGateway
from .cache import AbstractCacheStore
from .config import Arr
from .leg_state import RunContext, Title, resolve_entry_title
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
    LegFinished,
    LegSummary,
    LegTally,
    NeedsActionCause,
    NeedsActionFact,
    RecommendedGroup,
    ReleaseName,
    ReleaseSkipped,
    RunSummaryReady,
    ScanFinished,
    ScanStarted,
    ScopeFactory,
    Severity,
    StyledValue,
)
from .output.events import EntryState
from .pending_import import PendingImport
from .releases import SeadexDict
from .torrents import AddOutcome, ReleaseOutcome
from .wait_types import ImportWaitMode, PendingImportState

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
        self._emit(LegFinished(arr=arr))

    # --- self-contained ledger rows ------------------------------------------

    def log_no_episodes(self, title: str) -> None:
        """Report a title whose mapping resolved no episodes, as a self-contained (col-0) ledger row."""

        self._ledger(EntryState.NO_EPISODES, title)

    def log_arr_item_unmonitored(self, ctx: RunContext, item_title: str) -> None:
        """Report skipping an unmonitored item (bumps the tally, emits its row)."""

        ctx.stats.unmonitored += 1
        self._ledger(EntryState.UNMONITORED, item_title)

    def log_no_anilist_mappings(self, ctx: RunContext, title: str) -> None:
        """Report a title with no AniList mappings (bumps the tally, emits its row)."""

        ctx.stats.no_mappings += 1
        self._ledger(EntryState.NO_MAPPING, title)

    def log_ignored_anilist_id(self, ctx: RunContext, al_id: int) -> None:
        """Report an AniList ID skipped via the ignore list.

        No AniList lookup: an ignored id is dropped before the prefetch, so the arr's own title
        is the only one in hand and asking would cost a request per ignored id.
        """

        self._titled_row(EntryState.IGNORED, resolve_entry_title(al_id, (), ctx.arr_title))

    def log_no_sd_entry(self, ctx: RunContext, al_id: int) -> None:
        """Report an id with no SeaDex entry (bumps the tally, emits a titled row)."""

        ctx.stats.no_seadex_entry += 1
        self._titled_row(EntryState.NO_ENTRY, self._display_title(ctx, al_id, None))

    def log_seadex_outage_skip(self, ctx: RunContext, al_id: int) -> None:
        """Report a title whose SeaDex lookup was skipped (SeaDex unreachable)."""

        ctx.stats.seadex_unreachable += 1
        # The stored name wins: a prior run resolved it, so no lookup is needed.
        entry = self.cache_store.get_entry(ctx.arr, al_id)
        title = self._display_title(ctx, al_id, entry.name if entry is not None else None)
        self._titled_row(EntryState.SKIPPED, title)
        # Scope-free (the titled row closed the entry): the reason rides col-0.
        self.detail("status", StyledValue("lookup skipped (SeaDex unreachable)", Accent.DIM))

    def _display_title(self, ctx: RunContext, al_id: int, stored: str | None) -> Title:
        """The entry's label: the stored name, else AniList, else the arr's own title, else the id form."""

        return resolve_entry_title(al_id, (stored,) if stored else self.anilist.titles(al_id), ctx.arr_title)

    def _titled_row(self, state: EntryState, title: Title) -> None:
        """A ledger row for an id with no entry block, the id repeated only when the row shows a title."""

        self._ledger(state, title.name)
        if title.resolved:
            self.detail("anilist", StyledValue(str(title.al_id)))

    # --- entry-block headers -------------------------------------------------

    def log_al_title(
        self,
        ctx: RunContext,
        title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> None:
        """Open the active-entry block: a "checking" header plus coverage/URL.

        `coverage` is a one-line range (e.g. "S04 E01-E12"). None or "" renders the URL only.
        """

        # Remembered so add_torrent and the summary can attribute what they grab, and show the files we
        # mapped from the Arr even when a release's own file list can't be parsed.
        ctx.per_title.current_title = title
        ctx.per_title.current_url = sd_entry.url
        ctx.per_title.current_coverage = coverage

        self._open_entry(
            EntryHeader(
                EntryState.CHECKING,
                title,
                al_id=sd_entry.anilist_id,
                coverage=coverage,
                url=sd_entry.url,
                incomplete=sd_entry.is_incomplete,
            ),
        )

    def log_cached_entry(self, ctx: RunContext, al_id: int) -> None:
        """Emit a cached entry's self-contained block, read from the cache record."""

        ctx.stats.unchanged += 1

        entry = self.cache_store.get_entry(ctx.arr, al_id)
        title = self._display_title(ctx, al_id, entry.name if entry is not None else None)

        # A complete block: self-close so a later diagnostic attributes to the open item, not to this row.
        self._block(
            EntryHeader(
                EntryState.UNCHANGED,
                title.name,
                al_id=al_id,
                coverage=entry.coverage if entry is not None else None,
                url=entry.url if entry is not None else None,
            ),
        )

    # MISSING and ERRORED have no ledger vocabulary, so they render nothing inline.
    _PENDING_ENTRY_STATES: ClassVar[dict[PendingImportState, EntryState]] = {
        PendingImportState.DOWNLOADING: EntryState.QUEUED,
        PendingImportState.DOWNLOADED: EntryState.DOWNLOADED,
        PendingImportState.IMPORTED: EntryState.IMPORTED,
    }

    def log_pending_snapshot(self, state: PendingImportState, pending: PendingImport, series_id: int) -> None:
        """Emit a carried-over pending record's self-contained block inline in the series block.

        The header reads the record's claim on `series_id`. Bumps no counter: the engine owns the
        drop and count bookkeeping.
        """

        entry_state = self._PENDING_ENTRY_STATES.get(state)
        if entry_state is None:
            return
        claim = pending.claim_for(series_id)
        # Row style is renderer policy keyed on state, so the producer passes no style.
        self._block(
            EntryHeader(
                entry_state,
                pending.display_label,
                coverage=claim.coverage,
                url=claim.url,
            ),
        )

    # --- entry-block details -------------------------------------------------

    def log_no_seadex_releases(self, ctx: RunContext) -> None:
        """Report no suitable SeaDex releases (a status detail on the open entry)."""

        ctx.stats.no_releases += 1
        self.detail("status", StyledValue("no suitable releases on SeaDex", Accent.DIM))

    def log_seadex_action(
        self,
        seadex_dict: SeadexDict,
        results: list[ReleaseOutcome],
        *,
        preview: bool,
        waiting_to_import: bool,
    ) -> None:
        """Post the action block for a title that differs from SeaDex's pick, after the adding has run.

        Posts nothing when no release was added or already in qBittorrent.
        """

        # Nothing was added and nothing was already in qBittorrent, in a real run or a preview: the skipped, failed
        # and held-by-cap rows say why.
        if not results:
            return

        # A release neither qBittorrent nor its tracker page named has no name, and `ReleaseName.display` falls back
        # to its group.
        added: list[ReleaseName] = []
        downloading: list[ReleaseName] = []
        for r in results:
            if r.added:
                added.append(ReleaseName(r.name or "", r.group))
            elif r.outcome is AddOutcome.ALREADY_IN_QBIT:
                downloading.append(ReleaseName(r.name or "", r.group))

        # `added` first, so a preview says "would add" only for a title it would add something for.
        if not added:
            status = GrabStatus.ALREADY_DOWNLOADING
        elif preview:
            status = GrabStatus.WOULD_ADD
        else:
            status = GrabStatus.ADDING

        # Tag is a StrEnum, so sorting is by value.
        groups = tuple(
            RecommendedGroup(name=srg, tags=tuple(sorted(map(str, srg_item.tags))))
            for srg, srg_item in seadex_dict.items()
            if any(u.to_grab for u in srg_item.releases.values())
        )

        self._post(
            GrabAction(
                status=status,
                groups=groups,
                added=tuple(added),
                in_qbit=tuple(downloading),
                waiting_to_import=waiting_to_import,
            ),
        )

    def log_max_torrents_added(self, cap: int) -> None:
        """Report hitting the per-run torrent cap (advanced.max_torrents_to_add)."""

        # Close the entry first: the cap line is a run-level fact and the capped title logs nothing further, so a
        # still-open entry frontier would misplace it under that title.
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

        tally = LegTally.from_stats(ctx.stats)
        elapsed_s = (time.monotonic() - ctx.started_monotonic) if ctx.started_monotonic is not None else None

        self._emit(
            RunSummaryReady(
                summary=LegSummary(
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
