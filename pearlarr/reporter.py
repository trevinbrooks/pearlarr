"""The arr run reporter: the typed emission path the arr run logs through."""

import time
from typing import ClassVar

from seadex import EntryRecord

from .anilist_gateway import AniListGateway
from .arr_run_state import ArrRunState, Title, title_from_names
from .cache import AbstractCacheStore
from .config import Arr
from .output import (
    Accent,
    ArrRunFinished,
    ArrRunSummary,
    ArrRunSummaryReady,
    ArrRunTally,
    CapReached,
    CountsMark,
    CountsSource,
    Emit,
    GrabAction,
    GrabFailed,
    GrabStatus,
    GroupLabel,
    ItemStarted,
    NeedsActionCause,
    NeedsActionRow,
    ReleaseLabel,
    ReleaseSkipped,
    ScanFinished,
    ScanStarted,
    ScopeFactory,
    Severity,
    StyledValue,
    TitleDetail,
    TitleEvent,
    TitleHeader,
    TitleRow,
    TitleScope,
)
from .output.events import TitleState
from .pending_import import PendingImport
from .releases import SeadexDict
from .torrent_service import AddStatus, GrabResult
from .wait_types import ImportWaitMode, PendingImportState

# The first cause present wins. PRIVATE_ONLY can't co-occur with the `fallback` causes
# (private_releases is run-wide), so this only breaks ties between no-fallback and outdated.
_TIP_PRECEDENCE: tuple[NeedsActionCause, ...] = (
    NeedsActionCause.PRIVATE_ONLY,
    NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    NeedsActionCause.PRIVATE_ONLY_OUTDATED,
)


def _summary_tip(needs: tuple[NeedsActionRow, ...]) -> NeedsActionCause | None:
    """The cause whose guidance tip the summary shows, or None (renderer maps text)."""

    for cause in _TIP_PRECEDENCE:
        if any(fact.cause is cause for fact in needs):
            return cause
    return None


class RunReporter:
    """The arr run's emitting interface: each method emits a typed output event."""

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
        # The title scope currently open, None between titles. Boundaries and sibling rows close it first.
        self._title_scope: TitleScope | None = None

    # --- title-scope lifecycle + emit helpers --------------------------------

    def _close_title_scope(self) -> None:
        """Close the open title scope, if any (idempotent)."""

        if self._title_scope is not None:
            self._title_scope.close()
            self._title_scope = None

    def _open_title_scope(self, header: TitleHeader) -> None:
        """Close any open title scope, then open a fresh one carrying its header."""

        self._close_title_scope()
        self._title_scope = self._scopes.title(header)

    def _block(self, header: TitleHeader) -> None:
        """Emit a self-contained title block: open its scope, then close it."""

        self._open_title_scope(header)
        self._close_title_scope()

    def _post(self, fact: TitleEvent) -> None:
        """Post a title fact on the open scope, else emit it scope-free.

        The scope-free branch is load-bearing, not defensive: the titled-row paths post details after `_emit_title_row`
        closed the title scope.
        """

        if self._title_scope is not None:
            self._title_scope.post(fact)
        else:
            self._emit(fact)

    def detail(self, label: str, value: StyledValue, *, severity: Severity = Severity.INFO) -> None:
        """The ONE title-detail path: routes through the open scope, else scope-free."""

        self._post(TitleDetail(label=label, value=value, severity=severity))

    def post(self, fact: ReleaseSkipped | GrabFailed) -> None:
        """The ONE path for the typed release-level facts: via the open title scope, else scope-free."""

        self._post(fact)

    def _emit_title_row(self, state: TitleState, label: str) -> None:
        """Close any open title scope, then emit a scope-free (col-0) title row."""

        self._close_title_scope()
        self._emit(TitleRow(state, label))

    # --- arr run / item boundaries -------------------------------------------

    def log_arr_start(self, arr: Arr, n_items: int) -> None:
        """Announce the start of the scan (the per-arr scan-open boundary)."""

        self._close_title_scope()
        self._emit(ScanStarted(arr=arr, total=n_items))

    def log_arr_item_start(
        self,
        arr: Arr,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> None:
        """Announce the start of one Arr item (closes the previous item/title)."""

        self._close_title_scope()
        self._emit(ItemStarted(arr=arr, index=n_item, total=n_items, title=item_title))

    # The two close boundaries carry no `log_` prefix: they state a boundary rather than report one, and no
    # renderer draws a line for either.
    def scan_finished(self, arr: Arr) -> None:
        """Close the scan and its open title scope (the per-arr scan-close boundary)."""

        self._close_title_scope()
        self._emit(ScanFinished(arr=arr))

    def arr_run_finished(self, arr: Arr) -> None:
        """Close the arr run (the arr-run-close boundary). bootstrap emits it on unwind."""

        self._close_title_scope()
        self._emit(ArrRunFinished(arr=arr))

    # --- self-contained title rows -------------------------------------------

    def log_no_episodes(self, title: str) -> None:
        """Report a title whose mapping resolved no episodes, as a self-contained (col-0) title row."""

        self._emit_title_row(TitleState.NO_EPISODES, title)

    def log_arr_item_unmonitored(self, state: ArrRunState, item_title: str) -> None:
        """Report an unmonitored item (bumps the tally, emits its row)."""

        state.stats.unmonitored += 1
        self._emit_title_row(TitleState.UNMONITORED, item_title)

    def log_no_anilist_mappings(self, state: ArrRunState, title: str) -> None:
        """Report a library item with no AniList mappings (bumps the tally, emits its row)."""

        state.stats.no_mappings += 1
        self._emit_title_row(TitleState.NO_MAPPING, title)

    def log_ignored_anilist_id(self, state: ArrRunState, anilist_id: int) -> None:
        """Report an AniList ID the ignore list leaves out.

        No AniList lookup: an ignored id is dropped before the prefetch, so the arr item's own name
        is the only one in hand and asking would cost a request per ignored id.
        """

        self._titled_row(TitleState.IGNORED, title_from_names(anilist_id, (), state.arr_title))

    def log_no_sd_entry(self, state: ArrRunState, anilist_id: int) -> None:
        """Report a title with no SeaDex entry (bumps the tally, emits a titled row)."""

        state.stats.no_seadex_entry += 1
        self._titled_row(TitleState.NO_ENTRY, self._display_title(state, anilist_id, None))

    def log_seadex_unreachable(self, state: ArrRunState, anilist_id: int) -> None:
        """Report a title whose SeaDex lookup didn't happen (SeaDex unreachable)."""

        state.stats.seadex_unreachable += 1
        # The stored name wins: a prior run resolved it, so no lookup is needed.
        cached = self.cache_store.get_cached_title(state.arr, anilist_id)
        title = self._display_title(state, anilist_id, cached.name if cached is not None else None)
        self._titled_row(TitleState.NOT_CHECKED, title)
        # Scope-free (the titled row closed the title scope): the reason goes in col-0.
        self.detail("status", StyledValue("SeaDex unreachable", Accent.DIM))

    def _display_title(self, state: ArrRunState, anilist_id: int, stored: str | None) -> Title:
        """The title's label: the stored name, else AniList, else the arr item's own name, else the id form."""

        return title_from_names(anilist_id, (stored,) if stored else self.anilist.titles(anilist_id), state.arr_title)

    def _titled_row(self, state: TitleState, title: Title) -> None:
        """A title row for a title with no title block, the id repeated only when the row shows a name."""

        self._emit_title_row(state, title.name)
        if title.resolved:
            self.detail("anilist", StyledValue(str(title.anilist_id)))

    # --- title-block headers -------------------------------------------------

    def open_title_block(
        self,
        state: ArrRunState,
        title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> None:
        """Open the active title's block: a "checking" header plus coverage/URL.

        `coverage` is a one-line range (e.g. "S04 E01-E12"). None or "" renders the URL only.
        """

        # Remembered so grab_releases and the summary can attribute what they grab, and show the files we
        # read from the Arr even when a release's own file list can't be parsed.
        state.per_title.title_name = title
        state.per_title.entry_url = sd_entry.url
        state.per_title.entry_coverage = coverage

        self._open_title_scope(
            TitleHeader(
                TitleState.CHECKING,
                title,
                anilist_id=sd_entry.anilist_id,
                coverage=coverage,
                url=sd_entry.url,
                incomplete=sd_entry.is_incomplete,
            ),
        )

    def log_cached_title(self, state: ArrRunState, anilist_id: int) -> None:
        """Emit an unchanged title's self-contained block, read from its cached row."""

        state.stats.unchanged += 1

        cached = self.cache_store.get_cached_title(state.arr, anilist_id)
        title = self._display_title(state, anilist_id, cached.name if cached is not None else None)

        # A complete block: self-close so a later diagnostic attributes to the open item, not to this row.
        self._block(
            TitleHeader(
                TitleState.UNCHANGED,
                title.name,
                anilist_id=anilist_id,
                coverage=cached.coverage if cached is not None else None,
                url=cached.url if cached is not None else None,
            ),
        )

    # GONE and ERRORED have no title-row vocabulary, so they render nothing inline.
    _TITLE_STATE_FOR_PENDING_IMPORT: ClassVar[dict[PendingImportState, TitleState]] = {
        PendingImportState.DOWNLOADING: TitleState.DOWNLOADING,
        PendingImportState.DOWNLOADED: TitleState.DOWNLOADED,
        PendingImportState.IMPORTED: TitleState.IMPORTED,
    }

    def log_pending_import(self, state: PendingImportState, pending: PendingImport, series_id: int) -> None:
        """Emit a carried-over pending import's self-contained block inline in the series block.

        The header reads the pending import's claim on `series_id`. Bumps no counter: the wait manager owns the
        drop and count bookkeeping.
        """

        title_state = self._TITLE_STATE_FOR_PENDING_IMPORT.get(state)
        if title_state is None:
            return
        claim = pending.claim_for(series_id)
        # Row style is renderer policy keyed on state, so the reporter passes no style.
        self._block(
            TitleHeader(
                title_state,
                pending.display_label,
                coverage=claim.window_coverage,
                url=claim.url,
            ),
        )

    # --- title-block details -------------------------------------------------

    def log_no_seadex_releases(self, state: ArrRunState) -> None:
        """Report no suitable SeaDex releases (a status detail on the open title)."""

        state.stats.no_releases += 1
        self.detail("status", StyledValue("no suitable releases on SeaDex", Accent.DIM))

    def log_seadex_action(
        self,
        seadex_dict: SeadexDict,
        results: list[GrabResult],
        *,
        dry_run: bool,
        waiting_to_import: bool,
    ) -> None:
        """Post the action block for a title that differs from SeaDex's preferred releases, after the grabs have run.

        Posts nothing when no release was grabbed or already in qBittorrent.
        """

        # Nothing was grabbed and nothing was already in qBittorrent, in a real run or a dry run: the skipped, failed
        # and held-by-cap rows say why.
        if not results:
            return

        # A release neither qBittorrent nor its tracker page named has no name, and `ReleaseLabel.display` falls back
        # to its group.
        added: list[ReleaseLabel] = []
        downloading: list[ReleaseLabel] = []
        for r in results:
            if r.added:
                added.append(ReleaseLabel(r.name or "", r.group))
            elif r.outcome is AddStatus.ALREADY_IN_QBIT:
                downloading.append(ReleaseLabel(r.name or "", r.group))

        # `added` first, so a dry run says "would add" only for a title it would grab something for.
        if not added:
            status = GrabStatus.ALREADY_IN_QBIT
        elif dry_run:
            status = GrabStatus.WOULD_GRAB
        else:
            status = GrabStatus.GRABBING

        # Tag is a StrEnum, so sorting is by value.
        groups = tuple(
            GroupLabel(name=srg, tags=tuple(sorted(map(str, group_releases.tags))))
            for srg, group_releases in seadex_dict.items()
            if any(u.to_grab for u in group_releases.releases.values())
        )

        self._post(
            GrabAction(
                status=status,
                groups=groups,
                grabbed=tuple(added),
                in_qbit=tuple(downloading),
                waiting_to_import=waiting_to_import,
            ),
        )

    def log_max_torrents_added(self, cap: int) -> None:
        """Report hitting the per-arr-run torrent cap (advanced.max_torrents_to_add)."""

        # Close the title scope first: the cap line is an arr-run-level fact and the capped title logs nothing
        # further, so a still-open title scope would misplace it under that title.
        self._close_title_scope()
        self._emit(CapReached(cap=cap))

    # --- summary boundary ----------------------------------------------------

    def counts_mark(self) -> CountsMark:
        """The counts mark taken at the arr run's start. `log_arr_run_summary` diffs against it."""

        return self._counts().bound_mark()

    def log_arr_run_summary(self, state: ArrRunState, *, dry_run: bool, has_client: bool) -> None:
        """Emit the arr run summary."""

        self._close_title_scope()

        # The mark carries the counter it was taken on, so this diff can never read a different hub.
        since = state.counts_mark.since()

        dry_run_note = None
        if dry_run:
            dry_run_note = "nothing grabbed" if has_client else "qBittorrent not configured; nothing grabbed"

        tally = ArrRunTally.from_stats(state.stats)
        elapsed_s = (time.monotonic() - state.started_monotonic) if state.started_monotonic is not None else None

        self._emit(
            ArrRunSummaryReady(
                summary=ArrRunSummary(
                    arr=state.arr,
                    dry_run_note=dry_run_note,
                    grabbed_count=state.torrents_added,
                    tally=tally,
                    wait_mode_on=state.import_wait_mode is not ImportWaitMode.OFF,
                    warnings=since.warning,
                    # The errors property sums ERROR and CRITICAL.
                    errors=since.errors,
                    elapsed_s=elapsed_s,
                    tip=_summary_tip(tally.needs_action),
                ),
            ),
        )
