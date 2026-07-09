import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import qbittorrentapi
from seadex import EntryRecord

from .anilist_gateway import AniListGateway
from .cache import AbstractCacheStore
from .config import Arr
from .log import EntryState, log_counter
from .manual_import import ImportWaitMode, PendingImport, PendingState
from .output import (
    Accent,
    CapReached,
    Emit,
    EntryDetail,
    EntryFact,
    EntryHeader,
    EntryScope,
    GrabAction,
    GrabStatus,
    ItemStarted,
    LedgerRow,
    NeedsActionCause,
    NeedsActionFact,
    RecommendedGroup,
    ReleaseName,
    RunFinished,
    RunSummary,
    RunSummaryReady,
    RunTally,
    ScanFinished,
    ScanStarted,
    ScopeFactory,
    StyledValue,
)
from .seadex_types import SeadexDict
from .torrents import AddOutcome, ReleaseOutcome


@dataclass(frozen=True, slots=True)
class GrabRecord:
    """One grab, recorded for the end-of-run summary's "added" detail block.

    Replaces the ``{"title", "coverage", "url", "name", "group"}`` item dict.
    """

    title: str | None
    coverage: str | None
    url: str | None
    name: str | None
    group: str


class NeedsActionKind(Enum):
    """Why a title landed in the "needs action" block - the machine-readable gate.

    The summary's guidance tips key off this, never off the display ``reason``
    text (which can be reworded freely). PRIVATE_ONLY is the warn-mode skip;
    PRIVATE_ONLY_NO_FALLBACK is fallback mode that couldn't (no public
    alternative covers the held files) or wouldn't (an interactive private
    pick) fall back, so its tip must not suggest turning fallback on.
    PRIVATE_ONLY_STALE is fallback mode refusing to replace an owned stale
    copy of the preferred private release (an alternative exists; the
    fallback-never-supersedes rule holds it). GRAB_FAILED is a contained
    transient failure (tracker/qBittorrent down); the title stays uncached and
    retries next run, so it gets no tip.
    """

    PRIVATE_ONLY = auto()
    PRIVATE_ONLY_NO_FALLBACK = auto()
    PRIVATE_ONLY_STALE = auto()
    UNSUPPORTED_TRACKER = auto()
    GRAB_FAILED = auto()


@dataclass(frozen=True, slots=True)
class NeedsActionRecord:
    """One skip needing the user, recorded for the summary's "needs action" block.

    ``reason`` is the human display text; ``kind`` is the closed classification
    the reporter's guidance gates on.
    """

    title: str | None
    coverage: str | None
    group: str
    url: str | None
    reason: str
    kind: NeedsActionKind


@dataclass
class RunStats:
    """The per-run tally rendered by the end-of-run summary.

    Replaces the 10-key ``fresh_stats()`` dict: field names equal the old keys, so
    counter bumps and list appends produce identical tallies - but a typo now
    fails to compile instead of silently birthing a key.
    """

    checked: int = 0
    added: list[GrabRecord] = field(default_factory=list[GrabRecord])
    up_to_date: int = 0
    cached: int = 0
    no_seadex_entry: int = 0
    # Lookups skipped because SeaDex was unreachable this run - counted apart
    # from no_seadex_entry so an outage never reads as "SeaDex has no data".
    seadex_unreachable: int = 0
    no_releases: int = 0
    no_mappings: int = 0
    needs_action: list[NeedsActionRecord] = field(
        default_factory=list[NeedsActionRecord],
    )
    unmonitored: int = 0
    # Carried-over pending-import counts by current status (NEVER this-run grabs -
    # those stay `added`). Distinct int fields so a typo fails to compile instead
    # of silently birthing a key, and so the summary can render each on its own row.
    queued: int = 0
    importing: int = 0
    imported: int = 0


@dataclass
class RunContext:
    """Per-run state, created fresh at the top of each run.

    Replaces the run-scoped mutable ``self.*`` fields of the old god class so the
    decision engine, the torrent service, and the reporter can read and return
    data instead of mutating shared orchestrator state.
    """

    arr: Arr
    dry_run: bool = False
    # The run's resolved wait-for-completion mode (cli > config > default), stamped
    # in reset_run_stats; OFF makes every pending-import path a no-op.
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF
    stats: RunStats = field(default_factory=RunStats)
    torrents_added: int = 0
    # Title, SeaDex URL, and coverage of the entry currently being processed, so
    # grabs and the summary can attribute and link what they grab.
    current_title: str | None = None
    current_url: str | None = None
    current_coverage: str | None = None
    # Set per-title when a private-only release forces a skip, so
    # the caller knows not to cache the title as done; the group names ride along
    # for the run summary's "needs action" list.
    private_only_skipped: bool = False
    private_only_groups: list[str] = field(default_factory=list[str])
    # Set per-title when an owned-at-stale-size private pick is held because only
    # a fallback covers it (never a replacement): picks the summary row's kind.
    private_only_stale_held: bool = False
    # Set per-title when the Arr already owns a public fallback's files (the
    # owned-fallback soft-skip): drives the cache's fallback-satisfied marker.
    fallback_covered: bool = False
    # Set per-title when a recommended release is on a tracker we have no parser for
    # (so we can't grab it, but the user didn't deselect it): keeps the title from
    # being cached as done, and the group names ride along for the summary. The
    # skipped hashes are excluded from the cached hash set on a mixed (something
    # else grabbed) title, so the release is re-considered once a parser lands.
    unsupported_tracker_skipped: bool = False
    unsupported_tracker_groups: list[str] = field(default_factory=list[str])
    unsupported_tracker_hashes: list[str] = field(default_factory=list[str])
    # Run clock (monotonic, so an NTP/DST step can't yield negative elapsed) and
    # the logger-counter snapshot taken at the start, diffed for the summary.
    started_monotonic: float | None = None
    log_counts_at_start: dict[int, int] = field(default_factory=dict[int, int])
    # PendingImport records written THIS run (on a successful add), for the
    # end-of-run blocking pass; the durable copies live in cache_store under
    # ``pending_imports``, so this is just the fast in-memory list to wait on.
    pending_imports: list[PendingImport] = field(
        default_factory=list[PendingImport],
    )
    # The classified status of each CARRIED-OVER record touched this run (by the
    # per-series inline snapshot or the deferred reconcile), keyed by infohash.
    # Read by the pre-summary tally so each carried-over record is counted exactly
    # once by its known status (un-touched store records default to QUEUED). Never
    # holds a this-run grab (those stay `added`).
    pending_states: dict[str, PendingState] = field(
        default_factory=dict[str, PendingState],
    )


def is_preview(ctx: RunContext, qbit: qbittorrentapi.Client | None) -> bool:
    """A run is a no-op preview when a dry run was requested OR qBittorrent is not
    configured (nothing can actually be grabbed).

    Module-level so every per-run collaborator computes preview identically from
    the shared :class:`RunContext` + client, rather than each re-deriving it.
    """

    return ctx.dry_run or qbit is None


# The summary tip precedence: the first cause present wins. PRIVATE_ONLY can't
# co-occur with the fallback-mode kinds (private_releases is run-wide), so this
# only breaks ties between the two fallback kinds - no-fallback over stale.
_TIP_PRECEDENCE: tuple[NeedsActionCause, ...] = (
    NeedsActionCause.PRIVATE_ONLY,
    NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
    NeedsActionCause.PRIVATE_ONLY_STALE,
)


def _summary_tip(needs: tuple[NeedsActionFact, ...]) -> NeedsActionCause | None:
    """The cause whose guidance tip the summary shows, or None (renderer maps text).

    Derived from the MAPPED tally causes so the kind->cause mapping stays
    single-sited in :meth:`RunTally.from_stats`.
    """

    for cause in _TIP_PRECEDENCE:
        if any(fact.cause is cause for fact in needs):
            return cause
    return None


class RunReporter:
    """Owns the producer surface: each method EMITS a typed output event.

    Built once per arr instance with its stable collaborators (the ``emit`` seam,
    the logger whose LogCounter the summary diffs, the cache store, the AniList
    gateway). The methods hold every producer-side decision - stats bumps, ctx
    mutation, gates, title fallbacks - then state WHAT happened as an event; the
    scan-line builders (``output.scan_lines``) own the layout. An open entry block
    rides ``self._entry``: a CHECKING header opens one and its details stream
    through it (any boundary/sibling closes it, idempotently, first); a COMPLETE
    block (cached / carried-over pending) opens and self-closes via ``_block``, so
    a gap diagnostic keeps col 0 rather than indenting under the finished block.
    """

    def __init__(
        self,
        *,
        emit: Emit,
        logger: logging.Logger,
        cache_store: AbstractCacheStore,
        anilist: AniListGateway,
    ) -> None:
        self._emit = emit
        self._scopes = ScopeFactory(emit)
        self.logger = logger
        self.cache_store = cache_store
        self.anilist = anilist
        # The entry block currently open (a coverage/url-bearing header); None
        # between entries. Boundaries and sibling rows close it before emitting.
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
        """Emit a self-contained entry block: open its scope, then close it.

        For COMPLETE blocks (cached / carried-over pending) whose header carries
        the whole row (coverage/url baked in) and that accrue no followers -
        closing before return keeps a gap diagnostic (e.g. a retry WARNING while
        resolving the next title) at col 0, not indented under the finished block.
        """

        self._open_entry(header)
        self._close_entry()

    def _post(self, fact: EntryFact) -> None:
        """Post an entry fact on the open scope, else emit it scope-free.

        The scope-free arm is LOAD-BEARING, not defensive: the titled-row paths
        (no-entry / outage) post their anilist/status details AFTER ``_ledger``
        closed the entry, so those details ride col-0 by design.
        """

        if self._entry is not None:
            self._entry.post(fact)
        else:
            self._emit(fact)

    def _detail(self, label: str, value: StyledValue) -> None:
        """The ONE entry-detail path: routes through the open scope, else scope-free.

        Making this the only way to emit a detail keeps a post-close detail (the
        no-entry / outage paths, where ``_entry`` is already None) from being
        written against a stale ``self._entry`` out of habit.
        """

        self._post(EntryDetail(label=label, value=value))

    def _ledger(self, state: EntryState, label: str) -> None:
        """Close any open entry, then emit a scope-free (col-0) ledger row.

        The bare titled rows (unmonitored / no-mapping / ignored / no-entry /
        skipped / IN_RADARR / NO_EPISODES) are self-contained: they close the
        prior entry and never open one, so the row - and any diagnostic beside it -
        keeps today's col-0 punch-through.
        """

        self._close_entry()
        self._emit(LedgerRow(state, label))

    # --- run / item boundaries -----------------------------------------------

    def log_arr_start(self, arr: Arr, n_items: int) -> bool:
        """Announce the start of the run (the per-arr scan-open boundary).

        Args:
            arr: Type of arr instance
            n_items: Total number of shows/movies
        """

        self._close_entry()
        self._emit(ScanStarted(arr=arr, total=n_items))
        return True

    def log_arr_item_start(
        self,
        arr: Arr,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> bool:
        """Announce the start of one Arr item (closes the previous item/entry).

        Args:
            arr: Type of arr instance
            item_title: Title for the item
            n_item: Number for the show/movie
            n_items: Total number of shows/movies
        """

        self._close_entry()
        self._emit(ItemStarted(arr=arr, index=n_item, total=n_items, title=item_title))
        return True

    # The two close boundaries carry no ``log_`` prefix and return nothing: they
    # state a boundary rather than report one, and no renderer draws a line for
    # either (rich passes, legacy echoes nothing, the text sink skips them).

    def scan_finished(self, arr: Arr) -> None:
        """Close the scan and its open entry (the per-arr scan-close boundary).

        Args:
            arr: Type of arr instance
        """

        self._close_entry()
        self._emit(ScanFinished(arr=arr))

    def run_finished(self, arr: Arr) -> None:
        """Close the run (the leg-close boundary); bootstrap emits it on unwind.

        The entry close is defensive - ``scan_finished`` ran on every path here.

        Args:
            arr: Type of arr instance
        """

        self._close_entry()
        self._emit(RunFinished(arr=arr))

    # --- self-contained ledger rows ------------------------------------------

    def log_entry_status(self, state: EntryState, label: str) -> bool:
        """Emit a one-line entry status as a self-contained (col-0) ledger row.

        Args:
            state (EntryState): Which entry-level outcome this row reports
            label (str): What the state applies to (usually a title)
        """

        self._ledger(state, label)
        return True

    def log_arr_item_unmonitored(self, ctx: RunContext, item_title: str) -> bool:
        """Report skipping an unmonitored item (bumps the tally, emits its row).

        Args:
            ctx (RunContext): The run's state (stats tally).
            item_title (str): Item title
        """

        ctx.stats.unmonitored += 1
        self._ledger(EntryState.UNMONITORED, item_title)
        return True

    def log_no_anilist_mappings(self, ctx: RunContext, title: str) -> bool:
        """Report a title with no AniList mappings (bumps the tally, emits its row).

        Args:
            ctx (RunContext): The run's state (stats tally).
            title: Title for the item
        """

        ctx.stats.no_mappings += 1
        self._ledger(EntryState.NO_MAPPING, title)
        return True

    def log_ignored_anilist_id(self, al_id: int) -> bool:
        """Report an AniList ID skipped via the ignore list.

        Args:
            al_id (int): AniList ID
        """

        self._ledger(EntryState.IGNORED, f"AniList #{al_id}")
        return True

    def log_no_sd_entry(self, ctx: RunContext, al_id: int) -> bool:
        """Report an id with no SeaDex entry (bumps the tally, emits a titled row).

        Args:
            ctx (RunContext): The run's state (stats tally).
            al_id (int): Al ID
        """

        ctx.stats.no_seadex_entry += 1
        return self._log_titled_entry(EntryState.NO_ENTRY, al_id)

    def log_seadex_outage_skip(self, ctx: RunContext, al_id: int) -> bool:
        """Report a title whose SeaDex lookup was skipped (SeaDex unreachable).

        The outage skip is NOT a missing entry - the gateway warned once when
        SeaDex went unreachable - so the ledger says "skipped" with the reason on
        a detail line, and the tally lands in its own counter.

        Args:
            ctx (RunContext): The run's state (stats tally).
            al_id (int): Al ID
        """

        ctx.stats.seadex_unreachable += 1
        # Many outage-skipped ids were processed on a past run, so their name
        # sits in the cache row: prefer it over an AniList lookup, which in a
        # compound SeaDex+AniList outage would pay retry backoff per title.
        entry = self.cache_store.get_entry(ctx.arr, al_id)
        self._log_titled_entry(EntryState.SKIPPED, al_id, name=entry.name if entry is not None else None)
        # Scope-free (the titled row closed the entry): the reason rides col-0.
        self._detail("status", StyledValue("lookup skipped (SeaDex unreachable)", Accent.DIM))
        return True

    def _log_titled_entry(self, state: EntryState, al_id: int, *, name: str | None = None) -> bool:
        """A ledger row for an id with no SeaDex entry block to show.

        Renders the caller-supplied ``name`` when one is known (the outage path
        reads it off the cache row); otherwise resolves a human title live via
        AniList - through the gateway, so its retry log narrates any backoff.
        Either way the id rides its own "anilist" detail line when a title showed.
        """

        title = name if name is not None else self.anilist.title(al_id)
        self._ledger(state, title or f"AniList #{al_id}")
        # Only repeat the id on its own line when the ledger shows a title;
        # otherwise the ledger already reads "AniList #<id>" and a detail line
        # would just duplicate it. PLAIN accent -> style-less kv, as today.
        if title:
            self._detail("anilist", StyledValue(str(al_id)))
        return True

    # --- entry-block headers -------------------------------------------------

    def log_al_title(
        self,
        ctx: RunContext,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> bool:
        """Open the active-entry block: a focal "checking" header + coverage/URL.

        The entry being evaluated is the focal header of the title block, keyed on
        state CHECKING; its coverage/URL continuation and any details that follow
        ride the opened scope.

        Args:
            ctx (RunContext): The run's state (remembers the active title/url/coverage).
            anilist_title (str): Title for the AniList entry
            sd_entry: SeaDex entry
            coverage (str, optional): One-line coverage (e.g. "S04 E01-E12").
                Defaults to None / "" (e.g., a Radarr movie -> URL only)
        """

        # Remember title, URL, and coverage so add_torrent / the summary can
        # attribute and link what they grab, and show the same files we mapped
        # from the Arr even when a release's own file list can't be parsed.
        ctx.current_title = anilist_title
        ctx.current_url = sd_entry.url
        ctx.current_coverage = coverage

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
        return True

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        # Only the two dim-rendered cached states are admissible: the builder keys
        # row style on state, so a wider type would let a caller render a "cached"
        # row green/undimmed (the old always-grey50 invariant, now type-pinned).
        state: Literal[EntryState.UNCHANGED, EntryState.IN_RADARR] = EntryState.UNCHANGED,
    ) -> bool:
        """Emit a cached entry's self-contained block: a dim header plus its coverage/URL line.

        Cached entries have been unchanged since the last run, so they collapse to
        a dim header (state and title) and continuation lines carrying the stored
        season/episode coverage and, on its own line, the SeaDex URL. Everything is
        read from the cache record (written when the entry was first processed),
        with a name lookup only if the cache predates name storage.

        Args:
            ctx (RunContext): The run's state (stats tally).
            arr (Arr): Arr instance the entry is cached under
            al_id (int): AniList ID
            state (EntryState): Defaults to UNCHANGED (skipped because the SeaDex
                entry's update time matches the cache); pass IN_RADARR for entries
                already handled by a Radarr sync
        """

        ctx.stats.cached += 1

        # One row read serves the title, coverage, and URL below.
        entry = self.cache_store.get_entry(arr, al_id)
        title = entry.name if entry is not None else None
        if title is None:
            # Older cache without a stored name - fall back to a (gateway) lookup,
            # so its retry log narrates any backoff. None-gated: an empty stored
            # name must NOT trigger a lookup.
            title = self.anilist.title(al_id)
        if title is None:
            title = "(unknown title)"

        # A complete block (nothing follows it): self-close so a gap diagnostic
        # rides col 0, not indented under the finished row.
        self._block(
            EntryHeader(
                state,
                title,
                al_id=al_id,
                coverage=entry.coverage if entry is not None else None,
                url=entry.url if entry is not None else None,
            ),
        )
        return True

    # The carried-over pending states that get an inline entry header + a
    # scoreboard counter. MISSING / ERRORED are handled (drop / leave) by the
    # engine but have no ledger vocabulary, so they render nothing inline.
    _PENDING_ENTRY_STATES: dict[PendingState, EntryState] = {
        PendingState.QUEUED: EntryState.QUEUED,
        PendingState.IMPORTING: EntryState.IMPORTING,
        PendingState.IMPORTED: EntryState.IMPORTED,
    }

    def log_pending_snapshot(self, state: PendingState, pending: PendingImport) -> bool:
        """Emit a carried-over pending record's self-contained block inline in the series block.

        Emits the same titled header + coverage/link continuation as the other
        entry headers (so the carried-over record reads inside the series block and
        is self-attributed by its release title), for the three reportable states
        (``queued`` / ``importing`` / ``imported``). MISSING / ERRORED render
        nothing (no ledger vocabulary; the engine logs them at debug). This bumps
        NO counter - the engine owns the drop/count bookkeeping.

        Args:
            state (PendingState): The record's classified status this poll.
            pending (PendingImport): The carried-over record; its display label,
                coverage and SeaDex link attribute the row.
        """

        entry_state = self._PENDING_ENTRY_STATES.get(state)
        if entry_state is None:
            return False
        # The IMPORTED-green vs dim distinction is renderer policy keyed on state
        # (the builder does it) - the producer passes no style. A complete block
        # (the next record's reconcile may warn): self-close so that warning rides
        # col 0, not indented under this finished row.
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

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        """Report no suitable SeaDex releases (a status detail on the open entry).

        Args:
            ctx (RunContext): The run's state (stats tally).
        """

        ctx.stats.no_releases += 1
        self._detail("status", StyledValue("no suitable releases on SeaDex", Accent.DIM))
        return True

    def log_seadex_action(
        self,
        seadex_dict: SeadexDict,
        results: list[ReleaseOutcome],
        dry_run: bool = False,
        monitor_active: bool = False,
    ) -> bool:
        """Post the action block for a title that differs from SeaDex's pick.

        Called after the adding has run, so the status reflects what actually
        happened. Three outcomes: a fresh grab reads "adding" (a dry run reads
        "would add"); a recommended release already in the client from a PRIOR run -
        still downloading, not yet imported - reads "already downloading" (and,
        when the end-of-run monitor is active this session, "waiting to import");
        the genuine "you already own it" never reaches here. The block carries, in
        order: the status, then each recommended release group, then the
        per-release outcome (added / downloading).

        Args:
            seadex_dict (SeadexDict): SeaDex entries (used for the recommended groups)
            results (list): add_torrent's per-release outcomes (a preview run
                simulates its adds, so these are present on a dry run too)
            dry_run (bool): No torrent client, so nothing was really grabbed, but
                we'd have added everything. Defaults to False
            monitor_active (bool): The run will wait on / import pending torrents
                this session, so the "already downloading" line can promise the
                import. Defaults to False

        Returns:
            bool: True if a status block was posted; False if there was nothing to
                report (e.g., every release was skipped - the skip warning already
                explains that, so a status would only mislead)
        """

        # Nothing grabbed and nothing already present (e.g., all releases skipped
        # as private-only): leave the status to the inline "skipped" warning.
        if not results and not dry_run:
            return False

        # One pass over the outcomes: split added/downloading (a hashless/private
        # release has no name; the builder falls back to its group, so "" is fine).
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
            # Every result present-from-a-prior-run (none freshly added): the
            # torrent is in the client but still downloading / not yet imported.
            status = GrabStatus.ALREADY_DOWNLOADING

        # The recommended release group(s) - those flagged for download - with
        # their SeaDex tags (str-mapped + sorted: Tag is a StrEnum, so the order
        # is deterministic by value).
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

    def log_max_torrents_added(self, cap: int) -> bool:
        """Report hitting the per-run torrent cap (advanced.max_torrents_to_add).

        Args:
            cap (int): The configured cap that was reached.
        """

        # Close the entry first: the scan breaks here and _finalize_run's
        # reconcile runs before the summary, so a still-open ENTRY frontier would
        # misplace any reconcile diagnostics under the capped title.
        self._close_entry()
        self._emit(CapReached(cap=cap))
        return True

    # --- summary boundary ----------------------------------------------------

    def log_run_summary(self, ctx: RunContext, *, is_preview: bool, has_client: bool) -> bool:
        """Emit the end-of-run scoreboard (the summary boundary; closes the entry).

        The arr and the resolved wait mode are read off ``ctx``; the carried-over
        ``queued`` / ``importing`` / ``imported`` rows render only when the mode is
        not OFF (renderer policy off ``wait_mode_on``).

        Args:
            ctx (RunContext): The run's state (arr, wait mode, stats, totals, clock).
            is_preview (bool): The run grabbed nothing (dry run or no client).
            has_client (bool): A qBittorrent client is configured (distinguishes
                the dry-run note wording).
        """

        self._close_entry()

        # Warning/error counts come from the logger-level counter, diffed against
        # the snapshot taken when the run started.
        now_counts = log_counter(self.logger).snapshot()
        start_counts = ctx.log_counts_at_start

        def _delta(level: int) -> int:
            return now_counts.get(level, 0) - start_counts.get(level, 0)

        # A run grabs nothing when explicitly flagged dry, or when no client is
        # configured at all - the note wording distinguishes the two.
        dry_run_note = None
        if is_preview:
            dry_run_note = "nothing grabbed" if has_client else "qBittorrent not configured; nothing grabbed"

        tally = RunTally.from_stats(ctx.stats)
        elapsed_s = (time.monotonic() - ctx.started_monotonic) if ctx.started_monotonic is not None else None

        self._emit(
            RunSummaryReady(
                summary=RunSummary(
                    arr=ctx.arr,
                    dry_run=is_preview,
                    dry_run_note=dry_run_note,
                    added_count=ctx.torrents_added,
                    tally=tally,
                    wait_mode_on=ctx.import_wait_mode is not ImportWaitMode.OFF,
                    warnings=_delta(logging.WARNING),
                    # Today's n_errors sums ERROR and CRITICAL.
                    errors=_delta(logging.ERROR) + _delta(logging.CRITICAL),
                    elapsed_s=elapsed_s,
                    tip=_summary_tip(tally.needs_action),
                ),
            ),
        )
        return True
