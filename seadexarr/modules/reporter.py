import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import qbittorrentapi
from rich.text import Text
from seadex import EntryRecord

from .anilist_gateway import AniListGateway
from .cache import AbstractCacheStore
from .config import Arr
from .log import (
    EntryState,
    LogFormatter,
    arr_item_noun,
    count_noun,
    entry_string,
    format_elapsed,
    group_highlight,
    indent_string,
    log_counter,
    log_section_rule,
    log_styled,
    log_titled_rule,
)
from .manual_import import ImportWaitMode, PendingImport, PendingState
from .seadex_types import SeadexDict
from .torrents import AddOutcome, ReleaseOutcome

type SummaryRow = tuple[str, str | Text | None, str]
"""One labeled row in a summary per-entry block: ``(label, value, accent)``.

``value`` is the already-resolved field text (``None``/empty rows are dropped by
:func:`_summary_block` before rendering); ``accent`` is the row's rich style. It
admits a rich :class:`~rich.text.Text` as well as ``str``: the "torrent" row of
an "added" block carries the ``group_highlight`` Text whose inline group span the
console keeps (the file log stringifies it). The ``log_fmt.kv`` consumer accepts
exactly ``str | Text``.
"""


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


class RunReporter:
    """Owns the ``log_*`` presentation surface and the end-of-run summary.

    Built once per arr instance with the stable collaborators it needs (logger,
    log formatter, cache store, AniList gateway). The methods that touch run
    state take the :class:`RunContext` as their first argument; the rest are
    pure presentation.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        log_fmt: LogFormatter,
        cache_store: AbstractCacheStore,
        anilist: AniListGateway,
    ) -> None:
        self.logger = logger
        self.log_fmt = log_fmt
        self.cache_store = cache_store
        self.anilist = anilist

    def log_run_summary(
        self,
        ctx: RunContext,
        *,
        is_preview: bool,
        has_client: bool,
    ) -> bool:
        """Log the end-of-run scoreboard for an Arr run

        The arr and the resolved wait mode are read off ``ctx``; the carried-over
        ``queued`` / ``importing`` / ``imported`` rows render only when the mode
        is not OFF (so they never clutter a run with the feature off).

        Args:
            ctx (RunContext): The run's state (arr, wait mode, stats, totals, clock).
            is_preview (bool): The run grabbed nothing (dry run or no client).
            has_client (bool): A qBittorrent client is configured (distinguishes
                the dry-run note wording).
        """

        arr = ctx.arr
        stats = ctx.stats

        # Warning/error counts come from the logger-level counter, diffed
        # against the snapshot taken when the run started
        now_counts = log_counter(self.logger).snapshot()
        start_counts = ctx.log_counts_at_start

        def _delta(level: int) -> int:
            return now_counts.get(level, 0) - start_counts.get(level, 0)

        n_warnings = _delta(logging.WARNING)
        n_errors = _delta(logging.ERROR) + _delta(logging.CRITICAL)

        title = f"SeaDexArr ({arr.capitalize()}) run complete"
        # State dry-run once, here, scoping the whole summary - rather than also
        # tagging the "added" value (the same fact twice in one block). The file
        # log keeps the plain title; the annotation rides the console title only.
        rule_title = title
        # A run grabs nothing when explicitly flagged dry, or when no client is
        # configured at all - annotate (and later dim) the summary either way.
        is_dry_run = is_preview
        if is_dry_run:
            note = "nothing grabbed" if has_client else ("qBittorrent not configured; nothing grabbed")
            rule_title += f"   (DRY RUN — {note})"
        # A blank before the rule separates the last item from the summary.
        self.log_fmt.blank()
        log_titled_rule(self.logger, rule_title, heavy=True, message=title)
        # A blank under the title gives the scoreboard a gap below the header.
        self.log_fmt.blank()

        # The summary's key column is narrower than the per-title detail column:
        # "needs action" (12) is the widest key here, vs. "missing episodes" (16)
        # in entry details. A heavy rule separates the two blocks, so the differing
        # colon columns never sit adjacent. Wrap the formatter to fix width at 12.
        def summary_kv(key: str, value: str, *, value_style: str | None = None) -> None:
            self.log_fmt.kv(key, value, key_width=12, value_style=value_style)

        # A needs-action entry in the summary, rendered with the same labeled
        # gutter as added_detail so the two blocks read alike: the title hangs at
        # indent 2, then fixed fields sit at indent 3 beneath it. Unlike a grab
        # there's no torrent name to lean on, so the skipped private release
        # group IS named here. The whole block is yellow - it's the one section
        # asking the user to do something. The title is shown in full; it sits on
        # its own line above the fixed fields, so its length can't break the column.
        def _summary_block(
            title: str,
            title_style: str | None,
            rows: list[SummaryRow],
        ) -> None:
            # Shared layout for the summary's per-entry blocks: the title hangs
            # at indent 2, then labeled gutter fields sit beneath it at indent 3,
            # their values landing in the same column as the live "checking"
            # block. Each row carries its already-resolved accent.
            log_styled(self.logger, indent_string(title, level=2), title_style)
            for label, value, accent in rows:
                if not value:
                    continue
                self.log_fmt.kv(
                    label,
                    value,
                    value_style=accent,
                    indent=3,
                    key_width=7,
                    sep="",
                )

        def needs_detail(item: NeedsActionRecord) -> None:
            rows: list[SummaryRow] = [
                ("files", item.coverage, "grey50"),
                ("group", item.group, "yellow"),
                ("reason", item.reason, "yellow"),
                ("link", item.url, "grey50"),
            ]
            _summary_block(item.title or "(unknown title)", "yellow", rows)

        # A grab in the summary, rendered like the live per-entry "checking"
        # block: the title hangs at indent 2, then labeled gutter fields sit
        # beneath it at indent 3, their values landing in the same column (14) as
        # the live block. The grab is labeled "torrent" rather than "added" since
        # the whole section is already the added list. The recommended group is
        # called out at the front of the torrent name - highlighted in place when
        # the name already leads with it, or prepended in brackets otherwise - so
        # the group always reads first. A dry run dims the whole block (group accent
        # included) so the would-be grabs don't read as real. The title is shown
        # in full on its own line, so its length can't break the column.
        def added_detail(item: GrabRecord) -> None:
            torrent_value = group_highlight(
                item.name,
                item.group,
                group_style="grey50" if is_dry_run else "cyan",
                base_style="grey50" if is_dry_run else "green",
            )
            # A dry run dims the torrent value too (matching the dimmed title line
            # and the already-dim files/link) so the would-be grabs don't read as
            # real; files and link are dim either way.
            rows: list[SummaryRow] = [
                ("files", item.coverage, "grey50"),
                ("link", item.url, "grey50"),
                ("torrent", torrent_value, "grey50" if is_dry_run else "green"),
            ]
            _summary_block(
                item.title or "(unknown title)",
                "grey50" if is_dry_run else None,
                rows,
            )

        summary_kv("checked", str(stats.checked))

        # Needs-action sits ahead of "added" so anything still waiting on the
        # user surfaces first, before the (often longer) list of completed grabs.
        needs = stats.needs_action
        summary_kv(
            "needs action",
            str(len(needs)),
            value_style="yellow" if needs else None,
        )
        for item in needs:
            needs_detail(item)

        # The count is the authoritative torrents_added (covers the no-client
        # dry-run path too); the list is the per-grab detail from add_torrent.
        summary_kv(
            "added",
            str(ctx.torrents_added),
            value_style="green" if ctx.torrents_added else None,
        )
        for item in stats.added:
            added_detail(item)

        # Carried-over pending-import statuses (NEVER this-run grabs - those are the
        # `added` block above). Each row renders only when the feature is on AND the
        # value is non-zero, so a feature-off run is unchanged and there's never an
        # `added`+`queued` double line for a single torrent.
        if ctx.import_wait_mode is not ImportWaitMode.OFF:
            if stats.queued:
                summary_kv("queued", str(stats.queued), value_style="grey50")
            if stats.importing:
                summary_kv("importing", str(stats.importing), value_style="yellow")
            if stats.imported:
                summary_kv("imported", str(stats.imported), value_style="green")

        summary_kv("up to date", str(stats.up_to_date))
        summary_kv(
            "unchanged",
            f"{stats.cached}  (since last run)" if stats.cached else "0",
            value_style="grey50",
        )
        if stats.no_mappings:
            summary_kv("no mapping", str(stats.no_mappings))
        # Keep "no entry" (no SeaDex entry at all) separate from "no release"
        # (an entry exists but nothing suitable to grab) so they don't conflate
        if stats.no_seadex_entry:
            summary_kv("no entry", str(stats.no_seadex_entry))
        # Outage skips are neither: SeaDex was unreachable, so these titles
        # weren't looked up at all (they're re-checked next run).
        if stats.seadex_unreachable:
            summary_kv("seadex down", str(stats.seadex_unreachable), value_style="yellow")
        if stats.no_releases:
            summary_kv("no release", str(stats.no_releases))

        if stats.unmonitored:
            summary_kv("unmonitored", str(stats.unmonitored))

        summary_kv(
            "issues",
            f"{count_noun(n_warnings, 'warning')}, {count_noun(n_errors, 'error')}",
            value_style="bold red" if n_errors else ("yellow" if n_warnings else None),
        )
        if ctx.started_monotonic is not None:
            elapsed = format_elapsed(time.monotonic() - ctx.started_monotonic)
            summary_kv("elapsed", elapsed)

        # A single guidance line if anything was skipped purely for being
        # private-only, rather than repeating it per-entry during the run. Kept
        # at indent 1, so it reads as part of the summary block, not detached.
        # PRIVATE_ONLY can't co-occur with the fallback-mode kinds
        # (private_releases is run-wide); those two can, and the no-fallback tip
        # (which omits the fallback suggestion, already on) wins over the stale one.
        if any(item.kind is NeedsActionKind.PRIVATE_ONLY for item in needs):
            tip = "Tip: manually grab private releases or set private_releases: fallback to automatically grab public alternatives."
        elif any(item.kind is NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK for item in needs):
            tip = "Tip: no public alternative exists yet; the title is re-checked every run until one appears."
        elif any(item.kind is NeedsActionKind.PRIVATE_ONLY_STALE for item in needs):
            tip = (
                "Tip: your copies of these releases are outdated (their file sizes no longer match); "
                "update them from their private tracker, or delete the outdated files to let the "
                "public fallback stand in."
            )
        else:
            tip = None
        if tip is not None:
            log_styled(self.logger, indent_string(tip, level=1), "grey50")

        log_section_rule(self.logger, "=", width=self.log_fmt.line_length)

        return True

    # The carried-over pending states that get an inline ledger row + a scoreboard
    # counter. MISSING / ERRORED are handled (drop / leave) by the engine but have
    # no ledger vocabulary, so they render nothing inline.
    _PENDING_ENTRY_STATES: dict[PendingState, EntryState] = {
        PendingState.QUEUED: EntryState.QUEUED,
        PendingState.IMPORTING: EntryState.IMPORTING,
        PendingState.IMPORTED: EntryState.IMPORTED,
    }

    def log_pending_snapshot(
        self,
        state: PendingState,
        pending: PendingImport,
    ) -> bool:
        """Render a carried-over pending record's status inline in the series block.

        Emits the same titled ledger row + coverage/link continuation as the other
        entry rows (so the carried-over record reads inside the series block and is
        self-attributed by its release title), for the three reportable states
        (``queued`` / ``importing`` / ``imported``). MISSING / ERRORED render
        nothing (no ledger vocabulary; the engine logs them at debug). This bumps
        NO counter - the engine owns the drop/count bookkeeping - so a record is
        never double-counted.

        Args:
            state (PendingState): The record's classified status this poll.
            pending (PendingImport): The carried-over record; its display label,
                coverage and SeaDex link attribute the row.
        """

        entry_state = self._PENDING_ENTRY_STATES.get(state)
        if entry_state is None:
            return False
        style = "green" if entry_state is EntryState.IMPORTED else "grey50"
        self.log_entry_status(entry_state, pending.display_label, style=style)
        self.log_entry_coverage(pending.coverage, pending.url)
        return True

    def log_arr_start(
        self,
        arr: Arr,
        n_items: int,
    ) -> bool:
        """Produce a log message for the start of the run

        Args:
            arr: Type of arr instance
            n_items: Total number of shows/movies
        """

        # A blank before the rule separates the boot block from the run banner;
        # the gap UNDER this title is supplied by the first log_arr_item_start.
        self.log_fmt.blank()
        banner = f"Starting SeaDexArr ({arr.capitalize()}) for {arr_item_noun(arr, n_items)}"
        log_titled_rule(self.logger, banner, heavy=True)

        return True

    def log_entry_status(
        self,
        state: EntryState,
        label: str,
        style: str | None = "grey50",
    ) -> bool:
        """Log a one-line entry status as a fixed-column ledger row

        Renders "<state> <label>" at indent level 1, with state padded to a fixed
        width so the label lines up across rows (see entry_string). The state word
        carries the meaning, so there is no trailing note; season/episode coverage
        and the SeaDex URL ride a separate continuation line (log_entry_coverage).
        The indent is baked into the message, so the file log keeps it too.

        Args:
            state (EntryState): Which entry-level outcome this row reports
            label (str): What the state applies to (usually a title)
            style (str): Console style for the line. Defaults to "grey50" (dim);
                pass None for an emphasized line such as the active "checking" one
        """

        # A blank line before each ledger row separates entries within a title
        # block (and the first entry from its header)
        self.log_fmt.blank()
        log_styled(self.logger, indent_string(entry_string(state, label), level=1), style)

        return True

    def log_entry_coverage(
        self,
        coverage: str | None,
        url: str | None,
        style: str | None = "grey50",
        incomplete: bool = False,
    ) -> bool:
        """Log the season/episode coverage and SeaDex URL beneath an entry

        Two dim detail lines whose values sit directly beneath the entry's title
        (so they line up with each other and with the title): the season/episode
        coverage labeled "files", then the full SeaDex URL labeled "link".
        Either part may be absent - a Radarr movie has no episode coverage (link
        only) - and nothing is logged when both are absent. An incomplete SeaDex
        entry is flagged as an emphasized tail on the last line shown.

        Example:

            files S04 E01-E12
            link https://releases.moe/111852

        Args:
            coverage (str): One-line coverage, e.g. "S04 E01-E12" (maybe "")
            url (str): Full SeaDex URL (maybe None/"")
            style (str): Console style. Defaults to "grey50" (dim)
            incomplete (bool): Flag the SeaDex entry as incomplete. Defaults False
        """

        rows = [(label, value) for label, value in (("files", coverage), ("link", url)) if value]
        if not rows:
            return False

        for idx, (label, value) in enumerate(rows):
            # The incomplete flag rides the last line so it reads once, next to
            # the URL when there is one
            tail = "(marked incomplete on SeaDex)" if incomplete and idx == len(rows) - 1 else None
            self.log_fmt.detail(label, value, value_style=style, tail=tail)

        return True

    def log_arr_item_unmonitored(
        self,
        ctx: RunContext,
        item_title: str,
    ) -> bool:
        """Produce a log message if skipping because the item is unmonitored

        Args:
            ctx (RunContext): The run's state (stats tally).
            item_title (str): Item title
        """

        ctx.stats.unmonitored += 1
        return self.log_entry_status(
            EntryState.UNMONITORED,
            item_title,
        )

    def log_arr_item_start(
        self,
        arr: Arr,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> bool:
        """Produce a log message for the start of Arr item

        Args:
            arr: Type of arr instance
            item_title: Title for the item
            n_item: Number for the show/movie
            n_items: Total number of shows/movies
        """

        # A blank line before the separator rule sets each item's block apart
        # from the previous one (and from the run banner for the first item)
        self.log_fmt.blank()
        header = f"[{n_item}/{n_items}] {arr.capitalize()}: {item_title}"
        log_titled_rule(self.logger, header)

        return True

    def log_no_anilist_mappings(
        self,
        ctx: RunContext,
        title: str,
    ) -> bool:
        """Produce a log message for the case where no AniList mappings are found

        Args:
            ctx (RunContext): The run's state (stats tally).
            title: Title for the item
        """

        ctx.stats.no_mappings += 1
        return self.log_entry_status(
            EntryState.NO_MAPPING,
            title,
        )

    def log_ignored_anilist_id(
        self,
        al_id: int,
    ) -> bool:
        """Produce a log message when an AniList ID is skipped via the ignore list

        Args:
            al_id (int): AniList ID
        """

        return self.log_entry_status(
            EntryState.IGNORED,
            f"AniList #{al_id}",
        )

    def log_no_sd_entry(
        self,
        ctx: RunContext,
        al_id: int,
    ) -> bool:
        """Produce a log message if no SeaDex entry is found

        Args:
            ctx (RunContext): The run's state (stats tally).
            al_id (int): Al ID
        """

        ctx.stats.no_seadex_entry += 1
        return self._log_titled_entry(EntryState.NO_ENTRY, al_id)

    def log_seadex_outage_skip(
        self,
        ctx: RunContext,
        al_id: int,
    ) -> bool:
        """Log a title whose SeaDex lookup was skipped (SeaDex unreachable)

        The outage skip is NOT a missing entry - the gateway warned once when
        SeaDex went unreachable - so the ledger says "skipped" with the reason
        on a detail line, and the tally lands in its own counter.

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
        self.log_fmt.detail(
            "status",
            "lookup skipped (SeaDex unreachable)",
            value_style="grey50",
        )
        return True

    def _log_titled_entry(self, state: EntryState, al_id: int, *, name: str | None = None) -> bool:
        """A ledger row for an id with no SeaDex entry to show.

        Renders the caller-supplied ``name`` when one is known (the outage path
        reads it off the cache row); otherwise resolves a human title live via
        AniList - through the gateway, so its retry log narrates any backoff.
        Either way the id rides its own "anilist" detail line.
        """

        anilist_title = name if name is not None else self.anilist.title(al_id)
        self.log_entry_status(state, anilist_title or f"AniList #{al_id}")
        # Only repeat the id on its own line when the ledger shows a title;
        # otherwise the ledger already reads "AniList #<id>" and a detail line
        # would just duplicate it
        if anilist_title:
            self.log_fmt.detail("anilist", str(al_id))

        return True

    def log_al_title(
        self,
        ctx: RunContext,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> bool:
        """Log the active-entry header: a "checking" row and its coverage/URL line

        The entry being evaluated is the focal line of the title block, so it sits
        on the ledger (state "checking") undimmed. The dim continuation lines below
        carry the season/episode coverage and, on its own line, the full SeaDex
        URL, so you can see what it covers and where to find it; an incomplete
        SeaDex entry is flagged as an emphasized tail on the last of those lines.

        Args:
            ctx (RunContext): The run's state (remembers the active title/url/coverage).
            anilist_title (str): Title for the AniList entry
            sd_entry: SeaDex entry
            coverage (str, optional): One-line coverage (e.g. "S04 E01-E12").
                Defaults to None / "" (e.g., a Radarr movie -> URL only)
        """

        # Remember title, URL, and coverage so add_torrent / the summary can
        # attribute and link what they grab, and show the same files we mapped
        # from the Arr even when a release's own file list can't be parsed
        ctx.current_title = anilist_title
        ctx.current_url = sd_entry.url
        ctx.current_coverage = coverage

        # The active entry, on the ledger but undimmed (style=None) so it reads
        # as the focal line, not a no-op like the gray unchanged rows
        self.log_entry_status(EntryState.CHECKING, anilist_title, style=None)
        self.log_entry_coverage(
            coverage,
            sd_entry.url,
            incomplete=sd_entry.is_incomplete,
        )

        return True

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        state: EntryState = EntryState.UNCHANGED,
    ) -> bool:
        """Log a cached entry as a ledger row plus its coverage/URL line

        Cached entries have been unchanged since the last run, so they collapse to a dim
        ledger row (state and title) and continuation lines carrying the stored
        season/episode coverage and, on its own line, the SeaDex URL. Everything
        is read from the cache
        record (written when the entry was first processed), with a name lookup
        only if the cache predates name storage.

        Args:
            ctx (RunContext): The run's state (stats tally).
            arr (Arr): Arr instance the entry is cached under
            al_id (int): AniList ID
            state (EntryState): Defaults to UNCHANGED (skipped because the SeaDex
                entry's update time matches the cache); pass IN_RADARR for entries
                already handled by a Radarr sync
        """

        ctx.stats.cached += 1

        # One row read serves the title, coverage, and URL lines below (was a
        # SELECT name + SELECT coverage + SELECT url against the same row).
        entry = self.cache_store.get_entry(arr, al_id)
        anilist_title = entry.name if entry is not None else None
        if anilist_title is None:
            # Older cache without a stored name - fall back to a (gateway)
            # lookup, so its retry log narrates any backoff.
            anilist_title = self.anilist.title(al_id)
        if anilist_title is None:
            anilist_title = "(unknown title)"

        self.log_entry_status(state, anilist_title)
        self.log_entry_coverage(
            entry.coverage if entry is not None else None,
            entry.url if entry is not None else None,
        )

        return True

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        """Log if no suitable SeaDex releases are found

        Args:
            ctx (RunContext): The run's state (stats tally).
        """

        ctx.stats.no_releases += 1
        self.log_fmt.detail(
            "status",
            "no suitable releases on SeaDex",
            value_style="grey50",
        )

        return True

    def log_seadex_action(
        self,
        seadex_dict: SeadexDict,
        results: list[ReleaseOutcome],
        dry_run: bool = False,
        monitor_active: bool = False,
    ) -> bool:
        """Log the action block for a title that differs from SeaDex's pick

        Called after the adding has run, so the status reflects what actually
        happened rather than what we set out to do. Three outcomes: a fresh grab
        reads "adding" (a dry run reads "would add"); a recommended release
        already in the client from a PRIOR run - still downloading, not yet
        imported - reads "already downloading" (and, when the end-of-run monitor
        is active this session, "waiting to import"); the genuine "you already
        own it" never reaches here (that's the any_to_download=False path). The
        block is, in order: the status line, then each recommended release group,
        then the per-release outcome (added / downloading).

        Args:
            seadex_dict (SeadexDict): SeaDex entries (used for the recommended groups)
            results (list): add_torrent's per-release outcomes (a preview run
                simulates its adds, so these are present on a dry run too)
            dry_run (bool): No torrent client, so nothing was really grabbed, but
                we'd have added everything. Defaults to False
            monitor_active (bool): The run will wait on / import pending torrents
                this session (import_wait_mode != OFF, non-preview), so the
                "already downloading" line can promise the import. Defaults to False

        Returns:
            bool: True if a status block was logged; False if there was nothing
                to report (e.g., every release was skipped - the skip warning
                already explains that, so a status would only mislead)
        """

        added = dry_run or any(r.added for r in results)
        # Every result present-from-a-prior-run (none freshly added): the torrent
        # is in the client but still downloading / not yet imported.
        already_downloading = bool(results) and not added

        # Nothing grabbed and nothing already present (e.g., all releases skipped
        # as private-only): leave the status to the inline "skipped" warning
        if not results and not dry_run:
            return False

        if added:
            self.log_fmt.detail(
                "status",
                "would add SeaDex's recommended release (dry run)"
                if dry_run
                else "adding SeaDex's recommended release",
            )
        elif already_downloading:
            message = "SeaDex's pick is already downloading in qBittorrent"
            if monitor_active:
                message += " - waiting to import"
            self.log_fmt.detail("status", message, value_style="yellow")

        # The release group(s) we recommend (those flagged for download), tags too
        for srg, srg_item in seadex_dict.items():
            urls = srg_item.urls
            if any(u.download for u in urls.values()):
                tags = srg_item.tags
                if len(tags) > 0:
                    recommendation = f"{srg} [{', '.join(tags)}]"
                else:
                    recommendation = srg
                self.log_fmt.detail("group", recommendation, value_style="cyan")

        # Per-release outcome (qBittorrent path; a dry run has no names to show).
        # A hashless/private torrent has no name, so fall back to the release group
        # rather than rendering the literal "None".
        for r in results:
            if r.added:
                self.log_fmt.detail(
                    "would add" if dry_run else "added",
                    r.name or r.group,
                    value_style="green",
                )
            elif r.outcome is AddOutcome.ALREADY_ADDED:
                self.log_fmt.detail("downloading", r.name or r.group, value_style="yellow")

        return True

    def log_max_torrents_added(self) -> bool:
        """Produce a log message about hitting the maximum number of torrents added"""

        log_styled(
            self.logger,
            "Reached the maximum number of torrents for this run (advanced.max_torrents_to_add); stopping",
            "yellow",
        )

        return True
