"""Per-run state (``RunContext``) and presentation (``RunReporter``).

These two collaborators are the linchpin of the orchestrator's decoupling
(see ``REFACTOR_PLAN.md`` §6). ``RunContext`` is the per-run state bag that used
to live as scattered mutable ``self.*`` fields — the stats tally, the running
torrent count, the title/url/coverage currently being processed, the run clock,
and the logger-counter snapshot. ``RunReporter`` owns every ``log_*`` method and
the end-of-run summary; it reads and writes the context rather than reaching into
the orchestrator.

A fresh ``RunContext`` is built at the top of each run; the reporter is built once
and takes the context as an argument on the methods that need run state, so it
stays valid across runs. The orchestrator keeps thin ``log_*`` delegators (which
inject the current context) so the Sonarr/Radarr adapters call the same surface
as before.

Extracted from ``SeaDexArr`` in Phase 4b of the refactor; behaviour-preserving.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from seadex import EntryRecord

from .anilist import get_anilist_title
from .anilist_gateway import AniListGateway
from .cache import CacheStore
from .config import Arr
from .log import (
    LogFormatter,
    count_noun,
    entry_string,
    group_highlight,
    indent_string,
    rule_string,
)


def fresh_stats() -> dict:
    """Build an empty per-run stats tally for the end-of-run summary."""

    return {
        "checked": 0,
        "added": [],  # list of {"title", "coverage", "url", "name", "group"}
        "up_to_date": 0,
        "cached": 0,
        "no_seadex_entry": 0,
        "no_releases": 0,
        "no_mappings": 0,
        "needs_action": [],  # list of {"title", "reason"}
        "unmonitored": 0,
    }


@dataclass
class RunContext:
    """Per-run state, created fresh at the top of each run.

    Replaces the run-scoped mutable ``self.*`` fields of the old god class so the
    decision engine, the torrent service, and the reporter can read and return
    data instead of mutating shared orchestrator state.
    """

    arr: Arr
    dry_run: bool = False
    stats: dict = field(default_factory=fresh_stats)
    torrents_added: int = 0
    # Title, SeaDex URL, and coverage of the entry currently being processed, so
    # grabs and the summary can attribute and link what they grab.
    current_title: str | None = None
    current_url: str | None = None
    current_coverage: str | None = None
    # Set per-title when public_only forces a skip of a private-only release, so
    # the caller knows not to cache the title as done; the group names ride along
    # for the run summary's "needs action" list.
    public_only_skipped: bool = False
    public_only_groups: list[str] = field(default_factory=list)
    # Run clock (monotonic, so an NTP/DST step can't yield negative elapsed) and
    # the logger-counter snapshot taken at the start, diffed for the summary.
    started_monotonic: float | None = None
    log_counts_at_start: dict[int, int] = field(default_factory=dict)


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
        cache_store: CacheStore,
        anilist: AniListGateway,
    ) -> None:
        self.logger = logger
        self.log_fmt = log_fmt
        self.cache_store = cache_store
        self.anilist = anilist

    def log_run_summary(
        self,
        ctx: RunContext,
        arr: Arr,
        *,
        is_preview: bool,
        has_client: bool,
    ) -> bool:
        """Log the end-of-run scoreboard for an Arr run

        Args:
            ctx (RunContext): The run's state (stats, totals, clock).
            arr (Arr): Type of arr instance
            is_preview (bool): The run grabbed nothing (dry run or no client).
            has_client (bool): A qBittorrent client is configured (distinguishes
                the dry-run note wording).
        """

        stats = ctx.stats

        # Warning/error counts come from the logger-level counter, diffed
        # against the snapshot taken when the run started
        counter = getattr(self.logger, "seadex_counter", None)
        now_counts = counter.snapshot() if counter else {}
        start_counts = ctx.log_counts_at_start

        def _delta(level: int) -> int:
            return now_counts.get(level, 0) - start_counts.get(level, 0)

        n_warnings = _delta(logging.WARNING)
        n_errors = _delta(logging.ERROR) + _delta(logging.CRITICAL)

        title = f"SeaDexArr ({arr.capitalize()}) run complete"
        # State dry-run once, here, scoping the whole summary - rather than also
        # tagging the "added" value (the same fact twice in one block). The file
        # log keeps the plain title; the annotation rides the console rule_title.
        rule_title = title
        # A run grabs nothing when explicitly flagged dry, or when no client is
        # configured at all - annotate (and later dim) the summary either way.
        is_dry_run = is_preview
        if is_dry_run:
            note = "nothing grabbed" if has_client else (
                "no client; nothing grabbed"
            )
            rule_title += f"   (DRY RUN — {note})"
        self.logger.info(
            title,
            extra={
                "rule_title": rule_title,
                "rule_style": "bold cyan",
                "rule_heavy": True,
            },
        )

        # The summary's key column is narrower than the per-title detail column:
        # "needs action" (12) is the widest key here, vs. "missing episodes" (16)
        # in entry details. A heavy rule separates the two blocks, so the differing
        # colon columns never sit adjacent. Wrap the formatter to fix width at 12.
        def summary_kv(key: str, value: Any, **kwargs: Any) -> bool:
            return self.log_fmt.kv(key, value, key_width=12, **kwargs)

        # A needs-action entry in the summary, rendered with the same labeled
        # gutter as added_detail so the two blocks read alike: the title hangs at
        # indent 2, then fixed fields sit at indent 3 beneath it. Unlike a grab
        # there's no torrent name to lean on, so the skipped private release
        # group IS named here. The whole block is yellow - it's the one section
        # asking the user to do something. The title is shown in full; it sits on
        # its own line above the fixed fields, so its length can't break the column.
        def _summary_block(title: str, title_style: str | None, rows: list) -> None:
            # Shared layout for the summary's per-entry blocks: the title hangs
            # at indent 2, then labeled gutter fields sit beneath it at indent 3,
            # their values landing in the same column as the live "checking"
            # block. Each row carries its already-resolved accent.
            self.logger.info(
                indent_string(title, level=2),
                extra={"line_style": title_style},
            )
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

        def needs_detail(item: dict) -> None:
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("group", item.get("group"), "yellow"),
                ("reason", item.get("reason"), "yellow"),
                ("link", item.get("url"), "grey50"),
            ]
            _summary_block(item.get("title") or "(unknown title)", "yellow", rows)

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
        def added_detail(item: dict) -> None:
            torrent_value = group_highlight(
                item.get("name"),
                item.get("group"),
                group_style="grey50" if is_dry_run else "cyan",
                base_style="grey50" if is_dry_run else "green",
            )
            # A dry run dims the torrent value too (matching the dimmed title line
            # and the already-dim files/link) so the would-be grabs don't read as
            # real; files and link are dim either way.
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("link", item.get("url"), "grey50"),
                ("torrent", torrent_value, "grey50" if is_dry_run else "green"),
            ]
            _summary_block(
                item.get("title") or "(unknown title)",
                "grey50" if is_dry_run else None,
                rows,
            )

        summary_kv("checked", str(stats["checked"]))

        # Needs-action sits ahead of "added" so anything still waiting on the
        # user surfaces first, before the (often longer) list of completed grabs.
        needs = stats["needs_action"]
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
        for item in stats["added"]:
            added_detail(item)

        summary_kv("up to date", str(stats["up_to_date"]))
        summary_kv(
            "unchanged",
            f"{stats['cached']}  (since last run)"
            if stats["cached"]
            else "0",
            value_style="grey50",
        )
        if stats["no_mappings"]:
            summary_kv("no mapping", str(stats["no_mappings"]))
        # Keep "no entry" (no SeaDex entry at all) separate from "no release"
        # (an entry exists but nothing suitable to grab) so they don't conflate
        if stats["no_seadex_entry"]:
            summary_kv("no entry", str(stats["no_seadex_entry"]))
        summary_kv("no release", str(stats["no_releases"]))

        if stats["unmonitored"]:
            summary_kv("unmonitored", str(stats["unmonitored"]))

        summary_kv(
            "issues",
            f"{count_noun(n_warnings, 'warning')}, {count_noun(n_errors, 'error')}",
            value_style="bold red"
            if n_errors
            else ("yellow" if n_warnings else None),
        )
        if ctx.started_monotonic is not None:
            elapsed = self.log_fmt.format_elapsed(
                time.monotonic() - ctx.started_monotonic,
            )
            summary_kv("elapsed", elapsed)

        # A single guidance line if anything was skipped purely for being
        # private-only, rather than repeating it per-entry during the run. Kept
        # at indent 1, so it reads as part of the summary block, not detached.
        public_only_skipped = any(
            "public_only" in (item.get("reason") or "") for item in needs
        )
        if public_only_skipped:
            self.logger.info(
                indent_string(
                    "Tip: set public_only: false to allow private trackers, or "
                    "wait for a public release.",
                    level=1,
                ),
                extra={"line_style": "grey50"},
            )

        self.logger.info(
            rule_string(rule_char="=", total_length=self.log_fmt.line_length),
            extra={"rule_char": "="},
        )

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

        item_label = {
            Arr.RADARR: count_noun(n_items, "movie"),
            Arr.SONARR: count_noun(n_items, "series", "series"),
        }[arr]

        banner = f"Starting SeaDexArr ({arr.capitalize()}) for {item_label}"
        self.logger.info(
            banner,
            extra={
                "rule_title": banner,
                "rule_style": "bold cyan",
                "rule_heavy": True,
            },
        )

        return True

    def log_entry_status(
        self,
        state: str,
        label: str,
        style: str | None = "grey50",
    ) -> bool:
        """Log a one-line entry status as a fixed-column ledger row

        Renders "<state> <label>" at indent level 1, with state padded to a fixed
        width so the label lines up across rows (see entry_string). Used for the
        entry-level outcomes: unchanged, in radarr, checking, unmonitored,
        skipped, no mapping, ignored, and no entry. The state word carries the
        meaning, so there is no trailing note; season/episode coverage and the
        SeaDex URL ride a separate continuation line (log_entry_coverage). The
        indent is baked into the message, so the file log keeps it too.

        Args:
            state (str): Short state word, e.g. "unchanged" or "no entry"
            label (str): What the state applies to (usually a title)
            style (str): Console style for the line. Defaults to "grey50" (dim);
                pass None for an emphasized line such as the active "checking" one
        """

        # A blank line before each ledger row separates entries within a title
        # block (and the first entry from its header)
        self.log_fmt.blank()
        self.logger.info(
            indent_string(entry_string(state, label), level=1),
            extra={"line_style": style},
        )

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

        rows = [
            row for row in (("files", coverage), ("link", url)) if row[1]
        ]
        if not rows:
            return False

        for idx, (label, value) in enumerate(rows):
            # The incomplete flag rides the last line so it reads once, next to
            # the URL when there is one
            tail = (
                "(marked incomplete on SeaDex)"
                if incomplete and idx == len(rows) - 1
                else None
            )
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

        ctx.stats["unmonitored"] += 1
        return self.log_entry_status(
            "unmonitored",
            item_title,
        )

    # Both Ares reach the same "unmonitored" outcome, so this is just an alias
    log_anilist_item_unmonitored = log_arr_item_unmonitored

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
        self.logger.info(
            header,
            extra={"rule_title": header, "rule_style": "bold cyan"},
        )

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

        ctx.stats["no_mappings"] += 1
        return self.log_entry_status(
            "no mapping",
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
            "ignored",
            f"AniList #{al_id}",
        )

    def log_no_anilist_id(self) -> bool:
        """Produce a log message for the case where no AniList ID is found"""

        self.logger.debug(
            indent_string("-> No AL ID found. Continuing"),
        )
        self.logger.debug(
            rule_string(
                total_length=self.log_fmt.line_length,
            ),
        )

        return True

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

        ctx.stats["no_seadex_entry"] += 1

        # Resolve a human title so the line is meaningful. There's no SeaDex
        # entry and the id isn't cached (we only cache processed ids), so this
        # is a live AniList lookup; the id rides its own "anilist" detail line.
        anilist_title, self.anilist.al_cache = get_anilist_title(
            al_id,
            al_cache=self.anilist.al_cache,
        )
        self.log_entry_status(
            "no entry",
            anilist_title or f"AniList #{al_id}",
        )
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
        self.log_entry_status("checking", anilist_title, style=None)
        self.log_entry_coverage(
            coverage, sd_entry.url, incomplete=sd_entry.is_incomplete,
        )

        return True

    def log_cached_entry(
        self,
        ctx: RunContext,
        arr: Arr,
        al_id: int,
        state: str = "unchanged",
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
            state (str): State word. Defaults to "unchanged" (skipped because the
                SeaDex entry's update time matches the cache); pass "in radarr"
                for entries already handled by a Radarr sync
        """

        ctx.stats["cached"] += 1

        anilist_title = self.cache_store.get_cached_name(arr=arr, al_id=al_id)
        if anilist_title is None:
            # Older cache without a stored name - fall back to a lookup
            anilist_title, self.anilist.al_cache = get_anilist_title(
                al_id,
                al_cache=self.anilist.al_cache,
            )
        if anilist_title is None:
            anilist_title = "(unknown title)"

        self.log_entry_status(state, anilist_title)
        self.log_entry_coverage(
            self.cache_store.get_cached_field(arr, al_id, "coverage"),
            self.cache_store.get_cached_field(arr, al_id, "url"),
        )

        return True

    def log_no_seadex_releases(self, ctx: RunContext) -> bool:
        """Log if no suitable SeaDex releases are found

        Args:
            ctx (RunContext): The run's state (stats tally).
        """

        ctx.stats["no_releases"] += 1
        self.log_fmt.detail(
            "status",
            "no suitable releases on SeaDex",
            value_style="grey50",
        )

        return True

    def log_seadex_action(
        self,
        seadex_dict: dict,
        results: list,
        dry_run: bool = False,
    ) -> bool:
        """Log the action block for a title that differs from SeaDex's pick

        Called after the adding has run, so the status reflects what actually
        happened rather than what we set out to do: if a better release was
         grabbed, it reads "adding"; if every recommended release was already
        present, it reads "matches - keeping it". The block is, in order: the
        status line, then each recommended release group, then the per-release
        outcome (added / kept).

        Args:
            seadex_dict (dict): SeaDex entries (used for the recommended groups)
            results (list): add_torrent's per-release outcomes (empty on a dry
                run, where there are no client-reported names)
            dry_run (bool): No torrent client, so nothing was really grabbed,, but
                we'd have added everything. Defaults to False

        Returns:
            bool: True if a status block was logged; False if there was nothing
                to report (e.g., every release was skipped - the skip warning
                already explains that, so a status would only mislead)
        """

        added = dry_run or any(r.get("outcome") == "added" for r in results)

        # Nothing grabbed and nothing already present (e.g., all releases skipped
        # by public_only): leave the status to the inline "skipped" warning
        if not results and not dry_run:
            return False

        if added:
            self.log_fmt.detail(
                "status",
                "your copy differs from SeaDex's pick - adding a better release",
            )
        else:
            self.log_fmt.detail(
                "status",
                "your copy matches SeaDex's pick - keeping it",
                value_style="green",
            )

        # The release group(s) we recommend (those flagged for download), tags too
        for srg, srg_item in seadex_dict.items():

            urls = srg_item.get("urls", {})
            if any(u.get("download", False) for u in urls.values()):
                tags = srg_item.get("tags", [])
                if len(tags) > 0:
                    recommendation = f"{srg} [{', '.join(tags)}]"
                else:
                    recommendation = srg
                self.log_fmt.detail("group", recommendation, value_style="cyan")

        # Per-release outcome (qBittorrent path; a dry run has no names to show)
        for r in results:
            if r.get("outcome") == "added":
                self.log_fmt.detail("added", r.get("name"), value_style="green")
            else:
                self.log_fmt.detail("kept", r.get("name"))

        return True

    def log_max_torrents_added(self) -> bool:
        """Produce a log message about hitting the maximum number of torrents added"""

        self.logger.info(
            "Reached the maximum torrents for this run; stopping",
            extra={"line_style": "yellow"},
        )

        return True
