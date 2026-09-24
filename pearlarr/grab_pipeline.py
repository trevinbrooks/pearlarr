"""The grab "produce" side: add torrents, register pending records, write cache."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

from seadex import EntryRecord

from . import coverage as _coverage
from .cache import CacheRecord
from .config import PrivateReleaseAction
from .grab_placement import PendingSeed
from .log import count_noun
from .manual_import import ImportWaitMode
from .notify import GrabNotice
from .output import Accent, GrabFailed, ReleaseSkipped, SkipReason, StyledValue
from .pending_records import PendingRecords
from .reporter import (
    GrabRecord,
    NeedsActionKind,
    NeedsActionRecord,
    RunContext,
    is_preview,
)
from .seadex_types import SeadexDict, SeadexUrlItem
from .stamps import now_stamp, pending_cutoff, stamp_of
from .torrents import GRAB_FAILURES, PARSEABLE_TRACKERS, AddOutcome, AddResult, ReleaseOutcome

if TYPE_CHECKING:
    # Annotation-only: run_services imports this module at runtime (cycle).
    from .run_services import RunDeps


NO_SEEDS: Mapping[str, PendingSeed] = MappingProxyType({})
"""No seeds: the strategies build them only when a record can be persisted (the wait mode is on)."""


@dataclass(frozen=True)
class GrabRequest:
    """The resolved per-id payload for the shared grab tail."""

    al_id: int
    arr_title: str
    """The arr item's own title (the notification's byline when it adds information)."""
    entry_title: str
    """The entry's display title: AniList's, else the arr's, else the id form."""
    entry: EntryRecord
    """The SeaDex entry whole: the notification renders its url, notes, comparison links, and incomplete flag."""
    seadex_dict: SeadexDict
    torrent_hashes: list[str | None]
    cache_details: CacheRecord
    """The run's mutable `CacheRecord` accumulator: the frozen field pins the reference, not the dict's contents."""
    replaced_groups: tuple[str, ...]
    """The existing arr release groups this grab is replacing (the notifier's "Replacing" field)."""
    coverage: str = ""
    """Sonarr's episode coverage string ("" for Radarr)."""
    pending_seeds: Mapping[str, PendingSeed] = NO_SEEDS
    """One seed per flagged torrent with a video file, keyed by infohash (`NO_SEEDS` when the wait mode is off)."""
    input_missing_groups: tuple[str, ...] = ()
    """Groups with a url whose grab-time placement waited on a Sonarr read that failed this run, so a run
    may have been held: the title is never cached as done and re-checks next run."""


class GrabPipeline:
    """Adds the recommended release(s), registers pending records, writes the cache."""

    def __init__(
        self,
        *,
        deps: "RunDeps",
        ctx: RunContext,
    ) -> None:
        self._config = deps.config
        self._planner = deps.planner
        self.cache_store = deps.cache_store
        self._torrents = deps.torrents
        self._anilist = deps.anilist
        self._notifier = deps.notifier
        self._reporter = deps.reporter
        self.logger = deps.logger
        self.qbit = deps.qbit
        self._clock = deps.clock
        self._records = PendingRecords(deps.cache_store)
        # Rebound each run by begin_run to the same ctx the engine holds, so the grab bookkeeping stays in sync.
        self._ctx = ctx
        self._records.begin_run(ctx)

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the run context the grab bookkeeping reads/writes."""

        self._ctx = ctx
        self._records.begin_run(ctx)

    @property
    def records(self) -> PendingRecords:
        """The pending-record seam, bound to the current run (the hub hands it to every other reader)."""

        return self._records

    def _is_preview(self) -> bool:
        """A run is a no-op preview (nothing can be grabbed): explicit dry run, or qBittorrent not configured."""

        return is_preview(self._ctx, self.qbit)

    @property
    def _effective_cap(self) -> int | None:
        """The run-wide add cap, or None when uncapped (`0` disables it)."""
        cap = self._config.advanced.max_torrents_to_add
        return None if cap == 0 or self._is_preview() else cap

    def _cap_reached(self) -> bool:
        """Whether this run's adds have reached the cap (never under a preview or an uncapped run)."""

        cap = self._effective_cap
        return cap is not None and self._ctx.torrents_added >= cap

    def add_torrent(self, req: GrabRequest) -> tuple[int, list[ReleaseOutcome]]:
        """Add the request's torrent(s) to qBittorrent, holding the grabbable ones past the run's cap.

        Returns the added / already-downloading outcome lines rather than logging them (the caller emits the block).
        """

        n_torrents_added = 0
        results: list[ReleaseOutcome] = []

        for srg, srg_item in req.seadex_dict.items():
            for url_item in srg_item.urls.values():
                if not self._screen_url(srg, url_item):
                    continue
                if self._cap_reached():
                    self._hold_capped(url_item, req)
                    continue

                add_result = self._add_one_url(srg, url_item, req)
                if add_result is None:
                    continue

                results.append(add_result)
                if add_result.outcome is AddOutcome.ADDED:
                    self._ctx.torrents_added += 1
                    n_torrents_added += 1

        return n_torrents_added, results

    def _screen_url(self, srg: str, url_item: SeadexUrlItem) -> bool:
        """Whether a SeaDex url may be grabbed: flagged for download, public, on a selected tracker with a parser.

        A refused url posts its skip line and flags the title.
        """

        if not url_item.download:
            return False

        url = url_item.url
        tracker = url_item.tracker

        if not url_item.is_public:
            self._reporter.post(ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.PRIVATE_ONLY, url=url))
            self._ctx.per_title.private_only_skipped = True
            self._ctx.per_title.private_only_groups.append(srg)
            return False

        if tracker.casefold() not in self._config.seadex.trackers:
            self._reporter.post(
                ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.TRACKER_NOT_SELECTED, url=url),
            )
            return False

        # Invariant: an unparseable tracker never reaches TorrentService.add, whose raise is a defensive contract. This
        # skip and warn enforces it: handing one through unwinds the id's url loop, dropping later grabbable releases.
        if tracker not in PARSEABLE_TRACKERS:
            self._reporter.post(
                ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.UNSUPPORTED_TRACKER, url=url),
            )
            self._ctx.per_title.unsupported_tracker_skipped = True
            self._ctx.per_title.unsupported_tracker_groups.append(srg)
            if url_item.infohash is not None:
                self._ctx.per_title.unsupported_tracker_hashes.append(url_item.infohash)
            return False

        return True

    def _hold_capped(self, url_item: SeadexUrlItem, req: GrabRequest) -> None:
        """Hold a grabbable url past the cap: the title stays uncached, and a resident torrent keeps its claim.

        The tally lands before the save, so a raise in between never leaves the full-pass gate open.
        """

        if not self._ctx.per_title.held_by_cap:
            self._ctx.per_title.held_by_cap = True
            self._ctx.stats.held_by_cap += 1
        seed = self._seed_for(url_item, req)
        if seed is not None and seed.accreted:
            self._records.save(seed.record_at(now_stamp(), fresh=False))

    def _add_one_url(
        self,
        srg: str,
        url_item: SeadexUrlItem,
        req: GrabRequest,
    ) -> ReleaseOutcome | None:
        """Add one screened url, or None on a contained failure or a refused add.

        Both ADDED and ALREADY_ADDED persist the durable `PendingImport` (already-present means a prior-run grab).
        """

        url = url_item.url

        # An expected external failure (tracker or qBittorrent down) is contained to one warning here, so the loop
        # moves on and grab_and_cache leaves the title uncached for a retry next run.
        try:
            result = self._torrents.add(item=url_item, preview=self._is_preview())
        except GRAB_FAILURES as e:
            self._reporter.post(GrabFailed(group=srg, url=url, error=str(e)))
            self._ctx.per_title.grab_failed_groups.append(srg)
            return None

        if result.outcome is AddOutcome.ADDED:
            # Prefer the release's own parsed file list, falling back to the entry-level coverage so the summary's
            # files are never blank when a release's filenames couldn't be parsed.
            coverage_str = _coverage.coverage_string(url_item.episodes) or self._ctx.per_title.current_coverage
            self._ctx.stats.added.append(
                GrabRecord(
                    title=self._ctx.per_title.current_title,
                    coverage=coverage_str,
                    url=self._ctx.per_title.current_url,
                    name=result.name,
                    group=srg,
                ),
            )

        # ALREADY_ADDED is an earlier run's grab still awaiting import. The genuine "already own it" case is the
        # any_to_download=False branch, which never reaches add_torrent.
        if result.outcome in (AddOutcome.ADDED, AddOutcome.ALREADY_ADDED):
            self._register_pending_import(url_item, req, result)
            return ReleaseOutcome(outcome=result.outcome, name=result.name, group=srg)

        return None

    def _seed_for(self, url_item: SeadexUrlItem, req: GrabRequest) -> PendingSeed | None:
        """The url's seed (the strategy seeds nothing when the wait mode is off), or None on a preview or no hash."""

        if self._is_preview() or not url_item.infohash:
            return None
        return req.pending_seeds.get(url_item.infohash)

    def _register_pending_import(self, url_item: SeadexUrlItem, req: GrabRequest, result: AddResult) -> None:
        """Persist a grabbed or already-present release: a fresh `ADDED` tracks, an `ALREADY_ADDED` accretes or reacquires."""

        seed = self._seed_for(url_item, req)
        if seed is None:
            return
        if result.outcome is AddOutcome.ADDED:
            self._track_fresh(seed)
        elif seed.accreted:
            self._accrete_resident(seed)
        else:
            self._reacquire(seed, result.added_on)

    def _track_fresh(self, seed: PendingSeed) -> None:
        """A fresh add: the record enters the run list, its birth and every claim stamped now (a re-add too)."""

        self._records.insert_fresh(seed.record_at(now_stamp(), fresh=True))

    def _accrete_resident(self, seed: PendingSeed) -> None:
        """A torrent already downloading under a stored record: the entry's claim and placements join it.

        Reacquired only when the record is not this run's own grab (a torrent two entries list in one run).
        """

        record = seed.record_at(now_stamp(), fresh=False)
        self._records.save(record)
        if record.infohash not in self._ctx.pending_imports:
            self._ctx.reacquired_keys.add(record.infohash)

    def _reacquire(self, seed: PendingSeed, added_on: datetime | None) -> None:
        """A torrent qBittorrent holds with no record: tracked from its add time.

        Dropped when that time is already past `imports.pending_max_age_days`; stamped now when
        qBittorrent reports no add time.
        """

        stamp = now_stamp()
        if added_on is not None:
            max_age_days = self._config.imports.pending_max_age_days
            if added_on < pending_cutoff(max_age_days):
                self.logger.debug(
                    f"{seed.claim.title or seed.facts.infohash} has been in qBittorrent longer than "
                    f"{count_noun(max_age_days, 'day')}, not tracking it",
                )
                return
            stamp = stamp_of(added_on)
        # A reacquire, not a fresh grab: `save` refreshes without a run-list insert.
        record = seed.record_at(stamp, fresh=False)
        self._records.save(record)
        self._ctx.reacquired_keys.add(record.infohash)

    def _needs_action(self, groups: list[str], reason: str, kind: NeedsActionKind) -> NeedsActionRecord:
        """A needs-action record for the current title."""

        return NeedsActionRecord(
            title=self._ctx.per_title.current_title,
            coverage=self._ctx.per_title.current_coverage,
            group=", ".join(dict.fromkeys(groups)),
            url=self._ctx.per_title.current_url,
            reason=reason,
            kind=kind,
        )

    def _should_cache_as_done(self, *, added_this_title: int) -> bool:
        """Whether this title's outcome may be cached as done.

        Only if something was grabbed or nothing was skipped. A url held by the run cap, a fallback hold, a failed
        grab, or a failed Sonarr read under a placement vetoes it.
        """

        per_title = self._ctx.per_title
        # A non-interactive fallback-mode private hold means the fallback COULDN'T cover these files: never cache, so
        # every run re-checks and resurfaces it. Warn mode and interactive picks keep the plain gate.
        fallback_hold = (
            per_title.private_only_skipped
            and self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK
            and not self._config.advanced.interactive
        )
        # A contained grab failure means a release this title should have is missing: never cache, even on a partial
        # grab, so the next run retries (the completed add dedups).
        return (
            not per_title.held_by_cap
            and not fallback_hold
            and not per_title.grab_failed_groups
            and not per_title.input_missing_groups
            and (added_this_title > 0 or not (per_title.private_only_skipped or per_title.unsupported_tracker_skipped))
        )

    def _classify_needs_action(self) -> NeedsActionRecord | None:
        """The single needs-action row for a title NOT cached as done, or None.

        Flat guard-returns preserve the precedence private-only > unsupported-tracker > grab-failed > read-missed.
        """

        if self._ctx.per_title.private_only_skipped:
            reason, kind = self._private_only_reason()
            return self._needs_action(self._ctx.per_title.private_only_groups, reason, kind)

        if self._ctx.per_title.unsupported_tracker_skipped:
            return self._needs_action(
                self._ctx.per_title.unsupported_tracker_groups,
                "tracker not yet supported; grab manually",
                NeedsActionKind.UNSUPPORTED_TRACKER,
            )

        if self._ctx.per_title.grab_failed_groups:
            # No user action needed (the warning named it, the uncached title retries), but the summary must say why
            # the title is neither added nor up to date.
            return self._needs_action(
                self._ctx.per_title.grab_failed_groups,
                "grab failed; will retry next run",
                NeedsActionKind.GRAB_FAILED,
            )

        if self._ctx.per_title.input_missing_groups:
            # The placement may have held a run on the missing read, so the grab was judged coarsely: re-check.
            return self._needs_action(
                self._ctx.per_title.input_missing_groups,
                "a Sonarr read the placement needs failed; will retry next run",
                NeedsActionKind.PLACEMENT_INPUT_MISSING,
            )

        return None

    def _private_only_reason(self) -> tuple[str, NeedsActionKind]:
        """The (reason, kind) for a private-only hold, resolving the fallback-mode arms.

        The stale bit wins over a coexisting plain hold, keeping one row per title.
        """

        if self._config.seadex.private_releases is not PrivateReleaseAction.FALLBACK:
            return "private-only release; private releases not supported", NeedsActionKind.PRIVATE_ONLY
        if self._config.advanced.interactive:
            return (
                "hand-picked private release; private releases not supported",
                NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK,
            )
        if self._ctx.per_title.private_only_stale_held:
            return (
                (
                    "private-only release; your copy is outdated (its file size no longer matches) "
                    "and only a fallback covers it"
                ),
                NeedsActionKind.PRIVATE_ONLY_STALE,
            )
        return (
            "private-only release; no public alternative covers these files",
            NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK,
        )

    def grab_and_cache(self, req: GrabRequest) -> None:
        """Shared per-id tail: add torrents, notify, cache the outcome, pace the run."""

        any_to_download = self._planner.get_any_to_download(req.seadex_dict)
        # The strategy's placement facts land on the title's flags beside the add loop's own.
        self._ctx.per_title.input_missing_groups.extend(req.input_missing_groups)

        added_this_title = 0

        if not any_to_download:
            # A failed Sonarr read may have held a placement, so "already have it" is not claimed either.
            if not (self._ctx.per_title.private_only_skipped or self._ctx.per_title.input_missing_groups):
                self._ctx.stats.up_to_date += 1
                self._reporter.detail(
                    "status",
                    StyledValue("already have the recommended release", Accent.NOTE),
                )
        else:
            added_this_title = self._grab(req)

        if self._should_cache_as_done(added_this_title=added_this_title):
            # Unsupported-tracker hashes are excluded so the release is re-considered once a parser lands. Private-only
            # ones deliberately are not: private releases are never grabbed, so their quiet suppression is intended.
            skipped = set(self._ctx.per_title.unsupported_tracker_hashes)
            cacheable = [h for h in req.torrent_hashes if h is None or h not in skipped]
            # Always written: the partial-merge upsert would otherwise preserve a stale True.
            fallback_satisfied = self._ctx.per_title.fallback_covered or any(
                u.is_fallback and u.download for rg_item in req.seadex_dict.values() for u in rg_item.urls.values()
            )
            req.cache_details["torrent_hashes"] = cacheable
            req.cache_details["fallback_satisfied"] = fallback_satisfied
            self.cache_store.update_cache(
                self._ctx.arr,
                req.al_id,
                req.cache_details,
            )
        else:
            rec = self._classify_needs_action()
            if rec is not None:
                self._ctx.stats.needs_action.append(rec)

        self._clock.sleep(self._config.advanced.sleep_time)

    def _grab(self, req: GrabRequest) -> int:
        """Add this title's torrents and notify, returning the added count.

        The cap notice is logged here, once, by the title whose adds crossed it. The cache save belongs to the
        engine's finalize site.
        """

        # Fetched up front to keep the network calls in the run's request ordering.
        anilist_thumb = self._anilist.thumb(req.al_id)
        anilist_banner = self._anilist.banner(req.al_id)

        # add_torrent runs even in a preview: the service simulates the add, while the download-flag, private-release
        # and tracker filters still apply, so only releases that would really be grabbed are counted.
        before = self._ctx.torrents_added
        n_torrents_added, results = self.add_torrent(req)

        # Logged only now the outcome is known, so the status reads "adding" only when something was actually grabbed.
        self._reporter.log_seadex_action(
            req.seadex_dict,
            results,
            dry_run=self._is_preview(),
            monitor_active=(self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._is_preview()),
        )

        # Never on a preview: this is an outward notification. Built after the add so each group is labeled with what
        # actually happened.
        if self._notifier.enabled and n_torrents_added > 0 and not self._is_preview():
            self._notifier.push_grab(
                GrabNotice(
                    arr=self._ctx.arr,
                    arr_title=req.arr_title,
                    entry_title=req.entry_title,
                    entry=req.entry,
                    thumb_url=anilist_thumb,
                    banner_url=anilist_banner,
                    replaced_groups=req.replaced_groups,
                    seadex_dict=req.seadex_dict,
                    results=results,
                    failed_groups=frozenset(self._ctx.per_title.grab_failed_groups),
                    coverage=req.coverage,
                ),
            )

        if self._ctx.per_title.held_by_cap:
            self._reporter.detail("status", StyledValue("held by the run cap; grabbed next run", Accent.NOTE))

        cap = self._effective_cap
        if cap is not None and before < cap <= self._ctx.torrents_added:
            self._reporter.log_max_torrents_added(cap)

        return n_torrents_added
