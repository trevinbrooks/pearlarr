"""The grab "produce" side: add torrents, register pending records, write cache."""

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from seadex import EntryRecord

from . import coverage as _coverage
from .cache import CacheRecord, pending_cutoff, stamp_of
from .config import Arr, PrivateReleaseAction
from .log import count_noun
from .manual_import import ImportWaitMode, PendingImport
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
from .torrents import GRAB_FAILURES, PARSEABLE_TRACKERS, AddOutcome, AddResult, ReleaseOutcome

if TYPE_CHECKING:
    # Annotation-only: run_services imports this module at runtime (cycle).
    from .run_services import RunDeps


class GrabResult(NamedTuple):
    """`_grab`'s outcome."""

    cap_reached: bool
    added: int


@dataclass(frozen=True)
class GrabRequest:
    """The resolved per-id payload for the shared grab tail."""

    al_id: int
    item_title: str
    anilist_title: str
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
    pending_seeds: dict[str, PendingImport] | None = None


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
        self._records = PendingRecords(deps.cache_store)
        # Rebound each run by begin_run to the same ctx the engine holds, so the grab bookkeeping stays in sync.
        self._ctx = ctx
        self._records.begin_run(ctx)

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the run context the grab bookkeeping reads/writes."""

        self._ctx = ctx
        self._records.begin_run(ctx)

    def _is_preview(self) -> bool:
        """A run is a no-op preview (nothing can be grabbed): explicit dry run, or qBittorrent not configured."""

        return is_preview(self._ctx, self.qbit)

    @property
    def _effective_cap(self) -> int | None:
        """The run-wide add cap, or None when uncapped (`0` disables it)."""
        cap = self._config.advanced.max_torrents_to_add
        return None if cap == 0 or self._is_preview() else cap

    def add_torrent(self, req: GrabRequest) -> tuple[int, list[ReleaseOutcome]]:
        """Add the request's torrent(s) to qBittorrent.

        Returns the added / already-downloading outcome lines rather than logging them (the caller emits the block).
        """

        n_torrents_added = 0
        results: list[ReleaseOutcome] = []
        cap = self._effective_cap

        for srg, srg_item in req.seadex_dict.items():
            for url_item in srg_item.urls.values():
                add_result = self._add_one_url(srg, url_item, req)
                if add_result is None:
                    continue

                results.append(add_result)
                if add_result.outcome is not AddOutcome.ADDED:
                    continue

                self._ctx.torrents_added += 1
                n_torrents_added += 1
                if cap is not None and self._ctx.torrents_added >= cap:
                    return n_torrents_added, results

        return n_torrents_added, results

    def _add_one_url(
        self,
        srg: str,
        url_item: SeadexUrlItem,
        req: GrabRequest,
    ) -> ReleaseOutcome | None:
        """Resolve a single SeaDex url to an add outcome (or `None` to skip).

        Both ADDED and ALREADY_ADDED persist the durable `PendingImport` (already-present means a prior-run grab).
        """

        if not url_item.download:
            return None

        url = url_item.url
        tracker = url_item.tracker

        if not url_item.is_public:
            self._reporter.post(ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.PRIVATE_ONLY, url=url))
            self._ctx.per_title.private_only_skipped = True
            self._ctx.per_title.private_only_groups.append(srg)
            return None

        if tracker.casefold() not in self._config.seadex.trackers:
            self._reporter.post(
                ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.TRACKER_NOT_SELECTED, url=url),
            )
            return None

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
            return None

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

    def _register_pending_import(self, url_item: SeadexUrlItem, req: GrabRequest, result: AddResult) -> None:
        """Finalize the durable `PendingImport` for a grabbed or already-present release.

        A fresh `ADDED` inserts the seed as a this-run grab. An `ALREADY_ADDED` reacquire keeps a
        store-resident record as is, whatever its age (`prune_expired_pending` is the sole TTL
        authority for tracked records). A non-resident reacquire joins at qBittorrent's add time,
        stamped now when that time is unknown, and dropped when it is already past
        `imports.pending_max_age_days`.
        """

        seeds = req.pending_seeds
        if (
            self._ctx.import_wait_mode is ImportWaitMode.OFF
            or self._is_preview()
            or not url_item.infohash
            or not seeds
            or url_item.infohash not in seeds
        ):
            return
        pending = seeds[url_item.infohash]
        if result.outcome is AddOutcome.ADDED:
            self._records.insert_fresh(pending)
        elif self._records.has(pending.key):
            self._ctx.reacquired_keys.add(pending.key)
        else:
            if result.added_on is not None:
                max_age_days = self._config.imports.pending_max_age_days
                if result.added_on < pending_cutoff(max_age_days):
                    self.logger.debug(
                        f"{pending.display_label} has been in qBittorrent longer than "
                        f"{count_noun(max_age_days, 'day')}, not tracking it",
                    )
                    return
                pending = replace(pending, added_at=stamp_of(result.added_on))
            # A reacquire, not a fresh grab: `save` refreshes without a run-list insert.
            self._records.save(pending)
            self._ctx.reacquired_keys.add(pending.key)
        # One guard row per entry (Sonarr only): each per-release firing re-puts the same row, a no-op upsert.
        if self._ctx.arr is Arr.SONARR:
            self.cache_store.put_guards(self._ctx.arr, pending.al_id, pending.guards)

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

    def _should_cache_as_done(self, *, cap_reached: bool, added_this_title: int, grab_failed: bool) -> bool:
        """Whether this title's outcome may be cached as done.

        Only if something was grabbed or nothing was skipped. The run cap, a fallback hold, or a failed grab vetoes it.
        """

        # A non-interactive fallback-mode private hold means the fallback COULDN'T cover these files: never cache, so
        # every run re-checks and resurfaces it. Warn mode and interactive picks keep the plain gate.
        fallback_hold = (
            self._ctx.per_title.private_only_skipped
            and self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK
            and not self._config.advanced.interactive
        )
        return (
            not cap_reached
            and not fallback_hold
            and not grab_failed
            and (
                added_this_title > 0
                or not (self._ctx.per_title.private_only_skipped or self._ctx.per_title.unsupported_tracker_skipped)
            )
        )

    def _classify_needs_action(self, *, grab_failed: bool) -> NeedsActionRecord | None:
        """The single needs-action row for a title NOT cached as done, or None.

        Flat guard-returns preserve the precedence private-only > unsupported-tracker > grab-failed.
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

        if grab_failed:
            # No user action needed (the warning named it, the uncached title retries), but the summary must say why
            # the title is neither added nor up to date.
            return self._needs_action(
                self._ctx.per_title.grab_failed_groups,
                "grab failed; will retry next run",
                NeedsActionKind.GRAB_FAILED,
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

    def grab_and_cache(self, req: GrabRequest) -> bool:
        """Shared per-id tail: add torrents, notify, cache the outcome. True when the run-wide cap was hit."""

        any_to_download = self._planner.get_any_to_download(req.seadex_dict)

        # The cap can stop the url loop mid-title, so a capped title is never cached as done, only classified below.
        cap_reached = False
        added_this_title = 0

        if not any_to_download:
            if not self._ctx.per_title.private_only_skipped:
                self._ctx.stats.up_to_date += 1
                self._reporter.detail(
                    "status",
                    StyledValue("already have the recommended release", Accent.NOTE),
                )
        else:
            cap_reached, added_this_title = self._grab(req)

        # A contained grab failure means a release this title should have is missing: never cache, even on a partial
        # grab, so the next run retries (the completed add dedups).
        grab_failed = bool(self._ctx.per_title.grab_failed_groups)

        if self._should_cache_as_done(
            cap_reached=cap_reached,
            added_this_title=added_this_title,
            grab_failed=grab_failed,
        ):
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
            rec = self._classify_needs_action(grab_failed=grab_failed)
            if rec is not None:
                self._ctx.stats.needs_action.append(rec)

        # Stop only now the summary rows are recorded, skipping the throttle: no point pacing a run that's over.
        if cap_reached:
            return True

        time.sleep(self._config.advanced.sleep_time)

        return False

    def _grab(self, req: GrabRequest) -> GrabResult:
        """Add this title's torrents, notify, and honor the run-wide cap.

        The cap notice is logged here, but the cache save belongs to the engine's finalize site.
        """

        # Fetched up front to keep the network calls in the run's request ordering.
        anilist_thumb = self._anilist.thumb(req.al_id)
        anilist_banner = self._anilist.banner(req.al_id)

        # add_torrent runs even in a preview: the service simulates the add, while the download-flag, private-release
        # and tracker filters still apply, so only releases that would really be grabbed are counted.
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
                    arr_title=req.item_title,
                    al_title=req.anilist_title,
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

        cap = self._effective_cap
        if cap is not None and self._ctx.torrents_added >= cap:
            self._reporter.log_max_torrents_added(cap)
            return GrabResult(cap_reached=True, added=n_torrents_added)

        return GrabResult(cap_reached=False, added=n_torrents_added)
