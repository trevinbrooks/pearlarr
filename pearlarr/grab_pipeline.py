"""The grab "produce" side: add torrents, register pending records, write cache."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

from seadex import EntryRecord

from . import coverage as _coverage
from .arr_run_state import ArrRunState, GrabbedRow, NeedsActionRow, is_dry_run
from .cache import CachedTitleUpdate
from .config import PrivateReleaseAction
from .grab_placement import PendingImportUpdate
from .notify import GrabNotice
from .output import Accent, GrabFailed, NeedsActionCause, ReleaseSkipped, StyledValue
from .pending_import_store import PendingImportStore
from .releases import Release, SeadexDict, SkipReason
from .stamps import now_stamp, stamp_of
from .torrent_service import GRAB_FAILURES, PARSEABLE_TRACKERS, AddResult, AddStatus, GrabResult
from .wait_types import ImportWaitMode

if TYPE_CHECKING:
    # Annotation-only: arr_run_services imports this module at runtime (cycle).
    from .arr_run_services import ArrRunDeps


NO_PENDING_IMPORT_UPDATES: Mapping[str, PendingImportUpdate] = MappingProxyType({})
"""No seeds: the strategies build them only when a record can be persisted (the wait mode is on)."""


@dataclass(frozen=True, slots=True)
class SkipReport:
    """What the run reports for a url the placement held: its title's needs-action row."""

    cause: NeedsActionCause
    reason: str
    """The needs-action row's reason text."""


# The order is the precedence `_classify_needs_action` reports holds in: the first hand-import hold found here
# wins, and any other hold is reported only when no hand-import hold and no failed grab is.
_SKIP_REPORTS: Mapping[SkipReason, SkipReport] = MappingProxyType(
    {
        SkipReason.MISNUMBERED: SkipReport(
            NeedsActionCause.MISNUMBERED,
            "numbering does not match its SeaDex listing; import it by hand",
        ),
        SkipReason.TITLE_REFUTED: SkipReport(
            NeedsActionCause.TITLE_REFUTED,
            "file titles contradict its numbering; import it by hand",
        ),
        # No retry clears it: the row stays until the entry holds the group's files, so it outranks a failed read.
        SkipReason.PLACED_ELSEWHERE: SkipReport(
            NeedsActionCause.PLACED_ELSEWHERE,
            "its files place under another title SeaDex lists it with; any it could not place need a hand import",
        ),
        # The placement may have held a run on the missing read, so the grab was judged coarsely: re-check.
        SkipReason.INPUT_UNREAD: SkipReport(
            NeedsActionCause.PLACEMENT_INPUT_MISSING,
            "a read the placement needs failed; will retry next run",
        ),
    },
)


@dataclass(frozen=True)
class GrabRequest:
    """The resolved per-id payload for the shared grab tail."""

    al_id: int
    arr_title: str
    """The arr item's own title (the notification's byline when it adds information)."""
    title_name: str
    """The entry's display title: AniList's, else the arr's, else the id form."""
    entry: EntryRecord
    """The SeaDex entry whole: the notification renders its url, notes, comparison links, and incomplete flag."""
    seadex_dict: SeadexDict
    update: CachedTitleUpdate
    """The run's mutable `CacheRecord` accumulator: the frozen field pins the reference, not the dict's contents."""
    replaced_groups: tuple[str, ...]
    """The existing arr release groups this grab is replacing (the notifier's "Replacing" field)."""
    coverage: str = ""
    """Sonarr's episode coverage string ("" for Radarr)."""
    pending_import_updates: Mapping[str, PendingImportUpdate] = NO_PENDING_IMPORT_UPDATES
    """One seed per flagged torrent with a video file, keyed by its hash in qBittorrent (`NO_SEEDS` when the wait
    mode is off)."""
    input_unread_groups: tuple[str, ...] = ()
    """Groups with a url whose grab-time placement waited on a read that failed this run, so a run may have
    been held: the title is never cached as done and re-checks next run."""
    metadata_unread: bool = False
    """An entry of the series was looked up without its AniList record (`SonarrEpisodes.entries_unread`), or a
    series was read without its TVDB orders (`SonarrEpisodes.orders_unread`): the title is never cached as done and
    re-checks next run."""


def _reacquire_stamp(added_on: datetime | None) -> str:
    """The `added_at` of a new record for a torrent already in qBittorrent.

    Its add time there, or now when the client reports none.
    """

    return now_stamp() if added_on is None else stamp_of(added_on)


class GrabPipeline:
    """Adds the recommended release(s), registers pending records, writes the cache."""

    def __init__(
        self,
        *,
        deps: "ArrRunDeps",
        state: ArrRunState,
        pending_imports: PendingImportStore,
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
        # The hub's record seam, which the hub binds each run.
        self._pending_imports = pending_imports
        # Rebound each run by bind_arr_run to the same ctx the engine holds, so the grab bookkeeping stays in sync.
        self._state = state

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the run context the grab bookkeeping reads/writes."""

        self._state = state

    def _is_dry_run(self) -> bool:
        """A run is a no-op preview (nothing can be grabbed): explicit dry run, or qBittorrent not configured."""

        return is_dry_run(self._state, self.qbit)

    @property
    def _effective_cap(self) -> int | None:
        """The run-wide add cap, or None when uncapped (`0` disables it)."""
        cap = self._config.advanced.max_torrents_to_add
        return None if cap == 0 or self._is_dry_run() else cap

    def _cap_reached(self) -> bool:
        """Whether this run's adds have reached the cap (never under a preview or an uncapped run)."""

        cap = self._effective_cap
        return cap is not None and self._state.torrents_added >= cap

    def grab_releases(self, req: GrabRequest) -> tuple[int, list[GrabResult]]:
        """Add the request's torrent(s) to qBittorrent, holding the grabbable ones past the run's cap.

        Returns the added / already-downloading outcome lines rather than logging them (the caller emits the block).
        """

        n_torrents_added = 0
        results: list[GrabResult] = []

        for srg, group_releases in req.seadex_dict.items():
            for release in group_releases.releases.values():
                if not self._may_grab(srg, release):
                    continue
                if self._cap_reached():
                    self._hold_capped(release, req)
                    continue

                add_result = self._grab_release(srg, release, req)
                if add_result is None:
                    continue

                results.append(add_result)
                if add_result.outcome is AddStatus.ADDED:
                    self._state.torrents_added += 1
                    n_torrents_added += 1

        return n_torrents_added, results

    def _may_grab(self, srg: str, release: Release) -> bool:
        """Whether a SeaDex url may be grabbed: flagged, public, on a selected tracker with a parser, and not held.

        A refused url posts its skip line and flags the title.
        """

        if not release.to_grab:
            return False

        url = release.url
        tracker = release.tracker

        if not release.is_public:
            self._reporter.post(ReleaseSkipped(group=srg, tracker=tracker, reason=SkipReason.PRIVATE_ONLY, url=url))
            self._state.per_title.private_only_skipped = True
            self._state.per_title.private_only_groups.append(srg)
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
            self._state.per_title.unsupported_tracker_skipped = True
            self._state.per_title.unsupported_tracker_groups.append(srg)
            return False

        skip_reason = release.skip_reason
        if skip_reason is None:
            return True
        # Never added, and the title stays uncached for the next run.
        self._state.per_title.add_skipped(skip_reason, (srg,))
        self._reporter.post(
            ReleaseSkipped(
                group=srg,
                tracker=tracker,
                reason=skip_reason,
                url=url,
                counts=release.another_titles_counts,
            )
        )
        return False

    def _hold_capped(self, release: Release, req: GrabRequest) -> None:
        """Hold a grabbable url past the cap: the title stays uncached, and a resident torrent keeps its claim."""

        if not self._state.per_title.held_by_cap:
            self._state.per_title.held_by_cap = True
            self._state.stats.held_by_cap += 1
        update = self._update_for(release, req)
        if update is not None and (joined := update.joined()) is not None:
            self._pending_imports.save_with_claim(joined, update.claim)

    def _grab_release(
        self,
        srg: str,
        release: Release,
        req: GrabRequest,
    ) -> GrabResult | None:
        """Add one screened url and register its pending import: its outcome, or None on a contained failure."""

        url = release.url

        # An expected external failure (tracker or qBittorrent down) is contained to one warning here, so the loop
        # moves on and grab_and_cache leaves the title uncached for a retry next run.
        try:
            result = self._torrents.add(item=release, dry_run=self._is_dry_run())
        except GRAB_FAILURES as e:
            self._reporter.post(GrabFailed(group=srg, url=url, error=str(e)))
            self._state.per_title.grab_failed_groups.append(srg)
            return None

        if result.outcome is AddStatus.ADDED:
            # Prefer the release's own parsed file list, falling back to the entry-level coverage so the summary's
            # files are never blank when a release's filenames couldn't be parsed.
            coverage_str = _coverage.coverage_string(release.episodes) or self._state.per_title.entry_coverage
            self._state.stats.added.append(
                GrabbedRow(
                    title=self._state.per_title.title_name,
                    coverage=coverage_str,
                    url=self._state.per_title.entry_url,
                    name=result.name,
                    group=srg,
                ),
            )

        # ALREADY_ADDED is the url's torrent already in qBittorrent, SeaDex's or a cross-seed holding its files,
        # whoever added it, and it's tracked like a grab. The genuine "already own it" case is the
        # any_to_grab=False branch, which never reaches grab_releases.
        self._register_pending_import(release, req, result)
        return GrabResult(outcome=result.outcome, name=result.name, group=srg)

    def _update_for(self, release: Release, req: GrabRequest) -> PendingImportUpdate | None:
        """The url's seed (the strategy seeds nothing when the wait mode is off), or None on a preview or no hash."""

        qbit_hash = release.qbit_hash
        if self._is_dry_run() or not qbit_hash:
            return None
        return req.pending_import_updates.get(qbit_hash)

    def _register_pending_import(self, release: Release, req: GrabRequest, result: AddResult) -> None:
        """Persist a grabbed or already-present release: `ADDED` tracks, `ALREADY_ADDED` accretes or reacquires.

        A fresh add enters the run list stamped now. A torrent already in qBittorrent, whoever added it, joins its
        stored record or starts one, and is reacquired unless the record is this run's own grab (a torrent two
        entries list in one run).
        """

        update = self._update_for(release, req)
        if update is None:
            return
        if result.outcome is AddStatus.ADDED:
            self._pending_imports.insert_fresh(update.pending_import_at(now_stamp(), fresh=True), update.claim)
            return
        pending = update.pending_import_at(_reacquire_stamp(result.added_on), fresh=False)
        self._pending_imports.save_with_claim(pending, update.claim)
        if pending.infohash not in self._state.grabbed_this_run:
            self._state.reacquired_keys.add(pending.infohash)

    def _needs_action(self, groups: list[str], reason: str, cause: NeedsActionCause) -> NeedsActionRow:
        """A needs-action record for the current title."""

        return NeedsActionRow(
            title=self._state.per_title.title_name,
            coverage=self._state.per_title.entry_coverage,
            group=", ".join(dict.fromkeys(groups)),
            url=self._state.per_title.entry_url,
            reason=reason,
            cause=cause,
        )

    def _should_cache_as_done(self, *, added_this_title: int) -> bool:
        """Whether this title's outcome may be cached as done.

        Only if something was grabbed or nothing was skipped. A url held by the run cap, a fallback hold, a failed
        grab, a failed read under a placement, a url the placement held, or a check that read an entry without its
        AniList record or a series without its TVDB orders vetoes it.
        """

        per_title = self._state.per_title
        # A non-interactive fallback-mode private hold means nothing public was grabbed for these files: never cache,
        # so every run re-checks and resurfaces it. Warn mode and interactive picks keep the plain gate.
        private_only_no_fallback = (
            per_title.private_only_skipped
            and self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK
            and not self._config.advanced.interactive
        )
        # A contained grab failure means a release this title should have is missing: never cache, even on a partial
        # grab, so the next run retries (the completed add dedups).
        return (
            not per_title.held_by_cap
            and not private_only_no_fallback
            and not per_title.grab_failed_groups
            and not per_title.skipped_groups
            and not per_title.metadata_unread
            and (added_this_title > 0 or not (per_title.private_only_skipped or per_title.unsupported_tracker_skipped))
        )

    def _classify_needs_action(self) -> NeedsActionRow | None:
        """The single needs-action row for a title NOT cached as done, or None.

        Flat guard-returns keep the precedence private-only > unsupported-tracker > hand-import holds > grab-failed >
        the other holds, each in `_HOLD_ROWS` order. A hand import is what no retry brings, so its row outranks a
        failed grab's. A failed grab is a release the title should have, so its row outranks a url held as another
        title's or on a failed read.
        """

        if self._state.per_title.private_only_skipped:
            reason, cause = self._private_only_reason()
            return self._needs_action(self._state.per_title.private_only_groups, reason, cause)

        if self._state.per_title.unsupported_tracker_skipped:
            return self._needs_action(
                self._state.per_title.unsupported_tracker_groups,
                "tracker not yet supported; grab manually",
                NeedsActionCause.UNSUPPORTED_TRACKER,
            )

        hand_import_row = self._needs_action_row_for(
            skip_reason for skip_reason in _SKIP_REPORTS if skip_reason.left_for_hand_import
        )
        if hand_import_row is not None:
            return hand_import_row

        if self._state.per_title.grab_failed_groups:
            # No user action needed (the warning named it, the uncached title retries), but the summary must say why
            # the title is neither added nor up to date.
            return self._needs_action(
                self._state.per_title.grab_failed_groups,
                "grab failed; will retry next run",
                NeedsActionCause.GRAB_FAILED,
            )

        return self._needs_action_row_for(
            skip_reason for skip_reason in _SKIP_REPORTS if not skip_reason.left_for_hand_import
        )

    def _needs_action_row_for(self, holds: Iterable[SkipReason]) -> NeedsActionRow | None:
        """The row of the first of `holds` that held a group of this title, or None."""

        skipped_groups = self._state.per_title.skipped_groups
        for skip_reason in holds:
            groups = skipped_groups.get(skip_reason)
            if groups:
                hold_row = _SKIP_REPORTS[skip_reason]
                return self._needs_action(groups, hold_row.reason, hold_row.cause)
        return None

    def _private_only_reason(self) -> tuple[str, NeedsActionCause]:
        """The (reason, cause) for a private-only hold, resolving the fallback-mode arms.

        The stale bit wins over a coexisting plain hold, keeping one row per title.
        """

        if self._config.seadex.private_releases is not PrivateReleaseAction.FALLBACK:
            return "private-only release; private releases not supported", NeedsActionCause.PRIVATE_ONLY
        if self._config.advanced.interactive:
            return (
                "hand-picked private release; private releases not supported",
                NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
            )
        if self._state.per_title.private_only_outdated:
            return (
                (
                    "private-only release; your copy is outdated (its file size no longer matches) "
                    "and only a fallback covers it"
                ),
                NeedsActionCause.PRIVATE_ONLY_STALE,
            )
        return (
            "private-only release; no public alternative covers these files",
            NeedsActionCause.PRIVATE_ONLY_NO_FALLBACK,
        )

    def grab_and_cache(self, req: GrabRequest) -> None:
        """Shared per-id tail: add torrents, notify, cache the outcome."""

        any_to_grab = self._planner.any_to_grab(req.seadex_dict)
        # The strategy's placement facts land on the title's flags beside the add loop's own.
        self._state.per_title.add_skipped(SkipReason.INPUT_UNREAD, req.input_unread_groups)
        self._state.per_title.metadata_unread |= req.metadata_unread

        added_this_title = 0

        if not any_to_grab:
            # A failed Sonarr read may have held a placement, so "already have it" is not claimed either.
            if not (self._state.per_title.private_only_skipped or req.input_unread_groups):
                self._state.stats.up_to_date += 1
                self._reporter.detail(
                    "status",
                    StyledValue("already have the recommended release", Accent.NOTE),
                )
        else:
            added_this_title = self._grab(req)

        if self._should_cache_as_done(added_this_title=added_this_title):
            # Always written: the partial-merge upsert would otherwise preserve a stale True.
            req.update["fallback_satisfied"] = any(
                u.is_fallback and u.to_grab
                for group_releases in req.seadex_dict.values()
                for u in group_releases.releases.values()
            )
            self.cache_store.update_cached_title(
                self._state.arr,
                req.al_id,
                req.update,
            )
        else:
            rec = self._classify_needs_action()
            if rec is not None:
                self._state.stats.needs_action.append(rec)

    def _grab(self, req: GrabRequest) -> int:
        """Add this title's torrents and notify, returning the added count.

        The cap notice is logged here, once, by the title whose adds crossed it. The cache save belongs to the
        run loop.
        """

        # Fetched up front to keep the network calls in the run's request ordering.
        anilist_thumb = self._anilist.thumb(req.al_id)
        anilist_banner = self._anilist.banner(req.al_id)

        # grab_releases runs even in a preview: the service simulates the add, while the download-flag, private-release
        # and tracker filters still apply, so only releases that would really be grabbed are counted.
        before = self._state.torrents_added
        n_torrents_added, results = self.grab_releases(req)

        # Logged only now the outcome is known, so the status reads "adding" only when something was actually grabbed.
        dry_run = self._is_dry_run()
        self._reporter.log_seadex_action(
            req.seadex_dict,
            results,
            dry_run=dry_run,
            waiting_to_import=self._state.import_wait_mode is not ImportWaitMode.OFF and not dry_run,
        )

        # Never on a preview: this is an outward notification. Built after the add so each group is labeled with what
        # actually happened.
        if self._notifier.enabled and n_torrents_added > 0 and not dry_run:
            self._notifier.push_grab(
                GrabNotice(
                    arr=self._state.arr,
                    arr_title=req.arr_title,
                    title_name=req.title_name,
                    entry=req.entry,
                    thumb_url=anilist_thumb,
                    banner_url=anilist_banner,
                    replaced_groups=req.replaced_groups,
                    seadex_dict=req.seadex_dict,
                    results=results,
                    failed_groups=frozenset(self._state.per_title.grab_failed_groups),
                    coverage=req.coverage,
                ),
            )

        if self._state.per_title.held_by_cap:
            self._reporter.detail("status", StyledValue("held by the run cap; grabbed next run", Accent.NOTE))

        cap = self._effective_cap
        if cap is not None and before < cap <= self._state.torrents_added:
            self._reporter.log_max_torrents_added(cap)

        return n_torrents_added
