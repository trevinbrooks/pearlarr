"""The grab "produce" side: add torrents, register pending records, write cache.

Extracted from the old ``RunLoop`` god class. ``GrabPipeline`` owns the per-id
grab tail both strategies funnel into - add the recommended release(s) to
qBittorrent, persist the durable :class:`PendingImport` records the end-of-run
monitor waits on, notify, and write the cache outcome. It returns a pure bool
(cap-reached) and never calls back into the run loop;
:class:`~.run_services.RunServices` keeps a thin ``grab_and_cache`` delegator so
the strategy<->services contract is unchanged.

Binds the run :class:`RunContext` via :meth:`begin_run` (the same object the
run loop holds), so the grab bookkeeping the run summary reads stays in sync.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seadex import EntryRecord

from . import coverage as _coverage
from .cache import CacheRecord
from .config import PrivateReleaseAction
from .manual_import import ImportWaitMode, PendingImport
from .notify import GrabNotice
from .reporter import (
    GrabRecord,
    NeedsActionKind,
    NeedsActionRecord,
    RunContext,
    is_preview,
)
from .seadex_types import SeadexDict, SeadexUrlItem
from .torrents import GRAB_FAILURES, PARSEABLE_TRACKERS, AddOutcome, ReleaseOutcome

if TYPE_CHECKING:
    # Annotation-only: run_services imports this module at runtime (cycle).
    from .run_services import RunDeps


@dataclass(frozen=True)
class GrabRequest:
    """The resolved per-id payload for the shared grab tail.

    ``cache_details`` is the run's mutable :class:`CacheRecord` accumulator: the
    frozen field pins the reference, not the dict's contents (``grab_and_cache``
    still writes ``torrent_hashes`` into it before saving).
    """

    al_id: int
    item_title: str
    anilist_title: str
    # The SeaDex entry whole: the notification renders its url / notes /
    # comparison links / incomplete flag.
    entry: EntryRecord
    seadex_dict: SeadexDict
    torrent_hashes: list[str | None]
    cache_details: CacheRecord
    release_group: list[str | None] | None
    # Sonarr's episode coverage string ("" for Radarr - movies have none).
    coverage: str = ""
    pending_seeds: dict[str, PendingImport] | None = None


class GrabPipeline:
    """Adds the recommended release(s), registers pending records, writes the cache.

    Constructed once per run in :class:`~.run_services.RunServices` from the
    deps hub + the placeholder ctx (unpacked to private attrs here);
    :meth:`begin_run` rebinds the ctx each run. The hub's ``grab_and_cache``
    delegates here; ``_grab`` returns a pure bool (cap-reached) so the run loop
    owns the single finalize site.
    """

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
        self.log_fmt = deps.log_fmt
        self.qbit = deps.qbit
        # Seeded with the engine's placeholder ctx; rebound each run via begin_run
        # (the same object the engine holds, so the grab bookkeeping stays in sync).
        self._ctx = ctx
        # Groups whose release hit a contained grab failure (tracker/client down)
        # this title; set in _add_one_url, read + reset in grab_and_cache.
        self._grab_failed_groups: list[str] = []

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the run context the grab bookkeeping reads/writes."""

        self._ctx = ctx

    def _is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return is_preview(self._ctx, self.qbit)

    def add_torrent(
        self,
        torrent_dict: SeadexDict,
        pending_seeds: dict[str, PendingImport] | None = None,
    ) -> tuple[int, list[ReleaseOutcome]]:
        """Add torrent(s) to qBittorrent

        The per-release outcome lines (added / already-downloading) are NOT
        logged here; this returns them so the caller (log_seadex_action) can emit
        the whole block - added releases first, then already-downloading ones -
        with a status that reflects what actually happened: "adding" if anything
        was grabbed, "already downloading" if every recommended release was
        already in the client from a prior run. The "skipped" warnings
        (private-only, unselected tracker) are still logged inline, as they're
        independent of that status.

        Args:
            torrent_dict (dict): Dictionary of torrent info
            pending_seeds (dict[str, PendingImport] | None): The Sonarr strategy's
                ``infohash -> PendingImport`` seeds, finalized into a durable
                record on a successful add. Radarr passes None.

        Returns:
            tuple: (n_torrents_added, results), where results is a list of
                ``ReleaseOutcome``, one per release acted on, in order
        """

        n_torrents_added = 0
        results: list[ReleaseOutcome] = []
        cap = self._config.advanced.max_torrents_to_add

        for srg, srg_item in torrent_dict.items():
            for url_item in srg_item.urls.values():
                add_result = self._add_one_url(
                    srg,
                    url_item,
                    pending_seeds=pending_seeds,
                )
                if add_result is None:
                    continue

                results.append(add_result)
                if add_result.outcome is not AddOutcome.ADDED:
                    continue

                # Stop once max_torrents_to_add is reached
                self._ctx.torrents_added += 1
                n_torrents_added += 1
                if cap is not None and self._ctx.torrents_added >= cap:
                    return n_torrents_added, results

        return n_torrents_added, results

    def _add_one_url(
        self,
        srg: str,
        url_item: SeadexUrlItem,
        pending_seeds: dict[str, PendingImport] | None = None,
    ) -> ReleaseOutcome | None:
        """Resolve a single SeaDex url to an add outcome (or ``None`` to skip).

        Returns ``None`` for a release that's filtered out (not flagged for
        download, private-only, or an unselected tracker)
        and for a service ``add`` that neither added nor was already present. On
        an ``AddOutcome.ADDED`` the run-summary grab record is appended here; the
        caller owns the torrents_added/cap bookkeeping. On EITHER ``ADDED`` or
        ``ALREADY_ADDED`` (an already-present torrent is a prior-run grab still
        downloading / not yet imported) the durable :class:`PendingImport` record
        is persisted via :meth:`_register_pending_import` so the end-of-run monitor
        waits on it - when the feature is on, off-preview, and we hold its seed.
        """

        # If not flagged for download, then skip
        if not url_item.download:
            return None

        url = url_item.url
        tracker = url_item.tracker

        if not url_item.is_public:
            self.log_fmt.detail(
                "skipped",
                f"{srg} on {tracker} (private-only)",
                value_style="yellow",
                level=logging.WARNING,
            )
            self._ctx.private_only_skipped = True
            self._ctx.private_only_groups.append(srg)
            return None

        # Skip trackers not in the user's selected list
        if tracker.casefold() not in self._config.seadex.trackers:
            self.log_fmt.detail(
                "skipped",
                f"{url} (tracker {tracker} not in your selected list)",
                value_style="yellow",
            )
            return None

        # Skip trackers we have no parser for. Handing one to the service would
        # raise, unwinding this id's whole url loop (dropping any later grabbable
        # release too); skip+warn here so the loop continues, and flag the title so
        # it's not cached as done (re-checked once a parser / config change lands).
        if tracker not in PARSEABLE_TRACKERS:
            self.log_fmt.detail(
                "skipped",
                f"{url} (tracker {tracker} not yet supported)",
                value_style="yellow",
                level=logging.WARNING,
            )
            self._ctx.unsupported_tracker_skipped = True
            self._ctx.unsupported_tracker_groups.append(srg)
            if url_item.infohash is not None:
                self._ctx.unsupported_tracker_hashes.append(url_item.infohash)
            return None

        # The service parses the release URL by tracker and adds it to
        # qBittorrent, returning the add status and a display name (the
        # client's name, or the release title scraped from the source
        # page as a fallback). A preview run simulates the add. An expected
        # external failure (tracker or qBittorrent down/erroring) is contained
        # here to ONE warning - no traceback - so the loop moves on and
        # grab_and_cache leaves the title uncached for a retry next run.
        try:
            result = self._torrents.add(item=url_item, preview=self._is_preview())
        except GRAB_FAILURES as e:
            self.log_fmt.detail(
                "failed",
                f"could not grab {url}: {e}; will retry next run",
                value_style="yellow",
                level=logging.WARNING,
            )
            self._grab_failed_groups.append(srg)
            return None

        if result.outcome is AddOutcome.ADDED:
            # Record the grab for the end-of-run summary. Prefer the
            # release's own parsed file list (precise for multi-cour /
            # per-torrent grabs); fall back to the entry-level coverage we
            # mapped from the Arr so the summary's "files" is never blank
            # when a release's filenames couldn't be parsed (e.g. an OVA).
            coverage_str = _coverage.coverage_string(url_item.episodes) or self._ctx.current_coverage
            self._ctx.stats.added.append(
                GrabRecord(
                    title=self._ctx.current_title,
                    coverage=coverage_str,
                    url=self._ctx.current_url,
                    name=result.name,
                    group=srg,
                ),
            )

        # Persist the durable pending-import record on BOTH a fresh add and an
        # already-present torrent. ALREADY_ADDED here means "in the client from a
        # prior run, still downloading / not yet imported": we only reach this add
        # when the planner flagged a download, and the genuine "you already own it"
        # case is the any_to_download=False branch, which never calls add_torrent.
        # So the end-of-run monitor must wait on it too. Appending to
        # _ctx.pending_imports marks the infohash a this-run grab, so the per-series
        # snapshot / reconcile / tally skip it (no double-report, no early drop).
        if result.outcome in (AddOutcome.ADDED, AddOutcome.ALREADY_ADDED):
            self._register_pending_import(url_item, pending_seeds)
            return ReleaseOutcome(outcome=result.outcome, name=result.name, group=srg)

        return None

    def _register_pending_import(
        self,
        url_item: SeadexUrlItem,
        pending_seeds: dict[str, PendingImport] | None,
    ) -> None:
        """Finalize the durable :class:`PendingImport` for a grabbed/present release.

        Only on a real (non-preview) add of a release we hold a seed for, keyed by
        infohash so a re-add overwrites and a verified import deletes. The in-memory
        copy rides the run context for the fast end-of-run blocking pass and marks
        the infohash a this-run grab (excluded from the carried-over snapshot /
        reconcile / tally).

        Args:
            url_item (SeadexUrlItem): The release just handed to the client; its
                ``infohash`` keys the seed and the durable store.
            pending_seeds (dict[str, PendingImport] | None): The Sonarr strategy's
                ``infohash -> PendingImport`` seeds for this id (None for Radarr).
        """

        if (
            self._ctx.import_wait_mode is not ImportWaitMode.OFF
            and not self._is_preview()
            and url_item.infohash
            and pending_seeds
            and url_item.infohash in pending_seeds
        ):
            pending = pending_seeds[url_item.infohash]
            self.cache_store.put_pending(
                self._ctx.arr,
                url_item.infohash,
                pending.to_json(),
            )
            self._ctx.pending_imports.append(pending)

    def _needs_action(self, groups: list[str], reason: str, kind: NeedsActionKind) -> NeedsActionRecord:
        """A needs-action record for the current title: title/coverage/url come from
        the per-title context, the caller supplies the skipped groups, the display
        reason, and the machine-readable kind the summary's guidance gates on."""

        return NeedsActionRecord(
            title=self._ctx.current_title,
            coverage=self._ctx.current_coverage,
            group=", ".join(dict.fromkeys(groups)),
            url=self._ctx.current_url,
            reason=reason,
            kind=kind,
        )

    def _should_cache_as_done(self, *, cap_reached: bool, added_this_title: int, grab_failed: bool) -> bool:
        """Whether this title's outcome may be cached as done.

        Cache only when something was grabbed, or nothing was skipped: a
        private-only OR unsupported-tracker skip that left nothing grabbed keeps
        the title uncached, so it's re-checked once a public release / parser /
        config change lands. Three vetoes block the cache even on a partial
        grab: the run-wide cap (it can leave this title's later urls
        unattempted), a fallback hold, and a contained grab failure (so the
        next run retries).
        """

        # A non-interactive fallback-mode private hold means the fallback COULDN'T
        # cover these files: never cache the title so every run re-checks and
        # resurfaces it. Warn mode and interactive picks keep the plain gate.
        fallback_hold = (
            self._ctx.private_only_skipped
            and self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK
            and not self._config.advanced.interactive
        )
        return (
            not cap_reached
            and not fallback_hold
            and not grab_failed
            and (added_this_title > 0 or not (self._ctx.private_only_skipped or self._ctx.unsupported_tracker_skipped))
        )

    def grab_and_cache(self, req: GrabRequest) -> bool:
        """Shared per-id tail: add torrents, notify, then cache the outcome

        Identical across both Arrs once the (Arr-specific) ``seadex_dict`` and
        release-group info have been resolved (bundled into ``req``). Returns True
        only when max_torrents_to_add has been reached (after the needs-action
        tail has recorded this title's summary row; the engine's single finalize
        site does the save + summary), so the caller stops the whole run;
        otherwise False (move to the next id).
        """

        # Reset the per-title grab-failure note (set in _add_one_url, read below).
        self._grab_failed_groups = []

        # Check the release groups are matching, and get a bespoke list of torrents
        any_to_download = self._planner.get_any_to_download(req.seadex_dict)

        # Capture the running total before the add block so we can tell whether
        # THIS title actually grabbed anything
        torrents_before = self._ctx.torrents_added

        # Set when _grab hits max_torrents_to_add: the needs-action tail below
        # still runs (a contained grab failure on this title must land its summary
        # row even at the cap), but the per-title cache update is skipped - the cap
        # can stop the url loop mid-title - and True stops the run (the engine's
        # single finalize site does the actual cache save).
        cap_reached = False

        if not any_to_download:
            if not self._ctx.private_only_skipped:
                self._ctx.stats.up_to_date += 1
                self.log_fmt.detail(
                    "status",
                    "already have the recommended release",
                    value_style="blue",
                )
        else:
            cap_reached = self._grab(req)

        # Work out whether THIS title actually grabbed anything
        added_this_title = self._ctx.torrents_added - torrents_before

        # A contained grab failure (tracker/client down) means a release this
        # title should have is missing: never cache - even on a partial grab - so
        # the next run retries (the completed add dedups). Also read by the
        # needs-action chain below.
        grab_failed = bool(self._grab_failed_groups)

        # A cap-reached or veto-held title is never cached, but still falls
        # through to the needs-action classification below.
        if self._should_cache_as_done(
            cap_reached=cap_reached,
            added_this_title=added_this_title,
            grab_failed=grab_failed,
        ):
            # A mixed title (grabbed + unsupported-tracker skip) is cached, but the
            # skipped hashes are excluded so the release is re-considered on the
            # entry's next update once a parser lands. Private-only hashes are
            # deliberately NOT excluded: private releases are never grabbed, so
            # their quiet suppression is the intended behavior.
            skipped = set(self._ctx.unsupported_tracker_hashes)
            cacheable = [h for h in req.torrent_hashes if h is None or h not in skipped]
            # The fallback-satisfied marker: a fallback grab or the owned-fallback
            # soft-skip. Always written - the partial-merge upsert would otherwise
            # preserve a stale True after a later genuine grab.
            fallback_satisfied = self._ctx.fallback_covered or any(
                u.is_fallback and u.download for rg_item in req.seadex_dict.values() for u in rg_item.urls.values()
            )
            req.cache_details["torrent_hashes"] = cacheable
            req.cache_details["fallback_satisfied"] = fallback_satisfied
            self.cache_store.update_cache(
                self._ctx.arr,
                req.al_id,
                req.cache_details,
            )
        # A release was skipped for a reason outside the user's control (and either
        # nothing was added or a fallback hold / grab failure blocks the cache):
        # surface ONE needs-action reason for the title (private-only wins) so it
        # shows in the summary. In fallback mode the hold is a fallback that couldn't (no public
        # alternative covered the missing files) or wouldn't (the user's own
        # interactive private pick, or an owned-at-stale-size pick a fallback must
        # not replace) fall back - either way the tip must not suggest the
        # fallback already on.
        elif self._ctx.private_only_skipped:
            if self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK:
                if self._config.advanced.interactive:
                    reason = "hand-picked private release; private releases not supported"
                    kind = NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK
                elif self._ctx.private_only_stale_held:
                    # One row per title: the stale bit wins over a coexisting
                    # plain hold (self-correcting across runs; never cached).
                    reason = (
                        "private-only release; your copy is outdated (its file size no longer matches) "
                        "and only a fallback covers it"
                    )
                    kind = NeedsActionKind.PRIVATE_ONLY_STALE
                else:
                    reason = "private-only release; no public alternative covers these files"
                    kind = NeedsActionKind.PRIVATE_ONLY_NO_FALLBACK
            else:
                reason, kind = (
                    "private-only release; private releases not supported",
                    NeedsActionKind.PRIVATE_ONLY,
                )
            self._ctx.stats.needs_action.append(
                self._needs_action(self._ctx.private_only_groups, reason, kind),
            )
        elif self._ctx.unsupported_tracker_skipped:
            self._ctx.stats.needs_action.append(
                self._needs_action(
                    self._ctx.unsupported_tracker_groups,
                    "tracker not yet supported; grab manually",
                    NeedsActionKind.UNSUPPORTED_TRACKER,
                ),
            )
        elif grab_failed:
            # No user action needed - the warning named the failure and the
            # uncached title retries next run - but the summary must say why the
            # title is neither added nor up to date.
            self._ctx.stats.needs_action.append(
                self._needs_action(
                    self._grab_failed_groups,
                    "grab failed; will retry next run",
                    NeedsActionKind.GRAB_FAILED,
                ),
            )

        # Cap reached: stop the run now that the summary rows are recorded (no
        # point throttling a run that's over).
        if cap_reached:
            return True

        # Add in a wait, if required
        time.sleep(self._config.advanced.sleep_time)

        return False

    def _grab(self, req: GrabRequest) -> bool:
        """Add this title's torrents, notify, and honour the run-wide cap.

        Runs only when there's something to download. Returns True once
        max_torrents_to_add has been reached (the cap notice is logged here; the
        engine's finalize site does the cache save) so the caller stops the whole
        run; otherwise False.
        """

        # Resolve the AniList art (cover thumbnail + wide banner, the same
        # media node once cached) up front so the network call keeps its
        # position in the run's request ordering, even though it's only used
        # in the push below.
        anilist_thumb = self._anilist.thumb(req.al_id)
        anilist_banner = self._anilist.banner(req.al_id)

        # Add torrents to qBittorrent. add_torrent runs even in a preview
        # (no client / dry run): the service simulates the add, while the
        # download-flag, private-release and tracker filters still apply, so
        # only releases that would actually be grabbed are counted.
        n_torrents_added, results = self.add_torrent(
            torrent_dict=req.seadex_dict,
            pending_seeds=req.pending_seeds,
        )

        # Log the action block now the outcome is known, so the status reads
        # "adding" only when something was actually grabbed, "would add" on a
        # preview, else "already downloading" - the pick is in the client from a
        # prior run, still downloading.
        self._reporter.log_seadex_action(
            req.seadex_dict,
            results,
            dry_run=self._is_preview(),
            monitor_active=(self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._is_preview()),
        )

        # Push a message to Discord if we've added anything (never on a
        # preview - it's an outward notification). Built after the add so the
        # embed can label each group with what actually happened.
        if self._notifier.enabled and n_torrents_added > 0 and not self._is_preview():
            self._notifier.push_grab(
                GrabNotice(
                    arr=self._ctx.arr,
                    arr_title=req.item_title,
                    al_title=req.anilist_title,
                    entry=req.entry,
                    thumb_url=anilist_thumb,
                    banner_url=anilist_banner,
                    release_group=req.release_group,
                    seadex_dict=req.seadex_dict,
                    results=results,
                    failed_groups=frozenset(self._grab_failed_groups),
                    coverage=req.coverage,
                ),
            )

        cap = self._config.advanced.max_torrents_to_add
        if cap is not None and self._ctx.torrents_added >= cap:
            self._reporter.log_max_torrents_added(cap)
            # Cap reached: signal the run to stop with a pure bool. run_sync breaks
            # the scan and runs the single _finalize_run site (so the blocking/hybrid
            # pass still imports this run's records before the save + summary).
            return True

        return False
