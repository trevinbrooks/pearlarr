"""The grab "produce" side: add torrents, register pending records, write cache.

Extracted from :class:`~.seadex_arr.SeaDexArr`. ``GrabPipeline`` owns the per-id
grab tail both strategies funnel into - add the recommended release(s) to
qBittorrent, persist the durable :class:`PendingImport` records the end-of-run
monitor waits on, notify, and write the cache outcome. It returns a pure bool
(cap-reached) and never calls back into the engine; the engine keeps a thin
``grab_and_cache`` delegator so the strategy<->engine contract is unchanged.

Binds the run :class:`RunContext` via :meth:`begin_run` (the same object the
engine holds), so the grab bookkeeping the run summary reads stays in sync.
"""

import logging
import time
from dataclasses import dataclass

import qbittorrentapi

from . import coverage as _coverage
from .anilist_gateway import AniListGateway
from .cache import AbstractCacheStore, CacheRecord
from .config import AppConfig
from .log import LogFormatter
from .manual_import import ImportWaitMode, PendingImport
from .notify import Notifier
from .planner import DownloadPlanner
from .reporter import (
    GrabRecord,
    NeedsActionKind,
    NeedsActionRecord,
    RunContext,
    RunReporter,
    is_preview,
)
from .seadex_types import SeadexDict, SeadexUrlItem
from .torrents import PARSEABLE_TRACKERS, AddOutcome, ReleaseOutcome, TorrentService


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
    sd_url: str
    seadex_dict: SeadexDict
    torrent_hashes: list[str | None]
    cache_details: CacheRecord
    release_group: list[str | None] | None
    pending_seeds: dict[str, PendingImport] | None = None


class GrabPipeline:
    """Adds the recommended release(s), registers pending records, writes the cache.

    Constructed once per run in :class:`~.seadex_arr.SeaDexArr` from the unpacked
    deps + the placeholder ctx; :meth:`begin_run` rebinds the ctx each run. The
    engine's ``grab_and_cache`` delegates here; ``_grab`` returns a pure bool
    (cap-reached) so the engine owns the single finalize site.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        planner: DownloadPlanner,
        cache_store: AbstractCacheStore,
        torrents: TorrentService,
        anilist: AniListGateway,
        notifier: Notifier,
        reporter: RunReporter,
        log_fmt: LogFormatter,
        qbit: qbittorrentapi.Client | None,
        ctx: RunContext,
    ) -> None:
        self._config = config
        self._planner = planner
        self.cache_store = cache_store
        self._torrents = torrents
        self._anilist = anilist
        self._notifier = notifier
        self._reporter = reporter
        self.log_fmt = log_fmt
        self.qbit = qbit
        # Seeded with the engine's placeholder ctx; rebound each run via begin_run
        # (the same object the engine holds, so the grab bookkeeping stays in sync).
        self._ctx = ctx

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

        The per-release outcome lines (added / kept) are NOT logged here; this
        returns them so the caller (log_seadex_action) can print the whole block
        in order with a status that reflects what actually happened - "adding" if
        anything was grabbed, "keeping" if every recommended release was already
        present. The "skipped" warnings (private-only, unselected tracker) are
        still logged inline, as they're independent of that status.

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
            for url, url_item in srg_item.urls.items():
                add_result = self._add_one_url(
                    srg,
                    url,
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
        url: str,
        url_item: SeadexUrlItem,
        pending_seeds: dict[str, PendingImport] | None = None,
    ) -> ReleaseOutcome | None:
        """Resolve a single SeaDex url to an add outcome (or ``None`` to skip).

        Returns ``None`` for a release that's filtered out (not flagged for
        download, private-only under ``public_only``, or an unselected tracker)
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

        tracker = url_item.tracker

        if self._config.seadex.public_only and not url_item.is_public:
            self.log_fmt.detail(
                "skipped",
                f"{tracker} private-only (public_only on)",
                value_style="yellow",
                level=logging.WARNING,
            )
            self._ctx.public_only_skipped = True
            self._ctx.public_only_groups.append(srg)
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
            if url_item.hash is not None:
                self._ctx.unsupported_tracker_hashes.append(url_item.hash)
            return None

        # The service parses the release URL by tracker and adds it to
        # qBittorrent, returning the add status and a display name (the
        # client's name, or the release title scraped from the source
        # page as a fallback). A preview run simulates the add.
        success, torrent_name = self._torrents.add(
            url=url,
            tracker=tracker,
            torrent_hash=url_item.hash,
            preview=self._is_preview(),
        )

        if success is AddOutcome.ADDED:
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
                    name=torrent_name,
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
        if success in (AddOutcome.ADDED, AddOutcome.ALREADY_ADDED):
            self._register_pending_import(url_item, pending_seeds)
            return ReleaseOutcome(outcome=success, name=torrent_name, group=srg)

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
                ``hash`` keys the seed and the durable store.
            pending_seeds (dict[str, PendingImport] | None): The Sonarr strategy's
                ``infohash -> PendingImport`` seeds for this id (None for Radarr).
        """

        if (
            self._ctx.import_wait_mode is not ImportWaitMode.OFF
            and not self._is_preview()
            and url_item.hash
            and pending_seeds
            and url_item.hash in pending_seeds
        ):
            pending = pending_seeds[url_item.hash]
            self.cache_store.put_pending(
                self._ctx.arr,
                url_item.hash,
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

    def grab_and_cache(self, req: GrabRequest) -> bool:
        """Shared per-id tail: add torrents, notify, then cache the outcome

        Identical across both Arrs once the (Arr-specific) ``seadex_dict`` and
        release-group info have been resolved (bundled into ``req``). Returns True
        only when max_torrents_to_add has been reached (cache saved and summary
        logged), so the caller stops the whole run; otherwise False (move to the
        next id).
        """

        # Check the release groups are matching, and get a bespoke list of torrents
        any_to_download = self._planner.get_any_to_download(req.seadex_dict)

        # Capture the running total before the add block so we can tell whether
        # THIS title actually grabbed anything
        torrents_before = self._ctx.torrents_added

        if not any_to_download:
            if not self._ctx.public_only_skipped:
                self._ctx.stats.up_to_date += 1
                self.log_fmt.detail(
                    "status",
                    "already have the recommended release",
                    value_style="blue",
                )
        elif self._grab(req):
            # max_torrents_to_add reached: _grab logged the cap and the run stops
            # here (the per-title cache update below is deliberately skipped); the
            # engine's single finalize site does the actual cache save.
            return True

        # Work out whether THIS title actually grabbed anything
        added_this_title = self._ctx.torrents_added - torrents_before

        # Cache the title as done only when something was grabbed, or nothing was
        # skipped. A private-only OR unsupported-tracker skip that left nothing
        # grabbed keeps it uncached, so it's re-checked once a public release /
        # parser / config change lands.
        if added_this_title > 0 or not (self._ctx.public_only_skipped or self._ctx.unsupported_tracker_skipped):
            # A mixed title (grabbed + unsupported-tracker skip) is cached, but the
            # skipped hashes are excluded so the release is re-considered on the
            # entry's next update once a parser lands. Private-only hashes are
            # deliberately NOT excluded: public_only is a user-configured exclusion,
            # so its quiet suppression is the intended behavior.
            skipped = set(self._ctx.unsupported_tracker_hashes)
            cacheable = [h for h in req.torrent_hashes if h is None or h not in skipped]
            req.cache_details.update({"torrent_hashes": cacheable})
            self.cache_store.update_cache(
                self._ctx.arr,
                req.al_id,
                req.cache_details,
            )
        # Nothing added, but a release was skipped for a reason outside the user's
        # control: surface ONE needs-action reason for the title (private-only wins)
        # so it isn't cached as done and shows in the summary.
        elif self._ctx.public_only_skipped:
            self._ctx.stats.needs_action.append(
                self._needs_action(
                    self._ctx.public_only_groups,
                    "private-only release; public_only on",
                    NeedsActionKind.PRIVATE_ONLY,
                ),
            )
        elif self._ctx.unsupported_tracker_skipped:
            self._ctx.stats.needs_action.append(
                self._needs_action(
                    self._ctx.unsupported_tracker_groups,
                    "unsupported tracker; no parser yet",
                    NeedsActionKind.UNSUPPORTED_TRACKER,
                ),
            )

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

        # Resolve the AniList cover thumbnail (via the gateway) and build the
        # Discord embed fields for the grab. The thumb lookup is done up front
        # to preserve ordering even though it's only used in the push below.
        anilist_thumb = self._anilist.thumb(req.al_id)
        fields = self._notifier.build_fields(
            arr=self._ctx.arr,
            release_group=req.release_group,
            seadex_dict=req.seadex_dict,
        )

        # Add torrents to qBittorrent. add_torrent runs even in a preview
        # (no client / dry run): the service simulates the add, while the
        # download-flag, public_only and tracker filters still apply, so only
        # releases that would actually be grabbed are counted.
        n_torrents_added, results = self.add_torrent(
            torrent_dict=req.seadex_dict,
            pending_seeds=req.pending_seeds,
        )

        # Log the action block now the outcome is known, so the status reads
        # "adding" only when something was actually grabbed, "already downloading"
        # when the pick is already in the client mid-download, else "keeping".
        self._reporter.log_seadex_action(
            req.seadex_dict,
            results,
            dry_run=self._is_preview(),
            monitor_active=(self._ctx.import_wait_mode is not ImportWaitMode.OFF and not self._is_preview()),
        )

        # Push a message to Discord if we've added anything (never on a
        # preview - it's an outward notification)
        if self._notifier.enabled and n_torrents_added > 0 and not self._is_preview():
            self._notifier.push(
                arr_title=req.item_title,
                al_title=req.anilist_title,
                seadex_url=req.sd_url,
                fields=fields,
                thumb_url=anilist_thumb,
            )

        cap = self._config.advanced.max_torrents_to_add
        if cap is not None and self._ctx.torrents_added >= cap:
            self._reporter.log_max_torrents_added()
            # Cap reached: signal the run to stop with a pure bool. run_sync breaks
            # the scan and runs the single _finalize_run site (so the blocking/hybrid
            # pass still imports this run's records before the save + summary).
            return True

        return False
