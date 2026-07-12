"""The SeaDex release filter: builds the per-entry `seadex_dict` and applies the selection rules."""

import copy
import sys
from typing import TYPE_CHECKING, NamedTuple

from rich.console import Console
from seadex import EntryRecord, TorrentRecord

from .config import PRIVATE_TRACKERS, PrivateReleaseAction
from .console_caps import console_of
from .log import indent_string
from .output import Accent, StyledValue, hub_warn
from .reporter import RunContext
from .seadex_types import (
    ArrReleaseDict,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)

if TYPE_CHECKING:
    # Annotation-only: run_services imports this module at runtime (cycle).
    from .run_services import RunDeps


class FilterResult(NamedTuple):
    """The applied download plan, as the strategies consume it.

    The strategy-facing slice of the planner's `PlanResult`:
    the unique hashes to remember in the cache record (None for a hash-less
    private torrent) plus the same `seadex_dict` annotated in place with
    per-url `download` flags. The plan's skip outcome is not carried - it is
    folded onto the run context here, in `filter_downloads`.
    """

    torrent_hashes: list[str | None]
    seadex_dict: SeadexDict


def _is_public_torrent(torrent: TorrentRecord) -> bool:
    """Whether a torrent is on a public tracker (the run's is_public computation).

    `PRIVATE_TRACKERS` is read at call time, so the characterization tests can
    pin it on this module.
    """

    return torrent.tracker.is_public() and torrent.tracker.casefold() not in PRIVATE_TRACKERS


class SeadexReleaseFilter:
    """Turns a SeaDex `EntryRecord` into the run's filtered/ranked release dict.

    Owns no per-run caches; it binds the run `RunContext` (`begin_run`)
    only so `filter_downloads` can stamp the private-only skip flags the grab tail
    later reads. `RunServices` keeps same-named thin
    delegators (`get_seadex_dict` / `filter_seadex_interactive` /
    `filter_seadex_downloads`) forwarding here, so the strategy<->services
    contract is unchanged.
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
        self.logger = deps.logger
        self._reporter = deps.reporter
        # Seeded with the services hub's placeholder ctx; rebound each run via
        # begin_run (the same object the hub holds, so a write here is seen by the
        # grab tail).
        self._ctx = ctx

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the run context `filter_downloads` stamps private-only flags onto."""

        self._ctx = ctx

    def build(self, sd_entry: EntryRecord) -> SeadexDict:
        """Parse and filter a SeaDex entry into the run's release dict."""

        # The torrent records are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags. Casefold both sides (config strings vs the seadex Tag
        # str-enum's canonical case) so a natural-case rule like "dolby vision" matches.
        ignore_tags = {tag.casefold() for tag in self._config.seadex.ignore_tags}
        final_torrent_list = [t for t in sd_entry.torrents if ignore_tags.isdisjoint(tag.casefold() for tag in t.tags)]

        # Filter down by allowed trackers
        final_torrent_list = [t for t in final_torrent_list if t.tracker.casefold() in self._config.seadex.trackers]

        # The preferred picks: the want_best -> audio-preference cascade
        candidates = self._narrow_candidates(final_torrent_list)

        # If any preferred private pick isn't covered by the public picks' files
        # (per-group when its files are unknown) and the user chose fallback over
        # warn-and-wait, also offer the best PUBLIC alternatives (same cascade
        # over the public torrents not already picked). The private picks stay
        # in, both so the planner can see the Arr already has one, and so it can
        # warn when nothing public covers the same files.
        fallback_urls: set[str] = set()
        if self._config.seadex.private_releases is PrivateReleaseAction.FALLBACK:
            group_has_public: dict[str, bool] = {}
            group_has_blind_public: dict[str, bool] = {}
            for t in candidates:
                rg, is_pub = t.release_group, _is_public_torrent(t)
                group_has_public[rg] = group_has_public.get(rg, False) or is_pub
                # A public candidate with an UNKNOWN fileset: treated per-group as
                # covering (the cross-seed case), mirroring the private side.
                group_has_blind_public[rg] = group_has_blind_public.get(rg, False) or (is_pub and not t.files)
            public_file_names = {f.name for t in candidates if _is_public_torrent(t) for f in t.files}

            def needs_fallback(t: TorrentRecord) -> bool:
                """A private candidate needs a public alternative unless the public candidates cover its files.

                An unknown fileset on either side degrades to the per-group gate.
                """

                if _is_public_torrent(t):
                    return False
                if not t.files:
                    return not group_has_public[t.release_group]
                if group_has_blind_public[t.release_group]:
                    return False
                return not {f.name for f in t.files} <= public_file_names

            if any(needs_fallback(t) for t in candidates):
                preferred_urls = {t.url for t in candidates}
                fallbacks = self._narrow_candidates(
                    [t for t in final_torrent_list if _is_public_torrent(t) and t.url not in preferred_urls],
                )
                fallback_urls = {t.url for t in fallbacks}
                candidates = candidates + fallbacks

        # Pull out release groups, URLs, and various other useful info as a
        # dictionary
        seadex_release_groups: SeadexDict = {}
        for t in candidates:
            if t.release_group not in seadex_release_groups:
                seadex_release_groups[t.release_group] = SeadexReleaseGroupItem(urls={}, tags=t.tags)

            seadex_release_groups[t.release_group].urls[t.url] = SeadexUrlItem(
                url=t.url,
                files=[f.name for f in t.files],
                size=[f.size for f in t.files],
                tracker=t.tracker,
                is_public=_is_public_torrent(t),
                is_dual_audio=t.is_dual_audio,
                infohash=t.infohash,
                download=False,
                is_fallback=t.url in fallback_urls,
            )

        # Private releases are never grabbed, so within each release group drop
        # any private URL whose files the group's public URLs cover (an unknown
        # fileset counts as covered - the plain cross-seed case). We deliberately
        # do this per-group rather than across the whole list: a private URL with
        # uncovered files is kept for now and only filtered out later if the Arr
        # doesn't already have a matching download (see reduce_overlapping_downloads)
        for release_group_item in seadex_release_groups.values():
            urls = release_group_item.urls
            group_public_files = {f for u in urls.values() if u.is_public for f in u.files}
            if any(u.is_public for u in urls.values()):
                release_group_item.urls = {
                    url: u for url, u in urls.items() if u.is_public or not set(u.files) <= group_public_files
                }

        return seadex_release_groups

    def _narrow_candidates(self, torrents: list[TorrentRecord]) -> list[TorrentRecord]:
        """Narrow one candidate pool via the want_best -> audio-preference cascade.

        Each cut only applies when it leaves at least one torrent: narrow to
        'best'-tagged releases (when `want_best`), then to the preferred audio
        (dual when `prefer_dual_audio`, else single).
        """

        best = [t for t in torrents if t.is_best]
        if self._config.seadex.want_best and best:
            torrents = best

        if self._config.seadex.prefer_dual_audio:
            preferred_audio = [t for t in torrents if t.is_dual_audio]
        else:
            preferred_audio = [t for t in torrents if not t.is_dual_audio]
        return preferred_audio if preferred_audio else torrents

    def interactive_pick(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """If multiple matches are found, let the user filter them interactively."""

        # The prompt rows are interactive UI for the input() below, not log
        # events: they render straight to the terminal so they stay visible at
        # any configured log level and never tally into the issues summary.
        console = console_of(self.logger) or Console(file=sys.stdout)

        def say(row: str) -> None:
            # markup=False: the rows carry third-party text (SeaDex notes, group
            # names) - a stray "[" would otherwise raise MarkupError and abort
            # the whole arr run. Nothing here styles via markup anyway.
            console.print(row, highlight=False, soft_wrap=True, markup=False)

        say("Multiple releases found - pick which to grab")
        say(indent_string("SeaDex notes:"))

        notes = sd_entry.notes.split("\n")
        for n in notes:
            say(indent_string(n))
        say("")

        all_srgs = list(seadex_dict.keys())
        for s_i, s in enumerate(all_srgs):
            # Flag the non-preferred public stand-ins (private_releases: fallback)
            # so the pick is informed; a picked private release is refused later.
            fallback_tag = " (public fallback)" if any(u.is_fallback for u in seadex_dict[s].urls.values()) else ""
            say(indent_string(f"[{s_i}]: {s}{fallback_tag}"))

        srgs_to_grab = input(
            "Which release group(s)? Enter one number, a comma-separated list, or leave blank for all: ",
        )

        srgs_to_grab = srgs_to_grab.split(",")

        # Remove any blank entries
        while "" in srgs_to_grab:
            srgs_to_grab.remove("")

        # If we have some selections, parse down
        if len(srgs_to_grab) > 0:
            seadex_dict_filtered = {}
            for srg_idx in srgs_to_grab:
                try:
                    srg = all_srgs[int(srg_idx)]
                except (ValueError, IndexError):
                    # ValueError: a non-numeric entry (a typo); IndexError: out of
                    # range. Skip the bad token instead of abandoning the whole entry.
                    hub_warn(f"Skipping invalid selection {srg_idx!r}")
                    continue
                seadex_dict_filtered[srg] = copy.deepcopy(seadex_dict[srg])

            # Every token was invalid (blank input means "all" and never gets
            # here), so the title proceeds with zero releases - say so.
            if not seadex_dict_filtered:
                hub_warn("No valid selection - skipping this title")

            seadex_dict = seadex_dict_filtered

        return seadex_dict

    def filter_downloads(
        self,
        al_id: int,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> FilterResult:
        """Flip the switch on whether we're downloading each torrent or not.

        Thin orchestrator seam over the `DownloadPlanner`: pass it the
        entry's cached hashes, then apply the plan's private-only skip outcome back
        onto the run context the grab/cache tail still reads (the SkipNotice log
        lines, the private_only_skipped flag, and the skipped group names).
        """

        result = self._planner.plan(
            seadex_dict=seadex_dict,
            arr_release_dict=arr_release_dict,
            cached_hashes=self.cache_store.torrent_hashes(self._ctx.arr, al_id),
            ep_list=ep_list,
        )

        # The planner reports what to log rather than logging it; post each
        # private-only skip exactly as the inline call used to.
        for notice in result.skip_notices:
            self._reporter.detail(
                "skipped",
                StyledValue(f"{', '.join(notice.groups)} {notice.reason}", Accent.CAUTION),
                severity=notice.severity,
            )

        # Carry the skip flag/groups onto the run context (reset per title in the
        # prologue; add_torrent may append more before grab_and_cache reads them).
        if result.private_only_skipped:
            self._ctx.private_only_skipped = True
            self._ctx.private_only_groups.extend(result.private_only_groups)
        if result.private_only_stale_held:
            self._ctx.private_only_stale_held = True
        if result.fallback_covered:
            self._ctx.fallback_covered = True

        return FilterResult(result.torrent_hashes, result.seadex_dict)
