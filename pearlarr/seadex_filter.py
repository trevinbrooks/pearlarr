"""The SeaDex release filter: builds the per-entry `seadex_dict` and applies the selection rules."""

import sys
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

from rich.console import Console
from seadex import EntryRecord, TorrentRecord

from .arr_models import ArrFiles, SonarrEpisode
from .arr_run_state import ArrRunState
from .config import PRIVATE_TRACKERS, PrivateReleaseAction
from .console_features import console_of
from .output import Accent, StyledValue, hub_warn
from .output.console_text import indent_string
from .planner import GrabPlan
from .releases import GroupReleases, Release, SeadexDict, carries_ignored_tag, ignore_tag_set

if TYPE_CHECKING:
    # Annotation-only: arr_run_services imports this module at runtime (cycle).
    from .arr_run_services import ArrRunDeps


def _is_public_torrent(torrent: TorrentRecord) -> bool:
    """Whether a torrent is on a public tracker (the run's is_public computation)."""

    return torrent.tracker.is_public() and torrent.tracker.casefold() not in PRIVATE_TRACKERS


def _file_sizes(torrent: TorrentRecord) -> list[int]:
    """A torrent's listed file sizes - the coverage-comparison input and the release's `file_sizes`."""

    return [f.size for f in torrent.files]


def file_size_cover(listings: Iterable[Iterable[int]]) -> Counter[int]:
    """Merge per-torrent file-size multisets into one coverage multiset, elementwise max.

    Byte size is the identity two trackers' torrents of one upload always agree
    on: folder layout differs between trackers, and SeaDex's own per-file naming
    can too, but the bytes do not. Covering n files of one size requires ONE
    torrent genuinely carrying n - never n torrents carrying one each (a `+`
    merge would fake that), and never a set union (which would deflate a real
    n-copy torrent to one).
    """

    cover: Counter[int] = Counter()
    for listing in listings:
        cover |= Counter(listing)
    return cover


class SeadexReleaseFilter:
    """Turns a SeaDex `EntryRecord` into the run's filtered/ranked release dict.

    Owns no per-run caches. It binds the arr run's `ArrRunState` (`bind_arr_run`)
    only so `plan_grabs` can set the private-only skip flags the grab tail
    later reads. `ArrRunServices` keeps same-named thin
    delegators (`get_seadex_dict` / `filter_seadex_interactive` /
    `plan_grabs`) forwarding here, so the strategy<->services
    contract is unchanged.
    """

    def __init__(
        self,
        *,
        deps: "ArrRunDeps",
        state: ArrRunState,
    ) -> None:
        self._config = deps.config
        self._planner = deps.planner
        self.logger = deps.logger
        self._reporter = deps.reporter
        # Started with `ArrRunServices`'s placeholder state, rebound each arr run via
        # bind_arr_run (the same object `ArrRunServices` holds, so a write here is seen
        # by the grab tail).
        self._state = state

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the arr run state `plan_grabs` sets private-only flags on."""

        self._state = state

    def build(self, sd_entry: EntryRecord) -> SeadexDict:
        """Read and filter a SeaDex entry into the run's release dict."""

        # The torrents are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags, casefolded on both sides so a natural-case rule like "dolby vision" matches.
        ignore_tags = ignore_tag_set(self._config.seadex.ignore_tags)
        final_torrent_list = [t for t in sd_entry.torrents if not carries_ignored_tag(t, ignore_tags)]

        # Filter down by allowed trackers
        final_torrent_list = [t for t in final_torrent_list if t.tracker.casefold() in self._config.seadex.trackers]

        # The preferred releases: the audio-preference -> want_best cascade
        candidates = self._narrow_candidates(final_torrent_list)

        # If a preferred private release isn't covered by the public ones, offer the
        # best public fallbacks too (private_releases: fallback).
        candidates, fallback_urls = self._augment_with_public_fallbacks(final_torrent_list, candidates)

        # Pull out release groups, releases, and various other useful info as a
        # dictionary
        seadex_release_groups: SeadexDict = {}
        for t in candidates:
            if t.release_group not in seadex_release_groups:
                seadex_release_groups[t.release_group] = GroupReleases(releases={}, tags=t.tags)

            seadex_release_groups[t.release_group].releases[t.url] = Release(
                url=t.url,
                files=[f.name for f in t.files],
                file_sizes=_file_sizes(t),
                tracker=t.tracker,
                is_public=_is_public_torrent(t),
                is_dual_audio=t.is_dual_audio,
                infohash=t.infohash,
                to_grab=False,
                is_fallback=t.url in fallback_urls,
            )

        self._prune_covered_private(seadex_release_groups)

        return seadex_release_groups

    def _augment_with_public_fallbacks(
        self,
        final_torrent_list: list[TorrentRecord],
        candidates: list[TorrentRecord],
    ) -> tuple[list[TorrentRecord], set[str]]:
        """Offer the best public fallbacks for uncovered private releases (fallback mode).

        Returns the (maybe augmented) candidates and the set of fallback urls.
        Outside `private_releases: fallback` the candidates come back unchanged
        with an empty set, so the caller stays branch-free.
        """

        if self._config.seadex.private_releases is not PrivateReleaseAction.FALLBACK:
            return candidates, set()

        # If any preferred private release isn't covered by the public releases'
        # files (per-group when its files are unknown) and fallback was chosen over
        # warn, also offer the best PUBLIC fallbacks (same cascade over the public
        # torrents not already chosen). The private releases stay in, both so the
        # planner can see the Arr already has one, and so it can warn when nothing
        # public covers the same files.
        group_has_public: dict[str, bool] = {}
        group_has_blind_public: dict[str, bool] = {}
        for t in candidates:
            rg, is_pub = t.release_group, _is_public_torrent(t)
            group_has_public[rg] = group_has_public.get(rg, False) or is_pub
            # A public candidate with an UNKNOWN fileset: treated per-group as
            # covering (the cross-seed case), mirroring the private side.
            group_has_blind_public[rg] = group_has_blind_public.get(rg, False) or (is_pub and not t.files)
        # File-size multisets, not file paths: trackers lay the same files out
        # differently (one nesting extras in a folder the other lists flat), and a
        # path comparison would misread a covered private release as uncovered,
        # pulling in spurious fallbacks.
        public_cover = file_size_cover(_file_sizes(t) for t in candidates if _is_public_torrent(t))

        def needs_fallback(t: TorrentRecord) -> bool:
            """A private candidate needs a public fallback unless the public candidates cover its files.

            An unknown fileset on either side degrades to the per-group gate.
            """

            if _is_public_torrent(t):
                return False
            if not t.files:
                return not group_has_public[t.release_group]
            if group_has_blind_public[t.release_group]:
                return False
            return not Counter(_file_sizes(t)) <= public_cover

        if not any(needs_fallback(t) for t in candidates):
            return candidates, set()

        preferred_urls = {t.url for t in candidates}
        fallbacks = self._narrow_candidates(
            [t for t in final_torrent_list if _is_public_torrent(t) and t.url not in preferred_urls],
        )
        fallback_urls = {t.url for t in fallbacks}
        return candidates + fallbacks, fallback_urls

    def _prune_covered_private(self, seadex_release_groups: SeadexDict) -> None:
        """Drop each group's private releases whose files its public releases already cover.

        Private releases are never grabbed, so within each release group drop
        any private release whose files the group's public releases cover (an
        unknown fileset counts as covered - the plain cross-seed case). We
        deliberately do this per-group rather than across the whole list: a
        private release with uncovered files is kept for now and only filtered out
        later if the Arr doesn't already have a matching copy (see
        reduce_overlapping_grabs)
        """

        for group_releases in seadex_release_groups.values():
            urls = group_releases.releases
            # The gate is a public release's existence, not the cover's truthiness:
            # an empty-fileset public release covers (cross-seed) while merging nothing in.
            if not any(u.is_public for u in urls.values()):
                continue
            group_public_cover = file_size_cover(u.file_sizes for u in urls.values() if u.is_public)
            group_releases.releases = {
                url: u for url, u in urls.items() if u.is_public or not Counter(u.file_sizes) <= group_public_cover
            }

    def _narrow_candidates(self, torrents: list[TorrentRecord]) -> list[TorrentRecord]:
        """Narrow one candidate pool via the audio-preference -> want_best cascade.

        Each cut only applies when it leaves at least one torrent: narrow to the
        preferred audio (dual when `prefer_dual_audio`, else single), then to
        'best'-tagged releases within it (when `want_best`). Audio cuts first so
        a dual-audio alternative outranks a single-audio best. Bump
        `SELECTION_RULES_VERSION` when this order changes.
        """

        if self._config.seadex.prefer_dual_audio:
            preferred_audio = [t for t in torrents if t.is_dual_audio]
        else:
            preferred_audio = [t for t in torrents if not t.is_dual_audio]
        if preferred_audio:
            torrents = preferred_audio

        best = [t for t in torrents if t.is_best]
        return best if self._config.seadex.want_best and best else torrents

    def interactive_pick(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """If multiple release groups are found, filter them interactively."""

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
            # Mark the fallbacks (private_releases: fallback) so the pick is
            # informed. A picked private release is skipped later.
            fallback_tag = " (public fallback)" if any(u.is_fallback for u in seadex_dict[s].releases.values()) else ""
            say(indent_string(f"[{s_i}]: {s}{fallback_tag}"))

        raw = input(
            "Which release group(s)? Enter one number, a comma-separated list, or leave blank for all: ",
        )
        selections = [tok for tok in raw.split(",") if tok]

        # If we have some selections, narrow down
        if selections:
            seadex_dict_filtered: SeadexDict = {}
            for srg_idx in selections:
                try:
                    srg = all_srgs[int(srg_idx)]
                except (ValueError, IndexError):
                    # ValueError: a non-numeric token (a typo). IndexError: out of
                    # range. Skip the bad token instead of abandoning the whole title.
                    hub_warn(f"Skipping invalid selection {srg_idx!r}")
                    continue
                # The caller discards the original dict, so hand the group's releases over by reference.
                seadex_dict_filtered[srg] = seadex_dict[srg]

            # Every token was invalid (blank input means "all" and never gets
            # here), so the title proceeds with zero releases - say so.
            if not seadex_dict_filtered:
                hub_warn("No valid selection - skipping this title")

            seadex_dict = seadex_dict_filtered

        return seadex_dict

    def plan_grabs(
        self,
        seadex_dict: SeadexDict,
        arr_files: ArrFiles,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> GrabPlan:
        """Plan which releases to grab.

        Thin orchestrator over the `GrabPlanner`: run it, then apply the
        plan's private-only skip outcome back onto the arr run state the
        grab/cache tail reads (the `PrivateOnlyNotice` log lines, the
        private_only_skipped flag, and the skipped group names). The plan
        itself is returned whole, so a field added to `GrabPlan` reaches the
        strategies without a hand-synced mirror type.
        """

        result = self._planner.plan(
            seadex_dict=seadex_dict,
            arr_files=arr_files,
            ep_list=ep_list,
        )

        # The planner reports what to log rather than logging it, so each
        # private-only skip is posted here.
        for notice in result.skips.notices:
            self._reporter.detail(
                "skipped",
                StyledValue(f"{', '.join(notice.groups)} {notice.reason}", Accent.CAUTION),
                severity=notice.severity,
            )

        # Carry the skip flags/groups onto the arr run state (reset per title in
        # begin_check. grab_releases may append more before grab_and_cache reads them).
        self._state.per_title.add_private_only_notices(result.skips)

        return result
