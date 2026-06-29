import copy
import logging

from seadex import EntryRecord

from .cache import CacheStore
from .config import PRIVATE_TRACKERS, AppConfig
from .log import LogFormatter, indent_string
from .planner import DownloadPlanner
from .reporter import RunContext
from .seadex_types import (
    ArrReleaseDict,
    SeadexDict,
    SeadexReleaseGroupItem,
    SeadexUrlItem,
    SonarrEpisode,
)


class SeadexReleaseFilter:
    """Turns a SeaDex ``EntryRecord`` into the run's filtered/ranked release dict.

    Owns no per-run caches; it binds the run :class:`RunContext` (``begin_run``)
    only so ``filter_downloads`` can stamp the public_only skip flags the grab tail
    later reads. The engine keeps same-named thin delegators (``get_seadex_dict`` /
    ``filter_seadex_interactive`` / ``filter_seadex_downloads``) forwarding here, so
    the strategy<->services contract is unchanged.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        planner: DownloadPlanner,
        cache_store: CacheStore,
        logger: logging.Logger,
        log_fmt: LogFormatter,
        ctx: RunContext,
    ) -> None:
        self._config = config
        self._planner = planner
        self.cache_store = cache_store
        self.logger = logger
        self.log_fmt = log_fmt
        # Seeded with the engine's placeholder ctx; rebound each run via begin_run
        # (the same object the engine holds, so a write here is seen by the grab tail).
        self._ctx = ctx

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the run context ``filter_downloads`` stamps public_only flags onto."""

        self._ctx = ctx

    def build(self, sd_entry: EntryRecord) -> SeadexDict:
        """Parse and filter a SeaDex entry into the run's release dict.

        Args:
            sd_entry: SeaDex API query
        """

        # The torrent records are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags. Casefold both sides (config strings vs the seadex Tag
        # str-enum's canonical case) so a natural-case rule like "dolby vision" matches.
        ignore_tags = {tag.casefold() for tag in self._config.seadex.ignore_tags}
        final_torrent_list = [t for t in sd_entry.torrents if ignore_tags.isdisjoint(tag.casefold() for tag in t.tags)]

        # Filter down by allowed trackers
        final_torrent_list = [t for t in final_torrent_list if t.tracker.casefold() in self._config.seadex.trackers]

        # Find the 'best'-tagged releases up front: we narrow to them first (when
        # want_best is set and any exist), then apply the audio-preference filter
        # on that narrowed set below.
        best_torrents = [t for t in final_torrent_list if t.is_best]
        any_best = len(best_torrents) > 0

        # Narrow to 'best' releases when any exist
        if self._config.seadex.want_best and any_best:
            candidates = best_torrents
        else:
            candidates = final_torrent_list

        # Prefer dual-audio releases, but only when at least one exists
        if self._config.seadex.prefer_dual_audio:
            duals = [t for t in candidates if t.is_dual_audio]
            if len(duals) > 0:
                candidates = duals
        # Otherwise prefer non-dual-audio
        else:
            non_duals = [t for t in candidates if not t.is_dual_audio]
            if len(non_duals) > 0:
                candidates = non_duals

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
                is_public=t.tracker.is_public() and t.tracker.casefold() not in PRIVATE_TRACKERS,
                is_dual_audio=t.is_dual_audio,
                hash=t.infohash,
                download=False,
            )

        # If we only want public releases, then within each release group drop
        # any private URLs, so long as that group also has a public option. We
        # deliberately do this per-group rather than across the whole list: a
        # group that only has a private URL is kept for now and only filtered
        # out later if the Arr doesn't already have a matching download (see
        # reduce_overlapping_downloads)
        if self._config.seadex.public_only:
            for release_group_item in seadex_release_groups.values():
                urls = release_group_item.urls
                has_public = any(u.is_public for u in urls.values())
                if has_public:
                    release_group_item.urls = {url: u for url, u in urls.items() if u.is_public}

        return seadex_release_groups

    def interactive_pick(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        """If multiple matches are found, let the user filter them interactively.

        Args:
            seadex_dict: Dictionary of SeaDex releases
            sd_entry: SeaDex entry
        """

        self.logger.warning("Multiple releases found - pick which to grab")
        self.logger.info(
            indent_string("SeaDex notes:"),
        )

        notes = sd_entry.notes.split("\n")
        for n in notes:
            self.logger.warning(
                indent_string(n),
            )
        self.logger.warning(
            indent_string(""),
        )

        all_srgs = list(seadex_dict.keys())
        for s_i, s in enumerate(all_srgs):
            self.logger.warning(
                indent_string(f"[{s_i}]: {s}"),
            )

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
                    self.logger.warning(
                        indent_string(f"Skipping invalid selection: {srg_idx!r}"),
                    )
                    continue
                seadex_dict_filtered[srg] = copy.deepcopy(seadex_dict[srg])

            seadex_dict = seadex_dict_filtered

        return seadex_dict

    def filter_downloads(
        self,
        al_id: int,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> tuple[list[str | None], SeadexDict]:
        """Flip the switch on whether we're downloading each torrent or not.

        Thin orchestrator seam over the :class:`DownloadPlanner`: pass it the
        entry's cached hashes, then apply the plan's private-only skip outcome back
        onto the run context the grab/cache tail still reads (the SkipNotice log
        lines, the public_only_skipped flag, and the skipped group names).

        Args:
            al_id: AniList ID
            seadex_dict: Dictionary of SeaDex releases
            arr_release_dict: Dictionary of arr release properties
            ep_list: List of episodes. Defaults to None
        """

        arr = self._ctx.arr
        result = self._planner.plan(
            seadex_dict=seadex_dict,
            arr=arr,
            arr_release_dict=arr_release_dict,
            cached_hashes=self.cache_store.torrent_hashes(arr, al_id),
            ep_list=ep_list,
        )

        # The planner reports what to log rather than logging it; render each
        # private-only skip exactly as the inline call used to.
        for notice in result.skip_notices:
            self.log_fmt.detail(
                "skipped",
                f"{', '.join(notice.groups)} {notice.reason}",
                value_style="yellow",
                level=notice.level,
            )

        # Carry the skip flag/groups onto the run context (reset per title in the
        # prologue; add_torrent may append more before grab_and_cache reads them).
        if result.public_only_skipped:
            self._ctx.public_only_skipped = True
            self._ctx.public_only_groups.extend(result.public_only_groups)

        return result.torrent_hashes, result.seadex_dict
