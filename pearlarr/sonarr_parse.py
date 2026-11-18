"""Sonarr `/parse` cache collaborator: SeaDex filenames, read through the parse cache, for the placement.

`SonarrParseCache` owns the grab-time `/parse` of a release's filenames plus
the durable parse cache: cold-cache warm, TTL eviction, and the gather that
pairs each video file with its parse for `place_release`. The cache rows
themselves (`ParseCacheRows`, in `parse_cache`) are bound once on `ArrRunDeps`. The
episode collaborator is handed in. It has the series hash that pins unmatched rows,
and the order of each series, since a row is good only while its series stays in the
order it was stored under. It matches a parse again for a series Sonarr numbers by
another order (`EpisodeSource.aired_parse`). The cache keeps Sonarr's own answer,
never that one.
"""

import concurrent.futures
from collections.abc import Sequence

from .arr_models import ParsedFileInfo
from .arr_run_services import ArrRunDeps
from .grab_placement import ListedFile
from .output.console_text import count_noun
from .parse_cache import ParseFreshness
from .releases import SeadexDict
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import EpisodeSource, fetch_workers
from .sonarr_matching import NoAnswer, parse_without_answer
from .video_files import indexed_video_files


class SonarrParseCache:
    """Owns the grab-time `/parse` and the durable, freshness-checked parse cache, built once per run.

    Reads go through `rows`, so a write staged by `parsed_files` is visible to a later same-run read.
    """

    def __init__(self, deps: ArrRunDeps, sonarr: AbstractSonarrClient, episodes: EpisodeSource) -> None:
        """Bind the shared collaborators, the Sonarr client whose `/parse` the cache fills from, and the episode source.

        The episode source gives the run's series hash, each series' order, and the aired matches of a parse.
        """

        self.sonarr = sonarr
        self._episodes = episodes
        self._config = deps.config
        self.cache_store = deps.cache_store
        self.rows = deps.parse_cache_rows
        self.logger = deps.logger

    def _parse_for(self, f: str, *, window: ParseFreshness) -> ParsedFileInfo | None:
        """One file's parse, read-through the cache: a fresh hit, else Sonarr's answer (cached), else None.

        None is a parse left unread (not cached, re-queried on demand).
        """

        info = self.rows.read(f, window=window)
        if info is not None:
            return info
        info = self.sonarr.parse(f)
        if info is None:
            return None
        # Cache the result (unmatched parses are pinned to the series hash so
        # they aren't re-parsed every run) before acting on it.
        self.rows.write(f, info, window=window)
        if not info.matched_episodes:
            self.logger.debug(f"Sonarr could not parse episode for {f}")
        return info

    def _aired_parse_for(self, f: str, *, window: ParseFreshness, unread: set[int]) -> ParsedFileInfo | None:
        """One file's parse as the placement reads it: `_parse_for`'s, matched as an aired Sonarr matches it.

        Only a parse of a series Sonarr numbers by another order differs from the cached one. None is a parse left
        unread: Sonarr gave none, or an aired Sonarr's answer can't be known and `parse_without_answer` leaves it so.

        `unread` collects the series whose episodes couldn't be read. A failed read isn't kept, so without it every
        other file of such a series would try the read again, and warn again.
        """

        info = self._parse_for(f, window=window)
        if info is None:
            return None
        aired = NoAnswer.EPISODES_UNREAD if info.series_id in unread else self._episodes.aired_parse(info)
        if not isinstance(aired, NoAnswer):
            return aired
        if aired is NoAnswer.SONARR_FAILS:
            self.logger.debug(f"No parse for {f}: two aired episodes of its series share a number it is matched by")
        elif aired is NoAnswer.EPISODES_UNREAD and info.series_id is not None:
            unread.add(info.series_id)
        return parse_without_answer(info, aired)

    def _warm_parse_cache(self, names: Sequence[str], *, window: ParseFreshness) -> None:
        """Concurrently parse the distinct `names` not yet cached, so the gather reads them as hits.

        Only `sonarr.parse` runs in the pool (cache reads and writes stay on this thread). No-op when sequential.
        """

        workers = fetch_workers(self._config)
        if workers <= 1:
            return

        pending = [f for f in names if not self.rows.is_fresh(f, window=window)]

        if len(pending) <= 1:
            return

        def fetch(name: str) -> tuple[str, ParsedFileInfo | None]:
            # A RAISE leaves the parse unread (None: not cached, re-parsed on
            # demand) so one bad file can't abort the concurrent warm sweep.
            try:
                return name, self.sonarr.parse(name)
            except Exception:
                return name, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            results = list(pool.map(fetch, pending))

        for name, result in results:
            if result is None:  # request failed: don't cache an unread parse
                continue
            self.rows.write(name, result, window=window)

    def parsed_files(self, seadex_dict: SeadexDict) -> dict[str, tuple[ListedFile, ...]]:
        """Each release's video files with their sizes and Sonarr parses, keyed by url as the groups key their releases.

        A name goes to Sonarr once, within a run (the same file across overlapping groups) and across
        runs through the durable parse cache, its unmatched rows pinned by the run's series hash and each row of
        a series good only under the order the series was in when the row was stored. A parse of a series Sonarr
        numbers by another order is handed on with the episodes an aired Sonarr matches it to
        (`_aired_parse_for`). Every release gets a value, empty when it lists no video file. Never mutates
        `seadex_dict`.
        """

        # Cutoffs computed once per call (not per file), all anchored to one instant.
        window = ParseFreshness.open(self._episodes.series_hash, self._episodes.series_in_another_order)

        # Evict parse rows aged past that same cutoff so the stored parses stop
        # growing without bound. Staged like the writes below (committed at the
        # run's save point, discarded in a dry run). Only the first call per run finds stale
        # rows, later calls evict nothing.
        evicted = self.cache_store.evict_sonarr_parse(window.matched_cutoff)
        if evicted:
            self.logger.debug(f"Evicted {count_noun(evicted, 'stale Sonarr parse')}")

        # Each release's video files (creditless OP/ED, subs, fonts, audio dropped) with their listed sizes.
        listed = {
            url: [(f, release.file_sizes[idx]) for idx, f in indexed_video_files(release.files)]
            for group_releases in seadex_dict.values()
            for url, release in group_releases.releases.items()
        }
        self._warm_parse_cache(list(dict.fromkeys(f for files in listed.values() for f, _ in files)), window=window)
        unread: set[int] = set()
        return {
            url: tuple(ListedFile(f, size, self._aired_parse_for(f, window=window, unread=unread)) for f, size in files)
            for url, files in listed.items()
        }
