"""Sonarr `/parse` cache collaborator: SeaDex filenames, read through the parse cache, for the placement.

`SonarrParseCache` owns the grab-time `/parse` of a release's filenames plus
the durable parse cache: cold-cache warm, TTL eviction, and the gather that
pairs each video file with its parse for `place_release`. The cache rows
themselves (`ParseCacheRows`, in `parse_cache`) are bound once on `ArrRunDeps`. The
series hash that pins unmatched rows is threaded in per call
(`series_hash`) so this stays decoupled from the episode collaborator that
computes it.
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
from .sonarr_episodes import fetch_workers
from .video_files import indexed_video_files


class SonarrParseCache:
    """Owns the grab-time `/parse` and the durable, freshness-checked parse cache, built once per run.

    Reads go through `rows`, so a write staged by `parsed_files` is visible to a later same-run read.
    """

    def __init__(self, deps: ArrRunDeps, sonarr: AbstractSonarrClient) -> None:
        """Bind the shared collaborators and the strategy's Sonarr client, whose `/parse` the cache fills from."""

        self.sonarr = sonarr
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

    def parsed_files(
        self,
        seadex_dict: SeadexDict,
        *,
        series_hash: str,
    ) -> dict[str, tuple[ListedFile, ...]]:
        """Each release's video files with their sizes and Sonarr parses, keyed by url as the groups key their releases.

        A name goes to Sonarr once, within a run (the same file across overlapping groups) and across
        runs through the durable parse cache, its unmatched rows pinned by `series_hash`. Every release gets
        a value, empty when it lists no video file. Never mutates `seadex_dict`.
        """

        # Cutoffs computed once per call (not per file), all anchored to one instant.
        window = ParseFreshness.open(series_hash)

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
        return {
            url: tuple(ListedFile(f, size, self._parse_for(f, window=window)) for f, size in files)
            for url, files in listed.items()
        }
