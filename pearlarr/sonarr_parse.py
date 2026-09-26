"""Sonarr `/parse` cache collaborator: SeaDex filenames, read through the parse cache, for the placement.

`SonarrParseCache` owns the grab-time `/parse` of a release's filenames plus
the durable parse cache: cold-cache warm, TTL eviction, and the gather that
pairs each video file with its parse for `place_release`. The cache leaf
itself (`ParseRecords`, in `parse_records`) is bound once on `RunDeps`. The
series-id fingerprint that pins unmatched records is threaded in per call
(`series_fp`) so this stays decoupled from the episode collaborator that
computes it.
"""

import concurrent.futures
from collections.abc import Sequence

from .grab_placement import SeedFile
from .log import count_noun
from .parse_records import ParseWindow
from .run_services import RunDeps
from .seadex_types import ParsedFileInfo, SeadexDict
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import fetch_workers
from .video_files import video_file_entries


class SonarrParseCache:
    """Owns the grab-time `/parse` and the durable, freshness-checked parse cache, built once per run.

    Reads go through `records`, so a write staged by `parsed_files` is visible to a later same-run read.
    """

    def __init__(self, deps: RunDeps, sonarr: AbstractSonarrClient) -> None:
        """Bind the shared collaborators and the strategy's Sonarr client, whose `/parse` the cache fills from."""

        self.sonarr = sonarr
        self._config = deps.config
        self.cache_store = deps.cache_store
        self.records = deps.parse_records
        self.logger = deps.logger

    def _parse_for(self, f: str, *, window: ParseWindow) -> ParsedFileInfo | None:
        """One file's parse, read-through the cache: a fresh hit, else Sonarr's answer (cached), else None.

        None is a transport miss (not cached, re-queried on demand).
        """

        info = self.records.read(f, window=window)
        if info is not None:
            return info
        info = self.sonarr.parse(f)
        if info is None:
            return None
        # Cache the result (unmatched parses are series-fp pinned so they
        # aren't re-parsed every run) before acting on it.
        self.records.write(f, info, window=window)
        if not info.matched_episodes:
            self.logger.debug(f"Sonarr could not parse episode for {f}")
        return info

    def _warm_parse_cache(self, names: Sequence[str], *, window: ParseWindow) -> None:
        """Concurrently parse the distinct `names` not yet cached, so the gather reads them as hits.

        Only `sonarr.parse` runs in the pool (cache reads and writes stay on this thread). No-op when sequential.
        """

        workers = fetch_workers(self._config)
        if workers <= 1:
            return

        pending = [f for f in names if not self.records.is_fresh(f, window=window)]

        if len(pending) <= 1:
            return

        def fetch(name: str) -> tuple[str, ParsedFileInfo | None]:
            # A RAISE degrades to a transient miss (None: not cached, re-parsed on
            # demand) so one bad file can't abort the concurrent warm sweep.
            try:
                return name, self.sonarr.parse(name)
            except Exception:
                return name, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            results = list(pool.map(fetch, pending))

        for name, result in results:
            if result is None:  # request failed: don't cache a transient miss
                continue
            self.records.write(name, result, window=window)

    def parsed_files(
        self,
        seadex_dict: SeadexDict,
        *,
        series_fp: str,
    ) -> dict[str, tuple[SeedFile, ...]]:
        """Each url's video files with their sizes and Sonarr parses, keyed as the groups key their urls.

        A name goes to Sonarr once, within a run (the same file across overlapping groups) and across
        runs through the durable parse cache, its unmatched rows pinned by `series_fp`. Every url gets
        an entry, empty when it lists no video file. Never mutates `seadex_dict`.
        """

        # Cutoffs computed once per call (not per file), all anchored to one instant.
        window = ParseWindow.open(series_fp)

        # Evict parse records aged past that same cutoff so the block stops growing
        # without bound. Staged like the writes below (committed at the run's save
        # point, discarded in a preview). Only the first call per run finds stale
        # rows, later calls evict nothing.
        evicted = self.cache_store.evict_sonarr_parse(window.matched_cutoff)
        if evicted:
            self.logger.debug(f"Evicted {count_noun(evicted, 'stale Sonarr parse record')}")

        # Each url's video files (creditless OP/ED, subs, fonts, audio dropped) with their listed sizes.
        listed = {
            url: [(f, url_item.size[idx]) for idx, f in video_file_entries(url_item.files)]
            for release_group_item in seadex_dict.values()
            for url, url_item in release_group_item.urls.items()
        }
        self._warm_parse_cache(list(dict.fromkeys(f for files in listed.values() for f, _ in files)), window=window)
        return {
            url: tuple(SeedFile(f, size, self._parse_for(f, window=window)) for f, size in files)
            for url, files in listed.items()
        }
