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
import os
from collections.abc import Iterator, Sequence

from .grab_placement import SeedFile
from .log import count_noun
from .manual_import import path_leaf
from .parse_records import ParseWindow
from .run_services import RunDeps
from .seadex_types import ParsedFileInfo, SeadexDict
from .sonarr_client import AbstractSonarrClient
from .sonarr_episodes import fetch_workers

TORRENT_FILENAMES_TO_SKIP = [
    "NCED",
    "NCOP",
    "Creditless Ending",
    "Creditless Opening",
    "Creditless ED",
    "Creditless OP",
]

# File extensions that never map to an episode (subtitles, fonts, chapters,
# metadata, images, samples, ...). We skip these before querying Sonarr so we
# don't waste a round-trip on them. This is deliberately a deny-list rather than
# an allow-list of video extensions: the cost of missing one here is a single
# harmless API call (Sonarr just returns no episode), whereas an allow-list that
# omits an unusual container would silently drop a real episode.
NON_VIDEO_EXTENSIONS = {
    ".ass",
    ".srt",
    ".ssa",
    ".sub",
    ".idx",
    ".sup",
    ".vtt",
    # Matroska SUBTITLES (often SxxExx-named, so they'd parse). The video .mkv is kept.
    ".mks",
    ".nfo",
    ".txt",
    ".md",
    ".sfv",
    ".xml",
    ".json",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".torrent",
    ".url",
    ".rar",
    ".zip",
    ".7z",
    # Bundled manga volumes and books.
    ".cbz",
    ".cbr",
    ".cb7",
    ".pdf",
    ".epub",
    # Audio tracks + rip sidecars (OSTs, EAC logs/cuesheets) bundled in releases:
    # never an episode. `.mka` is Matroska audio (the video `.mkv` is kept).
    ".flac",
    ".mka",
    ".wav",
    ".aac",
    ".ac3",
    ".dts",
    ".dtshd",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".wv",
    ".tak",
    ".ape",
    ".cue",
    ".log",
    ".m3u8",
    # Checksum sidecars: Sonarr's manual-import scan never offers them, so a
    # seeded one sticks a record on "intended file missing" until the deadline.
    ".blake3",
    ".md5",
    ".sha2",
    ".sha256",
}


def is_video_candidate(basename: str) -> bool:
    """Whether a filename is an importable video (not a sub, font, or creditless OP/ED).

    The one home of the skip rules, so the seed, the import and the parse agree on what can be an episode.
    """

    if any(skip in basename for skip in TORRENT_FILENAMES_TO_SKIP):
        return False
    return os.path.splitext(basename)[1].lower() not in NON_VIDEO_EXTENSIONS


def video_file_entries(files: Sequence[str]) -> Iterator[tuple[int, str]]:
    """Yield `(index, basename)` for each importable video file in `files`, the index keying a size list."""

    for idx, name in enumerate(files):
        base = path_leaf(name)
        if is_video_candidate(base):
            yield idx, base


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
        listed = [
            (url, [(f, url_item.size[idx]) for idx, f in video_file_entries(url_item.files)])
            for release_group_item in seadex_dict.values()
            for url, url_item in release_group_item.urls.items()
        ]
        self._warm_parse_cache(list(dict.fromkeys(f for _, files in listed for f, _ in files)), window=window)
        return {
            url: tuple(SeedFile(f, size, self._parse_for(f, window=window)) for f, size in files)
            for url, files in listed
        }
