"""Sonarr `/parse` cache collaborator: SeaDex filenames -> season/episode.

`SonarrParseCache` owns the grab-time `/parse` of a release's filenames plus
the durable parse cache: cold-cache warm, TTL eviction, and the mapping loop.
The cache leaf itself (`ParseRecords`, in `parse_records`) is bound once on
`RunDeps` and shared with the seed builder, so both read one `ParsedFileInfo`
per file. The series-id fingerprint that pins unmatched records is threaded in
per call (`series_fp`) so this stays decoupled from the episode collaborator
that computes it.
"""

import concurrent.futures
import os
from collections.abc import Iterator, Sequence

from .log import count_noun
from .manual_import import path_leaf
from .parse_records import ParseWindow
from .run_services import RunDeps
from .seadex_types import EpisodeRecord, ParsedFileInfo, SeadexDict
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
    # Matroska SUBTITLES (often SxxExx-named, so they'd parse) - .mkv is kept.
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
    """Whether a filename is an importable video (not a sub/font/NCED/sample).

    The single source of the skip rules, so the seed, the import-time repair,
    and the parse all agree on which files are even candidates for an episode.
    Module-level (owned by none) since several collaborators share it.
    """

    if any(skip in basename for skip in TORRENT_FILENAMES_TO_SKIP):
        return False
    return os.path.splitext(basename)[1].lower() not in NON_VIDEO_EXTENSIONS


def video_file_entries(files: Sequence[str]) -> Iterator[tuple[int, str]]:
    """Yield `(index, basename)` for each importable video file in `files`.

    The one basename+skip iteration the warm pass, the parse loop, and the seed
    builder share. The index survives so an index-aligned size list stays usable.
    """

    for idx, name in enumerate(files):
        base = path_leaf(name)
        if is_video_candidate(base):
            yield idx, base


class SonarrParseCache:
    """Owns the grab-time `/parse` + the durable, freshness-checked parse cache.

    Constructed once per run in `SonarrSync` from the shared `RunDeps` and the
    strategy's Sonarr client. The cache is read-through `cache_store` (the same
    leaf the seed builder reads), so staged writes from `parse_episodes_from_seadex`
    are visible to a later same-run read.
    """

    def __init__(self, deps: RunDeps, sonarr: AbstractSonarrClient) -> None:
        """Bind the shared collaborators the parse cache reads.

        Args:
            deps: The shared collaborators (config/cache/logger unpacked
                off it).
            sonarr: The strategy's Sonarr client (its `/parse`).
        """

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

    def _warm_parse_cache(
        self,
        seadex_dict: SeadexDict,
        *,
        window: ParseWindow,
    ) -> None:
        """Concurrently parse the not-yet-cached files for one release.

        Cold-cache pre-pass: collapses the per-file `/parse` latency the same
        way `prefetch_episodes` does for episodes, deduping repeats across
        overlapping release groups. The mapping loop then reads from the warm
        cache. Only `sonarr.parse` runs in the pool. Cache reads/writes stay on
        the main thread. No-op when sequential (`sleep_time > 0`) or warm.
        """

        workers = fetch_workers(self._config)
        if workers <= 1:
            return

        pending: list[str] = []
        seen: set[str] = set()
        for srg_item in seadex_dict.values():
            for url_item in srg_item.urls.values():
                for _, f in video_file_entries(url_item.files):
                    if f in seen:
                        continue
                    seen.add(f)
                    if not self.records.is_fresh(f, window=window):
                        pending.append(f)

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

    def parse_episodes_from_seadex(
        self,
        seadex_dict: SeadexDict,
        *,
        series_fp: str,
    ) -> SeadexDict:
        """For files in a SeaDex release, parse this through Sonarr to get season/episode numbers.

        This gets an overall episode list per-release group, and also episode lists per-torrent,
        if there are multiple

        Parsed filenames are cached through the cache store, so a given
        filename is only ever sent to Sonarr once - both within a run, where
        the same file can appear across overlapping release groups, and across
        runs. The mapping is deterministic for a SeaDex release name, so this is
        safe. Only successful parses are cached, so a file becomes parseable as
        soon as its series is added to Sonarr.

        Args:
            seadex_dict: The releases to parse. Episode lists are attached to
                its items in place, and it is returned.
            series_fp: The run's series-id fingerprint, pinning unmatched
                records (from the episode collaborator).
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

        # Concurrently warm the cache for any not-yet-cached files so the mapping
        # loop below reads them as hits (no-op when sequential or already warm).
        self._warm_parse_cache(seadex_dict, window=window)

        for release_group_item in seadex_dict.values():
            # Set up an overall "all episodes" list (bound locally so the
            # appends below stay typed as list, not list | None)
            all_episodes: list[EpisodeRecord] = []
            release_group_item.all_episodes = all_episodes

            for url_item in release_group_item.urls.values():
                # Set up a list to parse episodes from files
                episodes: list[EpisodeRecord] = []
                url_item.episodes = episodes
                sizes = url_item.size

                # Video files only (NCED/NCOP, subs, fonts, audio dropped) - the
                # same rule the warm pass uses. The index keys the size list.
                for sd_file_idx, f in video_file_entries(url_item.files):
                    # Fresh cache hit, or query Sonarr and cache the result so it
                    # expires (re-validates) rather than being trusted forever.
                    info = self._parse_for(f, window=window)
                    if info is None or not info.matched_episodes:
                        continue

                    size = sizes[sd_file_idx]
                    for matched in info.matched_episodes:
                        season = matched.season_number
                        episode = matched.episode_number

                        self.logger.debug(f"{f} mapped to: S{season:02d}E{episode:02d}")

                        # EpisodeRecord is immutable, so the per-url and the
                        # release-group-wide lists can share one instance.
                        ep_record = EpisodeRecord(
                            season=season,
                            episode=episode,
                            size=size,
                        )
                        episodes.append(ep_record)
                        all_episodes.append(ep_record)

        return seadex_dict
