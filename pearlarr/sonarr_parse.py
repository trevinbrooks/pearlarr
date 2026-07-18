"""Sonarr `/parse` cache collaborator: SeaDex filenames -> season/episode.

`SonarrParseCache` owns the grab-time `/parse` of a release's filenames plus
the durable, freshness-checked parse cache (read-through `cache_store`):
cold-cache warm, per-file freshness, negative-record self-heal, and TTL
eviction. The series-id fingerprint that pins negative records is threaded in
per call (`series_fp`) so this stays decoupled from the episode collaborator
that computes it.
"""

import concurrent.futures
import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, NamedTuple, NotRequired, TypedDict, cast

from .cache import UPDATED_AT_STR_FORMAT, record_is_fresh
from .json_narrow import is_json_list, is_json_obj
from .log import count_noun
from .run_services import RunDeps
from .seadex_types import EpisodeRecord, ParsedEpisode, SeadexDict, SonarrParse
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

# How long a persisted Sonarr /parse result stays usable before it's re-queried.
# A filename's season/episode mapping is stable, but Sonarr's /parse depends on
# the current library, so a wrong-but-non-empty match could otherwise be trusted
# forever. Re-validate monthly so such an entry self-heals.
SONARR_PARSE_CACHE_TTL_DAYS = 30

# How long a NEGATIVE (confirmed empty) parse result stays usable. Pinned to the
# series-id set (see sonarr_episodes.sonarr_series_fingerprint) so adding the
# series self-heals it. This TTL is only a backstop, so it is short.
SONARR_PARSE_NEG_CACHE_TTL_DAYS = 7


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
        base = os.path.basename(name)
        if is_video_candidate(base):
            yield idx, base


class ParseWindow(NamedTuple):
    """The freshness window for one parse pass, computed once per call.

    Bundles the values the parse-cache freshness check and writes thread
    together. One window is built in `parse_episodes_from_seadex` and passed
    down to the fresh-check / writer.
    """

    now_str: str
    """Stamps new records."""

    cutoff: datetime
    """Bounds a positive record's TTL."""

    neg_cutoff: datetime
    """The short negative-record backstop."""

    series_fp: str
    """Pins negative records to the current series-id set (so a newly-added series self-heals)."""


class SonarrParseRecord(TypedDict):
    """One persisted Sonarr `/parse` cache record, keyed by filename."""

    fetched_at: str
    """Stamps the record for TTL eviction."""

    episodes: list[dict[str, int]]
    """The parsed season/episode list (empty for a negative record)."""

    series_fp: NotRequired[str]
    """`NotRequired` because only a negative record carries it (pinning it to the series-id set) - the
    freshness reader dispatches on exactly that presence."""

    full_season: NotRequired[bool]
    """`NotRequired` (mirrors `series_fp`): written only when Sonarr flagged `parsedEpisodeInfo.fullSeason`, so
    every pre-existing row reads False and ages out on the TTL - never a schema bump."""


def parsed_episodes(record: Mapping[str, object]) -> list[ParsedEpisode]:
    """Deserialize a parse record's `episodes` JSON into `ParsedEpisode`s.

    The sole reader of the persisted `{"season", "episode"}` objects (paired
    with `_serialize_episodes`), so those magic keys live at this one cache
    seam. A malformed entry is skipped, never crashed on.
    """

    raw = record.get("episodes")
    if not is_json_list(raw):
        return []
    episodes: list[ParsedEpisode] = []
    for entry in raw:
        if not is_json_obj(entry):
            continue
        season = entry.get("season")
        episode = entry.get("episode")
        if isinstance(season, int) and isinstance(episode, int):
            episodes.append(ParsedEpisode(season=season, episode=episode))
    return episodes


def parsed_full_season(record: Mapping[str, object]) -> bool:
    """Whether a parse record marks Sonarr's `parsedEpisodeInfo.fullSeason`.

    Reads the `NotRequired` key at the same cache seam as `parsed_episodes`
    (absent -> False, today's behavior), so the seed guard can refuse a
    full-season parse by Sonarr's own flag rather than only the span cap.
    """

    return bool(record.get("full_season", False))


def _serialize_episodes(episodes: list[ParsedEpisode]) -> list[dict[str, int]]:
    """Serialize `ParsedEpisode`s to the persisted `{"season", "episode"}` objects.

    The inverse of `parsed_episodes`: the cache.db entries stay JSON objects, so
    existing records read back unchanged (migration-free).
    """

    return [{"season": ep.season, "episode": ep.episode} for ep in episodes]


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
        self.logger = deps.logger

    @staticmethod
    def _sonarr_parse_is_fresh(
        record: dict[str, Any],
        *,
        window: ParseWindow,
    ) -> bool:
        """True if a persisted parse record is still usable.

        Positive (has episodes): stable mapping, valid for the 30-day
        `window.cutoff`. Negative (empty): valid only while the series-id set is
        unchanged (matching `window.series_fp`) and within the short
        `window.neg_cutoff` backstop, so a newly-added series self-heals. Legacy
        records (no fp) read stale.
        """

        if record.get("episodes"):
            return record_is_fresh(
                record,
                payload_key="episodes",
                cutoff=window.cutoff,
            )
        if record.get("series_fp") != window.series_fp:
            return False
        try:
            return datetime.strptime(record.get("fetched_at", ""), UPDATED_AT_STR_FORMAT) >= window.neg_cutoff
        except (TypeError, ValueError):
            return False

    def _write_parse_record(self, filename: str, parse: SonarrParse, *, window: ParseWindow) -> None:
        """Upsert a Sonarr parse-cache record (one builder for both shapes).

        A NEGATIVE record (empty `episodes`) carries the series fingerprint so
        it self-heals when the library changes. A POSITIVE one never does - the
        freshness reader dispatches on exactly that presence. `full_season` is
        written only when set (mirroring `series_fp`), so legacy rows read
        False. The episodes are serialized to `{"season", "episode"}` JSON
        objects at this seam.
        """

        record: SonarrParseRecord = {"fetched_at": window.now_str, "episodes": _serialize_episodes(parse.episodes)}
        if not parse.episodes:
            record["series_fp"] = window.series_fp
        if parse.full_season:
            record["full_season"] = True
        self.cache_store.put_sonarr_parse(filename, cast("dict[str, Any]", record))

    def _episodes_for(self, f: str, *, window: ParseWindow) -> list[ParsedEpisode]:
        """One file's season/episode records, read-through the parse cache.

        Empty means skip: a fresh negative record, a transient parse failure
        (not cached, re-queried on demand), and a fresh parse that found
        nothing all fold to `[]`. The full-season flag only rides the persisted
        record (for the seed guard), so this mapping read still yields pairs.
        """

        record = self.cache_store.get_sonarr_parse(f)
        if record is not None and self._sonarr_parse_is_fresh(record, window=window):
            return parsed_episodes(record)
        result = self.sonarr.parse(f)
        # None = request failed: skip without caching a transient miss.
        if result is None:
            return []
        # Cache the result (negatives are series-fp pinned so they aren't
        # re-parsed every run) before acting on it.
        self._write_parse_record(f, result, window=window)
        if not result.episodes:
            self.logger.debug(f"Sonarr could not parse episode for {f}")
        return result.episodes

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
                    record = self.cache_store.get_sonarr_parse(f)
                    if record is not None and self._sonarr_parse_is_fresh(record, window=window):
                        continue
                    pending.append(f)

        if len(pending) <= 1:
            return

        def fetch(name: str) -> tuple[str, SonarrParse | None]:
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
            self._write_parse_record(name, result, window=window)

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
            series_fp: The run's series-id fingerprint, pinning negative
                records (from the episode collaborator).
        """

        # Cutoffs computed once per call (not per file), all anchored to one instant.
        now = datetime.now()
        window = ParseWindow(
            now_str=now.strftime(UPDATED_AT_STR_FORMAT),
            cutoff=now - timedelta(days=SONARR_PARSE_CACHE_TTL_DAYS),
            neg_cutoff=now - timedelta(days=SONARR_PARSE_NEG_CACHE_TTL_DAYS),
            series_fp=series_fp,
        )

        # Evict parse records aged past that same cutoff so the block stops growing
        # without bound. Staged like the writes below (committed at the run's save
        # point, discarded in a preview). Only the first call per run finds stale
        # rows, later calls evict nothing.
        evicted = self.cache_store.evict_sonarr_parse(window.cutoff)
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
                    parsed = self._episodes_for(f, window=window)
                    if not parsed:
                        continue

                    size = sizes[sd_file_idx]
                    for ep in parsed:
                        season = ep.season
                        episode = ep.episode

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
