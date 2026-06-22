import copy
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from itertools import compress
from typing import Any
from urllib.request import urlretrieve
from xml.etree import ElementTree

import httpx
import qbittorrentapi
import requests
from seadex import EntryNotFoundError, EntryRecord, SeaDexEntry

from . import coverage as _coverage
from .anibridge import AniBridge
from .anilist import (
    ANILIST_BATCH_SIZE,
    get_anilist_thumb,
    get_anilist_title,
    get_query_batch,
)
from .cache import UPDATED_AT_STR_FORMAT, CacheStore
from .config import PRIVATE_TRACKERS, AppConfig
from .discord import discord_push
from .log import (
    LogFormatter,
    count_noun,
    entry_string,
    group_highlight,
    indent_string,
    rule_string,
    setup_logger,
)
from .planner import (
    get_all_seadex_rgs_per_episode,
    get_same_files_groups,
    normalize_rg,
)
from .torrent import (
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)

ANIME_IDS_URL = "https://raw.githubusercontent.com/Kometa-Team/Anime-IDs/refs/heads/master/anime_ids.json"
ANIME_IDS_FILE = "anime_ids.json"
ANIDB_MAPPINGS_URL = "https://raw.githubusercontent.com/Anime-Lists/anime-lists/refs/heads/master/anime-list-master.xml"
ANIDB_MAPPINGS_FILE = "anime-list-master.xml"

# anibridge-mappings ships daily-rolling release assets tagged by major version.
# Bump ANIBRIDGE_RELEASE when a new (breaking) major lands; the cache filename is
# versioned to match so an old-format file is never parsed by the new loader.
ANIBRIDGE_RELEASE = "v3"
ANIBRIDGE_MAPPINGS_URL = f"https://github.com/anibridge/anibridge-mappings/releases/download/{ANIBRIDGE_RELEASE}/mappings.min.json"
ANIBRIDGE_MAPPINGS_FILE = f"anibridge_mappings_{ANIBRIDGE_RELEASE}.json"


# --- Shared parse cache for the immutable mapping sources -------------------
#
# anime_ids.json, anime-list-master.xml and the anibridge graph are large,
# read-only files whose parsed and indexed forms never change once an arr
# instance is built. A scheduled cycle builds a SeaDexRadarr then a SeaDexSonarr
# in the same process, and each used to independently re-read and re-index all
# three. This memo caches the parsed result under a stable key, holding a single
# (mtime, value) slot per key: the second instance reuses the first's parse
# while the file is unchanged on disk, and a re-download (new mtime) replaces the
# slot rather than accumulating. The parsed objects are only ever read after
# construction, so sharing them is safe; the CLI run path is single-threaded, so
# the plain dict needs no locking.
_PARSED_MAPPING_CACHE: dict[str, tuple[float, Any]] = {}


def _load_mapping_by_mtime(
    path: str,
    parse: Callable[[str], Any],
    cache_key: str | None = None,
) -> Any:
    """Return ``parse(path)``, reusing a cached result while the mtime is unchanged.

    Args:
        path (str): File whose modification time gates the cache
        parse (Callable[[str], Any]): Builds the parsed value from the path
        cache_key (str | None): Cache slot to use; defaults to ``path``. Pass a
            distinct key when more than one product is derived from one file
            (e.g. the Anime-IDs map and its reverse index).
    """

    key = cache_key or path
    mtime = os.path.getmtime(path)

    cached = _PARSED_MAPPING_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    value = parse(path)
    _PARSED_MAPPING_CACHE[key] = (mtime, value)
    return value


def _parse_anime_mappings(path: str) -> dict:
    """Load the Kometa Anime-IDs JSON map from disk."""

    with open(path) as f:
        return json.load(f)


def _build_anime_mappings_index(anime_mappings: dict) -> dict[str, dict]:
    """Build reverse indexes over the Kometa Anime-IDs map for fast lookups

    Anime-IDs is a flat {name: mapping} dict of ~16k entries. Querying it by
    external id (the per-series hot path) used to mean a full scan per id type;
    this groups every mapping with an AniList id by each external id it carries,
    so get_mappings_from_anime_mappings becomes a dict lookup.

    Args:
        anime_mappings (dict): Parsed Anime-IDs map ({name: mapping})

    Returns:
        dict: field name -> {external id -> [mapping, ...]}, for the
            "tvdb_id", "tmdb_movie_id", "tmdb_show_id" and "imdb_id" fields
    """

    index: dict[str, dict] = {
        "tvdb_id": defaultdict(list),
        "tmdb_movie_id": defaultdict(list),
        "tmdb_show_id": defaultdict(list),
        "imdb_id": defaultdict(list),
    }
    for m in anime_mappings.values():
        if m.get("anilist_id") is None:
            continue
        for field, bucket in index.items():
            value = m.get(field)
            if value is not None:
                bucket[value].append(m)
    return index


def _parse_anime_mappings_index(path: str) -> dict[str, dict]:
    """Build the Anime-IDs reverse index over the memoized parse of ``path``."""

    return _build_anime_mappings_index(_load_mapping_by_mtime(path, _parse_anime_mappings))


def _parse_anidb_mappings(path: str) -> ElementTree.Element:
    """Parse the AniDB anime-list XML and return its root element."""

    return ElementTree.parse(path).getroot()


def _parse_anibridge(path: str, logger: logging.Logger | None) -> AniBridge:
    """Load the anibridge graph from disk and build its indexed view."""

    with open(path) as f:
        graph = json.load(f)

    return AniBridge(graph, logger=logger)


ALLOWED_ARRS = [
    "radarr",
    "sonarr",
]

# How long a persisted AniList response stays usable before it's re-fetched.
# title/format/coverImage are effectively static; episodes for a currently airing
# show drift, so this caps how stale that count can get (~one episode/week).
ANILIST_CACHE_TTL_DAYS = 7


class SeaDexArr(ABC):

    def __init__(
        self,
        arr: str = "sonarr",
        config: str = "config.yml",
        cache: str = "cache.json",
        logger: logging.Logger | None = None,
    ) -> None:
        """Base class for SeaDexArr instances

        Args:
            arr (str, optional): Which Arr is being run.
                Defaults to "sonarr".
            config (str, optional): Path to a config file.
                Defaults to "config.yml".
            cache (str, optional): Path to a cache file.
                Defaults to "cache.json".
            logger. Logging instance. Defaults to None,
                which will create one.
        """

        # Load, template-sync, and expose the config file as typed settings.
        # AppConfig owns the file lifecycle (copy-template-if-missing, parse,
        # key-order sync); self.config stays bound to the raw mapping for the
        # few arr-specific keys the subclasses still read directly.
        self._config = AppConfig.load(config, arr)
        self.config_file = config
        self.config = self._config.data

        # Ignore unmonitored flag
        self.ignore_unmonitored = self._config.ignore_unmonitored

        # A single keep-alive session shared by the raw Sonarr/Radarr API calls
        self.session = requests.Session()

        # qbit
        self.qbit: qbittorrentapi.Client
        qbit_info = self._config.qbit_info

        # Configured only when every qbit_info field has a value; with a missing
        # block or any null field, no client is created.
        if qbit_info is not None and all(
            qbit_info.get(key, None) is not None for key in qbit_info
        ):
            qbit = qbittorrentapi.Client(**qbit_info)

            try:
                qbit.auth_log_in()
            except qbittorrentapi.LoginFailed:
                raise ValueError(
                    "qBittorrent login failed - check the qbit_info host and "
                    "credentials in your config",
                )

            self.qbit = qbit

        self.ignore_seadex_update_times = self._config.ignore_seadex_update_times

        self.use_torrent_hash_to_filter = self._config.use_torrent_hash_to_filter

        # Hooks between torrents and Arts, and torrent number bookkeeping
        self.torrent_category = self._config.torrent_category
        self.torrent_tags = self._config.torrent_tags
        self.max_torrents_to_add = self._config.max_torrents_to_add
        self.torrents_added = 0

        # When True, simulate a run without grabbing torrents, writing the cache,
        # or sending notifications. Set per-run by run(); the no-op default here
        # keeps every method that consults it safe before run() is called.
        self.dry_run = False

        # Per-run tally for the end-of-run summary (reset at the start of run())
        self.stats = self._fresh_stats()
        self._run_started_monotonic = None
        self._log_counts_at_start = {}
        # Title, SeaDex URL, and season/episode coverage currently being
        # processed, so add_torrent and the summary can attribute what they grab
        # (and link to it, and show which files we mapped) without threading them
        # through every call
        self.current_title = None
        self.current_url = None
        self.current_coverage = None

        # Discord
        self.discord_url = self._config.discord_url

        # Flags for filtering torrents
        self.public_only = self._config.public_only
        self.prefer_dual_audio = self._config.prefer_dual_audio
        self.want_best = self._config.want_best

        # Set per-title when public_only forces us to skip a release that's only
        # available privately, so the caller knows not to cache the title as done.
        # public_only_groups collects the release-group name(s) that were skipped
        # for that reason, so the run summary can name them under "needs action"
        self.public_only_skipped = False
        self.public_only_groups: list[str] = []

        self.ignore_tags = self._config.ignore_tags

        # AniList IDs to skip entirely
        self.ignore_anilist_ids = self._config.ignore_anilist_ids

        # All trackers (public + private) by default; private are filtered later,
        # after the overlap check against what's already downloaded.
        self.trackers = self._config.trackers

        # Advanced settings
        self.sleep_time = self._config.sleep_time
        self.cache_time = self._config.cache_time

        # Get the mapping files
        anime_mappings_cfg = self._config.anime_mappings_cfg
        anidb_mappings_cfg = self._config.anidb_mappings_cfg
        anibridge_mappings_cfg = self._config.anibridge_mappings_cfg

        if anime_mappings_cfg is False:
            anime_mappings = {}
            anime_mappings_index = None
        elif anime_mappings_cfg is None:
            anime_mappings = self.get_anime_mappings()
            anime_mappings_index = self._get_anime_mappings_index()
        else:
            anime_mappings = anime_mappings_cfg
            anime_mappings_index = _build_anime_mappings_index(anime_mappings)

        if anidb_mappings_cfg is False:
            anidb_mappings = None
        elif anidb_mappings_cfg is None:
            anidb_mappings = self.get_anidb_mappings()
        else:
            anidb_mappings = anidb_mappings_cfg

        if anibridge_mappings_cfg is False:
            anibridge = None
        elif anibridge_mappings_cfg is None:
            anibridge = self.get_anibridge_mappings()
        else:
            # A config-provided value is treated as a raw anibridge graph dict
            anibridge = AniBridge(
                anibridge_mappings_cfg,
                logger=getattr(self, "logger", None),
            )

        self.anime_mappings = anime_mappings
        # Reverse indexes over the (large, ~16k-entry) Kometa Anime-IDs map so
        # get_mappings_from_anime_mappings is an O(matches) dict lookup instead of
        # three full scans of every mapping per series. Shared across instances
        # via the mtime memo when both arrs read the same on-disk file.
        self._anime_mappings_index = anime_mappings_index if anime_mappings else None
        self.anidb_mappings = anidb_mappings
        # Lazily-built {anidbid -> [element]} index over the AniDB XML, so the
        # specials/movie path in get_ep_list does a dict lookup instead of an
        # XPath scan of every <anime> element. Populated on first use.
        self._anidb_index: dict[str, list] | None = None
        self.anibridge = anibridge

        self.interactive = self._config.interactive

        if logger is None:
            self.logger = setup_logger(log_level=self._config.log_level)
        else:
            self.logger = logger

        # Instantiate the SeaDex API
        self.seadex = SeaDexEntry()

        self.al_cache = {}

        # Memoize get_anilist_ids mapping computation per identifying key, so
        # the prefetch pass and the main loop don't compute it twice per item
        self._anilist_ids_cache = {}

        # Per-run cache of the raw Sonarr episode fetch, keyed by series id. A
        # multi-season series maps to several AniList ids, each of which would
        # otherwise re-fetch the same whole-series episode list; cache it for the
        # run so the network round-trip happens once per series. Reset per run.
        self._ep_list_cache: dict[int, list] = {}

        # Load the cache (or create its schema) and reconcile the descriptor
        # against the current package version + config checksum. Each arr builds
        # its own store that reads the file fresh, so a scheduled Radarr->Sonarr
        # cycle hands off through cache.json rather than shared memory.
        self.cache_file = cache
        self.cache_store = CacheStore.load(cache, config_checksum=self._config.checksum())
        self.cache: dict[str, Any] = self.cache_store.data

        # All aligned detail rendering goes through this formatter, so the
        # presentation primitives (kv lines, blank separators, elapsed strings)
        # live on it rather than on the orchestration class. line_length is the
        # full width used for the run's separator rules.
        self.log_fmt = LogFormatter(self.logger)

    def close(self) -> None:
        """Close the shared HTTP session (release pooled connections)."""
        if self.session is not None:
            self.session.close()

    def get_anime_mappings(self) -> dict:
        """Get the anime IDs file"""

        self.get_external_mappings(
            f=ANIME_IDS_FILE,
            url=ANIME_IDS_URL,
        )

        return _load_mapping_by_mtime(ANIME_IDS_FILE, _parse_anime_mappings)


    def _get_anime_mappings_index(self) -> dict[str, dict]:
        """Reverse index over the on-disk Anime-IDs map, shared via the mtime memo.

        Must follow get_anime_mappings (which downloads/refreshes the file); it
        reuses that memoized parse instead of re-reading the JSON.
        """

        return _load_mapping_by_mtime(
            ANIME_IDS_FILE,
            _parse_anime_mappings_index,
            cache_key=f"{ANIME_IDS_FILE}#index",
        )


    def get_anidb_mappings(self) -> ElementTree.Element:
        """Get the AniDB mappings file"""

        self.get_external_mappings(
            f=ANIDB_MAPPINGS_FILE,
            url=ANIDB_MAPPINGS_URL,
        )

        return _load_mapping_by_mtime(ANIDB_MAPPINGS_FILE, _parse_anidb_mappings)


    def anidb_anime_by_id(self, anidb_id: int) -> list:
        """Return the AniDB XML <anime> element(s) for an AniDB id

        Builds (once, lazily) an "{anidbid -> [element]}" index over the parsed
        AniDB mappings, so callers do a dict lookup instead of an XPath scan of
        every <anime> element. Returns a list to preserve the caller's existing
        "0 / 1 / >1 match" handling.

        Args:
            anidb_id (int): AniDB id to look up
        """

        if self.anidb_mappings is None:
            return []

        if self._anidb_index is None:
            index: dict[str, list] = defaultdict(list)
            for anime in self.anidb_mappings.findall("anime"):
                anidbid = anime.get("anidbid")
                if anidbid is not None:
                    index[anidbid].append(anime)
            self._anidb_index = index

        return self._anidb_index.get(str(anidb_id), [])

    def get_anibridge_mappings(self) -> AniBridge:
        """Download the anibridge-mappings graph and build an indexed view.

        Returns:
            AniBridge: Parsed, indexed mappings ready for id lookups
        """

        self.get_external_mappings(
            f=ANIBRIDGE_MAPPINGS_FILE,
            url=ANIBRIDGE_MAPPINGS_URL,
        )

        # self.logger isn't set until after mappings load, so this is None during
        # __init__; the lambda lets the shared parse build AniBridge with it.
        logger = getattr(self, "logger", None)
        return _load_mapping_by_mtime(
            ANIBRIDGE_MAPPINGS_FILE,
            lambda path: _parse_anibridge(path, logger),
        )


    def get_external_mappings(
        self,
        f: str,
        url: str,
    ) -> bool:
        """Get an external mapping file, respecting a cache time

        Args:
            f (str): file on disk
            url (str): url to download the file from
        """

        if not os.path.exists(f):
            urlretrieve(url, f)

        f_mtime = os.path.getmtime(f)
        f_datetime = datetime.fromtimestamp(f_mtime)
        now_datetime = datetime.now()

        t_diff = now_datetime - f_datetime

        # If the file is older than the cache time, re-download
        if t_diff.days >= self.cache_time:
            urlretrieve(url, f)

        return True

    def get_seadex_entry(
        self,
        al_id: int,
    ) -> EntryRecord | None:
        """Get SeaDex entry from AniList ID

        Args:
            al_id (int): AniList ID
        """

        sd_entry = None
        try:
            sd_entry = self.seadex.from_id(al_id)
        except EntryNotFoundError:
            pass
        except httpx.ConnectError:
            self.logger.warning("Could not connect to SeaDex. Website may be down")

        return sd_entry

    def check_al_id_in_cache(
        self,
        arr: str,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """Whether the cached entry matches SeaDex's last-updated timestamp."""

        return self.cache_store.check_al_id_in_cache(arr, al_id, seadex_entry)

    def get_cached_name(
        self,
        arr: str,
        al_id: int,
    ) -> str | None:
        """Cached AniList title for an entry, reused without an AniList lookup."""

        return self.get_cached_field(arr, al_id, "name")

    def get_cached_field(
        self,
        arr: str,
        al_id: int,
        field: str,
    ) -> Any:
        """Read a single stored field from an entry's cache record, if present."""

        return self.cache_store.get_cached_field(arr, al_id, field)

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: str = "movie",
        log_ignored: bool = True,
    ) -> dict:
        """Get a list of entries that match on TVDB ID

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (str): TMDB type. Can be "movie" or "show"
            log_ignored (bool): Log a ledger row for each ignored AniList ID.
                Defaults to True; pass False from the prefetch pass so ignored
                ids aren't logged twice (once there, once in the main loop)
        """

        if tmdb_type not in ["movie", "show"]:
            raise ValueError("tmdb_type must be 'movie' or 'show'")

        # Check we have exactly one ID specified here
        non_none_sum = sum(v is not None for v in [tvdb_id, tmdb_id, imdb_id])

        if non_none_sum == 0:
            raise ValueError(
                "At least one of tvdb_id, tmdb_id, and imdb_id must be provided",
            )

        # The mapping computation is deterministic for a given set of
        # identifying args, so memoize it and only redo the per-call logging
        key = (tvdb_id, tmdb_id, imdb_id, tmdb_type)
        if key in self._anilist_ids_cache:
            anilist_mappings, ids_to_drop = self._anilist_ids_cache[key]
        else:
            anilist_mappings = {}

            # AniBridge is the primary source: its richer per-season episode
            # offsets win, so query it first and key results by AniList ID
            if self.anibridge:
                anilist_mappings = self.get_mappings_from_anibridge_mappings(
                    tvdb_id=tvdb_id,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    tmdb_type=tmdb_type,
                    anilist_mappings=anilist_mappings,
                )

            # Then fall back to the Kometa Anime IDs for anything AniBridge
            # doesn't cover (it only adds AniList IDs not already present)
            if self.anime_mappings:
                anilist_mappings = self.get_mappings_from_anime_mappings(
                    tvdb_id=tvdb_id,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    tmdb_type=tmdb_type,
                    anilist_mappings=anilist_mappings,
                )

            # Drop any AniList IDs the user has chosen to ignore
            ids_to_drop = [
                al_id
                for al_id in self.ignore_anilist_ids
                if al_id in anilist_mappings
            ]
            for al_id in ids_to_drop:
                del anilist_mappings[al_id]

            # Sort by AniList ID
            anilist_mappings = dict(sorted(anilist_mappings.items()))

            self._anilist_ids_cache[key] = (anilist_mappings, ids_to_drop)

        # Log ignored ids per-call (not just on the cache-filling call), so the
        # main loop still logs every ignored id even after the prefetch pass ran
        if log_ignored:
            for al_id in ids_to_drop:
                self.log_ignored_anilist_id(al_id=al_id)

        # Return a copy so a caller mutating the result can't corrupt the memo
        return dict(anilist_mappings)

    @staticmethod
    def _anilist_meta_is_fresh(record: dict | None) -> bool:
        """True if a persisted AniList record has a payload and is within TTL

        Shared by a load (which ids to seed) and save (which to keep vs. refresh),
        so the two never disagree about what "still good" means.
        """

        if not (record or {}).get("data"):
            return False
        try:
            stamp = datetime.strptime(
                (record or {}).get("fetched_at", ""), UPDATED_AT_STR_FORMAT,
            )
        except (TypeError, ValueError):
            return False
        return stamp >= datetime.now() - timedelta(days=ANILIST_CACHE_TTL_DAYS)

    def load_anilist_cache(self) -> None:
        """Seed the in-memory AniList cache from the persisted store

        AniList metadata (title / format / episodes / cover) is effectively
        static, so reusing what we fetched on previous runs is what keeps a run
        from re-querying AniList for ids it has already seen - the main cause of
        the rate-limit stalls. Entries older than ANILIST_CACHE_TTL_DAYS are
         skipped, so the data can't get arbitrarily stale (see prefetch_anilist /
        save_anilist_cache for the writing side).
        """

        meta = self.cache.get("anilist_meta", {})
        if not meta:
            return

        loaded = 0
        for id_str, record in meta.items():
            if not self._anilist_meta_is_fresh(record):
                continue
            try:
                self.al_cache[int(id_str)] = record["data"]
            except (TypeError, ValueError):
                continue
            loaded += 1

        if loaded:
            self.logger.debug(
                indent_string(f"Loaded {loaded} AniList entries from cache"),
            )

    def save_anilist_cache(self) -> None:
        """Persist any newly seen AniList responses back to the on-disk cache

        An entry that's already stored and still fresh keeps its original
        fetched_at (so the TTL actually expires it rather than resetting every
        run); a missing OR stale entry is (re)written with the current time, so
        an aged-out id is refreshed instead of being re-fetched on every run.
        """

        meta = self.cache.setdefault("anilist_meta", {})
        now_str = datetime.now().strftime(UPDATED_AT_STR_FORMAT)

        written = 0
        for al_id, data in self.al_cache.items():
            id_str = str(al_id)
            if self._anilist_meta_is_fresh(meta.get(id_str)):
                continue
            meta[id_str] = {"fetched_at": now_str, "data": data}
            written += 1

        # A preview keeps the warmed entries in memory but doesn't persist them,
        # so a preview leaves the on-disk cache untouched (gate lives in
        # save_cache).
        if written:
            self.save_cache()

    def prefetch_anilist(self, al_ids: Iterable[int]) -> None:
        """Warm the AniList cache for a set of ids in batched requests

        Fetches everything still missing from the cache in ANILIST_BATCH_SIZE-id
        "id_in" pages (one request per page) instead of one request per id on
        demand, then persists the results. This is what collapses a cold run's
        ~one-AniList-request-per-series into a handful, so the per-title loop
        rarely has to hit AniList one id at a time and trip its rate limit.

        Args:
            al_ids (iterable[int]): Candidate AniList IDs for this run
        """

        missing = sorted(
            {i for i in al_ids if i is not None and i not in self.al_cache},
        )
        if not missing:
            return

        # Surfaced at INFO (only when there's actually something to fetch, so
        # warm runs stay silent), so the upfront pause on a cold run is explained
        self.logger.info(
            indent_string(
                f"Prefetching {len(missing)} AniList entries "
                f"in batches of {ANILIST_BATCH_SIZE}",
            ),
            extra={"line_style": "grey50"},
        )

        for start in range(0, len(missing), ANILIST_BATCH_SIZE):
            chunk = missing[start:start + ANILIST_BATCH_SIZE]
            # Ids unknown to AniList are simply absent from the result; the
            # per-id helpers will try once more on demand and degrade gracefully
            for al_id, data in get_query_batch(chunk).items():
                self.al_cache[al_id] = data

        # Persist now (before the main loop) so the batch's work survives even an
        # early return - e.g., when max_torrents_to_add is hit mid-run
        self.save_anilist_cache()

    def get_mappings_from_anime_mappings(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: str = "movie",
        anilist_mappings: dict | None = None,
    ) -> dict:
        """Get mappings from the Anime ID mappings

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (str): TMDB type. Can be "movie" or "show"
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        if tmdb_type not in ["movie", "show"]:
            raise ValueError("tmdb_type must be 'movie' or 'show'")

        # Check we have exactly one ID specified here
        non_none_sum = sum(v is not None for v in [tvdb_id, tmdb_id, imdb_id])

        if non_none_sum == 0:
            raise ValueError(
                "At least one of tvdb_id, tmdb_id, and imdb_id must be provided",
            )

        index = self._anime_mappings_index
        if index is None:
            return anilist_mappings

        # Add the first mapping seen for each AniList id, matching the previous
        # "don't clobber an id another query already produced" behaviour
        def merge(field: str, value: Any) -> None:
            for m in index[field].get(value, ()):
                anilist_id = m["anilist_id"]
                if anilist_id not in anilist_mappings:
                    anilist_mappings[anilist_id] = m

        if tvdb_id is not None:
            merge("tvdb_id", tvdb_id)
        if tmdb_id is not None:
            merge(f"tmdb_{tmdb_type}_id", tmdb_id)
        if imdb_id is not None:
            merge("imdb_id", imdb_id)

        return anilist_mappings

    def get_mappings_from_anibridge_mappings(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: str = "movie",
        anilist_mappings: dict | None = None,
    ) -> dict:
        """Get mappings from the AniBridge mappings

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (str): TMDB type. Can be "movie" or "show"
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        anibridge = self.anibridge
        if not anibridge:
            return anilist_mappings

        if tmdb_type not in ["movie", "show"]:
            raise ValueError("tmdb_type must be 'movie' or 'show'")

        # Check we have exactly one ID specified here
        non_none_sum = sum(v is not None for v in [tvdb_id, tmdb_id, imdb_id])

        if non_none_sum == 0:
            raise ValueError(
                "At least one of tvdb_id, tmdb_id, and imdb_id must be provided",
            )

        # Add any AniList IDs the indexes resolve for the supplied ids, without
        # clobbering matches an earlier id already produced (tvdb > tmdb > imdb).
        def merge(found: dict) -> None:
            for anilist_id, entry in found.items():
                if anilist_id not in anilist_mappings:
                    anilist_mappings[anilist_id] = entry

        if tvdb_id is not None:
            merge(anibridge.lookup_by_tvdb(tvdb_id))
        if tmdb_id is not None:
            merge(anibridge.lookup_by_tmdb(tmdb_id, tmdb_type))
        if imdb_id is not None:
            merge(anibridge.lookup_by_imdb(imdb_id))

        return anilist_mappings

    def get_anilist_title(
        self,
        al_id: int,
    ) -> str:
        """Resolve and remember the AniList title for an ID (no logging)

        Fetches the title (via cache or a live AniList query) and stores it as
        the current title so later steps can attribute grabs to it. The entry
        header is logged separately by log_al_title, once episodes are known and
        the season/episode coverage can be shown.

        Args:
            al_id (int): AniList ID
        """

        anilist_title, self.al_cache = get_anilist_title(
            al_id,
            al_cache=self.al_cache,
        )

        # If the lookup came back empty (e.g., AniList was rate-limiting even
        # after retries), fall back to the id so the entry is still identifiable
        # rather than showing "None"
        if not anilist_title:
            anilist_title = f"AniList #{al_id}"

        self.current_title = anilist_title

        return anilist_title

    def get_seadex_dict(
        self,
        sd_entry: EntryRecord,
    ) -> dict:
        """Parse and filter SeaDex request

        Args:
            sd_entry: SeaDex API query
        """

        # The torrent records are only read here (a fresh dict is built per
        # release group below), so iterate them directly rather than deep-copying
        # the whole list of model objects on every entry.

        # Filter out any tags
        ignore_tags = set(self.ignore_tags)
        final_torrent_list = [
            t for t in sd_entry.torrents if ignore_tags.isdisjoint(t.tags)
        ]

        # Filter down by allowed trackers
        final_torrent_list = [
            t for t in final_torrent_list if t.tracker.casefold() in self.trackers
        ]

        # Pull out torrents tagged as best, so long as at least one
        # is tagged as best. Keep a copy so we can fall back if audio
        # preferences otherwise downgrade quality
        best_torrents = [t for t in final_torrent_list if t.is_best]
        any_best = len(best_torrents) > 0

        # Narrow to 'best' releases when any exist
        if self.want_best and any_best:
            candidates = best_torrents
        else:
            candidates = final_torrent_list

        # Prefer dual-audio releases, but only when at least one exists
        if self.prefer_dual_audio:
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
        seadex_release_groups = {}
        for t in candidates:

            if t.release_group not in seadex_release_groups:
                seadex_release_groups[t.release_group] = {"urls": {}}
                seadex_release_groups[t.release_group]["tags"] = t.tags

            seadex_release_groups[t.release_group]["urls"][t.url] = {
                "url": t.url,
                "files": [f.name for f in t.files],
                "size": [f.size for f in t.files],
                "tracker": t.tracker,
                "is_public": t.tracker.is_public() and t.tracker.casefold() not in PRIVATE_TRACKERS,
                "hash": t.infohash,
                "download": False,
            }

        # If we only want public releases, then within each release group drop
        # any private URLs, so long as that group also has a public option. We
        # deliberately do this per-group rather than across the whole list: a
        # group that only has a private URL is kept for now and only filtered
        # out later if the Arr doesn't already have a matching download (see
        # reduce_overlapping_downloads)
        if self.public_only:
            for release_group_item in seadex_release_groups.values():
                urls = release_group_item["urls"]
                has_public = any(u["is_public"] for u in urls.values())
                if has_public:
                    release_group_item["urls"] = {
                        url: u for url, u in urls.items() if u["is_public"]
                    }

        return seadex_release_groups

    def filter_seadex_interactive(
        self,
        seadex_dict: dict,
        sd_entry: EntryRecord,
    ) -> dict:
        """If multiple matches are found, let the user filter them interactively

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
            "Which release group(s)? Enter one number, a comma-separated list, "
            "or leave blank for all: ",
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
                except IndexError:
                    self.logger.warning(
                        indent_string(f"Index {srg_idx} is out of range"),
                    )
                    continue
                seadex_dict_filtered[srg] = copy.deepcopy(seadex_dict[srg])

            seadex_dict = copy.deepcopy(seadex_dict_filtered)

        return seadex_dict

    def get_seadex_fields(
        self,
        arr: str,
        al_id: int,
        release_group: list | str | None,
        seadex_dict: dict,
    ) -> tuple[list, str | None]:
        """Get fields for Discord post

        Args:
            arr: Type of arr instance
            al_id: AniList ID
            release_group: Arr release group
            seadex_dict: Dictionary of SeaDex releases
        """

        anilist_thumb, self.al_cache = get_anilist_thumb(
            al_id=al_id,
            al_cache=self.al_cache,
        )
        fields = []

        # The first field should be the Arr group. If it's empty, mention it's missing
        release_group_discord = copy.deepcopy(release_group)

        # Catch various edge cases: normalise to a non-empty list of strings
        if not release_group_discord:
            release_group_discord = ["None"]
        elif isinstance(release_group_discord, str):
            release_group_discord = [release_group_discord]

        field_dict = {
            "name": f"{arr.capitalize()} Release:",
            "value": "\n".join(release_group_discord),
        }
        fields.append(field_dict)

        # SeaDex options with links
        for srg, srg_item in seadex_dict.items():

            # URLs flagged for download in this group, in one pass
            urls_to_download = [
                url
                for url, u in srg_item.get("urls", {}).items()
                if u.get("download", False)
            ]

            if urls_to_download:

                # Include any tags in the string
                discord_value = ""
                tags = srg_item.get("tags", [])
                if len(tags) > 0:
                    discord_value += "Tags:\n"
                    discord_value += "\n".join(tags)
                    discord_value += "\n\n"

                # And include URLs for files we're downloading
                discord_value += "Links:\n"
                discord_value += "\n".join(urls_to_download)

                field_dict = {
                    "name": f"SeaDex recommendation: {srg}",
                    "value": f"{discord_value}",
                }

                fields.append(field_dict)

        return fields, anilist_thumb

    def filter_seadex_downloads(
        self,
        al_id: int,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        ep_list: list | None = None,
    ) -> tuple[list, dict]:
        """Flip the switch on whether we're downloading this torrent or not

        Args:
            al_id: AniList ID
            seadex_dict: Dictionary of SeaDex releases
            arr: Type of arr instance
            arr_release_dict: Dictionary of arr release properties
            ep_list: List of episodes. Defaults to None
        """

        if self.use_torrent_hash_to_filter:
            torrent_hashes, seadex_dict = self.filter_by_torrent_hash(
                al_id=al_id,
                seadex_dict=seadex_dict,
                arr=arr,
            )
        else:
            torrent_hashes, seadex_dict = self.filter_by_release_group(
                seadex_dict=seadex_dict,
                arr=arr,
                arr_release_dict=arr_release_dict,
                ep_list=ep_list,
            )

            # Also include any cached hashes
            cached_hashes = (
                self.cache.get("anilist_entries", {})
                .get(arr, {})
                .get(str(al_id), {})
                .get("torrent_hashes", [])
            )
            torrent_hashes.extend(cached_hashes)

        # Make sure the hashes are unique
        torrent_hashes = list(set(torrent_hashes))

        return torrent_hashes, seadex_dict

    def filter_by_torrent_hash(
        self,
        al_id: int,
        seadex_dict: dict,
        arr: str,
    ) -> tuple[list, dict]:
        """Select downloads if the torrent hash is not already in the cache

        Multiple "best" releases are all grabbed, except where several cover
        the same files (see reduce_overlapping_downloads), in which case only
        one is kept

        Args:
            al_id: AniList ID
            seadex_dict: Dictionary of SeaDex releases
            arr: Type of arr instance
        """

        cached_hashes = (
            self.cache.get("anilist_entries", {})
            .get(arr, {})
            .get(str(al_id), {})
            .get("torrent_hashes", [])
        )
        torrent_hashes = []

        for seadex_rg, seadex_rg_item in seadex_dict.items():

            self.logger.debug(
                indent_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.get("urls", {})
            for url_item in seadex_urls.values():

                url_hash = url_item.get("hash", None)

                # If the URL is already in the hash cache, then append but don't set to download
                torrent_hashes.append(url_hash)
                if url_hash not in cached_hashes:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {url_hash} not found in cache. "
                            f"Will add to downloads",
                        ),
                    )

                    url_item.update({"download": True})

                else:
                    self.logger.debug(
                        indent_string(
                            f"Torrent hash {url_hash} in cache. Will skip download",
                        ),
                    )

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        return torrent_hashes, seadex_dict

    def filter_by_release_group(
        self,
        seadex_dict: dict,
        arr: str,
        arr_release_dict: dict,
        ep_list: list | None = None,
    ) -> tuple[list, dict]:
        """Filter torrents by release group

        This is either an episode-by-episode for the Sonarr
        case where we can parse episodes, or a more blunt
        hammer just checking against anything for Radarr
        and weirdly named TV

        Args:
            seadex_dict: Dictionary of SeaDex releases
            arr: Type of arr instance
            arr_release_dict: Dictionary of arr release properties
            ep_list: List of episodes. Defaults to None
        """

        # The release-group names, used both for display (insertion order
        # preserved) and for membership tests below. A dict keys view already
        # supports `in` in O(1), so there's no need to materialize a list.
        arr_release_groups = arr_release_dict.keys()

        # And also just check if any release group matches
        # any Arr release tag
        seadex_keys = set(seadex_dict.keys())
        overlapping_results = any(rg in seadex_keys for rg in arr_release_groups)

        # Index the Sonarr episodes by (season, episode) once, shared by both
        # the overlap map below and the per-episode match loop: looking up a
        # parsed SeaDex (season, episode) is then an O(1) dict op rather than a
        # fresh scan of the whole list. First entry wins on a duplicate key
        # (Sonarr episodes are unique by season+episode).
        sonarr_by_key: dict = {}
        for sonarr_ep in ep_list or []:
            sonarr_by_key.setdefault(
                (
                    sonarr_ep.get("seasonNumber", 999),
                    sonarr_ep.get("episodeNumber", 999),
                ),
                sonarr_ep,
            )

        # If we have overlaps, get a note of them here, reusing the index above
        all_seadex_rgs_per_episode = get_all_seadex_rgs_per_episode(
            seadex_dict=seadex_dict,
            sonarr_by_key=sonarr_by_key,
        )

        # Resolve once: the per-episode debug lines below sit in the hot
        # matching loop, so this lets us skip building their f-strings on a
        # normal INFO run instead of formatting them only to discard them.
        debug_on = self.logger.isEnabledFor(logging.DEBUG)

        for seadex_rg, seadex_rg_item in seadex_dict.items():

            self.logger.debug(
                indent_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.get("urls", {})
            for url, url_item in seadex_urls.items():

                seadex_episodes = url_item.get("episodes", [])

                # Simple case, we have no episode mappings, so
                # just fall back to checking against release group
                if len(seadex_episodes) == 0:
                    if seadex_rg not in arr_release_groups and not overlapping_results:
                        self.logger.debug(
                            indent_string(
                                f"SeaDex release group {seadex_rg} not in {arr.capitalize()} releases: "
                                f"{', '.join([str(x) for x in arr_release_groups])} - will download {url}",
                            ),
                        )

                        url_item.update({"download": True})

                    # If the group matches, fall through to a size comparison
                    if seadex_rg in arr_release_groups:

                        seadex_file_sizes = url_item.get("size", [])
                        arr_file_sizes = arr_release_dict[seadex_rg].get("size", [])

                        if not isinstance(arr_file_sizes, list):
                            arr_file_sizes = [arr_file_sizes]

                        # If we have no overlaps at all, then add
                        if set(seadex_file_sizes).isdisjoint(arr_file_sizes):
                            self.logger.debug(
                                indent_string(
                                    f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                                    f"{', '.join([str(x) for x in arr_release_groups])}, but file sizes do not match - will download {url}",
                                ),
                            )

                            url_item.update({"download": True})

                        else:
                            self.logger.debug(
                                indent_string(
                                    f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                                    f"{', '.join([str(x) for x in arr_release_groups])}, and file sizes match",
                                ),
                            )

                else:

                    # At this point, we need an episode list from Sonarr
                    if ep_list is None:
                        self.logger.debug(
                            "Skipping per-episode check: no Sonarr episode list available",
                        )
                        continue

                    # For each episode we've parsed from the torrent, check if a) it exists in the Sonarr list, b) if
                    # the release group matches, and c) if the file sizes match. If there's any mismatch between release
                    # groups (and there are no alternatives), then flip download to True. If all the sizes mismatch,
                    # flip download to true

                    rg_matches = [False] * len(seadex_episodes)
                    size_matches = [False] * len(seadex_episodes)

                    for seadex_idx, seadex_ep in enumerate(seadex_episodes):

                        seadex_ep_season = seadex_ep.get("season", 888)
                        seadex_ep_episode = seadex_ep.get("episode", 888)
                        seadex_ep_size = seadex_ep.get("size", None)

                        # O(1) lookup into the indexed Sonarr episodes instead of
                        # re-scanning the whole list for every parsed episode
                        sonarr_ep = sonarr_by_key.get(
                            (seadex_ep_season, seadex_ep_episode),
                        )
                        if sonarr_ep is None:
                            continue

                        # Get the matched Sonarr episode's file size
                        sonarr_ep_size = sonarr_ep.get("episodeFile", {}).get(
                            "size", None,
                        )

                        # Do the sizes match?
                        size_match = sonarr_ep_size == seadex_ep_size

                        season_ep_str = (
                            f"S{seadex_ep_season:02d}E{seadex_ep_episode:02d}"
                        )

                        # Check SeaDex release group matches the episode release group in Sonarr
                        sonarr_rg = sonarr_ep.get("episodeFile", {}).get(
                            "releaseGroup", None,
                        )
                        sonarr_rg_normalized = normalize_rg(sonarr_rg)
                        seadex_rg_normalized = normalize_rg(seadex_rg)
                        # If not, flag as should be downloaded if it's not
                        # already in some overlapping release.
                        # normalized name indexes all_seadex_rgs_per_episode, so compare the normalized name
                        if (
                            sonarr_rg_normalized != seadex_rg_normalized
                            and sonarr_rg_normalized
                            not in all_seadex_rgs_per_episode["all"]
                        ):

                            # Avoid duplicating when another release already covers it
                            all_seadex_rg = all_seadex_rgs_per_episode.get(
                                season_ep_str, (),
                            )

                            if sonarr_rg_normalized not in all_seadex_rg:
                                if debug_on:
                                    self.logger.debug(
                                        indent_string(
                                            f"SeaDex release group {seadex_rg} differs from "
                                            f"{arr.capitalize()} release for "
                                            f"{season_ep_str} ({sonarr_rg}) and no other "
                                            f"recommended release covers it - will download {url}",
                                        ),
                                    )

                                url_item.update({"download": True})

                        else:

                            if debug_on:
                                self.logger.debug(
                                    indent_string(
                                        f"Found SeaDex match to {arr.capitalize()} "
                                        f"for {season_ep_str}.",
                                    ),
                                )
                                if not size_match:
                                    self.logger.debug(
                                        indent_string(
                                            f"-> Sizes are different: "
                                            f"{sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
                                        ),
                                    )
                                else:
                                    self.logger.debug(
                                        indent_string(
                                            f"-> Sizes match: {sonarr_ep_size}",
                                        ),
                                    )

                            rg_matches[seadex_idx] = True

                        # Now check against file size
                        if size_match:
                            size_matches[seadex_idx] = True

                    # If we have matched the release groups but not the file sizes, then flag that
                    # here and mark for download
                    size_matches = list(compress(size_matches, rg_matches))
                    if not any(size_matches) and len(size_matches) > 0:
                        self.logger.debug(
                            indent_string(
                                f"File sizes all differ for release group {seadex_rg} - will download {url}",
                            ),
                        )
                        url_item.update({"download": True})

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        self.reduce_overlapping_downloads(seadex_dict=seadex_dict)

        # Build the hash list from whatever is still flagged for download, so it
        # always matches the exact set of torrents we'll add. Private torrents
        # have no infohash, so skip those
        torrent_hashes = [
            url_item["hash"]
            for rg_item in seadex_dict.values()
            for url_item in rg_item.get("urls", {}).values()
            if url_item.get("download", False) and url_item.get("hash") is not None
        ]

        return torrent_hashes, seadex_dict

    def reduce_overlapping_downloads(
        self,
        seadex_dict: dict,
    ) -> None:
        """Reduce overlapping flagged downloads down to a single release group

        Where multiple preferred release groups cover the same files and the
        Arr doesn't already have any of them, we only want to grab one. If
        public_only is set, we prefer a public release group and drop the
        private ones. If the only options are private, we log an error and skip
        the title (without caching it as done) rather than grabbing a private
        release.

        Mutates the download flags on seadex_dict in place. Skipped entirely in
        interactive mode, where the user has already hand-picked what to grab.

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        # In interactive mode the user has explicitly chosen which releases to
        # grab, so don't second-guess them by dropping any
        if self.interactive:
            return

        def is_flagged(rg_item: dict) -> bool:
            return any(
                u.get("download", False) for u in rg_item.get("urls", {}).values()
            )

        def is_public_group(rg_item: dict) -> bool:
            return any(
                u.get("is_public", False) for u in rg_item.get("urls", {}).values()
            )

        def unflag(rg_item: dict) -> None:
            for u in rg_item.get("urls", {}).values():
                u["download"] = False

        same_files_groups = get_same_files_groups(seadex_dict)

        for same_files in same_files_groups:

            # Only the release groups the Arr doesn't already have are flagged
            flagged = [rg for rg in same_files if is_flagged(seadex_dict[rg])]
            if len(flagged) == 0:
                continue

            if self.public_only:
                public_flagged = [
                    rg for rg in flagged if is_public_group(seadex_dict[rg])
                ]

                if len(public_flagged) == 0:
                    # The Arr has none of these release groups, public_only is
                    # set, but none are available on a public tracker. Don't
                    # grab a private release, just log an error and skip. Flag
                    # the skip so the caller doesn't cache the title as done
                    self.log_fmt.detail(
                        "skipped",
                        f"{', '.join(flagged)} private-only (public_only on)",
                        value_style="yellow",
                        level=logging.WARNING,
                    )
                    self.public_only_skipped = True
                    self.public_only_groups.extend(flagged)
                    for rg in flagged:
                        unflag(seadex_dict[rg])
                    continue

                # Keep the first public release group, drop everything else
                keeper = public_flagged[0]
            else:
                # We don't care about public/private, just keep the first one
                keeper = flagged[0]

            for rg in flagged:
                if rg == keeper:
                    continue

                self.logger.debug(
                    indent_string(
                        f"Not downloading release group {rg}: release group "
                        f"{keeper} already covers the same files",
                    ),
                )
                unflag(seadex_dict[rg])

    @staticmethod
    def get_any_to_download(seadex_dict: dict) -> bool:
        """Check if any torrents are marked as to download

        Args:
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        return any(
            url_item.get("download", False)
            for rg_item in seadex_dict.values()
            for url_item in rg_item.get("urls", {}).values()
        )

    @staticmethod
    def format_episode_coverage(episodes: list) -> list | None:
        """Per-season season/episode coverage tuples for a torrent.

        Thin wrapper over :func:`coverage.format_episode_coverage`, relocated in
        the Phase 1 decomposition; kept on the class so subclasses can call it
        via ``self``.
        """

        return _coverage.format_episode_coverage(episodes)

    def coverage_string(self, episodes: list) -> str:
        """One-line season/episode coverage, e.g. "S04 E01-E12".

        Thin wrapper over :func:`coverage.coverage_string`.
        """

        return _coverage.coverage_string(episodes)

    @staticmethod
    def episodes_from_ep_list(ep_list: list | None, missing_only: bool = False) -> list:
        """Convert a Sonarr ep_list into {"season","episode"} coverage dicts.

        Thin wrapper over :func:`coverage.episodes_from_ep_list`.
        """

        return _coverage.episodes_from_ep_list(ep_list, missing_only=missing_only)

    def add_torrent(
        self,
        torrent_dict: dict,
        torrent_client: str = "qbit",
    ) -> tuple[int, list]:
        """Add torrent(s) to a torrent client

        The per-release outcome lines (added / kept) are NOT logged here; this
        returns them so the caller (log_seadex_action) can print the whole block
        in order with a status that reflects what actually happened - "adding" if
        anything was grabbed, "keeping" if every recommended release was already
        present. The "skipped" warnings (private-only, unselected tracker) are
        still logged inline, as they're independent of that status.

        Args:
            torrent_dict (dict): Dictionary of torrent info
            torrent_client (str): Torrent client to use. Options are
                "qbit" for qBittorrent. Defaults to "qbit"

        Returns:
            tuple: (n_torrents_added, results), where results is a list of
                {"outcome": "added" | "already have", "name": str, "group": str}
                dicts, one per release acted on, in order
        """

        n_torrents_added = 0
        results = []

        for srg, srg_item in torrent_dict.items():

            seadex_urls = srg_item.get("urls", {})
            for url, url_item in seadex_urls.items():

                # If not flagged for download, then skip
                download = url_item.get("download", False)
                if not download:
                    continue

                item_hash = url_item.get("hash", None)
                tracker = url_item.get("tracker", None)

                if self.public_only and not url_item.get("is_public", True):
                    self.log_fmt.detail(
                        "skipped",
                        f"{tracker} private-only (public_only on)",
                        value_style="yellow",
                        level=logging.WARNING,
                    )
                    self.public_only_skipped = True
                    self.public_only_groups.append(srg)
                    continue

                # Skip trackers not in the user's selected list
                if tracker.casefold() not in self.trackers:
                    self.log_fmt.detail(
                        "skipped",
                        f"{url} (tracker {tracker} not in your selected list)",
                        value_style="yellow",
                    )
                    continue

                # Each parser returns the download/magnet link plus the release's
                # human-readable title scraped from the source page, so we always
                # have a real name to show even when the client can't report one
                # (e.g. a private torrent with no info hash, or a dry run)

                # Nyaa
                if tracker.lower() == "nyaa":
                    parsed_url, source_name = get_nyaa_torrent(url=url)

                # AnimeToshio
                elif tracker.lower() == "animetosho":
                    parsed_url, source_name = get_animetosho_torrent(
                        url=url,
                        session=self.session,
                    )

                # RuTracker
                elif tracker.lower() == "rutracker":
                    parsed_url, source_name = get_rutracker_torrent(
                        url=url,
                        torrent_hash=item_hash,
                        session=self.session,
                    )

                # Otherwise, bug out
                else:
                    raise ValueError(f"Unable to parse torrent links from {tracker}")

                if parsed_url is None:
                    raise Exception("Have not managed to parse the torrent URL")

                if torrent_client == "qbit":
                    success, torrent_name = self.add_torrent_to_qbit(
                        url=url,
                        torrent_url=parsed_url,
                        torrent_hash=item_hash,
                    )

                else:
                    raise ValueError(f"Unsupported torrent client {torrent_client}")

                # Prefer the name qBittorrent reports; fall back to the release's
                # title from the source page rather than the raw download link
                if not torrent_name:
                    torrent_name = source_name

                if success == "torrent_added":
                    results.append(
                        {"outcome": "added", "name": torrent_name, "group": srg},
                    )

                    # Record the grab for the end-of-run summary. Prefer the
                    # release's own parsed file list (precise for multi-cour /
                    # per-torrent grabs); fall back to the entry-level coverage we
                    # mapped from the Arr so the summary's "files" is never blank
                    # when a release's filenames couldn't be parsed (e.g. an OVA).
                    coverage_str = self.coverage_string(
                        url_item.get("episodes", []),
                    ) or self.current_coverage
                    self.stats["added"].append(
                        {
                            "title": self.current_title,
                            "coverage": coverage_str,
                            "url": self.current_url,
                            "name": torrent_name,
                            "group": srg,
                        },
                    )

                    # Stop once max_torrents_to_add is reached
                    self.torrents_added += 1
                    n_torrents_added += 1
                    if self.max_torrents_to_add is not None:
                        if self.torrents_added >= self.max_torrents_to_add:
                            return n_torrents_added, results

                elif success == "torrent_already_added":
                    results.append(
                        {"outcome": "already have", "name": torrent_name, "group": srg},
                    )

                else:
                    raise ValueError(f"Cannot handle torrent client {torrent_client}")

        return n_torrents_added, results

    def add_torrent_to_qbit(
        self,
        url: str,
        torrent_url: str,
        torrent_hash: str | None,
    ) -> tuple[str, str | None]:
        """Add a torrent to qbittorrent

        Args:
            url (str): SeaDex URL
            torrent_url (str): Torrent URL to add to a client
            torrent_hash (str): Torrent hash

        Returns:
            tuple: (status, torrent_name), where status is one of
                "torrent_added" or "torrent_already_added", and torrent_name is
                the name reported by qBittorrent (or None if the torrent has no
                info hash to look up, in which case the caller uses the URL)
        """

        # A private torrent has no info hash, so we can't look it up by hash to
        # dedup or to read its name back; just add it and let qBittorrent dedup
        # internally. With a hash, skip the adding if it's already present
        if torrent_hash is not None and self.qbit is not None:
            torr_info = self.qbit.torrents_info(torrent_hashes=torrent_hash)
            torr_hashes = [i.hash for i in torr_info]

            if torrent_hash in torr_hashes:
                self.logger.debug(
                    indent_string(f"Torrent {url} already in qBittorrent"),
                )
                return "torrent_already_added", torr_info[0].name

        # Preview (dry run or no client): report it as added without touching the
        # client. With a client present the dedup lookup above still ran, so an
        # already-present torrent is reported accurately. There's no client-side
        # name to read back, so the caller falls back to the URL.
        if self._is_preview():
            return "torrent_added", None

        # Add the torrent
        result = self.qbit.torrents_add(
            urls=torrent_url,
            category=self.torrent_category,
            tags=self.torrent_tags,
        )
        if result != "Ok.":
            raise Exception("Failed to add torrent")

        # Look the torrent back up by hash so we can report its name. A private
        # torrent has no info hash to look up, so leave the name unset and let
        # the caller fall back to the URL
        torrent_name = None
        if torrent_hash is not None:
            added_info = self.qbit.torrents_info(torrent_hashes=torrent_hash)
            torrent_name = added_info[0].name if added_info else None

        return "torrent_added", torrent_name

    def _is_preview(self) -> bool:
        """A run is a no-op preview when an explicit dry run was requested OR
        qBittorrent is not configured (nothing can actually be grabbed)."""
        return self.dry_run or self.qbit is None

    def save_cache(self, sort: bool = True) -> None:
        """Persist the in-memory cache to disk, unless this run is a preview.

        Args:
            sort (bool): Sort anilist_entries by id before writing. Defaults to
                True so the persisted file is ordered by id; pass False to skip
                the sort on a hot write path.
        """

        self.cache_store.save(preview=self._is_preview(), sort=sort)

    def update_cache(self, arr: str, al_id: int, cache_details: dict | None = None) -> bool:
        """Merge ``cache_details`` into an entry's cache record (in-memory only).

        The run's save points flush it; see ``CacheStore.update_cache``.

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            cache_details (dict): Details for the cache entry. Defaults to None
        """

        return self.cache_store.update_cache(arr, al_id, cache_details)

    @staticmethod
    def _fresh_stats() -> dict:
        """Build an empty per-run stats tally for the end-of-run summary"""

        return {
            "checked": 0,
            "added": [],  # list of {"title", "coverage", "url", "name", "group"}
            "up_to_date": 0,
            "cached": 0,
            "no_seadex_entry": 0,
            "no_releases": 0,
            "no_mappings": 0,
            "needs_action": [],  # list of {"title", "reason"}
            "unmonitored": 0,
        }

    def reset_run_stats(self) -> bool:
        """Reset the per-run tally and start the run clock

        Warning/error counts are read from the logger-level counter by diffing
        a snapshot taken here against one taken when the summary is logged.
        """

        self.stats = self._fresh_stats()
        self.torrents_added = 0
        # Drop any episode lists cached from a previous run so a fresh run always
        # re-reads the current Sonarr library
        self._ep_list_cache = {}
        # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
        self._run_started_monotonic = time.monotonic()
        counter = getattr(self.logger, "seadex_counter", None)
        self._log_counts_at_start = counter.snapshot() if counter else {}

        return True

    # --- Run orchestration (shared template) --------------------------------
    #
    # Each subclass's run() is a thin wrapper over run_sync: both Arrs share the
    # whole scaffolding (reset stats, fetch items, optional single-id filter,
    # AniList prefetch, the per-item loop, and the end-of-run save + summary) and
    # differ only in how an item is fetched/identified and what the per-AniList-id
    # body does. The divergent pieces are the hooks (_get_all_items,
    # _filter_to_single_item, _item_anilist_ids, _process_al_id); the identical
    # per-id head and grab/cache tail are _al_id_prologue and _grab_and_cache.

    @abstractmethod
    def _get_all_items(self) -> list:
        """Fetch every Arr item (movie/series) that has an AniList mapping."""

    @abstractmethod
    def _filter_to_single_item(self, items: list, item_id: int) -> list:
        """Narrow the item list to the one matching a single CLI-supplied id."""

    @abstractmethod
    def _item_anilist_ids(self, item: Any, log_ignored: bool = True) -> dict:
        """Resolve the AniList ids an Arr item maps to (arr-specific id args)."""

    @abstractmethod
    def _process_al_id(
        self,
        arr: str,
        item: Any,
        item_title: str,
        al_id: int,
        mapping: dict,
    ) -> bool:
        """Handle one AniList id for an item; return True to stop the whole run.

        The per-id middle is the genuinely Arr-specific part (Radarr's single
        file vs. Sonarr's episode coverage); the shared head and tail live in
        _al_id_prologue and _grab_and_cache.
        """

    def run_sync(self, arr: str, item_id: int | None, dry_run: bool) -> bool:
        """Shared run scaffolding for both Arr syncers

        Args:
            arr (str): "radarr" or "sonarr"
            item_id (int | None): If set, only run for the single item with this
                id (TMDB for Radarr, TVDB for Sonarr)
            dry_run (bool): Simulate the run without grabbing torrents, writing
                the cache, or sending notifications
        """

        # Whether this is a no-op preview - consulted by the mutating helpers
        self.dry_run = dry_run

        # Reset the per-run tally and start the run clock
        self.reset_run_stats()

        all_items = self._get_all_items()

        # If we're targeting a single item, filter down to it
        if item_id is not None:
            all_items = self._filter_to_single_item(all_items, item_id)

        n_items = len(all_items)

        self.log_arr_start(arr=arr, n_items=n_items)

        # Warm the AniList cache before the per-item loop: reuse what past runs
        # fetched, then batch-fetch (id_in pages) everything still missing, so the
        # loop rarely hits AniList one id at a time and trips its rate limit.
        self.load_anilist_cache()
        prefetch_ids = set()
        for item in all_items:
            if not item.monitored and self.ignore_unmonitored:
                continue
            prefetch_ids.update(
                self._item_anilist_ids(item, log_ignored=False),
            )
        self.prefetch_anilist(prefetch_ids)

        for item_idx, item in enumerate(all_items):

            try:

                item_title = item.title

                self.log_arr_item_start(
                    arr=arr,
                    item_title=item_title,
                    n_item=item_idx + 1,
                    n_items=n_items,
                )

                # If we're not monitored, then skip if ignore_unmonitored is switched on
                if not item.monitored and self.ignore_unmonitored:
                    self.log_arr_item_unmonitored(item_title=item_title)
                    continue

                # Get the mappings from the Arr item to AniList
                al_mappings = self._item_anilist_ids(item)

                if len(al_mappings) == 0:
                    self.log_no_anilist_mappings(title=item_title)
                    continue

                for al_id, mapping in al_mappings.items():
                    # _process_al_id returns True only when max_torrents_to_add was
                    # reached - it has already saved the cache and logged the
                    # summary - so stop the whole run here. The original per-item
                    # post-loop max check is redundant with this early return (the
                    # in-block check fires after every add, so torrents_added can
                    # never reach the cap without _process_al_id stopping first),
                    # so it isn't repeated.
                    if self._process_al_id(
                        arr=arr,
                        item=item,
                        item_title=item_title,
                        al_id=al_id,
                        mapping=mapping,
                    ):
                        return True

            except Exception as e:
                title = getattr(item, "title", "unknown title")
                self.logger.error(
                    f"{title}: unexpected error: {e}", exc_info=True,
                )
                continue

        # Per-title update_cache calls only mutate memory now, so this end-of-run
        # save is what actually persists the run (and sorts by id on the way out)
        self.save_cache()
        self.log_run_summary(arr=arr)

        return True

    def _al_id_prologue(self, al_id: int | None) -> EntryRecord | None:
        """Shared per-AniList-id head: reset skip flags, tally, fetch SeaDex entry

        Returns the SeaDex entry to process, or None when the id should be
        skipped (no id, or no SeaDex entry) - the caller moves to the next id.

        Args:
            al_id (int | None): AniList id being processed; defensively None-checked
                since the mapping dicts are built from external data
        """

        # Reset the per-title public_only skip flag (and the skipped group names)
        # before we make any download decisions for this title
        self.public_only_skipped = False
        self.public_only_groups = []
        self.stats["checked"] += 1

        if al_id is None:
            self.log_no_anilist_id()
            return None

        # Get the SeaDex entry if it exists
        sd_entry = self.get_seadex_entry(al_id=al_id)
        if sd_entry is None:
            self.log_no_sd_entry(al_id=al_id)
            return None

        return sd_entry

    def _cached_entry_skip(
        self,
        arr: str,
        al_id: int,
        sd_entry: EntryRecord,
        sd_url: str,
        coverage: Callable[[], str],
    ) -> bool:
        """Shared cached-entry short-circuit for both Arr runners

        When the id is already cached and we're honoring SeaDex update times,
        backfill the url + coverage on legacy records that predate those fields,
        log the cached entry, and return True so the caller skips it. ``coverage``
        is a zero-arg callable so the (for Sonarr, episode-fetching) coverage
        lookup runs only on the one-time backfill, never on the common
        already-backfilled path.

        Args:
            arr (str): "radarr" or "sonarr"
            al_id (int): AniList id being processed
            sd_entry (EntryRecord): Resolved SeaDex entry
            sd_url (str): SeaDex entry URL stored on the backfilled record
            coverage (Callable[[], str]): Lazily builds the coverage string for
                the backfill ("" for a movie, a season/episode range for a series)
        """

        if not self.check_al_id_in_cache(arr=arr, al_id=al_id, seadex_entry=sd_entry):
            return False
        if self.ignore_seadex_update_times:
            return False

        # Backfill the enriched fields for records written before they existed,
        # so cached rows can still link to SeaDex (and, for series, show the
        # season/episode coverage). One-time per old entry.
        if not self.get_cached_field(arr, al_id, "url"):
            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details={"url": sd_url, "coverage": coverage()},
            )
        self.log_cached_entry(arr=arr, al_id=al_id)
        return True

    def _grab_and_cache(
        self,
        arr: str,
        al_id: int,
        item_title: str,
        anilist_title: str,
        sd_url: str,
        seadex_dict: dict,
        torrent_hashes: list,
        cache_details: dict,
        release_group: list | str | None,
    ) -> bool:
        """Shared per-id tail: add torrents, notify, then cache the outcome

        Identical across both Arrs once the (Arr-specific) seadex_dict and
        release-group info have been resolved. Returns True only when
        max_torrents_to_add has been reached (cache saved and summary logged),
        so the caller stops the whole run; otherwise False (move to the next id).

        Args:
            arr (str): "radarr" or "sonarr"
            al_id (int): AniList id being processed
            item_title (str): Arr item title (Discord notification heading)
            anilist_title (str): Resolved AniList title (non-None; Discord field)
            sd_url (str): SeaDex entry URL (non-None; Discord field)
            seadex_dict (dict): Filtered SeaDex releases
            torrent_hashes (list): Hashes to remember in the cache record
            cache_details (dict): Cache record being assembled for this id
            release_group (list | str | None): Arr release group(s) for the
                Discord fields
        """

        # Check the release groups are matching, and get a bespoke list of torrents
        any_to_download = self.get_any_to_download(seadex_dict=seadex_dict)

        # Capture the running total before the add block so we can tell whether
        # THIS title actually grabbed anything
        torrents_before = self.torrents_added

        if any_to_download:
            fields, anilist_thumb = self.get_seadex_fields(
                arr=arr,
                al_id=al_id,
                release_group=release_group,
                seadex_dict=seadex_dict,
            )

            # Add torrents to qBittorrent. add_torrent runs even in a preview
            # (no client / dry run): add_torrent_to_qbit simulates the add, while
            # the download-flag, public_only and tracker filters still apply, so
            # only releases that would actually be grabbed are counted.
            n_torrents_added, results = self.add_torrent(
                torrent_dict=seadex_dict,
                torrent_client="qbit",
            )

            # Log the action block now the outcome is known, so the status reads
            # "adding" only when something was actually grabbed (else "keeping")
            self.log_seadex_action(
                seadex_dict=seadex_dict,
                results=results,
                dry_run=self._is_preview(),
            )

            # Push a message to Discord if we've added anything (never on a
            # preview - it's an outward notification)
            if (
                self.discord_url is not None
                and n_torrents_added > 0
                and not self._is_preview()
            ):
                discord_push(
                    url=self.discord_url,
                    arr_title=item_title,
                    al_title=anilist_title,
                    seadex_url=sd_url,
                    fields=fields,
                    thumb_url=anilist_thumb,
                )

            if self.max_torrents_to_add is not None:
                if self.torrents_added >= self.max_torrents_to_add:
                    self.log_max_torrents_added()
                    self.save_cache()
                    self.log_run_summary(arr=arr)
                    return True

        elif not self.public_only_skipped:
            self.stats["up_to_date"] += 1
            self.log_fmt.detail(
                "status",
                "already have the recommended release",
                value_style="blue",
            )

        # Work out whether THIS title actually grabbed anything
        added_this_title = self.torrents_added - torrents_before

        # Update and save out the cache whenever something was grabbed for this
        # title, or when nothing was skipped at all. Leave the title uncached ONLY
        # when public_only skipped a release AND nothing else was grabbed for it -
        # so it's re-checked (and the skip re-logged as a reminder) on every run,
        # and retried once a public release appears or public_only is relaxed
        if added_this_title > 0 or not self.public_only_skipped:
            cache_details.update({"torrent_hashes": torrent_hashes})
            self.update_cache(
                arr=arr,
                al_id=al_id,
                cache_details=cache_details,
            )
        elif added_this_title == 0:
            # Record the private-only skip for the summary's "needs action" list,
            # attributed to this title - but only when nothing was actually added
            # for it. The coverage is whatever log_al_title recorded as current
            # (a season/episode string for Sonarr, None for a Radarr movie).
            self.stats["needs_action"].append(
                {
                    "title": self.current_title,
                    "coverage": self.current_coverage,
                    "group": ", ".join(
                        dict.fromkeys(self.public_only_groups),
                    ),
                    "url": self.current_url,
                    "reason": "private-only release; public_only on",
                },
            )

        # Add in a wait, if required
        time.sleep(self.sleep_time)

        return False

    def log_run_summary(self, arr: str) -> bool:
        """Log the end-of-run scoreboard for an Arr run

        Args:
            arr (str): Type of arr instance
        """

        if arr not in ALLOWED_ARRS:
            raise ValueError(f"arr must be one of: {ALLOWED_ARRS}")

        stats = self.stats

        # Warning/error counts come from the logger-level counter, diffed
        # against the snapshot taken when the run started
        counter = getattr(self.logger, "seadex_counter", None)
        now_counts = counter.snapshot() if counter else {}
        start_counts = self._log_counts_at_start

        def _delta(level: int) -> int:
            return now_counts.get(level, 0) - start_counts.get(level, 0)

        n_warnings = _delta(logging.WARNING)
        n_errors = _delta(logging.ERROR) + _delta(logging.CRITICAL)

        title = f"SeaDexArr ({arr.capitalize()}) run complete"
        # State dry-run once, here, scoping the whole summary - rather than also
        # tagging the "added" value (the same fact twice in one block). The file
        # log keeps the plain title; the annotation rides the console rule_title.
        rule_title = title
        # A run grabs nothing when explicitly flagged dry, or when no client is
        # configured at all - annotate (and later dim) the summary either way.
        is_dry_run = self._is_preview()
        if is_dry_run:
            note = "nothing grabbed" if self.qbit is not None else (
                "no client; nothing grabbed"
            )
            rule_title += f"   (DRY RUN — {note})"
        self.logger.info(
            title,
            extra={
                "rule_title": rule_title,
                "rule_style": "bold cyan",
                "rule_heavy": True,
            },
        )

        # The summary's key column is narrower than the per-title detail column:
        # "needs action" (12) is the widest key here, vs. "missing episodes" (16)
        # in entry details. A heavy rule separates the two blocks, so the differing
        # colon columns never sit adjacent. Wrap the formatter to fix width at 12.
        def summary_kv(key: str, value: Any, **kwargs: Any) -> bool:
            return self.log_fmt.kv(key, value, key_width=12, **kwargs)

        # A needs-action entry in the summary, rendered with the same labeled
        # gutter as added_detail so the two blocks read alike: the title hangs at
        # indent 2, then fixed fields sit at indent 3 beneath it. Unlike a grab
        # there's no torrent name to lean on, so the skipped private release
        # group IS named here. The whole block is yellow - it's the one section
        # asking the user to do something. The title is shown in full; it sits on
        # its own line above the fixed fields, so its length can't break the column.
        def _summary_block(title: str, title_style: str | None, rows: list) -> None:
            # Shared layout for the summary's per-entry blocks: the title hangs
            # at indent 2, then labeled gutter fields sit beneath it at indent 3,
            # their values landing in the same column as the live "checking"
            # block. Each row carries its already-resolved accent.
            self.logger.info(
                indent_string(title, level=2),
                extra={"line_style": title_style},
            )
            for label, value, accent in rows:
                if not value:
                    continue
                self.log_fmt.kv(
                    label,
                    value,
                    value_style=accent,
                    indent=3,
                    key_width=7,
                    sep="",
                )

        def needs_detail(item: dict) -> None:
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("group", item.get("group"), "yellow"),
                ("reason", item.get("reason"), "yellow"),
                ("link", item.get("url"), "grey50"),
            ]
            _summary_block(item.get("title") or "(unknown title)", "yellow", rows)

        # A grab in the summary, rendered like the live per-entry "checking"
        # block: the title hangs at indent 2, then labeled gutter fields sit
        # beneath it at indent 3, their values landing in the same column (14) as
        # the live block. The grab is labeled "torrent" rather than "added" since
        # the whole section is already the added list. The recommended group is
        # called out at the front of the torrent name - highlighted in place when
        # the name already leads with it, or prepended in brackets otherwise - so
        # the group always reads first. A dry run dims the whole block (group accent
        # included) so the would-be grabs don't read as real. The title is shown
        # in full on its own line, so its length can't break the column.
        def added_detail(item: dict) -> None:
            torrent_value = group_highlight(
                item.get("name"),
                item.get("group"),
                group_style="grey50" if is_dry_run else "cyan",
                base_style="grey50" if is_dry_run else "green",
            )
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("link", item.get("url"), "grey50"),
                ("torrent", torrent_value, "green"),
            ]
            # A dry run dims every value (matching the dimmed title line) so the
            # would-be grabs don't read as real
            if is_dry_run:
                rows = [(label, value, "grey50") for label, value, _ in rows]
            _summary_block(
                item.get("title") or "(unknown title)",
                "grey50" if is_dry_run else None,
                rows,
            )

        summary_kv("checked", str(stats["checked"]))

        # Needs-action sits ahead of "added" so anything still waiting on the
        # user surfaces first, before the (often longer) list of completed grabs.
        needs = stats["needs_action"]
        summary_kv(
            "needs action",
            str(len(needs)),
            value_style="yellow" if needs else None,
        )
        for item in needs:
            needs_detail(item)

        # The count is the authoritative torrents_added (covers the no-client
        # dry-run path too); the list is the per-grab detail from add_torrent.
        summary_kv(
            "added",
            str(self.torrents_added),
            value_style="green" if self.torrents_added else None,
        )
        for item in stats["added"]:
            added_detail(item)

        summary_kv("up to date", str(stats["up_to_date"]))
        summary_kv(
            "unchanged",
            f"{stats['cached']}  (since last run)"
            if stats["cached"]
            else "0",
            value_style="grey50",
        )
        if stats["no_mappings"]:
            summary_kv("no mapping", str(stats["no_mappings"]))
        # Keep "no entry" (no SeaDex entry at all) separate from "no release"
        # (an entry exists but nothing suitable to grab) so they don't conflate
        if stats["no_seadex_entry"]:
            summary_kv("no entry", str(stats["no_seadex_entry"]))
        summary_kv("no release", str(stats["no_releases"]))

        if stats["unmonitored"]:
            summary_kv("unmonitored", str(stats["unmonitored"]))

        summary_kv(
            "issues",
            f"{count_noun(n_warnings, 'warning')}, {count_noun(n_errors, 'error')}",
            value_style="bold red"
            if n_errors
            else ("yellow" if n_warnings else None),
        )
        if self._run_started_monotonic is not None:
            elapsed = self.log_fmt.format_elapsed(
                time.monotonic() - self._run_started_monotonic,
            )
            summary_kv("elapsed", elapsed)

        # A single guidance line if anything was skipped purely for being
        # private-only, rather than repeating it per-entry during the run. Kept
        # at indent 1, so it reads as part of the summary block, not detached.
        public_only_skipped = any(
            "public_only" in (item.get("reason") or "") for item in needs
        )
        if public_only_skipped:
            self.logger.info(
                indent_string(
                    "Tip: set public_only: false to allow private trackers, or "
                    "wait for a public release.",
                    level=1,
                ),
                extra={"line_style": "grey50"},
            )

        self.logger.info(
            rule_string(rule_char="=", total_length=self.log_fmt.line_length),
            extra={"rule_char": "="},
        )

        return True

    def log_arr_start(
        self,
        arr: str,
        n_items: int,
    ) -> bool:
        """Produce a log message for the start of the run

        Args:
            arr: Type of arr instance
            n_items: Total number of shows/movies
        """

        if arr not in ALLOWED_ARRS:
            raise ValueError(f"arr must be one of: {ALLOWED_ARRS}")

        item_label = {
            "radarr": count_noun(n_items, "movie"),
            "sonarr": count_noun(n_items, "series", "series"),
        }[arr]

        banner = f"Starting SeaDexArr ({arr.capitalize()}) for {item_label}"
        self.logger.info(
            banner,
            extra={
                "rule_title": banner,
                "rule_style": "bold cyan",
                "rule_heavy": True,
            },
        )

        return True

    def log_entry_status(
        self,
        state: str,
        label: str,
        style: str | None = "grey50",
    ) -> bool:
        """Log a one-line entry status as a fixed-column ledger row

        Renders "<state> <label>" at indent level 1, with state padded to a fixed
        width so the label lines up across rows (see entry_string). Used for the
        entry-level outcomes: unchanged, in radarr, checking, unmonitored,
        skipped, no mapping, ignored, and no entry. The state word carries the
        meaning, so there is no trailing note; season/episode coverage and the
        SeaDex URL ride a separate continuation line (log_entry_coverage). The
        indent is baked into the message, so the file log keeps it too.

        Args:
            state (str): Short state word, e.g. "unchanged" or "no entry"
            label (str): What the state applies to (usually a title)
            style (str): Console style for the line. Defaults to "grey50" (dim);
                pass None for an emphasized line such as the active "checking" one
        """

        # A blank line before each ledger row separates entries within a title
        # block (and the first entry from its header)
        self.log_fmt.blank()
        self.logger.info(
            indent_string(entry_string(state, label), level=1),
            extra={"line_style": style},
        )

        return True

    def log_entry_coverage(
        self,
        coverage: str | None,
        url: str | None,
        style: str | None = "grey50",
        incomplete: bool = False,
    ) -> bool:
        """Log the season/episode coverage and SeaDex URL beneath an entry

        Two dim detail lines whose values sit directly beneath the entry's title
        (so they line up with each other and with the title): the season/episode
        coverage labeled "files", then the full SeaDex URL labeled "link".
        Either part may be absent - a Radarr movie has no episode coverage (link
        only) - and nothing is logged when both are absent. An incomplete SeaDex
        entry is flagged as an emphasized tail on the last line shown.

        Example:

            files S04 E01-E12
            link https://releases.moe/111852

        Args:
            coverage (str): One-line coverage, e.g. "S04 E01-E12" (maybe "")
            url (str): Full SeaDex URL (maybe None/"")
            style (str): Console style. Defaults to "grey50" (dim)
            incomplete (bool): Flag the SeaDex entry as incomplete. Defaults False
        """

        rows = [
            row for row in (("files", coverage), ("link", url)) if row[1]
        ]
        if not rows:
            return False

        for idx, (label, value) in enumerate(rows):
            # The incomplete flag rides the last line so it reads once, next to
            # the URL when there is one
            tail = (
                "(marked incomplete on SeaDex)"
                if incomplete and idx == len(rows) - 1
                else None
            )
            self.log_fmt.detail(label, value, value_style=style, tail=tail)

        return True

    def log_arr_item_unmonitored(
        self,
        item_title: str,
    ) -> bool:
        """Produce a log message if skipping because the item is unmonitored

        Args:
            item_title (str): Item title
        """

        self.stats["unmonitored"] += 1
        return self.log_entry_status(
            "unmonitored",
            item_title,
        )

    # Both Ares reach the same "unmonitored" outcome, so this is just an alias
    log_anilist_item_unmonitored = log_arr_item_unmonitored

    def log_arr_item_start(
        self,
        arr: str,
        item_title: str,
        n_item: int,
        n_items: int,
    ) -> bool:
        """Produce a log message for the start of Arr item

        Args:
            arr: Type of arr instance
            item_title: Title for the item
            n_item: Number for the show/movie
            n_items: Total number of shows/movies
        """

        # A blank line before the separator rule sets each item's block apart
        # from the previous one (and from the run banner for the first item)
        self.log_fmt.blank()
        header = f"[{n_item}/{n_items}] {arr.capitalize()}: {item_title}"
        self.logger.info(
            header,
            extra={"rule_title": header, "rule_style": "bold cyan"},
        )

        return True

    def log_no_anilist_mappings(
        self,
        title: str,
    ) -> bool:
        """Produce a log message for the case where no AniList mappings are found

        Args:
            title: Title for the item
        """

        self.stats["no_mappings"] += 1
        return self.log_entry_status(
            "no mapping",
            title,
        )

    def log_ignored_anilist_id(
        self,
        al_id: int,
    ) -> bool:
        """Produce a log message when an AniList ID is skipped via the ignore list

        Args:
            al_id (int): AniList ID
        """

        return self.log_entry_status(
            "ignored",
            f"AniList #{al_id}",
        )

    def log_no_anilist_id(self) -> bool:
        """Produce a log message for the case where no AniList ID is found"""

        self.logger.debug(
            indent_string("-> No AL ID found. Continuing"),
        )
        self.logger.debug(
            rule_string(
                total_length=self.log_fmt.line_length,
            ),
        )

        return True

    def log_no_sd_entry(
        self,
        al_id: int,
    ) -> bool:
        """Produce a log message if no SeaDex entry is found

        Args:
            al_id (int): Al ID
        """

        self.stats["no_seadex_entry"] += 1

        # Resolve a human title so the line is meaningful. There's no SeaDex
        # entry and the id isn't cached (we only cache processed ids), so this
        # is a live AniList lookup; the id rides its own "anilist" detail line.
        anilist_title, self.al_cache = get_anilist_title(
            al_id,
            al_cache=self.al_cache,
        )
        self.log_entry_status(
            "no entry",
            anilist_title or f"AniList #{al_id}",
        )
        # Only repeat the id on its own line when the ledger shows a title;
        # otherwise the ledger already reads "AniList #<id>" and a detail line
        # would just duplicate it
        if anilist_title:
            self.log_fmt.detail("anilist", str(al_id))

        return True

    def log_al_title(
        self,
        anilist_title: str,
        sd_entry: EntryRecord,
        coverage: str | None = None,
    ) -> bool:
        """Log the active-entry header: a "checking" row and its coverage/URL line

        The entry being evaluated is the focal line of the title block, so it sits
        on the ledger (state "checking") undimmed. The dim continuation lines below
        carry the season/episode coverage and, on its own line, the full SeaDex
        URL, so you can see what it covers and where to find it; an incomplete
        SeaDex entry is flagged as an emphasized tail on the last of those lines.

        Args:
            anilist_title (str): Title for the AniList entry
            sd_entry: SeaDex entry
            coverage (str, optional): One-line coverage (e.g. "S04 E01-E12").
                Defaults to None / "" (e.g., a Radarr movie -> URL only)
        """

        # Remember title, URL, and coverage so add_torrent / the summary can
        # attribute and link what they grab, and show the same files we mapped
        # from the Arr even when a release's own file list can't be parsed
        self.current_title = anilist_title
        self.current_url = sd_entry.url
        self.current_coverage = coverage

        # The active entry, on the ledger but undimmed (style=None) so it reads
        # as the focal line, not a no-op like the gray unchanged rows
        self.log_entry_status("checking", anilist_title, style=None)
        self.log_entry_coverage(
            coverage, sd_entry.url, incomplete=sd_entry.is_incomplete,
        )

        return True

    def log_cached_entry(
        self,
        arr: str,
        al_id: int,
        state: str = "unchanged",
    ) -> bool:
        """Log a cached entry as a ledger row plus its coverage/URL line

        Cached entries have been unchanged since the last run, so they collapse to a dim
        ledger row (state and title) and continuation lines carrying the stored
        season/episode coverage and, on its own line, the SeaDex URL. Everything
        is read from the cache
        record (written when the entry was first processed), with a name lookup
        only if the cache predates name storage.

        Args:
            arr (str): Arr instance the entry is cached under
            al_id (int): AniList ID
            state (str): State word. Defaults to "unchanged" (skipped because the
                SeaDex entry's update time matches the cache); pass "in radarr"
                for entries already handled by a Radarr sync
        """

        self.stats["cached"] += 1

        anilist_title = self.get_cached_name(arr=arr, al_id=al_id)
        if anilist_title is None:
            # Older cache without a stored name - fall back to a lookup
            anilist_title, self.al_cache = get_anilist_title(
                al_id,
                al_cache=self.al_cache,
            )
        if anilist_title is None:
            anilist_title = "(unknown title)"

        self.log_entry_status(state, anilist_title)
        self.log_entry_coverage(
            self.get_cached_field(arr, al_id, "coverage"),
            self.get_cached_field(arr, al_id, "url"),
        )

        return True

    def log_no_seadex_releases(self) -> bool:
        """Log if no suitable SeaDex releases are found"""

        self.stats["no_releases"] += 1
        self.log_fmt.detail(
            "status",
            "no suitable releases on SeaDex",
            value_style="grey50",
        )

        return True

    def log_seadex_action(
        self,
        seadex_dict: dict,
        results: list,
        dry_run: bool = False,
    ) -> bool:
        """Log the action block for a title that differs from SeaDex's pick

        Called after the adding has run, so the status reflects what actually
        happened rather than what we set out to do: if a better release was
         grabbed, it reads "adding"; if every recommended release was already
        present, it reads "matches - keeping it". The block is, in order: the
        status line, then each recommended release group, then the per-release
        outcome (added / kept).

        Args:
            seadex_dict (dict): SeaDex entries (used for the recommended groups)
            results (list): add_torrent's per-release outcomes (empty on a dry
                run, where there are no client-reported names)
            dry_run (bool): No torrent client, so nothing was really grabbed,, but
                we'd have added everything. Defaults to False

        Returns:
            bool: True if a status block was logged; False if there was nothing
                to report (e.g., every release was skipped - the skip warning
                already explains that, so a status would only mislead)
        """

        added = dry_run or any(r.get("outcome") == "added" for r in results)

        # Nothing grabbed and nothing already present (e.g., all releases skipped
        # by public_only): leave the status to the inline "skipped" warning
        if not results and not dry_run:
            return False

        if added:
            self.log_fmt.detail(
                "status",
                "your copy differs from SeaDex's pick - adding a better release",
            )
        else:
            self.log_fmt.detail(
                "status",
                "your copy matches SeaDex's pick - keeping it",
                value_style="green",
            )

        # The release group(s) we recommend (those flagged for download), tags too
        for srg, srg_item in seadex_dict.items():

            urls = srg_item.get("urls", {})
            if any(u.get("download", False) for u in urls.values()):
                tags = srg_item.get("tags", [])
                if len(tags) > 0:
                    recommendation = f"{srg} [{', '.join(tags)}]"
                else:
                    recommendation = srg
                self.log_fmt.detail("group", recommendation, value_style="cyan")

        # Per-release outcome (qBittorrent path; a dry run has no names to show)
        for r in results:
            if r.get("outcome") == "added":
                self.log_fmt.detail("added", r.get("name"), value_style="green")
            else:
                self.log_fmt.detail("kept", r.get("name"))

        return True

    def log_max_torrents_added(self) -> bool:
        """Produce a log message about hitting the maximum number of torrents added"""

        self.logger.info(
            "Reached the maximum torrents for this run; stopping",
            extra={"line_style": "yellow"},
        )

        return True
