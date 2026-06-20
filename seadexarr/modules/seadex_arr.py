import copy
import json
import logging
import os
import shutil
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from hashlib import md5
from itertools import compress
from typing import Any
from urllib.request import urlretrieve
from xml.etree import ElementTree

import httpx
import qbittorrentapi
import requests
import yaml
from ruamel.yaml import YAML
from seadex import EntryNotFoundError, EntryRecord, SeaDexEntry

from .anilist import (
    ANILIST_BATCH_SIZE,
    get_anilist_thumb,
    get_anilist_title,
    get_query_batch,
)
from .log import (
    DETAIL_INDENT,
    DETAIL_KEY_WIDTH,
    KEY_WIDTH,
    count_noun,
    entry_string,
    indent_string,
    kv_string,
    left_aligned_string,
    rule_string,
    setup_logger,
)
from .anibridge import AniBridge
from .torrent import (
    get_animetosho_torrent,
    get_nyaa_torrent,
    get_rutracker_torrent,
)
from .. import __version__


def save_json(
    data: dict,
    out_file: str,
    sort_cache: bool = False,
) -> None:
    """Save JSON prettily

    Args:
        data (dict): Data to be saved
        out_file (str): Path to JSON file
        sort_cache (bool, optional): Whether to sort cache files by AniList ID. Defaults to False.
    """

    # Optionally sort this data
    if sort_cache:

        anilist_entries = data.get("anilist_entries")
        if anilist_entries is not None:
            for arr, arr_item in anilist_entries.items():
                keys = list(arr_item.keys())
                keys.sort(key=int)
                sorted_data = {key: arr_item[key] for key in keys}

                anilist_entries[arr] = sorted_data

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


ANIME_IDS_URL = "https://raw.githubusercontent.com/Kometa-Team/Anime-IDs/refs/heads/master/anime_ids.json"
ANIDB_MAPPINGS_URL = "https://raw.githubusercontent.com/Anime-Lists/anime-lists/refs/heads/master/anime-list-master.xml"

# anibridge-mappings ships daily-rolling release assets tagged by major version.
# Bump ANIBRIDGE_RELEASE when a new (breaking) major lands; the cache filename is
# versioned to match so an old-format file is never parsed by the new loader.
ANIBRIDGE_RELEASE = "v3"
ANIBRIDGE_MAPPINGS_URL = f"https://github.com/anibridge/anibridge-mappings/releases/download/{ANIBRIDGE_RELEASE}/mappings.min.json"
ANIBRIDGE_MAPPINGS_FILE = f"anibridge_mappings_{ANIBRIDGE_RELEASE}.json"

ALLOWED_ARRS = [
    "radarr",
    "sonarr",
]

PUBLIC_TRACKERS = {
    tracker.casefold()
    for tracker in [
        "Nyaa",
        "AnimeTosho",
        "AniDex",
        "RuTracker",
    ]
}

PRIVATE_TRACKERS = {
    tracker.casefold()
    for tracker in [
        "AB",
        "BeyondHD",
        "PassThePopcorn",
        "BroadcastTheNet",
        "HDBits",
        "Blutopia",
        "Aither",
    ]
}

UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"

# How long a persisted AniList response stays usable before it's re-fetched.
# title/format/coverImage are effectively static; episodes for a currently airing
# show drift, so this caps how stale that count can get (~one episode/week).
ANILIST_CACHE_TTL_DAYS = 7


def normalize_rg(name: str | None) -> str | None:
    """Normalize a release group name for comparison

    Lower-cases and strips surrounding whitespace and dashes so that the same
    group named slightly differently by Sonarr and SeaDex (e.g. "Era-Raws" vs.
    "era-raws ") compare equal. Returns None for a missing/blank name.

    Args:
        name (str | None): Release group name
    """

    if not name:
        return None
    return name.strip().strip("-").casefold()


def get_all_seadex_rgs_per_episode(
    seadex_dict: dict,
    ep_list: list | None,
) -> dict:
    """Get a list of all SeaDex releases per-episode

    Args:
        seadex_dict: Dictionary of SeaDex releases
        ep_list (list): List of episodes and info
    """

    all_seadex_rgs_per_episode = {"all": []}

    if len(seadex_dict) > 1:
        for seadex_rg, seadex_rg_item in seadex_dict.items():

            # Index by the normalized name so the membership checks in
            # filter_by_release_group are case- and dash-insensitive
            seadex_rg_normalized = normalize_rg(seadex_rg)

            seadex_urls = seadex_rg_item.get("urls", {})
            for _url, url_item in seadex_urls.items():

                seadex_episodes = url_item.get("episodes", [])

                # If we haven't managed to parse, then set this up as an
                # "all" episode fallback
                if len(seadex_episodes) == 0:
                    if seadex_rg_normalized not in all_seadex_rgs_per_episode["all"]:
                        all_seadex_rgs_per_episode["all"].append(seadex_rg_normalized)

                found_episodes = [False] * len(seadex_episodes)

                for seadex_idx, seadex_ep in enumerate(seadex_episodes):

                    if found_episodes[seadex_idx]:
                        continue

                    for sonarr_ep in ep_list:
                        sonarr_ep_season = sonarr_ep.get("seasonNumber", 999)
                        sonarr_ep_episode = sonarr_ep.get("episodeNumber", 999)

                        # Do we have a match?
                        if sonarr_ep_season == seadex_ep.get(
                            "season", 888,
                        ) and sonarr_ep_episode == seadex_ep.get("episode", 888):

                            season_key = (
                                f"S{sonarr_ep_season:02d}E{sonarr_ep_episode:02d}"
                            )
                            if season_key not in all_seadex_rgs_per_episode:
                                all_seadex_rgs_per_episode[season_key] = []

                            if seadex_rg_normalized not in all_seadex_rgs_per_episode[season_key]:
                                all_seadex_rgs_per_episode[season_key].append(seadex_rg_normalized)

                            found_episodes[seadex_idx] = True

    return all_seadex_rgs_per_episode


def get_episode_keys(all_episodes: Iterable[dict]) -> set:
    """Build the set of (season, episode) keys an episode list covers

    Reduces a release's parsed episode list to the set of (season, episode)
    pairs it contains, so different SeaDex release groups can be compared by
    what files they cover.

    Args:
        all_episodes (iterable): Parsed episode dicts with "season"/"episode"
    """

    return {(ep.get("season"), ep.get("episode")) for ep in all_episodes}


def get_same_files_groups(seadex_dict: dict) -> list:
    """Group SeaDex release groups that cover exactly the same files

    Release groups are grouped by their parsed episode coverage: two groups are
    only treated as covering the same files when their parsed episode lists are
    identical. This is deliberately stricter than "episodes overlap" -- groups
    that overlap without being equal (e.g., a full-season batch and a single
    cour) cover *different* files and must not be collapsed, or we'd silently
    drop episodes when keeping only one of them.

    Release groups with no episode parsing at all (e.g., Radarr movies) are
    treated as covering the same files. Release groups whose files couldn't be
    parsed (Sonarr parse failure, empty episode list) are each kept on their
    own: we can't prove what they cover, so we'd rather grab a duplicate than
    silently drop content. Returns a list of lists of release group names.

    Args:
        seadex_dict (dict): Dictionary of SeaDex releases
    """

    grouped = {}
    for rg, rg_item in seadex_dict.items():
        all_episodes = rg_item.get("all_episodes", None)

        if all_episodes is None:
            # No episode parsing for this Arr (e.g., Radarr): treat as one movie
            key = "__no_episode_parsing__"
        elif len(all_episodes) == 0:
            # Parsing ran but found nothing: keep this group on its own so we
            # never drop content we couldn't verify
            key = ("__unparsed__", rg)
        else:
            key = frozenset(get_episode_keys(all_episodes))

        # Insertion-ordered dict preserves first-seen group order for us
        grouped.setdefault(key, []).append(rg)

    return list(grouped.values())


def format_episode_ranges(episode_numbers: Iterable[int]) -> str:
    """Condense a set of episode numbers into a readable range string

    Contiguous runs are collapsed (e.g. [1, 2, 3] -> "E01-E03"), lone episodes
    are kept as-is (e.g. [5] -> "E05"), and gaps split into multiple comma-separated ranges (e.g. [1, 2, 3, 7, 8] -> "E01-E03, E07-E08").

    Args:
        episode_numbers (iterable): Episode numbers within a single season
    """

    episodes = sorted(set(episode_numbers))
    if not episodes:
        return ""

    # Walk the sorted episodes, breaking into runs wherever they aren't
    # consecutive
    runs = []
    run_start = run_end = episodes[0]
    for episode in episodes[1:]:
        if episode == run_end + 1:
            run_end = episode
        else:
            runs.append((run_start, run_end))
            run_start = run_end = episode
    runs.append((run_start, run_end))

    return ", ".join(
        f"E{start:02d}" if start == end else f"E{start:02d}-E{end:02d}"
        for start, end in runs
    )


class SeaDexArr:

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

        # If we don't have a config file, copy the sample to the current
        # working directory
        f_path = copy.deepcopy(__file__)
        config_template_path = os.path.join(
            os.path.dirname(f_path), "config_sample.yml",
        )
        if not os.path.exists(config):
            shutil.copy(config_template_path, config)
            raise FileNotFoundError(f"{config} not found. Copying template")

        self.config_file = config
        with open(config) as f:
            self.config = yaml.safe_load(f)

        # Check the config has all the same keys as the sample, if not add 'em in
        self.verify_config(
            config_path=config,
            config_template_path=config_template_path,
        )

        # Ignore unmonitored flag
        self.ignore_unmonitored = self.config.get(f"{arr}_ignore_unmonitored", False)

        # A single keep-alive session shared by the raw Sonarr/Radarr API calls
        self.session = requests.Session()

        # qbit
        self.qbit: qbittorrentapi.Client
        qbit_info = self.config.get("qbit_info", None)

        # Check we've got everything we need
        qbit_info_provided = all(
            qbit_info.get(key, None) is not None for key in qbit_info
        )
        if qbit_info_provided:
            qbit = qbittorrentapi.Client(**qbit_info)

            # Ensure this works
            try:
                qbit.auth_log_in()
            except qbittorrentapi.LoginFailed:
                raise ValueError(
                    "qBittorrent login failed - check the qbit_info host and "
                    "credentials in your config",
                )

            self.qbit = qbit

        self.ignore_seadex_update_times = self.config.get(
            "ignore_seadex_update_times", False,
        )

        self.use_torrent_hash_to_filter = self.config.get(
            "use_torrent_hash_to_filter", False,
        )

        # Hooks between torrents and Arts, and torrent number bookkeeping
        self.torrent_category = self.config.get(f"{arr}_torrent_category", None)
        self.torrent_tags = self.config.get("torrent_tags", None)
        self.max_torrents_to_add = self.config.get("max_torrents_to_add", None)
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
        self.discord_url = self.config.get("discord_url", None)

        # Flags for filtering torrents
        self.public_only = self.config.get("public_only", True)
        self.prefer_dual_audio = self.config.get("prefer_dual_audio", True)
        self.want_best = self.config.get("want_best", True)

        # Set per-title when public_only forces us to skip a release that's only
        # available privately, so the caller knows not to cache the title as done.
        # public_only_groups collects the release-group name(s) that were skipped
        # for that reason, so the run summary can name them under "needs action"
        self.public_only_skipped = False
        self.public_only_groups: list[str] = []

        ignore_tags = self.config.get("ignore_tags", None)
        if ignore_tags is None:
            ignore_tags = []
        self.ignore_tags = ignore_tags

        # AniList IDs to skip entirely
        ignore_anilist_ids = self.config.get("ignore_anilist_ids", None)
        if ignore_anilist_ids is None:
            ignore_anilist_ids = set()
        self.ignore_anilist_ids = {int(x) for x in ignore_anilist_ids}

        trackers = self.config.get("trackers", None)

        # If we don't have any trackers selected, build a list from public
        # and private trackers
        # Include all even if public_only is True, as these filter out releases before we check if they overlap with what's downloaded in Sonarr
        if trackers is None:
            trackers = PUBLIC_TRACKERS.union(PRIVATE_TRACKERS)

        self.trackers = {t.casefold() for t in trackers}

        # Advanced settings
        self.sleep_time = self.config.get("sleep_time", 2)
        self.cache_time = self.config.get("cache_time", 1)

        # Get the mapping files
        anime_mappings_cfg = self.config.get("anime_mappings", None)
        anidb_mappings_cfg = self.config.get("anidb_mappings", None)
        anibridge_mappings_cfg = self.config.get("anibridge_mappings", None)

        if anime_mappings_cfg is False:
            anime_mappings = {}
        elif anime_mappings_cfg is None:
            anime_mappings = self.get_anime_mappings()
        else:
            anime_mappings = anime_mappings_cfg

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
        self.anidb_mappings = anidb_mappings
        self.anibridge = anibridge

        self.interactive = self.config.get("interactive", False)

        if logger is None:
            log_level = self.config.get("log_level", "INFO")
            self.logger = setup_logger(log_level=log_level)
        else:
            self.logger = logger

        # Instantiate the SeaDex API
        self.seadex = SeaDexEntry()

        # Set up cache for AL API calls
        self.al_cache = {}

        # Memoize get_anilist_ids mapping computation per identifying key, so
        # the prefetch pass and the main loop don't compute it twice per item
        self._anilist_ids_cache = {}

        # Load in cache if it exists. Else create
        self.cache_file = cache
        if os.path.exists(cache):
            with open(cache) as f:
                cache = json.load(f)
        else:
            cache = self.setup_cache()
        self.cache: dict[str, Any] = cache

        # Check the package or config hasn't updated, else
        # edit the cache description
        self.check_cache_updates()

        self.log_line_length = 80

    def close(self) -> None:
        """Close the shared HTTP session (release pooled connections)."""
        if self.session is not None:
            self.session.close()

    def verify_config(
        self,
        config_path: str,
        config_template_path: str,
    ) -> bool:
        """Verify all the keys in the current config file match those in the template

        Args:
            config_path (str): Path to a config file
            config_template_path (str): Path to config template
        """

        with open(config_template_path) as f:
            config_template = YAML().load(f)

        # If the keys aren't in the right order, then
        # use the template as a base and inherit from
        # the main config
        if list(self.config.keys()) != list(config_template.keys()):

            new_config = copy.deepcopy(config_template)
            for key in config_template:
                if key in self.config:
                    new_config[key] = copy.deepcopy(self.config[key])
                else:
                    new_config[key] = copy.deepcopy(config_template[key])

            self.config = copy.deepcopy(new_config)

            # Save out
            with open(config_path, "w+") as f:
                YAML().dump(self.config, f)

        return True

    def setup_cache(self) -> dict:
        """Set up the cache file"""

        cache = {}

        with open(self.config_file, "rb") as f:
            config_hash = md5(f.read()).hexdigest()

        # Descriptor for the file so we know if things have changed
        description = {
            "seadexarr_version": __version__,
            "config_checksum": config_hash,
        }

        cache.update({"description": description})
        cache.update({"anilist_entries": {}})

        return cache

    def check_cache_updates(self) -> bool:
        """Check if anything's been updated, and if so update in cache"""

        # Check if SeaDexArr version has updated
        if (
            self.cache.get("description", {}).get("seadexarr_version", None)
            != __version__
        ):
            self.cache["description"]["seadexarr_version"] = __version__

        # Check if the config file has changed
        with open(self.config_file, "rb") as f:
            config_hash = md5(f.read()).hexdigest()
            if (
                self.cache.get("description", {}).get("config_checksum", None)
                != config_hash
            ):
                self.cache["description"]["config_checksum"] = config_hash

        return True

    def get_anime_mappings(self) -> dict:
        """Get the anime IDs file"""

        anime_mappings_file = os.path.join("anime_ids.json")

        # If a file doesn't exist, get it
        self.get_external_mappings(
            f=anime_mappings_file,
            url=ANIME_IDS_URL,
        )

        with open(anime_mappings_file) as f:
            return json.load(f)


    def get_anidb_mappings(self) -> ElementTree.Element:
        """Get the AniDB mappings file"""

        anidb_mappings_file = os.path.join("anime-list-master.xml")

        # If a file doesn't exist, get it
        self.get_external_mappings(
            f=anidb_mappings_file,
            url=ANIDB_MAPPINGS_URL,
        )

        return ElementTree.parse(anidb_mappings_file).getroot()


    def get_anibridge_mappings(self) -> AniBridge:
        """Download the anibridge-mappings graph and build an indexed view.

        Returns:
            AniBridge: Parsed, indexed mappings ready for id lookups
        """

        # If a file doesn't exist, get it
        self.get_external_mappings(
            f=ANIBRIDGE_MAPPINGS_FILE,
            url=ANIBRIDGE_MAPPINGS_URL,
        )

        with open(ANIBRIDGE_MAPPINGS_FILE) as f:
            graph = json.load(f)

        return AniBridge(graph, logger=getattr(self, "logger", None))


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

        # Check if this is older than the cache
        f_mtime = os.path.getmtime(f)
        f_datetime = datetime.fromtimestamp(f_mtime)
        now_datetime = datetime.now()

        # Get the time difference
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
        """Check if timestamps in the cache match when SeaDex entry was last updated

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            seadex_entry: SeaDex entry
        """
        sd_time = seadex_entry.updated_at
        sd_time_str = sd_time.strftime(UPDATED_AT_STR_FORMAT)
        cache_time = (
            self.cache.get("anilist_entries", {})
            .get(arr, {})
            .get(str(al_id), {})
            .get("updated_at")
        )

        return sd_time_str == cache_time

    def get_cached_name(
        self,
        arr: str,
        al_id: int,
    ) -> str | None:
        """Get the AniList title stored in the cache for an entry, if any

        The title is written into the cache alongside the timestamp when an
        entry is first processed, so it can be reused for cached entries
        without an additional AniList lookup.

        Args:
            arr (str): Arr instance the entry is cached under
            al_id (int): AniList ID

        Returns:
            str | None: Cached title, or None if not present
        """

        return self.get_cached_field(arr, al_id, "name")

    def get_cached_field(
        self,
        arr: str,
        al_id: int,
        field: str,
    ) -> Any:
        """Read a single stored field from an entry's cache record, if present

        Args:
            arr (str): Arr instance the entry is cached under
            al_id (int): AniList ID
            field (str): Cache field name (e.g. "name", "url", "coverage")

        Returns:
            The stored value, or None if absent
        """

        return (
            self.cache.get("anilist_entries", {})
            .get(arr, {})
            .get(str(al_id), {})
            .get(field)
        )

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
                record.get("fetched_at", ""), UPDATED_AT_STR_FORMAT,
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

        if tvdb_id is not None:
            anilist_mappings.update(
                {
                    m["anilist_id"]: m
                    for n, m in self.anime_mappings.items()
                    if m.get("tvdb_id", None) == tvdb_id
                    and m.get("anilist_id", None) is not None
                    and m.get("anilist_id", None) not in anilist_mappings
                },
            )
        if tmdb_id is not None:
            anilist_mappings.update(
                {
                    m["anilist_id"]: m
                    for n, m in self.anime_mappings.items()
                    if m.get(f"tmdb_{tmdb_type}_id", None) == tmdb_id
                    and m.get("anilist_id", None) is not None
                    and m.get("anilist_id", None) not in anilist_mappings
                },
            )
        if imdb_id is not None:
            anilist_mappings.update(
                {
                    m["anilist_id"]: m
                    for n, m in self.anime_mappings.items()
                    if m.get("imdb_id", None) == imdb_id
                    and m.get("anilist_id", None) is not None
                    and m.get("anilist_id", None) not in anilist_mappings
                },
            )

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

        final_torrent_list = copy.deepcopy(sd_entry.torrents)

        # Filter out any tags
        final_torrent_list = [
            t for t in final_torrent_list if len(set(self.ignore_tags).intersection(set(t.tags))) == 0
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

        # If the user wants only 'best' releases and any exist, narrow down to those
        if self.want_best and any_best:
            candidates = best_torrents
        else:
            candidates = final_torrent_list

        # Now, if we prefer dual audio, then remove any that aren't
        # tagged, so long as at least one is tagged
        if self.prefer_dual_audio:
            duals = [t for t in candidates if t.is_dual_audio]
            if len(duals) > 0:
                candidates = duals
        # Or, if it's False, do the opposite
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

        # Catch various edge cases
        if release_group_discord is None:
            release_group_discord = ["None"]
        if len(release_group_discord) == 0:
            release_group_discord = ["None"]
        if isinstance(release_group_discord, str):
            release_group_discord = [release_group]

        field_dict = {
            "name": f"{arr.capitalize()} Release:",
            "value": "\n".join(release_group_discord),
        }
        fields.append(field_dict)

        # SeaDex options with links
        for srg, srg_item in seadex_dict.items():

            # Check if we're actually downloading anything
            dl = [
                srg_item.get("urls", {}).get(x, {}).get("download", False)
                for x in srg_item["urls"]
            ]

            if any(dl):

                # Include any tags in the string
                discord_value = ""
                tags = srg_item.get("tags", [])
                if len(tags) > 0:
                    discord_value += "Tags:\n"
                    discord_value += "\n".join(tags)
                    discord_value += "\n\n"

                urls_to_download = [x for i, x in enumerate(srg_item["urls"]) if dl[i]]

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
                left_aligned_string(
                    f"Filtering for release group {seadex_rg}",
                ),
            )

            seadex_urls = seadex_rg_item.get("urls", {})
            for _url, url_item in seadex_urls.items():

                url_hash = url_item.get("hash", None)

                # If the URL is already in the hash cache, then append but don't set to download
                torrent_hashes.append(url_hash)
                if url_hash not in cached_hashes:
                    self.logger.debug(
                        left_aligned_string(
                            f"Torrent hash {url_hash} not found in cache. "
                            f"Will add to downloads",
                        ),
                    )

                    url_item.update({"download": True})

                else:
                    self.logger.debug(
                        left_aligned_string(
                            f"Torrent hash {url_hash} in cache. Will skip download",
                        ),
                    )

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        self.reduce_overlapping_downloads(seadex_dict=seadex_dict, arr=arr)

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

        # Get a simple list of the release groups
        arr_release_groups = list(arr_release_dict.keys())

        # And also just check if any release group matches
        # any Arr release tag
        seadex_keys = set(seadex_dict.keys())
        overlapping_results = any(rg in seadex_keys for rg in arr_release_groups)

        # If we have overlaps, get a note of them here
        all_seadex_rgs_per_episode = get_all_seadex_rgs_per_episode(
            seadex_dict=seadex_dict,
            ep_list=ep_list,
        )

        for seadex_rg, seadex_rg_item in seadex_dict.items():

            self.logger.debug(
                left_aligned_string(
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
                            left_aligned_string(
                                f"SeaDex release group {seadex_rg} not in {arr.capitalize()} releases: "
                                f"{', '.join([str(x) for x in arr_release_groups])} - will download {url}",
                            ),
                        )

                        url_item.update({"download": True})

                    # Else, if we match, then double-check against the size
                    if seadex_rg in arr_release_groups:

                        # Be a blunt hammer and just check intersections
                        seadex_file_sizes = url_item.get("size", [])
                        arr_file_sizes = arr_release_dict[seadex_rg].get("size", [])

                        if not isinstance(arr_file_sizes, list):
                            arr_file_sizes = [arr_file_sizes]

                        intersect = list(
                            filter(
                                lambda x: x in seadex_file_sizes,
                                arr_file_sizes,
                            ),
                        )

                        # If we have no overlaps at all, then add
                        if len(intersect) == 0:
                            self.logger.debug(
                                left_aligned_string(
                                    f"SeaDex release group {seadex_rg} in {arr.capitalize()} releases: "
                                    f"{', '.join([str(x) for x in arr_release_groups])}, but file sizes do not match - will download {url}",
                                ),
                            )

                            url_item.update({"download": True})

                        else:
                            self.logger.debug(
                                left_aligned_string(
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

                    found_episodes = [False] * len(seadex_episodes)
                    rg_matches = [False] * len(seadex_episodes)
                    size_matches = [False] * len(seadex_episodes)

                    for seadex_idx, seadex_ep in enumerate(seadex_episodes):

                        if found_episodes[seadex_idx]:
                            continue

                        for sonarr_ep in ep_list:

                            # Get Season, Episode, and size numbers for Sonarr and SeaDex
                            sonarr_ep_season = sonarr_ep.get("seasonNumber", 999)
                            sonarr_ep_episode = sonarr_ep.get("episodeNumber", 999)
                            sonarr_ep_size = sonarr_ep.get("episodeFile", {}).get(
                                "size", None,
                            )

                            seadex_ep_season = seadex_ep.get("season", 888)
                            seadex_ep_episode = seadex_ep.get("episode", 888)
                            seadex_ep_size = seadex_ep.get("size", None)

                            # Do we have a match?
                            if (
                                sonarr_ep_season == seadex_ep_season
                                and sonarr_ep_episode == seadex_ep_episode
                            ):

                                # Do the sizes match?
                                size_match = sonarr_ep_size == seadex_ep_size

                                season_ep_str = (
                                    f"S{sonarr_ep_season:02d}E{sonarr_ep_episode:02d}"
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

                                    # This check here is to make sure we don't duplicate
                                    # if there's overlap
                                    all_seadex_rg = all_seadex_rgs_per_episode.get(
                                        season_ep_str, [],
                                    )

                                    if sonarr_rg_normalized not in all_seadex_rg:
                                        self.logger.debug(
                                            left_aligned_string(
                                                f"SeaDex release group {seadex_rg} differs from "
                                                f"{arr.capitalize()} release for "
                                                f"{season_ep_str} ({sonarr_rg}) and no other "
                                                f"recommended release covers it - will download {url}",
                                            ),
                                        )

                                        url_item.update({"download": True})

                                else:

                                    self.logger.debug(
                                        left_aligned_string(
                                            f"Found SeaDex match to {arr.capitalize()} "
                                            f"for {season_ep_str}.",
                                        ),
                                    )
                                    if not size_match:
                                        self.logger.debug(
                                            left_aligned_string(
                                                f"-> Sizes are different: "
                                                f"{sonarr_ep_size} (Sonarr), {seadex_ep_size} (SeaDex)",
                                            ),
                                        )
                                    else:
                                        self.logger.debug(
                                            left_aligned_string(
                                                f"-> Sizes match: {sonarr_ep_size}",
                                            ),
                                        )

                                    rg_matches[seadex_idx] = True

                                # Now check against file size
                                if size_match:
                                    size_matches[seadex_idx] = True

                                found_episodes[seadex_idx] = True

                    # If we have matched the release groups but not the file sizes, then flag that
                    # here and mark for download
                    size_matches = list(compress(size_matches, rg_matches))
                    if not any(size_matches) and len(size_matches) > 0:
                        self.logger.debug(
                            left_aligned_string(
                                f"File sizes all differ for release group {seadex_rg} - will download {url}",
                            ),
                        )
                        url_item.update({"download": True})

        # Where multiple preferred release groups cover the same files and the
        # Arr has none of them, only grab one (preferring public if public_only)
        self.reduce_overlapping_downloads(seadex_dict=seadex_dict, arr=arr)

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
        arr: str,
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
            arr (str): Type of arr instance
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
                    self.log_detail(
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
                    left_aligned_string(
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

        any_to_download = False
        for rg in seadex_dict:

            if any_to_download:
                return any_to_download

            dl = [
                seadex_dict[rg]["urls"][x].get("download", False)
                for x in seadex_dict[rg]["urls"]
            ]
            if any(dl):
                any_to_download = True

        return any_to_download

    @staticmethod
    def format_episode_coverage(episodes: list) -> list | None:
        """Summarize the Sonarr season/episode coverage of a torrent, per season

        Returns a list of (season_label, episode_ranges) tuples, one per season
        the torrent covers, ordered by season. The season label is e.g. "S01"
        and the episode ranges condense contiguous runs, e.g. "E01-E12" or
        "E01-E03, E07-E12" for a season with a gap, or "E05" for a lone episode.

        Returns None when there is no parsed episode info (e.g., Radarr movies,
        or a Sonarr parse failure).

        Args:
            episodes (list): List of {"season", "episode", ...} dicts,
                as parsed onto each torrent's url_item
        """

        if not episodes:
            return None

        # Collect the episode numbers seen for each season
        episodes_by_season = {}
        for ep in episodes:
            season = ep.get("season")
            episode = ep.get("episode")
            if season is None or episode is None:
                continue
            episodes_by_season.setdefault(season, set()).add(episode)

        if not episodes_by_season:
            return None

        return [
            (f"S{season:02d}", format_episode_ranges(episodes_by_season[season]))
            for season in sorted(episodes_by_season)
        ]

    def coverage_string(self, episodes: list) -> str:
        """One-line season/episode coverage, e.g. "S04 E01-E12" or
        "S00 E10, S02 E01-E12". Returns "" when there's no parsed episode info
        (e.g., a Radarr movie), so callers can treat it as "URL only".

        Args:
            episodes (list): {"season", "episode"} dicts
        """

        coverage = self.format_episode_coverage(episodes)
        if not coverage:
            return ""
        return ", ".join(f"{label} {ranges}" for label, ranges in coverage)

    @staticmethod
    def episodes_from_ep_list(ep_list: list | None, missing_only: bool = False) -> list:
        """Convert a Sonarr ep_list into {"season","episode"} coverage dicts

        Sonarr episodes carry "seasonNumber"/"episodeNumber"; the coverage
        helpers expect "season"/"episode". Optionally, keep only missing episodes
        (no file on disk) to summarize what is still needed.

        Args:
            ep_list (list): Sonarr episode dicts
            missing_only (bool): Keep only episodes with no file. Defaults to False
        """

        episodes = []
        for ep in ep_list or []:
            if missing_only and ep.get("episodeFileId", 0) != 0:
                continue
            episodes.append(
                {
                    "season": ep.get("seasonNumber"),
                    "episode": ep.get("episodeNumber"),
                },
            )
        return episodes

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
                    self.log_detail(
                        "skipped",
                        f"{tracker} private-only (public_only on)",
                        value_style="yellow",
                        level=logging.WARNING,
                    )
                    self.public_only_skipped = True
                    self.public_only_groups.append(srg)
                    continue

                # If we don't have a tracker from our list selected, then
                # get out of here
                if tracker.casefold() not in self.trackers:
                    self.log_detail(
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
                    parsed_url, source_name = get_animetosho_torrent(url=url)

                # RuTracker
                elif tracker.lower() == "rutracker":
                    parsed_url, source_name = get_rutracker_torrent(
                        url=url,
                        torrent_hash=item_hash,
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
                        },
                    )

                    # Increment the number of torrents added, and if we've hit the limit, then
                    # jump out
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

    def save_cache(self) -> None:
        """Persist the in-memory cache to disk

        Skipped during a preview so a preview never writes state, mirroring
        update_cache.
        """

        if not self._is_preview():
            save_json(
                self.cache,
                self.cache_file,
                sort_cache=True,
            )

    def update_cache(self, arr: str, al_id: int, cache_details: dict | None = None) -> bool:
        """Update cache with useful info

        Args:
            arr (str): Arr instance
            al_id (int): AniList ID
            cache_details (dict): Details for the cache entry.
                Defaults to None
        """

        if cache_details is None:
            cache_details = {}

        # Turn datetime to string
        if "updated_at" in cache_details:
            cache_details["updated_at"] = cache_details["updated_at"].strftime(
                UPDATED_AT_STR_FORMAT,
            )

        # Add to cache and save out
        if arr not in self.cache["anilist_entries"]:
            self.cache["anilist_entries"][arr] = {}

        if str(al_id) not in self.cache["anilist_entries"][arr]:
            self.cache["anilist_entries"][arr][str(al_id)] = {}

        self.cache["anilist_entries"][arr][str(al_id)].update(cache_details)

        # Persist via save_cache, which keeps the preview gate in one place:
        # a preview keeps the in-memory cache consistent for the rest of the
        # run, but never persists it - so a preview never marks a title as done.
        self.save_cache()

        return True

    def _fresh_stats(self) -> dict:
        """Build an empty per-run stats tally for the end-of-run summary"""

        return {
            "checked": 0,
            "added": [],  # list of {"title", "group", "coverage"}
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
        # Monotonic so a wall-clock step (NTP, DST) can't yield negative elapsed
        self._run_started_monotonic = time.monotonic()
        counter = getattr(self.logger, "seadex_counter", None)
        self._log_counts_at_start = counter.snapshot() if counter else {}

        return True

    def log_kv(
        self,
        key: str,
        value: Any,
        value_style: str | None = None,
        level: int = logging.INFO,
        indent: int = 1,
        key_width: int = KEY_WIDTH,
        sep: str = " :",
        tail: str | None = None,
        tail_style: str = "yellow",
    ) -> bool:
        """Log an aligned "key : value" (or gutter "key value") detail line

        The file log stores the plain kv_string text; on the console the label
        is dimmed so the value reads first, and an optional value_style accents
        the outcome (e.g., green for "added").

        Args:
            key: Left-hand label
            value: Right-hand value
            value_style: Optional rich style for the value (e.g. "green")
            level: Logging level. Defaults to logging.INFO
            indent: Number of indent levels. Defaults to 1
            key_width: Column width the key is padded to. Defaults to KEY_WIDTH (16)
            sep: Separator after the padded key. Defaults to ":"; pass "" for
                the colon-less gutter format (see log_detail)
            tail: Optional emphasized suffix (console only), e.g., an "incomplete"
                note. Defaults to None
            tail_style: Style for the tail. Defaults to "yellow"
        """

        self.logger.log(
            level,
            kv_string(key, value, key_width=key_width, indent=indent, sep=sep),
            extra={
                "kv": {
                    "key": key,
                    "value": value,
                    "value_style": value_style,
                    "indent": indent,
                    "key_width": key_width,
                    "sep": sep,
                    "tail": tail,
                    "tail_style": tail_style,
                },
            },
        )

        return True

    def log_detail(
        self,
        label: str,
        value: Any,
        value_style: str | None = None,
        level: int = logging.INFO,
        tail: str | None = None,
        tail_style: str = "yellow",
    ) -> bool:
        """Log an entry-detail line: dim gutter label, value at the title column

        The colon-less "<label> <value>" form is used for everything indented under
        an entry (files / link / status / group / added / kept / missing /
        skipped / anilist). The value lands in the same column as the entry title,
        so the whole block reads as one aligned column; the label sits dimmed in
        the indent gutter and the value carries any accent color.

        Args:
            label: Gutter label, e.g. "files" or "added"
            value: The value text
            value_style: Optional rich style for the value (e.g. "green")
            level: Logging level. Defaults to logging.INFO
            tail: Optional emphasized suffix (console only). Defaults to None
            tail_style: Style for the tail. Defaults to "yellow"
        """

        return self.log_kv(
            label,
            value,
            value_style=value_style,
            level=level,
            indent=DETAIL_INDENT,
            key_width=DETAIL_KEY_WIDTH,
            sep="",
            tail=tail,
            tail_style=tail_style,
        )

    def log_blank(self) -> bool:
        """Emit a blank line to visually separate entries / item blocks"""

        self.logger.info("")
        return True

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format an elapsed number of seconds as e.g. "8s", "14m 03s" or "1h 02m 03s" """

        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

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
        # colon columns never sit adjacent. Wrap log_kv to fix the width at 12.
        def summary_kv(key: str, value: Any, **kwargs: Any) -> bool:
            return self.log_kv(key, value, key_width=12, **kwargs)

        # A needs-action entry in the summary, rendered with the same labeled
        # gutter as added_detail so the two blocks read alike: the title hangs at
        # indent 2, then fixed fields sit at indent 3 beneath it. Unlike a grab
        # there's no torrent name to lean on, so the skipped private release
        # group IS named here. The whole block is yellow - it's the one section
        # asking the user to do something. Titles are truncated so they can't wrap
        # onto a second line and break the column.
        def needs_detail(item: dict) -> None:
            t = item.get("title") or "(unknown title)"
            if len(t) > 38:
                t = t[:37] + "…"
            self.logger.info(
                indent_string(t, level=2), extra={"line_style": "yellow"},
            )
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("group", item.get("group"), "yellow"),
                ("reason", item.get("reason"), "yellow"),
                ("link", item.get("url"), "grey50"),
            ]
            for label, value, accent in rows:
                if not value:
                    continue
                self.log_kv(
                    label,
                    value,
                    value_style=accent,
                    indent=3,
                    key_width=7,
                    sep="",
                )

        # A grab in the summary, rendered like the live per-entry "checking"
        # block: the title hangs at indent 2, then labeled gutter fields sit
        # beneath it at indent 3, their values landing in the same column (14) as
        # the live block. The recommended group is dropped here (the torrent name
        # already carries it), and the grab is labeled "torrent" rather than
        # "added" since the whole section is already the added list. A dry run
        # dims the block so the would-be grabs don't read as real.
        def added_detail(item: dict) -> None:
            t = item.get("title") or "(unknown title)"
            if len(t) > 38:
                t = t[:37] + "…"
            self.logger.info(
                indent_string(t, level=2),
                extra={"line_style": "grey50" if is_dry_run else None},
            )
            rows = [
                ("files", item.get("coverage"), "grey50"),
                ("link", item.get("url"), "grey50"),
                ("torrent", item.get("name"), "green"),
            ]
            for label, value, accent in rows:
                if not value:
                    continue
                self.log_kv(
                    label,
                    value,
                    value_style="grey50" if is_dry_run else accent,
                    indent=3,
                    key_width=7,
                    sep="",
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
            elapsed = self._format_elapsed(
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
            rule_string(rule_char="=", total_length=self.log_line_length),
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
        self.log_blank()
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
            self.log_detail(label, value, value_style=style, tail=tail)

        return True

    def log_arr_item_unmonitored(
        self,
        arr: str,
        item_title: str,
    ) -> bool:
        """Produce a log message if skipping because the item is unmonitored

        Args:
            arr: Type of arr instance
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
        self.log_blank()
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
                total_length=self.log_line_length,
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
            self.log_detail("anilist", str(al_id))

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
        self.log_detail(
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
            self.log_detail(
                "status",
                "your copy differs from SeaDex's pick - adding a better release",
            )
        else:
            self.log_detail(
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
                self.log_detail("group", recommendation, value_style="cyan")

        # Per-release outcome (qBittorrent path; a dry run has no names to show)
        for r in results:
            if r.get("outcome") == "added":
                self.log_detail("added", r.get("name"), value_style="green")
            else:
                self.log_detail("kept", r.get("name"))

        return True

    def log_max_torrents_added(self) -> bool:
        """Produce a log message about hitting the maximum number of torrents added"""

        self.logger.info(
            "Reached the maximum torrents for this run; stopping",
            extra={"line_style": "yellow"},
        )

        return True
