"""ID-mapping resolution: external sources -> AniList ids.

``MappingResolver`` owns the three immutable mapping sources (the Kometa
Anime-IDs JSON, the AniDB anime-list XML, and the anibridge graph): it
downloads/refreshes them, parses and indexes them, and resolves an Arr's
external ids (TVDB / TMDB / IMDb) to the AniList ids they map to.

Extracted from ``SeaDexArr`` in Phase 3 of the refactor (see
``REFACTOR_PLAN.md``); behaviour-preserving. The module-global parse memo
(:data:`_PARSED_MAPPING_CACHE`) is deliberately module-level, not instance
state, so a scheduled Radarr->Sonarr cycle reuses the first instance's parse
of an unchanged on-disk file.
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar, cast
from urllib.request import urlretrieve
from xml.etree import ElementTree

from .anibridge import AniBridge


class TmdbType(StrEnum):
    """Which TMDB id space an external lookup is scoped to."""

    MOVIE = "movie"
    SHOW = "show"


def _validate_ids(
    tvdb_id: int | None,
    tmdb_id: int | None,
    imdb_id: str | None,
) -> None:
    """Raise if no external id was supplied (at least one is required)."""

    if (tvdb_id is None) and (tmdb_id is None) and (imdb_id is None):
        raise ValueError(
            "At least one of tvdb_id, tmdb_id, and imdb_id must be provided",
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
_PARSED_MAPPING_CACHE: dict[str, tuple[float, object]] = {}

_T = TypeVar("_T")


def _load_mapping_by_mtime(
    path: str,
    parse: Callable[[str], _T],
    cache_key: str | None = None,
) -> _T:
    """Return ``parse(path)``, reusing a cached result while the mtime is unchanged.

    Args:
        path (str): File whose modification time gates the cache
        parse (Callable[[str], _T]): Builds the parsed value from the path
        cache_key (str | None): Cache slot to use; defaults to ``path``. Pass a
            distinct key when more than one product is derived from one file
            (e.g. the Anime-IDs map and its reverse index).
    """

    key = cache_key or path
    mtime = os.path.getmtime(path)

    cached = _PARSED_MAPPING_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        # The slot is type-erased across keys (each holds whatever its parse fn
        # produced); the call's _T is recovered from ``parse``'s return type.
        return cast(_T, cached[1])

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


class MappingResolver:
    """Resolve external Arr ids to AniList ids via three mapping sources.

    Loads, downloads-if-stale, parses and indexes the Kometa Anime-IDs map, the
    AniDB anime-list XML, and the anibridge graph at construction, then answers
    id lookups. ``get_anilist_ids`` returns the AniList mappings plus the ids it
    dropped because the user chose to ignore them, so the caller (not the
    resolver) owns the per-id logging.
    """

    def __init__(
        self,
        *,
        cache_time: int,
        ignore_anilist_ids: set[int],
        anime_mappings_cfg: dict | bool | None,
        anidb_mappings_cfg: ElementTree.Element | bool | None,
        anibridge_mappings_cfg: dict | bool | None,
    ) -> None:
        """Load and index the mapping sources.

        Args:
            cache_time (int): Days a downloaded source stays usable before it's
                re-fetched.
            ignore_anilist_ids (set[int]): AniList ids to drop from every result.
            anime_mappings_cfg: ``False`` to disable, ``None`` to download the
                Kometa Anime-IDs map, or a pre-parsed map dict.
            anidb_mappings_cfg: ``False`` to disable, ``None`` to download the
                AniDB XML, or a pre-parsed root element.
            anibridge_mappings_cfg: ``False`` to disable, ``None`` to download
                the anibridge graph, or a raw anibridge graph dict.
        """

        self.cache_time = cache_time
        self.ignore_anilist_ids = ignore_anilist_ids

        # Memoize get_anilist_ids mapping computation per identifying key, so
        # the prefetch pass and the main loop don't compute it twice per item
        self._anilist_ids_cache: dict = {}

        if anime_mappings_cfg is False:
            anime_mappings = {}
            anime_mappings_index = None
        elif anime_mappings_cfg is None:
            anime_mappings = self.get_anime_mappings()
            anime_mappings_index = self._get_anime_mappings_index()
        else:
            # Neither disabled (False) nor download (None): a pre-parsed map dict.
            assert isinstance(anime_mappings_cfg, dict)
            anime_mappings = anime_mappings_cfg
            anime_mappings_index = _build_anime_mappings_index(anime_mappings)

        if anidb_mappings_cfg is False:
            anidb_mappings = None
        elif anidb_mappings_cfg is None:
            anidb_mappings = self.get_anidb_mappings()
        else:
            # Neither disabled (False) nor download (None): a pre-parsed root.
            assert isinstance(anidb_mappings_cfg, ElementTree.Element)
            anidb_mappings = anidb_mappings_cfg

        if anibridge_mappings_cfg is False:
            anibridge = None
        elif anibridge_mappings_cfg is None:
            anibridge = self.get_anibridge_mappings()
        else:
            # A config-provided value is treated as a raw anibridge graph dict.
            # Built with logger=None to preserve the previous behaviour (the
            # base class loaded mappings before its logger existed).
            assert isinstance(anibridge_mappings_cfg, dict)
            anibridge = AniBridge(anibridge_mappings_cfg, logger=None)

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

        # Built with logger=None to preserve the previous behaviour (the base
        # class loaded mappings before its logger existed).
        return _load_mapping_by_mtime(
            ANIBRIDGE_MAPPINGS_FILE,
            lambda path: _parse_anibridge(path, None),
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

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: TmdbType = TmdbType.MOVIE,
    ) -> tuple[dict, list]:
        """Resolve external ids to a sorted {AniList id -> mapping} dict

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.

        Returns:
            tuple: (anilist_mappings, ids_to_drop), where anilist_mappings is a
                fresh copy of the resolved mappings (so a caller mutating it
                can't corrupt the memo) and ids_to_drop is the list of ignored
                AniList ids removed from the result, for the caller to log.
        """

        _validate_ids(tvdb_id, tmdb_id, imdb_id)

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

        # Return a copy so a caller mutating the result can't corrupt the memo
        return dict(anilist_mappings), ids_to_drop

    def get_mappings_from_anime_mappings(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: TmdbType = TmdbType.MOVIE,
        anilist_mappings: dict | None = None,
    ) -> dict:
        """Get mappings from the Anime ID mappings

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        _validate_ids(tvdb_id, tmdb_id, imdb_id)

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
        tmdb_type: TmdbType = TmdbType.MOVIE,
        anilist_mappings: dict | None = None,
    ) -> dict:
        """Get mappings from the AniBridge mappings

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        anibridge = self.anibridge
        if not anibridge:
            return anilist_mappings

        _validate_ids(tvdb_id, tmdb_id, imdb_id)

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
