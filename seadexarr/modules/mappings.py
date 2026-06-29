"""ID-mapping resolution: external sources -> AniList ids.

``MappingResolver`` owns the three immutable mapping sources (the Kometa
Anime-IDs JSON, the AniDB anime-list XML, and the anibridge graph). It
downloads/refreshes each file, and - only when a file's *content* changes -
parses and indexes it once into a dedicated SQLite store (``mappings.db``, via
:class:`~seadexarr.modules.mapping_store.MappingStore`). Lookups are then served
from SQL, so a process whose source files are unchanged never re-parses or holds
the ~50MB of parsed structures resident. It resolves an Arr's external ids (TVDB /
TMDB / IMDb) to the AniList ids they map to.

Visibility: the resolver logs each source's download / cache-hit / parse step and
streams download progress, and the downloader uses a per-read socket timeout so a
stalled fetch fails (and is reported) instead of hanging the run forever.
"""

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Protocol, cast
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .anibridge import AniBridge, AniBridgeEntry, AniBridgeGraph, AniBridgeLookup
from .mapping_store import (
    INLINE_DIGEST,
    SOURCE_ANIBRIDGE,
    SOURCE_ANIDB,
    SOURCE_ANIME_IDS,
    AnimeIdRow,
    MappingStore,
)
from .paths import resolve_paths
from .seadex_types import TvdbMappings

type AnimeIdsRecord = dict[str, Any]
"""One Kometa Anime-IDs record (``{field: value}``).

A loosely-shaped producer dict: a flat record carrying mixed-typed fields
(``anilist_id``/``tvdb_id``/``tmdb_movie_id``/``imdb_id`` ints or strs,
``tvdb_season``/``tvdb_epoffset`` ints). It is read at the raw->typed boundary
(:func:`_entry_from_raw` / :func:`_anime_ids_rows`) and never modeled as a domain
object, so it stays a loose ``dict[str, Any]`` like
:data:`~seadexarr.modules.anibridge.AniBridgeEntry`.
"""

type AnimeIdsMap = dict[str, AnimeIdsRecord]
"""The parsed Kometa Anime-IDs JSON: ``{name -> record}`` (~16k entries)."""


class TmdbType(StrEnum):
    """Which TMDB id space an external lookup is scoped to."""

    MOVIE = "movie"
    SHOW = "show"


class MappingMode(Enum):
    """The two closed kinds of episode mapping a SeaDex mapping can carry.

    A mapping either ships explicit AniBridge ``tvdb_mappings`` (season ->
    episode ranges) or it doesn't, in which case the Sonarr path falls back to
    Anime-ID season/offset logic. There is no third kind, so dispatch on this
    enum needs no defensive default arm.
    """

    ANIME_IDS = "anime_ids"
    ANIBRIDGE = "anibridge"


@dataclass(frozen=True, slots=True)
class MappingEntry:
    """One resolved AniList mapping, as the sync strategies consume it.

    Built at the raw->typed boundary from the two producers: the AniBridge graph
    attaches ``tvdb_mappings`` (season -> episode ranges) only on a TVDB lookup,
    while the Kometa Anime-IDs records carry the flat ``tvdb_season`` /
    ``tvdb_epoffset`` fields. Both normalise into one typed record with attribute
    reads; the ``tvdb_mappings``-present-or-not distinction becomes the typed
    :attr:`mode` discriminant.
    """

    anilist_id: int
    tvdb_id: int | None = None
    tvdb_season: int = -1
    tvdb_epoffset: int = 0
    tvdb_mappings: TvdbMappings | None = None
    tmdb_movie_id: int | None = None
    imdb_id: str | None = None
    anidb_id: int | None = None

    @property
    def mode(self) -> MappingMode:
        """Which episode-mapping mode this entry drives.

        ANIBRIDGE iff ``tvdb_mappings`` was attached. Tested with ``is not
        None`` (not truthiness) on purpose: an *empty* ``tvdb_mappings`` dict is
        still ANIBRIDGE, exactly as the former ``"tvdb_mappings" in mapping``
        key-presence check was.
        """

        return MappingMode.ANIBRIDGE if self.tvdb_mappings is not None else MappingMode.ANIME_IDS


def _entry_from_raw(anilist_id: int, raw: AnimeIdsRecord | AniBridgeEntry) -> MappingEntry:
    """Build a :class:`MappingEntry` from a producer's raw dict.

    The one place a loosely-typed producer dict (an AniBridge ``_consumer_entry``
    dict) becomes a typed record. Defaults mirror the former ``.get(..., default)``
    reads exactly, so an AniBridge entry (which carries no ``tvdb_season`` /
    ``tvdb_epoffset``) lands on ``-1`` / ``0``.

    Args:
        anilist_id (int): AniList id for this entry.
        raw (AnimeIdsRecord | AniBridgeEntry): Producer dict; only the enumerated
            keys are read.
    """

    # Coalesce a present-but-null season/epoffset to the sentinel, matching
    # _anime_ids_rows (a JSON null would otherwise violate the int fields).
    season = raw.get("tvdb_season", -1)
    epoffset = raw.get("tvdb_epoffset", 0)
    return MappingEntry(
        anilist_id=anilist_id,
        tvdb_id=raw.get("tvdb_id"),
        tvdb_season=-1 if season is None else season,
        tvdb_epoffset=0 if epoffset is None else epoffset,
        tvdb_mappings=raw.get("tvdb_mappings"),
        tmdb_movie_id=raw.get("tmdb_movie_id"),
        imdb_id=raw.get("imdb_id"),
        anidb_id=raw.get("anidb_id"),
    )


def _entry_from_anime_row(row: AnimeIdRow) -> MappingEntry:
    """Build a :class:`MappingEntry` from a stored :class:`AnimeIdRow`.

    The SQL twin of :func:`_entry_from_raw` for Kometa records: a row never
    carries ``tvdb_mappings`` (only AniBridge does), so ``mode`` is ANIME_IDS.
    ``row.tmdb_show_id`` has no MappingEntry field, so it is simply not read.
    """

    return MappingEntry(
        anilist_id=row.anilist_id,
        tvdb_id=row.tvdb_id,
        tvdb_season=row.tvdb_season,
        tvdb_epoffset=row.tvdb_epoffset,
        tvdb_mappings=None,
        tmdb_movie_id=row.tmdb_movie_id,
        imdb_id=row.imdb_id,
        anidb_id=row.anidb_id,
    )


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
ANIBRIDGE_MAPPINGS_URL = (
    f"https://github.com/anibridge/anibridge-mappings/releases/download/{ANIBRIDGE_RELEASE}/mappings.min.json"
)
ANIBRIDGE_MAPPINGS_FILE = f"anibridge_mappings_{ANIBRIDGE_RELEASE}.json"

# Per-read socket timeout for a source download. A stalled connection raises after
# this many seconds (reported, then the run skips and retries) instead of hanging
# forever, which is the behaviour the previous timeout-less urlretrieve had.
DOWNLOAD_TIMEOUT_S = 30


def _file_digest(path: str) -> str:
    """sha256 of a file's bytes - the content key that gates a re-parse."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadProgress(Protocol):
    """A sink for streaming download progress - drives the boot cockpit's bar.

    Structural, so the boot view's step handle satisfies it without this data
    module importing the UI layer. ``fraction`` is 0-1 download completion;
    ``detail`` is a short human note (the file + MB).
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...


def _download_file(
    url: str,
    dest: str,
    *,
    timeout: int,
    logger: logging.Logger | None,
    label: str,
    progress: DownloadProgress | None = None,
) -> None:
    """Stream ``url`` to ``dest`` with a socket timeout and throttled progress.

    Writes to a ``.part`` temp and atomically renames on success, so a failed or
    stalled download never leaves a truncated source file for the next run to
    digest and trust. A per-read timeout bounds a stall (the read raises rather
    than blocking forever). Progress is reported about once per MB - to the boot
    cockpit's live bar when a ``progress`` sink is given, else (standalone use) as
    a throttled DEBUG line.
    """

    tmp = dest + ".part"
    try:
        req = Request(url, headers={"User-Agent": "seadexarr"})
        with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:  # noqa: S310 (trusted https sources)
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            next_mark = 1 << 20
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if got >= next_mark:
                    if progress is not None and total:
                        progress.progress(got / total, f"{label} · {got >> 20}/{total >> 20} MB")
                    elif logger is not None:
                        suffix = f"{got >> 20}/{total >> 20} MB ({got * 100 // total}%)" if total else f"{got >> 20} MB"
                        logger.debug(f"  ...downloading {label}: {suffix}")
                    next_mark = got + (1 << 20)
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def _parse_anime_mappings(path: str) -> AnimeIdsMap:
    """Load the Kometa Anime-IDs JSON map from disk."""

    with open(path) as f:
        # Raw JSON boundary: json.load returns Any; narrow to the known map shape.
        return cast("AnimeIdsMap", json.load(f))


def _anime_ids_rows(anime_mappings: AnimeIdsMap) -> list[AnimeIdRow]:
    """Flatten the Kometa map into anime_ids store rows (first-seen order).

    Every record yields a row (a NULL ``anilist_id`` is kept, so the library-filter
    candidate sets match the former full-map scan; id->entry lookups filter those
    out, as the former reverse index did). Each field uses the same
    ``.get(..., default)`` reads :func:`_entry_from_raw` used, so the SQL-served
    entry is identical.
    """

    rows: list[AnimeIdRow] = []
    for record in anime_mappings.values():
        # ``.get(key, default)`` only substitutes the default for an ABSENT key; a
        # present-but-null JSON value (``"tvdb_season": null``) returns None, which
        # the NOT NULL ``tvdb_season`` / ``tvdb_epoffset`` columns reject - and that
        # IntegrityError aborts the whole populate (then the :memory: fail-open
        # re-parses the same data and re-raises, taking down the entire run).
        # Coalesce an explicit null to the same sentinel an absent key gets.
        season = record.get("tvdb_season", -1)
        epoffset = record.get("tvdb_epoffset", 0)
        rows.append(
            AnimeIdRow(
                # NULL anilist_id is kept here but filtered out at query time; the
                # field is typed int per that IS NOT NULL contract, so cast the
                # producer's nullable value (it rides through as a stored NULL).
                anilist_id=cast("int", record.get("anilist_id")),
                tvdb_id=record.get("tvdb_id"),
                tvdb_season=-1 if season is None else season,
                tvdb_epoffset=0 if epoffset is None else epoffset,
                tmdb_movie_id=record.get("tmdb_movie_id"),
                tmdb_show_id=record.get("tmdb_show_id"),
                imdb_id=record.get("imdb_id"),
                anidb_id=record.get("anidb_id"),
            ),
        )
    return rows


def _parse_anidb_mappings(path: str) -> ElementTree.Element:
    """Parse the AniDB anime-list XML and return its root element."""

    return ElementTree.parse(path).getroot()


def _anidb_rows(root: ElementTree.Element) -> tuple[list[tuple[Any, ...]], list[tuple[int]]]:
    """Flatten the AniDB XML into ``anidb_mapping`` rows + the ambiguous-id set.

    Reproduces the former ``anidb_anime_by_id`` + ``_parse_anidb_mapping_dict``
    behaviour exactly, but once at populate time:

    * An anidb id appearing in more than one ``<anime>`` element is *ambiguous*
      (the former ``len(...) > 1`` -> raise case); it is recorded and stored with
      no mapping rows, and :meth:`MappingResolver.anidb_mapping_dict` raises on it.
    * For an unambiguous id, each ``<mapping-list>/<mapping>`` with text is parsed
      to ``{tvdb_ep: anidb_ep}`` keyed by ``tvdbseason``; a repeated season is
      last-wins, matching the former dict assignment. Malformed mappings are
      skipped (the old code only crashed if such an anime was looked up; populating
      every anime must tolerate what it never reached).
    """

    counts: dict[int, int] = {}
    season_maps: dict[int, dict[int, dict[int, int]]] = {}
    for anime in root.findall("anime"):
        anidbid = anime.get("anidbid")
        if anidbid is None:
            continue
        try:
            aid = int(anidbid)
        except ValueError:
            continue
        counts[aid] = counts.get(aid, 0) + 1

        season_map: dict[int, dict[int, int]] = {}
        for ms in anime.findall("mapping-list"):
            for i in ms.findall("mapping"):
                if not i.text:
                    continue
                try:
                    season = int(i.attrib["tvdbseason"])
                    pairs = [x.split("-") for x in i.text.strip(";").split(";")]
                    # orientation {tvdb_ep: anidb_ep}; last <mapping> per season wins
                    season_map[season] = {int(p[1]): int(p[0]) for p in pairs}
                except (KeyError, ValueError, IndexError):
                    continue
        season_maps[aid] = season_map

    rows: list[tuple[Any, ...]] = []
    for aid, season_map in season_maps.items():
        if counts[aid] > 1:
            # Ambiguous: never read (lookup raises first), so store no rows.
            continue
        for season, mapping in season_map.items():
            for tvdb_ep, anidb_ep in mapping.items():
                rows.append((aid, season, tvdb_ep, anidb_ep))

    ambiguous = [(aid,) for aid, count in counts.items() if count > 1]
    return rows, ambiguous


class MappingResolver:
    """Resolve external Arr ids to AniList ids via three SQL-backed sources.

    Downloads-if-stale, then - only when a source's content digest changed -
    parses and indexes it into ``mappings.db`` (owned here). Lookups are answered
    from SQL. ``get_anilist_ids`` returns the AniList mappings plus the ids it
    dropped because the user chose to ignore them, so the caller (not the
    resolver) owns the per-id logging.
    """

    def __init__(
        self,
        *,
        cache_time: int,
        ignore_anilist_ids: set[int],
        anime_mappings_cfg: AnimeIdsMap | bool | None,
        anidb_mappings_cfg: ElementTree.Element | bool | None,
        anibridge_mappings_cfg: AniBridgeGraph | bool | None,
        mappings_db: str = ":memory:",
        logger: logging.Logger | None = None,
        progress: DownloadProgress | None = None,
    ) -> None:
        """Open the store and load (parse-if-changed) the mapping sources.

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
            mappings_db (str): Path to the SQLite mapping cache; defaults to an
                in-memory db (tests / pre-parsed configs).
            logger (logging.Logger | None): For download/parse visibility (DEBUG;
                the boot cockpit owns the INFO-level startup narrative).
            progress (DownloadProgress | None): The boot cockpit's step handle, fed
                streaming download progress for the live bar. Defaults to None (no
                cockpit: progress falls back to throttled DEBUG lines).
        """

        self.cache_time = cache_time
        self.ignore_anilist_ids = ignore_anilist_ids
        self.logger = logger
        self._progress = progress

        # The downloaded sources are cached next to mappings.db in the data dir; an
        # in-memory db (tests / pre-parsed configs) falls back to the default data
        # dir, so a source is never written to the working directory.
        base = os.path.dirname(os.path.abspath(mappings_db)) if mappings_db != ":memory:" else resolve_paths().data_dir
        self._anime_ids_path = os.path.join(base, ANIME_IDS_FILE)
        self._anidb_path = os.path.join(base, ANIDB_MAPPINGS_FILE)
        self._anibridge_path = os.path.join(base, ANIBRIDGE_MAPPINGS_FILE)

        # Memoize get_anilist_ids per identifying key, so the prefetch pass and the
        # main loop don't recompute (and re-query) it twice per item.
        self._anilist_ids_cache: dict[
            tuple[int | None, int | None, str | None, TmdbType],
            tuple[dict[int, MappingEntry], list[int]],
        ] = {}

        # Set by _build; the AniBridge facade is SQL-backed (None when disabled).
        self.anibridge: AniBridge | None = None
        self._anime_enabled = False
        self._anidb_enabled = False

        self._store = MappingStore.open(mappings_db, logger=logger)
        # Close the store on ANY construction failure: the resolver is never
        # returned on a raise, so nobody else can call close() - a leak that would
        # accumulate a SQLite fd/WAL handle every scheduled cycle on a persistent
        # download/parse failure (the old resolver held no resources).
        try:
            try:
                self._build(anime_mappings_cfg, anidb_mappings_cfg, anibridge_mappings_cfg)
            except sqlite3.DatabaseError:
                # A read/write error against the on-disk db (e.g. disk full
                # mid-populate): the atomic replace already rolled back, so nothing
                # half-built persists and is_fresh stays false. Fall back to a fresh
                # :memory: store and re-parse so the run still works rather than
                # serving partial mappings.
                if mappings_db == ":memory:":
                    raise
                if self.logger is not None:
                    self.logger.warning(
                        "mappings.db unusable; rebuilding indexes in memory for this run",
                        exc_info=True,
                    )
                self._store.close()
                self._store = MappingStore.open(":memory:", logger=logger)
                self._build(anime_mappings_cfg, anidb_mappings_cfg, anibridge_mappings_cfg)
        except BaseException:
            self._store.close()
            raise

    # -- construction --------------------------------------------------------

    def _build(
        self,
        anime_cfg: AnimeIdsMap | bool | None,
        anidb_cfg: ElementTree.Element | bool | None,
        anibridge_cfg: AniBridgeGraph | bool | None,
    ) -> None:
        """Load each source into the store per its config (download / inline / off).

        Safe to call twice (the fail-open retry does): each ``replace_*`` fully
        replaces a source's rows, and the enabled flags / ``anibridge`` facade are
        simply re-set.
        """

        if anime_cfg is False:
            self._anime_enabled = False
        elif anime_cfg is None:
            self._anime_enabled = True
            self._load_anime_ids()
        else:
            assert isinstance(anime_cfg, dict)
            self._anime_enabled = True
            self._store.replace_anime_ids(INLINE_DIGEST, _anime_ids_rows(anime_cfg))

        if anidb_cfg is False:
            self._anidb_enabled = False
        elif anidb_cfg is None:
            self._anidb_enabled = True
            self._load_anidb()
        else:
            assert isinstance(anidb_cfg, ElementTree.Element)
            self._anidb_enabled = True
            rows, ambiguous = _anidb_rows(anidb_cfg)
            self._store.replace_anidb(INLINE_DIGEST, rows, ambiguous)

        if anibridge_cfg is False:
            self.anibridge = None
        elif anibridge_cfg is None:
            self._load_anibridge()
        else:
            assert isinstance(anibridge_cfg, dict)
            ab = AniBridge(anibridge_cfg, logger=self.logger)
            self._store.replace_anibridge(INLINE_DIGEST, *ab.to_rows())
            self.anibridge = AniBridge.from_store(self._store)

    def _maybe_download(self, file: str, url: str, label: str) -> None:
        """Download ``file`` if missing, or refresh it once it's past cache_time."""

        # The data dir exists in the normal CLI flow (ensure_data_dir); create it
        # here too so the in-memory/standalone fallback never fails on a first write.
        os.makedirs(os.path.dirname(file), exist_ok=True)

        if not os.path.exists(file):
            self._log(f"Downloading {label}")
            _download_file(
                url, file, timeout=DOWNLOAD_TIMEOUT_S, logger=self.logger, label=label, progress=self._progress
            )
            return

        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file))
        if age.days >= self.cache_time:
            self._log(f"Refreshing {label} (cached {age.days}d >= {self.cache_time}d)")
            try:
                _download_file(
                    url, file, timeout=DOWNLOAD_TIMEOUT_S, logger=self.logger, label=label, progress=self._progress
                )
            except OSError as e:
                # A transient blip refreshing a stale-but-valid cached source must not abort
                # the run: the atomic .part write left the cached file intact, so fall open to
                # it and warn (next cycle re-attempts). A first-ever download above stays fatal.
                if self.logger is not None:
                    self.logger.warning(f"Could not refresh {label} ({e}); using the cached copy")

    def _load_anime_ids(self) -> None:
        """Download + (re)index the Kometa Anime-IDs map only if its content changed."""

        self._maybe_download(self._anime_ids_path, ANIME_IDS_URL, "anime_ids.json")
        digest = _file_digest(self._anime_ids_path)
        if self._store.is_fresh(SOURCE_ANIME_IDS, digest):
            self._log("anime_ids.json unchanged; using cached index")
            return
        self._log("Parsing + indexing anime_ids.json ...")
        t0 = time.perf_counter()
        rows = _anime_ids_rows(_parse_anime_mappings(self._anime_ids_path))
        self._store.replace_anime_ids(digest, rows)
        self._log(f"Indexed anime_ids.json ({len(rows)} records, {time.perf_counter() - t0:.2f}s)")

    def _load_anidb(self) -> None:
        """Download + (re)index the AniDB anime-list XML only if its content changed."""

        self._maybe_download(self._anidb_path, ANIDB_MAPPINGS_URL, "anidb anime-list")
        digest = _file_digest(self._anidb_path)
        if self._store.is_fresh(SOURCE_ANIDB, digest):
            self._log("anidb anime-list unchanged; using cached index")
            return
        self._log("Parsing + indexing anidb anime-list ...")
        t0 = time.perf_counter()
        rows, ambiguous = _anidb_rows(_parse_anidb_mappings(self._anidb_path))
        self._store.replace_anidb(digest, rows, ambiguous)
        self._log(f"Indexed anidb anime-list ({len(rows)} mappings, {time.perf_counter() - t0:.2f}s)")

    def _load_anibridge(self) -> None:
        """Download + (re)index the anibridge graph only if its content changed."""

        self._maybe_download(self._anibridge_path, ANIBRIDGE_MAPPINGS_URL, "anibridge mappings")
        digest = _file_digest(self._anibridge_path)
        if self._store.is_fresh(SOURCE_ANIBRIDGE, digest):
            self._log("anibridge mappings unchanged; using cached index")
        else:
            self._log("Parsing + indexing anibridge mappings ...")
            t0 = time.perf_counter()
            with open(self._anibridge_path) as f:
                graph = json.load(f)
            ab = AniBridge(graph, logger=self.logger)
            self._store.replace_anibridge(digest, *ab.to_rows())
            self._log(f"Indexed anibridge mappings ({len(ab)} AniList ids, {time.perf_counter() - t0:.2f}s)")
        self.anibridge = AniBridge.from_store(self._store)

    def _log(self, message: str) -> None:
        """Emit a DEBUG line if a logger was injected (no-op otherwise).

        The boot cockpit owns the INFO-level startup narrative (the "Refreshing
        mappings" step + its live bar + finish line), so these per-source
        download/parse/cache-hit notes stay at DEBUG to keep a normal run calm.
        """

        if self.logger is not None:
            self.logger.debug(message)

    def sources_summary(self) -> str:
        """A compact "which sources are active" note for the boot step detail."""

        names: list[str] = []
        if self._anime_enabled:
            names.append("anime-ids")
        if self._anidb_enabled:
            names.append("anidb")
        if self.anibridge is not None:
            names.append("anibridge")
        return " · ".join(names) if names else "none enabled"

    def close(self) -> None:
        """Close the mapping store (idempotent); call once per cycle at teardown."""

        self._store.close()

    # -- library-filter id sets ---------------------------------------------

    def anime_id_set(self, column: str) -> set[int | str]:
        """DISTINCT external ids the Anime-IDs source carries for ``column``.

        Backs the library-filter candidate sets (symmetric with AniBridge's
        ``all_*`` sets), so ``collect_anime_items`` no longer scans the full map.
        Returns an empty set when the source is disabled.
        """

        if not self._anime_enabled:
            return set[int | str]()
        return self._store.anime_ids_distinct(column)

    @property
    def has_anidb(self) -> bool:
        """True when the AniDB source is enabled (the former ``anidb_mappings is not None``)."""

        return self._anidb_enabled

    # -- anidb episode mapping ----------------------------------------------

    def anidb_mapping_dict(self, anidb_id: int, tvdb_season: int) -> dict[int, dict[int, int]]:
        """Return ``{tvdb_season: {tvdb_ep: anidb_ep}}`` for an AniDB id + season.

        Replaces the former ``anidb_anime_by_id`` + ``_parse_anidb_mapping_dict``
        pair: ``{}`` when the source is disabled, the id is unknown, or it has no
        mapping for the season; raises the same ``ValueError`` when the id was
        ambiguous (appeared in more than one ``<anime>`` element).

        Args:
            anidb_id (int): AniDB id to look up.
            tvdb_season (int): The TVDB season AniList resolved to.
        """

        if not self._anidb_enabled:
            return {}
        if self._store.anidb_is_ambiguous(anidb_id):
            raise ValueError("Multiple AniDB mappings found. This should not happen!")
        rows = self._store.anidb_rows(anidb_id, tvdb_season)
        if not rows:
            return {}
        return {tvdb_season: dict(rows)}

    # -- anilist resolution --------------------------------------------------

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: TmdbType = TmdbType.MOVIE,
    ) -> tuple[dict[int, MappingEntry], list[int]]:
        """Resolve external ids to a sorted {AniList id -> mapping} dict.

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

        # The mapping computation is deterministic for a given set of identifying
        # args, so memoize it and only redo the per-call logging.
        key = (tvdb_id, tmdb_id, imdb_id, tmdb_type)
        if key in self._anilist_ids_cache:
            anilist_mappings, ids_to_drop = self._anilist_ids_cache[key]
        else:
            anilist_mappings: dict[int, MappingEntry] = {}

            # AniBridge is the primary source: its richer per-season episode
            # offsets win, so query it first and key results by AniList ID.
            if self.anibridge:
                anilist_mappings = self.get_mappings_from_anibridge_mappings(
                    tvdb_id=tvdb_id,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    tmdb_type=tmdb_type,
                    anilist_mappings=anilist_mappings,
                )

            # Then fall back to the Kometa Anime IDs for anything AniBridge
            # doesn't cover (it only adds AniList IDs not already present).
            if self._anime_enabled:
                anilist_mappings = self.get_mappings_from_anime_mappings(
                    tvdb_id=tvdb_id,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    tmdb_type=tmdb_type,
                    anilist_mappings=anilist_mappings,
                )

            # Drop any AniList IDs the user has chosen to ignore.
            ids_to_drop = [al_id for al_id in self.ignore_anilist_ids if al_id in anilist_mappings]
            for al_id in ids_to_drop:
                del anilist_mappings[al_id]

            # Sort by AniList ID.
            anilist_mappings = dict(sorted(anilist_mappings.items()))

            self._anilist_ids_cache[key] = (anilist_mappings, ids_to_drop)

        # Return fresh copies of BOTH so a caller mutating either can't corrupt the
        # memo (the entries are frozen, so a shallow dict/list copy is enough).
        return dict(anilist_mappings), list(ids_to_drop)

    def get_mappings_from_anime_mappings(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        tmdb_type: TmdbType = TmdbType.MOVIE,
        anilist_mappings: dict[int, MappingEntry] | None = None,
    ) -> dict[int, MappingEntry]:
        """Get mappings from the Anime ID mappings (served from SQL).

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary.
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        _validate_ids(tvdb_id, tmdb_id, imdb_id)

        if not self._anime_enabled:
            return anilist_mappings

        # Add the first row seen for each AniList id (rows come back in first-seen
        # order), matching the previous "don't clobber an id another query already
        # produced" behaviour.
        def merge(column: str, value: object) -> None:
            for row in self._store.anime_ids_lookup(column, value):
                if row.anilist_id not in anilist_mappings:
                    anilist_mappings[row.anilist_id] = _entry_from_anime_row(row)

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
        anilist_mappings: dict[int, MappingEntry] | None = None,
    ) -> dict[int, MappingEntry]:
        """Get mappings from the AniBridge mappings (served from SQL).

        Args:
            tvdb_id (int): TVDB ID
            tmdb_id (int): TMDB ID
            imdb_id (int): IMDb ID
            tmdb_type (TmdbType): Which TMDB id space the tmdb_id is in.
            anilist_mappings (dict): Dictionary of AniList mappings.
                Defaults to None, which will create a new dictionary.
        """

        if anilist_mappings is None:
            anilist_mappings = {}

        anibridge = self.anibridge
        if not anibridge:
            return anilist_mappings

        _validate_ids(tvdb_id, tmdb_id, imdb_id)

        # Add any AniList IDs the indexes resolve for the supplied ids, without
        # clobbering matches an earlier id already produced (tvdb > tmdb > imdb).
        def merge(found: AniBridgeLookup) -> None:
            for anilist_id, entry in found.items():
                if anilist_id not in anilist_mappings:
                    anilist_mappings[anilist_id] = _entry_from_raw(anilist_id, entry)

        if tvdb_id is not None:
            merge(anibridge.lookup_by_tvdb(tvdb_id))
        if tmdb_id is not None:
            merge(anibridge.lookup_by_tmdb(tmdb_id, tmdb_type))
        if imdb_id is not None:
            merge(anibridge.lookup_by_imdb(imdb_id))

        return anilist_mappings
