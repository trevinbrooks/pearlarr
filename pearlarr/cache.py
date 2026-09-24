"""Persistent run cache: `CacheStore` owns `cache.db`, its schema, freshness checks, and writes.

Writes stage in one deferred transaction and commit only at a non-preview `save`. Do NOT set
`isolation_level=None` in `_connect`: real autocommit commits staged writes and breaks that gate.
One `CacheStore` per arr, never shared across arrs or threads.
"""

import contextlib
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, NamedTuple, TypedDict, cast, override

from seadex import EntryRecord

from . import __version__
from .config import Arr
from .json_narrow import is_json_obj
from .log import LOG_NAME
from .manual_import import GuardFacts
from .output import hub_note
from .sqlite_util import connect as _sqlite_connect
from .sqlite_util import open_or_quarantine, rollback_and_close
from .stamps import parse_stamp, parse_stamp_or_none, stamp_of

# `CREATE TABLE IF NOT EXISTS` never alters an existing table: shape changes need a SCHEMA_VERSION bump + migration.
# anilist_meta / sonarr_parse expose `fetched_at` as a VIRTUAL generated column, indexed for the TTL sweep's DELETE.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    arr        TEXT    NOT NULL,
    al_id      INTEGER NOT NULL,
    name       TEXT,
    url        TEXT,
    coverage   TEXT,
    updated_at TEXT,
    fallback_satisfied INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (arr, al_id)
);

CREATE TABLE IF NOT EXISTS torrent_hashes (
    arr      TEXT    NOT NULL,
    al_id    INTEGER NOT NULL,
    -- A hashless release round-trips as the `_NO_HASH` sentinel (the column is NOT NULL).
    infohash TEXT NOT NULL,
    PRIMARY KEY (arr, al_id, infohash),
    FOREIGN KEY (arr, al_id) REFERENCES entries (arr, al_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_torrent_hashes_infohash ON torrent_hashes (infohash);

CREATE TABLE IF NOT EXISTS anilist_meta (
    al_id      INTEGER PRIMARY KEY,
    record     BLOB NOT NULL,
    fetched_at TEXT GENERATED ALWAYS AS (record ->> 'fetched_at') VIRTUAL
);
CREATE INDEX IF NOT EXISTS ix_anilist_meta_fetched ON anilist_meta (fetched_at);

CREATE TABLE IF NOT EXISTS sonarr_parse (
    filename   TEXT PRIMARY KEY,
    record     BLOB NOT NULL,
    fetched_at TEXT GENERATED ALWAYS AS (record ->> 'fetched_at') VIRTUAL
);
CREATE INDEX IF NOT EXISTS ix_sonarr_parse_fetched ON sonarr_parse (fetched_at);

CREATE TABLE IF NOT EXISTS pending_imports (
    arr      TEXT NOT NULL,
    -- One record per torrent (lowercase hash): every entry listing it rides the record as a claim.
    infohash TEXT NOT NULL,
    record   BLOB NOT NULL,
    PRIMARY KEY (arr, infohash)
);

CREATE TABLE IF NOT EXISTS guard_facts (
    arr    TEXT    NOT NULL,
    al_id  INTEGER NOT NULL,
    -- One guard-evidence row per entry, refreshed whole at each seed. No FK: an orphan row is inert.
    record BLOB    NOT NULL,
    PRIMARY KEY (arr, al_id)
);

CREATE TABLE IF NOT EXISTS history_checkpoints (
    arr        TEXT PRIMARY KEY,
    since_date TEXT    NOT NULL,
    last_id    INTEGER NOT NULL
);
"""

# Current cache.db schema version, stored in `PRAGMA user_version`.
SCHEMA_VERSION = 5

_LOG = logging.getLogger(f"{LOG_NAME}.cache")


class CacheSchemaError(RuntimeError):
    """The cache db was written by a newer pearlarr. Refuse to open it.

    Deliberately NOT a `sqlite3.DatabaseError`, so the quarantine path never eats it.
    """


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table via PRAGMA table_info."""

    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
    """v0 = any pre-versioning db. Add the columns that shipped after the first cut."""

    if not _has_column(conn, "entries", "fallback_satisfied"):
        conn.execute("ALTER TABLE entries ADD COLUMN fallback_satisfied INTEGER NOT NULL DEFAULT 0")


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """Rebuild `pending_imports` with `al_id` in the PK (SQLite cannot alter a PK)."""

    if _has_column(conn, "pending_imports", "al_id"):
        return
    conn.execute(
        "CREATE TABLE pending_imports_v2 ("
        "arr TEXT NOT NULL, infohash TEXT NOT NULL, "
        "al_id INTEGER NOT NULL DEFAULT 0, record BLOB NOT NULL, "
        "PRIMARY KEY (arr, infohash, al_id))",
    )
    conn.execute(
        "INSERT INTO pending_imports_v2 (arr, infohash, al_id, record) "
        "SELECT arr, infohash, COALESCE(record ->> 'al_id', 0), record FROM pending_imports",
    )
    conn.execute("DROP TABLE pending_imports")
    conn.execute("ALTER TABLE pending_imports_v2 RENAME TO pending_imports")


def _migrate_2_to_3(conn: sqlite3.Connection) -> None:
    """Backfill one `guard_facts` row per Sonarr entry, then strip `guards` from every blob."""

    if conn.execute("SELECT EXISTS (SELECT 1 FROM guard_facts)").fetchone()[0]:
        return
    # `record` stays a BARE column, so SQLite's min/max guarantee picks it from the MAX(added_at) row.
    # `json_type` drops a JSON-null guards value, which `->` would pass as 'null'.
    conn.execute(
        "INSERT INTO guard_facts (arr, al_id, record) "
        "SELECT arr, al_id, jsonb(record -> 'guards') FROM ("
        "SELECT arr, al_id, record, MAX(record ->> 'added_at') "
        "FROM pending_imports "
        "WHERE arr = 'sonarr' AND al_id != 0 "
        "AND json_type(record, '$.guards') = 'object' "
        "GROUP BY arr, al_id)",
    )
    conn.execute("UPDATE pending_imports SET record = jsonb_remove(record, '$.guards')")


def _migrate_3_to_4(conn: sqlite3.Connection) -> None:
    """Null the id-form names an AniList outage once stored, so the next cached read resolves them."""

    # The pattern is the frozen v3-era form of `reporter.unresolved_label`, spelled
    # out on purpose so the migration never tracks the live label.
    conn.execute("UPDATE entries SET name = NULL WHERE name LIKE 'AniList #%'")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PendingV4:
    """One v4 `pending_imports` row narrowed once: the per-entry record shape before the torrent-level fold."""

    al_id: int
    series_id: int
    title: str | None
    coverage: str | None
    url: str | None
    ordered_episode_ids: list[int]
    names: dict[str, Any]
    preowned_episode_ids: list[int]
    slice_coverage: str | None
    added_at: str
    release_group: str
    is_dual_audio: bool
    seadex_files: list[str]
    release_sizes: list[int]
    file_episode_map: dict[str, list[int]]
    episode_ids: list[int]
    excluded_files: list[str]
    awaiting_cleanup: bool

    @classmethod
    def from_row(cls, raw: dict[str, Any]) -> "_PendingV4":
        """Narrow a stored v4 dict with the defaults its reader applied."""

        return cls(
            al_id=raw.get("al_id", 0),
            series_id=raw.get("series_id", 0),
            title=raw.get("title"),
            coverage=raw.get("coverage"),
            url=raw.get("url"),
            ordered_episode_ids=list(raw.get("ordered_episode_ids", [])),
            names=dict(raw.get("names", {})),
            preowned_episode_ids=list(raw.get("preowned_episode_ids", [])),
            slice_coverage=raw.get("slice_coverage"),
            added_at=raw.get("added_at", ""),
            release_group=raw.get("release_group", ""),
            is_dual_audio=bool(raw.get("is_dual_audio", False)),
            seadex_files=list(raw.get("seadex_files", [])),
            release_sizes=list(raw.get("release_sizes", [])),
            file_episode_map={name: list(ids) for name, ids in raw.get("file_episode_map", {}).items()},
            episode_ids=list(raw.get("episode_ids", [])),
            excluded_files=list(raw.get("excluded_files", [])),
            awaiting_cleanup=bool(raw.get("awaiting_cleanup")),
        )

    def claim(self) -> dict[str, Any]:
        """The v5 claim this row becomes, its fields carried as stored (an unscoped window stays unscoped).

        Only a row from before the window (a legacy `episode_ids` list) derives one: its map's and the list's ids.
        """

        ordered = self.ordered_episode_ids
        if not ordered and self.episode_ids:
            ordered = sorted(
                {i for ids in self.file_episode_map.values() for i in ids if i} | {i for i in self.episode_ids if i}
            )
        return {
            "al_id": self.al_id,
            "series_id": self.series_id,
            "title": self.title,
            "coverage": self.coverage,
            "url": self.url,
            "ordered_episode_ids": ordered,
            "names": self.names,
            "preowned_episode_ids": self.preowned_episode_ids,
            "slice_coverage": self.slice_coverage,
            "claimed_at": self.added_at,
        }


def _fold_v4_rows(infohash: str, rows: Sequence[_PendingV4]) -> dict[str, Any]:
    """One torrent's v4 rows (`al_id` order) as the v5 record: the first row's facts, the maps unioned."""

    first = rows[0]
    if any(row.release_group != first.release_group or row.is_dual_audio != first.is_dual_audio for row in rows):
        _LOG.debug(f"pending records of {infohash} disagree on their release; keeping entry {first.al_id}'s")
    stamps = [moment for row in rows if (moment := parse_stamp_or_none(row.added_at)) is not None]
    file_episode_map: dict[str, list[int]] = {}
    for row in rows:
        for name, ids in row.file_episode_map.items():
            file_episode_map.setdefault(name, ids)
    excluded = dict.fromkeys(name for row in rows for name in row.excluded_files if name not in file_episode_map)
    return {
        "infohash": infohash,
        "release_group": first.release_group,
        "is_dual_audio": first.is_dual_audio,
        "seadex_files": first.seadex_files,
        "added_at": stamp_of(min(stamps)) if stamps else "",
        "file_episode_map": file_episode_map,
        "claims": [row.claim() for row in rows],
        "excluded_files": list(excluded),
        "release_sizes": first.release_sizes,
        "awaiting_cleanup": all(row.awaiting_cleanup for row in rows),
    }


def _migrate_4_to_5(conn: sqlite3.Connection) -> None:
    """Fold the per-entry `pending_imports` rows into one record per torrent, each old row a claim."""

    if not _has_column(conn, "pending_imports", "al_id"):
        return
    grouped: dict[tuple[str, str], list[_PendingV4]] = {}
    for arr, infohash, rec_json in conn.execute(
        "SELECT arr, LOWER(infohash), json(record) FROM pending_imports ORDER BY arr, LOWER(infohash), al_id",
    ):
        grouped.setdefault((arr, infohash), []).append(_PendingV4.from_row(json.loads(rec_json)))
    conn.execute(
        "CREATE TABLE pending_imports_v5 ("
        "arr TEXT NOT NULL, infohash TEXT NOT NULL, record BLOB NOT NULL, "
        "PRIMARY KEY (arr, infohash))",
    )
    conn.executemany(
        "INSERT INTO pending_imports_v5 (arr, infohash, record) VALUES (?, ?, jsonb(?))",
        [(arr, infohash, json.dumps(_fold_v4_rows(infohash, rows))) for (arr, infohash), rows in grouped.items()],
    )
    conn.execute("DROP TABLE pending_imports")
    conn.execute("ALTER TABLE pending_imports_v5 RENAME TO pending_imports")


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_0_to_1,
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
    4: _migrate_4_to_5,
}


def _ensure_schema(conn: sqlite3.Connection, path: str) -> None:
    """Ensure the schema and bring an older db up to `SCHEMA_VERSION`."""

    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise CacheSchemaError(
            f"Cache database at {path} uses schema v{version}, newer than this pearlarr understands "
            f"(v{SCHEMA_VERSION}) - it was written by a newer release - upgrade pearlarr, or move the "
            "file away to start a fresh cache",
        )
    fresh = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0] == 0
    # `executescript` implicitly COMMITs first, a no-op this early in load.
    conn.executescript(_SCHEMA)
    if fresh:
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return
    for step in range(version, SCHEMA_VERSION):
        # SQLite DDL is transactional, so a failed step rolls back whole, stamp included.
        conn.execute("BEGIN")
        try:
            _MIGRATIONS[step](conn)
            conn.execute(f"PRAGMA user_version={step + 1}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        hub_note(f"Upgraded cache database schema v{step} -> v{step + 1}")


def record_payload(record: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    """The non-empty payload object a persisted record holds under `key`, else None."""

    if not isinstance(record, dict):
        return None
    payload = record.get(key)
    return payload if is_json_obj(payload) and payload else None


def stamp_is_fresh(record: dict[str, Any], cutoff: datetime) -> bool:
    """True if the record's `fetched_at` parses and is at or after `cutoff`."""

    try:
        return parse_stamp(record.get("fetched_at", "")) >= cutoff
    except (TypeError, ValueError):
        return False


class CacheRecord(TypedDict, total=False):
    """The fixed shape of a per-entry cache update / a `cache_details` payload."""

    name: str
    url: str
    coverage: str
    updated_at: "str | datetime"
    fallback_satisfied: bool
    """Whether a public fallback satisfied the title."""
    torrent_hashes: list[str | None]
    """A remembered list can carry `None` (a hashless release), which the store preserves."""


@dataclass(frozen=True, slots=True)
class CachedEntry:
    """The scalar columns of one `entries` row, read in a single query."""

    updated_at: str | None
    name: str | None
    url: str | None
    coverage: str | None
    fallback_satisfied: bool


@dataclass(frozen=True, slots=True)
class HistoryCheckpoint:
    """One arr's history cursor: the last-seen record's date + monotone id."""

    since_date: str
    """The raw ISO8601 stamp of the newest seen record (arr-clock domain)."""
    last_id: int
    """The per-arr autoincrement id, for strict `record.id > last_id` dedup across the re-query overlap."""


_ENTRY_SCALAR_COLUMNS = ("name", "url", "coverage", "updated_at", "fallback_satisfied")

# Sentinel stored in `torrent_hashes.infohash` (NOT NULL) for a remembered `None`. A real infohash is never empty.
_NO_HASH = ""


class _JsonBlock(NamedTuple):
    """One JSONB block: its table and key column(s), interpolated into the `_json_*` helpers' SQL."""

    table: str
    key_cols: tuple[str, ...]


_ANILIST_META = _JsonBlock("anilist_meta", ("al_id",))
_SONARR_PARSE = _JsonBlock("sonarr_parse", ("filename",))
_PENDING_IMPORTS = _JsonBlock("pending_imports", ("arr", "infohash"))
_GUARD_FACTS = _JsonBlock("guard_facts", ("arr", "al_id"))


class CacheStats(NamedTuple):
    """Row counts per cache table plus the on-disk size in bytes."""

    entries: int
    torrent_hashes: int
    anilist_meta: int
    sonarr_parse: int
    pending_imports: int
    guard_facts: int
    size_bytes: int


def _arr_key(arr: Arr) -> str:
    """The text stored for an `Arr` (`"sonarr"` / `"radarr"`)."""

    return str(arr)


def selection_digest_key(arr: Arr) -> str:
    """The `kv` key holding the selection digest an arr's verdicts were vouched under."""

    return f"selection_digest_{_arr_key(arr)}"


def _connect(path: str, *, ensure_wal: bool = True) -> sqlite3.Connection:
    """Open a cache-db connection (see `sqlite_util.connect`)."""

    return _sqlite_connect(path, ensure_wal=ensure_wal, foreign_keys=ensure_wal)


class AbstractCacheStore(ABC):
    """The instance facade run collaborators depend on. The `load` / `open_readonly` constructors stay off it."""

    @abstractmethod
    def save(self, *, preview: bool) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def selection_stale(self, arr: Arr, digest: str) -> bool: ...
    @abstractmethod
    def vouch_selection(self, arr: Arr, digest: str) -> None: ...
    @abstractmethod
    def check_al_id_in_cache(self, arr: Arr, al_id: int, seadex_entry: EntryRecord) -> bool: ...
    @abstractmethod
    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None: ...
    @abstractmethod
    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]: ...
    @abstractmethod
    def update_cache(self, arr: Arr, al_id: int, cache_details: CacheRecord | None = None) -> None: ...
    @abstractmethod
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]: ...
    @abstractmethod
    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None: ...
    @abstractmethod
    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None: ...
    @abstractmethod
    def evict_anilist_meta(self, cutoff: datetime) -> int: ...
    @abstractmethod
    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None: ...
    @abstractmethod
    def evict_sonarr_parse(self, cutoff: datetime) -> int: ...
    @abstractmethod
    def get_pending(self, arr: Arr) -> dict[str, dict[str, Any]]: ...
    @abstractmethod
    def get_pending_record(self, arr: Arr, infohash: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]: ...
    @abstractmethod
    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None: ...
    @abstractmethod
    def drop_pending(self, arr: Arr, infohash: str) -> None: ...
    @abstractmethod
    def other_arr_holds(self, arr: Arr, infohash: str) -> bool: ...
    @abstractmethod
    def put_guards(self, arr: Arr, al_id: int, guards: GuardFacts) -> None: ...
    @abstractmethod
    def get_guards(self, arr: Arr) -> dict[int, GuardFacts]: ...
    @abstractmethod
    def get_history_checkpoint(self, arr: Arr) -> HistoryCheckpoint | None: ...
    @abstractmethod
    def put_history_checkpoint(self, arr: Arr, checkpoint: HistoryCheckpoint) -> None: ...
    @abstractmethod
    def own_download_ids(self, arr: Arr) -> frozenset[str]: ...
    @abstractmethod
    def stats(self) -> CacheStats: ...
    @abstractmethod
    def integrity_check(self) -> str: ...


class CacheStore(AbstractCacheStore):
    """Owns the cache database: schema, freshness checks, and persistence."""

    def __init__(self, conn: sqlite3.Connection, path: str, *, on_memory: bool) -> None:
        self._conn = conn
        self._path = path
        # True while backed by an in-memory db. The first non-preview save promotes it to `path`.
        self._on_memory = on_memory

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        *,
        config_checksum: str,
    ) -> "CacheStore":
        """Open the cache db (or an in-memory stand-in) and reconcile the descriptor.

        A missing file opens `:memory:`, promoted at the first real save. `config_checksum` is informational only.
        """

        exists = os.path.exists(path)
        # Fail-closed on transient errors, fail-open (quarantine + `:memory:`) on real corruption.
        conn, fell_back = open_or_quarantine(
            path if exists else ":memory:",
            connect_fn=_connect,
            ensure=lambda c: _ensure_schema(c, path),
            what="Cache database",
            recovery="started a fresh cache (titles will be re-checked; grab-dedup and "
            "pending-import tracking reset, so recent grabs may be re-offered)",
        )
        if fell_back:
            exists = False
        store = cls(conn, path, on_memory=not exists)
        store._reconcile(config_checksum)
        return store

    @classmethod
    def open_readonly(cls, path: str) -> "CacheStore":
        """Open an existing cache db for a read-only diagnostic (`stats`/`check`).

        No WAL pragmas (a diagnostic must not mutate the file's journal mode), no schema ensure, no quarantine.
        """

        return cls(_connect(path, ensure_wal=False), path, on_memory=False)

    def _reconcile(self, config_checksum: str) -> None:
        """Stamp the current package version and config checksum into `kv`."""

        self._set_kv("pearlarr_version", __version__)
        self._set_kv("config_checksum", config_checksum)

    @override
    def save(self, *, preview: bool) -> None:
        """Persist staged writes, unless this is a preview run."""

        # Invariant: a preview run never commits, so every staged write is discarded on close and preview mode can
        # never mark a title as handled.
        if preview:
            return
        if self._on_memory:
            self._promote()
        else:
            self._conn.commit()

    def _promote(self) -> None:
        """Promote the in-memory db to the on-disk file, durably."""

        self._conn.commit()
        tmp_path = self._path + ".promote.tmp"
        # Clear any temp left by a previously-aborted promote.
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                os.remove(tmp_path + suffix)
        disk: sqlite3.Connection | None = None
        try:
            disk = sqlite3.connect(tmp_path)
            self._conn.backup(disk)
            disk.close()
            disk = None
            os.replace(tmp_path, self._path)  # atomic: cache.db is never a torn file
        finally:
            if disk is not None:
                disk.close()
            # Remove the temp (and any sidecars) if we failed before the rename.
            for suffix in ("", "-wal", "-shm"):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path + suffix)
        self._conn.close()
        self._conn = _connect(self._path)
        self._on_memory = False

    @override
    def close(self) -> None:
        """Roll back any uncommitted writes and close the connection."""

        rollback_and_close(self._conn)

    # -- descriptor (kv) -----------------------------------------------------

    def _set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    @override
    def selection_stale(self, arr: Arr, digest: str) -> bool:
        """Whether the arr's cached verdicts predate `digest` (matching settings moved)."""

        stored = self._get_kv(selection_digest_key(arr))
        return stored is not None and stored != digest

    @override
    def vouch_selection(self, arr: Arr, digest: str) -> None:
        """Stage `digest` as the settings this arr's verdicts reflect. Only a full-library re-check may vouch."""

        self._set_kv(selection_digest_key(arr), digest)

    # -- per-entry records (entries + torrent_hashes) ------------------------

    @override
    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """True if the cached entry's timestamp matches the SeaDex entry's `updated_at`."""

        sd_time_str = stamp_of(seadex_entry.updated_at)
        row = self._conn.execute(
            "SELECT updated_at FROM entries WHERE arr = ? AND al_id = ?",
            (_arr_key(arr), al_id),
        ).fetchone()
        return bool(row) and row[0] == sd_time_str

    @override
    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None:
        """The scalar columns of an entry's row in one query, or None (the `torrent_hashes` child set is excluded)."""

        row = self._conn.execute(
            "SELECT updated_at, name, url, coverage, fallback_satisfied FROM entries WHERE arr = ? AND al_id = ?",
            (_arr_key(arr), al_id),
        ).fetchone()
        return None if row is None else CachedEntry(row[0], row[1], row[2], row[3], bool(row[4]))

    @override
    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]:
        """Torrent hashes remembered for an entry (empty if none). A `None` marker (hashless release) survives."""

        rows = self._conn.execute(
            "SELECT infohash FROM torrent_hashes WHERE arr = ? AND al_id = ? ORDER BY infohash",
            (_arr_key(arr), al_id),
        ).fetchall()
        return cast("list[str | None]", [None if r[0] == _NO_HASH else r[0] for r in rows])

    @override
    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Merge fields into an entry's record: only supplied ones, staged until a save point."""

        details: dict[str, Any] = dict(cache_details or {})

        updated_at = details.get("updated_at")
        if isinstance(updated_at, datetime):
            details["updated_at"] = stamp_of(updated_at)

        arr_key = _arr_key(arr)

        scalar = [c for c in _ENTRY_SCALAR_COLUMNS if c in details]
        if scalar:
            # One upsert, not INSERT-then-UPDATE: an existing row updates ONLY the supplied columns (partial merge).
            # The names come from the closed _ENTRY_SCALAR_COLUMNS tuple, so the interpolation is safe.
            cols = ", ".join(scalar)
            placeholders = ", ".join("?" for _ in scalar)
            assignments = ", ".join(f"{c} = excluded.{c}" for c in scalar)
            self._conn.execute(
                f"INSERT INTO entries (arr, al_id, {cols}) VALUES (?, ?, {placeholders}) "
                f"ON CONFLICT (arr, al_id) DO UPDATE SET {assignments}",
                (arr_key, al_id, *(details[c] for c in scalar)),
            )
        else:
            # No scalar fields: just ensure the row exists (the FK target for torrent_hashes).
            self._conn.execute(
                "INSERT INTO entries (arr, al_id) VALUES (?, ?) ON CONFLICT (arr, al_id) DO NOTHING",
                (arr_key, al_id),
            )

        if "torrent_hashes" in details:
            self._conn.execute(
                "DELETE FROM torrent_hashes WHERE arr = ? AND al_id = ?",
                (arr_key, al_id),
            )
            hashes: list[str | None] = details["torrent_hashes"] or []
            # A None marker stores as the _NO_HASH sentinel (the column is NOT NULL), and ON CONFLICT collapses
            # repeated sentinels so at most one marker is kept.
            self._conn.executemany(
                "INSERT INTO torrent_hashes (arr, al_id, infohash) VALUES (?, ?, ?) "
                "ON CONFLICT (arr, al_id, infohash) DO NOTHING",
                [(arr_key, al_id, _NO_HASH if h is None else h) for h in hashes],
            )

    # -- JSONB record blocks (shared plumbing) --------------------------------
    # Table/column names come only from the closed _JsonBlock constants, so the f-string SQL isn't an injection
    # surface (same pattern as stats()).

    def _json_get(self, block: _JsonBlock, key: tuple[int | str, ...]) -> dict[str, Any] | None:
        """The stored record under `key` in a JSONB block, or None."""

        where = " AND ".join(f"{c} = ?" for c in block.key_cols)
        row = self._conn.execute(
            f"SELECT json(record) FROM {block.table} WHERE {where}",
            key,
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _json_put(self, block: _JsonBlock, key: tuple[int | str, ...], record: dict[str, Any]) -> None:
        """Upsert a record into a JSONB block (staged, persisted at a save point)."""

        cols = ", ".join(block.key_cols)
        placeholders = ", ".join("?" for _ in block.key_cols)
        self._conn.execute(
            f"INSERT INTO {block.table} ({cols}, record) VALUES ({placeholders}, jsonb(?)) "
            f"ON CONFLICT ({cols}) DO UPDATE SET record = excluded.record",
            (*key, json.dumps(record)),
        )

    def _pending_rows(self, sql: str, params: tuple[int | str, ...]) -> dict[str, dict[str, Any]]:
        """Deserialize a `SELECT infohash, json(record)` pending-imports query, keyed by infohash."""

        return {infohash: json.loads(rec_json) for infohash, rec_json in self._conn.execute(sql, params)}

    def _evict_stale_json(self, block: _JsonBlock, cutoff: datetime) -> int:
        """Delete records older than `cutoff` (or stamp-less, which is otherwise un-evictable). Count deleted."""

        cursor = self._conn.execute(
            f"DELETE FROM {block.table} WHERE fetched_at < ? OR fetched_at IS NULL",
            (stamp_of(cutoff),),
        )
        return cursor.rowcount

    # -- AniList meta (JSONB + TTL) ------------------------------------------

    @override
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield `(al_id, record)` for every stored record, age unfiltered (see `stamp_is_fresh`)."""

        for al_id, rec_json in self._conn.execute(
            "SELECT al_id, json(record) FROM anilist_meta",
        ):
            yield al_id, json.loads(rec_json)

    @override
    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        """The stored `{"fetched_at", "data"}` record for an id, or None."""

        return self._json_get(_ANILIST_META, (al_id,))

    @override
    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None:
        """Upsert an AniList-meta record (staged, persisted at a save point)."""

        self._json_put(_ANILIST_META, (al_id,), record)

    # -- Sonarr parse cache (JSONB + TTL) ------------------------------------

    @override
    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        """The stored `{"fetched_at", "episodes"}` record for a filename, or None."""

        return self._json_get(_SONARR_PARSE, (filename,))

    @override
    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None:
        """Upsert a Sonarr parse record (staged, persisted at a save point)."""

        self._json_put(_SONARR_PARSE, (filename,), record)

    # -- pending imports -----------------------------------------------------

    @override
    def get_pending(self, arr: Arr) -> dict[str, dict[str, Any]]:
        """All pending-import records for an arr, keyed by infohash (snapshot)."""

        return self._pending_rows(
            "SELECT infohash, json(record) FROM pending_imports WHERE arr = ?",
            (_arr_key(arr),),
        )

    @override
    def get_pending_record(self, arr: Arr, infohash: str) -> dict[str, Any] | None:
        """One torrent's stored record, or None."""

        return self._json_get(_PENDING_IMPORTS, (_arr_key(arr), infohash))

    @override
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]:
        """The records with a claim on one Sonarr `series_id`, keyed by infohash."""

        return self._pending_rows(
            "SELECT infohash, json(record) FROM pending_imports WHERE arr = ? AND EXISTS ("
            "SELECT 1 FROM json_each(record, '$.claims') WHERE value ->> 'series_id' = ?)",
            (_arr_key(arr), series_id),
        )

    @override
    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None:
        """Upsert one torrent's record (staged, persisted at a save point)."""

        self._json_put(_PENDING_IMPORTS, (_arr_key(arr), infohash), record)

    @override
    def drop_pending(self, arr: Arr, infohash: str) -> None:
        """Delete one torrent's record."""

        self._conn.execute("DELETE FROM pending_imports WHERE arr = ? AND infohash = ?", (_arr_key(arr), infohash))

    @override
    def other_arr_holds(self, arr: Arr, infohash: str) -> bool:
        """Whether the OTHER arr holds a record on the torrent (an exact key match: hashes are stored lowercase)."""

        row = self._conn.execute(
            "SELECT 1 FROM pending_imports WHERE arr != ? AND infohash = ? LIMIT 1",
            (_arr_key(arr), infohash),
        ).fetchone()
        return row is not None

    @override
    def put_guards(self, arr: Arr, al_id: int, guards: GuardFacts) -> None:
        """Upsert the entry's guard-evidence row (staged, persisted at a save point)."""

        self._json_put(_GUARD_FACTS, (_arr_key(arr), al_id), asdict(guards))

    @override
    def get_guards(self, arr: Arr) -> dict[int, GuardFacts]:
        """The arr's guard rows for entries with LIVE pending records. Rows are immortal, nothing deletes one."""

        return {
            al_id: GuardFacts.from_json(json.loads(rec_json))
            for al_id, rec_json in self._conn.execute(
                "SELECT al_id, json(record) FROM guard_facts WHERE arr = ? AND al_id IN ("
                "SELECT e.value ->> 'al_id' FROM pending_imports p, json_each(p.record, '$.claims') e WHERE p.arr = ?)",
                (_arr_key(arr), _arr_key(arr)),
            )
        }

    # -- history checkpoints --------------------------------------------------

    @override
    def get_history_checkpoint(self, arr: Arr) -> HistoryCheckpoint | None:
        """The arr's stored history cursor, or None before the first advance."""

        row = self._conn.execute(
            "SELECT since_date, last_id FROM history_checkpoints WHERE arr = ?",
            (_arr_key(arr),),
        ).fetchone()
        return None if row is None else HistoryCheckpoint(since_date=row[0], last_id=row[1])

    @override
    def put_history_checkpoint(self, arr: Arr, checkpoint: HistoryCheckpoint) -> None:
        """Upsert the arr's history cursor (staged, persisted at a save point)."""

        self._conn.execute(
            "INSERT INTO history_checkpoints (arr, since_date, last_id) VALUES (?, ?, ?) "
            "ON CONFLICT (arr) DO UPDATE SET since_date = excluded.since_date, last_id = excluded.last_id",
            (_arr_key(arr), checkpoint.since_date, checkpoint.last_id),
        )

    @override
    def own_download_ids(self, arr: Arr) -> frozenset[str]:
        """Casefolded infohashes of our own grabs (remembered + pending) for an arr."""

        rows = self._conn.execute(
            "SELECT infohash FROM torrent_hashes WHERE arr = ? AND infohash != ? "
            "UNION SELECT infohash FROM pending_imports WHERE arr = ?",
            (_arr_key(arr), _NO_HASH, _arr_key(arr)),
        ).fetchall()
        return frozenset(str(row[0]).casefold() for row in rows)

    # -- maintenance: eviction, stats, integrity -----------------------------

    @override
    def evict_anilist_meta(self, cutoff: datetime) -> int:
        """Delete AniList-meta records older than `cutoff` (or stamp-less). Count."""

        return self._evict_stale_json(_ANILIST_META, cutoff)

    @override
    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        """Delete Sonarr parse records older than `cutoff` (or stamp-less). Count."""

        return self._evict_stale_json(_SONARR_PARSE, cutoff)

    def _count(self, table: str) -> int:
        """Row count of one table (the name comes from `stats`'s closed literals)."""

        row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    @override
    def stats(self) -> CacheStats:
        """Row counts per table plus the on-disk size in bytes, incl. WAL (0 while in memory)."""

        size = 0
        if not self._on_memory:
            for suffix in ("", "-wal"):
                with contextlib.suppress(OSError):
                    size += os.path.getsize(self._path + suffix)
        return CacheStats(
            entries=self._count("entries"),
            torrent_hashes=self._count("torrent_hashes"),
            anilist_meta=self._count("anilist_meta"),
            sonarr_parse=self._count("sonarr_parse"),
            pending_imports=self._count("pending_imports"),
            guard_facts=self._count("guard_facts"),
            size_bytes=size,
        )

    @override
    def integrity_check(self) -> str:
        """Run `PRAGMA quick_check` and return its result (`"ok"` when healthy)."""

        row = self._conn.execute("PRAGMA quick_check").fetchone()
        return str(row[0]) if row else "unknown"
