"""Persistent run cache: `CacheStore` owns `cache.db`, its schema, freshness checks, and writes.

Writes stage in one deferred transaction and commit only at a non-preview `save`. Do NOT set
`isolation_level=None` in `_connect`: real autocommit commits staged writes and breaks that gate.
One `CacheStore` per arr, never shared across arrs or threads.
"""

import contextlib
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, NamedTuple, TypedDict, cast, override

from seadex import EntryRecord

from . import __version__
from .config import Arr
from .manual_import import GuardFacts, PendingKey
from .output import hub_note
from .sqlite_util import connect as _sqlite_connect
from .sqlite_util import open_or_quarantine, rollback_and_close

# Timestamp format for cache record fields (`updated_at`, `fetched_at`).
UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"


def stamp_of(moment: datetime) -> str:
    """`moment` in `UPDATED_AT_STR_FORMAT` (record `added_at` stamps)."""

    return moment.strftime(UPDATED_AT_STR_FORMAT)


def now_stamp() -> str:
    """The current local time in `UPDATED_AT_STR_FORMAT` (record `added_at` stamps)."""

    return stamp_of(datetime.now())


def pending_cutoff(max_age_days: int) -> datetime:
    """The oldest add time a pending record may carry: now minus `imports.pending_max_age_days`."""

    return datetime.now() - timedelta(days=max_age_days)


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
    infohash TEXT NOT NULL,
    -- al_id is in the key: one torrent can be listed on several entries. 0 is the legacy sentinel.
    al_id    INTEGER NOT NULL DEFAULT 0,
    record   BLOB NOT NULL,
    PRIMARY KEY (arr, infohash, al_id)
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
SCHEMA_VERSION = 3


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


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_0_to_1,
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
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


def record_is_fresh(
    record: dict[str, Any] | None,
    *,
    payload_key: str,
    cutoff: datetime,
) -> bool:
    """True if a persisted record has a payload under `payload_key` and its `fetched_at` is within TTL."""

    if not isinstance(record, dict):
        return False
    if not record.get(payload_key):
        return False
    try:
        stamp = datetime.strptime(record.get("fetched_at", ""), UPDATED_AT_STR_FORMAT)
    except (TypeError, ValueError):
        return False
    return stamp >= cutoff


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
_PENDING_IMPORTS = _JsonBlock("pending_imports", ("arr", "infohash", "al_id"))
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
    def get_pending(self, arr: Arr) -> dict[PendingKey, dict[str, Any]]: ...
    @abstractmethod
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[PendingKey, dict[str, Any]]: ...
    @abstractmethod
    def put_pending(self, arr: Arr, key: PendingKey, record: dict[str, Any]) -> None: ...
    @abstractmethod
    def has_pending(self, arr: Arr, key: PendingKey) -> bool: ...
    @abstractmethod
    def drop_pending(self, arr: Arr, key: PendingKey) -> None: ...
    @abstractmethod
    def count_pending_for_infohash(self, infohash: str) -> int: ...
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

        sd_time_str = seadex_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT)
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
            details["updated_at"] = updated_at.strftime(UPDATED_AT_STR_FORMAT)

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

    def _pending_rows(self, sql: str, params: tuple[int | str, ...]) -> dict[PendingKey, dict[str, Any]]:
        """Deserialize a `SELECT infohash, al_id, json(record)` pending-imports query, keyed per record."""

        out: dict[PendingKey, dict[str, Any]] = {}
        for infohash, al_id, rec_json in self._conn.execute(sql, params):
            out[PendingKey(infohash, al_id)] = json.loads(rec_json)
        return out

    def _evict_stale_json(self, block: _JsonBlock, cutoff: datetime) -> int:
        """Delete records older than `cutoff` (or stamp-less, which is otherwise un-evictable). Count deleted."""

        cursor = self._conn.execute(
            f"DELETE FROM {block.table} WHERE fetched_at < ? OR fetched_at IS NULL",
            (cutoff.strftime(UPDATED_AT_STR_FORMAT),),
        )
        return cursor.rowcount

    # -- AniList meta (JSONB + TTL) ------------------------------------------

    @override
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield `(al_id, record)` for every stored record, TTL unfiltered (see `record_is_fresh`)."""

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
    def get_pending(self, arr: Arr) -> dict[PendingKey, dict[str, Any]]:
        """All pending-import records for an arr, keyed per record (snapshot)."""

        return self._pending_rows(
            "SELECT infohash, al_id, json(record) FROM pending_imports WHERE arr = ?",
            (_arr_key(arr),),
        )

    @override
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[PendingKey, dict[str, Any]]:
        """Pending-import records for one Sonarr `series_id` (a record without one yields NULL, excluded)."""

        return self._pending_rows(
            "SELECT infohash, al_id, json(record) FROM pending_imports WHERE arr = ? AND record ->> 'series_id' = ?",
            (_arr_key(arr), series_id),
        )

    @override
    def put_pending(self, arr: Arr, key: PendingKey, record: dict[str, Any]) -> None:
        """Upsert one record under its `PendingKey` (staged, persisted at a save point)."""

        self._json_put(_PENDING_IMPORTS, (_arr_key(arr), key.infohash, key.al_id), record)

    @override
    def has_pending(self, arr: Arr, key: PendingKey) -> bool:
        """Whether ONE pending record exists under its `PendingKey` (a keyed EXISTS, never a scan)."""

        row = self._conn.execute(
            "SELECT 1 FROM pending_imports WHERE arr = ? AND infohash = ? AND al_id = ? LIMIT 1",
            (_arr_key(arr), key.infohash, key.al_id),
        ).fetchone()
        return row is not None

    @override
    def drop_pending(self, arr: Arr, key: PendingKey) -> None:
        """Delete ONE pending record, never its siblings on the same torrent."""

        self._conn.execute(
            "DELETE FROM pending_imports WHERE arr = ? AND infohash = ? AND al_id = ?",
            (_arr_key(arr), key.infohash, key.al_id),
        )

    @override
    def count_pending_for_infohash(self, infohash: str) -> int:
        """How many pending records reference `infohash`, deliberately across BOTH arrs (not one arr's slice)."""

        row = self._conn.execute(
            "SELECT count(*) FROM pending_imports WHERE infohash = ?",
            (infohash,),
        ).fetchone()
        return int(row[0]) if row else 0

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
                "SELECT al_id, json(record) FROM guard_facts "
                "WHERE arr = ? AND al_id IN (SELECT al_id FROM pending_imports WHERE arr = ?)",
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
