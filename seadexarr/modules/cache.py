"""Persistent run cache: SQLite-backed store, schema ownership, freshness, writes.

``CacheStore`` owns the on-disk cache - one SQLite database (``cache.db``) - and
every read/write against its five logical blocks: the descriptor ``kv`` (package
version + config checksum), the per-arr ``entries`` plus their ``torrent_hashes``
child rows, the ``anilist_meta`` and ``sonarr_parse`` JSONB caches, and
``pending_imports``. It also owns the freshness check that decides whether a
title needs re-processing. Folding all five blocks here (they used to be poked
into a shared dict by three different modules) gives the cache file a single
owner.

Write model (preserves the pre-SQLite semantics exactly):

* Writes are *staged* in one deferred transaction and only persisted when a run
  reaches a save point and calls ``save(preview=False)`` -> ``COMMIT``.
* A preview run calls ``save(preview=True)`` -> no commit, so it never persists.
  Reads within the run still see the staged-but-uncommitted writes (same
  connection), exactly as the old in-memory dict did. ``close()`` rolls back
  anything still uncommitted.
* A hard kill mid-run loses at most the titles finished since the last save
  point; they're simply re-checked next run, never silently skipped - the safe
  direction.

This rests on the connection using **deferred** transaction control. The
connection factory (:func:`_connect`) pins it *explicitly*
(``autocommit=LEGACY_TRANSACTION_CONTROL`` + ``isolation_level="DEFERRED"``) rather
than leaning on the sqlite3 defaults, so a future Python flipping a default can't
silently break the preview gate. Do NOT set ``isolation_level=None`` / real
autocommit - every staged write would commit immediately and the gate would break.

A missing cache opens an **in-memory** database and is *promoted* to the real
file on the first non-preview ``save`` (via the sqlite3 backup API), so a preview
run on a system with no cache yet still writes nothing to disk.

Each arr instance constructs its own ``CacheStore``; a scheduled cycle runs Radarr
(which commits ``cache.db``) then Sonarr (which re-opens it), handing off through
the *file*. Do not share one ``CacheStore`` / connection across arrs or threads.
"""

import contextlib
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, NamedTuple, TypedDict, cast, override

from seadex import EntryRecord

from .config import Arr
from .seadex_types import coerce_int
from .sqlite_util import connect as _sqlite_connect
from .sqlite_util import open_or_quarantine, rollback_and_close
from .. import __version__

# Timestamp format for cache record fields (entry ``updated_at`` and the AniList
# meta / Sonarr parse ``fetched_at``). Lives here because the cache owns the
# record schema; consumers (the orchestrator and the Sonarr adapter) import it.
UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"

# One statement per block; ``IF NOT EXISTS`` so it's a no-op on an existing db.
# NOTE: there is NO schema-migration mechanism here - ``CREATE TABLE IF NOT EXISTS``
# creates a missing table but silently does NOT alter an existing one. So changing a
# column on a shipped table (e.g. relaxing a NOT NULL) won't reach an upgraded
# cache.db and will diverge or crash; such a change needs a real migration, or - as
# ``torrent_hashes`` does for its None marker - a storage trick that leaves the
# column type unchanged. anilist_meta / sonarr_parse store the record as a JSONB blob
# and expose the fetch timestamp as a VIRTUAL generated column indexed for the TTL
# sweep - the spike confirmed the index is used by ``DELETE ... WHERE fetched_at < ?``.
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
    PRIMARY KEY (arr, al_id)
);

CREATE TABLE IF NOT EXISTS torrent_hashes (
    arr      TEXT    NOT NULL,
    al_id    INTEGER NOT NULL,
    -- A SeaDex url's infohash can be ``None`` (a hashless release), and a remembered
    -- ``None`` IS a membership key the planner dedups on, so it must round-trip. The
    -- column stays NOT NULL (unchanged since the first release - ``CREATE TABLE IF
    -- NOT EXISTS`` would NOT migrate an existing db, so a nullable column would crash
    -- on an upgraded cache). ``None`` is persisted as the ``_NO_HASH`` sentinel and
    -- mapped back on read; a real infohash is never empty, so there's no collision.
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
    record   BLOB NOT NULL,
    PRIMARY KEY (arr, infohash)
);
"""


def record_is_fresh(
    record: dict[str, Any] | None,
    *,
    payload_key: str,
    ttl_days: int,
    cutoff: datetime | None = None,
) -> bool:
    """True if a persisted record has a payload and its ``fetched_at`` is within TTL.

    Shared freshness check for the raw, stringly-keyed cache records (the AniList
    ``anilist_meta`` records and the Sonarr parse-cache records), so the load
    (which ids to seed) and save (which to keep vs. refresh) sides never disagree
    about what "still good" means.

    Args:
        record (dict[str, Any] | None): The raw cache record, or None / a non-dict
            (treated as not fresh).
        payload_key (str): Key whose presence (and truthiness) marks a usable
            payload (e.g. ``"data"`` for AniList, ``"episodes"`` for Sonarr).
        ttl_days (int): TTL window in days, used to derive ``cutoff`` when one
            isn't supplied.
        cutoff (datetime | None): Precomputed freshness cutoff. Pass this once per
            loop so ``datetime.now()`` isn't recomputed per record; when None it's
            derived from ``ttl_days`` against the current time.
    """

    if not isinstance(record, dict):
        return False
    if not record.get(payload_key):
        return False
    try:
        stamp = datetime.strptime(record.get("fetched_at", ""), UPDATED_AT_STR_FORMAT)
    except (TypeError, ValueError):
        return False
    if cutoff is None:
        cutoff = datetime.now() - timedelta(days=ttl_days)
    return stamp >= cutoff


class CacheRecord(TypedDict, total=False):
    """The fixed shape of a per-entry cache update / a ``cache_details`` payload.

    ``total=False`` because producers assemble it incrementally (a movie carries
    no coverage at first; Sonarr fills coverage/url later). ``updated_at`` may hold
    a ``datetime`` at the producer and is strftime'd to ``str`` in place by
    :meth:`CacheStore.update_cache`, hence the union.
    """

    name: str
    url: str
    coverage: str
    updated_at: "str | datetime"
    # A SeaDex url's infohash is ``str | None`` and is appended unconditionally
    # (planner.filter_by_torrent_hash), so a remembered list can carry ``None``; the
    # store preserves those Nones because the planner dedups on None membership.
    torrent_hashes: list[str | None]


@dataclass(frozen=True, slots=True)
class CachedEntry:
    """The four scalar columns of one ``entries`` row, read in a single query.

    Lets a caller that needs several fields of the same ``(arr, al_id)`` row fetch
    them in one round-trip (see :meth:`CacheStore.get_entry`) instead of issuing a
    point ``SELECT`` per field. Each column is nullable on disk, so every field is
    ``str | None``.
    """

    updated_at: str | None
    name: str | None
    url: str | None
    coverage: str | None


# The four scalar columns of ``entries`` that ``update_cache`` may merge. A closed
# tuple so the partial-update path only touches columns actually supplied (the old
# dict ``.update`` left absent fields untouched - this preserves that).
_ENTRY_SCALAR_COLUMNS = ("name", "url", "coverage", "updated_at")

# Sentinel stored in ``torrent_hashes.infohash`` (a NOT NULL column) for a remembered
# ``None`` marker - a hashless release the planner still dedups on. A real infohash is
# never empty, so the empty string round-trips uniquely back to ``None`` on read.
_NO_HASH = ""


class _JsonBlock(NamedTuple):
    """One JSONB (``record`` BLOB) block: its table and key column(s).

    A closed allowlist - these names are interpolated into the ``_json_*``
    helpers' SQL, so they must only ever come from the constants below.
    """

    table: str
    key_cols: tuple[str, ...]


_ANILIST_META = _JsonBlock("anilist_meta", ("al_id",))
_SONARR_PARSE = _JsonBlock("sonarr_parse", ("filename",))
_PENDING_IMPORTS = _JsonBlock("pending_imports", ("arr", "infohash"))


class CacheStats(NamedTuple):
    """Row counts per cache table plus the on-disk size in bytes."""

    entries: int
    torrent_hashes: int
    anilist_meta: int
    sonarr_parse: int
    pending_imports: int
    size_bytes: int


def _arr_key(arr: Arr) -> str:
    """The text stored for an ``Arr`` (``"sonarr"`` / ``"radarr"``).

    ``Arr`` is a ``StrEnum`` so it already binds as its value, but coercing here
    keeps every SQL parameter an unambiguous ``str`` regardless of whether a call
    site passed the enum member or a bare string.
    """

    return str(arr)


def _coerce_arr(value: object) -> Arr | None:
    """Parse a legacy arr key (``"sonarr"`` / ``"radarr"``) to an ``Arr``, or None."""

    try:
        return Arr(value)
    except ValueError:
        return None


def _as_dict(value: object) -> dict[str, Any] | None:
    """Narrow a value read from untyped legacy JSON to a dict, or None.

    The boundary cast for migration: a hand-edited / legacy ``cache.json`` carries
    ``Any``-typed values, so each level is checked and pinned to ``dict[str, Any]``
    before the typed facade writers touch it.
    """

    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _connect(path: str, *, ensure_wal: bool = True) -> sqlite3.Connection:
    """Open a cache-db connection (see :func:`sqlite_util.connect`).

    Kept as the cache's own connection factory - the single place ``load`` /
    ``_promote`` / ``open_readonly`` go through, and the patch point the cache tests
    target - delegating the pragma/transaction-control plumbing to the shared
    helper. The cache has FK constraints (``torrent_hashes`` -> ``entries`` ON
    DELETE CASCADE), so foreign keys are enabled whenever WAL is (the writable run
    path); read-only diagnostics pass ``ensure_wal=False`` and get neither.

    Args:
        path (str): Database path (or ``":memory:"``).
        ensure_wal (bool): Apply the WAL (and, with it, foreign-keys) pragmas.
            Defaults to True.
    """

    return _sqlite_connect(path, ensure_wal=ensure_wal, foreign_keys=ensure_wal)


class AbstractCacheStore(ABC):
    """Nominal ABC base defining the instance facade run collaborators depend on.

    Both the real ``CacheStore`` and the test ``FakeCacheStore`` subclass this, so
    the checker enforces the whole facade on each via inheritance (and ``@override``)
    - neither can silently drift, and a fake missing a method won't instantiate. The
    two ``load`` / ``open_readonly`` constructors are not part of the instance surface
    and stay off the base.
    """

    @abstractmethod
    def save(self, *, preview: bool) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
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
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]: ...
    @abstractmethod
    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None: ...
    @abstractmethod
    def drop_pending(self, arr: Arr, infohash: str) -> None: ...
    @abstractmethod
    def stats(self) -> CacheStats: ...
    @abstractmethod
    def integrity_check(self) -> str: ...


class CacheStore(AbstractCacheStore):
    """Owns the cache database: schema, freshness checks, and persistence."""

    def __init__(self, conn: sqlite3.Connection, path: str, *, on_memory: bool) -> None:
        self._conn = conn
        self._path = path
        # True while backed by an in-memory db (the file didn't exist at load); the
        # first non-preview save promotes it to ``path``.
        self._on_memory = on_memory
        # Set to the legacy ``cache.json`` path when this run seeded itself from it,
        # so the promotion that creates ``cache.db`` also retires the old file.
        self._migrated_from: str | None = None

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        *,
        config_checksum: str,
        migrate_from: str | None = None,
        logger: logging.Logger | None = None,
    ) -> "CacheStore":
        """Open the cache db (or an in-memory stand-in) and reconcile the descriptor.

        An existing file is opened in place; a missing file opens ``:memory:`` so a
        preview run that never reaches a real save leaves no file behind. Either
        way the schema is ensured and the version/checksum descriptor is staged
        (committed at the first non-preview save).

        When there is no db yet but a legacy ``cache.json`` is present
        (``migrate_from``), its contents are seeded into the in-memory db; the
        normal promote-on-first-real-save then writes ``cache.db`` and retires the
        old file. A preview run seeds but never promotes, so it migrates nothing -
        the legacy file is left untouched for the next real run.

        Args:
            path (str): Path to the cache database file.
            config_checksum (str): Current config-file checksum, stamped into the
                descriptor so a changed config is recorded (informational; not used
                to invalidate records - entries are freshness-keyed already).
            migrate_from (str | None): Path to a legacy ``cache.json`` to seed from
                when no db exists yet. Defaults to None (no migration).
            logger (logging.Logger | None): For the one-line migration notice.
        """

        exists = os.path.exists(path)
        # Fail-closed on transient errors, fail-open (quarantine + :memory:) on real
        # corruption; the in-memory fallback is promoted on the first real save.
        # Schema is ensured before any staged write (executescript implicitly
        # COMMITs first, a no-op here since nothing is staged yet).
        conn, fell_back = open_or_quarantine(
            path if exists else ":memory:",
            connect_fn=_connect,
            ensure=lambda c: c.executescript(_SCHEMA),
            logger=logger,
            what="Cache database",
            recovery="started a fresh cache (entries will be re-checked this run).",
        )
        if fell_back:
            exists = False
        store = cls(conn, path, on_memory=not exists)
        if not exists and migrate_from and os.path.exists(migrate_from):
            store._seed_from_legacy_json(migrate_from, logger=logger)
        store._reconcile(config_checksum)
        return store

    @classmethod
    def open_readonly(cls, path: str) -> "CacheStore":
        """Open an existing cache db for a read-only diagnostic (``stats``/``check``).

        Applies only ``busy_timeout`` (NOT the WAL / foreign-keys pragmas, so a
        diagnostic never mutates the file's journal mode) and does NOT ensure the
        schema, reconcile the descriptor, or quarantine on corruption - the command
        should reflect the file as-is. A corrupt / not-a-database file raises
        :class:`sqlite3.DatabaseError` from the first read (in ``stats`` /
        ``integrity_check``); the caller is expected to catch and report it, since
        surfacing bad integrity is the whole point of those commands.
        """

        return cls(_connect(path, ensure_wal=False), path, on_memory=False)

    def _reconcile(self, config_checksum: str) -> None:
        """Stamp the current package version and config checksum into ``kv``."""

        self._set_kv("seadexarr_version", __version__)
        self._set_kv("config_checksum", config_checksum)

    def _seed_from_legacy_json(
        self,
        json_path: str,
        *,
        logger: logging.Logger | None,
    ) -> None:
        """One-time import of a legacy ``cache.json`` into the (in-memory) db.

        Reads the five legacy blocks and stages them through the normal facade
        writers, so a corrupt or partial old file degrades to "import what's
        readable" rather than crashing. Records are validated defensively because
        the old file was hand-editable JSON. ``description`` is skipped (the
        descriptor is re-stamped by :meth:`_reconcile`); legacy bare-list Sonarr
        parse records (the pre-TTL form) are skipped as stale.
        """

        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            if logger is not None:
                logger.warning(
                    f"Legacy cache at {json_path} is unreadable; starting fresh",
                )
            return
        data = _as_dict(raw)
        if data is None:
            return

        entries = _as_dict(data.get("anilist_entries"))
        for arr_str, recs_raw in (entries or {}).items():
            arr = _coerce_arr(arr_str)
            recs = _as_dict(recs_raw)
            if arr is None or recs is None:
                continue
            for al_id_str, rec_raw in recs.items():
                al_id = coerce_int(al_id_str)
                rec = _as_dict(rec_raw)
                if al_id is None or rec is None:
                    continue
                details: dict[str, Any] = {c: rec[c] for c in _ENTRY_SCALAR_COLUMNS if c in rec}
                hashes = rec.get("torrent_hashes")
                if isinstance(hashes, list):
                    details["torrent_hashes"] = cast("list[str | None]", hashes)
                self.update_cache(arr, al_id, cast("CacheRecord", details))

        meta = _as_dict(data.get("anilist_meta"))
        for al_id_str, rec_raw in (meta or {}).items():
            al_id = coerce_int(al_id_str)
            rec = _as_dict(rec_raw)
            if al_id is not None and rec is not None:
                self.put_anilist_meta(al_id, rec)

        parse = _as_dict(data.get("sonarr_parse_cache"))
        for filename, rec_raw in (parse or {}).items():
            # dict form only; the pre-TTL bare-list form is treated as stale.
            rec = _as_dict(rec_raw)
            if rec is not None:
                self.put_sonarr_parse(filename, rec)

        pending = _as_dict(data.get("pending_imports"))
        for arr_str, recs_raw in (pending or {}).items():
            arr = _coerce_arr(arr_str)
            recs = _as_dict(recs_raw)
            if arr is None or recs is None:
                continue
            for infohash, rec_raw in recs.items():
                rec = _as_dict(rec_raw)
                if rec is not None:
                    self.put_pending(arr, infohash, rec)

        self._migrated_from = json_path
        if logger is not None:
            logger.info(f"Migrating legacy cache {json_path} -> {self._path}")

    @override
    def save(self, *, preview: bool) -> None:
        """Persist staged writes - unless this is a preview run.

        The single commit chokepoint. A preview never commits (so it never
        persists, mirroring the old in-memory-only mutation); the first non-preview
        save on a still-in-memory db promotes it to the real file.

        Args:
            preview (bool): When True, leave writes staged/uncommitted (discarded on
                close) so the run persists nothing.
        """

        if preview:
            return
        if self._on_memory:
            self._promote()
        else:
            self._conn.commit()

    def _promote(self) -> None:
        """Promote the in-memory db to the on-disk file, durably.

        Commits the staged writes in memory, copies the whole db to a *temp* file via
        the sqlite3 backup API, then atomically renames it onto ``path`` and re-opens
        the file-backed connection through :func:`_connect`. Backing up to a temp +
        atomic rename means ``cache.db`` is only ever created from a COMPLETE copy: a
        crash or I/O error mid-copy leaves no 0-byte / partial ``cache.db`` for the
        next run to mistake for a real (empty) cache and skip the migration over.
        """

        self._conn.commit()
        tmp_path = self._path + ".promote.tmp"
        # Clear any temp left by a previously-aborted promote before reusing the name.
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
        # Swap the in-memory source for a fresh file-backed handle (pragmas applied).
        self._conn.close()
        self._conn = _connect(self._path)
        self._on_memory = False

        # If this db was seeded from a legacy cache.json, retire it now that the
        # real db exists (kept as a ``.migrated`` backup, and no longer re-seeded).
        if self._migrated_from and os.path.exists(self._migrated_from):
            with contextlib.suppress(OSError):
                os.replace(self._migrated_from, self._migrated_from + ".migrated")
            self._migrated_from = None

    @override
    def close(self) -> None:
        """Roll back any uncommitted writes and close the connection.

        Anything not flushed by a save point is dropped - the safe direction (those
        titles are re-checked next run). Idempotent enough for a ``finally`` block.
        """

        rollback_and_close(self._conn)

    # -- descriptor (kv) -----------------------------------------------------

    def _set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- per-entry records (entries + torrent_hashes) ------------------------

    @override
    def check_al_id_in_cache(
        self,
        arr: Arr,
        al_id: int,
        seadex_entry: EntryRecord,
    ) -> bool:
        """True if the cached entry's timestamp matches the SeaDex entry's.

        Args:
            arr (Arr): Arr instance.
            al_id (int): AniList ID.
            seadex_entry: SeaDex entry whose ``updated_at`` is compared.
        """

        sd_time_str = seadex_entry.updated_at.strftime(UPDATED_AT_STR_FORMAT)
        row = self._conn.execute(
            "SELECT updated_at FROM entries WHERE arr = ? AND al_id = ?",
            (_arr_key(arr), al_id),
        ).fetchone()
        return bool(row) and row[0] == sd_time_str

    @override
    def get_entry(self, arr: Arr, al_id: int) -> CachedEntry | None:
        """The four scalar columns of an entry's row in one query, or None.

        Folds what used to be a point ``SELECT`` per field into a single read for
        callers that need several columns of the same ``(arr, al_id)`` row (the
        cached-skip short-circuit and the cached-entry log line). Does NOT include
        the ``torrent_hashes`` child set - use :meth:`torrent_hashes` for that.
        """

        row = self._conn.execute(
            "SELECT updated_at, name, url, coverage FROM entries WHERE arr = ? AND al_id = ?",
            (_arr_key(arr), al_id),
        ).fetchone()
        return None if row is None else CachedEntry(row[0], row[1], row[2], row[3])

    @override
    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]:
        """Torrent hashes already remembered for an entry (empty if none).

        Used by the download planner to skip releases already grabbed. A remembered
        ``None`` marker (a hashless release) is preserved and round-trips, matching
        the planner's ``cached_hashes: list[str | None]`` membership check.

        Args:
            arr (Arr): Arr instance the entry is cached under.
            al_id (int): AniList ID.
        """

        rows = self._conn.execute(
            "SELECT infohash FROM torrent_hashes WHERE arr = ? AND al_id = ? ORDER BY infohash",
            (_arr_key(arr), al_id),
        ).fetchall()
        # Map the _NO_HASH sentinel back to the None marker it stands in for.
        return cast("list[str | None]", [None if r[0] == _NO_HASH else r[0] for r in rows])

    @override
    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> None:
        """Merge fields into an entry's record (staged; persisted at a save point).

        Mirrors the old dict ``.update``: only the supplied scalar fields are
        written (absent ones are left untouched), and a supplied ``torrent_hashes``
        replaces the entry's whole hash set. ``updated_at`` given as a ``datetime``
        is strftime'd in place.

        Args:
            arr (Arr): Arr instance.
            al_id (int): AniList ID.
            cache_details (CacheRecord): Fields to merge. Defaults to None (just
                ensures the entry row exists).
        """

        details: dict[str, Any] = dict(cache_details or {})

        updated_at = details.get("updated_at")
        if isinstance(updated_at, datetime):
            details["updated_at"] = updated_at.strftime(UPDATED_AT_STR_FORMAT)

        arr_key = _arr_key(arr)

        scalar = [c for c in _ENTRY_SCALAR_COLUMNS if c in details]
        if scalar:
            # One upsert instead of INSERT-then-UPDATE: insert the supplied columns,
            # or on an existing row update ONLY those columns (partial merge - absent
            # columns are left untouched). The column names come from the closed
            # _ENTRY_SCALAR_COLUMNS tuple, so the interpolation isn't an injection
            # surface.
            cols = ", ".join(scalar)
            placeholders = ", ".join("?" for _ in scalar)
            assignments = ", ".join(f"{c} = excluded.{c}" for c in scalar)
            self._conn.execute(
                f"INSERT INTO entries (arr, al_id, {cols}) VALUES (?, ?, {placeholders}) "  # noqa: S608
                f"ON CONFLICT (arr, al_id) DO UPDATE SET {assignments}",
                (arr_key, al_id, *(details[c] for c in scalar)),
            )
        else:
            # No scalar fields: just ensure the row exists (the FK target for
            # torrent_hashes) without clobbering existing fields.
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
            # Keep None markers (a hashless release the planner still dedups on at
            # planner.filter_by_torrent_hash). None is stored as the _NO_HASH sentinel
            # (the column is NOT NULL); ON CONFLICT then collapses duplicates -
            # including repeated sentinels - so at most one None marker is kept.
            self._conn.executemany(
                "INSERT INTO torrent_hashes (arr, al_id, infohash) VALUES (?, ?, ?) "
                "ON CONFLICT (arr, al_id, infohash) DO NOTHING",
                [(arr_key, al_id, _NO_HASH if h is None else h) for h in hashes],
            )

    # -- JSONB record blocks (shared plumbing) --------------------------------
    # Table/column names come only from the closed _JsonBlock constants, so the
    # f-string SQL isn't an injection surface (same pattern as stats()).

    def _json_get(self, block: _JsonBlock, key: tuple[int | str, ...]) -> dict[str, Any] | None:
        """The stored record under ``key`` in a JSONB block, or None."""

        where = " AND ".join(f"{c} = ?" for c in block.key_cols)
        row = self._conn.execute(
            f"SELECT json(record) FROM {block.table} WHERE {where}",  # noqa: S608
            key,
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _json_put(self, block: _JsonBlock, key: tuple[int | str, ...], record: dict[str, Any]) -> None:
        """Upsert a record into a JSONB block (staged; persisted at a save point)."""

        cols = ", ".join(block.key_cols)
        placeholders = ", ".join("?" for _ in block.key_cols)
        self._conn.execute(
            f"INSERT INTO {block.table} ({cols}, record) VALUES ({placeholders}, jsonb(?)) "  # noqa: S608
            f"ON CONFLICT ({cols}) DO UPDATE SET record = excluded.record",
            (*key, json.dumps(record)),
        )

    def _evict_stale_json(self, block: _JsonBlock, cutoff: datetime) -> int:
        """Delete records older than ``cutoff`` (or stamp-less); count deleted.

        Hits the block's indexed generated ``fetched_at`` column, so it's an index
        range-delete, not a scan. Staged like any write - committed at the next
        save point, discarded in a preview. A NULL ``fetched_at`` (a legacy /
        hand-edited record with no stamp) is unreadable AND would otherwise be
        un-evictable forever, so it's swept too.
        """

        cursor = self._conn.execute(
            f"DELETE FROM {block.table} WHERE fetched_at < ? OR fetched_at IS NULL",  # noqa: S608
            (cutoff.strftime(UPDATED_AT_STR_FORMAT),),
        )
        return cursor.rowcount

    # -- AniList meta (JSONB + TTL) ------------------------------------------

    @override
    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield ``(al_id, record)`` for every stored AniList-meta record.

        The record is the ``{"fetched_at": ..., "data": ...}`` shape; the caller
        applies its own TTL freshness check (see :func:`record_is_fresh`).
        """

        for al_id, rec_json in self._conn.execute(
            "SELECT al_id, json(record) FROM anilist_meta",
        ):
            yield al_id, json.loads(rec_json)

    @override
    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        """The stored ``{"fetched_at", "data"}`` record for an id, or None."""

        return self._json_get(_ANILIST_META, (al_id,))

    @override
    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None:
        """Upsert an AniList-meta record (staged; persisted at a save point)."""

        self._json_put(_ANILIST_META, (al_id,), record)

    # -- Sonarr parse cache (JSONB + TTL) ------------------------------------

    @override
    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        """The stored ``{"fetched_at", "episodes"}`` record for a filename, or None."""

        return self._json_get(_SONARR_PARSE, (filename,))

    @override
    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None:
        """Upsert a Sonarr parse record (staged; persisted at a save point)."""

        self._json_put(_SONARR_PARSE, (filename,), record)

    # -- pending imports -----------------------------------------------------

    @override
    def get_pending(self, arr: Arr) -> dict[str, dict[str, Any]]:
        """All pending-import records for an arr, keyed by infohash (snapshot).

        Returns a plain dict copy; mutating it does not touch the store (use
        :meth:`put_pending` / :meth:`drop_pending`).
        """

        out: dict[str, dict[str, Any]] = {}
        for infohash, rec_json in self._conn.execute(
            "SELECT infohash, json(record) FROM pending_imports WHERE arr = ?",
            (_arr_key(arr),),
        ):
            out[infohash] = json.loads(rec_json)
        return out

    @override
    def get_pending_for_series(self, arr: Arr, series_id: int) -> dict[str, dict[str, Any]]:
        """Pending-import records for one Sonarr ``series_id``, keyed by infohash.

        Same fresh-per-call snapshot as :meth:`get_pending` (a record dropped earlier
        this run is already absent), but the ``series_id`` filter is pushed into SQL
        via ``record ->> 'series_id'`` so only this series' records are deserialized -
        the per-series reconcile no longer re-parses every pending record once per
        series. ``series_id`` is stored as a JSON int (``PendingImport.series_id``),
        so the bound int compares directly; a record with no ``series_id`` yields NULL
        and is excluded, matching the old ``record.get("series_id") != series_id`` skip.
        """

        out: dict[str, dict[str, Any]] = {}
        for infohash, rec_json in self._conn.execute(
            "SELECT infohash, json(record) FROM pending_imports WHERE arr = ? AND record ->> 'series_id' = ?",
            (_arr_key(arr), series_id),
        ):
            out[infohash] = json.loads(rec_json)
        return out

    @override
    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None:
        """Upsert a pending-import record (staged; persisted at a save point)."""

        self._json_put(_PENDING_IMPORTS, (_arr_key(arr), infohash), record)

    @override
    def drop_pending(self, arr: Arr, infohash: str) -> None:
        """Delete a pending-import record (staged; persisted at a save point)."""

        self._conn.execute(
            "DELETE FROM pending_imports WHERE arr = ? AND infohash = ?",
            (_arr_key(arr), infohash),
        )

    # -- maintenance: eviction, stats, integrity -----------------------------

    @override
    def evict_anilist_meta(self, cutoff: datetime) -> int:
        """Delete AniList-meta records older than ``cutoff`` (or stamp-less); count.

        See :meth:`_evict_stale_json`: an indexed range-delete, staged like any
        write, that only frees rows the gateway already refuses to read (older
        than the same TTL).
        """

        return self._evict_stale_json(_ANILIST_META, cutoff)

    @override
    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        """Delete Sonarr parse records older than ``cutoff`` (or stamp-less); count.

        Mirrors :meth:`evict_anilist_meta` (see :meth:`_evict_stale_json`).
        """

        return self._evict_stale_json(_SONARR_PARSE, cutoff)

    def _count(self, table: str) -> int:
        """Row count of one table; the name comes from stats()'s closed literals."""

        row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
        return int(row[0]) if row else 0

    @override
    def stats(self) -> CacheStats:
        """Row counts per table plus the on-disk size in bytes (0 while in memory).

        A cheap health snapshot for the ``cache stats`` command / a run-end log:
        how big is each block, and how big is the db (incl. its WAL).
        """

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
            size_bytes=size,
        )

    @override
    def integrity_check(self) -> str:
        """Run ``PRAGMA quick_check`` and return its result (``"ok"`` when healthy)."""

        row = self._conn.execute("PRAGMA quick_check").fetchone()
        return str(row[0]) if row else "unknown"
