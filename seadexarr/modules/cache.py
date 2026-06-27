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

This rests on **autocommit being OFF** (the connection keeps the default
``isolation_level=""``, i.e. deferred). Do NOT switch it to autocommit / set
``isolation_level=None`` - every staged write would commit immediately and the
preview gate would break.

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
from collections.abc import Iterator
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, TypedDict, cast

from seadex import EntryRecord

from .config import Arr
from .. import __version__

# Timestamp format for cache record fields (entry ``updated_at`` and the AniList
# meta / Sonarr parse ``fetched_at``). Lives here because the cache owns the
# record schema; consumers (the orchestrator and the Sonarr adapter) import it.
UPDATED_AT_STR_FORMAT = "%Y-%m-%d %H:%M:%S"

# Wait this long for a write lock before raising, instead of failing instantly on
# a momentarily-locked db. The single-instance run lock makes contention rare, but
# this keeps a brief overlap (e.g. a lingering reader) from crashing a run.
_BUSY_TIMEOUT_MS = 5000

# One statement per block; ``IF NOT EXISTS`` so it's a no-op on an existing db.
# anilist_meta / sonarr_parse store the record as a JSONB blob and expose the
# fetch timestamp as a VIRTUAL generated column indexed for the TTL sweep - the
# spike confirmed the index is used by ``DELETE ... WHERE fetched_at < ?``.
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
    infohash TEXT    NOT NULL,
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
    stamp_key: str = "fetched_at",
    cutoff: datetime | None = None,
) -> bool:
    """True if a persisted record has a payload and its stamp is within TTL.

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
        stamp_key (str): Record key holding the strftime'd fetch timestamp.
            Defaults to ``"fetched_at"``.
        cutoff (datetime | None): Precomputed freshness cutoff. Pass this once per
            loop so ``datetime.now()`` isn't recomputed per record; when None it's
            derived from ``ttl_days`` against the current time.
    """

    if not isinstance(record, dict):
        return False
    if not record.get(payload_key):
        return False
    try:
        stamp = datetime.strptime(record.get(stamp_key, ""), UPDATED_AT_STR_FORMAT)
    except (TypeError, ValueError):
        return False
    if cutoff is None:
        cutoff = datetime.now() - timedelta(days=ttl_days)
    return stamp >= cutoff


class CacheField(StrEnum):
    """The stored fields of a per-entry cache record.

    A ``StrEnum`` so each member IS its column / key string: ``CacheField.URL``
    reads (and serialized as) ``"url"``. Kept as the public read vocabulary even
    though the backing store is now SQLite columns.
    """

    NAME = "name"
    URL = "url"
    COVERAGE = "coverage"
    UPDATED_AT = "updated_at"
    TORRENT_HASHES = "torrent_hashes"


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
    # (planner.filter_by_torrent_hash), so a remembered list can carry ``None``;
    # the store drops those Nones (they never matched anything).
    torrent_hashes: list[str | None]


# The four scalar columns of ``entries`` that ``update_cache`` may merge. Kept in
# a set so the partial-update path only touches columns actually supplied (the old
# dict ``.update`` left absent fields untouched - this preserves that).
_ENTRY_SCALAR_COLUMNS = ("name", "url", "coverage", "updated_at")


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


def _coerce_int(value: object) -> int | None:
    """Parse a legacy stringified id to an ``int``, or None for a bad value."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_dict(value: object) -> dict[str, Any] | None:
    """Narrow a value read from untyped legacy JSON to a dict, or None.

    The boundary cast for migration: a hand-edited / legacy ``cache.json`` carries
    ``Any``-typed values, so each level is checked and pinned to ``dict[str, Any]``
    before the typed facade writers touch it.
    """

    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _quarantine_corrupt(path: str, *, logger: logging.Logger | None) -> None:
    """Move an unreadable cache db (and its WAL/SHM) aside so a run can recover.

    Fail-open: rather than crash-loop on a corrupt/torn file, rename it to
    ``<path>.corrupt-<timestamp>`` (kept for inspection) and let the caller start a
    fresh cache. A fresh cache only costs one re-check pass - the safe direction.
    """

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = f"{path}.corrupt-{stamp}"
    with contextlib.suppress(OSError):
        os.replace(path, dest)
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            os.replace(path + suffix, dest + suffix)
    if logger is not None:
        logger.warning(
            f"Cache database at {path} was unreadable/corrupt; moved it to {dest} "
            "and started a fresh cache (entries will be re-checked this run).",
        )


class CacheStore:
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
        conn = sqlite3.connect(path if exists else ":memory:")
        store = cls(conn, path, on_memory=not exists)
        try:
            store._configure(conn)
        except sqlite3.DatabaseError:
            # An existing file that isn't a valid database (e.g. a torn write or a
            # stray non-db file) raises here on the first PRAGMA/DDL. Quarantine it
            # and start fresh in memory (promoted on the first real save), so a
            # corrupt cache fails open instead of crash-looping every run.
            conn.close()
            _quarantine_corrupt(path, logger=logger)
            conn = sqlite3.connect(":memory:")
            store = cls(conn, path, on_memory=True)
            store._configure(conn)
            exists = False
        if not exists and migrate_from and os.path.exists(migrate_from):
            store._seed_from_legacy_json(migrate_from, logger=logger)
        store._reconcile(config_checksum)
        return store

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        """Apply connection pragmas and ensure the schema.

        Runs before any staged data write, so the WAL/schema statements execute in
        autocommit and never tangle with the run's deferred write transaction. WAL
        is a no-op on an in-memory db (it reports ``memory``); that's fine.
        """

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.executescript(_SCHEMA)

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
                al_id = _coerce_int(al_id_str)
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
            al_id = _coerce_int(al_id_str)
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
        """Promote the in-memory db to the on-disk file via the sqlite3 backup API.

        Commits the staged writes in memory (free - no file yet), copies the whole
        db onto ``path``, then re-points at the file-backed connection. Subsequent
        writes land on the file and commit at the next save point.
        """

        self._conn.commit()
        disk = sqlite3.connect(self._path)
        self._conn.backup(disk)
        self._conn.close()
        # Re-apply the connection-scoped pragmas on the new file-backed handle
        # (foreign_keys is per-connection; WAL persists in the file once set).
        disk.execute("PRAGMA journal_mode=WAL")
        disk.execute("PRAGMA foreign_keys=ON")
        disk.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        self._conn = disk
        self._on_memory = False

        # If this db was seeded from a legacy cache.json, retire it now that the
        # real db exists (kept as a ``.migrated`` backup, and no longer re-seeded).
        if self._migrated_from and os.path.exists(self._migrated_from):
            with contextlib.suppress(OSError):
                os.replace(self._migrated_from, self._migrated_from + ".migrated")
            self._migrated_from = None

    def close(self) -> None:
        """Roll back any uncommitted writes and close the connection.

        Anything not flushed by a save point is dropped - the safe direction (those
        titles are re-checked next run). Idempotent enough for a ``finally`` block.
        """

        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()
        self._conn.close()

    # -- descriptor (kv) -----------------------------------------------------

    def _set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # -- per-entry records (entries + torrent_hashes) ------------------------

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

    def get_cached_name(self, arr: Arr, al_id: int) -> str | None:
        """The cached AniList title for an entry, if any (no AniList lookup)."""

        return cast("str | None", self.get_cached_field(arr, al_id, CacheField.NAME))

    def get_cached_field(
        self,
        arr: Arr,
        al_id: int,
        field: CacheField,
    ) -> object | None:
        """Read a single stored field from an entry's record, if present.

        Args:
            arr (Arr): Arr instance the entry is cached under.
            al_id (int): AniList ID.
            field (CacheField): Field to read (NAME / URL / COVERAGE / UPDATED_AT /
                TORRENT_HASHES).

        Returns:
            The stored value, or None if absent. TORRENT_HASHES returns the list.
        """

        if field == CacheField.TORRENT_HASHES:
            return self.torrent_hashes(arr, al_id)
        row = self._conn.execute(
            # field.value is one of the fixed scalar column names (closed enum), so
            # the f-string interpolation is not an injection surface.
            f"SELECT {field.value} FROM entries WHERE arr = ? AND al_id = ?",  # noqa: S608
            (_arr_key(arr), al_id),
        ).fetchone()
        return row[0] if row else None

    def torrent_hashes(self, arr: Arr, al_id: int) -> list[str | None]:
        """Torrent hashes already remembered for an entry (empty if none).

        Used by the download planner to skip releases already grabbed. The stored
        hashes are concrete strings; the element type is widened to ``str | None``
        to match the planner's ``cached_hashes: list[str | None]`` parameter.

        Args:
            arr (Arr): Arr instance the entry is cached under.
            al_id (int): AniList ID.
        """

        rows = self._conn.execute(
            "SELECT infohash FROM torrent_hashes WHERE arr = ? AND al_id = ? ORDER BY infohash",
            (_arr_key(arr), al_id),
        ).fetchall()
        return cast("list[str | None]", [r[0] for r in rows])

    def update_cache(
        self,
        arr: Arr,
        al_id: int,
        cache_details: CacheRecord | None = None,
    ) -> bool:
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

        # Ensure the parent row exists (the FK target for torrent_hashes) without
        # clobbering existing scalar fields.
        self._conn.execute(
            "INSERT INTO entries (arr, al_id) VALUES (?, ?) "
            "ON CONFLICT (arr, al_id) DO NOTHING",
            (arr_key, al_id),
        )

        scalar = [c for c in _ENTRY_SCALAR_COLUMNS if c in details]
        if scalar:
            assignments = ", ".join(f"{c} = ?" for c in scalar)
            params = [details[c] for c in scalar]
            params.extend((arr_key, al_id))
            self._conn.execute(
                f"UPDATE entries SET {assignments} WHERE arr = ? AND al_id = ?",  # noqa: S608
                params,
            )

        if "torrent_hashes" in details:
            self._conn.execute(
                "DELETE FROM torrent_hashes WHERE arr = ? AND al_id = ?",
                (arr_key, al_id),
            )
            hashes: list[str | None] = details["torrent_hashes"] or []
            self._conn.executemany(
                "INSERT INTO torrent_hashes (arr, al_id, infohash) VALUES (?, ?, ?) "
                "ON CONFLICT (arr, al_id, infohash) DO NOTHING",
                [(arr_key, al_id, h) for h in hashes if h is not None],
            )

        return True

    # -- AniList meta (JSONB + TTL) ------------------------------------------

    def iter_anilist_meta(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield ``(al_id, record)`` for every stored AniList-meta record.

        The record is the ``{"fetched_at": ..., "data": ...}`` shape; the caller
        applies its own TTL freshness check (see :func:`record_is_fresh`).
        """

        for al_id, rec_json in self._conn.execute(
            "SELECT al_id, json(record) FROM anilist_meta",
        ):
            yield al_id, json.loads(rec_json)

    def get_anilist_meta(self, al_id: int) -> dict[str, Any] | None:
        """The stored ``{"fetched_at", "data"}`` record for an id, or None."""

        row = self._conn.execute(
            "SELECT json(record) FROM anilist_meta WHERE al_id = ?",
            (al_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_anilist_meta(self, al_id: int, record: dict[str, Any]) -> None:
        """Upsert an AniList-meta record (staged; persisted at a save point)."""

        self._conn.execute(
            "INSERT INTO anilist_meta (al_id, record) VALUES (?, jsonb(?)) "
            "ON CONFLICT (al_id) DO UPDATE SET record = excluded.record",
            (al_id, json.dumps(record)),
        )

    # -- Sonarr parse cache (JSONB + TTL) ------------------------------------

    def iter_sonarr_parse(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``(filename, record)`` for every stored Sonarr parse record."""

        for filename, rec_json in self._conn.execute(
            "SELECT filename, json(record) FROM sonarr_parse",
        ):
            yield filename, json.loads(rec_json)

    def get_sonarr_parse(self, filename: str) -> dict[str, Any] | None:
        """The stored ``{"fetched_at", "episodes"}`` record for a filename, or None."""

        row = self._conn.execute(
            "SELECT json(record) FROM sonarr_parse WHERE filename = ?",
            (filename,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_sonarr_parse(self, filename: str, record: dict[str, Any]) -> None:
        """Upsert a Sonarr parse record (staged; persisted at a save point)."""

        self._conn.execute(
            "INSERT INTO sonarr_parse (filename, record) VALUES (?, jsonb(?)) "
            "ON CONFLICT (filename) DO UPDATE SET record = excluded.record",
            (filename, json.dumps(record)),
        )

    # -- pending imports -----------------------------------------------------

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

    def put_pending(self, arr: Arr, infohash: str, record: dict[str, Any]) -> None:
        """Upsert a pending-import record (staged; persisted at a save point)."""

        self._conn.execute(
            "INSERT INTO pending_imports (arr, infohash, record) VALUES (?, ?, jsonb(?)) "
            "ON CONFLICT (arr, infohash) DO UPDATE SET record = excluded.record",
            (_arr_key(arr), infohash, json.dumps(record)),
        )

    def drop_pending(self, arr: Arr, infohash: str) -> None:
        """Delete a pending-import record (staged; persisted at a save point)."""

        self._conn.execute(
            "DELETE FROM pending_imports WHERE arr = ? AND infohash = ?",
            (_arr_key(arr), infohash),
        )

    # -- maintenance: eviction, stats, integrity -----------------------------

    def evict_anilist_meta(self, cutoff: datetime) -> int:
        """Delete AniList-meta records older than ``cutoff``; return the count.

        Hits the indexed generated ``fetched_at`` column, so it's an index
        range-delete, not a scan. Staged like any write - committed at the next
        save point, discarded in a preview - so it only frees rows that the
        gateway already refuses to read (older than the same TTL).
        """

        cursor = self._conn.execute(
            "DELETE FROM anilist_meta WHERE fetched_at < ?",
            (cutoff.strftime(UPDATED_AT_STR_FORMAT),),
        )
        return cursor.rowcount

    def evict_sonarr_parse(self, cutoff: datetime) -> int:
        """Delete Sonarr parse records older than ``cutoff``; return the count."""

        cursor = self._conn.execute(
            "DELETE FROM sonarr_parse WHERE fetched_at < ?",
            (cutoff.strftime(UPDATED_AT_STR_FORMAT),),
        )
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        """Row counts per table plus the on-disk size in bytes (0 while in memory).

        A cheap health snapshot for the ``cache stats`` command / a run-end log:
        how big is each block, and how big is the db (incl. its WAL).
        """

        out: dict[str, int] = {}
        for table in ("entries", "torrent_hashes", "anilist_meta", "sonarr_parse", "pending_imports"):
            row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
            out[table] = int(row[0]) if row else 0
        size = 0
        if not self._on_memory:
            for suffix in ("", "-wal"):
                with contextlib.suppress(OSError):
                    size += os.path.getsize(self._path + suffix)
        out["size_bytes"] = size
        return out

    def integrity_check(self) -> str:
        """Run ``PRAGMA quick_check`` and return its result (``"ok"`` when healthy)."""

        row = self._conn.execute("PRAGMA quick_check").fetchone()
        return str(row[0]) if row else "unknown"
