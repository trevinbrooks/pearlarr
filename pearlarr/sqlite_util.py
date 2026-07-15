"""Shared SQLite primitives for the project's on-disk stores.

Both `cache` (`cache.db`) and `mapping_store` (`mappings.db`) need the same low-level
behavior: a connection pinned to *explicit* legacy/deferred transaction control,
a busy timeout, a corruption predicate that distinguishes a genuinely torn file
from a transient lock, and a fail-open quarantine of an unreadable file. Those
primitives live here so the two stores share ONE copy - a change to the
corruption policy (which gates a destructive rename) or the transaction-control
pin (which gates the cache's preview-write semantics) can't land in one store and
silently diverge in the other.

The store *classes* stay separate on purpose: their write models genuinely differ
(the cache stages writes behind a preview gate and promotes an in-memory db; the
mapping store does atomic per-source digest-gated replaces with a rebuild-on-format
-change). Only the connection/corruption plumbing is shared.
"""

import contextlib
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple

from .output import hub_warn

# Wait this long for a write lock before raising, instead of failing instantly on
# a momentarily-locked db. The single-instance run lock makes contention rare, but
# this keeps a brief overlap (e.g. a lingering reader) from crashing a run.
BUSY_TIMEOUT_MS = 5000


def connect(path: str, *, ensure_wal: bool = True, foreign_keys: bool = False) -> sqlite3.Connection:
    """Open a connection with the project's fixed pragmas + transaction control.

    The single place these connections are created, so every caller shares
    identical settings instead of re-typing the pragma trio. `busy_timeout` is
    applied FIRST so the WAL-mode switch and the schema statements that follow
    already honor it (rather than racing with a zero timeout).

    Transaction control is pinned *explicitly* to legacy/deferred rather than
    leaning on the sqlite3 defaults: legacy mode means an implicit `BEGIN`
    precedes the first DML and nothing commits until the owner calls `commit` -
    exactly what the cache's staged-write preview gate and the mapping store's
    atomic per-source replace both rely on - so a future Python flipping a default
    can't silently turn staged writes into immediate commits and break either. Do
    NOT set `isolation_level=None` / real autocommit. (Both attributes are set
    post-connect, before any transaction is open, so this is pure configuration.)

    A non-db / corrupt file raises on the WAL switch; the handle is closed so it
    doesn't leak before the caller decides whether to quarantine.

    Args:
        path: Database path, or `":memory:"`.
        ensure_wal: Apply the WAL-mode pragma (the writable run path).
            Read-only diagnostics pass False so they neither mutate the db's
            journal mode nor need the file to be a valid db just to open.
        foreign_keys: Apply `PRAGMA foreign_keys=ON` (only the cache has FK
            constraints - the mapping store has none).
    """

    conn = sqlite3.connect(path)
    conn.autocommit = sqlite3.LEGACY_TRANSACTION_CONTROL
    conn.isolation_level = "DEFERRED"
    try:
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        if ensure_wal:
            conn.execute("PRAGMA journal_mode=WAL")
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn


def _close_quietly(conn: sqlite3.Connection | None) -> None:
    """Close `conn` if open, swallowing sqlite teardown errors. No-op on None."""

    if conn is not None:
        with contextlib.suppress(sqlite3.Error):
            conn.close()


class OpenResult(NamedTuple):
    """An opened store connection plus whether it fell back to `:memory:`."""

    conn: sqlite3.Connection
    fell_back: bool


def open_or_quarantine(
    path: str,
    *,
    connect_fn: Callable[[str], sqlite3.Connection],
    ensure: Callable[[sqlite3.Connection], object],
    what: str,
    recovery: str,
) -> OpenResult:
    """Open `path` and ensure its schema, quarantining a corrupt file.

    The shared recovery policy for both stores. A transient/operational
    `DatabaseError` (locked, disk I/O) is NOT corruption: fail closed and
    re-raise rather than destructively quarantining a healthy db on a fluke. A
    real not-a-database / torn file is moved aside via `quarantine_corrupt`
    and a fresh `:memory:` db is returned instead, so a corrupt store fails
    open rather than crash-looping every run.

    Args:
        path: Database path to open (or `":memory:"`).
        connect_fn: The store's own connection factory (keeps its
            pragma choices and the tests' patch point).
        ensure: Ensures the schema on a fresh connection.
        what: Human noun for the quarantine log line.
        recovery: Trailing recovery clause for the quarantine log line.

    Returns:
        The open connection plus a fell-back flag - True when the file was
        quarantined and the connection is the in-memory fallback.
    """

    conn: sqlite3.Connection | None = None
    try:
        conn = connect_fn(path)
        ensure(conn)
    except sqlite3.DatabaseError as exc:
        _close_quietly(conn)
        if not is_corruption(exc):
            raise
        quarantine_corrupt(path, what=what, recovery=recovery)
        conn = connect_fn(":memory:")
        try:
            ensure(conn)
        except BaseException:
            _close_quietly(conn)
            raise
        return OpenResult(conn, True)
    except BaseException:
        # A non-sqlite failure from ensure (e.g. a schema-version refusal) still
        # owns an open handle - close it on the way out so it can't leak.
        _close_quietly(conn)
        raise
    return OpenResult(conn, False)


def rollback_and_close(conn: sqlite3.Connection) -> None:
    """Roll back anything uncommitted and close `conn`, swallowing sqlite errors.

    The shared `close()` tail for both stores, so their error behavior on a
    torn-down connection can't diverge. Idempotent enough for a `finally` block.
    """

    with contextlib.suppress(sqlite3.Error):
        conn.rollback()
    _close_quietly(conn)


def is_corruption(exc: sqlite3.DatabaseError) -> bool:
    """True if a DatabaseError signals an actually corrupt / not-a-database file.

    The quarantine path destroys (renames) the db, so it must fire ONLY on real
    corruption - never on a transient `OperationalError` (`SQLITE_BUSY` /
    `database is locked`, a disk I/O error), which would otherwise wipe a healthy
    file on a fluke. Keys on the SQLite extended/primary result code, with a message
    fallback for builds that don't surface one.
    """

    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        # Compare the primary result code (low 8 bits) so extended codes match too.
        primary = code & 0xFF
        if primary in (sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_FORMAT):
            return True
        # A known operational/transient code is explicitly NOT corruption.
        if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR):
            return False
    msg = str(exc).lower()
    return any(s in msg for s in ("not a database", "malformed", "file is encrypted"))


def quarantine_corrupt(
    path: str,
    *,
    what: str,
    recovery: str,
) -> None:
    """Move an unreadable db (and its WAL/SHM sidecars) aside so a run can recover.

    Fail-open: rather than crash-loop on a corrupt/torn file, rename it to
    `<path>.corrupt-<timestamp>` (kept for inspection) and let the caller start
    fresh. A fresh db only costs one re-derive pass - the safe direction.

    The warn line is built from `what` (a human noun, e.g. `"Cache database"`)
    and `recovery` (its trailing clause, e.g. `"started a fresh cache (...)."`,
    ending with the period).
    """

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = f"{path}.corrupt-{stamp}"
    with contextlib.suppress(OSError):
        os.replace(path, dest)
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(OSError):
            os.replace(path + suffix, dest + suffix)
    hub_warn(f"{what} at {path} was unreadable/corrupt - moved it to {dest} and {recovery}")
