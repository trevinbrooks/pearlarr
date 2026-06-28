"""SQLite cache of the parsed + indexed id-mapping sources (``mappings.db``).

The three immutable mapping sources - the Kometa Anime-IDs JSON, the AniDB
anime-list XML and the anibridge graph - are large, read-only files whose
parsed+indexed forms never change until the file is re-downloaded *with new
content*. Re-parsing and re-indexing them on every fresh process start is wasted
CPU (~1.2s) and resident memory (~51MB held for the process lifetime). This store
persists the indexed forms in a dedicated ``mappings.db`` (next to ``cache.db``),
keyed by each source's content digest, so a process whose source files are
unchanged answers lookups straight from SQL without re-parsing - and never has to
hold the full parsed structures in memory.

How this differs from :class:`~seadexarr.modules.cache.CacheStore` (deliberately a
separate file and a separate db):

* **Never preview-gated.** ``cache.db`` stages writes and only commits on a
  non-preview save; this store is a pure derived cache of an *immutable download*,
  so a freshly parsed index is always committed, even on ``--dry-run``. Each
  ``replace_*`` is its own atomic transaction.
* **Atomic populate.** A source's rows and its ``meta`` digest stamp are written
  and committed in *one* transaction. Stamping the digest separately would risk a
  kill leaving *digest-fresh + tables-empty*, which :meth:`is_fresh` would then
  trust and silently serve as empty mappings. The single transaction makes that
  state unreachable.
* **No migration; rebuild on format change.** There is no ``ALTER`` path. The
  table format is guarded by :data:`SCHEMA_VERSION` (stored in ``PRAGMA
  user_version``); a mismatch simply DROPs and recreates the tables - safe, because
  every row is re-derivable from the source files on the next run.
* **Fail-open.** A corrupt/not-a-database file is quarantined and a fresh
  ``:memory:`` store started (one re-parse, the safe direction), mirroring
  ``CacheStore``.

``MappingResolver`` builds this once per cycle inside ``single_instance_lock``, so
concurrent repopulation across processes is already prevented; the connection is
not shared across threads.
"""

import contextlib
import logging
import sqlite3
from collections.abc import Callable, Iterable

from .sqlite_util import connect, is_corruption, quarantine_corrupt

# Bump when the table layout below changes; a stored ``user_version`` that differs
# triggers a DROP+rebuild (the data is a pure cache, re-derived from the sources).
SCHEMA_VERSION = 1

# Source names used as ``meta`` keys and to select which tables a ``replace_*``
# clears. Kept as constants so the resolver and the store agree on the spelling.
SOURCE_ANIME_IDS = "anime_ids"
SOURCE_ANIBRIDGE = "anibridge"
SOURCE_ANIDB = "anidb"

# Digest stamped for a source populated from a pre-parsed config object rather than
# a downloaded file: it has no file to hash, and must never be considered fresh
# against a real file digest, so it is always repopulated.
INLINE_DIGEST = "<inline>"

# anime_ids query columns the resolver may filter / DISTINCT on. An allowlist
# because a column name cannot be a bound parameter; only these reach an f-string.
_ANIME_ID_COLUMNS = ("tvdb_id", "tmdb_movie_id", "tmdb_show_id", "imdb_id")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    name   TEXT PRIMARY KEY,
    digest TEXT NOT NULL
);

-- anilist_id is nullable on purpose: a Kometa record with an external id but no
-- anilist_id contributes to the library-filter candidate sets (anime_ids_distinct)
-- exactly as the former full-map scan did, but is excluded from the id->entry
-- lookups (which the former reverse index also skipped). See anime_ids_lookup.
CREATE TABLE IF NOT EXISTS anime_ids (
    anilist_id    INTEGER,
    tvdb_id       INTEGER,
    tvdb_season   INTEGER NOT NULL DEFAULT -1,
    tvdb_epoffset INTEGER NOT NULL DEFAULT 0,
    tmdb_movie_id INTEGER,
    tmdb_show_id  INTEGER,
    imdb_id       TEXT,
    anidb_id      INTEGER
);
CREATE INDEX IF NOT EXISTS ix_anime_ids_tvdb       ON anime_ids (tvdb_id);
CREATE INDEX IF NOT EXISTS ix_anime_ids_tmdb_movie ON anime_ids (tmdb_movie_id);
CREATE INDEX IF NOT EXISTS ix_anime_ids_tmdb_show  ON anime_ids (tmdb_show_id);
CREATE INDEX IF NOT EXISTS ix_anime_ids_imdb       ON anime_ids (imdb_id);

CREATE TABLE IF NOT EXISTS anibridge_entry (
    anilist_id         INTEGER PRIMARY KEY,
    anidb_id           INTEGER,
    imdb_id            TEXT,
    tmdb_movie_id      INTEGER,
    mal_id             INTEGER,
    first_tvdb_id      INTEGER,
    first_tmdb_show_id INTEGER
);

-- ext_id mixes int (tvdb/tmdb) and str (imdb). The column is declared BLOB so it
-- has BLOB (no) affinity: SQLite stores each value with its native type and does
-- no NUMERIC coercion, so an imdb string is never mangled and a bound int/str
-- matches the stored value of the same type through the index.
CREATE TABLE IF NOT EXISTS anibridge_xref (
    axis       TEXT NOT NULL,
    ext_id     BLOB NOT NULL,
    anilist_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_anibridge_xref ON anibridge_xref (axis, ext_id);

-- One row per (anilist, tvdb, season) episode range. A season that is *present*
-- but carries no ranges (an "s<N>" scope with an empty episode map) is stored as a
-- marker row with NULL start_ep: it must round-trip, because downstream an empty
-- range list means "whole season covered" whereas a missing season means "not
-- covered" - opposite outcomes. A real range always has a non-NULL start_ep.
CREATE TABLE IF NOT EXISTS anibridge_tvdb_range (
    anilist_id INTEGER NOT NULL,
    tvdb_id    INTEGER NOT NULL,
    season     INTEGER NOT NULL,
    start_ep   INTEGER,
    end_ep     INTEGER
);
CREATE INDEX IF NOT EXISTS ix_anibridge_range ON anibridge_tvdb_range (anilist_id, tvdb_id);

CREATE TABLE IF NOT EXISTS anidb_mapping (
    anidb_id    INTEGER NOT NULL,
    tvdb_season INTEGER NOT NULL,
    tvdb_ep     INTEGER NOT NULL,
    anidb_ep    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_anidb_mapping ON anidb_mapping (anidb_id, tvdb_season);

CREATE TABLE IF NOT EXISTS anidb_ambiguous (
    anidb_id INTEGER PRIMARY KEY
);
"""

# Which tables a source owns, so ``replace_<source>`` clears exactly its own rows.
_SOURCE_TABLES: dict[str, tuple[str, ...]] = {
    SOURCE_ANIME_IDS: ("anime_ids",),
    SOURCE_ANIBRIDGE: ("anibridge_entry", "anibridge_xref", "anibridge_tvdb_range"),
    SOURCE_ANIDB: ("anidb_mapping", "anidb_ambiguous"),
}

# Drop in FK-free order; all tables are independent so any order works. Derived
# from _SOURCE_TABLES (plus the shared ``meta`` table) so a new source's tables are
# dropped on a SCHEMA_VERSION rebuild automatically - no parallel hand-maintained
# list to forget (which would leave a stale table that CREATE IF NOT EXISTS skips).
_DROP_ALL = "".join(
    f"DROP TABLE IF EXISTS {t};\n" for t in ("meta", *(t for tables in _SOURCE_TABLES.values() for t in tables))
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the tables, rebuilding from scratch if the stored format is stale.

    A fresh db has ``user_version`` 0, which never equals :data:`SCHEMA_VERSION`, so
    the rebuild branch creates everything. An existing db at the current version
    just re-ensures the (already present) tables.
    """

    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        conn.executescript(_DROP_ALL)
        conn.executescript(_SCHEMA)
        # PRAGMA can't be parameterised; SCHEMA_VERSION is a trusted int constant.
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    else:
        conn.executescript(_SCHEMA)
    conn.commit()


class MappingStore:
    """Owns ``mappings.db``: schema, per-source freshness, atomic populate, queries."""

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._path = path

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def open(cls, path: str, *, logger: logging.Logger | None = None) -> "MappingStore":
        """Open (or create) the mappings db, quarantining a corrupt file.

        A missing file is created in place (this is a derived cache we *want* on
        disk, so there is no preview/in-memory staging like ``CacheStore``). A
        not-a-database/torn file is moved aside and a fresh ``:memory:`` store is
        returned so the run fails open (re-parsing into memory) instead of
        crash-looping.

        Args:
            path (str): Path to ``mappings.db`` (or ``":memory:"`` for tests).
            logger (logging.Logger | None): For the one-line quarantine notice.
        """

        conn: sqlite3.Connection | None = None
        try:
            conn = connect(path)
            _ensure_schema(conn)
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            if not is_corruption(exc):
                raise
            quarantine_corrupt(
                path,
                logger=logger,
                what="Mappings database",
                recovery="started a fresh one (sources will be re-parsed this run).",
            )
            conn = connect(":memory:")
            _ensure_schema(conn)
        return cls(conn, path)

    def close(self) -> None:
        """Roll back anything uncommitted and close the connection (idempotent)."""

        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    # -- freshness -----------------------------------------------------------

    def is_fresh(self, name: str, digest: str) -> bool:
        """True iff ``name``'s stored digest equals ``digest`` (so no re-parse)."""

        row = self._conn.execute("SELECT digest FROM meta WHERE name = ?", (name,)).fetchone()
        return row is not None and row[0] == digest

    # -- atomic populate -----------------------------------------------------

    def _replace(self, name: str, digest: str, write: Callable[[sqlite3.Connection], None]) -> None:
        """Clear ``name``'s tables, run ``write`` to repopulate, stamp the digest.

        All in ONE transaction (legacy/deferred control means the first DELETE opens
        it and ``commit`` closes it), so a kill mid-populate rolls back to the prior
        state - never digest-fresh + empty. ``write`` is a callback given the live
        connection; it only inserts rows.
        """

        try:
            for table in _SOURCE_TABLES[name]:
                self._conn.execute(f"DELETE FROM {table}")
            write(self._conn)
            self._conn.execute(
                "INSERT INTO meta (name, digest) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET digest = excluded.digest",
                (name, digest),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def replace_anime_ids(self, digest: str, rows: Iterable[tuple[object, ...]]) -> None:
        """Atomically replace the anime_ids rows.

        Args:
            digest (str): sha256 of the source file (or :data:`INLINE_DIGEST`).
            rows: ``(anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, tmdb_movie_id,
                tmdb_show_id, imdb_id, anidb_id)`` tuples.
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anime_ids (anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, "
                "tmdb_movie_id, tmdb_show_id, imdb_id, anidb_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

        self._replace(SOURCE_ANIME_IDS, digest, write)

    def replace_anibridge(
        self,
        digest: str,
        entries: Iterable[tuple[object, ...]],
        xrefs: Iterable[tuple[object, ...]],
        ranges: Iterable[tuple[object, ...]],
    ) -> None:
        """Atomically replace the anibridge tables.

        Args:
            digest (str): sha256 of the source file (or :data:`INLINE_DIGEST`).
            entries: ``(anilist_id, anidb_id, imdb_id, tmdb_movie_id, mal_id,
                first_tvdb_id, first_tmdb_show_id)`` tuples.
            xrefs: ``(axis, ext_id, anilist_id)`` reverse-index tuples.
            ranges: ``(anilist_id, tvdb_id, season, start_ep, end_ep)`` tuples;
                ``start_ep`` NULL marks a present-but-empty season.
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anibridge_entry (anilist_id, anidb_id, imdb_id, tmdb_movie_id, "
                "mal_id, first_tvdb_id, first_tmdb_show_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                entries,
            )
            conn.executemany(
                "INSERT INTO anibridge_xref (axis, ext_id, anilist_id) VALUES (?, ?, ?)",
                xrefs,
            )
            conn.executemany(
                "INSERT INTO anibridge_tvdb_range (anilist_id, tvdb_id, season, start_ep, end_ep) "
                "VALUES (?, ?, ?, ?, ?)",
                ranges,
            )

        self._replace(SOURCE_ANIBRIDGE, digest, write)

    def replace_anidb(
        self,
        digest: str,
        mappings: Iterable[tuple[object, ...]],
        ambiguous: Iterable[tuple[int]],
    ) -> None:
        """Atomically replace the anidb tables.

        Args:
            digest (str): sha256 of the source file (or :data:`INLINE_DIGEST`).
            mappings: ``(anidb_id, tvdb_season, tvdb_ep, anidb_ep)`` tuples.
            ambiguous: ``(anidb_id,)`` tuples for ids appearing in >1 ``<anime>``.
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anidb_mapping (anidb_id, tvdb_season, tvdb_ep, anidb_ep) VALUES (?, ?, ?, ?)",
                mappings,
            )
            conn.executemany(
                "INSERT INTO anidb_ambiguous (anidb_id) VALUES (?) ON CONFLICT (anidb_id) DO NOTHING",
                ambiguous,
            )

        self._replace(SOURCE_ANIDB, digest, write)

    # -- anime_ids queries ---------------------------------------------------

    def anime_ids_lookup(self, column: str, value: object) -> list[tuple[object, ...]]:
        """anime_ids rows matching ``column == value``, in first-seen (rowid) order.

        Returns the full row tuple ``(anilist_id, tvdb_id, tvdb_season,
        tvdb_epoffset, tmdb_movie_id, tmdb_show_id, imdb_id, anidb_id)`` so the
        caller can build a ``MappingEntry``. Rows with no ``anilist_id`` are
        excluded (the former reverse index skipped them); ``column`` must be one of
        :data:`_ANIME_ID_COLUMNS` (it is interpolated, so it is allowlisted).
        """

        if column not in _ANIME_ID_COLUMNS:
            raise ValueError(f"Unknown anime_ids column: {column!r}")
        return self._conn.execute(
            "SELECT anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, tmdb_movie_id, "
            f"tmdb_show_id, imdb_id, anidb_id FROM anime_ids "
            f"WHERE {column} = ? AND anilist_id IS NOT NULL ORDER BY rowid",
            (value,),
        ).fetchall()

    def anime_ids_distinct(self, column: str) -> set[object]:
        """The set of DISTINCT non-null ``column`` values in anime_ids.

        Used to build the library-filter candidate sets without scanning the map.
        """

        if column not in _ANIME_ID_COLUMNS:
            raise ValueError(f"Unknown anime_ids column: {column!r}")
        rows = self._conn.execute(
            f"SELECT DISTINCT {column} FROM anime_ids WHERE {column} IS NOT NULL",
        ).fetchall()
        return {r[0] for r in rows}

    # -- anibridge queries ---------------------------------------------------

    def anibridge_entries_for(
        self,
        axis: str,
        ext_id: object,
    ) -> list[tuple[int, object, object, object, object, object, object]]:
        """Every ``(anilist_id, *entry)`` row mapped to ``ext_id`` on ``axis``.

        One xref->entry JOIN so a lookup that resolves k AniList ids costs a single
        query instead of k per-id point lookups. Row shape: ``(anilist_id, anidb_id,
        imdb_id, tmdb_movie_id, mal_id, first_tvdb_id, first_tmdb_show_id)`` - the
        stored ``_consumer_entry`` picks the caller rebuilds the entry from. The
        INNER JOIN drops any xref row lacking an entry, but ``to_rows`` writes both
        from the same ``by_anilist`` map, so that pairing is structurally guaranteed.
        """

        return self._conn.execute(
            "SELECT x.anilist_id, e.anidb_id, e.imdb_id, e.tmdb_movie_id, e.mal_id, "
            "e.first_tvdb_id, e.first_tmdb_show_id "
            "FROM anibridge_xref x JOIN anibridge_entry e ON e.anilist_id = x.anilist_id "
            "WHERE x.axis = ? AND x.ext_id = ?",
            (axis, ext_id),
        ).fetchall()

    def anibridge_ranges_for(
        self,
        axis: str,
        ext_id: object,
        tvdb_id: int,
    ) -> list[tuple[int, int, int | None, int | None]]:
        """``(anilist_id, season, start_ep, end_ep)`` rows for an (axis, ext_id) set,
        scoped to ``tvdb_id``, in ``(anilist_id, populate)`` order.

        The batched twin of the former per-id range lookup: one xref->range JOIN
        fetches the ranges for every AniList id a tvdb lookup resolves, so the caller
        groups them by ``anilist_id`` and rebuilds each season's list in insertion
        order (``ORDER BY x.anilist_id, r.rowid`` -> parity with the in-memory build).
        A NULL ``start_ep`` row is the present-but-empty-season marker.
        """

        return self._conn.execute(
            "SELECT x.anilist_id, r.season, r.start_ep, r.end_ep "
            "FROM anibridge_xref x JOIN anibridge_tvdb_range r ON r.anilist_id = x.anilist_id "
            "WHERE x.axis = ? AND x.ext_id = ? AND r.tvdb_id = ? "
            "ORDER BY x.anilist_id, r.rowid",
            (axis, ext_id, tvdb_id),
        ).fetchall()

    def anibridge_distinct(self, axis: str) -> set[object]:
        """The set of all ext ids on ``axis`` (for the library-filter id sets)."""

        rows = self._conn.execute(
            "SELECT DISTINCT ext_id FROM anibridge_xref WHERE axis = ?",
            (axis,),
        ).fetchall()
        return {r[0] for r in rows}

    def anibridge_len(self) -> int:
        """Number of AniList entries (backs ``AniBridge.__len__`` / ``__bool__``)."""

        return self._conn.execute("SELECT COUNT(*) FROM anibridge_entry").fetchone()[0]

    # -- anidb queries -------------------------------------------------------

    def anidb_is_ambiguous(self, anidb_id: int) -> bool:
        """True iff ``anidb_id`` appeared in more than one ``<anime>`` element."""

        row = self._conn.execute(
            "SELECT 1 FROM anidb_ambiguous WHERE anidb_id = ?",
            (anidb_id,),
        ).fetchone()
        return row is not None

    def anidb_rows(self, anidb_id: int, tvdb_season: int) -> list[tuple[int, int]]:
        """``(tvdb_ep, anidb_ep)`` rows for ``anidb_id`` scoped to ``tvdb_season``."""

        return self._conn.execute(
            "SELECT tvdb_ep, anidb_ep FROM anidb_mapping WHERE anidb_id = ? AND tvdb_season = ?",
            (anidb_id, tvdb_season),
        ).fetchall()
