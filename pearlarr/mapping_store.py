"""SQLite cache of the parsed + indexed id-mapping sources (`mappings.db`).

The three immutable mapping sources - the Kometa Anime-IDs JSON, the AniDB
anime-list XML and the anibridge graph - are large, read-only files whose
parsed+indexed forms never change until the file is re-downloaded *with new
content*. Re-parsing and re-indexing them on every fresh process start is wasted
CPU (~1.2s) and resident memory (~51MB held for the process lifetime). This store
persists the indexed forms in a dedicated `mappings.db` (next to `cache.db`),
keyed by each source's content digest, so a process whose source files are
unchanged answers lookups straight from SQL without re-parsing - and never has to
hold the full parsed structures in memory.

How this differs from `CacheStore` (deliberately a separate file and a separate db):

* **Never preview-gated.** `cache.db` stages writes and only commits on a
  non-preview save; this store is a pure derived cache of an *immutable download*,
  so a freshly parsed index is always committed, even on `--dry-run`. Each
  `replace_*` is its own atomic transaction.
* **Atomic populate.** A source's rows and its `meta` digest stamp are written
  and committed in *one* transaction. Stamping the digest separately would risk a
  kill leaving *digest-fresh + tables-empty*, which `is_fresh` would then
  trust and silently serve as empty mappings. The single transaction makes that
  state unreachable.
* **No migration; rebuild on format change.** There is no `ALTER` path. The
  table format is guarded by `SCHEMA_VERSION` (stored in `PRAGMA
  user_version`); a mismatch simply DROPs and recreates the tables - safe, because
  every row is re-derivable from the source files on the next run.
* **Fail-open.** A corrupt/not-a-database file is quarantined and a fresh
  `:memory:` store started (one re-parse, the safe direction), mirroring
  `CacheStore`.

`MappingResolver` builds this once per cycle inside `single_instance_lock`, so
concurrent repopulation across processes is already prevented; the connection is
not shared across threads.
"""

import sqlite3
from collections.abc import Callable, Iterable
from typing import Literal, NamedTuple, overload

from .sqlite_util import connect, open_or_quarantine, rollback_and_close

# Bump when the table layout below changes; a stored `user_version` that differs
# triggers a DROP+rebuild (the data is a pure cache, re-derived from the sources).
SCHEMA_VERSION = 2

# Source names used as `meta` keys and to select which tables a `replace_*`
# clears. Kept as constants so the resolver and the store agree on the spelling.
SOURCE_ANIME_IDS = "anime_ids"
SOURCE_ANIBRIDGE = "anibridge"
SOURCE_ANIDB = "anidb"

# Digest stamped for a source populated from a pre-parsed config object rather than
# a downloaded file: it has no file to hash, and must never be considered fresh
# against a real file digest, so it is always repopulated.
INLINE_DIGEST = "<inline>"

# anime_ids query columns the resolver may filter / DISTINCT on. The Literal is
# the checker-facing vocabulary; the tuple is the runtime allowlist guarding the
# f-string interpolation (a column name cannot be a bound parameter). Keep in sync.
type AnimeIdColumn = Literal["tvdb_id", "tmdb_movie_id", "imdb_id"]
_ANIME_ID_COLUMNS = ("tvdb_id", "tmdb_movie_id", "imdb_id")

# anibridge xref axes - a DIFFERENT vocabulary from the anime_ids columns above
# ("tvdb", not "tvdb_id"). tvdb/tmdb ext ids are ints; imdb ids are strs.
type AniBridgeAxis = Literal["tvdb", "tmdb_movie", "imdb"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    name   TEXT PRIMARY KEY,
    digest TEXT NOT NULL
);

-- anilist_id is nullable on purpose: a Kometa record with an external id but no
-- anilist_id feeds the library-filter candidate sets (anime_ids_distinct) but is
-- excluded from the id->entry lookups. See anime_ids_lookup.
CREATE TABLE IF NOT EXISTS anime_ids (
    anilist_id    INTEGER,
    tvdb_id       INTEGER,
    tvdb_season   INTEGER NOT NULL DEFAULT -1,
    tvdb_epoffset INTEGER NOT NULL DEFAULT 0,
    tmdb_movie_id INTEGER,
    imdb_id       TEXT,
    anidb_id      INTEGER
);
CREATE INDEX IF NOT EXISTS ix_anime_ids_tvdb       ON anime_ids (tvdb_id);
CREATE INDEX IF NOT EXISTS ix_anime_ids_tmdb_movie ON anime_ids (tmdb_movie_id);
CREATE INDEX IF NOT EXISTS ix_anime_ids_imdb       ON anime_ids (imdb_id);

CREATE TABLE IF NOT EXISTS anibridge_entry (
    anilist_id    INTEGER PRIMARY KEY,
    anidb_id      INTEGER,
    imdb_id       TEXT,
    tmdb_movie_id INTEGER
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

# Which tables a source owns, so `replace_<source>` clears exactly its own rows.
_SOURCE_TABLES: dict[str, tuple[str, ...]] = {
    SOURCE_ANIME_IDS: ("anime_ids",),
    SOURCE_ANIBRIDGE: ("anibridge_entry", "anibridge_xref", "anibridge_tvdb_range"),
    SOURCE_ANIDB: ("anidb_mapping", "anidb_ambiguous"),
}

# Drop in FK-free order; all tables are independent so any order works. Derived
# from _SOURCE_TABLES (plus the shared `meta` table) so a new source's tables are
# dropped on a SCHEMA_VERSION rebuild automatically - no parallel hand-maintained
# list to forget (which would leave a stale table that CREATE IF NOT EXISTS skips).
_DROP_ALL = "".join(
    f"DROP TABLE IF EXISTS {t};\n" for t in ("meta", *(t for tables in _SOURCE_TABLES.values() for t in tables))
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the tables, rebuilding from scratch if the stored format is stale.

    A fresh db has `user_version` 0, which never equals `SCHEMA_VERSION`, so
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


class AnimeIdRow(NamedTuple):
    """One stored `anime_ids` row, in column order.

    A typed record (not a positional `tuple[object, ...]`) shared by the whole
    pipeline - `_anime_ids_rows` builds it, `replace_anime_ids` inserts it,
    `anime_ids_lookup` returns it, `_entry_from_anime_row` reads named fields -
    so the four column lists can't silently transpose. `anilist_id` is typed
    `int` because the only consumer (`anime_ids_lookup`) filters
    `anilist_id IS NOT NULL`, even though the column itself is nullable.
    """

    anilist_id: int
    tvdb_id: int | None
    tvdb_season: int
    tvdb_epoffset: int
    tmdb_movie_id: int | None
    imdb_id: str | None
    anidb_id: int | None


class AniBridgeEntryRow(NamedTuple):
    """One `anibridge_entry` row: the computed `_consumer_entry` picks.

    A typed record (not a positional `tuple[object, ...]`) shared by the producer
    (`AniBridge.to_rows`) and the consumer (`anibridge_entries_for`), so the two
    column lists agree by construction. Tuple-compatible, so it's built straight
    from a fetched row with `AniBridgeEntryRow(*row)`.
    """

    anilist_id: int
    anidb_id: int | None
    imdb_id: str | None
    tmdb_movie_id: int | None


class AniBridgeXrefRow(NamedTuple):
    """One `anibridge_xref` reverse-index row (`axis` -> ext id -> AniList id)."""

    axis: str
    ext_id: int | str
    anilist_id: int


class AniBridgeRangeRow(NamedTuple):
    """One `anibridge_tvdb_range` row; a NULL `start_ep` marks an empty season."""

    anilist_id: int
    tvdb_id: int
    season: int
    start_ep: int | None
    end_ep: int | None


class AniBridgeRows(NamedTuple):
    """The three anibridge row tables one graph flattens to (`AniBridge.to_rows`)."""

    entries: list[AniBridgeEntryRow]
    xrefs: list[AniBridgeXrefRow]
    ranges: list[AniBridgeRangeRow]


class AniBridgeRangeHit(NamedTuple):
    """One `MappingStore.anibridge_ranges_for` result row (tvdb-scoped)."""

    anilist_id: int
    season: int
    start_ep: int | None
    end_ep: int | None


class AnidbMappingRow(NamedTuple):
    """One `anidb_mapping` row: a tvdb episode -> anidb episode pair."""

    anidb_id: int
    tvdb_season: int
    tvdb_ep: int
    anidb_ep: int


class AnidbEpPair(NamedTuple):
    """One `anidb_rows` result: a season-scoped tvdb episode -> anidb episode pair.

    A typed record (not a positional `tuple[int, int]`) so the two ordered ints
    can't silently transpose; the caller builds `dict(rows)` off the field order.
    """

    tvdb_ep: int
    anidb_ep: int


def _ext_id_as[T: (int, str)](value: object, kind: type[T], axis: str) -> T:
    """Pin a stored anibridge ext id to the type its axis owns.

    The write path coerced these, so a mismatch means the db is corrupt - raise
    (with the type names only, never the value) instead of folding.
    """

    if not isinstance(value, kind):
        raise TypeError(f"anibridge xref {axis!r} ext_id holds {type(value).__name__}, expected {kind.__name__}")
    return value


class MappingStore:
    """Owns `mappings.db`: schema, per-source freshness, atomic populate, queries."""

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._path = path

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> "MappingStore":
        """Open (or create) the mappings db, quarantining a corrupt file.

        A missing file is created in place (this is a derived cache we *want* on
        disk, so there is no preview/in-memory staging like `CacheStore`). A
        not-a-database/torn file is moved aside and a fresh `:memory:` store is
        returned so the run fails open (re-parsing into memory) instead of
        crash-looping.

        Args:
            path: Path to `mappings.db` (or `":memory:"` for tests).
        """

        conn, _ = open_or_quarantine(
            path,
            connect_fn=connect,
            ensure=_ensure_schema,
            what="Mappings database",
            recovery="started a fresh one (sources will be re-parsed this run)",
        )
        return cls(conn, path)

    def close(self) -> None:
        """Roll back anything uncommitted and close the connection (idempotent)."""

        rollback_and_close(self._conn)

    # -- freshness -----------------------------------------------------------

    def is_fresh(self, name: str, digest: str) -> bool:
        """True iff `name`'s stored digest equals `digest` (so no re-parse)."""

        row = self._conn.execute("SELECT digest FROM meta WHERE name = ?", (name,)).fetchone()
        return row is not None and row[0] == digest

    # -- atomic populate -----------------------------------------------------

    def _replace(self, name: str, digest: str, write: Callable[[sqlite3.Connection], None]) -> None:
        """Clear `name`'s tables, run `write` to repopulate, stamp the digest.

        All in ONE transaction (legacy/deferred control means the first DELETE opens
        it and `commit` closes it), so a kill mid-populate rolls back to the prior
        state - never digest-fresh + empty. `write` is a callback given the live
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

    def replace_anime_ids(self, digest: str, rows: Iterable[AnimeIdRow]) -> None:
        """Atomically replace the anime_ids rows.

        Args:
            digest: sha256 of the source file (or `INLINE_DIGEST`).
            rows: `AnimeIdRow` tuples (column order), one per Kometa record.
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anime_ids (anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, "
                "tmdb_movie_id, imdb_id, anidb_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

        self._replace(SOURCE_ANIME_IDS, digest, write)

    def replace_anibridge(
        self,
        digest: str,
        rows: AniBridgeRows,
    ) -> None:
        """Atomically replace the anibridge tables.

        Args:
            digest: sha256 of the source file (or `INLINE_DIGEST`).
            rows: The three row tables one graph flattens to
                (a `range` row's `start_ep` NULL marks a present-but-empty
                season).
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anibridge_entry (anilist_id, anidb_id, imdb_id, tmdb_movie_id) VALUES (?, ?, ?, ?)",
                rows.entries,
            )
            conn.executemany(
                "INSERT INTO anibridge_xref (axis, ext_id, anilist_id) VALUES (?, ?, ?)",
                rows.xrefs,
            )
            conn.executemany(
                "INSERT INTO anibridge_tvdb_range (anilist_id, tvdb_id, season, start_ep, end_ep) "
                "VALUES (?, ?, ?, ?, ?)",
                rows.ranges,
            )

        self._replace(SOURCE_ANIBRIDGE, digest, write)

    def replace_anidb(
        self,
        digest: str,
        mappings: Iterable[AnidbMappingRow],
        ambiguous: Iterable[int],
    ) -> None:
        """Atomically replace the anidb tables.

        Args:
            digest: sha256 of the source file (or `INLINE_DIGEST`).
            mappings: `AnidbMappingRow` tuples (column order).
            ambiguous: anidb ids appearing in more than one `<anime>` element.
        """

        def write(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO anidb_mapping (anidb_id, tvdb_season, tvdb_ep, anidb_ep) VALUES (?, ?, ?, ?)",
                mappings,
            )
            conn.executemany(
                "INSERT INTO anidb_ambiguous (anidb_id) VALUES (?) ON CONFLICT (anidb_id) DO NOTHING",
                ((anidb_id,) for anidb_id in ambiguous),
            )

        self._replace(SOURCE_ANIDB, digest, write)

    # -- anime_ids queries ---------------------------------------------------

    def anime_ids_lookup(self, column: AnimeIdColumn, value: object) -> list[AnimeIdRow]:
        """`AnimeIdRow`s matching `column == value`, in first-seen (rowid) order.

        Returns the full row so the caller can build a `MappingEntry`. Rows with no
        `anilist_id` are excluded; `column` must be one of `_ANIME_ID_COLUMNS`
        (it is interpolated, so it is allowlisted). The SELECT column order
        matches `AnimeIdRow`'s fields.
        """

        if column not in _ANIME_ID_COLUMNS:
            raise ValueError(f"Unknown anime_ids column: {column!r}")
        rows = self._conn.execute(
            "SELECT anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, tmdb_movie_id, "
            f"imdb_id, anidb_id FROM anime_ids "
            f"WHERE {column} = ? AND anilist_id IS NOT NULL ORDER BY rowid",
            (value,),
        ).fetchall()
        return [AnimeIdRow(*row) for row in rows]

    @overload
    def anime_ids_distinct(self, column: Literal["tvdb_id", "tmdb_movie_id"]) -> set[int]: ...
    @overload
    def anime_ids_distinct(self, column: Literal["imdb_id"]) -> set[str]: ...
    def anime_ids_distinct(self, column: AnimeIdColumn) -> set[int] | set[str]:
        """The set of DISTINCT non-null `column` values in anime_ids.

        Used to build the library-filter candidate sets without scanning the map.
        Third-party Kometa JSON reaches these columns un-coerced, so a junk-typed
        value is possible: it is SKIPPED (junk in a candidate set never matched an
        arr id anyway).
        """

        if column not in _ANIME_ID_COLUMNS:
            raise ValueError(f"Unknown anime_ids column: {column!r}")
        rows = self._conn.execute(
            f"SELECT DISTINCT {column} FROM anime_ids WHERE {column} IS NOT NULL",
        ).fetchall()
        if column == "imdb_id":
            return {v for r in rows if isinstance(v := r[0], str)}
        return {v for r in rows if isinstance(v := r[0], int)}

    # -- anibridge queries ---------------------------------------------------

    def anibridge_entries_for(self, axis: str, ext_id: int | str) -> list[AniBridgeEntryRow]:
        """Every `AniBridgeEntryRow` mapped to `ext_id` on `axis`.

        One xref->entry JOIN so a lookup that resolves k AniList ids costs a single
        query instead of k per-id point lookups. The INNER JOIN drops any xref row
        lacking an entry, but `to_rows` writes both from the same `by_anilist`
        map, so that pairing is structurally guaranteed.
        """

        rows = self._conn.execute(
            "SELECT x.anilist_id, e.anidb_id, e.imdb_id, e.tmdb_movie_id "
            "FROM anibridge_xref x JOIN anibridge_entry e ON e.anilist_id = x.anilist_id "
            "WHERE x.axis = ? AND x.ext_id = ?",
            (axis, ext_id),
        ).fetchall()
        return [AniBridgeEntryRow(*row) for row in rows]

    def anibridge_ranges_for(
        self,
        axis: str,
        ext_id: int | str,
        tvdb_id: int,
    ) -> list[AniBridgeRangeHit]:
        """The `tvdb_id`-scoped `AniBridgeRangeHit` rows for an (axis, ext_id) lookup.

        Ordered by `(anilist_id, rowid)`. Batched: one xref->range JOIN
        fetches the ranges for every AniList id a tvdb lookup resolves, so the
        caller groups them by `anilist_id` and rebuilds each season's list in
        insertion order (`ORDER BY x.anilist_id, r.rowid` -> parity with the
        in-memory build). A NULL `start_ep` row is the present-but-empty-season
        marker.
        """

        rows = self._conn.execute(
            "SELECT x.anilist_id, r.season, r.start_ep, r.end_ep "
            "FROM anibridge_xref x JOIN anibridge_tvdb_range r ON r.anilist_id = x.anilist_id "
            "WHERE x.axis = ? AND x.ext_id = ? AND r.tvdb_id = ? "
            "ORDER BY x.anilist_id, r.rowid",
            (axis, ext_id, tvdb_id),
        ).fetchall()
        return [AniBridgeRangeHit(*row) for row in rows]

    @overload
    def anibridge_distinct(self, axis: Literal["tvdb", "tmdb_movie"]) -> set[int]: ...
    @overload
    def anibridge_distinct(self, axis: Literal["imdb"]) -> set[str]: ...
    def anibridge_distinct(self, axis: AniBridgeAxis) -> set[int] | set[str]:
        """The set of all ext ids on `axis` (for the library-filter id sets).

        Unlike the anime_ids columns, the AniBridge write path coerces every ext
        id per axis before it reaches the store, so a mismatched stored type is
        corruption: it RAISES rather than being skipped.
        """

        rows = self._conn.execute(
            "SELECT DISTINCT ext_id FROM anibridge_xref WHERE axis = ?",
            (axis,),
        ).fetchall()
        if axis == "imdb":
            return {_ext_id_as(r[0], str, axis) for r in rows}
        return {_ext_id_as(r[0], int, axis) for r in rows}

    def anibridge_len(self) -> int:
        """Number of AniList entries (backs `AniBridge.__len__` / `__bool__`)."""

        return self._conn.execute("SELECT COUNT(*) FROM anibridge_entry").fetchone()[0]

    # -- anidb queries -------------------------------------------------------

    def anidb_is_ambiguous(self, anidb_id: int) -> bool:
        """True iff `anidb_id` appeared in more than one `<anime>` element."""

        row = self._conn.execute(
            "SELECT 1 FROM anidb_ambiguous WHERE anidb_id = ?",
            (anidb_id,),
        ).fetchone()
        return row is not None

    def anidb_rows(self, anidb_id: int, tvdb_season: int) -> list[AnidbEpPair]:
        """`AnidbEpPair` rows for `anidb_id` scoped to `tvdb_season`.

        The SELECT column order matches `AnidbEpPair`'s fields.
        """

        rows = self._conn.execute(
            "SELECT tvdb_ep, anidb_ep FROM anidb_mapping WHERE anidb_id = ? AND tvdb_season = ?",
            (anidb_id, tvdb_season),
        ).fetchall()
        return [AnidbEpPair(*row) for row in rows]
