# pyright: strict
"""Behavioral tests for the SQLite-backed `CacheStore`.

Pins behavior, not internals: the cached title rows, the JSONB meta/parse caches,
pending imports, the stored version and config hash, and - critically -
the staged-write / dry-run gate (a dry run never persists, a real save does) and
the in-memory -> file promotion for a missing cache.
"""

import contextlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import pearlarr
from pearlarr.arr_models import ParsedFileInfo
from pearlarr.cache import (
    SCHEMA_VERSION,
    CachedTitleUpdate,
    CacheSchemaError,
    CacheStore,
    HistoryCheckpoint,
    row_payload,
    stamp_is_fresh,
)
from pearlarr.config import Arr
from pearlarr.log import LOG_NAME
from pearlarr.output import Diagnostic, Severity, install_hub
from pearlarr.parse_cache import parse_from_json, parse_to_json
from pearlarr.pending_import import OwnedAtGrab, PendingImport, SizeMatchedEpisode, TitleNames
from pearlarr.sqlite_util import is_corruption
from pearlarr.stamps import parse_stamp_or_none
from pearlarr.tvdb_orders import CachedOrders, OrderRow

from .builders import cached_as_done, claim_al_ids, make_entry_record, pending_import, title_claim
from .fakes import RecordingHub

# Stand-in for a config hash. `CacheStore` only stores and compares the
# value it is handed. It never computes one, so any string works here.
CONFIG_HASH = "0123456789abcdef0123456789abcdef"


def _cached_title_name(store: CacheStore, al_id: int, arr: Arr = Arr.SONARR) -> str | None:
    """The cached title's stored name via `get_cached_title` (None when the row is absent)."""

    cached = store.get_cached_title(arr, al_id)
    return None if cached is None else cached.name


def _raise_os_replace(*_args: object, **_kwargs: object) -> None:
    """A drop-in for `os.replace` that fails the atomic promote rename."""

    raise OSError("boom")


def _raise_locked(*_args: object, **_kwargs: object) -> sqlite3.Connection:
    """A drop-in for `connect` that simulates a transient open-time lock."""

    raise sqlite3.OperationalError("database is locked")


def _row(pending: PendingImport) -> dict[str, Any]:
    """The pending import as the store's JSON column reads it back (tuples as lists)."""

    return json.loads(json.dumps(pending.to_json()))


def _open(tmp_path: Path) -> contextlib.closing[CacheStore]:
    return contextlib.closing(CacheStore.load(str(tmp_path / "cache.db"), config_hash=CONFIG_HASH))


class TestSchemaAndDescriptor:
    """A missing db opens in-memory and reads empty. A saved db persists the version and config hash."""

    def test_missing_file_opens_in_memory(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            # Nothing on disk yet, and reads work against the empty in-memory schema.
            assert not (tmp_path / "cache.db").exists()
            assert store.get_cached_title(Arr.SONARR, 7) is None

    def test_descriptor_persists_version_and_checksum(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.save(dry_run=False)

        # Read kv with a RAW connection. CacheStore.load rewrites the version and config hash on
        # open (via _reconcile), so reopening through the facade would read back the
        # just-rewritten constants and pass even if save() never persisted them.
        db = tmp_path / "cache.db"
        assert db.exists()
        raw = sqlite3.connect(str(db))
        try:
            rows = dict(raw.execute("SELECT key, value FROM kv").fetchall())
        finally:
            raw.close()
        assert rows.get("pearlarr_version") == pearlarr.__version__
        assert rows.get("config_checksum") == CONFIG_HASH


# `torrent_hashes` as every schema through v6 declared it. The v6 -> v7 step drops it.
_TORRENT_HASHES_TABLE = """
CREATE TABLE torrent_hashes (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL, infohash TEXT NOT NULL,
    PRIMARY KEY (arr, al_id, infohash),
    FOREIGN KEY (arr, al_id) REFERENCES entries (arr, al_id) ON DELETE CASCADE);
"""


# The original (pre-versioning) shape of the cached title tables: `entries` without
# `fallback_satisfied`, with `user_version` 0 by default. Pins the v0 -> v1
# migration (the shape a first-release cache.db still has on disk).
_V0_SCHEMA = (
    """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL,
    name TEXT, url TEXT, coverage TEXT, updated_at TEXT,
    PRIMARY KEY (arr, al_id));
"""
    + _TORRENT_HASHES_TABLE
)


# The v1 shape of `pending_imports` (PK without al_id), as shipped through 1.0.x.
# Only the table the v1 -> v2 step rebuilds is declared: `_ensure_schema` creates
# the rest fresh, and the step must leave them alone.
_V1_PENDING_SCHEMA = """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE pending_imports (
    arr      TEXT NOT NULL,
    infohash TEXT NOT NULL,
    record   BLOB NOT NULL,
    PRIMARY KEY (arr, infohash));
"""


# The v2 shape (composite PK, guards still inline in each blob), as shipped in
# 1.1.0. Only the table the v2 -> v3 step reads/strips is declared - `_ensure_schema`
# creates the rest fresh, including the `guard_facts` table the step backfills.
_V2_PENDING_SCHEMA = """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE pending_imports (
    arr      TEXT NOT NULL,
    infohash TEXT NOT NULL,
    al_id    INTEGER NOT NULL DEFAULT 0,
    record   BLOB NOT NULL,
    PRIMARY KEY (arr, infohash, al_id));
"""


# The v3 shape of the tables `CacheStore.load` needs: `kv` plus the `entries`
# the v3 -> v4 step rewrites (a name may hold the id form an unreachable AniList once stored).
_V3_ENTRIES_SCHEMA = """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL,
    name TEXT, url TEXT, coverage TEXT, updated_at TEXT,
    fallback_satisfied INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (arr, al_id));
"""


# The table with the index a v6 file holds beside it.
_V6_TORRENT_HASHES_SCHEMA = (
    _TORRENT_HASHES_TABLE + "CREATE INDEX ix_torrent_hashes_infohash ON torrent_hashes (infohash);\n"
)


def _seed_v2_db(db: Path, rows: list[tuple[str, str, int, str]]) -> None:
    """A v2 db whose `pending_imports` holds the given (arr, infohash, al_id, json) rows."""

    raw = sqlite3.connect(str(db))
    raw.executescript(_V2_PENDING_SCHEMA)
    raw.executemany(
        "INSERT INTO pending_imports (arr, infohash, al_id, record) VALUES (?, ?, ?, jsonb(?))",
        rows,
    )
    raw.execute("PRAGMA user_version=2")
    raw.commit()
    raw.close()


# The v4 shape: the v2 pending table (still one row per title) plus the `guard_facts`
# rows the v2 -> v3 step backfilled. The v4 -> v5 step merges the rows per torrent.
_V4_PENDING_SCHEMA = (
    _V2_PENDING_SCHEMA
    + """
CREATE TABLE guard_facts (
    arr    TEXT    NOT NULL,
    al_id  INTEGER NOT NULL,
    record BLOB    NOT NULL,
    PRIMARY KEY (arr, al_id));
"""
)


def _seed_v4_db(
    db: Path, rows: Sequence[tuple[str, str, int, str]], guards: Sequence[tuple[str, int, str]] = ()
) -> None:
    """A v4 db holding the given (arr, infohash, al_id, json) pending rows and (arr, al_id, json) guard rows."""

    raw = sqlite3.connect(str(db))
    raw.executescript(_V4_PENDING_SCHEMA)
    raw.executemany(
        "INSERT INTO pending_imports (arr, infohash, al_id, record) VALUES (?, ?, ?, jsonb(?))",
        rows,
    )
    raw.executemany("INSERT INTO guard_facts (arr, al_id, record) VALUES (?, ?, jsonb(?))", list(guards))
    raw.execute("PRAGMA user_version=4")
    raw.commit()
    raw.close()


def _v4_blob(infohash: str, al_id: int, **fields: Any) -> dict[str, Any]:
    """A fully shaped v4 pending row: one title's row on `infohash`, `fields` overriding."""

    blob: dict[str, Any] = {
        "infohash": infohash,
        "series_id": 5,
        "al_id": al_id,
        "title": f"Entry {al_id}",
        "coverage": "S01",
        "url": f"https://releases.moe/{al_id}",
        "ordered_episode_ids": [101],
        "names": {"series": "Series", "anilist": ["Alias"]},
        "preowned_episode_ids": [],
        "slice_coverage": "S01 E01",
        "added_at": "2026-07-01 00:00:00",
        "release_group": "Grp",
        "is_dual_audio": False,
        "seadex_files": ["a.mkv", "b.mkv"],
        "release_sizes": [10, 20],
        "file_episode_map": {"a.mkv": [101]},
        "excluded_files": [],
        "awaiting_cleanup": False,
    }
    blob.update(fields)
    return blob


def _assert_round_trips(rows: dict[str, dict[str, Any]]) -> None:
    """Each migrated blob round-trips byte-for-byte as a pending import, plus empty sizes and its unplaced listed files.

    The listed files the blob neither mapped nor excluded come back as its hand imports with no order-only keys, and
    no row is a cross-seed.
    """

    for infohash, blob in rows.items():
        rebuilt = PendingImport.from_json(blob, guards={}).to_json()
        unplaced: list[str] = [
            name
            for name in blob["seadex_files"]
            if name not in blob["file_episode_map"] and name not in blob["excluded_files"]
        ]
        expected: dict[str, Any] = {
            **blob,
            "sizes_by_name": dict[str, int](),
            "hand_import_files": unplaced,
            "order_only": dict[str, Any](),
            "cross_seed": False,
        }
        assert json.dumps(rebuilt) == json.dumps(expected), infohash


def _columns(db: Path, table: str) -> list[str]:
    """`table`'s column names as the file holds them. Empty when the file has no such table."""

    raw = sqlite3.connect(str(db))
    try:
        return [str(name) for (name,) in raw.execute("SELECT name FROM pragma_table_info(?)", (table,))]
    finally:
        raw.close()


def _user_version(db: Path) -> int:
    raw = sqlite3.connect(str(db))
    try:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])
    finally:
        raw.close()


class TestSchemaVersionGate:
    """The schema-version gate: older dbs migrate step-by-step, newer dbs are refused."""

    def test_fresh_db_is_stamped_through_promote(self, tmp_path: Path) -> None:
        # The :memory: stand-in gets its schema version on create. The backup-API promote must
        # carry that version into the file it writes, or every fresh install would
        # look like a v0 db on its second run.
        with _open(tmp_path) as store:
            store.save(dry_run=False)
        assert _user_version(tmp_path / "cache.db") == SCHEMA_VERSION

    def test_v0_db_is_upgraded_in_place(self, tmp_path: Path) -> None:
        # The scenario the gate exists for: a db from before `fallback_satisfied`
        # shipped must be ALTERed current instead of crashing get_cached_title every run.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V0_SCHEMA)
        raw.execute("INSERT INTO entries (arr, al_id, name) VALUES ('sonarr', 7, 'Frieren')")
        raw.commit()
        raw.close()
        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default

        with _open(tmp_path) as store:
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.name == "Frieren"
            assert cached.fallback_satisfied is False  # backfilled column default
        # The migration committed at load time - durable even though the run's own
        # staged writes were rolled back by close().
        assert _user_version(db) == SCHEMA_VERSION
        # Each step announces itself as an INFO hub Diagnostic, one per step walked.
        steps = recording.of_type(Diagnostic)
        assert [s.message for s in steps] == [
            f"Upgraded cache database schema v{n} -> v{n + 1}" for n in range(SCHEMA_VERSION)
        ]
        assert all(s.severity is Severity.INFO and s.origin == LOG_NAME for s in steps)

    def test_manually_altered_v0_db_upgrades_cleanly(self, tmp_path: Path) -> None:
        # A v0 db that already got the ALTER by hand (the pre-gate bridge): the
        # guarded migration step must not trip over the existing column.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V0_SCHEMA)
        raw.execute("ALTER TABLE entries ADD COLUMN fallback_satisfied INTEGER NOT NULL DEFAULT 0")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            assert store.get_cached_title(Arr.SONARR, 7) is None  # reads work after the migration
        assert _user_version(db) == SCHEMA_VERSION

    def test_v1_pending_imports_walk_the_chain_to_torrent_records(self, tmp_path: Path) -> None:
        # The v1 -> v2 step grows the (arr, infohash) PK to (arr, infohash, al_id)
        # via a table rebuild, and the v4 -> v5 merge turns each row into a claim.
        # A legacy row with no al_id in its JSON lands under the 0 sentinel and
        # ends as its hash's one claim with al_id 0. A row that DOES carry al_id
        # in its JSON backfills the key from it.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V1_PENDING_SCHEMA)
        raw.execute(
            "INSERT INTO pending_imports (arr, infohash, record) VALUES ('sonarr', 'legacy', jsonb(?))",
            ('{"infohash": "legacy", "series_id": 5, "title": "Old Show"}',),
        )
        raw.execute(
            "INSERT INTO pending_imports (arr, infohash, record) VALUES ('sonarr', 'tagged', jsonb(?))",
            ('{"infohash": "tagged", "series_id": 6, "al_id": 9}',),
        )
        raw.execute("PRAGMA user_version=1")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            pending = store.get_pending_import_rows(Arr.SONARR)
            assert set(pending) == {"legacy", "tagged"}
            # The sentinel-keyed legacy row rehydrates as a claim with al_id 0.
            legacy = PendingImport.from_json(pending["legacy"], guards={})
            assert claim_al_ids(legacy) == (0,)
            assert legacy.claims[0].title == "Old Show"
            tagged = PendingImport.from_json(pending["tagged"], guards={})
            assert claim_al_ids(tagged) == (9,)
            assert tagged.series_ids == (6,)
            # The merged rows behave exactly like modern rows: keyed by hash, dropped by hash.
            store.drop_pending_import_row(Arr.SONARR, "legacy")
            assert set(store.get_pending_import_rows(Arr.SONARR)) == {"tagged"}
        assert _user_version(db) == SCHEMA_VERSION

    def test_v2_guard_backfill_latest_added_at_wins(self, tmp_path: Path) -> None:
        # The v2 -> v3 step: two rows of one title carry divergent frozen guard
        # copies - the exact divergence the guard_facts row kills. The newest
        # blob's copy (by added_at) becomes the title's row, and every blob is
        # stripped of its guards key. (A guard-less row NEWER than a
        # guard-carrying sibling is chronologically unreachable in released dbs.)
        db = tmp_path / "cache.db"
        old = {
            "infohash": "h1",
            "al_id": 9,
            "added_at": "2026-07-01 00:00:00",
            "guards": {"entry_groups": ["Old"]},
        }
        new = {
            "infohash": "h2",
            "al_id": 9,
            "added_at": "2026-07-02 00:00:00",
            "guards": {"entry_groups": ["New"], "stale_groups": ["Old"], "owned_episodes": [[101, 700]]},
        }
        _seed_v2_db(db, [("sonarr", "h1", 9, json.dumps(old)), ("sonarr", "h2", 9, json.dumps(new))])

        with _open(tmp_path) as store:
            assert store.get_guards(Arr.SONARR) == {
                9: OwnedAtGrab(
                    entry_groups=("New",), stale_groups=("Old",), owned_episodes=(SizeMatchedEpisode(101, 700),)
                ),
            }
            # The divergent per-row copies are gone for good.
            assert all("guards" not in rec for rec in store.get_pending_import_rows(Arr.SONARR).values())
        assert _user_version(db) == SCHEMA_VERSION

    def test_v2_guard_backfill_skips_ineligible_records(self, tmp_path: Path) -> None:
        # No row may come from: the legacy al_id=0 sentinel, a radarr row, a
        # guards-less blob, or a JSON-null guards value (a null row would poison
        # from_json at every read). The stray keys are still stripped everywhere.
        db = tmp_path / "cache.db"
        guards = {"entry_groups": ["G"]}
        stamp = "2026-07-01 00:00:00"
        _seed_v2_db(
            db,
            [
                ("sonarr", "s0", 0, json.dumps({"al_id": 0, "added_at": stamp, "guards": guards})),
                ("radarr", "r1", 3, json.dumps({"al_id": 3, "added_at": stamp, "guards": guards})),
                ("sonarr", "g1", 4, json.dumps({"al_id": 4, "added_at": stamp})),
                ("sonarr", "n1", 5, json.dumps({"al_id": 5, "added_at": stamp, "guards": None})),
            ],
        )

        with _open(tmp_path) as store:
            assert store.get_guards(Arr.SONARR) == {}
            assert store.get_guards(Arr.RADARR) == {}
            stripped = [
                *store.get_pending_import_rows(Arr.SONARR).values(),
                *store.get_pending_import_rows(Arr.RADARR).values(),
            ]
            assert len(stripped) == 4
            assert all("guards" not in rec for rec in stripped)

    def test_v2_migration_leaves_existing_guard_rows_alone(self, tmp_path: Path) -> None:
        # The step's idempotence guard: a v2 db already carrying guard
        # rows is left exactly as found (no backfill overwrites them).
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V2_PENDING_SCHEMA)
        raw.execute(
            "CREATE TABLE guard_facts (arr TEXT NOT NULL, al_id INTEGER NOT NULL, "
            "record BLOB NOT NULL, PRIMARY KEY (arr, al_id))",
        )
        raw.execute(
            "INSERT INTO guard_facts (arr, al_id, record) VALUES ('sonarr', 9, jsonb(?))",
            (json.dumps({"entry_groups": ["Kept"]}),),
        )
        raw.execute(
            "INSERT INTO pending_imports (arr, infohash, al_id, record) VALUES ('sonarr', 'h', 9, jsonb(?))",
            (json.dumps({"al_id": 9, "added_at": "2026-07-05 00:00:00", "guards": {"entry_groups": ["Newer"]}}),),
        )
        raw.execute("PRAGMA user_version=2")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            assert store.get_guards(Arr.SONARR) == {9: OwnedAtGrab(entry_groups=("Kept",))}

    def test_v3_id_form_names_are_nulled(self, tmp_path: Path) -> None:
        # The v3 -> v4 step: a name stored as the id-form fallback (AniList
        # unreachable at write time) is nulled so a later cached read resolves it. A
        # real name and an absent one survive untouched.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V3_ENTRIES_SCHEMA)
        raw.executemany(
            "INSERT INTO entries (arr, al_id, name) VALUES (?, ?, ?)",
            [("sonarr", 1, "AniList #1"), ("sonarr", 2, "Frieren"), ("radarr", 3, None)],
        )
        raw.execute("PRAGMA user_version=3")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            assert _cached_title_name(store, 1) is None
            assert _cached_title_name(store, 2) == "Frieren"
            assert _cached_title_name(store, 3, Arr.RADARR) is None
        assert _user_version(db) == SCHEMA_VERSION

    def test_v4_rows_of_one_torrent_fold_into_one_record(self, tmp_path: Path) -> None:
        # The v4 -> v5 step: one torrent's per-title rows become ONE row whose
        # claims sit in al_id order. The rows group by case fold and the key +
        # blob hash come out lowercase.
        # The release facts are the first row's, the file maps union with the first
        # row winning a shared name, an exclusion another row mapped is dropped,
        # `added_at` is the oldest stamp, and `awaiting_cleanup` is set when every row had it.
        # The union keeps distinct file-name keys byte-exact (an NFC and an NFD
        # spelling stay two keys): `normalize_name` reconciles them at read time.
        db = tmp_path / "cache.db"
        first = _v4_blob("abcd", 11, excluded_files=["b.mkv", "y.mkv"], awaiting_cleanup=True)
        second = _v4_blob(
            "ABCD",
            22,
            title="Second",
            coverage="S02",
            url="u2",
            ordered_episode_ids=[201, 202],
            names={"series": "Series", "anilist": ["Second Alias"]},
            preowned_episode_ids=[201],
            slice_coverage="S02 E01-E02",
            added_at="2026-07-02 00:00:00",
            release_group="Other",
            is_dual_audio=True,
            seadex_files=["z.mkv"],
            release_sizes=[30],
            file_episode_map={"b.mkv": [202], "a.mkv": [999]},
            excluded_files=["x.mkv"],
            awaiting_cleanup=True,
        )
        # Inserted newest first: the merge orders claims by al_id, not by row order.
        _seed_v4_db(db, [("sonarr", "ABCD", 22, json.dumps(second)), ("sonarr", "abcd", 11, json.dumps(first))])

        with _open(tmp_path) as store:
            rows = store.get_pending_import_rows(Arr.SONARR)
            assert rows == {
                "abcd": {
                    "infohash": "abcd",
                    "release_group": "Grp",
                    "is_dual_audio": False,
                    "seadex_files": ["a.mkv", "b.mkv"],
                    "added_at": "2026-07-01 00:00:00",
                    "file_episode_map": {"a.mkv": [101], "b.mkv": [202]},
                    "claims": [
                        {
                            "al_id": 11,
                            "series_id": 5,
                            "title": "Entry 11",
                            "coverage": "S01",
                            "url": "https://releases.moe/11",
                            "ordered_episode_ids": [101],
                            "names": {"series": "Series", "anilist": ["Alias"]},
                            "preowned_episode_ids": [],
                            "slice_coverage": "S01 E01",
                        },
                        {
                            "al_id": 22,
                            "series_id": 5,
                            "title": "Second",
                            "coverage": "S02",
                            "url": "u2",
                            "ordered_episode_ids": [201, 202],
                            "names": {"series": "Series", "anilist": ["Second Alias"]},
                            "preowned_episode_ids": [201],
                            "slice_coverage": "S02 E01-E02",
                        },
                    ],
                    "excluded_files": ["y.mkv", "x.mkv"],
                    "release_sizes": [10, 20],
                    "awaiting_cleanup": True,
                },
            }
            _assert_round_trips(rows)
            assert _columns(db, "pending_imports") == ["arr", "infohash", "record"]
        assert _user_version(db) == SCHEMA_VERSION

    def test_v4_fold_needs_every_cleanup_flag_and_keeps_the_arrs_apart(self, tmp_path: Path) -> None:
        # One row not awaiting cleanup clears the merged `awaiting_cleanup`. The other arr's row on the
        # same hash stays its own row, and the guard rows survive the step
        # to be read through the merged claims (the orphan stays unread).
        db = tmp_path / "cache.db"
        _seed_v4_db(
            db,
            [
                ("sonarr", "h", 11, json.dumps(_v4_blob("h", 11, awaiting_cleanup=True))),
                ("sonarr", "h", 22, json.dumps(_v4_blob("h", 22, awaiting_cleanup=False))),
                ("radarr", "h", 0, json.dumps(_v4_blob("h", 0, series_id=0, awaiting_cleanup=True))),
            ],
            guards=[
                ("sonarr", 11, json.dumps({"entry_groups": ["A"]})),
                ("sonarr", 22, json.dumps({"entry_groups": ["B"]})),
                ("sonarr", 33, json.dumps({"entry_groups": ["Orphan"]})),
            ],
        )

        with _open(tmp_path) as store:
            sonarr = store.get_pending_import_rows(Arr.SONARR)
            radarr = store.get_pending_import_rows(Arr.RADARR)
            assert PendingImport.from_json(sonarr["h"], guards={}).awaiting_cleanup is False
            assert claim_al_ids(PendingImport.from_json(sonarr["h"], guards={})) == (11, 22)
            assert PendingImport.from_json(radarr["h"], guards={}).awaiting_cleanup is True
            assert claim_al_ids(PendingImport.from_json(radarr["h"], guards={})) == (0,)
            assert store.other_arr_holds(Arr.SONARR, "h") is True
            assert store.get_guards(Arr.SONARR) == {
                11: OwnedAtGrab(entry_groups=("A",)),
                22: OwnedAtGrab(entry_groups=("B",)),
            }
            _assert_round_trips(sonarr)
            _assert_round_trips(radarr)

    def test_v4_fold_takes_the_oldest_parseable_stamp(self, tmp_path: Path) -> None:
        # `added_at` is the oldest row stamp that parses: a junk stamp is skipped,
        # and a torrent whose rows all carry junk gets an empty one.
        db = tmp_path / "cache.db"
        _seed_v4_db(
            db,
            [
                ("sonarr", "k", 1, json.dumps(_v4_blob("k", 1, added_at="junk"))),
                ("sonarr", "k", 2, json.dumps(_v4_blob("k", 2, added_at="2026-07-03 00:00:00"))),
                ("sonarr", "k", 3, json.dumps(_v4_blob("k", 3, added_at="2026-07-02 00:00:00"))),
                ("sonarr", "j", 1, json.dumps(_v4_blob("j", 1, added_at="junk"))),
            ],
        )

        with _open(tmp_path) as store:
            rows = store.get_pending_import_rows(Arr.SONARR)
            assert rows["k"]["added_at"] == "2026-07-02 00:00:00"
            assert rows["j"]["added_at"] == ""
            _assert_round_trips(rows)

    def test_v4_legacy_ids_fold_into_the_claims_window(self, tmp_path: Path) -> None:
        # A row from before the ordered window (no `ordered_episode_ids`) claims
        # the ids its file map and legacy `episode_ids` list had, sorted, zeros
        # dropped. A row that carried its window keeps it as is, and the legacy
        # list reaches no row.
        db = tmp_path / "cache.db"
        legacy = {
            "infohash": "old",
            "al_id": 3,
            "series_id": 5,
            "added_at": "2026-07-01 00:00:00",
            "file_episode_map": {"a.mkv": [2, 0]},
            "episode_ids": [3, 1, 0],
        }
        windowed = _v4_blob("new", 4, ordered_episode_ids=[9, 8], episode_ids=[1])
        _seed_v4_db(db, [("sonarr", "old", 3, json.dumps(legacy)), ("sonarr", "new", 4, json.dumps(windowed))])

        with _open(tmp_path) as store:
            rows = store.get_pending_import_rows(Arr.SONARR)
            assert all("episode_ids" not in row for row in rows.values())
            old = PendingImport.from_json(rows["old"], guards={})
            assert old.claims[0].ordered_episode_ids == (1, 2, 3)
            assert old.claims[0].names == TitleNames()
            # The merge carries the file map verbatim and only derives the window from it. The pending import drops
            # the zero.
            assert rows["old"]["file_episode_map"] == {"a.mkv": [2, 0]}
            assert dict(old.file_episode_map) == {"a.mkv": (2,)}
            new = PendingImport.from_json(rows["new"], guards={})
            assert new.claims[0].ordered_episode_ids == (9, 8)
            _assert_round_trips({"new": rows["new"]})

    def test_v4_unscoped_row_stays_unscoped_and_carries_its_names_verbatim(self, tmp_path: Path) -> None:
        # A v4 claim with no window (an empty window, no legacy list) claims nothing off
        # its file map, and the stored names dict goes as is, never re-encoded through
        # the live class.
        db = tmp_path / "cache.db"
        no_window_blob = _v4_blob("u", 5, ordered_episode_ids=[], names={"series": "Series"})
        _seed_v4_db(db, [("sonarr", "u", 5, json.dumps(no_window_blob))])

        with _open(tmp_path) as store:
            rows = store.get_pending_import_rows(Arr.SONARR)
            (claim,) = rows["u"]["claims"]
            assert claim["ordered_episode_ids"] == []
            assert claim["names"] == {"series": "Series"}
            assert PendingImport.from_json(rows["u"], guards={}).claims[0].names == TitleNames(series="Series")

    def test_v4_migration_is_a_no_op_on_reopen(self, tmp_path: Path) -> None:
        # The migrated db reopens as found: same rows, no step announced, and
        # a v4 `user_version` over a table already in the v5 shape (no al_id column) is
        # set current without touching the rows.
        db = tmp_path / "cache.db"
        _seed_v4_db(db, [("sonarr", "h", 11, json.dumps(_v4_blob("h", 11)))])
        with _open(tmp_path) as migrated:
            rows = migrated.get_pending_import_rows(Arr.SONARR)
        assert _user_version(db) == SCHEMA_VERSION

        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default
        with _open(tmp_path) as reopened:
            assert reopened.get_pending_import_rows(Arr.SONARR) == rows
        assert recording.of_type(Diagnostic) == []

        folded = tmp_path / "folded.db"
        raw = sqlite3.connect(str(folded))
        raw.executescript(_V1_PENDING_SCHEMA)  # the (arr, infohash) PK: v5's shape too
        raw.execute(
            "INSERT INTO pending_imports (arr, infohash, record) VALUES ('sonarr', 'h', jsonb(?))",
            (json.dumps(rows["h"]),),
        )
        raw.execute("PRAGMA user_version=4")
        raw.commit()
        raw.close()
        with contextlib.closing(CacheStore.load(str(folded), config_hash=CONFIG_HASH)) as store:
            assert store.get_pending_import_rows(Arr.SONARR) == rows
        assert _user_version(folded) == SCHEMA_VERSION

    def test_v5_entries_gain_a_season_type_column_left_null(self, tmp_path: Path) -> None:
        # The v5 -> v6 step: a cached title keeps its columns and reads no season type, and the
        # `tvdb_orders` table the step relies on `_SCHEMA` for is there to write.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V3_ENTRIES_SCHEMA)  # entries kept the v3 shape through v5
        raw.execute("INSERT INTO entries (arr, al_id, name) VALUES ('sonarr', 2, 'Frieren')")
        raw.execute("PRAGMA user_version=5")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            cached = store.get_cached_title(Arr.SONARR, 2)
            assert cached is not None
            assert (cached.name, cached.season_type) == ("Frieren", None)
            store.put_tvdb_orders(1, _ORDERS)
            assert store.get_tvdb_orders(1) == _ORDERS
        assert _user_version(db) == SCHEMA_VERSION

    def test_v6_torrent_hashes_table_is_dropped(self, tmp_path: Path) -> None:
        # The v6 -> v7 step: the table goes, and the cached title its rows hung off stays.
        db = tmp_path / "cache.db"
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 2, {"name": "Frieren", "url": "u"})
            store.save(dry_run=False)
        raw = sqlite3.connect(str(db))
        raw.executescript(_V6_TORRENT_HASHES_SCHEMA)
        raw.execute("INSERT INTO torrent_hashes (arr, al_id, infohash) VALUES ('sonarr', 2, 'abc')")
        raw.execute("PRAGMA user_version=6")
        raw.commit()
        raw.close()

        with _open(tmp_path) as store:
            cached = store.get_cached_title(Arr.SONARR, 2)
            assert cached is not None
            assert (cached.name, cached.url) == ("Frieren", "u")
        assert _user_version(db) == SCHEMA_VERSION
        assert _columns(db, "torrent_hashes") == []

    def test_fresh_db_starts_current_without_an_al_id_column(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        with _open(tmp_path) as store:
            store.save(dry_run=False)
        assert _user_version(db) == SCHEMA_VERSION
        assert _columns(db, "pending_imports") == ["arr", "infohash", "record"]

    def test_newer_schema_is_refused_not_quarantined(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
        raw.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
        raw.commit()
        raw.close()

        with pytest.raises(CacheSchemaError):
            CacheStore.load(str(db), config_hash=CONFIG_HASH)
        # Fail closed: the healthy newer db is left exactly where it was.
        assert db.exists()
        assert not list(tmp_path.glob("cache.db.corrupt-*"))


class TestEntries:
    """Cached title columns merge partially on update, format and clear timestamps, and stay isolated per arr."""

    def test_update_cache_is_a_partial_merge(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "Title", "url": "u"})
            # A later update with only one field must not wipe the others.
            store.update_cached_title(Arr.SONARR, 7, {"coverage": "S01"})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert (cached.name, cached.url, cached.coverage) == ("Title", "u", "S01")

    def test_clear_entry_stamp_clears_the_timestamp_and_nothing_else(self, tmp_path: Path) -> None:
        stamped: CachedTitleUpdate = {"name": "Title", "url": "u", "updated_at": datetime(2021, 6, 5)}
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, stamped)
            store.update_cached_title(Arr.RADARR, 7, stamped)

            store.uncache_title(Arr.SONARR, 7)

            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert (cached.updated_at, cached.name, cached.url) == (None, "Title", "u")
            # The other arr's row for the same id keeps its timestamp.
            other = store.get_cached_title(Arr.RADARR, 7)
            assert other is not None
            assert other.updated_at == "2021-06-05 00:00:00"

    def test_clear_entry_stamp_adds_no_row(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.uncache_title(Arr.SONARR, 7)
            assert store.get_cached_title(Arr.SONARR, 7) is None

    def test_update_cache_formats_datetime_timestamp(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.updated_at == "2021-06-05 04:03:02"

    def test_arrs_are_isolated(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "S"})
            store.update_cached_title(Arr.RADARR, 7, {"name": "R"})
            assert _cached_title_name(store, 7, Arr.SONARR) == "S"
            assert _cached_title_name(store, 7, Arr.RADARR) == "R"

    def test_get_entry_reads_all_columns_at_once(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(
                Arr.SONARR,
                7,
                {"name": "Title", "url": "u", "coverage": "S01", "updated_at": datetime(2021, 6, 5, 4, 3, 2)},
            )
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert (cached.name, cached.url, cached.coverage, cached.updated_at) == (
                "Title",
                "u",
                "S01",
                "2021-06-05 04:03:02",
            )
            # A missing row reads back as None, not an all-None cached title.
            assert store.get_cached_title(Arr.SONARR, 999) is None


class TestFallbackSatisfied:
    """`fallback_satisfied` defaults false, round-trips as a real bool, and survives partial updates that omit it."""

    def test_defaults_false_and_roundtrips_as_bool(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            # A row written without the key reads back False (NOT NULL DEFAULT 0)...
            store.update_cached_title(Arr.SONARR, 7, {"name": "Title"})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.fallback_satisfied is False
            # ...and a written True reads back as a real bool, not the stored int.
            store.update_cached_title(Arr.SONARR, 7, {"fallback_satisfied": True})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.fallback_satisfied is True

    def test_persists_across_save_and_reopen(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"fallback_satisfied": True})
            store.save(dry_run=False)

        with _open(tmp_path) as reopened:
            cached = reopened.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.fallback_satisfied is True

    def test_partial_update_preserves_until_rewritten(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"fallback_satisfied": True})
            # A partial merge that omits the key must not clear the marker...
            store.update_cached_title(Arr.SONARR, 7, {"coverage": "S01"})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.fallback_satisfied is True
            # ...but a supplied False overwrites it.
            store.update_cached_title(Arr.SONARR, 7, {"fallback_satisfied": False})
            cached = store.get_cached_title(Arr.SONARR, 7)
            assert cached is not None
            assert cached.fallback_satisfied is False


class TestPreviewGate:
    """The staged-write dry-run gate: a dry-run save never persists, only a real save does."""

    def test_preview_save_on_missing_file_writes_nothing(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "Title"})
            store.save(dry_run=True)
        assert not (tmp_path / "cache.db").exists()

    def test_real_save_on_missing_file_creates_and_persists(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "Title"})
            store.save(dry_run=False)

        assert (tmp_path / "cache.db").exists()
        with _open(tmp_path) as reopened:
            assert _cached_title_name(reopened, 7) == "Title"

    def test_preview_on_existing_db_does_not_mutate_committed_state(self, tmp_path: Path) -> None:
        # Establish a real, committed db.
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "Original"})
            store.save(dry_run=False)

        # A dry run stages changes but must persist none of them.
        with _open(tmp_path) as dry_run_store:
            dry_run_store.update_cached_title(Arr.SONARR, 7, {"name": "Changed"})
            dry_run_store.update_cached_title(Arr.SONARR, 8, {"name": "New"})
            dry_run_store.save(dry_run=True)

        with _open(tmp_path) as reopened:
            assert _cached_title_name(reopened, 7) == "Original"  # not "Changed"
            assert reopened.get_cached_title(Arr.SONARR, 8) is None  # never added


class TestSelectionDigest:
    """`selection_changed`/`store_selection_hash`: missing-key-fresh, per-arr, commit-gated."""

    def test_no_digest_on_record_reads_fresh(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            assert store.selection_changed(Arr.SONARR, "d1") is False

    def test_vouch_round_trips_through_a_real_save(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.store_selection_hash(Arr.SONARR, "d1")
            store.save(dry_run=False)

        with _open(tmp_path) as reopened:
            assert reopened.selection_changed(Arr.SONARR, "d1") is False
            assert reopened.selection_changed(Arr.SONARR, "d2") is True

    def test_preview_save_does_not_consume_a_pending_recheck(self, tmp_path: Path) -> None:
        # Matching prefs changed (the committed selection hash is old). A dry run
        # stages the new one, but its rollback must leave the re-check due
        # on the next real run.
        with _open(tmp_path) as store:
            store.store_selection_hash(Arr.SONARR, "old")
            store.save(dry_run=False)

        with _open(tmp_path) as previewing:
            assert previewing.selection_changed(Arr.SONARR, "new") is True
            previewing.store_selection_hash(Arr.SONARR, "new")
            previewing.save(dry_run=True)

        with _open(tmp_path) as real:
            assert real.selection_changed(Arr.SONARR, "new") is True

    def test_arrs_vouch_independently(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.store_selection_hash(Arr.SONARR, "d1")
            assert store.selection_changed(Arr.SONARR, "d2") is True
            assert store.selection_changed(Arr.RADARR, "d2") is False


class TestAnilistMeta:
    """AniList cached answers round-trip by id, are iterable, and a later `put` overwrites the prior answer."""

    def test_roundtrip_get_and_iter(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            rec = {"fetched_at": "2026-06-20 12:00:00", "data": {"Media": {"id": 1}}}
            store.put_anilist_meta(1, rec)
            assert store.get_anilist_meta(1) == rec
            assert store.get_anilist_meta(999) is None
            assert dict(store.iter_anilist_meta()) == {1: rec}

    def test_put_overwrites(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.put_anilist_meta(1, {"fetched_at": "2026-06-20 12:00:00", "data": {"a": 1}})
            store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {"a": 2}})
            meta = store.get_anilist_meta(1)
            assert meta is not None
            assert meta["data"] == {"a": 2}


class TestRecordFreshness:
    """The stored-row helpers: the payload half and the stamp half, split apart."""

    def test_record_payload_wants_a_non_empty_object(self) -> None:
        assert row_payload(None, "data") is None
        assert row_payload({"fetched_at": "2026-06-26 12:00:00"}, "data") is None
        assert row_payload({"data": {}}, "data") is None
        assert row_payload({"data": [{"x": 1}]}, "data") is None
        assert row_payload({"data": {"x": 1}}, "data") == {"x": 1}

    def test_stamp_is_fresh_parses_and_compares(self) -> None:
        cutoff = datetime(2026, 6, 20)
        assert stamp_is_fresh({"fetched_at": "2026-06-26 12:00:00"}, cutoff) is True
        assert stamp_is_fresh({"fetched_at": "2020-01-01 00:00:00"}, cutoff) is False
        assert stamp_is_fresh({}, cutoff) is False
        assert stamp_is_fresh({"fetched_at": 5}, cutoff) is False

    def test_parse_stamp_or_none_swallows_junk_and_non_strings(self) -> None:
        assert parse_stamp_or_none("2026-06-26 12:00:00") == datetime(2026, 6, 26, 12)
        assert parse_stamp_or_none("junk") is None
        assert parse_stamp_or_none("") is None
        # A raw row's stamp may not even be a string: the helper answers None, never raises.
        raw: dict[str, Any] = {"added_at": 5}
        assert parse_stamp_or_none(raw["added_at"]) is None


class TestSonarrParse:
    """Cached Sonarr parses round-trip whole keyed by file name, and a missing file name reads back None."""

    def test_roundtrip_get(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            info = ParsedFileInfo(season_number=1, episode_numbers=(2,))
            rec = {"fetched_at": "2026-06-20 12:00:00", "parse": parse_to_json(info)}
            store.put_sonarr_parse("file.mkv", rec)
            read_back = store.get_sonarr_parse("file.mkv")
            assert read_back == rec
            assert read_back is not None
            assert parse_from_json(read_back) == info
            assert store.get_sonarr_parse("missing.mkv") is None

    def test_series_fingerprint_key_round_trips(self, tmp_path: Path) -> None:
        # The unmatched parse's pin: the freshness check dispatches on its presence.
        with _open(tmp_path) as store:
            rec = {
                "fetched_at": "2026-06-20 12:00:00",
                "parse": parse_to_json(ParsedFileInfo()),
                "series_fp": "fp",
            }
            store.put_sonarr_parse("unmatched.mkv", rec)
            assert store.get_sonarr_parse("unmatched.mkv") == rec

    def test_legacy_record_reads_back_unreadable(self, tmp_path: Path) -> None:
        # A pre-existing row carries an episode list, not a whole parse: it reads
        # back verbatim and the parse check rejects it, so the file re-parses.
        with _open(tmp_path) as store:
            rec = {"fetched_at": "2026-06-20 12:00:00", "episodes": [{"season": 1, "episode": 1}]}
            store.put_sonarr_parse("legacy.mkv", rec)
            read_back = store.get_sonarr_parse("legacy.mkv")
            assert read_back == rec
            assert read_back is not None
            assert parse_from_json(read_back) is None


class TestPendingImports:
    """Pending imports are tracked per (arr, infohash), read keyed or by any claim's series id, and droppable."""

    def test_roundtrip_drop_and_arr_isolation(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            rec = _row(pending_import(infohash="h1", al_id=3, series_id=5))
            store.put_pending_import_row(Arr.SONARR, "h1", rec)
            store.put_pending_import_row(Arr.RADARR, "h2", {"infohash": "h2"})
            assert store.get_pending_import_rows(Arr.SONARR) == {"h1": rec}
            assert store.get_pending_import_rows(Arr.RADARR) == {"h2": {"infohash": "h2"}}
            # The keyed reads see only their own arr's row.
            assert store.get_pending_import_row(Arr.SONARR, "h1") == rec
            assert store.get_pending_import_row(Arr.RADARR, "h1") is None

            store.drop_pending_import_row(Arr.SONARR, "h1")
            assert store.get_pending_import_rows(Arr.SONARR) == {}
            assert store.get_pending_import_row(Arr.SONARR, "h1") is None
            # Dropping a missing key is a no-op.
            store.drop_pending_import_row(Arr.SONARR, "nope")

    def test_one_record_per_torrent_replaced_whole(self, tmp_path: Path) -> None:
        # One torrent listed on two titles is ONE row carrying both
        # claims: a re-put under the hash replaces the row whole (never a
        # second row), and the drop takes every claim with it.
        with _open(tmp_path) as store:
            store.put_pending_import_row(Arr.SONARR, "h", _row(pending_import(infohash="h", al_id=11, series_id=5)))
            both = _row(
                pending_import(
                    infohash="h",
                    claims=(title_claim(al_id=11, series_id=5), title_claim(al_id=22, series_id=5)),
                )
            )
            store.put_pending_import_row(Arr.SONARR, "h", both)
            assert store.get_pending_import_rows(Arr.SONARR) == {"h": both}
            assert store.stats().pending_imports == 1

            store.drop_pending_import_row(Arr.SONARR, "h")
            assert store.get_pending_import_rows(Arr.SONARR) == {}

    def test_other_arr_holds_matches_the_key_exactly(self, tmp_path: Path) -> None:
        # Every row key is the lowercase infohash, so the other arr's row is
        # found by exact key and never by case fold.
        with _open(tmp_path) as store:
            store.put_pending_import_row(Arr.RADARR, "abcd", {"infohash": "abcd"})

            assert store.other_arr_holds(Arr.SONARR, "abcd") is True
            assert store.other_arr_holds(Arr.SONARR, "ABCD") is False
            assert store.other_arr_holds(Arr.RADARR, "abcd") is False

    def test_other_arr_holds_ignores_the_arrs_own_record(self, tmp_path: Path) -> None:
        # The category gate asks whether the OTHER arr still claims the torrent:
        # the same arr's row never counts, a missing hash reads False, and the
        # other arr's row flips the answer for as long as it lives.
        with _open(tmp_path) as store:
            store.put_pending_import_row(Arr.SONARR, "h", pending_import(infohash="h", al_id=11).to_json())

            assert store.other_arr_holds(Arr.SONARR, "h") is False
            assert store.other_arr_holds(Arr.RADARR, "h") is True
            assert store.other_arr_holds(Arr.SONARR, "other") is False

            store.put_pending_import_row(Arr.RADARR, "h", {"infohash": "h"})
            assert store.other_arr_holds(Arr.SONARR, "h") is True
            store.drop_pending_import_row(Arr.RADARR, "h")
            assert store.other_arr_holds(Arr.SONARR, "h") is False

    def test_get_pending_for_series_matches_any_claim_in_sql(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            a = _row(pending_import(infohash="a", al_id=1, series_id=5))
            b = _row(pending_import(infohash="b", al_id=2, series_id=5))
            c = _row(pending_import(infohash="c", al_id=3, series_id=9))
            # A two-claim row spanning both series answers to either.
            both = _row(
                pending_import(
                    infohash="t",
                    claims=(title_claim(al_id=6, series_id=5), title_claim(al_id=7, series_id=9)),
                )
            )
            store.put_pending_import_row(Arr.SONARR, "a", a)
            store.put_pending_import_row(Arr.SONARR, "b", b)
            store.put_pending_import_row(Arr.SONARR, "c", c)
            store.put_pending_import_row(Arr.SONARR, "t", both)
            # A claim with no series id, and a row with no claims: never matched.
            store.put_pending_import_row(Arr.SONARR, "d", {"infohash": "d", "claims": [{"al_id": 4}]})
            store.put_pending_import_row(Arr.SONARR, "e", {"infohash": "e"})

            # Only rows with a claim on the series come back. The integer series_id binds directly.
            assert store.get_pending_import_rows_for_series(Arr.SONARR, 5) == {"a": a, "b": b, "t": both}
            assert store.get_pending_import_rows_for_series(Arr.SONARR, 9) == {"c": c, "t": both}
            assert store.get_pending_import_rows_for_series(Arr.SONARR, 404) == {}

            # Fresh per call: a drop is reflected immediately (no copy kept from an earlier call).
            store.drop_pending_import_row(Arr.SONARR, "a")
            assert store.get_pending_import_rows_for_series(Arr.SONARR, 5) == {"b": b, "t": both}


class TestGuardFacts:
    """One guard-evidence row per (arr, al_id), read-filtered to titles with a live claim."""

    def test_roundtrip_and_arr_isolation(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.put_pending_import_row(Arr.SONARR, "hs", pending_import(infohash="hs", al_id=7).to_json())
            store.put_pending_import_row(
                Arr.RADARR, "hr", pending_import(infohash="hr", al_id=7, series_id=0).to_json()
            )
            facts = OwnedAtGrab(
                entry_groups=("SubsPlease", "Erai-raws"),
                stale_groups=("HorribleSubs",),
                owned_episodes=(SizeMatchedEpisode(101, 700_000_000), SizeMatchedEpisode(102, 650_000_000)),
            )
            store.put_guards(Arr.SONARR, 7, facts)
            store.put_guards(Arr.RADARR, 7, OwnedAtGrab(entry_groups=("Beatrice-Raws",)))

            # Typed on both ends: owned_episodes come back as SizeMatchedEpisode tuples.
            assert store.get_guards(Arr.SONARR) == {7: facts}
            assert store.get_guards(Arr.RADARR) == {7: OwnedAtGrab(entry_groups=("Beatrice-Raws",))}

    def test_latest_put_wins(self, tmp_path: Path) -> None:
        # The fix's write semantic: a re-put overwrites the title's single row
        # whole, so no two reads can ever see divergent evidence for one title.
        with _open(tmp_path) as store:
            store.put_pending_import_row(Arr.SONARR, "h", pending_import(infohash="h", al_id=7).to_json())
            store.put_guards(Arr.SONARR, 7, OwnedAtGrab(entry_groups=("Old",)))
            newest = OwnedAtGrab(entry_groups=("New",), stale_groups=("Old",))
            store.put_guards(Arr.SONARR, 7, newest)
            assert store.get_guards(Arr.SONARR) == {7: newest}

    def test_orphan_rows_are_stored_but_never_read(self, tmp_path: Path) -> None:
        # There is deliberately no delete path (a title's claims on several
        # torrents share the row), so reads join live claims: the read stays
        # bounded by in-flight work, not all-time grab history, and an orphan
        # row is invisible until the title gets a claim again.
        with _open(tmp_path) as store:
            facts = OwnedAtGrab(entry_groups=("SubGroup",))
            store.put_guards(Arr.SONARR, 7, facts)
            assert store.get_guards(Arr.SONARR) == {}

            store.put_pending_import_row(Arr.SONARR, "h", pending_import(infohash="h", al_id=7).to_json())
            assert store.get_guards(Arr.SONARR) == {7: facts}

            # The last pending import dropping re-orphans the row: stored, no longer read.
            store.drop_pending_import_row(Arr.SONARR, "h")
            assert store.get_guards(Arr.SONARR) == {}
            assert store.stats().guard_facts == 1

    def test_reads_through_every_claim_of_a_record(self, tmp_path: Path) -> None:
        # The join walks every claim on a live pending import: both titles sharing one
        # torrent get their evidence back, a title claiming nothing stays unread.
        with _open(tmp_path) as store:
            seven = OwnedAtGrab(entry_groups=("A",))
            eight = OwnedAtGrab(entry_groups=("B",))
            store.put_guards(Arr.SONARR, 7, seven)
            store.put_guards(Arr.SONARR, 8, eight)
            store.put_guards(Arr.SONARR, 9, OwnedAtGrab(entry_groups=("C",)))
            shared = pending_import(infohash="h", claims=(title_claim(al_id=7), title_claim(al_id=8, series_id=8)))
            store.put_pending_import_row(Arr.SONARR, "h", shared.to_json())

            assert store.get_guards(Arr.SONARR) == {7: seven, 8: eight}

    def test_empty_store_reads_empty(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            assert store.get_guards(Arr.SONARR) == {}


_ORDERS = CachedOrders(
    fetched_at="2026-07-01 10:00:00",
    fresh_until="2026-07-08 10:00:00",
    orders={"official": (OrderRow(1, 1, 101, 1), OrderRow(1, 2, 102, 2)), "dvd": ()},
    sonarr_counts={"official": 2, "alternate": 2},
)


class TestTvdbOrders:
    """One row per series, replaced whole, and a row that no longer reads counts as uncached."""

    def test_a_record_round_trips_and_a_put_replaces_it(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            assert store.get_tvdb_orders(1) is None

            store.put_tvdb_orders(1, CachedOrders(fetched_at="x", fresh_until="x", orders={}, sonarr_counts=None))
            store.put_tvdb_orders(1, _ORDERS)

            assert store.get_tvdb_orders(1) == _ORDERS
            assert store.stats().tvdb_orders == 1

    def test_a_record_of_another_shape_reads_as_none(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        with _open(tmp_path) as store:
            store.save(dry_run=False)
        raw = sqlite3.connect(str(db))
        raw.execute("INSERT INTO tvdb_orders (tvdb_id, record) VALUES (1, jsonb(?))", (json.dumps({"orders": []}),))
        raw.commit()
        raw.close()

        with _open(tmp_path) as reopened:
            assert reopened.get_tvdb_orders(1) is None


class TestHistoryCheckpoints:
    """History checkpoints upsert per arr and respect the dry-run gate."""

    def test_roundtrip_upsert_and_arr_isolation(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            assert store.get_history_checkpoint(Arr.SONARR) is None

            store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T10:00:00Z", 12))
            store.put_history_checkpoint(Arr.RADARR, HistoryCheckpoint("2026-07-02T10:00:00Z", 3))
            assert store.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-01T10:00:00Z", 12)
            assert store.get_history_checkpoint(Arr.RADARR) == HistoryCheckpoint("2026-07-02T10:00:00Z", 3)

            # Upsert: a later advance replaces the arr's single row.
            store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-03T10:00:00Z", 40))
            assert store.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-03T10:00:00Z", 40)

    def test_preview_save_does_not_persist_checkpoint(self, tmp_path: Path) -> None:
        # The dry-run gate: a previewed run must never advance the cursor.
        db = tmp_path / "cache.db"
        with _open(tmp_path) as store:
            store.save(dry_run=False)  # promote so the dry run below has a real file
            store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T10:00:00Z", 12))
            store.save(dry_run=True)

        assert db.exists()
        with _open(tmp_path) as reopened:
            assert reopened.get_history_checkpoint(Arr.SONARR) is None


class TestPromoteFailure:
    """A failed atomic promote leaves no partial `cache.db` behind, and a later save still promotes for real."""

    def test_failed_promote_leaves_no_partial_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the promote rename fails (disk full / killed mid-copy), it must leave NO
        # partial cache.db - else the next run mistakes the torn file for a real
        # (empty) cache.
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "T"})
            # Scope the patch to this save only: the later `again.save` must promote for real.
            with monkeypatch.context() as mp, contextlib.suppress(OSError):
                mp.setattr("pearlarr.cache.os.replace", _raise_os_replace)
                store.save(dry_run=False)

        assert not (tmp_path / "cache.db").exists()  # no 0-byte orphan
        assert not list(tmp_path.glob("cache.db.promote*"))  # temp cleaned up

        # A fresh run promotes for real (the failed attempt left nothing torn behind).
        with _open(tmp_path) as again:
            again.update_cached_title(Arr.SONARR, 7, {"name": "T"})
            again.save(dry_run=False)
        assert (tmp_path / "cache.db").exists()

        with _open(tmp_path) as reopened:
            assert _cached_title_name(reopened, 7) == "T"


class TestRunLifecycle:
    """Replays the order the run loop drives a real `CacheStore` through.

    The run-loop tests mock `cache_store`, so this is the only check that the real
    load -> writes -> save(commit) -> close(rollback) -> reopen sequence behaves.
    """

    def test_run_call_order_persists_and_reloads(self, tmp_path: Path) -> None:
        # Run 1 (real): check one title the way the loop does, then commit + close.
        with _open(tmp_path) as store:
            store.update_cached_title(
                Arr.SONARR,
                7,
                {
                    "name": "Show",
                    "url": "u",
                    "coverage": "S01",
                    "updated_at": datetime(2026, 1, 2, 3, 4, 5),
                },
            )
            store.put_anilist_meta(7, {"fetched_at": "2026-06-26 12:00:00", "data": {"id": 7}})
            carried = _row(pending_import(infohash="aaa", al_id=7, series_id=5))
            store.put_pending_import_row(Arr.SONARR, "aaa", carried)
            store.save(dry_run=False)  # the run's commit, so the close's rollback has nothing to undo

        # Run 2: reopen -> cache hit, carried-over pending.
        with _open(tmp_path) as again:
            stamped_under = make_entry_record(updated_at=datetime(2026, 1, 2, 3, 4, 5))
            assert cached_as_done(again.get_cached_title(Arr.SONARR, 7), stamped_under)
            assert again.get_pending_import_rows(Arr.SONARR) == {"aaa": carried}
            # A completed import is dropped, and that drop persists across a save.
            again.drop_pending_import_row(Arr.SONARR, "aaa")
            again.save(dry_run=False)

        with _open(tmp_path) as final:
            assert final.get_pending_import_rows(Arr.SONARR) == {}


class TestMaintenance:
    """Eviction drops only stale or stampless answers and keeps fresh ones. Stats/integrity report accurate counts."""

    def test_evict_anilist_meta_drops_only_stale(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.put_anilist_meta(1, {"fetched_at": "2020-01-01 00:00:00", "data": {"x": 1}})
            store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
            assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1
            assert store.get_anilist_meta(1) is None
            assert store.get_anilist_meta(2) is not None

    def test_evict_sonarr_parse_drops_only_stale(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            parse = parse_to_json(ParsedFileInfo(season_number=1, episode_numbers=(1,)))
            store.put_sonarr_parse("old.mkv", {"fetched_at": "2020-01-01 00:00:00", "parse": parse})
            store.put_sonarr_parse("new.mkv", {"fetched_at": "2026-06-26 12:00:00", "parse": parse})
            assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
            assert store.get_sonarr_parse("old.mkv") is None
            assert store.get_sonarr_parse("new.mkv") is not None

    def test_evict_sweeps_stampless_records(self, tmp_path: Path) -> None:
        # An answer with no fetched_at -> NULL generated column. It is unreadable
        # (stamp_is_fresh rejects it) and must not become un-evictable dead weight.
        with _open(tmp_path) as store:
            store.put_anilist_meta(1, {"data": {"x": 1}})  # no fetched_at -> NULL
            store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
            assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1  # only the stampless
            assert store.get_anilist_meta(1) is None
            assert store.get_anilist_meta(2) is not None  # fresh, stamped -> kept
            store.put_sonarr_parse("nostamp.mkv", {"parse": {}})  # no fetched_at -> NULL
            assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
            assert store.get_sonarr_parse("nostamp.mkv") is None

    def test_stats_and_integrity(self, tmp_path: Path) -> None:
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "T"})
            store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {}})
            store.save(dry_run=False)  # promote to file so size_bytes > 0

            s = store.stats()
            assert s.entries == 1
            assert s.anilist_meta == 1
            assert s.sonarr_parse == 0
            assert s.pending_imports == 0
            assert s.size_bytes > 0
            assert store.integrity_check() == "ok"

    def test_stats_count_a_table_the_file_predates_as_empty(self, tmp_path: Path) -> None:
        # `cache stats` opens the file read-only, so a file no run has migrated yet lacks the newer tables.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V3_ENTRIES_SCHEMA)
        raw.execute("PRAGMA user_version=5")
        raw.commit()
        raw.close()

        with contextlib.closing(CacheStore.open_readonly(str(db))) as store:
            assert store.stats().tvdb_orders == 0


class TestCorruptStore:
    """Corruption detection fires only on real corruption, and only genuinely corrupt dbs get quarantined."""

    def test_is_corruption_distinguishes_corrupt_from_transient(self) -> None:
        # Quarantine wipes the db, so it must fire ONLY on real corruption - never on
        # a transient lock/IO error (which would destroy a healthy cache on a fluke).
        assert is_corruption(sqlite3.DatabaseError("file is not a database")) is True
        assert is_corruption(sqlite3.DatabaseError("database disk image is malformed")) is True
        assert is_corruption(sqlite3.OperationalError("database is locked")) is False
        assert is_corruption(sqlite3.OperationalError("disk I/O error")) is False

    def test_locked_healthy_db_is_not_quarantined(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A healthy db whose load() hits a non-corruption DatabaseError must NOT be
        # quarantined: the error propagates (fail closed) and the file is untouched.
        with _open(tmp_path) as store:
            store.update_cached_title(Arr.SONARR, 7, {"name": "Keep"})
            store.save(dry_run=False)
        db = tmp_path / "cache.db"

        # Simulate a transient lock at open time (e.g. the WAL switch hitting BUSY).
        with monkeypatch.context() as mp:
            mp.setattr("pearlarr.sqlite_util.connect", _raise_locked)
            raised = False
            try:
                CacheStore.load(str(db), config_hash=CONFIG_HASH)
            except sqlite3.OperationalError:
                raised = True

        assert raised  # propagated, not swallowed into a quarantine
        assert db.exists()  # healthy db left in place
        assert not list(tmp_path.glob("cache.db.corrupt-*"))  # never quarantined

    def test_corrupt_db_is_quarantined_and_recovered(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        db.write_text("this is not a sqlite database")  # torn-write stand-in
        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default

        # Must NOT raise - fail open to a fresh store.
        with _open(tmp_path) as store:
            assert store.get_cached_title(Arr.SONARR, 7) is None
            store.update_cached_title(Arr.SONARR, 7, {"name": "Recovered"})
            store.save(dry_run=False)

        # The corrupt file was moved aside. A fresh, working db took its place.
        assert len(list(tmp_path.glob("cache.db.corrupt-*"))) == 1
        with _open(tmp_path) as reopened:
            assert _cached_title_name(reopened, 7) == "Recovered"

        # The recovery notice names the state that was lost, not just "fresh cache".
        [notice] = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert "moved it to" in notice.message
        assert notice.message.endswith(
            "started a fresh cache (titles will be re-checked; "
            "pending-import tracking reset, so recent grabs may be re-offered)",
        )
