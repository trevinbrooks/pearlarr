# pyright: strict
"""Behavioral tests for the SQLite-backed `CacheStore`.

Pins behavior, not internals: the per-entry records + torrent-hash child rows,
the JSONB meta/parse caches, pending imports, the descriptor, and - critically -
the staged-write / preview gate (a preview never persists, a real save does) and
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
from pearlarr.cache import (
    SCHEMA_VERSION,
    CacheSchemaError,
    CacheStore,
    HistoryCheckpoint,
    record_payload,
    stamp_is_fresh,
)
from pearlarr.config import Arr
from pearlarr.log import LOG_NAME
from pearlarr.manual_import import EntryNames, GuardFacts, OwnedEpisode, PendingImport, newest_claimed_at_of
from pearlarr.output import Diagnostic, Severity, install_hub
from pearlarr.output.recording import RecordingHub
from pearlarr.parse_records import parsed_info, to_parse_record
from pearlarr.seadex_types import ParsedFileInfo
from pearlarr.sqlite_util import is_corruption
from pearlarr.stamps import parse_stamp_or_none

from .builders import claim_al_ids, entry_claim, make_entry_record, pending_import

# Stand-in for a config-file checksum. `CacheStore` only stamps and compares the
# value it is handed. It never computes one, so any string works here.
CHECKSUM = "0123456789abcdef0123456789abcdef"


def _entry_name(store: CacheStore, al_id: int, arr: Arr = Arr.SONARR) -> str | None:
    """The entry's stored name via `get_entry` (None when the row is absent)."""

    entry = store.get_entry(arr, al_id)
    return None if entry is None else entry.name


def _raise_os_replace(*_args: object, **_kwargs: object) -> None:
    """A drop-in for `os.replace` that fails the atomic promote rename."""

    raise OSError("boom")


def _raise_locked(*_args: object, **_kwargs: object) -> sqlite3.Connection:
    """A drop-in for `_connect` that simulates a transient open-time lock."""

    raise sqlite3.OperationalError("database is locked")


def _row(record: PendingImport) -> dict[str, Any]:
    """The record as the store's JSON column reads it back (tuples as lists)."""

    return json.loads(json.dumps(record.to_json()))


def _open(tmp_path: Path) -> CacheStore:
    return CacheStore.load(str(tmp_path / "cache.db"), config_checksum=CHECKSUM)


class TestSchemaAndDescriptor:
    """A missing db opens in-memory and reads empty. A saved db persists the version/checksum descriptor."""

    def test_missing_file_opens_in_memory(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        # Nothing on disk yet, and reads work against the empty in-memory schema.
        assert not (tmp_path / "cache.db").exists()
        assert store.get_entry(Arr.SONARR, 7) is None
        store.close()

    def test_descriptor_persists_version_and_checksum(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.save(preview=False)
        store.close()

        # Read kv with a RAW connection. CacheStore.load re-stamps the descriptor on
        # open (via _reconcile), so reopening through the facade would read back the
        # just-re-stamped constants and pass even if save() never persisted them.
        db = tmp_path / "cache.db"
        assert db.exists()
        raw = sqlite3.connect(str(db))
        try:
            rows = dict(raw.execute("SELECT key, value FROM kv").fetchall())
        finally:
            raw.close()
        assert rows.get("pearlarr_version") == pearlarr.__version__
        assert rows.get("config_checksum") == CHECKSUM


# The original (pre-versioning) shape of the per-entry tables: `entries` without
# `fallback_satisfied`, stamped `user_version` 0 by default. Pins the v0 -> v1
# upgrade path (the shape a first-release cache.db still has on disk).
_V0_SCHEMA = """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL,
    name TEXT, url TEXT, coverage TEXT, updated_at TEXT,
    PRIMARY KEY (arr, al_id));
CREATE TABLE torrent_hashes (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL, infohash TEXT NOT NULL,
    PRIMARY KEY (arr, al_id, infohash),
    FOREIGN KEY (arr, al_id) REFERENCES entries (arr, al_id) ON DELETE CASCADE);
"""


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
# the v3 -> v4 step rewrites (a name may hold the id form an outage once stored).
_V3_ENTRIES_SCHEMA = """
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries (
    arr TEXT NOT NULL, al_id INTEGER NOT NULL,
    name TEXT, url TEXT, coverage TEXT, updated_at TEXT,
    fallback_satisfied INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (arr, al_id));
"""


def _seed_v2_db(db: Path, rows: list[tuple[str, str, int, str]]) -> None:
    """A v2-stamped db whose `pending_imports` holds the given (arr, infohash, al_id, json) rows."""

    raw = sqlite3.connect(str(db))
    raw.executescript(_V2_PENDING_SCHEMA)
    raw.executemany(
        "INSERT INTO pending_imports (arr, infohash, al_id, record) VALUES (?, ?, ?, jsonb(?))",
        rows,
    )
    raw.execute("PRAGMA user_version=2")
    raw.commit()
    raw.close()


# The v4 shape: the v2 pending table (still one row per entry) plus the `guard_facts`
# rows the v2 -> v3 step backfilled. The v4 -> v5 step folds the rows per torrent.
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
    """A v4-stamped db holding the given (arr, infohash, al_id, json) pending rows and (arr, al_id, json) guard rows."""

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
    """A fully shaped v4 pending row: one entry's per-entry record on `infohash`, `fields` overriding."""

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


def _assert_round_trips(records: dict[str, dict[str, Any]]) -> None:
    """Every migrated blob re-serializes byte-for-byte through the record class, gaining only its empty size maps."""

    for infohash, blob in records.items():
        rebuilt = PendingImport.from_json(blob, guards={}).to_json()
        assert json.dumps(rebuilt) == json.dumps({**blob, "sizes_by_name": {}, "identified": {}}), infohash


def _pending_columns(db: Path) -> list[str]:
    """The `pending_imports` column names as the file holds them."""

    raw = sqlite3.connect(str(db))
    try:
        return [str(row[1]) for row in raw.execute("PRAGMA table_info(pending_imports)")]
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
        # The :memory: stand-in is stamped on create. The backup-API promote must
        # carry that stamp into the file it writes, or every fresh install would
        # look like a v0 db on its second run.
        store = _open(tmp_path)
        store.save(preview=False)
        store.close()
        assert _user_version(tmp_path / "cache.db") == SCHEMA_VERSION

    def test_v0_db_is_upgraded_in_place(self, tmp_path: Path) -> None:
        # The scenario the gate exists for: a db from before `fallback_satisfied`
        # shipped must be ALTERed current instead of crashing get_entry every run.
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.executescript(_V0_SCHEMA)
        raw.execute("INSERT INTO entries (arr, al_id, name) VALUES ('sonarr', 7, 'Frieren')")
        raw.commit()
        raw.close()
        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.name == "Frieren"
        assert entry.fallback_satisfied is False  # backfilled column default
        store.close()
        # The upgrade committed at load time - durable even though the run's own
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_entry(Arr.SONARR, 7) is None  # reads work post-upgrade
        store.close()
        assert _user_version(db) == SCHEMA_VERSION

    def test_v1_pending_imports_walk_the_chain_to_torrent_records(self, tmp_path: Path) -> None:
        # The v1 -> v2 step grows the (arr, infohash) PK to (arr, infohash, al_id)
        # via a table rebuild, and the v4 -> v5 fold turns each row into a claim.
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        pending = store.get_pending(Arr.SONARR)
        assert set(pending) == {"legacy", "tagged"}
        # The sentinel-keyed legacy row rehydrates as a claim with al_id 0.
        legacy = PendingImport.from_json(pending["legacy"], guards={})
        assert claim_al_ids(legacy) == (0,)
        assert legacy.claims[0].title == "Old Show"
        tagged = PendingImport.from_json(pending["tagged"], guards={})
        assert claim_al_ids(tagged) == (9,)
        assert tagged.series_ids == (6,)
        # The folded rows behave exactly like modern records: keyed by hash, dropped by hash.
        store.drop_pending(Arr.SONARR, "legacy")
        assert set(store.get_pending(Arr.SONARR)) == {"tagged"}
        store.close()
        assert _user_version(db) == SCHEMA_VERSION

    def test_v2_guard_backfill_latest_added_at_wins(self, tmp_path: Path) -> None:
        # The v2 -> v3 step: two records of one entry carry divergent frozen guard
        # copies - the exact divergence the guard_facts row kills. The newest
        # blob's copy (by added_at) becomes the entry's row, and every blob is
        # stripped of its guards key. (A guard-less record NEWER than a
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_guards(Arr.SONARR) == {
            9: GuardFacts(entry_groups=("New",), stale_groups=("Old",), owned_episodes=(OwnedEpisode(101, 700),)),
        }
        # The divergent per-record copies are gone for good.
        assert all("guards" not in rec for rec in store.get_pending(Arr.SONARR).values())
        store.close()
        assert _user_version(db) == SCHEMA_VERSION

    def test_v2_guard_backfill_skips_ineligible_records(self, tmp_path: Path) -> None:
        # No row may come from: the legacy al_id=0 sentinel, a radarr record, a
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_guards(Arr.SONARR) == {}
        assert store.get_guards(Arr.RADARR) == {}
        stripped = [*store.get_pending(Arr.SONARR).values(), *store.get_pending(Arr.RADARR).values()]
        assert len(stripped) == 4
        assert all("guards" not in rec for rec in stripped)
        store.close()

    def test_v2_migration_leaves_existing_guard_rows_alone(self, tmp_path: Path) -> None:
        # The step's idempotence guard: a v2-stamped db already carrying guard
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_guards(Arr.SONARR) == {9: GuardFacts(entry_groups=("Kept",))}
        store.close()

    def test_v3_id_form_names_are_nulled(self, tmp_path: Path) -> None:
        # The v3 -> v4 step: a name stored as the id-form fallback (an AniList
        # outage at write time) is nulled so a later cached read resolves it. A
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert _entry_name(store, 1) is None
        assert _entry_name(store, 2) == "Frieren"
        assert _entry_name(store, 3, Arr.RADARR) is None
        store.close()
        assert _user_version(db) == SCHEMA_VERSION

    def test_v4_rows_of_one_torrent_fold_into_one_record(self, tmp_path: Path) -> None:
        # The v4 -> v5 step: one torrent's per-entry rows become ONE record whose
        # claims sit in al_id order, each keeping its row's clock as its own. The
        # rows group by case fold and the key + record hash come out lowercase.
        # The release facts are the first row's, the maps union with the first
        # row winning a shared name, an exclusion another row mapped is dropped,
        # the birth is the oldest stamp, and the flag holds when every row had it.
        # The union keeps distinct file-name keys byte-exact (an NFC and an NFD
        # spelling stay two keys): `normalized_leaf` reconciles them at read time.
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
        # Inserted newest first: the fold orders claims by al_id, not by row order.
        _seed_v4_db(db, [("sonarr", "ABCD", 22, json.dumps(second)), ("sonarr", "abcd", 11, json.dumps(first))])

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        records = store.get_pending(Arr.SONARR)
        assert records == {
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
                        "claimed_at": "2026-07-01 00:00:00",
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
                        "claimed_at": "2026-07-02 00:00:00",
                    },
                ],
                "excluded_files": ["y.mkv", "x.mkv"],
                "release_sizes": [10, 20],
                "awaiting_cleanup": True,
            },
        }
        _assert_round_trips(records)
        assert _pending_columns(db) == ["arr", "infohash", "record"]
        store.close()
        assert _user_version(db) == SCHEMA_VERSION

    def test_v4_fold_needs_every_cleanup_flag_and_keeps_the_arrs_apart(self, tmp_path: Path) -> None:
        # One unflagged row clears the folded flag. The other arr's row on the
        # same hash stays its own record, and the guard rows survive the step
        # to be read through the folded claims (the orphan stays unread).
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        sonarr = store.get_pending(Arr.SONARR)
        radarr = store.get_pending(Arr.RADARR)
        assert PendingImport.from_json(sonarr["h"], guards={}).awaiting_cleanup is False
        assert claim_al_ids(PendingImport.from_json(sonarr["h"], guards={})) == (11, 22)
        assert PendingImport.from_json(radarr["h"], guards={}).awaiting_cleanup is True
        assert claim_al_ids(PendingImport.from_json(radarr["h"], guards={})) == (0,)
        assert store.other_arr_holds(Arr.SONARR, "h") is True
        assert store.get_guards(Arr.SONARR) == {
            11: GuardFacts(entry_groups=("A",)),
            22: GuardFacts(entry_groups=("B",)),
        }
        _assert_round_trips(sonarr)
        _assert_round_trips(radarr)
        store.close()

    def test_v4_fold_takes_the_oldest_parseable_stamp(self, tmp_path: Path) -> None:
        # The birth is the oldest row stamp that parses. A junk stamp is skipped
        # for the birth yet stays the claim's own clock verbatim, and a torrent
        # whose rows all carry junk is born unstamped.
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        records = store.get_pending(Arr.SONARR)
        assert records["k"]["added_at"] == "2026-07-02 00:00:00"
        assert [claim["claimed_at"] for claim in records["k"]["claims"]] == [
            "junk",
            "2026-07-03 00:00:00",
            "2026-07-02 00:00:00",
        ]
        assert records["j"]["added_at"] == ""
        assert newest_claimed_at_of(records["j"]) is None
        _assert_round_trips(records)
        store.close()

    def test_v4_legacy_ids_fold_into_the_claims_window(self, tmp_path: Path) -> None:
        # A row from before the ordered window (no `ordered_episode_ids`) claims
        # the ids its map and legacy `episode_ids` list held, sorted, zeros
        # dropped. A row that carried its window keeps it as is, and the legacy
        # list reaches no record.
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

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        records = store.get_pending(Arr.SONARR)
        assert all("episode_ids" not in record for record in records.values())
        old = PendingImport.from_json(records["old"], guards={})
        assert old.claims[0].ordered_episode_ids == (1, 2, 3)
        assert old.claims[0].names == EntryNames()
        # The map itself is carried verbatim: the fold only derives the window from it.
        assert dict(old.file_episode_map) == {"a.mkv": (2, 0)}
        new = PendingImport.from_json(records["new"], guards={})
        assert new.claims[0].ordered_episode_ids == (9, 8)
        _assert_round_trips({"new": records["new"]})
        store.close()

    def test_v4_unscoped_row_stays_unscoped_and_carries_its_names_verbatim(self, tmp_path: Path) -> None:
        # An unscoped v4 claim (an empty window, no legacy list) claims nothing off
        # its map, and the stored names dict rides as is, never re-encoded through
        # the live class.
        db = tmp_path / "cache.db"
        unscoped = _v4_blob("u", 5, ordered_episode_ids=[], names={"series": "Series"})
        _seed_v4_db(db, [("sonarr", "u", 5, json.dumps(unscoped))])

        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        records = store.get_pending(Arr.SONARR)
        (claim,) = records["u"]["claims"]
        assert claim["ordered_episode_ids"] == []
        assert claim["names"] == {"series": "Series"}
        assert PendingImport.from_json(records["u"], guards={}).claims[0].names == EntryNames(series="Series")
        store.close()

    def test_v4_migration_is_a_no_op_on_reopen(self, tmp_path: Path) -> None:
        # The migrated db reopens as found: same records, no step announced, and
        # a v4 stamp over a table already in the v5 shape (no al_id column) is
        # stamped current without touching the rows.
        db = tmp_path / "cache.db"
        _seed_v4_db(db, [("sonarr", "h", 11, json.dumps(_v4_blob("h", 11)))])
        migrated = CacheStore.load(str(db), config_checksum=CHECKSUM)
        records = migrated.get_pending(Arr.SONARR)
        migrated.close()
        assert _user_version(db) == SCHEMA_VERSION

        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default
        reopened = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert reopened.get_pending(Arr.SONARR) == records
        reopened.close()
        assert recording.of_type(Diagnostic) == []

        folded = tmp_path / "folded.db"
        raw = sqlite3.connect(str(folded))
        raw.executescript(_V1_PENDING_SCHEMA)  # the (arr, infohash) PK: v5's shape too
        raw.execute(
            "INSERT INTO pending_imports (arr, infohash, record) VALUES ('sonarr', 'h', jsonb(?))",
            (json.dumps(records["h"]),),
        )
        raw.execute("PRAGMA user_version=4")
        raw.commit()
        raw.close()
        store = CacheStore.load(str(folded), config_checksum=CHECKSUM)
        assert store.get_pending(Arr.SONARR) == records
        store.close()
        assert _user_version(folded) == SCHEMA_VERSION

    def test_fresh_db_starts_at_v5_without_an_al_id_column(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        store = _open(tmp_path)
        store.save(preview=False)
        store.close()
        assert _user_version(db) == SCHEMA_VERSION
        assert _pending_columns(db) == ["arr", "infohash", "record"]

    def test_newer_schema_is_refused_not_quarantined(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        raw = sqlite3.connect(str(db))
        raw.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
        raw.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
        raw.commit()
        raw.close()

        with pytest.raises(CacheSchemaError):
            CacheStore.load(str(db), config_checksum=CHECKSUM)
        # Fail closed: the healthy newer db is left exactly where it was.
        assert db.exists()
        assert not list(tmp_path.glob("cache.db.corrupt-*"))


class TestEntries:
    """Per-entry columns merge partially on update, format/match timestamps, and stay isolated per arr."""

    def test_update_cache_is_a_partial_merge(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title", "url": "u"})
        # A later update with only one field must not wipe the others.
        store.update_cache(Arr.SONARR, 7, {"coverage": "S01"})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert (entry.name, entry.url, entry.coverage) == ("Title", "u", "S01")
        store.close()

    def test_update_cache_formats_datetime_timestamp(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.updated_at == "2021-06-05 04:03:02"
        store.close()

    def test_check_al_id_in_cache_matches_timestamp(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        assert (
            store.check_al_id_in_cache(Arr.SONARR, 7, make_entry_record(updated_at=datetime(2021, 6, 5, 4, 3, 2)))
            is True
        )
        # Same id, different timestamp -> stale.
        assert (
            store.check_al_id_in_cache(Arr.SONARR, 7, make_entry_record(updated_at=datetime(2022, 1, 1, 0, 0, 0)))
            is False
        )
        # Unknown id -> no record -> no match.
        assert (
            store.check_al_id_in_cache(Arr.SONARR, 8, make_entry_record(updated_at=datetime(2021, 6, 5, 4, 3, 2)))
            is False
        )
        store.close()

    def test_arrs_are_isolated(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "S"})
        store.update_cache(Arr.RADARR, 7, {"name": "R"})
        assert _entry_name(store, 7, Arr.SONARR) == "S"
        assert _entry_name(store, 7, Arr.RADARR) == "R"
        store.close()

    def test_get_entry_reads_all_scalar_columns_at_once(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(
            Arr.SONARR,
            7,
            {"name": "Title", "url": "u", "coverage": "S01", "updated_at": datetime(2021, 6, 5, 4, 3, 2)},
        )
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert (entry.name, entry.url, entry.coverage, entry.updated_at) == (
            "Title",
            "u",
            "S01",
            "2021-06-05 04:03:02",
        )
        # A missing row reads back as None, not an all-None record.
        assert store.get_entry(Arr.SONARR, 999) is None
        store.close()


class TestFallbackSatisfied:
    """`fallback_satisfied` defaults false, round-trips as a real bool, and survives partial updates that omit it."""

    def test_defaults_false_and_roundtrips_as_bool(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        # A row written without the key reads back False (NOT NULL DEFAULT 0)...
        store.update_cache(Arr.SONARR, 7, {"name": "Title"})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is False
        # ...and a written True reads back as a real bool, not the stored int.
        store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": True})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is True
        store.close()

    def test_persists_across_save_and_reopen(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": True})
        store.save(preview=False)
        store.close()

        reopened = _open(tmp_path)
        entry = reopened.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is True
        reopened.close()

    def test_partial_update_preserves_until_rewritten(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": True})
        # A partial merge that omits the key must not clear the marker...
        store.update_cache(Arr.SONARR, 7, {"coverage": "S01"})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is True
        # ...but a supplied False overwrites it.
        store.update_cache(Arr.SONARR, 7, {"fallback_satisfied": False})
        entry = store.get_entry(Arr.SONARR, 7)
        assert entry is not None
        assert entry.fallback_satisfied is False
        store.close()


class TestTorrentHashes:
    """Torrent hashes preserve the None (hashless) marker and dedup it.

    Rewriting the set replaces it wholesale, and the shape stays compatible with a pre-existing NOT NULL schema.
    """

    def test_roundtrip_preserves_none_marker(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["aaa", "bbb", None]})
        # The None marker (a hashless release) is preserved - the planner dedups on
        # its membership, so dropping it would re-grab the release. Order is free.
        assert set(store.torrent_hashes(Arr.SONARR, 7)) == {"aaa", "bbb", None}
        # Missing entry -> empty list, never None.
        assert store.torrent_hashes(Arr.SONARR, 999) == []
        store.close()

    def test_duplicate_none_markers_collapse_to_one(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": [None, "aaa", None]})
        assert store.torrent_hashes(Arr.SONARR, 7) == [None, "aaa"]
        store.close()

    def test_rewrite_replaces_the_set(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["aaa", "bbb"]})
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["ccc"]})
        assert store.torrent_hashes(Arr.SONARR, 7) == ["ccc"]
        store.close()

    def test_none_marker_on_a_preexisting_db(self, tmp_path: Path) -> None:
        # Upgrade path: a pre-existing cache.db has `infohash TEXT NOT NULL`, and
        # CREATE TABLE IF NOT EXISTS will NOT alter it - so the None marker must
        # round-trip via the sentinel WITHOUT an IntegrityError, not lean on a schema
        # change that never reaches an existing db. (This test fails if torrent_hashes
        # is made nullable instead, since the old table stays NOT NULL.) The raw
        # schema tracks _SCHEMA's current shape: dbs from older schemas are
        # unsupported until real migrations land (manual ALTERs bridge the gap).
        db = str(tmp_path / "cache.db")
        raw = sqlite3.connect(db)
        raw.executescript(
            "CREATE TABLE entries (arr TEXT NOT NULL, al_id INTEGER NOT NULL, name TEXT, "
            "url TEXT, coverage TEXT, updated_at TEXT, "
            "fallback_satisfied INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (arr, al_id));"
            "CREATE TABLE torrent_hashes (arr TEXT NOT NULL, al_id INTEGER NOT NULL, "
            "infohash TEXT NOT NULL, PRIMARY KEY (arr, al_id, infohash), "
            "FOREIGN KEY (arr, al_id) REFERENCES entries (arr, al_id) ON DELETE CASCADE);",
        )
        raw.commit()
        raw.close()

        store = CacheStore.load(db, config_checksum=CHECKSUM)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["aaa", None]})  # must not raise
        assert set(store.torrent_hashes(Arr.SONARR, 7)) == {"aaa", None}
        store.close()


class TestPreviewGate:
    """The staged-write preview gate: a preview save never persists, only a real save does."""

    def test_preview_save_on_missing_file_writes_nothing(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title"})
        store.save(preview=True)
        store.close()
        assert not (tmp_path / "cache.db").exists()

    def test_real_save_on_missing_file_creates_and_persists(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title"})
        store.save(preview=False)
        store.close()

        assert (tmp_path / "cache.db").exists()
        reopened = _open(tmp_path)
        assert _entry_name(reopened, 7) == "Title"
        reopened.close()

    def test_preview_on_existing_db_does_not_mutate_committed_state(self, tmp_path: Path) -> None:
        # Establish a real, committed db.
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Original"})
        store.save(preview=False)
        store.close()

        # A preview run stages changes but must persist none of them.
        preview = _open(tmp_path)
        preview.update_cache(Arr.SONARR, 7, {"name": "Changed"})
        preview.update_cache(Arr.SONARR, 8, {"name": "New"})
        preview.save(preview=True)
        preview.close()

        reopened = _open(tmp_path)
        assert _entry_name(reopened, 7) == "Original"  # not "Changed"
        assert reopened.get_entry(Arr.SONARR, 8) is None  # never added
        reopened.close()


class TestSelectionDigest:
    """`selection_stale`/`vouch_selection`: missing-key-fresh, per-arr, commit-gated."""

    def test_no_digest_on_record_reads_fresh(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        assert store.selection_stale(Arr.SONARR, "d1") is False
        store.close()

    def test_vouch_round_trips_through_a_real_save(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.vouch_selection(Arr.SONARR, "d1")
        store.save(preview=False)
        store.close()

        reopened = _open(tmp_path)
        assert reopened.selection_stale(Arr.SONARR, "d1") is False
        assert reopened.selection_stale(Arr.SONARR, "d2") is True
        reopened.close()

    def test_preview_save_does_not_consume_a_pending_recheck(self, tmp_path: Path) -> None:
        # Matching prefs changed (the committed digest is old). A preview run
        # stages the new one, but its rollback must leave the re-check armed
        # for the next real run.
        store = _open(tmp_path)
        store.vouch_selection(Arr.SONARR, "old")
        store.save(preview=False)
        store.close()

        previewing = _open(tmp_path)
        assert previewing.selection_stale(Arr.SONARR, "new") is True
        previewing.vouch_selection(Arr.SONARR, "new")
        previewing.save(preview=True)
        previewing.close()

        real = _open(tmp_path)
        assert real.selection_stale(Arr.SONARR, "new") is True
        real.close()

    def test_arrs_vouch_independently(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.vouch_selection(Arr.SONARR, "d1")
        assert store.selection_stale(Arr.SONARR, "d2") is True
        assert store.selection_stale(Arr.RADARR, "d2") is False
        store.close()


class TestAnilistMeta:
    """AniList metadata records round-trip by id, are iterable, and a later `put` overwrites the prior record."""

    def test_roundtrip_get_and_iter(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        rec = {"fetched_at": "2026-06-20 12:00:00", "data": {"Media": {"id": 1}}}
        store.put_anilist_meta(1, rec)
        assert store.get_anilist_meta(1) == rec
        assert store.get_anilist_meta(999) is None
        assert dict(store.iter_anilist_meta()) == {1: rec}
        store.close()

    def test_put_overwrites(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"fetched_at": "2026-06-20 12:00:00", "data": {"a": 1}})
        store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {"a": 2}})
        meta = store.get_anilist_meta(1)
        assert meta is not None
        assert meta["data"] == {"a": 2}
        store.close()


class TestRecordFreshness:
    """The persisted-record readers: the payload half and the stamp half, split apart."""

    def test_record_payload_wants_a_non_empty_object(self) -> None:
        assert record_payload(None, "data") is None
        assert record_payload({"fetched_at": "2026-06-26 12:00:00"}, "data") is None
        assert record_payload({"data": {}}, "data") is None
        assert record_payload({"data": [{"x": 1}]}, "data") is None
        assert record_payload({"data": {"x": 1}}, "data") == {"x": 1}

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
        # A raw row's stamp may not even be a string: the reader answers None, never raises.
        raw: dict[str, Any] = {"added_at": 5}
        assert parse_stamp_or_none(raw["added_at"]) is None


class TestSonarrParse:
    """Parsed Sonarr records round-trip whole keyed by filename, and a missing filename reads back None."""

    def test_roundtrip_get(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        info = ParsedFileInfo(season_number=1, episode_numbers=(2,))
        rec = {"fetched_at": "2026-06-20 12:00:00", "parse": to_parse_record(info)}
        store.put_sonarr_parse("file.mkv", rec)
        read_back = store.get_sonarr_parse("file.mkv")
        assert read_back == rec
        assert read_back is not None
        assert parsed_info(read_back) == info
        assert store.get_sonarr_parse("missing.mkv") is None
        store.close()

    def test_series_fingerprint_key_round_trips(self, tmp_path: Path) -> None:
        # The unmatched record's pin: the freshness reader dispatches on its presence.
        store = _open(tmp_path)
        rec = {
            "fetched_at": "2026-06-20 12:00:00",
            "parse": to_parse_record(ParsedFileInfo()),
            "series_fp": "fp",
        }
        store.put_sonarr_parse("unmatched.mkv", rec)
        assert store.get_sonarr_parse("unmatched.mkv") == rec
        store.close()

    def test_legacy_record_reads_back_unreadable(self, tmp_path: Path) -> None:
        # A pre-existing row carries an episode list, not a whole parse: it reads
        # back verbatim and the parse reader refuses it, so the file re-parses.
        store = _open(tmp_path)
        rec = {"fetched_at": "2026-06-20 12:00:00", "episodes": [{"season": 1, "episode": 1}]}
        store.put_sonarr_parse("legacy.mkv", rec)
        read_back = store.get_sonarr_parse("legacy.mkv")
        assert read_back == rec
        assert read_back is not None
        assert parsed_info(read_back) is None
        store.close()


class TestPendingImports:
    """Pending records are tracked per (arr, infohash), read keyed or by any claim's series id, and droppable."""

    def test_roundtrip_drop_and_arr_isolation(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        rec = _row(pending_import(infohash="h1", al_id=3, series_id=5))
        store.put_pending(Arr.SONARR, "h1", rec)
        store.put_pending(Arr.RADARR, "h2", {"infohash": "h2"})
        assert store.get_pending(Arr.SONARR) == {"h1": rec}
        assert store.get_pending(Arr.RADARR) == {"h2": {"infohash": "h2"}}
        # The keyed reads see only their own arr's row.
        assert store.get_pending_record(Arr.SONARR, "h1") == rec
        assert store.get_pending_record(Arr.RADARR, "h1") is None

        store.drop_pending(Arr.SONARR, "h1")
        assert store.get_pending(Arr.SONARR) == {}
        assert store.get_pending_record(Arr.SONARR, "h1") is None
        # Dropping a missing key is a no-op.
        store.drop_pending(Arr.SONARR, "nope")
        store.close()

    def test_one_record_per_torrent_replaced_whole(self, tmp_path: Path) -> None:
        # One torrent listed on two AniList entries is ONE row carrying both
        # claims: a re-put under the hash replaces the record whole (never a
        # second row), and the drop takes every claim with it.
        store = _open(tmp_path)
        store.put_pending(Arr.SONARR, "h", _row(pending_import(infohash="h", al_id=11, series_id=5)))
        both = _row(
            pending_import(
                infohash="h",
                claims=(entry_claim(al_id=11, series_id=5), entry_claim(al_id=22, series_id=5)),
            )
        )
        store.put_pending(Arr.SONARR, "h", both)
        assert store.get_pending(Arr.SONARR) == {"h": both}
        assert store.stats().pending_imports == 1

        store.drop_pending(Arr.SONARR, "h")
        assert store.get_pending(Arr.SONARR) == {}
        store.close()

    def test_other_arr_holds_matches_the_key_exactly(self, tmp_path: Path) -> None:
        # Every record key is the lowercase infohash, so the other arr's row is
        # found by exact key and never by case fold.
        store = _open(tmp_path)
        store.put_pending(Arr.RADARR, "abcd", {"infohash": "abcd"})

        assert store.other_arr_holds(Arr.SONARR, "abcd") is True
        assert store.other_arr_holds(Arr.SONARR, "ABCD") is False
        assert store.other_arr_holds(Arr.RADARR, "abcd") is False
        store.close()

    def test_other_arr_holds_ignores_the_arrs_own_record(self, tmp_path: Path) -> None:
        # The category gate asks whether the OTHER arr still claims the torrent:
        # an arr's own record never counts, a missing hash reads False, and the
        # other arr's row flips the answer for as long as it lives.
        store = _open(tmp_path)
        store.put_pending(Arr.SONARR, "h", pending_import(infohash="h", al_id=11).to_json())

        assert store.other_arr_holds(Arr.SONARR, "h") is False
        assert store.other_arr_holds(Arr.RADARR, "h") is True
        assert store.other_arr_holds(Arr.SONARR, "other") is False

        store.put_pending(Arr.RADARR, "h", {"infohash": "h"})
        assert store.other_arr_holds(Arr.SONARR, "h") is True
        store.drop_pending(Arr.RADARR, "h")
        assert store.other_arr_holds(Arr.SONARR, "h") is False
        store.close()

    def test_get_pending_for_series_matches_any_claim_in_sql(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        a = _row(pending_import(infohash="a", al_id=1, series_id=5))
        b = _row(pending_import(infohash="b", al_id=2, series_id=5))
        c = _row(pending_import(infohash="c", al_id=3, series_id=9))
        # A two-claim record spanning both series answers to either.
        both = _row(
            pending_import(
                infohash="t",
                claims=(entry_claim(al_id=6, series_id=5), entry_claim(al_id=7, series_id=9)),
            )
        )
        store.put_pending(Arr.SONARR, "a", a)
        store.put_pending(Arr.SONARR, "b", b)
        store.put_pending(Arr.SONARR, "c", c)
        store.put_pending(Arr.SONARR, "t", both)
        # A claim with no series id, and a record with no claims: never matched.
        store.put_pending(Arr.SONARR, "d", {"infohash": "d", "claims": [{"al_id": 4}]})
        store.put_pending(Arr.SONARR, "e", {"infohash": "e"})

        # Only records with a claim on the series come back. The integer series_id binds directly.
        assert store.get_pending_for_series(Arr.SONARR, 5) == {"a": a, "b": b, "t": both}
        assert store.get_pending_for_series(Arr.SONARR, 9) == {"c": c, "t": both}
        assert store.get_pending_for_series(Arr.SONARR, 404) == {}

        # Fresh per call: a drop is reflected immediately (no stale snapshot).
        store.drop_pending(Arr.SONARR, "a")
        assert store.get_pending_for_series(Arr.SONARR, 5) == {"b": b, "t": both}
        store.close()


class TestGuardFacts:
    """One guard-evidence row per (arr, al_id), read-filtered to entries with a live claim."""

    def test_roundtrip_and_arr_isolation(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.put_pending(Arr.SONARR, "hs", pending_import(infohash="hs", al_id=7).to_json())
        store.put_pending(Arr.RADARR, "hr", pending_import(infohash="hr", al_id=7, series_id=0).to_json())
        facts = GuardFacts(
            entry_groups=("SubsPlease", "Erai-raws"),
            stale_groups=("HorribleSubs",),
            owned_episodes=(OwnedEpisode(101, 700_000_000), OwnedEpisode(102, 650_000_000)),
        )
        store.put_guards(Arr.SONARR, 7, facts)
        store.put_guards(Arr.RADARR, 7, GuardFacts(entry_groups=("Beatrice-Raws",)))

        # Typed on both ends: owned_episodes come back as OwnedEpisode tuples.
        assert store.get_guards(Arr.SONARR) == {7: facts}
        assert store.get_guards(Arr.RADARR) == {7: GuardFacts(entry_groups=("Beatrice-Raws",))}
        store.close()

    def test_latest_put_wins(self, tmp_path: Path) -> None:
        # The fix's write semantic: a re-seed overwrites the entry's single row
        # whole, so no two reads can ever see divergent evidence for one entry.
        store = _open(tmp_path)
        store.put_pending(Arr.SONARR, "h", pending_import(infohash="h", al_id=7).to_json())
        store.put_guards(Arr.SONARR, 7, GuardFacts(entry_groups=("Old",)))
        newest = GuardFacts(entry_groups=("New",), stale_groups=("Old",))
        store.put_guards(Arr.SONARR, 7, newest)
        assert store.get_guards(Arr.SONARR) == {7: newest}
        store.close()

    def test_orphan_rows_are_stored_but_never_read(self, tmp_path: Path) -> None:
        # There is deliberately no delete path (an entry's claims on several
        # torrents share the row), so reads join live claims: the read stays
        # bounded by in-flight work, not all-time grab history, and an orphan
        # row is invisible until the entry seeds again.
        store = _open(tmp_path)
        facts = GuardFacts(entry_groups=("SubGroup",))
        store.put_guards(Arr.SONARR, 7, facts)
        assert store.get_guards(Arr.SONARR) == {}

        store.put_pending(Arr.SONARR, "h", pending_import(infohash="h", al_id=7).to_json())
        assert store.get_guards(Arr.SONARR) == {7: facts}

        # The last record dropping re-orphans the row: stored, no longer read.
        store.drop_pending(Arr.SONARR, "h")
        assert store.get_guards(Arr.SONARR) == {}
        assert store.stats().guard_facts == 1
        store.close()

    def test_reads_through_every_claim_of_a_record(self, tmp_path: Path) -> None:
        # The join walks every claim on a live record: both entries sharing one
        # torrent get their evidence back, an entry claiming nothing stays unread.
        store = _open(tmp_path)
        seven = GuardFacts(entry_groups=("A",))
        eight = GuardFacts(entry_groups=("B",))
        store.put_guards(Arr.SONARR, 7, seven)
        store.put_guards(Arr.SONARR, 8, eight)
        store.put_guards(Arr.SONARR, 9, GuardFacts(entry_groups=("C",)))
        shared = pending_import(infohash="h", claims=(entry_claim(al_id=7), entry_claim(al_id=8, series_id=8)))
        store.put_pending(Arr.SONARR, "h", shared.to_json())

        assert store.get_guards(Arr.SONARR) == {7: seven, 8: eight}
        store.close()

    def test_empty_store_reads_empty(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        assert store.get_guards(Arr.SONARR) == {}
        store.close()


class TestHistoryCheckpoints:
    """History checkpoints upsert per arr and respect the preview gate."""

    def test_roundtrip_upsert_and_arr_isolation(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        assert store.get_history_checkpoint(Arr.SONARR) is None

        store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T10:00:00Z", 12))
        store.put_history_checkpoint(Arr.RADARR, HistoryCheckpoint("2026-07-02T10:00:00Z", 3))
        assert store.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-01T10:00:00Z", 12)
        assert store.get_history_checkpoint(Arr.RADARR) == HistoryCheckpoint("2026-07-02T10:00:00Z", 3)

        # Upsert: a later advance replaces the arr's single row.
        store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-03T10:00:00Z", 40))
        assert store.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-03T10:00:00Z", 40)
        store.close()

    def test_preview_save_does_not_persist_checkpoint(self, tmp_path: Path) -> None:
        # The dry-run gate: a previewed run must never advance the cursor.
        db = tmp_path / "cache.db"
        store = _open(tmp_path)
        store.save(preview=False)  # promote so the preview below has a real file
        store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T10:00:00Z", 12))
        store.save(preview=True)
        store.close()

        assert db.exists()
        reopened = _open(tmp_path)
        assert reopened.get_history_checkpoint(Arr.SONARR) is None
        reopened.close()


class TestPromoteFailure:
    """A failed atomic promote leaves no partial `cache.db` behind, and a later save still promotes for real."""

    def test_failed_promote_leaves_no_partial_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the promote rename fails (disk full / killed mid-copy), it must leave NO
        # partial cache.db - else the next run mistakes the torn file for a real
        # (empty) cache.
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "T"})
        # Scope the patch to this save only: the later `again.save` must promote for real.
        with monkeypatch.context() as mp, contextlib.suppress(OSError):
            mp.setattr("pearlarr.cache.os.replace", _raise_os_replace)
            store.save(preview=False)
        store.close()

        assert not (tmp_path / "cache.db").exists()  # no 0-byte orphan
        assert not list(tmp_path.glob("cache.db.promote*"))  # temp cleaned up

        # A fresh run promotes for real (the failed attempt left nothing torn behind).
        again = _open(tmp_path)
        again.update_cache(Arr.SONARR, 7, {"name": "T"})
        again.save(preview=False)
        again.close()
        assert (tmp_path / "cache.db").exists()

        reopened = _open(tmp_path)
        assert _entry_name(reopened, 7) == "T"
        reopened.close()


class TestRunLifecycle:
    """Replays the order the run loop drives a real `CacheStore` through.

    The run-loop tests mock `cache_store`, so this is the only check that the real
    load -> writes -> save(commit) -> close(rollback) -> reopen sequence behaves.
    """

    def test_run_call_order_persists_and_reloads(self, tmp_path: Path) -> None:
        db = str(tmp_path / "cache.db")

        # Run 1 (real): process one entry the way the loop does, then commit + close.
        store = CacheStore.load(db, config_checksum=CHECKSUM)
        store.update_cache(
            Arr.SONARR,
            7,
            {
                "name": "Show",
                "url": "u",
                "coverage": "S01",
                "updated_at": datetime(2026, 1, 2, 3, 4, 5),
                "torrent_hashes": ["aaa", "bbb"],
            },
        )
        store.put_anilist_meta(7, {"fetched_at": "2026-06-26 12:00:00", "data": {"id": 7}})
        carried = _row(pending_import(infohash="aaa", al_id=7, series_id=5))
        store.put_pending(Arr.SONARR, "aaa", carried)
        store.save(preview=False)  # mid/end-of-run commit
        store.close()  # finally: rollback is a no-op (already committed)

        # Run 2: reopen -> cache hit, remembered hashes, carried-over pending.
        again = CacheStore.load(db, config_checksum=CHECKSUM)
        assert again.check_al_id_in_cache(Arr.SONARR, 7, make_entry_record(updated_at=datetime(2026, 1, 2, 3, 4, 5)))
        assert again.torrent_hashes(Arr.SONARR, 7) == ["aaa", "bbb"]
        assert again.get_pending(Arr.SONARR) == {"aaa": carried}
        # A completed import is dropped, and that drop persists across a save.
        again.drop_pending(Arr.SONARR, "aaa")
        again.save(preview=False)
        again.close()

        final = CacheStore.load(db, config_checksum=CHECKSUM)
        assert final.get_pending(Arr.SONARR) == {}
        final.close()


class TestMaintenance:
    """Eviction drops only stale or stampless records and keeps fresh ones. Stats/integrity report accurate counts."""

    def test_evict_anilist_meta_drops_only_stale(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"fetched_at": "2020-01-01 00:00:00", "data": {"x": 1}})
        store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
        assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1
        assert store.get_anilist_meta(1) is None
        assert store.get_anilist_meta(2) is not None
        store.close()

    def test_evict_sonarr_parse_drops_only_stale(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        parse = to_parse_record(ParsedFileInfo(season_number=1, episode_numbers=(1,)))
        store.put_sonarr_parse("old.mkv", {"fetched_at": "2020-01-01 00:00:00", "parse": parse})
        store.put_sonarr_parse("new.mkv", {"fetched_at": "2026-06-26 12:00:00", "parse": parse})
        assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
        assert store.get_sonarr_parse("old.mkv") is None
        assert store.get_sonarr_parse("new.mkv") is not None
        store.close()

    def test_evict_sweeps_stampless_records(self, tmp_path: Path) -> None:
        # A record with no fetched_at -> NULL generated column. It is unreadable
        # (stamp_is_fresh rejects it) and must not become un-evictable dead weight.
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"data": {"x": 1}})  # no fetched_at -> NULL
        store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
        assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1  # only the stampless
        assert store.get_anilist_meta(1) is None
        assert store.get_anilist_meta(2) is not None  # fresh, stamped -> kept
        store.put_sonarr_parse("nostamp.mkv", {"parse": {}})  # no fetched_at -> NULL
        assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
        assert store.get_sonarr_parse("nostamp.mkv") is None
        store.close()

    def test_stats_and_integrity(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "T", "torrent_hashes": ["a", "b"]})
        store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {}})
        store.save(preview=False)  # promote to file so size_bytes > 0

        s = store.stats()
        assert s.entries == 1
        assert s.torrent_hashes == 2
        assert s.anilist_meta == 1
        assert s.sonarr_parse == 0
        assert s.pending_imports == 0
        assert s.size_bytes > 0
        assert store.integrity_check() == "ok"
        store.close()


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
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Keep"})
        store.save(preview=False)
        store.close()
        db = tmp_path / "cache.db"

        # Simulate a transient lock at open time (e.g. the WAL switch hitting BUSY).
        with monkeypatch.context() as mp:
            mp.setattr("pearlarr.cache._connect", _raise_locked)
            raised = False
            try:
                CacheStore.load(str(db), config_checksum=CHECKSUM)
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
        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_entry(Arr.SONARR, 7) is None
        store.update_cache(Arr.SONARR, 7, {"name": "Recovered"})
        store.save(preview=False)
        store.close()

        # The corrupt file was moved aside. A fresh, working db took its place.
        assert len(list(tmp_path.glob("cache.db.corrupt-*"))) == 1
        reopened = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert _entry_name(reopened, 7) == "Recovered"
        reopened.close()

        # The recovery notice names the state that was lost, not just "fresh cache".
        [notice] = [d for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert "moved it to" in notice.message
        assert notice.message.endswith(
            "started a fresh cache (titles will be re-checked; grab-dedup and "
            "pending-import tracking reset, so recent grabs may be re-offered)",
        )
