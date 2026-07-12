# pyright: strict
"""Behavioral tests for the SQLite-backed `CacheStore`.

Pins behavior, not internals: the per-entry records + torrent-hash child rows,
the JSONB meta/parse caches, pending imports, the descriptor, and - critically -
the staged-write / preview gate (a preview never persists; a real save does) and
the in-memory -> file promotion for a missing cache.
"""

import contextlib
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

import pearlarr
from pearlarr.modules.cache import SCHEMA_VERSION, CacheSchemaError, CacheStore, HistoryCheckpoint
from pearlarr.modules.config import Arr
from pearlarr.modules.log import LOG_NAME
from pearlarr.modules.output import Diagnostic, Severity, install_hub
from pearlarr.modules.output.recording import RecordingHub
from pearlarr.modules.sqlite_util import is_corruption

from .builders import make_entry_record

# Stand-in for a config-file checksum. `CacheStore` only stamps and compares the
# value it is handed; it never computes one, so any string works here.
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


def _open(tmp_path: Path) -> CacheStore:
    return CacheStore.load(str(tmp_path / "cache.db"), config_checksum=CHECKSUM)


class TestSchemaAndDescriptor:
    """A missing db opens in-memory and reads empty; a saved db persists the version/checksum descriptor."""

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


def _user_version(db: Path) -> int:
    raw = sqlite3.connect(str(db))
    try:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])
    finally:
        raw.close()


class TestSchemaVersionGate:
    """The schema-version gate: older dbs migrate step-by-step, newer dbs are refused."""

    def test_fresh_db_is_stamped_through_promote(self, tmp_path: Path) -> None:
        # The :memory: stand-in is stamped on create; the backup-API promote must
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
        # Each step announces itself as an INFO hub Diagnostic.
        (upgraded,) = recording.of_type(Diagnostic)
        assert upgraded.severity is Severity.INFO
        assert upgraded.message == "Upgraded cache database schema v0 -> v1"
        assert upgraded.origin == LOG_NAME

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


class TestSonarrParse:
    """Parsed Sonarr episode records round-trip keyed by filename, and a missing filename reads back None."""

    def test_roundtrip_get(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        rec = {"fetched_at": "2026-06-20 12:00:00", "episodes": [{"season": 1, "episode": 2}]}
        store.put_sonarr_parse("file.mkv", rec)
        assert store.get_sonarr_parse("file.mkv") == rec
        assert store.get_sonarr_parse("missing.mkv") is None
        store.close()


class TestPendingImports:
    """Pending imports are tracked per arr+infohash, droppable, and filterable by series id in SQL."""

    def test_roundtrip_drop_and_arr_isolation(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        rec = {"infohash": "h1", "series_id": 5, "episode_ids": [1, 2]}
        store.put_pending(Arr.SONARR, "h1", rec)
        store.put_pending(Arr.RADARR, "h2", {"infohash": "h2"})
        assert store.get_pending(Arr.SONARR) == {"h1": rec}
        assert store.get_pending(Arr.RADARR) == {"h2": {"infohash": "h2"}}

        store.drop_pending(Arr.SONARR, "h1")
        assert store.get_pending(Arr.SONARR) == {}
        # Dropping a missing infohash is a no-op.
        store.drop_pending(Arr.SONARR, "nope")
        store.close()

    def test_get_pending_for_series_filters_in_sql(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        a = {"infohash": "a", "series_id": 5}
        b = {"infohash": "b", "series_id": 5}
        c = {"infohash": "c", "series_id": 9}
        d = {"infohash": "d"}  # no series_id key -> excluded (record ->> 'series_id' is NULL)
        for h, rec in (("a", a), ("b", b), ("c", c), ("d", d)):
            store.put_pending(Arr.SONARR, h, rec)

        # Only this series' records come back; the integer series_id binds directly.
        assert store.get_pending_for_series(Arr.SONARR, 5) == {"a": a, "b": b}
        assert store.get_pending_for_series(Arr.SONARR, 9) == {"c": c}
        assert store.get_pending_for_series(Arr.SONARR, 404) == {}

        # Fresh per call: a drop is reflected immediately (no stale snapshot).
        store.drop_pending(Arr.SONARR, "a")
        assert store.get_pending_for_series(Arr.SONARR, 5) == {"b": b}
        store.close()


class TestHistoryCheckpoints:
    """History checkpoints upsert per arr and respect the preview gate.

    `own_download_ids` unions casefolded hashes across cached torrents and pending imports.
    """

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

    def test_own_download_ids_unions_and_casefolds(self, tmp_path: Path) -> None:
        store = _open(tmp_path)
        # Remembered hashes incl. a None marker (stored as the "" sentinel, excluded).
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["ABCDEF", None]})
        store.put_pending(Arr.SONARR, "FEDCBA", {"series_id": 7})
        store.put_pending(Arr.RADARR, "other", {})

        assert store.own_download_ids(Arr.SONARR) == frozenset({"abcdef", "fedcba"})
        assert store.own_download_ids(Arr.RADARR) == frozenset({"other"})
        store.close()


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
            mp.setattr("pearlarr.modules.cache.os.replace", _raise_os_replace)
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
        store.put_pending(Arr.SONARR, "aaa", {"infohash": "aaa", "series_id": 5})
        store.save(preview=False)  # mid/end-of-run commit
        store.close()  # finally: rollback is a no-op (already committed)

        # Run 2: reopen -> cache hit, remembered hashes, carried-over pending.
        again = CacheStore.load(db, config_checksum=CHECKSUM)
        assert again.check_al_id_in_cache(Arr.SONARR, 7, make_entry_record(updated_at=datetime(2026, 1, 2, 3, 4, 5)))
        assert again.torrent_hashes(Arr.SONARR, 7) == ["aaa", "bbb"]
        assert again.get_pending(Arr.SONARR) == {"aaa": {"infohash": "aaa", "series_id": 5}}
        # A completed import is dropped, and that drop persists across a save.
        again.drop_pending(Arr.SONARR, "aaa")
        again.save(preview=False)
        again.close()

        final = CacheStore.load(db, config_checksum=CHECKSUM)
        assert final.get_pending(Arr.SONARR) == {}
        final.close()


class TestMaintenance:
    """Eviction drops only stale or stampless records and keeps fresh ones; stats/integrity report accurate counts."""

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
        eps = [{"season": 1, "episode": 1}]
        store.put_sonarr_parse("old.mkv", {"fetched_at": "2020-01-01 00:00:00", "episodes": eps})
        store.put_sonarr_parse("new.mkv", {"fetched_at": "2026-06-26 12:00:00", "episodes": eps})
        assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
        assert store.get_sonarr_parse("old.mkv") is None
        assert store.get_sonarr_parse("new.mkv") is not None
        store.close()

    def test_evict_sweeps_stampless_records(self, tmp_path: Path) -> None:
        # A record with no fetched_at -> NULL generated column. It is unreadable
        # (record_is_fresh rejects it) and must not become un-evictable dead weight.
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"data": {"x": 1}})  # no fetched_at -> NULL
        store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
        assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1  # only the stampless
        assert store.get_anilist_meta(1) is None
        assert store.get_anilist_meta(2) is not None  # fresh, stamped -> kept
        store.put_sonarr_parse("nostamp.mkv", {"episodes": []})  # no fetched_at -> NULL
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
            mp.setattr("pearlarr.modules.cache._connect", _raise_locked)
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

        # The corrupt file was moved aside; a fresh, working db took its place.
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
