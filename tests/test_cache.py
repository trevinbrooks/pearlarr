"""Behavioural tests for the SQLite-backed ``CacheStore``.

Pins behaviour, not internals: the per-entry records + torrent-hash child rows,
the JSONB meta/parse caches, pending imports, the descriptor, and - critically -
the staged-write / preview gate (a preview never persists; a real save does) and
the in-memory -> file promotion for a missing cache.
"""

import contextlib
import json
import sqlite3
from datetime import datetime
from typing import Any
from unittest import mock

import seadexarr
from seadexarr.modules.cache import CacheField, CacheStore, _is_corruption
from seadexarr.modules.config import Arr

# Stand-in for a config-file checksum. ``CacheStore`` only stamps and compares the
# value it is handed; it never computes one, so any string works here.
CHECKSUM = "0123456789abcdef0123456789abcdef"


def _entry(dt: datetime) -> Any:
    """A stand-in SeaDex entry exposing only ``updated_at`` (typed Any so it
    satisfies the ``EntryRecord`` parameter without a real record)."""

    class _Entry:
        updated_at = dt

    return _Entry()


def _open(tmp_path) -> CacheStore:
    return CacheStore.load(str(tmp_path / "cache.db"), config_checksum=CHECKSUM)


class TestSchemaAndDescriptor:
    def test_missing_file_opens_in_memory(self, tmp_path) -> None:
        store = _open(tmp_path)
        # Nothing on disk yet, and reads work against the empty in-memory schema.
        assert not (tmp_path / "cache.db").exists()
        assert store.get_cached_name(Arr.SONARR, 7) is None
        store.close()

    def test_descriptor_persists_version_and_checksum(self, tmp_path) -> None:
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
        assert rows.get("seadexarr_version") == seadexarr.__version__
        assert rows.get("config_checksum") == CHECKSUM


class TestEntries:
    def test_update_and_read_back_fields(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title", "url": "u"})
        assert store.get_cached_name(Arr.SONARR, 7) == "Title"
        assert store.get_cached_field(Arr.SONARR, 7, CacheField.URL) == "u"
        assert store.get_cached_field(Arr.SONARR, 999, CacheField.NAME) is None
        store.close()

    def test_update_cache_is_a_partial_merge(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title", "url": "u"})
        # A later update with only one field must not wipe the others.
        store.update_cache(Arr.SONARR, 7, {"coverage": "S01"})
        assert store.get_cached_name(Arr.SONARR, 7) == "Title"
        assert store.get_cached_field(Arr.SONARR, 7, CacheField.URL) == "u"
        assert store.get_cached_field(Arr.SONARR, 7, CacheField.COVERAGE) == "S01"
        store.close()

    def test_update_cache_formats_datetime_timestamp(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        assert store.get_cached_field(Arr.SONARR, 7, CacheField.UPDATED_AT) == "2021-06-05 04:03:02"
        store.close()

    def test_check_al_id_in_cache_matches_timestamp(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        assert store.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2021, 6, 5, 4, 3, 2))) is True
        # Same id, different timestamp -> stale.
        assert store.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2022, 1, 1, 0, 0, 0))) is False
        # Unknown id -> no record -> no match.
        assert store.check_al_id_in_cache(Arr.SONARR, 8, _entry(datetime(2021, 6, 5, 4, 3, 2))) is False
        store.close()

    def test_arrs_are_isolated(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "S"})
        store.update_cache(Arr.RADARR, 7, {"name": "R"})
        assert store.get_cached_name(Arr.SONARR, 7) == "S"
        assert store.get_cached_name(Arr.RADARR, 7) == "R"
        store.close()


class TestTorrentHashes:
    def test_roundtrip_preserves_none_marker(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["aaa", "bbb", None]})
        # The None marker (a hashless release) is preserved - the planner dedups on
        # its membership, so dropping it would re-grab the release. Order is free.
        assert set(store.torrent_hashes(Arr.SONARR, 7)) == {"aaa", "bbb", None}
        # Missing entry -> empty list, never None.
        assert store.torrent_hashes(Arr.SONARR, 999) == []
        store.close()

    def test_duplicate_none_markers_collapse_to_one(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": [None, "aaa", None]})
        assert store.torrent_hashes(Arr.SONARR, 7) == [None, "aaa"]
        store.close()

    def test_rewrite_replaces_the_set(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["aaa", "bbb"]})
        store.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["ccc"]})
        assert store.torrent_hashes(Arr.SONARR, 7) == ["ccc"]
        store.close()

    def test_none_marker_on_a_preexisting_db(self, tmp_path) -> None:
        # Upgrade path: a cache.db from the first release has `infohash TEXT NOT NULL`,
        # and CREATE TABLE IF NOT EXISTS will NOT alter it - so the None marker must
        # round-trip via the sentinel WITHOUT an IntegrityError, not lean on a schema
        # change that never reaches an existing db. (This test fails if torrent_hashes
        # is made nullable instead, since the old table stays NOT NULL.)
        db = str(tmp_path / "cache.db")
        raw = sqlite3.connect(db)
        raw.executescript(
            "CREATE TABLE entries (arr TEXT NOT NULL, al_id INTEGER NOT NULL, name TEXT, "
            "url TEXT, coverage TEXT, updated_at TEXT, PRIMARY KEY (arr, al_id));"
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
    def test_preview_save_on_missing_file_writes_nothing(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title"})
        store.save(preview=True)
        store.close()
        assert not (tmp_path / "cache.db").exists()

    def test_real_save_on_missing_file_creates_and_persists(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Title"})
        store.save(preview=False)
        store.close()

        assert (tmp_path / "cache.db").exists()
        reopened = _open(tmp_path)
        assert reopened.get_cached_name(Arr.SONARR, 7) == "Title"
        reopened.close()

    def test_preview_on_existing_db_does_not_mutate_committed_state(self, tmp_path) -> None:
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
        assert reopened.get_cached_name(Arr.SONARR, 7) == "Original"  # not "Changed"
        assert reopened.get_cached_name(Arr.SONARR, 8) is None  # never added
        reopened.close()


class TestAnilistMeta:
    def test_roundtrip_get_and_iter(self, tmp_path) -> None:
        store = _open(tmp_path)
        rec = {"fetched_at": "2026-06-20 12:00:00", "data": {"Media": {"id": 1}}}
        store.put_anilist_meta(1, rec)
        assert store.get_anilist_meta(1) == rec
        assert store.get_anilist_meta(999) is None
        assert dict(store.iter_anilist_meta()) == {1: rec}
        store.close()

    def test_put_overwrites(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"fetched_at": "2026-06-20 12:00:00", "data": {"a": 1}})
        store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {"a": 2}})
        meta = store.get_anilist_meta(1)
        assert meta is not None
        assert meta["data"] == {"a": 2}
        store.close()


class TestSonarrParse:
    def test_roundtrip_get_and_iter(self, tmp_path) -> None:
        store = _open(tmp_path)
        rec = {"fetched_at": "2026-06-20 12:00:00", "episodes": [{"season": 1, "episode": 2}]}
        store.put_sonarr_parse("file.mkv", rec)
        assert store.get_sonarr_parse("file.mkv") == rec
        assert store.get_sonarr_parse("missing.mkv") is None
        assert dict(store.iter_sonarr_parse()) == {"file.mkv": rec}
        store.close()


class TestPendingImports:
    def test_roundtrip_drop_and_arr_isolation(self, tmp_path) -> None:
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


class TestLegacyMigration:
    @staticmethod
    def _legacy() -> dict[str, Any]:
        return {
            "description": {"seadexarr_version": "0.0.0", "config_checksum": "old"},
            "anilist_entries": {
                "sonarr": {
                    "7": {
                        "name": "T", "url": "u", "coverage": "S01",
                        "updated_at": "2021-06-05 04:03:02",
                        "torrent_hashes": ["aaa", None, "bbb"],
                    },
                },
            },
            "anilist_meta": {
                "7": {"fetched_at": "2026-06-26 12:00:00", "data": {"Media": {"id": 7}}},
            },
            "sonarr_parse_cache": {
                "file.mkv": {"fetched_at": "2026-06-26 12:00:00", "episodes": [{"season": 1, "episode": 2}]},
                "legacy.mkv": [{"season": 1, "episode": 1}],  # pre-TTL bare-list -> skipped
            },
            "pending_imports": {
                "sonarr": {"h1": {"infohash": "h1", "series_id": 5}},
            },
        }

    def test_migration_seeds_all_blocks_and_retires_legacy(self, tmp_path) -> None:
        legacy = tmp_path / "cache.json"
        legacy.write_text(json.dumps(self._legacy()))
        db = str(tmp_path / "cache.db")

        store = CacheStore.load(db, config_checksum=CHECKSUM, migrate_from=str(legacy))
        # Seeded rows are visible before promotion (staged in the in-memory db).
        assert store.get_cached_name(Arr.SONARR, 7) == "T"
        store.save(preview=False)  # promote -> create cache.db + retire legacy
        store.close()

        assert (tmp_path / "cache.db").exists()
        assert not legacy.exists()
        assert (tmp_path / "cache.json.migrated").exists()

        reopened = CacheStore.load(db, config_checksum=CHECKSUM)
        assert reopened.get_cached_field(Arr.SONARR, 7, CacheField.URL) == "u"
        assert reopened.get_cached_field(Arr.SONARR, 7, CacheField.COVERAGE) == "S01"
        # legacy ["aaa", None, "bbb"] -> None marker preserved (de-duped, order-free)
        assert set(reopened.torrent_hashes(Arr.SONARR, 7)) == {"aaa", "bbb", None}
        assert reopened.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2021, 6, 5, 4, 3, 2)))
        meta = reopened.get_anilist_meta(7)
        assert meta is not None and meta["data"] == {"Media": {"id": 7}}
        parse = reopened.get_sonarr_parse("file.mkv")
        assert parse is not None and parse["episodes"] == [{"season": 1, "episode": 2}]
        assert reopened.get_sonarr_parse("legacy.mkv") is None  # bare-list skipped
        assert reopened.get_pending(Arr.SONARR) == {"h1": {"infohash": "h1", "series_id": 5}}
        reopened.close()

    def test_preview_migration_persists_nothing(self, tmp_path) -> None:
        legacy = tmp_path / "cache.json"
        legacy.write_text(json.dumps(self._legacy()))
        db = str(tmp_path / "cache.db")

        store = CacheStore.load(db, config_checksum=CHECKSUM, migrate_from=str(legacy))
        store.save(preview=True)
        store.close()

        # A preview never promotes: no db written, the legacy file left untouched.
        assert not (tmp_path / "cache.db").exists()
        assert legacy.exists()

    def test_failed_promote_does_not_orphan_migration(self, tmp_path) -> None:
        # If the promote rename fails (disk full / killed mid-copy), it must leave NO
        # partial cache.db and must NOT retire the legacy file - else the next load
        # sees an (empty) db, skips migrate_from, and silently abandons the legacy
        # cache + its pending imports.
        legacy = tmp_path / "cache.json"
        legacy.write_text(json.dumps(self._legacy()))
        db = str(tmp_path / "cache.db")

        store = CacheStore.load(db, config_checksum=CHECKSUM, migrate_from=str(legacy))
        with (
            mock.patch("seadexarr.modules.cache.os.replace", side_effect=OSError("boom")),
            contextlib.suppress(OSError),
        ):
            store.save(preview=False)
        store.close()

        assert not (tmp_path / "cache.db").exists()  # no 0-byte orphan
        assert legacy.exists()  # legacy not retired
        assert not list(tmp_path.glob("cache.db.promote*"))  # temp cleaned up

        # A subsequent real run re-migrates everything (the data was never lost).
        again = CacheStore.load(db, config_checksum=CHECKSUM, migrate_from=str(legacy))
        assert again.get_cached_name(Arr.SONARR, 7) == "T"
        assert again.get_pending(Arr.SONARR) == {"h1": {"infohash": "h1", "series_id": 5}}
        again.save(preview=False)
        again.close()
        assert (tmp_path / "cache.db").exists()
        assert (tmp_path / "cache.json.migrated").exists()


class TestRunLifecycle:
    """Replays the order the run loop drives a real CacheStore through (the run-loop
    tests mock cache_store, so this is the only check that the real load -> writes ->
    save(commit) -> close(rollback) -> reopen sequence behaves)."""

    def test_run_call_order_persists_and_reloads(self, tmp_path) -> None:
        db = str(tmp_path / "cache.db")

        # Run 1 (real): process one entry the way the loop does, then commit + close.
        store = CacheStore.load(db, config_checksum=CHECKSUM)
        store.update_cache(Arr.SONARR, 7, {
            "name": "Show", "url": "u", "coverage": "S01",
            "updated_at": datetime(2026, 1, 2, 3, 4, 5),
            "torrent_hashes": ["aaa", "bbb"],
        })
        store.put_anilist_meta(7, {"fetched_at": "2026-06-26 12:00:00", "data": {"id": 7}})
        store.put_pending(Arr.SONARR, "aaa", {"infohash": "aaa", "series_id": 5})
        store.save(preview=False)  # mid/end-of-run commit
        store.close()              # finally: rollback is a no-op (already committed)

        # Run 2: reopen -> cache hit, remembered hashes, carried-over pending.
        again = CacheStore.load(db, config_checksum=CHECKSUM)
        assert again.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2026, 1, 2, 3, 4, 5)))
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
    def test_evict_anilist_meta_drops_only_stale(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.put_anilist_meta(1, {"fetched_at": "2020-01-01 00:00:00", "data": {"x": 1}})
        store.put_anilist_meta(2, {"fetched_at": "2026-06-26 12:00:00", "data": {"x": 2}})
        assert store.evict_anilist_meta(datetime(2026, 6, 20)) == 1
        assert store.get_anilist_meta(1) is None
        assert store.get_anilist_meta(2) is not None
        store.close()

    def test_evict_sonarr_parse_drops_only_stale(self, tmp_path) -> None:
        store = _open(tmp_path)
        eps = [{"season": 1, "episode": 1}]
        store.put_sonarr_parse("old.mkv", {"fetched_at": "2020-01-01 00:00:00", "episodes": eps})
        store.put_sonarr_parse("new.mkv", {"fetched_at": "2026-06-26 12:00:00", "episodes": eps})
        assert store.evict_sonarr_parse(datetime(2026, 6, 20)) == 1
        assert store.get_sonarr_parse("old.mkv") is None
        assert store.get_sonarr_parse("new.mkv") is not None
        store.close()

    def test_evict_sweeps_stampless_records(self, tmp_path) -> None:
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

    def test_stats_and_integrity(self, tmp_path) -> None:
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "T", "torrent_hashes": ["a", "b"]})
        store.put_anilist_meta(1, {"fetched_at": "2026-06-26 12:00:00", "data": {}})
        store.save(preview=False)  # promote to file so size_bytes > 0

        s = store.stats()
        assert s["entries"] == 1
        assert s["torrent_hashes"] == 2
        assert s["anilist_meta"] == 1
        assert s["sonarr_parse"] == 0
        assert s["pending_imports"] == 0
        assert s["size_bytes"] > 0
        assert store.integrity_check() == "ok"
        store.close()


class TestCorruptStore:
    def test_is_corruption_distinguishes_corrupt_from_transient(self) -> None:
        # Quarantine wipes the db, so it must fire ONLY on real corruption - never on
        # a transient lock/IO error (which would destroy a healthy cache on a fluke).
        assert _is_corruption(sqlite3.DatabaseError("file is not a database")) is True
        assert _is_corruption(sqlite3.DatabaseError("database disk image is malformed")) is True
        assert _is_corruption(sqlite3.OperationalError("database is locked")) is False
        assert _is_corruption(sqlite3.OperationalError("disk I/O error")) is False

    def test_locked_healthy_db_is_not_quarantined(self, tmp_path) -> None:
        # A healthy db whose load() hits a non-corruption DatabaseError must NOT be
        # quarantined: the error propagates (fail closed) and the file is untouched.
        store = _open(tmp_path)
        store.update_cache(Arr.SONARR, 7, {"name": "Keep"})
        store.save(preview=False)
        store.close()
        db = tmp_path / "cache.db"

        # Simulate a transient lock at open time (e.g. the WAL switch hitting BUSY).
        with mock.patch(
            "seadexarr.modules.cache._connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            raised = False
            try:
                CacheStore.load(str(db), config_checksum=CHECKSUM)
            except sqlite3.OperationalError:
                raised = True

        assert raised  # propagated, not swallowed into a quarantine
        assert db.exists()  # healthy db left in place
        assert not list(tmp_path.glob("cache.db.corrupt-*"))  # never quarantined

    def test_corrupt_db_is_quarantined_and_recovered(self, tmp_path) -> None:
        db = tmp_path / "cache.db"
        db.write_text("this is not a sqlite database")  # torn-write stand-in

        # Must NOT raise - fail open to a fresh store.
        store = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert store.get_cached_name(Arr.SONARR, 7) is None
        store.update_cache(Arr.SONARR, 7, {"name": "Recovered"})
        store.save(preview=False)
        store.close()

        # The corrupt file was moved aside; a fresh, working db took its place.
        assert len(list(tmp_path.glob("cache.db.corrupt-*"))) == 1
        reopened = CacheStore.load(str(db), config_checksum=CHECKSUM)
        assert reopened.get_cached_name(Arr.SONARR, 7) == "Recovered"
        reopened.close()
