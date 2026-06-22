"""Characterization tests for ``CacheStore``.

Pins the cache schema, freshness checks, and persistence (incl. the preview
write-gate) moved out of ``SeaDexArr`` in Phase 2 (see ``REFACTOR_PLAN.md``).
"""

import json
from datetime import datetime
from typing import Any

import seadexarr
from seadexarr.modules.cache import CacheStore, save_json
from seadexarr.modules.config import AppConfig


def _config(tmp_path, body: bytes = b"public_only: true\n") -> AppConfig:
    """An ``AppConfig`` backed by a real (tiny) file so ``checksum()`` works."""

    path = tmp_path / "config.yml"
    path.write_bytes(body)
    return AppConfig(path=str(path), arr="sonarr", data={})


def _entry(dt: datetime) -> Any:
    """A stand-in SeaDex entry exposing only ``updated_at`` (typed Any so it
    satisfies the ``EntryRecord`` parameter without a real record)."""

    class _Entry:
        updated_at = dt

    return _Entry()


class TestSchemaAndReconcile:
    def test_fresh_schema_on_missing_file(self, tmp_path) -> None:
        cfg = _config(tmp_path)
        store = CacheStore.load(str(tmp_path / "cache.json"), cfg)
        assert set(store.data) == {"description", "anilist_entries"}
        assert store.data["description"]["seadexarr_version"] == seadexarr.__version__
        assert store.data["description"]["config_checksum"] == cfg.checksum()
        assert store.data["anilist_entries"] == {}

    def test_reconcile_updates_stale_version(self, tmp_path) -> None:
        cfg = _config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({
            "description": {"seadexarr_version": "0.0.0", "config_checksum": cfg.checksum()},
            "anilist_entries": {},
        }))
        store = CacheStore.load(str(cache_path), cfg)
        assert store.data["description"]["seadexarr_version"] == seadexarr.__version__

    def test_reconcile_updates_changed_checksum(self, tmp_path) -> None:
        cfg = _config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({
            "description": {"seadexarr_version": seadexarr.__version__, "config_checksum": "stale"},
            "anilist_entries": {},
        }))
        store = CacheStore.load(str(cache_path), cfg)
        assert store.data["description"]["config_checksum"] == cfg.checksum()


class TestRecords:
    def test_update_cache_creates_nested_and_formats_timestamp(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), _config(tmp_path))
        store.update_cache("sonarr", 7, {"name": "Title", "updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        rec = store.data["anilist_entries"]["sonarr"]["7"]
        assert rec["name"] == "Title"
        assert rec["updated_at"] == "2021-06-05 04:03:02"

    def test_get_cached_field_and_name(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), _config(tmp_path))
        store.update_cache("sonarr", 7, {"name": "Title", "url": "u"})
        assert store.get_cached_name("sonarr", 7) == "Title"
        assert store.get_cached_field("sonarr", 7, "url") == "u"
        assert store.get_cached_field("sonarr", 7, "missing") is None
        assert store.get_cached_field("sonarr", 999, "name") is None

    def test_check_al_id_in_cache_matches_timestamp(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), _config(tmp_path))
        store.update_cache("sonarr", 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        assert store.check_al_id_in_cache("sonarr", 7, _entry(datetime(2021, 6, 5, 4, 3, 2))) is True
        # Same id, different timestamp -> stale.
        assert store.check_al_id_in_cache("sonarr", 7, _entry(datetime(2022, 1, 1, 0, 0, 0))) is False
        # Unknown id -> no record -> no match.
        assert store.check_al_id_in_cache("sonarr", 8, _entry(datetime(2021, 6, 5, 4, 3, 2))) is False


class TestPersistence:
    def test_save_skips_during_preview(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        store = CacheStore.load(str(cache_path), _config(tmp_path))  # load() never writes
        store.save(preview=True)
        assert not cache_path.exists()

    def test_save_persists_when_not_preview(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        store = CacheStore.load(str(cache_path), _config(tmp_path))
        store.save(preview=False)
        assert cache_path.exists()
        on_disk = json.loads(cache_path.read_text())
        assert on_disk["description"]["seadexarr_version"] == seadexarr.__version__

    def test_save_json_sorts_entries_by_int_id(self, tmp_path) -> None:
        out = tmp_path / "c.json"
        save_json({"anilist_entries": {"sonarr": {"10": {}, "2": {}, "1": {}}}}, str(out), sort_cache=True)
        on_disk = json.loads(out.read_text())
        assert list(on_disk["anilist_entries"]["sonarr"].keys()) == ["1", "2", "10"]
