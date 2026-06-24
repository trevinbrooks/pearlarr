"""Characterization tests for ``CacheStore``.

Pins the cache schema, freshness checks, and persistence (incl. the preview
write-gate) moved out of ``SeaDexArr`` during the refactor.
"""

import json
from datetime import datetime
from typing import Any

import seadexarr
from seadexarr.modules.cache import CacheField, CacheStore, save_json
from seadexarr.modules.config import Arr

# Stand-in for a config-file checksum. ``CacheStore`` only stamps and compares
# the value it is handed; it never computes one, so any string works here.
CHECKSUM = "0123456789abcdef0123456789abcdef"


def _entry(dt: datetime) -> Any:
    """A stand-in SeaDex entry exposing only ``updated_at`` (typed Any so it
    satisfies the ``EntryRecord`` parameter without a real record)."""

    class _Entry:
        updated_at = dt

    return _Entry()


class TestSchemaAndReconcile:
    def test_fresh_schema_on_missing_file(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), config_checksum=CHECKSUM)
        assert set(store.data) == {"description", "anilist_entries"}
        assert store.data["description"]["seadexarr_version"] == seadexarr.__version__
        assert store.data["description"]["config_checksum"] == CHECKSUM
        assert store.data["anilist_entries"] == {}

    def test_reconcile_updates_stale_version(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({
            "description": {"seadexarr_version": "0.0.0", "config_checksum": CHECKSUM},
            "anilist_entries": {},
        }))
        store = CacheStore.load(str(cache_path), config_checksum=CHECKSUM)
        assert store.data["description"]["seadexarr_version"] == seadexarr.__version__

    def test_reconcile_updates_changed_checksum(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({
            "description": {"seadexarr_version": seadexarr.__version__, "config_checksum": "stale"},
            "anilist_entries": {},
        }))
        store = CacheStore.load(str(cache_path), config_checksum=CHECKSUM)
        assert store.data["description"]["config_checksum"] == CHECKSUM


class TestRecords:
    def test_update_cache_creates_nested_and_formats_timestamp(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), config_checksum=CHECKSUM)
        store.update_cache(Arr.SONARR, 7, {"name": "Title", "updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        rec = store.data["anilist_entries"]["sonarr"]["7"]
        assert rec["name"] == "Title"
        assert rec["updated_at"] == "2021-06-05 04:03:02"

    def test_get_cached_field_and_name(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), config_checksum=CHECKSUM)
        store.update_cache(Arr.SONARR, 7, {"name": "Title", "url": "u"})
        assert store.get_cached_name(Arr.SONARR, 7) == "Title"
        assert store.get_cached_field(Arr.SONARR, 7, CacheField.URL) == "u"
        assert store.get_cached_field(Arr.SONARR, 999, CacheField.NAME) is None

    def test_check_al_id_in_cache_matches_timestamp(self, tmp_path) -> None:
        store = CacheStore.load(str(tmp_path / "cache.json"), config_checksum=CHECKSUM)
        store.update_cache(Arr.SONARR, 7, {"updated_at": datetime(2021, 6, 5, 4, 3, 2)})
        assert store.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2021, 6, 5, 4, 3, 2))) is True
        # Same id, different timestamp -> stale.
        assert store.check_al_id_in_cache(Arr.SONARR, 7, _entry(datetime(2022, 1, 1, 0, 0, 0))) is False
        # Unknown id -> no record -> no match.
        assert store.check_al_id_in_cache(Arr.SONARR, 8, _entry(datetime(2021, 6, 5, 4, 3, 2))) is False


class TestPersistence:
    def test_save_skips_during_preview(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        store = CacheStore.load(str(cache_path), config_checksum=CHECKSUM)  # load() never writes
        store.save(preview=True)
        assert not cache_path.exists()

    def test_save_persists_when_not_preview(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.json"
        store = CacheStore.load(str(cache_path), config_checksum=CHECKSUM)
        store.save(preview=False)
        assert cache_path.exists()
        on_disk = json.loads(cache_path.read_text())
        assert on_disk["description"]["seadexarr_version"] == seadexarr.__version__

    def test_save_json_sorts_entries_by_int_id(self, tmp_path) -> None:
        out = tmp_path / "c.json"
        save_json({"anilist_entries": {"sonarr": {"10": {}, "2": {}, "1": {}}}}, str(out), sort_cache=True)
        on_disk = json.loads(out.read_text())
        assert list(on_disk["anilist_entries"]["sonarr"].keys()) == ["1", "2", "10"]
