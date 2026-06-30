# pyright: strict
"""Parity guard: ``tests/builders.FakeCacheStore`` must observably match ``CacheStore``.

~50 tests trust ``FakeCacheStore`` as a drop-in for the SQLite ``CacheStore``, yet
nothing checks the two agree. A semantic drift - TTL eviction, the
entries/torrent_hashes split + ordering, the JSONB round-trip, the pending
series filter - would silently invalidate every test that leans on the fake. This
drives one identical op sequence through both and asserts every observable read is
equal, then the eviction counts and a post-drop/post-save re-read.
"""

from datetime import datetime
from pathlib import Path

from seadexarr.modules.cache import CacheField, CacheRecord, CacheStore, CacheStoreProtocol
from seadexarr.modules.config import Arr

from .builders import FakeCacheStore, make_entry_record

# fetched_at stamps straddling the eviction cutoff (UPDATED_AT_STR_FORMAT strings).
_OLD = "2020-01-01 00:00:00"
_NEW = "2030-01-01 00:00:00"
_CUTOFF = datetime(2025, 1, 1)
_ENTRY_STAMP = datetime(2026, 1, 1)


def _apply_ops(store: CacheStoreProtocol) -> None:
    """One identical mutating sequence: entries + hashes, the two TTL caches, pending."""

    store.update_cache(
        Arr.SONARR,
        7,
        CacheRecord(
            name="Show",
            url="https://releases.moe/7",
            coverage="S1",
            updated_at=_ENTRY_STAMP,
            torrent_hashes=[None, "hb", "ha"],
        ),
    )
    # A cross-arr row + a duplicate/None hash list, to exercise the dedup + None marker.
    store.update_cache(Arr.RADARR, 99, CacheRecord(name="Movie", torrent_hashes=["z", "z", None]))

    store.put_anilist_meta(7, {"fetched_at": _NEW, "data": {"title": "Show"}})
    store.put_anilist_meta(8, {"fetched_at": _OLD, "data": {"title": "Old"}})
    store.put_anilist_meta(9, {"data": {"title": "Stampless"}})  # unreadable stamp -> swept

    store.put_sonarr_parse("fresh.mkv", {"fetched_at": _NEW, "episodes": [1]})
    store.put_sonarr_parse("stale.mkv", {"fetched_at": _OLD, "episodes": [2]})

    store.put_pending(Arr.SONARR, "hashA", {"series_id": 7, "title": "A"})
    store.put_pending(Arr.SONARR, "hashB", {"series_id": 8, "title": "B"})


def _observe(store: CacheStoreProtocol) -> dict[str, object]:
    """Every observable read the fake-trusting tests rely on, as one comparable snapshot."""

    match_entry = make_entry_record(anilist_id=7, updated_at=_ENTRY_STAMP)
    stats = store.stats()
    return {
        "entry_s7": store.get_entry(Arr.SONARR, 7),
        "entry_missing": store.get_entry(Arr.SONARR, 4242),
        "name_s7": store.get_cached_name(Arr.SONARR, 7),
        "url_s7": store.get_cached_field(Arr.SONARR, 7, CacheField.URL),
        "coverage_s7": store.get_cached_field(Arr.SONARR, 7, CacheField.COVERAGE),
        "hashes_s7": store.torrent_hashes(Arr.SONARR, 7),
        "hashes_r99": store.torrent_hashes(Arr.RADARR, 99),
        "check_match": store.check_al_id_in_cache(Arr.SONARR, 7, match_entry),
        "check_wrong_arr": store.check_al_id_in_cache(Arr.RADARR, 7, match_entry),
        "anilist_7": store.get_anilist_meta(7),
        "anilist_all": dict(store.iter_anilist_meta()),
        "sonarr_parse_fresh": store.get_sonarr_parse("fresh.mkv"),
        "sonarr_parse_all": dict(store.iter_sonarr_parse()),
        "pending_sonarr": store.get_pending(Arr.SONARR),
        "pending_series7": store.get_pending_for_series(Arr.SONARR, 7),
        "stats_no_size": {k: v for k, v in stats.items() if k != "size_bytes"},
        "integrity": store.integrity_check(),
    }


def test_fake_cache_store_observably_matches_real(tmp_path: Path) -> None:
    fake = FakeCacheStore()
    real = CacheStore.load(str(tmp_path / "cache.db"), config_checksum="x")
    try:
        _apply_ops(fake)
        _apply_ops(real)
        assert _observe(fake) == _observe(real)

        # TTL eviction parity: OLD + stampless go, NEW stays (anilist 2 dropped, parse 1).
        assert fake.evict_anilist_meta(_CUTOFF) == real.evict_anilist_meta(_CUTOFF) == 2
        assert fake.evict_sonarr_parse(_CUTOFF) == real.evict_sonarr_parse(_CUTOFF) == 1

        fake.drop_pending(Arr.SONARR, "hashA")
        real.drop_pending(Arr.SONARR, "hashA")

        # save is a no-op for the fake and stages+promotes for the real; neither must
        # disturb the observable reads that follow.
        fake.save(preview=False)
        real.save(preview=False)

        assert _observe(fake) == _observe(real)
    finally:
        real.close()
        fake.close()
