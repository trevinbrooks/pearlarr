# pyright: strict
"""Parity guard: ``tests/builders.FakeCacheStore`` must observably match ``CacheStore``.

~50 tests trust ``FakeCacheStore`` as a drop-in for the SQLite ``CacheStore``, yet
nothing checks the two agree. A semantic drift - TTL eviction, the
entries/torrent_hashes split + ordering, the JSONB round-trip, the pending
series filter - would silently invalidate every test that leans on the fake. This
drives one identical op sequence through both and asserts every observable read is
equal, then the eviction counts and a post-drop/post-save re-read.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from seadexarr.modules.cache import AbstractCacheStore, CacheField, CacheRecord, CacheStore
from seadexarr.modules.config import Arr

from .builders import FakeCacheStore, make_entry_record

# fetched_at stamps straddling the eviction cutoff (UPDATED_AT_STR_FORMAT strings).
_OLD = "2020-01-01 00:00:00"
_NEW = "2030-01-01 00:00:00"
_CUTOFF = datetime(2025, 1, 1)
_ENTRY_STAMP = datetime(2026, 1, 1)


@contextmanager
def _both_stores(tmp_path: Path) -> Generator[tuple[FakeCacheStore, CacheStore]]:
    """The fake and a real SQLite store side by side, both closed on exit."""

    fake = FakeCacheStore()
    real = CacheStore.load(str(tmp_path / "cache.db"), config_checksum="x")
    try:
        yield fake, real
    finally:
        real.close()
        fake.close()


def _apply_ops(store: AbstractCacheStore) -> None:
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
    # Partial re-update of the SAME entry (omits name/url/updated_at): pins the scalar
    # merge (absent columns untouched, not clobbered) + the hash-set replace (not append).
    store.update_cache(Arr.SONARR, 7, CacheRecord(coverage="S2", torrent_hashes=[None, "hc", "ha"]))
    # A cross-arr row + a duplicate/None hash list, to exercise the dedup + None marker.
    store.update_cache(Arr.RADARR, 99, CacheRecord(name="Movie", torrent_hashes=["z", "z", None]))

    store.put_anilist_meta(7, {"fetched_at": _NEW, "data": {"title": "Show"}})
    store.put_anilist_meta(8, {"fetched_at": _OLD, "data": {"title": "Old"}})
    store.put_anilist_meta(9, {"data": {"title": "Stampless"}})  # unreadable stamp -> swept

    store.put_sonarr_parse("fresh.mkv", {"fetched_at": _NEW, "episodes": [1]})
    store.put_sonarr_parse("stale.mkv", {"fetched_at": _OLD, "episodes": [2]})

    store.put_pending(Arr.SONARR, "hashA", {"series_id": 7, "title": "A"})
    store.put_pending(Arr.SONARR, "hashB", {"series_id": 8, "title": "B"})


def _observe(store: AbstractCacheStore) -> dict[str, object]:
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


@dataclass(frozen=True)
class _JsonbBlock:
    """One whole-dict JSONB block's put/get/iter surface, for the isolation check.

    ``get`` returns the single record under this block's fixed key (or None);
    ``iter_records`` returns every record reachable through the block's collection
    reads (for ``pending`` that is both ``get_pending`` and ``get_pending_for_series``).
    """

    name: str
    put: Callable[[AbstractCacheStore, dict[str, Any]], None]
    get: Callable[[AbstractCacheStore], dict[str, Any] | None]
    iter_records: Callable[[AbstractCacheStore], list[dict[str, Any]]]


_ISO_IH = "hiso"
_ISO_AL = 4242
_ISO_FILE = "iso.mkv"
_ISO_SID = 7

# The three blocks the real store round-trips through JSON on both ends. Each carries
# series_id so the pending block's get_pending_for_series filter matches the record.
_JSONB_BLOCKS: tuple[_JsonbBlock, ...] = (
    _JsonbBlock(
        "pending",
        put=lambda s, r: s.put_pending(Arr.SONARR, _ISO_IH, r),
        get=lambda s: s.get_pending(Arr.SONARR).get(_ISO_IH),
        iter_records=lambda s: [
            *s.get_pending(Arr.SONARR).values(),
            *s.get_pending_for_series(Arr.SONARR, _ISO_SID).values(),
        ],
    ),
    _JsonbBlock(
        "anilist_meta",
        put=lambda s, r: s.put_anilist_meta(_ISO_AL, r),
        get=lambda s: s.get_anilist_meta(_ISO_AL),
        iter_records=lambda s: [rec for _al, rec in s.iter_anilist_meta()],
    ),
    _JsonbBlock(
        "sonarr_parse",
        put=lambda s, r: s.put_sonarr_parse(_ISO_FILE, r),
        get=lambda s: s.get_sonarr_parse(_ISO_FILE),
        iter_records=lambda s: [rec for _fn, rec in s.iter_sonarr_parse()],
    ),
)


def _scribble(rec: dict[str, Any], marker: str) -> None:
    """Mutate a record in place - top-level, nested, and a fresh key - so any leak
    into the store (a shallow copy OR a shared reference) surfaces on the next read."""

    rec["title"] = marker
    rec["nested"]["k"] = marker
    rec["injected"] = True


def _assert_pristine(rec: dict[str, Any] | None, block: str) -> dict[str, Any]:
    """The record is present and none of :func:`_scribble`'s mutations reached it."""

    assert rec is not None, block
    assert rec["title"] == "orig", block
    assert rec["nested"]["k"] == "v", block
    assert "injected" not in rec, block
    return rec


def _assert_block_snapshot_isolated(store: AbstractCacheStore, block: _JsonbBlock) -> None:
    """A JSONB record is caller-mutation-isolated on BOTH ends, like the real store's
    json.dumps (put) / json.loads (get, iter) round-trip: mutating the dict handed to
    ``put`` afterwards, or any dict returned by ``get`` / ``iter``, must not reach the
    store."""

    record: dict[str, Any] = {"series_id": _ISO_SID, "title": "orig", "nested": {"k": "v"}}
    block.put(store, record)

    # write side: mutate the caller's own dict after the put.
    _scribble(record, "leaked-via-put")
    after_put = _assert_pristine(block.get(store), block.name)

    # read side (get): mutate the returned copy.
    _scribble(after_put, "leaked-via-get")
    _assert_pristine(block.get(store), block.name)

    # read side (iter): mutate each iterated record.
    for rec in block.iter_records(store):
        _scribble(rec, "leaked-via-iter")
    for rec in block.iter_records(store):
        _assert_pristine(rec, block.name)


def test_jsonb_record_snapshot_mutation_does_not_leak(tmp_path: Path) -> None:
    """Every JSONB block isolates caller mutation on both ends. The real store (the
    json round-trip) is driven through the same assertion, so it proves the contract
    rather than an invented one - and pins the fake to it."""

    with _both_stores(tmp_path) as (fake, real):
        for block in _JSONB_BLOCKS:
            _assert_block_snapshot_isolated(fake, block)
            _assert_block_snapshot_isolated(real, block)


def test_fake_cache_store_observably_matches_real(tmp_path: Path) -> None:
    with _both_stores(tmp_path) as (fake, real):
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
