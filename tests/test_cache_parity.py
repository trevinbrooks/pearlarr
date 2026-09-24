# pyright: strict
"""Parity guard: `tests/builders.FakeCacheStore` must observably match `CacheStore`.

~50 tests trust `FakeCacheStore` as a drop-in for the SQLite `CacheStore`, yet
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

from pearlarr.cache import AbstractCacheStore, CacheRecord, CacheStore, HistoryCheckpoint
from pearlarr.config import Arr
from pearlarr.manual_import import GuardFacts

from .builders import FakeCacheStore, entry_claim, make_entry_record, pending_import

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
            fallback_satisfied=True,
        ),
    )
    # Partial re-update of the SAME entry (omits name/url/updated_at/marker): pins the
    # scalar merge (absent columns untouched - incl. the fallback marker, still True)
    # + the hash-set replace (not append).
    store.update_cache(Arr.SONARR, 7, CacheRecord(coverage="S2", torrent_hashes=[None, "hc", "ha"]))
    # A cross-arr row + a duplicate/None hash list, to exercise the dedup + None
    # marker. No fallback_satisfied key, pinning the default-False parity.
    store.update_cache(Arr.RADARR, 99, CacheRecord(name="Movie", torrent_hashes=["z", "z", None]))

    store.put_anilist_meta(7, {"fetched_at": _NEW, "data": {"title": "Show"}})
    store.put_anilist_meta(8, {"fetched_at": _OLD, "data": {"title": "Old"}})
    store.put_anilist_meta(9, {"data": {"title": "Stampless"}})  # unreadable stamp -> swept

    store.put_sonarr_parse("fresh.mkv", {"fetched_at": _NEW, "episodes": [1]})
    store.put_sonarr_parse("stale.mkv", {"fetched_at": _OLD, "episodes": [2]})

    # hashA carries TWO claims spanning two series: the series filter and the guard
    # join must both walk every claim, in the fake exactly as in SQLite's json_each.
    store.put_pending(
        Arr.SONARR,
        "hashA",
        pending_import(
            infohash="hashA",
            claims=(entry_claim(al_id=7, series_id=7, title="A"), entry_claim(al_id=9, series_id=8, title="A2")),
        ).to_json(),
    )
    store.put_pending(Arr.SONARR, "hashB", pending_import(infohash="hashB", al_id=8, series_id=8, title="B").to_json())
    # A Radarr record on hashA, so `other_arr_holds` answers differently per arr: a
    # fake ignoring the arr narrowing cannot match both.
    store.put_pending(Arr.RADARR, "hashA", pending_import(infohash="hashA", al_id=7, series_id=0, title="AR").to_json())

    # Guard rows: a same-id re-put (latest wins, al 7 has a live claim), al 9 live
    # only through hashA's second claim, plus an orphan sonarr row and a radarr row
    # with no claim, both stored but filtered out of every read (get_guards joins
    # live claims). The parity test's later drop of hashA re-orphans 7 and 9 too.
    store.put_guards(Arr.SONARR, 7, GuardFacts(entry_groups=("Old",)))
    store.put_guards(Arr.SONARR, 7, GuardFacts(entry_groups=("New",), stale_groups=("Old",)))
    store.put_guards(Arr.SONARR, 9, GuardFacts(entry_groups=("Second",)))
    store.put_guards(Arr.SONARR, 4242, GuardFacts(entry_groups=("Orphan",)))
    store.put_guards(Arr.RADARR, 99, GuardFacts(entry_groups=("MovieGrp",)))

    # History checkpoint: an initial write then an upsert (only the last survives).
    store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T00:00:00Z", 5))
    store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-02T00:00:00Z", 9))

    # Selection digest: an initial vouch then an upsert (only the last survives).
    store.vouch_selection(Arr.SONARR, "digest-old")
    store.vouch_selection(Arr.SONARR, "digest-new")


def _observe(store: AbstractCacheStore) -> dict[str, object]:
    """Every observable read the fake-trusting tests rely on, as one comparable snapshot."""

    match_entry = make_entry_record(anilist_id=7, updated_at=_ENTRY_STAMP)
    stats = store.stats()
    return {
        "entry_s7": store.get_entry(Arr.SONARR, 7),
        "entry_r99": store.get_entry(Arr.RADARR, 99),
        "entry_missing": store.get_entry(Arr.SONARR, 4242),
        "hashes_s7": store.torrent_hashes(Arr.SONARR, 7),
        "hashes_r99": store.torrent_hashes(Arr.RADARR, 99),
        "check_match": store.check_al_id_in_cache(Arr.SONARR, 7, match_entry),
        "check_wrong_arr": store.check_al_id_in_cache(Arr.RADARR, 7, match_entry),
        "anilist_7": store.get_anilist_meta(7),
        "anilist_all": dict(store.iter_anilist_meta()),
        "sonarr_parse_fresh": store.get_sonarr_parse("fresh.mkv"),
        "sonarr_parse_stale": store.get_sonarr_parse("stale.mkv"),
        "pending_sonarr": store.get_pending(Arr.SONARR),
        "pending_record_hasha": store.get_pending_record(Arr.SONARR, "hashA"),
        "pending_record_wrong_arr": store.get_pending_record(Arr.RADARR, "hashB"),
        "pending_record_missing": store.get_pending_record(Arr.SONARR, "nope"),
        "pending_series7": store.get_pending_for_series(Arr.SONARR, 7),
        "pending_series8": store.get_pending_for_series(Arr.SONARR, 8),
        "pending_series_missing": store.get_pending_for_series(Arr.SONARR, 4242),
        "has_pending_hasha": store.has_pending(Arr.SONARR, "hashA"),
        "has_pending_missing": store.has_pending(Arr.SONARR, "nope"),
        "other_arr_sonarr": store.other_arr_holds(Arr.SONARR, "hashA"),
        "other_arr_radarr": store.other_arr_holds(Arr.RADARR, "hashA"),
        "other_arr_own_only": store.other_arr_holds(Arr.SONARR, "hashB"),
        "other_arr_case": store.other_arr_holds(Arr.SONARR, "HASHA"),
        "other_arr_missing": store.other_arr_holds(Arr.SONARR, "nope"),
        "guards_sonarr": store.get_guards(Arr.SONARR),
        "guards_radarr": store.get_guards(Arr.RADARR),
        "checkpoint_sonarr": store.get_history_checkpoint(Arr.SONARR),
        "checkpoint_radarr": store.get_history_checkpoint(Arr.RADARR),
        "selection_current": store.selection_stale(Arr.SONARR, "digest-new"),
        "selection_moved": store.selection_stale(Arr.SONARR, "digest-old"),
        "selection_unvouched": store.selection_stale(Arr.RADARR, "digest-new"),
        "own_ids_sonarr": store.own_download_ids(Arr.SONARR),
        "own_ids_radarr": store.own_download_ids(Arr.RADARR),
        "stats_no_size": stats._replace(size_bytes=0),
        "integrity": store.integrity_check(),
    }


@dataclass(frozen=True)
class _JsonbBlock:
    """One whole-dict JSONB block's put/get/iter surface, for the isolation check.

    `get` returns the single record under this block's fixed key (or None).
    `iter_records` returns every record reachable through the block's collection
    reads (for `pending` that is both `get_pending` and `get_pending_for_series`).
    """

    name: str
    put: Callable[[AbstractCacheStore, dict[str, Any]], None]
    get: Callable[[AbstractCacheStore], dict[str, Any] | None]
    iter_records: Callable[[AbstractCacheStore], list[dict[str, Any]]]


_ISO_KEY = "hiso"
_ISO_AL = 4242
_ISO_FILE = "iso.mkv"
_ISO_SID = 7


def _sonarr_parse_records(store: AbstractCacheStore) -> list[dict[str, Any]]:
    """The sonarr_parse block's collection read (per-filename - it has no iterator)."""

    rec = store.get_sonarr_parse(_ISO_FILE)
    return [] if rec is None else [rec]


# The three blocks the real store round-trips through JSON on both ends. Each carries
# a claim on `_ISO_SID` so the pending block's get_pending_for_series filter matches the record.
# guard_facts is deliberately absent: the scribble check needs raw mutable dicts and
# `GuardFacts` is frozen (isolation by construction), so `_apply_ops` covers it instead.
_JSONB_BLOCKS: tuple[_JsonbBlock, ...] = (
    _JsonbBlock(
        "pending",
        put=lambda s, r: s.put_pending(Arr.SONARR, _ISO_KEY, r),
        get=lambda s: s.get_pending_record(Arr.SONARR, _ISO_KEY),
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
        iter_records=_sonarr_parse_records,
    ),
)


def _scribble(rec: dict[str, Any], marker: str) -> None:
    """Mutate a record in place: top-level, nested, and a fresh key.

    Any leak into the store (a shallow copy OR a shared reference) surfaces on the next read.
    """

    rec["title"] = marker
    rec["nested"]["k"] = marker
    rec["injected"] = True


def _assert_pristine(rec: dict[str, Any] | None, block: str) -> dict[str, Any]:
    """The record is present and none of `_scribble`'s mutations reached it."""

    assert rec is not None, block
    assert rec["title"] == "orig", block
    assert rec["nested"]["k"] == "v", block
    assert "injected" not in rec, block
    return rec


def _assert_block_snapshot_isolated(store: AbstractCacheStore, block: _JsonbBlock) -> None:
    """A JSONB record is caller-mutation-isolated on both ends, like the real store's json round-trip.

    Mutating the dict handed to `put` afterwards, or any dict returned by `get` / `iter`, must not reach the store.
    """

    record: dict[str, Any] = {"claims": [{"series_id": _ISO_SID}], "title": "orig", "nested": {"k": "v"}}
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
    """Every JSONB block isolates caller mutation on both ends.

    The real store (the json round-trip) is driven through the same assertion, so it
    proves the contract rather than an invented one - and pins the fake to it.
    """

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

        # save is a no-op for the fake and stages+promotes for the real. Neither must
        # disturb the observable reads that follow.
        fake.save(preview=False)
        real.save(preview=False)

        assert _observe(fake) == _observe(real)
