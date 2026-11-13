# pyright: strict
"""Parity guard: `tests/builders.FakeCacheStore` must observably match `CacheStore`.

~50 tests trust `FakeCacheStore` as a drop-in for the SQLite `CacheStore`, yet
nothing checks the two agree. A semantic drift - TTL eviction, the cached title
merge, the JSONB round-trip, the pending series filter - would
silently invalidate every test that leans on the fake. This
drives one identical op sequence through both and asserts every observable read is
equal, then the eviction counts and a post-drop/post-save re-read.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pearlarr.cache import AbstractCacheStore, CachedTitleUpdate, CacheStore, HistoryCheckpoint
from pearlarr.config import Arr
from pearlarr.pending_import import OwnedAtGrab
from pearlarr.tvdb_orders import CachedOrders, OrderRow

from .builders import FakeCacheStore, pending_import, title_claim

# fetched_at stamps straddling the eviction cutoff (STAMP_FORMAT strings).
_OLD = "2020-01-01 00:00:00"
_NEW = "2030-01-01 00:00:00"
_CUTOFF = datetime(2025, 1, 1)
_CACHED_TITLE_STAMP = datetime(2026, 1, 1)


@contextmanager
def _both_stores(tmp_path: Path) -> Generator[tuple[FakeCacheStore, CacheStore]]:
    """The fake and a real SQLite store side by side, both closed on exit."""

    fake = FakeCacheStore()
    real = CacheStore.load(str(tmp_path / "cache.db"), config_hash="x")
    try:
        yield fake, real
    finally:
        real.close()
        fake.close()


def _apply_ops(store: AbstractCacheStore) -> None:
    """One identical mutating sequence: cached titles, the two TTL caches, pending imports."""

    store.update_cached_title(
        Arr.SONARR,
        7,
        CachedTitleUpdate(
            name="Show",
            url="https://releases.moe/7",
            coverage="S1",
            updated_at=_CACHED_TITLE_STAMP,
            fallback_satisfied=True,
            season_type="dvd",
        ),
    )
    # Partial re-update of the SAME cached title (omits name/url/updated_at/marker): pins the
    # partial merge (absent columns untouched - incl. the fallback marker, still True).
    store.update_cached_title(Arr.SONARR, 7, CachedTitleUpdate(coverage="S2"))
    # A cross-arr row. No fallback_satisfied key, pinning the default-False parity.
    store.update_cached_title(Arr.RADARR, 99, CachedTitleUpdate(name="Movie"))
    # A row whose stamp is cleared loses its timestamp and nothing else. Clearing an id with no row adds none, and
    # neither does an empty update.
    store.update_cached_title(Arr.SONARR, 8, CachedTitleUpdate(name="Cleared", updated_at=_CACHED_TITLE_STAMP))
    store.uncache_title(Arr.SONARR, 8)
    store.uncache_title(Arr.SONARR, 4242)
    store.update_cached_title(Arr.SONARR, 4242, CachedTitleUpdate())

    store.put_anilist_meta(7, {"fetched_at": _NEW, "data": {"title": "Show"}})
    store.put_anilist_meta(8, {"fetched_at": _OLD, "data": {"title": "Old"}})
    store.put_anilist_meta(9, {"data": {"title": "Stampless"}})  # unreadable stamp -> swept

    store.put_sonarr_parse("fresh.mkv", {"fetched_at": _NEW, "episodes": [1]})
    store.put_sonarr_parse("stale.mkv", {"fetched_at": _OLD, "episodes": [2]})

    # hashA carries TWO claims spanning two series: the series filter and the guard
    # join must both walk every claim, in the fake exactly as in SQLite's json_each.
    store.put_pending_import_row(
        Arr.SONARR,
        "hashA",
        pending_import(
            infohash="hashA",
            claims=(title_claim(al_id=7, series_id=7, title="A"), title_claim(al_id=9, series_id=8, title="A2")),
        ).to_json(),
    )
    store.put_pending_import_row(
        Arr.SONARR, "hashB", pending_import(infohash="hashB", al_id=8, series_id=8, title="B").to_json()
    )
    # A Radarr row on hashA, so `other_arr_holds` answers differently per arr: a
    # fake ignoring the arr narrowing cannot match both.
    store.put_pending_import_row(
        Arr.RADARR, "hashA", pending_import(infohash="hashA", al_id=7, series_id=0, title="AR").to_json()
    )

    # Guard rows: a same-id re-put (latest wins, al 7 has a live claim), al 9 live
    # only through hashA's second claim, plus an orphan sonarr row and a radarr row
    # with no claim, both stored but filtered out of every read (get_guards joins
    # live claims). The parity test's later drop of hashA re-orphans 7 and 9 too.
    store.put_guards(Arr.SONARR, 7, OwnedAtGrab(entry_groups=("Old",)))
    store.put_guards(Arr.SONARR, 7, OwnedAtGrab(entry_groups=("New",), stale_groups=("Old",)))
    store.put_guards(Arr.SONARR, 9, OwnedAtGrab(entry_groups=("Second",)))
    store.put_guards(Arr.SONARR, 4242, OwnedAtGrab(entry_groups=("Orphan",)))
    store.put_guards(Arr.RADARR, 99, OwnedAtGrab(entry_groups=("MovieGrp",)))

    # History checkpoint: an initial write then an upsert (only the last survives).
    store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-01T00:00:00Z", 5))
    store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("2026-07-02T00:00:00Z", 9))

    # Selection hash: an initial store then an upsert (only the last survives).
    store.store_selection_hash(Arr.SONARR, "digest-old")
    store.store_selection_hash(Arr.SONARR, "digest-new")

    # TVDB orders: a row replaced whole (the dvd order goes), an empty listed order, a series with no orders.
    row = (OrderRow(1, 1, 101, 1),)
    store.put_tvdb_orders(5, CachedOrders(_OLD, _OLD, {"official": row, "dvd": row}, {"official": 1, "dvd": 1}))
    store.put_tvdb_orders(5, CachedOrders(_NEW, _NEW, {"official": row, "alt": ()}, {"official": 1}))
    store.put_tvdb_orders(6, CachedOrders(_NEW, _NEW, {}, None))


def _observe(store: AbstractCacheStore) -> dict[str, object]:
    """Every observable read the fake-trusting tests rely on, as one comparable snapshot."""

    stats = store.stats()
    return {
        "entry_s7": store.get_cached_title(Arr.SONARR, 7),
        "entry_r99": store.get_cached_title(Arr.RADARR, 99),
        "entry_missing": store.get_cached_title(Arr.SONARR, 4242),
        "entry_cleared": store.get_cached_title(Arr.SONARR, 8),
        "entry_wrong_arr": store.get_cached_title(Arr.RADARR, 7),
        "anilist_7": store.get_anilist_meta(7),
        "anilist_all": dict(store.iter_anilist_meta()),
        "sonarr_parse_fresh": store.get_sonarr_parse("fresh.mkv"),
        "sonarr_parse_stale": store.get_sonarr_parse("stale.mkv"),
        "pending_sonarr": store.get_pending_import_rows(Arr.SONARR),
        "pending_record_hasha": store.get_pending_import_row(Arr.SONARR, "hashA"),
        "pending_record_wrong_arr": store.get_pending_import_row(Arr.RADARR, "hashB"),
        "pending_record_missing": store.get_pending_import_row(Arr.SONARR, "nope"),
        "pending_series7": store.get_pending_import_rows_for_series(Arr.SONARR, 7),
        "pending_series8": store.get_pending_import_rows_for_series(Arr.SONARR, 8),
        "pending_series_missing": store.get_pending_import_rows_for_series(Arr.SONARR, 4242),
        "other_arr_sonarr": store.other_arr_holds(Arr.SONARR, "hashA"),
        "other_arr_radarr": store.other_arr_holds(Arr.RADARR, "hashA"),
        "other_arr_own_only": store.other_arr_holds(Arr.SONARR, "hashB"),
        "other_arr_case": store.other_arr_holds(Arr.SONARR, "HASHA"),
        "other_arr_missing": store.other_arr_holds(Arr.SONARR, "nope"),
        "guards_sonarr": store.get_guards(Arr.SONARR),
        "guards_radarr": store.get_guards(Arr.RADARR),
        "checkpoint_sonarr": store.get_history_checkpoint(Arr.SONARR),
        "checkpoint_radarr": store.get_history_checkpoint(Arr.RADARR),
        "selection_current": store.selection_changed(Arr.SONARR, "digest-new"),
        "selection_moved": store.selection_changed(Arr.SONARR, "digest-old"),
        "selection_unvouched": store.selection_changed(Arr.RADARR, "digest-new"),
        "tvdb_orders_5": store.get_tvdb_orders(5),
        "tvdb_orders_6": store.get_tvdb_orders(6),
        "tvdb_orders_missing": store.get_tvdb_orders(4242),
        "stats_no_size": stats._replace(size_bytes=0),
        "integrity": store.integrity_check(),
    }


@dataclass(frozen=True)
class _JsonbBlock:
    """One whole-dict JSONB block's put/get/iter surface, for the isolation check.

    `get` returns the single row under this block's fixed key (or None).
    `iter_rows` returns every row reachable through the block's collection
    reads (for `pending` that is both `get_pending_import_rows` and `get_pending_import_rows_for_series`).
    """

    name: str
    put: Callable[[AbstractCacheStore, dict[str, Any]], None]
    get: Callable[[AbstractCacheStore], dict[str, Any] | None]
    iter_rows: Callable[[AbstractCacheStore], list[dict[str, Any]]]


_ISO_KEY = "hiso"
_ISO_AL = 4242
_ISO_FILE = "iso.mkv"
_ISO_SID = 7


def _sonarr_parse_rows(store: AbstractCacheStore) -> list[dict[str, Any]]:
    """The sonarr_parse block's collection read (per-filename - it has no iterator)."""

    rec = store.get_sonarr_parse(_ISO_FILE)
    return [] if rec is None else [rec]


# The three blocks the real store round-trips through JSON on both ends. Each carries
# a claim on `_ISO_SID` so the pending block's get_pending_import_rows_for_series filter matches the row.
# guard_facts is deliberately absent: the scribble check needs raw mutable dicts and
# `OwnedAtGrab` is frozen (isolation by construction), so `_apply_ops` covers it instead.
_JSONB_BLOCKS: tuple[_JsonbBlock, ...] = (
    _JsonbBlock(
        "pending",
        put=lambda s, r: s.put_pending_import_row(Arr.SONARR, _ISO_KEY, r),
        get=lambda s: s.get_pending_import_row(Arr.SONARR, _ISO_KEY),
        iter_rows=lambda s: [
            *s.get_pending_import_rows(Arr.SONARR).values(),
            *s.get_pending_import_rows_for_series(Arr.SONARR, _ISO_SID).values(),
        ],
    ),
    _JsonbBlock(
        "anilist_meta",
        put=lambda s, r: s.put_anilist_meta(_ISO_AL, r),
        get=lambda s: s.get_anilist_meta(_ISO_AL),
        iter_rows=lambda s: [rec for _al, rec in s.iter_anilist_meta()],
    ),
    _JsonbBlock(
        "sonarr_parse",
        put=lambda s, r: s.put_sonarr_parse(_ISO_FILE, r),
        get=lambda s: s.get_sonarr_parse(_ISO_FILE),
        iter_rows=_sonarr_parse_rows,
    ),
)


def _scribble(rec: dict[str, Any], marker: str) -> None:
    """Mutate a row in place: top-level, nested, and a fresh key.

    Any leak into the store (a shallow copy OR a shared reference) surfaces on the next read.
    """

    rec["title"] = marker
    rec["nested"]["k"] = marker
    rec["injected"] = True


def _assert_not_mutated(rec: dict[str, Any] | None, block: str) -> dict[str, Any]:
    """The row is present and none of `_scribble`'s mutations reached it."""

    assert rec is not None, block
    assert rec["title"] == "orig", block
    assert rec["nested"]["k"] == "v", block
    assert "injected" not in rec, block
    return rec


def _assert_block_snapshot_isolated(store: AbstractCacheStore, block: _JsonbBlock) -> None:
    """A JSONB row is caller-mutation-isolated on both ends, like the real store's json round-trip.

    Mutating the dict handed to `put` afterwards, or any dict returned by `get` / `iter`, must not reach the store.
    """

    row: dict[str, Any] = {"claims": [{"series_id": _ISO_SID}], "title": "orig", "nested": {"k": "v"}}
    block.put(store, row)

    # write side: mutate the caller's own dict after the put.
    _scribble(row, "leaked-via-put")
    after_put = _assert_not_mutated(block.get(store), block.name)

    # read side (get): mutate the returned copy.
    _scribble(after_put, "leaked-via-get")
    _assert_not_mutated(block.get(store), block.name)

    # read side (iter): mutate each iterated row.
    for rec in block.iter_rows(store):
        _scribble(rec, "leaked-via-iter")
    for rec in block.iter_rows(store):
        _assert_not_mutated(rec, block.name)


def test_jsonb_row_snapshot_mutation_does_not_leak(tmp_path: Path) -> None:
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

        fake.drop_pending_import_row(Arr.SONARR, "hashA")
        real.drop_pending_import_row(Arr.SONARR, "hashA")

        # save is a no-op for the fake and stages+promotes for the real. Neither must
        # disturb the observable reads that follow.
        fake.save(dry_run=False)
        real.save(dry_run=False)

        assert _observe(fake) == _observe(real)
