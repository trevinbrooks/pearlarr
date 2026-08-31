"""Views and writes over one arr's durable pending-import rows plus the run list."""

from typing import Any

from .cache import AbstractCacheStore
from .manual_import import PendingImport, PendingKey, hydrate_pending, is_awaiting_cleanup
from .reporter import RunContext


class PendingRecords:
    """One seam for the pending-import store: raw and hydrated views, run-list-aware writes.

    Binds the cache store once. The run context (whose `arr` scopes every read and
    whose `pending_imports` is the run list) arrives via `begin_run` each run.
    """

    _ctx: RunContext
    """The current run's context. Never bound at construction: `begin_run` rebinds it every run."""

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context the views and writes read."""

        self._ctx = ctx

    def fresh_keys(self) -> set[PendingKey]:
        """Keys of the records written THIS run, tallied as `added` and never carried-over."""

        return set(self._ctx.pending_imports)

    def rows(self) -> dict[PendingKey, dict[str, Any]]:
        """Every stored row, raw."""

        return self._store.get_pending(self._ctx.arr)

    def has(self, key: PendingKey) -> bool:
        """Whether the store holds a record under `key`."""

        return self._store.has_pending(self._ctx.arr, key)

    def flagged(self) -> dict[PendingKey, dict[str, Any]]:
        """The cleanup-flagged rows, raw (the heal pass's working set)."""

        return {key: raw for key, raw in self.rows().items() if is_awaiting_cleanup(raw)}

    def active(self) -> dict[PendingKey, dict[str, Any]]:
        """Raw rows minus this-run grabs and cleanup-flagged leftovers."""

        fresh = self.fresh_keys()
        return {key: raw for key, raw in self.rows().items() if key not in fresh and not is_awaiting_cleanup(raw)}

    def hydrate(self, rows: dict[PendingKey, dict[str, Any]]) -> dict[PendingKey, PendingImport]:
        """Rehydrate `rows`, each record fed its entry's guard row (an empty `rows` skips the guard read)."""

        if not rows:
            return {}
        return hydrate_pending(rows, self._store.get_guards(self._ctx.arr))

    def active_records(self) -> dict[PendingKey, PendingImport]:
        """The `active` rows rehydrated: the carried-over working set of the end-of-run passes."""

        return self.hydrate(self.active())

    def for_series(self, series_id: int) -> dict[PendingKey, PendingImport]:
        """One series' rows rehydrated, cleanup-flagged rows excluded (the heal pass owns those).

        Guard feeds must NOT use this filtered read: a flagged record's files are on disk,
        so sonarr_import's trusted-groups read hydrates the unfiltered rows itself.
        """

        rows = {
            key: raw
            for key, raw in self._store.get_pending_for_series(self._ctx.arr, series_id).items()
            if not is_awaiting_cleanup(raw)
        }
        return self.hydrate(rows)

    def insert_fresh(self, pending: PendingImport) -> None:
        """Persist a this-run grab and enter it in the run list (it tallies as `added`)."""

        self._store.put_pending(self._ctx.arr, pending.key, pending.to_json())
        self._ctx.pending_imports[pending.key] = pending

    def save(self, pending: PendingImport) -> None:
        """Persist ONE record, refreshing any run-list copy but NEVER inserting one.

        A run-list upsert would silently convert a reacquire or demote into a fresh
        grab, skewing the carried-over tally and the heal's recount.
        """

        self._store.put_pending(self._ctx.arr, pending.key, pending.to_json())
        if pending.key in self._ctx.pending_imports:
            self._ctx.pending_imports[pending.key] = pending

    def drop(self, key: PendingKey) -> None:
        """Remove ONE record (`PendingKey`-scoped, never its siblings) from the store and the run list."""

        self._store.drop_pending(self._ctx.arr, key)
        self._ctx.pending_imports.pop(key, None)

    def count_arr_siblings(self, key: PendingKey) -> int:
        """This arr's OTHER claims on `key`'s torrent."""

        return self._store.count_arr_siblings(self._ctx.arr, key)

    def count_siblings_any_arr(self, key: PendingKey) -> int:
        """Both arrs' OTHER claims on `key`'s torrent."""

        return self._store.count_siblings_any_arr(self._ctx.arr, key)
