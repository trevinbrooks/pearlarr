"""Views and writes over one arr's durable pending-import rows plus the run list."""

from collections.abc import Iterable, Mapping
from typing import Any

from .cache import AbstractCacheStore
from .config import Arr
from .manual_import import GuardFacts, ImportProbe, PendingImport, hydrate_pending, is_awaiting_cleanup
from .reporter import RunContext


class PendingRecords:
    """One seam for the pending-import store: raw and hydrated views, run-list-aware writes.

    Binds the cache store once. The run context (whose `arr` scopes every read and
    whose `pending_imports` is the run list) arrives via `begin_run` each run.
    Every key is a torrent's infohash.
    """

    _ctx: RunContext
    """The current run's context. Never bound at construction: `begin_run` rebinds it every run."""

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def begin_run(self, ctx: RunContext) -> None:
        """Bind the fresh run context the views and writes read."""

        self._ctx = ctx

    def fresh_keys(self) -> set[str]:
        """Keys of the records written THIS run, tallied as `added` and never carried-over."""

        return set(self._ctx.pending_imports)

    def rows(self) -> dict[str, dict[str, Any]]:
        """Every stored row, raw."""

        return self._store.get_pending(self._ctx.arr)

    def rows_for_series(self, series_id: int) -> dict[str, dict[str, Any]]:
        """The raw rows with a claim on `series_id` (the store's own filter, not a full-table read)."""

        return self._store.get_pending_for_series(self._ctx.arr, series_id)

    def for_series(self, series_id: int, guards: Mapping[int, GuardFacts]) -> list[PendingImport]:
        """The records claiming `series_id`, rehydrated under `guards` (the rows the caller already read)."""

        return list(hydrate_pending(self.rows_for_series(series_id), guards).values())

    def flagged(self) -> dict[str, dict[str, Any]]:
        """The cleanup-flagged rows, raw (the heal pass's working set)."""

        return {key: raw for key, raw in self.rows().items() if is_awaiting_cleanup(raw)}

    def active(self) -> dict[str, dict[str, Any]]:
        """Raw rows minus this-run grabs and cleanup-flagged leftovers."""

        fresh = self.fresh_keys()
        return {key: raw for key, raw in self.rows().items() if key not in fresh and not is_awaiting_cleanup(raw)}

    def hydrate(self, rows: Mapping[str, dict[str, Any]]) -> dict[str, PendingImport]:
        """Rehydrate `rows`, each claim fed its entry's guard row (an empty `rows` skips the guard read)."""

        if not rows:
            return {}
        return hydrate_pending(rows, self.guards())

    def guards(self) -> dict[int, GuardFacts]:
        """The arr's guard rows for the entries with live records (one read, shared by a poll's hydrations)."""

        return self._store.get_guards(self._ctx.arr)

    def active_records(self) -> dict[str, PendingImport]:
        """The `active` rows rehydrated: the carried-over working set of the end-of-run passes."""

        return self.hydrate(self.active())

    def stored_records(self, hashes: Iterable[str]) -> dict[str, PendingImport]:
        """The store-resident records among `hashes`, rehydrated under one guards read."""

        rows = {
            infohash: raw
            for infohash in dict.fromkeys(hashes)
            if (raw := self._store.get_pending_record(self._ctx.arr, infohash)) is not None
        }
        return self.hydrate(rows)

    def insert_fresh(self, record: PendingImport) -> None:
        """Persist a this-run grab and enter it in the run list (it tallies as `added`)."""

        self._put(record)
        self._ctx.pending_imports[record.infohash] = record

    def save(self, record: PendingImport) -> None:
        """Persist ONE record, refreshing any run-list copy but NEVER inserting one.

        A run-list upsert would silently convert a reacquire or accretion into a fresh
        grab, skewing the carried-over tally and the heal's recount.
        """

        self._put(record)
        if record.infohash in self._ctx.pending_imports:
            self._ctx.pending_imports[record.infohash] = record

    def _put(self, record: PendingImport) -> None:
        """The store write plus every claim's guard row (Sonarr only: Radarr's import reads no guards)."""

        self._store.put_pending(self._ctx.arr, record.infohash, record.to_json())
        if self._ctx.arr is Arr.SONARR:
            for claim in record.claims:
                self._store.put_guards(self._ctx.arr, claim.al_id, claim.guards)

    def absorb_probe(self, record: PendingImport, probe: ImportProbe) -> PendingImport:
        """Persist a poll's import-time placements and exclusions onto the record and return the healed copy.

        Writes only when the healed record differs from the stored one.
        """

        healed = record.with_placements(probe.placements).with_exclusions(probe.exclusions)
        if healed == record:
            return record
        self.save(healed)
        return healed

    def drop(self, infohash: str) -> None:
        """Remove one torrent's record from the store and the run list."""

        self._store.drop_pending(self._ctx.arr, infohash)
        self._ctx.pending_imports.pop(infohash, None)

    def other_arr_holds(self, infohash: str) -> bool:
        """Whether the other arr still holds a record on the torrent (the category move defers on it)."""

        return self._store.other_arr_holds(self._ctx.arr, infohash)
