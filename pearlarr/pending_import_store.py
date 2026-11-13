"""Views and writes over one arr's durable pending-import rows plus the run list."""

from collections.abc import Iterable, Mapping
from typing import Any

from .arr_run_state import ArrRunState
from .cache import AbstractCacheStore
from .config import Arr
from .pending_import import OwnedAtGrab, PendingImport, TitleClaim, is_awaiting_cleanup, load_pending_imports
from .wait_types import AttemptResult


class PendingImportStore:
    """One seam for the pending-import store: raw and hydrated views, run-list-aware writes.

    Binds the cache store once. The run context (whose `arr` scopes every read and whose
    `pending_imports` is the run list) arrives via `begin_run` each run. Every key is a torrent's infohash.
    """

    _ctx: ArrRunState
    """The current run's context. Never bound at construction: `begin_run` rebinds it every run."""

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the fresh run context the views and writes read."""

        self._ctx = state

    def fresh_keys(self) -> set[str]:
        """Keys of the records written THIS run, tallied as `added` and never carried-over."""

        return set(self._ctx.grabbed_this_run)

    def rows(self) -> dict[str, dict[str, Any]]:
        """Every stored row, raw."""

        return self._store.get_pending_import_rows(self._ctx.arr)

    def rows_for_series(self, series_id: int) -> dict[str, dict[str, Any]]:
        """The raw rows with a claim on `series_id` (the store's own filter, not a full-table read)."""

        return self._store.get_pending_import_rows_for_series(self._ctx.arr, series_id)

    def imports_for_series(self, series_id: int, guards: Mapping[int, OwnedAtGrab]) -> dict[str, PendingImport]:
        """The records claiming `series_id`, keyed by infohash, rehydrated under the guards the caller already read."""

        return load_pending_imports(self.rows_for_series(series_id), guards)

    def flagged(self) -> dict[str, dict[str, Any]]:
        """The cleanup-flagged rows, raw (the heal pass's working set)."""

        return {key: raw for key, raw in self.rows().items() if is_awaiting_cleanup(raw)}

    def active_rows(self) -> dict[str, dict[str, Any]]:
        """Raw rows minus this-run grabs and cleanup-flagged leftovers."""

        fresh = self.fresh_keys()
        return {key: raw for key, raw in self.rows().items() if key not in fresh and not is_awaiting_cleanup(raw)}

    def load(self, rows: Mapping[str, dict[str, Any]]) -> dict[str, PendingImport]:
        """Rehydrate `rows`, each claim fed its entry's guard row (an empty `rows` skips the guard read)."""

        if not rows:
            return {}
        return load_pending_imports(rows, self.guards())

    def guards(self) -> dict[int, OwnedAtGrab]:
        """The arr's guard rows for the entries with live records (one read, shared by a poll's hydrations)."""

        return self._store.get_guards(self._ctx.arr)

    def active_imports(self) -> dict[str, PendingImport]:
        """The `active` rows rehydrated: the carried-over working set of the end-of-run passes."""

        return self.load(self.active_rows())

    def stored_pending_imports(self, hashes: Iterable[str]) -> dict[str, PendingImport]:
        """The store-resident records among `hashes`, rehydrated under one guards read."""

        rows = {
            infohash: raw
            for infohash in dict.fromkeys(hashes)
            if (raw := self._store.get_pending_import_row(self._ctx.arr, infohash)) is not None
        }
        return self.load(rows)

    def insert_fresh(self, pending: PendingImport, claim: TitleClaim) -> None:
        """Persist a this-run grab with `claim`'s guard row and enter it in the run list (it tallies as `added`)."""

        self._put(pending)
        self._put_guards(claim)
        self._ctx.grabbed_this_run[pending.infohash] = pending

    def save_with_claim(self, pending: PendingImport, claim: TitleClaim) -> None:
        """`save` plus `claim`'s guard row: the write of a grab whose claim joins or refreshes a stored record."""

        self.save(pending)
        self._put_guards(claim)

    def save(self, pending: PendingImport) -> None:
        """Persist ONE record (no guard row), refreshing any run-list copy but NEVER inserting one.

        A run-list upsert would silently convert a reacquire or accretion into a fresh
        grab, skewing the carried-over tally and the heal's recount.
        """

        self._put(pending)
        if pending.infohash in self._ctx.grabbed_this_run:
            self._ctx.grabbed_this_run[pending.infohash] = pending

    def _put(self, pending: PendingImport) -> None:
        """The record's row write."""

        self._store.put_pending_import_row(self._ctx.arr, pending.infohash, pending.to_json())

    def _put_guards(self, claim: TitleClaim) -> None:
        """The claim's guard row (Sonarr only: Radarr's import reads no guards).

        Written only by the grab that produced the evidence, never re-put from a hydrated copy that
        may trail a newer write.
        """

        if self._ctx.arr is Arr.SONARR:
            self._store.put_guards(self._ctx.arr, claim.al_id, claim.guards)

    def save_history_placements(self, pending: PendingImport, probe: AttemptResult) -> PendingImport:
        """Persist the files a poll's Sonarr history placed onto the record and return the healed copy.

        Writes only when the poll added something: the record hands itself back on a no-op.
        """

        healed = pending.with_placements(probe.placements)
        if healed is pending:
            return pending
        self.save(healed)
        return healed

    def drop(self, infohash: str) -> None:
        """Remove one torrent's record from the store and the run list."""

        self._store.drop_pending_import_row(self._ctx.arr, infohash)
        self._ctx.grabbed_this_run.pop(infohash, None)

    def drop_unimported(self, pending: PendingImport) -> None:
        """Remove the record of a torrent that left qBittorrent unimported, and have the next run check its titles.

        A title is cached as done once its torrent is added, so each claiming title's stamp is cleared. The check
        grabs the release again when the arr still lacks it.
        """

        self.drop(pending.infohash)
        for claim in pending.claims:
            self._store.uncache_title(self._ctx.arr, claim.al_id)

    def other_arr_holds(self, infohash: str) -> bool:
        """Whether the other arr still holds a record on the torrent (the category move defers on it)."""

        return self._store.other_arr_holds(self._ctx.arr, infohash)
