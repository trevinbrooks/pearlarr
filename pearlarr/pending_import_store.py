"""Views and writes over one arr's durable pending-import rows plus the run list."""

from collections.abc import Iterable, Mapping
from typing import Any

from .arr_run_state import ArrRunState
from .cache import AbstractCacheStore
from .config import Arr
from .pending_import import OwnedAtGrab, PendingImport, TitleClaim, is_awaiting_cleanup, load_pending_imports
from .wait_types import AttemptResult


class PendingImportStore:
    """One interface for the pending-import store: raw and loaded views, run-list-aware writes.

    Binds the cache store once. The arr run's context (whose `arr` every read filters on and whose
    `grabbed_this_run` is the run list) arrives via `bind_arr_run` each arr run. Every key is a torrent's hash.
    """

    _state: ArrRunState
    """The arr run's context. Never bound at construction: `bind_arr_run` rebinds it every arr run."""

    def __init__(self, cache_store: AbstractCacheStore) -> None:
        self._store = cache_store

    def bind_arr_run(self, state: ArrRunState) -> None:
        """Bind the fresh arr run context the views and writes read."""

        self._state = state

    def grabbed_keys(self) -> set[str]:
        """Keys of the pending imports written THIS run, tallied as `grabbed` and never carried-over."""

        return set(self._state.grabbed_this_run)

    def rows(self) -> dict[str, dict[str, Any]]:
        """Every stored row, raw."""

        return self._store.get_pending_import_rows(self._state.arr)

    def rows_for_series(self, series_id: int) -> dict[str, dict[str, Any]]:
        """The raw rows with a claim on `series_id` (a filter in the store, not a full-table read)."""

        return self._store.get_pending_import_rows_for_series(self._state.arr, series_id)

    def imports_for_series(self, series_id: int, guards: Mapping[int, OwnedAtGrab]) -> dict[str, PendingImport]:
        """The pending imports claiming `series_id`, keyed by hash, loaded under the guards the caller already read."""

        return load_pending_imports(self.rows_for_series(series_id), guards)

    def awaiting_cleanup_rows(self) -> dict[str, dict[str, Any]]:
        """The rows awaiting cleanup, raw (the working set for finishing their cleanup)."""

        return {key: raw for key, raw in self.rows().items() if is_awaiting_cleanup(raw)}

    def active_rows(self) -> dict[str, dict[str, Any]]:
        """Raw rows minus this-run grabs and rows awaiting cleanup."""

        grabbed = self.grabbed_keys()
        return {key: raw for key, raw in self.rows().items() if key not in grabbed and not is_awaiting_cleanup(raw)}

    def load(self, rows: Mapping[str, dict[str, Any]]) -> dict[str, PendingImport]:
        """Load `rows`, each claim fed its title's guard row (an empty `rows` skips the guard read)."""

        if not rows:
            return {}
        return load_pending_imports(rows, self.guards())

    def guards(self) -> dict[int, OwnedAtGrab]:
        """The arr's guard rows for the titles with live pending imports (one read, shared by a poll's loads)."""

        return self._store.get_guards(self._state.arr)

    def active_imports(self) -> dict[str, PendingImport]:
        """The `active` rows loaded: the carried-over working set of the wait passes."""

        return self.load(self.active_rows())

    def stored_pending_imports(self, hashes: Iterable[str]) -> dict[str, PendingImport]:
        """The stored pending imports among `hashes`, loaded under one guards read."""

        rows = {
            infohash: raw
            for infohash in dict.fromkeys(hashes)
            if (raw := self._store.get_pending_import_row(self._state.arr, infohash)) is not None
        }
        return self.load(rows)

    def insert_grabbed(self, pending: PendingImport, claim: TitleClaim) -> None:
        """Persist a this-run grab with `claim`'s guard row and enter it in the run list (it tallies as `grabbed`)."""

        self._put(pending)
        self._put_guards(claim)
        self._state.grabbed_this_run[pending.infohash] = pending

    def save_with_claim(self, pending: PendingImport, claim: TitleClaim) -> None:
        """`save` + `claim`'s guard row: the write of a grab whose claim joins or refreshes a stored pending import."""

        self.save(pending)
        self._put_guards(claim)

    def save(self, pending: PendingImport) -> None:
        """Persist ONE pending import (no guard row), refreshing any run-list copy but NEVER inserting one.

        A run-list upsert would silently convert a reacquire or a joined claim into a fresh
        grab, skewing the carried-over tally and the recount when a cleanup finishes.
        """

        self._put(pending)
        if pending.infohash in self._state.grabbed_this_run:
            self._state.grabbed_this_run[pending.infohash] = pending

    def _put(self, pending: PendingImport) -> None:
        """The pending import's row write."""

        self._store.put_pending_import_row(self._state.arr, pending.infohash, pending.to_json())

    def _put_guards(self, claim: TitleClaim) -> None:
        """The claim's guard row (Sonarr only: Radarr's import reads no guards).

        Written only by the grab that produced the evidence, never re-put from a loaded copy that
        may trail a newer write.
        """

        if self._state.arr is Arr.SONARR:
            self._store.put_guards(self._state.arr, claim.al_id, claim.guards)

    def save_history_placements(self, pending: PendingImport, result: AttemptResult) -> PendingImport:
        """Persist the files a poll's Sonarr history placed onto the pending import and return the updated copy.

        Writes only when the poll added something: the pending import hands itself back on a no-op.
        """

        healed = pending.with_placements(result.placements)
        if healed is pending:
            return pending
        self.save(healed)
        return healed

    def drop(self, infohash: str) -> None:
        """Remove one torrent's pending import from the store and the run list."""

        self._store.drop_pending_import_row(self._state.arr, infohash)
        self._state.grabbed_this_run.pop(infohash, None)

    def drop_unimported(self, pending: PendingImport) -> None:
        """Drop the pending import of a torrent that left qBittorrent unimported, so the next run checks its titles.

        A title is cached once its torrent is added, so each claiming title is uncached. The check grabs the release
        again when the arr still lacks it.
        """

        self.drop(pending.infohash)
        for claim in pending.claims:
            self._store.uncache_title(self._state.arr, claim.al_id)

    def other_arr_holds(self, infohash: str) -> bool:
        """Whether the other arr still has a pending import on the torrent (the category move is put off for it)."""

        return self._store.other_arr_holds(self._state.arr, infohash)
