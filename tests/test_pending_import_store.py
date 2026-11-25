# pyright: strict
"""Tests for the pending import store."""

from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.pending_import_store import PendingImportStore

from .builders import FakeCacheStore, pending_import, title_claim


def test_dropping_a_pending_import_takes_it_out_of_the_store_and_the_run_list() -> None:
    keep = pending_import(infohash="keep")
    drop = pending_import(infohash="drop")
    store = FakeCacheStore()
    state = ArrRunState(arr=Arr.SONARR)
    pending_imports = PendingImportStore(store)
    pending_imports.bind_arr_run(state)
    for grabbed in (keep, drop):
        pending_imports.insert_grabbed(grabbed, title_claim())

    pending_imports.drop("drop")

    assert set(store.get_pending_import_rows(Arr.SONARR)) == {"keep"}
    assert state.grabbed_this_run == {"keep": keep}
