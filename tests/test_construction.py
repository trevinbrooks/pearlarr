"""Construction-seam tests: the REAL ``__init__`` + ``begin_run`` rebind.

The rest of the suite builds the engine/strategy via ``make_bare_instance``
(``object.__new__``), which bypasses ``__init__``. These drive the real
``SeaDexArr`` / ``SonarrSync`` constructors off a hand-built ``RunDeps`` so the
collaborator wiring and the ``begin_run`` two-phase rebind have an in-suite guard
(previously only an offline smoke).
"""

from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_sonarr import SonarrSync

from .builders import make_run_deps


def test_engine_begin_run_rebinds_all_ctx_collaborators() -> None:
    # reset_run_stats swaps a fresh RunContext AND rebinds every ctx-holding
    # collaborator to it; a missed rebind would route a collaborator's writes into the
    # orphaned prior context (the drift the begin_run fold guards against).
    engine = SeaDexArr(make_run_deps(), Arr.SONARR)
    holders = [engine._filter, engine._grab_pipeline, engine._wait_manager]
    assert all(c._ctx is engine._ctx for c in holders)  # __init__'s placeholder bind

    first_ctx = engine._ctx
    engine.reset_run_stats(arr=Arr.SONARR, dry_run=False)

    assert engine._ctx is not first_ctx  # a fresh ctx was swapped in
    assert all(c._ctx is engine._ctx for c in holders)  # ...and all rebound to it


def test_sonarr_sync_init_shares_cache_store_for_staged_writes() -> None:
    # The parse-cache writer (SonarrParseCache) and the seed reader (ImportReconciler)
    # must share the strat's cache_store by identity, or a staged parse write would not
    # be visible to build_pending_seeds.
    deps = make_run_deps()
    engine = SeaDexArr(deps, Arr.SONARR)
    # The SonarrClient validates its connection on construction; stub it so the test
    # exercises the (network-independent) collaborator wiring, not a live Sonarr.
    with mock.patch("seadexarr.modules.seadex_sonarr.SonarrClient"):
        strat = SonarrSync(deps, engine)

    assert strat._parse.cache_store is deps.cache_store
    assert strat._reconciler.cache_store is deps.cache_store
