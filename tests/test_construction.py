# pyright: strict
# pyright: reportPrivateUsage=false
# These assert on the engine/strategy's private wiring (_filter / _ctx / _parse /
# _reconciler), which strict re-flags; the repo disables reportPrivateUsage for tests.
"""Construction-seam tests: the REAL ``__init__`` + ``begin_run`` rebind.

The rest of the suite builds the engine/strategy via ``make_bare_instance``
(``object.__new__``), which bypasses ``__init__``. These drive the real
``SeaDexArr`` / ``SonarrSync`` constructors off a hand-built ``RunDeps`` so the
collaborator wiring and the ``begin_run`` two-phase rebind have an in-suite guard
(previously only an offline smoke).
"""

from seadexarr.modules.config import Arr
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync

from .builders import make_run_deps
from .fakes import FakeRadarrClient, FakeSonarrClient


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
    # The real SonarrClient validates its connection on construction; inject a typed
    # fake through the sonarr_client seam so the (network-independent) collaborator
    # wiring runs off the REAL __init__ - an incomplete fake here is a pyright error
    # and un-instantiable, unlike the old stringly-typed monkeypatch.
    strat = SonarrSync(deps, engine, sonarr_client=FakeSonarrClient())

    assert strat._parse.cache_store is deps.cache_store
    assert strat._reconciler.cache_store is deps.cache_store


def test_radarr_sync_init_builds_without_network_via_client_seam() -> None:
    # The real RadarrClient hits the network on construction (arrapi fetches system
    # status), so RadarrSync's REAL __init__ was untestable before the radarr_client
    # seam; the injected fake also skips require_connection, so no keys are needed.
    deps = make_run_deps()
    engine = SeaDexArr(deps, Arr.RADARR)
    fake = FakeRadarrClient()

    strat = RadarrSync(deps, engine, radarr_client=fake)

    assert strat.radarr is fake
    assert strat._mappings is deps.mappings
    assert strat.anibridge is deps.mappings.anibridge
