# pyright: strict
# pyright: reportPrivateUsage=false
# These assert on the loop/hub/strategy's private wiring (_filter / _ctx / _parse /
# _reconciler), which strict re-flags; the repo disables reportPrivateUsage for tests.
"""Construction-seam tests: the REAL ``__init__`` + ``begin_run`` rebind.

The rest of the suite builds the hub/loop/strategy via ``make_bare_instance``
(``object.__new__``), which bypasses ``__init__``. These drive the real
``RunServices`` / ``SeaDexArr`` / ``SonarrSync`` constructors off a hand-built
``RunDeps`` so the collaborator wiring and the ``begin_run`` two-phase rebind
have an in-suite guard (previously only an offline smoke).
"""

from seadexarr.modules.config import Arr
from seadexarr.modules.run_services import RunServices
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync

from .builders import make_run_deps
from .fakes import FakeRadarrClient, FakeSonarrClient


def test_runner_adopts_placeholder_then_rebinds_fresh_ctx() -> None:
    deps = make_run_deps()
    services = RunServices(deps, Arr.SONARR)
    runner = SeaDexArr(deps, services)

    # Phase 1 (adoption): the runner adopts the hub's single placeholder - no
    # second mint - and every ctx-holding collaborator is bound to that object.
    holders = [services._filter, services._grab_pipeline, runner._wait_manager]
    assert runner._ctx is services.ctx
    assert all(c._ctx is runner._ctx for c in holders)

    # Phase 2 (fresh mint): reset_run_stats swaps a fresh RunContext AND rebinds
    # every ctx-holding collaborator to it; a missed rebind would route a
    # collaborator's writes into the orphaned prior context (the drift the
    # begin_run fold guards against).
    first_ctx = runner._ctx
    runner.reset_run_stats(dry_run=False)

    assert runner._ctx is not first_ctx  # a fresh ctx was swapped in
    assert services.ctx is runner._ctx  # the hub rebound to it
    assert all(c._ctx is runner._ctx for c in holders)  # ...and all five agree


def test_sonarr_sync_init_shares_cache_store_for_staged_writes() -> None:
    # The parse-cache writer (SonarrParseCache) and the seed reader (ImportReconciler)
    # must share the strat's cache_store by identity, or a staged parse write would not
    # be visible to build_pending_seeds.
    deps = make_run_deps()
    services = RunServices(deps, Arr.SONARR)
    # The real SonarrClient validates its connection on construction; inject a typed
    # fake through the sonarr_client seam so the (network-independent) collaborator
    # wiring runs off the REAL __init__ - an incomplete fake here is a pyright error
    # and un-instantiable, unlike the old stringly-typed monkeypatch.
    strat = SonarrSync(deps, services, sonarr_client=FakeSonarrClient())

    assert strat._parse.cache_store is deps.cache_store
    assert strat._reconciler.cache_store is deps.cache_store


def test_radarr_sync_init_builds_without_network_via_client_seam() -> None:
    # The real RadarrClient hits the network on construction (arrapi fetches system
    # status), so RadarrSync's REAL __init__ was untestable before the radarr_client
    # seam; the injected fake also skips require_connection, so no keys are needed.
    deps = make_run_deps()
    services = RunServices(deps, Arr.RADARR)
    fake = FakeRadarrClient()

    strat = RadarrSync(deps, services, radarr_client=fake)

    assert strat.radarr is fake
    assert strat._mappings is deps.mappings
    assert strat.anibridge is deps.mappings.anibridge
