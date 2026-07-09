# pyright: strict
# pyright: reportPrivateUsage=false
# These assert on the loop/hub/strategy's private wiring (_filter / _ctx / _parse /
# _reconciler), which strict re-flags; the repo disables reportPrivateUsage for tests.
"""Construction-seam tests: the REAL ``__init__`` + ``begin_run`` rebind.

The rest of the suite builds the hub/loop/strategy via ``make_bare_instance``
(``object.__new__``), which bypasses ``__init__``. These drive the real
``RunServices`` / ``RunLoop`` / ``SonarrSync`` constructors off a hand-built
``RunDeps`` so the collaborator wiring and the ``begin_run`` two-phase rebind
have an in-suite guard (previously only an offline smoke).
"""

import logging
from pathlib import Path

import httpx
import pytest

from seadexarr.modules import run_services
from seadexarr.modules.boot_flow import BootFlow
from seadexarr.modules.config import Arr
from seadexarr.modules.mappings import MappingResolver
from seadexarr.modules.run_loop import RunLoop
from seadexarr.modules.run_services import RunDeps, RunServices
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync

from .builders import make_bare_instance, make_config, make_run_deps
from .fakes import FakeRadarrClient, FakeSonarrClient


def _ctx_holders(runner: RunLoop, services: RunServices) -> list[object]:
    """Structurally discover every rebindable ctx-holder hanging off the pair.

    Introspective on purpose: a hand-enumerated holder list silently passes when
    a future collaborator is constructed with a ctx but missed by a ``begin_run``
    cascade - discovery here means its stale ``_ctx`` fails the identity asserts.
    """

    return [
        attr
        for owner in (runner, services)
        for attr in vars(owner).values()
        if hasattr(attr, "begin_run") and hasattr(attr, "_ctx")
    ]


def test_runner_adopts_placeholder_then_rebinds_fresh_ctx() -> None:
    deps = make_run_deps()
    services = RunServices(deps, Arr.SONARR)
    runner = RunLoop(deps, services)

    # Phase 1 (adoption): the runner adopts the hub's single placeholder - no
    # second mint - and every ctx-holding collaborator is bound to that object.
    holders = _ctx_holders(runner, services)
    # The discovery itself is load-bearing: wait manager + hub + filter + pipeline.
    assert len(holders) >= 4
    assert runner._ctx is services.ctx
    assert all(vars(c)["_ctx"] is runner._ctx for c in holders)

    # Phase 2 (fresh mint): reset_run_stats swaps a fresh RunContext AND rebinds
    # every ctx-holding collaborator to it; a missed rebind would route a
    # collaborator's writes into the orphaned prior context (the drift the
    # begin_run fold guards against).
    first_ctx = runner._ctx
    runner.reset_run_stats(dry_run=False)

    assert runner._ctx is not first_ctx  # a fresh ctx was swapped in
    assert services.ctx is runner._ctx  # the hub rebound to it
    assert all(vars(c)["_ctx"] is runner._ctx for c in holders)


def test_sonarr_sync_init_shares_cache_store_for_staged_writes() -> None:
    # The parse-cache writer (SonarrParseCache) and the seed reader (ImportReconciler)
    # must share the strat's cache_store by identity, or a staged parse write would not
    # be visible to build_pending_seeds.
    deps = make_run_deps()
    services = RunServices(deps, Arr.SONARR)
    # Inject a typed fake through the sonarr_client seam so the collaborator wiring
    # runs off the REAL __init__ without connection keys - an incomplete fake here
    # is a pyright error and un-instantiable, unlike the old stringly-typed
    # monkeypatch.
    strat = SonarrSync(deps, services, sonarr_client=FakeSonarrClient())

    assert strat._parse.cache_store is deps.cache_store
    assert strat._reconciler.cache_store is deps.cache_store


def test_radarr_sync_init_builds_without_network_via_client_seam() -> None:
    # The radarr_client seam lets RadarrSync's REAL __init__ run under test: the
    # injected fake skips require_connection, so no config keys are needed.
    deps = make_run_deps()
    services = RunServices(deps, Arr.RADARR)
    fake = FakeRadarrClient()

    strat = RadarrSync(deps, services, radarr_client=fake)

    assert strat.radarr is fake
    assert strat._mappings is deps.mappings
    assert strat.anibridge is deps.mappings.anibridge


def test_rundeps_build_pins_verify_ssl_to_the_arrs_knob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``RunDeps.build`` constructs the run's httpx client with THIS arr's
    ``verify_ssl`` (the per-arr escape hatch for a self-signed HTTPS arr).
    """

    seen: list[bool] = []
    real_factory = run_services.make_httpx_client

    def _record(*, verify: bool = True) -> httpx.Client:
        seen.append(verify)
        return real_factory(verify=verify)

    monkeypatch.setattr(run_services, "make_httpx_client", _record)
    deps = RunDeps.build(
        Arr.SONARR,
        cache=str(tmp_path / "cache.db"),
        logger=logging.getLogger("seadexarr.test"),
        mappings=make_bare_instance(MappingResolver),
        app_config=make_config(verify_ssl=False),
        web=httpx.Client(),
        boot=BootFlow(),
    )
    deps.close()

    assert seen == [False]


def test_sonarr_cross_check_builds_without_network_via_radarr_seam() -> None:
    # With ignore_movies_in_radarr on, SonarrSync.__init__ used to hard-build a
    # RadarrClient and eagerly fetch its library (in the constructor) - the one
    # arr client construction without a seam. The injected fake makes the REAL
    # __init__ testable with the feature ON; empty mappings -> no movies collected.
    deps = make_run_deps(config=make_config(url="http://sonarr", api_key="key", ignore_movies_in_radarr=True))
    services = RunServices(deps, Arr.SONARR)
    fake = FakeRadarrClient()

    strat = SonarrSync(deps, services, sonarr_client=FakeSonarrClient(), radarr_client=fake)

    assert strat.ignore_movies_in_radarr is True
    assert strat.all_radarr_movies == []
