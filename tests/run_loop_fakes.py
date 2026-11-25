# pyright: strict
"""Fakes and a run loop builder shared by the arr run loop tests.

`make_engine` builds an `ArrRunLoop` bare, without its constructor, over typed fakes of its collaborators, with
`_finish_arr_run` swapped for a `FinalizeRecorder` so a test can count the finalize calls.
"""

import logging

from pearlarr.anilist_gateway import AniListGateway
from pearlarr.arr_run_loop import ArrRunLoop
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.arr_strategy import ImportCompleter
from pearlarr.cache import AbstractCacheStore
from pearlarr.config import AppConfig, Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.output import CountsMark, SeverityCounter
from pearlarr.progress import ProgressCallback
from pearlarr.skyhook_gateway import SkyHookGateway

from .builders import FakeCacheStore, make_bare_instance, make_config, make_services
from .fakes import FakeSkyHookClient


class FakeGateway:
    """Stands in for the AniList/SeaDex gateways: a cache-warm no-op (0 fetched)."""

    # The loop reads each gateway's unreachable flag. Never unreachable unless a test sets it.
    unreachable: bool = False

    def load_cache(self) -> None:
        pass

    def prefetch(
        self,
        ids: object,
        *,
        dry_run: bool = False,
        progress: ProgressCallback | None = None,
    ) -> int:
        del ids, dry_run, progress
        return 0


class _FakeReporter:
    """The arr run loop's log hooks the scan and the arr run tail drive (each just acknowledges)."""

    def counts_mark(self) -> CountsMark:
        return SeverityCounter().bound_mark()

    def log_arr_start(self, arr: Arr, n_items: int) -> bool:
        return True

    def log_arr_item_start(self, arr: Arr, item_title: str, n_item: int, n_items: int) -> bool:
        return True

    def log_arr_item_unmonitored(self, state: ArrRunState, item_title: str) -> bool:
        return True

    def log_no_anilist_mappings(self, state: ArrRunState, title: str) -> bool:
        return True

    def log_no_seadex_releases(self, state: ArrRunState) -> bool:
        return True

    def scan_finished(self, arr: Arr) -> bool:
        return True

    def log_arr_run_summary(self, state: ArrRunState, *, dry_run: bool, has_client: bool) -> bool:
        return True

    def arr_run_finished(self, arr: Arr) -> bool:
        return True


class FakeBound:
    """A collaborator whose only arr run loop hook is the `bind_arr_run` state bind."""

    def bind_arr_run(self, state: ArrRunState, strategy: ImportCompleter | None = None) -> None:
        del state, strategy


class FinalizeRecorder:
    """A typed fake of `ArrRunLoop._finish_arr_run` that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def make_engine_services(*, qbit: object = None, **overrides: object) -> ArrRunServices:
    """A bare real `ArrRunServices` whose per-title collaborators are state-bind fakes.

    `qbit` set makes the run real.
    """

    return make_services(
        qbit=qbit,
        _filter=FakeBound(),
        _grab_pipeline=FakeBound(),
        _pending_imports=FakeBound(),
        _reporter=_FakeReporter(),
        **overrides,
    )


def make_engine(
    finalize: FinalizeRecorder | None,
    logger: logging.Logger,
    *,
    config: AppConfig | None = None,
    seadex: FakeGateway | None = None,
    cache_store: AbstractCacheStore | None = None,
    anilist: AniListGateway | FakeGateway | None = None,
    services: ArrRunServices | None = None,
    wait_manager: FakeBound | ImportWaitManager | None = None,
    skyhook: SkyHookGateway | None = None,
) -> ArrRunLoop:
    """A bare `ArrRunLoop` wired with typed fakes for the arr run loop's collaborators.

    The strategy reaches `run_arr` typed (it's an `ArrStrategy`). The rest are
    injected as bare attributes (the methods only read them), and `_finish_arr_run`
    is shadowed by the recorder so the single finalize site is observable
    (`None` keeps the real one). The `_services` object defaults to a dry-run
    `make_engine_services()` (the loop reads its `arr`, `bind_arr_run`, `mark_dirty`,
    `is_dry_run` and `pay_pause`). The `cache_store` backs the activity read's checkpoint.
    `config` overrides the loop's config (the activity-read toggle tests).
    `cache_store` shares one store across engines (the checkpoint replay tests).
    `anilist` swaps in a real gateway (the AniList boot-note tests) or a fake that is unreachable (the selection
    hash test).
    `services` swaps the services (the commit tests make it real and share its store).
    `wait_manager` swaps the state-bind wait manager (the contained-error commit test, the arr-run-start test).
    `skyhook` swaps in a gateway over a failing client (the SkyHook unreachable tests).
    """

    config = config if config is not None else make_config()
    store = cache_store if cache_store is not None else FakeCacheStore()
    services = services if services is not None else make_engine_services()
    shadowed = {} if finalize is None else {"_finish_arr_run": finalize}
    return make_bare_instance(
        ArrRunLoop,
        qbit=services.qbit,
        logger=logger,
        _config=config,
        _arr_config=config.for_arr(Arr.SONARR),
        _anilist=anilist if anilist is not None else FakeGateway(),
        _seadex=seadex if seadex is not None else FakeGateway(),
        _skyhook=skyhook if skyhook is not None else SkyHookGateway(client=FakeSkyHookClient(), cache_store=store),
        _reporter=_FakeReporter(),
        _services=services,
        _wait_manager=wait_manager if wait_manager is not None else FakeBound(),
        cache_store=store,
        **shadowed,
    )
