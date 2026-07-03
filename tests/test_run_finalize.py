# pyright: strict
"""Guards the single end-of-run finalize site.

When ``max_torrents_to_add`` is reached mid-run, ``_grab`` returns a pure bool
(it no longer finalizes); ``run_sync`` breaks the per-item scan and runs the ONE
post-loop ``_finalize_run`` site - the same site the normal end-of-run path
reaches. These pin both halves of that hoist so a future change can't silently
double-finalize or skip the blocking/import pass on the cap-reached break.

The strategy is the shared typed :class:`FakeStrategy` (an ``ArrSync`` recording
its ``process_al_id`` calls), the engine's collaborators are small typed fakes,
and ``_finalize_run`` is replaced by a typed recorder - so the contracts are
pinned by asserting recorded state.
"""

import logging

from seadexarr.modules.config import Arr
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.protocols import ImportCompleter
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr
from seadexarr.modules.seadex_types import ProgressSink

from .builders import make_bare_instance, make_config, make_services
from .fakes import CaptureHandler, FakeArrItem, FakeStrategy


class _FakeGateway:
    """Stands in for the AniList/SeaDex gateways: a cache-warm no-op (0 fetched)."""

    def load_cache(self) -> None:
        pass

    def prefetch(
        self,
        ids: object,
        *,
        preview: bool = False,
        progress: ProgressSink | None = None,
    ) -> int:
        del ids, preview, progress
        return 0


class _FakeReporter:
    """The run-loop log hooks the scan drives (each just acknowledges)."""

    def log_arr_start(self, arr: Arr, n_items: int) -> bool:
        return True

    def log_arr_item_start(self, arr: Arr, item_title: str, n_item: int, n_items: int) -> bool:
        return True

    def log_arr_item_unmonitored(self, ctx: RunContext, item_title: str) -> bool:
        return True

    def log_no_anilist_mappings(self, ctx: RunContext, title: str) -> bool:
        return True


class _FakeBound:
    """A collaborator whose only run-loop hook is the ``begin_run`` ctx bind."""

    def begin_run(self, ctx: RunContext, strategy: ImportCompleter | None = None) -> None:
        del ctx, strategy


class _FinalizeRecorder:
    """A typed stand-in for ``SeaDexArr._finalize_run`` that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _engine(finalize: _FinalizeRecorder, logger: logging.Logger) -> SeaDexArr:
    """A bare ``SeaDexArr`` wired with typed fakes for the run-loop collaborators.

    The strategy reaches ``run_sync`` typed (it's an ``ArrSync``); the rest are
    injected as bare attributes (the methods only read them), and ``_finalize_run``
    is shadowed by the recorder so the single finalize site is observable. The
    ``_services`` hub is a bare real ``RunServices`` (the loop reads its ``arr``,
    ``begin_run`` and ``is_preview``) whose per-id collaborators are ctx-bind fakes.
    """

    config = make_config()
    services = make_services(
        qbit=None,
        _filter=_FakeBound(),
        _grab_pipeline=_FakeBound(),
    )
    return make_bare_instance(
        SeaDexArr,
        qbit=None,
        logger=logger,
        _config=config,
        _arr_config=config.for_arr(Arr.SONARR),
        _anilist=_FakeGateway(),
        _seadex=_FakeGateway(),
        _reporter=_FakeReporter(),
        _services=services,
        _wait_manager=_FakeBound(),
        _finalize_run=finalize,
    )


class TestCapReachedFinalizesOnce:
    """A mid-run cap stops the scan and finalizes exactly once, at the single site."""

    def test_cap_reached_breaks_loop_and_finalizes_once(self, logger: logging.Logger) -> None:
        # Cap reached on the first id: process_al_id returns True (stop the run).
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A"), FakeArrItem(item_id=2, title="B")],
            anilist_ids={1: MappingEntry(anilist_id=1)},
            process_returns=True,
        )
        finalize = _FinalizeRecorder()

        _engine(finalize, logger).run_sync(
            strategy,
            item_id=None,
            dry_run=True,
        )

        # The cap stopped the scan after the first id: the second item is never reached.
        assert strategy.process_calls == [1]
        # ...and the single post-loop finalize ran exactly once.
        assert finalize.calls == 1


class TestPerIdErrorContainment:
    """One AniList id's failure is contained to that id, not the whole item."""

    def test_one_al_id_error_does_not_skip_siblings(self, logger: logging.Logger) -> None:
        strategy = FakeStrategy(
            items=[FakeArrItem(item_id=1, title="A")],
            anilist_ids={1: MappingEntry(anilist_id=1), 2: MappingEntry(anilist_id=2)},
            process_raises_on=1,
        )
        finalize = _FinalizeRecorder()
        capture = CaptureHandler()
        logger.addHandler(capture)
        try:
            _engine(finalize, logger).run_sync(
                strategy,
                item_id=None,
                dry_run=True,
            )
        finally:
            logger.removeHandler(capture)

        # The first id raised but was contained: the sibling id 2 is still processed.
        assert strategy.process_calls == [1, 2]
        # The per-id failure was logged at ERROR (containment is observable).
        assert any(r.levelno == logging.ERROR for r in capture.records)
        # The single finalize still ran once on the normal end-of-run path.
        assert finalize.calls == 1
