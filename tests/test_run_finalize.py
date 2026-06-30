# pyright: strict
"""Guards the single end-of-run finalize site.

When ``max_torrents_to_add`` is reached mid-run, ``_grab`` returns a pure bool
(it no longer finalizes); ``run_sync`` breaks the per-item scan and runs the ONE
post-loop ``_finalize_run`` site - the same site the normal end-of-run path
reaches. These pin both halves of that hoist so a future change can't silently
double-finalize or skip the blocking/import pass on the cap-reached break.

The strategy is a real typed :class:`FakeStrategy` (an ``ArrSync`` recording its
``process_al_id`` calls), the engine's collaborators are small typed fakes, and
``_finalize_run`` is replaced by a typed recorder - so the contracts are pinned
by asserting recorded state, not ``MagicMock`` call interactions, and the file
type-checks at strict.
"""

import logging
from typing import override

from seadexarr.modules.anilist_gateway import PrefetchProgress
from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import ImportProbe, ImportProgress, PendingImport
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.protocols import ArrSync, EpisodeProgress, ImportCompleter
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import SeaDexArr

from .builders import make_bare_instance, make_config, make_logger


class _Item:
    """A minimal Arr item satisfying the ``ArrItem`` protocol surface."""

    def __init__(self, *, item_id: int = 1, title: str = "Show", monitored: bool = True) -> None:
        self.id = item_id
        self.title = title
        self.imdbId: str | None = None
        self.monitored = monitored


class FakeStrategy(ArrSync[_Item]):
    """A typed, recording ``ArrSync`` for engine-orchestration tests.

    Records each ``process_al_id`` call (the al_id), and lets a test script the
    items, the resolved AniList ids, and whether ``process_al_id`` returns the
    cap-reached sentinel or raises - replacing the ``MagicMock`` strategy whose
    ``assert_called`` pinned the contract. The import hooks are unused here.
    """

    def __init__(
        self,
        *,
        items: list[_Item],
        anilist_ids: dict[int, MappingEntry],
        process_returns: bool = False,
        process_raises_on: int | None = None,
    ) -> None:
        self._items = items
        self._anilist_ids = anilist_ids
        self._process_returns = process_returns
        self._process_raises_on = process_raises_on
        self.process_calls: list[int] = []

    @override
    def get_items(self) -> list[_Item]:
        return self._items

    @override
    def filter_to_single(self, items: list[_Item], item_id: int) -> list[_Item]:
        return [i for i in items if i.id == item_id]

    @override
    def item_anilist_ids(self, item: _Item, log_ignored: bool = True) -> dict[int, MappingEntry]:
        return self._anilist_ids

    @property
    @override
    def warms_episodes(self) -> bool:
        return False

    @override
    def prefetch_episodes(self, items: list[_Item], *, progress: EpisodeProgress | None = None) -> int:
        return 0

    @override
    def process_al_id(self, item: _Item, item_title: str, al_id: int, mapping: MappingEntry) -> bool:
        self.process_calls.append(al_id)
        if self._process_raises_on is not None and al_id == self._process_raises_on:
            raise ValueError(f"boom on al_id {al_id}")
        return self._process_returns

    @override
    def pending_import_series_id(self, item: _Item) -> int | None:
        return None

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        *,
        force: bool = False,
        at_deadline: bool = False,
    ) -> ImportProbe:
        raise NotImplementedError  # not driven by the finalize tests

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        raise NotImplementedError  # not driven by the finalize tests


class _FakeGateway:
    """Stands in for the AniList/SeaDex gateways: a cache-warm no-op (0 fetched)."""

    def load_cache(self) -> None:
        pass

    def prefetch(
        self,
        ids: object,
        *,
        preview: bool = False,
        progress: PrefetchProgress | None = None,
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


class _CaptureHandler(logging.Handler):
    """Collects emitted records so a logged error can be asserted by level."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _engine(strategy: FakeStrategy, finalize: _FinalizeRecorder, logger: logging.Logger) -> SeaDexArr:
    """A bare ``SeaDexArr`` wired with typed fakes for the run-loop collaborators.

    The strategy reaches ``run_sync`` typed (it's an ``ArrSync``); the rest are
    injected as bare attributes (the methods only read them), and ``_finalize_run``
    is shadowed by the recorder so the single finalize site is observable.
    """

    config = make_config()
    return make_bare_instance(
        SeaDexArr,
        qbit=None,
        logger=logger,
        _config=config,
        _arr_config=config.for_arr(Arr.SONARR),
        _anilist=_FakeGateway(),
        _seadex=_FakeGateway(),
        _reporter=_FakeReporter(),
        _filter=_FakeBound(),
        _grab_pipeline=_FakeBound(),
        _wait_manager=_FakeBound(),
        _finalize_run=finalize,
    )


class TestCapReachedFinalizesOnce:
    """A mid-run cap stops the scan and finalizes exactly once, at the single site."""

    def test_cap_reached_breaks_loop_and_finalizes_once(self) -> None:
        # Cap reached on the first id: process_al_id returns True (stop the run).
        strategy = FakeStrategy(
            items=[_Item(item_id=1, title="A"), _Item(item_id=2, title="B")],
            anilist_ids={1: MappingEntry(anilist_id=1)},
            process_returns=True,
        )
        finalize = _FinalizeRecorder()

        result = _engine(strategy, finalize, make_logger()).run_sync(
            strategy,
            arr=Arr.SONARR,
            item_id=None,
            dry_run=True,
        )

        assert result is True
        # The cap stopped the scan after the first id: the second item is never reached.
        assert strategy.process_calls == [1]
        # ...and the single post-loop finalize ran exactly once.
        assert finalize.calls == 1


class TestPerIdErrorContainment:
    """One AniList id's failure is contained to that id, not the whole item."""

    def test_one_al_id_error_does_not_skip_siblings(self) -> None:
        strategy = FakeStrategy(
            items=[_Item(item_id=1, title="A")],
            anilist_ids={1: MappingEntry(anilist_id=1), 2: MappingEntry(anilist_id=2)},
            process_raises_on=1,
        )
        finalize = _FinalizeRecorder()
        logger = make_logger("test-run-finalize-containment")
        capture = _CaptureHandler()
        logger.addHandler(capture)
        try:
            _engine(strategy, finalize, logger).run_sync(
                strategy,
                arr=Arr.SONARR,
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
