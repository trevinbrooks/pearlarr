# pyright: strict
"""Shared, strict-typed test doubles.

The home for fakes used across more than one test module, written to type-check
at strict (no ``MagicMock``, no ``Any``). The guiding pattern: where a collaborator
is injected behind a typed seam (``ArrSync``, ``CacheStoreProtocol``), a small
concrete fake implements it and records what a test needs to assert - so contracts
are pinned by recorded state, not ``MagicMock`` call interactions.

Collaborators that the run machinery only reads as bare attributes (absorbed as
``Any`` by ``make_bare_instance``) don't need a shared fake; keep those local to
the test that drives them.
"""

import logging
from typing import override

from seadexarr.modules.manual_import import ImportProbe, ImportProgress, PendingImport
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.protocols import ArrSync, EpisodeProgress


class FakeArrItem:
    """A minimal item satisfying the ``ArrItem`` protocol surface.

    Sets the four attributes the run loop reads (``id`` / ``title`` / ``imdbId`` /
    ``monitored``); a single class stands in for both a Sonarr series and a Radarr
    movie since the shared loop only touches ``ArrItem``.
    """

    def __init__(self, *, item_id: int = 1, title: str = "Show", monitored: bool = True) -> None:
        self.id = item_id
        self.title = title
        self.imdbId: str | None = None
        self.monitored = monitored


class FakeStrategy(ArrSync[FakeArrItem]):
    """A typed, recording ``ArrSync`` for engine-orchestration tests.

    Records each ``process_al_id`` call (the al_id) and lets a test script the
    items, the resolved AniList ids, and whether ``process_al_id`` returns the
    cap-reached sentinel or raises - replacing a ``MagicMock`` strategy whose
    ``assert_called`` pinned the contract. The import hooks raise unless a test
    that drives them overrides this fake.
    """

    def __init__(
        self,
        *,
        items: list[FakeArrItem],
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
    def get_items(self) -> list[FakeArrItem]:
        return self._items

    @override
    def filter_to_single(self, items: list[FakeArrItem], item_id: int) -> list[FakeArrItem]:
        return [i for i in items if i.id == item_id]

    @override
    def item_anilist_ids(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, MappingEntry]:
        return self._anilist_ids

    @property
    @override
    def warms_episodes(self) -> bool:
        return False

    @override
    def prefetch_episodes(self, items: list[FakeArrItem], *, progress: EpisodeProgress | None = None) -> int:
        return 0

    @override
    def process_al_id(self, item: FakeArrItem, item_title: str, al_id: int, mapping: MappingEntry) -> bool:
        self.process_calls.append(al_id)
        if self._process_raises_on is not None and al_id == self._process_raises_on:
            raise ValueError(f"boom on al_id {al_id}")
        return self._process_returns

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
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
        raise NotImplementedError  # override in a test that drives the import hook

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        raise NotImplementedError  # override in a test that drives the import hook


class CaptureHandler(logging.Handler):
    """A logging handler that collects records, so a logged line/level can be asserted.

    Attach to a test's logger, run the code, then assert over ``records`` (e.g. a
    contained per-id failure logged at ``ERROR``) - the no-throw, structured way to
    pin logging behaviour without coupling to exact message strings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
