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
from collections.abc import Callable
from typing import override

from seadexarr.modules.manual_import import ImportProbe, ImportProgress, PendingImport
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.protocols import ArrSync, EpisodeProgress
from seadexarr.modules.seadex_types import (
    CommandResource,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    SonarrEpisode,
)


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


class FakeSonarrClient:
    """A typed, scriptable stand-in for ``SonarrClient``'s read/command surface.

    Each read returns a per-instance field a test presets or reassigns mid-test
    (e.g. ``fake.episodes_return = [...]``); the two import commands RECORD their
    typed call args, so a test asserts on recorded state (``execute_calls`` /
    ``candidate_calls``) instead of a ``MagicMock`` ``assert_called`` / ``call_args``.
    Absorbed as ``Any`` by ``make_sonarr_sync(sonarr=...)``, so it need not subclass
    the real client - it only has to answer the methods the strategy/executor call.
    """

    def __init__(
        self,
        *,
        queue: list[QueueRecord] | None = None,
        episodes: list[SonarrEpisode] | None = None,
        commands: list[CommandResource] | None = None,
        candidates: list[ManualImportCandidate] | None = None,
        quality_defs: list[QualityDefinition] | None = None,
        languages: list[Language] | None = None,
        parse: list[dict[str, int]] | None = None,
        parse_episode_info_fn: Callable[[str], ParsedFileInfo | None] | None = None,
        execute_command_id: int | None = None,
        command_status: CommandResource | None = None,
        refresh_count: int | None = 7,
    ) -> None:
        self.queue_return: list[QueueRecord] = queue or []
        self.episodes_return: list[SonarrEpisode] | None = [] if episodes is None else episodes
        self.commands_return: list[CommandResource] = commands or []
        self.candidates_return: list[ManualImportCandidate] | None = candidates
        self.quality_defs_return: list[QualityDefinition] = quality_defs or []
        self.languages_return: list[Language] = languages or []
        self.parse_return: list[dict[str, int]] | None = parse
        self.parse_episode_info_fn: Callable[[str], ParsedFileInfo | None] = parse_episode_info_fn or (lambda _f: None)
        self.execute_command_id = execute_command_id
        self.command_status_return = (
            command_status if command_status is not None else CommandResource(status="completed")
        )
        self.refresh_count = refresh_count
        # Recorded calls (the typed replacement for MagicMock's assert_called /
        # call_args / call_count): the import commands keep their full args; the
        # plain reads keep a count / arg-list so a test can assert (not-)called.
        self.candidate_calls: list[tuple[PendingImport, bool]] = []
        self.execute_calls: list[tuple[list[ManualImportFile], str]] = []
        self.queue_calls: int = 0
        self.episodes_calls: list[int] = []
        self.refresh_calls: int = 0

    def queue(self) -> list[QueueRecord]:
        self.queue_calls += 1
        return self.queue_return

    def list_commands(self) -> list[CommandResource]:
        return self.commands_return

    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        del quiet
        self.episodes_calls.append(series_id)
        return self.episodes_return

    def parse(self, filename: str) -> list[dict[str, int]] | None:
        del filename
        return self.parse_return

    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None:
        return self.parse_episode_info_fn(filename)

    def refresh_monitored_downloads(self) -> int | None:
        self.refresh_calls += 1
        return self.refresh_count

    def command_status(self, command_id: int) -> CommandResource:
        del command_id
        return self.command_status_return

    def quality_definitions(self) -> list[QualityDefinition]:
        return self.quality_defs_return

    def languages(self) -> list[Language]:
        return self.languages_return

    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
        filter_existing_files: bool = False,
    ) -> list[ManualImportCandidate] | None:
        self.candidate_calls.append((pending, filter_existing_files))
        return self.candidates_return

    def manual_import_execute(
        self,
        *,
        files: list[ManualImportFile],
        import_mode: str = "auto",
    ) -> int | None:
        self.execute_calls.append((files, import_mode))
        return self.execute_command_id


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
