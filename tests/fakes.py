# pyright: strict
"""Shared, strict-typed test doubles.

The home for fakes used across more than one test module, written to type-check
at strict (no ``MagicMock``, no ``Any``). The guiding pattern: where a collaborator
is injected behind a typed seam (``ArrSync``, ``AbstractCacheStore``), a small
concrete fake implements it and records what a test needs to assert - so contracts
are pinned by recorded state.

Collaborators that the run machinery only reads as bare attributes (absorbed as
``Any`` by ``make_bare_instance``) don't need a shared fake; keep those local to
the test that drives them.
"""

import io
import logging
import re
from collections.abc import Callable
from typing import override

from seadexarr.modules.manual_import import ImportProbe, ImportProgress, PendingImport
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.protocols import ArrSync
from seadexarr.modules.radarr_client import AbstractRadarrClient
from seadexarr.modules.seadex_types import (
    CommandResource,
    HistoryRecord,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    MovieFile,
    ParsedFileInfo,
    ProgressSink,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    SonarrEpisode,
    SonarrItem,
)
from seadexarr.modules.sonarr_client import AbstractSonarrClient

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Drop ANSI escape sequences so assertions see the plain characters."""

    return _ANSI.sub("", text)


class TtyStringIO(io.StringIO):
    """An in-memory stream that claims to be a terminal (drives the rich/TTY arms)."""

    @override
    def isatty(self) -> bool:
        return True


class FakeClock:
    """A monotonic-ish clock the tests advance by hand, for stable durations."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


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
    cap-reached sentinel or raises. The import hooks raise unless a test that
    drives them overrides this fake.
    """

    def __init__(
        self,
        *,
        items: list[FakeArrItem],
        anilist_ids: dict[int, MappingEntry],
        process_returns: bool = False,
        process_raises_on: int | None = None,
        history: list[HistoryRecord] | None = None,
    ) -> None:
        self._items = items
        self._anilist_ids = anilist_ids
        self._process_returns = process_returns
        self._process_raises_on = process_raises_on
        self.process_calls: list[int] = []
        # Scripted history; reassign to None mid-test to script the failure path.
        self.history: list[HistoryRecord] | None = [] if history is None else history
        self.history_calls: list[str] = []

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
    def prefetch_episodes(self, items: list[FakeArrItem], *, progress: ProgressSink | None = None) -> int:
        return 0

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        self.history_calls.append(date)
        return self.history

    @override
    def process_al_id(self, item: FakeArrItem, al_id: int, mapping: MappingEntry) -> bool:
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


class FakeSonarrClient(AbstractSonarrClient):
    """A typed, scriptable stand-in for the :class:`AbstractSonarrClient` surface.

    Each read returns a per-instance field a test presets or reassigns mid-test
    (e.g. ``fake.episodes_return = [...]``); the two import commands RECORD their
    typed call args, so a test asserts on recorded state (``execute_calls`` /
    ``candidate_calls``). Subclasses the ``AbstractSonarrClient`` ABC, so it's
    nominally checked against the real client's full method surface - a
    missing method is a static ``reportAbstractUsage`` error and an
    un-instantiable ``TypeError``, not a silently-absorbed ``Any``.
    """

    def __init__(
        self,
        *,
        all_series: list[SonarrItem] | None = None,
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
        history_since: list[HistoryRecord] | None = None,
    ) -> None:
        self.all_series_return: list[SonarrItem] = all_series or []
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
        self.history_since_return: list[HistoryRecord] | None = [] if history_since is None else history_since
        # Recorded calls: the import commands keep their full args; the plain reads
        # keep a count / arg-list so a test can assert (not-)called.
        self.candidate_calls: list[PendingImport] = []
        self.execute_calls: list[tuple[list[ManualImportFile], str]] = []
        self.all_series_calls: int = 0
        self.queue_calls: int = 0
        self.episodes_calls: list[int] = []
        self.refresh_calls: int = 0
        self.history_calls: list[str] = []

    @override
    def all_series(self) -> list[SonarrItem]:
        self.all_series_calls += 1
        return self.all_series_return

    @override
    def queue(self) -> list[QueueRecord]:
        self.queue_calls += 1
        return self.queue_return

    @override
    def list_commands(self) -> list[CommandResource]:
        return self.commands_return

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        del quiet
        self.episodes_calls.append(series_id)
        return self.episodes_return

    @override
    def parse(self, filename: str) -> list[dict[str, int]] | None:
        del filename
        return self.parse_return

    @override
    def parse_episode_info(self, filename: str) -> ParsedFileInfo | None:
        return self.parse_episode_info_fn(filename)

    @override
    def refresh_monitored_downloads(self) -> int | None:
        self.refresh_calls += 1
        return self.refresh_count

    @override
    def command_status(self, command_id: int) -> CommandResource:
        del command_id
        return self.command_status_return

    @override
    def quality_definitions(self) -> list[QualityDefinition]:
        return self.quality_defs_return

    @override
    def languages(self) -> list[Language]:
        return self.languages_return

    @override
    def manual_import_candidates(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None:
        self.candidate_calls.append(pending)
        return self.candidates_return

    @override
    def manual_import_execute(
        self,
        *,
        files: list[ManualImportFile],
        import_mode: str = "auto",
    ) -> int | None:
        self.execute_calls.append((files, import_mode))
        return self.execute_command_id

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        self.history_calls.append(date)
        return self.history_since_return


class FakeRadarrClient(AbstractRadarrClient):
    """A typed, scriptable stand-in for the :class:`AbstractRadarrClient` surface.

    Mirrors :class:`FakeSonarrClient`: reads return per-instance fields a test
    presets, and ``movie_files`` RECORDS the ids it was asked for. Subclasses the
    ABC, so a missing method is a static ``reportAbstractUsage`` error and an
    un-instantiable ``TypeError``.
    """

    def __init__(
        self,
        *,
        movies: list[RadarrItem] | None = None,
        movie_files: list[MovieFile] | None = None,
        history_since: list[HistoryRecord] | None = None,
    ) -> None:
        self.movies_return: list[RadarrItem] = movies or []
        self.movie_files_return: list[MovieFile] = movie_files or []
        self.history_since_return: list[HistoryRecord] | None = [] if history_since is None else history_since
        self.movie_files_calls: list[int] = []
        self.history_calls: list[str] = []

    @override
    def all_movies(self) -> list[RadarrItem]:
        return self.movies_return

    @override
    def movie_files(self, movie_id: int) -> list[MovieFile]:
        self.movie_files_calls.append(movie_id)
        return self.movie_files_return

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        self.history_calls.append(date)
        return self.history_since_return


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
