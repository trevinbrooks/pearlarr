# pyright: strict
"""Shared, strict-typed test doubles. A fake driven by a single test stays in that test."""

import io
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast, final, override

import httpx
import pytest
import qbittorrentapi

from pearlarr.aired_view import NoView
from pearlarr.arr_http import ArrHttp, DeleteOutcome
from pearlarr.arr_models import (
    AlternateTitle,
    ArrFiles,
    CommandResource,
    DownloadClientConfig,
    HistoryPage,
    HistoryRecord,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    MovieFile,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    RemotePathMapping,
    SonarrEpisode,
    SonarrItem,
)
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_strategy import ArrStrategy
from pearlarr.clock import Clock
from pearlarr.mappings import TitleMapping
from pearlarr.output import (
    ArrRunSummaryReady,
    CapReached,
    Diagnostic,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    OutputHub,
    ReleaseSkipped,
    Renderer,
    ScanStarted,
    Severity,
    TitleDetail,
    TitleHeader,
    TitleRow,
    install_hub,
)
from pearlarr.output.console_lines import ConsoleLine
from pearlarr.output.scan_lines import scan_event_lines
from pearlarr.pending_import import PendingImport
from pearlarr.placement_types import EpisodeIndex, episode_index
from pearlarr.progress import ProgressCallback
from pearlarr.radarr_client import AbstractRadarrClient
from pearlarr.scene_numbers import SceneMapping
from pearlarr.series_reads import AiredParseSource, SeriesRows
from pearlarr.skyhook_client import AbstractSkyHookClient, SkyHookNoReply, SkyHookReply
from pearlarr.sonarr_client import AbstractSonarrClient
from pearlarr.sonarr_episodes import EpisodeSource, TitleEpisodes
from pearlarr.sonarr_matching import NoAnswer
from pearlarr.tvdb_orders import AIRED_ORDER, SonarrSeriesOrders, is_another_order
from pearlarr.wait_types import AttemptKind, AttemptResult, ImportProgress
from pearlarr.xem_client import AbstractXemClient, XemNoReply

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Drop ANSI escape sequences."""

    return _ANSI.sub("", text)


# `isinstance` raises on the `ScanEvent` alias. Keep the tuple unannotated so it narrows to exactly `ScanEvent`.
SCAN_EVENT_TYPES = (
    ScanStarted,
    ItemStarted,
    TitleHeader,
    TitleDetail,
    TitleRow,
    ReleaseSkipped,
    GrabFailed,
    GrabAction,
    CapReached,
    ArrRunSummaryReady,
)


def scan_lines_from_events(events: Iterable[Event]) -> list[ConsoleLine]:
    """Re-derive the legacy scan lines a recorded event stream produces."""

    lines: list[ConsoleLine] = []
    for event in events:
        if isinstance(event, SCAN_EVENT_TYPES):
            lines.extend(scan_event_lines(event))
    return lines


def debug_logger(name: str) -> logging.Logger:
    """A logger `caplog` sees at DEBUG: no handlers of its own, propagating to the root."""

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    return logger


@final
class RecordingRenderer(Renderer):
    """A renderer that records everything it is handed (events + lifecycle calls)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.whens: list[float] = []
        self.runs_begun = 0
        self.levels: list[int] = []
        self.closed = False

    @override
    def handle(self, event: Event, when: float) -> None:
        self.events.append(event)
        self.whens.append(when)

    @override
    def begin_run(self) -> None:
        self.runs_begun += 1

    @override
    def set_level(self, level: int) -> None:
        self.levels.append(level)

    @override
    def close(self) -> None:
        self.closed = True

    def of_type[E](self, cls: type[E]) -> list[E]:
        """The recorded events of one type, in emit order."""

        return [event for event in self.events if isinstance(event, cls)]


@final
class RecordingHub:
    """A real OutputHub over one RecordingRenderer.

    Handles mint real ScopeIds and dedup/containment behave exactly as in
    production.
    """

    def __init__(self) -> None:
        self.renderer = RecordingRenderer()
        self.hub = OutputHub([self.renderer])

    @property
    def events(self) -> list[Event]:
        """The recorded events, in emit order."""

        return self.renderer.events

    def emit(self, event: Event) -> None:
        """Emit through the real hub."""

        self.hub.emit(event)

    def of_type[E](self, cls: type[E]) -> list[E]:
        """The recorded events of one type, in emit order."""

        return self.renderer.of_type(cls)


def install_recording_hub() -> RecordingHub:
    """Install a fresh `RecordingHub` as the process hub. Conftest's autouse teardown restores the default."""

    recording = RecordingHub()
    install_hub(recording.hub)
    return recording


def bind_arr_http(url: str = "http://arr.test") -> tuple[ArrHttp, RecordingHub]:
    """A bound arr transport (backoff sleeps stubbed) and a fresh recording hub. Conftest closes the client."""

    recording = install_recording_hub()
    http = ArrHttp.bind(client=httpx.Client(), url=url, api_key="testkey", label="Sonarr", sleep=lambda _s: None)
    return http, recording


@dataclass(frozen=True)
class _CategoryRow:
    """One qBittorrent info row, as far as the category read goes."""

    category: str


class QbitCategoryReads:
    """A qBittorrent stub for the two reads the category resolver makes.

    Every torrent sits in `category`, which a test may change between reads: None is a torrent qBittorrent
    doesn't have, and an exception is a failed read. `preferences` answers the settings read, each answer
    once and the last one from then on.
    """

    def __init__(
        self,
        category: str | Exception | None = None,
        *preferences: dict[str, object] | Exception,
    ) -> None:
        self.category = category
        self._preferences = list(preferences) or [{}]

    def torrents_info(self, *, torrent_hashes: str) -> list[_CategoryRow]:
        """The row for whichever hash is asked, in the category set now."""

        del torrent_hashes
        if isinstance(self.category, Exception):
            raise self.category
        return [] if self.category is None else [_CategoryRow(self.category)]

    def app_preferences(self) -> dict[str, object]:
        """The next scripted settings answer."""

        answer = self._preferences.pop(0) if len(self._preferences) > 1 else self._preferences[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def client(self) -> qbittorrentapi.Client:
        """This stub as the SDK client it stands in for."""

        # The sanctioned leaf-client hatch: the SDK client has no interface to subclass.
        return cast("qbittorrentapi.Client", self)


def diagnostic_messages(recording: RecordingHub, severity: Severity | None = None) -> list[str]:
    """The recorded `Diagnostic` messages, optionally filtered to one exact severity."""

    return [d.message for d in recording.of_type(Diagnostic) if severity is None or d.severity is severity]


class TtyStringIO(io.StringIO):
    """An in-memory stream that claims to be a terminal."""

    @override
    def isatty(self) -> bool:
        return True


class AsciiStringIO(io.StringIO):
    """An in-memory stream whose claimed encoding can't encode glyphs."""

    encoding = "ascii"


class RecordingProgress:
    """A `ProgressCallback` that records each (fraction, detail) update, in call order."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, fraction: float, detail: str | None = None) -> None:
        """Record one update."""

        self.updates.append((fraction, detail))


_CLOCK_USE_LIMIT = 10_000
"""Reads plus sleeps one `FakeClock` allows, far past the hundred or so the busiest test takes."""


class FakeClock(Clock):
    """A hand-driven `Clock`: `sleep` records and advances by `step`, `tick` by hand, callable as `now`.

    Past `_CLOCK_USE_LIMIT` reads and sleeps it fails the test, since only a loop that never ends gets there.
    """

    def __init__(self, step: float = 0.0) -> None:
        self.t = 0.0
        self._step = step
        self._uses = 0
        self.sleeps: list[float] = []
        """The requested duration of each `sleep` call, in call order."""

    @override
    def monotonic(self) -> float:
        self._count_use()
        return self.t

    @override
    def sleep(self, seconds: float) -> None:
        # Ignore the requested duration: advancing by the fixed step exercises the
        # deadline arithmetic without ever really sleeping.
        self._count_use()
        self.sleeps.append(seconds)
        self.t += self._step

    def _count_use(self) -> None:
        self._uses += 1
        if self._uses > _CLOCK_USE_LIMIT:
            pytest.fail(f"a FakeClock was used {_CLOCK_USE_LIMIT} times: the loop calling it never ends")

    def __call__(self) -> float:
        return self.monotonic()

    def tick(self, seconds: float) -> None:
        """Advance the fake time by `seconds`."""

        self.t += seconds


class FakeArrItem:
    """A minimal fake of the `ArrItem` surface: a series or a movie."""

    def __init__(self, *, item_id: int = 1, title: str = "Show", monitored: bool = True) -> None:
        self.id = item_id
        self.title = title
        self.imdbId: str | None = None
        self.monitored = monitored


class FakeStrategy(ArrStrategy[FakeArrItem]):
    """A typed `ArrStrategy` with scripted returns, for arr-run-loop tests."""

    def __init__(
        self,
        *,
        items: list[FakeArrItem],
        anilist_ids: dict[int, TitleMapping],
        services: ArrRunServices | None = None,
        history: list[HistoryRecord] | None = None,
        supports_blocking_wait: bool = True,
    ) -> None:
        self._items = items
        self._anilist_ids = anilist_ids
        # Every processed id is held past the cap on these services' run state, as the grab pipeline keeps one: the
        # title's hold and the run tally.
        self._cap_services = services
        self._supports_blocking_wait = supports_blocking_wait
        # Reassign to None mid-test to script the failure path.
        self.history: list[HistoryRecord] | None = [] if history is None else history
        self.history_calls: list[str] = []
        # Assign to script the titles the start-of-run hook names.
        self.check_again: set[int] = set()
        self.start_of_run_calls: list[list[int]] = []

    @override
    def get_items(self) -> list[FakeArrItem]:
        return self._items

    @override
    def filter_to_single(self, items: list[FakeArrItem], item_id: int) -> list[FakeArrItem]:
        return [i for i in items if i.id == item_id]

    @override
    def start_of_run(self, items: list[FakeArrItem]) -> set[int]:
        self.start_of_run_calls.append([item.id for item in items])
        return set(self.check_again)

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return self._anilist_ids

    @property
    @override
    def warms_episodes(self) -> bool:
        return False

    @override
    def prefetch_episodes(self, items: list[FakeArrItem], *, progress: ProgressCallback | None = None) -> int:
        return 0

    @override
    def prefetch_orders(self, items: list[FakeArrItem], *, progress: ProgressCallback | None = None) -> int:
        return 0

    @property
    @override
    def reads_scene_numbers(self) -> bool:
        return False

    @override
    def prefetch_scene_numbers(self, items: list[FakeArrItem], *, progress: ProgressCallback | None = None) -> int:
        return 0

    @override
    def history_since(self, date: str) -> list[HistoryRecord] | None:
        self.history_calls.append(date)
        return self.history

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        if self._cap_services is not None:
            self._cap_services.state.per_title.held_by_cap = True
            self._cap_services.state.stats.held_by_cap += 1

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        return None

    @override
    def import_completed(
        self,
        pending: PendingImport,
        content_path: str,
        attempt_kind: AttemptKind,
    ) -> AttemptResult:
        raise NotImplementedError  # override in a test that drives the import hook

    @override
    def import_progress(self, pending: PendingImport) -> ImportProgress:
        raise NotImplementedError  # override in a test that drives the import hook

    @property
    @override
    def supports_blocking_wait(self) -> bool:
        return self._supports_blocking_wait


NEUTRAL_RESCAN = CommandResource(
    name="RefreshMonitoredDownloads", status="completed", started=datetime(2000, 1, 1, tzinfo=UTC)
)
"""A completed rescan from long before any add: an idle Sonarr's command list."""


class FakeSonarrClient(AbstractSonarrClient):
    """A typed, scriptable fake of `AbstractSonarrClient`."""

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
        parse_fn: Callable[[str], ParsedFileInfo | None] | None = None,
        execute_command_id: int | None = None,
        command_status: CommandResource | None = None,
        command_status_script: list[CommandResource] | None = None,
        refresh_count: int | None = 7,
        history_since: list[HistoryRecord] | None = None,
        folder_candidates: list[ManualImportCandidate] | None = None,
        history_page: HistoryPage | None = None,
        path_mappings: list[RemotePathMapping] | None = None,
        completed_download_handling: bool = True,
    ) -> None:
        self.all_series_return: list[SonarrItem] = all_series or []
        # Reassign to None after construction to script a failed queue read.
        self.queue_return: list[QueueRecord] | None = queue or []
        self.episodes_return: list[SonarrEpisode] | None = [] if episodes is None else episodes
        # Reassign to None after construction to script a failed command-list read.
        self.commands_return: list[CommandResource] | None = [NEUTRAL_RESCAN] if commands is None else commands
        self.candidates_return: list[ManualImportCandidate] | None = candidates
        self.quality_defs_return: list[QualityDefinition] = quality_defs or []
        self.languages_return: list[Language] = languages or []
        # The scripted `/parse` result per filename (None = Sonarr could not parse it this call).
        self.parse_fn: Callable[[str], ParsedFileInfo | None] = parse_fn or (lambda _f: None)
        self.execute_command_id = execute_command_id
        self.command_status_return = (
            command_status if command_status is not None else CommandResource(status="completed")
        )
        self.command_status_script: list[CommandResource] = command_status_script or []
        self.refresh_count = refresh_count
        self.history_since_return: list[HistoryRecord] | None = [] if history_since is None else history_since
        # The folder-fetch trio defaults to None (the "failed" reads).
        self.folder_candidates_return: list[ManualImportCandidate] | None = folder_candidates
        self.history_page_return: HistoryPage | None = history_page
        self.path_mappings_return: list[RemotePathMapping] | None = path_mappings
        self.download_client_config_return = DownloadClientConfig(
            enable_completed_download_handling=completed_download_handling,
        )
        self.queue_delete_return: DeleteOutcome = DeleteOutcome.OK
        self.candidate_calls: list[PendingImport] = []
        self.parse_calls: list[str] = []
        self.execute_calls: list[tuple[list[ManualImportFile], str]] = []
        self.all_series_calls: int = 0
        self.queue_calls: int = 0
        self.queue_delete_calls: list[int] = []
        self.episodes_calls: list[int] = []
        self.refresh_calls: int = 0
        self.history_calls: list[str] = []
        self.folder_candidate_calls: list[tuple[str, str]] = []
        self.download_history_calls: list[str] = []
        self.path_mapping_calls: int = 0
        self.download_client_config_calls: int = 0

    @override
    def all_series(self) -> list[SonarrItem]:
        self.all_series_calls += 1
        return self.all_series_return

    @override
    def queue(self) -> list[QueueRecord] | None:
        self.queue_calls += 1
        return self.queue_return

    @override
    def queue_delete(self, queue_id: int) -> DeleteOutcome:
        self.queue_delete_calls.append(queue_id)
        return self.queue_delete_return

    @override
    def list_commands(self) -> list[CommandResource] | None:
        return None if self.commands_return is None else list(self.commands_return)

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        del quiet
        self.episodes_calls.append(series_id)
        return self.episodes_return

    @override
    def parse(self, filename: str) -> ParsedFileInfo | None:
        self.parse_calls.append(filename)
        return self.parse_fn(filename)

    @override
    def refresh_monitored_downloads(self) -> int | None:
        self.refresh_calls += 1
        return self.refresh_count

    @override
    def command_status(self, command_id: int) -> CommandResource:
        del command_id
        if self.command_status_script:
            return self.command_status_script.pop(0)
        return self.command_status_return

    @override
    def quality_definitions(self) -> list[QualityDefinition]:
        return self.quality_defs_return

    @override
    def languages(self) -> list[Language]:
        return self.languages_return

    @override
    def manual_import_candidates_by_download_id(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None:
        self.candidate_calls.append(pending)
        return self.candidates_return

    @override
    def manual_import_candidates_by_folder(
        self,
        *,
        folder: str,
        title: str,
    ) -> list[ManualImportCandidate] | None:
        self.folder_candidate_calls.append((folder, title))
        return self.folder_candidates_return

    @override
    def history_for_download(self, *, download_id: str) -> HistoryPage | None:
        self.download_history_calls.append(download_id)
        return self.history_page_return

    @override
    def remote_path_mappings(self) -> list[RemotePathMapping] | None:
        self.path_mapping_calls += 1
        return self.path_mappings_return

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

    @override
    def download_client_config(self) -> DownloadClientConfig:
        self.download_client_config_calls += 1
        return self.download_client_config_return


class HookedSonarrClient(FakeSonarrClient):
    """A fake whose state a test changes at a chosen read. Each hook runs after its read's answer is taken."""

    on_queue: Callable[[], None] | None = None
    on_rescan: Callable[[], None] | None = None
    on_fetch: Callable[[], None] | None = None
    on_episodes: Callable[[], None] | None = None
    on_languages: Callable[[], None] | None = None

    @override
    def queue(self) -> list[QueueRecord] | None:
        answer = super().queue()
        if self.on_queue is not None:
            self.on_queue()
        return answer

    @override
    def refresh_monitored_downloads(self) -> int | None:
        answer = super().refresh_monitored_downloads()
        if self.on_rescan is not None:
            self.on_rescan()
        return answer

    @override
    def manual_import_candidates_by_download_id(
        self,
        *,
        pending: PendingImport,
    ) -> list[ManualImportCandidate] | None:
        answer = super().manual_import_candidates_by_download_id(pending=pending)
        if self.on_fetch is not None:
            self.on_fetch()
        return answer

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        answer = super().episodes(series_id, quiet=quiet)
        if self.on_episodes is not None:
            self.on_episodes()
        return answer

    @override
    def languages(self) -> list[Language]:
        answer = super().languages()
        if self.on_languages is not None:
            self.on_languages()
        return answer


class PerSeriesSonarrClient(FakeSonarrClient):
    """A `FakeSonarrClient` whose episode read is scripted per series: an unscripted series fails the read."""

    def __init__(
        self,
        episodes_by_series: dict[int, list[SonarrEpisode]],
        *,
        candidates: list[ManualImportCandidate],
    ) -> None:
        super().__init__(candidates=candidates, execute_command_id=42)
        self.episodes_by_series = episodes_by_series

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        del quiet
        self.episodes_calls.append(series_id)
        return self.episodes_by_series.get(series_id)


class FakeRadarrClient(AbstractRadarrClient):
    """A typed, scriptable fake of `AbstractRadarrClient`."""

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


class FakeSkyHookClient(AbstractSkyHookClient):
    """SkyHook scripted per (TVDB id, order type).

    An unscripted pair answers `SkyHookNoReply.NOT_FOUND`, as SkyHook does for a series TVDB lacks. A fetch of a
    `failing` pair marks SkyHook unreachable, and every fetch after it is `SkyHookNoReply.UNREACHABLE`, as the real
    client makes it.
    """

    def __init__(
        self,
        replies: Mapping[tuple[int, str], SkyHookReply] | None = None,
        *,
        failing: frozenset[tuple[int, str]] = frozenset(),
    ) -> None:
        self._replies = dict(replies or {})
        self._failing = failing
        self._unreachable = False
        self.fetch_calls: list[tuple[int, str]] = []
        """Each fetch's (TVDB id, order type), in call order, the ones answered unreachable among them."""

    @property
    @override
    def unreachable(self) -> bool:
        return self._unreachable

    @override
    def fetch(self, tvdb_id: int, order_type: str) -> SkyHookReply | SkyHookNoReply:
        self.fetch_calls.append((tvdb_id, order_type))
        if self._unreachable or (tvdb_id, order_type) in self._failing:
            self._unreachable = True
            return SkyHookNoReply.UNREACHABLE
        return self._replies.get((tvdb_id, order_type), SkyHookNoReply.NOT_FOUND)


class FakeXemClient(AbstractXemClient):
    """TheXEM scripted per TVDB id, recording each series a request goes out for in `asked`.

    An unscripted series answers with no mappings, as TheXEM does for a series it doesn't list. A read of an
    `unusable` series is `XemNoReply.UNUSABLE` and stops nothing. A read of a `failing` series marks TheXEM
    unreachable, and every read after it is `XemNoReply.UNREACHABLE` with nothing asked, as the real client makes it.
    """

    def __init__(
        self,
        mappings: Mapping[int, tuple[SceneMapping, ...]] | None = None,
        *,
        unusable: frozenset[int] = frozenset(),
        failing: frozenset[int] = frozenset(),
    ) -> None:
        self._mappings = dict(mappings or {})
        self._unusable = unusable
        self._failing = failing
        self._unreachable = False
        self.asked: list[int] = []

    @property
    @override
    def unreachable(self) -> bool:
        return self._unreachable

    @override
    def mappings(self, tvdb_id: int) -> tuple[SceneMapping, ...] | XemNoReply:
        if self._unreachable:
            return XemNoReply.UNREACHABLE
        self.asked.append(tvdb_id)
        if tvdb_id in self._failing:
            self._unreachable = True
            return XemNoReply.UNREACHABLE
        if tvdb_id in self._unusable:
            return XemNoReply.UNUSABLE
        return self._mappings.get(tvdb_id, ())


class ScriptedEpisodes(EpisodeSource, AiredParseSource, SeriesRows):
    """A `SonarrEpisodes` and its `SeriesReads` scripted by hand: the series list off `sonarr`, one window per AniList id.

    `title_episodes` records its calls, raises for an `ambiguous` id, and serves `windows[anilist_id]` when windows are
    given, else the whole series list. Without a client every read is `NoView.READ_FAILED`, and a series among the
    `no_views` has no view for the reason given. Every series is numbered by `order`, and Sonarr sent no
    count of its episodes. The `unread_series` have no stored order answer, and the `unread_anilist_ids` no AniList
    answer. `counts` are AniList's episode counts by AniList id, None for an id left out. No series has an alternate
    title, and a parse is taken as already matched the way an aired Sonarr matches it.
    """

    def __init__(
        self,
        sonarr: FakeSonarrClient | None = None,
        *,
        windows: Mapping[int, list[SonarrEpisode] | None] | None = None,
        ambiguous: frozenset[int] = frozenset(),
        order: str = AIRED_ORDER,
        no_views: Mapping[int, NoView] | None = None,
        counts: Mapping[int, int] | None = None,
        unread_series: frozenset[int] = frozenset(),
        unread_anilist_ids: frozenset[int] = frozenset(),
    ) -> None:
        self._sonarr = sonarr
        self._windows = windows
        self._ambiguous = ambiguous
        self._order = order
        self._no_views: Mapping[int, NoView] = no_views or {}
        self._counts: Mapping[int, int] = counts or {}
        self._unread_series = unread_series
        self._unread_anilist_ids = unread_anilist_ids
        self._last_known: dict[int, list[SonarrEpisode]] = {}
        self.calls: list[tuple[int, int]] = []
        """Each `title_episodes` call's (series id, AniList id), in call order."""

    @property
    @override
    def series_hash(self) -> str:
        return "fp"

    @property
    @override
    def series_in_another_order(self) -> Mapping[int, str]:
        """Every series the client lists, when the scripted order is other than the aired one."""

        if self._sonarr is None or not is_another_order(self._order):
            return {}
        return {series.id: self._order for series in self._sonarr.all_series_return}

    def _series_episodes(self, series_id: int) -> list[SonarrEpisode] | NoView:
        """The whole-series list off the client, or why the series has no view."""

        if series_id in self._no_views:
            return self._no_views[series_id]
        episodes = None if self._sonarr is None else self._sonarr.episodes(series_id)
        if episodes is None:
            return NoView.READ_FAILED
        self._last_known[series_id] = episodes
        return episodes

    @override
    def title_episodes(self, series_id: int, anilist_id: int, mapping: TitleMapping) -> TitleEpisodes | NoView:
        """The title's window, recorded. Raises for an `ambiguous` id as the real resolver does."""

        del mapping
        self.calls.append((series_id, anilist_id))
        if anilist_id in self._ambiguous:
            raise ValueError("ambiguous")
        if self._windows is None:
            episodes = self._series_episodes(series_id)
            if isinstance(episodes, NoView):
                return episodes
        else:
            episodes = self._windows.get(anilist_id)
            if episodes is None:
                return NoView.READ_FAILED
        return TitleEpisodes(aired=tuple(episodes))

    def series_index(self, series_id: int) -> EpisodeIndex | None:
        """The whole series' index with no other order read onto it, None for a series with no view."""

        episodes = self._series_episodes(series_id)
        return None if isinstance(episodes, NoView) else episode_index(episodes)

    @override
    def read_episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """The whole-series list off the client, None for a series with no view."""

        episodes = self._series_episodes(series_id)
        return None if isinstance(episodes, NoView) else episodes

    @override
    def viewed_episodes(self, series_id: int) -> list[SonarrEpisode] | None:
        """The series list as last read off the client, read now if it never was. None for a series with no view."""

        if series_id in self._last_known and series_id not in self._no_views:
            return self._last_known[series_id]
        episodes = self._series_episodes(series_id)
        return None if isinstance(episodes, NoView) else episodes

    @override
    def last_known_episodes(self, series_id: int) -> list[SonarrEpisode]:
        """The series list as last read off the client, none for a series never read."""

        return self._last_known.get(series_id, [])

    @override
    def in_another_order(self, series_id: int) -> bool:
        """Whether the scripted order, which every series is numbered by, is other than the aired one."""

        del series_id
        return is_another_order(self._order)

    def library_orders(self, series_id: int) -> SonarrSeriesOrders:
        """Every series is numbered by the scripted order, with no count."""

        del series_id
        return SonarrSeriesOrders(tvdb_id=0, season_type=self._order, counts=None)

    def alternate_titles(self, series_id: int) -> tuple[AlternateTitle, ...]:
        """No series has one."""

        del series_id
        return ()

    @override
    def aired_parse(self, info: ParsedFileInfo) -> ParsedFileInfo | NoAnswer | NoView:
        """The parse as it is: a scripted parse is one an aired Sonarr gives."""

        return info

    def sonarr_files(self, episodes: Iterable[SonarrEpisode]) -> ArrFiles:
        """The arr has no files."""

        del episodes
        return ArrFiles()

    @override
    def episode_count(self, anilist_id: int) -> int | None:
        """The scripted count, None for an unscripted id."""

        return self._counts.get(anilist_id)

    def metadata_unread(self, anilist_ids: Iterable[int], series_ids: Iterable[int] = ()) -> bool:
        """Whether any of the titles is one of the `unread_anilist_ids`, or any of the series one of the `unread_series`."""

        return not (self._unread_anilist_ids.isdisjoint(anilist_ids) and self._unread_series.isdisjoint(series_ids))

    def title_unread(self, anilist_id: int, series_id: int) -> bool:
        """Whether the title is one of the `unread_anilist_ids`. No series' orders are ever behind here."""

        return anilist_id in self._unread_anilist_ids


class CaptureHandler(logging.Handler):
    """A logging handler that collects records into `records`."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
