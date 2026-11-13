# pyright: strict
# pyright: reportPrivateUsage=false
# These reach into the strategy's private collaborators (strat._episodes /
# strat._executor) to pin the seam. Strict re-flags that and the repo disables
# reportPrivateUsage for tests.
"""Seam tests for the composition split.

These pin the contract between the run machinery and the Arr strategies: each
`ArrStrategy` hook reaches the shared pipeline only through the injected
`ArrRunServices` the strategy holds as `self._services` - and since the hub
split out of the old god class, that seam is literal: the scripted fake below
is a real `ArrRunServices` subclass. The strategies are built bare
(`object.__new__`) so no live Sonarr/Radarr client is constructed.
"""

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, NamedTuple, cast, override

import pytest
from seadex import EntryRecord

from pearlarr import sonarr_import as sonarr_import_module
from pearlarr.arr_activity import HISTORY_MAX_LOOKBACK_DAYS
from pearlarr.arr_categories import ArrCategoryResolver
from pearlarr.arr_http import DeleteOutcome
from pearlarr.arr_models import (
    AlternateTitle,
    ArrFiles,
    CommandResource,
    HistoryPage,
    HistoryRecord,
    Language,
    ManualImportCandidate,
    MovieFile,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    RemotePathMapping,
    SeasonTypeCount,
    SonarrEpisode,
    SonarrItem,
)
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import Arr
from pearlarr.episode_state import EpisodeFileStatus, GrabbedGroups, ImportSnapshot, trusted_groups
from pearlarr.file_paths import normalize_name
from pearlarr.grab_pipeline import NO_PENDING_IMPORT_UPDATES, GrabRequest
from pearlarr.grab_placement import EntryPlacements, ListedFile, place_release
from pearlarr.import_quality import resolve_language_objects
from pearlarr.mappings import ExternalIds, MappingSource, ResolvedMappings, TitleMapping
from pearlarr.output import Accent, Severity, StyledValue, TitleDetail
from pearlarr.pending_import import (
    FileEpisodeMap,
    OwnedAtGrab,
    PendingImport,
    SizeMatchedEpisode,
    TitleNames,
    TorrentGroup,
)
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.placement_types import FetchedListings, TorrentListing, episode_index
from pearlarr.planner import GrabPlan
from pearlarr.progress import ProgressCallback
from pearlarr.radarr_strategy import _HISTORY_SKEW_HOURS, RadarrStrategy
from pearlarr.releases import EpisodeKey, OrderOnlyKey, ReleaseEpisode, SeadexDict, SkipReason
from pearlarr.sonarr_episodes import sonarr_series_hash
from pearlarr.sonarr_state import DownloadHistoryState, HistoryImport, placements_from_history
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.stamps import parse_stamp, stamp_of
from pearlarr.tvdb_orders import AIRED_ORDER, NO_INDEX_ORDERS, IndexOrders, OrderRow
from pearlarr.wait_types import (
    AttemptKind,
    AttemptResult,
    EffectStatus,
    HandImportFile,
    ImportProgress,
    ImportWaitMode,
    LeftForHandImport,
    PauseReason,
    PendingImportState,
)

from .builders import (
    PENDING_IMPORT_ADDED,
    SEP,
    FakeCacheStore,
    FakeClock,
    claim_al_ids,
    episode_snapshot,
    grab_plan,
    make_bare_instance,
    make_config,
    make_entry_record,
    make_fetched_categories,
    make_grabbing_title,
    make_import_wait_manager,
    make_logger,
    make_radarr_sync,
    make_release,
    make_run_deps,
    make_sonarr_episodes,
    make_sonarr_sync,
    manual_candidate,
    new_torrent,
    numbered_names,
    parsed_info,
    pending_import,
    queue_record,
    rg_group,
    sonarr_ep,
    title_claim,
    title_window,
    torrent_listing,
    two_claim_pending_import,
)
from .fakes import (
    NEUTRAL_RESCAN,
    FakeRadarrClient,
    FakeSonarrClient,
    HookedSonarrClient,
    QbitCategoryReads,
    RecordingHub,
    ScriptedEpisodes,
    debug_logger,
    diagnostic_messages,
    install_recording_hub,
)


class _Item:
    """A stand-in Arr item exposing whatever id attributes a test sets.

    Declares the full `ArrItem` surface so it structurally satisfies the
    `SonarrItem` / `RadarrItem` protocols. Each test sets only the attributes
    the hook under test actually reads.
    """

    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tvdbId: int
    tmdbId: int
    seasonType: str = AIRED_ORDER
    seasonTypes: tuple[SeasonTypeCount, ...] = ()
    alternateTitles: tuple[AlternateTitle, ...] = ()

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


class GetAniListIdsCall(NamedTuple):
    """One recorded `get_anilist_ids` call (the args the strategy forwarded).

    Defaults mirror the seam's own defaults, so a recorded call equals an expected
    one constructed with only the fields the strategy actually varied.
    """

    ids: ExternalIds
    log_ignored: bool = True


class _FakeRunServices(ArrRunServices):
    """A scripted `ArrRunServices` - the seam a strategy holds as `self._services`.

    A REAL `ArrRunServices` subclass (the seam name is literal now, so it also
    satisfies the constructors' typed `services` parameter), with its own
    `__init__` that never calls the heavy real one. Every member of the
    strategy-facing surface is overridden: each scriptable result is a
    constructor arg, and the methods whose call a test asserts RECORD it - so
    the contract is pinned by recorded state.
    """

    def __init__(
        self,
        *,
        anilist_ids: dict[int, TitleMapping] | None = None,
        ignored: dict[int, TitleMapping] | None = None,
        prologue_entry: EntryRecord | None = None,
        anilist_title: str = "Title",
        cached_skip: bool = False,
        needs_scan: bool = True,
        seadex_dict: SeadexDict | None = None,
        interactive_result: SeadexDict | None = None,
        filter_downloads_result: GrabPlan | None = None,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
        arr: Arr = Arr.SONARR,
        cache_store: FakeCacheStore | None = None,
    ) -> None:
        # The record seam the strategies read stored records through, bound as a run would bind it.
        self.cache_store = cache_store if cache_store is not None else FakeCacheStore()
        self._pending_imports = PendingImportStore(self.cache_store)
        self._pending_imports.bind_arr_run(ArrRunState(arr=arr, import_wait_mode=import_wait_mode))
        self._anilist_ids = anilist_ids or {}
        self._ignored = ignored or {}
        self._prologue_entry = prologue_entry
        self._anilist_title = anilist_title
        self._cached_skip = cached_skip
        self._needs_scan = needs_scan
        self._seadex_dict: SeadexDict = seadex_dict if seadex_dict is not None else {}
        self._interactive_result = interactive_result
        self._filter_downloads_result = filter_downloads_result
        self._import_wait_mode = import_wait_mode
        self.get_anilist_ids_calls: list[GetAniListIdsCall] = []
        self.al_id_prologue_calls: list[int] = []
        self.no_episodes_titles: list[str] = []
        self.unmonitored_titles: list[str] = []
        self.missing_episodes: list[str] = []
        self.log_al_title_calls: list[str] = []
        self.cached_skip_coverages: list[Callable[[], str]] = []
        self.get_seadex_dict_calls: list[EntryRecord] = []
        self.interactive_calls: list[SeadexDict] = []
        self.filter_downloads_calls: list[tuple[SeadexDict, ArrFiles]] = []
        self.grab_requests: list[GrabRequest] = []
        self.no_releases_calls: list[tuple[int, CachedTitleUpdate]] = []
        self.no_releases_unread: list[int] = []
        self.invalid_selection_skips = 0
        self.pauses_owed = 0

    @property
    @override
    def pending_imports(self) -> PendingImportStore:
        return self._pending_imports

    @override
    def title_needs_check(self, al_id: int) -> bool:
        del al_id
        return self._needs_scan

    @override
    def title_mappings(self, ids: ExternalIds, log_ignored: bool = True) -> dict[int, TitleMapping]:
        self.get_anilist_ids_calls.append(GetAniListIdsCall(ids, log_ignored))
        return self._anilist_ids

    @override
    def mappings_with_ignored(self, ids: ExternalIds) -> ResolvedMappings:
        del ids
        return ResolvedMappings(kept=self._anilist_ids, ignored=self._ignored)

    @override
    def begin_check(self, al_id: int) -> EntryRecord | None:
        self.al_id_prologue_calls.append(al_id)
        return self._prologue_entry

    @override
    def is_unchanged(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        coverage: Callable[[], str],
    ) -> bool:
        del al_id, sd_entry
        self.cached_skip_coverages.append(coverage)
        return self._cached_skip

    @override
    def resolve_title(self, al_id: int) -> Title:
        return Title(al_id=al_id, name=self._anilist_title, names=TitleNames(anilist=(self._anilist_title,)))

    @override
    def new_cached_title_update(self, title: Title, sd_entry: EntryRecord) -> CachedTitleUpdate:
        # A pure builder over its arguments (no hub state), so the real one serves the fake.
        return super().new_cached_title_update(title, sd_entry)

    @override
    def get_seadex_dict(self, sd_entry: EntryRecord) -> SeadexDict:
        self.get_seadex_dict_calls.append(sd_entry)
        return self._seadex_dict

    @override
    def filter_seadex_interactive(
        self,
        seadex_dict: SeadexDict,
        sd_entry: EntryRecord,
    ) -> SeadexDict:
        del sd_entry
        self.interactive_calls.append(seadex_dict)
        return seadex_dict if self._interactive_result is None else self._interactive_result

    @override
    def filter_seadex_downloads(
        self,
        seadex_dict: SeadexDict,
        arr_files: ArrFiles,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> GrabPlan:
        del ep_list
        self.filter_downloads_calls.append((seadex_dict, arr_files))
        if self._filter_downloads_result is not None:
            return self._filter_downloads_result
        return grab_plan(seadex_dict)

    @property
    @override
    def import_wait_mode(self) -> ImportWaitMode:
        return self._import_wait_mode

    @override
    def report_no_releases(self, al_id: int, update: CachedTitleUpdate, *, metadata_unread: bool = False) -> None:
        self.no_releases_calls.append((al_id, update))
        if metadata_unread:
            self.no_releases_unread.append(al_id)

    @override
    def report_invalid_selection(self) -> None:
        self.invalid_selection_skips += 1

    @override
    def owe_pause(self) -> None:
        self.pauses_owed += 1

    @override
    def grab_and_cache(self, req: GrabRequest) -> None:
        self.grab_requests.append(req)

    @override
    def log_no_episodes(self, title: str) -> None:
        self.no_episodes_titles.append(title)

    @override
    def log_arr_item_unmonitored(self, item_title: str) -> None:
        self.unmonitored_titles.append(item_title)

    @override
    def log_missing_episodes(self, missing: str) -> None:
        self.missing_episodes.append(missing)

    @override
    def open_title_block(self, title: str, sd_entry: EntryRecord, coverage: str | None = None) -> None:
        del sd_entry, coverage
        self.log_al_title_calls.append(title)


def test_fake_overrides_the_full_public_surface() -> None:
    """`@override` catches renames and signature drift but NOT additions.

    A new public `ArrRunServices` member would be silently inherited here (real
    body over fake state) and only fail at test runtime. Reflection closes
    that gap - extend the fake when this fails.
    """

    # Loop-driven members (the runner calls these, strategies never do), so the
    # fake doesn't script them.
    lifecycle = {"bind_arr_run", "state", "is_dry_run", "mark_dirty", "pay_pause", "selection_changed"}
    surface = {
        name
        for name, member in vars(ArrRunServices).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    overridden = {name for name in vars(_FakeRunServices) if not name.startswith("_")}
    assert surface - lifecycle <= overridden


def test_the_hub_logs_missing_episodes_as_a_caution_detail() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_missing_episodes("S01 E02")

    assert recording.of_type(TitleDetail) == [TitleDetail("missing", StyledValue("S01 E02", Accent.CAUTION))]


class _ScriptedListings:
    """A scripted listings collaborator: per-infohash listings and size identities, or a SeaDex outage."""

    def __init__(
        self,
        listed: Mapping[str, TorrentListing] | None = None,
        episode_by_size: Mapping[int, int] | None = None,
        *,
        unreachable: bool = False,
    ) -> None:
        self._read = None if unreachable else FetchedListings(listed or {}, episode_by_size or {})
        self.reads: list[tuple[int, ResolvedMappings]] = []

    def read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        self.reads.append((series_id, resolved))
        return self._read


class _ScriptedParse:
    """Parse collaborator serving scripted files per url (no Sonarr round-trip), none for an unscripted url."""

    def __init__(self, files_by_url: dict[str, tuple[ListedFile, ...]] | None = None) -> None:
        self._files = files_by_url or {}

    def parsed_files(self, seadex_dict: SeadexDict, *, series_hash: str) -> dict[str, tuple[ListedFile, ...]]:
        del series_hash
        return {
            url: self._files.get(url, ()) for group_releases in seadex_dict.values() for url in group_releases.releases
        }


class TestItemAnilistIdsDelegates:
    """item_anilist_ids resolves through the held services, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        run = _FakeRunServices(anilist_ids={7: TitleMapping(anilist_id=7)})
        strat = make_bare_instance(RadarrStrategy, _services=run)

        result = strat.item_mappings(_Item(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: TitleMapping(anilist_id=7)}
        assert run.get_anilist_ids_calls == [
            GetAniListIdsCall(ExternalIds(tmdb_movie=42, imdb="tt7"), log_ignored=False)
        ]

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrStrategy, _services=run)

        strat.item_mappings(_Item(tvdbId=99, imdbId="tt9"))

        assert run.get_anilist_ids_calls == [GetAniListIdsCall(ExternalIds(tvdb=99, imdb="tt9"), log_ignored=True)]


class TestFilterToSingle:
    """filter_to_single narrows by the arr's external id (no engine needed)."""

    def test_radarr_matches_tmdb_id(self) -> None:
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())
        items: list[RadarrItem] = [_Item(tmdbId=1), _Item(tmdbId=2)]

        assert strat.filter_to_single(items, 2) == [items[1]]
        assert strat.filter_to_single(items, 7) == []

    def test_sonarr_matches_tvdb_id(self) -> None:
        strat = make_bare_instance(SonarrStrategy, logger=make_logger())
        items: list[SonarrItem] = [_Item(tvdbId=10), _Item(tvdbId=20)]

        assert strat.filter_to_single(items, 10) == [items[0]]


class TestRunStartHook:
    """get_items doubles as the run-start hook: it resets the per-run scratch.

    The episode reset/fingerprint now lives on the SonarrEpisodes collaborator.
    This stays at strategy level to pin that get_items actually routes through it.
    """

    def test_sonarr_get_items_clears_ep_list_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        series: list[SonarrItem] = [_Item(id=5, tvdbId=50), _Item(id=7, tvdbId=70)]
        strat = make_sonarr_sync(ep_list_cache={5: [sonarr_ep(1, 1)]})
        strat._episodes._orderings = {5: NO_INDEX_ORDERS}
        strat._episodes._picks_logged = {11}

        def _all_series() -> list[SonarrItem]:
            return series

        monkeypatch.setattr(strat._episodes, "get_all_sonarr_series", _all_series)

        result = strat.get_items()

        assert result == series
        # The episode collaborator drops its caches + re-fingerprints as it enumerates.
        assert strat._episodes._ep_list_cache == {}
        assert strat._episodes._orderings == {}
        assert strat._episodes._picks_logged == set()
        assert strat._episodes._series_hash == sonarr_series_hash([5, 7])


class TestSonarrPrefetchDelegates:
    """prefetch_episodes is a thin hook over the episode collaborator's warm."""

    def test_sonarr_prefetch_routes_to_episodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strat = make_sonarr_sync()
        prefetch_calls: list[tuple[list[SonarrItem], ProgressCallback | None]] = []

        def _prefetch(items: list[SonarrItem], *, progress: ProgressCallback | None = None) -> int:
            prefetch_calls.append((items, progress))
            return 3

        monkeypatch.setattr(strat._episodes, "prefetch", _prefetch)
        items: list[SonarrItem] = [_Item(id=1)]

        assert strat.prefetch_episodes(items, progress=None) == 3
        assert prefetch_calls == [(items, None)]


class TestRadarrPrefetchEpisodes:
    """Radarr has no episodes: the warm hook is a no-op that warms nothing."""

    def test_returns_zero_and_skips_sink(self) -> None:
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())
        calls: list[tuple[float, str | None]] = []

        class _Rec:
            def progress(self, fraction: float, detail: str | None = None) -> None:
                calls.append((fraction, detail))

        assert strat.prefetch_episodes([_Item(tmdbId=1)], progress=_Rec()) == 0
        assert calls == []  # no episodes -> sink never driven

    def test_warms_episodes_is_false(self) -> None:
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())
        assert strat.warms_episodes is False


class TestProcessAlIdThreadsServices:
    """The per-id head runs through the held services. A missing entry stops this id."""

    def test_radarr_no_seadex_entry_stops_at_the_prologue(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(RadarrStrategy, _services=run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_seadex_entry_stops_at_the_prologue(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrStrategy, _services=run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_episodes_resolved_skips_explicitly(self) -> None:
        # An anime-id mapping that resolves to [] (season not in Sonarr / offset past
        # the end): skip with a no-episodes row, never mislabeled "unmonitored"
        # and never falling through to grab orphans - and NO AniBridge warning.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[]))
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.no_episodes_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert run.pauses_owed == 1
        # anime-id empty is NOT the AniBridge case -> no warning surfaced.
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_sonarr_anibridge_empty_map_skips_with_warning(self) -> None:
        # The AniBridge no-usable-ranges case (a real empty-{} tvdb entry: source
        # ANIBRIDGE, mode ANIBRIDGE): the NO_EPISODES skip PLUS a visible WARNING
        # naming the cause. The warning keys off source (so it also covers the
        # degraded imdb/tmdb case), so the entry must carry source=ANIBRIDGE as a
        # real one does. Fails on the unfixed path, which silently grabbed nothing.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[]))
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(
            _Item(id=1, title="Title"),
            5,
            TitleMapping(anilist_id=5, tvdb_mappings={}, source=MappingSource.ANIBRIDGE),
        )
        assert run.no_episodes_titles == ["Title"]
        assert run.log_al_title_calls == []
        # AniBridge-specific notice surfaced.
        assert any(
            "AniBridge has no usable season ranges" in message
            for message in diagnostic_messages(recording, Severity.WARNING)
        )

    def test_sonarr_all_unmonitored_episodes_skip_with_an_unmonitored_row(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        unmonitored = sonarr_ep(1, 1).model_copy(update={"monitored": False})
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=[unmonitored])),
            _config=make_config(sleep_time=0, ignore_unmonitored=True),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.unmonitored_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert run.pauses_owed == 1

    def test_sonarr_reports_the_windows_missing_episodes(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        window = [sonarr_ep(1, 1), sonarr_ep(1, 2, episode_file_id=0)]
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=window)),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.missing_episodes == ["S01 E02"]

    def test_sonarr_all_invalid_selection_routes_to_shared_skip(self) -> None:
        # The interactive pick rejected every token: the strategy must route to
        # report_invalid_selection (which persists nothing) and never reach the
        # grab tail - the Sonarr twin of the Radarr guard in test_seadex_dict.
        seadex_dict: SeadexDict = {}
        for name in ("GroupA", "GroupB"):
            seadex_dict.update(_one_group_dict(name))
        run = _FakeRunServices(
            prologue_entry=make_entry_record(),
            seadex_dict=seadex_dict,
            interactive_result={},
        )
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=[sonarr_ep(1, 1)])),
            _parse=_ScriptedParse(),
            _listings=_ScriptedListings(),
            _config=make_config(interactive=True, sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=1, title="Title", tvdbId=10, imdbId=None), 5, TitleMapping(anilist_id=5))
        assert run.invalid_selection_skips == 1
        assert run.grab_requests == []
        assert run.no_releases_calls == []


class _QbitState(NamedTuple):
    """The two `state_enum` booleans `poll_torrent` reads."""

    is_complete: bool = True
    is_errored: bool = False


class _CompletedRow(NamedTuple):
    """One complete qBittorrent info row (the fields `poll_torrent` reads)."""

    content_path: str
    state_enum: _QbitState = _QbitState()
    progress: float = 1.0


class _CompletedQbit:
    """A qBittorrent stub whose heavy poll always reads one complete torrent."""

    def __init__(self, content_path: str) -> None:
        self._row = _CompletedRow(content_path)

    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[_CompletedRow]:
        del torrent_hashes
        return [self._row]


_RESCAN_NAME = "RefreshMonitoredDownloads"
_RESCAN_AFTER_ADD = CommandResource(
    name=_RESCAN_NAME, status="completed", started=PENDING_IMPORT_ADDED + timedelta(days=1)
)
_RESCAN_BEFORE_ADD = CommandResource(
    name=_RESCAN_NAME, status="completed", started=PENDING_IMPORT_ADDED - timedelta(days=1)
)


def _make_sonarr_for_import(
    *,
    candidates: list[ManualImportCandidate] | None,
    queue: list[QueueRecord] | None = None,
    episodes: list[SonarrEpisode] | None = None,
    quality_defs: list[QualityDefinition] | None = None,
    languages: list[Language] | None = None,
    commands: list[CommandResource] | None = None,
    command_status: CommandResource | None = None,
    command_status_script: list[CommandResource] | None = None,
    cmd_id: int | None = 42,
    config_overrides: dict[str, list[str] | str] | None = None,
    cache_store: FakeCacheStore | None = None,
    clock: FakeClock | None = None,
    categories: ArrCategoryResolver | None = None,
) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A bare `SonarrStrategy` plus its scripted `self.sonarr` `HookedSonarrClient`.

    The fake returns the given queue records, current episodes, manual-import
    candidates, quality definitions, languages and an execute command id -
    everything `import_completed` reaches over the network - and records the two
    import commands so a test asserts on recorded state. `episodes` defaults to
    empty, so the target episodes have NO file yet (they need importing) and the
    done-check never short-circuits. Pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. `queue` defaults to empty (Sonarr
    isn't tracking the download, so the strategy steps in). `commands` lists the
    commands beside the fake's neutral rescan, so no list given here reads as a
    restart. The fake's refresh / command-status defaults resolve immediately so
    the rescan never waits.
    """

    sonarr = HookedSonarrClient(
        queue=queue,
        episodes=episodes,
        commands=None if commands is None else [NEUTRAL_RESCAN, *commands],
        command_status=command_status,
        command_status_script=command_status_script,
        candidates=candidates,
        quality_defs=quality_defs,
        languages=languages,
        execute_command_id=cmd_id,
    )
    overrides: dict[str, list[str] | str] = config_overrides or {}
    strat = make_sonarr_sync(
        sonarr=sonarr,
        config=make_config(**overrides),
        cache_store=cache_store or FakeCacheStore(),
        clock=clock,
        categories=categories,
    )
    return strat, sonarr


def _posted(sonarr: FakeSonarrClient) -> list[list[tuple[str, list[int]]]]:
    """The `(path, episode ids)` of every file in each manual import sent to Sonarr, one list per import."""

    return [[(f.path, f.episodeIds) for f in files] for files, _mode in sonarr.execute_calls]


class _RangeFile(NamedTuple):
    """The pending record of one range file, the strategy that imports it, and the Sonarr fake it talks to."""

    pending: PendingImport
    strat: SonarrStrategy
    sonarr: FakeSonarrClient


class TestImportCompletedQueueState:
    """import_completed reads the queue + episode files before stepping in."""

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_importing_defers_with_credit(self, attempt_kind: AttemptKind) -> None:
        # Sonarr is copying this download (importing): never double-import, and
        # the copy is Sonarr's wait, so the monitor credits it (a serial pack
        # import must not expire the record at the deadline). Counts kept.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importing")],
        )

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert result.pause_reason is PauseReason.IMPORT
        assert result.command_issued is False
        assert (result.imported_count, result.file_map_episode_count) == (0, 1)
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_queue_downloading_waits_without_credit(self) -> None:
        # Sonarr's view lags qBittorrent's "complete": the record's own wait, so
        # the ready clock keeps running.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "downloading")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.paused is False
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_clean_pending_retries_until_forced(self) -> None:
        # A clean importPending: defer to Sonarr unless forced.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="ok")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert sonarr.candidate_calls == []

    def test_clean_pending_at_deadline_defers_while_a_pass_runs(self) -> None:
        # Sonarr imports packs serially: a clean pending row behind a running
        # pass is next in line, not stalled. The deadline attempt neither steps
        # in past the pass nor graduates; the wait is credited.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="ok")],
            commands=[_started_process_pass()],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.pause_reason is PauseReason.BUSY
        assert result.command_issued is False
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_clean_pending_before_deadline_credits_only_while_sonarr_is_busy(self) -> None:
        # The command check runs ahead of the clean-pending wait: a running
        # pass credits the poll, an idle Sonarr that has not imported the row
        # burns its clock (the deadline step-in still meets it).
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="ok")],
            commands=[_started_process_pass()],
        )

        busy = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        idle = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert busy.pause_reason is PauseReason.BUSY
        assert idle.paused is False
        assert sonarr.candidate_calls == []

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_outage_retries_instead_of_stepping_in(self, attempt_kind: AttemptKind) -> None:
        # A failed queue read must never read as "not tracked" (stepping in would
        # race an import Sonarr may be running). Waiting with no deferral credit
        # fails closed on BOTH passes: the poll loop retries, and a persistent
        # outage burns the ready deadline so the record graduates still pending.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
        )
        sonarr.queue_return = None

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert result.paused is False
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_queue_read_recovers_on_a_later_poll(self) -> None:
        # The outage is retried via the poll loop: once the queue answers again
        # the same record proceeds to the normal step-in.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
        )
        sonarr.queue_return = None

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.queue_return = []
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert first.all_done is False
        assert first.command_issued is False
        assert second.command_issued is True
        assert len(sonarr.execute_calls) == 1

    def test_running_disk_command_defers_even_forced(self) -> None:
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            commands=[_started_process_pass()],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.all_done is False
        # A foreign command pauses the ready clock (BUSY, never a "still
        # importing" claim); the monitor's credit cap bounds the walk-away.
        assert result.pause_reason is PauseReason.BUSY
        assert result.command_issued is False
        assert (result.imported_count, result.file_map_episode_count) == (0, 1)
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_own_running_import_defers_other_records_with_credit(self) -> None:
        # Record A's issued copy is still running while record B polls: B defers,
        # and because the running command is OURS (an id we POSTed this
        # run) the probe reads `deferred` - the monitor credits B's wait back to
        # its ready deadline instead of burning it.
        pending_a = pending_import(infohash="abc123")
        pending_b = pending_import(infohash="def456")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cmd_id=42,
        )

        first = strat.import_completed(pending_a, "/d", AttemptKind.ORDINARY)
        sonarr.commands_return = [_inflight_manual_import("ABC123", command_id=42)]
        result = strat.import_completed(pending_b, "/e", AttemptKind.ORDINARY)

        assert first.command_issued is True
        assert result.all_done is False
        assert result.paused is True
        assert result.command_issued is False  # not B's command
        assert len(sonarr.execute_calls) == 1  # only A's import was POSTed

    def test_in_flight_import_for_this_record_reads_honest_flags(self) -> None:
        # A still-running ManualImport covering THIS record (e.g. POSTed by a
        # prior run): checked ahead of the anonymous disk guard, so the probe
        # reads `command_issued` (an at-deadline outcome must say "still
        # importing", never "not ready") and `deferred` (our own copy's runtime
        # never burns the clock).
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            commands=[_inflight_manual_import("ABC123")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.command_issued is True
        assert result.paused is True
        assert sonarr.execute_calls == []

    def test_two_claim_record_defers_on_the_torrents_running_import(self) -> None:
        # Two entries claim one torrent. While the torrent's command runs
        # (download-id match), the record's poll defers with credit: a
        # same-download command is ours by definition, and the copy serializes
        # every claim's slice.
        pending = pending_import(
            infohash="abc123",
            claims=(title_claim(), title_claim(al_id=777, series_id=8, title="Other")),
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            commands=[_inflight_manual_import("ABC123")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.paused is True
        assert result.command_issued is True
        assert sonarr.execute_calls == []

    def test_unproven_folder_import_waits_with_credit_but_no_claim(self) -> None:
        # A no-download-id ManualImport matching only by path overlap (possibly
        # a foreign folder import): waits, never stacking a duplicate, and the
        # wait is credited (an import is running either way), but unproven, so
        # no "still importing" claim.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            commands=[
                CommandResource.model_validate(
                    {
                        "id": 7000,
                        "name": "ManualImport",
                        "status": "started",
                        "body": {"files": [{"path": "/d/Show - 01 [1080p].mkv"}]},
                    },
                ),
            ],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.pause_reason is PauseReason.IMPORT
        assert result.command_issued is False
        assert sonarr.execute_calls == []

    def test_step_in_post_is_a_credited_wait(self) -> None:
        # The accepted POST is our copy in flight (ISSUED, credited once by the
        # monitor), so a deadline step-in is not graduated on the same poll; the
        # next poll finds the command started under its issued id (IMPORT).
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cmd_id=42,
        )

        issued = strat.import_completed(pending, "/d", AttemptKind.FINAL)
        sonarr.commands_return = [
            CommandResource.model_validate(
                {
                    "id": 42,
                    "name": "ManualImport",
                    "status": "started",
                    "body": {"files": [{"path": "/d/Show - 01 [1080p].mkv"}]},
                },
            ),
        ]
        in_flight = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert (issued.command_issued, issued.pause_reason) == (True, PauseReason.ISSUED)
        assert (in_flight.command_issued, in_flight.pause_reason) == (True, PauseReason.IMPORT)
        assert len(sonarr.execute_calls) == 1

    def test_run_start_clears_the_issued_command_ids(self) -> None:
        # A stale id from a previous run must never claim a foreign command as
        # ours (ownership is deliberately per-run; the same-download case
        # survives restarts via its download-id match instead).
        pending = pending_import(infohash="abc123")
        strat, _sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cmd_id=42,
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        executor = strat._reconciler._executor

        assert executor._is_auto_import_command(42) is True
        executor.reset()
        assert executor._is_auto_import_command(42) is False

    def test_an_excluded_file_is_not_imported_once_the_mapped_targets_are_in(self) -> None:
        done_file = "Show - 01 [1080p].mkv"
        excluded_file = "Show - 02 [1080p].mkv"
        pending = pending_import(
            file_episode_map={done_file: [101]},
            ordered_episode_ids=[101, 102],
            seadex_files=[done_file, excluded_file],
            excluded_files=[normalize_name(excluded_file)],
        )
        sonarr = FakeSonarrClient(
            episodes=[
                sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
                SonarrEpisode.model_validate({"id": 102, "seasonNumber": 1, "episodeNumber": 2}),
            ],
            candidates=[manual_candidate(f"/d/{done_file}"), manual_candidate(f"/d/{excluded_file}")],
            quality_defs=[],
            languages=[],
            execute_command_id=42,
        )
        strat = make_sonarr_sync(sonarr=sonarr, config=make_config(), cache_store=FakeCacheStore())

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert sonarr.execute_calls == []

    @pytest.mark.parametrize(
        ("command_status", "command_status_script", "refresh_calls"),
        [
            pytest.param(
                None,
                [CommandResource(status="started"), CommandResource(status="completed")],
                1,
                id="slow-completes",
            ),
            pytest.param(CommandResource(status="started"), None, 2, id="never-completes"),
        ],
    )
    def test_rescan_marker_lands_only_on_a_completed_rescan(
        self,
        command_status: CommandResource | None,
        command_status_script: list[CommandResource] | None,
        refresh_calls: int,
    ) -> None:
        clock = FakeClock()
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            command_status=command_status,
            command_status_script=command_status_script,
            clock=clock,
        )
        pending = pending_import(infohash="abc123")

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        clock.tick(30)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == refresh_calls

    def test_rescan_asks_again_when_a_restart_orphans_it(self) -> None:
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            command_status_script=[CommandResource(status="orphaned")],
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 2

    def test_rescan_throttle_measures_from_the_rescan_end(self) -> None:
        sonarr = FakeSonarrClient(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            command_status_script=[CommandResource(status="started")] * 3,
            execute_command_id=42,
        )
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(),
            cache_store=FakeCacheStore(),
            clock=FakeClock(step=10),
        )

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="def456"), "/e", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    def test_rescan_throttle_holds_after_a_rejected_post(self) -> None:
        # The stamp lands on every exit, a rejected rescan included, so the
        # next record in the pass never retries it inside the window.
        sonarr = FakeSonarrClient(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            execute_command_id=42,
            refresh_count=None,
        )
        strat = make_sonarr_sync(
            sonarr=sonarr,
            config=make_config(),
            cache_store=FakeCacheStore(),
            clock=FakeClock(step=10),
        )

        strat.import_completed(pending_import(infohash="abc123"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="def456"), "/e", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1

    def test_clean_pending_forced_steps_in(self) -> None:
        # force=True (snapshot / final monitor poll): stop deferring, issue the
        # import. The copy is async, so this reads command_issued, NOT a
        # verified all_done.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="ok")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.command_issued is True
        assert result.all_done is False
        assert len(sonarr.execute_calls) == 1

    def test_pending_with_warning_steps_in(self) -> None:
        # A warning-flagged importPending is a failed Sonarr import attempt (e.g.
        # the file wasn't visible on its mount yet) that Sonarr won't reliably
        # retry - waiting would only burn the readiness deadline, so step in.
        # (Observed live: four flagged-pending torrents sat 10+ minutes untouched.)
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="warning")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1

    def test_target_already_recommended_drops_record(self) -> None:
        # Episode files are the source of truth for "already imported": the target
        # episode already holds the recommended group's file -> done, no scan.
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert sonarr.candidate_calls == []

    def test_target_holding_another_entry_group_is_protected(self) -> None:
        # The overwrite guard spans every group the entry's picks carried at
        # grab time, not just grabbed ones: a file from a pick we never grabbed
        # reads RECOMMENDED -> done, never imported over.
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            guards=OwnedAtGrab(entry_groups=("SubGroup", "OtherPick")),
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="OtherPick")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert sonarr.candidate_calls == []

    @staticmethod
    def _misfiled() -> tuple[PendingImport, SonarrStrategy, FakeSonarrClient]:
        """Our E02 file (200) filed on E01 as well: both hold our group at a listed size, E01 not its own."""

        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101], "Show - 02 [1080p].mkv": [102]},
            seadex_files=["Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"],
            release_sizes=[100, 200],
            sizes_by_name={"show - 01 [1080p].mkv": 100, "show - 02 [1080p].mkv": 200},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), manual_candidate("/d/Show - 02 [1080p].mkv")],
            episodes=[
                sonarr_ep(1, 1, ep_id=101, episode_file_id=12, release_group="SubGroup", size=200),
                sonarr_ep(1, 2, ep_id=102, episode_file_id=12, release_group="SubGroup", size=200),
            ],
        )
        return pending, strat, sonarr

    def test_a_misfiled_target_is_imported_over_then_its_sibling_then_verifies(self) -> None:
        pending, strat, sonarr = self._misfiled()
        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        # Importing file 01 onto 101 deletes the shared episode file, so 102 is left with none.
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=11, release_group="SubGroup", size=100),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=11, release_group="SubGroup", size=100),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=13, release_group="SubGroup", size=200),
        ]

        third = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[("/d/Show - 01 [1080p].mkv", [101])], [("/d/Show - 02 [1080p].mkv", [102])]]
        assert (first.command_issued, second.command_issued, third.all_done) == (True, True, True)

    @staticmethod
    def _range_file(episode: SonarrEpisode, guards: OwnedAtGrab) -> _RangeFile:
        """A record whose one E01-02 file (size 300) is placed on both episodes, E01 holding `episode` and E02 empty."""

        name = "Show - S01E01-02 [1080p].mkv"
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={name: [101, 102]},
            release_sizes=[300],
            sizes_by_name={name.casefold(): 300},
            guards=guards,
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}")],
            episodes=[episode, sonarr_ep(1, 2, ep_id=102, episode_file_id=0)],
        )
        return _RangeFile(pending, strat, sonarr)

    def test_a_range_file_replaces_our_own_copy_beside_its_open_episode(self) -> None:
        # E01 already holds our file at its listed size, so the import posts both episodes and replaces that copy.
        own = sonarr_ep(1, 1, ep_id=101, episode_file_id=11, release_group="SubGroup", size=300)
        pending, strat, sonarr = self._range_file(own, OwnedAtGrab())
        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, episode_file_id=12, release_group="SubGroup", size=300),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=12, release_group="SubGroup", size=300),
        ]

        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[("/d/Show - S01E01-02 [1080p].mkv", [101, 102])]]
        assert (first.command_issued, second.all_done) == (True, True)

    def test_a_range_file_leaves_another_releases_recommended_file_alone(self) -> None:
        other = sonarr_ep(1, 1, ep_id=101, episode_file_id=11, release_group="OtherPick", size=900)
        pending, strat, sonarr = self._range_file(other, OwnedAtGrab(entry_groups=("SubGroup", "OtherPick")))

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[("/d/Show - S01E01-02 [1080p].mkv", [102])]]

    def test_target_holding_an_untagged_file_we_identified_is_protected(self) -> None:
        # The grab-time size identification rides the record through to the
        # import: an untagged file the planner named reads RECOMMENDED, so the
        # pack we grabbed for the REST of the season never copies over it.
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            guards=OwnedAtGrab(owned_episodes=(SizeMatchedEpisode(101, 700),)),
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group=None, size=700)],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert sonarr.candidate_calls == []

    def test_flagged_record_with_files_gone_demotes_through_the_reconciler(self) -> None:
        # The heal's demote reads the POST's ISSUED reason off the real probe:
        # a flagged record whose files are gone re-imports and drops back to an
        # ordinary carryover (the fix's wiring, not a scripted probe).
        recording = install_recording_hub()
        pending = pending_import(infohash="abc123", awaiting_cleanup=True)
        store = FakeCacheStore()
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cache_store=store,
        )
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        mgr = make_import_wait_manager(qbit=_CompletedQbit("/d"), cache_store=store, strategy=strat)

        mgr._cleanup.finish_cleanups()

        assert len(sonarr.execute_calls) == 1
        assert mgr._pending_imports.rows()[pending.infohash].get("awaiting_cleanup") is False
        assert mgr._state.pending_import_states[pending.infohash] is PendingImportState.DOWNLOADED
        assert any("no longer present" in w for w in diagnostic_messages(recording, Severity.WARNING))

    def test_store_persisted_guard_row_protects_through_the_snapshot_seam(self) -> None:
        # THE fix path end to end: the persisted blob carries NO guards - the
        # entry's guard_facts row alone must feed the rehydrated record's
        # protection decision when the carried-over snapshot drives the import.
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            guards=OwnedAtGrab(entry_groups=("SubGroup", "OtherPick")),
        )
        store = FakeCacheStore()
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="OtherPick")],
            cache_store=store,
        )
        (claim,) = pending.claims
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        store.put_guards(Arr.SONARR, claim.al_id, claim.guards)
        mgr = make_import_wait_manager(
            qbit=_CompletedQbit("/d"),
            cache_store=store,
            strategy=strat,
        )

        mgr.report_imports_for_series(claim.series_id)

        # The row's facts reached the decision: the on-disk OtherPick file reads
        # RECOMMENDED - verified done, never manually imported over - and the
        # finished record leaves the store. The IMPORTED classification pins that
        # arm (a MISSING drop would also empty the store).
        assert sonarr.candidate_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert mgr._state.pending_import_states[pending.infohash] is PendingImportState.IMPORTED

    def test_import_blocked_steps_in_with_our_mapping(self) -> None:
        # Sonarr can't auto-import (importBlocked) -> our authoritative manual
        # import takes over and ISSUES the command. The copy is async, so right
        # after issuing the probe reads command_issued (NOT all_done).
        # A later monitor cycle flips to all_done once the episode files land.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importBlocked", status="warning")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert result.all_done is False
        assert len(sonarr.candidate_calls) == 1
        assert len(sonarr.execute_calls) == 1

    def test_later_poll_observes_freshly_imported_files(self) -> None:
        # Regression: import verification reads the episode FILES as the source of
        # truth, so the episode list must be re-fetched each poll, never served
        # stale from the per-run cache. Poll 1 (target absent) issues the import
        # (command_issued). Once the copy lands, poll 2 must observe the
        # file -> all_done, WITHOUT re-issuing. A stale cache would
        # keep all_done False forever (the monitor times out as "still
        # importing", and in move mode the import is never confirmed at all).
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importBlocked", status="warning")],
        )

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert first.all_done is False
        assert first.command_issued is True

        # The copy landed: the target episode now holds the recommended file.
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]

        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert second.all_done is True
        assert len(sonarr.execute_calls) == 1

    def test_not_in_queue_steps_in(self) -> None:
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        assert len(sonarr.candidate_calls) == 1
        assert len(sonarr.execute_calls) == 1


def _started_process_pass() -> CommandResource:
    """Sonarr's own ProcessMonitoredDownloads pass, executing right now."""

    return CommandResource.model_validate({"name": "ProcessMonitoredDownloads", "status": "started"})


def _inflight_manual_import(infohash: str, *, status: str = "started", command_id: int = 0) -> CommandResource:
    """A ManualImport command whose one file carries `infohash` as downloadId."""

    return CommandResource.model_validate(
        {
            "id": command_id,
            "name": "ManualImport",
            "status": status,
            "body": {"files": [{"downloadId": infohash, "episodeIds": [101]}]},
        },
    )


class TestCloseTracked:
    """close_tracked dismisses the download's leftover queue entry through any one row's id."""

    def test_dismisses_by_the_first_usable_row_id(self) -> None:
        # Sonarr closes the WHOLE tracked download off any one queue id, so the
        # first row with a usable id (junk folds to 0) is deleted, exactly once.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[
                queue_record("ABC123", "importPending", status="warning"),  # id 0: unusable
                queue_record("ABC123", "importPending", status="warning", queue_id=11),
                queue_record("ABC123", "importPending", status="warning", queue_id=12),
            ],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]

    def test_other_downloads_rows_are_not_touched(self) -> None:
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("FFFF", "importPending", queue_id=9)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_unknown_series_rows_are_left_for_sonarr(self) -> None:
        # Dismissing an unknown-series entry 500s in Sonarr without recording
        # the ignore; the row drops once the torrent leaves the watched category.
        # SKIPPED, never FAILED: an unknown-series entry can't strand the record.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", status="warning", queue_id=11, series_id=0)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_already_clear_queue_is_a_no_op(self) -> None:
        # Sonarr closed the download itself (one import covered the full grab).
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(candidates=None)

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_queue_outage_fails_the_effect_without_a_guessed_delete(self) -> None:
        # A failed read already retried inside the transport: FAILED (kept for a
        # later run), never an in-run RETRY and never a guessed delete.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(candidates=None)
        sonarr.queue_return = None

        assert strat.remove_from_queue(pending) is EffectStatus.FAILED
        assert sonarr.queue_calls == 1
        assert sonarr.queue_delete_calls == []

    def test_successful_close_notes_the_removal(self) -> None:
        recording = install_recording_hub()
        strat, _ = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )

        status = strat.remove_from_queue(pending_import(infohash="abc123", title="Show"))

        assert status is EffectStatus.DONE
        assert any("from Sonarr's queue" in m for m in diagnostic_messages(recording, Severity.INFO))

    def test_failed_delete_asks_for_a_retry_and_stays_quiet(self) -> None:
        # The client's fail-open warning already fired; no success note on top.
        recording = install_recording_hub()
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.FAILED

        status = strat.remove_from_queue(pending_import(infohash="abc123"))

        assert status is EffectStatus.RETRY
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording, Severity.INFO) == []

    def test_gone_queue_entry_is_quiet_success(self) -> None:
        # Sonarr already dismissed the entry (its refresh cleared it first): DONE
        # with zero diagnostics, not the coalesced could-not-remove warning.
        recording = install_recording_hub()
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.GONE

        status = strat.remove_from_queue(pending_import(infohash="abc123"))

        assert status is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording) == []


class TestScratchResetParity:
    """`reset()` reassigns the per-run scratch wholesale: post-reset state == freshly built state."""

    def test_executor_and_scanner_reset_to_the_init_state(self) -> None:
        strat, _ = _make_sonarr_for_import(candidates=None)
        executor = strat._executor
        scratch = executor._scratch
        scratch.quality_defs = []
        scratch.languages = []
        scratch.warned_unplaceable.add("h")
        scratch.warned_empty_folder.add("h")
        scratch.warned_default_quality = True
        scratch.last_rescan_monotonic = 1.0
        scratch.rescanned.add("h")
        scratch.kept_candidates["h"] = sonarr_import_module._KeptCandidates(
            "/d", sonarr_import_module._FetchedCandidates(candidates=[], history=None), 1.0
        )
        scratch.issued_command_ids.add(42)
        scratch.completed_download_handling = False
        scan = executor.candidate_fetcher._scratch
        scan.folder_pinned["h"] = True
        scan.history_states["h"] = DownloadHistoryState(history_closed=True)
        scan.path_mappings = []
        scan.translated_paths["/a"] = "/b"

        executor.reset()

        assert executor._scratch == sonarr_import_module._ImportScratch()
        assert executor.candidate_fetcher._scratch == sonarr_import_module._FetchScratch()


class TestInFlightManualImportGuard:
    """A ManualImport already running for this download must not be re-issued."""

    def test_in_flight_command_suppresses_reissue(self) -> None:
        # Queue empty (Sonarr dropped the torrent while importing server-side), but
        # a started ManualImport for this infohash is still running -> wait, don't
        # stack a duplicate.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        # Honest flags: a command for this record IS running (so an at-deadline
        # outcome reads "still importing", never "not ready") and its runtime is
        # credited back to the ready deadline.
        assert result.command_issued is True
        assert result.paused is True
        assert sonarr.execute_calls == []
        assert sonarr.candidate_calls == []

    def test_in_flight_guard_holds_even_when_forced(self) -> None:
        # Cross-run regression: the carried-over reconcile path always forces, and
        # that is the path that loops. force=True overrides Sonarr's clean-pending
        # deferral, NOT an already-running command -> still suppress the re-issue.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.command_issued is True
        assert result.paused is True
        assert sonarr.execute_calls == []

    def test_completed_command_does_not_suppress(self) -> None:
        # A finished ManualImport is not in flight, so it must never wedge us: with
        # the queue empty we step in and issue our import as before.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123", status="completed")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1

    def test_queued_import_for_other_download_defers(self) -> None:
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("OTHERHASH", status="queued")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.BUSY
        assert result.command_issued is False
        assert sonarr.execute_calls == []


class TestImportCompletedPayload:
    """import_completed assigns episodes from OUR map, never Sonarr's parse."""

    def test_payload_uses_pending_fields_not_candidate(self) -> None:
        pending = pending_import(
            series_id=7,
            release_group="SubGroup",
            infohash="HASH",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        # The candidate carries a *different* in-context quality and no
        # authoritative episode/series info. The payload must ignore those.
        candidate = manual_candidate(
            "/downloads/Show - 01 [1080p].mkv",
            quality={"quality": {"name": "HDTV-720p"}},
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        result = strat.import_completed(pending, "/downloads/Show", AttemptKind.ORDINARY)

        # The command was issued. The copy is async, so the probe carries
        # command_issued without all_done right after issuing.
        assert result.all_done is False
        assert result.command_issued is True
        assert sonarr.candidate_calls == [pending]
        files = sonarr.execute_calls[0][0]
        assert len(files) == 1
        entry = files[0]
        assert entry.seriesId == 7
        assert entry.episodeIds == [101]
        assert entry.releaseGroup == "SubGroup"
        assert entry.downloadId == "HASH"
        assert entry.path == "/downloads/Show - 01 [1080p].mkv"

    def test_sonarr_structured_quality_wins_over_ours(self) -> None:
        # Sonarr already parsed the release as (bluray, 1080). Our filename parse
        # would say WEB-DL, but Sonarr's structured parse takes precedence.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p][WEB-DL].mkv": [101]},
            seadex_files=["Show - 01 [1080p][WEB-DL].mkv"],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p][WEB-DL].mkv",
            quality={"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
        )
        quality_defs = [
            QualityDefinition.model_validate(
                {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
            ),
            QualityDefinition.model_validate(
                {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
            ),
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=quality_defs,
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        entry = sonarr.execute_calls[0][0][0]
        quality = entry.quality
        assert quality is not None
        inner = quality.quality
        assert inner is not None
        assert inner.name == "Bluray-1080p"
        revision = quality.revision
        assert revision is not None
        assert revision.version == 1

    def test_our_parse_fills_when_sonarr_quality_unknown(self) -> None:
        # Sonarr couldn't parse the release (Unknown). Our filename parse of
        # (web, 1080) fills both axes -> WEBDL-1080p, and a real quality is emitted.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p][WEB-DL].mkv": [101]},
            seadex_files=["Show - 01 [1080p][WEB-DL].mkv"],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p][WEB-DL].mkv",
            quality={"quality": {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}},
        )
        quality_defs = [
            QualityDefinition.model_validate(
                {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
            ),
            QualityDefinition.model_validate(
                {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
            ),
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=quality_defs,
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        entry = sonarr.execute_calls[0][0][0]
        quality = entry.quality
        assert quality is not None
        inner = quality.quality
        assert inner is not None
        assert inner.name == "WEBDL-1080p"

    def test_matches_disk_name_across_nfd_normalization(self) -> None:
        # The seed map is keyed by an NFC name. The on-disk leaf arrives NFD
        # (macOS). Normalization on both sides still matches -> the file imports,
        # never "no authoritative mapping".
        nfc = "Café - 01 [1080p].mkv"  # composed e-acute
        nfd = "Café - 01 [1080p].mkv"  # decomposed
        pending = pending_import(
            file_episode_map={nfc: [101]},
            seadex_files=[nfc],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[manual_candidate(f"/d/{nfd}")])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert result.command_issued is True
        assert sonarr.execute_calls[0][0][0].episodeIds == [101]

    def test_candidate_not_in_our_map_is_never_imported(self) -> None:
        # Strict-honor: a file Sonarr found that ISN'T in our map (e.g. an episode
        # our mapping gave to another preferred torrent) is never imported. Here our
        # one intended file is also present and IS imported.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidates = [
            manual_candidate("/d/Show - 01 [1080p].mkv"),
            manual_candidate("/d/Show - 02 [1080p].mkv"),  # not in our map
        ]
        strat, sonarr = _make_sonarr_for_import(candidates=candidates)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_sample_candidate_is_skipped(self) -> None:
        pending = pending_import(
            file_episode_map={
                "Show - 01 [1080p].mkv": [101],
                "Show - 01 [1080p].sample.mkv": [101],
            },
            seadex_files=["Show - 01 [1080p].mkv", "Show - 01 [1080p].sample.mkv"],
        )
        good = manual_candidate("/d/Show - 01 [1080p].mkv")
        sample = manual_candidate(
            "/d/Show - 01 [1080p].sample.mkv",
            rejections=[{"reason": "Sample"}],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[good, sample])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        # The good file's import command was issued.
        # The sample is never queued.
        assert result.all_done is False
        assert result.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_already_imported_rejection_does_not_skip_missing_group_file(self) -> None:
        # Bug fix (grab-then-skip): the episode already holds a MISSING-GROUP file
        # (UNKNOWN_GROUP -> still needing our recommended import), and Sonarr offers
        # the candidate WITH an "already imported" rejection (it fires whenever the
        # episode has any file on disk). That rejection must NOT veto our import:
        # we grabbed this exactly to replace the unidentifiable file, so we step in
        # and ISSUE the command (command_issued, the copy is async).
        pending = pending_import(
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group=None)],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert result.all_done is False
        assert len(sonarr.execute_calls) == 1

    def test_missing_group_import_then_recognized_terminates(self) -> None:
        # Loop-termination regression (mirrors the import->recognize round-trip):
        # poll 1 imports over a missing-group file (command_issued).
        # The imported file now carries OUR group, so poll 2 reads it as RECOMMENDED ->
        # nothing needed -> all_done, with NO re-issue. Proves the fix can't loop.
        pending = pending_import(
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group=None)],
        )

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert first.command_issued is True
        assert first.all_done is False

        # The import landed: episode 101 now holds our recommended group's file.
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]

        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert second.all_done is True
        assert len(sonarr.execute_calls) == 1

    def test_intended_file_missing_from_disk_retries(self) -> None:
        # Our map intends a file Sonarr can't see yet -> never dropped, retried
        # (no command issued, no files present).
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        # A different file is on disk. Ours isn't there yet.
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Unrelated.mkv")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.command_issued is False
        assert result.all_done is False
        assert sonarr.execute_calls == []

    def test_transient_candidate_scan_retries(self) -> None:
        # A None candidates result (timeout / non-200) is transient -> retry.
        strat, sonarr = _make_sonarr_for_import(candidates=None)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert sonarr.execute_calls == []

    def test_languages_follow_dual_audio_flag(self) -> None:
        pending = pending_import(
            is_dual_audio=True,
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        languages = [Language(id=1, name="English"), Language(id=8, name="Japanese")]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            languages=languages,
            config_overrides={"import_languages_dual": ["Japanese", "English"]},
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        names = [lang.name for lang in sonarr.execute_calls[0][0][0].languages]
        assert names == ["Japanese", "English"]

    def test_failed_execute_retries(self) -> None:
        # Sonarr rejected the import command (busy / locked) -> retry, not give up.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate], cmd_id=None)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert result.command_issued is False
        # The execute WAS attempted (Sonarr rejected it -> command_issued False).
        # A regression returning early without even trying would still set False here.
        assert len(sonarr.execute_calls) == 1

    def test_quality_defs_and_languages_cached_per_run(self) -> None:
        # Quality definitions + languages are fetched lazily ONCE and cached on the
        # executor for the rest of the run. Run 1 caches the (a) values. Before run 2
        # the source changes to (b), but a cached run must not re-fetch, so both polls
        # keep the (a) values. Drop the lazy-fetch guard and run 2 refetches (b),
        # flipping the executor's cache to (b) -> these assertions fail.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        defs_a = [
            QualityDefinition.model_validate({"quality": {"id": 1, "name": "A", "source": "web", "resolution": 1080}}),
        ]
        defs_b = [
            QualityDefinition.model_validate(
                {"quality": {"id": 2, "name": "B", "source": "bluray", "resolution": 1080}},
            ),
        ]
        langs_a = [Language(id=1, name="English")]
        langs_b = [Language(id=8, name="Japanese")]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=defs_a,
            languages=langs_a,
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        # The source changes between polls. A cached run must ignore it.
        sonarr.quality_defs_return = defs_b
        sonarr.languages_return = langs_b
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert strat._executor._scratch.quality_defs == defs_a
        assert strat._executor._scratch.languages == langs_a


class TestRecordSnapshot:
    """One poll's view over every series a record spans: an id routes to its index, an unknown id reads ABSENT."""

    @staticmethod
    def _snapshot() -> ImportSnapshot:
        # Series 7 holds 101 under the record's own group, series 8 a bare 201.
        return ImportSnapshot(
            _two_series_pending_import(),
            {
                7: episode_snapshot(
                    episodes=episode_index([sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]),
                    trusted=trusted_groups(OwnedAtGrab(), GrabbedGroups(TorrentGroup("SubGroup", ()))),
                ),
                8: episode_snapshot(episodes=episode_index([sonarr_ep(1, 1, ep_id=201, episode_file_id=0)])),
            },
            {},
        )

    def test_series_of_reads_the_index_holding_the_id(self) -> None:
        snapshot = self._snapshot()

        assert (snapshot.series_of(101), snapshot.series_of(201), snapshot.series_of(999)) == (7, 8, None)

    def test_statuses_classify_under_each_series_in_the_asked_order(self) -> None:
        # The grouping by series never reorders the answer, a repeat collapses, and an id no index
        # holds reads ABSENT rather than raising.
        statuses = self._snapshot().file_map_statuses([201, 999, 101, 201], {})

        assert list(statuses.by_id) == [201, 999, 101]
        assert statuses.by_id == {
            201: EpisodeFileStatus.ABSENT,
            999: EpisodeFileStatus.ABSENT,
            101: EpisodeFileStatus.TRUSTED,
        }


_SHOW_FILE = "Show - 01 [1080p].mkv"
_OTHER_FILE = "Other - 01 [1080p].mkv"


class _PerSeriesSonarr(FakeSonarrClient):
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


class _CountingStore(FakeCacheStore):
    """A `FakeCacheStore` counting its guard and pending-record reads."""

    def __init__(self) -> None:
        super().__init__()
        self.get_guards_calls = 0
        self.get_pending_import_row_calls = 0

    @override
    def get_guards(self, arr: Arr) -> dict[int, OwnedAtGrab]:
        self.get_guards_calls += 1
        return super().get_guards(arr)

    @override
    def get_pending_import_row(self, arr: Arr, infohash: str) -> dict[str, Any] | None:
        self.get_pending_import_row_calls += 1
        return super().get_pending_import_row(arr, infohash)


def _two_series_pending_import(
    files: FileEpisodeMap | None = None,
    *,
    other_window: tuple[int, ...] = (201,),
) -> PendingImport:
    """One torrent claimed on series 7 (window 101) and series 8 (`other_window`), its listing fully mapped."""

    mapped = {_SHOW_FILE: (101,), _OTHER_FILE: (201,)} if files is None else files
    return two_claim_pending_import(
        series_ids=(7, 8),
        windows=((101,), other_window),
        titles=("Show", "Other"),
        infohash="abc123",
        file_episode_map=mapped,
        seadex_files=list(mapped),
    )


def _posted_series(sonarr: FakeSonarrClient) -> dict[str, int]:
    """Each POSTed file's `seriesId` keyed by its path, across every execute."""

    return {entry.path: entry.seriesId for files, _ in sonarr.execute_calls for entry in files}


class TestMultiSeriesImport:
    """A two-series record posts each file under the series its claim names and verifies as one record."""

    @staticmethod
    def _strat(sonarr: FakeSonarrClient, store: FakeCacheStore | None = None) -> SonarrStrategy:
        return make_sonarr_sync(sonarr=sonarr, config=make_config(), cache_store=store or FakeCacheStore())

    @staticmethod
    def _candidates(*names: str) -> list[ManualImportCandidate]:
        return [manual_candidate(f"/d/{name}") for name in names]

    def test_each_file_posts_under_its_claims_series(self) -> None:
        # Both claims are scoped, so a file's first id names its claim whatever the indexes hold
        # (here nothing: every series reads an empty index).
        pending = _two_series_pending_import()
        strat, sonarr = _make_sonarr_for_import(candidates=self._candidates(_SHOW_FILE, _OTHER_FILE))

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": 7, f"/d/{_OTHER_FILE}": 8}

    def test_an_unscoped_claims_file_posts_under_the_index_holding_its_id(self) -> None:
        # The series-8 claim is unscoped, so no window holds 201: the same-poll index of series 8 does.
        pending = _two_series_pending_import(other_window=())
        sonarr = _PerSeriesSonarr(
            {7: [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)], 8: [sonarr_ep(1, 1, ep_id=201, episode_file_id=0)]},
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )

        result = self._strat(sonarr).import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": 7, f"/d/{_OTHER_FILE}": 8}

    def test_a_file_no_claim_or_index_routes_waits_and_warns_at_the_deadline(self) -> None:
        # 301 sits in neither claim's window and in no series index: nothing is POSTed, the poll
        # waits, and only the deadline attempt names the file.
        recording = install_recording_hub()
        stray = "Stray - 01 [1080p].mkv"
        pending = _two_series_pending_import({stray: (301,)})
        strat, sonarr = _make_sonarr_for_import(candidates=self._candidates(stray))

        polled = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert polled == replace(AttemptResult.waiting(), file_map_episode_count=1)
        assert diagnostic_messages(recording, Severity.WARNING) == []

        deadline = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert deadline == replace(AttemptResult.waiting(), file_map_episode_count=1)
        assert sonarr.execute_calls == []
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "resolve to no series" in warning
        assert normalize_name(stray) in warning

    def test_an_empty_index_still_posts_under_the_claims_series(self) -> None:
        # Series 8's index is empty: the claim's window still routes 201 there.
        pending = _two_series_pending_import()
        sonarr = _PerSeriesSonarr(
            {7: [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)], 8: []},
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )

        result = self._strat(sonarr).import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": 7, f"/d/{_OTHER_FILE}": 8}

    def test_a_one_series_record_posts_under_its_only_series_without_an_index(self) -> None:
        # An unscoped claim and an empty index leave no window and no index to route by: the record's
        # one series is where its files go.
        pending = pending_import()
        strat, sonarr = _make_sonarr_for_import(candidates=self._candidates(_SHOW_FILE))
        sonarr.episodes_return = []

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": pending.claims[0].series_id}

    def test_imported_only_when_every_series_targets_are_done(self) -> None:
        # Series 7's target already holds the record's group while series 8's is bare: the record is
        # not done, and only the series-8 file is POSTed. Once that lands, the next poll verifies.
        pending = _two_series_pending_import()
        sonarr = _PerSeriesSonarr(
            {
                7: [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")],
                8: [sonarr_ep(1, 1, ep_id=201, episode_file_id=0)],
            },
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )
        strat = self._strat(sonarr)

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert first.all_done is False
        assert first.command_issued is True
        assert (first.imported_count, first.file_map_episode_count) == (1, 2)
        assert _posted_series(sonarr) == {f"/d/{_OTHER_FILE}": 8}

        sonarr.episodes_by_series[8] = [sonarr_ep(1, 1, ep_id=201, release_group="SubGroup")]
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert second.all_done is True
        assert (second.imported_count, second.file_map_episode_count) == (2, 2)
        assert len(sonarr.execute_calls) == 1

    def test_history_rows_land_under_the_admitting_claim(self) -> None:
        # A row counts only where a claim on its series admits the episode: the scoped series-7 claim
        # admits 101 alone, the unscoped series-8 claim admits every row on series 8, and a row on a
        # series no claim holds is dropped.
        second, misfiled, stray = "Other - 02 [1080p].mkv", "Show - 02 [1080p].mkv", "Stray - 01 [1080p].mkv"
        pending = pending_import(
            infohash="abc123",
            file_episode_map={},
            seadex_files=[_SHOW_FILE, _OTHER_FILE, second, misfiled, stray],
            claims=(
                title_claim(al_id=1, series_id=7, title="Show", ordered_episode_ids=(101,)),
                title_claim(al_id=2, series_id=8, title="Other"),
            ),
        )
        rows = [
            HistoryImport(f"/d/{_SHOW_FILE}", episode_id=101, series_id=7),
            HistoryImport(f"/d/{_OTHER_FILE}", episode_id=201, series_id=8),
            HistoryImport(f"/d/{second}", episode_id=202, series_id=8),
            HistoryImport(f"/d/{misfiled}", episode_id=102, series_id=7),
            HistoryImport(f"/d/{stray}", episode_id=901, series_id=9),
        ]

        assert placements_from_history(rows, pending) == {
            normalize_name(_SHOW_FILE): (101,),
            normalize_name(_OTHER_FILE): (201,),
            normalize_name(second): (202,),
        }

    def test_seed_statuses_read_the_guards_once_for_every_series(self) -> None:
        # One guards read serves every claimed series' trust policy, while each series' episodes are
        # fetched fresh on their own.
        store = _CountingStore()
        sonarr = _PerSeriesSonarr({7: [], 8: []}, candidates=[])
        strat = self._strat(sonarr, store)

        progress = strat.import_progress(_two_series_pending_import())

        assert store.get_guards_calls == 1
        assert sonarr.episodes_calls == [7, 8]
        assert progress == ImportProgress(0, 2, determinate=True)

    def test_each_claimed_id_is_judged_under_its_own_claims_guards(self) -> None:
        # One series, two claims on it: A's plan judged group G stale, B's picks carry G. G files sit on
        # both windows' episodes, so A's id still needs importing while B's is done (one of two), where the
        # series' merged evidence would have trusted G for both. One episode fetch serves both claims.
        pending = pending_import(
            file_episode_map={_SHOW_FILE: [101], _OTHER_FILE: [102]},
            claims=(
                title_claim(al_id=1, ordered_episode_ids=(101,), guards=OwnedAtGrab(stale_groups=("G",))),
                title_claim(al_id=2, ordered_episode_ids=(102,), guards=OwnedAtGrab(entry_groups=("G",))),
            ),
        )
        sonarr = _PerSeriesSonarr(
            {7: [sonarr_ep(1, 1, ep_id=101, release_group="G"), sonarr_ep(1, 2, ep_id=102, release_group="G")]},
            candidates=[],
        )

        progress = self._strat(sonarr).import_progress(pending)

        assert progress == ImportProgress(1, 2, determinate=True)
        assert sonarr.episodes_calls == [7]

    def test_a_misplaced_episode_still_counts_as_owed(self) -> None:
        # Our 200 file sits on both episodes: 102 holds its own file, 101 still lacks the 100 one.
        pending = pending_import(
            file_episode_map={_SHOW_FILE: [101], _OTHER_FILE: [102]},
            release_sizes=[100, 200],
            sizes_by_name={normalize_name(_SHOW_FILE): 100, normalize_name(_OTHER_FILE): 200},
        )
        sonarr = _PerSeriesSonarr(
            {
                7: [
                    sonarr_ep(1, 1, ep_id=101, size=200, release_group="SubGroup"),
                    sonarr_ep(1, 2, ep_id=102, size=200, release_group="SubGroup"),
                ]
            },
            candidates=[],
        )

        progress = self._strat(sonarr).import_progress(pending)

        assert progress == ImportProgress(1, 2, determinate=True)

    def test_an_id_outside_every_window_is_judged_under_the_lone_unscoped_claim(self) -> None:
        # One series: A's window holds 101 and its picks carry G, B is unscoped and its plan judged G stale.
        # 102 sits outside every window, so B alone judges it and it still needs importing (one of two
        # done), where the series' merged evidence would have trusted G for both.
        pending = pending_import(
            file_episode_map={_SHOW_FILE: [101], _OTHER_FILE: [102]},
            claims=(
                title_claim(al_id=1, ordered_episode_ids=(101,), guards=OwnedAtGrab(entry_groups=("G",))),
                title_claim(al_id=2, guards=OwnedAtGrab(stale_groups=("G",))),
            ),
        )
        sonarr = _PerSeriesSonarr(
            {7: [sonarr_ep(1, 1, ep_id=101, release_group="G"), sonarr_ep(1, 2, ep_id=102, release_group="G")]},
            candidates=[],
        )

        progress = self._strat(sonarr).import_progress(pending)

        assert progress == ImportProgress(1, 2, determinate=True)
        assert sonarr.episodes_calls == [7]

    def test_an_id_two_windows_hold_is_judged_and_netted_under_the_first_claim(self) -> None:
        # A's window (101-104) and B's (103-104) overlap on G files. 103 and 104 are judged under A, whose
        # plan judged G stale, so they still need importing, and B's preowned 103 nets nothing out of the
        # bar: every target the statuses judged is counted, where a union of preowned ids would drop one.
        files = {f"show - {n:02d}.mkv": [100 + n] for n in range(1, 5)}
        pending = pending_import(
            file_episode_map=files,
            seadex_files=list(files),
            claims=(
                title_claim(al_id=1, ordered_episode_ids=(101, 102, 103, 104), guards=OwnedAtGrab(stale_groups=("G",))),
                title_claim(
                    al_id=2,
                    ordered_episode_ids=(103, 104),
                    preowned_episode_ids=(103,),
                    guards=OwnedAtGrab(entry_groups=("G",)),
                ),
            ),
        )
        episodes = [sonarr_ep(1, n, ep_id=100 + n, release_group="G") for n in range(1, 5)]
        sonarr = _PerSeriesSonarr({7: episodes}, candidates=[])

        assert self._strat(sonarr).import_progress(pending) == ImportProgress(0, 4, determinate=True)


def _import_history(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` record, as the import-evidence match reads it."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


class TestRadarrImportCompletedHistory:
    """Radarr reconciles a completed download off its OWN import history (evidence, never action).

    Radarr's completed-download handling does the import; `import_completed` only
    reads history as proof so the record can drop and the category gate release.
    """

    def test_matching_import_event_verifies_files(self) -> None:
        strat = make_radarr_sync(radarr=FakeRadarrClient(history_since=[_import_history("h")]))

        result = strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is True

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([], id="no-event-yet"),
            pytest.param([_import_history("h", event="grabbed")], id="non-import-event"),
            pytest.param(None, id="history-outage"),
        ],
    )
    def test_missing_evidence_keeps_the_record_retrying(self, history: list[HistoryRecord] | None) -> None:
        # No import proof (none yet, a non-import event, or Radarr down): fail
        # closed: waiting keeps the record pending, never verified on a missing read.
        radarr = FakeRadarrClient()
        radarr.history_since_return = history
        strat = make_radarr_sync(radarr=radarr)

        result = strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is False

    def test_matches_download_id_case_insensitively(self) -> None:
        # Radarr uppercases the stored downloadId; our infohash is lowercase.
        strat = make_radarr_sync(radarr=FakeRadarrClient(history_since=[_import_history("ABCDEF")]))

        result = strat.import_completed(pending_import(infohash="abcdef"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is True

    def test_history_fetched_once_per_pass(self) -> None:
        # Two records in one reconcile pass share ONE history fetch (memoized).
        radarr = FakeRadarrClient(history_since=[_import_history("h")])
        strat = make_radarr_sync(radarr=radarr)

        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="other"), "/d", AttemptKind.ORDINARY)

        assert len(radarr.history_calls) == 1

    def test_run_start_resets_the_memo(self) -> None:
        # get_items (the run-start hook) clears the memo so a stale window can't
        # carry into the next run's reconcile.
        radarr = FakeRadarrClient(history_since=[_import_history("h")])
        strat = make_radarr_sync(radarr=radarr)

        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)
        strat.get_items()
        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert len(radarr.history_calls) == 2

    @pytest.mark.parametrize("age_days", [None, HISTORY_MAX_LOOKBACK_DAYS + 10], ids=["no-records", "past-the-cap"])
    def test_history_window_starts_at_the_cap_when_no_stamp_is_newer(self, age_days: int | None) -> None:
        store = FakeCacheStore()
        if age_days is not None:
            stamp = stamp_of(datetime.now() - timedelta(days=age_days))
            store.put_pending_import_row(Arr.RADARR, "h", pending_import(infohash="h", added_at=stamp).to_json())
        strat = make_radarr_sync(cache_store=store)
        back = timedelta(days=HISTORY_MAX_LOOKBACK_DAYS, hours=_HISTORY_SKEW_HOURS)

        before = datetime.now(UTC)
        start = strat._history_query_start()
        after = datetime.now(UTC)

        assert before - back <= start <= after - back

    def test_history_window_starts_at_the_oldest_stamp_inside_the_cap(self) -> None:
        older = stamp_of(datetime.now() - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS - 10))
        newer = stamp_of(datetime.now() - timedelta(days=1))
        store = FakeCacheStore()
        store.put_pending_import_row(Arr.RADARR, "new", pending_import(infohash="new", added_at=newer).to_json())
        store.put_pending_import_row(Arr.RADARR, "old", pending_import(infohash="old", added_at=older).to_json())
        strat = make_radarr_sync(cache_store=store)

        start = strat._history_query_start()

        assert start == parse_stamp(older).astimezone(UTC) - timedelta(hours=_HISTORY_SKEW_HOURS)

    def test_pending_import_series_id_is_none(self) -> None:
        # Radarr records carry series_id=0, so the per-item snapshot hook key is
        # None (short-circuits the per-item snapshot - reconcile runs in the tail).
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())

        assert strat.pending_import_series_id(_Item(id=5)) is None

    def test_import_progress_is_indeterminate_zero(self) -> None:
        # Radarr never enters the blocking monitor, so the fast-lane bar hook
        # returns the safe "no bar, promote nothing" value.
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())

        assert strat.import_progress(pending_import()) == ImportProgress(0, 0, determinate=False)

    def test_does_not_support_the_blocking_monitor(self) -> None:
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())

        assert strat.supports_blocking_wait is False


_GRAB_URL = "https://nyaa.si/1"
_GRAB_FILES = ("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv")
_EP1 = sonarr_ep(1, 1, ep_id=101, episode_file_id=0)
_EP2 = sonarr_ep(1, 2, ep_id=102, episode_file_id=0)
_STAMP = "2026-01-01 00:00:00"


def _grab_episodes(
    sonarr: FakeSonarrClient,
    *,
    unread_series: frozenset[int] = frozenset(),
    unread_al_ids: frozenset[int] = frozenset(),
) -> ScriptedEpisodes:
    """Entry 5's window is episode 102 of `sonarr`'s series.

    The `unread_series` have no order record, and the `unread_al_ids` no AniList record.
    """

    return ScriptedEpisodes(sonarr, windows={5: [_EP2]}, unread_series=unread_series, unread_al_ids=unread_al_ids)


def _sonarr_grab_run(mode: ImportWaitMode, *, store: FakeCacheStore, episodes: ScriptedEpisodes) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for entry 5 (series 7) listing one two-file torrent.

    The series maps to entries 4 and 5 and to the ignored entry 6.
    """

    seadex: SeadexDict = {
        "NAN0": rg_group(
            {_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=list(_GRAB_FILES))}
        ),
    }
    run = _FakeRunServices(
        anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
        ignored={6: TitleMapping(anilist_id=6)},
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show Cour 2",
        seadex_dict=seadex,
        filter_downloads_result=grab_plan(seadex),
        import_wait_mode=mode,
        cache_store=store,
    )
    parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(_GRAB_FILES, start=1)}
    strat = make_bare_instance(
        SonarrStrategy,
        _services=run,
        _episodes=episodes,
        _parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in _GRAB_FILES)}),
        _listings=_ScriptedListings(),
        _config=make_config(sleep_time=0),
        logger=make_logger(),
    )
    strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))
    return run


def _run_title_check(
    files: tuple[ListedFile, ...],
    window: list[SonarrEpisode],
    listings: _ScriptedListings,
    *,
    episode_count: int | None = None,
) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for entry 5 over `window` (all of series 7), listing one torrent of `files`.

    `episode_count` is AniList's count for entry 5, None when it gives none.
    """

    seadex: SeadexDict = {
        "NAN0": rg_group(
            {_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=[f.file_name for f in files])}
        ),
    }
    run = _FakeRunServices(
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show",
        seadex_dict=seadex,
        filter_downloads_result=grab_plan(seadex),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=FakeCacheStore(),
    )
    strat = make_bare_instance(
        SonarrStrategy,
        _services=run,
        _episodes=ScriptedEpisodes(
            FakeSonarrClient(episodes=window),
            windows={5: window},
            counts={} if episode_count is None else {5: episode_count},
        ),
        _parse=_ScriptedParse({_GRAB_URL: files}),
        _listings=listings,
        _config=make_config(sleep_time=0),
        logger=make_logger(),
    )
    strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))
    return run


class TestSonarrProcessAlIdSeeds:
    """A Sonarr grab places each listed torrent under the entry's scope, a stored one as its record's leftover."""

    def test_a_stored_torrent_is_placed_as_its_records_leftover(self) -> None:
        # Cour 1's record already maps file 01. Cour 2 flags the same torrent: only file 02 is placed,
        # under the stored claim's window and this entry's, and the seed accretes onto the record.
        store = FakeCacheStore()
        resident = pending_import(
            infohash="h1",
            al_id=22,
            series_id=7,
            title="Show Cour 1",
            ordered_episode_ids=(101,),
            file_episode_map={_GRAB_FILES[0]: [101]},
            seadex_files=list(_GRAB_FILES),
        )
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        sonarr = FakeSonarrClient(episodes=[_EP1, _EP2])

        run = _sonarr_grab_run(ImportWaitMode.BLOCKING, store=store, episodes=_grab_episodes(sonarr))

        (req,) = run.grab_requests
        (update,) = req.pending_import_updates.values()
        assert update.pending_import == resident
        assert update.file_map == {normalize_name(_GRAB_FILES[1]): (102,)}
        assert update.claim.al_id == 5
        assert update.claim.ordered_episode_ids == (102,)
        pending = update.pending_import_at(_STAMP, fresh=False)
        assert claim_al_ids(pending) == (22, 5)
        assert dict(pending.file_episode_map) == {
            normalize_name(name): (ep,) for name, ep in zip(_GRAB_FILES, (101, 102), strict=True)
        }
        # The planner judges the url by the whole map's episodes inside this entry.
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == [ReleaseEpisode(season=1, episode=2, size=1000)]
        assert req.input_unread_groups == ()
        # The stored claim's window came off the per-run whole-series read.
        assert sonarr.episodes_calls == [7]

    def test_a_specials_pack_the_listing_contradicts_is_held(self) -> None:
        # The series' entries list the pack over specials 2, 4 and 6, three as it is wide, and its `1` is none of
        # them: the pack counts by another TVDB state. Nothing is placed for the planner, and the url is held.
        specials = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)]
        files = tuple(f"Show.S00E{n:02d}.mkv" for n in range(1, 4))
        seadex: SeadexDict = {
            "NAN0": rg_group({_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=list(files))}),
        }
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            anilist_title="Show Specials",
            seadex_dict=seadex,
            filter_downloads_result=grab_plan(seadex),
            import_wait_mode=ImportWaitMode.BLOCKING,
            cache_store=FakeCacheStore(),
        )
        parses = {name: parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n, name in enumerate(files, 1)}
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=specials), windows={5: [specials[1], specials[3]]}),
            _parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in files)}),
            _listings=_ScriptedListings({"h1": torrent_listing(title_window(1, (502, 504, 506)))}),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.MISNUMBERED
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == []
        assert req.pending_import_updates["h1"].file_map == {}
        assert req.input_unread_groups == ()

    def test_a_file_placed_by_size_over_its_name_warns_once_per_url(self) -> None:
        # Two specials named as each other.
        specials = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in (1, 2)]
        files = tuple(
            ListedFile(f"Show.S00E0{n}.mkv", 1000 * n, parsed_info(season=0, episodes=(n,), matched=((0, n),)))
            for n in (1, 2)
        )
        recording = install_recording_hub()

        run = _run_title_check(files, specials, _ScriptedListings(episode_by_size={1000: 502, 2000: 501}))

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert update.file_map == {"show.s00e01.mkv": (502,), "show.s00e02.mkv": (501,)}
        assert diagnostic_messages(recording, Severity.WARNING) == [
            (
                f"{_GRAB_URL}: show.s00e01.mkv goes on S00E02 because its size matches SeaDex's S00E02 release "
                "(its name reads as S00E01)"
                " (1 more file in the debug log)"
            )
        ]

    @pytest.mark.parametrize(
        ("episode_count", "placed"),
        [
            pytest.param(2, {"ova - 01 [grp].mkv": (501,), "ova - 02 [grp].mkv": (504,)}, id="the-windows-two"),
            pytest.param(3, {}, id="an-episode-sonarr-lacks"),
        ],
    )
    def test_the_entrys_anilist_episode_count_reaches_its_placement(
        self, episode_count: int, placed: dict[str, tuple[int, ...]]
    ) -> None:
        # Two specials numbered apart take the run only when they are the title's every episode.
        specials = [
            sonarr_ep(0, 1, ep_id=501, episode_file_id=0, air_date="2016-03-01"),
            sonarr_ep(0, 4, ep_id=504, episode_file_id=0, air_date="2016-06-01"),
        ]
        names = [*numbered_names("ova", 2), *numbered_names("short", 3)]
        files = tuple(ListedFile(name, 1000, parsed_info()) for name in names)

        run = _run_title_check(files, specials, _ScriptedListings(), episode_count=episode_count)

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == placed

    def test_an_unread_listing_places_nothing_and_holds_the_url(self) -> None:
        # SeaDex is down, so the windows the torrent runs under and the size identities are both unknown. Nothing
        # is placed, the url is held, and the title isn't cached as done since its inputs weren't all read.
        files = tuple(
            ListedFile(name, 1000, parsed_info(season=1, episodes=(n,))) for n, name in enumerate(_GRAB_FILES, 1)
        )

        run = _run_title_check(files, [_EP1, _EP2], _ScriptedListings(unreachable=True))

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.INPUT_UNREAD
        assert req.pending_import_updates["h1"].file_map == {}
        assert req.input_unread_groups == ("NAN0",)

    def test_the_listing_read_gets_the_ignored_entries_too(self) -> None:
        # Entry 6 shares the series but is on the ignore list. It's never processed, but a torrent it lists is
        # still its files, so the series' listing read gets it beside the entries the run processes.
        kept, ignored = {5: TitleMapping(anilist_id=5)}, {6: TitleMapping(anilist_id=6)}
        seadex: SeadexDict = {
            "NAN0": rg_group({_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=["a.mkv"])}),
        }
        run = _FakeRunServices(
            anilist_ids=kept,
            ignored=ignored,
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            seadex_dict=seadex,
            filter_downloads_result=grab_plan(seadex),
            import_wait_mode=ImportWaitMode.BLOCKING,
        )
        listings = _ScriptedListings()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1]), windows={5: [_EP1]}),
            _parse=_ScriptedParse(),
            _listings=listings,
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, kept[5])

        assert listings.reads == [(7, ResolvedMappings(kept=kept, ignored=ignored))]

    def test_an_ignored_entrys_listed_files_are_never_grabbed_onto_a_sibling(self) -> None:
        # Entry 5, the season, is on the ignore list, and entry 6 is the specials. The torrent is the season's
        # bare-numbered run, which entry 6's window alone would take as its specials. The season's listed window
        # decides the files first, so entry 6 grabs none of them and the url is held as another title's.
        specials = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in (1, 2)]
        files = tuple(
            ListedFile(name, 1000 * n, parsed_info(season=0, absolutes=(n,), matched=((1, n),)))
            for n, name in enumerate(_GRAB_FILES, 1)
        )
        seadex: SeadexDict = {
            "NAN0": rg_group(
                {_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=list(_GRAB_FILES))}
            ),
        }
        run = _FakeRunServices(
            anilist_ids={6: TitleMapping(anilist_id=6)},
            ignored={5: TitleMapping(anilist_id=5)},
            prologue_entry=make_entry_record(url="https://releases.moe/6"),
            seadex_dict=seadex,
            filter_downloads_result=grab_plan(seadex),
            import_wait_mode=ImportWaitMode.BLOCKING,
            cache_store=FakeCacheStore(),
        )
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1, _EP2, *specials]), windows={6: specials}),
            _parse=_ScriptedParse({_GRAB_URL: files}),
            _listings=_ScriptedListings(
                {"h1": torrent_listing(title_window(5, (101, 102), ignored=True), title_window(6, (501, 502)))}
            ),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 6, TitleMapping(anilist_id=6))

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert update.file_map == {}
        assert update.excluded == tuple(normalize_name(name) for name in _GRAB_FILES)
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.PLACED_ELSEWHERE

    def test_wait_mode_off_reads_no_record_and_seeds_nothing(self) -> None:
        store = _CountingStore()
        sonarr = FakeSonarrClient(episodes=[_EP1, _EP2])

        run = _sonarr_grab_run(ImportWaitMode.OFF, store=store, episodes=_grab_episodes(sonarr))

        (req,) = run.grab_requests
        assert req.pending_import_updates == NO_PENDING_IMPORT_UPDATES
        assert store.get_pending_import_row_calls == 0
        # The placement still runs: the planner's coverage never depends on the wait mode.
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == [ReleaseEpisode(season=1, episode=2, size=1000)]


class TestPlacementDebugLine:
    """The grab's debug line per url says what placed and why the url waited."""

    def test_a_url_held_on_a_failed_read_names_the_read_once(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = debug_logger("pearlarr-placement-line")
        strat = make_bare_instance(SonarrStrategy, logger=logger)
        grabbing_title = make_grabbing_title(episode_index([_EP1]), episode_index([_EP1]))
        placement = place_release(
            [ListedFile("Show - 01.mkv", 1000, None)], grabbing_title, new_torrent(grabbing_title)
        )

        with caplog.at_level("DEBUG"):
            strat._log_placements(EntryPlacements(grabbing_title, {_GRAB_URL: placement}))

        assert placement.skip_reason is SkipReason.INPUT_UNREAD
        assert [r.message for r in caplog.records] == [f"{_GRAB_URL}: placed nothing (not grabbed: a read failed)"]

    def test_a_file_left_for_a_hand_import_is_named_with_why(self, caplog: pytest.LogCaptureFixture) -> None:
        # A Blu-ray pack numbers 13 files for a 12-episode season: the DVD order splits the opener in two halves
        # Sonarr doesn't list, numbers the aired second episode on after them, and keeps the whole opener last.
        season = [sonarr_ep(4, n, ep_id=400 + n, episode_file_id=0, tvdb_id=400 + n) for n in range(1, 13)]
        dvd = (OrderRow(7, 1, 9001, None), OrderRow(7, 2, 9002, None))
        dvd += (*(OrderRow(7, n + 1, 400 + n, None) for n in range(2, 13)), OrderRow(7, 21, 401, None))
        series = episode_index(season, IndexOrders({"dvd": dvd}, None))
        names = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
        grabbing_title = make_grabbing_title(series, series, names=TitleNames("Show", ("Show Ending",)))
        logger = debug_logger("pearlarr-placement-line-orders")
        strat = make_bare_instance(SonarrStrategy, logger=logger)
        placement = place_release(
            [ListedFile(name, 1000, parsed_info()) for name in names], grabbing_title, new_torrent(grabbing_title)
        )

        with caplog.at_level("DEBUG"):
            strat._log_placements(EntryPlacements(grabbing_title, {_GRAB_URL: placement}))

        halves = ", ".join(
            f"{normalize_name(names[n - 1])} (another TVDB order, not in Sonarr's list: DVD S07E0{n})" for n in (1, 2)
        )
        assert [r.message for r in caplog.records] == [
            f"{_GRAB_URL}: placed S04 E02-E12; left for a hand import: {halves}"
        ]


class TestSonarrProcessAlIdMetadataUnread:
    """A grab asks whether its check read an entry without its AniList record or a series without its TVDB orders.

    The entries are every entry of the title's series, the series are the title's and every stored record's.
    """

    @pytest.mark.parametrize(
        ("unread_series", "unread_al_ids", "expected"),
        [
            pytest.param(frozenset[int](), frozenset[int](), False, id="all-read"),
            pytest.param(frozenset({7}), frozenset[int](), True, id="its-own-series"),
            pytest.param(frozenset({8}), frozenset[int](), True, id="a-stored-records-series"),
            pytest.param(frozenset[int](), frozenset({5}), True, id="its-own-entry"),
            pytest.param(frozenset[int](), frozenset({4}), True, id="another-entry-of-its-series"),
            pytest.param(frozenset[int](), frozenset({6}), True, id="an-ignored-entry-of-its-series"),
        ],
    )
    def test_the_grab_request_carries_whether_its_metadata_is_unread(
        self, unread_series: frozenset[int], unread_al_ids: frozenset[int], expected: bool
    ) -> None:
        # Series 8's record already holds the torrent entry 5 lists, so the grab places it under that record too.
        store = FakeCacheStore()
        resident = pending_import(infohash="h1", al_id=22, series_id=8, seadex_files=list(_GRAB_FILES))
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        episodes = _grab_episodes(
            FakeSonarrClient(episodes=[_EP1, _EP2]), unread_series=unread_series, unread_al_ids=unread_al_ids
        )

        run = _sonarr_grab_run(ImportWaitMode.BLOCKING, store=store, episodes=episodes)

        (req,) = run.grab_requests
        assert req.metadata_unread is expected

    @pytest.mark.parametrize(
        ("unread_al_ids", "unread"),
        [
            pytest.param(frozenset({5}), [5], id="its-own-entry"),
            pytest.param(frozenset({4}), [], id="another-entry-of-its-series"),
        ],
    )
    def test_a_title_with_no_suitable_release_asks_about_its_own_entry_alone(
        self, unread_al_ids: frozenset[int], unread: list[int]
    ) -> None:
        run = _FakeRunServices(
            anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            anilist_title="Show Cour 2",
            seadex_dict={},
            cache_store=FakeCacheStore(),
        )
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            _episodes=_grab_episodes(FakeSonarrClient(episodes=[_EP1, _EP2]), unread_al_ids=unread_al_ids),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))

        assert [al_id for al_id, _ in run.no_releases_calls] == [5]
        assert run.no_releases_unread == unread


class TestRadarrProcessAlIdSeeds:
    """A Radarr grab seeds a `PendingImportUpdate` per download+hash, keyed by its hash in qBittorrent, gated on wait mode."""

    @staticmethod
    def _run_process(
        mode: ImportWaitMode, *, store: FakeCacheStore | None = None, cross_seed: str | None = None
    ) -> _FakeRunServices:
        """Drive `check_title` to grab_and_cache and return the recording services hub.

        `cross_seed` marks the url as held by that torrent.
        """

        item = make_release(url="https://nyaa.si/1", infohash="h1", download=True)
        item.cross_seed_hash = cross_seed
        seadex: SeadexDict = {"NAN0": rg_group({"https://nyaa.si/1": item})}
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/9"),
            anilist_title="A Movie",
            seadex_dict=seadex,
            filter_downloads_result=grab_plan(seadex),
            import_wait_mode=mode,
            arr=Arr.RADARR,
            cache_store=store,
        )
        strat = make_bare_instance(
            RadarrStrategy,
            _services=run,
            radarr=FakeRadarrClient(),
            logger=make_logger(),
            _config=make_config(),
        )
        strat.check_title(_Item(id=5, title="Movie"), 42, TitleMapping(anilist_id=42))
        return run

    def test_seeds_the_record_with_empty_sonarr_fields(self) -> None:
        run = self._run_process(ImportWaitMode.BLOCKING)

        (req,) = run.grab_requests
        (update,) = req.pending_import_updates.values()
        # Carried fields: the real torrent identity + the one claim's anilist display context.
        assert update.facts.infohash == "h1"
        assert update.facts.release_group == "NAN0"
        assert update.claim.al_id == 42
        assert update.claim.title == "A Movie"
        assert update.claim.url == "https://releases.moe/9"
        # A torrent the store does not hold starts a new record.
        assert update.pending_import is None
        # Sonarr-domain fields deliberately stay empty for a Radarr record.
        assert update.claim.series_id == 0
        assert update.file_map == {}
        assert update.facts.seadex_files == ()
        assert update.claim.ordered_episode_ids == ()
        assert update.claim.window_coverage is None
        pending = update.pending_import_at(_STAMP, fresh=True)
        # display_label renders sensibly from these (title · group).
        assert pending.display_label == f"A Movie{SEP}NAN0"
        assert pending.file_map_episode_ids() == []
        assert pending.added_at == _STAMP

    def test_seed_carries_the_stored_record(self) -> None:
        # A torrent another Radarr entry already holds: the seed accretes this entry's claim onto it.
        store = FakeCacheStore()
        resident = pending_import(infohash="h1", al_id=9, series_id=0, title="Other Movie", file_episode_map={})
        store.put_pending_import_row(Arr.RADARR, "h1", resident.to_json())

        run = self._run_process(ImportWaitMode.BLOCKING, store=store)

        (req,) = run.grab_requests
        (update,) = req.pending_import_updates.values()
        assert update.pending_import == resident
        assert claim_al_ids(update.pending_import_at(_STAMP, fresh=False)) == (9, 42)

    def test_a_marked_url_seeds_onto_its_cross_seeds_record(self) -> None:
        store = FakeCacheStore()
        resident = pending_import(infohash="c1", al_id=9, series_id=0, title="Other Movie", file_episode_map={})
        store.put_pending_import_row(Arr.RADARR, "c1", resident.to_json())

        run = self._run_process(ImportWaitMode.BLOCKING, store=store, cross_seed="c1")

        (req,) = run.grab_requests
        update = req.pending_import_updates["c1"]
        assert update.pending_import == resident
        assert (update.facts.infohash, update.facts.is_cross_seed) == ("c1", True)

    def test_no_seeds_when_wait_mode_off(self) -> None:
        store = _CountingStore()

        run = self._run_process(ImportWaitMode.OFF, store=store)

        (req,) = run.grab_requests
        assert req.pending_import_updates == NO_PENDING_IMPORT_UPDATES
        assert store.get_pending_import_row_calls == 0


class TestManualImportWarningGating:
    """The import warns loudly only at the deadline. Otherwise it's debug.

    A missing intended file on an early poll is expected (the copy hasn't landed),
    so it must NOT inflate the summary's warning count. Only the final attempt
    (at_deadline) warns loudly that a still-missing file is terminal.
    """

    @staticmethod
    def _strat_with_missing_file() -> tuple[SonarrStrategy, PendingImport]:
        # Our map intends a file that isn't on disk yet (only an unrelated file is),
        # so run_auto_import always finds it missing and retries.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, _ = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Unrelated.mkv")],
        )
        # caplog captures via the root logger's propagation, so use a propagating
        # DEBUG logger here (make_logger disables propagation for quiet runs). The
        # missing-file line is logged by the import executor, which holds the same
        # logger as the strat in production - so set both (the strat's plus the
        # executor's) to mirror that and let caplog see the executor's record.
        logger = debug_logger("pearlarr-warning-gating")
        strat.logger = logger
        strat._executor.logger = logger
        return strat, pending

    def test_missing_off_deadline_is_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, pending = self._strat_with_missing_file()
        recording = install_recording_hub()

        with caplog.at_level("DEBUG"):
            result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert not any(
            "not visible to Sonarr" in message for message in diagnostic_messages(recording, Severity.WARNING)
        )
        assert any("not visible to Sonarr" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    def test_missing_at_deadline_warns_loudly(self) -> None:
        strat, pending = self._strat_with_missing_file()
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert result.all_done is False
        assert any("not visible to Sonarr" in message for message in diagnostic_messages(recording, Severity.WARNING))

    @staticmethod
    def _strat_with_unplaceable(*names: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # The mapped file plus `names`, which the map lacks, beside the bare target episode.
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), *(manual_candidate(f"/d/{n}") for n in names)],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
        )
        logger = debug_logger("pearlarr-warning-gating")
        strat._executor.logger = logger
        return strat, sonarr

    def test_grab_time_excluded_unplaceable_file_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # A sibling record's slice of a shared torrent was decided at grab time:
        # this record still imports its own file, and the unplaceable sibling
        # file is a debug note, not a "could not be matched" warning.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling], excluded_files=[normalize_name(sibling)]
        )
        strat, sonarr = self._strat_with_unplaceable(sibling)
        recording = install_recording_hub()

        with caplog.at_level("DEBUG"):
            result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert not any("could not be matched" in m for m in diagnostic_messages(recording, Severity.WARNING))
        assert any("1 file excluded on record" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    def test_partial_exclusion_warns_the_remainder_once(self) -> None:
        # One excluded plus one genuinely unplaceable file: the warning names
        # only the remainder, and the once-per-run memo is consumed by it.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling, "Extras.mkv"],
            excluded_files=[normalize_name(sibling)],
        )
        strat, _ = self._strat_with_unplaceable(sibling, "Extras.mkv")
        recording = install_recording_hub()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be matched" in m]
        assert len(warnings) == 1
        assert "1 file could not be matched" in warnings[0]

    def test_a_disk_file_seadexs_list_lacks_warns_as_a_skip(self) -> None:
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv"])
        strat, sonarr = self._strat_with_unplaceable("Show - S02E01 [1080p].mkv")
        recording = install_recording_hub()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert [f.path for f in sonarr.execute_calls[0][0]] == ["/d/Show - 01 [1080p].mkv"]
        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be matched" in m]
        assert len(warnings) == 1

    def test_a_duplicate_leaf_the_map_lacks_is_reported_once(self) -> None:
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv", "a/Extras.mkv", "b/Extras.mkv"])
        strat, sonarr = self._strat_with_unplaceable("a/Extras.mkv", "b/Extras.mkv")
        recording = install_recording_hub()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert [f.path for f in sonarr.execute_calls[0][0]] == ["/d/Show - 01 [1080p].mkv"]
        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be matched" in m]
        assert len(warnings) == 1
        assert "1 file could not be matched" in warnings[0]


class TestUnmatchedProbe:
    """A download whose every on-disk file the map lacks is an outcome on its first poll, not a wait.

    The import places nothing, so the next poll would find the same files: no parse or episode read is waited on.
    """

    _NAMES = ("Movie Part 1.mkv", "Movie Part 2.mkv")

    @classmethod
    def _unplaced_pair(cls, *, excluded: tuple[str, ...] = ()) -> PendingImport:
        # Two files the grab placed nowhere, for one target episode.
        return pending_import(
            file_episode_map={},
            ordered_episode_ids=[101],
            seadex_files=list(cls._NAMES),
            excluded_files=[normalize_name(name) for name in excluded],
        )

    @classmethod
    def _strat(cls, *, served: bool = True, file_group: str | None = None) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # `file_group` names the group whose file the target episode already holds (None: no file).
        episode = sonarr_ep(1, 1, ep_id=101, release_group=file_group, episode_file_id=1 if file_group else 0)
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in cls._NAMES],
            episodes=[episode] if served else None,
        )
        return strat, sonarr

    def test_all_unmatched_returns_the_names_without_a_warning(self) -> None:
        pending = self._unplaced_pair()
        strat, sonarr = self._strat()
        recording = install_recording_hub()

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert first.unplaced_files == self._NAMES
        assert first.all_done is False
        assert first.command_issued is False
        assert first.pause_reason is PauseReason.NONE
        assert second == first
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_an_empty_episode_index_still_reads_unmatched(self) -> None:
        pending = self._unplaced_pair()
        strat, _ = self._strat(served=False)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == self._NAMES

    def test_every_file_excluded_at_grab_time_still_reads_unmatched(self) -> None:
        # Nothing of the record's own is on disk and nothing is authoritative: a recorded exclusion
        # silences the warning, never the verdict, so the record graduates instead of waiting out its deadline.
        pending = self._unplaced_pair(excluded=self._NAMES)
        strat, _ = self._strat()

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == self._NAMES

    def test_the_unmatched_names_leave_out_the_excluded_files(self) -> None:
        pending = self._unplaced_pair(excluded=self._NAMES[1:])
        strat, sonarr = self._strat()

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == self._NAMES[:1]
        assert sonarr.execute_calls == []

    def test_a_hand_import_that_filled_every_intended_episode_reads_imported(self) -> None:
        # The files still sit in the folder (a copy import), and the map still lacks them. The
        # intended episode holds the record's own group now: done, never re-reported.
        pending = self._unplaced_pair()
        strat, sonarr = self._strat(file_group="SubGroup")
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result == AttemptResult.imported()
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_an_untagged_hand_import_at_a_listed_size_reads_imported(self) -> None:
        # Sonarr reads no group off a name that carries none. The file's size says it is one of the torrent's.
        pending = pending_import(
            file_episode_map={}, ordered_episode_ids=[101], seadex_files=list(self._NAMES), release_sizes=(1000,)
        )
        strat, sonarr = self._strat()
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, size=1000)]

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result == AttemptResult.imported()
        assert sonarr.execute_calls == []

    def test_another_groups_copy_on_the_target_keeps_the_verdict_unmatched(self) -> None:
        # The stale copy this grab replaces is not ours: the human still has the import to do.
        pending = self._unplaced_pair()
        strat, _ = self._strat(file_group="OtherPick")

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == self._NAMES

    def test_a_seeded_file_beside_an_unmatched_one_imports_and_warns_once(self) -> None:
        # Something to import means the partial arm: today's once-a-run warning, no unmatched outcome.
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv", "Extras.mkv"])
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), manual_candidate("/d/Extras.mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
        )
        recording = install_recording_hub()

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert first.command_issued is True
        assert first.unplaced_files == ()
        assert sonarr.execute_calls[0][0][0].path == "/d/Show - 01 [1080p].mkv"
        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be matched" in m]
        assert len(warnings) == 1


class TestHeldBackFiles:
    """A file the grab left for a hand import keeps the record pending once every placed file is in."""

    # Each file's number is one episode early. The titles say where the titled two go; the plain file belongs on 601.
    _NAMES = ("show - 01 - The Return [grp].mkv", "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv")
    _TITLES = ("Pilot Flight", "The Return", "Festival Night")
    _SIZE_ON: ClassVar[dict[int, int]] = {601: 1003, 602: 1001, 603: 1002}

    @classmethod
    def _record(cls) -> PendingImport:
        return pending_import(
            file_episode_map={normalize_name(cls._NAMES[0]): [602], normalize_name(cls._NAMES[1]): [603]},
            hand_import_files=(normalize_name(cls._NAMES[2]),),
            seadex_files=list(cls._NAMES),
            release_sizes=tuple(sorted(cls._SIZE_ON.values())),
            ordered_episode_ids=(601, 602, 603),
        )

    @classmethod
    def _episodes(cls, *with_files: int) -> list[SonarrEpisode]:
        """The window, each episode of `filled` holding the record's file at its listed size."""

        return [
            sonarr_ep(1, n, ep_id=ep_id, title=title, size=cls._SIZE_ON[ep_id], release_group="SubGroup")
            if ep_id in with_files
            else sonarr_ep(1, n, ep_id=ep_id, title=title, episode_file_id=0)
            for n, (ep_id, title) in enumerate(zip(cls._SIZE_ON, cls._TITLES, strict=True), start=1)
        ]

    @classmethod
    def _strat(cls, *extras: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        """The download's files plus `extras` on disk, the placed two in."""

        return _make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in (*cls._NAMES, *extras)],
            episodes=cls._episodes(602, 603),
        )

    def test_the_held_back_file_keeps_the_record_and_is_named_every_poll(self) -> None:
        strat, sonarr = self._strat()
        recording = install_recording_hub()

        first = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)
        second = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert first.left_for_hand_import.names == (self._NAMES[2],)
        assert first.all_done is False
        assert second == first
        assert sonarr.execute_calls == []
        assert not any("could not be matched" in m for m in diagnostic_messages(recording, Severity.WARNING))

    def test_a_record_stored_without_hand_import_files_keeps_its_unplaced_file_pending(self) -> None:
        strat, sonarr = self._strat()
        pending = self._record()
        raw = pending.to_json()
        del raw["hand_import_files"]
        stored = PendingImport.from_json(raw, guards={claim.al_id: claim.guards for claim in pending.claims})

        result = strat.import_completed(stored, "/d", AttemptKind.ORDINARY)

        assert result.left_for_hand_import.names == (self._NAMES[2],)
        assert result.all_done is False
        assert sonarr.execute_calls == []

    def test_the_record_retires_once_a_hand_import_fills_its_window(self) -> None:
        strat, sonarr = self._strat()
        result = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = self._episodes(601, 602, 603)

        done = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert result.left_for_hand_import.names == (self._NAMES[2],)
        assert done.all_done is True
        assert done.left_for_hand_import.names == ()

    def test_an_untagged_hand_import_at_a_listed_size_retires_the_record(self) -> None:
        strat, sonarr = self._strat()
        no_group = sonarr_ep(1, 1, ep_id=601, title=self._TITLES[0], size=self._SIZE_ON[601], release_group=None)
        sonarr.episodes_return = [no_group, *self._episodes(602, 603)[1:]]

        done = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert done.all_done is True
        assert done.left_for_hand_import.names == ()

    def test_a_plain_skip_beside_the_held_back_file_warns_once_for_itself_alone(self) -> None:
        strat, _ = self._strat("Extras.mkv")
        recording = install_recording_hub()

        result = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)
        strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert result.left_for_hand_import.names == (self._NAMES[2],)
        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be matched" in m]
        assert len(warnings) == 1
        assert ": 1 file could not be matched" in warnings[0]

    def test_a_plain_skip_alone_retires(self) -> None:
        strat, _ = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), manual_candidate("/d/Extras.mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")],
        )
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv", "Extras.mkv"])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert result.left_for_hand_import.names == ()


class TestOrderOnlyHalves:
    """A bare-numbered pack whose split opener's halves only the DVD order numbers, its other eleven files in."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
    _KEYS: ClassVar[tuple[OrderOnlyKey, ...]] = (
        OrderOnlyKey("dvd", EpisodeKey(7, 1)),
        OrderOnlyKey("dvd", EpisodeKey(7, 2)),
    )

    def test_the_probe_keys_each_half_by_its_leaf_and_names_the_empty_opener(self) -> None:
        run = [normalize_name(name) for name in self._RUN]
        pending = pending_import(
            release_group="grp",
            seadex_files=self._RUN,
            file_episode_map={name: [400 + n] for n, name in enumerate(run[2:], start=2)},
            hand_import_files=run[:2],
            order_only=dict(zip(run[:2], self._KEYS, strict=True)),
            ordered_episode_ids=[400 + n for n in range(1, 13)],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in self._RUN],
            episodes=[
                sonarr_ep(4, 1, ep_id=401, episode_file_id=0),
                *(sonarr_ep(4, n, ep_id=400 + n, release_group="grp") for n in range(2, 13)),
            ],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.left_for_hand_import == LeftForHandImport(
            files=tuple(HandImportFile(name, key) for name, key in zip(self._RUN[:2], self._KEYS, strict=True)),
            missing=(EpisodeKey(4, 1),),
        )
        assert sonarr.execute_calls == []


class TestDefaultQualityWarning:
    """An unmatched `imports.default_quality` warns once per run, at the consume seam.

    `quality_parts_from_name` stays pure (its silent-empty return is pinned
    elsewhere). The executor is where the configured name meets the run's real
    Sonarr definitions, and it runs once per FILE, hence the once-per-run guard.
    """

    @staticmethod
    def _strat_with_default(name: str | None) -> tuple[SonarrStrategy, FakeSonarrClient, RecordingHub]:
        # The candidate + definitions resolve cleanly, so the only quality-related
        # warning in play is the unmatched-default one.
        quality = {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}}
        overrides: dict[str, list[str] | str] = {} if name is None else {"import_default_quality": name}
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv", quality=quality)],
            quality_defs=[QualityDefinition.model_validate(quality)],
            config_overrides=overrides,
        )
        return strat, sonarr, install_recording_hub()

    @staticmethod
    def _default_quality_warnings(recording: RecordingHub) -> list[str]:
        return [message for message in diagnostic_messages(recording, Severity.WARNING) if "default_quality" in message]

    def test_unmatched_name_warns_once_per_run(self) -> None:
        pending = pending_import()
        strat, sonarr, recording = self._strat_with_default("Blueray-1080p")

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        # Both polls stepped in (rebuilt the payload), yet the typo warned once.
        assert len(sonarr.execute_calls) == 2
        assert self._default_quality_warnings(recording) == [
            "imports.default_quality 'Blueray-1080p' matches no Sonarr quality definition - ignoring it",
        ]

        # The guard is per-run scratch: the run-start reset re-arms it.
        strat._executor.reset()
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert len(self._default_quality_warnings(recording)) == 2

    def test_matched_name_does_not_warn(self) -> None:
        strat, sonarr, recording = self._strat_with_default("WEBDL-1080p")

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []

    def test_unset_name_does_not_warn(self) -> None:
        strat, sonarr, recording = self._strat_with_default(None)

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []


def _imported_history(*rows: tuple[int, str]) -> HistoryPage:
    """A dead-tracked page whose import rows carry `(episodeId, droppedPath)` for series 7, above the grab."""

    records: list[dict[str, object]] = [
        {
            "id": 10 + index,
            "eventType": "downloadFolderImported",
            "date": "2026-01-02T00:00:00Z",
            "seriesId": 7,
            "episodeId": episode_id,
            "data": {"droppedPath": path},
        }
        for index, (episode_id, path) in enumerate(rows)
    ]
    records.append({"id": 1, "eventType": "grabbed", "date": "2026-01-01T00:00:00Z"})
    return HistoryPage.model_validate({"records": records, "totalRecords": len(records)})


def _closed_history() -> HistoryPage:
    """A history page whose newest relevant event is a prior full import."""

    return HistoryPage.model_validate(
        {
            "records": [
                {"id": 3, "eventType": "downloadFolderImported", "date": "2026-06-20T06:15:30Z"},
                {"id": 1, "eventType": "grabbed", "date": "2026-06-19T00:00:00Z"},
            ],
        },
    )


def _clean_history() -> HistoryPage:
    """A history page whose newest relevant event is a grab (genuinely downloading)."""

    return HistoryPage.model_validate(
        {"records": [{"id": 9, "eventType": "grabbed", "date": "2026-07-17T00:00:00Z"}]},
    )


def _tv_mapping() -> RemotePathMapping:
    """The incident-shaped mapping: qBittorrent's `/d` is Sonarr's `/remote/tv`."""

    return RemotePathMapping.model_validate({"host": "seedbox", "remotePath": "/d/", "localPath": "/remote/tv/"})


class TestFolderScanFallback:
    """The dead-loop cure: a downloadId scan that can't serve the download probes history and scans the folder.

    A download Sonarr's history maps to Imported/Failed/Ignored is queue-hidden
    and its `downloadId=` scan never serves it (a 500 on v4, `[]` on v5). Without
    the fallback every poll would defer "for a later run" for as long as the
    torrent stayed in qBittorrent.
    """

    @staticmethod
    def _strat(
        *,
        history: HistoryPage | None,
        folder_candidates: list[ManualImportCandidate] | None,
        mappings: list[RemotePathMapping] | None = None,
        config_overrides: dict[str, list[str] | str] | None = None,
    ) -> tuple[SonarrStrategy, HookedSonarrClient]:
        # candidates=None: the downloadId scan fails (the 500 shape) -> fallback.
        strat, sonarr = _make_sonarr_for_import(candidates=None, config_overrides=config_overrides)
        sonarr.history_page_return = history
        sonarr.folder_candidates_return = folder_candidates
        sonarr.path_mappings_return = mappings if mappings is not None else []
        return strat, sonarr

    @staticmethod
    def _answer_empty_by_id(
        sonarr: FakeSonarrClient,
        *,
        commands: tuple[CommandResource, ...] = (_RESCAN_AFTER_ADD,),
    ) -> None:
        """Script the downloadId scan to answer `[]` under the given command list."""

        sonarr.candidates_return = []
        sonarr.commands_return = list(commands)

    @staticmethod
    def _poll_then_empty_the_folder(
        strat: SonarrStrategy, sonarr: FakeSonarrClient, pending: PendingImport
    ) -> AttemptResult:
        """Poll with the episode index unread, then again once the folder scans empty and only 101 holds a file."""

        sonarr.episodes_return = []
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.folder_candidates_return = []
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
        return strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

    def test_dead_tracked_imports_from_translated_folder_without_download_id(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        # The folder scan received the TRANSLATED (Sonarr-visible) path.
        assert [folder for folder, _ in sonarr.folder_candidate_calls] == ["/remote/tv/Show"]
        [(files, mode)] = sonarr.execute_calls
        # Configured mode auto is forced to copy: the untracked Execute branch
        # with Auto resolves to MOVE and would rip the seeding files.
        assert mode == "copy"
        [entry] = files
        assert entry.downloadId is None
        # The wire dump must OMIT the key (unset), never send null.
        assert "downloadId" not in entry.model_dump(exclude_unset=True)
        assert entry.episodeIds == [101]

    def test_dead_tracked_explicit_mode_is_honored(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
            config_overrides={"import_mode": "move"},
        )

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        [(_, mode)] = sonarr.execute_calls
        assert mode == "move"

    def test_clean_probe_keeps_download_id_and_copies(self) -> None:
        # A transient blip on a genuinely-downloading torrent: the folder scan
        # substitutes for the candidates, and the entries keep the downloadId so
        # the import still resolves the queue row. Auto becomes copy all the same:
        # nothing here proves Sonarr tracks the download, and an untracked Auto
        # import moves the files out of the seeding torrent.
        strat, sonarr = self._strat(
            history=_clean_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert result.command_issued is True
        [(files, mode)] = sonarr.execute_calls
        assert mode == "copy"
        assert files[0].downloadId == "ABC123"

    def test_probe_failure_defaults_to_status_quo_and_is_not_memoized(self) -> None:
        strat, sonarr = self._strat(
            history=None,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        # Status-quo-safe default: entries keep the downloadId (converges even
        # in the dead-tracked state, just noisily - bounded, never a loop).
        assert all(files[0].downloadId == "ABC123" for files, _ in sonarr.execute_calls)
        # A FAILED probe is never memoized: the second activation re-probed.
        assert sonarr.download_history_calls == ["abc123", "abc123"]

    def test_clean_verdict_is_read_again_next_poll(self) -> None:
        strat, sonarr = self._strat(
            history=_clean_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.history_page_return = _closed_history()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        [_, (files, mode)] = sonarr.execute_calls
        assert mode == "copy"
        assert files[0].downloadId is None

    def test_dead_verdict_memoized_and_noted_once_per_run(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        # The dead-tracked note is debug-level; caplog needs a propagating logger.
        logger = debug_logger("pearlarr-dead-verdict")
        strat._executor.candidate_fetcher.logger = logger
        pending = pending_import()

        with caplog.at_level("DEBUG"):
            strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
            strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.download_history_calls == ["abc123"]
        notes = [r for r in caplog.records if "recorded this download as imported" in r.message]
        assert len(notes) == 1
        assert notes[0].levelname == "DEBUG"

    def test_empty_folder_scan_does_not_pin_and_download_id_scan_recovers(self) -> None:
        # 200 [] = the folder isn't visible to Sonarr (or the translation is
        # wrong). Pinning on it would wedge the record. Instead the next poll
        # retries the recoverable downloadId scan first.
        strat, sonarr = self._strat(history=_closed_history(), folder_candidates=[])
        pending = pending_import()

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert result.command_issued is False
        assert sonarr.execute_calls == []

        # The transient heals: the downloadId scan answers again and wins.
        sonarr.candidates_return = [manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert len(sonarr.candidate_calls) == 2
        assert len(sonarr.folder_candidate_calls) == 1
        assert sonarr.execute_calls[0][0][0].downloadId == "ABC123"

    def test_dead_tracked_empty_folder_warns_only_at_the_deadline(self, caplog: pytest.LogCaptureFixture) -> None:
        # An empty folder right after completion is expected (Sonarr's mount
        # lags), so ordinary polls note it at debug; the genuinely stuck shape
        # (an unmapped record, still empty at the deadline) warns once a run,
        # re-armed by reset. `_closed_history()` carries no import rows, so the
        # rebuild yields nothing and the record stays unmapped.
        recording = install_recording_hub()
        strat, _ = self._strat(history=_closed_history(), folder_candidates=[])
        logger = debug_logger("pearlarr-folder-scan")
        strat._executor.logger = logger
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        with caplog.at_level("DEBUG"):
            strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
            strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert diagnostic_messages(recording, Severity.WARNING) == []
        assert [r.levelname for r in caplog.records if "found no files" in r.message] == ["DEBUG", "DEBUG"]

        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 1
        assert "/d/Show" in warnings[0]

        strat._executor.reset()
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 2

    def test_dead_tracked_empty_folder_stays_quiet_once_the_seed_verifies(self) -> None:
        # An accounted-only record after a move-mode import: the by-id scan is
        # dead, the folder is empty, and the seeded file already verifies. The
        # deadline poll reads imported and prints nothing.
        recording = install_recording_hub()
        strat, sonarr = self._strat(history=_closed_history(), folder_candidates=[])
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            ordered_episode_ids=[101],
            seadex_files=["Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"],
            excluded_files=["show - 02 [1080p].mkv"],
        )

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is True
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_seeded_unverified_record_warns_once_with_the_folder_reason(self) -> None:
        # A seeded record whose file is neither in the folder nor on its
        # episode: the deadline prints ONE warning, the missing-files one,
        # carrying the empty-folder reason rather than a second line.
        recording = install_recording_hub()
        strat, _ = self._strat(history=_closed_history(), folder_candidates=[])

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "1 intended file not visible to Sonarr" in warning
        assert "a scan of /d/Show found no files" in warning

    def test_dead_tracked_empty_folder_warns_for_an_unmapped_record_at_the_deadline(self) -> None:
        # Nothing to verify and nothing to place: the deadline says so once,
        # the ordinary poll stays quiet, and no placement is invented.
        recording = install_recording_hub()
        strat, _ = self._strat(history=_closed_history(), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        polled = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        assert diagnostic_messages(recording, Severity.WARNING) == []

        deadline = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "found no files" in warning
        assert (polled.all_done, deadline.all_done) == (False, False)
        assert deadline.placements == {}

    def test_never_scanned_record_verifies_from_sonarr_history(self) -> None:
        # Sonarr imported the download before any scan saw its files: the
        # by-id scan is dead, the folder is empty, and the map is empty. Its
        # history rows stand in for the scan, the rebuilt map verifies on the
        # episode file, and the placements ride the probe with nothing POSTed.
        recording = install_recording_hub()
        name = "Show - 01 [1080p].mkv"
        strat, sonarr = self._strat(history=_imported_history((101, f"/d/Show/{name}")), folder_candidates=[])
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert result.placements == {normalize_name(name): (101,)}
        assert pending.file_episode_map == {}
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == []
        assert len(sonarr.folder_candidate_calls) == 1
        assert len(sonarr.download_history_calls) == 1

    def test_history_rebuild_leaves_seeds_and_exclusions_alone(self) -> None:
        # A seeded file Sonarr placed elsewhere stays a genuine mismatch, and
        # an excluded (sibling slice) file never lands on this record.
        file_map, excluded = "Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"
        strat, sonarr = self._strat(
            history=_imported_history((102, f"/d/Show/{file_map}"), (102, f"/d/Show/{excluded}")),
            folder_candidates=[],
        )
        sonarr.episodes_return = [sonarr_ep(1, 2, ep_id=102, release_group="SubGroup")]
        pending = pending_import(
            file_episode_map={file_map: [101]},
            ordered_episode_ids=[101],
            seadex_files=[file_map, excluded],
            excluded_files=[normalize_name(excluded)],
        )

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.placements == {}

    def test_rebuilt_map_that_does_not_verify_waits_without_pinning_it(self) -> None:
        # The history row names an episode holding no recommended file (the
        # import was replaced or deleted since): the deadline prints the one
        # missing-files warning with the folder reason, and the history
        # placement is NOT attached, so the next run re-derives it.
        recording = install_recording_hub()
        name = "Show - 01 [1080p].mkv"
        strat, _ = self._strat(history=_imported_history((101, f"/d/Show/{name}")), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.placements == {}
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "1 intended file not visible to Sonarr" in warning
        assert "a scan of /d/Show found no files" in warning

    def test_covered_record_never_rebuilds(self) -> None:
        # A mapped record verifies or waits on its own seed: the rows are
        # never consulted, so a conflicting row changes nothing.
        name = "Show - 01 [1080p].mkv"
        strat, _ = self._strat(history=_imported_history((102, f"/d/Show/{name}")), folder_candidates=[])
        pending = pending_import()

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.placements == {}

    def test_empty_poll_then_files_never_warns(self) -> None:
        # The common shape: the first scan lands before the folder is visible,
        # the next sees the files. No warning was ever due.
        recording = install_recording_hub()
        strat, sonarr = self._strat(history=_closed_history(), folder_candidates=[])
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.folder_candidates_return = [manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert not any("found no files" in w for w in diagnostic_messages(recording, Severity.WARNING))

    def test_clean_verdict_empty_folder_stays_quiet(self) -> None:
        # A transient by-id blip self-heals next poll, so no warning fires at
        # all (the by-id client read is quiet, the executor owns the noise).
        recording = install_recording_hub()
        strat, _ = self._strat(history=_clean_history(), folder_candidates=[])

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_nonempty_folder_scan_pins_for_the_run_and_reset_unpins(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        # Pinned after the nonempty scan: the second poll went straight to the
        # folder (one downloadId attempt total), and the run fetched the
        # remote path mappings exactly once.
        assert len(sonarr.candidate_calls) == 1
        assert len(sonarr.folder_candidate_calls) == 2
        assert sonarr.path_mapping_calls == 1

        # The pin is per-run scratch: reset re-arms the downloadId scan.
        strat._executor.reset()
        sonarr.candidates_return = [manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        assert len(sonarr.candidate_calls) == 2

    def test_empty_download_id_scan_of_a_dead_tracked_download_imports_from_its_folder(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr)

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        [(files, mode)] = sonarr.execute_calls
        assert mode == "copy"
        assert files[0].downloadId is None

    def test_empty_download_id_scan_waits_without_the_proof(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr, commands=(_RESCAN_BEFORE_ADD,))

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_dead_tracked_record_without_the_proof_rescans_then_imports_from_its_folder(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr, commands=(_RESCAN_BEFORE_ADD,))

        def prove() -> None:
            sonarr.commands_return = [_RESCAN_AFTER_ADD]

        sonarr.on_rescan = prove

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.refresh_calls == 1
        [(files, _mode)] = sonarr.execute_calls
        assert files[0].downloadId is None

    @pytest.mark.parametrize("history", [_clean_history(), None], ids=["clean", "unreadable"])
    def test_folder_scan_is_never_kept(self, history: HistoryPage | None) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        sonarr.queue_return = [queue_record("ABC123", "importBlocked", status="warning")]

        def start_a_pass() -> None:
            sonarr.commands_return = [_RESCAN_BEFORE_ADD, _started_process_pass()]

        sonarr.on_fetch = start_a_pass
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.folder_candidates_return = None
        sonarr.commands_return = [_RESCAN_BEFORE_ADD]
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert blocked.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    @pytest.mark.parametrize("history", [_clean_history(), None], ids=["clean", "unreadable"])
    def test_empty_download_id_scan_waits_unless_history_reads_dead(self, history: HistoryPage | None) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr)

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_empty_download_id_scan_of_a_queued_download_waits(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr)
        sonarr.queue_return = [queue_record("ABC123", "importPending", status="warning")]

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_empty_download_id_scan_with_an_empty_folder_waits_and_warns_at_the_deadline(self) -> None:
        recording = install_recording_hub()
        first, second = "Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"
        strat, sonarr = self._strat(history=_imported_history((101, f"/d/Show/{first}")), folder_candidates=[])
        self._answer_empty_by_id(sonarr)
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], seadex_files=[first, second])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "found no files" in warning

    @pytest.mark.parametrize(
        ("by_id", "rebuilt"),
        [(None, {normalize_name("Show - 01 [1080p].mkv"): (101,)}), ([], {})],
        ids=["failed", "empty"],
    )
    def test_folder_pinned_record_rebuilds_from_history_only_after_a_failed_download_id_scan(
        self,
        by_id: list[ManualImportCandidate] | None,
        rebuilt: FileEpisodeMap,
    ) -> None:
        first, second = "Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"
        strat, sonarr = self._strat(
            history=_imported_history((101, f"/d/Show/{first}")),
            folder_candidates=[manual_candidate(f"/d/Show/{first}"), manual_candidate(f"/d/Show/{second}")],
        )
        self._answer_empty_by_id(sonarr)
        sonarr.candidates_return = by_id
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], seadex_files=[first, second])

        result = self._poll_then_empty_the_folder(strat, sonarr, pending)

        assert result.placements == rebuilt

    def test_failed_download_id_scan_then_an_empty_answer_never_rebuilds_from_history(self) -> None:
        first, second = "Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"
        strat, sonarr = self._strat(history=_imported_history((101, f"/d/Show/{first}")), folder_candidates=None)
        self._answer_empty_by_id(sonarr)
        sonarr.candidates_return = None
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], seadex_files=[first, second])
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.candidates_return = []
        sonarr.folder_candidates_return = []

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is False

    def test_translated_in_flight_command_suppresses_reissue(self) -> None:
        # A dead-tracked import POSTs translated paths and no downloadId.
        # The next poll's guard must recognize it through the memoized translation
        # (the scripted command has no episode ids, so only the path arm can).
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        assert len(sonarr.execute_calls) == 1

        sonarr.commands_return = [
            CommandResource.model_validate(
                {
                    "name": "ManualImport",
                    "status": "started",
                    "body": {"files": [{"path": "/remote/tv/Show/Show - 01 [1080p].mkv", "episodeIds": []}]},
                },
            ),
        ]
        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert len(sonarr.execute_calls) == 1

    def test_single_file_content_path_scans_the_file(self) -> None:
        # A single-FILE torrent's content_path IS the file. Sonarr's folder=
        # param accepts a file path (its FileExists arm).
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )

        result = strat.import_completed(pending_import(), "/d/Show - 01 [1080p].mkv", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert [folder for folder, _ in sonarr.folder_candidate_calls] == ["/remote/tv/Show - 01 [1080p].mkv"]


class TestUnwatchedTorrentImport:
    """A finished torrent in a category Sonarr doesn't watch imports from its folder.

    Sonarr never tracks such a torrent: its `downloadId=` scan answers `[]` for good and its history holds
    nothing. Once the queue and a completed rescan show Sonarr isn't tracking it, the folder is read and the
    record's map imports the files, with no downloadId and by copy.
    """

    @staticmethod
    def _strat(
        category: str = "manual",
        *,
        folder_candidates: list[ManualImportCandidate] | None = None,
    ) -> tuple[SonarrStrategy, HookedSonarrClient, QbitCategoryReads]:
        """Sonarr watching `tv-sonarr`, the record's torrent in `category`, and Sonarr not tracking it.

        The downloadId scan answers `[]`, the queue is empty, a rescan finished after the add, and the
        history holds only the neutral page.
        """

        qbit = QbitCategoryReads(category)
        categories = make_fetched_categories(grab="tv-sonarr", post_import=None, qbit=qbit.client())
        strat, sonarr = _make_sonarr_for_import(candidates=[], categories=categories)
        sonarr.commands_return = [_RESCAN_AFTER_ADD]
        sonarr.history_page_return = HistoryPage.model_validate({"records": []})
        sonarr.folder_candidates_return = (
            [manual_candidate("/d/Show/Show - 01 [1080p].mkv")] if folder_candidates is None else folder_candidates
        )
        sonarr.path_mappings_return = []
        return strat, sonarr, qbit

    def test_imports_from_the_folder_by_the_records_map_without_a_download_id(self) -> None:
        strat, sonarr, _ = self._strat()

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert result.command_issued is True
        [(files, mode)] = sonarr.execute_calls
        assert mode == "copy"
        [entry] = files
        assert entry.episodeIds == [101]
        assert "downloadId" not in entry.model_dump(exclude_unset=True)

    def test_a_torrent_in_the_watched_category_waits_for_sonarr(self) -> None:
        strat, sonarr, _ = self._strat("tv-sonarr")

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_a_download_sonarr_still_holds_a_queue_row_for_is_not_unwatched(self) -> None:
        # Sonarr v4 goes on tracking a torrent moved out of its category until it restarts.
        strat, sonarr, _ = self._strat()
        sonarr.queue_return = [queue_record("ABC123", "importPending", status="warning")]

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    def test_the_category_is_asked_again_every_poll(self) -> None:
        strat, sonarr, qbit = self._strat()
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        qbit.category = "tv-sonarr"
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        # Moved into the watched category, the download gets its id back. Still a folder import, so it copies.
        posted = [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls]
        assert posted == [(None, "copy"), (None, "copy"), ("ABC123", "copy")]

    def test_a_failed_download_id_scan_of_an_unwatched_torrent_omits_the_id(self) -> None:
        strat, sonarr, _ = self._strat()
        sonarr.candidates_return = None

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        [(files, mode)] = sonarr.execute_calls
        assert (files[0].downloadId, mode) == (None, "copy")

    def test_an_empty_folder_warns_at_the_deadline_with_the_folder_reason(self) -> None:
        recording = install_recording_hub()
        strat, sonarr, _ = self._strat(folder_candidates=[])

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert sonarr.execute_calls == []
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "1 intended file not visible to Sonarr" in warning
        assert "a scan of /d/Show found no files" in warning


class TestResolveLanguageObjects:
    """resolve_language_objects maps names to {id,name}, dropping unknowns."""

    def test_resolves_in_request_order(self) -> None:
        defs = [Language(id=1, name="English"), Language(id=8, name="Japanese")]

        result = resolve_language_objects(["Japanese", "English"], defs)

        assert result == [Language(id=8, name="Japanese"), Language(id=1, name="English")]

    def test_skips_unknown_names(self) -> None:
        defs = [Language(id=8, name="Japanese")]

        result = resolve_language_objects(["Japanese", "Klingon"], defs)

        assert result == [Language(id=8, name="Japanese")]

    def test_matches_case_insensitively(self) -> None:
        defs = [Language(id=8, name="Japanese")]

        result = resolve_language_objects(["japanese"], defs)

        assert result == [Language(id=8, name="Japanese")]


class _FakeRadarr:
    """Minimal Radarr client: scripts (and records) the per-movie movie-file list."""

    def __init__(self, files: list[MovieFile]) -> None:
        self._files = files
        self.movie_files_calls: list[int] = []

    def movie_files(self, movie_id: int) -> list[MovieFile]:
        self.movie_files_calls.append(movie_id)
        return self._files


def _one_group_dict(srg: str) -> SeadexDict:
    """A one-group `SeadexDict` keyed by `srg`, carrying a single URL record."""

    url = f"https://nyaa.si/{srg}"
    return {srg: rg_group({url: make_release(url=url)})}


class TestRadarrProcessAlIdSeam:
    """process_al_id's movie middle threads its facts through the held services."""

    @staticmethod
    def _make_strat(
        run: _FakeRunServices,
        *,
        files: list[MovieFile] | None = None,
        interactive: bool = False,
    ) -> tuple[RadarrStrategy, _FakeRadarr]:
        radarr = _FakeRadarr(files or [])
        strat = make_bare_instance(
            RadarrStrategy,
            _services=run,
            radarr=radarr,
            logger=make_logger(),
            _config=make_config(interactive=interactive),
        )
        return strat, radarr

    def test_cached_entry_short_circuits_before_any_movie_read(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), cached_skip=True)
        strat, radarr = self._make_strat(run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        # Neither the SeaDex parse nor the movie files were reached.
        assert run.get_seadex_dict_calls == []
        assert radarr.movie_files_calls == []
        # The lazy backfill coverage passed is the movie's empty-coverage thunk.
        [coverage] = run.cached_skip_coverages
        assert coverage() == ""

    def test_no_releases_routes_to_shared_skip_with_movie_cache_details(self) -> None:
        entry = make_entry_record(anilist_id=5, url="https://releases.moe/5")
        run = _FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict={},
        )
        strat, _ = self._make_strat(run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        expected: CachedTitleUpdate = {
            "name": "Movie Title",
            "updated_at": entry.updated_at,
            "url": "https://releases.moe/5",
            "coverage": "",
        }
        assert run.no_releases_calls == [(5, expected)]
        assert run.grab_requests == []

    def test_happy_path_threads_releases_into_grab_request(self) -> None:
        entry = make_entry_record(anilist_id=5, url="https://releases.moe/5")
        seadex_dict = _one_group_dict("SubGroup")
        filtered = _one_group_dict("SubGroup")
        run = _FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict=seadex_dict,
            filter_downloads_result=grab_plan(filtered),
        )
        strat, _ = self._make_strat(run, files=[MovieFile(release_group="OldGroup", size=100)])

        strat.check_title(_Item(id=3, title="Item Title"), 5, TitleMapping(anilist_id=5))
        # The download filter received the movie's accumulated releases.
        assert run.filter_downloads_calls == [(seadex_dict, ArrFiles(by_group={"OldGroup": (100,)}))]
        [req] = run.grab_requests
        assert req.al_id == 5
        assert req.arr_title == "Item Title"
        assert req.title_name == "Movie Title"
        # The SeaDex entry rides the request whole (the notifier reads its
        # url/notes/comparisons), so pin identity, not a copied field.
        assert req.entry is entry
        # The grab consumes the FILTERED dict, not the pre-filter dict.
        assert req.seadex_dict is filtered
        # replaced_groups carries the movie's tagged groups (all of them).
        assert req.replaced_groups == ("OldGroup",)

    def test_multi_edition_movie_forwards_every_release_group(self) -> None:
        # A multi-edition movie holds files from two groups. The GrabRequest (and
        # so the notifier) must see both, in the release dict's insertion order -
        # not just the first file's group.
        run = _FakeRunServices(
            prologue_entry=make_entry_record(),
            anilist_title="Movie Title",
            seadex_dict=_one_group_dict("SubGroup"),
            filter_downloads_result=grab_plan(_one_group_dict("SubGroup")),
        )
        strat, _ = self._make_strat(
            run,
            files=[
                MovieFile(release_group="GroupA", size=100),
                MovieFile(release_group="GroupB", size=200),
            ],
        )

        strat.check_title(_Item(id=3, title="Item Title"), 5, TitleMapping(anilist_id=5))

        [req] = run.grab_requests
        assert req.replaced_groups == ("GroupA", "GroupB")

    def _interactive_call_count(self, *, interactive: bool, n_groups: int) -> int:
        """Run one process_al_id and count the interactive-filter invocations."""

        seadex_dict: SeadexDict = {}
        for idx in range(n_groups):
            seadex_dict.update(_one_group_dict(f"Group{idx}"))
        run = _FakeRunServices(prologue_entry=make_entry_record(), seadex_dict=seadex_dict)
        strat, _ = self._make_strat(run, interactive=interactive)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        return len(run.interactive_calls)

    def test_interactive_prompts_only_for_multi_release_interactive(self) -> None:
        assert self._interactive_call_count(interactive=True, n_groups=2) == 1
        assert self._interactive_call_count(interactive=True, n_groups=1) == 0
        assert self._interactive_call_count(interactive=False, n_groups=2) == 0


class TestRadarrFiles:
    """radarr_files accumulates sizes per group and never hard-errors."""

    def test_multiple_distinct_groups_kept_not_errored(self) -> None:
        # VU3: 2 distinct groups no longer raise (which skipped the movie every run).
        # The fold carries both so the planner dedups against each.
        radarr = _FakeRadarr([MovieFile(release_group="A", size=100), MovieFile(release_group="B", size=200)])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(by_group={"A": (100,), "B": (200,)})

    def test_same_group_sizes_accumulate(self) -> None:
        # CB6: two files of one group keep BOTH sizes (the old comprehension collapsed
        # to the last).
        radarr = _FakeRadarr([MovieFile(release_group="A", size=100), MovieFile(release_group="A", size=200)])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(by_group={"A": (100, 200)})

    def test_untagged_files_fold_to_untagged_sizes(self) -> None:
        # No {None: [None]} placeholder and no null key: untagged files carry
        # their sizes on `no_group`, and a no-files movie folds to the empty record.
        radarr = _FakeRadarr([MovieFile(release_group=None, size=100), MovieFile(release_group="A", size=200)])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(by_group={"A": (200,)}, no_group=(100,))

    def test_no_files_returns_empty_record(self) -> None:
        radarr = _FakeRadarr([])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles()

    def test_untagged_unreadable_size_folds_to_zero(self) -> None:
        # An unreadable untagged size folds to 0, which disqualifies the whole
        # ownership multiset (fail-closed, like Sonarr's untagged pass): the
        # readable sizes alone must not let a pick claim the copy.
        radarr = _FakeRadarr([MovieFile(release_group=None, size=500), MovieFile(release_group=None, size=None)])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(no_group=(500, 0))


class TestSonarrFiles:
    """sonarr_files feeds the planner's untagged pass nothing."""

    def test_untagged_episode_files_never_reach_untagged(self) -> None:
        # PIN: an untagged episode file contributes NOTHING to the record - it
        # travels on the ep_list into the planner's identity pass instead. The
        # planner merges both arrs' untagged sources, so a file fed through
        # both would count twice and could then never be owned.
        episodes = make_sonarr_episodes()

        files = episodes.sonarr_files(
            [
                sonarr_ep(1, 1, ep_id=101, release_group=None, size=700),
                sonarr_ep(1, 2, ep_id=102, release_group="RG", size=800),
            ]
        )

        assert files == ArrFiles(by_group={"RG": (800,)})
        assert files.no_group == ()


class TestArrFiles:
    """The record's own contract: deep-frozen, hashable, one shared fold."""

    def test_tagged_detaches_and_rejects_mutation(self) -> None:
        source = {"RG": (100,)}
        releases = ArrFiles(by_group=source)

        source["Other"] = (5,)

        assert dict(releases.by_group) == {"RG": (100,)}
        with pytest.raises(TypeError):
            cast("MutableMapping[str, tuple[int, ...]]", releases.by_group)["X"] = (1,)

    def test_equal_records_hash_equal_across_key_order(self) -> None:
        # dict eq ignores insertion order, so the hash must too.
        first = ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,))
        second = ArrFiles(by_group={"B": (2,), "A": (1,)}, no_group=(3,))

        assert first == second
        assert hash(first) == hash(second)

    def test_from_files_keeps_or_drops_untagged_by_arr(self) -> None:
        # One fold for both arrs: only the untagged destination differs. The
        # unreadable tagged size is dropped; the group keeps its name.
        files = [
            MovieFile(release_group=None, size=700),
            MovieFile(release_group="RG", size=800),
            MovieFile(release_group="RG", size=None),
        ]

        assert ArrFiles.from_files(files, keep_no_group=True) == ArrFiles(by_group={"RG": (800,)}, no_group=(700,))
        assert ArrFiles.from_files(files, keep_no_group=False) == ArrFiles(by_group={"RG": (800,)})

    def test_groups_label_renders_untagged_and_empty(self) -> None:
        assert ArrFiles(by_group={"A": (1,)}, no_group=(2,)).groups_label() == "A, (none)"
        assert ArrFiles().groups_label() == "(no files)"

    def test_group_count_matches_the_rendered_entries(self) -> None:
        # Untagged files fold to the single `(none)` entry, so they count once.
        assert ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,)).group_count() == 3
        assert ArrFiles().group_count() == 0

    def test_replaced_groups_lists_every_tagged_name(self) -> None:
        assert ArrFiles(by_group={"A": (1,), "B": ()}).groups() == ("A", "B")
