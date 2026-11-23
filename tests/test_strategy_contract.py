# pyright: strict
# pyright: reportPrivateUsage=false
# These reach into the strategy's private collaborators (strat._episodes /
# strat._executor) to pin the interface. Strict re-flags that and the repo disables
# reportPrivateUsage for tests.
"""Interface tests for the composition split.

These pin the contract between the arr run machinery and the Arr strategies: each
`ArrStrategy` hook reaches the shared pipeline only through the injected
`ArrRunServices` the strategy keeps as `self._services` - and since the services
split out of the old god class, that interface is literal: the scripted fake below
is a real `ArrRunServices` subclass. The strategies are built bare
(`object.__new__`) so no live Sonarr/Radarr client is constructed.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import ClassVar, NamedTuple, override

import pytest
from seadex import EntryRecord

from pearlarr.aired_view import AiredSeries, NoView, aired_series
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
    MatchedEpisode,
    MovieFile,
    ParsedFileInfo,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    SeasonTypeCount,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
)
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import Arr
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.grab_placement import ListedFile, PlacedEntry, place_release
from pearlarr.mappings import ExternalIds, MappingSource, ResolvedMappings, TitleMapping
from pearlarr.output import Accent, Severity, StyledValue, TitleDetail
from pearlarr.pending_follow import PendingFollower
from pearlarr.pending_import import (
    FileEpisodeMap,
    OwnedAtGrab,
    PendingImport,
    SizeMatchedEpisode,
    TitleNames,
)
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.placement_types import FetchedListings, TorrentListing, episode_index
from pearlarr.planner import GrabPlan
from pearlarr.radarr_strategy import _HISTORY_SKEW_HOURS, RadarrStrategy
from pearlarr.releases import EpisodeKey, OrderOnlyKey, SeadexDict, SkipReason
from pearlarr.series_reads import SeriesReads
from pearlarr.sonarr_episodes import SonarrEpisodes, sonarr_series_hash
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.stamps import parse_stamp, stamp_of
from pearlarr.tvdb_orders import AIRED_ORDER, OrderRow, SeriesNumbering, SonarrSeriesOrders
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
    FakeCacheStore,
    FakeClock,
    ScriptedAniListClient,
    claim_anilist_ids,
    every_series,
    grab_plan,
    held_series_episodes,
    make_anilist_gateway,
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
    parsed_info,
    pending_import,
    placed_episode,
    queue_record,
    rg_group,
    scripted_roles,
    sonarr_ep,
    stand_in_ep,
    stored_orders,
    stored_scene_mappings,
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
    """A fake Arr item exposing whatever id attributes a test sets.

    Declares the full `ArrItem` surface so it structurally satisfies the
    `SonarrItem` / `RadarrItem` protocols. Each test sets only the attributes
    the hook under test actually reads.
    """

    id: int
    title: str = ""
    imdbId: str | None
    monitored: bool
    tvdbId: int
    tmdbId: int
    seasonType: str = AIRED_ORDER
    seasonTypes: tuple[SeasonTypeCount, ...] = ()
    alternateTitles: tuple[AlternateTitle, ...] = ()
    totalEpisodeCount: int | None = None
    order_sent: bool = False

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


class TitleMappingsCall(NamedTuple):
    """One recorded `title_mappings` call (the args the strategy forwarded).

    Defaults mirror the method's own defaults, so a recorded call equals an expected
    one constructed with only the fields the strategy actually varied.
    """

    ids: ExternalIds
    log_ignored: bool = True


class _FakeRunServices(ArrRunServices):
    """A scripted `ArrRunServices` - the object a strategy keeps as `self._services`.

    A REAL `ArrRunServices` subclass (the class name is literal now, so it also
    satisfies the constructors' typed `services` parameter), with its own
    `__init__` that never calls the expensive real one. Every member of the
    strategy-facing interface is overridden: each scriptable result is a
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
        plan_grabs_result: GrabPlan | None = None,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
        arr: Arr = Arr.SONARR,
        cache_store: FakeCacheStore | None = None,
    ) -> None:
        # The pending import store the strategies read stored pending imports through, bound as an arr run would
        # bind it.
        self.cache_store = cache_store if cache_store is not None else FakeCacheStore()
        self._arr = arr
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
        self._plan_grabs_result = plan_grabs_result
        self._import_wait_mode = import_wait_mode
        self.title_mappings_calls: list[TitleMappingsCall] = []
        self.anilist_id_prologue_calls: list[int] = []
        self.no_episodes_titles: list[str] = []
        self.unmonitored_titles: list[str] = []
        self.missing_episodes: list[str] = []
        self.numbers_orders: list[str] = []
        self.left_out: list[str] = []
        self.log_al_title_calls: list[str] = []
        self.title_block_coverages: list[str | None] = []
        self.cached_skip_coverages: list[Callable[[], str]] = []
        self.get_seadex_dict_calls: list[EntryRecord] = []
        self.interactive_calls: list[SeadexDict] = []
        self.plan_grabs_calls: list[tuple[SeadexDict, ArrFiles]] = []
        self.planned_title_episodes: list[Sequence[SonarrEpisode] | None] = []
        self.grab_requests: list[GrabRequest] = []
        self.no_releases_calls: list[tuple[int, CachedTitleUpdate]] = []
        self.no_releases_unread: list[int] = []
        self.nothing_to_grab_calls: list[tuple[int, CachedTitleUpdate]] = []
        self.nothing_to_grab_unread: list[int] = []
        self.invalid_selection_skips = 0
        self.pauses_owed = 0

    @property
    @override
    def pending_imports(self) -> PendingImportStore:
        return self._pending_imports

    @override
    def title_needs_check(self, anilist_id: int) -> bool:
        del anilist_id
        return self._needs_scan

    @override
    def cached_numbering(self, anilist_id: int) -> SeriesNumbering | None:
        cached = self.cache_store.get_cached_title(self._arr, anilist_id)
        return None if cached is None else cached.numbering

    @override
    def title_mappings(self, ids: ExternalIds, log_ignored: bool = True) -> dict[int, TitleMapping]:
        self.title_mappings_calls.append(TitleMappingsCall(ids, log_ignored))
        return self._anilist_ids

    @override
    def mappings_with_ignored(self, ids: ExternalIds) -> ResolvedMappings:
        del ids
        return ResolvedMappings(kept=self._anilist_ids, ignored=self._ignored)

    @override
    def begin_check(self, anilist_id: int) -> EntryRecord | None:
        self.anilist_id_prologue_calls.append(anilist_id)
        return self._prologue_entry

    @override
    def is_unchanged(
        self,
        anilist_id: int,
        sd_entry: EntryRecord,
        coverage: Callable[[], str],
    ) -> bool:
        del anilist_id, sd_entry
        self.cached_skip_coverages.append(coverage)
        return self._cached_skip

    @override
    def resolve_title(self, anilist_id: int) -> Title:
        return Title(anilist_id=anilist_id, name=self._anilist_title, names=TitleNames(anilist=(self._anilist_title,)))

    @override
    def new_cached_title_update(self, title: Title, sd_entry: EntryRecord) -> CachedTitleUpdate:
        # A pure builder over its arguments (no services state), so the real one serves the fake.
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
    def plan_grabs(
        self,
        seadex_dict: SeadexDict,
        arr_files: ArrFiles,
        title_episodes: Sequence[SonarrEpisode] | None = None,
    ) -> GrabPlan:
        self.plan_grabs_calls.append((seadex_dict, arr_files))
        self.planned_title_episodes.append(title_episodes)
        if self._plan_grabs_result is not None:
            return self._plan_grabs_result
        return grab_plan(seadex_dict)

    @property
    @override
    def import_wait_mode(self) -> ImportWaitMode:
        return self._import_wait_mode

    @override
    def report_no_releases(self, anilist_id: int, update: CachedTitleUpdate, *, metadata_unread: bool = False) -> None:
        self.no_releases_calls.append((anilist_id, update))
        if metadata_unread:
            self.no_releases_unread.append(anilist_id)

    @override
    def cache_nothing_to_grab(
        self, anilist_id: int, update: CachedTitleUpdate, *, metadata_unread: bool = False
    ) -> None:
        self.nothing_to_grab_calls.append((anilist_id, update))
        if metadata_unread:
            self.nothing_to_grab_unread.append(anilist_id)

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
    def log_numbers_order(self, order: str) -> None:
        self.numbers_orders.append(order)

    @override
    def log_aired_only(self, left_out: str) -> None:
        self.left_out.append(left_out)

    @override
    def open_title_block(self, title: str, sd_entry: EntryRecord, coverage: str | None = None) -> None:
        del sd_entry
        self.log_al_title_calls.append(title)
        self.title_block_coverages.append(coverage)


def test_fake_overrides_the_full_public_surface() -> None:
    """`@override` catches renames and signature drift but NOT additions.

    A new public `ArrRunServices` member would be silently inherited here (real
    body over fake state) and only fail at test runtime. Reflection closes
    that gap - extend the fake when this fails.
    """

    # Loop-driven members (the arr run loop calls these, strategies never do), so the
    # fake doesn't script them.
    lifecycle = {"bind_arr_run", "state", "is_dry_run", "mark_dirty", "pay_pause", "selection_changed"}
    surface = {
        name
        for name, member in vars(ArrRunServices).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    overridden = {name for name in vars(_FakeRunServices) if not name.startswith("_")}
    assert surface - lifecycle <= overridden


def test_the_services_log_missing_episodes_as_a_caution_detail() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_missing_episodes("S01 E02")

    assert recording.of_type(TitleDetail) == [TitleDetail("missing", StyledValue("S01 E02", Accent.CAUTION))]


def test_the_services_log_whose_numbers_a_titles_lines_print_as_a_dim_detail_naming_sonarrs_order() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_numbers_order("dvd")

    assert recording.of_type(TitleDetail) == [TitleDetail("numbers", StyledValue("by Sonarr's DVD order", Accent.DIM))]


def test_the_services_log_the_files_left_out_as_a_note_detail_labeled_aired() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_aired_only("S01 E02 (1 of 2 files, NAN0)")

    assert recording.of_type(TitleDetail) == [
        TitleDetail(
            "left out",
            StyledValue("aired episodes not in Sonarr's order: S01 E02 (1 of 2 files, NAN0)", Accent.NOTE),
        )
    ]


class _ScriptedListings:
    """A scripted listings collaborator: per-hash listings and size matches."""

    def __init__(
        self,
        listed: Mapping[str, TorrentListing] | None = None,
        episode_by_size: Mapping[int, int] | None = None,
    ) -> None:
        self._read = FetchedListings(listed or {}, episode_by_size or {})

    def read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings:
        del series_id, resolved
        return self._read


class _ScriptedParse:
    """Parse collaborator serving scripted files per release (no Sonarr round-trip), none for an unscripted release."""

    def __init__(self, files_by_url: dict[str, tuple[ListedFile, ...]] | None = None) -> None:
        self._files = files_by_url or {}

    def parsed_files(self, seadex_dict: SeadexDict) -> dict[str, tuple[ListedFile, ...]]:
        return {
            url: self._files.get(url, ()) for group_releases in seadex_dict.values() for url in group_releases.releases
        }


class TestItemMappingsDelegates:
    """item_mappings resolves through the services it keeps, with arr-specific ids."""

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrStrategy, _services=run)

        strat.item_mappings(_Item(tvdbId=99, imdbId="tt9"))

        assert run.title_mappings_calls == [TitleMappingsCall(ExternalIds(tvdb=99, imdb="tt9"), log_ignored=True)]


class TestFilterToSingle:
    """filter_to_single narrows by the arr's external id (no arr run loop needed)."""

    def test_radarr_matches_tmdb_id(self) -> None:
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())
        items: list[RadarrItem] = [_Item(tmdbId=1), _Item(tmdbId=2)]

        assert strat.filter_to_single(items, 2) == [items[1]]
        assert strat.filter_to_single(items, 7) == []

    def test_sonarr_matches_tvdb_id(self) -> None:
        strat = make_bare_instance(SonarrStrategy, logger=make_logger())
        items: list[SonarrItem] = [_Item(tvdbId=10), _Item(tvdbId=20)]

        assert strat.filter_to_single(items, 10) == [items[0]]


class TestArrRunStartHook:
    """get_items doubles as the arr run's start hook: it resets the per-run scratch.

    The episode reset/series hash now lives on the SonarrEpisodes collaborator.
    This stays at strategy level to pin that get_items actually routes through it.
    """

    def test_sonarr_get_items_clears_the_series_episodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        series: list[SonarrItem] = [_Item(id=5, tvdbId=50), _Item(id=7, tvdbId=70)]
        strat = make_sonarr_sync(sonarr=FakeSonarrClient(all_series=series), series_episodes={5: [sonarr_ep(1, 1)]})
        strat._episodes._picks_logged = {11}
        monkeypatch.setattr(strat._episodes, "mapped_series", every_series)

        result = strat.get_items()

        assert result == series
        # The episode collaborator drops its caches + recomputes the series hash as it enumerates.
        assert held_series_episodes(strat._episodes) == {}
        assert strat._episodes._picks_logged == set()
        assert strat._reads.series_hash == sonarr_series_hash([5, 7])


class TestCheckTitleThreadsServices:
    """The per-title head runs through the services it keeps. No entry stops this title."""

    def test_radarr_no_seadex_entry_stops_at_begin_check(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(RadarrStrategy, _services=run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.anilist_id_prologue_calls == [5]

    def test_sonarr_no_seadex_entry_stops_at_begin_check(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrStrategy, _services=run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.anilist_id_prologue_calls == [5]

    def test_sonarr_anibridge_empty_mapping_stops_with_warning(self) -> None:
        # The AniBridge no-usable-ranges case (a real empty-{} tvdb mapping: source
        # ANIBRIDGE, kind EPISODE_RANGES): the NO_EPISODES row PLUS a visible WARNING
        # naming the cause. The warning keys off source (so it also covers the
        # imdb/tmdb case without TVDB mappings), so the mapping must carry source=ANIBRIDGE as a
        # real one does. Fails on the unfixed path, which silently grabbed nothing.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[]))
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            **scripted_roles(episodes, run.pending_imports),
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

    def test_sonarr_all_unmonitored_episodes_stop_with_an_unmonitored_row(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        unmonitored = sonarr_ep(1, 1).model_copy(update={"monitored": False})
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            **scripted_roles(ScriptedEpisodes(FakeSonarrClient(episodes=[unmonitored])), run.pending_imports),
            _config=make_config(sleep_time=0, ignore_unmonitored=True),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.unmonitored_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert run.pauses_owed == 1

    def test_sonarr_all_invalid_selection_routes_to_shared_stop(self) -> None:
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
            **scripted_roles(ScriptedEpisodes(FakeSonarrClient(episodes=[sonarr_ep(1, 1)])), run.pending_imports),
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
    """A qBittorrent stub whose poll always reads one complete torrent."""

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


_FILELESS_EPISODES = [sonarr_ep(1, number, ep_id=100 + number, episode_file_id=0) for number in range(1, 10)]
"""Episodes 101 to 109 of one season, none with a file."""


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

    The fake returns the given queue records, current episodes, ManualImport
    candidates, quality definitions, languages and an execute command id -
    everything `import_completed` reaches over the network - and records the two
    import commands so a test asserts on recorded state. `episodes` defaults to
    `_FILELESS_EPISODES`, so the file map episodes have NO file yet (they need importing)
    and the done-check never short-circuits. Pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. `queue` defaults to empty (Sonarr
    isn't tracking the download, so the strategy runs an auto import). `commands` lists the
    commands beside the fake's neutral rescan, so no list given here reads as a
    restart. The fake's refresh / command-status defaults resolve immediately so
    the rescan never waits.
    """

    sonarr = HookedSonarrClient(
        queue=queue,
        episodes=_FILELESS_EPISODES if episodes is None else episodes,
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
    """The `(path, episode ids)` of every file in each auto import sent to Sonarr, one list per import."""

    return [[(f.path, f.episodeIds) for f in files] for files, _mode in sonarr.execute_calls]


class TestImportCompletedQueueState:
    """import_completed reads the queue + episode files before an auto import."""

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_importing_waits_with_the_clock_paused(self, attempt_kind: AttemptKind) -> None:
        # Sonarr is copying this download (importing): never double-import, and
        # the copy is Sonarr's wait, so the wait pass pauses the clock for it (a serial
        # torrent import must not expire the pending import at the deadline). Counts kept.
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

    def test_clean_import_pending_retries_until_forced(self) -> None:
        # A clean importPending: leave it to Sonarr unless forced.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[queue_record("ABC123", "importPending", status="ok")],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert sonarr.candidate_calls == []

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_unreachable_retries_instead_of_an_auto_import(self, attempt_kind: AttemptKind) -> None:
        # A failed queue read must never read as "not tracked" (an auto import would
        # race an import Sonarr may be running). Waiting with the clock running
        # fails closed on BOTH attempt kinds: the poll loop retries, and a queue that stays
        # unreachable burns the import deadline so the pending import finishes still not imported.
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

    def test_auto_import_post_is_a_paused_wait(self) -> None:
        # The accepted POST is pearlarr's copy in flight (ISSUED, paused once by the
        # wait pass), so a final auto import is not finished on the same poll. The
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
        # next pending import in the wait pass never retries it inside the window.
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

    def test_clean_import_pending_forced_runs_an_auto_import(self) -> None:
        # AttemptKind.FINAL (snapshot / final blocking poll): stop waiting, issue the
        # import. The copy is async, so this reads command_issued, NOT
        # all_done.
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

    def test_file_map_episode_holding_a_file_with_no_group_pearlarr_size_matched_is_protected(self) -> None:
        # The grab-time size match is carried on the pending import through to the
        # import: a file with no group the planner named reads TRUSTED, so the
        # torrent pearlarr grabbed for the REST of the season never copies over it.
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

    def test_a_pending_import_awaiting_cleanup_with_files_gone_demotes_through_the_reconciler(self) -> None:
        # The cleanup's demote reads the POST's ISSUED reason off the real attempt result:
        # a pending import awaiting cleanup whose files are gone re-imports and drops back to an
        # ordinary carryover (the fix's wiring, not a scripted attempt result).
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
        assert any("no longer has every file imported" in w for w in diagnostic_messages(recording, Severity.WARNING))

    def test_store_persisted_guard_row_protects_through_the_pending_import_snapshot(self) -> None:
        # THE fix path end to end: the persisted blob carries NO guards - the
        # title's guard_facts row alone must feed the loaded pending import's
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
        store.put_guards(Arr.SONARR, claim.anilist_id, claim.guards)
        mgr = make_import_wait_manager(
            qbit=_CompletedQbit("/d"),
            cache_store=store,
            strategy=strat,
        )

        mgr.report_imports_for_series(claim.series_id)

        # The row's facts reached the decision: the on-disk OtherPick file reads
        # TRUSTED - done, never auto imported over - and the
        # finished pending import leaves the store. The IMPORTED classification pins that
        # branch (a GONE drop would also empty the store).
        assert sonarr.candidate_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert mgr._state.pending_import_states[pending.infohash] is PendingImportState.IMPORTED


class TestCloseTracked:
    """close_tracked dismisses the download's remaining queue record through any one row's id."""

    def test_other_downloads_rows_are_not_touched(self) -> None:
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("FFFF", "importPending", queue_id=9)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_unknown_series_rows_are_left_for_sonarr(self) -> None:
        # Dismissing an unknown-series queue record 500s in Sonarr without recording
        # the ignore; the row drops once the torrent leaves the watched category.
        # SKIPPED, never FAILED: an unknown-series queue record can't strand the pending import.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", status="warning", queue_id=11, series_id=0)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_queue_unreachable_fails_the_effect_without_a_guessed_delete(self) -> None:
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

    def test_gone_queue_record_is_quiet_success(self) -> None:
        # Sonarr already dismissed the queue record (its refresh cleared it first): DONE
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


class TestImportCompletedPayload:
    """import_completed takes episodes from pearlarr's file map, never Sonarr's parse."""

    def test_payload_uses_pending_import_fields_not_candidate(self) -> None:
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

        # The command was issued. The copy is async, so the attempt result carries
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
        # The sample is never sent.
        assert result.all_done is False
        assert result.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

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


def _two_series_pending_import(files: FileEpisodeMap | None = None) -> PendingImport:
    """One torrent claimed on series 7 (window 101) and series 8 (window 201), its listing fully placed."""

    mapped = {_SHOW_FILE: (101,), _OTHER_FILE: (201,)} if files is None else files
    return two_claim_pending_import(
        series_ids=(7, 8),
        windows=((101,), (201,)),
        titles=("Show", "Other"),
        infohash="abc123",
        file_episode_map=mapped,
        seadex_files=list(mapped),
    )


def _posted_series(sonarr: FakeSonarrClient) -> dict[str, int]:
    """Each POSTed file's `seriesId` keyed by its path, across every execute."""

    return {entry.path: entry.seriesId for files, _ in sonarr.execute_calls for entry in files}


class TestMultiSeriesImport:
    """A two-series pending import posts each file under the series its claim names and is done as one."""

    @staticmethod
    def _candidates(*names: str) -> list[ManualImportCandidate]:
        return [manual_candidate(f"/d/{name}") for name in names]

    def test_a_file_no_claim_or_index_routes_waits_and_warns_at_the_deadline(self) -> None:
        # 301 sits in neither claim's window and in no series index: nothing is POSTed, the poll
        # waits, and only the deadline attempt names the file.
        recording = install_recording_hub()
        stray = "Stray - 01 [1080p].mkv"
        pending = _two_series_pending_import({stray: (301,)})
        strat, sonarr = _make_sonarr_for_import(candidates=self._candidates(stray))

        polled = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert polled == replace(AttemptResult.waiting(), file_map_episode_count=1, pending=pending)
        assert diagnostic_messages(recording, Severity.WARNING) == []

        deadline = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert deadline == replace(AttemptResult.waiting(), file_map_episode_count=1, pending=pending)
        assert sonarr.execute_calls == []
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "resolve to no series" in warning
        assert normalize_file_name(stray) in warning

    def test_imported_only_when_every_series_file_map_episodes_are_done(self) -> None:
        # Series 7's file map episode already holds the pending import's group while series 8's has no file: the
        # pending import is not done, and only the series-8 file is POSTed. Once that is imported, the next poll
        # finds it done.
        pending = _two_series_pending_import()
        sonarr = _PerSeriesSonarr(
            {
                7: [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")],
                8: [sonarr_ep(1, 1, ep_id=201, episode_file_id=0)],
            },
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )
        strat = make_sonarr_sync(sonarr=sonarr, config=make_config(), cache_store=FakeCacheStore())

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


_FIRST_FILE = "Show - 01 [1080p].mkv"
_SECOND_FILE = "Show - 02 [1080p].mkv"

_TWO_EPISODES_STORED = pending_import(
    file_episode_map={_FIRST_FILE: (101,), _SECOND_FILE: (102,)},
    seadex_files=[_FIRST_FILE, _SECOND_FILE],
    ordered_episode_ids=(101, 102),
    tvdb_ids={101: 9001, 102: 9002},
)
"""A two-file pending import of title 1 on series 7, stored while row 101 held TVDB episode 9001 and row 102 held
9002."""


def _moved_down_one() -> list[SonarrEpisode]:
    """Series 7 with an episode numbered ahead of the stored two: 9001 sits on row 102 and 9002 on a row 103.

    No row has a file.
    """

    return [sonarr_ep(1, number, ep_id=100 + number, tvdb_id=8999 + number, episode_file_id=0) for number in (1, 2, 3)]


def _both_files() -> list[ManualImportCandidate]:
    """Both of `_TWO_EPISODES_STORED`'s files in the download folder."""

    return [manual_candidate(f"/d/{_FIRST_FILE}"), manual_candidate(f"/d/{_SECOND_FILE}")]


class TestImportFollowsEpisodeIds:
    """An import acts on the pending import with each id on the row that holds its TVDB episode at that read."""

    def test_each_file_is_posted_on_the_row_that_holds_its_episode(self) -> None:
        strat, sonarr = _make_sonarr_for_import(candidates=_both_files(), episodes=_moved_down_one())

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[(f"/d/{_FIRST_FILE}", [102]), (f"/d/{_SECOND_FILE}", [103])]]
        assert result.followed is True

    def test_a_hand_import_running_on_a_row_followed_to_holds_the_import(self) -> None:
        by_hand = CommandResource.model_validate(
            {"id": 7000, "name": "ManualImport", "status": "started", "body": {"files": [{"episodeIds": [103]}]}}
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=_both_files(), episodes=_moved_down_one(), commands=[by_hand]
        )

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.IMPORT
        assert sonarr.execute_calls == []

    def test_a_later_attempt_on_the_pending_import_as_stored_does_not_warn_of_the_file_again(self) -> None:
        # An attempt that ends in an error saves nothing, so the next one follows the stored pending import again.
        recording = install_recording_hub()
        only_the_first = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, episode_file_id=0)]
        strat, _ = _make_sonarr_for_import(candidates=_both_files(), episodes=only_the_first)

        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)
        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        left_out = normalize_file_name(_SECOND_FILE)
        warnings = [message for message in diagnostic_messages(recording, Severity.WARNING) if left_out in message]
        assert len(warnings) == 1


def _import_history(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` record, as the import-evidence match reads it."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


class TestRadarrImportCompletedHistory:
    """Radarr reconciles a completed download off its OWN import history (evidence, never action).

    Radarr's completed-download handling does the import; `import_completed` only
    reads history as proof so the pending import can drop and the category gate release.
    """

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([], id="no-event-yet"),
            pytest.param([_import_history("h", event="grabbed")], id="non-import-event"),
            pytest.param(None, id="history-outage"),
        ],
    )
    def test_missing_evidence_keeps_the_pending_import_retrying(self, history: list[HistoryRecord] | None) -> None:
        # No import proof (none yet, a non-import event, or Radarr unreachable): fail
        # closed: waiting keeps the pending import waiting, never done on an unread history.
        radarr = FakeRadarrClient()
        radarr.history_since_return = history
        strat = make_radarr_sync(radarr=radarr)

        result = strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is False

    def test_history_fetched_once_per_wait_pass(self) -> None:
        # Two pending imports in one wait pass share ONE history fetch (memoized).
        radarr = FakeRadarrClient(history_since=[_import_history("h")])
        strat = make_radarr_sync(radarr=radarr)

        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="other"), "/d", AttemptKind.ORDINARY)

        assert len(radarr.history_calls) == 1

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
        # Radarr pending imports carry series_id=0, so the per-item snapshot hook key is
        # None (short-circuits the per-item snapshot - reconcile runs in the wait pass).
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())

        assert strat.pending_import_series_id(_Item(id=5)) is None

    def test_import_progress_is_indeterminate_zero(self) -> None:
        # Radarr never enters the blocking wait, so the progress poll bar hook
        # returns the safe "no bar, promote nothing" value.
        strat = make_bare_instance(RadarrStrategy, logger=make_logger())

        assert strat.import_progress(pending_import()) == ImportProgress(0, 0, determinate=False)


_GRAB_URL = "https://nyaa.si/1"
_GRAB_FILES = ("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv")
_EP1 = sonarr_ep(1, 1, ep_id=101, episode_file_id=0)
_EP2 = sonarr_ep(1, 2, ep_id=102, episode_file_id=0)
_STAMP = "2026-01-01 00:00:00"


def _sonarr_grab_run(*, store: FakeCacheStore, episodes: ScriptedEpisodes) -> _FakeRunServices:
    """Drive a blocking run's Sonarr `check_title` for title 5 (series 7) listing one two-file torrent.

    The series has mappings for titles 4 and 5 and for the ignored title 6. The plan grabs the torrent.
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
        plan_grabs_result=grab_plan(seadex),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=store,
    )
    parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(_GRAB_FILES, start=1)}
    strat = make_bare_instance(
        SonarrStrategy,
        _services=run,
        **scripted_roles(episodes, run.pending_imports),
        _parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in _GRAB_FILES)}),
        _listings=_ScriptedListings(),
        _config=make_config(sleep_time=0),
        logger=make_logger(),
    )
    strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))
    return run


class TestSonarrCheckTitlePendingImportUpdates:
    """A Sonarr grab places each listed torrent under the title's window.

    A stored torrent is placed as its pending import's leftover.
    """

    def test_a_stored_torrent_is_placed_as_its_pending_imports_leftover(self) -> None:
        # Cour 1's pending import already has file 01 in its file map. Cour 2 marks the same torrent to grab: only
        # file 02 is placed, under the stored claim's window and this title's, and the pending import update merges
        # onto the pending import.
        store = FakeCacheStore()
        resident = pending_import(
            infohash="h1",
            anilist_id=22,
            series_id=7,
            title="Show Cour 1",
            ordered_episode_ids=(101,),
            file_episode_map={_GRAB_FILES[0]: [101]},
            seadex_files=list(_GRAB_FILES),
        )
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        sonarr = FakeSonarrClient(episodes=[_EP1, _EP2])

        run = _sonarr_grab_run(store=store, episodes=ScriptedEpisodes(sonarr, windows={5: [_EP2]}))

        (req,) = run.grab_requests
        (update,) = req.pending_import_updates.values()
        assert update.pending_import == resident
        assert update.file_map == {normalize_file_name(_GRAB_FILES[1]): (102,)}
        assert update.claim.anilist_id == 5
        assert update.claim.ordered_episode_ids == (102,)
        pending = update.pending_import_at(_STAMP, restart=False)
        assert claim_anilist_ids(pending) == (22, 5)
        assert dict(pending.file_episode_map) == {
            normalize_file_name(name): (ep,) for name, ep in zip(_GRAB_FILES, (101, 102), strict=True)
        }
        # The planner reads the release by the whole file map's episodes inside this title's window.
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == [placed_episode(season=1, episode=2, size=1000)]
        assert req.input_unread_groups == ()
        # The stored claim's window came off the per-run whole-series read.
        assert sonarr.episodes_calls == [7]

    def test_a_specials_torrent_the_listing_contradicts_is_skipped(self) -> None:
        # The series' entries list the torrent over specials 2, 4 and 6, three as it is wide, and its `1` is none of
        # them: the torrent counts by another TVDB state. Nothing is placed for the planner, and the release is
        # skipped.
        specials = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)]
        files = tuple(f"Show.S00E{n:02d}.mkv" for n in range(1, 4))
        seadex: SeadexDict = {
            "NAN0": rg_group({_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", download=True, files=list(files))}),
        }
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            anilist_title="Show Specials",
            seadex_dict=seadex,
            plan_grabs_result=grab_plan(seadex),
            import_wait_mode=ImportWaitMode.BLOCKING,
            cache_store=FakeCacheStore(),
        )
        parses = {name: parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n, name in enumerate(files, 1)}
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            **scripted_roles(
                ScriptedEpisodes(FakeSonarrClient(episodes=specials), windows={5: [specials[1], specials[3]]}),
                run.pending_imports,
            ),
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


_MOVED_SPECIAL = sonarr_ep(0, 1, ep_id=150, tvdb_id=9001, episode_file_id=0)
"""TVDB episode 9001 of series 7 as a special on row 150, after it left the season's first episode."""

_SECOND_ROW = sonarr_ep(1, 2, ep_id=102, tvdb_id=9002, episode_file_id=0)
"""The season's second episode of series 7, on the row it was stored on."""


def _stored_on_the_first_row(*, tvdb_ids: Mapping[int, int] | None) -> PendingImport:
    """Torrent "h1" as title 22 stored it: file 01 on row 101 of series 7, held with `tvdb_ids`."""

    return pending_import(
        infohash="h1",
        anilist_id=22,
        series_id=7,
        title="Show Cour 1",
        ordered_episode_ids=(101,),
        file_episode_map={_GRAB_FILES[0]: [101]},
        seadex_files=list(_GRAB_FILES),
        tvdb_ids=tvdb_ids,
    )


def _store_holding(pending: PendingImport) -> FakeCacheStore:
    """A store holding `pending`, with titles 9, 22 and 23 cached."""

    store = FakeCacheStore()
    store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
    for anilist_id in (9, 22, 23):
        store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": _STAMP})
    return store


def _title_stamps(store: FakeCacheStore) -> dict[int, str | None]:
    """The cache stamp of each title `_store_holding` cached."""

    cached_titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (9, 22, 23)}
    return {anilist_id: cached.updated_at for anilist_id, cached in cached_titles.items() if cached is not None}


class TestSonarrCheckTitleFollowsStoredPendingImports:
    """A check follows each stored pending import of the entry's torrents before it places them."""

    def test_a_stored_pending_import_that_cannot_be_followed_is_not_joined_and_its_torrent_is_unread(self) -> None:
        stored = _stored_on_the_first_row(tvdb_ids=None)
        store = _store_holding(stored)
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=[_MOVED_SPECIAL, _SECOND_ROW]), windows={5: [_SECOND_ROW]}
        )

        run = _sonarr_grab_run(store=store, episodes=episodes)

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.PLACEMENT_INPUT_UNREAD
        assert req.input_unread_groups == ("NAN0",)
        assert req.pending_import_updates["h1"].pending_import is None
        assert store.get_pending_import_row(Arr.SONARR, "h1") == stored.to_json()
        assert _title_stamps(store) == {9: _STAMP, 22: _STAMP, 23: _STAMP}


_STAND_IN_E2 = stand_in_ep(1, 2, tvdb_id=9002)
"""The stand-in for a second episode that Sonarr's order of series 7 has no row for."""

_LISTED_EP1 = _EP1.renumbered(season=1, episode=1, absolute=None)
"""Series 7's first episode as its view holds it when the DVD order lists it under its aired number."""


def _check_title_in_another_order(window: list[SonarrEpisode]) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 over `window`, all of series 7 as aired, numbered by its DVD order.

    The entry lists one torrent of `_GRAB_FILES`, named as the season's first two episodes, which the plan grabs.
    """

    release = make_release(url=_GRAB_URL, infohash="h1", download=True, files=list(_GRAB_FILES))
    seadex: SeadexDict = {"NAN0": rg_group({_GRAB_URL: release})}
    run = _FakeRunServices(
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show",
        seadex_dict=seadex,
        plan_grabs_result=grab_plan(seadex),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=FakeCacheStore(),
    )
    episodes = ScriptedEpisodes(FakeSonarrClient(episodes=window), windows={5: window}, order="dvd")
    parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(_GRAB_FILES, start=1)}
    strat = make_bare_instance(
        SonarrStrategy,
        _services=run,
        **scripted_roles(episodes, run.pending_imports),
        _parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in _GRAB_FILES)}),
        _listings=_ScriptedListings(),
        _config=make_config(sleep_time=0),
        logger=make_logger(),
    )
    strat.check_title(_Item(id=7, title="Show", tvdbId=70, imdbId=None), 5, TitleMapping(anilist_id=5))
    return run


class TestSonarrCheckTitleOnASeriesInAnotherOrder:
    """A title of a series Sonarr numbers by another order is checked by its aired episodes.

    What Sonarr holds is read off the episodes it has a row for, and the title's window keeps every aired episode.
    """

    def test_a_cached_rows_coverage_backfill_is_of_the_episodes_sonarr_has(self) -> None:
        run = _check_title_in_another_order([_LISTED_EP1, _STAND_IN_E2])

        (coverage,) = run.cached_skip_coverages
        assert coverage() == "S01 E01"

    def test_a_grabs_files_on_episodes_sonarr_has_no_row_for_are_reported_in_one_row(self) -> None:
        run = _check_title_in_another_order([_LISTED_EP1, _STAND_IN_E2])

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is None
        assert run.left_out == ["S01 E02 (1 of 2 files, NAN0)"]

    @pytest.mark.parametrize(
        ("reason", "not_stored"),
        [
            pytest.param(NoView.ORDERS_UNREAD, "aired episodes", id="aired-episodes"),
            pytest.param(NoView.SCENE_UNREAD, "scene numbers from TheXEM", id="scene-numbers"),
        ],
    )
    def test_a_series_with_no_aired_view_yet_is_left_unchecked_with_a_warning_of_what_is_not_stored(
        self, reason: NoView, not_stored: str
    ) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            **scripted_roles(
                ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1]), order="dvd", no_views={7: reason}),
                run.pending_imports,
            ),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show"), 5, TitleMapping(anilist_id=5))

        (warning,) = diagnostic_messages(recording, Severity.WARNING)
        assert warning == (
            f"Show not checked - Sonarr numbers its series by the DVD order and the series' {not_stored} are not "
            "stored yet - checking again next run"
        )
        assert (run.no_episodes_titles, run.log_al_title_calls, run.pauses_owed) == ([], [], 1)
        assert (run.grab_requests, run.no_releases_calls, run.nothing_to_grab_calls) == ([], [], [])


class TestSonarrCheckTitleOnASeriesInAiredOrder:
    """A title of a series in aired order is left unchecked, quietly, when its series can't be read."""

    def test_a_series_whose_episodes_could_not_be_read_is_left_unchecked_quietly(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrStrategy,
            _services=run,
            **scripted_roles(ScriptedEpisodes(), run.pending_imports),
            _config=make_config(sleep_time=0),
            logger=make_logger(),
        )

        strat.check_title(_Item(id=7, title="Show"), 5, TitleMapping(anilist_id=5))

        assert diagnostic_messages(recording) == []
        assert (run.no_episodes_titles, run.log_al_title_calls, run.pauses_owed) == ([], [], 0)
        assert (run.grab_requests, run.no_releases_calls) == ([], [])


@dataclass(frozen=True, slots=True, kw_only=True)
class _SeriesInSonarr:
    """Series 7 as Sonarr has it, and how Sonarr parses each file of the one torrent SeaDex lists for it."""

    series: SonarrSeries
    episodes: list[SonarrEpisode]
    parses: Mapping[str, ParsedFileInfo]
    orders: Mapping[str, tuple[OrderRow, ...]] | None = None
    """The TVDB orders stored for the series, with TheXEM read as mapping none of its episodes. None when nothing
    is stored."""


def _series_seven(season_type: str, alternate_title: dict[str, object]) -> SonarrSeries:
    """Sonarr's row of series 7 as the wire sends it, numbered by `season_type`, with its one alternate title."""

    return SonarrSeries.model_validate(
        {"id": 7, "title": "Show", "tvdbId": 70, "seasonType": season_type, "alternateTitles": [alternate_title]}
    )


def _sequel_in_dvd_order() -> _SeriesInSonarr:
    """A sequel's four files numbered as a first season, on a series Sonarr numbers by its DVD order.

    TVDB's aired order has two seasons, of three and four episodes. The DVD order is one season of the first six, so
    Sonarr lists the aired S02E01 to S02E03 as S01E04 to S01E06 and has no row for the aired S02E04. It reads
    "Show Zoku S01E0n" by the second season's name and matches nothing. An aired Sonarr matches the second season's
    episodes.
    """

    aired = (
        *(OrderRow(1, n, 9000 + n, n) for n in (1, 2, 3)),
        *(OrderRow(2, n, 9003 + n, 3 + n) for n in (1, 2, 3, 4)),
    )
    dvd = tuple(OrderRow(1, n, 9000 + n, None) for n in range(1, 7))
    names = [f"Show Zoku S01E{n:02d}.mkv" for n in (1, 2, 3, 4)]
    return _SeriesInSonarr(
        series=_series_seven("dvd", {"title": "Show Zoku", "sceneSeasonNumber": 2}),
        episodes=[sonarr_ep(1, n, ep_id=10 + n, tvdb_id=9000 + n, episode_file_id=0) for n in range(1, 7)],
        parses={
            name: ParsedFileInfo(season_number=1, episode_numbers=(n,), series_title="Show Zoku", series_id=7)
            for n, name in enumerate(names, start=1)
        },
        orders={"official": aired, "dvd": dvd},
    )


def _check_second_season(
    monkeypatch: pytest.MonkeyPatch, library: _SeriesInSonarr, episode_by_size: Mapping[int, int] | None = None
) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for title 5, the second season of series 7, over the run's own episode reads.

    The library read, the parse cache and each parse's matched episodes are the real ones. The plan grabs the
    torrent, whose nth file is n thousand bytes, and AniList knows no title. `episode_by_size` is the episode id
    SeaDex lists a one-episode release of each size for.
    """

    names = list(library.parses)
    sizes = [1000 * n for n in range(1, len(names) + 1)]
    release = make_release(url=_GRAB_URL, infohash="h1", download=True, files=names, size=sizes)
    seadex: SeadexDict = {"NAN0": rg_group({_GRAB_URL: release})}
    cache = FakeCacheStore()
    if library.orders is not None:
        cache.put_tvdb_orders(library.series.tvdbId, stored_orders(library.orders))
        cache.put_scene_mappings(library.series.tvdbId, stored_scene_mappings())
    run = _FakeRunServices(
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show Second Season",
        seadex_dict=seadex,
        plan_grabs_result=grab_plan(seadex),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=cache,
    )
    sonarr = FakeSonarrClient(all_series=[library.series], episodes=library.episodes, parse_fn=library.parses.get)
    deps = replace(
        make_run_deps(cache_store=cache), anilist=make_anilist_gateway(ScriptedAniListClient(title=None), cache)
    )
    reads = SeriesReads(sonarr, deps.skyhook, deps.xem)
    episodes = SonarrEpisodes(deps, sonarr, run, reads)
    monkeypatch.setattr(episodes, "mapped_series", every_series)
    episodes.refresh_library()
    strat = make_bare_instance(
        SonarrStrategy,
        _services=run,
        _episodes=episodes,
        _reads=reads,
        _follower=PendingFollower(run.pending_imports, reads, make_logger()),
        _parse=SonarrParseCache(deps, sonarr, reads),
        _listings=_ScriptedListings(episode_by_size=episode_by_size),
        _config=make_config(sleep_time=0),
        logger=make_logger(),
    )
    strat.check_title(library.series, 5, TitleMapping(anilist_id=5, tvdb_season=2))
    return run


class TestSonarrCheckTitleOverSonarrsParses:
    """A title's check places a file by what the library read says of its series' other names."""

    def test_a_file_named_by_an_alternate_title_of_the_series_lands_by_its_number_in_the_season(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The group numbers the second season from one under the series' other name, and Sonarr matched the file
        # into the first season. The number is read within the season only because the run knows the name.
        name = "[grp] Shou - 01.mkv"
        library = _SeriesInSonarr(
            series=_series_seven(AIRED_ORDER, {"title": "Shou"}),
            episodes=[
                *(sonarr_ep(1, n, ep_id=10 + n, absolute=n, episode_file_id=0) for n in (1, 2, 3, 4)),
                *(sonarr_ep(2, n, ep_id=14 + n, absolute=4 + n, episode_file_id=0) for n in (1, 2, 3)),
            ],
            parses={
                name: ParsedFileInfo(
                    season_number=0,
                    absolute_episode_numbers=(1,),
                    series_title="Shou",
                    matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=11),),
                    series_id=7,
                )
            },
        )

        run = _check_second_season(monkeypatch, library)

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == {normalize_file_name(name): (15,)}


class TestSonarrCheckTitlePrintsSonarrsNumbers:
    """On a series Sonarr numbers by another order, a title's lines print each episode by the number Sonarr lists."""

    def test_the_titles_rows_and_stored_coverage_are_in_sonarrs_numbers_under_the_named_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _check_second_season(monkeypatch, _sequel_in_dvd_order())

        (req,) = run.grab_requests
        claim = req.pending_import_updates["h1"].claim
        assert (run.title_block_coverages, run.missing_episodes) == (["S01 E04-E06"], ["S01 E04-E06"])
        assert (req.update.get("coverage"), claim.window_coverage, claim.claimed_coverage) == ("S01 E04-E06",) * 3
        assert run.numbers_orders == ["dvd"]

    def test_a_file_placed_by_size_over_its_name_is_warned_of_by_sonarrs_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # SeaDex lists a one-episode release of the first file's size for the aired S02E02, which Sonarr lists as
        # S01E05, and one of the second file's size for the aired S02E01.
        recording = install_recording_hub()

        _check_second_season(monkeypatch, _sequel_in_dvd_order(), episode_by_size={1000: 15, 2000: 14})

        assert diagnostic_messages(recording, Severity.WARNING) == [
            (
                f"{_GRAB_URL}: show zoku s01e01.mkv goes on S01E05 because its size matches SeaDex's S01E05 release "
                "(its name reads as S01E01)"
                " (1 more file in the debug log)"
            )
        ]


class TestPlacementDebugLine:
    """The grab's debug line per release says what placed and why the release was skipped."""

    def test_a_release_skipped_on_a_failed_read_names_the_read_once(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = debug_logger("pearlarr-placement-line")
        strat = make_bare_instance(SonarrStrategy, logger=logger)
        series = episode_index([_EP1])
        grabbing_title = make_grabbing_title(series, series)
        placement = place_release(
            [ListedFile("Show - 01.mkv", 1000, None)], grabbing_title, new_torrent(grabbing_title)
        )

        with caplog.at_level("DEBUG"):
            strat._log_placements(PlacedEntry(grabbing_title, {_GRAB_URL: placement}), [series])

        assert placement.skip_reason is SkipReason.PLACEMENT_INPUT_UNREAD
        assert [r.message for r in caplog.records] == [f"{_GRAB_URL}: placed nothing (not grabbed: a read failed)"]

    def test_a_file_left_for_a_hand_import_is_named_with_why(self, caplog: pytest.LogCaptureFixture) -> None:
        # A Blu-ray torrent numbers 13 files for a 12-episode season: the DVD order splits the opener in two halves
        # Sonarr doesn't list, numbers the aired second episode on after them, and keeps the whole opener last.
        season = [sonarr_ep(4, n, ep_id=400 + n, episode_file_id=0, tvdb_id=400 + n) for n in range(1, 13)]
        dvd = (OrderRow(7, 1, 9001, None), OrderRow(7, 2, 9002, None))
        dvd += (*(OrderRow(7, n + 1, 400 + n, None) for n in range(2, 13)), OrderRow(7, 21, 401, None))
        series = episode_index(season, {"dvd": dvd})
        names = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
        grabbing_title = make_grabbing_title(series, series, names=TitleNames("Show", ("Show Ending",)))
        logger = debug_logger("pearlarr-placement-line-orders")
        strat = make_bare_instance(SonarrStrategy, logger=logger)
        placement = place_release(
            [ListedFile(name, 1000, parsed_info()) for name in names], grabbing_title, new_torrent(grabbing_title)
        )

        with caplog.at_level("DEBUG"):
            strat._log_placements(PlacedEntry(grabbing_title, {_GRAB_URL: placement}), [series])

        halves = ", ".join(
            f"{normalize_file_name(names[n - 1])} (another TVDB order, not in Sonarr's list: DVD S07E0{n})"
            for n in (1, 2)
        )
        assert [r.message for r in caplog.records] == [
            f"{_GRAB_URL}: placed S04 E02-E12; left for a hand import: {halves}"
        ]

    def test_on_a_series_in_another_order_every_placed_file_is_named_by_sonarrs_numbers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The same 13 files, and Sonarr numbers the series by the DVD order: it has a row for each half, where they
        # land, and lists the aired second to twelfth episodes as S07E03 to S07E13.
        aired = tuple(OrderRow(4, n, 400 + n, None) for n in range(1, 13))
        dvd = (OrderRow(7, 1, 9001, None), OrderRow(7, 2, 9002, None))
        dvd += (*(OrderRow(7, n + 1, 400 + n, None) for n in range(2, 13)), OrderRow(7, 21, 401, None))
        rows = [sonarr_ep(7, row.episode, ep_id=row.tvdb_id, episode_file_id=0, tvdb_id=row.tvdb_id) for row in dvd]
        view = aired_series(rows, SonarrSeriesOrders(70, "dvd", None), stored_orders({AIRED_ORDER: aired, "dvd": dvd}))
        assert isinstance(view, AiredSeries)
        series = episode_index(view.aired, {"dvd": dvd}, view.own_only)
        names = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
        grabbing_title = make_grabbing_title(series, series, names=TitleNames("Show", ("Show Ending",)))
        strat = make_bare_instance(SonarrStrategy, logger=debug_logger("pearlarr-placement-line-own-rows"))
        placement = place_release(
            [ListedFile(name, 1000, parsed_info()) for name in names], grabbing_title, new_torrent(grabbing_title)
        )

        with caplog.at_level("DEBUG"):
            strat._log_placements(PlacedEntry(grabbing_title, {_GRAB_URL: placement}), [series])

        on_rows = ", ".join(f"{normalize_file_name(names[n - 1])} (S07 E0{n})" for n in (1, 2))
        assert [r.message for r in caplog.records] == [
            f"{_GRAB_URL}: placed S07 E03-E13; on episodes the aired order lacks: {on_rows}"
        ]

    def test_a_file_on_an_episode_sonarr_has_no_row_for_is_named_with_its_aired_episode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        series = episode_index([_LISTED_EP1, _STAND_IN_E2])
        grabbing_title = make_grabbing_title(series, series)
        files = [ListedFile(name, 1000, parsed_info(season=1, episodes=(n,))) for n, name in enumerate(_GRAB_FILES, 1)]
        strat = make_bare_instance(SonarrStrategy, logger=debug_logger("pearlarr-placement-line-aired"))
        placement = place_release(files, grabbing_title, new_torrent(grabbing_title))

        with caplog.at_level("DEBUG"):
            strat._log_placements(PlacedEntry(grabbing_title, {_GRAB_URL: placement}), [series])

        (line,) = [r.message for r in caplog.records]
        assert line == (
            f"{_GRAB_URL}: placed S01 E01; set aside: {normalize_file_name(_GRAB_FILES[1])} "
            "(aired episode not in Sonarr's order: S01 E02)"
        )


class TestRadarrCheckTitlePendingImportUpdates:
    """A Radarr grab builds a `PendingImportUpdate` per release to grab, keyed by its hash in qBittorrent."""

    @staticmethod
    def _run_process(*, store: FakeCacheStore, cross_seed: str) -> _FakeRunServices:
        """Drive a blocking run's `check_title` to grab_and_cache and return the recording services.

        `cross_seed` marks the release as carried by that torrent.
        """

        item = make_release(url="https://nyaa.si/1", infohash="h1", download=True)
        item.cross_seed_hash = cross_seed
        seadex: SeadexDict = {"NAN0": rg_group({"https://nyaa.si/1": item})}
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/9"),
            anilist_title="A Movie",
            seadex_dict=seadex,
            plan_grabs_result=grab_plan(seadex),
            import_wait_mode=ImportWaitMode.BLOCKING,
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

    def test_a_release_to_grab_is_stored_on_its_cross_seeds_pending_import(self) -> None:
        store = FakeCacheStore()
        resident = pending_import(infohash="c1", anilist_id=9, series_id=0, title="Other Movie", file_episode_map={})
        store.put_pending_import_row(Arr.RADARR, "c1", resident.to_json())

        run = self._run_process(store=store, cross_seed="c1")

        (req,) = run.grab_requests
        update = req.pending_import_updates["c1"]
        assert update.pending_import == resident
        assert (update.facts.infohash, update.facts.is_cross_seed) == ("c1", True)


class TestAutoImportWarningGating:
    """An import warns of what is final and new, once per run. What is early or already reported is a debug line.

    An intended file absent on an early poll is expected (the copy hasn't finished), and a file the grab
    excluded was reported when it was set aside, so neither is warned of. A file that can't be placed is.
    """

    def test_missing_off_deadline_is_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # The file map intends a file that isn't on disk yet (only an unrelated file is).
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101]})
        strat, _ = _make_sonarr_for_import(candidates=[manual_candidate("/d/Unrelated.mkv")])
        # The absent-file line is the import executor's. caplog needs a propagating logger.
        strat._executor.logger = debug_logger("pearlarr-warning-gating")
        recording = install_recording_hub()

        with caplog.at_level("DEBUG"):
            result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert not any(
            "not visible to Sonarr" in message for message in diagnostic_messages(recording, Severity.WARNING)
        )
        assert any("not visible to Sonarr" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    @staticmethod
    def _strat_with_unplaceable(*names: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # The file map's file plus `names`, which the file map lacks, beside the empty file map episode.
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), *(manual_candidate(f"/d/{n}") for n in names)],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
        )
        # The excluded-file note is the import executor's debug line. caplog needs a propagating logger.
        strat._executor.logger = debug_logger("pearlarr-warning-gating")
        return strat, sonarr

    def test_grab_time_excluded_unplaceable_file_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # A sibling pending import's part of a shared torrent was placed at grab time:
        # this pending import still imports its own file, and the unplaceable sibling
        # file is a debug note, not a "could not be placed" warning.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling], excluded_files=[normalize_file_name(sibling)]
        )
        strat, sonarr = self._strat_with_unplaceable(sibling)
        recording = install_recording_hub()

        with caplog.at_level("DEBUG"):
            result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert not any("could not be placed" in m for m in diagnostic_messages(recording, Severity.WARNING))
        assert any(
            "1 excluded file already reported when set aside" in r.message and r.levelname == "DEBUG"
            for r in caplog.records
        )

    def test_partial_exclusion_warns_the_remainder_once(self) -> None:
        # One excluded plus one genuinely unplaceable file: the warning names
        # only the remainder, and the once-per-run memo is consumed by it.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling, "Extras.mkv"],
            excluded_files=[normalize_file_name(sibling)],
        )
        strat, _ = self._strat_with_unplaceable(sibling, "Extras.mkv")
        recording = install_recording_hub()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be placed" in m]
        assert len(warnings) == 1
        assert "1 file could not be placed" in warnings[0]


class TestUnmatchedAttemptResult:
    """A download whose every on-disk file the file map lacks is an outcome on its first poll, not a wait.

    The import places nothing, so the next poll would find the same files: no parse or episode read is waited on.
    """

    _NAMES = ("Movie Part 1.mkv", "Movie Part 2.mkv")

    @classmethod
    def _unplaced_pair(cls) -> PendingImport:
        # Two files the grab placed nowhere, for one window episode.
        return pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=list(cls._NAMES))

    @classmethod
    def _strat(cls, *, served: bool = True) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # The window episode holds no file. With `served` off the series has no episodes at all.
        return _make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in cls._NAMES],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)] if served else [],
        )

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

    def test_a_series_with_no_episodes_waits(self) -> None:
        pending = self._unplaced_pair()
        strat, _ = self._strat(served=False)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result == replace(AttemptResult.waiting(), pending=pending)


class TestLeftForHandImportFiles:
    """A file the grab left for a hand import keeps the pending import stored once every placed file is in."""

    # Each file's number is one episode early. The episode titles say where the titled two go. The plain file
    # belongs on 601.
    _NAMES = ("show - 01 - The Return [grp].mkv", "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv")
    _TITLES = ("Pilot Flight", "The Return", "Festival Night")
    _SIZE_ON: ClassVar[dict[int, int]] = {601: 1003, 602: 1001, 603: 1002}

    @classmethod
    def _record(cls) -> PendingImport:
        return pending_import(
            file_episode_map={normalize_file_name(cls._NAMES[0]): [602], normalize_file_name(cls._NAMES[1]): [603]},
            hand_import_files=(normalize_file_name(cls._NAMES[2]),),
            seadex_files=list(cls._NAMES),
            release_sizes=tuple(sorted(cls._SIZE_ON.values())),
            ordered_episode_ids=(601, 602, 603),
        )

    @classmethod
    def _episodes(cls, *with_files: int) -> list[SonarrEpisode]:
        """The window, each episode of `with_files` holding the pending import's file at its listed size."""

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

    def test_the_file_left_for_hand_import_keeps_the_pending_import_and_is_named_every_poll(self) -> None:
        strat, sonarr = self._strat()
        recording = install_recording_hub()

        first = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)
        second = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert first.left_for_hand_import.names == (self._NAMES[2],)
        assert first.all_done is False
        assert second == first
        assert sonarr.execute_calls == []
        assert not any("could not be placed" in m for m in diagnostic_messages(recording, Severity.WARNING))

    def test_a_hand_import_with_no_group_at_a_listed_size_finishes_the_pending_import(self) -> None:
        strat, sonarr = self._strat()
        no_group = sonarr_ep(1, 1, ep_id=601, title=self._TITLES[0], size=self._SIZE_ON[601], release_group=None)
        sonarr.episodes_return = [no_group, *self._episodes(602, 603)[1:]]

        done = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert done.all_done is True
        assert done.left_for_hand_import.names == ()


class TestOrderOnlyHalves:
    """A name-numbered torrent whose split opener's halves only the DVD order numbers, its other eleven files in."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
    _KEYS: ClassVar[tuple[OrderOnlyKey, ...]] = (
        OrderOnlyKey("dvd", EpisodeKey(7, 1)),
        OrderOnlyKey("dvd", EpisodeKey(7, 2)),
    )

    def test_the_attempt_result_keys_each_half_by_its_file_name_and_names_the_empty_opener(self) -> None:
        run = [normalize_file_name(name) for name in self._RUN]
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
    """An unmatched `imports.default_quality` warns once per run, in the executor that uses it.

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

        # Both polls ran the auto import (rebuilt the payload), yet the typo warned once.
        assert len(sonarr.execute_calls) == 2
        assert self._default_quality_warnings(recording) == [
            "imports.default_quality 'Blueray-1080p' matches no Sonarr quality definition - ignoring it",
        ]

        # The guard is per-run scratch: the arr run's start reset restarts it.
        strat._executor.reset()
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert len(self._default_quality_warnings(recording)) == 2

    def test_matched_name_does_not_warn(self) -> None:
        strat, sonarr, recording = self._strat_with_default("WEBDL-1080p")

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []


def _imported_history(*rows: tuple[int, str]) -> HistoryPage:
    """A closed-history page whose import rows carry `(episodeId, droppedPath)` for series 7, above the grab."""

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


class TestFolderFetchFallback:
    """The closed-history loop cure: a downloadId fetch that can't serve the download reads history and the folder.

    A download Sonarr's history maps to Imported/Failed/Ignored is queue-hidden
    and its `downloadId=` fetch never serves it (a 500 on v4, `[]` on v5). Without
    the fallback every poll would wait "for a later run" for as long as the
    torrent stayed in qBittorrent.
    """

    @staticmethod
    def _strat(
        *, history: HistoryPage, folder_candidates: list[ManualImportCandidate]
    ) -> tuple[SonarrStrategy, HookedSonarrClient]:
        # candidates=None: the downloadId fetch fails (the 500 shape) -> fallback.
        strat, sonarr = _make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = history
        sonarr.folder_candidates_return = folder_candidates
        sonarr.path_mappings_return = []
        return strat, sonarr

    @staticmethod
    def _answer_empty_by_id(
        sonarr: FakeSonarrClient,
        *,
        commands: tuple[CommandResource, ...] = (_RESCAN_AFTER_ADD,),
    ) -> None:
        """Script the downloadId fetch to answer `[]` under the given command list."""

        sonarr.candidates_return = []
        sonarr.commands_return = list(commands)

    @staticmethod
    def _poll_then_empty_the_folder(
        strat: SonarrStrategy, sonarr: FakeSonarrClient, pending: PendingImport
    ) -> AttemptResult:
        """Poll with the episode index unread, then again once the folder fetch is empty and only 101 holds a file."""

        sonarr.episodes_return = []
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.folder_candidates_return = []
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
        return strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

    def test_history_closed_verdict_memoized_and_noted_once_per_run(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        # The history-closed note is debug-level. caplog needs a propagating logger.
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

    def test_history_closed_empty_folder_warns_only_at_the_deadline(self, caplog: pytest.LogCaptureFixture) -> None:
        # An empty folder right after completion is expected (Sonarr's mount
        # lags), so ordinary polls note it at debug; the genuinely stuck shape
        # (a pending import with an empty file map, still empty at the deadline) warns once a run,
        # restarted by reset. `_closed_history()` carries no import rows, so the
        # rebuild yields nothing and the file map stays empty.
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

    def test_empty_download_id_fetch_waits_without_the_proof(self) -> None:
        strat, sonarr = self._strat(
            history=_closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr, commands=(_RESCAN_BEFORE_ADD,))

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    @pytest.mark.parametrize(
        ("by_id", "rebuilt"),
        [(None, {normalize_file_name("Show - 01 [1080p].mkv"): (101,)}), ([], {})],
        ids=["failed", "empty"],
    )
    def test_folder_pinned_pending_import_rebuilds_from_history_only_after_a_failed_download_id_fetch(
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

        assert result.pending is not None
        assert result.pending.file_episode_map == rebuilt


class TestUnwatchedTorrentImport:
    """A finished torrent in a category Sonarr doesn't watch imports from its folder.

    Sonarr never tracks such a torrent: its `downloadId=` fetch answers `[]` for good and its history holds
    nothing. Once the queue and a completed rescan show Sonarr isn't tracking it, the folder is read and the
    pending import's file map imports the files, with no downloadId and by copy.
    """

    @staticmethod
    def _strat(
        *,
        folder_candidates: list[ManualImportCandidate] | None = None,
    ) -> tuple[SonarrStrategy, HookedSonarrClient, QbitCategoryReads]:
        """Sonarr watching `tv-sonarr`, the pending import's torrent in `manual`, and Sonarr not tracking it.

        The downloadId fetch answers `[]`, the queue is empty, a rescan finished after the add, and the
        history holds only the neutral page.
        """

        qbit = QbitCategoryReads("manual")
        categories = make_fetched_categories(grab="tv-sonarr", post_import=None, qbit=qbit.client())
        strat, sonarr = _make_sonarr_for_import(candidates=[], categories=categories)
        sonarr.commands_return = [_RESCAN_AFTER_ADD]
        sonarr.history_page_return = HistoryPage.model_validate({"records": []})
        sonarr.folder_candidates_return = (
            [manual_candidate("/d/Show/Show - 01 [1080p].mkv")] if folder_candidates is None else folder_candidates
        )
        sonarr.path_mappings_return = []
        return strat, sonarr, qbit

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

    def test_an_empty_folder_warns_at_the_deadline_with_the_folder_reason(self) -> None:
        recording = install_recording_hub()
        strat, sonarr, _ = self._strat(folder_candidates=[])

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert sonarr.execute_calls == []
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "1 intended file not visible to Sonarr" in warning
        assert "reading /d/Show found no files" in warning


class _FakeRadarr:
    """Minimal Radarr client: scripts (and records) the per-movie movie-file list."""

    def __init__(self, files: list[MovieFile]) -> None:
        self._files = files
        self.movie_files_calls: list[int] = []

    def movie_files(self, movie_id: int) -> list[MovieFile]:
        self.movie_files_calls.append(movie_id)
        return self._files


def _one_group_dict(srg: str) -> SeadexDict:
    """A one-group `SeadexDict` keyed by `srg`, carrying a single release."""

    url = f"https://nyaa.si/{srg}"
    return {srg: rg_group({url: make_release(url=url)})}


class TestRadarrCheckTitle:
    """check_title's movie middle threads its facts through the services it keeps."""

    @staticmethod
    def _make_strat(run: _FakeRunServices, *, interactive: bool = False) -> tuple[RadarrStrategy, _FakeRadarr]:
        radarr = _FakeRadarr([])
        strat = make_bare_instance(
            RadarrStrategy,
            _services=run,
            radarr=radarr,
            logger=make_logger(),
            _config=make_config(interactive=interactive),
        )
        return strat, radarr

    def test_cached_title_short_circuits_before_any_movie_read(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), cached_skip=True)
        strat, radarr = self._make_strat(run)

        strat.check_title(_Item(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        # Neither the SeaDex read nor the movie files were reached.
        assert run.get_seadex_dict_calls == []
        assert radarr.movie_files_calls == []
        # The lazy backfill coverage passed is the movie's empty-coverage thunk.
        [coverage] = run.cached_skip_coverages
        assert coverage() == ""

    def test_no_releases_routes_to_shared_stop_with_movie_cache_details(self) -> None:
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

    def _interactive_call_count(self, *, interactive: bool, n_groups: int) -> int:
        """Run one check_title and count the interactive-filter invocations."""

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
    """radarr_files never hard-errors: a size it can't read counts as zero."""

    def test_unreadable_size_with_no_group_converts_to_zero(self) -> None:
        # An unreadable size with no group converts to 0, which disqualifies the whole
        # ownership multiset (fail-closed, like the planner's size matching of Sonarr files with no group): the
        # readable sizes alone must not let a release claim the copy.
        radarr = _FakeRadarr([MovieFile(release_group=None, size=500), MovieFile(release_group=None, size=None)])
        strat = make_bare_instance(RadarrStrategy, radarr=radarr)

        assert strat.radarr_files(7) == ArrFiles(no_group=(500, 0))


class TestSonarrFiles:
    """sonarr_files feeds the planner no sizes for files with no group."""

    def test_episode_files_with_no_group_never_reach_no_group(self) -> None:
        # PIN: an episode file with no group contributes NOTHING to the `ArrFiles` - it
        # travels on the title's episodes into the planner's size matching instead. The
        # planner merges both arrs' no-group sources, so a file fed through
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
    """The type's own contract: deep-frozen, hashable, one shared conversion."""

    def test_equal_arr_files_hash_equal_across_key_order(self) -> None:
        # dict eq ignores insertion order, so the hash must too.
        first = ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,))
        second = ArrFiles(by_group={"B": (2,), "A": (1,)}, no_group=(3,))

        assert first == second
        assert hash(first) == hash(second)

    def test_from_files_keeps_or_drops_no_group_by_arr(self) -> None:
        # One conversion for both arrs: only the no-group destination differs. The
        # unreadable size with a group is dropped. The group keeps its name.
        files = [
            MovieFile(release_group=None, size=700),
            MovieFile(release_group="RG", size=800),
            MovieFile(release_group="RG", size=None),
        ]

        assert ArrFiles.from_files(files, keep_no_group=True) == ArrFiles(by_group={"RG": (800,)}, no_group=(700,))
        assert ArrFiles.from_files(files, keep_no_group=False) == ArrFiles(by_group={"RG": (800,)})

    def test_groups_label_renders_no_group_and_empty(self) -> None:
        assert ArrFiles(by_group={"A": (1,)}, no_group=(2,)).groups_label() == "A, (none)"
        assert ArrFiles().groups_label() == "(no files)"

    def test_group_count_matches_the_rendered_entries(self) -> None:
        # Files with no group convert to the single `(none)` entry, so they count once.
        assert ArrFiles(by_group={"A": (1,), "B": (2,)}, no_group=(3,)).group_count() == 3
        assert ArrFiles().group_count() == 0
