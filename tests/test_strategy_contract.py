# pyright: strict
# pyright: reportPrivateUsage=false
# The placement debug line tests call the strategy's private `_log_placements`. Strict re-flags that and the repo
# disables reportPrivateUsage for tests.
"""The Sonarr and Radarr strategies: a title's check, its import, and the files each arr holds.

A title's check runs over `_FakeRunServices`, a real `ArrRunServices` subclass that records what the strategy hands
the run (the grab request, the rows it caches, the lines it prints). An import runs over a scripted Sonarr and is
judged by the commands posted to it and the attempt result.
"""

import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import ClassVar, NamedTuple, override

import pytest
from seadex import EntryRecord

from pearlarr.aired_view import AiredSeries, NoView, aired_series
from pearlarr.arr_activity import HISTORY_MAX_LOOKBACK_DAYS, format_history_date
from pearlarr.arr_http import DeleteOutcome
from pearlarr.arr_models import (
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
    RadarrMovie,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
)
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import AppConfig, Arr
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.grab_placement import ListedFile, PlacedEntry, place_release
from pearlarr.import_quality import resolve_language_objects
from pearlarr.mappings import (
    ExternalIds,
    MappingResolver,
    MappingSource,
    MappingSources,
    ResolvedMappings,
    TitleMapping,
)
from pearlarr.output import Accent, Severity, StyledValue, TitleDetail
from pearlarr.pending_follow import PendingFollower
from pearlarr.pending_import import (
    FileEpisodeMap,
    OwnedAtGrab,
    PendingImport,
    SizeMatchedEpisode,
    TitleNames,
    TvdbIds,
)
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.placement_types import FetchedListings, TorrentListing, episode_index
from pearlarr.planner import GrabPlan
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.releases import EpisodeKey, OrderOnlyKey, PrintedKey, SeadexDict, SkipReason
from pearlarr.series_reads import SeriesReads
from pearlarr.sonarr_episodes import SonarrEpisodes
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.stamps import parse_stamp, stamp_of
from pearlarr.tvdb_orders import AIRED_ORDER, OrderRow, SeriesNumbering, SonarrSeriesOrders
from pearlarr.wait_types import (
    NO_PROGRESS,
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
    FakeCacheStore,
    FakeClock,
    ScriptedAniListClient,
    claim_anilist_ids,
    grab_plan,
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
    numbered_names,
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
    title_claim,
    title_window,
    torrent_listing,
    two_claim_pending_import,
)
from .fakes import (
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
from .sonarr_import_builders import (
    RESCAN_AFTER_ADD,
    RESCAN_BEFORE_ADD,
    closed_history,
    inflight_manual_import,
    make_sonarr_for_import,
    started_process_pass,
)


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
        plan_marks: frozenset[str] = frozenset(),
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
        self._plan_marks = plan_marks
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
        # The planner marks the releases it grabs on the dict it was handed.
        for group_releases in seadex_dict.values():
            for url, release in group_releases.releases.items():
                if url in self._plan_marks:
                    release.plan_grab()
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
    """A scripted listings collaborator: per-hash listings and size matches, or SeaDex unreachable."""

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
    """Parse collaborator serving scripted files per release (no Sonarr round-trip), none for an unscripted release."""

    def __init__(self, files_by_url: dict[str, tuple[ListedFile, ...]] | None = None) -> None:
        self._files = files_by_url or {}

    def parsed_files(self, seadex_dict: SeadexDict) -> dict[str, tuple[ListedFile, ...]]:
        return {
            url: self._files.get(url, ()) for group_releases in seadex_dict.values() for url in group_releases.releases
        }


def _sonarr_checking(
    run: _FakeRunServices,
    episodes: ScriptedEpisodes | None = None,
    *,
    config: AppConfig | None = None,
    parse: _ScriptedParse | None = None,
    listings: _ScriptedListings | None = None,
) -> SonarrStrategy:
    """A `SonarrStrategy` over `run`, its episode reads `episodes`, built without its constructor.

    The parse and the listings are scripted (none by default), and the config waits no time between titles unless
    `config` says otherwise.
    """

    episodes = episodes if episodes is not None else ScriptedEpisodes()
    return make_bare_instance(
        SonarrStrategy,
        _services=run,
        **scripted_roles(episodes, run.pending_imports),
        _parse=parse or _ScriptedParse(),
        _listings=listings or _ScriptedListings(),
        _config=config or make_config(sleep_time=0),
        logger=make_logger(),
    )


def _radarr_checking(
    run: _FakeRunServices, radarr: FakeRadarrClient | None = None, *, interactive: bool = False
) -> RadarrStrategy:
    """A `RadarrStrategy` built by its constructor over `run` and `radarr`, in interactive mode if `interactive`."""

    deps = make_run_deps(config=make_config(interactive=interactive))
    return RadarrStrategy(deps, run, radarr if radarr is not None else FakeRadarrClient())


def _mapping_one_series(deps: ArrRunDeps, *, tvdb_id: int, anilist_id: int) -> ArrRunDeps:
    """`deps` whose mappings map TVDB series `tvdb_id` to AniList title `anilist_id`, and nothing else."""

    mapped = MappingSources(
        anime={"Show": {"anilist_id": anilist_id, "tvdb_id": tvdb_id}}, anidb=False, anibridge=False
    )
    return replace(
        deps, resolver=MappingResolver(refresh_days=1, ignore_anilist_ids=set(), web=deps.web, sources=mapped)
    )


def _sonarr_mapping_tvdb_50(sonarr: FakeSonarrClient, run: ArrRunServices | None = None) -> SonarrStrategy:
    """A `SonarrStrategy` built by its constructor over `sonarr`, where only TVDB series 50 maps, to AniList 500."""

    deps = _mapping_one_series(make_run_deps(config=make_config(sleep_time=0)), tvdb_id=50, anilist_id=500)
    return SonarrStrategy(deps, run or ArrRunServices(deps, Arr.SONARR), sonarr_client=sonarr)


class TestItemMappingsDelegates:
    """item_mappings resolves through the services it keeps, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        run = _FakeRunServices(anilist_ids={7: TitleMapping(anilist_id=7)})
        strat = _radarr_checking(run)

        result = strat.item_mappings(RadarrMovie(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: TitleMapping(anilist_id=7)}
        assert run.title_mappings_calls == [
            TitleMappingsCall(ExternalIds(tmdb_movie=42, imdb="tt7"), log_ignored=False)
        ]

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = _FakeRunServices()
        strat = _sonarr_checking(run)

        strat.item_mappings(SonarrSeries(tvdbId=99, imdbId="tt9"))

        assert run.title_mappings_calls == [TitleMappingsCall(ExternalIds(tvdb=99, imdb="tt9"), log_ignored=True)]


class TestFilterToSingle:
    """filter_to_single keeps the item with the arr's external id, and warns when there is none."""

    @pytest.mark.parametrize(
        ("wanted", "kept", "warnings"),
        [
            pytest.param(20, [SonarrSeries(tvdbId=20)], [], id="listed"),
            pytest.param(
                7, [], ["No anime series with TVDB ID 7 found in Sonarr - check the --tvdb-id value"], id="not-listed"
            ),
        ],
    )
    def test_sonarr_keeps_the_series_of_the_tvdb_id(
        self, wanted: int, kept: list[SonarrItem], warnings: list[str]
    ) -> None:
        recording = install_recording_hub()

        result = make_sonarr_sync().filter_to_single([SonarrSeries(tvdbId=10), SonarrSeries(tvdbId=20)], wanted)

        assert (result, diagnostic_messages(recording, Severity.WARNING)) == (kept, warnings)

    @pytest.mark.parametrize(
        ("wanted", "kept", "warnings"),
        [
            pytest.param(2, [RadarrMovie(tmdbId=2)], [], id="listed"),
            pytest.param(
                7, [], ["No anime movie with TMDB ID 7 found in Radarr - check the --tmdb-id value"], id="not-listed"
            ),
        ],
    )
    def test_radarr_keeps_the_movie_of_the_tmdb_id(
        self, wanted: int, kept: list[RadarrItem], warnings: list[str]
    ) -> None:
        recording = install_recording_hub()

        result = make_radarr_sync().filter_to_single([RadarrMovie(tmdbId=1), RadarrMovie(tmdbId=2)], wanted)

        assert (result, diagnostic_messages(recording, Severity.WARNING)) == (kept, warnings)


class TestArrRunStartHook:
    """get_items doubles as the arr run's start: it lists the mapped series, and the run reads them afresh."""

    def test_sonarr_lists_only_the_series_with_an_anilist_mapping(self) -> None:
        mapped = SonarrSeries(id=5, title="Show", tvdbId=50)
        strat = _sonarr_mapping_tvdb_50(FakeSonarrClient(all_series=[mapped, SonarrSeries(id=7, tvdbId=70)]))

        assert strat.get_items() == [mapped]

    def test_a_title_checked_in_the_next_run_reads_the_episodes_sonarr_holds_then(self) -> None:
        # The first run reads episode 1 with no file. Its file lands before the second run, whose check must not
        # go by the first run's rows.
        series = SonarrSeries(id=5, title="Show", tvdbId=50)
        sonarr = FakeSonarrClient(all_series=[series], episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)])
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        strat = _sonarr_mapping_tvdb_50(sonarr, run)

        strat.get_items()
        strat.check_title(series, 500, TitleMapping(anilist_id=500))
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]
        strat.get_items()
        strat.check_title(series, 500, TitleMapping(anilist_id=500))

        assert run.missing_episodes == ["S01 E01"]


class TestSonarrCheckTitleBeforeTheGrab:
    """A series' check stops before the grab when there is nothing to grab.

    No SeaDex entry, no episodes, only unmonitored episodes or no pick left stops the title. The window's missing
    episodes are shown.
    """

    def test_sonarr_no_seadex_entry_stops_at_begin_check(self) -> None:
        run = _FakeRunServices()
        strat = _sonarr_checking(run)

        strat.check_title(SonarrSeries(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.anilist_id_prologue_calls == [5]

    @pytest.mark.parametrize(
        ("mapping", "warned"),
        [
            pytest.param(TitleMapping(anilist_id=5), False, id="season-not-in-sonarr"),
            pytest.param(
                TitleMapping(anilist_id=5, tvdb_mappings={}, source=MappingSource.ANIBRIDGE),
                True,
                id="anibridge-no-usable-ranges",
            ),
        ],
    )
    def test_sonarr_no_episodes_resolved_stops_and_warns_only_of_an_empty_anibridge_mapping(
        self, mapping: TitleMapping, warned: bool
    ) -> None:
        # A mapping that resolves to no episode stops the title on a no-episodes row: never as unmonitored and
        # never on to the grab. AniBridge's empty {} tvdb mapping also warns of its cause. The warning keys off
        # the source, so the mapping carries source=ANIBRIDGE as a real one does.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        recording = install_recording_hub()
        strat = _sonarr_checking(run, ScriptedEpisodes(FakeSonarrClient(episodes=[])))

        strat.check_title(SonarrSeries(id=1, title="Title"), 5, mapping)
        assert run.no_episodes_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert (run.nothing_to_grab_calls, run.pauses_owed) == ([], 1)
        warnings = diagnostic_messages(recording, Severity.WARNING)
        assert any("AniBridge has no usable season ranges" in message for message in warnings) is warned

    def test_sonarr_all_unmonitored_episodes_stop_with_an_unmonitored_row(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        unmonitored = sonarr_ep(1, 1).model_copy(update={"monitored": False})
        strat = _sonarr_checking(
            run,
            ScriptedEpisodes(FakeSonarrClient(episodes=[unmonitored])),
            config=make_config(sleep_time=0, ignore_unmonitored=True),
        )

        strat.check_title(SonarrSeries(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.unmonitored_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert run.pauses_owed == 1

    def test_sonarr_reports_the_windows_missing_episodes(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        window = [sonarr_ep(1, 1), sonarr_ep(1, 2, episode_file_id=0)]
        strat = _sonarr_checking(run, ScriptedEpisodes(FakeSonarrClient(episodes=window)))

        strat.check_title(SonarrSeries(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.missing_episodes == ["S01 E02"]

    def test_sonarr_all_invalid_selection_routes_to_shared_stop(self) -> None:
        # The interactive pick rejected every token: the strategy must route to
        # report_invalid_selection (which persists nothing) and never reach the
        # grab tail - the Sonarr twin of the Radarr guard in test_seadex_dict.
        run = _FakeRunServices(
            prologue_entry=make_entry_record(),
            seadex_dict=_groups("GroupA", "GroupB"),
            interactive_result={},
        )
        strat = _sonarr_checking(
            run,
            ScriptedEpisodes(FakeSonarrClient(episodes=[sonarr_ep(1, 1)])),
            config=make_config(interactive=True, sleep_time=0),
        )

        strat.check_title(SonarrSeries(id=1, title="Title", tvdbId=10), 5, TitleMapping(anilist_id=5))
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


def _posted(sonarr: FakeSonarrClient) -> list[list[tuple[str, list[int]]]]:
    """The `(path, episode ids)` of every file in each auto import sent to Sonarr, one list per import."""

    return [[(f.path, f.episodeIds) for f in files] for files, _mode in sonarr.execute_calls]


_IMPORTING = [queue_record("ABC123", "importing")]
_CLEAN_PENDING = [queue_record("ABC123", "importPending", status="ok")]


class TestImportCompletedQueueState:
    """import_completed reads the queue + episode files before an auto import."""

    @pytest.mark.parametrize(
        ("queue", "commands", "attempt_kind", "outcome"),
        [
            # Sonarr is copying this download. That is Sonarr's wait, so the clock pauses, and nothing is posted
            # over it even at the deadline.
            pytest.param(_IMPORTING, [], AttemptKind.ORDINARY, (PauseReason.IMPORT, False, 0), id="importing"),
            pytest.param(_IMPORTING, [], AttemptKind.FINAL, (PauseReason.IMPORT, False, 0), id="importing-final"),
            # Sonarr's view lags qBittorrent's "complete": the pending import's own wait, on a running clock.
            pytest.param(
                [queue_record("ABC123", "downloading")],
                [],
                AttemptKind.ORDINARY,
                (PauseReason.NONE, False, 0),
                id="downloading",
            ),
            # A clean importPending is left to Sonarr until the final attempt, which posts the import itself.
            pytest.param(_CLEAN_PENDING, [], AttemptKind.ORDINARY, (PauseReason.NONE, False, 0), id="clean-pending"),
            pytest.param(
                _CLEAN_PENDING, [], AttemptKind.FINAL, (PauseReason.ISSUED, True, 1), id="clean-pending-final"
            ),
            # Sonarr imports torrents one at a time. While its own pass runs, this download may be next in line,
            # so the clock pauses and even the final attempt posts nothing past it.
            pytest.param(
                _CLEAN_PENDING,
                [started_process_pass()],
                AttemptKind.ORDINARY,
                (PauseReason.BUSY, False, 0),
                id="clean-pending-behind-a-running-pass",
            ),
            pytest.param(
                _CLEAN_PENDING,
                [started_process_pass()],
                AttemptKind.FINAL,
                (PauseReason.BUSY, False, 0),
                id="clean-pending-behind-a-running-pass-final",
            ),
            pytest.param(
                [], [started_process_pass()], AttemptKind.FINAL, (PauseReason.BUSY, False, 0), id="untracked-final"
            ),
            # A warning on the importPending row is an import Sonarr tried and won't reliably retry.
            pytest.param(
                [queue_record("ABC123", "importPending", status="warning")],
                [],
                AttemptKind.ORDINARY,
                (PauseReason.ISSUED, True, 1),
                id="pending-with-a-warning",
            ),
            # A ManualImport of this download is already running, as after a restart: no second one is stacked on
            # it, and the wait reads as an import in flight. A finished one holds nothing back.
            pytest.param(
                [],
                [inflight_manual_import("ABC123")],
                AttemptKind.FINAL,
                (PauseReason.IMPORT, True, 0),
                id="its-own-import-running-final",
            ),
            pytest.param(
                [],
                [inflight_manual_import("abc123", status="completed")],
                AttemptKind.ORDINARY,
                (PauseReason.ISSUED, True, 1),
                id="its-own-import-completed",
            ),
            pytest.param(
                [],
                [inflight_manual_import("OTHERHASH", status="queued")],
                AttemptKind.ORDINARY,
                (PauseReason.BUSY, False, 0),
                id="another-downloads-import-queued",
            ),
        ],
    )
    def test_the_queue_state_and_running_commands_decide_between_a_wait_and_an_auto_import(
        self,
        queue: list[QueueRecord],
        commands: list[CommandResource],
        attempt_kind: AttemptKind,
        outcome: tuple[PauseReason, bool, int],
    ) -> None:
        # `outcome` is the pause the attempt asks for, whether an import command for this download is in flight,
        # and how many imports were posted.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")], queue=queue, commands=commands
        )

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert (result.imported_count, result.file_map_episode_count) == (0, 1)
        assert (result.pause_reason, result.command_issued, len(sonarr.execute_calls)) == outcome

    @pytest.mark.parametrize("attempt_kind", [AttemptKind.ORDINARY, AttemptKind.FINAL])
    def test_queue_unreachable_retries_instead_of_an_auto_import(self, attempt_kind: AttemptKind) -> None:
        # A failed queue read must never read as "not tracked" (an auto import would
        # race an import Sonarr may be running). Waiting with the clock running
        # fails closed on BOTH attempt kinds: the poll loop retries, and a queue that stays
        # unreachable burns the import deadline so the pending import finishes still not imported.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
        )
        sonarr.queue_return = None

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert result.paused is False
        assert sonarr.execute_calls == []

    def test_auto_import_post_is_a_paused_wait(self) -> None:
        # The accepted POST is pearlarr's copy in flight (ISSUED, paused once by the
        # wait pass), so a final auto import is not finished on the same poll. The
        # next poll finds the command started under its issued id (IMPORT).
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(
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

    def test_an_episode_holding_this_torrents_file_for_another_episode_is_imported_over(self) -> None:
        # This torrent's E02 file (size 200) is filed on E01 as well. Both rows hold its group at a listed size,
        # but E01's is not its own file, so E01 is still owed and its own file is posted onto it.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={_FIRST_FILE: [101], _SECOND_FILE: [102]},
            seadex_files=[_FIRST_FILE, _SECOND_FILE],
            release_sizes=[100, 200],
            sizes_by_name={_FIRST_FILE.casefold(): 100, _SECOND_FILE.casefold(): 200},
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=_both_files(),
            episodes=[
                sonarr_ep(1, n, ep_id=100 + n, episode_file_id=12, release_group="SubGroup", size=200) for n in (1, 2)
            ],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[(f"/d/{_FIRST_FILE}", [101])]]
        assert (result.imported_count, result.file_map_episode_count) == (1, 2)

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
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group=None, size=700)],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert sonarr.execute_calls == []

    def test_a_pending_import_awaiting_cleanup_with_files_gone_demotes_through_the_reconciler(self) -> None:
        # The cleanup's demote reads the POST's ISSUED reason off the real attempt result:
        # a pending import awaiting cleanup whose files are gone re-imports and drops back to an
        # ordinary carryover (the fix's wiring, not a scripted attempt result).
        recording = install_recording_hub()
        pending = pending_import(infohash="abc123", awaiting_cleanup=True)
        store = FakeCacheStore()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cache_store=store,
        )
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_import_wait_manager(qbit=_CompletedQbit("/d"), cache_store=store, strategy=strat, state=state)

        mgr.finish_cleanups()

        assert len(sonarr.execute_calls) == 1
        row = store.get_pending_import_row(Arr.SONARR, pending.infohash)
        assert row is not None
        assert row["awaiting_cleanup"] is False
        assert state.pending_import_states[pending.infohash] is PendingImportState.DOWNLOADED
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
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="OtherPick")],
            cache_store=store,
        )
        (claim,) = pending.claims
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        store.put_guards(Arr.SONARR, claim.anilist_id, claim.guards)
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_import_wait_manager(qbit=_CompletedQbit("/d"), cache_store=store, strategy=strat, state=state)

        mgr.report_imports_for_series(claim.series_id)

        # The row's facts reached the decision: the on-disk OtherPick file reads
        # TRUSTED - done, never auto imported over - and the
        # finished pending import leaves the store. The IMPORTED classification pins that
        # branch (a GONE drop would also empty the store).
        assert sonarr.execute_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert state.pending_import_states[pending.infohash] is PendingImportState.IMPORTED


class TestCloseTracked:
    """close_tracked dismisses the download's remaining queue record through any one row's id."""

    def test_dismisses_by_the_first_usable_row_id(self) -> None:
        # Sonarr closes the whole tracked download off any one queue id, so the first row with a usable id is
        # deleted, exactly once.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=queue_id) for queue_id in (0, 11, 12)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]

    def test_other_downloads_rows_are_not_touched(self) -> None:
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", status="warning", queue_id=11, series_id=0)],
        )

        assert strat.remove_from_queue(pending) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_queue_unreachable_fails_the_effect_without_a_guessed_delete(self) -> None:
        # A failed read already retried inside the transport: FAILED (kept for a
        # later run), never an in-run RETRY and never a guessed delete.
        pending = pending_import(infohash="abc123")
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.queue_return = None

        assert strat.remove_from_queue(pending) is EffectStatus.FAILED
        assert sonarr.queue_delete_calls == []

    def test_successful_close_notes_the_removal(self) -> None:
        recording = install_recording_hub()
        strat, _ = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )

        status = strat.remove_from_queue(pending_import(infohash="abc123", title="Show"))

        assert status is EffectStatus.DONE
        assert any("from Sonarr's queue" in m for m in diagnostic_messages(recording, Severity.INFO))

    def test_failed_delete_asks_for_a_retry_and_stays_quiet(self) -> None:
        # The client's fail-open warning already fired; no success note on top.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(
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

    def test_the_posted_file_carries_the_pending_imports_series_episodes_group_and_hash(self) -> None:
        # Sonarr's candidate for the file names no series or episodes of its own: they come from the pending import.
        pending = pending_import(
            series_id=7,
            release_group="SubGroup",
            infohash="HASH",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate("/downloads/Show - 01 [1080p].mkv")])

        result = strat.import_completed(pending, "/downloads/Show", AttemptKind.ORDINARY)

        # The copy runs after the post, so the attempt has issued the command and is not done yet.
        assert (result.all_done, result.command_issued) == (False, True)
        [(files, _mode)] = sonarr.execute_calls
        assert [(f.path, f.seriesId, f.episodeIds, f.releaseGroup, f.downloadId) for f in files] == [
            ("/downloads/Show - 01 [1080p].mkv", 7, [101], "SubGroup", "HASH")
        ]

    @pytest.mark.parametrize(
        ("sonarr_parsed", "posted"),
        [
            pytest.param(
                {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
                "Bluray-1080p",
                id="sonarrs-parse-wins-over-the-file-name",
            ),
            pytest.param(
                {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0},
                "WEBDL-1080p",
                id="the-file-name-fills-an-unknown",
            ),
        ],
    )
    def test_the_posted_quality_is_sonarrs_parse_else_the_file_names(
        self, sonarr_parsed: dict[str, object], posted: str
    ) -> None:
        name = "Show - 01 [1080p][WEB-DL].mkv"
        pending = pending_import(file_episode_map={name: [101]}, seadex_files=[name])
        qualities = (
            {"id": 3, "name": "WEBDL-1080p", "source": "web"},
            {"id": 7, "name": "Bluray-1080p", "source": "bluray"},
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}", quality={"quality": sonarr_parsed})],
            quality_defs=[QualityDefinition.model_validate({"quality": {**q, "resolution": 1080}}) for q in qualities],
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        quality = sonarr.execute_calls[0][0][0].quality
        assert quality is not None
        assert quality.quality is not None
        assert quality.quality.name == posted

    def test_a_decomposed_disk_name_still_matches_its_composed_file_map_name(self) -> None:
        # The file map holds the composed e-acute. A macOS mount hands the name back decomposed.
        name = "Café - 01 [1080p].mkv"
        composed, decomposed = unicodedata.normalize("NFC", name), unicodedata.normalize("NFD", name)
        pending = pending_import(file_episode_map={composed: [101]}, seadex_files=[composed])
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate(f"/d/{decomposed}")])

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[(f"/d/{decomposed}", [101])]]

    def test_a_failed_candidate_fetch_waits_and_calls_no_file_missing(self) -> None:
        # A fetch that failed says nothing of the folder, where an empty answer says the file isn't there.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(candidates=None)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.FINAL)

        assert (result.all_done, sonarr.execute_calls) == (False, [])
        assert diagnostic_messages(recording, Severity.WARNING) == []

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
        strat, sonarr = make_sonarr_for_import(candidates=[good, sample])

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
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(candidates=[candidate], cmd_id=None)

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


def _two_series_pending_import(
    files: FileEpisodeMap | None = None, *, other_window: tuple[int, ...] = (201,)
) -> PendingImport:
    """One torrent claimed on series 7 (window 101) and series 8 (`other_window`), its listing fully placed."""

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
    """A two-series pending import posts each file under the series its claim names and is done as one."""

    @staticmethod
    def _candidates(*names: str) -> list[ManualImportCandidate]:
        return [manual_candidate(f"/d/{name}") for name in names]

    @pytest.mark.parametrize(
        ("pending", "series_8_row"),
        [
            # A claim's window names its file's id, whatever the indexes hold (here no index holds 201).
            pytest.param(_two_series_pending_import(), 202, id="the-claims-window-names-the-id"),
            # The series-8 claim has no window, so the index holding 201 says where its file goes.
            pytest.param(_two_series_pending_import(other_window=()), 201, id="no-window-so-the-index-holding-it"),
        ],
    )
    def test_each_file_posts_under_its_own_claims_series(self, pending: PendingImport, series_8_row: int) -> None:
        sonarr = _PerSeriesSonarr(
            {
                7: [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
                8: [sonarr_ep(1, 1, ep_id=series_8_row, episode_file_id=0)],
            },
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )
        strat = make_sonarr_sync(sonarr=sonarr, config=make_config(), cache_store=FakeCacheStore())

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": 7, f"/d/{_OTHER_FILE}": 8}

    def test_a_file_no_claim_or_index_routes_waits_and_warns_at_the_deadline(self) -> None:
        # 301 sits in neither claim's window and in no series index: nothing is POSTed, the poll
        # waits, and only the deadline attempt names the file.
        recording = install_recording_hub()
        stray = "Stray - 01 [1080p].mkv"
        pending = _two_series_pending_import({stray: (301,)})
        strat, sonarr = make_sonarr_for_import(candidates=self._candidates(stray))

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

    def test_two_claims_on_one_series_each_judge_their_own_episode_by_their_own_guards(self) -> None:
        # A's plan marked group G outdated and B's releases carry G. G files sit on both windows' episodes, so
        # A's episode still needs importing while B's is done. The series' merged guards would trust G for both.
        pending = pending_import(
            file_episode_map={_SHOW_FILE: [101], _OTHER_FILE: [102]},
            claims=(
                title_claim(anilist_id=1, ordered_episode_ids=(101,), guards=OwnedAtGrab(stale_groups=("G",))),
                title_claim(anilist_id=2, ordered_episode_ids=(102,), guards=OwnedAtGrab(entry_groups=("G",))),
            ),
        )
        on_disk = [sonarr_ep(1, n, ep_id=100 + n, release_group="G") for n in (1, 2)]
        strat, _ = make_sonarr_for_import(candidates=[], episodes=on_disk)

        assert strat.import_progress(pending) == ImportProgress(1, 2, determinate=True)


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
        strat, sonarr = make_sonarr_for_import(candidates=_both_files(), episodes=_moved_down_one())

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert _posted(sonarr) == [[(f"/d/{_FIRST_FILE}", [102]), (f"/d/{_SECOND_FILE}", [103])]]
        assert result.followed is True

    def test_the_counts_and_the_result_are_on_the_rows_followed_to_and_the_store_is_left_alone(self) -> None:
        store = FakeCacheStore()
        store.put_pending_import_row(Arr.SONARR, "abc123", _TWO_EPISODES_STORED.to_json())
        rows = [*_moved_down_one()[:2], sonarr_ep(1, 3, ep_id=103, tvdb_id=9002, release_group="SubGroup")]
        strat, _ = make_sonarr_for_import(candidates=_both_files(), episodes=rows, cache_store=store)

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert (result.imported_count, result.file_map_episode_count) == (1, 2)
        assert strat.import_progress(_TWO_EPISODES_STORED) == ImportProgress(1, 2, determinate=True)
        assert result.pending is not None
        assert result.pending.file_episode_map == {
            normalize_file_name(_FIRST_FILE): (102,),
            normalize_file_name(_SECOND_FILE): (103,),
        }
        assert result.pending.tvdb_ids == TvdbIds({102: 9001, 103: 9002})
        assert store.get_pending_import_row(Arr.SONARR, "abc123") == _TWO_EPISODES_STORED.to_json()

    def test_a_pending_import_imported_by_the_ids_it_stores_ends_imported_as_stored(self) -> None:
        # Both files went in on the stored rows before the episodes moved down. Following would post the second
        # file again, onto row 103.
        rows = [
            sonarr_ep(1, 1, ep_id=101, tvdb_id=9000, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, tvdb_id=9001, release_group="SubGroup"),
            sonarr_ep(1, 3, ep_id=103, tvdb_id=9002, episode_file_id=0),
        ]
        strat, sonarr = make_sonarr_for_import(candidates=_both_files(), episodes=rows)

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert (result.all_done, result.followed, result.pending) == (True, False, _TWO_EPISODES_STORED)
        assert sonarr.execute_calls == []

    def test_a_progress_poll_never_counts_a_pending_import_the_follow_took_a_file_from(self) -> None:
        only_the_first = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, release_group="SubGroup")]
        strat, _ = make_sonarr_for_import(candidates=[], episodes=only_the_first)

        assert strat.import_progress(_TWO_EPISODES_STORED) == NO_PROGRESS

    def test_a_file_sonarrs_history_placed_is_stored_with_the_tvdb_episode_its_row_has_at_that_poll(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = _imported_history((101, f"/d/Show/{_FIRST_FILE}"))
        sonarr.folder_candidates_return = []
        sonarr.path_mappings_return = []
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, release_group="SubGroup")]
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[_FIRST_FILE])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.pending is not None
        assert result.pending.file_episode_map == {normalize_file_name(_FIRST_FILE): (101,)}
        assert result.pending.tvdb_ids == TvdbIds({101: 9001})

    def test_a_hand_import_running_on_a_row_followed_to_holds_the_import(self) -> None:
        by_hand = CommandResource.model_validate(
            {"id": 7000, "name": "ManualImport", "status": "started", "body": {"files": [{"episodeIds": [103]}]}}
        )
        strat, sonarr = make_sonarr_for_import(candidates=_both_files(), episodes=_moved_down_one(), commands=[by_hand])

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.IMPORT
        assert sonarr.execute_calls == []

    def test_a_later_attempt_on_the_pending_import_as_stored_does_not_warn_of_the_file_again(self) -> None:
        # An attempt that ends in an error saves nothing, so the next one follows the stored pending import again.
        recording = install_recording_hub()
        only_the_first = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, episode_file_id=0)]
        strat, _ = make_sonarr_for_import(candidates=_both_files(), episodes=only_the_first)

        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)
        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        left_out = normalize_file_name(_SECOND_FILE)
        warnings = [message for message in diagnostic_messages(recording, Severity.WARNING) if left_out in message]
        assert len(warnings) == 1


class TestImportOnOwnOnlyRows:
    """A file placed on an own-only row is imported like any other file of the pending import."""

    def test_files_on_such_rows_are_posted_on_them_and_the_pending_import_ends_once_they_are_in(self) -> None:
        # The aired opener 401 is two halves, rows 9001 and 9002, in the order Sonarr numbers the series by. The
        # first two files sit on the halves, outside the title's window, and the whole opener's row never gets one.
        names = [f"Show - {n:02d} [1080p].mkv" for n in (1, 2, 3)]
        pending = pending_import(
            file_episode_map=dict(zip(names, [(9001,), (9002,), (402,)], strict=True)),
            seadex_files=names,
            ordered_episode_ids=(401, 402),
            tvdb_ids={row: row for row in (9001, 9002, 401, 402)},
        )
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate(f"/d/{name}") for name in names])

        def poll(*held: int) -> AttemptResult:
            # One poll with a file of the pending import's group on each row in `held`.
            sonarr.episodes_return = [
                sonarr_ep(1, n, ep_id=row, tvdb_id=row, episode_file_id=int(row in held), release_group="SubGroup")
                for n, row in ((1, 9001), (2, 9002), (3, 402), (21, 401))
            ]
            return strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        first = poll(402)
        second = poll(9001, 9002, 402)

        assert _posted(sonarr) == [[(f"/d/{names[0]}", [9001]), (f"/d/{names[1]}", [9002])]]
        assert (first.all_done, first.imported_count, first.file_map_episode_count) == (False, 1, 3)
        assert second.all_done is True


def _import_history(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` record, as the import-evidence match reads it."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


class TestRadarrImportCompletedHistory:
    """Radarr reconciles a completed download off its OWN import history (evidence, never action).

    Radarr's completed-download handling does the import; `import_completed` only
    reads history as proof so the pending import can drop and the category gate release.
    """

    @pytest.mark.parametrize(
        ("history", "done"),
        [
            pytest.param([_import_history("h")], True, id="import-event"),
            # Radarr stores the download id uppercased.
            pytest.param([_import_history("H")], True, id="import-event-id-uppercased"),
            # No import proof (none yet, a non-import event, or Radarr unreachable): fail
            # closed: waiting keeps the pending import waiting, never done on an unread history.
            pytest.param([], False, id="no-event-yet"),
            pytest.param([_import_history("h", event="grabbed")], False, id="non-import-event"),
            pytest.param(None, False, id="history-outage"),
        ],
    )
    def test_only_an_import_event_for_the_download_ends_the_pending_import(
        self, history: list[HistoryRecord] | None, done: bool
    ) -> None:
        radarr = FakeRadarrClient()
        radarr.history_since_return = history
        strat = make_radarr_sync(radarr=radarr)

        result = strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is done

    def test_history_fetched_once_per_wait_pass(self) -> None:
        # Two pending imports in one wait pass share ONE history fetch (memoized).
        radarr = FakeRadarrClient(history_since=[_import_history("h")])
        strat = make_radarr_sync(radarr=radarr)

        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="other"), "/d", AttemptKind.ORDINARY)

        assert len(radarr.history_calls) == 1

    def test_the_next_arr_run_reads_the_history_again(self) -> None:
        # The import landed between two runs. A history held over from the first would never show it.
        radarr = FakeRadarrClient()
        strat = make_radarr_sync(radarr=radarr)
        pending = pending_import(infohash="h")

        earlier = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        radarr.history_since_return = [_import_history("h")]
        strat.get_items()
        later = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (earlier.all_done, later.all_done) == (False, True)

    def test_history_is_asked_for_no_further_back_than_the_cap(self) -> None:
        store, radarr = FakeCacheStore(), FakeRadarrClient()
        long_ago = stamp_of(datetime.now() - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS + 10))
        pending = pending_import(infohash="h", added_at=long_ago)
        store.put_pending_import_row(Arr.RADARR, "h", pending.to_json())
        strat = make_radarr_sync(radarr=radarr, cache_store=store)
        # Two hours under the cap, for a Radarr clock a little behind pearlarr's.
        back = timedelta(days=HISTORY_MAX_LOOKBACK_DAYS, hours=2)

        earliest = format_history_date(datetime.now(UTC) - back)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        latest = format_history_date(datetime.now(UTC) - back)

        assert [earliest <= since <= latest for since in radarr.history_calls] == [True]

    def test_history_window_starts_at_the_oldest_stamp_inside_the_cap(self) -> None:
        older = stamp_of(datetime.now() - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS - 10))
        newer = stamp_of(datetime.now() - timedelta(days=1))
        store, radarr = FakeCacheStore(), FakeRadarrClient()
        latest = pending_import(infohash="new", added_at=newer)
        store.put_pending_import_row(Arr.RADARR, "new", latest.to_json())
        store.put_pending_import_row(Arr.RADARR, "old", pending_import(infohash="old", added_at=older).to_json())
        strat = make_radarr_sync(radarr=radarr, cache_store=store)

        strat.import_completed(latest, "/d", AttemptKind.ORDINARY)

        # Two hours before the stamp, for a Radarr clock a little behind pearlarr's.
        start = parse_stamp(older).astimezone(UTC) - timedelta(hours=2)
        assert radarr.history_calls == [format_history_date(start)]

    def test_pending_import_series_id_is_none(self) -> None:
        # Radarr pending imports carry series_id=0, so the per-item snapshot hook key is
        # None (short-circuits the per-item snapshot - reconcile runs in the wait pass).
        strat = make_radarr_sync()

        assert strat.pending_import_series_id(RadarrMovie(id=5)) is None

    def test_import_progress_is_indeterminate_zero(self) -> None:
        # Radarr never enters the blocking wait, so the progress poll bar hook
        # returns the safe "no bar, promote nothing" value.
        strat = make_radarr_sync()

        assert strat.import_progress(pending_import()) == ImportProgress(0, 0, determinate=False)


_GRAB_URL = "https://nyaa.si/1"
_GRAB_FILES = ("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv")
_EP1 = sonarr_ep(1, 1, ep_id=101, episode_file_id=0)
_EP2 = sonarr_ep(1, 2, ep_id=102, episode_file_id=0)
_STAMP = "2026-01-01 00:00:00"
_SERIES_7 = SonarrSeries(id=7, title="Show", tvdbId=70)
"""Series 7 as Sonarr lists it."""


def _sonarr_grab_run(
    *,
    store: FakeCacheStore,
    episodes: ScriptedEpisodes,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    grab: bool = True,
    files: tuple[ListedFile, ...] | None = None,
    listings: _ScriptedListings | None = None,
) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 (series 7) listing one torrent, in a blocking run unless `mode` says.

    The series has mappings for titles 4 and 5 and for the ignored title 6. The torrent is `files`, or `_GRAB_FILES`
    named as the season's first two episodes. The plan grabs it unless `grab` is off.
    """

    if files is None:
        parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(_GRAB_FILES, start=1)}
        files = tuple(ListedFile(name, 1000, parses[name]) for name in _GRAB_FILES)
    release = make_release(url=_GRAB_URL, infohash="h1", files=[f.file_name for f in files])
    seadex: SeadexDict = {"NAN0": rg_group({_GRAB_URL: release})}
    run = _FakeRunServices(
        anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
        ignored={6: TitleMapping(anilist_id=6)},
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show Cour 2",
        seadex_dict=seadex,
        plan_marks=frozenset({_GRAB_URL}) if grab else frozenset(),
        import_wait_mode=mode,
        cache_store=store,
    )
    strat = _sonarr_checking(run, episodes, parse=_ScriptedParse({_GRAB_URL: files}), listings=listings)
    strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))
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
            normalize_file_name(_GRAB_FILES[0]): (101,),
            normalize_file_name(_GRAB_FILES[1]): (102,),
        }
        # The planner reads the release by the whole file map's episodes inside this title's window.
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == [placed_episode(season=1, episode=2, size=1000)]
        assert req.input_unread_groups == ()

    def test_a_specials_torrent_the_listing_contradicts_is_skipped(self) -> None:
        # The series' entries list the torrent over specials 2, 4 and 6, three as it is wide, and its `1` is none of
        # them: the torrent counts by another TVDB state. Nothing is placed for the planner, and the release is
        # skipped.
        specials = [sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)]
        files = tuple(f"Show.S00E{n:02d}.mkv" for n in range(1, 4))
        seadex: SeadexDict = {
            "NAN0": rg_group({_GRAB_URL: make_release(url=_GRAB_URL, infohash="h1", files=list(files))}),
        }
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            anilist_title="Show Specials",
            seadex_dict=seadex,
            plan_marks=frozenset({_GRAB_URL}),
            import_wait_mode=ImportWaitMode.BLOCKING,
            cache_store=FakeCacheStore(),
        )
        parses = {name: parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n, name in enumerate(files, 1)}
        strat = _sonarr_checking(
            run,
            ScriptedEpisodes(FakeSonarrClient(episodes=specials), windows={5: [specials[1], specials[3]]}),
            parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in files)}),
            listings=_ScriptedListings({"h1": torrent_listing(title_window(1, (502, 504, 506)))}),
        )

        strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.MISNUMBERED
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == []
        assert req.pending_import_updates["h1"].file_map == {}
        assert req.input_unread_groups == ()

    @pytest.mark.parametrize(
        ("episode_count", "placed"),
        [
            pytest.param(2, {"ova - 01 [grp].mkv": (501,), "ova - 02 [grp].mkv": (504,)}, id="the-windows-two"),
            pytest.param(3, {}, id="an-episode-sonarr-lacks"),
        ],
    )
    def test_the_titles_anilist_episode_count_reaches_its_placement(
        self, episode_count: int, placed: dict[str, tuple[int, ...]]
    ) -> None:
        # Two specials numbered apart take the sequence only when they are the title's every episode.
        specials = [
            sonarr_ep(0, 1, ep_id=501, episode_file_id=0, air_date="2016-03-01"),
            sonarr_ep(0, 4, ep_id=504, episode_file_id=0, air_date="2016-06-01"),
        ]
        names = [*numbered_names("ova", 2), *numbered_names("short", 3)]
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=specials), windows={5: specials}, counts={5: episode_count}
        )

        run = _sonarr_grab_run(
            store=FakeCacheStore(),
            episodes=episodes,
            files=tuple(ListedFile(name, 1000, parsed_info()) for name in names),
        )

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == placed

    def test_an_unread_listing_places_nothing_and_skips_the_release(self) -> None:
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1, _EP2]), windows={5: [_EP2]})

        run = _sonarr_grab_run(store=FakeCacheStore(), episodes=episodes, listings=_ScriptedListings(unreachable=True))

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is SkipReason.PLACEMENT_INPUT_UNREAD
        assert req.pending_import_updates["h1"].file_map == {}
        assert req.input_unread_groups == ("NAN0",)

    def test_the_listing_read_gets_the_ignored_titles_too(self) -> None:
        listings = _ScriptedListings()
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1, _EP2]), windows={5: [_EP2]})

        _sonarr_grab_run(store=FakeCacheStore(), episodes=episodes, listings=listings)

        kept = {4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)}
        assert listings.reads == [(7, ResolvedMappings(kept=kept, ignored={6: TitleMapping(anilist_id=6)}))]

    def test_with_the_wait_off_a_stored_torrent_is_placed_and_no_pending_import_update_is_built(self) -> None:
        # A run that waits on no import reads no stored pending import. This one was stored without TVDB ids, so a
        # check that read it could not follow it and would leave its torrent unread.
        store = FakeCacheStore()
        resident = pending_import(
            infohash="h1",
            series_id=7,
            file_episode_map={_GRAB_FILES[0]: [101]},
            seadex_files=list(_GRAB_FILES),
            tvdb_ids=None,
        )
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1, _EP2]), windows={5: [_EP2]})

        run = _sonarr_grab_run(store=store, episodes=episodes, mode=ImportWaitMode.OFF)

        (req,) = run.grab_requests
        assert req.pending_import_updates == {}
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes == [placed_episode(season=1, episode=2, size=1000)]
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

    def test_a_stored_pending_import_is_followed_before_its_torrent_is_placed(self) -> None:
        store = _store_holding(_stored_on_the_first_row(tvdb_ids={101: 9001}))
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=[_MOVED_SPECIAL, _SECOND_ROW]), windows={5: [_SECOND_ROW]}
        )

        run = _sonarr_grab_run(store=store, episodes=episodes)

        (req,) = run.grab_requests
        pending = req.pending_import_updates["h1"].pending_import_at(_STAMP, restart=False)
        assert dict(pending.file_episode_map) == {
            normalize_file_name(_GRAB_FILES[0]): (150,),
            normalize_file_name(_GRAB_FILES[1]): (102,),
        }
        assert [claim.ordered_episode_ids for claim in pending.claims] == [(150,), (102,)]

    @pytest.mark.parametrize(
        ("stored", "rows", "stamps"),
        [
            pytest.param(
                _stored_on_the_first_row(tvdb_ids={101: 9001}),
                (150,),
                {9: _STAMP, 22: None, 23: _STAMP},
                id="one-series",
            ),
            pytest.param(
                two_claim_pending_import(
                    anilist_ids=(22, 23),
                    series_ids=(7, 8),
                    windows=((101,), (201,)),
                    infohash="h1",
                    file_episode_map={_GRAB_FILES[0]: (101,), _GRAB_FILES[1]: (201,)},
                    seadex_files=list(_GRAB_FILES),
                    tvdb_ids={101: 9001, 201: 9101},
                ),
                (150, 250),
                {9: _STAMP, 22: None, 23: None},
                id="two-series",
            ),
        ],
    )
    def test_a_followed_pending_import_is_saved_when_nothing_is_grabbed_and_its_titles_are_uncached(
        self, stored: PendingImport, rows: tuple[int, ...], stamps: dict[int, str | None]
    ) -> None:
        store = _store_holding(stored)
        sonarr = _PerSeriesSonarr(
            {7: [_MOVED_SPECIAL, _SECOND_ROW], 8: [sonarr_ep(1, 2, ep_id=250, tvdb_id=9101, episode_file_id=0)]},
            candidates=[],
        )

        run = _sonarr_grab_run(store=store, episodes=ScriptedEpisodes(sonarr, windows={5: [_SECOND_ROW]}), grab=False)

        (req,) = run.grab_requests
        row = store.get_pending_import_row(Arr.SONARR, "h1")
        assert req.pending_import_updates == {}
        assert row is not None
        assert row["file_episode_map"] == {
            normalize_file_name(name): [ep] for name, ep in zip(_GRAB_FILES, rows, strict=False)
        }
        assert _title_stamps(store) == stamps

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


def _check_title_in_another_order(
    window: list[SonarrEpisode], *, grab: bool = True, ignore_unmonitored: bool = False
) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 over `window`, all of series 7 as aired, numbered by its DVD order.

    The entry lists one torrent of `_GRAB_FILES`, named as the season's first two episodes, which the plan grabs
    unless `grab` is off.
    """

    release = make_release(url=_GRAB_URL, infohash="h1", files=list(_GRAB_FILES))
    seadex: SeadexDict = {"NAN0": rg_group({_GRAB_URL: release})}
    run = _FakeRunServices(
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show",
        seadex_dict=seadex,
        plan_marks=frozenset({_GRAB_URL}) if grab else frozenset(),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=FakeCacheStore(),
    )
    episodes = ScriptedEpisodes(FakeSonarrClient(episodes=window), windows={5: window}, order="dvd")
    parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(_GRAB_FILES, start=1)}
    strat = _sonarr_checking(
        run,
        episodes,
        config=make_config(sleep_time=0, ignore_unmonitored=ignore_unmonitored),
        parse=_ScriptedParse({_GRAB_URL: tuple(ListedFile(name, 1000, parses[name]) for name in _GRAB_FILES)}),
    )
    strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))
    return run


class TestSonarrCheckTitleOnASeriesInAnotherOrder:
    """A title of a series Sonarr numbers by another order is checked by its aired episodes.

    What Sonarr holds is read off the episodes it has a row for, and the title's window keeps every aired episode.
    """

    def test_a_title_whose_every_episode_in_sonarr_is_unmonitored_stops_as_unmonitored(self) -> None:
        unmonitored = _LISTED_EP1.model_copy(update={"monitored": False})

        run = _check_title_in_another_order([unmonitored, _STAND_IN_E2], ignore_unmonitored=True)

        assert (run.unmonitored_titles, run.log_al_title_calls) == (["Show"], [])

    def test_a_cached_rows_coverage_backfill_is_of_the_episodes_sonarr_has(self) -> None:
        run = _check_title_in_another_order([_LISTED_EP1, _STAND_IN_E2])

        (coverage,) = run.cached_skip_coverages
        assert coverage() == "S01 E01"

    def test_the_plan_is_made_over_the_episodes_sonarr_has(self) -> None:
        run = _check_title_in_another_order([_LISTED_EP1, _STAND_IN_E2])

        assert run.planned_title_episodes == [(_LISTED_EP1,)]

    @pytest.mark.parametrize(
        ("window", "grab", "skip_reason", "left_out"),
        [
            pytest.param(
                [_LISTED_EP1, _STAND_IN_E2], True, None, ["S01 E02 (1 of 2 files, NAN0)"], id="one-of-its-files"
            ),
            pytest.param(
                [stand_in_ep(1, 1, tvdb_id=9001), _STAND_IN_E2, sonarr_ep(1, 3, ep_id=103, episode_file_id=0)],
                True,
                SkipReason.AIRED_ONLY,
                ["S01 E01-E02 (2 of 2 files, NAN0)"],
                id="its-every-file",
            ),
            pytest.param([_LISTED_EP1, _STAND_IN_E2], False, None, [], id="a-release-the-plan-does-not-grab"),
        ],
    )
    def test_a_grabs_files_on_episodes_sonarr_has_no_row_for_are_reported_in_one_row(
        self, window: list[SonarrEpisode], grab: bool, skip_reason: SkipReason | None, left_out: list[str]
    ) -> None:
        run = _check_title_in_another_order(window, grab=grab)

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[_GRAB_URL].skip_reason is skip_reason
        assert run.left_out == left_out

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
        strat = _sonarr_checking(
            run, ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1]), order="dvd", no_views={7: reason})
        )

        strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))

        (warning,) = diagnostic_messages(recording, Severity.WARNING)
        assert warning == (
            f"Show not checked - Sonarr numbers its series by the DVD order and the series' {not_stored} are not "
            "stored yet - checking again next run"
        )
        assert (run.no_episodes_titles, run.log_al_title_calls, run.pauses_owed) == ([], [], 1)
        assert (run.grab_requests, run.no_releases_calls, run.nothing_to_grab_calls) == ([], [], [])


class TestSonarrCheckTitleOnASeriesInAiredOrder:
    """A title of a series in aired order says nothing of its numbers, and nothing when its series can't be read."""

    def test_the_title_block_says_nothing_of_its_numbers(self) -> None:
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[_EP1, _EP2]), windows={5: [_EP1, _EP2]})

        run = _sonarr_grab_run(store=FakeCacheStore(), episodes=episodes)

        assert (run.numbers_orders, run.left_out) == ([], [])

    def test_a_series_whose_episodes_could_not_be_read_is_left_unchecked_quietly(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        recording = install_recording_hub()
        strat = _sonarr_checking(run, ScriptedEpisodes())

        strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))

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
    library: _SeriesInSonarr, episode_by_size: Mapping[int, int] | None = None
) -> _FakeRunServices:
    """Drive a Sonarr `check_title` for title 5, the second season of series 7, over the run's own episode reads.

    The library read, the parse cache and each parse's matched episodes are the real ones. The plan grabs the
    torrent, whose nth file is n thousand bytes, and AniList knows no title. `episode_by_size` is the episode id
    SeaDex lists a one-episode release of each size for.
    """

    names = list(library.parses)
    sizes = [1000 * n for n in range(1, len(names) + 1)]
    release = make_release(url=_GRAB_URL, infohash="h1", files=names, size=sizes)
    seadex: SeadexDict = {"NAN0": rg_group({_GRAB_URL: release})}
    cache = FakeCacheStore()
    if library.orders is not None:
        cache.put_tvdb_orders(library.series.tvdbId, stored_orders(library.orders))
        cache.put_scene_mappings(library.series.tvdbId, stored_scene_mappings())
    run = _FakeRunServices(
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show Second Season",
        seadex_dict=seadex,
        plan_marks=frozenset({_GRAB_URL}),
        import_wait_mode=ImportWaitMode.BLOCKING,
        cache_store=cache,
    )
    sonarr = FakeSonarrClient(all_series=[library.series], episodes=library.episodes, parse_fn=library.parses.get)
    deps = replace(
        make_run_deps(cache_store=cache), anilist=make_anilist_gateway(ScriptedAniListClient(title=None), cache)
    )
    deps = _mapping_one_series(deps, tvdb_id=library.series.tvdbId, anilist_id=5)
    reads = SeriesReads(sonarr, deps.skyhook, deps.xem)
    episodes = SonarrEpisodes(deps, sonarr, run, reads)
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
    """A title's check places a file by what the library read says of its series: the order and the other names."""

    def test_a_sequels_files_numbered_as_a_first_season_land_on_the_aired_second_season(self) -> None:
        library = _sequel_in_dvd_order()
        first, second, third, fourth = (normalize_file_name(name) for name in library.parses)

        run = _check_second_season(library)

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert update.file_map == {first: (14,), second: (15,), third: (16,)}
        assert update.excluded == (fourth,)
        assert run.left_out == ["S02 E04 (1 of 4 files, NAN0)"]
        assert [(ep.aired_key, ep.printed) for ep in req.seadex_dict["NAN0"].releases[_GRAB_URL].episodes] == [
            (EpisodeKey(2, n), PrintedKey(EpisodeKey(1, 3 + n))) for n in (1, 2, 3)
        ]

    def test_a_file_named_by_an_alternate_title_of_the_series_lands_by_its_number_in_the_season(self) -> None:
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

        run = _check_second_season(library)

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == {normalize_file_name(name): (15,)}


class TestSonarrCheckTitlePrintsSonarrsNumbers:
    """On a series Sonarr numbers by another order, a title's lines print each episode by the number Sonarr lists."""

    def test_the_titles_rows_and_stored_coverage_are_in_sonarrs_numbers_under_the_named_order(self) -> None:
        run = _check_second_season(_sequel_in_dvd_order())

        (req,) = run.grab_requests
        claim = req.pending_import_updates["h1"].claim
        assert (run.title_block_coverages, run.missing_episodes) == (["S01 E04-E06"], ["S01 E04-E06"])
        assert (req.update.get("coverage"), claim.window_coverage, claim.claimed_coverage) == ("S01 E04-E06",) * 3
        assert run.numbers_orders == ["dvd"]

    def test_a_file_placed_by_size_over_its_name_is_warned_of_by_sonarrs_number(self) -> None:
        # SeaDex lists a one-episode release of the first file's size for the aired S02E02, which Sonarr lists as
        # S01E05, and one of the second file's size for the aired S02E01.
        recording = install_recording_hub()

        _check_second_season(_sequel_in_dvd_order(), episode_by_size={1000: 15, 2000: 14})

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


class TestSonarrCheckTitleMetadataUnread:
    """A grab asks whether its check read a title without its AniList answer or a series without its TVDB orders.

    The titles are every title of the title's series, the series are the title's and every stored pending import's.
    A title that stops before a grab asks about itself alone.
    """

    @pytest.mark.parametrize(
        ("unread_series", "unread_anilist_ids", "expected"),
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
        self, unread_series: frozenset[int], unread_anilist_ids: frozenset[int], expected: bool
    ) -> None:
        # Series 8's pending import already holds the torrent title 5 lists, so the grab places it under that pending
        # import too.
        store = FakeCacheStore()
        resident = pending_import(infohash="h1", anilist_id=22, series_id=8, seadex_files=list(_GRAB_FILES))
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=[_EP1, _EP2]),
            windows={5: [_EP2]},
            unread_series=unread_series,
            unread_anilist_ids=unread_anilist_ids,
        )

        run = _sonarr_grab_run(store=store, episodes=episodes)

        (req,) = run.grab_requests
        assert req.metadata_unread is expected

    @pytest.mark.parametrize(
        ("window", "order", "stopped_on"),
        [
            # No release to grab caches the title as having none. No episode in Sonarr's order caches it as
            # having nothing to grab.
            pytest.param([_EP2], AIRED_ORDER, ([5], list[int]()), id="no-suitable-release"),
            pytest.param([_STAND_IN_E2], "dvd", (list[int](), [5]), id="no-episode-in-sonarrs-order"),
        ],
    )
    def test_a_title_that_stops_before_a_grab_is_not_unread_for_another_entry_of_its_series(
        self, window: list[SonarrEpisode], order: str, stopped_on: tuple[list[int], list[int]]
    ) -> None:
        run = _FakeRunServices(
            anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
            prologue_entry=make_entry_record(url="https://releases.moe/5"),
            anilist_title="Show Cour 2",
            seadex_dict={},
            cache_store=FakeCacheStore(),
        )
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=window), windows={5: window}, order=order, unread_anilist_ids=frozenset({4})
        )
        strat = _sonarr_checking(run, episodes)

        strat.check_title(_SERIES_7, 5, TitleMapping(anilist_id=5))

        no_releases = [anilist_id for anilist_id, _ in run.no_releases_calls]
        assert (no_releases, [anilist_id for anilist_id, _ in run.nothing_to_grab_calls]) == stopped_on
        assert (run.no_releases_unread, run.nothing_to_grab_unread) == ([], [])


class TestRadarrCheckTitlePendingImportUpdates:
    """A Radarr grab builds a `PendingImportUpdate` per release to grab, keyed by its hash in qBittorrent.

    A run that waits on no import builds none.
    """

    @staticmethod
    def _run_process(
        *, store: FakeCacheStore, cross_seed: str | None = None, mode: ImportWaitMode = ImportWaitMode.BLOCKING
    ) -> _FakeRunServices:
        """Drive `check_title` to grab_and_cache and return the recording services, in a blocking run unless `mode` says.

        `cross_seed` marks the release as carried by that torrent.
        """

        item = make_release(url="https://nyaa.si/1", infohash="h1")
        item.cross_seed_hash = cross_seed
        seadex: SeadexDict = {"NAN0": rg_group({"https://nyaa.si/1": item})}
        run = _FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/9"),
            anilist_title="A Movie",
            seadex_dict=seadex,
            plan_marks=frozenset({"https://nyaa.si/1"}),
            import_wait_mode=mode,
            arr=Arr.RADARR,
            cache_store=store,
        )
        strat = _radarr_checking(run)
        strat.check_title(RadarrMovie(id=5, title="Movie"), 42, TitleMapping(anilist_id=42))
        return run

    def test_a_release_to_grab_starts_a_pending_import_claimed_by_the_movies_title(self) -> None:
        run = self._run_process(store=FakeCacheStore())

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert (update.claim.anilist_id, update.claim.series_id) == (42, 0)
        assert (update.claim.title, update.claim.url) == ("A Movie", "https://releases.moe/9")
        assert (update.pending_import, update.file_map) == (None, {})

    def test_no_pending_import_updates_when_wait_mode_off(self) -> None:
        run = self._run_process(store=FakeCacheStore(), mode=ImportWaitMode.OFF)

        (req,) = run.grab_requests
        assert req.pending_import_updates == {}

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
    """An import warns of what is final and new, once per run. What is early or already reported is not warned of.

    An intended file absent on an early poll is expected (the copy hasn't finished), and a file the grab
    excluded was reported when it was set aside, so neither is warned of. A file that can't be placed is.
    """

    @pytest.mark.parametrize(
        ("attempt_kind", "warned"),
        [
            pytest.param(AttemptKind.ORDINARY, False, id="before-the-deadline"),
            pytest.param(AttemptKind.FINAL, True, id="at-the-deadline"),
        ],
    )
    def test_a_file_not_on_disk_is_warned_of_only_at_the_deadline(
        self, attempt_kind: AttemptKind, warned: bool
    ) -> None:
        # The file map intends a file that isn't on disk yet (only an unrelated file is). Early on the copy may
        # just not have finished.
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101]})
        strat, _ = make_sonarr_for_import(candidates=[manual_candidate("/d/Unrelated.mkv")])
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        warnings = diagnostic_messages(recording, Severity.WARNING)
        assert any("not visible to Sonarr" in message for message in warnings) is warned

    @staticmethod
    def _strat_with_unplaceable(*names: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # The file map's file plus `names`, which the file map lacks, beside the empty file map episode.
        return make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), *(manual_candidate(f"/d/{n}") for n in names)],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
        )

    def test_grab_time_excluded_unplaceable_file_never_warns(self) -> None:
        # A sibling pending import's part of a shared torrent was placed at grab time:
        # this pending import still imports its own file, and the unplaceable sibling
        # file was reported when the grab set it aside, so it is not a "could not be placed" warning.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling], excluded_files=[normalize_file_name(sibling)]
        )
        strat, sonarr = self._strat_with_unplaceable(sibling)
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert not any("could not be placed" in m for m in diagnostic_messages(recording, Severity.WARNING))

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
    def _unplaced_pair(cls, *, excluded: tuple[str, ...] = ()) -> PendingImport:
        # Two files the grab placed nowhere, for one window episode.
        return pending_import(
            file_episode_map={},
            ordered_episode_ids=[101],
            seadex_files=list(cls._NAMES),
            excluded_files=[normalize_file_name(name) for name in excluded],
        )

    @classmethod
    def _strat(cls, *, served: bool = True, file_group: str | None = None) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # `file_group` names the group whose file the window episode already holds (None: no file). With `served`
        # off the series has no episodes at all.
        episode = sonarr_ep(1, 1, ep_id=101, release_group=file_group, episode_file_id=1 if file_group else 0)
        return make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in cls._NAMES],
            episodes=[episode] if served else [],
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

    @pytest.mark.parametrize(
        ("excluded", "file_group", "unplaced"),
        [
            pytest.param(_NAMES, None, _NAMES, id="every-file-excluded-at-the-grab"),
            pytest.param(_NAMES[1:], None, _NAMES[:1], id="one-file-excluded-at-the-grab"),
            pytest.param((), "OtherPick", _NAMES, id="another-groups-copy-on-the-window-episode"),
        ],
    )
    def test_the_unmatched_names_are_the_files_still_to_import_by_hand(
        self, excluded: tuple[str, ...], file_group: str | None, unplaced: tuple[str, ...]
    ) -> None:
        pending = self._unplaced_pair(excluded=excluded)
        strat, sonarr = self._strat(file_group=file_group)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == unplaced
        assert sonarr.execute_calls == []


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

        return make_sonarr_for_import(
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

    def test_a_plain_skip_alone_finishes(self) -> None:
        strat, _ = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), manual_candidate("/d/Extras.mkv")],
            episodes=[
                sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
                sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
            ],
        )
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv", "Extras.mkv"], ordered_episode_ids=[101, 102])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert result.left_for_hand_import.names == ()


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
        strat, sonarr = make_sonarr_for_import(
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
        strat, sonarr = make_sonarr_for_import(
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

        # The next arr run warns of it again.
        strat.get_items()
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert len(self._default_quality_warnings(recording)) == 2

    @pytest.mark.parametrize("name", [pytest.param("WEBDL-1080p", id="matched"), pytest.param(None, id="unset")])
    def test_a_matched_or_unset_name_does_not_warn(self, name: str | None) -> None:
        strat, sonarr, recording = self._strat_with_default(name)

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


def _clean_history() -> HistoryPage:
    """A history page whose newest relevant event is a grab (genuinely downloading)."""

    return HistoryPage.model_validate(
        {"records": [{"id": 9, "eventType": "grabbed", "date": "2026-07-17T00:00:00Z"}]},
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
        *, history: HistoryPage | None, folder_candidates: list[ManualImportCandidate], import_mode: str = "auto"
    ) -> tuple[SonarrStrategy, HookedSonarrClient]:
        # candidates=None: the downloadId fetch fails (the 500 shape) -> fallback.
        strat, sonarr = make_sonarr_for_import(candidates=None, config_overrides={"import_mode": import_mode})
        sonarr.history_page_return = history
        sonarr.folder_candidates_return = folder_candidates
        sonarr.path_mappings_return = []
        return strat, sonarr

    @staticmethod
    def _answer_empty_by_id(
        sonarr: FakeSonarrClient,
        *,
        commands: tuple[CommandResource, ...] = (RESCAN_AFTER_ADD,),
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

    @pytest.mark.parametrize(
        ("history", "import_mode", "posted"),
        [
            pytest.param(_clean_history(), "auto", [("ABC123", "copy"), (None, "copy")], id="clean-history"),
            pytest.param(None, "auto", [("ABC123", "copy"), (None, "copy")], id="unreadable-history"),
            pytest.param(closed_history(), "move", [(None, "move"), (None, "move")], id="configured-move"),
        ],
    )
    def test_a_folder_import_keeps_the_download_id_until_history_closes_and_copies_unless_a_mode_is_set(
        self, history: HistoryPage | None, import_mode: str, posted: list[tuple[str | None, str]]
    ) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
            import_mode=import_mode,
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.history_page_return = closed_history()
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls] == posted

    def test_a_closed_history_is_read_once_per_run(self) -> None:
        # Kept for the run: a closed history doesn't reopen, and every poll would read it again.
        strat, sonarr = self._strat(
            history=closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.download_history_calls == ["abc123"]

    def test_history_closed_empty_folder_warns_only_at_the_deadline(self) -> None:
        # An empty folder right after completion is expected (Sonarr's mount
        # lags), so ordinary polls don't warn; the genuinely stuck shape
        # (a pending import with an empty file map, still empty at the deadline) warns once a run,
        # and again in the next run. `closed_history()` carries no import rows, so the
        # rebuild yields nothing and the file map stays empty.
        recording = install_recording_hub()
        strat, _ = self._strat(history=closed_history(), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert diagnostic_messages(recording, Severity.WARNING) == []

        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 1
        assert "/d/Show" in warnings[0]

        strat.get_items()
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 2

    def test_empty_download_id_fetch_with_an_empty_folder_waits_and_warns_at_the_deadline(self) -> None:
        recording = install_recording_hub()
        strat, sonarr = self._strat(history=closed_history(), folder_candidates=[])
        self._answer_empty_by_id(sonarr)
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "found no files" in warning

    def test_rebuilt_file_map_that_is_not_done_waits_without_pinning_it(self) -> None:
        recording = install_recording_hub()
        name = "Show - 01 [1080p].mkv"
        strat, _ = self._strat(history=_imported_history((101, f"/d/Show/{name}")), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.pending is pending
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "1 intended file not visible to Sonarr" in warning
        assert "reading /d/Show found no files" in warning

    def test_folder_fetch_is_never_kept(self) -> None:
        strat, sonarr = self._strat(
            history=_clean_history(), folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        )
        sonarr.queue_return = [queue_record("ABC123", "importBlocked", status="warning")]

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_fetch = start_a_pass
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.folder_candidates_return = None
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert blocked.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    @pytest.mark.parametrize(
        ("history", "commands"),
        [
            pytest.param(closed_history(), (RESCAN_BEFORE_ADD,), id="no-rescan-since-the-add"),
            pytest.param(_clean_history(), (RESCAN_AFTER_ADD,), id="clean-history"),
            pytest.param(None, (RESCAN_AFTER_ADD,), id="unreadable-history"),
        ],
    )
    def test_empty_download_id_fetch_waits_without_the_proof_or_aclosed_history(
        self, history: HistoryPage | None, commands: tuple[CommandResource, ...]
    ) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr, commands=commands)

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
        category: str = "manual",
        *,
        folder_candidates: list[ManualImportCandidate] | None = None,
    ) -> tuple[SonarrStrategy, HookedSonarrClient, QbitCategoryReads]:
        """Sonarr watching `tv-sonarr`, the pending import's torrent in `category`, and Sonarr not tracking it.

        The downloadId fetch answers `[]`, the queue is empty, a rescan finished after the add, and the
        history holds only the neutral page.
        """

        qbit = QbitCategoryReads(category)
        categories = make_fetched_categories(grab="tv-sonarr", post_import=None, qbit=qbit.client())
        strat, sonarr = make_sonarr_for_import(candidates=[], categories=categories)
        sonarr.commands_return = [RESCAN_AFTER_ADD]
        sonarr.history_page_return = HistoryPage.model_validate({"records": []})
        sonarr.folder_candidates_return = (
            [manual_candidate("/d/Show/Show - 01 [1080p].mkv")] if folder_candidates is None else folder_candidates
        )
        sonarr.path_mappings_return = []
        return strat, sonarr, qbit

    @pytest.mark.parametrize(
        ("category", "by_id", "queue", "posted"),
        [
            pytest.param("tv-sonarr", [], [], [], id="a-watched-category-waits-for-sonarr"),
            # Sonarr v4 goes on tracking a torrent moved out of its category until it restarts.
            pytest.param(
                "manual",
                [],
                [queue_record("ABC123", "importPending", status="warning")],
                [],
                id="a-queue-row-waits-for-sonarr",
            ),
            pytest.param("manual", None, [], [(None, "copy")], id="a-failed-fetch-omits-the-id"),
        ],
    )
    def test_only_a_torrent_sonarr_neither_watches_nor_tracks_imports_without_its_id(
        self,
        category: str,
        by_id: list[ManualImportCandidate] | None,
        queue: list[QueueRecord],
        posted: list[tuple[str | None, str]],
    ) -> None:
        strat, sonarr, _ = self._strat(category)
        sonarr.candidates_return = by_id
        sonarr.queue_return = queue

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls] == posted

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


class TestResolveLanguageObjects:
    """A configured language name matches Sonarr's whatever its case, and one Sonarr doesn't define is dropped."""

    @pytest.mark.parametrize(
        ("names", "resolved"),
        [
            pytest.param(["JAPANESE"], ["Japanese"], id="another-case"),
            pytest.param(["Klingon", "English"], ["English"], id="an-unknown-name"),
        ],
    )
    def test_a_name_resolves_to_sonarrs_language_of_that_name(self, names: list[str], resolved: list[str]) -> None:
        defs = [Language(id=1, name="English"), Language(id=8, name="Japanese")]

        assert [language.name for language in resolve_language_objects(names, defs)] == resolved


def _groups(*names: str) -> SeadexDict:
    """A `SeadexDict` with one release from each group of `names`."""

    return {name: rg_group({f"https://nyaa.si/{name}": make_release(url=f"https://nyaa.si/{name}")}) for name in names}


class TestRadarrCheckTitle:
    """A movie's check stops on no SeaDex entry, a cached title or no releases. A grab carries its files to the plan."""

    def test_radarr_no_seadex_entry_stops_at_begin_check(self) -> None:
        run = _FakeRunServices()
        strat = _radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        assert run.anilist_id_prologue_calls == [5]

    def test_a_cached_title_stops_with_an_empty_coverage_backfill(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), cached_skip=True)
        strat = _radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))

        assert (run.grab_requests, run.no_releases_calls, run.nothing_to_grab_calls) == ([], [], [])
        # A movie has no episodes, so the coverage a cached row is backfilled with is empty.
        [coverage] = run.cached_skip_coverages
        assert coverage() == ""

    def test_no_releases_routes_to_shared_stop_with_movie_cache_details(self) -> None:
        entry = make_entry_record(anilist_id=5, url="https://releases.moe/5")
        run = _FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict={},
        )
        strat = _radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        expected: CachedTitleUpdate = {
            "name": "Movie Title",
            "updated_at": entry.updated_at,
            "url": "https://releases.moe/5",
            "coverage": "",
        }
        assert run.no_releases_calls == [(5, expected)]
        assert run.grab_requests == []

    def test_a_grab_carries_the_movies_files_to_the_plan_and_every_group_they_are_of(self) -> None:
        entry = make_entry_record(anilist_id=5, url="https://releases.moe/5")
        seadex_dict = _groups("SubGroup")
        filtered = _groups("SubGroup")
        run = _FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict=seadex_dict,
            plan_grabs_result=grab_plan(filtered),
        )
        files = [
            MovieFile(release_group="GroupA", size=100),
            MovieFile(release_group="GroupA", size=150),
            MovieFile(release_group="GroupB", size=200),
        ]
        strat = _radarr_checking(run, FakeRadarrClient(movie_files=files))

        strat.check_title(RadarrMovie(id=3, title="Item Title"), 5, TitleMapping(anilist_id=5))

        assert run.plan_grabs_calls == [(seadex_dict, ArrFiles(by_group={"GroupA": (100, 150), "GroupB": (200,)}))]
        [req] = run.grab_requests
        assert (req.anilist_id, req.arr_title, req.title_name) == (5, "Item Title", "Movie Title")
        assert req.entry is entry
        assert req.seadex_dict is filtered
        assert req.replaced_groups == ("GroupA", "GroupB")

    @pytest.mark.parametrize(
        ("interactive", "seadex_dict", "prompts"),
        [
            pytest.param(True, _groups("GroupA", "GroupB"), 1, id="interactive-two-groups"),
            pytest.param(True, _groups("GroupA"), 0, id="interactive-one-group"),
            pytest.param(False, _groups("GroupA", "GroupB"), 0, id="not-interactive"),
        ],
    )
    def test_only_an_interactive_run_with_two_groups_asks_which_to_grab(
        self, interactive: bool, seadex_dict: SeadexDict, prompts: int
    ) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), seadex_dict=seadex_dict)
        strat = _radarr_checking(run, interactive=interactive)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))

        assert len(run.interactive_calls) == prompts


class TestRadarrFiles:
    """radarr_files never hard-errors: a size it can't read counts as zero."""

    def test_unreadable_size_with_no_group_converts_to_zero(self) -> None:
        # An unreadable size with no group converts to 0, which disqualifies the whole
        # ownership multiset (fail-closed, like the planner's size matching of Sonarr files with no group): the
        # readable sizes alone must not let a release claim the copy.
        radarr = FakeRadarrClient(
            movie_files=[MovieFile(release_group=None, size=500), MovieFile(release_group=None, size=None)]
        )
        strat = make_radarr_sync(radarr=radarr)

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
    """The type's own contract: hashable, one shared conversion, one label for its groups."""

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
