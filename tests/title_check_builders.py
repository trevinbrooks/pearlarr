# pyright: strict
"""Builders for the strategies' title check tests: a recording run services, a scripted parse and listings.

The Sonarr checks share series 7 and the one torrent its title lists, and `sonarr_grab_run` drives a check of it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import override

from seadex import EntryRecord

from pearlarr.arr_models import ArrFiles, SonarrEpisode, SonarrSeries
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.arr_run_state import ArrRunState, Title
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import AppConfig, Arr
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import (
    AnimeIdsRawMapping,
    ExternalIds,
    MappingResolver,
    MappingSources,
    ResolvedMappings,
    TitleMapping,
)
from pearlarr.pending_import import TitleNames
from pearlarr.pending_import_store import PendingImportStore
from pearlarr.placement_types import FetchedListings
from pearlarr.planner import GrabPlan
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.releases import SeadexDict
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import SeriesNumbering
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    ScriptedAniListClient,
    grab_plan,
    make_bare_instance,
    make_config,
    make_entry_record,
    make_logger,
    make_release,
    make_run_deps,
    make_torrent_record,
    parsed_info,
    rg_group,
    scripted_roles,
    sonarr_ep,
    stand_in_ep,
    stored_orders,
)
from .fakes import FakeRadarrClient, FakeSonarrClient, PerSeriesSonarrClient, ScriptedEpisodes

GRAB_URL = "https://nyaa.si/1"
"""The URL of the one torrent a Sonarr title check of series 7 lists."""

GRAB_FILES = ("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv")
"""The torrent's two files, named as the season's first two episodes."""

EP1 = sonarr_ep(1, 1, ep_id=101, episode_file_id=0)
"""Series 7's first episode, on row 101 with no file."""

EP2 = sonarr_ep(1, 2, ep_id=102, episode_file_id=0)
"""Series 7's second episode, on row 102 with no file."""

SERIES_7 = SonarrSeries(id=7, title="Show", tvdbId=70)
"""Series 7 as Sonarr lists it."""

STAND_IN_E2 = stand_in_ep(1, 2, tvdb_id=9002)
"""The stand-in for a second episode that Sonarr's order of series 7 has no row for."""


class FakeRunServices(ArrRunServices):
    """A scripted `ArrRunServices` - the object a strategy keeps as `self._services`.

    A real `ArrRunServices` subclass, so it fits the constructors' typed `services`
    parameter, with its own `__init__` that never calls the expensive real one. Every
    member of the strategy-facing interface is overridden: each scriptable result is a
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
        del ids, log_ignored
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


class ScriptedListings:
    """A scripted listings collaborator: no torrent listed and no size matched, or SeaDex unreachable."""

    def __init__(self, *, unreachable: bool = False) -> None:
        self._read = None if unreachable else FetchedListings({}, {})

    def read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        del series_id, resolved
        return self._read


class ScriptedParse:
    """Parse collaborator serving scripted files per release (no Sonarr round-trip), none for an unscripted release."""

    def __init__(self, files_by_url: dict[str, tuple[ListedFile, ...]] | None = None) -> None:
        self._files = files_by_url or {}

    def parsed_files(self, seadex_dict: SeadexDict) -> dict[str, tuple[ListedFile, ...]]:
        return {
            url: self._files.get(url, ()) for group_releases in seadex_dict.values() for url in group_releases.releases
        }


SERIES_8 = SonarrSeries(id=8, title="Show Zoku", tvdbId=80)
"""Series 8 as Sonarr lists it, a series a stored pending import may claim."""

SERIES_8_E2 = sonarr_ep(1, 2, ep_id=250, tvdb_id=9101, episode_file_id=0)
"""Series 8's one row: its second episode, on row 250 with no file."""

TITLE_5 = TitleMapping(anilist_id=5, tvdb_season=1, tvdb_epoffset=1)
"""Title 5 of series 7: its first season from the second episode on."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Unread:
    """What a check read without an answer: SeaDex's entries, AniList's titles and SkyHook's series."""

    seadex: bool = False
    """Whether SeaDex is unreachable for every entry but the one the check began on."""
    anilist_ids: frozenset[int] = frozenset()
    """The titles whose AniList lookup comes while AniList is unreachable."""
    series_ids: frozenset[int] = frozenset()
    """The series SkyHook has never answered for."""


def sonarr_checking(run: FakeRunServices, sonarr: FakeSonarrClient, deps: ArrRunDeps | None = None) -> SonarrStrategy:
    """A `SonarrStrategy` built by its constructor over `run` and `sonarr`, past the library read an arr run starts on.

    The deps are `make_run_deps` over the run's cache store unless given, and then they must share it.
    """

    deps = deps or make_run_deps(cache_store=run.cache_store)
    assert deps.cache_store is run.cache_store
    strat = SonarrStrategy(deps, run, sonarr_client=sonarr)
    strat.start_of_run(strat.get_items())
    return strat


def sonarr_grab_run(
    episodes: list[SonarrEpisode],
    *,
    store: FakeCacheStore | None = None,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    grab: bool = True,
    unread: Unread = Unread(),
) -> FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 of series 7, whose rows are `episodes`, in a blocking run unless `mode` says.

    The series has mappings for titles 4 and 5 and for the ignored title 6, each with a SeaDex entry. Title 5's lists
    one torrent, of `GRAB_FILES`, which Sonarr parses as the season's first two episodes, and the others list none.
    The plan grabs it unless `grab` is off. Sonarr also has series 8, with the one row `SERIES_8_E2`. Every source
    answers but the ones `unread` names. `store` is the cache the run starts with, empty unless given.
    """

    torrent = make_torrent_record(infohash="h1", file_names=GRAB_FILES)
    entry = make_entry_record(anilist_id=5, url="https://releases.moe/5", torrents=(torrent,))
    entries = {4: make_entry_record(anilist_id=4), 5: entry, 6: make_entry_record(anilist_id=6)}
    release = make_release(url=GRAB_URL, infohash="h1", files=list(GRAB_FILES), size=[1000] * len(GRAB_FILES))
    run = FakeRunServices(
        anilist_ids={4: TitleMapping(anilist_id=4), 5: TITLE_5},
        ignored={6: TitleMapping(anilist_id=6)},
        prologue_entry=entry,
        anilist_title="Show Cour 2",
        seadex_dict={"NAN0": rg_group({GRAB_URL: release})},
        plan_marks=frozenset({GRAB_URL}) if grab else frozenset(),
        import_wait_mode=mode,
        cache_store=store,
    )
    for series in (SERIES_7, SERIES_8):
        if series.id not in unread.series_ids:
            run.cache_store.put_tvdb_orders(series.tvdbId, stored_orders({}))
    deps = make_run_deps(
        cache_store=run.cache_store,
        seadex=FakeSeaDexSource({5: entry} if unread.seadex else entries, unreachable=unread.seadex),
        anilist_client=ScriptedAniListClient("Show Cour 2", failing=unread.anilist_ids),
    )
    parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(GRAB_FILES, start=1)}
    sonarr = PerSeriesSonarrClient({7: episodes, 8: [SERIES_8_E2]}, candidates=[])
    sonarr.all_series_return = [SERIES_7, SERIES_8]
    sonarr.parse_fn = parses.get
    sonarr_checking(run, sonarr, deps).check_title(SERIES_7, 5, TITLE_5)
    return run


def scripted_checking(
    run: FakeRunServices,
    episodes: ScriptedEpisodes | None = None,
    *,
    config: AppConfig | None = None,
    parse: ScriptedParse | None = None,
    listings: ScriptedListings | None = None,
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
        _parse=parse or ScriptedParse(),
        _listings=listings or ScriptedListings(),
        _config=config or make_config(sleep_time=0),
        logger=make_logger(),
    )


def scripted_grab_run(
    *,
    episodes: ScriptedEpisodes,
    store: FakeCacheStore | None = None,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    grab: bool = True,
    files: tuple[ListedFile, ...] | None = None,
    listings: ScriptedListings | None = None,
) -> FakeRunServices:
    """Drive a Sonarr `check_title` for title 5 (series 7) listing one torrent, in a blocking run unless `mode` says.

    The series has mappings for titles 4 and 5 and for the ignored title 6. The torrent is `files`, or `GRAB_FILES`
    named as the season's first two episodes. The plan grabs it unless `grab` is off. `store` is the cache the run
    starts with, empty unless given.
    """

    if files is None:
        parses = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(GRAB_FILES, start=1)}
        files = tuple(ListedFile(name, 1000, parses[name]) for name in GRAB_FILES)
    release = make_release(url=GRAB_URL, infohash="h1", files=[f.file_name for f in files])
    seadex: SeadexDict = {"NAN0": rg_group({GRAB_URL: release})}
    run = FakeRunServices(
        anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
        ignored={6: TitleMapping(anilist_id=6)},
        prologue_entry=make_entry_record(url="https://releases.moe/5"),
        anilist_title="Show Cour 2",
        seadex_dict=seadex,
        plan_marks=frozenset({GRAB_URL}) if grab else frozenset(),
        import_wait_mode=mode,
        cache_store=store,
    )
    strat = scripted_checking(run, episodes, parse=ScriptedParse({GRAB_URL: files}), listings=listings)
    strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))
    return run


def radarr_checking(
    run: FakeRunServices, radarr: FakeRadarrClient | None = None, *, interactive: bool = False
) -> RadarrStrategy:
    """A `RadarrStrategy` built by its constructor over `run` and `radarr`, in interactive mode if `interactive`."""

    deps = make_run_deps(config=make_config(interactive=interactive))
    return RadarrStrategy(deps, run, radarr if radarr is not None else FakeRadarrClient())


def seadex_groups(*names: str) -> SeadexDict:
    """A `SeadexDict` with one release from each group of `names`."""

    return {name: rg_group({f"https://nyaa.si/{name}": make_release(url=f"https://nyaa.si/{name}")}) for name in names}


def mapping_one_series(deps: ArrRunDeps, *, tvdb_id: int, anilist_id: int) -> ArrRunDeps:
    """`deps` whose mappings map TVDB series `tvdb_id` to AniList title `anilist_id`, and nothing else."""

    return _mapping_only(deps, {"anilist_id": anilist_id, "tvdb_id": tvdb_id})


def mapping_one_imdb_title(deps: ArrRunDeps, *, imdb_id: str, anilist_id: int) -> ArrRunDeps:
    """`deps` whose mappings map IMDb title `imdb_id` to AniList title `anilist_id`, and nothing else."""

    return _mapping_only(deps, {"anilist_id": anilist_id, "imdb_id": imdb_id})


def mapping_one_ignored_imdb_title(deps: ArrRunDeps, *, imdb_id: str, anilist_id: int) -> ArrRunDeps:
    """`deps` whose mappings map IMDb title `imdb_id` to AniList title `anilist_id`, which the ignore list names."""

    return _mapping_only(deps, {"anilist_id": anilist_id, "imdb_id": imdb_id}, ignored={anilist_id})


def _mapping_only(deps: ArrRunDeps, row: AnimeIdsRawMapping, ignored: set[int] | None = None) -> ArrRunDeps:
    """`deps` whose mappings are the one Anime-IDs row `row`, with the titles in `ignored` on the ignore list."""

    mapped = MappingSources(anime={"Show": row}, anidb=False, anibridge=False)
    return replace(
        deps,
        resolver=MappingResolver(refresh_days=1, ignore_anilist_ids=ignored or set(), web=deps.web, sources=mapped),
    )
