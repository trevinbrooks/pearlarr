# pyright: strict
"""Builders for the strategies' title check tests: a recording run services, a scripted parse and listings."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import override

from seadex import EntryRecord

from pearlarr.arr_models import ArrFiles, SonarrEpisode
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
    grab_plan,
    make_bare_instance,
    make_config,
    make_logger,
    make_release,
    make_run_deps,
    rg_group,
    scripted_roles,
)
from .fakes import FakeRadarrClient, ScriptedEpisodes


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
    """A scripted listings collaborator: no torrent listed and the given size matches, or SeaDex unreachable."""

    def __init__(self, episode_by_size: Mapping[int, int] | None = None, *, unreachable: bool = False) -> None:
        self._read = None if unreachable else FetchedListings({}, episode_by_size or {})

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


def sonarr_checking(
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
