# pyright: strict
# pyright: reportPrivateUsage=false
# These reach into the strategy's private collaborators (strat._episodes /
# strat._executor) to pin the seam. Strict re-flags that and the repo disables
# reportPrivateUsage for tests.
"""Seam tests for the composition split.

These pin the contract between the run machinery and the Arr strategies: each
`ArrSync` hook reaches the shared pipeline only through the injected
`RunServices` the strategy holds as `self._services` - and since the hub
split out of the old god class, that seam is literal: the scripted fake below
is a real `RunServices` subclass. The strategies are built bare
(`object.__new__`) so no live Sonarr/Radarr client is constructed.
"""

import logging
from collections.abc import Callable
from typing import NamedTuple, override

import pytest
from seadex import EntryRecord

from pearlarr.cache import CacheRecord
from pearlarr.config import Arr
from pearlarr.grab_pipeline import GrabRequest
from pearlarr.log import EntryState
from pearlarr.manual_import import ImportProgress, ImportReadiness, ImportWaitMode, PendingImport
from pearlarr.mappings import ExternalIds, MappingEntry, MappingSource
from pearlarr.output import Severity
from pearlarr.output.recording import RecordingHub
from pearlarr.run_services import RunServices
from pearlarr.seadex_filter import FilterResult
from pearlarr.seadex_radarr import RadarrSync
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import (
    ArrReleaseDict,
    CommandResource,
    HistoryPage,
    Language,
    ManualImportCandidate,
    MovieFile,
    ProgressSink,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    RemotePathMapping,
    SeadexDict,
    SonarrEpisode,
    SonarrItem,
)
from pearlarr.sonarr_episodes import sonarr_series_fingerprint
from pearlarr.sonarr_import_plan import resolve_language_objects

from .builders import (
    FakeCacheStore,
    make_bare_instance,
    make_config,
    make_entry_record,
    make_logger,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
    rg_group,
    sonarr_ep,
    url_item,
)
from .fakes import FakeSonarrClient, diagnostic_messages, install_recording_hub


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

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


class GetAniListIdsCall(NamedTuple):
    """One recorded `get_anilist_ids` call (the args the strategy forwarded).

    Defaults mirror the seam's own defaults, so a recorded call equals an expected
    one constructed with only the fields the strategy actually varied.
    """

    ids: ExternalIds
    log_ignored: bool = True


class CheckAlIdInCacheCall(NamedTuple):
    """One recorded `check_al_id_in_cache` call (its explicit cross-arr `arr`)."""

    arr: Arr
    al_id: int
    seadex_entry: EntryRecord


class LogCachedEntryCall(NamedTuple):
    """One recorded `log_cached_entry` call (its explicit cross-arr `arr`)."""

    arr: Arr
    al_id: int
    state: EntryState = EntryState.UNCHANGED


class _FakeRunServices(RunServices):
    """A scripted `RunServices` - the seam a strategy holds as `self._services`.

    A REAL `RunServices` subclass (the seam name is literal now, so it also
    satisfies the constructors' typed `services` parameter), with its own
    `__init__` that never calls the heavy real one. Every member of the
    strategy-facing surface is overridden: each scriptable result is a
    constructor arg, and the methods whose call a test asserts RECORD it - so
    the contract is pinned by recorded state.
    """

    def __init__(
        self,
        *,
        anilist_ids: dict[int, MappingEntry] | None = None,
        prologue_entry: EntryRecord | None = None,
        anilist_title: str = "Title",
        cached_skip: bool = False,
        al_id_in_cache: bool = False,
        needs_scan: bool = True,
        seadex_dict: SeadexDict | None = None,
        interactive_result: SeadexDict | None = None,
        filter_downloads_result: FilterResult | None = None,
        grab_result: bool = False,
        no_releases_result: bool = False,
        import_wait_mode: ImportWaitMode = ImportWaitMode.OFF,
        selection_stale: bool = False,
    ) -> None:
        self._selection_stale = selection_stale
        self._anilist_ids = anilist_ids or {}
        self._prologue_entry = prologue_entry
        self._anilist_title = anilist_title
        self._cached_skip = cached_skip
        self._al_id_in_cache = al_id_in_cache
        self._needs_scan = needs_scan
        self._seadex_dict: SeadexDict = seadex_dict if seadex_dict is not None else {}
        self._interactive_result = interactive_result
        self._filter_downloads_result = filter_downloads_result
        self._grab_result = grab_result
        self._no_releases_result = no_releases_result
        self._import_wait_mode = import_wait_mode
        self.get_anilist_ids_calls: list[GetAniListIdsCall] = []
        self.al_id_prologue_calls: list[int] = []
        self.check_al_id_in_cache_calls: list[CheckAlIdInCacheCall] = []
        self.log_entry_status_calls: list[tuple[EntryState, str]] = []
        self.log_al_title_calls: list[str] = []
        self.log_cached_entry_calls: list[LogCachedEntryCall] = []
        self.cached_skip_coverages: list[Callable[[], str]] = []
        self.get_seadex_dict_calls: list[EntryRecord] = []
        self.interactive_calls: list[SeadexDict] = []
        self.filter_downloads_calls: list[tuple[int, SeadexDict, ArrReleaseDict]] = []
        self.grab_requests: list[GrabRequest] = []
        self.no_releases_calls: list[tuple[int, CacheRecord]] = []
        self.invalid_selection_skips = 0

    @property
    @override
    def selection_stale(self) -> bool:
        return self._selection_stale

    @override
    def check_al_id_in_cache(self, arr: Arr, al_id: int, seadex_entry: EntryRecord) -> bool:
        self.check_al_id_in_cache_calls.append(CheckAlIdInCacheCall(arr, al_id, seadex_entry))
        return self._al_id_in_cache

    @override
    def al_id_needs_scan(self, al_id: int) -> bool:
        del al_id
        return self._needs_scan

    @override
    def get_anilist_ids(
        self,
        ids: ExternalIds,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        self.get_anilist_ids_calls.append(
            GetAniListIdsCall(ids, log_ignored),
        )
        return self._anilist_ids

    @override
    def al_id_prologue(self, al_id: int) -> EntryRecord | None:
        self.al_id_prologue_calls.append(al_id)
        return self._prologue_entry

    @override
    def cached_entry_skip(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        coverage: Callable[[], str],
    ) -> bool:
        del al_id, sd_entry
        self.cached_skip_coverages.append(coverage)
        return self._cached_skip

    @override
    def get_anilist_title(self, al_id: int) -> str:
        del al_id
        return self._anilist_title

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
        al_id: int,
        seadex_dict: SeadexDict,
        arr_release_dict: ArrReleaseDict,
        ep_list: list[SonarrEpisode] | None = None,
    ) -> FilterResult:
        del ep_list
        self.filter_downloads_calls.append((al_id, seadex_dict, arr_release_dict))
        if self._filter_downloads_result is not None:
            return self._filter_downloads_result
        return FilterResult([], seadex_dict)

    @property
    @override
    def import_wait_mode(self) -> ImportWaitMode:
        return self._import_wait_mode

    @override
    def no_releases_skip(self, al_id: int, cache_details: CacheRecord) -> bool:
        self.no_releases_calls.append((al_id, cache_details))
        return self._no_releases_result

    @override
    def invalid_selection_skip(self) -> bool:
        self.invalid_selection_skips += 1
        return False

    @override
    def grab_and_cache(self, req: GrabRequest) -> bool:
        self.grab_requests.append(req)
        return self._grab_result

    @override
    def log_entry_status(self, state: EntryState, label: str) -> None:
        self.log_entry_status_calls.append((state, label))

    @override
    def log_anilist_item_unmonitored(self, item_title: str) -> None:
        del item_title

    @override
    def log_al_title(self, anilist_title: str, sd_entry: EntryRecord, coverage: str | None = None) -> None:
        del sd_entry, coverage
        self.log_al_title_calls.append(anilist_title)

    @override
    def log_cached_entry(self, arr: Arr, al_id: int, state: EntryState = EntryState.UNCHANGED) -> None:
        self.log_cached_entry_calls.append(LogCachedEntryCall(arr, al_id, state))


def test_fake_overrides_the_full_public_surface() -> None:
    """`@override` catches renames and signature drift but NOT additions.

    A new public `RunServices` member would be silently inherited here (real
    body over fake state) and only fail at test runtime. Reflection closes
    that gap - extend the fake when this fails.
    """

    # Loop-driven members (the runner calls these, strategies never do), so the
    # fake doesn't script them.
    lifecycle = {"begin_run", "ctx", "is_preview", "mark_dirty"}
    surface = {
        name
        for name, member in vars(RunServices).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    overridden = {name for name in vars(_FakeRunServices) if not name.startswith("_")}
    assert surface - lifecycle <= overridden


class _FakeEpisodes:
    """Minimal episode collaborator: scripts `get_ep_list`'s resolved episode list."""

    series_fp = "fp"

    def __init__(self, *, ep_list: list[SonarrEpisode] | None) -> None:
        self._ep_list = ep_list

    def get_ep_list(
        self,
        sonarr_series_id: int,
        al_id: int,
        mapping: MappingEntry,
    ) -> list[SonarrEpisode] | None:
        del sonarr_series_id, al_id, mapping
        return self._ep_list

    def get_sonarr_release_dict(self, ep_list: list[SonarrEpisode]) -> ArrReleaseDict:
        del ep_list
        return {}


class _PassThroughParse:
    """Parse collaborator that returns the dict unchanged (no Sonarr round-trip)."""

    def parse_episodes_from_seadex(self, seadex_dict: SeadexDict, *, series_fp: str) -> SeadexDict:
        del series_fp
        return seadex_dict


class TestItemAnilistIdsDelegates:
    """item_anilist_ids resolves through the held services, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        run = _FakeRunServices(anilist_ids={7: MappingEntry(anilist_id=7)})
        strat = make_bare_instance(RadarrSync, _services=run)

        result = strat.item_anilist_ids(_Item(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: MappingEntry(anilist_id=7)}
        assert run.get_anilist_ids_calls == [GetAniListIdsCall(ExternalIds(tmdb=42, imdb="tt7"), log_ignored=False)]

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrSync, _services=run)

        strat.item_anilist_ids(_Item(tvdbId=99, imdbId="tt9"))

        assert run.get_anilist_ids_calls == [GetAniListIdsCall(ExternalIds(tvdb=99, imdb="tt9"), log_ignored=True)]


class TestFilterToSingle:
    """filter_to_single narrows by the arr's external id (no engine needed)."""

    def test_radarr_matches_tmdb_id(self) -> None:
        strat = make_bare_instance(RadarrSync, logger=make_logger())
        items: list[RadarrItem] = [_Item(tmdbId=1), _Item(tmdbId=2)]

        assert strat.filter_to_single(items, 2) == [items[1]]
        assert strat.filter_to_single(items, 7) == []

    def test_sonarr_matches_tvdb_id(self) -> None:
        strat = make_bare_instance(SonarrSync, logger=make_logger())
        items: list[SonarrItem] = [_Item(tvdbId=10), _Item(tvdbId=20)]

        assert strat.filter_to_single(items, 10) == [items[0]]


class TestRunStartHook:
    """get_items doubles as the run-start hook: it resets the per-run scratch.

    The episode reset/fingerprint now lives on the SonarrEpisodes collaborator.
    This stays at strategy level to pin that get_items actually routes through it.
    """

    def test_sonarr_get_items_clears_ep_list_cache(self) -> None:
        series: list[SonarrItem] = [_Item(id=5), _Item(id=7)]
        strat = make_sonarr_sync(ep_list_cache={5: [sonarr_ep(1, 1)]})

        def _all_series() -> list[SonarrItem]:
            return series

        strat._episodes.get_all_sonarr_series = _all_series

        result = strat.get_items()

        assert result == series
        # The episode collaborator drops its cache + re-fingerprints as it enumerates.
        assert strat._episodes._ep_list_cache == {}
        assert strat._episodes._series_fp == sonarr_series_fingerprint([5, 7])


class TestSonarrPrefetchDelegates:
    """prefetch_episodes is a thin hook over the episode collaborator's warm."""

    def test_sonarr_prefetch_routes_to_episodes(self) -> None:
        strat = make_sonarr_sync()
        prefetch_calls: list[tuple[list[SonarrItem], ProgressSink | None]] = []

        def _prefetch(items: list[SonarrItem], *, progress: ProgressSink | None = None) -> int:
            prefetch_calls.append((items, progress))
            return 3

        strat._episodes.prefetch = _prefetch
        items: list[SonarrItem] = [_Item(id=1)]

        assert strat.prefetch_episodes(items, progress=None) == 3
        assert prefetch_calls == [(items, None)]


class TestRadarrPrefetchEpisodes:
    """Radarr has no episodes: the warm hook is a no-op that warms nothing."""

    def test_returns_zero_and_skips_sink(self) -> None:
        strat = make_bare_instance(RadarrSync, logger=make_logger())
        calls: list[tuple[float, str | None]] = []

        class _Rec:
            def progress(self, fraction: float, detail: str | None = None) -> None:
                calls.append((fraction, detail))

        assert strat.prefetch_episodes([_Item(tmdbId=1)], progress=_Rec()) == 0
        assert calls == []  # no episodes -> sink never driven

    def test_warms_episodes_is_false(self) -> None:
        strat = make_bare_instance(RadarrSync, logger=make_logger())
        assert strat.warms_episodes is False


class TestProcessAlIdThreadsServices:
    """The per-id head runs through the held services. A missing entry stops this id."""

    def test_radarr_no_seadex_entry_returns_false(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(RadarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5)) is False
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_seadex_entry_returns_false(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5)) is False
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_episodes_resolved_skips_explicitly(self) -> None:
        # An anime-id mapping that resolves to [] (season not in Sonarr / offset past
        # the end): skip with the NO_EPISODES status, never mislabeled "unmonitored"
        # and never falling through to grab orphans - and NO AniBridge warning.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = _FakeEpisodes(ep_list=[])
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=make_logger(),
        )

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is False
        assert run.log_entry_status_calls == [(EntryState.NO_EPISODES, "Title")]
        assert run.log_al_title_calls == []
        # anime-id empty is NOT the AniBridge case -> no warning surfaced.
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_sonarr_anibridge_empty_map_skips_with_warning(self) -> None:
        # The AniBridge no-usable-ranges case (a real empty-{} tvdb entry: source
        # ANIBRIDGE, mode ANIBRIDGE): the NO_EPISODES skip PLUS a visible WARNING
        # naming the cause. The warning keys off source (so it also covers the
        # degraded imdb/tmdb case), so the entry must carry source=ANIBRIDGE as a
        # real one does. Fails on the unfixed path, which silently grabbed nothing.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = _FakeEpisodes(ep_list=[])
        recording = install_recording_hub()
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=make_logger(),
        )

        result = strat.process_al_id(
            _Item(id=1, title="Title"),
            5,
            MappingEntry(anilist_id=5, tvdb_mappings={}, source=MappingSource.ANIBRIDGE),
        )

        assert result is False
        assert run.log_entry_status_calls == [(EntryState.NO_EPISODES, "Title")]
        assert run.log_al_title_calls == []
        # AniBridge-specific notice surfaced.
        assert any(
            "AniBridge has no usable season ranges" in message
            for message in diagnostic_messages(recording, Severity.WARNING)
        )

    def test_sonarr_all_invalid_selection_routes_to_shared_skip(self) -> None:
        # The interactive pick rejected every token: the strategy must route to
        # invalid_selection_skip (which persists nothing) and never reach the
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
            SonarrSync,
            _services=run,
            _episodes=_FakeEpisodes(ep_list=[sonarr_ep(1, 1)]),
            _parse=_PassThroughParse(),
            _config=make_config(interactive=True, sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=make_logger(),
        )

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is False
        assert run.invalid_selection_skips == 1
        assert run.grab_requests == []
        assert run.no_releases_calls == []

    def test_sonarr_movie_in_radarr_cache_dedups_via_explicit_arr(self) -> None:
        # The Sonarr->Radarr dedup MUST pass an explicit Arr.RADARR to both the
        # cache check and the cached-entry log (the cross-arr invariant: ctx.arr
        # is SONARR here and would look in the wrong cache). Pinned by recording.
        entry = make_entry_record()
        run = _FakeRunServices(prologue_entry=entry, al_id_in_cache=True)
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=_FakeEpisodes(ep_list=[]),
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=True,
            logger=make_logger(),
        )

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is False
        assert run.check_al_id_in_cache_calls == [CheckAlIdInCacheCall(Arr.RADARR, 5, entry)]
        assert run.log_cached_entry_calls == [LogCachedEntryCall(Arr.RADARR, 5, EntryState.IN_RADARR)]

    def test_selection_stale_bypasses_the_radarr_cache_dedup(self) -> None:
        # A moved matching setting makes every cached verdict suspect (the shared
        # seadex config drives both arrs), so the cross-arr Radarr short-circuit is
        # skipped just like ignore_seadex_update_times - the id flows on to a fresh
        # evaluation instead of a stale IN_RADARR skip. Radarr's own run re-vouches.
        run = _FakeRunServices(prologue_entry=make_entry_record(), al_id_in_cache=True, selection_stale=True)
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=_FakeEpisodes(ep_list=[]),
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=True,
            all_radarr_movies=None,
            logger=make_logger(),
        )

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is False
        assert run.check_al_id_in_cache_calls == []  # the stale run never trusts the cross-arr cache


def _ep_with_file(ep_id: int, *, group: str | None) -> SonarrEpisode:
    """A current Sonarr episode that already holds a file from `group`."""

    return SonarrEpisode.model_validate(
        {"id": ep_id, "episodeFileId": ep_id * 10, "episodeFile": {"releaseGroup": group}},
    )


def _make_sonarr_for_import(
    *,
    candidates: list[ManualImportCandidate] | None,
    queue: list[QueueRecord] | None = None,
    episodes: list[SonarrEpisode] | None = None,
    quality_defs: list[QualityDefinition] | None = None,
    languages: list[Language] | None = None,
    commands: list[CommandResource] | None = None,
    cmd_id: int | None = 42,
    config_overrides: dict[str, list[str] | str] | None = None,
) -> tuple[SonarrSync, FakeSonarrClient]:
    """A bare `SonarrSync` plus its scripted `self.sonarr` `FakeSonarrClient`.

    The fake returns the given queue records, current episodes, manual-import
    candidates, quality definitions, languages and an execute command id -
    everything `import_completed` reaches over the network - and records the two
    import commands so a test asserts on recorded state. `episodes` defaults to
    empty, so the target episodes have NO file yet (they need importing) and the
    done-check never short-circuits. Pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. `queue` defaults to empty (Sonarr
    isn't tracking the download, so the strategy steps in). `commands` defaults to
    empty (no in-flight ManualImport, so the dedup guard never trips). The fake's
    refresh / command-status defaults resolve immediately so the rescan never waits.
    """

    sonarr = FakeSonarrClient(
        queue=queue,
        episodes=episodes,
        commands=commands,
        candidates=candidates,
        quality_defs=quality_defs,
        languages=languages,
        execute_command_id=cmd_id,
    )
    overrides: dict[str, list[str] | str] = config_overrides or {}
    strat = make_sonarr_sync(
        sonarr=sonarr,
        config=make_config(**overrides),
        cache_store=FakeCacheStore(),
    )
    return strat, sonarr


def _queue_record(infohash: str, state: str, *, status: str = "ok") -> QueueRecord:
    """One Sonarr queue record matching a download by infohash + tracked state.

    Built through `QueueRecord.model_validate` from the raw API field names so the
    record mirrors exactly what `SonarrClient.queue` parses at the boundary.
    """

    return QueueRecord.model_validate(
        {
            "downloadId": infohash,
            "trackedDownloadState": state,
            "trackedDownloadStatus": status,
        },
    )


class TestImportCompletedQueueState:
    """import_completed reads the queue + episode files before stepping in."""

    def test_sonarr_importing_retries_without_stepping_in(self) -> None:
        # Sonarr is mid-import (importing) -> wait for it, don't double-import.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importing")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.files_present is False
        assert sonarr.candidate_calls == []
        assert sonarr.execute_calls == []

    def test_clean_pending_retries_until_forced(self) -> None:
        # A clean importPending: defer to Sonarr (RETRY) unless forced.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importPending", status="ok")],
        )

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.files_present is False
        assert sonarr.candidate_calls == []

    def test_clean_pending_forced_steps_in(self) -> None:
        # force=True (snapshot / final monitor poll): stop deferring, issue the
        # import. The copy is async, so this reads RETRY + command_issued, NOT a
        # verified files_present.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importPending", status="ok")],
        )

        probe = strat.import_completed(pending, "/d", force=True)

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert probe.files_present is False
        assert len(sonarr.execute_calls) == 1

    def test_pending_with_warning_waits(self) -> None:
        # importPending waits (PENDING_CLEAN) even with a warning: stepping in on a
        # still-pending record races Sonarr's own import and double-imports. So we
        # retry without issuing a command and let Sonarr settle.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importPending", status="warning")],
        )

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        assert sonarr.execute_calls == []

    def test_target_already_recommended_drops_record(self) -> None:
        # Episode files are the source of truth for "already imported": the target
        # episode already holds the recommended group's file -> done, no scan.
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[_ep_with_file(101, group="SubGroup")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.IMPORTED
        assert probe.files_present is True
        assert sonarr.candidate_calls == []

    def test_import_blocked_steps_in_with_our_mapping(self) -> None:
        # Sonarr can't auto-import (importBlocked) -> our authoritative manual
        # import takes over and ISSUES the command. The copy is async, so right
        # after issuing the probe reads RETRY + command_issued (NOT files_present).
        # A later monitor cycle flips to files_present once the episode files land.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importBlocked", status="warning")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert probe.files_present is False
        assert len(sonarr.candidate_calls) == 1
        assert len(sonarr.execute_calls) == 1

    def test_later_poll_observes_freshly_imported_files(self) -> None:
        # Regression: import verification reads the episode FILES as the source of
        # truth, so the episode list must be re-fetched each poll, never served
        # stale from the per-run cache. Poll 1 (target absent) issues the import
        # (RETRY + command_issued). Once the copy lands, poll 2 must observe the
        # file -> IMPORTED + files_present, WITHOUT re-issuing. A stale cache would
        # keep files_present False forever (the monitor times out as "still
        # importing", and in move mode the import is never confirmed at all).
        pending = pending_import(
            infohash="abc123",
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importBlocked", status="warning")],
        )

        first = strat.import_completed(pending, "/d")
        assert first.files_present is False
        assert first.command_issued is True

        # The copy landed: the target episode now holds the recommended file.
        sonarr.episodes_return = [_ep_with_file(101, group="SubGroup")]

        second = strat.import_completed(pending, "/d")
        assert second.readiness is ImportReadiness.IMPORTED
        assert second.files_present is True
        # Episodes were re-read fresh each poll (not cached), and the landed import
        # was detected before any second execute.
        assert len(sonarr.episodes_calls) == 2
        assert len(sonarr.execute_calls) == 1

    def test_not_in_queue_steps_in(self) -> None:
        # Sonarr isn't tracking the download (our holding category) -> step in,
        # issuing the import command (RETRY + command_issued until the copy lands).
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert len(sonarr.candidate_calls) == 1
        # Stepping in must ISSUE the import, not just scan candidates.
        assert len(sonarr.execute_calls) == 1


def _inflight_manual_import(infohash: str, *, status: str = "started") -> CommandResource:
    """A ManualImport command whose one file carries `infohash` as downloadId."""

    return CommandResource.model_validate(
        {
            "name": "ManualImport",
            "status": status,
            "body": {"files": [{"downloadId": infohash, "episodeIds": [101]}]},
        },
    )


class TestInFlightManualImportGuard:
    """A ManualImport already running for this download must not be re-issued."""

    def test_in_flight_command_suppresses_reissue(self) -> None:
        # Queue empty (Sonarr dropped the torrent while importing server-side), but
        # a started ManualImport for this infohash is still running -> wait, don't
        # stack a duplicate.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        assert sonarr.execute_calls == []
        assert sonarr.candidate_calls == []

    def test_in_flight_guard_holds_even_when_forced(self) -> None:
        # Cross-run regression: the carried-over reconcile path always forces, and
        # that is the path that loops. force=True overrides Sonarr's clean-pending
        # deferral, NOT an already-running command -> still suppress the re-issue.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123")],
        )

        probe = strat.import_completed(pending, "/d", force=True)

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        assert sonarr.execute_calls == []

    def test_completed_command_does_not_suppress(self) -> None:
        # A finished ManualImport is not in flight, so it must never wedge us: with
        # the queue empty we step in and issue our import as before.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("abc123", status="completed")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1

    def test_in_flight_for_other_download_does_not_suppress(self) -> None:
        # An in-flight ManualImport for a DIFFERENT torrent must not block ours.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
            commands=[_inflight_manual_import("OTHERHASH")],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1


class TestImportCompletedPayload:
    """import_completed assigns episodes from OUR map, never Sonarr's parse."""

    def test_payload_uses_pending_fields_not_candidate(self) -> None:
        pending = pending_import(
            series_id=7,
            release_group="SubGroup",
            infohash="HASH",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        # The candidate carries a *different* in-context quality and no
        # authoritative episode/series info. The payload must ignore those.
        candidate = manual_candidate(
            "/downloads/Show - 01 [1080p].mkv",
            quality={"quality": {"name": "HDTV-720p"}},
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        probe = strat.import_completed(pending, "/downloads/Show")

        # The command was issued. The copy is async, so the probe is RETRY +
        # command_issued (not yet files_present) right after issuing.
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
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
            episode_ids=[101],
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

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
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
            episode_ids=[101],
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

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
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
            episode_ids=[101],
            seadex_files=[nfc],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[manual_candidate(f"/d/{nfd}")])

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert sonarr.execute_calls[0][0][0].episodeIds == [101]

    def test_candidate_not_in_our_map_is_never_imported(self) -> None:
        # Strict-honor: a file Sonarr found that ISN'T in our map (e.g. an episode
        # our mapping gave to another preferred torrent) is never imported. Here our
        # one intended file is also present and IS imported.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidates = [
            manual_candidate("/d/Show - 01 [1080p].mkv"),
            manual_candidate("/d/Show - 02 [1080p].mkv"),  # not in our map
        ]
        strat, sonarr = _make_sonarr_for_import(candidates=candidates)

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_sample_candidate_is_skipped(self) -> None:
        pending = pending_import(
            file_episode_map={
                "Show - 01 [1080p].mkv": [101],
                "Show - 01 [1080p].sample.mkv": [101],
            },
            episode_ids=[],
            seadex_files=["Show - 01 [1080p].mkv", "Show - 01 [1080p].sample.mkv"],
        )
        good = manual_candidate("/d/Show - 01 [1080p].mkv")
        sample = manual_candidate(
            "/d/Show - 01 [1080p].sample.mkv",
            rejections=[{"reason": "Sample"}],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[good, sample])

        probe = strat.import_completed(pending, "/d")

        # The good file's import command was issued (RETRY + command_issued).
        # The sample is never queued.
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_already_imported_rejection_does_not_skip_missing_group_file(self) -> None:
        # Bug fix (grab-then-skip): the episode already holds a MISSING-GROUP file
        # (UNKNOWN_GROUP -> still needing our recommended import), and Sonarr offers
        # the candidate WITH an "already imported" rejection (it fires whenever the
        # episode has any file on disk). That rejection must NOT veto our import:
        # we grabbed this exactly to replace the unidentifiable file, so we step in
        # and ISSUE the command (RETRY + command_issued, the copy is async).
        pending = pending_import(
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            episodes=[_ep_with_file(101, group=None)],
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert probe.files_present is False
        assert len(sonarr.execute_calls) == 1

    def test_missing_group_import_then_recognized_terminates(self) -> None:
        # Loop-termination regression (mirrors the import->recognize round-trip):
        # poll 1 imports over a missing-group file (RETRY + command_issued).
        # The imported file now carries OUR group, so poll 2 reads it as RECOMMENDED ->
        # nothing needed -> IMPORTED, with NO re-issue. Proves the fix can't loop.
        pending = pending_import(
            release_group="SubGroup",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            episodes=[_ep_with_file(101, group=None)],
        )

        first = strat.import_completed(pending, "/d")
        assert first.command_issued is True
        assert first.files_present is False

        # The import landed: episode 101 now holds our recommended group's file.
        sonarr.episodes_return = [_ep_with_file(101, group="SubGroup")]

        second = strat.import_completed(pending, "/d")
        assert second.readiness is ImportReadiness.IMPORTED
        assert second.files_present is True
        assert len(sonarr.execute_calls) == 1

    def test_intended_file_missing_from_disk_retries(self) -> None:
        # Our map intends a file Sonarr can't see yet -> never dropped, retried
        # (no command issued, no files present).
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        # A different file is on disk. Ours isn't there yet.
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Unrelated.mkv")],
        )

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        assert probe.files_present is False
        assert sonarr.execute_calls == []

    def test_transient_candidate_scan_retries(self) -> None:
        # A None candidates result (timeout / non-200) is transient -> retry.
        strat, sonarr = _make_sonarr_for_import(candidates=None)

        probe = strat.import_completed(pending_import(), "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert sonarr.execute_calls == []

    def test_languages_follow_dual_audio_flag(self) -> None:
        pending = pending_import(
            is_dual_audio=True,
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        languages = [Language(id=1, name="English"), Language(id=8, name="Japanese")]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            languages=languages,
            config_overrides={"import_languages_dual": ["Japanese", "English"]},
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        names = [lang.name for lang in sonarr.execute_calls[0][0][0].languages]
        assert names == ["Japanese", "English"]

    def test_failed_execute_retries(self) -> None:
        # Sonarr rejected the import command (busy / locked) -> retry, not give up.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate], cmd_id=None)

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        # The execute WAS attempted (Sonarr rejected it -> command_issued False).
        # A regression returning RETRY without even trying would still set False here.
        assert len(sonarr.execute_calls) == 1

    def test_quality_defs_and_languages_cached_per_run(self) -> None:
        # Quality definitions + languages are fetched lazily ONCE and cached on the
        # executor for the rest of the run. Run 1 caches the (a) values. Before run 2
        # the source changes to (b), but a cached run must not re-fetch, so both polls
        # keep the (a) values. Drop the lazy-fetch guard and run 2 refetches (b),
        # flipping the executor's cache to (b) -> these assertions fail.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
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

        strat.import_completed(pending, "/d")

        # The source changes between polls. A cached run must ignore it.
        sonarr.quality_defs_return = defs_b
        sonarr.languages_return = langs_b
        strat.import_completed(pending, "/d")

        assert strat._executor._quality_defs_cache == defs_a
        assert strat._executor._languages_cache == langs_a


class TestRadarrImportCompletedNoOp:
    """Radarr is out of scope: its import_completed is a no-op returning LEAVE."""

    def test_returns_leave(self) -> None:
        strat = make_bare_instance(RadarrSync, logger=make_logger())

        probe = strat.import_completed(pending_import(), "/d")
        assert probe.readiness is ImportReadiness.LEAVE
        assert probe.files_present is False

    def test_pending_import_series_id_is_none(self) -> None:
        # Radarr movies record no pending imports, so the snapshot hook key is None
        # (short-circuits the per-item snapshot entirely).
        strat = make_bare_instance(RadarrSync, logger=make_logger())

        assert strat.pending_import_series_id(_Item(id=5)) is None

    def test_import_progress_is_indeterminate_zero(self) -> None:
        # Radarr records no pending imports. The progress hook returns the safe
        # "no bar, promote nothing" value.
        strat = make_bare_instance(RadarrSync, logger=make_logger())

        assert strat.import_progress(pending_import()) == ImportProgress(0, 0, determinate=False)


class TestManualImportWarningGating:
    """The import warns loudly only at the deadline. Otherwise it's debug.

    A missing intended file on an early poll is expected (the copy hasn't landed),
    so it must NOT inflate the summary's warning count. Only the final attempt
    (at_deadline) warns loudly that a still-missing file is terminal.
    """

    @staticmethod
    def _strat_with_missing_file() -> tuple[SonarrSync, PendingImport]:
        # Our map intends a file that isn't on disk yet (only an unrelated file is),
        # so run_manual_import always finds it missing and retries.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, _ = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Unrelated.mkv")],
        )
        # caplog captures via the root logger's propagation, so use a propagating
        # DEBUG logger here (make_logger disables propagation for quiet runs). The
        # missing-file line is logged by the import executor, which holds the same
        # logger as the strat in production - so set both (the strat's plus the
        # executor's) to mirror that and let caplog see the executor's record.
        logger = logging.getLogger("pearlarr-warning-gating")
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.DEBUG)
        strat.logger = logger
        strat._executor.logger = logger
        return strat, pending

    def test_missing_off_deadline_is_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, pending = self._strat_with_missing_file()
        recording = install_recording_hub()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=False)

        assert probe.readiness is ImportReadiness.RETRY
        assert not any(
            "not visible to Sonarr" in message for message in diagnostic_messages(recording, Severity.WARNING)
        )
        assert any("not visible to Sonarr" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    def test_missing_at_deadline_warns_loudly(self) -> None:
        strat, pending = self._strat_with_missing_file()
        recording = install_recording_hub()

        probe = strat.import_completed(pending, "/d", at_deadline=True)

        assert probe.readiness is ImportReadiness.RETRY
        assert any("not visible to Sonarr" in message for message in diagnostic_messages(recording, Severity.WARNING))


class TestDefaultQualityWarning:
    """An unmatched `imports.default_quality` warns once per run, at the consume seam.

    `quality_axes_from_name` stays pure (its silent-empty return is pinned
    elsewhere). The executor is where the configured name meets the run's real
    Sonarr definitions, and it runs once per FILE - hence the once-per-run guard.
    """

    @staticmethod
    def _strat_with_default(name: str | None) -> tuple[SonarrSync, FakeSonarrClient, RecordingHub]:
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

        strat.import_completed(pending, "/d")
        strat.import_completed(pending, "/d")

        # Both polls stepped in (rebuilt the payload), yet the typo warned once.
        assert len(sonarr.execute_calls) == 2
        assert self._default_quality_warnings(recording) == [
            "imports.default_quality 'Blueray-1080p' matches no Sonarr quality definition - ignoring it",
        ]

        # The guard is per-run scratch: the run-start reset re-arms it.
        strat._executor.reset()
        strat.import_completed(pending, "/d")
        assert len(self._default_quality_warnings(recording)) == 2

    def test_matched_name_does_not_warn(self) -> None:
        strat, sonarr, recording = self._strat_with_default("WEBDL-1080p")

        strat.import_completed(pending_import(), "/d")

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []

    def test_unset_name_does_not_warn(self) -> None:
        strat, sonarr, recording = self._strat_with_default(None)

        strat.import_completed(pending_import(), "/d")

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []


def _dead_history() -> HistoryPage:
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
    """The dead-loop cure: a failed downloadId scan probes history and scans the folder.

    A download Sonarr's history maps to Imported/Failed/Ignored is queue-hidden
    and its `downloadId=` scan 500s forever. Without the fallback every poll
    deferred "for a later run" until the pending TTL silently dropped it.
    """

    @staticmethod
    def _strat(
        *,
        history: HistoryPage | None,
        folder_candidates: list[ManualImportCandidate] | None,
        mappings: list[RemotePathMapping] | None = None,
        config_overrides: dict[str, list[str] | str] | None = None,
    ) -> tuple[SonarrSync, FakeSonarrClient]:
        # candidates=None: the downloadId scan fails (the 500 shape) -> fallback.
        strat, sonarr = _make_sonarr_for_import(candidates=None, config_overrides=config_overrides)
        sonarr.history_page_return = history
        sonarr.folder_candidates_return = folder_candidates
        sonarr.path_mappings_return = mappings if mappings is not None else []
        return strat, sonarr

    def test_dead_tracked_imports_from_translated_folder_without_download_id(self) -> None:
        recording = install_recording_hub()
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )

        probe = strat.import_completed(pending_import(), "/d/Show")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
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
        notes = diagnostic_messages(recording, Severity.INFO)
        assert any("recorded this download as imported on 2026-06-20" in note for note in notes)

    def test_dead_tracked_explicit_mode_is_honored(self) -> None:
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
            config_overrides={"import_mode": "move"},
        )

        strat.import_completed(pending_import(), "/d/Show")

        [(_, mode)] = sonarr.execute_calls
        assert mode == "move"

    def test_clean_probe_keeps_download_id_and_configured_mode(self) -> None:
        # A transient blip on a genuinely-downloading torrent: the folder scan
        # substitutes for the candidates, but the entries stay status-quo so the
        # tracked lifecycle (copy-not-move, queue resolution) is preserved.
        strat, sonarr = self._strat(
            history=_clean_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )

        probe = strat.import_completed(pending_import(), "/d/Show")

        assert probe.command_issued is True
        [(files, mode)] = sonarr.execute_calls
        assert mode == "auto"
        assert files[0].downloadId == "abc123"

    def test_probe_failure_defaults_to_status_quo_and_is_not_memoized(self) -> None:
        strat, sonarr = self._strat(
            history=None,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show")
        strat.import_completed(pending, "/d/Show")

        # Status-quo-safe default: entries keep the downloadId (converges even
        # in the dead-tracked state, just noisily - bounded, never a loop).
        assert all(files[0].downloadId == "abc123" for files, _ in sonarr.execute_calls)
        # A FAILED probe is never memoized: the second activation re-probed.
        assert sonarr.history_probe_calls == ["abc123", "abc123"]

    def test_dead_verdict_memoized_and_noted_once_per_run(self) -> None:
        recording = install_recording_hub()
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show")
        strat.import_completed(pending, "/d/Show")

        assert sonarr.history_probe_calls == ["abc123"]
        notes = [n for n in diagnostic_messages(recording, Severity.INFO) if "recorded this download" in n]
        assert len(notes) == 1

    def test_empty_folder_scan_does_not_pin_and_download_id_scan_recovers(self) -> None:
        # 200 [] = the folder isn't visible to Sonarr (or the translation is
        # wrong). Pinning on it would wedge the record. Instead the next poll
        # retries the recoverable downloadId scan first.
        strat, sonarr = self._strat(history=_dead_history(), folder_candidates=[])
        pending = pending_import()

        probe = strat.import_completed(pending, "/d/Show")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False
        assert sonarr.execute_calls == []

        # The transient heals: the downloadId scan answers again and wins.
        sonarr.candidates_return = [manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        probe = strat.import_completed(pending, "/d/Show")

        assert probe.command_issued is True
        assert len(sonarr.candidate_calls) == 2
        assert len(sonarr.folder_candidate_calls) == 1
        assert sonarr.execute_calls[0][0][0].downloadId == "abc123"

    def test_dead_tracked_empty_folder_warns_once_per_run(self) -> None:
        # The genuinely stuck shape (by-id never works, folder empty): warned
        # once a run, re-armed by reset.
        recording = install_recording_hub()
        strat, _ = self._strat(history=_dead_history(), folder_candidates=[])
        pending = pending_import()

        strat.import_completed(pending, "/d/Show")
        strat.import_completed(pending, "/d/Show")

        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 1
        assert "/d/Show" in warnings[0]

        strat._executor.reset()
        strat.import_completed(pending, "/d/Show")
        warnings = [w for w in diagnostic_messages(recording, Severity.WARNING) if "found no files" in w]
        assert len(warnings) == 2

    def test_clean_verdict_empty_folder_stays_quiet(self) -> None:
        # A transient by-id blip self-heals next poll, so no warning fires at
        # all (the by-id client read is quiet, the executor owns the noise).
        recording = install_recording_hub()
        strat, _ = self._strat(history=_clean_history(), folder_candidates=[])

        strat.import_completed(pending_import(), "/d/Show")

        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_nonempty_folder_scan_pins_for_the_run_and_reset_unpins(self) -> None:
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show")
        strat.import_completed(pending, "/d/Show")

        # Pinned after the nonempty scan: the second poll went straight to the
        # folder (one downloadId attempt total), and the run fetched the
        # remote path mappings exactly once.
        assert len(sonarr.candidate_calls) == 1
        assert len(sonarr.folder_candidate_calls) == 2
        assert sonarr.path_mapping_calls == 1

        # The pin is per-run scratch: reset re-arms the downloadId scan.
        strat._executor.reset()
        sonarr.candidates_return = [manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        strat.import_completed(pending, "/d/Show")
        assert len(sonarr.candidate_calls) == 2

    def test_empty_download_id_scan_is_not_a_trigger(self) -> None:
        # [] (not None) means Sonarr ANSWERED "no files visible yet" - the
        # existing retry semantics apply and the fallback stays out of it.
        strat, sonarr = _make_sonarr_for_import(candidates=[])

        probe = strat.import_completed(pending_import(), "/d/Show")

        assert probe.readiness is ImportReadiness.RETRY
        assert sonarr.history_probe_calls == []
        assert sonarr.folder_candidate_calls == []

    def test_translated_in_flight_command_suppresses_reissue(self) -> None:
        # A dead-tracked import POSTs translated paths and no downloadId.
        # The next poll's guard must recognize it through the memoized translation
        # (the scripted command has no episode ids, so only the path arm can).
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show")
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
        probe = strat.import_completed(pending, "/d/Show")

        assert probe.readiness is ImportReadiness.RETRY
        assert len(sonarr.execute_calls) == 1

    def test_single_file_content_path_scans_the_file(self) -> None:
        # A single-FILE torrent's content_path IS the file. Sonarr's folder=
        # param accepts a file path (its FileExists arm).
        strat, sonarr = self._strat(
            history=_dead_history(),
            folder_candidates=[manual_candidate("/remote/tv/Show - 01 [1080p].mkv")],
            mappings=[_tv_mapping()],
        )

        probe = strat.import_completed(pending_import(), "/d/Show - 01 [1080p].mkv")

        assert probe.command_issued is True
        assert [folder for folder, _ in sonarr.folder_candidate_calls] == ["/remote/tv/Show - 01 [1080p].mkv"]


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
    return {srg: rg_group({url: url_item(url=url)})}


class TestRadarrProcessAlIdSeam:
    """process_al_id's movie middle threads its facts through the held services."""

    @staticmethod
    def _make_strat(
        run: _FakeRunServices,
        *,
        files: list[MovieFile] | None = None,
        interactive: bool = False,
    ) -> tuple[RadarrSync, _FakeRadarr]:
        radarr = _FakeRadarr(files or [])
        strat = make_bare_instance(
            RadarrSync,
            _services=run,
            radarr=radarr,
            logger=make_logger(),
            _config=make_config(interactive=interactive),
        )
        return strat, radarr

    def test_cached_entry_short_circuits_before_any_movie_read(self) -> None:
        run = _FakeRunServices(prologue_entry=make_entry_record(), cached_skip=True)
        strat, radarr = self._make_strat(run)

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is False
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
            no_releases_result=True,  # a sentinel: pins the pass-through return
        )
        strat, _ = self._make_strat(run)

        result = strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))

        assert result is True
        expected: CacheRecord = {
            "name": "Movie Title",
            "updated_at": entry.updated_at,
            "torrent_hashes": [],
            "url": "https://releases.moe/5",
            "coverage": "",
        }
        assert run.no_releases_calls == [(5, expected)]
        assert run.grab_requests == []

    def test_happy_path_threads_release_dict_into_grab_request(self) -> None:
        entry = make_entry_record(anilist_id=5, url="https://releases.moe/5")
        seadex_dict = _one_group_dict("SubGroup")
        filtered = _one_group_dict("SubGroup")
        run = _FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict=seadex_dict,
            filter_downloads_result=FilterResult(["feedface"], filtered),
            grab_result=True,
        )
        strat, _ = self._make_strat(run, files=[MovieFile(release_group="OldGroup", size=100)])

        result = strat.process_al_id(_Item(id=3, title="Item Title"), 5, MappingEntry(anilist_id=5))

        # grab_and_cache's scripted bool passes straight through.
        assert result is True
        # The download filter received the movie's accumulated release dict.
        assert run.filter_downloads_calls == [(5, seadex_dict, {"OldGroup": [100]})]
        [req] = run.grab_requests
        assert req.al_id == 5
        assert req.item_title == "Item Title"
        assert req.anilist_title == "Movie Title"
        # The SeaDex entry rides the request whole (the notifier reads its
        # url/notes/comparisons), so pin identity, not a copied field.
        assert req.entry is entry
        # The grab consumes the FILTERED dict + hashes, not the pre-filter dict.
        assert req.seadex_dict is filtered
        assert req.torrent_hashes == ["feedface"]
        # replaced_groups carries the movie's release-dict keys (all of them).
        assert req.replaced_groups == ("OldGroup",)

    def test_multi_edition_movie_forwards_every_release_group(self) -> None:
        # A multi-edition movie holds files from two groups. The GrabRequest (and
        # so the notifier) must see both, in the release dict's insertion order -
        # not just the first file's group.
        run = _FakeRunServices(
            prologue_entry=make_entry_record(),
            anilist_title="Movie Title",
            seadex_dict=_one_group_dict("SubGroup"),
            filter_downloads_result=FilterResult([], _one_group_dict("SubGroup")),
            grab_result=True,
        )
        strat, _ = self._make_strat(
            run,
            files=[
                MovieFile(release_group="GroupA", size=100),
                MovieFile(release_group="GroupB", size=200),
            ],
        )

        strat.process_al_id(_Item(id=3, title="Item Title"), 5, MappingEntry(anilist_id=5))

        [req] = run.grab_requests
        assert req.replaced_groups == ("GroupA", "GroupB")

    def _interactive_call_count(self, *, interactive: bool, n_groups: int) -> int:
        """Run one process_al_id and count the interactive-filter invocations."""

        seadex_dict: SeadexDict = {}
        for idx in range(n_groups):
            seadex_dict.update(_one_group_dict(f"Group{idx}"))
        run = _FakeRunServices(prologue_entry=make_entry_record(), seadex_dict=seadex_dict)
        strat, _ = self._make_strat(run, interactive=interactive)

        strat.process_al_id(_Item(id=1, title="Title"), 5, MappingEntry(anilist_id=5))
        return len(run.interactive_calls)

    def test_interactive_prompts_only_for_multi_release_interactive(self) -> None:
        assert self._interactive_call_count(interactive=True, n_groups=2) == 1
        assert self._interactive_call_count(interactive=True, n_groups=1) == 0
        assert self._interactive_call_count(interactive=False, n_groups=2) == 0


class TestRadarrReleaseDict:
    """get_radarr_release_dict accumulates sizes per group and never hard-errors."""

    def test_multiple_distinct_groups_kept_not_errored(self) -> None:
        # VU3: 2 distinct groups no longer raise (which skipped the movie every run).
        # The dict carries both so the planner dedups against each.
        radarr = _FakeRadarr([MovieFile(release_group="A", size=100), MovieFile(release_group="B", size=200)])
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {"A": [100], "B": [200]}

    def test_same_group_sizes_accumulate(self) -> None:
        # CB6: two files of one group keep BOTH sizes (the old comprehension collapsed
        # to the last).
        radarr = _FakeRadarr([MovieFile(release_group="A", size=100), MovieFile(release_group="A", size=200)])
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {"A": [100, 200]}

    def test_no_files_returns_none_marker(self) -> None:
        radarr = _FakeRadarr([])
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {None: [None]}
