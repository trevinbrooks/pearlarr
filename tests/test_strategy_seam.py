# pyright: strict
# pyright: reportPrivateUsage=false
# These reach into the strategy's private collaborators (strat._episodes /
# strat._executor) to pin the seam; strict re-flags that and the repo disables
# reportPrivateUsage for tests.
"""Seam tests for the composition split.

These pin the contract between the run machinery and the Arr strategies: each
``ArrSync`` hook reaches the shared pipeline only through the injected
``RunServices`` the strategy holds as ``self._services``. The strategies are
built bare (``object.__new__``) so no live Sonarr/Radarr client is constructed.
"""

import logging
from collections.abc import Callable
from typing import NamedTuple

import pytest
from seadex import EntryRecord

from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import (
    ImportReadiness,
    PendingImport,
    resolve_language_objects,
)
from seadexarr.modules.mappings import MappingEntry, MappingSource
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    CommandResource,
    Language,
    ManualImportCandidate,
    MovieFile,
    ProgressSink,
    QualityDefinition,
    QueueRecord,
    RadarrItem,
    SonarrEpisode,
    SonarrItem,
)
from seadexarr.modules.sonarr_episodes import sonarr_series_fingerprint

from .builders import (
    FakeCacheStore,
    make_bare_instance,
    make_config,
    make_entry_record,
    make_logger,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
    sonarr_ep,
)
from .fakes import CaptureHandler, FakeSonarrClient


class _Item:
    """A stand-in Arr item exposing whatever id attributes a test sets.

    Declares the full ``ArrItem`` surface so it structurally satisfies the
    ``SonarrItem`` / ``RadarrItem`` protocols; each test sets only the attributes
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
    """One recorded ``get_anilist_ids`` call (the kwargs the strategy forwarded).

    Defaults mirror the seam's own defaults, so a recorded call equals an expected
    one constructed with only the fields the strategy actually varied.
    """

    tvdb_id: int | None = None
    tmdb_id: int | None = None
    imdb_id: str | None = None
    log_ignored: bool = True


class _FakeRunServices:
    """Typed stand-in for the ``RunServices`` seam a strategy holds as ``self._services``.

    Carries ONLY the methods the ``ArrSync`` hooks call; each scriptable result is
    a constructor arg, and the methods whose call a test asserts RECORD it - so the
    contract is pinned by recorded state. Absorbed as a bare attribute by
    ``make_bare_instance``, so it need not satisfy the full ``RunServices`` protocol;
    it only answers what the hook under test reaches.
    """

    def __init__(
        self,
        *,
        anilist_ids: dict[int, MappingEntry] | None = None,
        prologue_entry: EntryRecord | None = None,
        anilist_title: str = "Title",
        cached_skip: bool = False,
    ) -> None:
        self._anilist_ids = anilist_ids or {}
        self._prologue_entry = prologue_entry
        self._anilist_title = anilist_title
        self._cached_skip = cached_skip
        self.get_anilist_ids_calls: list[GetAniListIdsCall] = []
        self.al_id_prologue_calls: list[int] = []
        self.log_entry_status_calls: list[tuple[EntryState, str]] = []
        self.log_al_title_calls: list[str] = []

    def get_anilist_ids(
        self,
        tvdb_id: int | None = None,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        log_ignored: bool = True,
    ) -> dict[int, MappingEntry]:
        self.get_anilist_ids_calls.append(
            GetAniListIdsCall(tvdb_id, tmdb_id, imdb_id, log_ignored),
        )
        return self._anilist_ids

    def al_id_prologue(self, al_id: int) -> EntryRecord | None:
        self.al_id_prologue_calls.append(al_id)
        return self._prologue_entry

    def cached_entry_skip(
        self,
        al_id: int,
        sd_entry: EntryRecord,
        sd_url: str,
        coverage: Callable[[], str],
    ) -> bool:
        del al_id, sd_entry, sd_url, coverage
        return self._cached_skip

    def get_anilist_title(self, al_id: int) -> str:
        del al_id
        return self._anilist_title

    def log_entry_status(self, state: EntryState, label: str, style: str | None = "grey50") -> bool:
        del style
        self.log_entry_status_calls.append((state, label))
        return True

    def log_al_title(self, anilist_title: str, sd_entry: EntryRecord, coverage: str | None = None) -> bool:
        del sd_entry, coverage
        self.log_al_title_calls.append(anilist_title)
        return True


class _FakeEpisodes:
    """Minimal episode collaborator: scripts ``get_ep_list``'s resolved episode list."""

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


def _capture_logger(name: str) -> tuple[logging.Logger, CaptureHandler]:
    """A fresh, isolated DEBUG logger plus the handler that captures its records."""

    logger = logging.getLogger(name)
    logger.handlers.clear()
    capture = CaptureHandler()
    logger.addHandler(capture)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, capture


class TestItemAnilistIdsDelegates:
    """item_anilist_ids resolves through the held services, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        run = _FakeRunServices(anilist_ids={7: MappingEntry(anilist_id=7)})
        strat = make_bare_instance(RadarrSync, _services=run)

        result = strat.item_anilist_ids(_Item(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: MappingEntry(anilist_id=7)}
        assert run.get_anilist_ids_calls == [GetAniListIdsCall(tmdb_id=42, imdb_id="tt7", log_ignored=False)]

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrSync, _services=run)

        strat.item_anilist_ids(_Item(tvdbId=99, imdbId="tt9"))

        assert run.get_anilist_ids_calls == [GetAniListIdsCall(tvdb_id=99, imdb_id="tt9", log_ignored=True)]


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

    The episode reset/fingerprint now lives on the SonarrEpisodes collaborator;
    this stays at strategy level to pin that get_items actually routes through it.
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
    """The per-id head runs through the held services; a missing entry stops this id."""

    def test_radarr_no_seadex_entry_returns_false(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(RadarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_seadex_entry_returns_false(self) -> None:
        run = _FakeRunServices()
        strat = make_bare_instance(SonarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        assert run.al_id_prologue_calls == [5]

    def test_sonarr_no_episodes_resolved_skips_explicitly(self) -> None:
        # An anime-id mapping that resolves to [] (season not in Sonarr / offset past
        # the end): skip with the NO_EPISODES status, never mislabeled "unmonitored"
        # and never falling through to grab orphans - and NO AniBridge warning.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = _FakeEpisodes(ep_list=[])
        logger, capture = _capture_logger("seadexarr-seam-no-episodes")
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=logger,
        )

        result = strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5))

        assert result is False
        assert run.log_entry_status_calls == [(EntryState.NO_EPISODES, "Title")]
        assert run.log_al_title_calls == []
        # anime-id empty is NOT the AniBridge case -> no warning surfaced.
        assert not any(r.levelno >= logging.WARNING for r in capture.records)

    def test_sonarr_anibridge_empty_map_skips_with_warning(self) -> None:
        # The AniBridge no-usable-ranges case (a real empty-{} tvdb entry: source
        # ANIBRIDGE, mode ANIBRIDGE): the NO_EPISODES skip PLUS a visible WARNING
        # naming the cause. The warning keys off source (so it also covers the
        # degraded imdb/tmdb case), so the entry must carry source=ANIBRIDGE as a
        # real one does. Fails on the unfixed path, which silently grabbed nothing.
        run = _FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        episodes = _FakeEpisodes(ep_list=[])
        logger, capture = _capture_logger("seadexarr-seam-anibridge")
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=logger,
        )

        result = strat.process_al_id(
            _Item(id=1),
            "Title",
            5,
            MappingEntry(anilist_id=5, tvdb_mappings={}, source=MappingSource.ANIBRIDGE),
        )

        assert result is False
        assert run.log_entry_status_calls == [(EntryState.NO_EPISODES, "Title")]
        assert run.log_al_title_calls == []
        # AniBridge-specific notice surfaced.
        assert any(r.levelno >= logging.WARNING for r in capture.records)


def _ep_with_file(ep_id: int, *, group: str | None) -> SonarrEpisode:
    """A current Sonarr episode that already holds a file from ``group``."""

    return SonarrEpisode.from_api(
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
    config_overrides: dict[str, list[str]] | None = None,
) -> tuple[SonarrSync, FakeSonarrClient]:
    """A bare ``SonarrSync`` plus its scripted ``self.sonarr`` :class:`FakeSonarrClient`.

    The fake returns the given queue records, current episodes, manual-import
    candidates, quality definitions, languages and an execute command id -
    everything ``import_completed`` reaches over the network - and records the two
    import commands so a test asserts on recorded state. ``episodes`` defaults to
    empty, so the target episodes have NO file yet (they need importing) and the
    done-check never short-circuits; pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. ``queue`` defaults to empty (Sonarr
    isn't tracking the download, so the strategy steps in). ``commands`` defaults to
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
    overrides: dict[str, list[str]] = config_overrides or {}
    strat = make_sonarr_sync(
        sonarr=sonarr,
        config=make_config(**overrides),
        cache_store=FakeCacheStore(),
    )
    return strat, sonarr


def _queue_record(infohash: str, state: str, *, status: str = "ok") -> QueueRecord:
    """One Sonarr queue record matching a download by infohash + tracked state.

    Built through ``QueueRecord.from_api`` from the raw API field names so the
    record mirrors exactly what ``SonarrClient.queue`` parses at the boundary.
    """

    return QueueRecord.from_api(
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
        # after issuing the probe reads RETRY + command_issued (NOT files_present);
        # a later monitor cycle flips to files_present once the episode files land.
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
        # (RETRY + command_issued); once the copy lands, poll 2 must observe the
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
    """A ManualImport command whose one file carries ``infohash`` as downloadId."""

    return CommandResource.from_api(
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
        # authoritative episode/series info; the payload must ignore those.
        candidate = manual_candidate(
            "/downloads/Show - 01 [1080p].mkv",
            quality={"quality": {"name": "HDTV-720p"}},
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        probe = strat.import_completed(pending, "/downloads/Show")

        # The command was issued; the copy is async, so the probe is RETRY +
        # command_issued (not yet files_present) right after issuing.
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert sonarr.candidate_calls == [pending]
        files = sonarr.execute_calls[0][0]
        assert len(files) == 1
        entry = files[0]
        assert entry["seriesId"] == 7
        assert entry["episodeIds"] == [101]
        assert entry["releaseGroup"] == "SubGroup"
        assert entry["downloadId"] == "HASH"
        assert entry["path"] == "/downloads/Show - 01 [1080p].mkv"

    def test_sonarr_structured_quality_wins_over_ours(self) -> None:
        # Sonarr already parsed the release as (bluray, 1080); our filename parse
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
        quality_defs: list[QualityDefinition] = [
            {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
            {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=quality_defs,
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        entry = sonarr.execute_calls[0][0][0]
        quality = entry.get("quality")
        assert quality is not None
        inner = quality.get("quality")
        assert inner is not None
        assert inner.get("name") == "Bluray-1080p"
        revision = quality.get("revision")
        assert revision is not None
        assert revision.get("version") == 1

    def test_our_parse_fills_when_sonarr_quality_unknown(self) -> None:
        # Sonarr couldn't parse the release (Unknown); our filename parse of
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
        quality_defs: list[QualityDefinition] = [
            {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
            {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=quality_defs,
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        entry = sonarr.execute_calls[0][0][0]
        quality = entry.get("quality")
        assert quality is not None
        inner = quality.get("quality")
        assert inner is not None
        assert inner.get("name") == "WEBDL-1080p"

    def test_matches_disk_name_across_nfd_normalization(self) -> None:
        # The seed map is keyed by an NFC name; the on-disk leaf arrives NFD
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
        assert sonarr.execute_calls[0][0][0]["episodeIds"] == [101]

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
        paths = [f["path"] for f in sonarr.execute_calls[0][0]]
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

        # The good file's import command was issued (RETRY + command_issued); the
        # sample is never queued.
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        paths = [f["path"] for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_already_imported_rejection_does_not_skip_missing_group_file(self) -> None:
        # Bug fix (grab-then-skip): the episode already holds a MISSING-GROUP file
        # (UNKNOWN_GROUP -> still needing our recommended import), and Sonarr offers
        # the candidate WITH an "already imported" rejection (it fires whenever the
        # episode has any file on disk). That rejection must NOT veto our import:
        # we grabbed this exactly to replace the unidentifiable file, so we step in
        # and ISSUE the command (RETRY + command_issued; the copy is async).
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
        # poll 1 imports over a missing-group file (RETRY + command_issued); the
        # imported file now carries OUR group, so poll 2 reads it as RECOMMENDED ->
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
        # A different file is on disk; ours isn't there yet.
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
        languages: list[Language] = [
            {"id": 1, "name": "English"},
            {"id": 8, "name": "Japanese"},
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            languages=languages,
            config_overrides={"import_languages_dual": ["Japanese", "English"]},
        )

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        names = [lang["name"] for lang in sonarr.execute_calls[0][0][0]["languages"]]
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
        # The execute WAS attempted (Sonarr rejected it -> command_issued False); a
        # regression returning RETRY without even trying would still set False here.
        assert len(sonarr.execute_calls) == 1

    def test_quality_defs_and_languages_cached_per_run(self) -> None:
        # Quality definitions + languages are fetched lazily ONCE and cached on the
        # executor for the rest of the run. Run 1 caches the (a) values; before run 2
        # the source changes to (b), but a cached run must not re-fetch, so both polls
        # keep the (a) values. Drop the lazy-fetch guard and run 2 refetches (b),
        # flipping the executor's cache to (b) -> these assertions fail.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        defs_a: list[QualityDefinition] = [
            {"quality": {"id": 1, "name": "A", "source": "web", "resolution": 1080}},
        ]
        defs_b: list[QualityDefinition] = [
            {"quality": {"id": 2, "name": "B", "source": "bluray", "resolution": 1080}},
        ]
        langs_a: list[Language] = [{"id": 1, "name": "English"}]
        langs_b: list[Language] = [{"id": 8, "name": "Japanese"}]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate],
            quality_defs=defs_a,
            languages=langs_a,
        )

        strat.import_completed(pending, "/d")

        # The source changes between polls; a cached run must ignore it.
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


class TestManualImportWarningGating:
    """The import warns loudly only at the deadline; otherwise it's debug.

    A missing intended file on an early poll is expected (the copy hasn't landed),
    so it must NOT inflate the summary's warning count; only the final attempt
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
        logger = logging.getLogger("seadexarr-warning-gating")
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.DEBUG)
        strat.logger = logger
        strat._executor.logger = logger
        return strat, pending

    def test_missing_off_deadline_is_debug_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, pending = self._strat_with_missing_file()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=False)

        assert probe.readiness is ImportReadiness.RETRY
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not any("not visible to Sonarr" in r.message for r in warnings)
        assert any("not visible to Sonarr" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    def test_missing_at_deadline_warns_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        strat, pending = self._strat_with_missing_file()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=True)

        assert probe.readiness is ImportReadiness.RETRY
        assert any("not visible to Sonarr" in r.message and r.levelname == "WARNING" for r in caplog.records)


class TestResolveLanguageObjects:
    """resolve_language_objects maps names to {id,name}, dropping unknowns."""

    def test_resolves_in_request_order(self) -> None:
        defs: list[Language] = [
            {"id": 1, "name": "English"},
            {"id": 8, "name": "Japanese"},
        ]

        result = resolve_language_objects(["Japanese", "English"], defs)

        assert result == [
            {"id": 8, "name": "Japanese"},
            {"id": 1, "name": "English"},
        ]

    def test_skips_unknown_names(self) -> None:
        defs: list[Language] = [{"id": 8, "name": "Japanese"}]

        result = resolve_language_objects(["Japanese", "Klingon"], defs)

        assert result == [{"id": 8, "name": "Japanese"}]

    def test_matches_case_insensitively(self) -> None:
        defs: list[Language] = [{"id": 8, "name": "Japanese"}]

        result = resolve_language_objects(["japanese"], defs)

        assert result == [{"id": 8, "name": "Japanese"}]


class _FakeRadarr:
    """Minimal Radarr client: scripts the per-movie movie-file list."""

    def __init__(self, files: list[MovieFile]) -> None:
        self._files = files

    def movie_files(self, movie_id: int) -> list[MovieFile]:
        del movie_id
        return self._files


class TestRadarrReleaseDict:
    """get_radarr_release_dict accumulates sizes per group and never hard-errors."""

    def test_multiple_distinct_groups_kept_not_errored(self) -> None:
        # VU3: 2 distinct groups no longer raise (which skipped the movie every run);
        # the dict carries both so the planner dedups against each.
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
