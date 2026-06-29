"""Seam tests for the composition split.

These pin the contract between the run machinery and the Arr strategies: each
``ArrSync`` hook reaches the shared pipeline only through the injected
``RunServices`` the strategy holds as ``self._services``. The strategies are
built bare (``object.__new__``) so no live Sonarr/Radarr client is constructed.
"""

import logging
from types import SimpleNamespace
from unittest import mock

from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import (
    ImportReadiness,
    PendingImport,
    resolve_language_objects,
)
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    CommandResource,
    Language,
    ManualImportCandidate,
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
    make_logger,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
)


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


class TestItemAnilistIdsDelegates:
    """item_anilist_ids resolves through the held services, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        run = mock.MagicMock()
        run.get_anilist_ids.return_value = {7: {}}
        strat = make_bare_instance(RadarrSync, _services=run)

        result = strat.item_anilist_ids(_Item(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: {}}
        run.get_anilist_ids.assert_called_once_with(
            tmdb_id=42,
            imdb_id="tt7",
            tmdb_type="movie",
            log_ignored=False,
        )

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = mock.MagicMock()
        strat = make_bare_instance(SonarrSync, _services=run)

        strat.item_anilist_ids(_Item(tvdbId=99, imdbId="tt9"))

        run.get_anilist_ids.assert_called_once_with(
            tvdb_id=99,
            imdb_id="tt9",
            log_ignored=True,
        )


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
        series = [_Item(id=5), _Item(id=7)]
        strat = make_sonarr_sync(_ep_list_cache={5: ["stale"]})
        strat._episodes.get_all_sonarr_series = mock.MagicMock(return_value=series)

        result = strat.get_items()

        assert result == series
        # The episode collaborator drops its cache + re-fingerprints as it enumerates.
        assert strat._episodes._ep_list_cache == {}
        assert strat._episodes._series_fp == sonarr_series_fingerprint([5, 7])


class TestSonarrPrefetchDelegates:
    """prefetch_episodes is a thin hook over the episode collaborator's warm."""

    def test_sonarr_prefetch_routes_to_episodes(self) -> None:
        strat = make_sonarr_sync()
        strat._episodes.prefetch = mock.MagicMock(return_value=3)
        items: list[SonarrItem] = [_Item(id=1)]

        assert strat.prefetch_episodes(items, progress=None) == 3
        strat._episodes.prefetch.assert_called_once_with(items, progress=None)


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
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None
        strat = make_bare_instance(RadarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        run.al_id_prologue.assert_called_once_with(5)

    def test_sonarr_no_seadex_entry_returns_false(self) -> None:
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None
        strat = make_bare_instance(SonarrSync, _services=run)

        assert strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        run.al_id_prologue.assert_called_once_with(5)

    def test_sonarr_no_episodes_resolved_skips_explicitly(self) -> None:
        # An anime-id mapping that resolves to [] (season not in Sonarr / offset past
        # the end): skip with the NO_EPISODES status, never mislabeled "unmonitored"
        # and never falling through to grab orphans - and NO AniBridge warning.
        run = mock.MagicMock()
        run.al_id_prologue.return_value = mock.MagicMock()  # a SeaDex entry exists
        run.cached_entry_skip.return_value = False
        run.get_anilist_title.return_value = "Title"
        episodes = mock.MagicMock()
        episodes.get_ep_list.return_value = []
        logger = mock.MagicMock()
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
        run.log_entry_status.assert_called_once_with(EntryState.NO_EPISODES, "Title")
        run.log_al_title.assert_not_called()
        logger.warning.assert_not_called()  # anime-id empty is NOT the AniBridge case

    def test_sonarr_anibridge_empty_map_skips_with_warning(self) -> None:
        # The AniBridge no-usable-ranges case (tvdb_mappings={} -> mode ANIBRIDGE): the
        # NO_EPISODES skip PLUS a visible WARNING naming the cause. Fails on the unfixed
        # path, which silently grabbed nothing / mislabeled the entry.
        run = mock.MagicMock()
        run.al_id_prologue.return_value = mock.MagicMock()
        run.cached_entry_skip.return_value = False
        run.get_anilist_title.return_value = "Title"
        episodes = mock.MagicMock()
        episodes.get_ep_list.return_value = []
        logger = mock.MagicMock()
        strat = make_bare_instance(
            SonarrSync,
            _services=run,
            _episodes=episodes,
            _config=make_config(sleep_time=0),
            ignore_movies_in_radarr=False,
            logger=logger,
        )

        result = strat.process_al_id(_Item(id=1), "Title", 5, MappingEntry(anilist_id=5, tvdb_mappings={}))

        assert result is False
        run.log_entry_status.assert_called_once_with(EntryState.NO_EPISODES, "Title")
        run.log_al_title.assert_not_called()
        assert logger.warning.called  # AniBridge-specific notice surfaced


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
    config_overrides: dict | None = None,
) -> tuple[SonarrSync, mock.MagicMock]:
    """A bare ``SonarrSync`` plus its scripted ``self.sonarr`` MagicMock.

    The mock returns the given queue records, current episodes, manual-import
    candidates, quality definitions, languages and an execute command id -
    everything ``import_completed`` reaches over the network. ``episodes`` defaults
    to empty, so the target episodes have NO file yet (they need importing) and the
    done-check never short-circuits; pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. ``queue`` defaults to empty (Sonarr
    isn't tracking the download, so the strategy steps in). ``commands`` defaults to
    empty (no in-flight ManualImport, so the dedup guard never trips). ``refresh`` /
    ``command_status`` resolve immediately so the rescan never really waits.
    """

    sonarr = mock.MagicMock()
    sonarr.queue.return_value = queue or []
    sonarr.list_commands.return_value = commands or []
    sonarr.episodes.return_value = episodes if episodes is not None else []
    sonarr.parse.return_value = []
    sonarr.refresh_monitored_downloads.return_value = 7
    sonarr.command_status.return_value = CommandResource(status="completed")
    sonarr.manual_import_candidates.return_value = candidates
    sonarr.quality_definitions.return_value = quality_defs or []
    sonarr.languages.return_value = languages or []
    sonarr.manual_import_execute.return_value = cmd_id
    strat = make_sonarr_sync(
        sonarr=sonarr,
        logger=make_logger(),
        log_fmt=mock.MagicMock(),
        _config=make_config(**(config_overrides or {})),
        _last_refresh_monotonic=None,
        _ep_list_cache={},
        _parse_info_cache={},
        _warned_unplaceable=set(),
        cache_store=FakeCacheStore(),
    )
    return strat, sonarr


def _queue_record(
    infohash: str,
    state: str,
    *,
    status: str = "ok",
    messages: list | None = None,
) -> QueueRecord:
    """One Sonarr queue record matching a download by infohash + tracked state.

    Built through ``QueueRecord.from_api`` from the raw API field names so the
    record mirrors exactly what ``SonarrClient.queue`` parses at the boundary.
    """

    return QueueRecord.from_api(
        {
            "downloadId": infohash,
            "trackedDownloadState": state,
            "trackedDownloadStatus": status,
            "statusMessages": messages or [],
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
        sonarr.manual_import_candidates.assert_not_called()
        sonarr.manual_import_execute.assert_not_called()

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
        sonarr.manual_import_candidates.assert_not_called()

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
        sonarr.manual_import_execute.assert_called_once()

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
        sonarr.manual_import_execute.assert_not_called()

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
        sonarr.manual_import_candidates.assert_not_called()

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
        sonarr.manual_import_candidates.assert_called_once()
        sonarr.manual_import_execute.assert_called_once()

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
        sonarr.episodes.return_value = [_ep_with_file(101, group="SubGroup")]

        second = strat.import_completed(pending, "/d")
        assert second.readiness is ImportReadiness.IMPORTED
        assert second.files_present is True
        # Episodes were re-read fresh each poll (not cached), and the landed import
        # was detected before any second execute.
        assert sonarr.episodes.call_count == 2
        sonarr.manual_import_execute.assert_called_once()

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
        sonarr.manual_import_candidates.assert_called_once()


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
        sonarr.manual_import_execute.assert_not_called()
        sonarr.manual_import_candidates.assert_not_called()

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
        sonarr.manual_import_execute.assert_not_called()

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
        sonarr.manual_import_execute.assert_called_once()

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

        assert probe.command_issued is True
        sonarr.manual_import_execute.assert_called_once()


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
        sonarr.manual_import_candidates.assert_called_once_with(
            pending=pending,
            filter_existing_files=False,
        )
        (_, kwargs) = sonarr.manual_import_execute.call_args
        files = kwargs["files"]
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

        strat.import_completed(pending, "/d")

        (_, kwargs) = sonarr.manual_import_execute.call_args
        entry = kwargs["files"][0]
        assert entry["quality"]["quality"]["name"] == "Bluray-1080p"
        assert entry["quality"]["revision"]["version"] == 1

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

        strat.import_completed(pending, "/d")

        (_, kwargs) = sonarr.manual_import_execute.call_args
        entry = kwargs["files"][0]
        assert entry["quality"]["quality"]["name"] == "WEBDL-1080p"

    def test_matches_disk_name_across_nfd_normalization(self) -> None:
        # The seed map is keyed by an NFC name; the on-disk leaf arrives NFD
        # (macOS). Normalization on both sides still matches -> the file imports,
        # never "no authoritative mapping".
        nfc = "Café - 01 [1080p].mkv"  # composed e-acute
        nfd = "Café - 01 [1080p].mkv"  # decomposed
        pending = pending_import(
            file_episode_map={nfc: [101]},
            episode_ids=[101],
            seadex_files=[nfc],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[manual_candidate(f"/d/{nfd}")])

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        (_, kwargs) = sonarr.manual_import_execute.call_args
        assert kwargs["files"][0]["episodeIds"] == [101]

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

        strat.import_completed(pending, "/d")

        (_, kwargs) = sonarr.manual_import_execute.call_args
        paths = [f["path"] for f in kwargs["files"]]
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
        (_, kwargs) = sonarr.manual_import_execute.call_args
        paths = [f["path"] for f in kwargs["files"]]
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
        sonarr.manual_import_execute.assert_called_once()

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
        sonarr.episodes.return_value = [_ep_with_file(101, group="SubGroup")]

        second = strat.import_completed(pending, "/d")
        assert second.readiness is ImportReadiness.IMPORTED
        assert second.files_present is True
        sonarr.manual_import_execute.assert_called_once()

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
        sonarr.manual_import_execute.assert_not_called()

    def test_transient_candidate_scan_retries(self) -> None:
        # A None candidates result (timeout / non-200) is transient -> retry.
        strat, sonarr = _make_sonarr_for_import(candidates=None)

        probe = strat.import_completed(pending_import(), "/d")
        assert probe.readiness is ImportReadiness.RETRY
        sonarr.manual_import_execute.assert_not_called()

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

        strat.import_completed(pending, "/d")

        (_, kwargs) = sonarr.manual_import_execute.call_args
        names = [lang["name"] for lang in kwargs["files"][0]["languages"]]
        assert names == ["Japanese", "English"]

    def test_failed_execute_retries(self) -> None:
        # Sonarr rejected the import command (busy / locked) -> retry, not give up.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        strat, _ = _make_sonarr_for_import(candidates=[candidate], cmd_id=None)

        probe = strat.import_completed(pending, "/d")
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is False

    def test_quality_defs_and_languages_cached_per_run(self) -> None:
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        strat.import_completed(pending, "/d")
        strat.import_completed(pending, "/d")

        # Fetched lazily once and reused for the rest of the run.
        assert sonarr.quality_definitions.call_count == 1
        assert sonarr.languages.call_count == 1


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
    def _strat_with_missing_file() -> tuple[SonarrSync, "PendingImport"]:
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

    def test_missing_off_deadline_is_debug_not_warning(self, caplog) -> None:
        strat, pending = self._strat_with_missing_file()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=False)

        assert probe.readiness is ImportReadiness.RETRY
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not any("not visible to Sonarr" in r.message for r in warnings)
        assert any("not visible to Sonarr" in r.message and r.levelname == "DEBUG" for r in caplog.records)

    def test_missing_at_deadline_warns_loudly(self, caplog) -> None:
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


class TestRadarrReleaseDict:
    """get_radarr_release_dict accumulates sizes per group and never hard-errors."""

    def test_multiple_distinct_groups_kept_not_errored(self) -> None:
        # VU3: 2 distinct groups no longer raise (which skipped the movie every run);
        # the dict carries both so the planner dedups against each.
        radarr = mock.MagicMock()
        radarr.movie_files.return_value = [
            SimpleNamespace(release_group="A", size=100),
            SimpleNamespace(release_group="B", size=200),
        ]
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {"A": [100], "B": [200]}

    def test_same_group_sizes_accumulate(self) -> None:
        # CB6: two files of one group keep BOTH sizes (the old comprehension collapsed
        # to the last).
        radarr = mock.MagicMock()
        radarr.movie_files.return_value = [
            SimpleNamespace(release_group="A", size=100),
            SimpleNamespace(release_group="A", size=200),
        ]
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {"A": [100, 200]}

    def test_no_files_returns_none_marker(self) -> None:
        radarr = mock.MagicMock()
        radarr.movie_files.return_value = []
        strat = make_bare_instance(RadarrSync, radarr=radarr)

        assert strat.get_radarr_release_dict(7) == {None: [None]}
