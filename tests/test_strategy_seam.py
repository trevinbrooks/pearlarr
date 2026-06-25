"""Seam tests for the composition split.

These pin the contract between the run machinery and the Arr strategies: each
``ArrSync`` hook reaches the shared pipeline only through the injected
``RunServices`` the strategy holds as ``self._services``. The strategies are
built bare (``object.__new__``) so no live Sonarr/Radarr client is constructed.
"""

import logging
import types
from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import (
    ImportReadiness,
    PendingImport,
    resolve_language_objects,
    resolve_quality_model,
)
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import (
    ManualImportCandidate,
    RadarrItem,
    SonarrEpisode,
    SonarrItem,
)

from .builders import (
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
            tmdb_id=42, imdb_id="tt7", tmdb_type="movie", log_ignored=False,
        )

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        run = mock.MagicMock()
        strat = make_bare_instance(SonarrSync, _services=run)

        strat.item_anilist_ids(_Item(tvdbId=99, imdbId="tt9"))

        run.get_anilist_ids.assert_called_once_with(
            tvdb_id=99, imdb_id="tt9", log_ignored=True,
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
    """get_items doubles as the run-start hook: it resets the per-run scratch."""

    def test_sonarr_get_items_clears_ep_list_cache(self) -> None:
        strat = make_bare_instance(SonarrSync, _ep_list_cache={5: ["stale"]})
        strat.get_all_sonarr_series = mock.MagicMock(return_value=["series"])

        result = strat.get_items()

        assert result == ["series"]
        assert strat._ep_list_cache == {}


class TestProcessAlIdThreadsServices:
    """The per-id head runs through the held services; a missing entry stops this id."""

    def test_radarr_no_seadex_entry_returns_false(self) -> None:
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None
        strat = make_bare_instance(RadarrSync, _services=run)

        assert strat.process_al_id(Arr.RADARR, _Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        run.al_id_prologue.assert_called_once_with(5)

    def test_sonarr_no_seadex_entry_returns_false(self) -> None:
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None
        strat = make_bare_instance(SonarrSync, _services=run)

        assert strat.process_al_id(Arr.SONARR, _Item(id=1), "Title", 5, MappingEntry(anilist_id=5)) is False
        run.al_id_prologue.assert_called_once_with(5)


def _ep_with_file(ep_id: int, *, group: str | None) -> SonarrEpisode:
    """A current Sonarr episode that already holds a file from ``group``."""

    return SonarrEpisode.from_api(
        {"id": ep_id, "episodeFileId": ep_id * 10, "episodeFile": {"releaseGroup": group}},
    )


def _make_sonarr_for_import(
    *,
    candidates: list[ManualImportCandidate] | None,
    queue: list[dict] | None = None,
    episodes: list[SonarrEpisode] | None = None,
    quality_defs: list[dict] | None = None,
    languages: list[dict] | None = None,
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
    isn't tracking the download, so the strategy steps in). ``refresh`` /
    ``command_status`` resolve immediately so the rescan never really waits.
    """

    sonarr = mock.MagicMock()
    sonarr.queue.return_value = queue or []
    sonarr.episodes.return_value = episodes if episodes is not None else []
    sonarr.parse.return_value = []
    sonarr.refresh_monitored_downloads.return_value = 7
    sonarr.command_status.return_value = {"status": "completed"}
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
        cache_store=types.SimpleNamespace(data={}),
    )
    return strat, sonarr


def _queue_record(
    infohash: str, state: str, *, status: str = "ok", messages: list | None = None,
) -> dict:
    """One Sonarr queue record matching a download by infohash + tracked state."""

    return {
        "downloadId": infohash,
        "trackedDownloadState": state,
        "trackedDownloadStatus": status,
        "statusMessages": messages or [],
    }


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


class TestImportCompletedPayload:
    """import_completed assigns episodes from OUR map, never Sonarr's parse."""

    def test_payload_uses_pending_fields_not_candidate(self) -> None:
        pending = pending_import(
            series_id=7,
            release_group="SubGroup",
            infohash="HASH",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
            season_number=1,
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

    def test_our_regex_quality_wins_over_candidate(self) -> None:
        # Filename says 1080p WEB-DL -> our parse yields "WEBDL-1080p"; the
        # candidate's in-context "HDTV-720p" must lose.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p][WEB-DL].mkv": [101]},
            episode_ids=[101],
            seadex_files=["Show - 01 [1080p][WEB-DL].mkv"],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p][WEB-DL].mkv",
            quality={"quality": {"name": "HDTV-720p"}},
        )
        quality_defs = [
            {"quality": {"id": 3, "name": "WEBDL-1080p", "resolution": 1080}},
        ]
        strat, sonarr = _make_sonarr_for_import(
            candidates=[candidate], quality_defs=quality_defs,
        )

        strat.import_completed(pending, "/d")

        (_, kwargs) = sonarr.manual_import_execute.call_args
        entry = kwargs["files"][0]
        assert entry["quality"]["quality"]["name"] == "WEBDL-1080p"
        assert entry["quality"]["revision"]["version"] == 1

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

    def test_already_imported_candidate_drops_record(self) -> None:
        # The only candidate is one Sonarr already imported itself -> nothing to
        # queue, but it IS placed, so the files are verified present and the record
        # is dropped (IMPORTED + files_present).
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        probe = strat.import_completed(pending, "/d")

        assert probe.readiness is ImportReadiness.IMPORTED
        assert probe.files_present is True
        sonarr.manual_import_execute.assert_not_called()

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
        languages = [
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
    """_manual_import warns loudly only at the deadline; otherwise it's debug.

    A missing intended file on an early poll is expected (the copy hasn't landed),
    so it must NOT inflate the summary's warning count; only the final attempt
    (at_deadline) warns loudly that a still-missing file is terminal.
    """

    @staticmethod
    def _strat_with_missing_file() -> tuple[SonarrSync, "PendingImport"]:
        # Our map intends a file that isn't on disk yet (only an unrelated file is),
        # so _manual_import always finds it missing and retries.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, _ = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Unrelated.mkv")],
        )
        # caplog captures via the root logger's propagation, so use a propagating
        # DEBUG logger here (make_logger disables propagation for quiet runs).
        logger = logging.getLogger("seadexarr-warning-gating")
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.DEBUG)
        strat.logger = logger
        return strat, pending

    def test_missing_off_deadline_is_debug_not_warning(self, caplog) -> None:
        strat, pending = self._strat_with_missing_file()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=False)

        assert probe.readiness is ImportReadiness.RETRY
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not any("not visible to Sonarr" in r.message for r in warnings)
        assert any(
            "not visible to Sonarr" in r.message and r.levelname == "DEBUG"
            for r in caplog.records
        )

    def test_missing_at_deadline_warns_loudly(self, caplog) -> None:
        strat, pending = self._strat_with_missing_file()

        with caplog.at_level("DEBUG"):
            probe = strat.import_completed(pending, "/d", at_deadline=True)

        assert probe.readiness is ImportReadiness.RETRY
        assert any(
            "not visible to Sonarr" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )


class TestResolveQualityModel:
    """resolve_quality_model maps a name to a QualityModel, case-insensitively."""

    def test_matches_case_insensitively(self) -> None:
        defs = [
            {"quality": {"id": 1, "name": "HDTV-720p"}},
            {"quality": {"id": 3, "name": "WEBDL-1080p"}},
        ]

        model = resolve_quality_model("webdl-1080p", defs)

        assert model is not None
        quality = model.get("quality")
        assert quality is not None
        assert quality.get("id") == 3
        assert quality.get("name") == "WEBDL-1080p"
        assert model.get("revision") == {"version": 1, "real": 0, "isRepack": False}

    def test_no_match_returns_none(self) -> None:
        defs = [{"quality": {"id": 1, "name": "HDTV-720p"}}]

        assert resolve_quality_model("Bluray-2160p", defs) is None

    def test_empty_defs_returns_none(self) -> None:
        assert resolve_quality_model("WEBDL-1080p", []) is None


class TestResolveLanguageObjects:
    """resolve_language_objects maps names to {id,name}, dropping unknowns."""

    def test_resolves_in_request_order(self) -> None:
        defs = [
            {"id": 1, "name": "English"},
            {"id": 8, "name": "Japanese"},
        ]

        result = resolve_language_objects(["Japanese", "English"], defs)

        assert result == [
            {"id": 8, "name": "Japanese"},
            {"id": 1, "name": "English"},
        ]

    def test_skips_unknown_names(self) -> None:
        defs = [{"id": 8, "name": "Japanese"}]

        result = resolve_language_objects(["Japanese", "Klingon"], defs)

        assert result == [{"id": 8, "name": "Japanese"}]

    def test_matches_case_insensitively(self) -> None:
        defs = [{"id": 8, "name": "Japanese"}]

        result = resolve_language_objects(["japanese"], defs)

        assert result == [{"id": 8, "name": "Japanese"}]
