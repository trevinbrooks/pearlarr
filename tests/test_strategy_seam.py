"""Seam tests for the composition split (see ``REFACTOR_PLAN.md``).

These pin the contract between the run machinery and the Arr strategies: each
``ArrSync`` hook reaches the shared pipeline only through the injected
``RunServices`` the strategy holds as ``self._services``. The strategies are
built bare (``object.__new__``) so no live Sonarr/Radarr client is constructed.
"""

from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import (
    ImportReadiness,
    resolve_language_objects,
    resolve_quality_model,
)
from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync
from seadexarr.modules.seadex_types import RadarrItem, SonarrItem

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


def _make_sonarr_for_import(
    *,
    candidates: list[dict] | None,
    queue: list[dict] | None = None,
    quality_defs: list[dict] | None = None,
    languages: list[dict] | None = None,
    cmd_id: int | None = 42,
    config_overrides: dict | None = None,
) -> tuple[SonarrSync, mock.MagicMock]:
    """A bare ``SonarrSync`` plus its scripted ``self.sonarr`` MagicMock.

    The mock returns the given queue records, manual-import candidates, quality
    definitions and languages, and a command id from ``manual_import_execute`` -
    everything ``import_completed`` reaches over the network - so the test can
    assert on the payload without a live Sonarr. ``queue`` defaults to empty (so
    Sonarr isn't tracking the download and the strategy steps in with its own
    manual import); ``refresh_monitored_downloads`` / ``command_status`` are
    stubbed to return immediately so the rescan never really waits. The mock is
    returned alongside the strategy so assertions read it through a
    ``MagicMock``-typed handle (``strat.sonarr`` is statically a ``SonarrClient``).
    """

    sonarr = mock.MagicMock()
    sonarr.queue.return_value = queue or []
    sonarr.refresh_monitored_downloads.return_value = 7
    sonarr.command_status.return_value = {"status": "completed"}
    sonarr.manual_import_candidates.return_value = candidates
    sonarr.quality_definitions.return_value = quality_defs or []
    sonarr.languages.return_value = languages or []
    sonarr.manual_import_execute.return_value = cmd_id
    strat = make_sonarr_sync(
        sonarr=sonarr,
        logger=make_logger(),
        _config=make_config(**(config_overrides or {})),
        _last_refresh_monotonic=None,
    )
    return strat, sonarr


def _queue_record(infohash: str, state: str) -> dict:
    """One Sonarr queue record matching a download by infohash + tracked state."""

    return {"downloadId": infohash, "trackedDownloadState": state}


class TestImportCompletedQueueState:
    """import_completed branches on Sonarr's queue before stepping in itself."""

    def test_sonarr_importing_retries_without_stepping_in(self) -> None:
        # Sonarr is mid-import (importing) -> wait for it, don't double-import.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importing")],
        )

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.RETRY
        sonarr.manual_import_candidates.assert_not_called()
        sonarr.manual_import_execute.assert_not_called()

    def test_sonarr_imported_drops_record(self) -> None:
        # Sonarr already imported it -> drop the record, never step in.
        pending = pending_import(infohash="abc123")
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "imported")],
        )

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.IMPORTED
        sonarr.manual_import_candidates.assert_not_called()

    def test_import_blocked_steps_in_with_our_mapping(self) -> None:
        # Sonarr can't auto-import (importBlocked) -> our authoritative manual
        # import takes over and queues the command.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[_queue_record("ABC123", "importBlocked")],
        )

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.IMPORTED
        sonarr.manual_import_candidates.assert_called_once()
        sonarr.manual_import_execute.assert_called_once()

    def test_not_in_queue_steps_in(self) -> None:
        # Sonarr isn't tracking the download (our holding category) -> step in.
        pending = pending_import(
            infohash="abc123",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[101],
        )
        strat, sonarr = _make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            queue=[],
        )

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.IMPORTED
        sonarr.manual_import_candidates.assert_called_once()


class TestImportCompletedPayload:
    """import_completed overrides every field from PendingImport, not the parse."""

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

        result = strat.import_completed(pending, "/downloads/Show")

        assert result is ImportReadiness.IMPORTED
        sonarr.manual_import_candidates.assert_called_once_with(
            folder="/downloads/Show",
            series_id=7,
            season_number=1,
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

    def test_sample_candidate_is_skipped(self) -> None:
        pending = pending_import(
            file_episode_map={
                "Show - 01 [1080p].mkv": [101],
                "Show - 01 [1080p].sample.mkv": [101],
            },
            episode_ids=[],
        )
        good = manual_candidate("/d/Show - 01 [1080p].mkv")
        sample = manual_candidate(
            "/d/Show - 01 [1080p].sample.mkv",
            rejections=[{"reason": "Sample"}],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[good, sample])

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.IMPORTED
        (_, kwargs) = sonarr.manual_import_execute.call_args
        paths = [f["path"] for f in kwargs["files"]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_already_imported_candidate_drops_record(self) -> None:
        # The only candidate is one Sonarr already imported itself -> nothing for
        # us to do, but it IS imported, so drop the record (don't re-attempt).
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            episode_ids=[],
        )
        candidate = manual_candidate(
            "/d/Show - 01 [1080p].mkv",
            rejections=["Episode file already imported"],
        )
        strat, sonarr = _make_sonarr_for_import(candidates=[candidate])

        result = strat.import_completed(pending, "/d")

        assert result is ImportReadiness.IMPORTED
        sonarr.manual_import_execute.assert_not_called()

    def test_no_candidates_yet_retries(self) -> None:
        # Sonarr reports no files at the path yet (mount not visible) -> retry.
        strat, sonarr = _make_sonarr_for_import(candidates=[])

        assert strat.import_completed(pending_import(), "/d") is ImportReadiness.RETRY
        sonarr.manual_import_execute.assert_not_called()

    def test_transient_candidate_scan_retries(self) -> None:
        # A None candidates result (timeout / non-200) is transient -> retry.
        strat, sonarr = _make_sonarr_for_import(candidates=None)

        assert strat.import_completed(pending_import(), "/d") is ImportReadiness.RETRY
        sonarr.manual_import_execute.assert_not_called()

    def test_unmapped_file_leaves_pending(self) -> None:
        # Candidate basename isn't in the map and there's no single-file
        # fallback (two unmatched), so it's skipped -> nothing importable. Retrying
        # the same files won't help, so leave it for a later run.
        pending = pending_import(
            file_episode_map={"Other.mkv": [101]},
            episode_ids=[],
        )
        candidates = [
            manual_candidate("/d/Mystery A.mkv"),
            manual_candidate("/d/Mystery B.mkv"),
        ]
        strat, sonarr = _make_sonarr_for_import(candidates=candidates)

        assert strat.import_completed(pending, "/d") is ImportReadiness.LEAVE
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
            config_overrides={
                "import_languages_dual": ["Japanese", "English"],
            },
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

        assert strat.import_completed(pending, "/d") is ImportReadiness.RETRY

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

        assert strat.import_completed(pending_import(), "/d") is ImportReadiness.LEAVE


class TestResolveQualityModel:
    """resolve_quality_model maps a name to a QualityModel, case-insensitively."""

    def test_matches_case_insensitively(self) -> None:
        defs = [
            {"quality": {"id": 1, "name": "HDTV-720p"}},
            {"quality": {"id": 3, "name": "WEBDL-1080p"}},
        ]

        model = resolve_quality_model("webdl-1080p", defs)

        assert model is not None
        assert model["quality"]["id"] == 3
        assert model["quality"]["name"] == "WEBDL-1080p"
        assert model["revision"] == {"version": 1, "real": 0, "isRepack": False}

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
