"""Characterization tests for the pure manual-import helpers.

Pins the deterministic decision helpers in
:mod:`seadexarr.modules.manual_import`: the ``(season, episode) -> id`` map, the
filename quality parse, the layered quality/language/episode-id selection, the
wait-mode resolution, and the :class:`PendingImport` JSON round-trip. All pure,
no network or disk; :class:`SonarrEpisode` is built directly via
:meth:`SonarrEpisode.from_api` so the suite doesn't depend on ``builders.py``.
"""

from typing import cast

from seadexarr.modules.manual_import import (
    ImportReadiness,
    ImportWaitMode,
    PendingImport,
    QualitySelection,
    QueueVerdict,
    WaitOutcome,
    assign_episode_ids,
    build_episode_id_map,
    classify_queue_states,
    derive_languages,
    parse_quality_from_filename,
    resolve_language_objects,
    resolve_wait_mode,
    select_quality,
)
from seadexarr.modules.seadex_types import SONARR_MISSING_KEY, SonarrEpisode


def _ep(*, ep_id: int, season: int | None, episode: int | None) -> SonarrEpisode:
    """A ``SonarrEpisode`` from the raw fields the id map reads."""

    return SonarrEpisode.from_api(
        {"id": ep_id, "seasonNumber": season, "episodeNumber": episode},
    )


class TestBuildEpisodeIdMap:
    def test_normal_seasoned_episodes(self) -> None:
        eps = [
            _ep(ep_id=11, season=1, episode=1),
            _ep(ep_id=12, season=1, episode=2),
            _ep(ep_id=21, season=2, episode=1),
        ]
        assert build_episode_id_map(eps) == {
            (1, 1): 11,
            (1, 2): 12,
            (2, 1): 21,
        }

    def test_absolute_numbering_season(self) -> None:
        # A series using absolute numbering still keys on (season, episode);
        # season 0/1 with large episode numbers is just data.
        eps = [
            _ep(ep_id=101, season=1, episode=24),
            _ep(ep_id=102, season=1, episode=25),
        ]
        assert build_episode_id_map(eps) == {(1, 24): 101, (1, 25): 102}

    def test_missing_season_and_episode_use_sentinel_no_collision(self) -> None:
        eps = [
            _ep(ep_id=5, season=None, episode=None),
            _ep(ep_id=6, season=1, episode=1),
        ]
        result = build_episode_id_map(eps)
        assert result[(SONARR_MISSING_KEY, SONARR_MISSING_KEY)] == 5
        assert result[(1, 1)] == 6

    def test_first_wins_on_duplicate_key(self) -> None:
        eps = [
            _ep(ep_id=7, season=1, episode=1),
            _ep(ep_id=8, season=1, episode=1),
        ]
        assert build_episode_id_map(eps) == {(1, 1): 7}

    def test_zero_id_skipped(self) -> None:
        eps = [
            _ep(ep_id=0, season=1, episode=1),
            _ep(ep_id=9, season=1, episode=2),
        ]
        assert build_episode_id_map(eps) == {(1, 2): 9}


class TestParseQualityFromFilename:
    def test_2160p_webdl(self) -> None:
        name = "Show.S01E01.2160p.WEB-DL.x265.mkv"
        assert parse_quality_from_filename(name) == "WEBDL-2160p"

    def test_1080p_bluray(self) -> None:
        name = "[Group] Show - 01 [BluRay 1080p HEVC].mkv"
        assert parse_quality_from_filename(name) == "Bluray-1080p"

    def test_no_resolution_returns_none(self) -> None:
        assert parse_quality_from_filename("Show.S01E01.WEB-DL.mkv") is None

    def test_case_insensitive(self) -> None:
        assert parse_quality_from_filename("show 720p webrip.mkv") == "WEBRip-720p"
        assert parse_quality_from_filename("SHOW 480P HDTV.MKV") == "HDTV-480p"

    def test_remux_maps_to_remux_name(self) -> None:
        assert (
            parse_quality_from_filename("Show.2160p.BluRay.Remux.mkv") == "Remux-2160p"
        )

    def test_bare_web_treated_as_webdl(self) -> None:
        assert parse_quality_from_filename("Show.1080p.WEB.mkv") == "WEBDL-1080p"

    def test_no_source_defaults_to_webdl(self) -> None:
        assert parse_quality_from_filename("Show - 01 [1080p].mkv") == "WEBDL-1080p"


class TestSelectQuality:
    def test_ours_wins(self) -> None:
        sel = select_quality(
            our_name="Bluray-2160p",
            candidate_quality={"quality": {"name": "WEBDL-1080p"}},
            default_name="HDTV-720p",
        )
        assert sel == QualitySelection(source="ours", name="Bluray-2160p", model=None)

    def test_sonarr_in_context_when_no_ours(self) -> None:
        candidate = {"quality": {"name": "WEBDL-1080p"}}
        sel = select_quality(
            our_name=None, candidate_quality=candidate, default_name="HDTV-720p",
        )
        assert sel == QualitySelection(source="sonarr", name=None, model=candidate)

    def test_sonarr_nested_quality_quality_name(self) -> None:
        candidate = {"quality": {"quality": {"name": "Bluray-720p"}}}
        sel = select_quality(
            our_name=None, candidate_quality=candidate, default_name=None,
        )
        assert sel.source == "sonarr"
        assert sel.model is candidate

    def test_unknown_candidate_falls_through_to_default(self) -> None:
        candidate = {"quality": {"name": "Unknown"}}
        sel = select_quality(
            our_name=None, candidate_quality=candidate, default_name="HDTV-720p",
        )
        assert sel == QualitySelection(source="default", name="HDTV-720p", model=None)

    def test_missing_candidate_name_falls_through_to_default(self) -> None:
        sel = select_quality(
            our_name=None, candidate_quality={"quality": {}}, default_name="HDTV-720p",
        )
        assert sel == QualitySelection(source="default", name="HDTV-720p", model=None)

    def test_default_when_no_ours_no_candidate(self) -> None:
        sel = select_quality(
            our_name=None, candidate_quality=None, default_name="HDTV-720p",
        )
        assert sel == QualitySelection(source="default", name="HDTV-720p", model=None)

    def test_unknown_when_nothing_available(self) -> None:
        sel = select_quality(our_name=None, candidate_quality=None, default_name=None)
        assert sel == QualitySelection(source="unknown", name=None, model=None)

    def test_unknown_candidate_and_no_default_is_unknown(self) -> None:
        candidate = {"quality": {"name": "Unknown"}}
        sel = select_quality(
            our_name=None, candidate_quality=candidate, default_name=None,
        )
        assert sel == QualitySelection(source="unknown", name=None, model=None)


class TestDeriveLanguages:
    def test_dual_audio_returns_dual(self) -> None:
        assert derive_languages(True, ["Japanese", "English"], ["Japanese"]) == [
            "Japanese",
            "English",
        ]

    def test_single_audio_returns_single(self) -> None:
        assert derive_languages(False, ["Japanese", "English"], ["Japanese"]) == [
            "Japanese",
        ]


class TestResolveWaitMode:
    def test_cli_wins_over_config(self) -> None:
        assert (
            resolve_wait_mode(ImportWaitMode.OFF, ImportWaitMode.HYBRID)
            is ImportWaitMode.OFF
        )

    def test_config_used_when_no_cli(self) -> None:
        assert (
            resolve_wait_mode(None, ImportWaitMode.BLOCKING)
            is ImportWaitMode.BLOCKING
        )

    def test_default_off_when_neither(self) -> None:
        assert resolve_wait_mode(None, None) is ImportWaitMode.OFF


class TestAssignEpisodeIds:
    def test_basename_match(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["a.mkv", "b.mkv"],
            file_episode_map={"a.mkv": [11], "b.mkv": [12, 13]},
            flat_fallback=[],
        )
        assert result == {"a.mkv": [11], "b.mkv": [12, 13]}

    def test_single_file_fallback(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["only.mkv"],
            file_episode_map={},
            flat_fallback=[42],
        )
        assert result == {"only.mkv": [42]}

    def test_unmapped_omitted_when_multiple_unmatched(self) -> None:
        # Two unmatched files -> the single-file rule does not fire -> both omitted.
        result = assign_episode_ids(
            candidate_basenames=["x.mkv", "y.mkv"],
            file_episode_map={},
            flat_fallback=[42],
        )
        assert result == {}

    def test_mapped_and_single_unmatched_uses_fallback(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["a.mkv", "extra.mkv"],
            file_episode_map={"a.mkv": [11]},
            flat_fallback=[99],
        )
        assert result == {"a.mkv": [11], "extra.mkv": [99]}

    def test_zero_id_never_assigned_from_map(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["a.mkv"],
            file_episode_map={"a.mkv": [0]},
            flat_fallback=[],
        )
        assert result == {}

    def test_zero_id_stripped_from_fallback(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["only.mkv"],
            file_episode_map={},
            flat_fallback=[0],
        )
        assert result == {}

    def test_zero_ids_filtered_but_real_ids_kept(self) -> None:
        result = assign_episode_ids(
            candidate_basenames=["a.mkv"],
            file_episode_map={"a.mkv": [0, 7]},
            flat_fallback=[],
        )
        assert result == {"a.mkv": [7]}


class TestPendingImportRoundTrip:
    def test_to_json_from_json_round_trip(self) -> None:
        pending = PendingImport(
            infohash="abc123",
            series_id=55,
            file_episode_map={"ep1.mkv": [11], "ep2.mkv": [12]},
            episode_ids=[11, 12],
            release_group="Era-Raws",
            is_dual_audio=True,
            season_number=2,
            seadex_files=["ep1.mkv", "ep2.mkv"],
            title="Some Show",
            added_at="2026-06-24 12:00:00",
        )
        assert PendingImport.from_json(pending.to_json()) == pending

    def test_from_json_tolerates_missing_keys(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h", "series_id": 1})
        assert rebuilt.infohash == "h"
        assert rebuilt.series_id == 1
        assert rebuilt.file_episode_map == {}
        assert rebuilt.episode_ids == []
        assert rebuilt.release_group == ""
        assert rebuilt.is_dual_audio is False
        assert rebuilt.season_number is None
        assert rebuilt.seadex_files == []
        assert rebuilt.title is None
        assert rebuilt.added_at == ""

    def test_none_season_round_trips(self) -> None:
        pending = PendingImport(
            infohash="h",
            series_id=1,
            file_episode_map={},
            episode_ids=[],
            release_group="RG",
            is_dual_audio=False,
            season_number=None,
            seadex_files=[],
            title=None,
            added_at="2026-06-24 00:00:00",
        )
        assert PendingImport.from_json(pending.to_json()) == pending


def test_wait_outcome_members_exist() -> None:
    # The public enum surface wave-2 dispatch depends on.
    assert {o.name for o in WaitOutcome} == {
        "COMPLETE",
        "ERRORED",
        "TIMED_OUT",
        "MISSING",
    }


def test_import_readiness_members_exist() -> None:
    # The tri-state the engine's blocking loop dispatches on.
    assert {o.name for o in ImportReadiness} == {"IMPORTED", "RETRY", "LEAVE"}


class TestClassifyQueueStates:
    """classify_queue_states reduces per-episode tracked states to one verdict."""

    def test_empty_steps_in(self) -> None:
        # Sonarr isn't tracking the download -> we own the import.
        assert classify_queue_states([]) is QueueVerdict.STEP_IN

    def test_all_imported_is_done(self) -> None:
        assert classify_queue_states(["imported", "imported"]) is QueueVerdict.DONE

    def test_import_blocked_steps_in(self) -> None:
        # importBlocked wins even when other episodes are still importing.
        assert (
            classify_queue_states(["importing", "importBlocked"])
            is QueueVerdict.STEP_IN
        )

    def test_downloading_waits(self) -> None:
        assert classify_queue_states(["downloading"]) is QueueVerdict.WAIT

    def test_importing_waits(self) -> None:
        assert classify_queue_states(["importPending", "importing"]) is QueueVerdict.WAIT

    def test_active_beats_imported(self) -> None:
        # A partly-imported pack with one episode still importing -> keep waiting.
        assert classify_queue_states(["imported", "importing"]) is QueueVerdict.WAIT

    def test_failed_steps_in(self) -> None:
        # Sonarr gave up; we have the files + authoritative mapping -> step in.
        assert classify_queue_states(["failed"]) is QueueVerdict.STEP_IN

    def test_case_insensitive(self) -> None:
        assert classify_queue_states(["IMPORTBLOCKED"]) is QueueVerdict.STEP_IN


class TestResolveLanguageObjectsDefensive:
    """resolve_language_objects survives a blank/None or malformed name list.

    Regression for the crash where blank YAML import_languages_* parsed to None
    and the helper did ``for name in None`` -> TypeError.
    """

    def test_none_names_returns_empty(self) -> None:
        # cast: the guard exists because the runtime value can violate the
        # ``list[str]`` annotation (blank YAML -> None), which is what we test.
        names = cast("list[str]", None)
        assert resolve_language_objects(names, [{"id": 8, "name": "Japanese"}]) == []

    def test_non_string_names_are_skipped(self) -> None:
        defs = [{"id": 8, "name": "Japanese"}]
        names = cast("list[str]", [None, 5, "Japanese"])

        result = resolve_language_objects(names, defs)

        assert result == [{"id": 8, "name": "Japanese"}]
