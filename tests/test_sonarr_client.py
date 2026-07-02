# pyright: strict
"""Direct tests for ``SonarrClient``, the Sonarr REST adapter.

Each test builds a REAL ``SonarrClient`` (whose ``__init__`` constructs an
``arrapi`` client that probes ``GET /api/v3/system/status``) over a
``responses``-mocked ``requests`` boundary, then drives one method and asserts
the request URL / body it builds AND the decoded return view its ``from_api``
parsers produce. Bodies come from the captured ``tests/fixtures/sonarr`` JSON
where one exists (queue / manual-import / command-list / quality-definitions),
otherwise a minimal inline body. POST bodies are asserted via
``responses``' ``json_params_matcher`` (no Any-typed body reads); GET request
shape is read off ``rsps.calls[-1].request.url``.
"""

import logging

import pytest
import requests
import responses
from responses import matchers

from seadexarr.modules.manual_import import PendingImport
from seadexarr.modules.seadex_types import (
    CommandResource,
    ManualImportFile,
    ParsedFileInfo,
    QueueRecord,
)
from seadexarr.modules.sonarr_client import SonarrClient

from .http_mock import sonarr_fixture

_URL = "http://sonarr.test"
_BASE = f"{_URL}/api/v3"
_KEY = "testkey"


def _make_client(rsps: responses.RequestsMock) -> SonarrClient:
    """Register arrapi's construction probe and build a real ``SonarrClient``.

    ``responses`` patches the global ``requests`` adapter, so both arrapi's own
    session (the ``system/status`` probe) and the shared session handed to the
    client are intercepted.
    """

    rsps.add(responses.GET, f"{_BASE}/system/status", json={"version": "3.0.10"})
    return SonarrClient(
        url=_URL,
        api_key=_KEY,
        session=requests.Session(),
        logger=logging.getLogger("seadexarr.test"),
    )


def _make_pending(*, infohash: str, title: str) -> PendingImport:
    """A minimal ``PendingImport`` carrying only the fields the scan reads."""

    return PendingImport(
        infohash=infohash,
        series_id=1,
        file_episode_map={},
        episode_ids=[],
        release_group="",
        is_dual_audio=False,
        seadex_files=[],
        title=title,
        added_at="",
    )


# --- queue() ----------------------------------------------------------------


def test_queue_decodes_records_and_builds_request() -> None:
    """``queue()`` pulls the whole queue in one paged request and narrows each
    record to a ``QueueRecord`` view.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/queue", json=sonarr_fixture("queue.json"))
        records = client.queue()
        url = rsps.calls[-1].request.url

    assert len(records) == 3
    assert records[0] == QueueRecord(
        download_id="B7640FF13A2ADCA981B821D03CEBD1B569798459",
        state="downloading",
        status="ok",
    )
    assert url is not None
    assert "pageSize=1000" in url
    assert "includeUnknownSeriesItems=true" in url


def test_queue_non_200_returns_empty() -> None:
    """A non-200 queue read falls back to an empty list (caller treats as untracked)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/queue", status=500)
        assert client.queue() == []


def test_queue_request_error_returns_empty() -> None:
    """A transient request error (a timeout raises RequestException) also falls
    back to [] instead of unwinding the poll loop.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/queue", body=requests.exceptions.ConnectionError("boom"))
        assert client.queue() == []


# --- episodes() -------------------------------------------------------------


def test_episodes_decodes_sorted_and_builds_request() -> None:
    """``episodes()`` pulls one series' episodes season/episode-sorted, narrowing
    each to a ``SonarrEpisode``; the request pins seriesId + the include flags.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/episode", json=sonarr_fixture("episodes_228_bahamut.json"))
        episodes = client.episodes(228)
        url = rsps.calls[-1].request.url

    assert episodes is not None
    assert len(episodes) == 13
    # sorted: the lone S00 special leads, S01E12 trails (decode + order in one).
    assert (episodes[0].season_number, episodes[0].episode_number, episodes[0].id) == (0, 1, 8475)
    assert (episodes[-1].season_number, episodes[-1].episode_number, episodes[-1].id) == (1, 12, 8487)
    assert url is not None
    assert "seriesId=228" in url
    assert "includeImages=false" in url
    assert "includeEpisodeFile=true" in url
    assert "apikey=testkey" in url


def test_episodes_non_200_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 episode read returns None and warns (the caller skips the id)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/episode", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.episodes(228)

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_episodes_quiet_suppresses_unreachable_warning(caplog: pytest.LogCaptureFixture) -> None:
    """``quiet=True`` still returns None on a non-200 but emits NO warning - the
    concurrent prefetch path, retried/logged on the main thread instead.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/episode", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.episodes(228, quiet=True)

    assert result is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_episodes_request_error_returns_none() -> None:
    """A transient request error (connection drop) is swallowed to None."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/episode", body=requests.exceptions.ConnectionError("boom"))
        assert client.episodes(228, quiet=True) is None


# --- parse() ----------------------------------------------------------------


def test_parse_skips_entries_missing_season_or_episode() -> None:
    """``parse()`` drops any parsed entry missing a season OR episode number,
    keeping only the fully-resolved ``{season, episode}`` mappings.
    """

    body: dict[str, object] = {
        "episodes": [
            {"seasonNumber": 1, "episodeNumber": 1},
            {"episodeNumber": 5},  # no seasonNumber -> dropped
        ],
    }
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", json=body)
        parsed = client.parse("Cool.Anime.S01E01.mkv")

    assert parsed == [{"season": 1, "episode": 1}]


def test_parse_clean_no_match_returns_empty_list() -> None:
    """A clean 200 where Sonarr matched no episode returns ``[]`` (a *confirmed*
    no-match the caller may negative-cache) - distinct from a failure's None.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", json={"episodes": []})
        assert client.parse("Unmatched.Release.mkv") == []


def test_parse_non_200_returns_none() -> None:
    """A non-200 parse returns None (a failure that must NOT be cached)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", status=500)
        assert client.parse("Cool.Anime.S01E01.mkv") is None


def test_parse_request_error_returns_none() -> None:
    """A transient request error returns None (also uncacheable)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", body=requests.exceptions.ConnectionError("boom"))
        assert client.parse("Cool.Anime.S01E01.mkv") is None


# --- parse_episode_info() (series-AGNOSTIC parsedEpisodeInfo) ----------------


def test_parse_episode_info_decodes_season_episode() -> None:
    """An ``SxxExx`` release decodes to its season + episode numbers; the request
    carries the title + apikey.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", json=sonarr_fixture("parse_bahamut_s01e01.json"))
        info = client.parse_episode_info("Bahamut.S01E01.mkv")
        url = rsps.calls[-1].request.url

    assert info == ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
        absolute_episode_numbers=(),
        special=False,
    )
    assert url is not None
    assert "title=" in url
    assert "apikey=testkey" in url


def test_parse_episode_info_decodes_absolute() -> None:
    """An absolute-numbered release decodes to its absolute numbers (season 0, no
    SxxExx episode numbers) - the case Sonarr's series-matched parse misses.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", json=sonarr_fixture("parse_toloveru_abs14.json"))
        info = client.parse_episode_info("ToLoveRu.-.14.mkv")

    assert info == ParsedFileInfo(
        season_number=0,
        episode_numbers=(),
        absolute_episode_numbers=(14,),
        special=False,
    )


def test_parse_episode_info_non_200_returns_none() -> None:
    """A non-200 parse leaves the file for retry (returns None)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", status=500)
        assert client.parse_episode_info("Bahamut.S01E01.mkv") is None


def test_parse_episode_info_request_error_returns_none() -> None:
    """A transient request error leaves the file for retry (returns None)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/parse", body=requests.exceptions.ConnectionError("boom"))
        assert client.parse_episode_info("Bahamut.S01E01.mkv") is None


# --- manual_import_candidates() ---------------------------------------------


def test_manual_import_candidates_decodes_and_uppercases_downloadid() -> None:
    """The scan keys on the UPPERCASED infohash (no ``seriesId``) and narrows
    each candidate to its ``path`` / ``quality`` / ``rejections``.
    """

    pending = _make_pending(
        infohash="abcdef0123456789abcdef0123456789abcdef01",
        title="Yamada-kun",
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/manualimport", json=sonarr_fixture("manualimport_yamada.json"))
        candidates = client.manual_import_candidates(pending=pending)
        url = rsps.calls[-1].request.url

    assert candidates is not None
    assert len(candidates) == 2
    first = candidates[0]
    assert first.path == (
        "/downloads/Yamada-kun.and.the.Seven.Witches.S00.480p.DVDRip.Opus2.0.x264-Headpatter/"
        "Yamada-kun.and.the.Seven.Witches.S00E01.480p.DVDRip.Opus2.0.x264-Headpatter.mkv"
    )
    assert first.quality is not None
    assert first.rejections[0].reason == "Unknown Series"
    assert url is not None
    assert "downloadId=ABCDEF0123456789ABCDEF0123456789ABCDEF01" in url
    assert "filterExistingFiles=false" in url


def test_manual_import_candidates_non_200_returns_none() -> None:
    """A non-200 scan returns None (caller keeps waiting / re-asks)."""

    pending = _make_pending(infohash="a" * 40, title="Yamada-kun")
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/manualimport", status=500)
        assert client.manual_import_candidates(pending=pending) is None


def test_manual_import_candidates_request_error_returns_none() -> None:
    """A transient request error (slow mount / drop) returns None, not []."""

    pending = _make_pending(infohash="a" * 40, title="Yamada-kun")
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/manualimport", body=requests.exceptions.ConnectionError("boom"))
        assert client.manual_import_candidates(pending=pending) is None


# --- manual_import_execute() / refresh_monitored_downloads() (POST /command) -


def test_manual_import_execute_posts_body_and_returns_id() -> None:
    """The ``ManualImport`` command POSTs ``{name, importMode, files}`` and returns
    the queued command id; ``import_mode`` threads straight into the body.
    """

    file: ManualImportFile = {
        "path": "/downloads/show/ep01.mkv",
        "seriesId": 42,
        "episodeIds": [101],
        "releaseGroup": "SubsPlease",
        "downloadId": "A" * 40,
        "languages": [{"id": 1, "name": "English"}],
    }
    expected_body: dict[str, object] = {
        "name": "ManualImport",
        "importMode": "move",
        "files": [file],
    }
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(
            responses.POST,
            f"{_BASE}/command",
            json={"id": 4242},
            match=[matchers.json_params_matcher(expected_body)],
        )
        command_id = client.manual_import_execute(files=[file], import_mode="move")
        url = rsps.calls[-1].request.url

    assert command_id == 4242
    assert url is not None
    assert url.startswith(f"{_BASE}/command")


def test_manual_import_execute_non_2xx_returns_none() -> None:
    """A non-2xx command POST leaves the import pending (returns ``None``)."""

    file: ManualImportFile = {
        "path": "/downloads/show/ep01.mkv",
        "seriesId": 42,
        "episodeIds": [101],
        "releaseGroup": "SubsPlease",
        "downloadId": "A" * 40,
        "languages": [{"id": 1, "name": "English"}],
    }
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.POST, f"{_BASE}/command", status=400, json={})
        assert client.manual_import_execute(files=[file]) is None


def test_post_command_request_error_returns_none() -> None:
    """A transient error on the command POST (now timeout-bounded) returns None
    rather than raising through the import path.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.POST, f"{_BASE}/command", body=requests.exceptions.ConnectionError("boom"))
        assert client.refresh_monitored_downloads() is None


def test_refresh_monitored_downloads_posts_command_name() -> None:
    """``RefreshMonitoredDownloads`` POSTs only ``{name}`` and returns its id."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(
            responses.POST,
            f"{_BASE}/command",
            json={"id": 77},
            match=[matchers.json_params_matcher({"name": "RefreshMonitoredDownloads"})],
        )
        command_id = client.refresh_monitored_downloads()

    assert command_id == 77


# --- command_status() / list_commands() -------------------------------------


def test_command_status_decodes() -> None:
    """A single-command GET narrows to a ``CommandResource`` with status/result."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(
            responses.GET,
            f"{_BASE}/command/55",
            json={"id": 55, "status": "completed", "result": "successful"},
        )
        status = client.command_status(55)

    assert status.id == 55
    assert status.status == "completed"
    assert status.result == "successful"


def test_command_status_non_200_returns_default() -> None:
    """A non-200 status read yields a default (status-None) ``CommandResource``."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/command/9", status=503)
        assert client.command_status(9) == CommandResource()


def test_command_status_request_error_returns_default() -> None:
    """A transient request error also yields the default ``CommandResource`` (the
    caller treats the import as unverified and leaves it pending).
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/command/9", body=requests.exceptions.ConnectionError("boom"))
        assert client.command_status(9) == CommandResource()


def test_list_commands_decodes_with_nested_files() -> None:
    """The command LIST narrows each command, including the nested ``body.files``
    each in-flight ManualImport carries (the guard's match signal).
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/command", json=sonarr_fixture("command_list.json"))
        commands = client.list_commands()

    assert len(commands) == 5
    first = commands[0]
    assert first.name == "ManualImport"
    assert first.status == "started"
    assert first.message == "Processing file 4 of 8"
    assert first.files[0].download_id == "3333333333333333333333333333333333333333"
    assert first.files[0].series_id == 169
    assert first.files[0].episode_ids == (6605,)


def test_list_commands_non_200_returns_empty() -> None:
    """A non-200 command list reads as "nothing in flight" (empty list)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/command", status=503)
        assert client.list_commands() == []


# --- quality_definitions() / languages() ------------------------------------


def test_quality_definitions_returns_raw_list() -> None:
    """Quality definitions are passed through verbatim (the resolver re-emits the
    nested ``quality`` into the outgoing payload).
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/qualitydefinition", json=sonarr_fixture("qualitydefinitions.json"))
        defs = client.quality_definitions()
        url = rsps.calls[-1].request.url

    assert defs[0].get("quality") == {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}
    assert url is not None
    assert "apikey=testkey" in url


def test_languages_returns_raw_list() -> None:
    """Languages are passed through verbatim (POSTed into the file payload)."""

    body: list[object] = [{"id": 1, "name": "English"}, {"id": 8, "name": "Japanese"}]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/language", json=body)
        result = client.languages()

    assert result == [{"id": 1, "name": "English"}, {"id": 8, "name": "Japanese"}]


def test_quality_definitions_non_200_returns_empty() -> None:
    """A non-200 quality-definitions read falls back to an empty list."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/qualitydefinition", status=500)
        assert client.quality_definitions() == []


def test_languages_non_200_returns_empty() -> None:
    """A non-200 languages read falls back to an empty list."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/language", status=500)
        assert client.languages() == []
