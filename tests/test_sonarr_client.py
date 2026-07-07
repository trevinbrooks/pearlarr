# pyright: strict
# pyright: reportPrivateUsage=false
# _make_client stubs the bound ArrHttp's sleep so fail-open tests don't wait
# out real retry backoffs; strict re-flags that private write.
"""Direct tests for ``SonarrClient``, the Sonarr REST adapter.

Each test builds a REAL ``SonarrClient`` (construction is network-free), then
drives one method and asserts the request URL / body it builds AND the decoded
return view its ``from_api`` parsers produce. The endpoints ride the
httpx-based ``ArrHttp`` and are mocked via ``respx``; the one still on
``requests`` (``history_since``) via ``responses``. Bodies come from the
captured ``tests/fixtures/sonarr`` JSON where one exists (queue /
manual-import / command-list / quality-definitions), otherwise a minimal
inline body. POST bodies are asserted by decoding the captured request content
(no Any-typed body reads); GET request shape is read off
``route.calls.last.request.url``.
"""

import json
import logging

import httpx
import pytest
import requests
import responses
import respx

from seadexarr.modules.manual_import import PendingImport
from seadexarr.modules.seadex_types import (
    CommandResource,
    HistoryRecord,
    ManualImportFile,
    ParsedFileInfo,
    QueueRecord,
    SonarrItem,
)
from seadexarr.modules.sonarr_client import SonarrClient

from .http_mock import sonarr_fixture

_URL = "http://sonarr.test"
_BASE = f"{_URL}/api/v3"
_KEY = "testkey"


def _make_client() -> SonarrClient:
    """Build a real ``SonarrClient`` (construction is network-free).

    The endpoints still on ``requests`` are mocked per test through
    ``responses``; the endpoints migrated onto ``ArrHttp`` ride the httpx
    client, mocked through ``respx``. The bound helper's ``sleep`` is stubbed
    out so fail-open tests don't wait out real backoffs.
    """

    client = SonarrClient(
        url=_URL,
        api_key=_KEY,
        session=requests.Session(),
        http=httpx.Client(),
        logger=logging.getLogger("seadexarr.test"),
    )
    client._http.sleep = lambda _s: None
    return client


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


# --- all_series() -------------------------------------------------------------

# A minimal ``/api/v3/series`` record: the consumed item fields plus a couple of
# extras proving unknown keys are ignored by ``SonarrSeries.from_api``.
_SERIES_BODY: dict[str, object] = {
    "id": 228,
    "title": "Undefeated Bahamut Chronicle",
    "monitored": True,
    "tvdbId": 299502,
    "imdbId": "tt5311514",
    "sortTitle": "undefeated bahamut chronicle",
    "seasonFolder": True,
}


@respx.mock
def test_all_series_parses_into_sonarr_item_shape() -> None:
    """``all_series`` parses each raw record into a ``SonarrSeries`` satisfying
    the ``SonarrItem`` protocol (checked from ``object``: the runtime
    counterpart of the client's typed claim), with correctly-typed id fields.
    """

    route = respx.get(f"{_BASE}/series").respond(json=[_SERIES_BODY])
    client = _make_client()
    series: list[object] = list(client.all_series())

    [show] = series
    assert isinstance(show, SonarrItem)
    assert show.id == 228
    assert show.title == "Undefeated Bahamut Chronicle"
    assert show.tvdbId == 299502
    assert show.imdbId == "tt5311514"
    assert show.monitored is True
    request = route.calls.last.request
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in str(request.url)
    assert request.headers["X-Api-Key"] == _KEY


@respx.mock
def test_all_series_skips_non_dict_elements() -> None:
    """A stray non-object element in the series array is skipped, never crashed on."""

    respx.get(f"{_BASE}/series").respond(json=[_SERIES_BODY, "stray", 42])
    series = _make_client().all_series()

    assert [s.id for s in series] == [228]


# --- queue() ----------------------------------------------------------------


@respx.mock
def test_queue_decodes_records_and_builds_request() -> None:
    """``queue()`` pulls the whole queue in one paged request and narrows each
    record to a ``QueueRecord`` view.
    """

    route = respx.get(f"{_BASE}/queue").respond(json=sonarr_fixture("queue.json"))
    client = _make_client()
    records = client.queue()

    assert len(records) == 3
    assert records[0] == QueueRecord(
        download_id="B7640FF13A2ADCA981B821D03CEBD1B569798459",
        state="downloading",
        status="ok",
    )
    request = route.calls.last.request
    url = str(request.url)
    assert "pageSize=1000" in url
    assert "includeUnknownSeriesItems=true" in url
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == _KEY


def _queue_page(total: int, hashes: list[str]) -> dict[str, object]:
    """One raw paged ``/queue`` body carrying ``totalRecords`` and the records."""

    return {
        "totalRecords": total,
        "records": [
            {"downloadId": h, "trackedDownloadState": "downloading", "trackedDownloadStatus": "ok"} for h in hashes
        ],
    }


@respx.mock
def test_queue_paginates_until_total_records_covered() -> None:
    """A queue larger than one page is fetched page by page until totalRecords
    is covered, never silently truncated at the first page.
    """

    pages = [
        httpx.Response(200, json=_queue_page(3, ["HASH0", "HASH1"])),
        httpx.Response(200, json=_queue_page(3, ["HASH2"])),
    ]
    seen_urls: list[str] = []

    def _serve(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return pages.pop(0)

    respx.get(f"{_BASE}/queue").mock(side_effect=_serve)
    client = _make_client()
    records = client.queue()

    assert [r.download_id for r in records] == ["HASH0", "HASH1", "HASH2"]
    assert len(seen_urls) == 2
    assert "page=1" in seen_urls[0]
    assert "page=2" in seen_urls[1]


@respx.mock
def test_queue_later_page_failure_keeps_fetched_records() -> None:
    """A failed LATER page returns what was already fetched (partial beats empty
    for the caller's "not tracked -> fall back to own scan" logic).
    """

    route = respx.get(f"{_BASE}/queue")
    # Page 1 succeeds; page 2 stays 500 through the transport retries.
    route.side_effect = [httpx.Response(200, json=_queue_page(3, ["HASH0", "HASH1"]))] + [httpx.Response(500)] * 10
    client = _make_client()
    records = client.queue()

    assert [r.download_id for r in records] == ["HASH0", "HASH1"]


@respx.mock
def test_queue_non_200_returns_empty() -> None:
    """A non-200 queue read falls back to an empty list (caller treats as untracked)."""

    respx.get(f"{_BASE}/queue").respond(status_code=404)
    assert _make_client().queue() == []


@respx.mock
def test_queue_request_error_returns_empty() -> None:
    """A transient request error (a timeout raises an httpx error) also falls
    back to [] instead of unwinding the poll loop.
    """

    respx.get(f"{_BASE}/queue").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().queue() == []


# --- episodes() -------------------------------------------------------------


@respx.mock
def test_episodes_decodes_sorted_and_builds_request() -> None:
    """``episodes()`` pulls one series' episodes season/episode-sorted, narrowing
    each to a ``SonarrEpisode``; the request pins seriesId + the include flags.
    """

    route = respx.get(f"{_BASE}/episode").respond(json=sonarr_fixture("episodes_228_bahamut.json"))
    episodes = _make_client().episodes(228)

    assert episodes is not None
    assert len(episodes) == 13
    # sorted: the lone S00 special leads, S01E12 trails (decode + order in one).
    assert (episodes[0].season_number, episodes[0].episode_number, episodes[0].id) == (0, 1, 8475)
    assert (episodes[-1].season_number, episodes[-1].episode_number, episodes[-1].id) == (1, 12, 8487)
    request = route.calls.last.request
    url = str(request.url)
    assert "seriesId=228" in url
    assert "includeImages=false" in url
    assert "includeEpisodeFile=true" in url
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
def test_episodes_missing_numbers_sort_first_without_crashing() -> None:
    """A record missing seasonNumber/episodeNumber sorts first (as -1) instead
    of raising a ``None < int`` TypeError and killing the whole fetch.
    """

    body = [
        {"id": 2, "seasonNumber": 1, "episodeNumber": 2},
        {"id": 9},
        {"id": 1, "seasonNumber": 1, "episodeNumber": 1},
    ]
    respx.get(f"{_BASE}/episode").respond(json=body)
    episodes = _make_client().episodes(228)

    assert episodes is not None
    assert [ep.id for ep in episodes] == [9, 1, 2]


@respx.mock
def test_episodes_skips_non_dict_elements() -> None:
    """A stray non-object element in the episode array is skipped, never crashed on."""

    respx.get(f"{_BASE}/episode").respond(
        json=[{"id": 1, "seasonNumber": 1, "episodeNumber": 1}, "stray", 42],
    )
    episodes = _make_client().episodes(228)

    assert episodes is not None
    assert [ep.id for ep in episodes] == [1]


@respx.mock
def test_episodes_non_200_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 episode read returns None and warns (the caller skips the id)."""

    respx.get(f"{_BASE}/episode").respond(status_code=500)
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        result = client.episodes(228)

    assert result is None
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert warning.getMessage() == "Could not fetch episodes for series 228 from Sonarr (status code 500); skipping"


@respx.mock
def test_episodes_quiet_suppresses_unreachable_warning(caplog: pytest.LogCaptureFixture) -> None:
    """``quiet=True`` still returns None on a non-200 but emits NO warning - the
    concurrent prefetch path, retried/logged on the main thread instead.
    """

    respx.get(f"{_BASE}/episode").respond(status_code=500)
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        result = client.episodes(228, quiet=True)

    assert result is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@respx.mock
def test_episodes_request_error_returns_none() -> None:
    """A transient request error (connection drop) is swallowed to None."""

    respx.get(f"{_BASE}/episode").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().episodes(228, quiet=True) is None


# --- parse() ----------------------------------------------------------------


@respx.mock
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
    respx.get(f"{_BASE}/parse").respond(json=body)

    assert _make_client().parse("Cool.Anime.S01E01.mkv") == [{"season": 1, "episode": 1}]


@respx.mock
def test_parse_clean_no_match_returns_empty_list() -> None:
    """A clean 200 where Sonarr matched no episode returns ``[]`` (a *confirmed*
    no-match the caller may negative-cache) - distinct from a failure's None.
    """

    respx.get(f"{_BASE}/parse").respond(json={"episodes": []})
    assert _make_client().parse("Unmatched.Release.mkv") == []


@respx.mock
def test_parse_non_200_returns_none() -> None:
    """A non-200 parse returns None (a failure that must NOT be cached)."""

    respx.get(f"{_BASE}/parse").respond(status_code=500)
    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None


@respx.mock
def test_parse_request_error_returns_none() -> None:
    """A transient request error returns None (also uncacheable)."""

    respx.get(f"{_BASE}/parse").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None


# --- parse_episode_info() (series-AGNOSTIC parsedEpisodeInfo) ----------------


@respx.mock
def test_parse_episode_info_decodes_season_episode() -> None:
    """An ``SxxExx`` release decodes to its season + episode numbers; the request
    carries the title in the URL and the api key in the X-Api-Key header.
    """

    route = respx.get(f"{_BASE}/parse").respond(json=sonarr_fixture("parse_bahamut_s01e01.json"))
    info = _make_client().parse_episode_info("Bahamut.S01E01.mkv")

    assert info == ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
        absolute_episode_numbers=(),
        special=False,
    )
    request = route.calls.last.request
    url = str(request.url)
    assert "title=" in url
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
def test_parse_episode_info_decodes_absolute() -> None:
    """An absolute-numbered release decodes to its absolute numbers (season 0, no
    SxxExx episode numbers) - the case Sonarr's series-matched parse misses.
    """

    respx.get(f"{_BASE}/parse").respond(json=sonarr_fixture("parse_toloveru_abs14.json"))
    info = _make_client().parse_episode_info("ToLoveRu.-.14.mkv")

    assert info == ParsedFileInfo(
        season_number=0,
        episode_numbers=(),
        absolute_episode_numbers=(14,),
        special=False,
    )


@respx.mock
def test_parse_episode_info_non_200_returns_none() -> None:
    """A non-200 parse leaves the file for retry (returns None)."""

    respx.get(f"{_BASE}/parse").respond(status_code=500)
    assert _make_client().parse_episode_info("Bahamut.S01E01.mkv") is None


@respx.mock
def test_parse_episode_info_request_error_returns_none() -> None:
    """A transient request error leaves the file for retry (returns None)."""

    respx.get(f"{_BASE}/parse").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().parse_episode_info("Bahamut.S01E01.mkv") is None


# --- manual_import_candidates() ---------------------------------------------


@respx.mock
def test_manual_import_candidates_decodes_and_uppercases_downloadid() -> None:
    """The scan keys on the UPPERCASED infohash (no ``seriesId``) and narrows
    each candidate to its ``path`` / ``quality`` / ``rejections``.
    """

    pending = _make_pending(
        infohash="abcdef0123456789abcdef0123456789abcdef01",
        title="Yamada-kun",
    )
    route = respx.get(f"{_BASE}/manualimport").respond(json=sonarr_fixture("manualimport_yamada.json"))
    candidates = _make_client().manual_import_candidates(pending=pending)

    assert candidates is not None
    assert len(candidates) == 2
    first = candidates[0]
    assert first.path == (
        "/downloads/Yamada-kun.and.the.Seven.Witches.S00.480p.DVDRip.Opus2.0.x264-Headpatter/"
        "Yamada-kun.and.the.Seven.Witches.S00E01.480p.DVDRip.Opus2.0.x264-Headpatter.mkv"
    )
    assert first.quality is not None
    assert first.rejections[0].reason == "Unknown Series"
    url = str(route.calls.last.request.url)
    assert "downloadId=ABCDEF0123456789ABCDEF0123456789ABCDEF01" in url
    assert "filterExistingFiles=false" in url


@respx.mock
def test_manual_import_candidates_non_200_returns_none() -> None:
    """A non-200 scan returns None (caller keeps waiting / re-asks)."""

    pending = _make_pending(infohash="a" * 40, title="Yamada-kun")
    respx.get(f"{_BASE}/manualimport").respond(status_code=500)
    assert _make_client().manual_import_candidates(pending=pending) is None


@respx.mock
def test_manual_import_candidates_request_error_returns_none() -> None:
    """A transient request error (slow mount / drop) returns None, not []."""

    pending = _make_pending(infohash="a" * 40, title="Yamada-kun")
    respx.get(f"{_BASE}/manualimport").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().manual_import_candidates(pending=pending) is None


# --- manual_import_execute() / refresh_monitored_downloads() (POST /command) -


@respx.mock
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
    route = respx.post(f"{_BASE}/command").respond(json={"id": 4242})
    command_id = _make_client().manual_import_execute(files=[file], import_mode="move")

    assert command_id == 4242
    request = route.calls.last.request
    assert json.loads(request.content) == expected_body
    url = str(request.url)
    assert url.startswith(f"{_BASE}/command")
    # The POST authenticates through the header too, never the query string.
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
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
    respx.post(f"{_BASE}/command").respond(status_code=400, json={})
    assert _make_client().manual_import_execute(files=[file]) is None


@respx.mock
def test_post_command_request_error_returns_none() -> None:
    """A transient error on the command POST (never retried - not idempotent)
    returns None rather than raising through the import path.
    """

    route = respx.post(f"{_BASE}/command").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().refresh_monitored_downloads() is None
    assert route.call_count == 1  # ONE attempt: a retry could double-queue


@respx.mock
def test_refresh_monitored_downloads_posts_command_name() -> None:
    """``RefreshMonitoredDownloads`` POSTs only ``{name}`` and returns its id."""

    route = respx.post(f"{_BASE}/command").respond(json={"id": 77})
    command_id = _make_client().refresh_monitored_downloads()

    assert command_id == 77
    assert json.loads(route.calls.last.request.content) == {"name": "RefreshMonitoredDownloads"}


# --- command_status() / list_commands() -------------------------------------


@respx.mock
def test_command_status_decodes() -> None:
    """A single-command GET narrows to a ``CommandResource`` with status/result."""

    respx.get(f"{_BASE}/command/55").respond(json={"id": 55, "status": "completed", "result": "successful"})
    status = _make_client().command_status(55)

    assert status.id == 55
    assert status.status == "completed"
    assert status.result == "successful"


@respx.mock
def test_command_status_non_200_returns_default() -> None:
    """A non-200 status read yields a default (status-None) ``CommandResource``."""

    respx.get(f"{_BASE}/command/9").respond(status_code=503)
    assert _make_client().command_status(9) == CommandResource()


@respx.mock
def test_command_status_request_error_returns_default() -> None:
    """A transient request error also yields the default ``CommandResource`` (the
    caller treats the import as unverified and leaves it pending).
    """

    respx.get(f"{_BASE}/command/9").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().command_status(9) == CommandResource()


@respx.mock
def test_list_commands_decodes_with_nested_files() -> None:
    """The command LIST narrows each command, including the nested ``body.files``
    each in-flight ManualImport carries (the guard's match signal).
    """

    respx.get(f"{_BASE}/command").respond(json=sonarr_fixture("command_list.json"))
    commands = _make_client().list_commands()

    assert len(commands) == 5
    first = commands[0]
    assert first.name == "ManualImport"
    assert first.status == "started"
    assert first.message == "Processing file 4 of 8"
    assert first.files[0].download_id == "3333333333333333333333333333333333333333"
    assert first.files[0].series_id == 169
    assert first.files[0].episode_ids == (6605,)


@respx.mock
def test_list_commands_non_200_returns_empty() -> None:
    """A non-200 command list reads as "nothing in flight" (empty list)."""

    respx.get(f"{_BASE}/command").respond(status_code=503)
    assert _make_client().list_commands() == []


# --- history_since() ----------------------------------------------------------


def test_history_since_decodes_records_and_builds_request() -> None:
    """``history_since()`` narrows each record to a ``HistoryRecord`` (incl. the
    case-insensitive ``data`` reason key and a null ``downloadId``); the request
    pins the date + the include flags off.
    """

    body: list[object] = [
        {
            "id": 12,
            "seriesId": 4,
            "date": "2026-07-01T10:00:00Z",
            "eventType": "episodeFileDeleted",
            "downloadId": "ABC123",
            "data": {"Reason": "Upgrade"},
        },
        {
            "id": 13,
            "seriesId": 5,
            "date": "2026-07-01T11:00:00Z",
            "eventType": "downloadFolderImported",
            "downloadId": None,
            "data": {"reason": "MissingFromDisk"},
        },
    ]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", json=body)
        records = client.history_since("2026-06-30T08:00:00Z")
        request = rsps.calls[-1].request

    assert records == [
        HistoryRecord(
            id=12,
            date="2026-07-01T10:00:00Z",
            item_id=4,
            event_type="episodeFileDeleted",
            download_id="ABC123",
            reason="Upgrade",
        ),
        HistoryRecord(
            id=13,
            date="2026-07-01T11:00:00Z",
            item_id=5,
            event_type="downloadFolderImported",
            download_id=None,
            reason="MissingFromDisk",
        ),
    ]
    url = request.url
    assert url is not None
    assert "date=2026-06-30T08%3A00%3A00Z" in url
    assert "includeSeries=false" in url
    assert "includeEpisode=false" in url
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


def test_history_since_non_200_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 history read returns None (the activity scan fails open)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    # The single warning for a failed history fetch states its consequence too
    # (the activity monitor only debug-logs, so this line is all the user sees).
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    expected = "Could not fetch Sonarr history (status code 500); skipping activity detection this run"
    assert warning.getMessage() == expected


def test_history_since_request_error_returns_none() -> None:
    """A transient request error is swallowed to None (fail-open)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", body=requests.exceptions.ConnectionError("boom"))
        assert client.history_since("2026-06-30T08:00:00Z") is None


def test_history_since_non_json_body_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A 200 with a non-JSON body (e.g. a proxy login page) fails open to None."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", body="<html>login</html>", content_type="text/html")
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_history_since_non_array_payload_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A JSON object (not the expected array) fails open to None."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", json={"message": "unauthorized"})
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_history_since_skips_non_dict_elements() -> None:
    """Stray non-object array elements are dropped, not crashed on."""

    body: list[object] = [
        {"id": 1, "seriesId": 2, "date": "2026-07-01T10:00:00Z", "eventType": "grabbed"},
        "stray",
        42,
    ]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client()
        rsps.add(responses.GET, f"{_BASE}/history/since", json=body)
        records = client.history_since("2026-06-30T08:00:00Z")

    assert records == [HistoryRecord(id=1, date="2026-07-01T10:00:00Z", item_id=2, event_type="grabbed")]


def test_trailing_slash_url_is_normalized() -> None:
    """A trailing-slash base url must not become a ``//api`` join - live Sonarr
    302s that to the login page, breaking every raw endpoint.
    """

    client = SonarrClient(
        url=f"{_URL}/",
        api_key=_KEY,
        session=requests.Session(),
        http=httpx.Client(),
        logger=logging.getLogger("seadexarr.test"),
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f"{_BASE}/history/since", json=[])
        assert client.history_since("2026-06-30T08:00:00Z") == []


# --- quality_definitions() / languages() ------------------------------------


@respx.mock
def test_quality_definitions_returns_raw_list() -> None:
    """Quality definitions are passed through verbatim (the resolver re-emits the
    nested ``quality`` into the outgoing payload).
    """

    route = respx.get(f"{_BASE}/qualitydefinition").respond(json=sonarr_fixture("qualitydefinitions.json"))
    defs = _make_client().quality_definitions()

    assert defs[0].get("quality") == {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}
    request = route.calls.last.request
    assert "apikey" not in str(request.url)
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
def test_languages_returns_raw_list() -> None:
    """Languages are passed through verbatim (POSTed into the file payload)."""

    body: list[object] = [{"id": 1, "name": "English"}, {"id": 8, "name": "Japanese"}]
    respx.get(f"{_BASE}/language").respond(json=body)

    assert _make_client().languages() == [{"id": 1, "name": "English"}, {"id": 8, "name": "Japanese"}]


@respx.mock
def test_quality_definitions_non_200_returns_empty() -> None:
    """A non-200 quality-definitions read falls back to an empty list."""

    respx.get(f"{_BASE}/qualitydefinition").respond(status_code=500)
    assert _make_client().quality_definitions() == []


@respx.mock
def test_languages_non_200_returns_empty() -> None:
    """A non-200 languages read falls back to an empty list."""

    respx.get(f"{_BASE}/language").respond(status_code=500)
    assert _make_client().languages() == []
