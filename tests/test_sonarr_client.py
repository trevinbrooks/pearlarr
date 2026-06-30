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

import requests
import responses
from responses import matchers

from seadexarr.modules.manual_import import PendingImport
from seadexarr.modules.seadex_types import (
    CommandResource,
    ManualImportFile,
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
        seadex_sizes=[],
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
