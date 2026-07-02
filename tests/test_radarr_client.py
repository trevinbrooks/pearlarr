# pyright: strict
"""Direct tests for ``RadarrClient.movie_files``, the raw Radarr endpoint.

Mirrors ``test_sonarr_client``: a REAL ``RadarrClient`` (whose ``__init__``
constructs an ``arrapi`` client that probes ``GET /api/v3/system/status``) over a
``responses``-mocked ``requests`` boundary. Pins the decode into the typed
``MovieFile`` view and the degrade-to-empty guard (non-200 / transient request
error -> ``[]`` + a warning), so a Radarr outage never unwinds the run.
"""

import logging

import pytest
import requests
import responses

from seadexarr.modules.radarr_client import RadarrClient
from seadexarr.modules.seadex_types import MovieFile

_URL = "http://radarr.test"
_BASE = f"{_URL}/api/v3"
_KEY = "testkey"


def _make_client(rsps: responses.RequestsMock) -> RadarrClient:
    """Register arrapi's construction probe and build a real ``RadarrClient``."""

    rsps.add(responses.GET, f"{_BASE}/system/status", json={"version": "5.0.0"})
    return RadarrClient(
        url=_URL,
        api_key=_KEY,
        session=requests.Session(),
        logger=logging.getLogger("seadexarr.test"),
    )


def test_movie_files_decodes_records_and_builds_request() -> None:
    body: list[object] = [{"releaseGroup": "SubsPlease", "size": 123, "id": 9}]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/moviefile", json=body)
        files = client.movie_files(7)
        url = rsps.calls[-1].request.url

    assert files == [MovieFile(release_group="SubsPlease", size=123)]
    assert url is not None
    assert "movieId=7" in url
    assert "apikey=testkey" in url


def test_movie_files_non_200_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A Radarr 500 degrades to [] with a warning (was a JSONDecodeError mid-run)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/moviefile", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            files = client.movie_files(7)

    assert files == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_movie_files_request_error_returns_empty() -> None:
    """A transient request error (timeout / connection drop) degrades to []."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/moviefile", body=requests.exceptions.ConnectionError("boom"))
        assert client.movie_files(7) == []
