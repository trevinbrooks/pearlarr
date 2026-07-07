# pyright: strict
"""Direct tests for the ``radarr_client`` module's contracts.

Mirrors ``test_sonarr_client``: a REAL ``RadarrClient`` (whose ``__init__``
constructs an ``arrapi`` client that probes ``GET /api/v3/system/status``) over a
``responses``-mocked ``requests`` boundary. Pins the decode into the typed
``MovieFile`` view and the degrade-to-empty guard (non-200 / transient request
error -> ``[]`` + a warning), so a Radarr outage never unwinds the run; plus the
``all_movies`` cast against arrapi drift and the ``collect_anime_movies`` wiring.
"""

import logging
from collections.abc import Set as AbstractSet

import pytest
import requests
import responses

from seadexarr.modules.radarr_client import RadarrClient, collect_anime_movies
from seadexarr.modules.seadex_types import HistoryRecord, MovieFile, RadarrItem

from .builders import make_bare_instance

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
        request = rsps.calls[-1].request

    assert files == [MovieFile(release_group="SubsPlease", size=123)]
    url = request.url
    assert url is not None
    assert "movieId=7" in url
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


def test_movie_files_non_200_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A Radarr 500 degrades to [] with a warning (was a JSONDecodeError mid-run)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/moviefile", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            files = client.movie_files(7)

    assert files == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert warning.getMessage() == "Could not fetch files for movie 7 from Radarr (status code 500); assuming none"


def test_movie_files_request_error_returns_empty() -> None:
    """A transient request error (timeout / connection drop) degrades to []."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/moviefile", body=requests.exceptions.ConnectionError("boom"))
        assert client.movie_files(7) == []


def test_history_since_decodes_records_and_builds_request() -> None:
    """``history_since()`` keys the item id on ``movieId`` and the request pins
    ``includeMovie=false``; a record with no ``data`` map parses to a None reason.
    """

    body: list[object] = [
        {
            "id": 3,
            "movieId": 9,
            "date": "2026-07-01T10:00:00Z",
            "eventType": "movieFileDeleted",
            "downloadId": "abc123",
            "data": {"reason": "upgrade"},
        },
        {
            "id": 4,
            "movieId": 10,
            "date": "2026-07-01T11:00:00Z",
            "eventType": "downloadFolderImported",
        },
    ]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/history/since", json=body)
        records = client.history_since("2026-06-30T08:00:00Z")
        request = rsps.calls[-1].request

    assert records == [
        HistoryRecord(
            id=3,
            date="2026-07-01T10:00:00Z",
            item_id=9,
            event_type="movieFileDeleted",
            download_id="abc123",
            reason="upgrade",
        ),
        HistoryRecord(
            id=4,
            date="2026-07-01T11:00:00Z",
            item_id=10,
            event_type="downloadFolderImported",
            download_id=None,
            reason=None,
        ),
    ]
    url = request.url
    assert url is not None
    assert "date=2026-06-30T08%3A00%3A00Z" in url
    assert "includeMovie=false" in url
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


def test_history_since_non_200_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 history read returns None with a warning (fail-open)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/history/since", status=500)
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_history_since_request_error_returns_none() -> None:
    """A transient request error is swallowed to None (fail-open)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/history/since", body=requests.exceptions.ConnectionError("boom"))
        assert client.history_since("2026-06-30T08:00:00Z") is None


def test_history_since_non_json_body_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A 200 with a non-JSON body fails open to None (the shared-helper hardening)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/history/since", body="<html>login</html>", content_type="text/html")
        with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
            result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_trailing_slash_url_is_normalized() -> None:
    """A trailing-slash base url must not become a ``//api`` join (login redirect)."""

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f"{_BASE}/system/status", json={"version": "5.0.0"})
        client = RadarrClient(
            url=f"{_URL}/",
            api_key=_KEY,
            session=requests.Session(),
            logger=logging.getLogger("seadexarr.test"),
        )
        rsps.add(responses.GET, f"{_BASE}/history/since", json=[])
        assert client.history_since("2026-06-30T08:00:00Z") == []


# A realistic Radarr v3 ``/api/v3/movie`` record. Every attribute the run READS
# (id/title/tmdbId/imdbId/monitored) is non-None on purpose: arrapi's partial-
# reload magic re-fetches ``/movie/{id}`` on any None attribute read.
_MOVIE_BODY: dict[str, object] = {
    "id": 9,
    "title": "Your Name.",
    "sortTitle": "your name",
    "sizeOnDisk": 4_806_820_247,
    "status": "released",
    "overview": "Two strangers find themselves linked in a bizarre way.",
    "inCinemas": "2016-08-26T00:00:00Z",
    "images": [],
    "year": 2016,
    "hasFile": True,
    "studio": "CoMix Wave Films",
    "path": "/movies/Your Name (2016)",
    "monitored": True,
    "minimumAvailability": "announced",
    "isAvailable": True,
    "runtime": 106,
    "cleanTitle": "yourname",
    "imdbId": "tt5311514",
    "tmdbId": 372058,
    "titleSlug": "372058",
    "certification": "PG",
    "genres": ["Animation", "Drama", "Romance"],
    "tags": [],
    "added": "2023-01-15T12:00:00Z",
    "qualityProfileId": 1,
    "originalTitle": "Kimi no Na wa.",
}


def test_all_movies_parses_into_radarr_item_shape() -> None:
    """The arrapi movies satisfy ``RadarrItem`` with correctly-typed id fields.

    ``all_movies`` casts arrapi's untyped objects to ``list[RadarrItem]``
    unchecked, so this pins the runtime shape (via the ``@runtime_checkable``
    protocol, checked from ``object`` since the static type is the cast's claim)
    and the exact typed values against arrapi parse drift.
    """

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        client = _make_client(rsps)
        rsps.add(responses.GET, f"{_BASE}/movie", json=[_MOVIE_BODY])
        movies: list[object] = list(client.all_movies())

    [movie] = movies
    assert isinstance(movie, RadarrItem)
    assert movie.id == 9
    assert movie.title == "Your Name."
    assert movie.tmdbId == 372058
    assert movie.imdbId == "tt5311514"
    assert movie.monitored is True


class _Movie:
    """A structural ``RadarrItem`` stand-in with the ids a test presets."""

    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tmdbId: int

    def __init__(self, movie_id: int, title: str, *, tmdb_id: int, imdb_id: str | None) -> None:
        self.id = movie_id
        self.title = title
        self.tmdbId = tmdb_id
        self.imdbId = imdb_id
        self.monitored = True


class _StubRadarrApi:
    """Stands in for ``RadarrClient``'s ``_api`` leaf: a preset raw movie list."""

    def __init__(self, movies: list[_Movie]) -> None:
        self._movies = movies

    def all_movies(self) -> list[_Movie]:
        return list(self._movies)


class _RecordingIdSets:
    """Recording ``AnimeIdSets``: preset per-column id sets, calls recorded."""

    def __init__(self, sets: dict[str, set[int | str]]) -> None:
        self._sets = sets
        self.calls: list[str] = []

    def anime_id_set(self, column: str) -> AbstractSet[int | str]:
        self.calls.append(column)
        return self._sets.get(column, set())


def test_collect_anime_movies_wires_id_spaces() -> None:
    """The candidate sets are pulled for (tmdb_movie_id, imdb_id), in that order,
    an imdb-only match is kept, and ``anibridge=None`` degrades to empty sets."""

    movies = [
        _Movie(1, "Imdb Only", tmdb_id=111, imdb_id="tt1"),
        _Movie(2, "Unmatched", tmdb_id=222, imdb_id="tt2"),
    ]
    client = make_bare_instance(RadarrClient, _api=_StubRadarrApi(movies))
    id_sets = _RecordingIdSets({"imdb_id": {"tt1"}})

    kept = collect_anime_movies(client, id_sets, None)

    assert id_sets.calls == ["tmdb_movie_id", "imdb_id"]
    assert [m.id for m in kept] == [1]


def test_movie_file_from_api_nullability() -> None:
    """Missing releaseGroup/size parse to None; unknown keys are ignored."""

    assert MovieFile.from_api({}) == MovieFile(None, None)
    parsed = MovieFile.from_api({"releaseGroup": "SubsPlease", "size": 123, "id": 9, "unknown": True})
    assert parsed == MovieFile(release_group="SubsPlease", size=123)
