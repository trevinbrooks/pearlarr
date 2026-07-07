# pyright: strict
# pyright: reportPrivateUsage=false
# _make_client stubs the bound ArrHttp's sleep so fail-open tests don't wait
# out real retry backoffs; strict re-flags that private write.
"""Direct tests for the ``radarr_client`` module's contracts.

Mirrors ``test_sonarr_client``: a REAL ``RadarrClient`` (construction is
network-free) with every endpoint riding the httpx-based ``ArrHttp``, mocked
via ``respx``. Pins the decode into the typed ``MovieFile`` /
``RadarrMovie`` views and the degrade-to-empty guard (non-200 / transient
request error -> ``[]`` + a warning), so a Radarr outage never unwinds the
run; plus the ``collect_anime_movies`` wiring.
"""

import logging
from collections.abc import Set as AbstractSet

import httpx
import pytest
import respx

from seadexarr.modules.radarr_client import RadarrClient, collect_anime_movies
from seadexarr.modules.seadex_types import HistoryRecord, MovieFile, RadarrItem, RadarrMovie

from .fakes import FakeRadarrClient

_URL = "http://radarr.test"
_BASE = f"{_URL}/api/v3"
_KEY = "testkey"


def _make_client() -> RadarrClient:
    """Build a real ``RadarrClient`` (construction is network-free).

    The bound ``ArrHttp``'s ``sleep`` is stubbed out so fail-open tests don't
    wait out real retry backoffs.
    """

    client = RadarrClient(
        url=_URL,
        api_key=_KEY,
        http=httpx.Client(),
        logger=logging.getLogger("seadexarr.test"),
    )
    client._http.sleep = lambda _s: None
    return client


@respx.mock
def test_movie_files_decodes_records_and_builds_request() -> None:
    body: list[object] = [{"releaseGroup": "SubsPlease", "size": 123, "id": 9}]
    route = respx.get(f"{_BASE}/moviefile").respond(json=body)
    files = _make_client().movie_files(7)

    assert files == [MovieFile(release_group="SubsPlease", size=123)]
    request = route.calls.last.request
    url = str(request.url)
    assert "movieId=7" in url
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
def test_movie_files_non_200_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A Radarr 404 degrades to [] with a warning (was a JSONDecodeError mid-run)."""

    respx.get(f"{_BASE}/moviefile").respond(status_code=404)
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        files = client.movie_files(7)

    assert files == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert warning.getMessage() == "Could not fetch files for movie 7 from Radarr (status code 404); assuming none"


@respx.mock
def test_movie_files_request_error_returns_empty() -> None:
    """A transient request error (timeout / connection drop) degrades to []."""

    respx.get(f"{_BASE}/moviefile").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().movie_files(7) == []


@respx.mock
def test_movie_files_non_json_body_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A 200 with an HTML body (reverse-proxy page) fails open to [] - never an abort."""

    respx.get(f"{_BASE}/moviefile").respond(content=b"<html>login</html>", content_type="text/html")
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        files = client.movie_files(7)

    assert files == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "non-JSON body" in warning.getMessage()


@respx.mock
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
    route = respx.get(f"{_BASE}/history/since").respond(json=body)
    records = _make_client().history_since("2026-06-30T08:00:00Z")

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
    request = route.calls.last.request
    url = str(request.url)
    assert "date=2026-06-30T08%3A00%3A00Z" in url
    assert "includeMovie=false" in url
    assert "apikey" not in url
    assert request.headers["X-Api-Key"] == "testkey"


@respx.mock
def test_history_since_non_200_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A non-200 history read returns None with a warning (fail-open)."""

    respx.get(f"{_BASE}/history/since").respond(status_code=500)
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@respx.mock
def test_history_since_request_error_returns_none() -> None:
    """A transient request error is swallowed to None (fail-open)."""

    respx.get(f"{_BASE}/history/since").mock(side_effect=httpx.ConnectError("boom"))
    assert _make_client().history_since("2026-06-30T08:00:00Z") is None


@respx.mock
def test_history_since_non_json_body_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A 200 with a non-JSON body fails open to None (the shared-helper hardening)."""

    respx.get(f"{_BASE}/history/since").respond(content=b"<html>login</html>", content_type="text/html")
    client = _make_client()
    with caplog.at_level(logging.WARNING, logger="seadexarr.test"):
        result = client.history_since("2026-06-30T08:00:00Z")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@respx.mock
def test_trailing_slash_url_is_normalized() -> None:
    """A trailing-slash base url must not become a ``//api`` join (login redirect)."""

    client = RadarrClient(
        url=f"{_URL}/",
        api_key=_KEY,
        http=httpx.Client(),
        logger=logging.getLogger("seadexarr.test"),
    )
    respx.get(f"{_BASE}/history/since").respond(json=[])
    assert client.history_since("2026-06-30T08:00:00Z") == []


# A minimal ``/api/v3/movie`` record: the consumed item fields plus a couple of
# extras proving unknown keys are ignored by ``RadarrMovie.from_api``.
_MOVIE_BODY: dict[str, object] = {
    "id": 9,
    "title": "Your Name.",
    "monitored": True,
    "imdbId": "tt5311514",
    "tmdbId": 372058,
    "sortTitle": "your name",
    "year": 2016,
}


@respx.mock
def test_all_movies_parses_into_radarr_item_shape() -> None:
    """``all_movies`` parses each raw record into a ``RadarrMovie`` satisfying
    the ``RadarrItem`` protocol (checked from ``object``: the runtime
    counterpart of the client's typed claim), with correctly-typed id fields.
    """

    route = respx.get(f"{_BASE}/movie").respond(json=[_MOVIE_BODY])
    client = _make_client()
    movies: list[object] = list(client.all_movies())

    [movie] = movies
    assert isinstance(movie, RadarrItem)
    assert movie.id == 9
    assert movie.title == "Your Name."
    assert movie.tmdbId == 372058
    assert movie.imdbId == "tt5311514"
    assert movie.monitored is True
    request = route.calls.last.request
    # The key rides the X-Api-Key header, never the URL (it would leak via logs).
    assert "apikey" not in str(request.url)
    assert request.headers["X-Api-Key"] == _KEY


@respx.mock
def test_all_movies_skips_non_dict_elements() -> None:
    """A stray non-object element in the movie array is skipped, never crashed on."""

    respx.get(f"{_BASE}/movie").respond(json=[_MOVIE_BODY, "stray", 42])
    movies = _make_client().all_movies()

    assert [m.id for m in movies] == [9]


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

    movies: list[RadarrItem] = [
        RadarrMovie(id=1, title="Imdb Only", tmdbId=111, imdbId="tt1"),
        RadarrMovie(id=2, title="Unmatched", tmdbId=222, imdbId="tt2"),
    ]
    client = FakeRadarrClient(movies=movies)
    id_sets = _RecordingIdSets({"imdb_id": {"tt1"}})

    kept = collect_anime_movies(client, id_sets, None)

    assert id_sets.calls == ["tmdb_movie_id", "imdb_id"]
    assert [m.id for m in kept] == [1]


def test_movie_file_from_api_nullability() -> None:
    """Missing releaseGroup/size parse to None; unknown keys are ignored."""

    assert MovieFile.from_api({}) == MovieFile(None, None)
    parsed = MovieFile.from_api({"releaseGroup": "SubsPlease", "size": 123, "id": 9, "unknown": True})
    assert parsed == MovieFile(release_group="SubsPlease", size=123)
