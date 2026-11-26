# pyright: strict
"""Direct tests for the `radarr_client` module's contracts.

Mirrors `test_sonarr_client`: a REAL `RadarrClient` (construction is
network-free) with every endpoint going through the httpx-based `ArrHttp`, mocked
via `respx`. Pins the decode into the typed `MovieFile` /
`RadarrMovie` views and the degrade-to-empty guard (non-200 / transient
request error -> `[]` + a warning), so an unreachable Radarr never unwinds the
run, plus the `collect_anime_movies` wiring.
"""

from collections.abc import Set as AbstractSet
from typing import override

import httpx
import respx

from pearlarr.arr_http import ArrHttp
from pearlarr.arr_models import HistoryRecord, MovieFile, RadarrItem, RadarrMovie
from pearlarr.mapping_store import IdKind
from pearlarr.mappings import KnownExternalIds
from pearlarr.output import Severity
from pearlarr.radarr_client import RadarrClient, collect_anime_movies

from .fakes import FakeRadarrClient, diagnostic_messages, install_recording_hub

_URL = "http://radarr.test"
_BASE = f"{_URL}/api/v3"


def _make_client() -> RadarrClient:
    """Build a real `RadarrClient` (construction is network-free).

    The bind's `sleep` is stubbed out so fail-open tests don't wait out real
    retry backoffs.
    """

    return RadarrClient(
        http=ArrHttp.bind(
            client=httpx.Client(),
            url=_URL,
            api_key="testkey",
            label="Radarr",
            sleep=lambda _s: None,
        ),
    )


@respx.mock
def test_movie_files_decodes_resources_and_builds_request() -> None:
    body: list[object] = [{"releaseGroup": "SubsPlease", "size": 123, "id": 9}]
    route = respx.get(f"{_BASE}/moviefile").respond(json=body)
    files = _make_client().movie_files(7)

    assert files == [MovieFile(release_group="SubsPlease", size=123)]
    assert "movieId=7" in str(route.calls.last.request.url)


@respx.mock
def test_movie_files_non_200_returns_empty_and_warns() -> None:
    """A failed read gives [] with a warning, so the caller goes on as if the movie had no files."""

    respx.get(f"{_BASE}/moviefile").respond(status_code=404)
    recording = install_recording_hub()
    files = _make_client().movie_files(7)

    assert files == []
    [warning] = diagnostic_messages(recording, Severity.WARNING)
    assert warning == "Could not fetch files for movie 7 from Radarr (status code 404) - assuming none"


@respx.mock
def test_history_since_decodes_records_and_builds_request() -> None:
    """`history_since()` keys the item id on `movieId` and the request pins `includeMovie=false`.

    A record with no `data` object parses to a `None` reason.
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
    url = str(route.calls.last.request.url)
    assert "date=2026-06-30T08%3A00%3A00Z" in url
    assert "includeMovie=false" in url


# A minimal `/api/v3/movie` resource: the consumed item fields plus a couple of
# extras proving unknown keys are ignored by `RadarrMovie`.
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
    """`all_movies` parses each raw resource into a `RadarrMovie` satisfying the `RadarrItem` protocol.

    This is checked from `object` (the runtime counterpart of the client's
    typed claim), with correctly-typed id fields.
    """

    respx.get(f"{_BASE}/movie").respond(json=[_MOVIE_BODY])
    client = _make_client()
    movies: list[object] = list(client.all_movies())

    [movie] = movies
    assert isinstance(movie, RadarrItem)
    assert movie.id == 9
    assert movie.title == "Your Name."
    assert movie.tmdbId == 372058
    assert movie.imdbId == "tt5311514"
    assert movie.monitored is True


class _RecordingIdSets(KnownExternalIds):
    """Recording `KnownExternalIds`: preset per-id-kind sets, calls recorded."""

    def __init__(self, sets: dict[str, set[int | str]]) -> None:
        self._sets = sets
        self.calls: list[str] = []

    @override
    def known_external_ids(self, kind: IdKind) -> AbstractSet[int | str]:
        self.calls.append(kind)
        return self._sets.get(kind, set())


def test_collect_anime_movies_wires_id_spaces() -> None:
    """The candidate sets are pulled for (tmdb_movie_id, imdb_id), in that order.

    An imdb-only match is kept, and `anibridge=None` degrades to empty sets.
    """

    movies: list[RadarrItem] = [
        RadarrMovie(id=1, title="Imdb Only", tmdbId=111, imdbId="tt1"),
        RadarrMovie(id=2, title="Unmatched", tmdbId=222, imdbId="tt2"),
    ]
    client = FakeRadarrClient(movies=movies)
    id_sets = _RecordingIdSets({"imdb_id": {"tt1"}})

    kept = collect_anime_movies(client, id_sets, None)

    assert id_sets.calls == ["tmdb_movie_id", "imdb_id"]
    assert [m.id for m in kept] == [1]


def test_movie_file_nullability() -> None:
    """Missing releaseGroup/size parse to None. Unknown keys are ignored."""

    assert MovieFile.model_validate({}) == MovieFile(release_group=None, size=None)
    parsed = MovieFile.model_validate({"releaseGroup": "SubsPlease", "size": 123, "id": 9, "unknown": True})
    assert parsed == MovieFile(release_group="SubsPlease", size=123)
