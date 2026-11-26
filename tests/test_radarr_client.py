# pyright: strict
"""Tests for `RadarrClient`, the Radarr REST adapter, and for `collect_anime_movies`.

Each test drives a real client over `respx` and checks what one method decides: the request it sends, what
it reads back, and what its caller gets when the read fails. How a failed request is retried, coalesced and
worded belongs to the shared handle and is shown once in `test_arr_http`.
"""

from collections.abc import Set as AbstractSet
from typing import override

import httpx
import pytest
import respx

from pearlarr.arr_http import ArrHttp
from pearlarr.arr_models import MovieFile, RadarrItem, RadarrMovie
from pearlarr.boundary import BoundaryContractError
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
    """Each of the movie's files reads into its group and size, and a file Radarr names no group for reads as None."""

    body: list[object] = [{"releaseGroup": "SubsPlease", "size": 123}, {"size": 456}]
    route = respx.get(f"{_BASE}/moviefile").respond(json=body)
    files = _make_client().movie_files(7)

    assert files == [MovieFile(release_group="SubsPlease", size=123), MovieFile(release_group=None, size=456)]
    assert dict(route.calls.last.request.url.params) == {"movieId": "7"}


@respx.mock
def test_movie_files_non_200_returns_empty_and_warns() -> None:
    """A failed read gives [] with a warning, so the caller goes on as if the movie had no files."""

    respx.get(f"{_BASE}/moviefile").respond(status_code=500)
    recording = install_recording_hub()
    files = _make_client().movie_files(7)

    assert files == []
    [warning] = diagnostic_messages(recording, Severity.WARNING)
    assert warning == "Could not fetch files for movie 7 from Radarr (status code 500) - assuming none"


@respx.mock
def test_history_since_asks_for_the_records_alone() -> None:
    """Radarr's history read asks for every event since the date, with the movie flag off.

    How each record reads, `movieId` as its item id included, is the model's (`test_arr_models`,
    `test_boundary_parity`).
    """

    route = respx.get(f"{_BASE}/history/since").respond(json=[])
    _make_client().history_since("2026-06-30T08:00:00Z")

    assert dict(route.calls.last.request.url.params) == {"date": "2026-06-30T08:00:00Z", "includeMovie": "false"}


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
    """`all_movies` reads each movie into the `RadarrItem` fields a run keys on."""

    respx.get(f"{_BASE}/movie").respond(json=[_MOVIE_BODY])
    [movie] = _make_client().all_movies()

    assert movie.id == 9
    assert movie.title == "Your Name."
    assert movie.tmdbId == 372058
    assert movie.imdbId == "tt5311514"
    assert movie.monitored is True


@respx.mock
def test_all_movies_all_invalid_payload_raises_contract_error() -> None:
    """A movie list where nothing reads as a movie raises, so a broken answer never reads as an empty library."""

    respx.get(f"{_BASE}/movie").respond(json=["junk", 42])
    with pytest.raises(BoundaryContractError):
        _make_client().all_movies()


class _PresetIdSets(KnownExternalIds):
    """The Anime-IDs candidate set for each id kind, preset."""

    def __init__(self, sets: dict[IdKind, set[int | str]]) -> None:
        self._sets = sets

    @override
    def known_external_ids(self, kind: IdKind) -> AbstractSet[int | str]:
        return self._sets.get(kind, set())


def test_collect_anime_movies_wires_id_spaces() -> None:
    """A movie is kept when its TMDB id or its IMDb id is a mapped one, each read from its own field."""

    movies: list[RadarrItem] = [
        RadarrMovie(id=1, title="Imdb Only", tmdbId=111, imdbId="tt1"),
        RadarrMovie(id=2, title="Tmdb Only", tmdbId=222, imdbId="tt2"),
        RadarrMovie(id=3, title="Unmatched", tmdbId=333, imdbId="tt3"),
    ]
    id_sets = _PresetIdSets({"tmdb_movie_id": {222}, "imdb_id": {"tt1"}})

    kept = collect_anime_movies(FakeRadarrClient(movies=movies), id_sets, None)

    assert [m.id for m in kept] == [1, 2]
