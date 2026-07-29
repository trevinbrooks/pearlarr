# pyright: strict
"""The download-client category fallback (`resolve_arr_categories`).

Pins the per-category precedence (explicit config > the arr's own qBittorrent
download client > blank), the at-most-once fetch (and no fetch at all when
both categories are explicit or no transport is bound), the per-arr field
names (`tvCategory`/`tvImportedCategory` vs `movieCategory`/
`movieImportedCategory`), and the fail-open matrix: a fetch failure, a
missing/disabled qBittorrent client, and blank/junk fields all leave blank
categories blank.
"""

from collections.abc import Sequence
from dataclasses import replace

import httpx
import respx

from pearlarr.arr_categories import ArrCategories, resolve_arr_categories
from pearlarr.arr_http import ArrHttp
from pearlarr.config import Arr, ArrSettings
from pearlarr.output import Diagnostic, Severity, install_hub
from pearlarr.output.recording import RecordingHub

_URL = "http://arr.test"


def _bind() -> tuple[ArrHttp, RecordingHub]:
    """A bound transport plus a fresh recording hub for the fail-open warnings."""

    recording = RecordingHub()
    install_hub(recording.hub)  # conftest teardown restores the default
    http = ArrHttp.bind(client=httpx.Client(), url=_URL, api_key="k", label="Sonarr", sleep=lambda _s: None)
    return http, recording


def _client(
    fields: Sequence[object],
    *,
    enable: bool = True,
    implementation: str = "QBittorrent",
) -> dict[str, object]:
    """One realistic `DownloadClientResource` body carrying `fields` (opaque JSON, junk allowed)."""

    return {
        "enable": enable,
        "protocol": "torrent",
        "priority": 1,
        "name": "qBittorrent",
        "fields": fields,
        "implementationName": "qBittorrent",
        "implementation": implementation,
        "configContract": "QBittorrentSettings",
        "id": 1,
    }


_SONARR_FIELDS: list[dict[str, object]] = [
    {"name": "host", "value": "localhost"},
    {"name": "port", "value": 8080},
    {"name": "tvCategory", "value": "tv-sonarr"},
    {"name": "tvImportedCategory", "value": "sonarr-done"},
]


@respx.mock
def test_blank_categories_adopt_the_sonarr_client_values() -> None:
    # The enabled qBittorrent client's categories fill both blanks; a disabled
    # sibling ahead of it is passed over.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[_client(_SONARR_FIELDS, enable=False), _client(_SONARR_FIELDS)],
    )
    http, recording = _bind()

    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(), http)

    assert resolved == ArrCategories(grab="tv-sonarr", post_import="sonarr-done")
    assert route.call_count == 1
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_radarr_reads_the_movie_field_names() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(
                [
                    {"name": "movieCategory", "value": "radarr"},
                    {"name": "movieImportedCategory", "value": "radarr-done"},
                ],
            ),
        ],
    )
    http, _ = _bind()

    resolved = resolve_arr_categories(Arr.RADARR, ArrSettings(), http)

    assert resolved == ArrCategories(grab="radarr", post_import="radarr-done")


@respx.mock
def test_explicit_config_wins_without_a_fetch() -> None:
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[_client(_SONARR_FIELDS)])
    http, _ = _bind()
    config = ArrSettings(torrent_category="anime", post_import_category="done")

    resolved = resolve_arr_categories(Arr.SONARR, config, http)

    assert resolved == ArrCategories(grab="anime", post_import="done")
    assert route.call_count == 0


@respx.mock
def test_a_lone_blank_adopts_only_its_own_fallback() -> None:
    # Per-category precedence: the explicit grab category stays, the blank
    # post-import one adopts the client's - via the one fetch.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[_client(_SONARR_FIELDS)])
    http, _ = _bind()

    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(torrent_category="anime"), http)

    assert resolved == ArrCategories(grab="anime", post_import="sonarr-done")
    assert route.call_count == 1


def test_no_transport_passes_the_config_through() -> None:
    # http=None (preview run, or missing connection keys): no fetch, and an
    # empty-string category folds to the same None a blank YAML key yields.
    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(torrent_category=""), None)

    assert resolved == ArrCategories(grab=None, post_import=None)


@respx.mock
def test_fetch_failure_fails_open_with_one_warning() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(status_code=500)
    http, recording = _bind()

    # The production handle is a no-retry clone (`RunDeps.build`); mirror it.
    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(), replace(http, retries=0))

    assert resolved == ArrCategories(grab=None, post_import=None)
    [warning] = recording.of_type(Diagnostic)
    assert warning.severity is Severity.WARNING
    expected = "Could not fetch the Sonarr download clients (status code 500) - blank categories stay blank"
    assert warning.message == expected


@respx.mock
def test_no_enabled_qbittorrent_client_fails_open() -> None:
    # A disabled qBittorrent client and an enabled non-qBittorrent one both
    # miss: blanks stay blank, quietly (a DEBUG breadcrumb, no hub warning).
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(_SONARR_FIELDS, enable=False),
            _client([{"name": "tvCategory", "value": "tv"}], implementation="Transmission"),
        ],
    )
    http, recording = _bind()

    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(), http)

    assert resolved == ArrCategories(grab=None, post_import=None)
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_blank_and_junk_category_fields_fail_open() -> None:
    # tvCategory is empty, tvImportedCategory absent, and the junk entries
    # (non-object, junk-typed value) are skipped without dropping the client.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(
                [
                    "junk",
                    {"name": "tvCategory", "value": ""},
                    {"name": "tvImportedCategory", "value": 7},
                ],
            ),
        ],
    )
    http, _ = _bind()

    resolved = resolve_arr_categories(Arr.SONARR, ArrSettings(), http)

    assert resolved == ArrCategories(grab=None, post_import=None)
