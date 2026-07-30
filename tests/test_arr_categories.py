# pyright: strict
"""The download-client category fallback (`ArrCategoryResolver`).

Pins the per-category precedence (explicit config > the arr's own qBittorrent
download client > blank) and the `""` opt-out, the LAZY fetch contract (none
at construction, one at first use, memoized on success ONLY - a failed fetch
is retried at the next use), the priority pick among several enabled clients,
the per-arr field names (`tvCategory`/`tvImportedCategory` vs
`movieCategory`/`movieImportedCategory`), and the fail-open matrix: a fetch
failure, a missing/disabled qBittorrent client, and blank/junk fields all
leave omitted categories blank.
"""

from collections.abc import Sequence
from dataclasses import replace

import httpx
import respx

from pearlarr.arr_categories import ArrCategoryResolver
from pearlarr.config import Arr, ArrSettings
from pearlarr.output import Diagnostic, Severity
from pearlarr.output.recording import RecordingHub

from .fakes import bind_arr_http, diagnostic_messages

_URL = "http://arr.test"


def _resolver(
    config: ArrSettings | None = None,
    *,
    arr: Arr = Arr.SONARR,
) -> tuple[ArrCategoryResolver, RecordingHub]:
    """A resolver over a freshly bound transport (the production shape: retries=0)."""

    http, recording = bind_arr_http(_URL)
    return ArrCategoryResolver(arr, config or ArrSettings(), replace(http, retries=0)), recording


def _client(
    fields: Sequence[object],
    *,
    enable: object = True,
    implementation: str = "QBittorrent",
    priority: object = 1,
) -> dict[str, object]:
    """One realistic `DownloadClientResource` body carrying `fields` (opaque JSON, junk allowed)."""

    return {
        "enable": enable,
        "protocol": "torrent",
        "priority": priority,
        "name": "qBittorrent",
        "fields": fields,
        "implementationName": "qBittorrent",
        "implementation": implementation,
        "configContract": "QBittorrentSettings",
        "id": 1,
    }


def _sonarr_fields(grab: str = "tv-sonarr", post_import: str = "sonarr-done") -> list[dict[str, object]]:
    return [
        {"name": "host", "value": "localhost"},
        {"name": "port", "value": 8080},
        {"name": "tvCategory", "value": grab},
        {"name": "tvImportedCategory", "value": post_import},
    ]


@respx.mock
def test_omitted_categories_adopt_the_sonarr_client_values_with_one_lazy_fetch() -> None:
    # Nothing is fetched at construction; the first use fetches once and the
    # success is memoized - a disabled sibling ahead of the client is passed over.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[_client(_sonarr_fields(), enable=False), _client(_sonarr_fields())],
    )
    resolver, recording = _resolver()

    assert route.call_count == 0
    assert resolver.grab() == "tv-sonarr"
    assert resolver.post_import() == "sonarr-done"
    assert resolver.grab() == "tv-sonarr"
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
    resolver, _ = _resolver(arr=Arr.RADARR)

    assert resolver.grab() == "radarr"
    assert resolver.post_import() == "radarr-done"


@respx.mock
def test_explicit_config_wins_without_a_fetch() -> None:
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[_client(_sonarr_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="anime", post_import_category="done"))

    assert resolver.grab() == "anime"
    assert resolver.post_import() == "done"
    assert route.call_count == 0


@respx.mock
def test_a_blank_string_opts_out_without_a_fetch() -> None:
    # The explicit opt-out: `""` (or whitespace-only) means no category at
    # all - never the client's - and costs no fetch.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[_client(_sonarr_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="", post_import_category="   "))

    assert resolver.grab() is None
    assert resolver.post_import() is None
    assert route.call_count == 0


@respx.mock
def test_a_lone_omitted_category_adopts_only_its_own_fallback() -> None:
    # Per-category precedence: the explicit grab category stays, the omitted
    # post-import one adopts the client's - via the one fetch.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[_client(_sonarr_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.grab() == "anime"
    assert resolver.post_import() == "sonarr-done"
    assert route.call_count == 1


def test_no_transport_stays_blank_and_fetchless() -> None:
    # http=None (preview run, or missing connection keys): omitted categories
    # stay blank and the explicit value still passes through.
    resolver = ArrCategoryResolver(Arr.SONARR, ArrSettings(torrent_category="anime"), None)

    assert resolver.grab() == "anime"
    assert resolver.post_import() is None


@respx.mock
def test_priority_beats_list_order_between_enabled_clients() -> None:
    # Two enabled qBittorrent clients: the lowest priority NUMBER wins even
    # from behind; junk priority folds to 50 (last place), so a malformed
    # record never outranks a clean one.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(_sonarr_fields("junky", "junky-done"), priority="high"),
            _client(_sonarr_fields("low", "low-done"), priority=10),
            _client(_sonarr_fields("preferred", "preferred-done"), priority=1),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab() == "preferred"
    assert resolver.post_import() == "preferred-done"


@respx.mock
def test_equal_priorities_keep_list_order() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(_sonarr_fields("first", "first-done"), priority=25),
            _client(_sonarr_fields("second", "second-done"), priority=25),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab() == "first"


@respx.mock
def test_junk_enable_spellings_read_as_disabled() -> None:
    # bool("false") / bool([0]) would be True: the lax fold reads both as
    # disabled, so junk records never steal the pick from a clean sibling.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(_sonarr_fields("stringly", "stringly-done"), enable="false"),
            _client(_sonarr_fields("listy", "listy-done"), enable=[0]),
            _client(_sonarr_fields()),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab() == "tv-sonarr"


@respx.mock
def test_fetch_failure_fails_open_with_one_warning_and_retries_at_next_use() -> None:
    # The miss is NOT memoized: the next use fetches again (an arr blip costs
    # only the work racing it), and the one warning states the consequence.
    route = respx.get(f"{_URL}/api/v3/downloadclient")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=[_client(_sonarr_fields())]),
    ]
    resolver, recording = _resolver()

    assert resolver.grab() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch the Sonarr download clients (status code 500) - blank categories stay blank",
    ]

    assert resolver.post_import() == "sonarr-done"
    assert resolver.grab() == "tv-sonarr"  # the healed fetch is memoized
    assert route.call_count == 2


@respx.mock
def test_no_enabled_qbittorrent_client_fails_open_quietly_and_memoizes() -> None:
    # A disabled qBittorrent client and an enabled non-qBittorrent one both
    # miss: blanks stay blank, quietly (a DEBUG breadcrumb, no hub warning) -
    # and the successful fetch memoizes its negative answer.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            _client(_sonarr_fields(), enable=False),
            _client([{"name": "tvCategory", "value": "tv"}], implementation="Transmission"),
        ],
    )
    resolver, recording = _resolver()

    assert resolver.grab() is None
    assert resolver.post_import() is None
    assert route.call_count == 1
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
    resolver, _ = _resolver()

    assert resolver.grab() is None
    assert resolver.post_import() is None
