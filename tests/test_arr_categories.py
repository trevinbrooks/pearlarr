# pyright: strict
"""The download-client category fallback (`ArrCategoryResolver`).

Pins the per-category precedence (explicit config > the arr's own qBittorrent
download client > blank) and the `""` opt-out, the LAZY fetch contract (none
at construction, one at first use, memoized on success ONLY - a failed fetch
is retried at the next use), the priority choice among several enabled clients,
the per-arr field names (`tvCategory`/`tvImportedCategory` vs
`movieCategory`/`movieImportedCategory`), and the fail-open matrix: a fetch
failure, an absent/disabled qBittorrent client, and blank/junk fields all
leave omitted categories blank. `is_unwatched` answers off the fetched
clients first (a fetched blank is authoritative False), the
configured grab on a failed fetch, and False when both are unknown.
`watch` answers off the fetched clients only: every enabled qBittorrent
client counts, a blank category or an absent client watches everything, and
a subcategory counts under its parent unless qBittorrent says its
subcategories are off. Whatever could not be read answers `UNREAD`.
"""

from dataclasses import replace

import httpx
import pytest
import qbittorrentapi
import respx

from pearlarr.arr_categories import ArrCategoryResolver, CategorySources, Watch
from pearlarr.config import Arr, ArrSettings
from pearlarr.output import Diagnostic, Severity

from .builders import download_client_json, sonarr_client_fields
from .fakes import QbitCategoryReads, RecordingHub, bind_arr_http, diagnostic_messages

_URL = "http://arr.test"


def _resolver(
    config: ArrSettings | None = None,
    *,
    arr: Arr = Arr.SONARR,
    qbit: QbitCategoryReads | None = None,
) -> tuple[ArrCategoryResolver, RecordingHub]:
    """A resolver over a freshly bound transport (the production shape: retries=0) and a qBittorrent stub."""

    http, recording = bind_arr_http(_URL)
    sources = CategorySources(replace(http, retries=0), (qbit or QbitCategoryReads()).client())
    return ArrCategoryResolver(arr, config or ArrSettings(), sources), recording


def _sonarr_watches_tv_sonarr() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])


@respx.mock
def test_omitted_categories_adopt_the_sonarr_client_values_with_one_lazy_fetch() -> None:
    # Nothing is fetched at construction; the first use fetches once and the
    # success is memoized - a disabled sibling ahead of the client is passed over.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[download_client_json(sonarr_client_fields(), enable=False), download_client_json(sonarr_client_fields())],
    )
    resolver, recording = _resolver()

    assert route.call_count == 0
    assert resolver.grab_category() == "tv-sonarr"
    assert resolver.post_import_category() == "sonarr-done"
    assert resolver.grab_category() == "tv-sonarr"
    assert route.call_count == 1
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_radarr_reads_the_movie_field_names() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(
                [
                    {"name": "movieCategory", "value": "radarr"},
                    {"name": "movieImportedCategory", "value": "radarr-done"},
                ],
            ),
        ],
    )
    resolver, _ = _resolver(arr=Arr.RADARR)

    assert resolver.grab_category() == "radarr"
    assert resolver.post_import_category() == "radarr-done"


@respx.mock
def test_explicit_config_wins_without_a_fetch() -> None:
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="anime", post_import_category="done"))

    assert resolver.grab_category() == "anime"
    assert resolver.post_import_category() == "done"
    assert route.call_count == 0


@respx.mock
def test_a_blank_string_opts_out_without_a_fetch() -> None:
    # The explicit opt-out: `""` (or whitespace-only) means no category at
    # all - never the client's - and costs no fetch.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="", post_import_category="   "))

    assert resolver.grab_category() is None
    assert resolver.post_import_category() is None
    assert route.call_count == 0


@respx.mock
def test_a_lone_omitted_category_adopts_only_its_own_fallback() -> None:
    # Per-category precedence: the explicit grab category stays, the omitted
    # post-import one adopts the client's - via the one fetch.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.grab_category() == "anime"
    assert resolver.post_import_category() == "sonarr-done"
    assert route.call_count == 1


_ANIME_THEN_DONE = ArrSettings(torrent_category="anime", post_import_category="done")


@respx.mock
@pytest.mark.parametrize(
    ("config", "with_post_import", "expected"),
    [
        pytest.param(_ANIME_THEN_DONE, True, ("anime", "done"), id="both"),
        pytest.param(_ANIME_THEN_DONE, False, ("anime",), id="grab-alone"),
        pytest.param(ArrSettings(torrent_category="anime"), False, ("anime",), id="an-omitted-one-not-asked-for"),
        pytest.param(
            ArrSettings(torrent_category="anime", post_import_category="anime"), True, ("anime",), id="one-name-once"
        ),
        pytest.param(ArrSettings(torrent_category="", post_import_category="done"), True, ("done",), id="blank-grab"),
    ],
)
def test_download_categories_names_each_category_once_and_no_blank(
    config: ArrSettings, with_post_import: bool, expected: tuple[str, ...]
) -> None:
    # None of these needs the arr's clients, which can't be read here.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(500)
    resolver, _ = _resolver(config)

    assert resolver.download_categories(with_post_import=with_post_import) == expected


@respx.mock
@pytest.mark.parametrize(
    ("config", "with_post_import", "expected"),
    [
        # The configured grab category alone is no answer while the post-import one can't be read.
        pytest.param(
            ArrSettings(torrent_category="anime"), True, ("anime", "sonarr-done"), id="an-omitted-post-import"
        ),
        pytest.param(ArrSettings(post_import_category="done"), False, ("tv-sonarr",), id="an-omitted-grab"),
    ],
)
def test_download_categories_waits_for_a_category_left_to_the_arrs_clients(
    config: ArrSettings, with_post_import: bool, expected: tuple[str, ...]
) -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=[download_client_json(sonarr_client_fields())]),
    ]
    resolver, _ = _resolver(config)

    assert resolver.download_categories(with_post_import=with_post_import) is None
    assert resolver.download_categories(with_post_import=with_post_import) == expected


def test_no_transport_stays_blank_and_fetchless() -> None:
    # sources=None (dry run, or absent connection keys): omitted categories
    # stay blank and the explicit value still passes through.
    resolver = ArrCategoryResolver(Arr.SONARR, ArrSettings(torrent_category="anime"), None)

    assert resolver.grab_category() == "anime"
    assert resolver.post_import_category() is None


@respx.mock
def test_priority_beats_list_order_between_enabled_clients() -> None:
    # Two enabled qBittorrent clients: the lowest priority NUMBER wins even
    # from behind; junk priority converts to 50 (last place), so a malformed
    # client never outranks a clean one.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields("junky", "junky-done"), priority="high"),
            download_client_json(sonarr_client_fields("low", "low-done"), priority=10),
            download_client_json(sonarr_client_fields("preferred", "preferred-done"), priority=1),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab_category() == "preferred"
    assert resolver.post_import_category() == "preferred-done"


@respx.mock
def test_equal_priorities_keep_list_order() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields("first", "first-done"), priority=25),
            download_client_json(sonarr_client_fields("second", "second-done"), priority=25),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab_category() == "first"


@respx.mock
def test_junk_enable_spellings_read_as_disabled() -> None:
    # bool("false") / bool([0]) would be True: the lax conversion reads both as
    # disabled, so junk clients never steal the choice from a clean sibling.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields("stringly", "stringly-done"), enable="false"),
            download_client_json(sonarr_client_fields("listy", "listy-done"), enable=[0]),
            download_client_json(sonarr_client_fields()),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab_category() == "tv-sonarr"


@respx.mock
def test_fetch_failure_fails_open_with_one_warning_and_retries_at_next_use() -> None:
    # The failed fetch is NOT memoized: the next use fetches again (an arr blip costs
    # only the work racing it), and the one warning states the consequence.
    route = respx.get(f"{_URL}/api/v3/downloadclient")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=[download_client_json(sonarr_client_fields())]),
    ]
    resolver, recording = _resolver()

    assert resolver.grab_category() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Could not fetch the Sonarr download clients (status code 500) - blank categories stay blank, "
            "and post-import moves and imports from unwatched categories wait"
        ),
    ]

    assert resolver.post_import_category() == "sonarr-done"
    assert resolver.grab_category() == "tv-sonarr"  # the recovered fetch is memoized
    assert route.call_count == 2


@respx.mock
def test_no_enabled_qbittorrent_client_fails_open_quietly_and_memoizes() -> None:
    # A disabled qBittorrent client and an enabled non-qBittorrent one both
    # count for nothing: blanks stay blank, quietly (a DEBUG line, no hub warning) -
    # and the successful fetch memoizes its negative answer.
    route = respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields(), enable=False),
            download_client_json([{"name": "tvCategory", "value": "tv"}], implementation="Transmission"),
        ],
    )
    resolver, recording = _resolver()

    assert resolver.grab_category() is None
    assert resolver.post_import_category() is None
    assert route.call_count == 1
    assert recording.of_type(Diagnostic) == []


@respx.mock
def test_blank_and_junk_category_fields_fail_open() -> None:
    # tvCategory is empty, tvImportedCategory absent, and the junk entries
    # (non-object, junk-typed value) are skipped without dropping the client.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(
                [
                    "junk",
                    {"name": "tvCategory", "value": ""},
                    {"name": "tvImportedCategory", "value": 7},
                ],
            ),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.grab_category() is None
    assert resolver.post_import_category() is None


# --- is_unwatched() ---------------------------------------------------------


@respx.mock
def test_is_unwatched_when_the_fetched_grab_category_differs() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])
    resolver, _ = _resolver()

    assert resolver.is_unwatched("sonarr-done") is True
    assert resolver.is_unwatched("tv-sonarr") is False  # post == watched: the queue record stays


@respx.mock
def test_is_unwatched_fetched_blank_is_authoritative_false() -> None:
    # A blank TvCategory means Sonarr sees every torrent, so no move ever clears
    # the queue record - even when pearlarr's grab category is configured.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[download_client_json([{"name": "tvCategory", "value": ""}])],
    )
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.is_unwatched("sonarr-done") is False


@respx.mock
def test_is_unwatched_fetch_failure_falls_back_to_the_configured_grab() -> None:
    # The single-attempt fetch handle can blip. The configured grab approximates
    # the watched category so one dropped connection can't force the explicit
    # delete (and its permanent Ignored marker) on a category-preferring setup.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(status_code=500)
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.is_unwatched("sonarr-done") is True
    assert resolver.is_unwatched("anime") is False


@respx.mock
def test_is_unwatched_fetch_failure_without_config_stays_false() -> None:
    # Both sources unknown: conservative False keeps the explicit close.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(status_code=500)
    resolver, _ = _resolver()

    assert resolver.is_unwatched("sonarr-done") is False


def test_is_unwatched_without_transport_reads_the_configured_grab() -> None:
    resolver = ArrCategoryResolver(Arr.SONARR, ArrSettings(torrent_category="anime"), None)

    assert resolver.is_unwatched("sonarr-done") is True
    assert resolver.is_unwatched("anime") is False


def test_is_unwatched_explicit_blank_opt_out_is_false() -> None:
    # `""` opts out of categories entirely: nothing is watched by pearlarr's reckoning.
    resolver = ArrCategoryResolver(Arr.SONARR, ArrSettings(torrent_category=""), None)

    assert resolver.is_unwatched("sonarr-done") is False


@respx.mock
def test_is_unwatched_compares_exactly() -> None:
    # qBittorrent categories are case-sensitive, so the compare must be too.
    respx.get(f"{_URL}/api/v3/downloadclient").respond(json=[download_client_json(sonarr_client_fields())])
    resolver, _ = _resolver()

    assert resolver.is_unwatched("TV-Sonarr") is True


# --- watch() -----------------------------------------------------------------


@respx.mock
def test_watch_goes_by_the_categories_the_clients_name() -> None:
    # Torrents are added to `anime` and Sonarr watches `tv-sonarr`, so the run's grabs are the unwatched ones.
    _sonarr_watches_tv_sonarr()
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.category_watch("tv-sonarr") is Watch.WATCHED
    assert resolver.category_watch("anime") is Watch.UNWATCHED
    assert resolver.category_watch("") is Watch.UNWATCHED  # an uncategorized torrent
    assert resolver.category_watch("TV-Sonarr") is Watch.UNWATCHED  # case-sensitive, as qBittorrent is
    assert resolver.category_watch("tv-sonarr-done") is Watch.UNWATCHED  # a longer name is another category
    assert resolver.category_watch("tv") is Watch.UNWATCHED  # and so is the parent of a watched one


@respx.mock
def test_every_enabled_client_counts_as_watching() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields("tv-sonarr", "sonarr-done"), priority=1),
            download_client_json(sonarr_client_fields("anime", "anime-done"), priority=50),
            download_client_json(sonarr_client_fields("old", "old-done"), enable=False),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.category_watch("anime") is Watch.WATCHED
    assert resolver.is_unwatched("anime") is False
    assert resolver.category_watch("old") is Watch.UNWATCHED


@respx.mock
@pytest.mark.parametrize("preferences", [{"use_subcategories": True}, {}], ids=["on", "not-reported"])
def test_a_subcategory_is_watched_under_the_category_a_client_names(preferences: dict[str, object]) -> None:
    _sonarr_watches_tv_sonarr()
    resolver, _ = _resolver(qbit=QbitCategoryReads(None, preferences))

    assert resolver.category_watch("tv-sonarr/done") is Watch.WATCHED
    assert resolver.is_unwatched("tv-sonarr/done") is False


@respx.mock
def test_with_subcategories_off_a_name_under_a_watched_one_is_its_own_category() -> None:
    _sonarr_watches_tv_sonarr()
    resolver, _ = _resolver(qbit=QbitCategoryReads(None, {"use_subcategories": False}))

    assert resolver.category_watch("tv-sonarr/done") is Watch.UNWATCHED
    assert resolver.is_unwatched("tv-sonarr/done") is True
    assert resolver.category_watch("tv-sonarr") is Watch.WATCHED


@respx.mock
def test_an_unreadable_subcategory_setting_leaves_a_subcategory_unread_until_it_reads() -> None:
    _sonarr_watches_tv_sonarr()
    down = qbittorrentapi.APIConnectionError("down")
    resolver, _ = _resolver(qbit=QbitCategoryReads(None, down, down, {"use_subcategories": False}))

    assert resolver.category_watch("tv-sonarr/done") is Watch.UNREAD
    assert resolver.is_unwatched("tv-sonarr/done") is False
    assert resolver.category_watch("tv-sonarr/done") is Watch.UNWATCHED


@respx.mock
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
def test_a_client_with_a_blank_category_watches_every_category(blank: str) -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[
            download_client_json(sonarr_client_fields(), priority=1),
            download_client_json([{"name": "tvCategory", "value": blank}], priority=50),
        ],
    )
    resolver, _ = _resolver()

    assert resolver.category_watch("sonarr-done") is Watch.WATCHED
    assert resolver.is_unwatched("sonarr-done") is False


@respx.mock
def test_with_no_qbittorrent_client_nothing_is_called_unwatched() -> None:
    respx.get(f"{_URL}/api/v3/downloadclient").respond(
        json=[download_client_json([{"name": "tvCategory", "value": "tv"}], implementation="Transmission")],
    )
    resolver, _ = _resolver()

    assert resolver.category_watch("sonarr-done") is Watch.WATCHED


@respx.mock
def test_a_failed_fetch_is_unread_and_asked_again() -> None:
    # No configured approximation here, unlike `is_unwatched`: a wrong answer imports a download twice.
    respx.get(f"{_URL}/api/v3/downloadclient").mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json=[download_client_json(sonarr_client_fields())])],
    )
    resolver, _ = _resolver(ArrSettings(torrent_category="anime"))

    assert resolver.category_watch("sonarr-done") is Watch.UNREAD
    assert resolver.category_watch("sonarr-done") is Watch.UNWATCHED


# --- torrent_watch() ---------------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    ("category", "watch"),
    [
        pytest.param("manual", Watch.UNWATCHED, id="another-category"),
        pytest.param("", Watch.UNWATCHED, id="uncategorized"),
        pytest.param("tv-sonarr", Watch.WATCHED, id="the-watched-category"),
        pytest.param(None, Watch.UNREAD, id="no-such-torrent"),
        pytest.param(qbittorrentapi.APIConnectionError("down"), Watch.UNREAD, id="failed-read"),
    ],
)
def test_torrent_watch_goes_by_the_category_qbittorrent_holds_the_torrent_in(
    category: str | Exception | None,
    watch: Watch,
) -> None:
    _sonarr_watches_tv_sonarr()
    resolver, _ = _resolver(qbit=QbitCategoryReads(category))

    assert resolver.torrent_watch("h") is watch
