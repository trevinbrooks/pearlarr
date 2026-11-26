# pyright: strict
"""Direct unit tests for the pure helpers and retry path in `anilist_client`.

These are otherwise only exercised incidentally. The classification / parsing /
extraction helpers take plain `dict` bodies, so they are tested with no mocks
at all. `AniListClient`'s retry loop is exercised through its public queries,
faked at the `httpx` boundary with the `respx` library (no `unittest.mock`).

The `AniListMediaNode` tests pin its all-None empty-dict validation, the
not-found node contract the AniList gateway depends on.
"""

import json
import re
import time
from typing import cast

import httpx
import pytest
import respx
from pydantic import AliasPath
from respx.models import Call

from pearlarr.anilist_client import (
    API_URL,
    BATCH_QUERY,
    MAX_RETRIES,
    QUERY,
    AniListClient,
    AniListMediaNode,
    lacks_air_dates,
    media_from_body,
)
from pearlarr.boundary import FuzzyDate
from pearlarr.output import Diagnostic, Severity
from pearlarr.paths import PROJECT_URL

from .fakes import diagnostic_messages, install_recording_hub

_FOUND: dict[str, object] = {"data": {"Media": {"id": 1, "episodes": 12}}}
# AniList's answer past its rate limit, with HTTP 429 or, when it soft-throttles, HTTP 200.
_THROTTLED: dict[str, object] = {"errors": [{"message": "Too Many Requests.", "status": 429}], "data": None}
# AniList's answer for an id it doesn't have.
_NOT_FOUND: dict[str, object] = {"data": {"Media": None}, "errors": [{"message": "Not Found.", "status": 404}]}


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Each wait the client asks for, in order, in place of really sleeping."""

    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)
    return waits


def _client() -> AniListClient:
    """A wire client for tests with no narration asserts."""

    return AniListClient(client=httpx.Client())


# --- reading a stored answer --------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_NOT_FOUND, id="unknown id"),
        # What `query` hands back from the request AniList refused.
        pytest.param({"data": None}, id="dataless"),
    ],
)
def test_an_answer_without_a_media_reads_as_the_empty_node(body: dict[str, object]) -> None:
    recording = install_recording_hub()

    assert media_from_body(body) == AniListMediaNode()
    assert recording.of_type(Diagnostic) == []


def test_a_media_that_does_not_fit_the_node_reads_as_the_empty_node_with_one_warning() -> None:
    """AniList sends episodes as a number, so only a stored answer the node now reads differently looks like this."""

    recording = install_recording_hub()

    assert media_from_body({"data": {"Media": {"id": 1, "episodes": "twelve"}}}) == AniListMediaNode()
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Ignoring malformed AniList Media node (episodes: int_parsing)"
    ]


@pytest.mark.parametrize(
    ("media", "lacks"),
    [
        pytest.param({"id": 7, "episodes": 12}, True, id="an id and no startDate key"),
        # AniList answers a date it doesn't know with null parts, and the key is still there.
        pytest.param({"id": 7, "startDate": {"year": None, "month": None, "day": None}}, False, id="null date parts"),
        pytest.param(None, False, id="no such id"),
    ],
)
def test_a_stored_media_without_a_start_date_key_lacks_its_air_dates(
    media: dict[str, object] | None, lacks: bool
) -> None:
    assert lacks_air_dates({"data": {"Media": media}}) is lacks


# --- what the queries ask for -----------------------------------------------


# The node reads a date from all three parts of AniList's date object.
_FUZZY_DATE: object = FuzzyDate


def _node_paths() -> set[str]:
    """What `AniListMediaNode` reads of a Media, as paths: `episodes`, `title.english`, each part of a date."""

    paths: set[str] = set()
    for name, field in AniListMediaNode.model_fields.items():
        alias = field.validation_alias
        if isinstance(alias, AliasPath):
            paths.add(".".join(str(step) for step in alias.path))
            continue
        key = alias if isinstance(alias, str) else name
        if field.annotation is _FUZZY_DATE:
            paths.update(f"{key}.{part}" for part in ("year", "month", "day"))
        else:
            paths.add(key)
    return paths


def _selected_paths(query: str) -> set[str]:
    """What a query asks of each Media, as paths: `episodes`, `title.english`."""

    # The Media selection is everything after the query's last argument list, and nests one level deep.
    selection = query.rsplit(") {", 1)[1]
    paths: set[str] = set()
    for field, subfields in cast("list[tuple[str, str]]", re.findall(r"(\w+)\s*(?:\{([^}]*)\})?", selection)):
        if subfields:
            paths.update(f"{field}.{sub}" for sub in subfields.split())
        else:
            paths.add(field)
    return paths


@pytest.mark.parametrize("query", [pytest.param(QUERY, id="single"), pytest.param(BATCH_QUERY, id="batch")])
def test_each_query_asks_for_everything_the_node_reads(query: str) -> None:
    """Whatever the node reads that a query leaves out would read as missing for every title."""

    assert _node_paths() - _selected_paths(query) == set()


# --- AniListClient.query / query_batch (respx-faked httpx boundary) --------------------------


@respx.mock
@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(200, _FOUND, id="found"),
        pytest.param(404, _NOT_FOUND, id="unknown id"),
    ],
)
def test_an_answer_with_data_comes_back_as_sent(status: int, body: dict[str, object]) -> None:
    """A Media, or AniList's answer that it has no such id, takes one request and leaves AniList reachable."""

    recording = install_recording_hub()
    client = _client()
    route = respx.post(API_URL).respond(status_code=status, json=body)

    assert client.query(1) == body
    assert route.call_count == 1
    assert client.unreachable is False
    assert diagnostic_messages(recording, Severity.WARNING) == []


@respx.mock
def test_every_post_carries_the_project_referer() -> None:
    """AniList refuses a Referer-less request, so the single and batch POSTs both name the project."""

    route = respx.post(API_URL).respond(json={"data": {"Media": {"id": 1}}})

    client = _client()
    client.query(1)
    client.query_batch([1, 2])

    assert route.call_count == 2
    calls = cast("list[Call]", route.calls)
    assert [call.request.headers["Referer"] for call in calls] == [PROJECT_URL, PROJECT_URL]


@respx.mock
def test_each_query_asks_anilist_for_the_ids_it_is_given() -> None:
    """A wrong id would read another title's answer as this one's."""

    route = respx.post(API_URL).respond(json={"data": {"Media": None}})

    client = _client()
    client.query(21)
    client.query_batch([21, 22])

    single, batch = cast("list[Call]", route.calls)
    assert str(single.request.url) == str(batch.request.url) == "https://graphql.anilist.co/"
    assert json.loads(single.request.content) == {"query": QUERY, "variables": {"id": 21}}
    assert json.loads(batch.request.content) == {"query": BATCH_QUERY, "variables": {"ids": [21, 22]}}


@respx.mock
@pytest.mark.usefixtures("slept")
@pytest.mark.parametrize(
    "first",
    [
        # A proxy's 429 page carries no throttle error, so the status alone asks again.
        pytest.param(httpx.Response(429, text="<html>Too Many Requests</html>"), id="429"),
        pytest.param(httpx.Response(200, json=_THROTTLED), id="429 riding a 200"),
        pytest.param(
            httpx.Response(200, json={"errors": [{"message": "Internal Server Error", "status": 500}], "data": None}),
            id="500 riding a 200",
        ),
        # Throttle wording with no status, in any case.
        pytest.param(
            httpx.Response(200, json={"errors": [{"message": "Too Many Requests."}], "data": None}),
            id="too many requests",
        ),
        pytest.param(
            httpx.Response(200, json={"errors": [{"message": "Rate Limit Exceeded"}], "data": None}), id="rate limit"
        ),
        pytest.param(
            httpx.Response(200, json={"errors": [{"message": "Request throttled"}], "data": None}), id="throttle"
        ),
    ],
)
def test_a_rate_limit_or_server_error_is_asked_for_again(first: httpx.Response) -> None:
    """The client waits and asks again, and the answer that follows is the one that comes back."""

    route = respx.post(API_URL).mock(side_effect=[first, httpx.Response(200, json=_FOUND)])

    assert _client().query(1) == _FOUND
    assert route.call_count == 2


@respx.mock
def test_query_batch_reshapes_page_media_by_id() -> None:
    """A Page body is re-keyed to {id: single-id-shaped body}, so the results can populate the run cache directly.

    Junk, id-less and non-int-id elements in the media array are skipped (an id
    unknown to AniList is simply absent, never an error).
    """

    page: dict[str, object] = {
        "data": {"Page": {"media": [{"id": 1, "episodes": 12}, "junk", {"episodes": 9}, {"id": "7", "episodes": 3}]}},
    }
    route = respx.post(API_URL).respond(json=page)

    out = _client().query_batch([1, 2])

    assert route.call_count == 1
    assert out == {1: {"data": {"Media": {"id": 1, "episodes": 12}}}}


# --- retry narration and marking AniList unreachable --------------------------


_DISABLED_MESSAGE = "The AniList API has been temporarily disabled due to severe stability issues."
_DISABLED_BODY: dict[str, object] = {"errors": [{"message": _DISABLED_MESSAGE, "status": 403}], "data": None}
_JUNK_THEN_DISABLED: dict[str, object] = {
    "errors": ["junk", {"message": _DISABLED_MESSAGE, "status": 403}],
    "data": None,
}
_SKIPPING = "skipping further lookups this run"


@respx.mock
@pytest.mark.parametrize(
    ("first", "waited", "line"),
    [
        pytest.param(
            httpx.Response(429, json=_THROTTLED, headers={"Retry-After": "42"}),
            42,
            "AniList rate-limited - waiting 42s (retry 1/3)",
            id="retry-after",
        ),
        pytest.param(
            httpx.Response(429, json=_THROTTLED, headers={"Retry-After": "0"}),
            1,
            "AniList rate-limited - waiting 1s (retry 1/3)",
            id="under a second",
        ),
        pytest.param(
            httpx.Response(429, json=_THROTTLED, headers={"Retry-After": "120"}),
            60,
            "AniList rate-limited - waiting 60s (retry 1/3)",
            id="past a minute",
        ),
        pytest.param(
            httpx.Response(503, text="<html>Service Unavailable</html>", headers={"Retry-After": "5"}),
            5,
            "AniList returned HTTP 503 - waiting 5s (retry 1/3)",
            id="503",
        ),
    ],
)
def test_a_retry_waits_out_retry_after_kept_between_a_second_and_a_minute(
    slept: list[float], first: httpx.Response, waited: int, line: str
) -> None:
    """The user is told why and how long, so a long wait doesn't look like a hang."""

    recording = install_recording_hub()
    respx.post(API_URL).mock(side_effect=[first, httpx.Response(200, json=_FOUND)])

    _client().query(1)

    assert slept == [waited]
    assert [(d.severity, d.message) for d in recording.of_type(Diagnostic)] == [(Severity.INFO, line)]


@respx.mock
def test_without_retry_after_each_wait_doubles(slept: list[float]) -> None:
    """Three server errors in a row are waited out, and the answer to the last retry comes back."""

    bad_gateway = httpx.Response(502, text="<html>Bad Gateway</html>")
    respx.post(API_URL).mock(side_effect=[bad_gateway, bad_gateway, bad_gateway, httpx.Response(200, json=_FOUND)])

    assert _client().query(1) == _FOUND
    # Each wait has up to a second of jitter on top.
    assert [int(wait) for wait in slept] == [1, 2, 4]


@respx.mock
def test_a_network_error_that_clears_is_waited_out_quietly(slept: list[float]) -> None:
    """A dropped connection is asked again after a short wait that only the debug log mentions."""

    recording = install_recording_hub()
    respx.post(API_URL).mock(side_effect=[httpx.ConnectError("connection refused"), httpx.Response(200, json=_FOUND)])

    assert _client().query(1) == _FOUND
    assert [int(wait) for wait in slept] == [1]
    assert [d.severity for d in recording.of_type(Diagnostic)] == [Severity.DEBUG]


@respx.mock
@pytest.mark.usefixtures("slept")
@pytest.mark.parametrize(
    ("effect", "returned"),
    [
        pytest.param(httpx.Response(429, json={"data": None}), {"data": None}, id="throttled"),
        pytest.param(httpx.ConnectError("down"), {}, id="unreachable"),
    ],
)
def test_give_up_marks_anilist_unreachable(effect: httpx.Response | Exception, returned: dict[str, object]) -> None:
    """Exhausting the retries warns ONCE and marks AniList unreachable: later calls never touch the wire."""

    recording = install_recording_hub()
    client = _client()
    route = respx.post(API_URL).mock(side_effect=[effect] * (MAX_RETRIES + 1))

    assert client.query(1) == returned
    assert client.unreachable is True
    assert client.query(2) == {}
    assert client.query_batch([3, 4]) == {}

    assert route.call_count == MAX_RETRIES + 1
    (warning,) = diagnostic_messages(recording, Severity.WARNING)
    assert warning == f"AniList request failed after {MAX_RETRIES} retries, {_SKIPPING}"


@respx.mock
@pytest.mark.parametrize(
    ("response", "returned", "detail"),
    [
        pytest.param(
            httpx.Response(403, json=_DISABLED_BODY),
            _DISABLED_BODY,
            f"rejected the request (HTTP 403: {_DISABLED_MESSAGE})",
            id="403 refusal",
        ),
        pytest.param(
            httpx.Response(200, json=_DISABLED_BODY),
            _DISABLED_BODY,
            f"rejected the request (HTTP 200: {_DISABLED_MESSAGE})",
            id="refusal riding 200",
        ),
        pytest.param(
            httpx.Response(403, text="<html>forbidden</html>"), {}, "gave no usable answer (HTTP 403)", id="html 403"
        ),
        pytest.param(
            httpx.Response(200, json={"data": None}),
            {"data": None},
            "gave no usable answer (HTTP 200)",
            id="dataless 200",
        ),
        # AniList sends neither of these: a body that isn't an object, and junk ahead of the real error.
        pytest.param(httpx.Response(200, json=[1, 2, 3]), {}, "gave no usable answer (HTTP 200)", id="json array"),
        pytest.param(
            httpx.Response(403, json=_JUNK_THEN_DISABLED),
            _JUNK_THEN_DISABLED,
            f"rejected the request (HTTP 403: {_DISABLED_MESSAGE})",
            id="junk before the error",
        ),
    ],
)
def test_dataless_answer_marks_anilist_unreachable_once(
    response: httpx.Response, returned: dict[str, object], detail: str
) -> None:
    """Any dataless answer marks AniList unreachable on the first request: one warning, the body still returned, then no traffic."""

    recording = install_recording_hub()
    client = _client()
    route = respx.post(API_URL).mock(return_value=response)

    body = client.query(1)
    assert client.unreachable is True
    assert client.query(2) == {}
    assert client.query_batch([3]) == {}

    assert body == returned
    assert route.call_count == 1
    (warning,) = diagnostic_messages(recording, Severity.WARNING)
    assert warning == f"AniList {detail}, {_SKIPPING}"


@respx.mock
def test_refusal_message_is_collapsed_and_clamped() -> None:
    """The quoted message is whitespace-collapsed and clamped, so the warning stays one readable line."""

    recording = install_recording_hub()
    ragged = "down\n  for\t maintenance " + "x" * 200
    body: dict[str, object] = {"errors": [{"message": ragged, "status": 403}], "data": None}
    respx.post(API_URL).respond(status_code=403, json=body)

    _client().query(1)

    (warning,) = diagnostic_messages(recording, Severity.WARNING)
    quoted = warning.removeprefix("AniList rejected the request (HTTP 403: ").removesuffix(f"), {_SKIPPING}")
    assert quoted.startswith("down for maintenance xxx")
    assert len(quoted) == 160


# --- AniListMediaNode -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="null"),
        pytest.param({"year": None, "month": None, "day": None}, id="null parts"),
        pytest.param({"year": 2020, "month": 8, "day": None}, id="no day"),
        pytest.param({"year": 2020, "month": 2, "day": 30}, id="no such day"),
    ],
)
def test_an_anilist_date_without_every_part_is_no_date(raw: dict[str, int | None] | None) -> None:
    assert AniListMediaNode.model_validate({"id": 1, "startDate": raw}) == AniListMediaNode(id=1)
