# pyright: strict
"""The TheXEM wire client over `respx`: what a reply reads as, and the unreachable flag."""

from typing import cast

import httpx
import pytest
import respx
from respx.models import Call

from pearlarr.output import Diagnostic, Severity
from pearlarr.scene_numbers import SceneMapping
from pearlarr.xem_client import XEM_URL, XemClient, XemNoReply

from .fakes import FakeXemClient, RecordingHub, install_recording_hub
from .http_mock import xem_fixture

_SERIES = 79214
_MAPPINGS = (
    SceneMapping(0, 1, 2, 1, 1),
    SceneMapping(0, 2, 0, 2, 0),
    SceneMapping(0, 3, 3, 11, 11),
    SceneMapping(0, 4, 3, 11, 11),
    SceneMapping(2, 20, 3, 22, 22),
)
"""What `tvdb_79214.json` maps."""

_NO_CONNECTION: dict[str, object] = {"result": "failure", "data": [], "message": "no single connection"}
"""TheXEM's reply for a series it maps to no other site."""

_UNLISTED: dict[str, object] = {
    "result": "failure",
    "data": [],
    "message": "no show with the tvdb_id 70350 found",
}
"""TheXEM's reply for a series it doesn't list."""


def _client() -> XemClient:
    """A real client over respx, backoffs stubbed out."""

    return XemClient(client=httpx.Client(), sleep=lambda _s: None)


def _warnings(recording: RecordingHub) -> list[str]:
    return [d.message for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]


def _read(tvdb_id: int, body: object) -> tuple[SceneMapping, ...] | XemNoReply:
    """One read answered with `body`."""

    with respx.mock:
        respx.get(XEM_URL).respond(json=body)
        return _client().mappings(tvdb_id)


def _success(*rows: object) -> dict[str, object]:
    """A reply with `rows`, worded as TheXEM words one that has mappings."""

    return {
        "result": "success",
        "data": list(rows),
        "message": f"full mapping for {_SERIES} on tvdb. this was a cached version",
    }


def _unusable(detail: str) -> str:
    return f"TheXEM's answer for TVDB {_SERIES} can't be used ({detail}) - keeping what is stored for the series"


def test_a_reply_reads_as_its_mappings_in_the_order_listed() -> None:
    assert _read(_SERIES, xem_fixture("tvdb_79214.json")) == _MAPPINGS


def test_a_request_carries_the_tvdb_id_and_the_origin_and_nothing_else() -> None:
    with respx.mock:
        route = respx.get(XEM_URL).respond(json=_NO_CONNECTION)
        _client().mappings(_SERIES)

        (call,) = cast("list[Call]", route.calls)

    assert str(call.request.url) == f"https://thexem.info/map/all?id={_SERIES}&origin=tvdb"


def test_a_row_with_no_scene_side_is_dropped() -> None:
    recording = install_recording_hub()

    # Every row of the reply maps its episode to another site only.
    assert _read(80900, xem_fixture("tvdb_80900.json")) == ()
    assert _warnings(recording) == []


@pytest.mark.parametrize(
    "body", [pytest.param(_NO_CONNECTION, id="mapped-to-no-site"), pytest.param(_UNLISTED, id="not-listed")]
)
def test_a_failure_thexem_has_for_a_series_without_mappings_reads_as_no_mappings(body: dict[str, object]) -> None:
    recording = install_recording_hub()

    assert _read(70350, body) == ()
    assert _warnings(recording) == []


@pytest.mark.parametrize(
    ("message", "said"),
    [
        pytest.param("try again later", "failure: try again later", id="short"),
        pytest.param(
            "upstream\n  timed out " * 20, ("failure: " + " ".join(["upstream timed out"] * 20))[:160], id="long"
        ),
    ],
)
def test_a_failure_worded_another_way_leaves_the_series_unread_and_is_quoted_on_one_clamped_line(
    message: str, said: str
) -> None:
    recording = install_recording_hub()

    reply = _read(_SERIES, {"result": "failure", "data": [], "message": message})

    assert reply is XemNoReply.UNUSABLE
    assert _warnings(recording) == [_unusable(said)]


@pytest.mark.parametrize(
    ("scene", "tvdb"),
    [
        pytest.param({"season": 3, "episode": 22}, {"season": 2, "episode": 20, "absolute": 47}, id="number-missing"),
        pytest.param(
            {"season": 3, "episode": 22.5, "absolute": 22},
            {"season": 2, "episode": 20, "absolute": 47},
            id="not-a-whole-number",
        ),
        pytest.param(
            {"season": 3, "episode": 22, "absolute": 22},
            {"season": 2, "episode": None, "absolute": 47},
            id="tvdb-number-null",
        ),
    ],
)
def test_one_row_with_a_number_that_does_not_read_leaves_the_series_unread(
    scene: dict[str, object], tvdb: dict[str, object]
) -> None:
    recording = install_recording_hub()
    good = {"scene": {"season": 2, "episode": 1, "absolute": 1}, "tvdb": {"season": 0, "episode": 1, "absolute": 0}}

    reply = _read(_SERIES, _success(good, {"scene": scene, "tvdb": tvdb}))

    assert reply is XemNoReply.UNUSABLE
    assert _warnings(recording) == [_unusable("its mappings failed validation")]


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        pytest.param({"error": "not found"}, "no result: no message", id="another-object"),
        pytest.param(["maintenance"], "<record>: model_type", id="not-an-object"),
    ],
)
def test_json_that_is_not_a_mappings_reply_leaves_the_series_unread(body: object, detail: str) -> None:
    recording = install_recording_hub()

    reply = _read(_SERIES, body)

    assert reply is XemNoReply.UNUSABLE
    assert _warnings(recording) == [_unusable(detail)]


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        pytest.param(httpx.Response(503), "HTTP 503", id="5xx"),
        pytest.param(httpx.Response(429), "HTTP 429", id="rate-limited"),
        # What TheXEM answers a client name it doesn't take.
        pytest.param(httpx.Response(403, text="<html>Forbidden</html>"), "HTTP 403", id="refused"),
        pytest.param(httpx.Response(404), "HTTP 404", id="moved-route"),
        # Only a 200 is read, whatever the body.
        pytest.param(httpx.Response(500, json=_NO_CONNECTION), "HTTP 500", id="no-mappings-body-on-a-5xx"),
        # A page served in TheXEM's place says nothing of the series asked for.
        pytest.param(httpx.Response(200, text="<html>Maintenance</html>"), "a reply that isn't JSON", id="a-page"),
        pytest.param(httpx.ConnectTimeout("timed out"), "ConnectTimeout", id="timeout"),
    ],
)
def test_thexem_stays_unreachable_and_later_reads_send_nothing(answer: httpx.Response | Exception, detail: str) -> None:
    recording = install_recording_hub()
    client = _client()

    with respx.mock:
        route = respx.get(XEM_URL)
        if isinstance(answer, httpx.Response):
            route.respond(answer.status_code, content=answer.content)
        else:
            route.side_effect = answer
        first = client.mappings(_SERIES)
        sent = route.call_count
        second = client.mappings(80900)

        assert route.call_count == sent

    assert first is second is XemNoReply.UNREACHABLE
    assert client.unreachable is True
    assert _warnings(recording) == [
        f"TheXEM failed ({detail} for TVDB {_SERIES}) - keeping the scene numbers already stored until it answers"
    ]


def test_an_answer_that_cannot_be_used_is_for_that_series_only() -> None:
    install_recording_hub()
    client = _client()

    with respx.mock:
        respx.get(XEM_URL, params={"id": "5"}).respond(json={"result": "failure", "data": [], "message": "busy"})
        respx.get(XEM_URL, params={"id": str(_SERIES)}).respond(json=xem_fixture("tvdb_79214.json"))
        unread = client.mappings(5)
        served = client.mappings(_SERIES)

    assert unread is XemNoReply.UNUSABLE
    assert served == _MAPPINGS
    assert client.unreachable is False


def test_the_fake_answers_one_read_sequence_as_the_real_client_does() -> None:
    install_recording_hub()
    # Served, no mappings, unusable, unreachable, then a series that would be served if the unreachable flag let
    # it through.
    reads = [_SERIES, 70350, 5, 6, _SERIES]
    fake = FakeXemClient({_SERIES: _MAPPINGS}, unusable=frozenset({5}), failing=frozenset({6}))
    real = _client()

    with respx.mock:
        respx.get(XEM_URL, params={"id": str(_SERIES)}).respond(json=xem_fixture("tvdb_79214.json"))
        respx.get(XEM_URL, params={"id": "70350"}).respond(json=_UNLISTED)
        respx.get(XEM_URL, params={"id": "5"}).respond(json={"result": "failure", "data": [], "message": "busy"})
        respx.get(XEM_URL, params={"id": "6"}).respond(503)
        from_real = [(real.mappings(tvdb_id), real.unreachable) for tvdb_id in reads]

    assert [(fake.mappings(tvdb_id), fake.unreachable) for tvdb_id in reads] == from_real
