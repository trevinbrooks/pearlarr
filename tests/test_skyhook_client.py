# pyright: strict
"""The SkyHook wire client over `respx`: reply validation, the 4xx miss, and the outage latch."""

import httpx
import pytest
import respx

from pearlarr.output import Diagnostic, Severity
from pearlarr.output.recording import RecordingHub
from pearlarr.seadex_types import SkyHookReply
from pearlarr.skyhook_client import SKYHOOK_URL, SkyHookClient, SkyHookMiss

from .fakes import install_recording_hub
from .http_mock import skyhook_fixture

_SERIES = 112961


def _client() -> SkyHookClient:
    """A real client over respx, backoffs stubbed out."""

    return SkyHookClient(client=httpx.Client(), sleep=lambda _s: None)


def _warnings(recording: RecordingHub) -> list[str]:
    return [d.message for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]


def _reply(tvdb_id: int, order_type: str, body: object) -> SkyHookReply | SkyHookMiss:
    """One fetch answered with `body`."""

    with respx.mock:
        respx.get(f"{SKYHOOK_URL}/{tvdb_id}/{order_type}").respond(json=body)
        return _client().fetch(tvdb_id, order_type)


def test_a_reply_reads_with_its_rows_and_absolutes() -> None:
    reply = _reply(_SERIES, "official", skyhook_fixture("tvdb_112961_official.json"))

    assert isinstance(reply, SkyHookReply)
    assert reply.status == "Ended"
    assert [t.type for t in reply.season_types] == ["official", "dvd"]
    assert reply.episodes is not None
    assert [(e.season_number, e.episode_number, e.absolute_episode_number) for e in reply.episodes] == [
        (0, 1, None),
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),
        (1, 4, 4),
    ]


@pytest.mark.parametrize(
    ("asked", "order_type"),
    [
        pytest.param(299502, "dvd", id="another-series"),
        pytest.param(_SERIES, "alternate", id="unlisted-order"),
    ],
)
def test_a_reply_that_does_not_match_the_request_comes_back_empty(asked: int, order_type: str) -> None:
    recording = install_recording_hub()
    # The dvd reply of 112961, served for a request it doesn't answer.
    body = skyhook_fixture("tvdb_112961_dvd.json")

    reply = _reply(asked, order_type, body)

    assert isinstance(reply, SkyHookReply)
    assert reply.episodes == ()
    detail = "the reply is for another series or order"
    assert _warnings(recording) == [
        f"SkyHook sent a malformed {order_type} order for TVDB {asked} ({detail}) - storing it empty"
    ]


def test_a_reply_that_fails_validation_comes_back_empty_with_a_warning() -> None:
    recording = install_recording_hub()

    reply = _reply(_SERIES, "official", {"status": "Ended", "episodes": []})

    assert isinstance(reply, SkyHookReply)
    assert (reply.tvdb_id, reply.episodes, reply.season_types) == (_SERIES, (), ())
    (warning,) = _warnings(recording)
    assert warning.startswith(f"SkyHook sent a malformed official order for TVDB {_SERIES} (")


def test_a_listed_order_with_no_rows_comes_back_empty_without_a_warning() -> None:
    recording = install_recording_hub()

    reply = _reply(71776, "dvd", skyhook_fixture("tvdb_71776_dvd.json"))

    assert isinstance(reply, SkyHookReply)
    assert reply.episodes == ()
    assert reply.lists("dvd")
    assert _warnings(recording) == []


def test_one_bad_row_empties_the_order_and_keeps_the_header() -> None:
    recording = install_recording_hub()
    body = {
        "tvdbId": _SERIES,
        "status": "Ended",
        "seasonTypes": [{"type": "official", "episodeCount": 4}, {"type": "dvd", "episodeCount": 4}],
        "episodes": [
            {"tvdbId": 2273831, "seasonNumber": 1, "episodeNumber": 1},
            {"seasonNumber": 1, "episodeNumber": 1},
        ],
    }

    reply = _reply(_SERIES, "dvd", body)

    assert isinstance(reply, SkyHookReply)
    assert reply.episodes == ()
    assert reply.status == "Ended"
    assert [t.type for t in reply.season_types] == ["official", "dvd"]
    assert _warnings(recording) == [
        f"SkyHook sent a malformed dvd order for TVDB {_SERIES} (an episode row failed validation) - storing it empty"
    ]


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        pytest.param(httpx.Response(503), "HTTP 503", id="5xx"),
        pytest.param(httpx.ConnectTimeout("timed out"), "ConnectTimeout", id="timeout"),
    ],
)
def test_an_outage_latches_and_later_fetches_send_nothing(answer: httpx.Response | Exception, detail: str) -> None:
    recording = install_recording_hub()
    client = _client()

    with respx.mock:
        route = respx.get(url__startswith=SKYHOOK_URL)
        if isinstance(answer, httpx.Response):
            route.respond(answer.status_code)
        else:
            route.side_effect = answer
        first = client.fetch(_SERIES, "official")
        sent = route.call_count
        second = client.fetch(71776, "official")

        assert route.call_count == sent

    assert first is second is SkyHookMiss.OUTAGE
    assert client.outage is True
    assert _warnings(recording) == [f"SkyHook request failed ({detail}) - skipping TVDB order fetches this run"]


def test_a_client_error_is_a_miss_for_that_series_only() -> None:
    client = _client()

    with respx.mock:
        respx.get(f"{SKYHOOK_URL}/1/official").respond(404)
        respx.get(f"{SKYHOOK_URL}/{_SERIES}/official").respond(json=skyhook_fixture("tvdb_112961_official.json"))
        missed = client.fetch(1, "official")
        served = client.fetch(_SERIES, "official")

    assert missed is SkyHookMiss.UNKNOWN
    assert isinstance(served, SkyHookReply)
    assert client.outage is False
