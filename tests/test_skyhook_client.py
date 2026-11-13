# pyright: strict
"""The SkyHook wire client over `respx`: reply validation, not found, and the unreachable flag."""

import httpx
import pytest
import respx

from pearlarr.output import Diagnostic, Severity
from pearlarr.skyhook_client import SKYHOOK_URL, SkyHookClient, SkyHookNoReply, SkyHookReply

from .fakes import FakeSkyHookClient, RecordingHub, install_recording_hub
from .http_mock import skyhook_fixture

_SERIES = 112961
_NOT_FOUND = {
    "message": "NotFound",
    "description": "thetvdb.com does not have a series with the tvdbid 1 and the language eng",
}
"""SkyHook's 404 body for a series TVDB doesn't have."""


def _client() -> SkyHookClient:
    """A real client over respx, backoffs stubbed out."""

    return SkyHookClient(client=httpx.Client(), sleep=lambda _s: None)


def _warnings(recording: RecordingHub) -> list[str]:
    return [d.message for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]


def _reply(tvdb_id: int, order_type: str, body: object) -> SkyHookReply | SkyHookNoReply:
    """One fetch answered with `body`."""

    with respx.mock:
        respx.get(f"{SKYHOOK_URL}/{tvdb_id}/{order_type}").respond(json=body)
        return _client().fetch(tvdb_id, order_type)


def test_a_reply_reads_with_its_rows_and_absolutes() -> None:
    reply = _reply(_SERIES, "official", skyhook_fixture("tvdb_112961_official.json"))

    assert isinstance(reply, SkyHookReply)
    assert reply.status == "Ended"
    assert [t.type for t in reply.season_types] == ["official", "dvd"]
    assert [(e.season_number, e.episode_number, e.absolute_episode_number) for e in reply.episodes] == [
        (0, 1, None),
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),
        (1, 4, 4),
    ]


@pytest.mark.parametrize(
    ("asked", "order_type", "warning"),
    [
        pytest.param(
            299502,
            "dvd",
            f"SkyHook answered TVDB 299502 with TVDB {_SERIES}'s dvd order - storing it empty",
            id="another-series",
        ),
        pytest.param(
            _SERIES,
            "alternate",
            f"SkyHook sent a malformed alternate order for TVDB {_SERIES} (the reply doesn't list that order) - "
            "storing it empty",
            id="unlisted-order",
        ),
    ],
)
def test_a_reply_that_does_not_match_the_request_comes_back_empty(asked: int, order_type: str, warning: str) -> None:
    recording = install_recording_hub()
    # The dvd reply of 112961, served for a request it doesn't answer.
    body = skyhook_fixture("tvdb_112961_dvd.json")

    reply = _reply(asked, order_type, body)

    assert isinstance(reply, SkyHookReply)
    assert reply.episodes == ()
    assert _warnings(recording) == [warning]


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
        pytest.param(httpx.Response(429), "HTTP 429", id="rate-limited"),
        # SkyHook sits behind Cloudflare, whose edge block is a host-wide 403 page.
        pytest.param(httpx.Response(403, text="<html>Attention Required</html>"), "HTTP 403", id="edge-block"),
        # A route SkyHook doesn't serve 404s with no body, unlike its own not-found.
        pytest.param(httpx.Response(404), "HTTP 404", id="moved-route"),
        # Only a 404 says the series is missing, whatever the body.
        pytest.param(httpx.Response(500, json=_NOT_FOUND), "HTTP 500", id="not-found-body-on-a-5xx"),
        pytest.param(httpx.ConnectTimeout("timed out"), "ConnectTimeout", id="timeout"),
    ],
)
def test_an_outage_latches_and_later_fetches_send_nothing(answer: httpx.Response | Exception, detail: str) -> None:
    recording = install_recording_hub()
    client = _client()

    with respx.mock:
        route = respx.get(url__startswith=SKYHOOK_URL)
        if isinstance(answer, httpx.Response):
            route.respond(answer.status_code, content=answer.content)
        else:
            route.side_effect = answer
        first = client.fetch(_SERIES, "official")
        sent = route.call_count
        second = client.fetch(71776, "official")

        assert route.call_count == sent

    assert first is second is SkyHookNoReply.UNREACHABLE
    assert client.unreachable is True
    assert _warnings(recording) == [
        f"SkyHook failed ({detail} for TVDB {_SERIES} official) - re-checking cached titles until it answers"
    ]


def test_skyhooks_own_not_found_is_a_miss_for_that_series_only() -> None:
    client = _client()

    with respx.mock:
        respx.get(f"{SKYHOOK_URL}/1/official").respond(404, json=_NOT_FOUND)
        respx.get(f"{SKYHOOK_URL}/{_SERIES}/official").respond(json=skyhook_fixture("tvdb_112961_official.json"))
        missed = client.fetch(1, "official")
        served = client.fetch(_SERIES, "official")

    assert missed is SkyHookNoReply.NOT_FOUND
    assert isinstance(served, SkyHookReply)
    assert client.unreachable is False


def test_the_fake_answers_one_fetch_sequence_as_the_real_client_does() -> None:
    install_recording_hub()
    official = skyhook_fixture("tvdb_112961_official.json")
    # Served, not found, unreachable, then a pair that would be served if the unreachable flag let it through.
    fetches = [(_SERIES, "official"), (1, "official"), (71776, "official"), (_SERIES, "official")]
    fake = FakeSkyHookClient(
        {(_SERIES, "official"): SkyHookReply.model_validate(official)}, failing=frozenset({(71776, "official")})
    )
    real = _client()

    with respx.mock:
        respx.get(f"{SKYHOOK_URL}/{_SERIES}/official").respond(json=official)
        respx.get(f"{SKYHOOK_URL}/1/official").respond(404, json=_NOT_FOUND)
        respx.get(f"{SKYHOOK_URL}/71776/official").respond(503)
        from_real = [(real.fetch(*pair), real.unreachable) for pair in fetches]

    assert [(fake.fetch(*pair), fake.unreachable) for pair in fetches] == from_real
