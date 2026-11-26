# pyright: strict
"""Reusable `respx` scaffolding for the Sonarr and Nyaa HTTP boundaries.

Shared by the end-to-end smoke (`test_e2e_smoke`), the adapter tests and the grab tests,
so the per-endpoint registration lives in one importable place instead of being
duplicated inline. The bodies come from the captured fixtures in
`tests/fixtures/sonarr/` (real Sonarr responses), so a test only names the
endpoints it exercises. Bodies are typed `object` (opaque JSON handed straight
to `respx`'s `json=`) - a test that needs to read into one casts at its
own call site.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from urllib.parse import quote

import httpx
import respx
from respx.models import Call, CallList

_SONARR_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"
_SKYHOOK_FIXTURES = Path(__file__).parent / "fixtures" / "skyhook"
_XEM_FIXTURES = Path(__file__).parent / "fixtures" / "xem"


def sonarr_fixture(name: str) -> object:
    """Decode a captured Sonarr fixture body by file name."""

    return json.loads((_SONARR_FIXTURES / name).read_text())


def skyhook_fixture(name: str) -> object:
    """Decode a captured SkyHook reply by file name, cut down to a few rows and keys."""

    return json.loads((_SKYHOOK_FIXTURES / name).read_text())


def xem_fixture(name: str) -> object:
    """Decode a captured TheXEM mappings reply by file name, cut down to a few rows."""

    return json.loads((_XEM_FIXTURES / name).read_text())


def requests_sent(calls: CallList) -> list[httpx.Request]:
    """The requests behind a respx route's or router's calls, oldest first."""

    # respx declares its call list as a bare `list`, so its items have no type until one is named here.
    return [call.request for call in cast("list[Call]", calls)]


def register_sonarr_reads(
    base: str,
    *,
    episodes: object,
    parse: object,
    quality_definitions: list[object] | None = None,
    languages: list[object] | None = None,
) -> None:
    """Register the read endpoints a Sonarr sync hits on the OFF-mode dry-run path.

    All registered on the ambient respx router (every raw arr endpoint goes through
    the httpx-based `ArrHttp`). `base` is the `http://host/api/v3`
    prefix. `episode` drives the per-series fetch. `parse` (matched on the
    base path, query ignored) replays a captured parse for every SeaDex
    filename. `qualitydefinition`/`language` default to empty (only touched
    when an import payload is built). `history/since` replays an empty window
    so the activity read stays quiet. The library fetch (`/series`) is
    registered by the test itself.
    """

    respx.get(f"{base}/episode").respond(json=episodes)
    respx.get(f"{base}/parse").respond(json=parse)
    respx.get(f"{base}/qualitydefinition").respond(json=quality_definitions or [])
    respx.get(f"{base}/language").respond(json=languages or [])
    respx.get(f"{base}/history/since").respond(json=[])


def nyaa_torrent_link(number: int) -> str:
    """The link Nyaa serves release `number`'s torrent file at, the one a grab hands qBittorrent."""

    return f"https://nyaa.si/download/{number}.torrent"


def nyaa_disposition(file_name: str) -> dict[str, str]:
    """The download header Nyaa sends: the file name twice, percent-encoded, the second as RFC 6266 `filename*`."""

    encoded = quote(file_name)
    return {"content-disposition": f"inline; filename=\"{encoded}\"; filename*=UTF-8''{encoded}"}


def serve_nyaa_torrents(*numbers: int) -> None:
    """Nyaa serving each numbered release's torrent file, named for its number, on the ambient respx router."""

    serve_named_nyaa_torrents({number: f"Release {number}" for number in numbers})


def serve_named_nyaa_torrents(names: Mapping[int, str]) -> None:
    """Nyaa serving each release's torrent file under its name, `names[number]`, on the ambient respx router."""

    for number, name in names.items():
        respx.head(nyaa_torrent_link(number)).respond(headers=nyaa_disposition(f"{name}.torrent"))


def nyaa_lacks_torrents(*numbers: int) -> None:
    """Nyaa answering 404 for each numbered release's torrent file, as it does for one taken down."""

    for number in numbers:
        respx.head(nyaa_torrent_link(number)).respond(404)
