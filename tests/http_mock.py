# pyright: strict
"""Reusable ``responses`` scaffolding for the Sonarr HTTP boundary.

Shared by the end-to-end smoke (``test_e2e_smoke``) and the adapter tests (T1.3),
so the per-endpoint registration lives in one importable place instead of being
duplicated inline. The bodies come from the captured fixtures in
``tests/fixtures/sonarr/`` (real Sonarr responses), so a test only names the
endpoints it exercises. Bodies are typed ``object`` (opaque JSON handed straight
to ``responses``' ``json=``) - a test that needs to read into one casts at its
own call site.
"""

import json
from pathlib import Path

import responses

_SONARR_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"


def sonarr_fixture(name: str) -> object:
    """Decode a captured Sonarr fixture body by file name."""

    return json.loads((_SONARR_FIXTURES / name).read_text())


def register_sonarr_reads(
    rsps: responses.RequestsMock,
    base: str,
    *,
    episodes: object,
    parse: object,
    quality_definitions: list[object] | None = None,
    languages: list[object] | None = None,
) -> None:
    """Register the requests-based read endpoints a Sonarr sync hits on the
    OFF-mode preview path.

    ``base`` is the ``http://host/api/v3`` prefix. ``episode`` drives the
    per-series fetch; ``parse`` (matched on the base path, query ignored)
    replays a captured parse for every SeaDex filename.
    ``qualitydefinition``/``language`` default to empty (only touched when an
    import payload is built); ``history/since`` replays an empty window so the
    activity scan stays quiet. The library fetch (``/series``) rides the
    httpx-based ``ArrHttp``, so a test registers it through respx instead.
    """

    rsps.add(responses.GET, f"{base}/episode", json=episodes)
    rsps.add(responses.GET, f"{base}/parse", json=parse)
    rsps.add(responses.GET, f"{base}/qualitydefinition", json=quality_definitions or [])
    rsps.add(responses.GET, f"{base}/language", json=languages or [])
    rsps.add(responses.GET, f"{base}/history/since", json=[])
