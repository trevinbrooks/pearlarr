# pyright: strict
"""Parity net for the Tier-2.75 boundary-type port (2.75a).

Pins today's read -> resolve -> re-emit behaviour AT THE WIRE so the pydantic
port cannot change it: golden ManualImport POST bodies across
``resolve_quality``'s branches (verbatim unknown-key passthrough, the
dict-falsy empty-quality fallbacks), per-field fail-open folding on
queue/history records, and the candidate rejections folding. Every test drives
a REAL client over ``respx`` (no network) - the seams that survive the port
unchanged.
"""

import json
import logging

import httpx
import respx

from seadexarr.modules.arr_http import ArrHttp
from seadexarr.modules.manual_import import ImportReadiness
from seadexarr.modules.radarr_client import RadarrClient, make_radarr_client
from seadexarr.modules.seadex_types import HistoryRecord, ImportRejection, QueueRecord
from seadexarr.modules.sonarr_client import SonarrClient
from seadexarr.modules.sonarr_import import ImportExecutor
from seadexarr.modules.sonarr_import_plan import EpisodeSnapshot, QueueVerdict, classify_queue
from seadexarr.modules.sonarr_mapper import FileEpisodeMapper

from .builders import make_run_deps, pending_import

_URL = "http://sonarr.test"
_BASE = f"{_URL}/api/v3"
_KEY = "testkey"


def _make_sonarr_client() -> SonarrClient:
    """A real ``SonarrClient`` over respx (backoffs stubbed out)."""

    return SonarrClient(
        http=ArrHttp.bind(
            client=httpx.Client(),
            url=_URL,
            api_key=_KEY,
            label="Sonarr",
            sleep=lambda _s: None,
        ),
        logger=logging.getLogger("seadexarr.test.boundary-parity"),
    )


def _make_radarr() -> RadarrClient:
    """A real ``RadarrClient`` over respx."""

    return make_radarr_client(
        url="http://radarr.test",
        api_key=_KEY,
        http=httpx.Client(),
    )


# --- golden ManualImport POST bodies (resolve_quality branches) ---------------
#
# Each test drives ``ImportExecutor.run_manual_import`` end-to-end - raw
# candidate JSON in, exact ``/api/v3/command`` body out - so the golden pins the
# full read -> resolve -> re-emit round-trip at the wire, not the in-memory
# shapes the port replaces.

_CANDIDATE_PATH = "/d/Show - 01 [1080p].mkv"


def _drive_manual_import(
    *,
    candidate: dict[str, object],
    quality_defs: list[object],
    languages: list[object],
) -> object:
    """Run one manual import over respx and return the decoded POSTed body."""

    respx.get(f"{_BASE}/manualimport").respond(json=[candidate])
    respx.get(f"{_BASE}/qualitydefinition").respond(json=quality_defs)
    respx.get(f"{_BASE}/language").respond(json=languages)
    post_route = respx.post(f"{_BASE}/command").respond(json={"id": 55})

    client = _make_sonarr_client()
    executor = ImportExecutor(make_run_deps(), client, FileEpisodeMapper(client))
    probe = executor.run_manual_import(
        pending_import(),
        "/d",
        snapshot=EpisodeSnapshot(episodes_by_id={}, recommended_groups=set()),
    )

    assert probe.readiness is ImportReadiness.RETRY
    assert probe.command_issued is True
    body: object = json.loads(post_route.calls.last.request.content)
    return body


def _golden_body(*, quality: dict[str, object], languages: list[object]) -> dict[str, object]:
    """The exact expected ``/api/v3/command`` body for the one-file import."""

    return {
        "name": "ManualImport",
        "importMode": "auto",
        "files": [
            {
                "path": _CANDIDATE_PATH,
                "seriesId": 7,
                "episodeIds": [101],
                "releaseGroup": "SubGroup",
                "downloadId": "abc123",
                "languages": languages,
                "quality": quality,
            },
        ],
    }


# The synthesized last-resort quality: never an omitted key (an omitted quality
# is exactly what crashed Sonarr's FileNameBuilder.AddQualityTokens).
_UNKNOWN_QUALITY: dict[str, object] = {
    "quality": {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0},
    "revision": {"version": 1, "real": 0, "isRepack": False},
}


@respx.mock
def test_golden_body_matched_definition_reemits_definition_quality() -> None:
    """A resolved (source, resolution) re-emits the matched definition's nested
    quality VERBATIM (unknown keys included) plus the default revision; a
    language definition with no id resolves to an explicit ``"id": null``.
    """

    definition_quality: dict[str, object] = {
        "id": 3,
        "name": "WEBDL-1080p",
        "source": "web",
        "resolution": 1080,
        "modifier": "none",
    }
    body = _drive_manual_import(
        candidate={
            "path": _CANDIDATE_PATH,
            "quality": {"quality": {"source": "web", "resolution": 1080}},
            "rejections": [],
        },
        quality_defs=[{"quality": definition_quality, "title": "WEBDL-1080p", "weight": 19}],
        languages=[{"name": "Japanese"}],
    )

    assert body == _golden_body(
        quality={
            "quality": definition_quality,
            "revision": {"version": 1, "real": 0, "isRepack": False},
        },
        languages=[{"id": None, "name": "Japanese"}],
    )


@respx.mock
def test_golden_body_no_match_reemits_candidate_verbatim_with_unknown_keys() -> None:
    """With no matching definition, Sonarr's own candidate model is re-emitted
    verbatim - unknown keys survive at BOTH nesting levels (inside ``quality``
    and inside ``quality.quality``).
    """

    candidate_quality: dict[str, object] = {
        "quality": {
            "id": 7,
            "name": "Bluray-1080p",
            "source": "bluray",
            "resolution": 1080,
            "customInner": True,
        },
        "revision": {"version": 2, "real": 0, "isRepack": True},
        "customOuter": [1, 2],
    }
    body = _drive_manual_import(
        candidate={"path": _CANDIDATE_PATH, "quality": candidate_quality, "rejections": []},
        quality_defs=[],
        languages=[{"id": 8, "name": "Japanese"}],
    )

    assert body == _golden_body(
        quality=candidate_quality,
        languages=[{"id": 8, "name": "Japanese"}],
    )


@respx.mock
def test_golden_body_quality_absent_synthesizes_unknown() -> None:
    """A candidate carrying NO quality at all falls back to the explicit
    synthesized Unknown (never an omitted quality key).
    """

    body = _drive_manual_import(
        candidate={"path": _CANDIDATE_PATH, "rejections": []},
        quality_defs=[],
        languages=[{"id": 8, "name": "Japanese"}],
    )

    assert body == _golden_body(quality=_UNKNOWN_QUALITY, languages=[{"id": 8, "name": "Japanese"}])


@respx.mock
def test_golden_body_empty_quality_object_synthesizes_unknown() -> None:
    """``"quality": {}`` on the candidate is falsy TODAY and must keep
    synthesizing Unknown after the port (the dict-falsy -> model-truthy trap at
    the candidate level).
    """

    body = _drive_manual_import(
        candidate={"path": _CANDIDATE_PATH, "quality": {}, "rejections": []},
        quality_defs=[],
        languages=[{"id": 8, "name": "Japanese"}],
    )

    assert body == _golden_body(quality=_UNKNOWN_QUALITY, languages=[{"id": 8, "name": "Japanese"}])


@respx.mock
def test_golden_body_empty_inner_quality_synthesizes_unknown() -> None:
    """``"quality": {"quality": {}}`` - present model, empty INNER quality - is
    falsy today in ``resolve_quality``'s fallback tail and must keep
    synthesizing Unknown after the port (the same trap one level down).
    """

    body = _drive_manual_import(
        candidate={"path": _CANDIDATE_PATH, "quality": {"quality": {}}, "rejections": []},
        quality_defs=[],
        languages=[{"id": 8, "name": "Japanese"}],
    )

    assert body == _golden_body(quality=_UNKNOWN_QUALITY, languages=[{"id": 8, "name": "Japanese"}])


# --- QueueRecord: per-field fail-open + the importPending invariant ----------


@respx.mock
def test_queue_record_folds_junk_per_field_without_dropping_the_record() -> None:
    """Junk in individual queue-record fields folds that FIELD to None; the
    record itself (and its healthy siblings) always survives.
    """

    respx.get(f"{_BASE}/queue").respond(
        json={
            "records": [
                {
                    "downloadId": 123,
                    "trackedDownloadState": ["importing"],
                    "trackedDownloadStatus": "warning",
                    "extraKey": "x",
                },
                {
                    "downloadId": "ABC123",
                    "trackedDownloadState": "downloading",
                    "trackedDownloadStatus": "ok",
                },
            ],
            "totalRecords": 2,
        },
    )

    assert _make_sonarr_client().queue() == [
        QueueRecord(download_id=None, state=None, status="warning"),
        QueueRecord(download_id="ABC123", state="downloading", status="ok"),
    ]


@respx.mock
def test_import_pending_always_classifies_pending_clean() -> None:
    """Live invariant: EVERY ``importPending`` record - even with a warning
    status - classifies as PENDING_CLEAN (wait), never STEP_IN (stepping in
    double-imports).
    """

    respx.get(f"{_BASE}/queue").respond(
        json={
            "records": [
                {
                    "downloadId": "ABC123",
                    "trackedDownloadState": "importPending",
                    "trackedDownloadStatus": "warning",
                },
            ],
            "totalRecords": 1,
        },
    )

    records = _make_sonarr_client().queue()
    states = [record.state for record in records if record.state]
    assert states == ["importPending"]
    assert classify_queue(states) is QueueVerdict.PENDING_CLEAN


# --- HistoryRecord: per-field folds, record never drops ----------------------


@respx.mock
def test_history_junk_item_id_folds_to_zero_and_the_record_is_kept() -> None:
    """A junk ``seriesId`` folds to 0 without dropping the record - a dropped
    record would be a missed dirty-mark and a lagging checkpoint.
    """

    respx.get(f"{_BASE}/history/since").respond(
        json=[
            {
                "id": 21,
                "seriesId": {"weird": True},
                "date": "2026-07-01T10:00:00Z",
                "eventType": "grabbed",
                "downloadId": "ABC123",
            },
        ],
    )

    assert _make_sonarr_client().history_since("2026-06-30T08:00:00Z") == [
        HistoryRecord(
            id=21,
            date="2026-07-01T10:00:00Z",
            item_id=0,
            event_type="grabbed",
            download_id="ABC123",
        ),
    ]


@respx.mock
def test_history_reason_key_resolves_case_insensitively() -> None:
    """The ``data`` reason key is found regardless of case (nets a naive
    fixed-alias port - an alias cannot do case-insensitivity).
    """

    respx.get(f"{_BASE}/history/since").respond(
        json=[
            {"id": 1, "seriesId": 4, "date": "d", "eventType": "grabbed", "data": {"REASON": "Upgrade"}},
            {"id": 2, "seriesId": 4, "date": "d", "eventType": "grabbed", "data": {"ReAsOn": "MissingFromDisk"}},
        ],
    )

    records = _make_sonarr_client().history_since("2026-06-30T08:00:00Z")
    assert records is not None
    assert [record.reason for record in records] == ["Upgrade", "MissingFromDisk"]


@respx.mock
def test_history_missing_item_id_defaults_to_zero() -> None:
    """A record with neither ``seriesId`` nor ``movieId`` keeps item_id 0 (the
    downstream ``item_id <= 0`` drop applies later, never at the parse).
    """

    respx.get(f"{_BASE}/history/since").respond(
        json=[{"id": 5, "date": "2026-07-01T10:00:00Z", "eventType": "grabbed"}],
    )

    assert _make_sonarr_client().history_since("2026-06-30T08:00:00Z") == [
        HistoryRecord(id=5, date="2026-07-01T10:00:00Z", item_id=0, event_type="grabbed"),
    ]


@respx.mock
def test_radarr_history_item_id_reads_movie_id_and_junk_folds() -> None:
    """Radarr's item id comes from ``movieId``; a null one folds to 0 with the
    record kept (same per-field posture as Sonarr's).
    """

    respx.get("http://radarr.test/api/v3/history/since").respond(
        json=[
            {"id": 3, "movieId": 9, "date": "d", "eventType": "grabbed"},
            {"id": 4, "movieId": None, "date": "d", "eventType": "grabbed"},
        ],
    )

    assert _make_radarr().history_since("2026-06-30T08:00:00Z") == [
        HistoryRecord(id=3, date="d", item_id=9, event_type="grabbed"),
        HistoryRecord(id=4, date="d", item_id=0, event_type="grabbed"),
    ]


# --- ManualImportCandidate: rejections folding --------------------------------


@respx.mock
def test_rejections_fold_strings_and_dicts_and_skip_other_shapes() -> None:
    """One rejections list mixing a bare string (older Sonarr), a proper
    ``{reason}`` object, a reason-less object, and non-str/non-dict junk: the
    first three fold to ``ImportRejection``s, the junk entries are skipped.
    """

    respx.get(f"{_BASE}/manualimport").respond(
        json=[
            {
                "path": "/d/x.mkv",
                "rejections": [
                    "Sample file",
                    {"reason": "Episode already imported"},
                    {"nope": 1},
                    42,
                    None,
                    ["nested"],
                ],
            },
        ],
    )

    candidates = _make_sonarr_client().manual_import_candidates(pending=pending_import())
    assert candidates is not None
    [candidate] = candidates
    assert candidate.path == "/d/x.mkv"
    assert candidate.quality is None
    assert candidate.rejections == (
        ImportRejection(reason="Sample file"),
        ImportRejection(reason="Episode already imported"),
        ImportRejection(reason=None),
    )


@respx.mock
def test_junk_path_drops_the_candidate_but_keeps_siblings() -> None:
    """A candidate whose ``path`` is a type lie (non-str) fails validation and
    is dropped whole (skip + warn); well-formed siblings survive. The old dict
    walk passed the lie through and crashed later at ``os.path.basename``.
    """

    respx.get(f"{_BASE}/manualimport").respond(
        json=[
            {"path": 42, "rejections": []},
            {"path": "/d/kept.mkv", "rejections": []},
        ],
    )

    candidates = _make_sonarr_client().manual_import_candidates(pending=pending_import())
    assert candidates is not None
    [candidate] = candidates
    assert candidate.path == "/d/kept.mkv"
