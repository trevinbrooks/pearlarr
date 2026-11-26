# pyright: strict
"""Tests for `SonarrClient`, the Sonarr REST adapter.

Each test drives a real client over `respx` and checks what one method decides: the request it sends, what
it reads back, and what its caller gets, with which warning, when the read fails. How a failed request is
retried, coalesced and worded belongs to the shared handle and is shown once in `test_arr_http`. Bodies come
from the captured `tests/fixtures/sonarr` JSON where one exists, so they match what Sonarr really sends.
"""

import json
import logging
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
import respx
from respx.models import Call

from pearlarr.arr_http import GET_RETRIES, ArrHttp, DeleteOutcome
from pearlarr.arr_models import (
    CommandResource,
    DownloadClientConfig,
    Language,
    ManualImportFile,
    MatchedEpisode,
    ParsedFileInfo,
    Quality,
    QueueRecord,
)
from pearlarr.boundary import BoundaryContractError
from pearlarr.output import Severity
from pearlarr.pending_import import NO_TVDB_IDS, PendingImport
from pearlarr.sonarr_client import SonarrClient

from .builders import title_claim
from .fakes import diagnostic_messages, install_recording_hub
from .http_mock import sonarr_fixture

_URL = "http://sonarr.test"
_BASE = f"{_URL}/api/v3"


def _make_client() -> SonarrClient:
    """Build a real `SonarrClient` (construction is network-free).

    Every endpoint goes through the bound `ArrHttp` over the httpx client, mocked
    through `respx`. The bind's `sleep` is stubbed out so fail-open tests
    don't wait out real backoffs.
    """

    return SonarrClient(
        http=ArrHttp.bind(
            client=httpx.Client(),
            url=_URL,
            api_key="testkey",
            label="Sonarr",
            sleep=lambda _s: None,
        ),
        logger=logging.getLogger("pearlarr.test"),
    )


def _make_pending_import(*, infohash: str, title: str) -> PendingImport:
    """A minimal `PendingImport` carrying only the fields the ManualImport fetch reads."""

    return PendingImport(
        infohash=infohash,
        release_group="",
        is_dual_audio=False,
        seadex_files=(),
        added_at="",
        file_episode_map={},
        claims=(title_claim(title=title),),
        tvdb_ids=NO_TVDB_IDS,
    )


# --- all_series() -------------------------------------------------------------

# A minimal `/api/v3/series` resource: the consumed item fields plus a couple of
# extras proving unknown keys are ignored by `SonarrSeries`.
_SERIES_BODY: dict[str, object] = {
    "id": 228,
    "title": "Undefeated Bahamut Chronicle",
    "monitored": True,
    "tvdbId": 299502,
    "imdbId": "tt5311514",
    "sortTitle": "undefeated bahamut chronicle",
    "seasonFolder": True,
}


@respx.mock
def test_all_series_parses_into_sonarr_item_shape() -> None:
    """`all_series` reads each series into the `SonarrItem` fields a run keys on."""

    respx.get(f"{_BASE}/series").respond(json=[_SERIES_BODY])

    [show] = _make_client().all_series()
    assert show.id == 228
    assert show.title == "Undefeated Bahamut Chronicle"
    assert show.tvdbId == 299502
    assert show.imdbId == "tt5311514"
    assert show.monitored is True


@respx.mock
def test_all_series_all_invalid_payload_raises_contract_error() -> None:
    """A non-empty series payload validating to NOTHING raises the typed contract error (fail-closed).

    It never reads as an empty library.
    """

    respx.get(f"{_BASE}/series").respond(json=["junk", 42])
    with pytest.raises(BoundaryContractError):
        _make_client().all_series()


# --- queue() ----------------------------------------------------------------


@respx.mock
def test_queue_decodes_records_and_builds_request() -> None:
    """`queue()` pulls the whole queue in one paged request and narrows each record to a `QueueRecord` view."""

    route = respx.get(f"{_BASE}/queue").respond(json=sonarr_fixture("queue.json"))
    client = _make_client()
    records = client.queue()

    assert records is not None
    assert len(records) == 3
    assert records[0] == QueueRecord(
        id=1516314797,
        download_id="B7640FF13A2ADCA981B821D03CEBD1B569798459",
        state="downloading",
        status="ok",
    )
    assert dict(route.calls.last.request.url.params) == {
        "page": "1",
        "pageSize": "1000",
        "includeUnknownSeriesItems": "true",
    }


def _queue_page(total: int, hashes: list[str]) -> dict[str, object]:
    """One raw paged `/queue` body carrying `totalRecords` and the records."""

    return {
        "totalRecords": total,
        "records": [
            {"downloadId": h, "trackedDownloadState": "downloading", "trackedDownloadStatus": "ok"} for h in hashes
        ],
    }


@respx.mock
def test_queue_paginates_until_total_records_covered() -> None:
    """A queue larger than one page is fetched page by page until totalRecords is covered.

    It is never silently truncated at the first page.
    """

    route = respx.get(f"{_BASE}/queue")
    route.side_effect = [
        httpx.Response(200, json=_queue_page(3, ["HASH0", "HASH1"])),
        httpx.Response(200, json=_queue_page(3, ["HASH2"])),
    ]
    records = _make_client().queue()

    assert records is not None
    assert [r.download_id for r in records] == ["HASH0", "HASH1", "HASH2"]
    assert [call.request.url.params["page"] for call in cast("list[Call]", route.calls)] == ["1", "2"]


@respx.mock
def test_queue_stops_at_an_empty_page_whatever_the_total_says() -> None:
    """A download that leaves the queue between page reads makes the total stale.

    The page that comes back empty ends the read, where trusting the total would page forever.
    """

    route = respx.get(f"{_BASE}/queue")
    route.side_effect = [
        httpx.Response(200, json=_queue_page(3, ["HASH0", "HASH1"])),
        httpx.Response(200, json=_queue_page(3, [])),
    ]
    records = _make_client().queue()

    assert records is not None
    assert [r.download_id for r in records] == ["HASH0", "HASH1"]


@respx.mock
def test_queue_later_page_failure_fails_the_whole_read() -> None:
    """A failed LATER page returns None, never the partial pages.

    A partial queue misreads a tracked download as untracked, so the caller
    must see the failure and retry instead.
    """

    route = respx.get(f"{_BASE}/queue")
    # Page 1 succeeds. Page 2 stays 500 through the transport retries.
    route.side_effect = [httpx.Response(200, json=_queue_page(3, ["HASH0", "HASH1"]))] + [httpx.Response(500)] * 10
    recording = install_recording_hub()

    assert _make_client().queue() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch the Sonarr queue (status code 500) - will retry",
    ]


# --- queue_delete() ---------------------------------------------------------


@respx.mock
def test_queue_delete_builds_request() -> None:
    """`queue_delete` removes the row queue-only: the torrent stays in the client, nothing is blocklisted."""

    route = respx.delete(f"{_BASE}/queue/1516314797").respond(json={})
    client = _make_client()

    assert client.queue_delete(1516314797) is DeleteOutcome.OK
    assert dict(route.calls.last.request.url.params) == {"removeFromClient": "false", "blocklist": "false"}


@respx.mock
def test_queue_delete_non_200_fails_open() -> None:
    """A failed delete returns FAILED with a warning, never a crash: the close is best-effort."""

    respx.delete(f"{_BASE}/queue/7").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().queue_delete(7) is DeleteOutcome.FAILED
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not remove the finished download from Sonarr's queue (status code 500)",
    ]


# --- episodes() -------------------------------------------------------------


@respx.mock
def test_episodes_decodes_sorted_and_builds_request() -> None:
    """`episodes()` pulls one series' episodes season/episode-sorted, narrowing each to a `SonarrEpisode`.

    The request pins seriesId + the include flags.
    """

    route = respx.get(f"{_BASE}/episode").respond(json=sonarr_fixture("episodes_228_bahamut.json"))
    episodes = _make_client().episodes(228)

    assert episodes is not None
    assert len(episodes) == 13
    # sorted: the lone S00 special leads, S01E12 trails (decode + order in one).
    assert (episodes[0].season_number, episodes[0].episode_number, episodes[0].id) == (0, 1, 8475)
    assert (episodes[-1].season_number, episodes[-1].episode_number, episodes[-1].id) == (1, 12, 8487)
    assert dict(route.calls.last.request.url.params) == {
        "seriesId": "228",
        "includeImages": "false",
        "includeEpisodeFile": "true",
    }


@respx.mock
def test_episodes_missing_numbers_sort_first_without_crashing() -> None:
    """An episode missing seasonNumber/episodeNumber sorts first (as -1).

    This avoids raising a `None < int` TypeError and killing the whole fetch.
    """

    body = [
        {"id": 2, "seasonNumber": 1, "episodeNumber": 2},
        {"id": 9},
        {"id": 1, "seasonNumber": 1, "episodeNumber": 1},
    ]
    respx.get(f"{_BASE}/episode").respond(json=body)
    episodes = _make_client().episodes(228)

    assert episodes is not None
    assert [ep.id for ep in episodes] == [9, 1, 2]


@respx.mock
def test_episodes_non_200_returns_none_and_warns() -> None:
    """A non-200 episode read returns None and warns (the caller skips the id)."""

    respx.get(f"{_BASE}/episode").respond(status_code=500)
    recording = install_recording_hub()
    result = _make_client().episodes(228)

    assert result is None
    [warning] = diagnostic_messages(recording, Severity.WARNING)
    assert warning == "Could not fetch episodes for series 228 from Sonarr (status code 500) - skipping"


@respx.mock
def test_episodes_quiet_suppresses_unreachable_warning() -> None:
    """`quiet=True` still returns None on a non-200 but emits NO warning.

    This is the concurrent prefetch path, retried/logged on the main thread
    instead.
    """

    respx.get(f"{_BASE}/episode").respond(status_code=500)
    recording = install_recording_hub()
    result = _make_client().episodes(228, quiet=True)

    assert result is None
    assert diagnostic_messages(recording, Severity.WARNING) == []


# --- parse() (the one parse: agnostic numbers + matched pairs) ---------------


@respx.mock
def test_parse_decodes_season_episode() -> None:
    """An `SxxExx` name decodes to its own numbers plus Sonarr's matched episode.

    The request carries the name in the URL.
    """

    route = respx.get(f"{_BASE}/parse").respond(json=sonarr_fixture("parse_bahamut_s01e01.json"))
    info = _make_client().parse("Show.S01E01.mkv")

    assert info == ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
        series_title="Saijaku Muhai no Bahamut",
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=8476),),
        series_id=228,
    )
    assert route.calls.last.request.url.params["title"] == "Show.S01E01.mkv"


@respx.mock
def test_parse_decodes_absolute() -> None:
    """An absolute-numbered name decodes to its absolute numbers (season 0, no SxxExx episode numbers).

    The name alone carries no `(season, episode)`. Sonarr's series-matched
    `episodes` array comes along as `matched_episodes`, the exact pass's
    in-set fallback for exactly this shape.
    """

    respx.get(f"{_BASE}/parse").respond(json=sonarr_fixture("parse_glimmerzu_abs14.json"))
    info = _make_client().parse("GlimmerZu.-.14.mkv")

    assert info == ParsedFileInfo(
        season_number=0,
        episode_numbers=(),
        absolute_episode_numbers=(14,),
        series_title="To Glimmer-Zu",
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=14, id=2886),),
        series_id=72,
    )


@respx.mock
def test_parse_junk_matched_entry_drops_the_whole_array() -> None:
    """One unreadable `episodes[]` element drops the WHOLE array, the name's own numbers surviving.

    All-or-nothing on purpose: dropping the bad element alone would shorten a
    span and slip the every-pair check.
    """

    body: dict[str, object] = {
        "parsedEpisodeInfo": {"seasonNumber": 1, "episodeNumbers": [1]},
        "episodes": [
            {"seasonNumber": 1, "episodeNumber": 1},
            {"episodeNumber": 5},  # no seasonNumber
        ],
    }
    respx.get(f"{_BASE}/parse").respond(json=body)

    assert _make_client().parse("Cool.Anime.S01E01.mkv") == ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
    )


@respx.mock
def test_parse_reads_full_season_flag() -> None:
    """`parse()` lifts `parsedEpisodeInfo.fullSeason` onto the result.

    A bare-"S05" name Sonarr matches to a whole season carries the flag the
    grab-time placement rejects on, even when the season is small enough to slip
    under the span cap. Absent/false reads False.
    """

    respx.get(f"{_BASE}/parse").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "episodes": [{"seasonNumber": 5, "episodeNumber": 1}],
                    "parsedEpisodeInfo": {"fullSeason": True},
                },
            ),
            httpx.Response(200, json={"episodes": [{"seasonNumber": 1, "episodeNumber": 1}]}),
        ],
    )
    flagged = _make_client().parse("Show.S05.mkv")
    assert flagged is not None and flagged.full_season is True
    plain = _make_client().parse("Show.S01E01.mkv")
    assert plain is not None and plain.full_season is False


@respx.mock
def test_parse_clean_no_match_returns_an_empty_parse() -> None:
    """A clean 200 where Sonarr matched no episode returns the empty parse.

    This is a *confirmed* no-match the caller may cache (pinned to the series
    hash), distinct from a failure's None. A missing `episodes` key is
    the same clean no-match.
    """

    respx.get(f"{_BASE}/parse").mock(
        side_effect=[
            httpx.Response(200, json={"episodes": []}),
            httpx.Response(200, json={}),
        ],
    )
    assert _make_client().parse("Unmatched.Release.mkv") == ParsedFileInfo()
    assert _make_client().parse("Unmatched.Release.mkv") == ParsedFileInfo()


@respx.mock
def test_parse_wrong_shape_episodes_returns_none() -> None:
    """A 200 whose `episodes` is present but not a list (a mangled response) returns the uncacheable None.

    This is NOT the cacheable empty parse.
    """

    respx.get(f"{_BASE}/parse").mock(
        side_effect=[
            httpx.Response(200, json={"episodes": {"mangled": True}}),
            httpx.Response(200, json={"episodes": None}),
        ],
    )
    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None
    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None


@respx.mock
def test_parse_malformed_payload_warns_and_returns_none() -> None:
    """A payload the parse model refuses warns with the field summary and returns None."""

    respx.get(f"{_BASE}/parse").respond(json={"parsedEpisodeInfo": {"seasonNumber": "one"}})
    recording = install_recording_hub()

    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Could not parse Cool.Anime.S01E01.mkv via Sonarr "
            "(malformed response: parsedEpisodeInfo.seasonNumber: int_parsing)"
        ),
    ]


@respx.mock
def test_parse_non_200_returns_none() -> None:
    """A non-200 parse returns None (a failure that must NOT be cached) with a warning naming the file."""

    respx.get(f"{_BASE}/parse").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().parse("Cool.Anime.S01E01.mkv") is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not parse Cool.Anime.S01E01.mkv via Sonarr (status code 500)",
    ]


# --- manual_import_candidates_by_download_id() ---------------------------------------------


@respx.mock
def test_manual_import_candidates_decodes_and_uppercases_downloadid() -> None:
    """The ManualImport fetch keys on the UPPERCASED infohash (no `seriesId`).

    It narrows each candidate to its `path` / `quality` / `rejections`.
    """

    pending = _make_pending_import(
        infohash="abcdef0123456789abcdef0123456789abcdef01",
        title="Yamada-kun",
    )
    route = respx.get(f"{_BASE}/manualimport").respond(json=sonarr_fixture("manualimport_yamada.json"))
    candidates = _make_client().manual_import_candidates_by_download_id(pending=pending)

    assert candidates is not None
    assert len(candidates) == 2
    first = candidates[0]
    assert first.path == (
        "/downloads/Yamada-kun.and.the.Seven.Witches.S00.480p.DVDRip.Opus2.0.x264-Headpatter/"
        "Yamada-kun.and.the.Seven.Witches.S00E01.480p.DVDRip.Opus2.0.x264-Headpatter.mkv"
    )
    assert first.quality is not None
    assert first.rejections[0].reason == "Unknown Series"
    assert dict(route.calls.last.request.url.params) == {
        "downloadId": "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
        "filterExistingFiles": "false",
    }
    # The fetch must use the long ManualImport timeout (slow remote mounts).
    timeout = cast("dict[str, float | None]", route.calls.last.request.extensions["timeout"])
    assert timeout == {"connect": 120, "read": 120, "write": 120, "pool": 120}


@respx.mock
def test_manual_import_candidates_non_200_returns_none_silently() -> None:
    """A non-200 fetch returns None with NO warning.

    The executor owns the fallback's messaging, since a warn here would brand every
    poll of a download whose history is closed with a misleading "will retry" right before the folder
    fetch handles it.
    """

    recording = install_recording_hub()
    pending = _make_pending_import(infohash="a" * 40, title="Yamada-kun")
    respx.get(f"{_BASE}/manualimport").respond(status_code=500)
    assert _make_client().manual_import_candidates_by_download_id(pending=pending) is None
    assert diagnostic_messages(recording, Severity.WARNING) == []


# --- manual_import_candidates_by_folder() (GET /manualimport?folder=) --------


@respx.mock
def test_folder_fetch_sends_folder_and_never_download_or_series_id() -> None:
    """CONTRACT: the folder fetch carries NEITHER `downloadId` NOR `seriesId`.

    A `downloadId` re-enters the poisoned tracked branch (the NRE the fallback
    exists to dodge). A `seriesId` makes the controller read the LIBRARY folder
    instead of pearlarr's. Candidates decode exactly like the downloadId fetch.
    """

    route = respx.get(f"{_BASE}/manualimport").respond(json=sonarr_fixture("manualimport_yamada.json"))
    candidates = _make_client().manual_import_candidates_by_folder(
        folder="/remote/torrents/Show S01",
        title="Yamada-kun",
    )

    assert candidates is not None
    assert len(candidates) == 2
    assert dict(route.calls.last.request.url.params) == {
        "folder": "/remote/torrents/Show S01",
        "filterExistingFiles": "false",
    }
    # The fetch must use the long ManualImport timeout (slow remote mounts).
    timeout = cast("dict[str, float | None]", route.calls.last.request.extensions["timeout"])
    assert timeout == {"connect": 120, "read": 120, "write": 120, "pool": 120}


@respx.mock
def test_folder_fetch_non_200_returns_none_and_warns() -> None:
    """A non-200 folder fetch is None, so the import retries, with a warning naming the title."""

    respx.get(f"{_BASE}/manualimport").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().manual_import_candidates_by_folder(folder="/d", title="Yamada-kun") is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch folder-scan import candidates for Yamada-kun (status code 500) - will retry",
    ]


@respx.mock
def test_folder_fetch_empty_list_decodes_to_empty() -> None:
    """200 `[]` (an invisible/untranslated folder) decodes to [], never None."""

    respx.get(f"{_BASE}/manualimport").respond(json=[])
    assert _make_client().manual_import_candidates_by_folder(folder="/nonexistent", title="t") == []


# --- history_for_download() (GET /history) -----------------------------------


@respx.mock
def test_history_for_download_decodes_and_pins_paging() -> None:
    """The history read pins page-1 / pageSize-100 / date-descending explicitly and uppercases the hash.

    A 23-file batch has ~46+ events. An unlucky default page size would yield a
    false answer from a partial newest-first window.
    """

    envelope: dict[str, object] = {
        "page": 1,
        "pageSize": 100,
        "totalRecords": 2,
        "records": [
            {"id": 32289, "eventType": "downloadFolderImported", "date": "2026-06-20T06:15:30Z"},
            {"id": 31000, "eventType": "grabbed", "date": "2026-06-20T05:00:00Z"},
        ],
    }
    route = respx.get(f"{_BASE}/history").respond(json=envelope)
    page = _make_client().history_for_download(download_id="abcdef0123456789abcdef0123456789abcdef01")

    assert page is not None
    assert [r.event_type for r in page.records] == ["downloadFolderImported", "grabbed"]
    assert page.records[0].date == "2026-06-20T06:15:30Z"
    assert dict(route.calls.last.request.url.params) == {
        "downloadId": "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
        "page": "1",
        "pageSize": "100",
        "sortKey": "date",
        "sortDirection": "descending",
    }


@respx.mock
def test_history_for_download_non_200_returns_none() -> None:
    """A failed history read is None ("no answer") - never read as clean history."""

    respx.get(f"{_BASE}/history").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().history_for_download(download_id="a" * 40) is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not read Sonarr's history for a download (status code 500) - assuming it is healthy",
    ]


@respx.mock
def test_history_for_download_junk_record_is_skipped_not_fatal() -> None:
    """A junk `records[]` element skips. The page (and its answer) survives."""

    envelope: dict[str, object] = {
        "records": ["junk", {"eventType": "downloadFailed", "date": "2026-01-01T00:00:00Z"}],
    }
    respx.get(f"{_BASE}/history").respond(json=envelope)
    page = _make_client().history_for_download(download_id="a" * 40)

    assert page is not None
    assert [r.event_type for r in page.records] == ["downloadFailed"]


# --- remote_path_mappings() (GET /remotepathmapping) -------------------------


@respx.mock
def test_remote_path_mappings_decodes() -> None:
    """Mappings decode host/remotePath/localPath. None only on failure."""

    respx.get(f"{_BASE}/remotepathmapping").respond(
        json=[
            {
                "host": "seedbox.example",
                "remotePath": "/home/u/torrents/4k-tv/",
                "localPath": "/remote/torrents/4k-tv/",
                "id": 5,
            },
        ],
    )
    mappings = _make_client().remote_path_mappings()

    assert mappings is not None
    [mapping] = mappings
    assert mapping.host == "seedbox.example"
    assert mapping.remote_path == "/home/u/torrents/4k-tv/"
    assert mapping.local_path == "/remote/torrents/4k-tv/"


@respx.mock
def test_remote_path_mappings_non_200_returns_none() -> None:
    """Failure is None, distinct from "no mappings configured" ([])."""

    respx.get(f"{_BASE}/remotepathmapping").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().remote_path_mappings() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch Sonarr's remote path mappings (status code 500) - using download paths as-is",
    ]


# --- wait-path no-retry handle -------------------------------------------------


@respx.mock
def test_wait_path_polls_make_exactly_one_attempt() -> None:
    """The three wait pass polls use the no-retry handle: ONE attempt each.

    manual_import_candidates / the folder fetch / the history read run inside
    the wait pass's poll loop, which IS the retry mechanism, while in-call retries
    would only stretch each poll and multiply identical warnings.
    """

    manualimport = respx.get(f"{_BASE}/manualimport").respond(status_code=500)
    history = respx.get(f"{_BASE}/history").respond(status_code=500)
    client = _make_client()
    pending = _make_pending_import(infohash="a" * 40, title="Yamada-kun")

    assert client.manual_import_candidates_by_download_id(pending=pending) is None
    assert manualimport.call_count == 1
    assert client.manual_import_candidates_by_folder(folder="/downloads/x", title="Yamada-kun") is None
    assert manualimport.call_count == 2
    assert client.history_for_download(download_id="a" * 40) is None
    assert history.call_count == 1


@respx.mock
def test_sweep_reads_keep_the_retry_budget() -> None:
    """The non-wait reads (episodes, parse) use the primary retrying handle.

    `/parse` is a string parse, never a disk read, so one transport serves the
    scan and the wait pass alike.
    """

    episodes = respx.get(f"{_BASE}/episode").respond(status_code=500)
    parse = respx.get(f"{_BASE}/parse").respond(status_code=500)
    client = _make_client()

    assert client.episodes(1, quiet=True) is None
    assert episodes.call_count == GET_RETRIES + 1
    assert client.parse("Show.S01E01.mkv") is None
    assert parse.call_count == GET_RETRIES + 1


# --- manual_import_execute() / refresh_monitored_downloads() (POST /command) -


@respx.mock
def test_manual_import_execute_posts_body_and_returns_id() -> None:
    """The `ManualImport` command POSTs `{name, importMode, files}` and returns the queued command id.

    `import_mode` threads straight into the body.
    """

    file = ManualImportFile(
        path="/downloads/show/ep01.mkv",
        seriesId=42,
        episodeIds=[101],
        releaseGroup="SubsPlease",
        downloadId="A" * 40,
        languages=[Language(id=1, name="English")],
    )
    expected_body: dict[str, object] = {
        "name": "ManualImport",
        "importMode": "move",
        "files": [
            {
                "path": "/downloads/show/ep01.mkv",
                "seriesId": 42,
                "episodeIds": [101],
                "releaseGroup": "SubsPlease",
                "downloadId": "A" * 40,
                "languages": [{"id": 1, "name": "English"}],
            },
        ],
    }
    route = respx.post(f"{_BASE}/command").respond(json={"id": 4242})
    command_id = _make_client().manual_import_execute(files=[file], import_mode="move")

    assert command_id == 4242
    assert json.loads(route.calls.last.request.content) == expected_body


@respx.mock
def test_post_command_request_error_returns_none() -> None:
    """A failed command POST returns None, never raising through the import path, with a warning naming the command."""

    respx.post(f"{_BASE}/command").mock(side_effect=httpx.ConnectError("boom"))
    recording = install_recording_hub()

    assert _make_client().refresh_monitored_downloads() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not queue RefreshMonitoredDownloads command (request failed (ConnectError)) - will retry",
    ]


@respx.mock
def test_post_command_2xx_non_object_warns_and_returns_none() -> None:
    """A 2xx command POST whose body isn't a JSON object warns before returning None.

    Sonarr may still have queued the command, so leave a warning.
    """

    respx.post(f"{_BASE}/command").respond(status_code=201, json=[])
    recording = install_recording_hub()

    assert _make_client().refresh_monitored_downloads() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not confirm the RefreshMonitoredDownloads command was queued (unexpected payload) - will retry",
    ]


@respx.mock
def test_refresh_monitored_downloads_posts_command_name() -> None:
    """`RefreshMonitoredDownloads` POSTs only `{name}` and returns its id."""

    route = respx.post(f"{_BASE}/command").respond(json={"id": 77})
    command_id = _make_client().refresh_monitored_downloads()

    assert command_id == 77
    assert json.loads(route.calls.last.request.content) == {"name": "RefreshMonitoredDownloads"}


# --- command_status() / list_commands() -------------------------------------


@respx.mock
def test_command_status_decodes() -> None:
    """A single-command GET narrows to a `CommandResource` with status/result."""

    respx.get(f"{_BASE}/command/55").respond(
        json={"id": 55, "status": "completed", "result": "successful", "started": "2026-10-23T03:57:47Z"},
    )
    status = _make_client().command_status(55)

    assert status.id == 55
    assert status.status == "completed"
    assert status.result == "successful"
    assert status.started == datetime(2026, 10, 23, 3, 57, 47, tzinfo=UTC)


@respx.mock
def test_command_status_non_200_returns_default() -> None:
    """A non-200 status read yields a default (status-None) `CommandResource`.

    The caller treats the import as unverified and leaves the pending import as it is.
    """

    respx.get(f"{_BASE}/command/9").respond(status_code=503)
    recording = install_recording_hub()

    assert _make_client().command_status(9) == CommandResource()
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch status for command 9 (status code 503) - leaving the import unverified",
    ]


@respx.mock
def test_list_commands_decodes_with_nested_files() -> None:
    """The command LIST narrows each command.

    This includes the nested `body.files` each in-flight ManualImport carries
    (the guard's match signal).
    """

    respx.get(f"{_BASE}/command").respond(json=sonarr_fixture("command_list.json"))
    commands = _make_client().list_commands()

    assert commands is not None
    assert len(commands) == 5
    first = commands[0]
    assert first.name == "ManualImport"
    assert first.status == "started"
    assert first.message == "Processing file 4 of 8"
    assert first.files[0].download_id == "3333333333333333333333333333333333333333"
    assert first.files[0].series_id == 169
    assert first.files[0].episode_ids == (6605,)


@respx.mock
def test_list_commands_non_200_returns_none() -> None:
    """A non-200 command list reads as unreadable (None)."""

    respx.get(f"{_BASE}/command").respond(status_code=503)
    recording = install_recording_hub()

    assert _make_client().list_commands() is None
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch the Sonarr command list (status code 503) - will retry",
    ]


@pytest.mark.parametrize("value", [1729655860, "not a time", "2026-10-23T03:57:40"])
@respx.mock
def test_list_commands_decodes_an_unzoned_started_as_none(value: int | str) -> None:
    """A command start time that isn't a zoned ISO 8601 timestamp decodes as None."""

    respx.get(f"{_BASE}/command").respond(
        json=[{"id": 1, "name": "RefreshMonitoredDownloads", "status": "completed", "started": value}]
    )
    commands = _make_client().list_commands()

    assert commands is not None
    assert commands[0].started is None


# --- history_since() ----------------------------------------------------------


@respx.mock
def test_history_since_asks_for_the_records_alone() -> None:
    """Sonarr's history read asks for every event since the date, with the series and episode flags off.

    How each record reads, `seriesId` as its item id included, is the model's (`test_arr_models`,
    `test_boundary_parity`).
    """

    route = respx.get(f"{_BASE}/history/since").respond(json=[])
    _make_client().history_since("2026-06-30T08:00:00Z")

    assert dict(route.calls.last.request.url.params) == {
        "date": "2026-06-30T08:00:00Z",
        "includeSeries": "false",
        "includeEpisode": "false",
    }


# --- quality_definitions() / languages() ------------------------------------


@respx.mock
def test_quality_definitions_decodes_nested_quality_verbatim() -> None:
    """Each definition carries its nested `quality`, which the import payload re-emits."""

    respx.get(f"{_BASE}/qualitydefinition").respond(json=sonarr_fixture("qualitydefinitions.json"))
    defs = _make_client().quality_definitions()

    assert defs[0].quality == Quality(id=0, name="Unknown", source="unknown", resolution=0)


@respx.mock
def test_languages_decodes_id_name_objects() -> None:
    """Languages decode to the {id, name} objects the resolver matches/re-builds."""

    body: list[object] = [{"id": 1, "name": "English"}, {"id": 8, "name": "Japanese"}]
    respx.get(f"{_BASE}/language").respond(json=body)

    assert _make_client().languages() == [Language(id=1, name="English"), Language(id=8, name="Japanese")]


@respx.mock
def test_quality_definitions_non_200_returns_empty() -> None:
    """A non-200 quality-definitions read falls back to an empty list."""

    respx.get(f"{_BASE}/qualitydefinition").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().quality_definitions() == []
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch quality definitions from Sonarr (status code 500) - using other quality sources",
    ]


@respx.mock
def test_languages_non_200_returns_empty() -> None:
    """A non-200 languages read falls back to an empty list."""

    respx.get(f"{_BASE}/language").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().languages() == []
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not fetch languages from Sonarr (status code 500) - using the candidate's languages",
    ]


# --- download_client_config() --------------------------------------------------


@respx.mock
def test_download_client_config_decodes_the_switch() -> None:
    """A healthy read carries Sonarr's completed-download-handling switch through."""

    respx.get(f"{_BASE}/config/downloadclient").respond(json={"enableCompletedDownloadHandling": False})
    config = _make_client().download_client_config()

    assert config.enable_completed_download_handling is False


@respx.mock
def test_download_client_config_non_200_assumes_cdh_on() -> None:
    """A failed read falls back to the defaults (CDH on), deferring rather than racing Sonarr."""

    respx.get(f"{_BASE}/config/downloadclient").respond(status_code=500)
    recording = install_recording_hub()

    assert _make_client().download_client_config() == DownloadClientConfig()
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Could not read Sonarr's download client config (status code 500) - assuming completed download handling is on",
    ]


@respx.mock
def test_download_client_config_malformed_body_warns_and_defaults() -> None:
    """A 200 with a junk payload warns once and falls back to the defaults."""

    respx.get(f"{_BASE}/config/downloadclient").respond(json={"enableCompletedDownloadHandling": "junk"})
    recording = install_recording_hub()
    config = _make_client().download_client_config()

    assert config == DownloadClientConfig()
    assert diagnostic_messages(recording, Severity.WARNING) == [
        (
            "Could not read Sonarr's download client config "
            "(malformed response: enableCompletedDownloadHandling: bool_parsing) - "
            "assuming completed download handling is on"
        ),
    ]
