# pyright: strict
"""Offline end-to-end pins for the manual-import folder-scan fallback.

These drive the REAL CLI composition root (`cli.run_single`) against live
loopback HTTP servers - a mock Sonarr and a mock qBittorrent, the same wire
surface `scripts/demo/demo_mocks.py` serves - so every hop is the production
stack: `ArrHttp` over httpx to the Sonarr mock, `qbittorrentapi` over its own
session to the qBittorrent mock. The carried-over pending record is seeded
straight into cache.db and every mapping source is disabled, so the scan finds
zero series, nothing leaves loopback, and the end-of-run blocking monitor is
the only thing driving the import.

Three scenarios pin the dead-loop cure end to end:

* DEAD-TRACKED: the downloadId scan 500s forever (Sonarr's poisoned tracked
  download), history's newest relevant event is `downloadFolderImported`, and
  the ManualImport command reports `failed` - yet the run converges to
  IMPORTED, because the episode FILES are the source of truth (the poisoned
  Execute tail in real Sonarr fails the command AFTER the per-file imports
  land). The POSTed entries omit `downloadId` and force `importMode: copy`.
* TRANSIENT: the downloadId scan fails once and heals; the folder scan's one
  activation returns `[]`, which must NOT pin folder mode - the next poll goes
  back to the downloadId scan and imports through the normal tracked shape.
* UNSCANNABLE: both scans fail every poll; the record defers (never drops)
  and the folder-scan warn template fires.
"""

import json
import re
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, NamedTuple, cast, override
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

import pearlarr.arr_http as arr_http
from pearlarr.cache import UPDATED_AT_STR_FORMAT, CacheStore
from pearlarr.cli import run_single
from pearlarr.config import AppConfig, Arr
from pearlarr.json_narrow import is_json_list, is_json_obj
from pearlarr.manual_import import PendingImport, normalize_basename
from pearlarr.output import current_hub
from pearlarr.paths import DATA_DIR_ENV
from pearlarr.seadex_types import Json

from .builders import make_config
from .http_mock import sonarr_fixture

_COMMAND_ID = re.compile(r"/api/v3/command/(\d+)")

_INFOHASH = "ab12" * 10
_SERIES_ID = 301
_GROUP = "Thighs"
_TITLE = "Demo Batch"
_CONTENT_PATH = "/downloads/Demo Batch S01 [Thighs]"
_EPISODES: tuple[tuple[int, str], ...] = (
    (301001, "Demo Batch - S01E01 (BD 1080p) [Thighs].mkv"),
    (301002, "Demo Batch - S01E02 (BD 1080p) [Thighs].mkv"),
)
_IMPORTED_EVENT_DATE = "2026-06-20T14:05:11Z"
_DEAD_TRACKED_NOTE = (
    "Sonarr recorded this download as imported on 2026-06-20 and won't serve it by id - "
    "importing from its folder instead"
)

_JP_EN: Json = [{"id": 8, "name": "Japanese"}, {"id": 1, "name": "English"}]


@dataclass(frozen=True)
class _Scenario:
    """One mock-Sonarr behavior matrix for the poisoned download's scans."""

    heal_download_scan: bool
    """Whether the downloadId scan heals (200 + candidates) once the folder
    scan has been consulted - the transient-blip timeline; False keeps it a
    permanent 500 (the dead-tracked NRE)."""

    folder_scan: Literal["candidates", "empty", "error"]
    """What the folder scan serves: real candidates, a 200 `[]` (folder not
    visible), or a 500."""

    history_event: str
    """The newest relevant `/api/v3/history` eventType (decides the probe
    verdict: `downloadFolderImported` -> dead-tracked, `grabbed` -> clean)."""

    ready_timeout: int = 30
    """The run's `imports.ready_timeout`; small for the deadline scenario."""


_DEAD_TRACKED = _Scenario(heal_download_scan=False, folder_scan="candidates", history_event="downloadFolderImported")
_TRANSIENT = _Scenario(heal_download_scan=True, folder_scan="empty", history_event="grabbed")
_UNSCANNABLE = _Scenario(
    heal_download_scan=False,
    folder_scan="error",
    history_event="downloadFolderImported",
    ready_timeout=1,
)


class _Request(NamedTuple):
    """One recorded HTTP request (query values flattened to their first value)."""

    method: str
    path: str
    params: dict[str, str]


@dataclass
class _World:
    """Lock-guarded mutable state both mock servers serve from and record into."""

    scenario: _Scenario
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: list[_Request] = field(default_factory=list["_Request"])
    download_scan_statuses: list[int] = field(default_factory=list[int])
    folder_scans: int = 0
    files_landed: bool = False
    cmd_seq: int = 100
    manual_commands: list[tuple[int, dict[str, Json]]] = field(default_factory=list[tuple[int, dict[str, Json]]])

    def record(self, request: _Request) -> None:
        with self.lock:
            self.requests.append(request)

    def requests_for(self, path: str) -> list[_Request]:
        """Every recorded request to `path` (exact match), in arrival order."""

        with self.lock:
            return [r for r in self.requests if r.path == path]

    def next_command(self, name: str, body: dict[str, Json]) -> int:
        """Allocate a command id; a ManualImport also lands the files.

        Landing on POST is the real timeline the dead-tracked pin relies on:
        Sonarr's per-file imports land BEFORE the poisoned Execute tail fails
        the command, so the mock lands files while the command reads `failed`.
        """

        with self.lock:
            self.cmd_seq += 1
            if name.casefold() == "manualimport":
                self.manual_commands.append((self.cmd_seq, body))
                self.files_landed = True
            return self.cmd_seq

    def manual_command_ids(self) -> set[int]:
        with self.lock:
            return {cid for cid, _ in self.manual_commands}

    def download_scan_response(self) -> list[Json] | None:
        """The next downloadId-scan body (None = serve a 500) - and count it.

        The transient heal is keyed on "the folder scan ran", not an attempt
        count, so the fixture stays valid across in-call retry policy changes:
        one POLL fails (however many attempts that takes), the fallback's
        folder scan activates once, and the NEXT poll's scan is healthy.
        """

        with self.lock:
            healed = self.scenario.heal_download_scan and self.folder_scans > 0
            self.download_scan_statuses.append(200 if healed else 500)
            return _candidates() if healed else None

    def folder_scan_response(self) -> list[Json] | None:
        """The next folder-scan body (None = serve a 500) - and count the activation."""

        with self.lock:
            self.folder_scans += 1
            match self.scenario.folder_scan:
                case "candidates":
                    return _candidates()
                case "empty":
                    return []
                case "error":
                    return None

    def episodes_payload(self) -> list[Json]:
        """The series' episodes: bare until the ManualImport lands the files."""

        with self.lock:
            landed = self.files_landed
        rows: list[Json] = []
        for number, (ep_id, name) in enumerate(_EPISODES, start=1):
            row: dict[str, Json] = {
                "id": ep_id,
                "seriesId": _SERIES_ID,
                "seasonNumber": 1,
                "episodeNumber": number,
                "monitored": True,
                "hasFile": landed,
                "episodeFileId": ep_id * 100 if landed else 0,
            }
            if landed:
                row["episodeFile"] = {
                    "id": ep_id * 100,
                    "relativePath": f"Season 01/{name}",
                    "releaseGroup": _GROUP,
                    "size": 1000,
                }
            rows.append(row)
        return rows

    def history_payload(self) -> Json:
        """The paged `/api/v3/history` envelope, date-descending.

        The newest row is an irrelevant `episodeFileDeleted` the classifier
        must skip; the scenario's event decides the verdict beneath it.
        """

        return {
            "page": 1,
            "pageSize": 100,
            "sortKey": "date",
            "sortDirection": "descending",
            "totalRecords": 3,
            "records": [
                {"id": 913, "eventType": "episodeFileDeleted", "date": "2026-07-15T06:10:00Z"},
                {
                    "id": 912,
                    "eventType": self.scenario.history_event,
                    "date": _IMPORTED_EVENT_DATE,
                    "downloadId": _INFOHASH.upper(),
                },
                {"id": 901, "eventType": "grabbed", "date": "2026-06-20T12:00:00Z", "downloadId": _INFOHASH.upper()},
            ],
        }

    def command_list_payload(self) -> list[Json]:
        """Every issued ManualImport, reported `failed` (the poisoned Execute tail)."""

        with self.lock:
            return [
                {"id": cid, "name": "ManualImport", "status": "failed", "body": {"files": body.get("files")}}
                for cid, body in self.manual_commands
            ]


def _candidates() -> list[Json]:
    """The on-disk candidates either scan mode serves for the download folder."""

    out: list[Json] = []
    for number, (_, name) in enumerate(_EPISODES, start=1):
        out.append(
            {
                "id": number,
                "path": f"{_CONTENT_PATH}/{name}",
                "relativePath": name,
                "name": name,
                "size": 1000,
                "quality": {
                    "quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
                    "revision": {"version": 1, "real": 0, "isRepack": False},
                },
                "languages": _JP_EN,
                "rejections": [],
            },
        )
    return out


class _MockServer(ThreadingHTTPServer):
    """A loopback mock server carrying the shared world for its handler."""

    daemon_threads = True

    def __init__(self, handler: type[BaseHTTPRequestHandler], world: _World) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.world = world

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _JsonHandler(BaseHTTPRequestHandler):
    """Shared base: typed world access, JSON/text responses, quiet logs."""

    @property
    def world(self) -> _World:
        return cast("_MockServer", self.server).world

    def _record(self) -> _Request:
        parsed = urlparse(self.path)
        request = _Request(
            self.command,
            parsed.path,
            {key: values[0] for key, values in parse_qs(parsed.query).items()},
        )
        self.world.record(request)
        return request

    def _read_body(self) -> dict[str, Json]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        parsed: object = json.loads(raw or b"{}")
        return parsed if is_json_obj(parsed) else {}

    def _send_json(self, body: Json, status: int = 200) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, text: str, cookie: str | None = None) -> None:
        payload = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @override
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Quiet: the tests assert on the recorded requests, not stderr."""


class _SonarrHandler(_JsonHandler):
    """The slice of the Sonarr v3 API a monitor-only offline run touches."""

    def do_GET(self) -> None:
        request = self._record()
        route = request.path
        if route == "/api/v3/series":
            self._send_json([])
        elif route == "/api/v3/history/since":
            self._send_json([])
        elif route == "/api/v3/episode":
            self._send_json(self.world.episodes_payload())
        elif route == "/api/v3/qualitydefinition":
            self._send_json(cast("Json", sonarr_fixture("qualitydefinitions.json")))
        elif route == "/api/v3/language":
            self._send_json(_JP_EN)
        elif route == "/api/v3/remotepathmapping":
            self._send_json([])
        elif route == "/api/v3/queue":
            self._send_json({"page": 1, "pageSize": 1000, "totalRecords": 0, "records": []})
        elif route == "/api/v3/history":
            self._send_json(self.world.history_payload())
        elif route == "/api/v3/manualimport":
            self._manual_import(request)
        elif route == "/api/v3/command":
            self._send_json(self.world.command_list_payload())
        elif (match := _COMMAND_ID.fullmatch(route)) is not None:
            cid = int(match.group(1))
            status = "failed" if cid in self.world.manual_command_ids() else "completed"
            self._send_json({"id": cid, "name": "ManualImport", "status": status})
        else:
            self._send_json({}, 404)

    def _manual_import(self, request: _Request) -> None:
        """Route the two scan modes by their query shape."""

        if "downloadId" in request.params:
            body = self.world.download_scan_response()
        elif "folder" in request.params:
            body = self.world.folder_scan_response()
        else:
            self._send_json({}, 404)
            return
        if body is None:
            self._send_json({"message": "Object reference not set to an instance of an object"}, 500)
        else:
            self._send_json(body)

    def do_POST(self) -> None:
        request = self._record()
        if request.path != "/api/v3/command":
            self._send_json({}, 404)
            return
        body = self._read_body()
        name = body.get("name")
        name = name if isinstance(name, str) else ""
        cid = self.world.next_command(name, body)
        self._send_json({"id": cid, "name": name, "status": "queued"}, 201)


class _QbitHandler(_JsonHandler):
    """The slice of the qBittorrent WebUI API the wait monitor touches."""

    def _info_rows(self) -> list[Json]:
        return [
            {
                "hash": _INFOHASH,
                "name": f"{_TITLE} S01 [{_GROUP}]",
                "size": 2000,
                "total_size": 2000,
                "progress": 1.0,
                "dlspeed": 0,
                "eta": 8640000,
                "completed": 2000,
                "state": "stalledUP",
                "content_path": _CONTENT_PATH,
                "save_path": "/downloads",
                "category": "anime",
                "tags": "",
                "added_on": 1750000000,
                "amount_left": 0,
            },
        ]

    def do_HEAD(self) -> None:
        # qbittorrentapi probes reachability with a bare HEAD; 200 it quietly.
        _ = self._record()
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        route = self._record().path
        if route == "/api/v2/app/webapiVersion":
            self._send_text("2.9.3")
        elif route == "/api/v2/app/version":
            self._send_text("v4.6.7")
        elif route == "/api/v2/torrents/info":
            self._send_json(self._info_rows())
        elif route == "/api/v2/app/preferences":
            self._send_json({"save_path": "/downloads"})
        elif route == "/api/v2/transfer/info":
            self._send_json({"connection_status": "connected", "dl_info_speed": 0, "up_info_speed": 0})
        else:
            self._send_json({})

    def do_POST(self) -> None:
        route = self._record().path
        _ = self._read_body_bytes()
        if route == "/api/v2/auth/login":
            self._send_text("Ok.", cookie="SID=e2edemodemo; path=/")
        elif route == "/api/v2/auth/logout":
            self._send_text("")
        elif route == "/api/v2/torrents/info":
            self._send_json(self._info_rows())
        else:
            self._send_text("")

    def _read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""


@contextmanager
def _serve(world: _World) -> Generator[tuple[str, str]]:
    """Run both mock servers on ephemeral loopback ports for the with-block."""

    sonarr = _MockServer(_SonarrHandler, world)
    qbit = _MockServer(_QbitHandler, world)
    threading.Thread(target=sonarr.serve_forever, daemon=True).start()
    threading.Thread(target=qbit.serve_forever, daemon=True).start()
    try:
        yield sonarr.url, qbit.url
    finally:
        for server in (sonarr, qbit):
            server.shutdown()
            server.server_close()


def _seed_pending(cache_path: Path, checksum: str) -> None:
    """Plant the carried-over record a prior run's grab would have left behind."""

    record = PendingImport(
        infohash=_INFOHASH,
        series_id=_SERIES_ID,
        file_episode_map={normalize_basename(name): [ep_id] for ep_id, name in _EPISODES},
        episode_ids=[],
        release_group=_GROUP,
        is_dual_audio=False,
        seadex_files=[name for _, name in _EPISODES],
        title=_TITLE,
        added_at=datetime.now().strftime(UPDATED_AT_STR_FORMAT),
        coverage="S01 E01-E02",
        url=None,
        ordered_episode_ids=[ep_id for ep_id, _ in _EPISODES],
    )
    store = CacheStore.load(str(cache_path), config_checksum=checksum)
    try:
        store.put_pending(Arr.SONARR, _INFOHASH, record.to_json())
        store.save(preview=False)
    finally:
        store.close()


def _pending_after(cache_path: Path, checksum: str) -> frozenset[str]:
    """The pending infohashes still in the durable store after the run."""

    store = CacheStore.load(str(cache_path), config_checksum=checksum)
    try:
        return frozenset(store.get_pending(Arr.SONARR))
    finally:
        store.close()


class _RunOutcome(NamedTuple):
    """One offline run's observable results."""

    ok: bool
    world: _World
    pending_after: frozenset[str]


def _drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: _Scenario) -> _RunOutcome:
    """One real `run_single(sonarr=True)` against the scenario's mock servers."""

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    # Keep the in-call GET retries but drop their backoff base (jitter stays),
    # so a deliberately-failing scan doesn't stretch each poll by seconds.
    monkeypatch.setattr(arr_http, "BACKOFF_BASE_S", 0.0)

    world = _World(scenario)
    with _serve(world) as (sonarr_url, qbit_url):
        # qBittorrent configured -> a real (non-preview) run whose blocking
        # monitor drives the carried-over record; 1s polls keep it quick.
        config = make_config(
            url=sonarr_url,
            api_key="testkey",
            host=qbit_url,
            username="demo",
            password="demo-password",
            anime_mappings=False,
            anidb_mappings=False,
            anibridge_mappings=False,
            wait_mode="hybrid",
            import_poll_interval=1,
            progress_poll_interval=0,
            import_ready_timeout=scenario.ready_timeout,
        )
        config_path = tmp_path / "config.yml"
        config_path.write_bytes(yaml.safe_dump(config.model_dump(mode="json")).encode())
        # Owner-only, or the boot warns about a world-readable API key.
        config_path.chmod(0o600)
        # The run's own checksum (file bytes + env overlay), so the seeded
        # descriptor matches what the run will stamp.
        checksum = AppConfig.load(str(config_path)).checksum()
        cache_path = tmp_path / "cache.db"
        _seed_pending(cache_path, checksum)

        ok = run_single(sonarr=True)

    return _RunOutcome(ok, world, _pending_after(cache_path, checksum))


def _flat(text: str) -> str:
    """Console text with all wrapping collapsed, for whole-sentence asserts."""

    return " ".join(text.split())


def _body_files(body: dict[str, Json]) -> list[dict[str, Json]]:
    """The `files` array of a recorded ManualImport body, as typed objects."""

    files = body.get("files")
    if not is_json_list(files):
        return []
    return [entry for entry in files if is_json_obj(entry)]


def test_dead_tracked_download_imports_via_folder_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome = _drive(tmp_path, monkeypatch, _DEAD_TRACKED)
    out = _flat(capsys.readouterr().out)

    # Converged to IMPORTED although every downloadId scan 500'd AND the
    # ManualImport command reported `failed`: the episode FILES decided.
    assert outcome.ok is True
    assert outcome.pending_after == frozenset()
    assert 'imported title="Demo Batch · Thighs" files=2' in out
    assert "complete imported=1 deferred=0 failed=0" in out
    assert current_hub().counts.mark().errors == 0

    # The dated hub note, exactly once (memoized verdict, once per record per run).
    assert out.count(_DEAD_TRACKED_NOTE) == 1

    # CONTRACT: the folder request carries folder + filterExistingFiles and
    # NEITHER downloadId (tracked-branch re-entry) NOR seriesId (library scan).
    folder_scans = [r for r in outcome.world.requests_for("/api/v3/manualimport") if "folder" in r.params]
    assert folder_scans
    for request in folder_scans:
        assert request.params["folder"] == _CONTENT_PATH
        assert request.params["filterExistingFiles"] == "false"
        assert "downloadId" not in request.params
        assert "seriesId" not in request.params

    # The probe hit the paged history endpoint with the paging pinned explicitly.
    history = outcome.world.requests_for("/api/v3/history")
    assert history
    assert history[0].params == {
        "downloadId": _INFOHASH.upper(),
        "page": "1",
        "pageSize": "100",
        "sortKey": "date",
        "sortDirection": "descending",
    }

    # The once-per-run mapping fetch happened (and translated to a no-op).
    assert outcome.world.requests_for("/api/v3/remotepathmapping")

    # ONE ManualImport, importMode forced to copy (config `auto`), and every
    # file entry OMITS downloadId - the dead-tracked untracked-branch shape.
    commands = outcome.world.manual_commands
    assert len(commands) == 1
    body = commands[0][1]
    assert body.get("name") == "ManualImport"
    assert body.get("importMode") == "copy"
    files = _body_files(body)
    assert len(files) == len(_EPISODES)
    for entry in files:
        assert "downloadId" not in entry
        assert entry.get("seriesId") == _SERIES_ID
        assert entry.get("episodeIds")
        path = entry.get("path")
        assert isinstance(path, str) and path.startswith(f"{_CONTENT_PATH}/")


def test_transient_scan_blip_does_not_pin_folder_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome = _drive(tmp_path, monkeypatch, _TRANSIENT)
    out = _flat(capsys.readouterr().out)

    # Converged through the NORMAL path after the one-poll blip.
    assert outcome.ok is True
    assert outcome.pending_after == frozenset()
    assert 'imported title="Demo Batch · Thighs" files=2' in out
    assert "complete imported=1 deferred=0 failed=0" in out
    assert current_hub().counts.mark().errors == 0

    # The empty folder scan ran exactly once - no pin, so the next poll went
    # straight back to the downloadId scan, which had healed.
    assert outcome.world.folder_scans == 1
    statuses = outcome.world.download_scan_statuses
    assert statuses[0] == 500
    assert statuses[-1] == 200

    # The probe ran once (clean verdict), and no dead-tracked note was emitted.
    assert len(outcome.world.requests_for("/api/v3/history")) == 1
    assert "won't serve it by id" not in out

    # The healthy tracked shape: every entry KEEPS the downloadId, and the
    # configured `auto` import mode reaches the wire unchanged.
    commands = outcome.world.manual_commands
    assert len(commands) == 1
    body = commands[0][1]
    assert body.get("importMode") == "auto"
    files = _body_files(body)
    assert len(files) == len(_EPISODES)
    for entry in files:
        assert entry.get("downloadId") == _INFOHASH


def test_unscannable_download_defers_with_folder_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome = _drive(tmp_path, monkeypatch, _UNSCANNABLE)
    out = _flat(capsys.readouterr().out)

    # Both scans failing is a DEFERRAL, not a drop or a crash: the record
    # survives for a later run and the run itself still completes.
    assert outcome.ok is True
    assert outcome.pending_after == frozenset({_INFOHASH})
    assert 'not ready title="Demo Batch · Thighs"' in out
    assert "complete imported=0 deferred=1 failed=0" in out
    assert current_hub().counts.mark().errors == 0

    # The folder-mode warn template fired, naming the record.
    assert f"Could not fetch folder-scan import candidates for {_TITLE}" in out

    # The dead-tracked note still fired exactly once (probe succeeded, verdict
    # memoized), and nothing was ever POSTed.
    assert out.count(_DEAD_TRACKED_NOTE) == 1
    assert outcome.world.manual_commands == []
