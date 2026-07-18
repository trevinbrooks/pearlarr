"""Stateful mock Sonarr (:8989) + mock qBittorrent (:8080) for the demo recording.

Maintainer tooling behind `scripts/demo/record.sh`, which regenerates
`docs/assets/demo_run.gif`; nothing here ships in the package.

Timeline: the run grabs Frieren (PMR) and FMAB (McBalls). Frieren completes first,
Sonarr reports it importBlocked -> Pearlarr steps in with a manual import. FMAB
completes later and Sonarr "imports" it itself (importPending, then files land
progressively).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, override
from urllib.parse import parse_qs, urlparse

from seadex import SeaDexEntry, Tracker

from pearlarr.json_narrow import is_json_list, is_json_obj
from pearlarr.seadex_types import Json

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "sonarr"
DEMO_FIXTURES = Path(__file__).resolve().parent / "fixtures"

SXXEYY = re.compile(r"S(\d{1,2})E(\d{2,3})")
BEBOP_ABS = re.compile(r"Cowboy Bebop - (\d{2,3})")
BTIH = re.compile(r"btih:([0-9a-fA-F]{40})")
NYAA_ID = re.compile(r"nyaa\.si/(?:download|view)/(\d+)")

FRIEREN_DL_S = 13.0
FMAB_DL_S = 21.0
QUEUE_LAG_S = 1.0
FMAB_PENDING_S = 6.0
CMD_RUN_S = 2.0
IMPORT_FILE_S = 0.09  # per-file landing cadence once an import starts

JP_EN: Json = [{"id": 8, "name": "Japanese"}, {"id": 1, "name": "English"}]


def now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class MediaFile:
    """One file inside a torrent."""

    name: str
    size: int


@dataclass(frozen=True)
class QualityStub:
    """The quadruple Sonarr nests under `quality.quality` on files and candidates."""

    quality_id: int
    name: str
    source: str
    resolution: int

    def payload(self) -> Json:
        """The full `quality` object, revision included."""

        return {
            "quality": {
                "id": self.quality_id,
                "name": self.name,
                "source": self.source,
                "resolution": self.resolution,
            },
            "revision": {"version": 1, "real": 0, "isRepack": False},
        }


@dataclass(frozen=True)
class Role:
    """One torrent the demo grabs, plus everything both mocks serve about it."""

    name: str
    duration: float
    series_key: str
    import_group: str
    quality: QualityStub
    infohash: str
    view_id: str
    files: tuple[MediaFile, ...]
    import_files: dict[int, MediaFile]  # episode number -> the file that lands for it

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def build_role(
    seadex_client: SeaDexEntry,
    al_id: int,
    group: str,
    *,
    name: str,
    duration: float,
    series_key: str,
    quality: QualityStub,
) -> Role:
    """Pull `group`'s Nyaa torrent off the live SeaDex entry for `al_id`."""

    entry = seadex_client.from_id(al_id)
    for torrent in entry.torrents:
        if torrent.tracker is not Tracker.NYAA or torrent.release_group != group or torrent.infohash is None:
            continue
        m = NYAA_ID.search(torrent.url)
        files = tuple(MediaFile(f.name, f.size) for f in torrent.files)
        import_files: dict[int, MediaFile] = {}
        for f in files:
            ep = SXXEYY.search(f.name)
            if ep and int(ep.group(1)) == 1:
                import_files[int(ep.group(2))] = f
        return Role(
            name=name,
            duration=duration,
            series_key=series_key,
            import_group=group,
            quality=quality,
            infohash=torrent.infohash.lower(),
            view_id=m.group(1) if m else "",
            files=files,
            import_files=import_files,
        )
    raise SystemExit(f"no Nyaa torrent for alID={al_id} group={group}")


class MockState:
    """Mutable cross-thread state: what was added when, plus the manual-import command."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._added: dict[str, float] = {}
        self._cmd_seq = 100
        self._manual_cmd: tuple[int, float] | None = None

    def record_add(self, infohash: str) -> None:
        with self._lock:
            self._added.setdefault(infohash, now())

    def added_snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._added)

    def added_at(self, infohash: str) -> float | None:
        with self._lock:
            return self._added.get(infohash)

    def next_command(self, *, manual: bool) -> int:
        with self._lock:
            self._cmd_seq += 1
            if manual:
                self._manual_cmd = (self._cmd_seq, now())
            return self._cmd_seq

    def manual_command(self) -> tuple[int, float] | None:
        with self._lock:
            return self._manual_cmd


@dataclass(frozen=True)
class World:
    """Everything the handlers serve, built once at startup."""

    roles: dict[str, Role]
    episodes: dict[str, tuple[dict[str, Json], ...]]
    state: MockState

    def by_hash(self, infohash: str) -> Role | None:
        wanted = infohash.lower()
        return next((r for r in self.roles.values() if r.infohash == wanted), None)

    def by_series(self, series_key: str) -> Role | None:
        return next((r for r in self.roles.values() if r.series_key == series_key), None)


_world: World  # assigned in main() before the servers accept requests


def load_json(path: Path) -> Json:
    """Load a fixture that must be a JSON container."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not (is_json_list(raw) or is_json_obj(raw)):
        raise SystemExit(f"{path} is not a JSON container")
    return raw


def load_object_rows(path: Path) -> tuple[dict[str, Json], ...]:
    """Load a fixture that must be a JSON array of objects."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_json_list(raw):
        raise SystemExit(f"{path} is not a JSON array")
    rows: list[dict[str, Json]] = []
    for item in raw:
        if not is_json_obj(item):
            raise SystemExit(f"{path} holds a non-object row")
        rows.append(item)
    return tuple(rows)


def int_field(row: dict[str, Json], key: str) -> int:
    """Read an int field off a fixture row, failing loudly on fixture drift."""

    value = row.get(key)
    if not isinstance(value, bool) and isinstance(value, int):
        return value
    raise SystemExit(f"expected int {key!r} in fixture row, got {value!r}")


def complete_at(role: Role) -> float | None:
    t0 = _world.state.added_at(role.infohash)
    return None if t0 is None else t0 + role.duration


def import_started_at(role: Role) -> float | None:
    """When this role's files begin landing in Sonarr (None = not yet)."""

    done = complete_at(role)
    if done is None or now() < done:
        return None
    if role is _world.roles["frieren"]:
        cmd = _world.state.manual_command()
        return None if cmd is None else cmd[1] + CMD_RUN_S
    return done + QUEUE_LAG_S + FMAB_PENDING_S


def files_landed(role: Role) -> int:
    t0 = import_started_at(role)
    if t0 is None or now() < t0:
        return 0
    total = len(role.import_files)
    return min(total, int((now() - t0) / IMPORT_FILE_S) + 1)


def episodes_response(series_key: str) -> Json:
    base = _world.episodes.get(series_key, ())
    role = _world.by_series(series_key)
    if role is None:
        return list(base)
    landed = files_landed(role)
    if landed == 0:
        return list(base)
    numbers = sorted(role.import_files)[:landed]
    landed_set = set(numbers)
    out: list[Json] = []
    for ep in base:
        n = int_field(ep, "episodeNumber")
        # Landed episodes get the NEW file even when one already exists - the
        # upgrade path (FMAB) replaces files, and the landing check must see it.
        if n not in landed_set:
            out.append(ep)
            continue
        landed_file = role.import_files[n]
        sid = int_field(ep, "seriesId")
        updated = dict(ep)
        updated["hasFile"] = True
        updated["episodeFileId"] = sid * 500000 + n
        episode_file: dict[str, Json] = {
            "seriesId": sid,
            "seasonNumber": 1,
            "relativePath": f"Season 01/{landed_file.name}",
            "path": f"/tv/imported/Season 01/{landed_file.name}",
            "size": landed_file.size,
            "dateAdded": "2026-01-01T00:00:00Z",
            "releaseGroup": role.import_group,
            "languages": JP_EN,
            "quality": role.quality.payload(),
            "customFormats": [],
            "id": sid * 500000 + n,
        }
        updated["episodeFile"] = episode_file
        out.append(updated)
    return out


def queue_records() -> list[Json]:
    records: list[Json] = []
    frieren = _world.roles["frieren"]
    done = complete_at(frieren)
    stepped_in = _world.state.manual_command() is not None
    if done is not None and now() >= done + QUEUE_LAG_S and not stepped_in:
        records.append(
            {
                "downloadId": frieren.infohash.upper(),
                "trackedDownloadState": "importBlocked",
                "trackedDownloadStatus": "warning",
                "status": "completed",
                "title": frieren.name,
                "size": frieren.size,
                "sizeleft": 0,
                "statusMessages": [
                    {"title": frieren.name, "messages": ["Import blocked: episode identity could not be determined"]},
                ],
                "protocol": "torrent",
                "downloadClient": "qBittorrent",
                "episodeHasFile": False,
                "id": 9001,
            },
        )
    fmab = _world.roles["fmab"]
    done = complete_at(fmab)
    import_t0 = import_started_at(fmab)
    if done is not None and now() >= done + QUEUE_LAG_S and (import_t0 is None or now() < import_t0):
        records.append(
            {
                "downloadId": fmab.infohash.upper(),
                "trackedDownloadState": "importPending",
                "trackedDownloadStatus": "ok",
                "status": "completed",
                "title": fmab.name,
                "size": fmab.size,
                "sizeleft": 0,
                "statusMessages": [],
                "protocol": "torrent",
                "downloadClient": "qBittorrent",
                "episodeHasFile": False,
                "id": 9002,
            },
        )
    return records


def parse_title(title: str) -> list[Json]:
    m = SXXEYY.search(title)
    if m:
        return [{"seasonNumber": int(m.group(1)), "episodeNumber": int(m.group(2))}]
    m = BEBOP_ABS.search(title)
    if m:
        return [{"seasonNumber": 1, "episodeNumber": int(m.group(1))}]
    return []


def manual_candidates(download_id: str) -> list[Json]:
    role = _world.by_hash(download_id)
    if role is None:
        return []
    folder = f"/downloads/{role.name}"
    out: list[Json] = []
    for f in role.files:
        if not f.name.endswith(".mkv"):
            continue
        out.append(
            {
                "path": f"{folder}/{f.name}",
                "relativePath": f.name,
                "name": f.name,
                "size": f.size,
                "downloadId": role.infohash.upper(),
                "quality": role.quality.payload(),
                "languages": JP_EN,
                "rejections": [],
                "id": abs(hash(f.name)) % 10**8,
            },
        )
    return out


class QuietHandler(BaseHTTPRequestHandler):
    """Shared base: JSON responses, and no per-request stderr spam."""

    def _send_json(self, body: Json, status: int = 200) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @override
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Quiet the happy path; errors still land in mocks.log via log_error."""


class SonarrHandler(QuietHandler):
    """The slice of the Sonarr v3 API a demo run touches."""

    # Rough real-world latencies so the boot steps don't all read 0.00s.
    LATENCY: ClassVar[dict[str, float]] = {
        "/api/v3/series": 0.30,
        "/api/v3/episode": 0.35,
        "/api/v3/history/since": 0.10,
        "/api/v3/qualitydefinition": 0.05,
        "/api/v3/language": 0.05,
    }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)
        time.sleep(self.LATENCY.get(route, 0.0))
        if route == "/api/v3/series":
            self._send_json(load_json(DEMO_FIXTURES / "series_demo.json"))
        elif route == "/api/v3/episode":
            self._send_json(episodes_response(query.get("seriesId", ["?"])[0]))
        elif route == "/api/v3/history/since":
            self._send_json([])
        elif route == "/api/v3/qualitydefinition":
            self._send_json(load_json(FIXTURES / "qualitydefinitions.json"))
        elif route == "/api/v3/language":
            self._send_json([])
        elif route == "/api/v3/remotepathmapping":
            # No mappings: the folder-scan fallback's once-per-run fetch must
            # never warn in an offline run.
            self._send_json([])
        elif route == "/api/v3/parse":
            title = query.get("title", [""])[0]
            self._send_json({"title": title, "episodes": parse_title(title)})
        elif route == "/api/v3/queue":
            records = queue_records()
            self._send_json({"page": 1, "pageSize": 1000, "totalRecords": len(records), "records": records})
        elif route == "/api/v3/manualimport":
            self._send_json(manual_candidates(query.get("downloadId", [""])[0]))
        elif re.fullmatch(r"/api/v3/command/\d+", route):
            cmd_id = int(route.rsplit("/", 1)[1])
            cmd = _world.state.manual_command()
            if cmd and cmd[0] == cmd_id:
                status = "completed" if now() - cmd[1] >= CMD_RUN_S else "started"
                self._send_json({"id": cmd_id, "name": "ManualImport", "status": status})
            else:
                self._send_json({"id": cmd_id, "name": "RefreshMonitoredDownloads", "status": "completed"})
        elif route == "/api/v3/command":
            cmd = _world.state.manual_command()
            body: list[Json] = []
            if cmd and now() - cmd[1] < CMD_RUN_S:
                body.append({"id": cmd[0], "name": "ManualImport", "status": "started"})
            self._send_json(body)
        else:
            sys.stderr.write(f"SONARR UNEXPECTED GET: {self.path}\n")
            self._send_json({}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        route = urlparse(self.path).path
        if route == "/api/v3/command":
            parsed: object = json.loads(raw or b"{}")
            body = parsed if is_json_obj(parsed) else {}
            name = body.get("name", "")
            name = name if isinstance(name, str) else ""
            cmd_id = _world.state.next_command(manual=name.casefold() == "manualimport")
            self._send_json({"id": cmd_id, "name": name, "status": "queued"}, 201)
        else:
            sys.stderr.write(f"SONARR UNEXPECTED POST: {self.path}\n")
            self._send_json({}, 404)


class QbitHandler(QuietHandler):
    """The slice of the qBittorrent WebUI API a demo run touches."""

    def _send_text(self, text: str, status: int = 200, cookie: str | None = None) -> None:
        payload = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _info_rows(self, hashes_param: str | None) -> list[Json]:
        wanted = [h.casefold() for h in hashes_param.split("|")] if hashes_param else None
        rows: list[Json] = []
        for infohash, t0 in _world.state.added_snapshot().items():
            if wanted is not None and infohash.casefold() not in wanted:
                continue
            role = _world.by_hash(infohash)
            if role is None:
                continue
            progress = min(1.0, (now() - t0) / role.duration)
            size = role.size
            speed = int(size / role.duration) if progress < 1.0 else 0
            rows.append(
                {
                    "hash": infohash,
                    "name": role.name,
                    "size": size,
                    "total_size": size,
                    "progress": progress,
                    "dlspeed": speed,
                    "eta": int((1.0 - progress) * role.duration) if progress < 1.0 else 8640000,
                    "completed": int(progress * size),
                    "state": "downloading" if progress < 1.0 else "stalledUP",
                    "content_path": f"/downloads/{role.name}",
                    "save_path": "/downloads",
                    "category": "anime",
                    "tags": "pearlarr",
                    "added_on": int(time.time()),
                    "amount_left": int((1.0 - progress) * size),
                },
            )
        return rows

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)
        if route == "/api/v2/app/webapiVersion":
            self._send_text("2.9.3")
        elif route == "/api/v2/app/version":
            self._send_text("v4.6.7")
        elif route == "/api/v2/torrents/info":
            self._send_json(self._info_rows(query.get("hashes", [None])[0]))
        elif route == "/api/v2/app/preferences":
            self._send_json({"save_path": "/downloads"})
        elif route == "/api/v2/transfer/info":
            self._send_json({"connection_status": "connected", "dl_info_speed": 0, "up_info_speed": 0})
        else:
            sys.stderr.write(f"QBIT UNEXPECTED GET: {self.path}\n")
            self._send_json({})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        route = urlparse(self.path).path
        body = parse_qs(raw.decode("utf-8", "replace"))
        if route == "/api/v2/auth/login":
            time.sleep(0.65)  # a realistic WebUI login round-trip
            self._send_text("Ok.", cookie="SID=demodemodemo; path=/")
        elif route == "/api/v2/auth/logout":
            self._send_text("")
        elif route == "/api/v2/torrents/info":
            self._send_json(self._info_rows(body.get("hashes", [None])[0]))
        elif route == "/api/v2/torrents/add":
            url = body.get("urls", [""])[0]
            role: Role | None = None
            m = BTIH.search(url)
            if m:
                role = _world.by_hash(m.group(1))
            if role is None:
                m = NYAA_ID.search(url)
                if m:
                    view_id = m.group(1)
                    role = next((r for r in _world.roles.values() if r.view_id == view_id), None)
            if role is None:
                sys.stderr.write(f"QBIT ADD UNMATCHED: {url[:120]}\n")
            else:
                _world.state.record_add(role.infohash)
                sys.stdout.write(f"qbit: added {role.name[:50]}...\n")
                sys.stdout.flush()
            self._send_text("Ok.")
        else:
            sys.stderr.write(f"QBIT UNEXPECTED POST: {self.path} body={raw[:120]!r}\n")
            self._send_text("")


def build_world() -> World:
    """Fetch the two roles' live SeaDex torrents and load the episode fixtures."""

    sys.stdout.write("fetching infohashes from SeaDex...\n")
    sys.stdout.flush()
    seadex_client = SeaDexEntry()
    roles = {
        "frieren": build_role(
            seadex_client,
            154587,
            "PMR",
            name="Frieren Beyond Journey's End S01 (BD Remux 1080p AVC FLAC AAC) [Dual Audio] [PMR]",
            duration=FRIEREN_DL_S,
            series_key="301",
            quality=QualityStub(20, "Bluray-1080p Remux", "blurayRaw", 1080),
        ),
        "fmab": build_role(
            seadex_client,
            5114,
            "McBalls",
            name="[McBalls] Fullmetal Alchemist Brotherhood (BD 1080p HEVC Opus) [Dual Audio]",
            duration=FMAB_DL_S,
            series_key="303",
            quality=QualityStub(7, "Bluray-1080p", "bluray", 1080),
        ),
    }
    for key, role in roles.items():
        sys.stdout.write(f"{key}: hash={role.infohash} view={role.view_id}\n")
    episodes = {
        "301": load_object_rows(DEMO_FIXTURES / "episodes_301_frieren.json"),
        "302": load_object_rows(DEMO_FIXTURES / "episodes_302_bebop.json"),
        "303": load_object_rows(DEMO_FIXTURES / "episodes_303_fmab.json"),
    }
    return World(roles=roles, episodes=episodes, state=MockState())


def main() -> None:
    global _world
    _world = build_world()
    sonarr = ThreadingHTTPServer(("127.0.0.1", 8989), SonarrHandler)
    qbit = ThreadingHTTPServer(("127.0.0.1", 8080), QbitHandler)
    threading.Thread(target=sonarr.serve_forever, daemon=True).start()
    sys.stdout.write("mock sonarr on :8989, mock qbit on :8080\n")
    sys.stdout.flush()
    qbit.serve_forever()


if __name__ == "__main__":
    main()
