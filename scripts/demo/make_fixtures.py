"""Build the demo Sonarr fixtures: Frieren (missing), Cowboy Bebop (up to date), FMAB (upgrade).

Maintainer tooling behind `scripts/demo/record.sh`, which runs this only when
`fixtures/` is empty; the generated files are committed so re-records don't
drift with SeaDex. Needs network - the Bebop file sizes come off the live entry.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from seadex import SeaDexEntry, Tracker

from pearlarr.json_narrow import is_json_list, is_json_obj
from pearlarr.seadex_types import Json

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "sonarr"
OUT = Path(__file__).resolve().parent / "fixtures"

BLURAY1080: Json = {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}
WEB720: Json = {"id": 5, "name": "WEBDL-720p", "source": "web", "resolution": 720}
JP_EN: Json = [{"id": 8, "name": "Japanese"}, {"id": 1, "name": "English"}]

BEBOP_FILE = re.compile(r"\[JySzE\] Cowboy Bebop - (\d{2}) \[v3\]\.mkv")


@dataclass(frozen=True)
class SeriesSpec:
    """One demo series row for series_demo.json."""

    series_id: int
    title: str
    sort_title: str
    tvdb_id: int
    year: int
    slug: str
    episode_count: int
    status: str


SERIES = (
    SeriesSpec(
        301, "Frieren: Beyond Journey's End", "frieren beyond journeys end", 424536, 2023, "frieren", 28, "ended"
    ),
    SeriesSpec(302, "Cowboy Bebop", "cowboy bebop", 76885, 1998, "cowboy-bebop", 26, "ended"),
    SeriesSpec(
        303, "Fullmetal Alchemist: Brotherhood", "fullmetal alchemist brotherhood", 116391, 2009, "fmab", 64, "ended"
    ),
)


@dataclass(frozen=True)
class EpFileSpec:
    """The varying bits of an on-disk episodeFile payload."""

    name: str
    group: str
    quality: Json
    size: int
    path_root: str


def series_template() -> dict[str, Json]:
    """The real series row (id 228) the demo rows are cloned from."""

    raw: object = json.loads((FIXTURES / "series_subset.json").read_text(encoding="utf-8"))
    if not is_json_list(raw):
        raise SystemExit("series_subset.json is not a JSON array")
    for item in raw:
        if is_json_obj(item) and item.get("id") == 228:
            return item
    raise SystemExit("series 228 template missing from series_subset.json")


def series_row(template: dict[str, Json], spec: SeriesSpec) -> dict[str, Json]:
    s = copy.deepcopy(template)
    s["id"] = spec.series_id
    s["title"] = spec.title
    s["sortTitle"] = spec.sort_title
    s["tvdbId"] = spec.tvdb_id
    s["year"] = spec.year
    s["titleSlug"] = spec.slug
    s["path"] = f"/tv/{spec.title.replace(':', ' -')}"
    s["monitored"] = True
    s["status"] = spec.status
    missing = spec.series_id == 301
    seasons: list[Json] = [
        {
            "seasonNumber": 1,
            "monitored": True,
            "statistics": {
                "episodeFileCount": 0 if missing else spec.episode_count,
                "episodeCount": spec.episode_count,
                "totalEpisodeCount": spec.episode_count,
                "sizeOnDisk": 0,
                "releaseGroups": [],
                "percentOfEpisodes": 0.0 if missing else 100.0,
            },
        },
    ]
    s["seasons"] = seasons
    return s


def episode(sid: int, n: int, file_payload: Json | None) -> dict[str, Json]:
    ep: dict[str, Json] = {
        "seriesId": sid,
        "tvdbId": sid * 10000 + n,
        "episodeFileId": sid * 100000 + n if file_payload is not None else 0,
        "seasonNumber": 1,
        "episodeNumber": n,
        "title": f"Episode {n}",
        "airDate": "2020-01-01",
        "airDateUtc": "2020-01-01T15:00:00Z",
        "runtime": 24,
        "overview": "",
        "hasFile": file_payload is not None,
        "monitored": True,
        "unverifiedSceneNumbering": False,
        "id": sid * 1000 + n,
    }
    if file_payload is not None:
        ep["episodeFile"] = file_payload
    return ep


def ep_file(sid: int, n: int, spec: EpFileSpec) -> dict[str, Json]:
    return {
        "seriesId": sid,
        "seasonNumber": 1,
        "relativePath": f"Season 01/{spec.name}",
        "path": f"{spec.path_root}/Season 01/{spec.name}",
        "size": spec.size,
        "dateAdded": "2024-01-01T00:00:00Z",
        "releaseGroup": spec.group,
        "languages": JP_EN,
        "quality": {"quality": spec.quality, "revision": {"version": 1, "real": 0, "isRepack": False}},
        "customFormats": [],
        "id": sid * 100000 + n,
    }


def bebop_episodes() -> list[Json]:
    """The JySzE Bluray set (SeaDex's best) already on disk.

    Uses the REAL per-file sizes from the live SeaDex file list - the ownership
    check compares sizes, so fabricated ones read as an outdated copy and
    trigger a re-grab.
    """

    entry = SeaDexEntry().from_id(1)
    specs: dict[int, EpFileSpec] = {}
    for torrent in entry.torrents:
        if torrent.tracker is not Tracker.NYAA or torrent.release_group != "JySzE":
            continue
        for f in torrent.files:
            m = BEBOP_FILE.fullmatch(f.name)
            if m and 1 <= int(m.group(1)) <= 26:
                specs[int(m.group(1))] = EpFileSpec(
                    name=f.name, group="JySzE", quality=BLURAY1080, size=f.size, path_root="/tv/Cowboy Bebop"
                )
    if len(specs) != 26:
        raise SystemExit(f"expected 26 Bebop episode files, got {len(specs)}")
    return [episode(302, n, ep_file(302, n, spec)) for n, spec in sorted(specs.items())]


def write_fixture(name: str, body: Json) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(body, indent=1), encoding="utf-8")


def main() -> None:
    template = series_template()
    write_fixture("series_demo.json", [series_row(template, spec) for spec in SERIES])

    # Frieren: 28 monitored episodes, nothing on disk.
    write_fixture("episodes_301_frieren.json", [episode(301, n, None) for n in range(1, 29)])

    write_fixture("episodes_302_bebop.json", bebop_episodes())

    # FMAB: full Erai-raws WEB 720p set - outdated next to the SeaDex pick.
    fmab = [
        episode(
            303,
            n,
            ep_file(
                303,
                n,
                EpFileSpec(
                    name=f"[Erai-raws] Fullmetal Alchemist Brotherhood - {n:02d} [720p].mkv",
                    group="Erai-raws",
                    quality=WEB720,
                    size=350_000_000 + n,
                    path_root="/tv/Fullmetal Alchemist - Brotherhood",
                ),
            ),
        )
        for n in range(1, 65)
    ]
    write_fixture("episodes_303_fmab.json", fmab)

    sys.stdout.write(f"wrote {sorted(p.name for p in OUT.iterdir())}\n")


if __name__ == "__main__":
    main()
