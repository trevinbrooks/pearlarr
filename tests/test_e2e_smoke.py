# pyright: strict
"""End-to-end smoke test: one full Sonarr pass through the REAL composition root.

This is the single test that proves ``cli.run_single`` -> ``_run_arrs`` ->
``RunDeps.build`` -> ``SeaDexArr.run_sync`` -> ``SonarrSync`` hooks actually run a
sync wired together, with ONLY the external network leaves faked:

* the SeaDex library, faked at the gateway's httpx boundary (``SeaDexEntry``);
* qBittorrent, left unconfigured so the whole run is a perpetual preview;
* the Sonarr + AniList HTTP, mocked at the ``requests`` boundary via ``responses``;
* the Nyaa source (``pynyaa``/httpx, which ``responses`` can't intercept), faked
  at ``torrents.get_nyaa_torrent``.

Everything in between is the real wiring. The id flows by hand-wired three-way
agreement: the captured ``series`` fixture's ``tvdbId`` (299502) -> an inline
``anime_mappings`` entry -> AniList id 20920 -> the faked SeaDex entry. The fakes
record, so the assertions prove that id actually flowed end to end (a vacuous run
that resolved nothing would still return True - the recorded calls are what make
this non-hollow).
"""

import logging
from pathlib import Path

import pytest
import responses
import yaml
from seadex import EntryRecord

import seadexarr.modules.seadex_gateway as seadex_gateway
import seadexarr.modules.torrents as torrents
from seadexarr.modules.cli import run_single
from seadexarr.modules.log import LogCounter
from seadexarr.modules.manual_import import ImportWaitMode

from .builders import make_config, make_entry_record, make_torrent_record
from .http_mock import register_sonarr_reads, sonarr_fixture

# Three-way id agreement: series fixture id 228 carries tvdbId 299502; the inline
# mapping ties it to AniList 20920; the faked SeaDex entry answers for 20920.
_TVDB = 299502
_ANILIST = 20920
_BASE = "http://sonarr.test/api/v3"
_ANILIST_URL = "https://graphql.anilist.co"
_NYAA_RELEASE_URL = "https://nyaa.si/1"

# A minimal valid AniList batch body (data.Page.media) so the real title-prefetch
# succeeds in one request instead of retrying a blocked endpoint.
_ANILIST_BODY: dict[str, object] = {
    "data": {
        "Page": {
            "media": [
                {
                    "id": _ANILIST,
                    "title": {"romaji": "Undefeated Bahamut Chronicle", "english": None, "native": None},
                    "episodes": 12,
                    "format": "TV",
                    "status": "FINISHED",
                    "coverImage": {"large": None},
                    "siteUrl": None,
                },
            ],
        },
    },
}


def test_sonarr_run_drives_real_composition_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The SeaDex entry the resolved id maps to: one grabbable Nyaa release whose
    # single file the Sonarr matching will parse (driving the real /parse adapter).
    entry = make_entry_record(
        anilist_id=_ANILIST,
        torrents=(
            make_torrent_record(
                url=_NYAA_RELEASE_URL,
                file_names=("Undefeated Bahamut Chronicle - S01E01 [1080p].mkv",),
            ),
        ),
    )

    # Fake the two external leaves; both record so the resolved id's flow is provable.
    filter_calls: list[str] = []
    nyaa_calls: list[str] = []

    class _FakeSeaDexEntry:
        """Stand-in for the SeaDex lib's ``SeaDexEntry`` (the gateway's httpx leaf)."""

        def __init__(self) -> None: ...

        def from_filter(self, query: str) -> list[EntryRecord]:
            filter_calls.append(query)
            return [entry]

        def from_id(self, al_id: int) -> EntryRecord:
            del al_id
            return entry

    def _fake_get_nyaa_torrent(url: str) -> tuple[str, str]:
        nyaa_calls.append(url)
        return ("magnet:?xt=urn:btih:" + "a" * 40, "Undefeated Bahamut Chronicle - S01E01 [1080p]")

    monkeypatch.setattr(seadex_gateway, "SeaDexEntry", _FakeSeaDexEntry)
    monkeypatch.setattr(torrents, "get_nyaa_torrent", _fake_get_nyaa_torrent)

    # A real config.yml on disk (run_single reads it via resolve_paths): Sonarr creds,
    # qBittorrent unset -> preview, and the one inline tvdb->anilist mapping that lets
    # the REAL resolver resolve a live id with no network (anidb/anibridge disabled).
    monkeypatch.setenv("SEADEX_ARR_DATA_DIR", str(tmp_path))
    config = make_config(
        url="http://sonarr.test",
        api_key="testkey",
        anime_mappings={"Bahamut": {"anilist_id": _ANILIST, "tvdb_id": _TVDB}},
        anidb_mappings=False,
        anibridge_mappings=False,
    )
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config.model_dump(mode="json")))

    # The Sonarr + AniList HTTP boundary. responses patches the requests adapter
    # globally, so both the shared Session and arrapi's own Session are intercepted.
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        register_sonarr_reads(
            rsps,
            _BASE,
            series=sonarr_fixture("series_subset.json"),
            episodes=sonarr_fixture("episodes_228_bahamut.json"),
            parse=sonarr_fixture("parse_bahamut_s01e01.json"),
        )
        rsps.add(responses.POST, _ANILIST_URL, json=_ANILIST_BODY)

        result = run_single(sonarr=True, import_wait_mode=ImportWaitMode.OFF)

        fired = {f"{call.request.method or ''} {(call.request.url or '').split('?')[0]}" for call in rsps.calls}

    # The real composition root ran one full Sonarr pass with zero real network.
    assert result is True
    # The inline tvdb->anilist mapping resolved id 20920 and the gateway was consulted
    # for it - the anti-vacuity guard: a run that resolved nothing never gets here.
    assert any(str(_ANILIST) in query for query in filter_calls)
    # The real Sonarr adapters drove the library fetch + per-file parse over the wire.
    assert f"GET {_BASE}/series" in fired
    assert f"GET {_BASE}/episode" in fired
    assert f"GET {_BASE}/parse" in fired
    # The resolved entry's release reached the (preview) grab at the torrent source.
    assert nyaa_calls == [_NYAA_RELEASE_URL]
    # ...and the whole pass logged no error (a swallowed failure would tally here).
    counter = getattr(logging.getLogger("SeaDexArr"), "seadex_counter", None)
    assert isinstance(counter, LogCounter)
    assert counter.counts.get(logging.ERROR, 0) == 0
    assert counter.counts.get(logging.CRITICAL, 0) == 0
