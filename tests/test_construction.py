# pyright: strict
"""How an arr run is put together, seen through what its runs do.

Bootstrap builds a run's parts once, and the loop hands each of them the run's fresh state as the run starts, so what
a title's check marks shows in the run's summary. A Radarr run checks every movie a mapping source knows.

`ArrRunDeps.build` sets up what the parts share. The arr client verifies certificates as the arr's `verify_ssl` says.
qBittorrent is reached with the config's options, and a grab goes to it under the arr's category with the config's
tags. A category left to the arr's download client is read when it is first used, never while the run boots.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import qbittorrentapi
import respx
from seadex import Tracker

from pearlarr import arr_run_services
from pearlarr.arr_http import ARR_REQUEST_TIMEOUT_S
from pearlarr.arr_models import RadarrMovie
from pearlarr.arr_run_services import ArrRunDeps
from pearlarr.boot_flow import BootFlow
from pearlarr.config import AppConfig, Arr
from pearlarr.mappings import MappingResolver, MappingSources
from pearlarr.output import ArrRunSummaryReady, Diagnostic, NeedsActionCause
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.releases import Release

from .builders import (
    FakeSeaDexSource,
    download_client_json,
    drive_arr_run,
    make_config,
    make_entry_record,
    make_resolver,
    make_run_deps,
    make_torrent_record,
    sonarr_client_fields,
)
from .fakes import FakeRadarrClient, diagnostic_messages, install_recording_hub
from .http_mock import serve_nyaa_torrents
from .wait_fakes import FakeTorrent

# --- a run's parts act on the run's state -----------------------------------------------------------------------


def _radarr_run(deps: ArrRunDeps, radarr: FakeRadarrClient) -> None:
    """A Radarr run over `radarr`'s library, its strategy built as bootstrap builds it."""

    drive_arr_run(deps, lambda services: RadarrStrategy(deps, services, radarr_client=radarr), arr=Arr.RADARR)


_YOUR_NAME = RadarrMovie(id=42, title="Your Name.", tmdbId=372058)
"""A movie in Radarr, AniList title 21519."""


def test_a_radarr_run_lists_the_private_release_it_cannot_grab_as_needing_action() -> None:
    # The release is marked private-only where SeaDex's entry is planned and listed where the title's check ends, two
    # parts of the run that each read the run's state.
    entry = make_entry_record(
        anilist_id=21519,
        torrents=(
            make_torrent_record(
                release_group="Priv",
                tracker=Tracker.ANIMEBYTES,
                url="https://animebytes.tv/torrents.php?id=1",
                infohash=None,
                file_names=("Your Name (2016) [BD 1080p].mkv",),
            ),
        ),
    )
    deps = make_run_deps(
        arr=Arr.RADARR,
        seadex=FakeSeaDexSource({21519: entry}),
        anime_ids={"Your Name": {"anilist_id": 21519, "tmdb_movie_id": 372058}},
    )
    recording = install_recording_hub()

    _radarr_run(deps, FakeRadarrClient(movies=[_YOUR_NAME]))

    (summary,) = recording.of_type(ArrRunSummaryReady)
    assert [(row.group, row.cause) for row in summary.summary.tally.needs_action] == [
        ("Priv", NeedsActionCause.PRIVATE_ONLY)
    ]


def test_a_radarr_run_checks_a_movie_only_anibridge_maps() -> None:
    # The run asks SeaDex for each movie it lists, under the movie's AniList title.
    seadex = FakeSeaDexSource()
    anibridge = MappingSources(anime=False, anidb=False, anibridge={"anilist:21519": {"tmdb_movie:372058": {}}})
    deps = replace(
        make_run_deps(arr=Arr.RADARR, seadex=seadex),
        resolver=MappingResolver(refresh_days=1, ignore_anilist_ids=set(), sources=anibridge, web=httpx.Client()),
    )

    _radarr_run(deps, FakeRadarrClient(movies=[_YOUR_NAME]))

    assert seadex.entry_calls == [21519]


# --- what ArrRunDeps.build sets up ------------------------------------------------------------------------------


class _QbitStandIn:
    """The `qbittorrentapi.Client` build makes, standing in for qBittorrent: its login succeeds and it lists nothing.

    It keeps the settings it is built with, and the category and tags of each torrent it is asked to add.
    """

    def __init__(self, **settings: object) -> None:
        self.settings = settings
        self.added: list[tuple[str | None, list[str] | None]] = []

    def auth_log_in(self) -> None:
        pass

    def torrents_add(self, *, urls: str, category: str | None, tags: list[str] | None) -> str:
        del urls
        self.added.append((category, tags))
        return "Ok."

    def torrents_info(self, *, torrent_hashes: str) -> list[FakeTorrent]:
        del torrent_hashes
        return []


type _Build = Callable[[AppConfig], ArrRunDeps]
"""`ArrRunDeps.build` for a Sonarr run over a config."""


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_Build]:
    """`ArrRunDeps.build` as bootstrap calls it for a Sonarr run, each one built closed after the test.

    A qBittorrent it is configured with is a `_QbitStandIn`.
    """

    monkeypatch.setattr(qbittorrentapi, "Client", _QbitStandIn)
    built: list[ArrRunDeps] = []

    def _build(config: AppConfig) -> ArrRunDeps:
        deps = ArrRunDeps.build(
            Arr.SONARR,
            cache=str(tmp_path / "cache.db"),
            logger=logging.getLogger("pearlarr.test"),
            resolver=make_resolver(),
            app_config=config,
            web=httpx.Client(),
            boot=BootFlow(),
        )
        built.append(deps)
        return deps

    yield _build
    for deps in built:
        deps.close()


@pytest.mark.parametrize(
    ("config", "verified"),
    [
        pytest.param(make_config(), True, id="by-default"),
        pytest.param(make_config(verify_ssl=False), False, id="verify-ssl-off"),
    ],
)
def test_the_arr_client_verifies_certificates_as_the_arrs_verify_ssl_says(
    monkeypatch: pytest.MonkeyPatch, build: _Build, config: AppConfig, verified: bool
) -> None:
    # Off for an arr behind a self-signed certificate. httpx keeps a client's setting to itself, so it is read where
    # build asks for the client.
    asked: list[bool] = []
    real_factory = arr_run_services.make_arr_client

    def _record(*, verify: bool = True) -> httpx.Client:
        asked.append(verify)
        return real_factory(verify=verify)

    monkeypatch.setattr(arr_run_services, "make_arr_client", _record)

    build(config)

    assert asked == [verified]


_SONARR_AND_QBITTORRENT = make_config(
    url="http://sonarr", api_key="key", host="http://qbit:8080", username="u", password="p", tags=["seadex"]
)
"""Sonarr and qBittorrent configured, the categories left to Sonarr's download client and the torrents tagged."""

_DOWNLOAD_CLIENTS = "http://sonarr/api/v3/downloadclient"


class TestCategoriesLeftToTheArr:
    """A category left to the arr's download client is read when it is first used.

    Never while the run boots, so an arr restarting under the boot costs the run nothing. A read the arr fails is
    tried once and warns: that use goes without, and the next one asks again.
    """

    @respx.mock
    def test_an_arr_failing_while_the_run_boots_still_gives_the_first_use_its_category(self, build: _Build) -> None:
        route = respx.get(_DOWNLOAD_CLIENTS).respond(500)
        recording = install_recording_hub()
        deps = build(_SONARR_AND_QBITTORRENT)
        route.respond(json=[download_client_json(sonarr_client_fields())])

        assert deps.categories.post_import_category() == "sonarr-done"
        assert recording.of_type(Diagnostic) == []

    @respx.mock
    def test_a_read_the_arr_fails_warns_and_the_next_use_asks_again(self, build: _Build) -> None:
        respx.get(_DOWNLOAD_CLIENTS).side_effect = [
            httpx.Response(500),
            httpx.Response(200, json=[download_client_json(sonarr_client_fields())]),
        ]
        recording = install_recording_hub()
        deps = build(_SONARR_AND_QBITTORRENT)

        assert deps.categories.grab_category() is None
        assert diagnostic_messages(recording) == [
            (
                "Could not fetch the Sonarr download clients (status code 500) - blank categories stay blank, "
                "and post-import moves and imports from unwatched categories wait"
            ),
        ]
        assert deps.categories.grab_category() == "tv-sonarr"


@respx.mock
def test_a_grab_goes_to_qbittorrent_under_the_arrs_category_with_the_configs_tags(build: _Build) -> None:
    # The grab and the post-import move ask the one read of Sonarr's download clients.
    clients = respx.get(_DOWNLOAD_CLIENTS).respond(json=[download_client_json(sonarr_client_fields())])
    serve_nyaa_torrents(1)
    deps = build(_SONARR_AND_QBITTORRENT)
    assert isinstance(deps.qbit, _QbitStandIn)
    release = Release(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, infohash="ab" * 20)

    deps.torrents.add(item=release, dry_run=False)

    assert deps.qbit.added == [("tv-sonarr", ["seadex"])]
    assert (deps.categories.post_import_category(), clients.call_count) == ("sonarr-done", 1)


_ARR_TIMEOUT = {"REQUESTS_ARGS": {"timeout": ARR_REQUEST_TIMEOUT_S}}
"""The qBittorrent setting that times its requests out as the arr's are."""


@pytest.mark.parametrize(
    ("options", "settings"),
    [
        pytest.param({}, _ARR_TIMEOUT, id="by-default"),
        pytest.param(
            {"VERIFY_WEBUI_CERTIFICATE": False}, {"VERIFY_WEBUI_CERTIFICATE": False, **_ARR_TIMEOUT}, id="with-options"
        ),
        pytest.param({"REQUESTS_ARGS": {"timeout": 60}}, {"REQUESTS_ARGS": {"timeout": 60}}, id="own-timeout"),
    ],
)
def test_qbittorrent_is_reached_with_the_configs_options_timing_out_as_the_arr_unless_they_say(
    build: _Build, options: dict[str, object], settings: dict[str, object]
) -> None:
    # A wedged qBittorrent socket can't hang a poll.
    deps = build(make_config(host="http://qbit:8080", username="u", password="p", options=options))

    assert isinstance(deps.qbit, _QbitStandIn)
    assert deps.qbit.settings == {"host": "http://qbit:8080", "username": "u", "password": "p", **settings}
