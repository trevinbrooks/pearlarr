# pyright: strict
# pyright: reportPrivateUsage=false
# These assert on the loop/services/strategy's private wiring (_filter / _state / _parse /
# _reconciler), which strict re-flags. The repo disables reportPrivateUsage for tests.
"""How an arr run is put together, seen through what its runs do.

Bootstrap builds a run's parts once, and the loop hands each of them the run's fresh state as the run starts: what a
title's check marks shows in the run's summary, and what it grabs is counted once and waited on by the run's own
wait. A Radarr run lists every movie a mapping source knows.

`ArrRunDeps.build` sets up what the parts share. The arr client verifies certificates as the arr's `verify_ssl` says.
qBittorrent is reached with the config's options, and a grab goes to it under the arr's category with the config's
tags. A category left to the arr's download client is read when it is first used, never while the run boots.
"""

import logging
from pathlib import Path
from typing import override

import httpx
import pytest
import qbittorrentapi
import respx
from seadex import Tracker

from pearlarr import arr_run_services
from pearlarr.arr_http import REWARN_INTERVAL_S
from pearlarr.arr_models import RadarrMovie
from pearlarr.arr_run_services import ArrRunDeps, ArrRunServices
from pearlarr.boot_flow import BootFlow
from pearlarr.config import AppConfig, Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import ArrRunSummaryReady, NeedsActionCause, RowFinished
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import ImportWaitMode, Outcome

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    attempt_result,
    drive_arr_run,
    grab_request,
    make_config,
    make_entry_record,
    make_resolver,
    make_run_deps,
    make_torrent_record,
    one_release_dict,
    pending_import_update,
)
from .fakes import FakeArrItem, FakeRadarrClient, FakeSonarrClient, install_recording_hub
from .http_mock import serve_nyaa_torrents
from .wait_fakes import ALL_IN, FakeQbit, FakeTorrent, QbitStep, RecordingStrategy

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


_HASH = "ab" * 20
"""The info hash of the Nyaa release the grabs here add."""


class _ListsWhatItAdded(FakeQbit):
    """qBittorrent holding nothing until it is asked to add a torrent, and from then on holding it finished."""

    def __init__(self) -> None:
        finished: list[QbitStep] = [FakeTorrent(is_complete=True, content_path="/downloads/done")]
        super().__init__({_HASH: finished})

    @override
    def torrents_info(self, *, torrent_hashes: str | list[str]) -> list[FakeTorrent]:
        return super().torrents_info(torrent_hashes=torrent_hashes) if self.added else []


class _GrabsOneRelease(RecordingStrategy):
    """Sonarr's side of a run over series 7 under title 1, whose check grabs one Nyaa release.

    Sonarr imports every file of it when asked.
    """

    def __init__(self, services: ArrRunServices) -> None:
        super().__init__(attempts=(attempt_result(),), progress=(ALL_IN,))
        self._services = services

    @override
    def get_items(self) -> list[FakeArrItem]:
        return [FakeArrItem(item_id=7)]

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {1: TitleMapping(anilist_id=1)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        entry = self._services.begin_check(anilist_id)
        assert entry is not None
        self._services.grab_and_cache(
            grab_request(
                anilist_id=anilist_id,
                entry=entry,
                seadex_dict=one_release_dict(srg="NAN0", infohash=_HASH),
                pending_import_updates={_HASH: pending_import_update(_HASH)},
            )
        )


@respx.mock
def test_a_blocking_run_counts_its_grab_once_and_imports_it() -> None:
    # The grab enters the torrent in the run's list of what it grabbed, which the summary counts and the wait reads.
    serve_nyaa_torrents(1)
    deps = make_run_deps(qbit=_ListsWhatItAdded().client(), seadex=FakeSeaDexSource({1: make_entry_record()}))
    recording = install_recording_hub()

    drive_arr_run(deps, _GrabsOneRelease, import_wait_mode=ImportWaitMode.BLOCKING)

    (summary,) = recording.of_type(ArrRunSummaryReady)
    tally = summary.summary.tally
    # Counted as this run's grab, never also as a download an earlier run left.
    assert ([row.group for row in tally.grabbed], tally.downloading) == (["NAN0"], 0)
    assert [(row.label, row.outcome) for row in recording.of_type(RowFinished)] == [("Show · NAN0", Outcome.IMPORTED)]


def test_services_read_selection_change_from_the_store_at_construction() -> None:
    # The real __init__ compares the store's selection hash against the config's:
    # an older hash turns on the arr-run-wide re-check. A matching hash or a fresh
    # store (no hash stored yet) stays quiet.
    config = make_config()

    vouched_old = FakeCacheStore()
    vouched_old.store_selection_hash(Arr.SONARR, "a-previous-digest")
    assert ArrRunServices(make_run_deps(config=config, cache_store=vouched_old), Arr.SONARR).selection_changed is True

    vouched_current = FakeCacheStore()
    vouched_current.store_selection_hash(Arr.SONARR, config.selection_hash())
    assert (
        ArrRunServices(make_run_deps(config=config, cache_store=vouched_current), Arr.SONARR).selection_changed is False
    )

    assert ArrRunServices(make_run_deps(config=config), Arr.SONARR).selection_changed is False


def test_sonarr_strategy_init_shares_cache_store_for_staged_writes() -> None:
    # The parse-cache writer (SonarrParseCache) and the pending import update builder (ImportReconciler,
    # through the services' pending import store) must share the strat's cache_store by identity, or
    # a staged parse write would not be visible to build_pending_import_updates.
    deps = make_run_deps()
    services = ArrRunServices(deps, Arr.SONARR)
    # Inject a typed fake through the sonarr_client argument so the collaborator wiring
    # runs off the REAL __init__ without connection keys - an incomplete fake here
    # is a pyright error and un-instantiable, unlike the old stringly-typed
    # monkeypatch.
    strat = SonarrStrategy(deps, services, sonarr_client=FakeSonarrClient())

    assert strat._parse.cache_store is deps.cache_store
    assert strat._reconciler._pending_imports is services.pending_imports
    # `PendingImportStore._store` is private, and this identity is the invariant the comment above names.
    assert services.pending_imports._store is deps.cache_store


def test_radarr_strategy_init_builds_without_network_via_the_client_argument() -> None:
    # The radarr_client argument lets RadarrStrategy's REAL __init__ run under test: the
    # injected fake skips require_connection, so no config keys are needed.
    deps = make_run_deps()
    services = ArrRunServices(deps, Arr.RADARR)
    fake = FakeRadarrClient()

    strat = RadarrStrategy(deps, services, radarr_client=fake)

    assert strat.radarr is fake
    assert strat._resolver is deps.resolver
    assert strat.anibridge is deps.resolver.anibridge


# --- what ArrRunDeps.build sets up ------------------------------------------------------------------------------


def _build_deps(app_config: AppConfig, tmp_path: Path) -> ArrRunDeps:
    """`ArrRunDeps.build` over the throwaway wiring every construction test shares."""

    return ArrRunDeps.build(
        Arr.SONARR,
        cache=str(tmp_path / "cache.db"),
        logger=logging.getLogger("pearlarr.test"),
        resolver=make_resolver(),
        app_config=app_config,
        web=httpx.Client(),
        boot=BootFlow(),
    )


def test_arr_run_deps_build_pins_verify_ssl_to_the_arrs_knob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`ArrRunDeps.build` constructs the arr run's httpx client with THIS arr's `verify_ssl`.

    It's the per-arr escape hatch for a self-signed HTTPS arr.
    """

    seen: list[bool] = []
    real_factory = arr_run_services.make_arr_client

    def _record(*, verify: bool = True) -> httpx.Client:
        seen.append(verify)
        return real_factory(verify=verify)

    monkeypatch.setattr(arr_run_services, "make_arr_client", _record)
    deps = _build_deps(make_config(verify_ssl=False), tmp_path)
    deps.close()

    assert seen == [False]


class _PassingQbit:
    """A fake `qbittorrentapi.Client` whose login succeeds (build's qBittorrent client)."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def auth_log_in(self) -> None:
        pass


@respx.mock
def test_arr_run_deps_build_wires_lazy_categories_fetched_at_first_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With qbit configured and omitted categories, `build` fetches NOTHING - the first use does, once."""

    monkeypatch.setattr(qbittorrentapi, "Client", _PassingQbit)
    route = respx.get("http://sonarr/api/v3/downloadclient").respond(
        json=[
            {
                "enable": True,
                "implementation": "QBittorrent",
                "fields": [
                    {"name": "tvCategory", "value": "tv-sonarr"},
                    {"name": "tvImportedCategory", "value": "sonarr-done"},
                ],
            },
        ],
    )
    config = make_config(url="http://sonarr", api_key="key", host="http://qbit:8080", username="u", password="p")

    deps = _build_deps(config, tmp_path)
    try:
        # Boot did no category I/O: an arr restarting under the boot must not
        # cost the arr run its post-import moves (the use-time fetch retries).
        assert route.call_count == 0
        assert deps.categories.grab_category() == "tv-sonarr"
        assert deps.categories.post_import_category() == "sonarr-done"
        assert route.call_count == 1  # memoized after the one success
        # The torrent adapter asks the same resolver at add time.
        assert deps.torrents.categories is deps.categories
    finally:
        deps.close()


@respx.mock
def test_arr_run_deps_build_skips_the_category_fetch_without_qbit(tmp_path: Path) -> None:
    """No qBittorrent client (perpetual dry run): no category is ever applied, so `build` never contacts the arr."""

    route = respx.get("http://sonarr/api/v3/downloadclient").respond(json=[])
    config = make_config(url="http://sonarr", api_key="key")

    deps = _build_deps(config, tmp_path)
    deps.close()

    # No transport was bound, so even USING the resolver stays fetchless.
    assert deps.categories.grab_category() is None
    assert deps.categories.post_import_category() is None
    assert route.call_count == 0


def test_arr_run_deps_build_binds_arr_http_when_configured(tmp_path: Path) -> None:
    """`build` binds this arr's transport once: label from the arr, default rewarn interval."""

    deps = _build_deps(make_config(url="http://sonarr", api_key="key"), tmp_path)
    deps.close()

    assert deps.arr_http is not None
    assert deps.arr_http.label == "Sonarr"
    assert deps.arr_http.base_url == "http://sonarr"
    # progress_line_interval is the Sonarr client's own replace, never build's bind.
    assert deps.arr_http.rewarn_s == REWARN_INTERVAL_S


def test_arr_run_deps_build_leaves_arr_http_none_without_connection_keys(tmp_path: Path) -> None:
    """Missing keys: `build` stays raise-free - require_connection fires at strategy construction instead."""

    deps = _build_deps(make_config(), tmp_path)
    deps.close()

    assert deps.arr_http is None
    # The resolver's None-http contract: no transport, so reads stay fetchless blanks.
    assert deps.categories.grab_category() is None


def test_arr_run_deps_build_category_handle_derives_from_arr_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The category handle is the arr transport's no-retry derivation - one shared set of failure streaks."""

    monkeypatch.setattr(qbittorrentapi, "Client", _PassingQbit)
    config = make_config(url="http://sonarr", api_key="key", host="http://qbit:8080", username="u", password="p")

    deps = _build_deps(config, tmp_path)
    deps.close()

    assert deps.arr_http is not None
    sources = deps.categories._sources
    assert sources is not None
    assert sources.http.retries == 0
    assert sources.http.streaks is deps.arr_http.streaks
