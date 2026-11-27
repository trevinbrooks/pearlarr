# pyright: strict
"""Tests for a Sonarr title's check, up to the grab request it hands the run.

They cover the stops before a grab, how a grab places each listed torrent and builds its pending import updates, the
stored pending imports a check follows first, and whether a grab's metadata was read. A series Sonarr numbers by
another order is in `test_sonarr_check_title_orders.py`.

A check runs over `FakeRunServices`, a real `ArrRunServices` subclass that records what the strategy hands the run
(the grab request, the rows it caches, the lines it prints).
"""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from pearlarr.arr_models import SonarrEpisode
from pearlarr.config import Arr
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import MappingSource, TitleMapping
from pearlarr.output import Severity
from pearlarr.pending_import import PendingImport
from pearlarr.releases import SeadexDict, SkipReason
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import AIRED_ORDER
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    claim_anilist_ids,
    make_config,
    make_entry_record,
    make_release,
    make_run_deps,
    make_torrent_record,
    numbered_names,
    parsed_info,
    pending_import,
    placed_episode,
    rg_group,
    sonarr_ep,
    two_claim_pending_import,
)
from .fakes import FakeSonarrClient, ScriptedEpisodes, diagnostic_messages, install_recording_hub
from .title_check_builders import (
    EP1,
    EP2,
    GRAB_FILES,
    GRAB_URL,
    SERIES_7,
    STAND_IN_E2,
    FakeRunServices,
    ScriptedListings,
    scripted_checking,
    scripted_grab_run,
    seadex_groups,
    sonarr_grab_run,
)

_STAMP = "2026-01-01 00:00:00"


class TestSonarrCheckTitleBeforeTheGrab:
    """A series' check stops before the grab when there is nothing to grab.

    No SeaDex entry, no episodes, only unmonitored episodes or no pick left stops the title. The window's missing
    episodes are shown.
    """

    def test_sonarr_title_with_no_seadex_entry_grabs_nothing_and_owes_no_pause(self) -> None:
        run = FakeRunServices()
        strat = scripted_checking(run)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))

        # The run reports the missing entry under the title's id.
        assert run.anilist_id_prologue_calls == [5]
        assert (run.grab_requests, run.pauses_owed) == ([], 0)

    @pytest.mark.parametrize(
        ("mapping", "warnings"),
        [
            pytest.param(TitleMapping(anilist_id=5), [], id="season-not-in-sonarr"),
            pytest.param(
                TitleMapping(anilist_id=5, tvdb_mappings={}, source=MappingSource.ANIBRIDGE),
                ["AniBridge has no usable season ranges for Title - not checked"],
                id="anibridge-no-usable-ranges",
            ),
        ],
    )
    def test_sonarr_no_episodes_resolved_stops_and_warns_only_of_an_empty_anibridge_mapping(
        self, mapping: TitleMapping, warnings: list[str]
    ) -> None:
        # A mapping that resolves to no episode stops the title on a no-episodes row: never as unmonitored and
        # never on to the grab. AniBridge's empty {} tvdb mapping also warns of its cause. The warning keys off
        # the source, so the mapping carries source=ANIBRIDGE as a real one does.
        run = FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        recording = install_recording_hub()
        strat = scripted_checking(run, ScriptedEpisodes(FakeSonarrClient(episodes=[])))

        strat.check_title(SERIES_7, 5, mapping)
        assert run.no_episodes_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert (run.nothing_to_grab_calls, run.pauses_owed) == ([], 1)
        assert diagnostic_messages(recording, Severity.WARNING) == warnings

    def test_sonarr_all_unmonitored_episodes_stop_with_an_unmonitored_row(self) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        sonarr = FakeSonarrClient(episodes=[sonarr_ep(1, 1).model_copy(update={"monitored": False})])
        strat = SonarrStrategy(make_run_deps(config=make_config(ignore_unmonitored=True)), run, sonarr_client=sonarr)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))
        assert run.unmonitored_titles == ["Title"]
        assert run.log_al_title_calls == []
        assert run.pauses_owed == 1

    def test_sonarr_one_monitored_episode_keeps_the_title_going(self) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Title")
        sonarr = FakeSonarrClient(episodes=[sonarr_ep(1, 1).model_copy(update={"monitored": False}), sonarr_ep(1, 2)])
        strat = SonarrStrategy(make_run_deps(config=make_config(ignore_unmonitored=True)), run, sonarr_client=sonarr)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))
        assert run.unmonitored_titles == []
        assert run.log_al_title_calls == ["Title"]

    def test_sonarr_reports_the_windows_missing_episodes(self) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record())
        sonarr = FakeSonarrClient(episodes=[sonarr_ep(1, 1), sonarr_ep(1, 2, episode_file_id=0)])
        strat = SonarrStrategy(make_run_deps(), run, sonarr_client=sonarr)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))
        assert run.missing_episodes == ["S01 E02"]

    def test_sonarr_all_invalid_selection_routes_to_shared_stop(self) -> None:
        # The interactive pick rejected every token: the strategy must route to
        # report_invalid_selection (which persists nothing) and never reach the
        # grab tail - the Sonarr twin of the Radarr guard in test_seadex_dict.
        run = FakeRunServices(
            prologue_entry=make_entry_record(),
            seadex_dict=seadex_groups("GroupA", "GroupB"),
            interactive_result={},
        )
        strat = scripted_checking(
            run,
            ScriptedEpisodes(FakeSonarrClient(episodes=[sonarr_ep(1, 1)])),
            config=make_config(interactive=True),
        )

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))
        assert run.invalid_selection_skips == 1
        assert run.grab_requests == []
        assert run.no_releases_calls == []


class TestSonarrCheckTitlePendingImportUpdates:
    """A Sonarr grab places each listed torrent under the title's window.

    A stored torrent is placed as its pending import's leftover.
    """

    def test_a_stored_torrent_is_placed_as_its_pending_imports_leftover(self) -> None:
        # Cour 1's pending import already has file 01 in its file map. Cour 2 marks the same torrent to grab: only
        # file 02 is placed, under the stored claim's window and this title's, and the pending import update merges
        # onto the pending import.
        store = FakeCacheStore()
        resident = pending_import(
            infohash="h1",
            anilist_id=22,
            ordered_episode_ids=(101,),
            file_episode_map={GRAB_FILES[0]: [101]},
            seadex_files=list(GRAB_FILES),
        )
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())

        run = sonarr_grab_run([EP1, EP2], store=store)

        (req,) = run.grab_requests
        (update,) = req.pending_import_updates.values()
        assert update.pending_import == resident
        assert update.file_map == {normalize_file_name(GRAB_FILES[1]): (102,)}
        assert (update.claim.anilist_id, update.claim.series_id) == (5, 7)
        assert update.claim.ordered_episode_ids == (102,)
        pending = update.pending_import_at(_STAMP, restart=False)
        assert claim_anilist_ids(pending) == (22, 5)
        assert dict(pending.file_episode_map) == {
            normalize_file_name(GRAB_FILES[0]): (101,),
            normalize_file_name(GRAB_FILES[1]): (102,),
        }
        # The planner reads the release by the whole file map's episodes inside this title's window.
        assert req.seadex_dict["NAN0"].releases[GRAB_URL].episodes == [placed_episode(season=1, episode=2, size=1000)]
        assert req.input_unread_groups == ()

    def test_the_listing_that_skips_a_specials_torrent_counts_the_ignored_titles_entry(self) -> None:
        # Title 5 has specials 2 and 4. Its entry lists the torrent, and so does the entry of title 6, which the
        # series' mappings ignore, on special 6: three specials, as the torrent is wide, and its `1` is none of them.
        # The torrent counts by another TVDB state, so nothing is placed and every file is left for a hand import.
        files = ("Show.S00E01.mkv", "Show.S00E02.mkv", "Show.S00E03.mkv")
        torrent = make_torrent_record(infohash="h1", file_names=files)
        own_entry = make_entry_record(torrents=(torrent,))
        seadex = FakeSeaDexSource({5: own_entry, 6: make_entry_record(torrents=(torrent,))})
        own = TitleMapping(anilist_id=5, tvdb_mappings={0: [(2, 2), (4, 4)]})
        seadex_dict: SeadexDict = {
            "NAN0": rg_group({GRAB_URL: make_release(url=GRAB_URL, infohash="h1", files=list(files), size=[1000] * 3)})
        }
        run = FakeRunServices(
            anilist_ids={5: own},
            ignored={6: TitleMapping(anilist_id=6, tvdb_mappings={0: [(6, 6)]})},
            prologue_entry=own_entry,
            seadex_dict=seadex_dict,
            plan_marks=frozenset({GRAB_URL}),
            import_wait_mode=ImportWaitMode.BLOCKING,
        )
        parses = {name: parsed_info(season=0, episodes=(n,), matched=((0, n),)) for n, name in enumerate(files, 1)}
        sonarr = FakeSonarrClient(
            episodes=[sonarr_ep(0, n, ep_id=500 + n, episode_file_id=0) for n in range(1, 7)], parse_fn=parses.get
        )
        strat = SonarrStrategy(make_run_deps(seadex=seadex), run, sonarr_client=sonarr)

        strat.check_title(SERIES_7, 5, own)

        (req,) = run.grab_requests
        release = req.seadex_dict["NAN0"].releases[GRAB_URL]
        assert (release.skip_reason, release.episodes) == (SkipReason.MISNUMBERED, [])
        update = req.pending_import_updates["h1"]
        assert (update.file_map, update.hand_import_files) == (
            {},
            ("show.s00e01.mkv", "show.s00e02.mkv", "show.s00e03.mkv"),
        )
        assert req.input_unread_groups == ()

    @pytest.mark.parametrize(
        ("episode_count", "placed"),
        [
            pytest.param(2, {"ova - 01 [grp].mkv": (501,), "ova - 02 [grp].mkv": (504,)}, id="the-windows-two"),
            pytest.param(3, {}, id="an-episode-sonarr-lacks"),
        ],
    )
    def test_the_titles_anilist_episode_count_reaches_its_placement(
        self, episode_count: int, placed: dict[str, tuple[int, ...]]
    ) -> None:
        # Two specials numbered apart take the sequence only when they are the title's every episode.
        specials = [
            sonarr_ep(0, 1, ep_id=501, episode_file_id=0, air_date="2016-03-01"),
            sonarr_ep(0, 4, ep_id=504, episode_file_id=0, air_date="2016-06-01"),
        ]
        names = [*numbered_names("ova", 2), *numbered_names("short", 3)]
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=specials), counts={5: episode_count})

        run = scripted_grab_run(episodes=episodes, files=tuple(ListedFile(name, 1000, parsed_info()) for name in names))

        (req,) = run.grab_requests
        assert req.pending_import_updates["h1"].file_map == placed

    def test_an_unread_listing_places_nothing_and_skips_the_release(self) -> None:
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=[EP1, EP2]), windows={5: [EP2]})

        run = scripted_grab_run(episodes=episodes, listings=ScriptedListings(unreachable=True))

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[GRAB_URL].skip_reason is SkipReason.PLACEMENT_INPUT_UNREAD
        assert req.pending_import_updates["h1"].file_map == {}
        assert req.input_unread_groups == ("NAN0",)

    def test_with_the_wait_off_a_stored_torrent_is_placed_and_no_pending_import_update_is_built(self) -> None:
        # A run that waits on no import reads no stored pending import. This one was stored without TVDB ids, so a
        # check that read it could not follow it and would leave its torrent unread.
        store = FakeCacheStore()
        resident = pending_import(infohash="h1", seadex_files=list(GRAB_FILES), tvdb_ids=None)
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())

        run = sonarr_grab_run([EP1, EP2], store=store, mode=ImportWaitMode.OFF)

        (req,) = run.grab_requests
        assert req.pending_import_updates == {}
        assert req.seadex_dict["NAN0"].releases[GRAB_URL].episodes == [placed_episode(season=1, episode=2, size=1000)]
        assert req.input_unread_groups == ()


_MOVED_SPECIAL = sonarr_ep(0, 1, ep_id=150, tvdb_id=9001, episode_file_id=0)
"""TVDB episode 9001 of series 7 as a special on row 150, after it left the season's first episode."""


_SECOND_ROW = sonarr_ep(1, 2, ep_id=102, tvdb_id=9002, episode_file_id=0)
"""The season's second episode of series 7, on the row it was stored on."""


def _stored_on_the_first_row(*, tvdb_ids: Mapping[int, int] | None) -> PendingImport:
    """Torrent "h1" as title 22 stored it: file 01 on row 101 of series 7, held with `tvdb_ids`."""

    return pending_import(
        infohash="h1",
        anilist_id=22,
        ordered_episode_ids=(101,),
        file_episode_map={GRAB_FILES[0]: [101]},
        seadex_files=list(GRAB_FILES),
        tvdb_ids=tvdb_ids,
    )


def _store_holding(pending: PendingImport) -> FakeCacheStore:
    """A store holding `pending`, with titles 9, 22 and 23 cached."""

    store = FakeCacheStore()
    store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
    for anilist_id in (9, 22, 23):
        store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": _STAMP})
    return store


def _title_stamps(store: FakeCacheStore) -> dict[int, str | None]:
    """The cache stamp of each title `_store_holding` cached."""

    cached_titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (9, 22, 23)}
    return {anilist_id: cached.updated_at for anilist_id, cached in cached_titles.items() if cached is not None}


class TestSonarrCheckTitleFollowsStoredPendingImports:
    """A check follows each stored pending import of the entry's torrents before it places them."""

    def test_a_stored_pending_import_is_followed_before_its_torrent_is_placed(self) -> None:
        store = _store_holding(_stored_on_the_first_row(tvdb_ids={101: 9001}))

        run = sonarr_grab_run([_MOVED_SPECIAL, _SECOND_ROW], store=store)

        (req,) = run.grab_requests
        pending = req.pending_import_updates["h1"].pending_import_at(_STAMP, restart=False)
        assert dict(pending.file_episode_map) == {
            normalize_file_name(GRAB_FILES[0]): (150,),
            normalize_file_name(GRAB_FILES[1]): (102,),
        }
        assert [claim.ordered_episode_ids for claim in pending.claims] == [(150,), (102,)]

    @pytest.mark.parametrize(
        ("stored", "followed", "stamps"),
        [
            pytest.param(
                _stored_on_the_first_row(tvdb_ids={101: 9001}),
                {"show - 01 [1080p].mkv": [150]},
                {9: _STAMP, 22: None, 23: _STAMP},
                id="one-series",
            ),
            # Series 8's TVDB episode 9101 has moved from row 201 to row 250 (`SERIES_8_E2`).
            pytest.param(
                two_claim_pending_import(
                    anilist_ids=(22, 23),
                    series_ids=(7, 8),
                    windows=((101,), (201,)),
                    infohash="h1",
                    file_episode_map={GRAB_FILES[0]: (101,), GRAB_FILES[1]: (201,)},
                    seadex_files=list(GRAB_FILES),
                    tvdb_ids={101: 9001, 201: 9101},
                ),
                {"show - 01 [1080p].mkv": [150], "show - 02 [1080p].mkv": [250]},
                {9: _STAMP, 22: None, 23: None},
                id="two-series",
            ),
        ],
    )
    def test_a_followed_pending_import_is_saved_when_nothing_is_grabbed_and_its_titles_are_uncached(
        self, stored: PendingImport, followed: dict[str, list[int]], stamps: dict[int, str | None]
    ) -> None:
        store = _store_holding(stored)

        run = sonarr_grab_run([_MOVED_SPECIAL, _SECOND_ROW], store=store, grab=False)

        (req,) = run.grab_requests
        row = store.get_pending_import_row(Arr.SONARR, "h1")
        assert req.pending_import_updates == {}
        assert row is not None
        assert row["file_episode_map"] == followed
        assert _title_stamps(store) == stamps

    def test_a_stored_pending_import_that_cannot_be_followed_is_not_joined_and_its_torrent_is_unread(self) -> None:
        # Stored with no TVDB ids, on a row Sonarr no longer has. Only one awaiting its cleanup outlasts the run's
        # start like that.
        stored = replace(_stored_on_the_first_row(tvdb_ids=None), awaiting_cleanup=True)
        store = _store_holding(stored)

        run = sonarr_grab_run([_MOVED_SPECIAL, _SECOND_ROW], store=store)

        (req,) = run.grab_requests
        assert req.seadex_dict["NAN0"].releases[GRAB_URL].skip_reason is SkipReason.PLACEMENT_INPUT_UNREAD
        assert req.input_unread_groups == ("NAN0",)
        assert req.pending_import_updates["h1"].pending_import is None
        assert store.get_pending_import_row(Arr.SONARR, "h1") == stored.to_json()
        assert _title_stamps(store) == {9: _STAMP, 22: _STAMP, 23: _STAMP}


class TestSonarrCheckTitleMetadataUnread:
    """A grab asks whether its check read a title without its AniList answer or a series without its TVDB orders.

    The titles are every title of the title's series, the series are the title's and every stored pending import's.
    A title that stops before a grab asks about itself alone.
    """

    @pytest.mark.parametrize(
        ("unread_series", "unread_anilist_ids", "expected"),
        [
            pytest.param(frozenset[int](), frozenset[int](), False, id="all-read"),
            pytest.param(frozenset({7}), frozenset[int](), True, id="its-own-series"),
            pytest.param(frozenset({8}), frozenset[int](), True, id="a-stored-records-series"),
            pytest.param(frozenset[int](), frozenset({5}), True, id="its-own-entry"),
            pytest.param(frozenset[int](), frozenset({4}), True, id="another-entry-of-its-series"),
            pytest.param(frozenset[int](), frozenset({6}), True, id="an-ignored-entry-of-its-series"),
        ],
    )
    def test_the_grab_request_carries_whether_its_metadata_is_unread(
        self, unread_series: frozenset[int], unread_anilist_ids: frozenset[int], expected: bool
    ) -> None:
        # Series 8's pending import already holds the torrent title 5 lists, so the grab places it under that pending
        # import too.
        store = FakeCacheStore()
        resident = pending_import(infohash="h1", series_id=8, seadex_files=list(GRAB_FILES))
        store.put_pending_import_row(Arr.SONARR, "h1", resident.to_json())
        episodes = ScriptedEpisodes(
            FakeSonarrClient(episodes=[EP1, EP2]), unread_series=unread_series, unread_anilist_ids=unread_anilist_ids
        )

        run = scripted_grab_run(store=store, episodes=episodes)

        (req,) = run.grab_requests
        assert req.metadata_unread is expected

    @pytest.mark.parametrize(
        ("window", "order", "stopped_on"),
        [
            # No release to grab caches the title as having none. No episode in Sonarr's order caches it as
            # having nothing to grab.
            pytest.param([EP2], AIRED_ORDER, ([5], list[int]()), id="no-suitable-release"),
            pytest.param([STAND_IN_E2], "dvd", (list[int](), [5]), id="no-episode-in-sonarrs-order"),
        ],
    )
    def test_a_title_that_stops_before_a_grab_is_not_unread_for_another_entry_of_its_series(
        self, window: list[SonarrEpisode], order: str, stopped_on: tuple[list[int], list[int]]
    ) -> None:
        run = FakeRunServices(
            anilist_ids={4: TitleMapping(anilist_id=4), 5: TitleMapping(anilist_id=5)},
            prologue_entry=make_entry_record(),
            seadex_dict={},
        )
        episodes = ScriptedEpisodes(FakeSonarrClient(episodes=window), order=order, unread_anilist_ids=frozenset({4}))
        strat = scripted_checking(run, episodes)

        strat.check_title(SERIES_7, 5, TitleMapping(anilist_id=5))

        no_releases = [anilist_id for anilist_id, _ in run.no_releases_calls]
        assert (no_releases, [anilist_id for anilist_id, _ in run.nothing_to_grab_calls]) == stopped_on
        assert (run.no_releases_unread, run.nothing_to_grab_unread) == ([], [])
