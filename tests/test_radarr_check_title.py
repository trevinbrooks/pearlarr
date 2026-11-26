# pyright: strict
"""Tests for a Radarr movie's check, up to the grab request it hands the run.

They cover the stops before a grab, what a grab carries to the plan, and the pending import update each release to
grab gets. A check runs over `FakeRunServices`, which records what the strategy hands the run.
"""

import pytest

from pearlarr.arr_models import ArrFiles, MovieFile, RadarrMovie
from pearlarr.cache import CachedTitleUpdate
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.releases import SeadexDict
from pearlarr.wait_types import ImportWaitMode

from .builders import FakeCacheStore, grab_plan, make_entry_record, make_release, pending_import, rg_group
from .fakes import FakeRadarrClient
from .title_check_builders import FakeRunServices, radarr_checking, seadex_groups


class TestRadarrCheckTitle:
    """A movie's check stops on no SeaDex entry, a cached title or no releases. A grab carries its files to the plan."""

    def test_radarr_title_with_no_seadex_entry_grabs_nothing_and_owes_no_pause(self) -> None:
        run = FakeRunServices()
        strat = radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))

        # The run reports the missing entry under the title's id.
        assert run.anilist_id_prologue_calls == [5]
        assert (run.grab_requests, run.pauses_owed) == ([], 0)

    def test_a_cached_title_stops_with_an_empty_coverage_backfill(self) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record(), cached_skip=True)
        strat = radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))

        assert (run.grab_requests, run.no_releases_calls, run.nothing_to_grab_calls) == ([], [], [])
        # A movie has no episodes, so the coverage a cached row is backfilled with is empty.
        [coverage] = run.cached_skip_coverages
        assert coverage() == ""

    def test_no_releases_routes_to_shared_stop_with_movie_cache_details(self) -> None:
        entry = make_entry_record(url="https://releases.moe/5")
        run = FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict={},
        )
        strat = radarr_checking(run)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))
        expected: CachedTitleUpdate = {
            "name": "Movie Title",
            "updated_at": entry.updated_at,
            "url": "https://releases.moe/5",
            "coverage": "",
        }
        assert run.no_releases_calls == [(5, expected)]
        assert run.grab_requests == []

    def test_a_grab_carries_the_movies_files_to_the_plan_and_every_group_they_are_of(self) -> None:
        entry = make_entry_record()
        seadex_dict = seadex_groups("SubGroup")
        filtered = seadex_groups("SubGroup")
        run = FakeRunServices(
            prologue_entry=entry,
            anilist_title="Movie Title",
            seadex_dict=seadex_dict,
            plan_grabs_result=grab_plan(filtered),
        )
        files = [
            MovieFile(release_group="GroupA", size=100),
            MovieFile(release_group="GroupA", size=150),
            MovieFile(release_group="GroupB", size=200),
        ]
        strat = radarr_checking(run, FakeRadarrClient(movie_files=files))

        strat.check_title(RadarrMovie(id=3, title="Item Title"), 5, TitleMapping(anilist_id=5))

        assert run.plan_grabs_calls == [(seadex_dict, ArrFiles(by_group={"GroupA": (100, 150), "GroupB": (200,)}))]
        [req] = run.grab_requests
        assert (req.anilist_id, req.arr_title, req.title_name) == (5, "Item Title", "Movie Title")
        assert req.entry is entry
        assert req.seadex_dict is filtered
        assert req.replaced_groups == ("GroupA", "GroupB")

    @pytest.mark.parametrize(
        ("interactive", "seadex_dict", "prompts"),
        [
            pytest.param(True, seadex_groups("GroupA", "GroupB"), 1, id="interactive-two-groups"),
            pytest.param(True, seadex_groups("GroupA"), 0, id="interactive-one-group"),
            pytest.param(False, seadex_groups("GroupA", "GroupB"), 0, id="not-interactive"),
        ],
    )
    def test_only_an_interactive_run_with_two_groups_asks_which_to_grab(
        self, interactive: bool, seadex_dict: SeadexDict, prompts: int
    ) -> None:
        run = FakeRunServices(prologue_entry=make_entry_record(), seadex_dict=seadex_dict)
        strat = radarr_checking(run, interactive=interactive)

        strat.check_title(RadarrMovie(id=1, title="Title"), 5, TitleMapping(anilist_id=5))

        assert len(run.interactive_calls) == prompts


class TestRadarrCheckTitlePendingImportUpdates:
    """A Radarr grab builds a `PendingImportUpdate` per release to grab, keyed by its hash in qBittorrent.

    A run that waits on no import builds none.
    """

    @staticmethod
    def _run_process(
        *, store: FakeCacheStore, cross_seed: str | None = None, mode: ImportWaitMode = ImportWaitMode.BLOCKING
    ) -> FakeRunServices:
        """Drive `check_title` to grab_and_cache and return the recording services, in a blocking run unless `mode` says.

        `cross_seed` marks the release as carried by that torrent.
        """

        item = make_release(url="https://nyaa.si/1", infohash="h1")
        item.cross_seed_hash = cross_seed
        seadex: SeadexDict = {"NAN0": rg_group({"https://nyaa.si/1": item})}
        run = FakeRunServices(
            prologue_entry=make_entry_record(url="https://releases.moe/9"),
            anilist_title="A Movie",
            seadex_dict=seadex,
            plan_marks=frozenset({"https://nyaa.si/1"}),
            import_wait_mode=mode,
            arr=Arr.RADARR,
            cache_store=store,
        )
        strat = radarr_checking(run)
        strat.check_title(RadarrMovie(id=5, title="Movie"), 42, TitleMapping(anilist_id=42))
        return run

    def test_a_release_to_grab_starts_a_pending_import_claimed_by_the_movies_title(self) -> None:
        run = self._run_process(store=FakeCacheStore())

        (req,) = run.grab_requests
        update = req.pending_import_updates["h1"]
        assert (update.claim.anilist_id, update.claim.series_id) == (42, 0)
        assert (update.claim.title, update.claim.url) == ("A Movie", "https://releases.moe/9")
        assert (update.pending_import, update.file_map) == (None, {})

    def test_no_pending_import_updates_when_wait_mode_off(self) -> None:
        run = self._run_process(store=FakeCacheStore(), mode=ImportWaitMode.OFF)

        (req,) = run.grab_requests
        assert req.pending_import_updates == {}

    def test_a_release_to_grab_is_stored_on_its_cross_seeds_pending_import(self) -> None:
        store = FakeCacheStore()
        resident = pending_import(infohash="c1", series_id=0, file_episode_map={})
        store.put_pending_import_row(Arr.RADARR, "c1", resident.to_json())

        run = self._run_process(store=store, cross_seed="c1")

        (req,) = run.grab_requests
        update = req.pending_import_updates["c1"]
        assert update.pending_import == resident
        assert (update.facts.infohash, update.facts.is_cross_seed) == ("c1", True)
