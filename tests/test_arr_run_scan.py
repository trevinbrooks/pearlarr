# pyright: strict
"""Tests for an arr run up to the end of its scan.

What the boot steps show, which library items the scan checks and what it shows for each, and what the scan commits
as it goes and the pause it keeps between titles.

Each test runs a whole Sonarr run, the loop and its services built by their constructors. The arr's side is a
strategy that lists a library and records each title the scan asks it to check, or the real Sonarr strategy where
only it shows the outcome. The outcome is read off the events the run showed, the titles the scan asked to check,
and the cache the run left, reopened where what a stop keeps is the point. How the run ends once the scan is done is
in `test_arr_run_finish`, and what a run checks again and stores for the next run in `test_arr_run_recheck`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest

from pearlarr.arr_models import SonarrSeries
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.cache import CacheStore
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import (
    ArrRunSummaryReady,
    BootReady,
    BootStepFinished,
    Diagnostic,
    ItemStarted,
    ScanStarted,
    ScopeClosed,
    ScopeKind,
    ScopeOpened,
    Severity,
    TitleRow,
)
from pearlarr.output.events import TitleState
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import ImportWaitMode, OutcomeKind

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    ScriptedAniListClient,
    drive_arr_run,
    grab_request,
    make_config,
    make_entry_record,
    make_run_deps,
    sonarr_file_imported,
)
from .fakes import (
    FakeArrItem,
    FakeClock,
    FakeSonarrClient,
    FakeStrategy,
    RecordingHub,
    install_recording_hub,
)
from .wait_fakes import FakeQbit


class _Library(FakeStrategy):
    """The arr's side of a scan over `items`, each mapping one title, whose AniList id is the item's id.

    `checked` holds the item of each title the scan asked it to check, by the item's title. The `unmapped` item maps
    no title.
    """

    def __init__(self, *items: FakeArrItem, unmapped: str | None = None) -> None:
        super().__init__(items=list(items), anilist_ids={})
        self._unmapped = unmapped
        self.checked: list[str] = []

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {} if item.title == self._unmapped else {item.id: TitleMapping(anilist_id=item.id)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        self.checked.append(item.title)


def _scan(strategy: FakeStrategy, *, item_id: int | None = None, ignore_unmonitored: bool = False) -> RecordingHub:
    """One whole Sonarr run of `strategy`, and what it showed. `item_id` narrows it as `--tvdb-id` does."""

    config = make_config(url="http://sonarr", api_key="key", ignore_unmonitored=ignore_unmonitored)
    recording = install_recording_hub()
    drive_arr_run(make_run_deps(config=config), lambda _: strategy, item_id=item_id)
    return recording


def _rows(recording: RecordingHub) -> list[tuple[TitleState, str]]:
    """The one-line title rows the run showed, each by its state and label."""

    return [(row.state, row.label) for row in recording.of_type(TitleRow)]


def test_the_scan_shows_its_item_total_and_each_items_place_in_it() -> None:
    library = _Library(FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan"))

    recording = _scan(library)

    assert recording.of_type(ScanStarted) == [ScanStarted(arr=Arr.SONARR, total=2)]
    assert recording.of_type(ItemStarted) == [
        ItemStarted(arr=Arr.SONARR, index=1, total=2, title="Frieren"),
        ItemStarted(arr=Arr.SONARR, index=2, total=2, title="Dandadan"),
    ]


class TestWhichItemsAreChecked:
    """Which of the library's items the scan asks to check.

    An unmonitored item is checked like any other unless the run is set to ignore it. Then it is shown as
    unmonitored and left out of the scan's checks and of the SeaDex fetch before them. A single-item run scans only
    the item asked for, and an item with no AniList mapping is shown as having none.
    """

    def test_an_unmonitored_item_is_checked_by_default(self) -> None:
        library = _Library(
            FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan", monitored=False)
        )

        recording = _scan(library)

        assert library.checked == ["Frieren", "Dandadan"]
        assert _rows(recording) == []

    def test_an_unmonitored_item_is_shown_and_left_unchecked_when_set_to_ignore_it(self) -> None:
        library = _Library(
            FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan", monitored=False)
        )

        recording = _scan(library, ignore_unmonitored=True)

        assert library.checked == ["Frieren"]
        assert _rows(recording) == [(TitleState.UNMONITORED, "Dandadan")]

    def test_the_seadex_fetch_counts_only_the_monitored_items_when_set_to_ignore_the_rest(self) -> None:
        library = _Library(FakeArrItem(item_id=1), FakeArrItem(item_id=2), FakeArrItem(item_id=3, monitored=False))

        recording = _scan(library, ignore_unmonitored=True)

        [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching SeaDex entries"]
        assert step.detail == "2 entries"

    def test_a_single_item_run_scans_only_the_item_asked_for(self) -> None:
        library = _Library(FakeArrItem(item_id=1), FakeArrItem(item_id=2, title="Dandadan"))

        recording = _scan(library, item_id=2)

        assert library.checked == ["Dandadan"]
        assert recording.of_type(ItemStarted) == [ItemStarted(arr=Arr.SONARR, index=1, total=1, title="Dandadan")]

    def test_an_item_with_no_anilist_mapping_is_shown_as_having_none(self) -> None:
        library = _Library(FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2), unmapped="Frieren")

        recording = _scan(library)

        assert _rows(recording) == [(TitleState.NO_MAPPING, "Frieren")]


def test_an_ignored_title_is_shown_once_by_the_name_of_its_series() -> None:
    # Series 3 maps AniList 1 and 2, and 2 is ignored. The imported file has the activity read ask for its titles
    # before the scan, as the prefetch does. Title 1, with no SeaDex entry, keeps the series off an undecided case:
    # a series whose only title is ignored shows a no-mapping row too.
    config = make_config(url="http://sonarr", api_key="key", ignore_anilist_ids={2})
    deps = make_run_deps(
        config=config,
        anime_ids={
            "Frieren": {"anilist_id": 1, "tvdb_id": 50, "tvdb_season": 1},
            "Frieren Season 2": {"anilist_id": 2, "tvdb_id": 50, "tvdb_season": 2},
        },
    )
    series = SonarrSeries.model_validate({"id": 3, "title": "Sousou no Frieren", "tvdbId": 50})
    sonarr = FakeSonarrClient(all_series=[series], history_since=[sonarr_file_imported(3)])
    recording = install_recording_hub()

    drive_arr_run(deps, lambda services: SonarrStrategy(deps, services, sonarr_client=sonarr))

    assert _rows(recording) == [(TitleState.IGNORED, "Sousou no Frieren"), (TitleState.NO_ENTRY, "Sousou no Frieren")]


class _TitleFails(_Library):
    """A library whose items each map titles 1 and 2, the check of title 1 failing."""

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {1: TitleMapping(anilist_id=1), 2: TitleMapping(anilist_id=2)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        if anilist_id == 1:
            raise RuntimeError("check failed")
        super().check_title(item, anilist_id, mapping)


class _SnapshotFails(_Library):
    """A library whose `failing` item's import snapshot fails once its title is checked."""

    def __init__(self, *items: FakeArrItem, failing: str) -> None:
        super().__init__(*items)
        self._failing = failing

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        if item.title == self._failing:
            raise RuntimeError("snapshot failed")
        return None


def _errors(recording: RecordingHub) -> list[Diagnostic]:
    """The errors the run showed."""

    return [d for d in recording.of_type(Diagnostic) if d.severity is Severity.ERROR]


class TestOneFailureIsContained:
    """A title or a library item that fails is shown as an error with its traceback, and the scan moves on.

    A failing title leaves the item's other titles checked, and a failing item the library's other items. The run
    ends with its summary all the same, the error counted in it.
    """

    def test_a_failing_title_leaves_the_items_other_titles_checked(self) -> None:
        library = _TitleFails(FakeArrItem(item_id=7, title="Frieren"))

        recording = _scan(library)

        # Title 2's check, after title 1's failed.
        assert library.checked == ["Frieren"]
        (error,) = _errors(recording)
        assert error.message == "Frieren (AniList #1): unexpected error (check failed) - moving on to the next title"
        assert error.trace is not None
        (summary,) = recording.of_type(ArrRunSummaryReady)
        assert summary.summary.errors == 1

    def test_a_failing_item_leaves_the_librarys_other_items_checked(self) -> None:
        library = _SnapshotFails(
            FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan"), failing="Frieren"
        )
        recording = install_recording_hub()

        # The scan takes each item's import snapshot only while the wait is on, which takes qBittorrent.
        deps = make_run_deps(qbit=FakeQbit().client())
        drive_arr_run(deps, lambda _: library, import_wait_mode=ImportWaitMode.DEFERRED)

        assert library.checked == ["Frieren", "Dandadan"]
        (error,) = _errors(recording)
        assert error.message == "Frieren: unexpected error (snapshot failed) - moving on to the next library item"
        assert error.trace is not None
        (summary,) = recording.of_type(ArrRunSummaryReady)
        assert summary.summary.errors == 1


_NOTED = OutcomeKind.IMPORTED
"""The outcome a boot step that finished with a note shows."""


_WARNED = OutcomeKind.LEFT
"""The outcome a boot step that finished with a warning shows."""


class TestBootNotes:
    """Each boot step ends on a note of what it fetched, or on a warning when its source was unreachable.

    An unreachable source's step never claims a count it didn't fetch.
    """

    @pytest.mark.parametrize(
        ("titles", "seadex_unreachable", "anilist_failing", "anilist_step", "seadex_step"),
        [
            pytest.param({1}, False, frozenset[int](), ("1 title", _NOTED), ("1 entry", _NOTED), id="healthy"),
            pytest.param(set[int](), False, frozenset[int](), ("cached", _NOTED), ("cached", _NOTED), id="no-titles"),
            pytest.param({1}, True, frozenset[int](), ("1 title", _NOTED), ("unreachable", _WARNED), id="no-seadex"),
            pytest.param({1}, False, frozenset({1}), ("unavailable", _WARNED), ("1 entry", _NOTED), id="no-anilist"),
        ],
    )
    def test_each_boot_step_notes_what_it_fetched_or_warns_that_its_source_was_unreachable(
        self,
        titles: set[int],
        seadex_unreachable: bool,
        anilist_failing: frozenset[int],
        anilist_step: tuple[str, OutcomeKind],
        seadex_step: tuple[str, OutcomeKind],
    ) -> None:
        # Both series map the same titles, so each prefetch fetches each title once.
        library = FakeStrategy(
            items=[FakeArrItem(item_id=1), FakeArrItem(item_id=2)],
            anilist_ids={anilist_id: TitleMapping(anilist_id=anilist_id) for anilist_id in titles},
        )
        deps = make_run_deps(
            seadex=FakeSeaDexSource(unreachable=seadex_unreachable),
            anilist_client=ScriptedAniListClient(failing=anilist_failing),
        )
        recording = install_recording_hub()

        drive_arr_run(deps, lambda _: library)

        assert [(step.label, step.detail, step.outcome) for step in recording.of_type(BootStepFinished)] == [
            ("Fetching Sonarr library", "2 series", _NOTED),
            ("Checking Sonarr activity", "none", _NOTED),
            ("Fetching AniList metadata", *anilist_step),
            ("Fetching SeaDex entries", *seadex_step),
        ]


def test_the_boot_view_closes_on_its_ready_line_before_the_scan_starts() -> None:
    # The console's live boot view draws the boot steps, and a scan row shown while it is open reflows above an old
    # spinner.
    recording = _scan(_Library())

    (boot,) = [opened.scope for opened in recording.of_type(ScopeOpened) if opened.scope.kind is ScopeKind.BOOT]
    order = [
        type(event)
        for event in recording.events
        if isinstance(event, BootReady | ScanStarted) or event == ScopeClosed(scope=boot)
    ]
    assert order == [BootReady, ScopeClosed, ScanStarted]


class _Staging(FakeStrategy):
    """A library of items 1 and 2, item 1 mapping titles 11 and 12 and item 2 titles 21 and 22.

    Each title's check ends with nothing to grab, which stages the title's cached row and owes the pacing pause, as a
    real check does. The check of title `stop_at` stops the run once its row is staged, as the SIGTERM handler would.
    """

    def __init__(self, services: ArrRunServices, *, stop_at: int | None = None) -> None:
        super().__init__(items=[FakeArrItem(item_id=1), FakeArrItem(item_id=2)], anilist_ids={})
        self._services = services
        self._stop_at = stop_at

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {anilist_id: TitleMapping(anilist_id=anilist_id) for anilist_id in (item.id * 10 + 1, item.id * 10 + 2)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        self._services.report_no_releases(anilist_id, {})
        if anilist_id == self._stop_at:
            raise SystemExit(0)


class _StagingFirstSnapshotFails(_Staging):
    """A `_Staging` library whose item 1 import snapshot fails once its titles are checked."""

    @override
    def pending_import_series_id(self, item: FakeArrItem) -> int | None:
        if item.id == 1:
            raise RuntimeError("snapshot failed")
        return None


class _StopOnSleepClock(FakeClock):
    """A `FakeClock` whose `stop_on`-th sleep raises `SystemExit`, as the SIGTERM handler would mid-pause."""

    def __init__(self, stop_on: int) -> None:
        super().__init__()
        self._stop_on = stop_on

    @override
    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if len(self.sleeps) == self._stop_on:
            raise SystemExit(0)


def _scan_into(
    cache: Path,
    strategy: Callable[[ArrRunServices], _Staging],
    *,
    qbit: FakeQbit | None,
    clock: FakeClock | None = None,
) -> None:
    """One whole Sonarr run of `strategy` over the cache file at `cache`, closed after it as bootstrap closes it.

    `qbit` None is a run with no qBittorrent configured, so a dry run. AniList knows every title.
    """

    store = CacheStore.load(str(cache), config_hash="")
    try:
        client = None if qbit is None else qbit.client()
        deps = make_run_deps(cache_store=store, qbit=client, clock=clock, anilist_client=ScriptedAniListClient())
        drive_arr_run(deps, strategy)
    finally:
        store.close()


def _saved_titles(cache: Path) -> list[int]:
    """The titles of the `_Staging` library the cache file at `cache` holds cached rows of, read fresh."""

    store = CacheStore.load(str(cache), config_hash="")
    try:
        return [anilist_id for anilist_id in (11, 12, 21, 22) if store.get_cached_title(Arr.SONARR, anilist_id)]
    finally:
        store.close()


class TestPerItemCommit:
    """A real run commits after each library item, so a stop keeps what finished and rolls back the item in flight.

    A dry run commits nothing.
    """

    def test_a_stop_mid_item_keeps_the_finished_items(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Stopped on item 2's second title, after its first title's row was staged.
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _Staging(services, stop_at=22), qbit=FakeQbit())

        assert _saved_titles(cache) == [11, 12]

    def test_an_item_that_raised_still_commits_what_it_staged(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Item 1's snapshot fails after its titles, an error the scan contains. The scan takes each item's snapshot
        # while imports are waited on, as they are by default. Item 2 stops before reaching its own.
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _StagingFirstSnapshotFails(services, stop_at=21), qbit=FakeQbit())

        assert _saved_titles(cache) == [11, 12]

    def test_a_stop_in_the_pacing_pause_keeps_the_item_before_it(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        # Every title owes a pause. The second one slept is the one before item 2's first title.
        with pytest.raises(SystemExit):
            _scan_into(cache, _Staging, qbit=FakeQbit(), clock=_StopOnSleepClock(stop_on=2))

        assert _saved_titles(cache) == [11, 12]

    def test_a_dry_run_keeps_nothing(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.db"

        _scan_into(cache, _Staging, qbit=None)

        assert _saved_titles(cache) == []

    @pytest.mark.parametrize(
        ("qbit", "next_fetch"),
        [pytest.param(FakeQbit(), "cached", id="real"), pytest.param(None, "4 titles", id="dry-run")],
    )
    def test_only_a_real_run_keeps_the_anilist_answers_its_prefetch_fetched_before_the_scan(
        self, tmp_path: Path, qbit: FakeQbit | None, next_fetch: str
    ) -> None:
        # Stopped in the first title, before any library item commits. The next run fetches only what wasn't kept.
        cache = tmp_path / "cache.db"
        with pytest.raises(SystemExit):
            _scan_into(cache, lambda services: _Staging(services, stop_at=11), qbit=qbit)
        recording = install_recording_hub()

        _scan_into(cache, _Staging, qbit=FakeQbit())

        [step] = [step for step in recording.of_type(BootStepFinished) if step.label == "Fetching AniList metadata"]
        assert step.detail == next_fetch


type _Ending = Callable[[ArrRunServices, int], None]
"""How a title's check ends, given the run's services and the title's AniList id."""


def _grab(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends at the grab tail."""

    services.grab_and_cache(grab_request(anilist_id=anilist_id))


def _no_releases(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends with no release to grab."""

    services.report_no_releases(anilist_id, {})


def _invalid_pick(services: ArrRunServices, anilist_id: int) -> None:
    """The check ends with an interactive pick that chose nothing valid."""

    del anilist_id
    services.report_invalid_selection()


_PACED_ENTRIES = {anilist_id: make_entry_record(anilist_id=anilist_id) for anilist_id in (1, 2, 3)}
"""SeaDex's entries of the `_Endings` titles."""


class _Endings(FakeStrategy):
    """A library of one item whose titles 1, 2 and 3 are checked in turn, each through the unchanged gate first.

    A title the gate sends on ends its check as `ending` says.
    """

    def __init__(self, services: ArrRunServices, ending: _Ending) -> None:
        super().__init__(items=[FakeArrItem()], anilist_ids={i: TitleMapping(anilist_id=i) for i in _PACED_ENTRIES})
        self._services = services
        self._ending = ending

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        entry = self._services.begin_check(anilist_id)
        if entry is not None and not self._services.is_unchanged(anilist_id, entry, lambda: ""):
            self._ending(self._services, anilist_id)


class TestPacingPause:
    """A check that queried owes the `sleep_time` pause, slept once before the next title.

    A title left unchanged queried nothing and owes none.
    """

    @pytest.mark.parametrize(
        "ending",
        [
            pytest.param(_grab, id="grab"),
            pytest.param(_no_releases, id="no-releases"),
            pytest.param(_invalid_pick, id="invalid-pick"),
        ],
    )
    def test_a_check_that_queried_owes_one_pause_slept_before_the_next_title(self, ending: _Ending) -> None:
        # An earlier run left titles 2 and 3 unchanged, so the first's pause is all the run owes.
        cache = FakeCacheStore()
        for anilist_id in (2, 3):
            entry = _PACED_ENTRIES[anilist_id]
            cache.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": entry.updated_at, "url": entry.url})
        clock = FakeClock()
        deps = make_run_deps(
            config=make_config(url="http://sonarr", api_key="key", sleep_time=3),
            cache_store=cache,
            seadex=FakeSeaDexSource(_PACED_ENTRIES),
            clock=clock,
        )

        drive_arr_run(deps, lambda services: _Endings(services, ending))

        assert clock.sleeps == [3]
