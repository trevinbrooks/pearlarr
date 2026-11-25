# pyright: strict
"""Tests for an arr run's scan: which library items it checks and what it shows for each.

Each test runs a whole Sonarr run, the loop and its services built by their constructors. The arr's side is a
strategy that lists a library and records each title the scan asks it to check, or the real Sonarr strategy where
only it shows the outcome. The outcome is read off the events the run showed and the titles the scan asked to check.
How the run ends once the scan is done is in `test_arr_run_finish`.
"""

from typing import override

from pearlarr.arr_models import HistoryRecord, SonarrSeries
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import (
    ArrRunSummaryReady,
    BootStepFinished,
    Diagnostic,
    ItemStarted,
    ScanStarted,
    Severity,
    TitleRow,
)
from pearlarr.output.events import TitleState
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import ImportWaitMode

from .builders import drive_arr_run, make_config, make_run_deps
from .fakes import FakeArrItem, FakeSonarrClient, FakeStrategy, RecordingHub, install_recording_hub
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
        library = _Library(
            FakeArrItem(item_id=1, title="Frieren"),
            FakeArrItem(item_id=2, title="Dandadan"),
            FakeArrItem(item_id=3, title="Kaiju No. 8", monitored=False),
        )

        recording = _scan(library, ignore_unmonitored=True)

        [step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Fetching SeaDex entries"]
        assert step.detail == "2 entries"

    def test_a_single_item_run_scans_only_the_item_asked_for(self) -> None:
        library = _Library(FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan"))

        recording = _scan(library, item_id=2)

        assert library.checked == ["Dandadan"]
        assert recording.of_type(ItemStarted) == [ItemStarted(arr=Arr.SONARR, index=1, total=1, title="Dandadan")]

    def test_an_item_with_no_anilist_mapping_is_shown_as_having_none(self) -> None:
        library = _Library(
            FakeArrItem(item_id=1, title="Frieren"), FakeArrItem(item_id=2, title="Dandadan"), unmapped="Frieren"
        )

        recording = _scan(library)

        assert _rows(recording) == [(TitleState.NO_MAPPING, "Frieren")]


_FILE_IMPORTED = HistoryRecord(
    id=1, date="2026-07-06T10:00:00Z", item_id=7, event_type="downloadFolderImported", download_id=None, reason=None
)
"""Sonarr's history record of a file imported into series 7."""


def test_an_ignored_title_is_shown_once_by_the_name_of_its_series() -> None:
    # Series 7 maps AniList 1 and 2, and 2 is ignored. The imported file has the activity read ask for its titles
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
    series = SonarrSeries.model_validate({"id": 7, "title": "Sousou no Frieren", "tvdbId": 50})
    sonarr = FakeSonarrClient(all_series=[series], history_since=[_FILE_IMPORTED])
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
