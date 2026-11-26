# pyright: strict
"""What an arr run checks again of the titles an earlier run cached, and what it stores for the next run.

A run checks again the cached titles Sonarr's history names, the ones the strategy's start-of-run hook names, and
every one once the selection settings changed. A full scan stores the settings and the history read's checkpoint
for the next run. Each test runs whole Sonarr runs, the loop and its services built by their constructors, over a
strategy that lists a library and passes each title through the unchanged gate or checks it no further. The history
read itself is in `test_arr_activity`, and the rest of the scan in `test_arr_run_scan`.
"""

from dataclasses import dataclass
from typing import override

import pytest

from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.cache import HistoryCheckpoint
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import BootStepFinished, TitleHeader
from pearlarr.output.events import TitleState

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    ScriptedAniListClient,
    drive_arr_run,
    make_config,
    make_entry_record,
    make_run_deps,
    sonarr_file_imported,
)
from .fakes import FakeArrItem, FakeStrategy, RecordingHub, diagnostic_messages, install_recording_hub
from .wait_fakes import FakeQbit

_ENTRIES = {anilist_id: make_entry_record(anilist_id=anilist_id) for anilist_id in (1, 2)}
"""SeaDex's entries of the `_CachedLibrary` titles."""


class _CachedLibrary(FakeStrategy):
    """Sonarr's side of a library of series 1 and 2, each mapping one title, whose AniList id is the series' id.

    Each title goes through the unchanged gate as both strategies send it, and is checked no further. `history` is
    Sonarr's history, and the start-of-run hook names the titles in `check_again`.
    """

    def __init__(self, services: ArrRunServices, history: list[HistoryRecord], check_again: frozenset[int]) -> None:
        super().__init__(items=[FakeArrItem(item_id=1), FakeArrItem(item_id=2)], anilist_ids={}, history=history)
        self.check_again = set(check_again)
        self._services = services

    @override
    def item_mappings(self, item: FakeArrItem, log_ignored: bool = True) -> dict[int, TitleMapping]:
        return {item.id: TitleMapping(anilist_id=item.id)}

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        entry = self._services.begin_check(anilist_id)
        if entry is not None:
            self._services.is_unchanged(anilist_id, entry, lambda: "")


def _cache_left_unchanged(checkpoint: HistoryCheckpoint | None = None) -> FakeCacheStore:
    """A cache an earlier run left both `_CachedLibrary` titles unchanged in, with `checkpoint` its activity read's.

    Each row has the url a run writes, so the gate never fills one in as it does for an older row.
    """

    cache = FakeCacheStore()
    for anilist_id, entry in _ENTRIES.items():
        cache.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": entry.updated_at, "url": entry.url})
    if checkpoint is not None:
        cache.put_history_checkpoint(Arr.SONARR, checkpoint)
    return cache


def _shown_unchanged(recording: RecordingHub) -> list[int | None]:
    """The titles the run showed as unchanged, so left unchecked."""

    return [header.anilist_id for header in recording.of_type(TitleHeader) if header.state is TitleState.UNCHANGED]


def _activity_note(recording: RecordingHub) -> str | None:
    """The note the run's activity read ended its boot step on. None when the run showed no such step."""

    notes = [step.detail for step in recording.of_type(BootStepFinished) if step.label == "Checking Sonarr activity"]
    return notes[0] if notes else None


@dataclass(frozen=True, kw_only=True)
class _Situation:
    """What a run over the `_CachedLibrary` meets.

    `imported_into` is the series Sonarr's history shows a file imported into since `checkpoint`, None for no
    history. The two settings are the config's, and `check_again` the titles the start-of-run hook names.
    """

    imported_into: int | None = 1
    checkpoint: HistoryCheckpoint | None = None
    detect_activity: bool = True
    ignore_update_times: bool = False
    check_again: frozenset[int] = frozenset()


_GAP = HistoryCheckpoint("2020-01-01T00:00:00Z", 1)
"""A checkpoint from further back than the history read reaches, so the history since it can't all be read.

Its record 1 came before `sonarr_file_imported`'s.
"""


class TestWhatARunChecksAgain:
    """A run checks again the cached titles Sonarr's history or the start-of-run hook names.

    A title is named by a file change in its series since the last run, and every title is when that history can't
    all be read. A history read that names no title leaves the hook's titles named. Either setting that turns the
    history read off skips its boot step, and with the ignore flag every title is checked anyway.
    """

    @pytest.mark.parametrize(
        ("situation", "unchanged", "note"),
        [
            pytest.param(_Situation(), [2], "1 title to check again", id="touched"),
            pytest.param(_Situation(imported_into=99), [1, 2], "none", id="not-in-the-library"),
            pytest.param(
                _Situation(imported_into=99, checkpoint=_GAP),
                [],
                "history gap - checking every title again",
                id="history-gap",
            ),
            pytest.param(_Situation(detect_activity=False), [1, 2], None, id="detection-off"),
            pytest.param(_Situation(ignore_update_times=True), [], None, id="ignoring-update-times"),
            pytest.param(
                _Situation(imported_into=None, check_again=frozenset({2})), [1], "none", id="start-of-run-hook"
            ),
        ],
    )
    def test_a_run_checks_again_the_titles_its_history_read_and_start_of_run_hook_name(
        self, situation: _Situation, unchanged: list[int], note: str | None
    ) -> None:
        config = make_config(
            url="http://sonarr",
            api_key="key",
            detect_arr_activity=situation.detect_activity,
            ignore_seadex_update_times=situation.ignore_update_times,
        )
        history = [] if situation.imported_into is None else [sonarr_file_imported(situation.imported_into)]
        cache = _cache_left_unchanged(situation.checkpoint)
        deps = make_run_deps(config=config, cache_store=cache, seadex=FakeSeaDexSource(_ENTRIES))
        recording = install_recording_hub()

        drive_arr_run(deps, lambda services: _CachedLibrary(services, history, situation.check_again))

        assert _shown_unchanged(recording) == unchanged
        assert _activity_note(recording) == note

    def test_a_checkpoint_held_by_an_unreachable_seadex_replays_its_history_on_the_next_healthy_run(self) -> None:
        # The run that found SeaDex unreachable checked nothing, so the file imported into series 1 still drives a
        # check on the next run. Both are real runs, as only a real run's stores are saved.
        cache = _cache_left_unchanged()
        history = [sonarr_file_imported(1)]
        unreachable = make_run_deps(
            cache_store=cache, qbit=FakeQbit().client(), seadex=FakeSeaDexSource(unreachable=True)
        )
        drive_arr_run(unreachable, lambda services: _CachedLibrary(services, history, frozenset()))
        recording = install_recording_hub()

        healthy = make_run_deps(cache_store=cache, qbit=FakeQbit().client(), seadex=FakeSeaDexSource(_ENTRIES))
        drive_arr_run(healthy, lambda services: _CachedLibrary(services, history, frozenset()))

        assert _shown_unchanged(recording) == [2]
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint(history[0].date, history[0].id)


_OLDER_SELECTION = "an-older-selection"
"""The selection hash an earlier full run stored, under selection settings changed since."""


class _Series3(FakeStrategy):
    """A library of series 3 mapping title 11, its history the one import.

    `services` holds each title past the cap, and with `fails` the title's check raises.
    """

    def __init__(self, services: ArrRunServices | None = None, *, fails: bool = False) -> None:
        super().__init__(
            items=[FakeArrItem(item_id=3)],
            anilist_ids={11: TitleMapping(anilist_id=11)},
            services=services,
            history=[sonarr_file_imported(3)],
        )
        self._fails = fails

    @override
    def check_title(self, item: FakeArrItem, anilist_id: int, mapping: TitleMapping) -> None:
        if self._fails:
            raise RuntimeError("check failed")
        super().check_title(item, anilist_id, mapping)


@dataclass(frozen=True, kw_only=True)
class _Coverage:
    """How far a run's scan covered the library.

    That is its narrowing to one series, the sources unreachable or holding a title, and a failed check.
    """

    item_id: int | None = None
    seadex_unreachable: bool = False
    anilist_unreachable: bool = False
    held_past_the_cap: bool = False
    title_fails: bool = False


class TestSelectionRecheck:
    """A run over changed selection settings checks the cached titles again, and a full scan stores the new settings.

    The note says why the titles are checked again. A single-item run checks only its series and can't store the
    settings, so the note would overstate it and recur. With the ignore flag every title is checked anyway. The run's
    activity checkpoint is stored with the settings or not at all.
    """

    _NOTE = "Selection settings changed - checking cached titles again"

    @pytest.mark.parametrize(
        ("changed", "item_id", "ignore_update_times", "announced"),
        [
            pytest.param(True, None, False, True, id="changed"),
            pytest.param(False, None, False, False, id="unchanged"),
            pytest.param(True, 3, False, False, id="single-item"),
            pytest.param(True, None, True, False, id="ignoring-update-times"),
        ],
    )
    def test_a_changed_selection_is_announced_only_on_a_full_run_that_reads_update_times(
        self, changed: bool, item_id: int | None, ignore_update_times: bool, announced: bool
    ) -> None:
        config = make_config(url="http://sonarr", api_key="key", ignore_seadex_update_times=ignore_update_times)
        cache = FakeCacheStore()
        cache.store_selection_hash(Arr.SONARR, _OLDER_SELECTION if changed else config.selection_hash())
        recording = install_recording_hub()

        drive_arr_run(make_run_deps(config=config, cache_store=cache), lambda _: _Series3(), item_id=item_id)

        assert (self._NOTE in diagnostic_messages(recording)) is announced

    @pytest.mark.parametrize(
        ("coverage", "stored"),
        [
            pytest.param(_Coverage(), True, id="full"),
            pytest.param(_Coverage(item_id=3), False, id="single-item"),
            pytest.param(_Coverage(seadex_unreachable=True), False, id="no-seadex"),
            # A title AniList left unread isn't cached, so the next run checks it whatever this one stored.
            pytest.param(_Coverage(anilist_unreachable=True), True, id="no-anilist"),
            # So is a title held past the cap.
            pytest.param(_Coverage(held_past_the_cap=True), True, id="held-past-the-cap"),
            # And one whose check failed: blocking the stores, one flaky title would have the library checked again
            # every run.
            pytest.param(_Coverage(title_fails=True), True, id="a-title-failed"),
        ],
    )
    def test_a_full_scan_seadex_answered_stores_the_selection_and_the_checkpoint(
        self, coverage: _Coverage, stored: bool
    ) -> None:
        # Left unstored, the next run checks the cached titles again and replays the same activity. The run is a real
        # one, as only a real run's stores are saved.
        cache = FakeCacheStore()
        cache.store_selection_hash(Arr.SONARR, _OLDER_SELECTION)
        deps = make_run_deps(
            cache_store=cache,
            qbit=FakeQbit().client(),
            seadex=FakeSeaDexSource(unreachable=coverage.seadex_unreachable),
            anilist_client=ScriptedAniListClient(failing=frozenset({11} if coverage.anilist_unreachable else ())),
        )

        drive_arr_run(
            deps,
            lambda services: _Series3(services if coverage.held_past_the_cap else None, fails=coverage.title_fails),
            item_id=coverage.item_id,
        )

        selection_stored = not cache.selection_changed(Arr.SONARR, deps.config.selection_hash())
        checkpoint_stored = cache.get_history_checkpoint(Arr.SONARR) is not None
        assert selection_stored is stored
        assert checkpoint_stored is stored
