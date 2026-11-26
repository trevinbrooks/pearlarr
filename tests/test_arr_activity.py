# pyright: strict
"""The arr-side activity read (`ArrActivityReader`), and what a run checks again by it.

The reader asks the arr's history for the records since its checkpoint, from a little over a day before it, as the
two clocks may differ. With no checkpoint it asks from as far back as it reaches. A checkpoint from further back
than that, or one it can't read, has it ask from there too and have every title checked again, since the history it
skips may hide a change. It skips the records it has already seen and names the items a file change touched,
pearlarr's own grabs among them. A failed read names nothing and leaves the checkpoint as it was.

The run half runs whole Sonarr runs over a library an earlier run cached unchanged, reading the titles the run
showed unchanged and the note its activity read ended its boot step on. What a run stores of the read for the next
run is in `test_arr_run_scan`.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import override

import pytest

from pearlarr.arr_activity import (
    HISTORY_MAX_LOOKBACK_DAYS,
    HISTORY_QUERY_OVERLAP_HOURS,
    ArrActivityReader,
    format_history_date,
    parse_history_date,
)
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
    drive_arr_run,
    make_config,
    make_entry_record,
    make_logger,
    make_run_deps,
    sonarr_file_imported,
)
from .fakes import CaptureHandler, FakeArrItem, FakeStrategy, RecordingHub, install_recording_hub
from .wait_fakes import FakeQbit

_NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)


def _rec(
    record_id: int,
    *,
    item_id: int = 1,
    event: str = "downloadFolderImported",
    download_id: str | None = None,
    reason: str | None = None,
    date: str = "2026-07-06T10:00:00Z",
) -> HistoryRecord:
    return HistoryRecord(
        id=record_id,
        date=date,
        item_id=item_id,
        event_type=event,
        download_id=download_id,
        reason=reason,
    )


class _Fetch:
    """A recording fake `history_since`: scripted records, dates recorded."""

    def __init__(self, records: list[HistoryRecord] | None) -> None:
        self.records = records
        self.calls: list[str] = []

    def __call__(self, date: str) -> list[HistoryRecord] | None:
        self.calls.append(date)
        return self.records


def _activity_reader(cache: FakeCacheStore | None = None) -> tuple[ArrActivityReader, FakeCacheStore]:
    cache = cache if cache is not None else FakeCacheStore()
    return ArrActivityReader(Arr.SONARR, cache, make_logger()), cache


class TestDateHelpers:
    """`parse_history_date`/`format_history_date` round-trip.

    Naive stamps assume UTC, offsets convert, garbage parses to `None`.
    """

    def test_round_trip(self) -> None:
        assert parse_history_date(format_history_date(_NOW)) == _NOW

    def test_naive_stamp_is_assumed_utc(self) -> None:
        assert parse_history_date("2026-07-06T12:00:00") == _NOW

    def test_offset_stamp_converts_to_utc(self) -> None:
        assert parse_history_date("2026-07-06T14:00:00+02:00") == _NOW

    def test_garbage_is_none(self) -> None:
        assert parse_history_date("not-a-date") is None


class TestRead:
    """`ArrActivityReader.read` windows/dedups history queries off the stored checkpoint, clamping bad or future dates to a full rescan, and filters touched ids by event type."""

    def test_bootstrap_queries_lookback_window_and_stores_max_id_checkpoint(self) -> None:
        activity, cache = _activity_reader()
        fetch = _Fetch([_rec(5, item_id=3), _rec(7, item_id=4, date="2026-07-06T11:00:00Z")])

        scan = activity.read(fetch, now=_NOW)

        assert scan.touched == frozenset({3, 4})
        assert scan.rescan_all is False
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T11:00:00Z", 7)

    def test_id_cursor_dedup_skips_already_seen_records(self) -> None:
        activity, cache = _activity_reader()
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(_NOW), 10))
        fetch = _Fetch([_rec(9, item_id=1), _rec(10, item_id=1), _rec(11, item_id=2)])

        assert activity.read(fetch, now=_NOW).touched == frozenset({2})

    def test_query_date_overlaps_behind_the_checkpoint(self) -> None:
        activity, cache = _activity_reader()
        since = _NOW - timedelta(hours=2)
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(since), 1))
        fetch = _Fetch([])

        assert activity.read(fetch, now=_NOW).rescan_all is False
        assert fetch.calls == [format_history_date(since - timedelta(hours=HISTORY_QUERY_OVERLAP_HOURS))]

    def test_checkpoint_beyond_lookback_clamps_and_signals_rescan(self) -> None:
        # A gap the clamp truncates: the un-queried stretch may hide changes,
        # so the scan demands a full re-check while the cursor still advances.
        activity, cache = _activity_reader()
        since = _NOW - timedelta(days=40)
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(since), 1))
        fetch = _Fetch([_rec(50, item_id=3, date="2026-07-06T09:00:00Z")])

        scan = activity.read(fetch, now=_NOW)

        assert scan.rescan_all is True
        assert scan.touched == frozenset()
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T09:00:00Z", 50)

    def test_gap_with_empty_window_still_signals_rescan(self) -> None:
        # No fresh records to advance the cursor: coverage stays broken, so the
        # rescan repeats next run until history produces a new stamp.
        activity, cache = _activity_reader()
        since = _NOW - timedelta(days=40)
        stored = HistoryCheckpoint(format_history_date(since), 1)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = activity.read(_Fetch([]), now=_NOW)

        assert scan.rescan_all is True
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_future_checkpoint_date_clamped_to_now(self) -> None:
        activity, cache = _activity_reader()
        cache.put_history_checkpoint(
            Arr.SONARR,
            HistoryCheckpoint(format_history_date(_NOW + timedelta(hours=1)), 1),
        )
        fetch = _Fetch([])

        assert activity.read(fetch, now=_NOW).rescan_all is False
        assert fetch.calls == [format_history_date(_NOW - timedelta(hours=HISTORY_QUERY_OVERLAP_HOURS))]

    def test_unparseable_checkpoint_date_replays_lookback_and_rescans(self) -> None:
        activity, cache = _activity_reader()
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("garbage", 1))
        fetch = _Fetch([])

        scan = activity.read(fetch, now=_NOW)

        assert scan.rescan_all is True
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]

    def test_event_type_filtering(self) -> None:
        activity, _ = _activity_reader()
        fetch = _Fetch(
            [
                _rec(1, item_id=1, event="grabbed"),
                _rec(2, item_id=2, event="episodeFileRenamed"),
                _rec(3, item_id=3, event="downloadFailed"),
                _rec(4, item_id=4, event="downloadIgnored"),
                _rec(5, item_id=5, event="downloadFolderImported"),
                _rec(6, item_id=6, event="seriesFolderImported"),
                _rec(7, item_id=7, event="movieFolderImported"),
                _rec(8, item_id=8, event="episodeFileDeleted"),
                _rec(9, item_id=9, event="movieFileDeleted", reason="MissingFromDisk"),
            ],
        )

        assert activity.read(fetch, now=_NOW).touched == frozenset({5, 6, 7, 8, 9})

    def test_upgrade_reason_delete_suppressed_both_casings(self) -> None:
        activity, cache = _activity_reader()
        fetch = _Fetch(
            [
                _rec(1, item_id=1, event="episodeFileDeleted", reason="upgrade"),
                _rec(2, item_id=2, event="episodeFileDeleted", reason="Upgrade"),
            ],
        )

        assert activity.read(fetch, now=_NOW).touched == frozenset()
        # The window was non-empty, so the cursor still advances on commit.
        activity.stage_checkpoint()
        checkpoint = cache.get_history_checkpoint(Arr.SONARR)
        assert checkpoint is not None
        assert checkpoint.last_id == 2

    @pytest.mark.parametrize(
        ("arr", "event"),
        [(Arr.SONARR, "downloadFolderImported"), (Arr.RADARR, "movieFolderImported")],
    )
    def test_pearlarrs_grab_import_touches_its_item(self, arr: Arr, event: str) -> None:
        # A pending grab of pearlarr's is set up so a re-added filter on pearlarr's grabs fails here: the arr may import
        # only part of a grab.
        cache = FakeCacheStore()
        cache.put_pending_import_row(arr, "beef01", {"infohash": "beef01"})
        activity = ArrActivityReader(arr, cache, make_logger())
        fetch = _Fetch(
            [
                _rec(1, item_id=1, event=event, download_id="BEEF01"),  # pending grab
                _rec(2, item_id=2, event=event, download_id="cafe00"),  # someone else's grab
            ],
        )

        assert activity.read(fetch, now=_NOW).touched == frozenset({1, 2})

    def test_committed_pearlarrs_grab_import_is_not_touched_again(self) -> None:
        # The committed cursor dedups the overlap's replay, so the next run re-checks nothing.
        activity, cache = _activity_reader()
        cache.put_pending_import_row(Arr.SONARR, "beef01", {"infohash": "beef01"})
        records = [_rec(1, item_id=2, download_id="BEEF01")]

        assert activity.read(_Fetch(records), now=_NOW).touched == frozenset({2})
        activity.stage_checkpoint()
        next_run, _ = _activity_reader(cache)
        assert next_run.read(_Fetch(records), now=_NOW).touched == frozenset()

    def test_item_id_zero_dropped(self) -> None:
        activity, _ = _activity_reader()
        fetch = _Fetch([_rec(1, item_id=0), _rec(2, item_id=6)])

        assert activity.read(fetch, now=_NOW).touched == frozenset({6})

    def test_fetch_failure_fails_open(self) -> None:
        logger = make_logger()
        logger.setLevel(logging.DEBUG)
        capture = CaptureHandler()
        logger.addHandler(capture)
        cache = FakeCacheStore()
        activity = ArrActivityReader(Arr.SONARR, cache, logger)
        try:
            scan = activity.read(_Fetch(None), now=_NOW)
        finally:
            logger.removeHandler(capture)

        assert scan.touched == frozenset()
        assert scan.rescan_all is False
        # The fetch helper (arr_http) owns the operator-facing warning. The reader
        # leaves only a debug line, so one failure never warns twice.
        assert not any(r.levelno == logging.WARNING for r in capture.records)
        assert any(r.levelno == logging.DEBUG for r in capture.records)
        # No pending checkpoint either: commit is a no-op.
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) is None

    def test_fetch_failure_on_a_gap_keeps_the_gap_detectable(self) -> None:
        # Fail-open beats rescan: nothing this run, but the untouched checkpoint
        # re-detects the gap next run.
        activity, cache = _activity_reader()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(days=40)), 1)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = activity.read(_Fetch(None), now=_NOW)

        assert scan.rescan_all is False
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_empty_response_leaves_checkpoint_unchanged(self) -> None:
        activity, cache = _activity_reader()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        assert activity.read(_Fetch([]), now=_NOW).touched == frozenset()
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_stale_only_window_leaves_checkpoint_unchanged(self) -> None:
        activity, cache = _activity_reader()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        assert activity.read(_Fetch([_rec(11), _rec(12)]), now=_NOW).touched == frozenset()
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_empty_date_record_does_not_stash_a_checkpoint(self) -> None:
        # An empty stored date would collapse the next window to the overlap.
        # keep the old cursor and let the id dedup drop the re-delivery.
        activity, cache = _activity_reader()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = activity.read(_Fetch([_rec(20, item_id=5, date="")]), now=_NOW)

        assert scan.touched == frozenset({5})
        activity.stage_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored


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
