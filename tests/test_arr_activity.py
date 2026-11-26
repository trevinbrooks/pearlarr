# pyright: strict
"""The arr-side activity read (`ArrActivityReader`).

The reader asks the arr's history for the records since its checkpoint, from a little over a day before it, as the
two clocks may differ. With no checkpoint it asks from as far back as it reaches. A checkpoint from further back
than that, or one it can't read, has it ask from there too and have every title checked again, since the history it
skips may hide a change. It skips the records it has already seen and names the items a file change touched,
pearlarr's own grabs among them. A failed read names nothing and leaves the checkpoint as it was.

What a run checks again by the read, and stores of it for the next run, is in `test_arr_run_recheck`.
"""

import logging
from datetime import UTC, datetime, timedelta

import pytest

from pearlarr.arr_activity import (
    HISTORY_MAX_LOOKBACK_DAYS,
    HISTORY_QUERY_OVERLAP_HOURS,
    ArrActivityReader,
    format_history_date,
    parse_history_date,
)
from pearlarr.arr_models import HistoryRecord
from pearlarr.cache import HistoryCheckpoint
from pearlarr.config import Arr

from .builders import (
    FakeCacheStore,
    make_logger,
)
from .fakes import CaptureHandler

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
