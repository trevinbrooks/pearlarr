# pyright: strict
# pyright: reportPrivateUsage=false
# The run-loop section reads the services hub's private dirty set (the wiring
# under test); strict re-flags that and the repo disables it for tests.
"""The arr-side activity scan (`ArrActivityMonitor`) and its run-loop wiring.

Unit half: the checkpoint window math (lookback bootstrap / overlap / clamp),
the id-cursor dedup, the coverage-gap rescan signal, the event + upgrade-reason
+ own-hash + item-id filters, and the fail-open contract. Run-loop half (reusing
`test_run_finalize`'s `_engine` harness): touched items flow into
`RunServices._dirty_al_ids` (all items on a gap), the checkpoint is staged
only on full-library, non-capped, non-outage runs (held drift signals replay
on the next healthy run), and either config toggle disables the fetch
entirely.
"""

import logging
from datetime import UTC, datetime, timedelta

from pearlarr.arr_activity import (
    HISTORY_MAX_LOOKBACK_DAYS,
    HISTORY_QUERY_OVERLAP_HOURS,
    ArrActivityMonitor,
    format_history_date,
    parse_history_date,
)
from pearlarr.boot_flow import BootFlow
from pearlarr.cache import HistoryCheckpoint
from pearlarr.config import AppConfig, Arr
from pearlarr.mappings import MappingEntry
from pearlarr.run_loop import RunLoop
from pearlarr.seadex_types import HistoryRecord

from .builders import FakeCacheStore, make_config, make_logger
from .fakes import CaptureHandler, FakeArrItem, FakeStrategy
from .test_run_finalize import _engine, _FakeGateway, _FinalizeRecorder

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
    """A recording `history_since` stand-in: scripted records, dates recorded."""

    def __init__(self, records: list[HistoryRecord] | None) -> None:
        self.records = records
        self.calls: list[str] = []

    def __call__(self, date: str) -> list[HistoryRecord] | None:
        self.calls.append(date)
        return self.records


def _monitor(cache: FakeCacheStore | None = None) -> tuple[ArrActivityMonitor, FakeCacheStore]:
    cache = cache if cache is not None else FakeCacheStore()
    return ArrActivityMonitor(Arr.SONARR, cache, make_logger()), cache


class TestDateHelpers:
    """`parse_history_date`/`format_history_date` round-trip; naive stamps assume UTC, offsets convert, garbage parses to `None`."""

    def test_round_trip(self) -> None:
        assert parse_history_date(format_history_date(_NOW)) == _NOW

    def test_naive_stamp_is_assumed_utc(self) -> None:
        assert parse_history_date("2026-07-06T12:00:00") == _NOW

    def test_offset_stamp_converts_to_utc(self) -> None:
        assert parse_history_date("2026-07-06T14:00:00+02:00") == _NOW

    def test_garbage_is_none(self) -> None:
        assert parse_history_date("not-a-date") is None


class TestScan:
    """`ArrActivityMonitor.scan` windows/dedups history queries off the stored checkpoint, clamping bad or future dates to a full rescan, and filters touched ids by event type."""

    def test_bootstrap_queries_lookback_window_and_stores_max_id_checkpoint(self) -> None:
        monitor, cache = _monitor()
        fetch = _Fetch([_rec(5, item_id=3), _rec(7, item_id=4, date="2026-07-06T11:00:00Z")])

        scan = monitor.scan(fetch, now=_NOW)

        assert scan.touched == frozenset({3, 4})
        assert scan.rescan_all is False
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T11:00:00Z", 7)

    def test_id_cursor_dedup_skips_already_seen_records(self) -> None:
        monitor, cache = _monitor()
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(_NOW), 10))
        fetch = _Fetch([_rec(9, item_id=1), _rec(10, item_id=1), _rec(11, item_id=2)])

        assert monitor.scan(fetch, now=_NOW).touched == frozenset({2})

    def test_query_date_overlaps_behind_the_checkpoint(self) -> None:
        monitor, cache = _monitor()
        since = _NOW - timedelta(hours=2)
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(since), 1))
        fetch = _Fetch([])

        assert monitor.scan(fetch, now=_NOW).rescan_all is False
        assert fetch.calls == [format_history_date(since - timedelta(hours=HISTORY_QUERY_OVERLAP_HOURS))]

    def test_checkpoint_beyond_lookback_clamps_and_signals_rescan(self) -> None:
        # A gap the clamp truncates: the un-queried stretch may hide changes,
        # so the scan demands a full re-check while the cursor still advances.
        monitor, cache = _monitor()
        since = _NOW - timedelta(days=40)
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(since), 1))
        fetch = _Fetch([_rec(50, item_id=3, date="2026-07-06T09:00:00Z")])

        scan = monitor.scan(fetch, now=_NOW)

        assert scan.rescan_all is True
        assert scan.touched == frozenset()
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T09:00:00Z", 50)

    def test_gap_with_empty_window_still_signals_rescan(self) -> None:
        # No fresh records to advance the cursor: coverage stays broken, so the
        # rescan repeats next pass until history produces a new stamp.
        monitor, cache = _monitor()
        since = _NOW - timedelta(days=40)
        stored = HistoryCheckpoint(format_history_date(since), 1)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = monitor.scan(_Fetch([]), now=_NOW)

        assert scan.rescan_all is True
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_future_checkpoint_date_clamped_to_now(self) -> None:
        monitor, cache = _monitor()
        cache.put_history_checkpoint(
            Arr.SONARR,
            HistoryCheckpoint(format_history_date(_NOW + timedelta(hours=1)), 1),
        )
        fetch = _Fetch([])

        assert monitor.scan(fetch, now=_NOW).rescan_all is False
        assert fetch.calls == [format_history_date(_NOW - timedelta(hours=HISTORY_QUERY_OVERLAP_HOURS))]

    def test_unparseable_checkpoint_date_replays_lookback_and_rescans(self) -> None:
        monitor, cache = _monitor()
        cache.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint("garbage", 1))
        fetch = _Fetch([])

        scan = monitor.scan(fetch, now=_NOW)

        assert scan.rescan_all is True
        assert fetch.calls == [format_history_date(_NOW - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS))]

    def test_event_type_filtering(self) -> None:
        monitor, _ = _monitor()
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

        assert monitor.scan(fetch, now=_NOW).touched == frozenset({5, 6, 7, 8, 9})

    def test_upgrade_reason_delete_suppressed_both_casings(self) -> None:
        monitor, cache = _monitor()
        fetch = _Fetch(
            [
                _rec(1, item_id=1, event="episodeFileDeleted", reason="upgrade"),
                _rec(2, item_id=2, event="episodeFileDeleted", reason="Upgrade"),
            ],
        )

        assert monitor.scan(fetch, now=_NOW).touched == frozenset()
        # The window was non-empty, so the cursor still advances on commit.
        monitor.commit_checkpoint()
        checkpoint = cache.get_history_checkpoint(Arr.SONARR)
        assert checkpoint is not None
        assert checkpoint.last_id == 2

    def test_own_hash_suppression_is_case_insensitive(self) -> None:
        cache = FakeCacheStore()
        cache.update_cache(Arr.SONARR, 7, {"torrent_hashes": ["ABCDEF", None]})
        cache.put_pending(Arr.SONARR, "beef01", {"series_id": 7})
        monitor, _ = _monitor(cache)
        fetch = _Fetch(
            [
                _rec(1, item_id=1, download_id="abcdef"),  # remembered grab
                _rec(2, item_id=2, download_id="BEEF01"),  # pending-imports-only hash
                _rec(3, item_id=3, download_id="cafe00"),  # someone else's grab
                _rec(4, item_id=4, download_id=None),  # no hash: kept
            ],
        )

        assert monitor.scan(fetch, now=_NOW).touched == frozenset({3, 4})

    def test_item_id_zero_dropped(self) -> None:
        monitor, _ = _monitor()
        fetch = _Fetch([_rec(1, item_id=0), _rec(2, item_id=6)])

        assert monitor.scan(fetch, now=_NOW).touched == frozenset({6})

    def test_fetch_failure_fails_open(self) -> None:
        logger = make_logger()
        logger.setLevel(logging.DEBUG)
        capture = CaptureHandler()
        logger.addHandler(capture)
        cache = FakeCacheStore()
        monitor = ArrActivityMonitor(Arr.SONARR, cache, logger)
        try:
            scan = monitor.scan(_Fetch(None), now=_NOW)
        finally:
            logger.removeHandler(capture)

        assert scan.touched == frozenset()
        assert scan.rescan_all is False
        # The fetch helper (arr_http) owns the user-facing warning; the monitor
        # leaves only a debug breadcrumb, so one failure never warns twice.
        assert not any(r.levelno == logging.WARNING for r in capture.records)
        assert any(r.levelno == logging.DEBUG for r in capture.records)
        # No pending checkpoint either: commit is a no-op.
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) is None

    def test_fetch_failure_on_a_gap_keeps_the_gap_detectable(self) -> None:
        # Fail-open beats rescan: nothing this run, but the untouched checkpoint
        # re-detects the gap next pass.
        monitor, cache = _monitor()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(days=40)), 1)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = monitor.scan(_Fetch(None), now=_NOW)

        assert scan.rescan_all is False
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_empty_response_leaves_checkpoint_unchanged(self) -> None:
        monitor, cache = _monitor()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        assert monitor.scan(_Fetch([]), now=_NOW).touched == frozenset()
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_stale_only_window_leaves_checkpoint_unchanged(self) -> None:
        monitor, cache = _monitor()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        assert monitor.scan(_Fetch([_rec(11), _rec(12)]), now=_NOW).touched == frozenset()
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored

    def test_empty_date_record_does_not_stash_a_checkpoint(self) -> None:
        # An empty stored date would collapse the next window to the overlap;
        # keep the old cursor and let the id dedup absorb the re-delivery.
        monitor, cache = _monitor()
        stored = HistoryCheckpoint(format_history_date(_NOW - timedelta(hours=3)), 12)
        cache.put_history_checkpoint(Arr.SONARR, stored)

        scan = monitor.scan(_Fetch([_rec(20, item_id=5, date="")]), now=_NOW)

        assert scan.touched == frozenset({5})
        monitor.commit_checkpoint()
        assert cache.get_history_checkpoint(Arr.SONARR) == stored


class TestRunLoopActivityWiring:
    """The run loop's activity block, driven through the `_engine` harness."""

    @staticmethod
    def _strategy(
        *,
        history: list[HistoryRecord] | None = None,
        process_returns: bool = False,
    ) -> FakeStrategy:
        return FakeStrategy(
            items=[FakeArrItem(item_id=3, title="A")],
            anilist_ids={11: MappingEntry(anilist_id=11)},
            process_returns=process_returns,
            history=history,
        )

    @staticmethod
    def _run(
        strategy: FakeStrategy,
        logger: logging.Logger,
        *,
        item_id: int | None = None,
        config: AppConfig | None = None,
    ) -> RunLoop:
        engine = _engine(_FinalizeRecorder(), logger, config=config)
        engine.run_sync(strategy, item_id=item_id, dry_run=True, boot=BootFlow())
        return engine

    def test_touched_item_marks_its_anilist_ids_dirty(self, logger: logging.Logger) -> None:
        strategy = self._strategy(history=[_rec(1, item_id=3)])
        engine = self._run(strategy, logger)

        assert engine._services._dirty_al_ids == {11}
        assert len(strategy.history_calls) == 1

    def test_untouched_items_stay_clean(self, logger: logging.Logger) -> None:
        strategy = self._strategy(history=[_rec(1, item_id=99)])  # not in the library
        engine = self._run(strategy, logger)

        assert engine._services._dirty_al_ids == set()

    def test_history_gap_marks_everything_dirty(self, logger: logging.Logger) -> None:
        # Same untouched history as above, but a checkpoint beyond the lookback:
        # broken coverage re-checks the whole library.
        strategy = self._strategy(history=[_rec(50, item_id=99)])
        engine = _engine(_FinalizeRecorder(), logger)
        stale = datetime.now(UTC) - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS + 10)
        engine.cache_store.put_history_checkpoint(Arr.SONARR, HistoryCheckpoint(format_history_date(stale), 1))
        engine.run_sync(strategy, item_id=None, dry_run=True, boot=BootFlow())

        assert engine._services._dirty_al_ids == {11}

    def test_full_run_stages_the_checkpoint(self, logger: logging.Logger) -> None:
        strategy = self._strategy(history=[_rec(1, item_id=3, date="2026-07-06T10:00:00Z")])
        engine = self._run(strategy, logger)

        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T10:00:00Z", 1)

    def test_single_item_run_does_not_advance_the_checkpoint(self, logger: logging.Logger) -> None:
        strategy = self._strategy(history=[_rec(1, item_id=3)])
        engine = self._run(strategy, logger, item_id=3)

        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is None

    def test_capped_run_does_not_advance_the_checkpoint(self, logger: logging.Logger) -> None:
        strategy = self._strategy(history=[_rec(1, item_id=3)], process_returns=True)
        engine = self._run(strategy, logger)

        assert engine.cache_store.get_history_checkpoint(Arr.SONARR) is None

    @staticmethod
    def _outage_engine(logger: logging.Logger, cache: FakeCacheStore) -> RunLoop:
        gateway = _FakeGateway()
        gateway.outage = True
        return _engine(_FinalizeRecorder(), logger, seadex=gateway, cache_store=cache)

    def test_outage_run_holds_the_checkpoint(self, logger: logging.Logger) -> None:
        # A SeaDex-outage run skips every lookup, so committing would consume
        # drift events the run never acted on (the 2026-07-15 incident shape).
        cache = FakeCacheStore()
        engine = self._outage_engine(logger, cache)
        engine.run_sync(self._strategy(history=[_rec(1, item_id=3)]), item_id=None, dry_run=True, boot=BootFlow())

        assert engine._services._dirty_al_ids == {11}  # detection itself still ran
        assert cache.get_history_checkpoint(Arr.SONARR) is None

    def test_consecutive_outage_runs_keep_holding(self, logger: logging.Logger) -> None:
        cache = FakeCacheStore()
        for _ in range(2):
            engine = self._outage_engine(logger, cache)
            engine.run_sync(self._strategy(history=[_rec(1, item_id=3)]), item_id=None, dry_run=True, boot=BootFlow())

        assert cache.get_history_checkpoint(Arr.SONARR) is None

    def test_held_checkpoint_replays_dirty_on_the_next_healthy_run(self, logger: logging.Logger) -> None:
        # Outage run first: dirty derived but unactionable, checkpoint held.
        cache = FakeCacheStore()
        outage = self._outage_engine(logger, cache)
        outage.run_sync(self._strategy(history=[_rec(1, item_id=3)]), item_id=None, dry_run=True, boot=BootFlow())

        # Healthy run over the SAME store: the held cursor replays the same
        # records through the id dedup, re-derives the dirty mark, and commits.
        healthy = _engine(_FinalizeRecorder(), logger, cache_store=cache)
        healthy.run_sync(
            self._strategy(history=[_rec(1, item_id=3, date="2026-07-06T10:00:00Z")]),
            item_id=None,
            dry_run=True,
            boot=BootFlow(),
        )

        assert healthy._services._dirty_al_ids == {11}
        assert cache.get_history_checkpoint(Arr.SONARR) == HistoryCheckpoint("2026-07-06T10:00:00Z", 1)

    def test_detect_toggle_off_skips_the_fetch(self, logger: logging.Logger) -> None:
        strategy = self._strategy()
        self._run(strategy, logger, config=make_config(detect_arr_activity=False))

        assert strategy.history_calls == []

    def test_ignore_update_times_skips_the_fetch(self, logger: logging.Logger) -> None:
        strategy = self._strategy()
        self._run(strategy, logger, config=make_config(ignore_seadex_update_times=True))

        assert strategy.history_calls == []
