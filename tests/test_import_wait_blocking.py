# pyright: strict
"""Tests for the blocking end pass, which waits on and imports every pending import.

They cover how an import is confirmed, what moves the import deadline, how a row ends without an import, an
interrupt, how the rows tell this run's grabs from carried-over ones, and the download bars.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, override

import pytest
import qbittorrentapi

from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.output import SPARK_SAMPLES, Diagnostic, Phase, Severity
from pearlarr.wait_types import AttemptKind, EffectStatus, ImportProgress, Outcome, PauseReason

from .builders import SEP, FakeCacheStore, attempt_result, pending_import
from .fakes import FakeClock, diagnostic_messages, install_recording_hub
from .wait_fakes import (
    CategoryQbit,
    FakeQbit,
    FakeTorrent,
    QbitStep,
    RecordingStrategy,
    RecordingWaitView,
    make_orchestration_manager,
    run_blocking_wait,
    stored_awaiting_cleanup,
)


class TestBlockingWaitImport:
    """Each finished torrent imports from its own path, confirmed by a poll or by a progress read between polls."""

    def test_two_torrents_each_import_from_their_own_path(self) -> None:
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=True)])
        qbit = FakeQbit(
            {
                "fast": [FakeTorrent(is_complete=True, content_path="/fast")],
                "slow": [FakeTorrent(progress=0.5), FakeTorrent(is_complete=True, content_path="/slow")],
            },
        )
        pending_imports = [pending_import(infohash="fast"), pending_import(infohash="slow")]
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            grabbed=pending_imports,
            clock=FakeClock(step=30),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert [(call.infohash, call.content_path) for call in strategy.import_calls] == [
            ("fast", "/fast"),
            ("slow", "/slow"),
        ]
        assert [view.final(key).outcome for key in ("fast", "slow")] == [Outcome.IMPORTED, Outcome.IMPORTED]
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    def test_progress_poll_fills_bar_and_promotes_before_the_next_poll(self) -> None:
        # Between polls the cheap progress poll fills the "files inserted" bar
        # as files get done and promotes the row to IMPORTED the instant all are done
        # - no second import_completed poll, no RefreshMonitoredDownloads.
        strategy = RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True, file_map_episode_count=3)],
            progress=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),
                ImportProgress(3, 3, determinate=True),  # all done -> promote
            ],
        )
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            progress_poll_interval=5,
        )

        # Only ONE poll. The progress poll did the rest.
        assert len(strategy.import_calls) == 1
        assert [(t.import_done, t.import_total) for t in view.seen("h", Phase.IMPORTING)] == [(0, 3), (1, 3), (2, 3)]
        assert view.final("h").outcome is Outcome.IMPORTED
        assert stored_awaiting_cleanup(store) == {}

    def test_with_the_progress_refresh_off_only_the_next_poll_confirms_an_import(self) -> None:
        # A progress read between the polls would find every file in and finish the row before poll 2.
        strategy = RecordingStrategy(
            attempts=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True),
            ],
            progress=[ImportProgress(1, 1, determinate=True)],
        )
        _, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            progress_poll_interval=0,
        )

        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY, AttemptKind.ORDINARY]
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_a_failed_progress_read_is_skipped_and_the_next_poll_imports(self) -> None:
        strategy = RecordingStrategy(
            attempts=[
                attempt_result(all_done=False, command_issued=True),
                attempt_result(all_done=True),
            ],
            progress=[RuntimeError("progress boom")],
        )
        _, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            progress_poll_interval=5,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert [call.attempt_kind for call in strategy.import_calls] == [AttemptKind.ORDINARY, AttemptKind.ORDINARY]


class TestBlockingWaitImportDeadline:
    """What moves an importing row's deadline: a rise in done files moves it out, an import in flight pauses it.

    A pause holds the deadline only up to a cap, and a row that ends there reads as what held it: still importing
    or Sonarr busy.
    """

    def test_a_progress_poll_rise_re_anchors_the_deadline(self) -> None:
        # The cheap progress poll sees a file get done (1/3 -> 2/3) even though the
        # poll's counts stay indeterminate. That rise alone must push the
        # deadline out one more poll (t=90, not t=60).
        strategy = RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True)],
            progress=[
                ImportProgress(1, 3, determinate=True),
                ImportProgress(2, 3, determinate=True),  # the rise, at t=10
            ],
        )
        _, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            step=5,
            import_timeout=60,
            progress_poll_interval=5,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        # Polls at t=0/30/60/90: the t=60 poll is in-bound only because the
        # t=10 progress-poll rise re-anchored the deadline to fire at t=70.
        assert len(strategy.import_calls) == 4
        assert strategy.import_calls[-1].attempt_kind is AttemptKind.FINAL

    def test_final_attempt_rescued_by_a_same_poll_rise(self) -> None:
        # The forced deadline attempt itself reports a fresh rise (0/3 -> 1/3 at
        # t=60): the row must NOT finish that poll - the done file re-anchors
        # the deadline instead - and only the NEXT quiet deadline (t=120) ends it.
        strategy = RecordingStrategy(
            attempts=[
                attempt_result(
                    all_done=False,
                    command_issued=True,
                    imported_count=done,
                    file_map_episode_count=3,
                )
                for done in (0, 0, 1)
            ],
        )
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert stored_awaiting_cleanup(store) == {"h": False}
        # t=0/30 in-bound, t=60 forced-but-rescued, t=90 in-bound, t=120 forced.
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, False, True]

    def test_final_auto_import_waits_its_own_copy_out(self) -> None:
        # Two stalled polls, then the final attempt POSTs (ISSUED, paused):
        # the row is not finished "still importing" on that same poll, the
        # copy in flight keeps pausing the clock, and the done-check ends IMPORTED.
        strategy = RecordingStrategy(
            attempts=[
                attempt_result(all_done=False),
                attempt_result(all_done=False),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                attempt_result(all_done=True),
            ],
        )
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.IMPORTED
        assert stored_awaiting_cleanup(store) == {}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True, True, True]

    def test_fresh_post_pauses_once_per_row(self) -> None:
        # A POST whose files all fail inside Sonarr is re-POSTed every poll:
        # only the first pauses the clock, so the retry loop finishes at the
        # ordinary deadline (t=60), never going on to the cap.
        strategy = RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED)],
        )
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert stored_awaiting_cleanup(store) == {"h": False}
        assert [c.attempt_kind.at_deadline for c in strategy.import_calls] == [False, False, True]

    def test_pause_cap_reads_still_importing_for_an_import_pause(self) -> None:
        # Sonarr's own import (or an unproven one) still running at the cap:
        # no command of pearlarr's, yet the label says "still importing", not "not
        # ready". Ready 60s + cap (6x) 360s -> finished by t=420.
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=False, pause_reason=PauseReason.IMPORT)])
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.STILL_IMPORTING
        assert stored_awaiting_cleanup(store) == {"h": False}
        assert strategy.import_calls[-1].attempt_kind.at_deadline is True
        assert len(strategy.import_calls) == 15

    def test_pause_cap_reads_sonarr_busy_for_another_downloads_disk_command(self) -> None:
        # A foreign disk command holding the line to the cap: "sonarr busy",
        # left for the next run, never a false "still importing".
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=False, pause_reason=PauseReason.BUSY)])
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            import_timeout=60,
        )

        assert view.final("h").outcome is Outcome.SONARR_BUSY
        assert stored_awaiting_cleanup(store) == {"h": False}


class TestBlockingWaitEndsWithoutAnImport:
    """A download that never finishes or an attempt that raises ends its row, and the pending import stays stored."""

    @pytest.mark.parametrize(
        "reading",
        [
            pytest.param(FakeTorrent(progress=0.5), id="still-downloading"),
            # qBittorrent down for the whole wait still ends it at the download timeout.
            pytest.param(qbittorrentapi.APIConnectionError("down"), id="qbittorrent-unreachable"),
        ],
    )
    def test_a_download_that_never_finishes_times_out_and_is_left_for_the_next_run(self, reading: QbitStep) -> None:
        strategy = RecordingStrategy()
        qbit = CategoryQbit({"h": [reading]})
        store, view = run_blocking_wait(
            strategy,
            qbit,
            post_import_category="pearlarr-done",
            download_timeout=60,
        )

        assert strategy.import_calls == []
        assert view.final("h").outcome is Outcome.DOWNLOAD_TIMED_OUT
        assert stored_awaiting_cleanup(store) == {"h": False}
        assert qbit.set_category_calls == []

    def test_import_exception_is_swallowed_announced_and_pending_import_left(self) -> None:
        # A failing import (e.g. malformed Sonarr response) must NOT propagate
        # and abort _finish_arr_run's cache save: the failure lands as an ERROR
        # Diagnostic carrying its traceback and the pending import is left.
        strategy = RecordingStrategy(attempts=[RuntimeError("boom")])
        recording = install_recording_hub()
        store, view = run_blocking_wait(
            strategy,
            FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
        )

        assert view.final("h").outcome is Outcome.ATTEMPT_FAILED
        assert stored_awaiting_cleanup(store) == {"h": False}
        (error,) = recording.of_type(Diagnostic)
        assert error.severity is Severity.ERROR
        assert "Auto import failed" in error.message
        assert error.trace is not None


class _InterruptSleepClock(FakeClock):
    """A clock whose sleep raises KeyboardInterrupt (Ctrl-C mid-wait)."""

    @override
    def sleep(self, seconds: float) -> None:
        raise KeyboardInterrupt


class _StopSleepClock(FakeClock):
    """A clock whose sleep raises `SystemExit` (a SIGTERM stop mid-wait)."""

    @override
    def sleep(self, seconds: float) -> None:
        raise SystemExit(0)


class _InterruptedDropStore(FakeCacheStore):
    """A store where a Ctrl-C lands in the pending import drop."""

    @override
    def drop_pending_import_row(self, arr: Arr, infohash: str) -> None:
        raise KeyboardInterrupt


class _InterruptedPutStore(FakeCacheStore):
    """A store where a Ctrl-C lands in the write that marks a pending import awaiting cleanup."""

    @override
    def put_pending_import_row(self, arr: Arr, infohash: str, row: dict[str, Any]) -> None:
        if row.get("awaiting_cleanup"):
            raise KeyboardInterrupt
        super().put_pending_import_row(arr, infohash, row)


class TestBlockingWaitInterrupt:
    """A Ctrl-C or a stop mid-wait notes what is left, and an import whose store write it cut off is not counted."""

    @pytest.mark.parametrize(
        ("clock_type", "ending"),
        [
            pytest.param(_InterruptSleepClock, nullcontext(), id="ctrl-c-ends-the-wait-and-the-run-goes-on"),
            pytest.param(_StopSleepClock, pytest.raises(SystemExit), id="a-sigterm-stop-propagates"),
        ],
    )
    def test_an_interrupted_wait_notes_what_is_left(
        self, clock_type: type[FakeClock], ending: AbstractContextManager[object]
    ) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.3)]}),
            strategy=RecordingStrategy(),
            store=store,
            grabbed=[pending_import(infohash="h")],
            clock=clock_type(),
        )
        recording = install_recording_hub()

        with ending:
            mgr.run_blocking()

        assert stored_awaiting_cleanup(store) == {"h": False}
        assert diagnostic_messages(recording) == ["Wait interrupted - 1 left for the next run"]

    def test_interrupt_mid_poll_still_finishes_that_polls_finished_rows(self) -> None:
        # A torrent that finished in the SAME poll the interrupt lands in
        # must still reach the view (the except handler pushes one final snapshot), so
        # the console tally/outcome lines can never undercount the returned WaitResult.
        # h1 is gone, so it finishes on its first poll. The Ctrl-C lands on h2's poll, before that poll's push.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h2": [KeyboardInterrupt()]}),
            strategy=RecordingStrategy(),
            grabbed=[pending_import(infohash="h1"), pending_import(infohash="h2")],
        )
        view = RecordingWaitView()
        recording = install_recording_hub()

        result = mgr.run_blocking(view=view)  # must not raise

        assert result is not None and result.waited == 1
        assert view.final("h1").outcome is Outcome.GONE  # the interrupt-time push carried it
        assert diagnostic_messages(recording) == ["Wait interrupted - 1 left for the next run"]

    @pytest.mark.parametrize(
        ("store_type", "removals"),
        [
            pytest.param(_InterruptedDropStore, [EffectStatus.DONE], id="in-the-drop"),
            # The queue removal fails, so the import is kept awaiting cleanup: that write is where the Ctrl-C lands.
            pytest.param(_InterruptedPutStore, [EffectStatus.FAILED], id="in-the-awaiting-cleanup-write"),
        ],
    )
    def test_a_ctrl_c_in_the_store_write_of_an_import_counts_no_import(
        self, store_type: type[FakeCacheStore], removals: list[EffectStatus]
    ) -> None:
        # The run's tally counts the result rows, so an import counts only once its store write went through.
        store = store_type()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=True)], removals=removals),
            store=store,
            stored_pending_imports=[pending_import(infohash="h")],
        )
        view = RecordingWaitView()

        result = mgr.run_blocking(view=view)

        assert result is not None
        assert result.rows == ()
        assert view.final("h").outcome is None
        assert stored_awaiting_cleanup(store) == {"h": False}


class TestBlockingWaitRowDiscriminator:
    """Wait rows carry `carried_over`, so _finish_arr_run can tell them from fresh grabs."""

    def test_wait_rows_tell_fresh_from_carried_over(self) -> None:
        # One fresh grab + one carried-over pending import, both import: only the
        # carried-over row is marked, even though the fresh import leaves
        # `grabbed_this_run` mid-pass.
        strategy = RecordingStrategy(
            attempts=[attempt_result(all_done=True)],
        )
        fresh = pending_import(infohash="f", title="Fresh")
        carried = pending_import(infohash="c", title="Carried")
        qbit = FakeQbit(
            {
                "f": [FakeTorrent(is_complete=True, content_path="/f")],
                "c": [FakeTorrent(is_complete=True, content_path="/c")],
            },
        )
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            state=run,
            stored_pending_imports=[carried],
            grabbed=[fresh],
            clock=FakeClock(step=30),
        )

        result = mgr.run_blocking()

        assert result is not None
        assert {row.label: row.carried_over for row in result.rows} == {
            f"Fresh{SEP}SubGroup": False,
            f"Carried{SEP}SubGroup": True,
        }
        # Imported rows leave the store, so neither converts to a pending import state, and
        # the `imported` bump stays _finish_arr_run's.
        assert run.pending_import_states == {}
        assert run.stats.imported == 0


@dataclass(frozen=True)
class _Bar:
    """A downloading row's download bar as the wait screen shows it."""

    fraction: float
    speed_bps: int | None
    eta_s: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    speed_history: tuple[int, ...] = ()


def _bars(view: RecordingWaitView, key: str) -> list[_Bar]:
    """The torrent's download bar in each pushed snapshot that showed it downloading, in push order."""

    return [
        _Bar(t.fraction, t.speed_bps, t.eta_s, t.bytes_done, t.bytes_total, t.speed_history)
        for t in view.seen(key, Phase.DOWNLOADING)
    ]


_DONE = FakeTorrent(is_complete=True, content_path="/d", progress=1.0)
"""A torrent's last reading: downloaded, ready to import."""


class TestBlockingWaitDownloadBars:
    """Each poll shows the downloading rows' bars, and a refresh between polls moves them and nothing else."""

    _POLLED = FakeTorrent(progress=0.3, dlspeed=100, eta=130, completed=300, size=1000)
    _POLLED_BAR = _Bar(0.3, 100, 130, 300, 1000, (100,))

    @pytest.mark.parametrize(
        ("wants_bars", "refreshed", "bars"),
        [
            pytest.param(
                True,
                FakeTorrent(progress=0.6, dlspeed=999, eta=40, completed=600, size=1000),
                [_POLLED_BAR, _Bar(0.6, 999, 40, 600, 1000, (100,))],
                id="moved",
            ),
            # Even at 100% the row stays downloading: only a poll finishes it.
            pytest.param(
                True,
                FakeTorrent(progress=1.0, dlspeed=250, eta=5, completed=1000, size=1000),
                [_POLLED_BAR, _Bar(1.0, 250, 5, 1000, 1000, (100,))],
                id="moved-to-full",
            ),
            pytest.param(True, _POLLED, [_POLLED_BAR], id="unchanged"),
            pytest.param(True, qbittorrentapi.APIConnectionError("down"), [_POLLED_BAR], id="the-read-fails"),
            pytest.param(
                False,
                FakeTorrent(progress=0.6, dlspeed=999, eta=40, completed=600, size=1000),
                [_POLLED_BAR],
                id="a-view-without-bars",
            ),
        ],
    )
    def test_a_refresh_between_polls_moves_only_the_download_bar(
        self, wants_bars: bool, refreshed: QbitStep, bars: list[_Bar]
    ) -> None:
        # The speed history is sampled once per poll, never by a refresh.
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [self._POLLED, _DONE]}, download_progress={"h": [refreshed]}),
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            grabbed=[pending_import(infohash="h")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()
        view.wants_download_progress = wants_bars

        mgr.run_blocking(view=view)

        assert _bars(view, "h") == bars
        assert view.final("h").outcome is Outcome.IMPORTED

    def test_a_refresh_moves_each_downloading_rows_own_bar(self) -> None:
        mgr = make_orchestration_manager(
            qbit=FakeQbit(
                {
                    "a": [FakeTorrent(progress=0.3, dlspeed=100), _DONE],
                    "b": [FakeTorrent(progress=0.3, dlspeed=100), _DONE],
                },
                download_progress={
                    "a": [FakeTorrent(progress=0.6, dlspeed=999)],
                    "b": [FakeTorrent(progress=0.8, dlspeed=500)],
                },
            ),
            strategy=RecordingStrategy(),
            grabbed=[pending_import(infohash="a"), pending_import(infohash="b")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        assert _bars(view, "a") == [_Bar(0.3, 100, speed_history=(100,)), _Bar(0.6, 999, speed_history=(100,))]
        assert _bars(view, "b") == [_Bar(0.3, 100, speed_history=(100,)), _Bar(0.8, 500, speed_history=(100,))]

    def test_a_refresh_leaves_an_importing_rows_bar_alone(self) -> None:
        # "imp" is importing with 1 file of 3 in while "dl" still downloads. qBittorrent reads "imp" at 100%,
        # but an importing row's bar counts the files Sonarr has in.
        importing = attempt_result(all_done=False, command_issued=True, imported_count=1, file_map_episode_count=3)
        mgr = make_orchestration_manager(
            qbit=FakeQbit(
                {"dl": [FakeTorrent(progress=0.3), _DONE], "imp": [_DONE]},
                download_progress={"dl": [FakeTorrent(progress=0.6)]},
            ),
            strategy=RecordingStrategy(attempts=[importing, attempt_result(all_done=True)]),
            grabbed=[pending_import(infohash="dl"), pending_import(infohash="imp")],
            import_poll_interval=30,
            progress_poll_interval=5,
            clock=FakeClock(step=5),
        )
        view = RecordingWaitView()

        mgr.run_blocking(view=view)

        # The first poll's frame, then the one the refresh pushed when "dl"'s bar moved.
        assert [(t.fraction, t.import_done, t.import_total) for t in view.seen("imp", Phase.IMPORTING)] == [
            (1 / 3, 1, 3),
            (1 / 3, 1, 3),
        ]

    def test_a_poll_qbittorrent_misses_keeps_the_last_bar_and_samples_no_speed(self) -> None:
        # No 0% flash and no fake stall in the sparkline: the blip poll shows the last real reading again.
        store, view = run_blocking_wait(
            RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            FakeQbit(
                {
                    "h": [
                        FakeTorrent(progress=0.3, dlspeed=100),
                        qbittorrentapi.APIConnectionError("blip"),
                        FakeTorrent(progress=0.4, dlspeed=200),
                        _DONE,
                    ]
                }
            ),
            progress_poll_interval=0,
        )

        assert _bars(view, "h") == [
            _Bar(0.3, 100, speed_history=(100,)),
            _Bar(0.3, 100, speed_history=(100,)),
            _Bar(0.4, 200, speed_history=(100, 200)),
        ]
        assert stored_awaiting_cleanup(store) == {}

    @pytest.mark.parametrize(
        ("readings", "history"),
        [
            pytest.param(
                [
                    FakeTorrent(progress=0.1, dlspeed=100),
                    FakeTorrent(progress=0.2),
                    FakeTorrent(progress=0.3, dlspeed=300),
                ],
                (100, 0, 300),
                id="a-stall-is-a-zero",
            ),
            pytest.param(
                [FakeTorrent(progress=0.1, dlspeed=100), *[FakeTorrent(progress=0.3, dlspeed=300)] * SPARK_SAMPLES],
                (300,) * SPARK_SAMPLES,
                id="the-oldest-samples-fall-off",
            ),
        ],
    )
    def test_the_speed_sparkline_takes_one_sample_per_poll(
        self, readings: list[QbitStep], history: tuple[int, ...]
    ) -> None:
        _, view = run_blocking_wait(
            RecordingStrategy(),
            FakeQbit({"h": [*readings, _DONE]}),
            progress_poll_interval=0,
        )

        assert _bars(view, "h")[-1].speed_history == history
