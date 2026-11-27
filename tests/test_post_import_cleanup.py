# pyright: strict
"""Tests for the cleanup after a confirmed import: the post-import category move and the queue record removal.

They cover the gates on each, the retries inside a run, and the owed cleanup a later run re-checks and finishes.
"""

from dataclasses import replace

import httpx
import pytest
import qbittorrentapi
import respx

from pearlarr.arr_categories import ArrCategoryResolver, CategorySources
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import AppConfig, Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.output import Severity, TitleHeader
from pearlarr.wait_types import (
    AttemptKind,
    AttemptResult,
    EffectStatus,
    Outcome,
    PauseReason,
    PendingImportState,
)

from .builders import (
    SEP,
    FakeCacheStore,
    attempt_result,
    download_client_json,
    make_categories,
    make_config,
    make_fetched_categories,
    pending_import,
    sonarr_client_fields,
    two_claim_pending_import,
)
from .fakes import FakeClock, bind_arr_http, diagnostic_messages, install_recording_hub
from .wait_fakes import (
    ALL_IN,
    TORRENT_H_COMPLETE,
    CategoryQbit,
    FakeQbit,
    FakeTorrent,
    QbitStep,
    RecordingStrategy,
    make_orchestration_manager,
    stored_awaiting_cleanup,
)

_ARR_URL = "http://arr.test"


def _reading_sonarrs_clients(config: AppConfig, qbit: FakeQbit) -> ArrCategoryResolver:
    """A resolver reading Sonarr's download clients from `_ARR_URL`, with no retry inside a read, like a run's."""

    http, _ = bind_arr_http(_ARR_URL)
    return make_categories(config, sources=CategorySources(replace(http, retries=0), qbit.client()))


class TestPostImportCategory:
    """A confirmed import moves its torrent to the arr's post-import category."""

    @staticmethod
    def _imported_manager(
        qbit: CategoryQbit, *, store: FakeCacheStore, state: ArrRunState, post_import_category: str | None
    ) -> ImportWaitManager:
        """A manager whose one carried-over pending import's files are done, so the series report confirms it.

        A `None` category routes through `make_config` to the blank-drop, so it
        exercises the same "left unset" default a real config file yields.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=RecordingStrategy(progress=[ALL_IN]),
            store=store,
            state=state,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            config=make_config(post_import_category=post_import_category),
        )

    @pytest.mark.parametrize(
        ("category", "set_errors", "expected_sets", "expected_created"),
        [
            # No 409 -> no create. Also pins the gate's scope: the pending import is stored
            # during the move (drop-last), and only the OTHER arr's row puts it off.
            pytest.param("pearlarr-done", [], [("pearlarr-done", "h")], [], id="clean-move"),
            # qBittorrent 409s an unknown category: create it, then re-apply.
            pytest.param(
                "pearlarr-done",
                [qbittorrentapi.Conflict409Error("unknown category")],
                [("pearlarr-done", "h")],
                ["pearlarr-done"],
                id="409-creates-then-retries",
            ),
            # One transient client error recovers on the in-run retry: the move still lands.
            pytest.param(
                "pearlarr-done",
                [qbittorrentapi.APIConnectionError("down")],
                [("pearlarr-done", "h")],
                [],
                id="client-error-recovers-on-retry",
            ),
            # Category left unset (the real config-file default): no call at all.
            pytest.param(None, [], [], [], id="unconfigured-no-call"),
        ],
    )
    def test_a_confirmed_import_moves_its_torrent_to_the_post_import_category_when_one_is_set(
        self,
        category: str | None,
        set_errors: list[Exception],
        expected_sets: list[tuple[str, str]],
        expected_created: list[str],
    ) -> None:
        qbit = CategoryQbit(TORRENT_H_COMPLETE, set_errors=set_errors)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = self._imported_manager(qbit, store=store, state=state, post_import_category=category)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == expected_sets
        assert qbit.created_categories == expected_created
        # The import itself always goes through: the pending import drops and the arr run counts it imported.
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert state.pending_import_states == {"h": PendingImportState.IMPORTED}

    def test_a_move_that_fails_every_retry_keeps_the_pending_import_for_a_later_cleanup(self) -> None:
        # The import still counts, and one warning names the effect left to finish.
        recording = install_recording_hub()
        errors: list[Exception] = [qbittorrentapi.APIConnectionError("down")] * 3
        qbit = CategoryQbit(TORRENT_H_COMPLETE, set_errors=errors)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = self._imported_manager(qbit, store=store, state=state, post_import_category="pearlarr-done")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert state.pending_import_states == {"h": PendingImportState.IMPORTED}
        assert stored_awaiting_cleanup(store) == {"h": True}
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "post-import cleanup" in warning
        assert "(category move)" in warning

    def test_a_blocking_wait_import_of_a_two_title_torrent_moves_and_closes_once(self) -> None:
        pending = two_claim_pending_import(infohash="h")
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=True)])
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            grabbed=[pending],
            config=make_config(post_import_category="pearlarr-done"),
            clock=FakeClock(step=30),
        )

        mgr.run_blocking()

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}


class TestRemoveFromQueue:
    """A confirmed import removes the queue record Sonarr leaves behind.

    One pending import spans every title on the torrent, so the close consults no sibling
    (see `ImportCompleter.remove_from_queue`). Only the category move waits on the other arr.
    """

    @pytest.mark.parametrize(
        ("torrents", "remove_from_queue", "moves", "closes"),
        [
            pytest.param(TORRENT_H_COMPLETE, True, [("pearlarr-done", "h")], ["h"], id="imported-closes"),
            pytest.param(TORRENT_H_COMPLETE, False, [("pearlarr-done", "h")], [], id="turned-off-never-closes"),
            pytest.param({}, True, [], [], id="gone-torrent-is-left-alone"),
        ],
    )
    def test_only_a_confirmed_import_closes_the_queue_record_and_only_when_turned_on(
        self,
        torrents: dict[str, list[QbitStep]],
        remove_from_queue: bool,
        moves: list[tuple[str, str]],
        closes: list[str],
    ) -> None:
        strategy = RecordingStrategy(progress=[ALL_IN])
        qbit = CategoryQbit(torrents)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            config=make_config(remove_from_queue=remove_from_queue, post_import_category="pearlarr-done"),
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert strategy.removal_calls == closes
        assert store.get_pending_import_rows(Arr.SONARR) == {}  # imported or gone, the pending import drops


class TestCleanupCategoryResolve:
    """The cleanup reads the post-import category once, and the move and the queue removal both act on that read."""

    @respx.mock
    def test_a_failed_client_read_moves_nothing_and_the_queue_removal_still_runs(self) -> None:
        # Sonarr answers a second read, naming a post-import category it doesn't watch. Had the removal read
        # again, it would skip for a move that never ran.
        respx.get(f"{_ARR_URL}/api/v3/downloadclient").mock(
            side_effect=[httpx.Response(500), httpx.Response(200, json=[download_client_json(sonarr_client_fields())])]
        )
        strategy = RecordingStrategy(progress=[ALL_IN])
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            categories=_reading_sonarrs_clients(make_config(), qbit),
        )

        mgr.report_imports_for_series(7)

        # What happens today, not a decision: with no category read there is no move to make, so the cleanup
        # finishes and drops the pending import, and the torrent never reaches the post-import category.
        assert qbit.set_category_calls == []
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}


class TestCloseSkipOnMoveToUnwatched:
    """A move to an unwatched category supersedes the explicit queue removal.

    The gate reads `is_unwatched`, never this run's move outcome: firing the
    delete (a permanent "download ignored" history row) in a transiently
    failed-move window is exactly what the skip exists to prevent.
    """

    @staticmethod
    def _manager(
        *,
        grab: str | None,
        post_import: str | None,
        qbit: FakeQbit,
        strategy: RecordingStrategy,
        store: FakeCacheStore,
    ) -> ImportWaitManager:
        """A manager whose resolver already fetched the arr's client categories (preloaded, transportless)."""

        return make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            categories=make_fetched_categories(make_config(), grab=grab, post_import=post_import, qbit=qbit.client()),
        )

    @pytest.mark.parametrize(
        ("grab", "move_errors", "moves", "closes", "kept"),
        [
            pytest.param("tv-sonarr", [], [("sonarr-done", "h")], [], [], id="moved-out-of-sight"),
            # The move fails on every retry.
            pytest.param(
                "tv-sonarr", [qbittorrentapi.APIConnectionError("down")] * 3, [], [], ["h"], id="move-failed-this-run"
            ),
            # A blank category on Sonarr's client watches every category, so no move clears the queue record.
            pytest.param(None, [], [("sonarr-done", "h")], ["h"], [], id="sonarr-watches-every-category"),
        ],
    )
    def test_only_a_move_out_of_sonarrs_sight_takes_the_place_of_the_explicit_close(
        self,
        grab: str | None,
        move_errors: list[Exception],
        moves: list[tuple[str, str]],
        closes: list[str],
        kept: list[str],
    ) -> None:
        strategy = RecordingStrategy(progress=[ALL_IN])
        qbit = CategoryQbit(TORRENT_H_COMPLETE, set_errors=move_errors)
        store = FakeCacheStore()
        mgr = self._manager(grab=grab, post_import="sonarr-done", qbit=qbit, strategy=strategy, store=store)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert strategy.removal_calls == closes
        assert list(store.get_pending_import_rows(Arr.SONARR)) == kept


class TestMoveKeepsAnUnseenTorrentOut:
    """The post-import move never carries a torrent Sonarr can't see into a category Sonarr watches.

    Sonarr has no record of that download. Once the torrent showed up in its category, Sonarr would
    import the files again by its own reading of their names.
    """

    @staticmethod
    def _manager(
        qbit: CategoryQbit, store: FakeCacheStore, *, post_import: str, awaiting_cleanup: bool = False
    ) -> ImportWaitManager:
        """A manager over one stored pending import of series 7 whose files are in, Sonarr's clients watching `tv-sonarr`.

        With `awaiting_cleanup`, the import went through on an earlier run and only its cleanup is owed.
        """

        return make_orchestration_manager(
            qbit=qbit,
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=True)], progress=[ALL_IN]),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=awaiting_cleanup)],
            categories=make_fetched_categories(
                make_config(), grab="tv-sonarr", post_import=post_import, qbit=qbit.client()
            ),
        )

    def test_a_failed_category_read_is_retried_and_never_moves_blind(self) -> None:
        unseen = FakeTorrent(is_complete=True, content_path="/d", category="manual")
        # The poll reads the torrent first, then the move's read fails once.
        qbit = CategoryQbit({"h": [unseen, qbittorrentapi.APIConnectionError("down"), unseen]})
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @pytest.mark.parametrize(
        ("waiting_in", "post_import"),
        [("manual", "tv-sonarr/done"), ("tv-sonarr/sub", "tv-sonarr")],
        ids=["into-a-subcategory", "out-of-a-subcategory"],
    )
    def test_an_unreadable_subcategory_setting_holds_the_move_it_would_decide(
        self,
        waiting_in: str,
        post_import: str,
    ) -> None:
        qbit = CategoryQbit(
            {"h": [FakeTorrent(is_complete=True, content_path="/d", category=waiting_in)]},
            preferences_error=qbittorrentapi.APIConnectionError("down"),
        )
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import=post_import)

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert stored_awaiting_cleanup(store) == {"h": True}

    @pytest.mark.parametrize(
        ("waiting_in", "moves"),
        [
            pytest.param("manual", [], id="unseen-stays-out"),
            pytest.param("tv-sonarr", [("tv-sonarr/done", "h")], id="seen-moves"),
        ],
    )
    def test_only_a_torrent_sonarr_already_sees_moves_into_a_watched_subcategory(
        self, waiting_in: str, moves: list[tuple[str, str]]
    ) -> None:
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category=waiting_in)]})
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr/done")

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == moves
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @pytest.mark.parametrize(
        "torrents",
        [{"h": [FakeTorrent(is_complete=True, content_path="/d", category="tv-sonarr/done")]}, {}],
        ids=["already-there", "gone-from-qbittorrent"],
    )
    def test_an_unreadable_subcategory_setting_never_holds_a_move_that_carries_nothing(
        self, torrents: dict[str, list[QbitStep]]
    ) -> None:
        # An owed cleanup, the one a torrent gone from qBittorrent still reaches.
        qbit = CategoryQbit(torrents, preferences_error=qbittorrentapi.APIConnectionError("down"))
        store = FakeCacheStore()
        mgr = self._manager(qbit, store, post_import="tv-sonarr/done", awaiting_cleanup=True)

        mgr.finish_cleanups()

        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @respx.mock
    def test_unreadable_sonarr_clients_hold_the_move_for_a_later_run(self) -> None:
        # Only Sonarr's clients say whether it watches `sonarr-done`. The configured grab category stands in
        # for them at the queue removal, never at the move.
        respx.get(f"{_ARR_URL}/api/v3/downloadclient").respond(status_code=500)
        config = make_config(torrent_category="anime", post_import_category="sonarr-done")
        strategy = RecordingStrategy(progress=[ALL_IN])
        qbit = CategoryQbit({"h": [FakeTorrent(is_complete=True, content_path="/d", category="manual")]})
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            categories=_reading_sonarrs_clients(config, qbit),
        )

        mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == []
        assert stored_awaiting_cleanup(store) == {"h": True}


class TestCloseRetrySchedule:
    """The close effect goes through the bounded in-run retry schedule off the injected clock."""

    @pytest.mark.parametrize(
        ("close_results", "sleeps", "kept"),
        [
            pytest.param([EffectStatus.RETRY, EffectStatus.DONE], [1.0], [], id="recovers-after-one-pause"),
            pytest.param([EffectStatus.RETRY] * 3, [1.0, 3.0], [True], id="retries-run-out"),
            # FAILED is a queue read that already retried inside the transport.
            pytest.param([EffectStatus.FAILED], [], [True], id="failed-is-final"),
        ],
    )
    def test_the_close_retries_on_the_schedule_and_keeps_the_pending_import_when_it_gives_up(
        self, close_results: list[EffectStatus], sleeps: list[float], kept: list[bool]
    ) -> None:
        recording = install_recording_hub()
        clock = FakeClock()
        strategy = RecordingStrategy(progress=[ALL_IN], removals=close_results)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(TORRENT_H_COMPLETE),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
            clock=clock,
        )

        mgr.report_imports_for_series(7)

        assert len(strategy.removal_calls) == len(close_results)
        assert clock.sleeps == sleeps
        assert list(stored_awaiting_cleanup(store).values()) == kept
        assert ["(queue removal)" in warn for warn in diagnostic_messages(recording, Severity.WARNING)] == kept


class TestCleanupRetry:
    """Pending imports awaiting cleanup: the observe steps skip them, `finish_cleanups` finishes them once per run."""

    @pytest.mark.parametrize(
        ("run2_close", "stored", "notes"),
        [
            pytest.param(
                EffectStatus.DONE,
                {},
                [f"Finished the leftover post-import cleanup for Show{SEP}SubGroup"],
                id="next-run-finishes",
            ),
            pytest.param(EffectStatus.FAILED, {"h": True}, [], id="next-run-fails-again"),
        ],
    )
    def test_keep_then_next_run_finishes_the_cleanup_or_keeps_again(
        self, run2_close: EffectStatus, stored: dict[str, bool], notes: list[str]
    ) -> None:
        # Run 1: the import is confirmed but the close fails -> awaiting cleanup, still
        # counted IMPORTED this run. Run 2: finishing the cleanup closes and drops,
        # or keeps the pending import again when the close still fails.
        run1_strategy = RecordingStrategy(
            progress=[ALL_IN],
            removals=[EffectStatus.FAILED],
        )
        store = FakeCacheStore()
        run1_state = ArrRunState(arr=Arr.SONARR)
        run1 = make_orchestration_manager(
            qbit=FakeQbit(TORRENT_H_COMPLETE),
            strategy=run1_strategy,
            store=store,
            state=run1_state,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )

        run1.report_imports_for_series(7)
        run1.finish_cleanups()

        assert stored_awaiting_cleanup(store) == {"h": True}
        assert run1_state.pending_import_states == {"h": PendingImportState.IMPORTED}
        # The finish leaves a cleanup that failed this run for the next one.
        assert run1_strategy.import_calls == []

        recording = install_recording_hub()
        run2_strategy = RecordingStrategy(removals=[run2_close])
        run2_state = ArrRunState(arr=Arr.SONARR)
        # The same durable store, next run. The torrent is gone from qBittorrent by then, the one case where the owed
        # cleanup runs without a fresh files check.
        run2 = make_orchestration_manager(qbit=FakeQbit({}), strategy=run2_strategy, store=store, state=run2_state)

        run2.finish_cleanups()
        run2.tally_carried_over_into_stats()

        assert run2_strategy.removal_calls == ["h"]
        assert stored_awaiting_cleanup(store) == stored
        assert diagnostic_messages(recording, Severity.INFO) == notes
        # Run 1 counted the import, so run 2 counts nothing for it either way.
        stats = run2_state.stats
        assert (run2_state.pending_import_states, stats.downloading, stats.downloaded) == ({}, 0, 0)

    def test_pending_imports_awaiting_cleanup_are_invisible_to_every_observe_pass(self) -> None:
        strategy = RecordingStrategy()
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(TORRENT_H_COMPLETE),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=True)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert recording.of_type(TitleHeader) == []
        assert mgr.run_deferred() is None
        assert mgr.run_blocking() is None
        assert strategy.import_calls == []
        assert stored_awaiting_cleanup(store) == {"h": True}

    def test_the_cleanup_finish_leaves_a_pending_import_that_owes_no_cleanup_to_the_wait(self) -> None:
        strategy = RecordingStrategy()
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(TORRENT_H_COMPLETE),
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h")],
        )

        mgr.finish_cleanups()

        assert strategy.import_calls == []
        assert stored_awaiting_cleanup(store) == {"h": False}


_OWED_CLEANUP = pending_import(infohash="h", awaiting_cleanup=True)
"""Pending import "h", imported on an earlier run with its cleanup still owed."""


class TestCleanupRetryVerification:
    """The cleanup finish re-checks through the standard import check before any effect fires."""

    def test_unobserved_qbittorrent_stays_awaiting_cleanup_without_effects(self) -> None:
        strategy = RecordingStrategy()
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [qbittorrentapi.APIConnectionError("down")]}),
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
        )

        mgr.finish_cleanups()

        assert strategy.import_calls == []
        assert strategy.removal_calls == []
        assert stored_awaiting_cleanup(store) == {"h": True}
        assert state.pending_import_states == {}

    def test_done_files_run_the_effects_at_the_final_attempt(self) -> None:
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=True)])
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[_OWED_CLEANUP],
            config=make_config(post_import_category="pearlarr-done"),
        )

        mgr.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}

    @pytest.mark.parametrize(
        "waiting",
        [
            pytest.param(attempt_result(all_done=False), id="no-command"),
            pytest.param(
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.IMPORT),
                id="command-still-in-flight",
            ),
        ],
    )
    def test_transient_retry_attempt_result_stays_awaiting_cleanup(self, waiting: AttemptResult) -> None:
        # Only a re-import issued just now says the files are gone. A blip, a busy queue or a command
        # still in flight is ambiguous: keep it awaiting cleanup and retry next run.
        strategy = RecordingStrategy(attempts=[waiting])
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit(TORRENT_H_COMPLETE),
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
        )

        mgr.finish_cleanups()

        assert [c.attempt_kind for c in strategy.import_calls] == [AttemptKind.FINAL]
        assert strategy.removal_calls == []
        assert stored_awaiting_cleanup(store) == {"h": True}
        assert state.pending_import_states == {}

    def test_files_gone_and_reimport_issued_demotes_to_a_carryover(self) -> None:
        # The files vanished and a fresh re-import command just went in (the
        # ISSUED reason): the pending import demotes to an ordinary carryover, no effect
        # fires, and this same run's wait pass takes it up.
        recording = install_recording_hub()
        strategy = RecordingStrategy(
            attempts=[attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED)],
        )
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        store = FakeCacheStore()
        state = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            state=state,
            stored_pending_imports=[_OWED_CLEANUP],
            config=make_config(post_import_category="pearlarr-done"),
        )

        mgr.finish_cleanups()

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == []
        assert stored_awaiting_cleanup(store) == {"h": False}
        # The blocking wait tallies before its pass, so the summary counts the row downloaded.
        assert state.pending_import_states == {"h": PendingImportState.DOWNLOADED}
        [warn] = diagnostic_messages(recording, Severity.WARNING)
        assert "no longer has every file imported" in warn

        result = mgr.run_deferred()

        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.IMPORT_IN_PROGRESS, True)]
