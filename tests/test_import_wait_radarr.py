# pyright: strict
"""Tests for Radarr's pending imports at the end of a run.

A Radarr pending import finishes once Radarr's history shows its import. A torrent a Radarr grab shares with a
Sonarr grab moves to the post-import category only once both arrs imported it.
"""

import pytest

from pearlarr.arr_models import HistoryRecord
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.import_wait import ImportWaitManager

from .builders import (
    FakeCacheStore,
    attempt_result,
    make_categories,
    make_config,
    make_import_wait_manager,
    make_radarr_sync,
    make_run_deps,
    pending_import,
    radarr_history_record,
)
from .fakes import FakeRadarrClient
from .wait_fakes import (
    ALL_IN,
    TORRENT_H_COMPLETE,
    CategoryQbit,
    RecordingStrategy,
    make_orchestration_manager,
)


def _radarr_reconcile_manager(
    *, qbit: CategoryQbit, history: list[HistoryRecord] | None, store: FakeCacheStore, post_import_category: str
) -> ImportWaitManager:
    """A manager wired for a Radarr run: a real `RadarrStrategy` reconciling off scripted history.

    The manager and the strategy share `store`, so a pending import the reconcile
    drops is gone for the cross-arr category count. The `_state.arr` is RADARR, so
    the arr-scoped steps read the Radarr pending imports.
    """

    config = make_config(radarr_post_import_category=post_import_category)
    radarr = FakeRadarrClient()
    radarr.history_since_return = history
    deps = make_run_deps(
        config=config,
        cache_store=store,
        qbit=qbit.client(),
        arr=Arr.RADARR,
        categories=make_categories(config, Arr.RADARR),
    )
    return make_import_wait_manager(
        deps,
        state=ArrRunState(arr=Arr.RADARR),
        strategy=make_radarr_sync(radarr=radarr, config=config, cache_store=store),
    )


class TestCrossArrCategoryGate:
    """A torrent shared by a Sonarr grab and a Radarr grab moves only after BOTH import."""

    def test_sonarr_import_puts_off_the_move_until_radarr_reconciles(self) -> None:
        # One torrent, two pending imports: a Sonarr title's part (imports first) and a Radarr
        # movie. The Sonarr import must NOT move the category while the Radarr
        # pending import still claims the hash; the later Radarr reconcile fires the move.
        store = FakeCacheStore()
        sonarr_pending = pending_import(infohash="h", series_id=7)
        radarr_pending = pending_import(infohash="h", series_id=0)
        qbit = CategoryQbit(TORRENT_H_COMPLETE)

        # Sonarr run: its part is confirmed (files done) and drops, but the Radarr
        # pending import still claims the hash cross-arr -> the move is put off.
        sonarr_mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=RecordingStrategy(progress=[ALL_IN]),
            store=store,
            stored_pending_imports=[sonarr_pending],
            config=make_config(post_import_category="pearlarr-done"),
        )
        store.put_pending_import_row(Arr.RADARR, radarr_pending.infohash, radarr_pending.to_json())

        sonarr_mgr.report_imports_for_series(7)

        assert qbit.set_category_calls == []  # the Radarr pending import still claims the hash
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}

        # Radarr run: the movie import is confirmed -> no claim remains -> the move fires.
        radarr_mgr = _radarr_reconcile_manager(
            qbit=qbit,
            history=[radarr_history_record("h", event="movieFolderImported")],
            store=store,
            post_import_category="pearlarr-done",
        )

        radarr_mgr.run_deferred()

        assert qbit.set_category_calls == [("pearlarr-done", "h")]
        assert store.get_pending_import_rows(Arr.RADARR) == {}

    @pytest.mark.parametrize(
        ("awaiting_cleanup", "radarr_awaiting_cleanup"),
        [
            pytest.param(False, False, id="first-import"),
            pytest.param(True, False, id="cleanup-finish"),
            # Radarr's import went through but its cleanup is still to finish, and that cleanup makes the move.
            pytest.param(False, True, id="radarr-awaiting-its-own-cleanup"),
        ],
    )
    def test_radarr_on_the_torrent_holds_the_move_and_never_the_queue_removal(
        self, awaiting_cleanup: bool, radarr_awaiting_cleanup: bool
    ) -> None:
        strategy = RecordingStrategy(attempts=[attempt_result(all_done=True)], progress=[ALL_IN])
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=qbit,
            strategy=strategy,
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=7, awaiting_cleanup=awaiting_cleanup)],
            config=make_config(post_import_category="pearlarr-done"),
        )
        movie = pending_import(infohash="h", series_id=0, awaiting_cleanup=radarr_awaiting_cleanup)
        store.put_pending_import_row(Arr.RADARR, "h", movie.to_json())

        mgr.report_imports_for_series(7)
        mgr.finish_cleanups()

        assert qbit.set_category_calls == []
        assert strategy.removal_calls == ["h"]
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert set(store.get_pending_import_rows(Arr.RADARR)) == {"h"}


class TestRadarrReconcile:
    """A Radarr pending import is done once Radarr's history shows its import, and waits until then."""

    @pytest.mark.parametrize(
        ("history", "awaiting_cleanup", "moves", "kept"),
        [
            pytest.param([], False, [], [False], id="no-event-yet"),
            pytest.param(None, False, [], [False], id="history-outage"),
            pytest.param(None, True, [], [True], id="history-outage-while-awaiting-cleanup"),
            pytest.param(
                [radarr_history_record("h", event="movieFolderImported")],
                True,
                [("pearlarr-done", "h")],
                [],
                id="event-finishes-the-cleanup",
            ),
        ],
    )
    def test_only_an_import_event_in_radarrs_history_moves_the_torrent_and_drops_the_pending_import(
        self,
        history: list[HistoryRecord] | None,
        awaiting_cleanup: bool,
        moves: list[tuple[str, str]],
        kept: list[bool],
    ) -> None:
        store = FakeCacheStore()
        movie = pending_import(infohash="h", series_id=0, awaiting_cleanup=awaiting_cleanup)
        store.put_pending_import_row(Arr.RADARR, "h", movie.to_json())
        qbit = CategoryQbit(TORRENT_H_COMPLETE)
        mgr = _radarr_reconcile_manager(qbit=qbit, history=history, store=store, post_import_category="pearlarr-done")

        mgr.finish_cleanups()
        mgr.run_deferred()

        assert qbit.set_category_calls == moves
        assert [bool(row.get("awaiting_cleanup")) for row in store.get_pending_import_rows(Arr.RADARR).values()] == kept
