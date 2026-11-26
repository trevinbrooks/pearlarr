# pyright: strict
"""Tests for the series report of carried-over pending imports.

The report shows each one once, as qBittorrent and the files read. A pending import whose torrent left
qBittorrent is dropped, and its claiming titles are checked again on the next run.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import ClassVar

import pytest

from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.output import Severity, TitleHeader
from pearlarr.output.events import TitleState
from pearlarr.pending_import import PendingImport
from pearlarr.torrent_service import AddStatus
from pearlarr.wait_types import NO_PROGRESS, ImportProgress, ImportWaitMode, Outcome, PendingImportState

from .builders import (
    SEP,
    FakeCacheStore,
    FakeTorrents,
    attempt_result,
    grab_request,
    make_grab_pipeline,
    one_release_dict,
    pending_import,
    pending_import_update,
    title_claim,
    two_claim_pending_import,
)
from .fakes import FakeClock, diagnostic_messages, install_recording_hub
from .wait_fakes import (
    TORRENT_H_COMPLETE,
    FakeQbit,
    FakeTorrent,
    QbitStep,
    RecordingStrategy,
    make_orchestration_manager,
    stored_awaiting_cleanup,
)


class TestReportImportsForSeries:
    """The series report shows each carried-over pending import of the series once, as qBittorrent and the files read.

    It only reads: it never attempts an import, and the `imported` count is the arr run finish's.
    """

    @pytest.mark.parametrize(
        ("torrent", "progress", "state", "shown", "stored"),
        [
            pytest.param(
                FakeTorrent(is_complete=True, content_path="/d"),
                ImportProgress(1, 1, determinate=True),
                PendingImportState.IMPORTED,
                [(TitleState.IMPORTED, f"Show{SEP}SubGroup")],
                {},
                id="every-file-imported",
            ),
            pytest.param(
                FakeTorrent(is_complete=True),
                NO_PROGRESS,
                PendingImportState.DOWNLOADED,
                [(TitleState.DOWNLOADED, f"Show{SEP}SubGroup")],
                {"h": False},
                id="files-not-in-yet",
            ),
            # The wait pass shows the error on its row. The tally counts an errored torrent nowhere.
            pytest.param(
                FakeTorrent(is_errored=True), NO_PROGRESS, PendingImportState.ERRORED, [], {"h": False}, id="errored"
            ),
        ],
    )
    def test_a_carried_over_pending_import_is_reported_as_qbittorrent_and_the_files_read(
        self,
        torrent: FakeTorrent,
        progress: ImportProgress,
        state: PendingImportState,
        shown: list[tuple[TitleState, str]],
        stored: dict[str, bool],
    ) -> None:
        strategy = RecordingStrategy(progress=[progress])
        store = FakeCacheStore()
        run = ArrRunState(arr=Arr.SONARR)
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [torrent]}),
            strategy=strategy,
            store=store,
            state=run,
            stored_pending_imports=[pending_import(infohash="h", series_id=7)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert [(h.state, h.title) for h in recording.of_type(TitleHeader)] == shown
        assert run.pending_import_states == {"h": state}
        assert stored_awaiting_cleanup(store) == stored
        assert strategy.import_calls == []
        assert run.stats.imported == 0

    def test_a_carried_over_torrent_gone_from_qbittorrent_is_warned_about_and_dropped(self) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}),
            strategy=RecordingStrategy(),
            store=store,
            stored_pending_imports=[pending_import(series_id=7)],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert f"Show{SEP}SubGroup is gone from qBittorrent" in warning
        assert stored_awaiting_cleanup(store) == {}

    @pytest.mark.parametrize(
        ("series_id", "state"),
        [
            # A stored pending import re-seen in qBittorrent this run was already reported by its title block.
            # Only the tally counts it.
            pytest.param(7, ArrRunState(arr=Arr.SONARR, reacquired_keys={"h"}), id="seen-again-this-run"),
            pytest.param(
                7,
                ArrRunState(arr=Arr.SONARR, grabbed_this_run={"h": pending_import(infohash="h", series_id=7)}),
                id="grabbed-this-run",
            ),
            pytest.param(99, ArrRunState(arr=Arr.SONARR), id="on-another-series"),
        ],
    )
    def test_a_pending_import_the_series_report_skips_is_left_alone(self, series_id: int, state: ArrRunState) -> None:
        store = FakeCacheStore()
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent()]}),
            strategy=RecordingStrategy(),
            store=store,
            stored_pending_imports=[pending_import(infohash="h", series_id=series_id)],
            state=state,
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)

        assert recording.of_type(TitleHeader) == []
        assert state.pending_import_states == {}
        assert stored_awaiting_cleanup(store) == {"h": False}

    def test_a_two_series_pending_import_is_reported_once_under_the_series_that_scans_first(self) -> None:
        pending = pending_import(
            infohash="h",
            claims=(
                title_claim(anilist_id=11, series_id=7, window_coverage="S01 E01-E12", url="https://releases.moe/11"),
                title_claim(anilist_id=22, series_id=8, window_coverage="S02 E01-E12", url="https://releases.moe/22"),
            ),
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(progress=0.5)]}),
            strategy=RecordingStrategy(),
            stored_pending_imports=[pending],
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(8)
        mgr.report_imports_for_series(7)

        assert [(h.state, h.coverage, h.url) for h in recording.of_type(TitleHeader)] == [
            (TitleState.DOWNLOADING, "S02 E01-E12", "https://releases.moe/22"),
        ]

    def test_a_torrent_the_grab_found_already_in_qbittorrent_is_left_to_the_end_pass_and_imports_there(self) -> None:
        # The grab starts its pending import and its title block reports it, so the series report leaves it be.
        state = ArrRunState(arr=Arr.SONARR, import_wait_mode=ImportWaitMode.BLOCKING)
        store = FakeCacheStore()
        pipeline = make_grab_pipeline(
            _torrents=FakeTorrents({"h1": (AddStatus.ALREADY_IN_QBIT, "Show")}), cache_store=store, _state=state
        )
        mgr = make_orchestration_manager(
            qbit=FakeQbit({"h1": [FakeTorrent(is_complete=True, content_path="/d")]}),
            strategy=RecordingStrategy(attempts=[attempt_result(all_done=True)]),
            store=store,
            state=state,
            clock=FakeClock(step=30),
        )
        pipeline.grab_releases(
            grab_request(
                seadex_dict=one_release_dict(srg="NAN0", infohash="h1"),
                pending_import_updates={"h1": pending_import_update("h1", series_id=7)},
            ),
        )
        recording = install_recording_hub()

        mgr.report_imports_for_series(7)
        result = mgr.run_blocking()

        assert recording.of_type(TitleHeader) == []
        assert result is not None
        assert [(row.outcome, row.carried_over) for row in result.rows] == [(Outcome.IMPORTED, True)]
        assert stored_awaiting_cleanup(store) == {}


def _report_series_7(mgr: ImportWaitManager) -> None:
    mgr.report_imports_for_series(7)


def _run_end_pass(mgr: ImportWaitManager) -> None:
    mgr.run_deferred()


class TestGoneTorrentHasItsTitlesCheckedAgain:
    """Dropping a pending import whose torrent left qBittorrent clears its titles' stamps: the next run checks them."""

    _CACHED_ON = "2021-06-05 00:00:00"
    _ALL_CACHED: ClassVar[Mapping[int, str | None]] = {5: _CACHED_ON, 6: _CACHED_ON, 9: _CACHED_ON}
    """Every title's stamp as `_store` writes it."""
    _CLAIMING_CHECKED_AGAIN: ClassVar[Mapping[int, str | None]] = {5: None, 6: None, 9: _CACHED_ON}
    """Titles 5 and 6, which claim the torrent, uncached. Title 9 never claimed it."""

    @classmethod
    def _store(cls) -> FakeCacheStore:
        """Titles 5, 6 and 9 cached."""

        store = FakeCacheStore()
        for anilist_id in (5, 6, 9):
            store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": cls._CACHED_ON})
        return store

    @staticmethod
    def _record() -> PendingImport:
        """Torrent "h", claimed by title 5 on series 7 and title 6 on series 8."""

        return two_claim_pending_import(anilist_ids=(5, 6), series_ids=(7, 8), infohash="h")

    @staticmethod
    def _stamps(store: FakeCacheStore) -> dict[int, str | None]:
        cached_titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (5, 6, 9)}
        return {anilist_id: cached.updated_at for anilist_id, cached in cached_titles.items() if cached is not None}

    @pytest.mark.parametrize(
        ("torrents", "awaiting_cleanup", "run", "stamps"),
        [
            pytest.param({}, False, _report_series_7, _CLAIMING_CHECKED_AGAIN, id="gone-at-the-scan-report"),
            pytest.param({}, False, _run_end_pass, _CLAIMING_CHECKED_AGAIN, id="gone-at-the-end-pass"),
            pytest.param(TORRENT_H_COMPLETE, False, _report_series_7, _ALL_CACHED, id="imported-at-the-scan-report"),
            pytest.param(TORRENT_H_COMPLETE, False, _run_end_pass, _ALL_CACHED, id="imported-at-the-end-pass"),
            pytest.param({}, True, ImportWaitManager.finish_cleanups, _ALL_CACHED, id="gone-after-a-confirmed-import"),
        ],
    )
    def test_a_drop_clears_the_claiming_titles_stamps_only_when_nothing_was_imported(
        self,
        torrents: dict[str, list[QbitStep]],
        awaiting_cleanup: bool,
        run: Callable[[ImportWaitManager], None],
        stamps: Mapping[int, str | None],
    ) -> None:
        store = self._store()
        mgr = make_orchestration_manager(
            qbit=FakeQbit(torrents),
            strategy=RecordingStrategy(
                attempts=[attempt_result(all_done=True)], progress=[ImportProgress(1, 1, determinate=True)]
            ),
            store=store,
            stored_pending_imports=[replace(self._record(), awaiting_cleanup=awaiting_cleanup)],
        )

        run(mgr)

        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert self._stamps(store) == stamps

    def test_the_other_arrs_row_keeps_its_stamp(self) -> None:
        store = self._store()
        store.update_cached_title(Arr.RADARR, 5, {"updated_at": self._CACHED_ON})
        mgr = make_orchestration_manager(
            qbit=FakeQbit({}), strategy=RecordingStrategy(), store=store, stored_pending_imports=[self._record()]
        )

        mgr.report_imports_for_series(7)

        radarr_row = store.get_cached_title(Arr.RADARR, 5)
        assert radarr_row is not None
        assert radarr_row.updated_at == self._CACHED_ON
