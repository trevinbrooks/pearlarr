# pyright: strict
"""Tests for a Sonarr import that reads the download's folder because Sonarr won't serve the download by id.

That happens to a download whose history Sonarr closed, and to a torrent in a category Sonarr doesn't watch.
"""

import pytest

from pearlarr.arr_models import CommandResource, HistoryPage, ManualImportCandidate, QueueRecord
from pearlarr.file_paths import normalize_file_name
from pearlarr.output import Severity
from pearlarr.pending_import import FileEpisodeMap, PendingImport
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import AttemptKind, AttemptResult, PauseReason

from .builders import make_fetched_categories, manual_candidate, pending_import, queue_record, sonarr_ep
from .fakes import FakeSonarrClient, HookedSonarrClient, QbitCategoryReads, diagnostic_messages, install_recording_hub
from .sonarr_import_builders import (
    RESCAN_AFTER_ADD,
    RESCAN_BEFORE_ADD,
    closed_history,
    imported_history,
    make_sonarr_for_import,
    started_process_pass,
)


def _clean_history() -> HistoryPage:
    """A history page whose newest relevant event is a grab (genuinely downloading)."""

    return HistoryPage.model_validate(
        {"records": [{"id": 9, "eventType": "grabbed", "date": "2026-07-17T00:00:00Z"}]},
    )


_EMPTY_FOLDER_WARNING = (
    "Show · SubGroup: Sonarr won't serve this download by id and reading its folder found no files (/d/Show)"
    " - will retry"
)
"""The deadline's warning for a download Sonarr won't serve by id, whose folder `/d/Show` reads empty."""


_NOT_VISIBLE_IN_EMPTY_FOLDER_WARNING = (
    "/d/Show: 1 intended file not visible to Sonarr for Show · SubGroup"
    " (it won't serve the download by id and reading /d/Show found no files) - will retry"
)
"""The deadline's warning for the file map's one file, when the download's folder `/d/Show` reads empty too."""


class TestFolderFetchFallback:
    """The closed-history loop cure: a downloadId fetch that can't serve the download reads history and the folder.

    A download Sonarr's history maps to Imported/Failed/Ignored is queue-hidden
    and its `downloadId=` fetch never serves it (a 500 on v4, `[]` on v5). Without
    the fallback every poll would wait "for a later run" for as long as the
    torrent stayed in qBittorrent.
    """

    @staticmethod
    def _strat(
        *, history: HistoryPage | None, folder_candidates: list[ManualImportCandidate], import_mode: str = "auto"
    ) -> tuple[SonarrStrategy, HookedSonarrClient]:
        # candidates=None: the downloadId fetch fails (the 500 shape) -> fallback.
        strat, sonarr = make_sonarr_for_import(candidates=None, config_overrides={"import_mode": import_mode})
        sonarr.history_page_return = history
        sonarr.folder_candidates_return = folder_candidates
        sonarr.path_mappings_return = []
        return strat, sonarr

    @staticmethod
    def _answer_empty_by_id(
        sonarr: FakeSonarrClient,
        *,
        commands: tuple[CommandResource, ...] = (RESCAN_AFTER_ADD,),
    ) -> None:
        """Script the downloadId fetch to answer `[]` under the given command list."""

        sonarr.candidates_return = []
        sonarr.commands_return = list(commands)

    @staticmethod
    def _poll_then_empty_the_folder(
        strat: SonarrStrategy, sonarr: FakeSonarrClient, pending: PendingImport
    ) -> AttemptResult:
        """Poll with the episode index unread, then again once the folder fetch is empty and only 101 holds a file."""

        sonarr.episodes_return = []
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.folder_candidates_return = []
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
        ]
        return strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

    @pytest.mark.parametrize(
        ("history", "import_mode", "posted"),
        [
            pytest.param(_clean_history(), "auto", [("ABC123", "copy"), (None, "copy")], id="clean-history"),
            pytest.param(None, "auto", [("ABC123", "copy"), (None, "copy")], id="unreadable-history"),
            pytest.param(closed_history(), "move", [(None, "move"), (None, "move")], id="configured-move"),
        ],
    )
    def test_a_folder_import_keeps_the_download_id_until_history_closes_and_copies_unless_a_mode_is_set(
        self, history: HistoryPage | None, import_mode: str, posted: list[tuple[str | None, str]]
    ) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
            import_mode=import_mode,
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.history_page_return = closed_history()
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls] == posted

    def test_a_closed_history_is_read_once_per_run(self) -> None:
        # Kept for the run: a closed history doesn't reopen, and every poll would read it again.
        strat, sonarr = self._strat(
            history=closed_history(),
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.download_history_calls == ["abc123"]

    def test_history_closed_empty_folder_warns_only_at_the_deadline(self) -> None:
        # An empty folder right after completion is expected (Sonarr's mount
        # lags), so ordinary polls don't warn; the genuinely stuck shape
        # (a pending import with an empty file map, still empty at the deadline) warns once a run,
        # and again in the next run. `closed_history()` carries no import rows, so the
        # rebuild yields nothing and the file map stays empty.
        recording = install_recording_hub()
        strat, _ = self._strat(history=closed_history(), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert diagnostic_messages(recording, Severity.WARNING) == []

        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        assert diagnostic_messages(recording, Severity.WARNING) == [_EMPTY_FOLDER_WARNING]

        strat.get_items()
        strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)
        assert diagnostic_messages(recording, Severity.WARNING) == [_EMPTY_FOLDER_WARNING, _EMPTY_FOLDER_WARNING]

    def test_empty_download_id_fetch_with_an_empty_folder_waits_and_warns_at_the_deadline(self) -> None:
        recording = install_recording_hub()
        strat, sonarr = self._strat(history=closed_history(), folder_candidates=[])
        self._answer_empty_by_id(sonarr)
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert diagnostic_messages(recording, Severity.WARNING) == [_EMPTY_FOLDER_WARNING]

    def test_rebuilt_file_map_that_is_not_done_waits_without_pinning_it(self) -> None:
        recording = install_recording_hub()
        name = "Show - 01 [1080p].mkv"
        strat, _ = self._strat(history=imported_history((101, f"/d/Show/{name}")), folder_candidates=[])
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[name])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert result.pending is pending
        assert diagnostic_messages(recording, Severity.WARNING) == [_NOT_VISIBLE_IN_EMPTY_FOLDER_WARNING]

    def test_folder_fetch_is_never_kept(self) -> None:
        strat, sonarr = self._strat(
            history=_clean_history(), folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")]
        )
        sonarr.queue_return = [queue_record("ABC123", "importBlocked", status="warning")]

        def start_a_pass() -> None:
            sonarr.commands_return = [RESCAN_BEFORE_ADD, started_process_pass()]

        sonarr.on_fetch = start_a_pass
        pending = pending_import()

        blocked = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        sonarr.on_fetch = None
        sonarr.folder_candidates_return = None
        sonarr.commands_return = [RESCAN_BEFORE_ADD]
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert blocked.pause_reason is PauseReason.BUSY
        assert sonarr.execute_calls == []

    @pytest.mark.parametrize(
        ("history", "commands"),
        [
            pytest.param(closed_history(), (RESCAN_BEFORE_ADD,), id="no-rescan-since-the-add"),
            pytest.param(_clean_history(), (RESCAN_AFTER_ADD,), id="clean-history"),
            pytest.param(None, (RESCAN_AFTER_ADD,), id="unreadable-history"),
        ],
    )
    def test_empty_download_id_fetch_waits_without_the_proof_or_a_closed_history(
        self, history: HistoryPage | None, commands: tuple[CommandResource, ...]
    ) -> None:
        strat, sonarr = self._strat(
            history=history,
            folder_candidates=[manual_candidate("/d/Show/Show - 01 [1080p].mkv")],
        )
        self._answer_empty_by_id(sonarr, commands=commands)

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert sonarr.execute_calls == []

    @pytest.mark.parametrize(
        ("by_id", "rebuilt"),
        [(None, {normalize_file_name("Show - 01 [1080p].mkv"): (101,)}), ([], {})],
        ids=["failed", "empty"],
    )
    def test_folder_pinned_pending_import_rebuilds_from_history_only_after_a_failed_download_id_fetch(
        self,
        by_id: list[ManualImportCandidate] | None,
        rebuilt: FileEpisodeMap,
    ) -> None:
        first, second = "Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"
        strat, sonarr = self._strat(
            history=imported_history((101, f"/d/Show/{first}")),
            folder_candidates=[manual_candidate(f"/d/Show/{first}"), manual_candidate(f"/d/Show/{second}")],
        )
        self._answer_empty_by_id(sonarr)
        sonarr.candidates_return = by_id
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101, 102], seadex_files=[first, second])

        result = self._poll_then_empty_the_folder(strat, sonarr, pending)

        assert result.pending is not None
        assert result.pending.file_episode_map == rebuilt


class TestUnwatchedTorrentImport:
    """A finished torrent in a category Sonarr doesn't watch imports from its folder.

    Sonarr never tracks such a torrent: its `downloadId=` fetch answers `[]` for good and its history holds
    nothing. Once the queue and a completed rescan show Sonarr isn't tracking it, the folder is read and the
    pending import's file map imports the files, with no downloadId and by copy.
    """

    @staticmethod
    def _strat(
        category: str = "manual",
        *,
        folder_candidates: list[ManualImportCandidate] | None = None,
    ) -> tuple[SonarrStrategy, HookedSonarrClient, QbitCategoryReads]:
        """Sonarr watching `tv-sonarr`, the pending import's torrent in `category`, and Sonarr not tracking it.

        The downloadId fetch answers `[]`, the queue is empty, a rescan finished after the add, and the
        history holds only the neutral page.
        """

        qbit = QbitCategoryReads(category)
        categories = make_fetched_categories(grab="tv-sonarr", post_import=None, qbit=qbit.client())
        strat, sonarr = make_sonarr_for_import(candidates=[], categories=categories)
        sonarr.commands_return = [RESCAN_AFTER_ADD]
        sonarr.history_page_return = HistoryPage.model_validate({"records": []})
        sonarr.folder_candidates_return = (
            [manual_candidate("/d/Show/Show - 01 [1080p].mkv")] if folder_candidates is None else folder_candidates
        )
        sonarr.path_mappings_return = []
        return strat, sonarr, qbit

    @pytest.mark.parametrize(
        ("category", "by_id", "queue", "posted"),
        [
            pytest.param("tv-sonarr", [], [], [], id="a-watched-category-waits-for-sonarr"),
            # Sonarr v4 goes on tracking a torrent moved out of its category until it restarts.
            pytest.param(
                "manual",
                [],
                [queue_record("ABC123", "importPending", status="warning")],
                [],
                id="a-queue-row-waits-for-sonarr",
            ),
            pytest.param("manual", None, [], [(None, "copy")], id="a-failed-fetch-omits-the-id"),
        ],
    )
    def test_only_a_torrent_sonarr_neither_watches_nor_tracks_imports_without_its_id(
        self,
        category: str,
        by_id: list[ManualImportCandidate] | None,
        queue: list[QueueRecord],
        posted: list[tuple[str | None, str]],
    ) -> None:
        strat, sonarr, _ = self._strat(category)
        sonarr.candidates_return = by_id
        sonarr.queue_return = queue

        strat.import_completed(pending_import(), "/d/Show", AttemptKind.ORDINARY)

        assert [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls] == posted

    def test_the_category_is_asked_again_every_poll(self) -> None:
        strat, sonarr, qbit = self._strat()
        pending = pending_import()

        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)
        qbit.category = "tv-sonarr"
        strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        # Moved into the watched category, the download gets its id back. Still a folder import, so it copies.
        posted = [(files[0].downloadId, mode) for files, mode in sonarr.execute_calls]
        assert posted == [(None, "copy"), (None, "copy"), ("ABC123", "copy")]

    def test_an_empty_folder_warns_at_the_deadline_with_the_folder_reason(self) -> None:
        recording = install_recording_hub()
        strat, sonarr, _ = self._strat(folder_candidates=[])

        result = strat.import_completed(pending_import(), "/d/Show", AttemptKind.FINAL)

        assert result.all_done is False
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == [_NOT_VISIBLE_IN_EMPTY_FOLDER_WARNING]
