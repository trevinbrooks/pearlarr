# pyright: strict
"""Tests that each save of the wait writes the pending import an attempt followed onto Sonarr's rows.

Only the titles whose ids moved lose their cache stamp, so the next run checks them again.
"""

from dataclasses import replace

import pytest

from pearlarr.config import Arr
from pearlarr.import_wait import ImportWaitManager
from pearlarr.pending_follow import follow
from pearlarr.pending_import import PendingImport
from pearlarr.wait_types import AttemptResult, EffectStatus, PauseReason

from .builders import FakeCacheStore, attempt_result, pending_import, sonarr_ep, two_claim_pending_import
from .wait_fakes import FakeQbit, FakeTorrent, RecordingStrategy, RecordingWaitView, make_orchestration_manager

_MOVED_DOWN_ONE = [
    sonarr_ep(1, number, ep_id=100 + number, tvdb_id=8999 + number, episode_file_id=0) for number in (1, 2, 3)
]
"""A read of series 7 with TVDB episode 9001 on row 102 and 9002 on row 103."""


_FOLLOWED_ROW = {"show - 01 [1080p].mkv": [102], "show - 02 [1080p].mkv": [103]}
"""The file map `_followed` gives `_two_titles_stored`, as the stored row holds it."""


def _two_titles_stored(*, awaiting_cleanup: bool = False) -> PendingImport:
    """Torrent "h" of series 7, claimed by title 5 on row 101 and title 6 on row 102.

    Stored while row 101 held TVDB episode 9001 and row 102 held 9002.
    """

    return two_claim_pending_import(
        anilist_ids=(5, 6),
        windows=((101,), (102,)),
        infohash="h",
        file_episode_map={"show - 01 [1080p].mkv": (101,), "show - 02 [1080p].mkv": (102,)},
        seadex_files=("Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv"),
        tvdb_ids={101: 9001, 102: 9002},
        awaiting_cleanup=awaiting_cleanup,
    )


def _followed(pending: PendingImport) -> PendingImport:
    """`pending` as an attempt that read `_MOVED_DOWN_ONE` ends with it."""

    followed = follow(pending, {7: _MOVED_DOWN_ONE})
    assert followed is not None
    return followed.pending


class TestWaitSavesTheFollowedPendingImport:
    """The pending import an attempt followed is what every save of the wait writes."""

    @staticmethod
    def _manager(strategy: RecordingStrategy, store: FakeCacheStore, pending: PendingImport) -> ImportWaitManager:
        """A manager over `store` with `pending` stored and its torrent complete in qBittorrent."""

        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        return make_orchestration_manager(
            qbit=FakeQbit({"h": [FakeTorrent(is_complete=True, content_path="/d")]}), strategy=strategy, store=store
        )

    @pytest.mark.parametrize(
        ("answer", "awaiting_cleanup"),
        [
            # Every file is still in, and the queue removal fails again.
            pytest.param(attempt_result(all_done=True), True, id="failed-retire-keeps"),
            pytest.param(
                attempt_result(all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED),
                False,
                id="reissue-demotes",
            ),
        ],
    )
    def test_the_cleanup_pass_after_an_attempt_that_followed_keeps_the_followed_ids(
        self, answer: AttemptResult, awaiting_cleanup: bool
    ) -> None:
        pending = _two_titles_stored(awaiting_cleanup=True)
        result = replace(answer, pending=_followed(pending), followed=True)
        strategy = RecordingStrategy(attempts=[result], removals=[EffectStatus.FAILED])
        store = FakeCacheStore()
        mgr = self._manager(strategy, store, pending)

        mgr.finish_cleanups()

        row = store.get_pending_import_rows(Arr.SONARR)["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (awaiting_cleanup, _FOLLOWED_ROW)

    @pytest.mark.parametrize(
        ("stored", "ended", "followed", "uncached"),
        [
            pytest.param(_two_titles_stored(), _followed(_two_titles_stored()), True, [5, 6], id="followed-ids"),
            pytest.param(
                pending_import(infohash="h", anilist_id=5, file_episode_map={}),
                pending_import(infohash="h", anilist_id=5),
                False,
                [],
                id="placements-alone",
            ),
        ],
    )
    def test_a_wait_attempt_stores_what_it_ended_with_and_uncaches_only_titles_whose_ids_moved(
        self, stored: PendingImport, ended: PendingImport, followed: bool, uncached: list[int]
    ) -> None:
        store = FakeCacheStore()
        for anilist_id in (5, 6, 9):
            store.update_cached_title(Arr.SONARR, anilist_id, {"updated_at": "2021-06-05 00:00:00"})
        issued = attempt_result(
            all_done=False, command_issued=True, pause_reason=PauseReason.ISSUED, pending=ended, followed=followed
        )
        mgr = self._manager(RecordingStrategy(attempts=[issued]), store, stored)

        mgr.run_deferred(view=RecordingWaitView())

        assert store.get_pending_import_rows(Arr.SONARR)["h"] == ended.to_json()
        titles = {anilist_id: store.get_cached_title(Arr.SONARR, anilist_id) for anilist_id in (5, 6, 9)}
        assert [key for key, title in titles.items() if title is not None and title.updated_at is None] == uncached

    def test_a_wait_import_whose_cleanup_fails_is_kept_with_the_followed_ids_and_still_counted(self) -> None:
        pending = _two_titles_stored()
        imported = attempt_result(all_done=True, pending=_followed(pending), followed=True)
        store = FakeCacheStore()
        mgr = self._manager(RecordingStrategy(attempts=[imported], removals=[EffectStatus.FAILED]), store, pending)

        result = mgr.run_deferred(view=RecordingWaitView())

        row = store.get_pending_import_rows(Arr.SONARR)["h"]
        assert (row.get("awaiting_cleanup"), row["file_episode_map"]) == (True, _FOLLOWED_ROW)
        assert result is not None
        assert result.carried_over_imported == 1
