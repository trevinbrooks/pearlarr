# pyright: strict
"""Tests for which Sonarr rows an import posts each file on.

A file goes under the series its claim names, on the row that holds its TVDB episode at that read, own-only rows
included. A row that already holds a file gets this one only while it is still owed it.
"""

from dataclasses import replace

import pytest

from pearlarr.arr_models import CommandResource, ManualImportCandidate, SonarrEpisode
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.file_paths import normalize_file_name
from pearlarr.output import Severity
from pearlarr.pending_import import FileEpisodeMap, OwnedAtGrab, PendingImport, SizeMatchedEpisode, TvdbIds
from pearlarr.wait_types import NO_PROGRESS, AttemptKind, AttemptResult, ImportProgress, PauseReason, PendingImportState

from .builders import (
    FakeCacheStore,
    make_import_wait_manager,
    make_run_deps,
    make_sonarr_sync,
    manual_candidate,
    pending_import,
    sonarr_ep,
    title_claim,
    two_claim_pending_import,
)
from .fakes import FakeSonarrClient, PerSeriesSonarrClient, diagnostic_messages, install_recording_hub
from .sonarr_import_builders import (
    FIRST_FILE,
    SECOND_FILE,
    both_files,
    imported_history,
    make_sonarr_for_import,
    posted_imports,
)
from .wait_fakes import FakeQbit, FakeTorrent, stored_awaiting_cleanup

_SHOW_FILE = "Show - 01 [1080p].mkv"


_OTHER_FILE = "Other - 01 [1080p].mkv"


def _two_series_pending_import(
    files: FileEpisodeMap | None = None, *, other_window: tuple[int, ...] = (201,)
) -> PendingImport:
    """One torrent claimed on series 7 (window 101) and series 8 (`other_window`), its listing fully placed."""

    mapped = {_SHOW_FILE: (101,), _OTHER_FILE: (201,)} if files is None else files
    return two_claim_pending_import(
        series_ids=(7, 8),
        windows=((101,), other_window),
        titles=("Show", "Other"),
        file_episode_map=mapped,
        seadex_files=list(mapped),
    )


def _posted_series(sonarr: FakeSonarrClient) -> dict[str, int]:
    """Each POSTed file's `seriesId` keyed by its path, across every execute."""

    return {entry.path: entry.seriesId for files, _ in sonarr.execute_calls for entry in files}


class TestMultiSeriesImport:
    """A two-series pending import posts each file under the series its claim names and is done as one."""

    @staticmethod
    def _candidates(*names: str) -> list[ManualImportCandidate]:
        return [manual_candidate(f"/d/{name}") for name in names]

    @pytest.mark.parametrize(
        ("pending", "series_8_row"),
        [
            # A claim's window names its file's id, whatever the indexes hold (here no index holds 201).
            pytest.param(_two_series_pending_import(), 202, id="the-claims-window-names-the-id"),
            # The series-8 claim has no window, so the index holding 201 says where its file goes.
            pytest.param(_two_series_pending_import(other_window=()), 201, id="no-window-so-the-index-holding-it"),
        ],
    )
    def test_each_file_posts_under_its_own_claims_series(self, pending: PendingImport, series_8_row: int) -> None:
        sonarr = PerSeriesSonarrClient(
            {
                7: [sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
                8: [sonarr_ep(1, 1, ep_id=series_8_row, episode_file_id=0)],
            },
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )
        strat = make_sonarr_sync(sonarr=sonarr)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert _posted_series(sonarr) == {f"/d/{_SHOW_FILE}": 7, f"/d/{_OTHER_FILE}": 8}

    def test_a_file_no_claim_or_index_routes_waits_and_warns_at_the_deadline(self) -> None:
        # 301 sits in neither claim's window and in no series index: nothing is POSTed, the poll
        # waits, and only the deadline attempt names the file.
        recording = install_recording_hub()
        stray = "Stray - 01 [1080p].mkv"
        pending = _two_series_pending_import({stray: (301,)})
        strat, sonarr = make_sonarr_for_import(candidates=self._candidates(stray))

        polled = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert polled == replace(AttemptResult.waiting(), file_map_episode_count=1, pending=pending)
        assert diagnostic_messages(recording, Severity.WARNING) == []

        deadline = strat.import_completed(pending, "/d", AttemptKind.FINAL)

        assert deadline == replace(AttemptResult.waiting(), file_map_episode_count=1, pending=pending)
        assert sonarr.execute_calls == []
        [warning] = diagnostic_messages(recording, Severity.WARNING)
        assert "resolve to no series" in warning
        assert normalize_file_name(stray) in warning

    def test_imported_only_when_every_series_file_map_episodes_are_done(self) -> None:
        # Series 7's file map episode already holds the pending import's group while series 8's has no file: the
        # pending import is not done, and only the series-8 file is POSTed. Once that is imported, the next poll
        # finds it done.
        pending = _two_series_pending_import()
        sonarr = PerSeriesSonarrClient(
            {
                7: [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")],
                8: [sonarr_ep(1, 1, ep_id=201, episode_file_id=0)],
            },
            candidates=self._candidates(_SHOW_FILE, _OTHER_FILE),
        )
        strat = make_sonarr_sync(sonarr=sonarr)

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert first.all_done is False
        assert first.command_issued is True
        assert (first.imported_count, first.file_map_episode_count) == (1, 2)
        assert _posted_series(sonarr) == {f"/d/{_OTHER_FILE}": 8}

        sonarr.episodes_by_series[8] = [sonarr_ep(1, 1, ep_id=201, release_group="SubGroup")]
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert second.all_done is True
        assert (second.imported_count, second.file_map_episode_count) == (2, 2)
        assert len(sonarr.execute_calls) == 1


_TWO_EPISODES_STORED = pending_import(
    file_episode_map={FIRST_FILE: (101,), SECOND_FILE: (102,)},
    seadex_files=[FIRST_FILE, SECOND_FILE],
    ordered_episode_ids=(101, 102),
    tvdb_ids={101: 9001, 102: 9002},
)
"""A two-file pending import of title 1 on series 7, stored while row 101 held TVDB episode 9001 and row 102 held
9002."""


def _moved_down_one() -> list[SonarrEpisode]:
    """Series 7 with an episode numbered ahead of the stored two: 9001 sits on row 102 and 9002 on a row 103.

    No row has a file.
    """

    return [sonarr_ep(1, number, ep_id=100 + number, tvdb_id=8999 + number, episode_file_id=0) for number in (1, 2, 3)]


class TestImportFollowsEpisodeIds:
    """An import acts on the pending import with each id on the row that holds its TVDB episode at that read."""

    def test_each_file_is_posted_on_the_row_that_holds_its_episode(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=both_files(), episodes=_moved_down_one())

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == [[(f"/d/{FIRST_FILE}", [102]), (f"/d/{SECOND_FILE}", [103])]]
        assert result.followed is True

    def test_the_counts_and_the_result_are_on_the_rows_followed_to_and_the_store_is_left_alone(self) -> None:
        store = FakeCacheStore()
        store.put_pending_import_row(Arr.SONARR, "abc123", _TWO_EPISODES_STORED.to_json())
        rows = [*_moved_down_one()[:2], sonarr_ep(1, 3, ep_id=103, tvdb_id=9002, release_group="SubGroup")]
        strat, _ = make_sonarr_for_import(candidates=both_files(), episodes=rows, cache_store=store)

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert (result.imported_count, result.file_map_episode_count) == (1, 2)
        assert strat.import_progress(_TWO_EPISODES_STORED) == ImportProgress(1, 2, determinate=True)
        assert result.pending is not None
        assert result.pending.file_episode_map == {
            normalize_file_name(FIRST_FILE): (102,),
            normalize_file_name(SECOND_FILE): (103,),
        }
        assert result.pending.tvdb_ids == TvdbIds({102: 9001, 103: 9002})
        assert store.get_pending_import_row(Arr.SONARR, "abc123") == _TWO_EPISODES_STORED.to_json()

    def test_a_pending_import_imported_by_the_ids_it_stores_ends_imported_as_stored(self) -> None:
        # Both files went in on the stored rows before the episodes moved down. Following would post the second
        # file again, onto row 103.
        rows = [
            sonarr_ep(1, 1, ep_id=101, tvdb_id=9000, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=102, tvdb_id=9001, release_group="SubGroup"),
            sonarr_ep(1, 3, ep_id=103, tvdb_id=9002, episode_file_id=0),
        ]
        strat, sonarr = make_sonarr_for_import(candidates=both_files(), episodes=rows)

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert (result.all_done, result.followed, result.pending) == (True, False, _TWO_EPISODES_STORED)
        assert sonarr.execute_calls == []

    def test_a_progress_poll_never_counts_a_pending_import_the_follow_took_a_file_from(self) -> None:
        only_the_first = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, release_group="SubGroup")]
        strat, _ = make_sonarr_for_import(candidates=[], episodes=only_the_first)

        assert strat.import_progress(_TWO_EPISODES_STORED) == NO_PROGRESS

    def test_the_progress_on_a_failed_episode_read_is_indeterminate(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=[])
        sonarr.episodes_return = None

        assert strat.import_progress(pending_import()) == NO_PROGRESS

    def test_a_file_sonarrs_history_placed_is_stored_with_the_tvdb_episode_its_row_has_at_that_poll(self) -> None:
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.history_page_return = imported_history((101, f"/d/Show/{FIRST_FILE}"))
        sonarr.folder_candidates_return = []
        sonarr.path_mappings_return = []
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, release_group="SubGroup")]
        pending = pending_import(file_episode_map={}, ordered_episode_ids=[101], seadex_files=[FIRST_FILE])

        result = strat.import_completed(pending, "/d/Show", AttemptKind.ORDINARY)

        assert result.pending is not None
        assert result.pending.file_episode_map == {normalize_file_name(FIRST_FILE): (101,)}
        assert result.pending.tvdb_ids == TvdbIds({101: 9001})

    def test_a_hand_import_running_on_a_row_followed_to_holds_the_import(self) -> None:
        by_hand = CommandResource.model_validate(
            {"id": 7000, "name": "ManualImport", "status": "started", "body": {"files": [{"episodeIds": [103]}]}}
        )
        strat, sonarr = make_sonarr_for_import(candidates=both_files(), episodes=_moved_down_one(), commands=[by_hand])

        result = strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        assert result.pause_reason is PauseReason.IMPORT
        assert sonarr.execute_calls == []

    def test_a_later_attempt_on_the_pending_import_as_stored_does_not_warn_of_the_file_again(self) -> None:
        # An attempt that ends in an error saves nothing, so the next one follows the stored pending import again.
        recording = install_recording_hub()
        only_the_first = [sonarr_ep(1, 1, ep_id=101, tvdb_id=9001, episode_file_id=0)]
        strat, _ = make_sonarr_for_import(candidates=both_files(), episodes=only_the_first)

        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)
        strat.import_completed(_TWO_EPISODES_STORED, "/d", AttemptKind.ORDINARY)

        left_out = normalize_file_name(SECOND_FILE)
        warnings = [message for message in diagnostic_messages(recording, Severity.WARNING) if left_out in message]
        assert len(warnings) == 1


class TestImportOnOwnOnlyRows:
    """A file placed on an own-only row is imported like any other file of the pending import."""

    def test_files_on_such_rows_are_posted_on_them_and_the_pending_import_ends_once_they_are_in(self) -> None:
        # The aired opener 401 is two halves, rows 9001 and 9002, in the order Sonarr numbers the series by. The
        # first two files sit on the halves, outside the title's window, and the whole opener's row never gets one.
        names = ["Show - 01 [1080p].mkv", "Show - 02 [1080p].mkv", "Show - 03 [1080p].mkv"]
        pending = pending_import(
            file_episode_map={names[0]: (9001,), names[1]: (9002,), names[2]: (402,)},
            seadex_files=names,
            ordered_episode_ids=(401, 402),
            tvdb_ids={9001: 9001, 9002: 9002, 401: 401, 402: 402},
        )
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate(f"/d/{name}") for name in names])
        third = sonarr_ep(1, 3, ep_id=402, tvdb_id=402, release_group="SubGroup")
        opener = sonarr_ep(1, 21, ep_id=401, tvdb_id=401, episode_file_id=0)

        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=9001, tvdb_id=9001, episode_file_id=0),
            sonarr_ep(1, 2, ep_id=9002, tvdb_id=9002, episode_file_id=0),
            third,
            opener,
        ]
        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        sonarr.episodes_return = [
            sonarr_ep(1, 1, ep_id=9001, tvdb_id=9001, release_group="SubGroup"),
            sonarr_ep(1, 2, ep_id=9002, tvdb_id=9002, release_group="SubGroup"),
            third,
            opener,
        ]
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == [[(f"/d/{names[0]}", [9001]), (f"/d/{names[1]}", [9002])]]
        assert (first.all_done, first.imported_count, first.file_map_episode_count) == (False, 1, 3)
        assert second.all_done is True


class TestImportOverEpisodeFiles:
    """Whether an episode that already holds a file is still owed this pending import's file.

    An episode holding another episode's file, or a group only another claim on the series trusts, is still owed.
    One holding a file the guard trusts is not, whether the trust comes from a size matched at the grab or from guards
    stored apart from the pending import. A pending import awaiting its cleanup whose files are gone is imported again.
    """

    def test_an_episode_holding_this_torrents_file_for_another_episode_is_imported_over(self) -> None:
        # This torrent's E02 file (size 200) is filed on E01 as well. Both rows hold its group at a listed size,
        # but E01's is not its own file, so E01 is still owed and its own file is posted onto it.
        pending = pending_import(
            file_episode_map={FIRST_FILE: [101], SECOND_FILE: [102]},
            seadex_files=[FIRST_FILE, SECOND_FILE],
            release_sizes=[100, 200],
            sizes_by_name={FIRST_FILE.casefold(): 100, SECOND_FILE.casefold(): 200},
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=both_files(),
            episodes=[sonarr_ep(1, n, ep_id=100 + n, release_group="SubGroup", size=200) for n in (1, 2)],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == [[(f"/d/{FIRST_FILE}", [101])]]
        assert (result.imported_count, result.file_map_episode_count) == (1, 2)

    def test_two_claims_on_one_series_each_judge_their_own_episode_by_their_own_guards(self) -> None:
        # B's releases carry group G and A's don't. G files sit on both windows' episodes, so A's episode still
        # needs importing while B's is done. The series' merged guards would trust G for both.
        pending = pending_import(
            file_episode_map={_SHOW_FILE: [101], _OTHER_FILE: [102]},
            claims=(
                title_claim(anilist_id=1, ordered_episode_ids=(101,)),
                title_claim(anilist_id=2, ordered_episode_ids=(102,), guards=OwnedAtGrab(entry_groups=("G",))),
            ),
        )
        on_disk = [sonarr_ep(1, n, ep_id=100 + n, release_group="G") for n in (1, 2)]
        strat, _ = make_sonarr_for_import(candidates=[], episodes=on_disk)

        assert strat.import_progress(pending) == ImportProgress(1, 2, determinate=True)

    def test_file_map_episode_holding_a_file_with_no_group_pearlarr_size_matched_is_protected(self) -> None:
        # The grab-time size match is carried on the pending import through to the
        # import: a file with no group the planner named reads TRUSTED, so the
        # torrent pearlarr grabbed for the REST of the season never copies over it. A download already in gets no
        # command at all, not even a rescan.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            guards=OwnedAtGrab(owned_episodes=(SizeMatchedEpisode(101, 700),)),
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group=None, size=700)],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert (sonarr.execute_calls, sonarr.refresh_calls) == ([], 0)

    def test_store_persisted_guard_row_protects_through_the_pending_import_snapshot(self) -> None:
        # THE fix path end to end: the persisted blob carries NO guards - the
        # title's guard_facts row alone must feed the loaded pending import's
        # protection decision when the carried-over snapshot drives the import.
        pending = pending_import(
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
            guards=OwnedAtGrab(entry_groups=("SubGroup", "OtherPick")),
        )
        store = FakeCacheStore()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            episodes=[sonarr_ep(1, 1, ep_id=101, release_group="OtherPick")],
            cache_store=store,
        )
        (claim,) = pending.claims
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        store.put_guards(Arr.SONARR, claim.anilist_id, claim.guards)
        state = ArrRunState(arr=Arr.SONARR)
        qbit = FakeQbit({pending.infohash: [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_import_wait_manager(
            make_run_deps(cache_store=store, qbit=qbit.client()), state=state, strategy=strat
        )

        mgr.report_imports_for_series(claim.series_id)

        # The row's facts reached the decision: the on-disk OtherPick file reads
        # TRUSTED - done, never auto imported over - and the
        # finished pending import leaves the store. The IMPORTED classification pins that
        # branch (a GONE drop would also empty the store).
        assert sonarr.execute_calls == []
        assert store.get_pending_import_rows(Arr.SONARR) == {}
        assert state.pending_import_states[pending.infohash] is PendingImportState.IMPORTED

    def test_a_pending_import_awaiting_cleanup_with_files_gone_demotes_through_the_reconciler(self) -> None:
        # The cleanup's demote reads the POST's ISSUED reason off the real attempt result:
        # a pending import awaiting cleanup whose files are gone re-imports and drops back to an
        # ordinary carryover (the fix's wiring, not a scripted attempt result).
        recording = install_recording_hub()
        pending = pending_import(awaiting_cleanup=True)
        store = FakeCacheStore()
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv")],
            cache_store=store,
        )
        store.put_pending_import_row(Arr.SONARR, pending.infohash, pending.to_json())
        state = ArrRunState(arr=Arr.SONARR)
        qbit = FakeQbit({pending.infohash: [FakeTorrent(is_complete=True, content_path="/d")]})
        mgr = make_import_wait_manager(
            make_run_deps(cache_store=store, qbit=qbit.client()), state=state, strategy=strat
        )

        mgr.finish_cleanups()

        assert len(sonarr.execute_calls) == 1
        assert stored_awaiting_cleanup(store) == {pending.infohash: False}
        assert state.pending_import_states[pending.infohash] is PendingImportState.DOWNLOADED
        assert any("no longer has every file imported" in w for w in diagnostic_messages(recording, Severity.WARNING))
