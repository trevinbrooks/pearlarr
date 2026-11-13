# pyright: strict
"""Direct tests for the wait pass's types: the pending states, the import probe, poll classification, and telemetry."""

from types import SimpleNamespace
from typing import cast

import pytest

from pearlarr.releases import EpisodeKey
from pearlarr.wait_types import (
    NO_ATTEMPT,
    AttemptResult,
    DownloadProgress,
    HandImportFile,
    LeftForHandImport,
    PauseReason,
    PendingImportState,
    TorrentStatus,
    classify_pending_import,
    download_progress_of,
)

from .builders import order_only_hand_import_files
from .test_pending_import import HALVES


class TestPendingStateAndProbe:
    """The shared carried-over status vocabulary + the import probe value object."""

    def test_pending_state_members(self) -> None:
        assert {s.name for s in PendingImportState} == {
            "DOWNLOADING",
            "DOWNLOADED",
            "IMPORTED",
            "ERRORED",
            "GONE",
        }

    def test_pending_state_is_its_string(self) -> None:
        assert PendingImportState.DOWNLOADED == "downloaded"
        assert PendingImportState.DOWNLOADING == "queued"

    def test_imported_probe_claims_the_files_without_a_command(self) -> None:
        result = AttemptResult.imported()
        assert result.all_done is True
        assert result.command_issued is False

    def test_waiting_probe_carries_the_attempt_flags(self) -> None:
        result = AttemptResult.waiting(command_issued=True, pause_reason=PauseReason.IMPORT)
        assert result.all_done is False
        assert result.command_issued is True
        assert result.pause_reason is PauseReason.IMPORT
        assert result.paused is True
        assert result.attempted is True

    def test_waiting_probe_defaults_to_no_deferral(self) -> None:
        result = AttemptResult.waiting()
        assert result.pause_reason is PauseReason.NONE
        assert result.paused is False

    def test_leave_probe_records_no_attempt(self) -> None:
        assert NO_ATTEMPT.attempted is False
        assert NO_ATTEMPT.all_done is False
        assert NO_ATTEMPT.command_issued is False

    def test_a_probes_placements_are_detached_and_read_only(self) -> None:
        placed: dict[str, tuple[int, ...]] = {"show - 01.mkv": (101,)}
        result = AttemptResult(all_done=False, command_issued=True, placements=placed)
        placed["show - 02.mkv"] = (102,)
        mutable = cast("dict[str, tuple[int, ...]]", result.placements)

        with pytest.raises(TypeError):
            mutable["show - 02.mkv"] = (102,)
        assert dict(result.placements) == {"show - 01.mkv": (101,)}

    @pytest.mark.parametrize(
        ("left", "explained"),
        [
            pytest.param(order_only_hand_import_files(), True, id="both keyed, an episode missing"),
            pytest.param(
                order_only_hand_import_files(missing=(), untrusted=(EpisodeKey(4, 1),)),
                True,
                id="both keyed, an episode holding another file",
            ),
            pytest.param(
                LeftForHandImport(
                    files=(order_only_hand_import_files().files[0], HandImportFile(HALVES[1])),
                    missing=(EpisodeKey(4, 1),),
                ),
                False,
                id="one of two keyed",
            ),
            pytest.param(order_only_hand_import_files(missing=()), False, id="no unfilled episode to name"),
            pytest.param(LeftForHandImport(missing=(EpisodeKey(4, 1),)), False, id="no file"),
        ],
    )
    def test_the_order_explains_only_keyed_files_with_an_unfilled_episode(
        self, left: LeftForHandImport, explained: bool
    ) -> None:
        assert left.explained_by_order is explained


class TestClassifyPending:
    """`classify_pending_import` folds a poll's outcome + the files-present flag into a state."""

    def test_missing(self) -> None:
        assert classify_pending_import(TorrentStatus.GONE, False) is PendingImportState.GONE

    def test_errored(self) -> None:
        assert classify_pending_import(TorrentStatus.ERRORED, False) is PendingImportState.ERRORED

    def test_still_downloading_is_queued(self) -> None:
        assert classify_pending_import(None, False) is PendingImportState.DOWNLOADING

    def test_complete_and_files_present_is_imported(self) -> None:
        assert classify_pending_import(TorrentStatus.COMPLETE, True) is PendingImportState.IMPORTED

    def test_complete_without_files_is_importing(self) -> None:
        # The copy is still in flight -> downloaded, never imported, until the
        # files are verified present.
        assert classify_pending_import(TorrentStatus.COMPLETE, False) is PendingImportState.DOWNLOADED


class TestSanitizeTorrentTelemetry:
    """MUTATION PIN: pins the pure telemetry sanitizer's clamps and sentinel folds.

    Covers the numeric-string `_as_float` path too - a cluster of ~10 surviving mutants.
    """

    @pytest.mark.parametrize(
        ("progress", "dlspeed", "eta", "completed", "size", "expected"),
        [
            # All-None getattr reads fold to the empty telemetry.
            (None, None, None, None, None, DownloadProgress(0.0, None, None, None, None)),
            # A clean row passes through untouched.
            (0.64, 3_200_000, 130, 1_800, 2_900, DownloadProgress(0.64, 3_200_000, 130, 1_800, 2_900)),
            # NaN progress (float and string) folds to 0.0, not a poisoned bar.
            (float("nan"), 100, 130, 50, 200, DownloadProgress(0.0, 100, 130, 50, 200)),
            ("nan", None, None, None, None, DownloadProgress(0.0, None, None, None, None)),
            # Numeric-string progress parses. Junk folds to 0.0.
            ("0.75", None, None, None, None, DownloadProgress(0.75, None, None, None, None)),
            ("fast", None, None, None, None, DownloadProgress(0.0, None, None, None, None)),
            # Progress clamps to [0, 1] on both ends.
            (1.5, None, None, None, None, DownloadProgress(1.0, None, None, None, None)),
            (-0.25, None, None, None, None, DownloadProgress(0.0, None, None, None, None)),
            # Idle (0) and negative speeds read as "no speed", never a 0 B/s row.
            (0.5, 0, None, None, None, DownloadProgress(0.5, None, None, None, None)),
            (0.5, -5, None, None, None, DownloadProgress(0.5, None, None, None, None)),
            # qBittorrent's 8_640_000 "infinite" eta and a 0/negative eta are unknown.
            # The last finite second still renders.
            (0.5, 100, 8_640_000, None, None, DownloadProgress(0.5, 100, None, None, None)),
            (0.5, 100, 0, None, None, DownloadProgress(0.5, 100, None, None, None)),
            (0.5, 100, 8_639_999, None, None, DownloadProgress(0.5, 100, 8_639_999, None, None)),
            # Zero/negative byte counts are unknown, not empty-progress readings.
            (0.5, None, None, 0, 0, DownloadProgress(0.5, None, None, None, None)),
            (0.5, None, None, -3, -1, DownloadProgress(0.5, None, None, None, None)),
            # An over-count clamps done to the total, never a >100% bar.
            (0.5, None, None, 500, 200, DownloadProgress(0.5, None, None, 200, 200)),
            # Bytes done without a known total still renders.
            (0.5, None, None, 100, None, DownloadProgress(0.5, None, None, 100, None)),
        ],
    )
    def test_edge_inputs(
        self,
        progress: object,
        dlspeed: object,
        eta: object,
        completed: object,
        size: object,
        expected: DownloadProgress,
    ) -> None:
        row = SimpleNamespace(progress=progress, dlspeed=dlspeed, eta=eta, completed=completed, size=size)

        assert download_progress_of(row) == expected

    def test_attrless_row_folds_to_the_zero_reading(self) -> None:
        # The fields are read best-effort off the row: a row missing them all is the empty telemetry.
        assert download_progress_of(object()) == DownloadProgress(0.0, None, None, None, None)


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in TorrentStatus} == {"COMPLETE", "ERRORED", "GONE"}
