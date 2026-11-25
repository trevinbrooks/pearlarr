# pyright: strict
"""Tests for removing an imported download's leftover record from Sonarr's queue.

The removal runs over a scripted Sonarr and is judged by the status it returns, what it deleted and what it noted.
"""

from pearlarr.arr_http import DeleteOutcome
from pearlarr.output import Severity
from pearlarr.wait_types import EffectStatus

from .builders import pending_import, queue_record
from .fakes import diagnostic_messages, install_recording_hub
from .sonarr_import_builders import make_sonarr_for_import


class TestQueueRecordRemoval:
    """An imported download's leftover queue record is removed through the id of any one of its rows."""

    def test_the_first_row_with_a_usable_id_is_deleted_once(self) -> None:
        # Sonarr closes the whole download off any one of its queue ids.
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=queue_id) for queue_id in (0, 11, 12)],
        )

        assert strat.remove_from_queue(pending_import()) is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]

    def test_other_downloads_rows_are_not_touched(self) -> None:
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("FFFF", "importPending", queue_id=9)],
        )

        assert strat.remove_from_queue(pending_import()) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_unknown_series_rows_are_left_for_sonarr(self) -> None:
        # Sonarr answers a delete of one with a 500 and records nothing, and the row drops once the torrent leaves the
        # category Sonarr watches. So the removal is skipped, never failed, and can't hold up the pending import.
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", status="warning", queue_id=11, series_id=0)],
        )

        assert strat.remove_from_queue(pending_import()) is EffectStatus.SKIPPED
        assert sonarr.queue_delete_calls == []

    def test_an_unreadable_queue_fails_the_removal_without_a_guessed_delete(self) -> None:
        # The client already retried the read, so the removal is left for a later run, not retried in this one.
        strat, sonarr = make_sonarr_for_import(candidates=None)
        sonarr.queue_return = None

        assert strat.remove_from_queue(pending_import()) is EffectStatus.FAILED
        assert sonarr.queue_delete_calls == []

    def test_a_removal_is_noted(self) -> None:
        recording = install_recording_hub()
        strat, _ = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )

        status = strat.remove_from_queue(pending_import(title="Show"))

        assert status is EffectStatus.DONE
        assert diagnostic_messages(recording, Severity.INFO) == [
            "Removed the imported download Show · SubGroup from Sonarr's queue"
        ]

    def test_failed_delete_asks_for_a_retry_and_stays_quiet(self) -> None:
        # The client already warned of the failure.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.FAILED

        status = strat.remove_from_queue(pending_import())

        assert status is EffectStatus.RETRY
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording, Severity.INFO) == []

    def test_a_record_already_gone_is_a_quiet_success(self) -> None:
        # Sonarr dismissed it first, as its own refresh can. That is neither a failure nor a removal to note.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(
            candidates=None,
            queue=[queue_record("ABC123", "importPending", queue_id=11)],
        )
        sonarr.queue_delete_return = DeleteOutcome.GONE

        status = strat.remove_from_queue(pending_import())

        assert status is EffectStatus.DONE
        assert sonarr.queue_delete_calls == [11]
        assert diagnostic_messages(recording) == []
