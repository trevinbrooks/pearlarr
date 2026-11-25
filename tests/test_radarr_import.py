# pyright: strict
"""Tests for a Radarr import, which reads Radarr's own history for proof that Radarr imported the download.

How the end of a run acts on that proof is in `test_import_wait_radarr.py`.
"""

from datetime import UTC, datetime, timedelta

import pytest

from pearlarr.arr_activity import HISTORY_MAX_LOOKBACK_DAYS, format_history_date
from pearlarr.arr_models import HistoryRecord, RadarrMovie
from pearlarr.config import Arr
from pearlarr.stamps import parse_stamp, stamp_of
from pearlarr.wait_types import AttemptKind, ImportProgress

from .builders import FakeCacheStore, make_radarr_sync, pending_import
from .fakes import FakeRadarrClient


def _import_history(download_id: str, *, event: str = "movieFolderImported") -> HistoryRecord:
    """One Radarr `/history/since` record, as the import-evidence match reads it."""

    return HistoryRecord.model_validate({"eventType": event, "downloadId": download_id, "movieId": 5})


class TestRadarrImportCompletedHistory:
    """Radarr reconciles a completed download off its OWN import history (evidence, never action).

    Radarr's completed-download handling does the import; `import_completed` only
    reads history as proof so the pending import can drop and the category gate release.
    """

    @pytest.mark.parametrize(
        ("history", "done"),
        [
            pytest.param([_import_history("h")], True, id="import-event"),
            # Radarr stores the download id uppercased.
            pytest.param([_import_history("H")], True, id="import-event-id-uppercased"),
            # No import proof (none yet, a non-import event, or Radarr unreachable): fail
            # closed: waiting keeps the pending import waiting, never done on an unread history.
            pytest.param([], False, id="no-event-yet"),
            pytest.param([_import_history("h", event="grabbed")], False, id="non-import-event"),
            pytest.param(None, False, id="history-outage"),
        ],
    )
    def test_only_an_import_event_for_the_download_ends_the_pending_import(
        self, history: list[HistoryRecord] | None, done: bool
    ) -> None:
        radarr = FakeRadarrClient()
        radarr.history_since_return = history
        strat = make_radarr_sync(radarr=radarr)

        result = strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)

        assert result.all_done is done

    def test_history_fetched_once_per_wait_pass(self) -> None:
        # Two pending imports in one wait pass share ONE history fetch (memoized).
        radarr = FakeRadarrClient(history_since=[_import_history("h")])
        strat = make_radarr_sync(radarr=radarr)

        strat.import_completed(pending_import(infohash="h"), "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending_import(infohash="other"), "/d", AttemptKind.ORDINARY)

        assert len(radarr.history_calls) == 1

    def test_the_next_arr_run_reads_the_history_again(self) -> None:
        # The import landed between two runs. A history held over from the first would never show it.
        radarr = FakeRadarrClient()
        strat = make_radarr_sync(radarr=radarr)
        pending = pending_import(infohash="h")

        earlier = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        radarr.history_since_return = [_import_history("h")]
        strat.get_items()
        later = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert (earlier.all_done, later.all_done) == (False, True)

    def test_history_is_asked_for_no_further_back_than_the_cap(self) -> None:
        store, radarr = FakeCacheStore(), FakeRadarrClient()
        long_ago = stamp_of(datetime.now() - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS + 10))
        pending = pending_import(infohash="h", added_at=long_ago)
        store.put_pending_import_row(Arr.RADARR, "h", pending.to_json())
        strat = make_radarr_sync(radarr=radarr, cache_store=store)
        # Two hours under the cap, for a Radarr clock a little behind pearlarr's.
        back = timedelta(days=HISTORY_MAX_LOOKBACK_DAYS, hours=2)

        earliest = format_history_date(datetime.now(UTC) - back)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        latest = format_history_date(datetime.now(UTC) - back)

        assert [earliest <= since <= latest for since in radarr.history_calls] == [True]

    def test_history_window_starts_at_the_oldest_stamp_inside_the_cap(self) -> None:
        older = stamp_of(datetime.now() - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS - 10))
        newer = stamp_of(datetime.now() - timedelta(days=1))
        store, radarr = FakeCacheStore(), FakeRadarrClient()
        latest = pending_import(infohash="new", added_at=newer)
        store.put_pending_import_row(Arr.RADARR, "new", latest.to_json())
        store.put_pending_import_row(Arr.RADARR, "old", pending_import(infohash="old", added_at=older).to_json())
        strat = make_radarr_sync(radarr=radarr, cache_store=store)

        strat.import_completed(latest, "/d", AttemptKind.ORDINARY)

        # Two hours before the stamp, for a Radarr clock a little behind pearlarr's.
        start = parse_stamp(older).astimezone(UTC) - timedelta(hours=2)
        assert radarr.history_calls == [format_history_date(start)]

    def test_pending_import_series_id_is_none(self) -> None:
        # Radarr pending imports carry series_id=0, so the per-item snapshot hook key is
        # None (short-circuits the per-item snapshot - reconcile runs in the wait pass).
        strat = make_radarr_sync()

        assert strat.pending_import_series_id(RadarrMovie(id=5)) is None

    def test_import_progress_is_indeterminate_zero(self) -> None:
        # Radarr never enters the blocking wait, so the progress poll bar hook
        # returns the safe "no bar, promote nothing" value.
        strat = make_radarr_sync()

        assert strat.import_progress(pending_import()) == ImportProgress(0, 0, determinate=False)
