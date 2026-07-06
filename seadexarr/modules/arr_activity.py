"""Arr-side activity detection between SeaDex passes.

The cached-entry skip keys only on SeaDex's ``updated_at``, so a file the arr
replaced under an *unchanged* SeaDex entry (quality upgrade, manual grab,
re-download after delete) would never be re-detected. Each run therefore polls
the arr's ``/api/v3/history/since`` once, maps the file-state-changing records
to touched item ids, and the run loop marks their AniList ids dirty - bypassing
the skip for exactly those ids.

Pure scan logic: no imports from the run machinery (the loop injects the fetch
callable), so the module stays cycle-free.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import final

from .cache import AbstractCacheStore, HistoryCheckpoint
from .config import Arr
from .seadex_types import HistoryRecord

# Re-query overlap behind the checkpoint (absorbs tz/clock skew between the arr
# server and us) and the hard lookback cap for a long-idle checkpoint.
HISTORY_QUERY_OVERLAP_HOURS = 26
HISTORY_MAX_LOOKBACK_DAYS = 30

# File-state-changing events (casefolded). Deletes are checked against the
# upgrade reason (an upgrade-delete always pairs with an import event);
# grabbed/renamed/failed/ignored change no file state and are excluded.
_IMPORT_EVENTS = frozenset({"downloadfolderimported", "seriesfolderimported", "moviefolderimported"})
_DELETE_EVENTS = frozenset({"episodefiledeleted", "moviefiledeleted"})

_HISTORY_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True, slots=True)
class ActivityScan:
    """One scan's outcome: the touched item ids, or a re-scan-everything signal.

    ``rescan_all`` is set when history coverage is broken - a checkpoint older
    than the lookback window, or an unreadable stored date - so the id cursor
    may have skipped file changes; the caller must treat every entry as dirty
    once (``touched`` is empty then).
    """

    touched: frozenset[int]
    rescan_all: bool


def parse_history_date(value: str) -> datetime | None:
    """Parse an arr ISO8601 stamp to an aware UTC datetime, or None."""

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_history_date(value: datetime) -> str:
    """Render an aware datetime as the arr-facing UTC ISO8601 stamp."""

    return value.astimezone(UTC).strftime(_HISTORY_DATE_FORMAT)


@final
class ArrActivityMonitor:
    """One arr's per-run history scan + checkpoint advance.

    ``scan`` reads the stored checkpoint, queries the overlapped window, dedups
    on the id cursor and stashes the advanced checkpoint;
    ``commit_checkpoint`` stages it through the cache store - only called by the
    run loop when the pass covered the whole library, and only persisted at a
    non-preview save point (so a dry run never advances the cursor).
    """

    def __init__(self, arr: Arr, cache_store: AbstractCacheStore, logger: logging.Logger) -> None:
        self._arr = arr
        self._cache_store = cache_store
        self._logger = logger
        self._pending_checkpoint: HistoryCheckpoint | None = None

    def scan(
        self,
        fetch: Callable[[str], list[HistoryRecord] | None],
        *,
        now: datetime | None = None,
    ) -> ActivityScan:
        """Scan the window since the checkpoint for arr-side file changes.

        Fetch failure (None) fails open: warn, mark nothing dirty, leave the
        checkpoint untouched (a coverage gap is then re-detected next pass). An
        empty window stashes no checkpoint either (the bootstrap retries next
        pass). Own grabs - records whose ``downloadId`` matches a remembered or
        pending infohash - are suppressed. Broken coverage (checkpoint beyond
        the lookback, or an unreadable stored date) returns ``rescan_all``
        instead of a touched set.

        Args:
            fetch (Callable[[str], list[HistoryRecord] | None]): The strategy's
                ``history_since`` (takes the ISO8601 query date).
            now (datetime | None): Injectable clock for tests (aware UTC).
        """

        now = now if now is not None else datetime.now(UTC)
        floor = now - timedelta(days=HISTORY_MAX_LOOKBACK_DAYS)
        checkpoint = self._cache_store.get_history_checkpoint(self._arr)
        rescan_all = False
        if checkpoint is None:
            # Bootstrap: no cursor at all, so cover the full lookback once.
            query_date = floor
        else:
            parsed = parse_history_date(checkpoint.since_date)
            if parsed is None:
                # Unreadable cursor date: coverage unknown - replay the full
                # lookback and re-check everything once.
                query_date = floor
                rescan_all = True
            else:
                query_date = min(parsed, now) - timedelta(hours=HISTORY_QUERY_OVERLAP_HOURS)
                if query_date < floor:
                    # The clamp truncates the window: events between the true
                    # cursor and the floor are unreachable, so re-check everything.
                    query_date = floor
                    rescan_all = True

        records = fetch(format_history_date(query_date))
        if records is None:
            self._logger.warning(
                f"Could not read {self._arr.capitalize()} history; skipping activity detection this run",
            )
            return ActivityScan(touched=frozenset(), rescan_all=False)

        last_id = checkpoint.last_id if checkpoint is not None else 0
        fresh = [record for record in records if record.id > last_id]
        if fresh:
            newest = max(fresh, key=lambda record: record.id)
            # An empty date would collapse the next window to the overlap; keep
            # the old cursor and let the id dedup absorb the re-delivery.
            if newest.date:
                self._pending_checkpoint = HistoryCheckpoint(since_date=newest.date, last_id=newest.id)
        if rescan_all or not fresh:
            return ActivityScan(touched=frozenset(), rescan_all=rescan_all)

        own = self._cache_store.own_download_ids(self._arr)
        touched: set[int] = set()
        for record in fresh:
            if not _is_file_change(record):
                continue
            if record.download_id is not None and record.download_id.casefold() in own:
                continue
            if record.item_id <= 0:
                continue
            touched.add(record.item_id)
        return ActivityScan(touched=frozenset(touched), rescan_all=False)

    def commit_checkpoint(self) -> None:
        """Stage the advanced checkpoint, if the scan produced one."""

        if self._pending_checkpoint is None:
            return
        self._cache_store.put_history_checkpoint(self._arr, self._pending_checkpoint)


def _is_file_change(record: HistoryRecord) -> bool:
    """Whether a history record changed arr file state (import / real delete)."""

    event = record.event_type.casefold()
    if event in _IMPORT_EVENTS:
        return True
    if event in _DELETE_EVENTS:
        return (record.reason or "").casefold() != "upgrade"
    return False
